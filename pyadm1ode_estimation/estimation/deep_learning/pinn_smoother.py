"""PINN state-trajectory smoother (Variant A).

A continuous-time collocation PINN used as an *offline* state reconstructor
over a window: it fits a network ``t -> x_hat(t)`` (the full 41-state ADM1da
vector) to sparse noisy sensor measurements while enforcing the ADM1da ODE as
a soft constraint, then returns the whole posterior trajectory. It conforms to
:class:`~pyadm1ode_estimation.estimation.base.BatchEstimator`.

Because the data loss and the physics collocation use *independent* time ranges,
the same per-window fit also **forecasts**: extend the collocation window past
the last measurement and the ADM1 ODE propagates the state into the data-free
tail (see :meth:`PinnSmoother.fit`). It stays a from-scratch per-window fit — no
pre-training.

The three loss terms are

* **data**    ``|| (h(x_hat(t_i)) - y_i) / σ ||²`` at measurement times,
* **physics** ``|| (dx_hat/dt - f(x_hat)) / scale ||²`` at collocation times,
* **prior**   ``|| (x_hat(t0) - x_prior) / scale ||²`` anchoring the start.

``f`` is :func:`pyadm1.core.adm1_torch.adm1da_rhs_torch` and ``h`` is a
:class:`~pyadm1ode_estimation.estimation.deep_learning.observation_torch.TorchObservationModel`
— the differentiable RHS and measurement map. The substrate feed for the
window is baked into the model parameters (``Adm1TorchParams``), so the network
input is time only.

Design choices (see the DL overview doc): the network predicts a *log-deviation
from a prior* — ``x(t) = x_prior * exp(raw(t))`` with the output layer zero-
initialised, so the initial trajectory equals the prior and stays strictly
positive. The physics residual is normalised per state by a characteristic
scale so no single (stiff acid-base) equation dominates. Uncertainty is
optional MC-Dropout. Loss weighting (``lambda_phys``) and the scales are the
tuning knobs for the stiff ADM1 system.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from pyadm1.core.adm1_torch import (
    Adm1TorchParams,
    adm1da_rhs_torch,
    gas_equilibrium_torch,
)

from ..base import TrajectoryEstimate
from .observation_torch import TorchObservationModel
from .pinn import ADM1PINN

_STATE_SIZE = 41
_N_LIQUID = 37  # states 0-36; the 4 gas-phase states (37-40) can be slaved


class PinnSmoother:
    """Offline PINN state-trajectory smoother; see module docstring."""

    def __init__(
        self,
        params: Adm1TorchParams,
        obs: TorchObservationModel,
        x_prior: Sequence[float],
        x_scale: Sequence[float] | None = None,
        hidden_layers: Sequence[int] = (64, 64, 64),
        dropout: float = 0.0,
        lambda_phys: float = 1.0,
        lambda_prior: float = 1.0,
        physics_scaling: str = "rate",
        rate_floor: float = 1.0,
        physics_weight: Sequence[float] | None = None,
        quasi_steady_gas: bool = False,
        gas_n_iter: int = 20,
        device: str = "cpu",
        seed: int = 0,
        dtype: torch.dtype = torch.float64,
    ):
        torch.manual_seed(seed)
        self.params = params
        self.obs = obs
        self.dtype = dtype
        self.device = device
        self.lambda_phys = float(lambda_phys)
        self.lambda_prior = float(lambda_prior)
        self.dropout = float(dropout)
        # Physics-residual normalisation. "state" divides by the state magnitude
        # (a relative rate); on the stiff ADM1 that lets the fast acid-base terms
        # (k_A_B = 1e8) dominate the loss. "rate" divides by the current |f_i|
        # (floored at ``rate_floor * state magnitude``), i.e. a *relative* ODE
        # residual, so no single equation dominates by sheer rate magnitude.
        if physics_scaling not in ("state", "rate"):
            raise ValueError(
                f"physics_scaling must be 'state' or 'rate', got {physics_scaling!r}."
            )
        self.physics_scaling = physics_scaling
        # ``rate_floor`` scales the rate-scaling floor. The default 1.0 floors at
        # the full state magnitude, which makes a *slow* drift (|f| << magnitude,
        # e.g. biomass/VFA changing over days) contribute a near-zero residual —
        # the network then parks such states at the prior. A small floor (e.g.
        # 0.05) keeps the relative residual near 1 for a wrongly-flat slow state,
        # so slow dynamics stay visible in the loss.
        self.rate_floor = float(rate_floor)
        # With quasi_steady_gas the network only predicts the 37 liquid states;
        # the 4 gas-phase pressures are solved from the liquid state each call
        # (removes the knife-edge free pTOTAL). Physics + prior are enforced on
        # the 37 free states only (the gas ODEs hold by construction).
        self.quasi_steady_gas = bool(quasi_steady_gas)
        self._n_free = _N_LIQUID if quasi_steady_gas else _STATE_SIZE
        self._gas_n_iter = int(gas_n_iter)

        x_prior = np.asarray(x_prior, dtype=float)
        if x_prior.shape != (_STATE_SIZE,):
            raise ValueError(
                f"x_prior must have shape ({_STATE_SIZE},), got {x_prior.shape}."
            )
        # Strictly-positive base for the log transform; a true-zero prior would
        # otherwise pin that state at zero for all time.
        base = np.clip(x_prior, 1.0e-8, None)
        scale = (
            base
            if x_scale is None
            else np.clip(np.asarray(x_scale, dtype=float), 1.0e-12, None)
        )

        self._x_prior = torch.tensor(x_prior, dtype=dtype, device=device)
        self._base = torch.tensor(base, dtype=dtype, device=device)
        self._scale = torch.tensor(scale, dtype=dtype, device=device)
        self._sigma = obs.noise_std_tensor(dtype=dtype, device=device)

        # Per-state physics weight (on the ``_n_free`` free states). Lets slow,
        # weakly-observed states be up-weighted so the optimiser bothers to move
        # them instead of leaving them at the prior. Default: uniform.
        if physics_weight is None:
            self._phys_weight = torch.ones(self._n_free, dtype=dtype, device=device)
        else:
            w = np.asarray(physics_weight, dtype=float)
            if w.shape != (self._n_free,):
                raise ValueError(
                    f"physics_weight must have shape ({self._n_free},), got {w.shape}."
                )
            self._phys_weight = torch.tensor(w, dtype=dtype, device=device)

        self.net = ADM1PINN(
            input_dim=1,
            output_dim=self._n_free,
            hidden_layers=list(hidden_layers),
            dropout=dropout,
        ).to(device=device, dtype=dtype)
        self._zero_init_output()

        # Persistent state for warm-started rolling updates (see update()).
        self._opt: torch.optim.Optimizer | None = None
        self._t0_origin: float | None = None

    # -- network -> physical state --------------------------------------
    def _zero_init_output(self) -> None:
        """Zero the final layer so raw(t)=0 and x(t)=x_prior at initialisation."""
        last = self.net.net[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()

    def _forward_state(self, t: torch.Tensor) -> torch.Tensor:
        """``t (N, 1) -> x (N, 41)``  via  ``x = base * exp(clamp(raw))``.

        In quasi-steady-gas mode the network outputs the 37 liquid states and
        the 4 gas-phase pressures are solved from them, then concatenated.
        """
        raw = torch.exp(torch.clamp(self.net(t), -10.0, 10.0))
        if self.quasi_steady_gas:
            x_liq = self._base[:_N_LIQUID] * raw
            gas = gas_equilibrium_torch(x_liq, self.params, n_iter=self._gas_n_iter)
            return torch.cat([x_liq, gas], dim=-1)
        return self._base * raw

    def _dxdt(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Autograd time-derivative of the ``_n_free`` free states, ``(N, n_free)``."""
        cols: list[torch.Tensor] = []
        ones = torch.ones_like(x[:, 0])
        for i in range(self._n_free):
            g = torch.autograd.grad(
                x[:, i], t, grad_outputs=ones, create_graph=True, retain_graph=True
            )[0]
            cols.append(g[:, 0])
        return torch.stack(cols, dim=-1)

    # -- fit ------------------------------------------------------------
    def fit(
        self,
        obs_times: Sequence[float],
        obs_values: np.ndarray | dict[str, Sequence[float]],
        t0: float,
        t1: float,
        n_collocation: int = 200,
        epochs: int = 2000,
        lr: float = 1.0e-3,
        grad_clip: float | None = 1.0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Train the smoother over ``[t0, t1]``.

        Args:
            obs_times: Measurement times, shape ``(To,)`` [days].
            obs_values: Either an array ``(To, n_channels)`` in the observation
                model's channel order, or a mapping ``{channel: values}``.
                ``NaN`` entries are treated as missing (gated) and skipped.
            t0, t1: **Collocation window** bounds [days] — the ODE residual is
                enforced across all of ``[t0, t1]`` and the prior is anchored at
                ``t0``. This is *independent* of the measurement extent.
            n_collocation: Number of uniform physics-residual points in the window.
            epochs, lr: Adam training schedule.

        Forecasting (physics-driven, still a per-window fit — no pre-training):
            the data loss lives only at ``obs_times``, so if ``t1`` runs **past
            the last measurement** the tail ``(max(obs_times), t1]`` has no data
            and the ADM1 ODE alone (``dx/dt = f(x)``) carries the state forward
            from the data-anchored end — an initial-value continuation. Pass the
            historical measurements as ``obs_times`` and set ``t1`` to the forecast
            horizon; ``estimate(t)`` for ``t`` beyond the data is then a forecast
            (put enough ``n_collocation`` points to cover the tail densely).

        Returns:
            Loss history dict with keys ``loss``, ``data``, ``phys``, ``prior``.
        """
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._opt = opt  # keep for warm-started update() calls
        self._t0_origin = float(t0)  # the growing-window anchor
        return self._run_epochs(
            obs_times,
            obs_values,
            t0,
            t1,
            self._x_prior,
            opt,
            n_collocation,
            epochs,
            grad_clip,
            verbose,
        )

    def _run_epochs(
        self,
        obs_times: Sequence[float],
        obs_values: np.ndarray | dict[str, Sequence[float]],
        t0: float,
        t1: float,
        x_prior_anchor: torch.Tensor,
        opt: torch.optim.Optimizer,
        n_collocation: int,
        epochs: int,
        grad_clip: float | None,
        verbose: bool,
    ) -> dict[str, list[float]]:
        """Core training loop shared by :meth:`fit` and :meth:`update`.

        Trains over ``[t0, t1]`` with the given optimiser and prior anchor
        (``x_prior_anchor`` pins ``x̂(t0)``). Does **not** re-initialise the
        network, so a reused optimiser + weights give a warm start.
        """
        t_obs = torch.tensor(
            np.asarray(obs_times, dtype=float).reshape(-1, 1),
            dtype=self.dtype,
            device=self.device,
        )
        y_obs = self._as_value_array(obs_values, len(obs_times))
        y_obs_t = torch.tensor(y_obs, dtype=self.dtype, device=self.device)
        valid = ~torch.isnan(y_obs_t)
        n_valid = valid.sum().clamp(min=1)

        t_coll = torch.linspace(
            t0, t1, n_collocation, dtype=self.dtype, device=self.device
        ).reshape(-1, 1)
        t_coll.requires_grad_(True)
        t0_t = torch.tensor([[t0]], dtype=self.dtype, device=self.device)
        n = self._n_free

        history: dict[str, list[float]] = {
            "loss": [],
            "data": [],
            "phys": [],
            "prior": [],
        }
        self.net.train()
        for ep in range(epochs):
            opt.zero_grad()

            # data loss (masked by observed cells)
            y_pred = self.obs.predict(self._forward_state(t_obs))
            res_d = (y_pred - y_obs_t) / self._sigma
            res_d = torch.where(valid, res_d, torch.zeros_like(res_d))
            l_data = (res_d**2).sum() / n_valid

            # physics loss on the free states only (in QSS mode the 4 gas ODEs
            # are satisfied by construction, so only the 37 liquid ODEs remain)
            x_c = self._forward_state(t_coll)
            f = adm1da_rhs_torch(x_c, self.params)
            f_free = f[..., :n]
            if self.physics_scaling == "rate":
                # relative residual: divide by |f| (floored at ``rate_floor *
                # state scale`` so near-equilibrium states with f≈0 don't blow up,
                # while slow drifts stay visible for a small rate_floor).
                denom = torch.clamp(
                    f_free.detach().abs(), min=self.rate_floor * self._scale[:n]
                )
            else:
                denom = self._scale[:n]
            res_p = (self._dxdt(t_coll, x_c) - f_free) / denom
            # per-state weight (default uniform): up-weight slow / weakly-observed
            # states so the optimiser moves them instead of parking them.
            l_phys = (self._phys_weight[:n] * res_p**2).mean()

            # prior anchor at t0 (free states)
            res_pr = (
                self._forward_state(t0_t)[..., :n] - x_prior_anchor[:n]
            ) / self._scale[:n]
            l_prior = (res_pr**2).mean()

            loss = l_data + self.lambda_phys * l_phys + self.lambda_prior * l_prior
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), grad_clip)
            opt.step()

            ld, lp, lpr, lo = (
                float(v.detach()) for v in (l_data, l_phys, l_prior, loss)
            )
            history["loss"].append(lo)
            history["data"].append(ld)
            history["phys"].append(lp)
            history["prior"].append(lpr)
            if verbose and (ep % max(1, epochs // 10) == 0 or ep == epochs - 1):
                print(
                    f"[{ep:5d}] loss={lo:.4e} data={ld:.4e} phys={lp:.4e} prior={lpr:.4e}"
                )

        return history

    # -- warm-started rolling update ------------------------------------
    def update(
        self,
        obs_times: Sequence[float],
        obs_values: np.ndarray | dict[str, Sequence[float]],
        t1: float,
        *,
        window: float | None = None,
        n_collocation: int = 200,
        epochs: int = 100,
        lr: float | None = 3.0e-4,
        grad_clip: float | None = 1.0,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        """Warm-started incremental re-fit for rolling operation.

        Reuses the current network weights **and** optimiser state instead of
        retraining from scratch, so each new step (e.g. a new hourly sample)
        converges in far fewer ``epochs``. Requires a prior :meth:`fit`.

        Window mode is chosen by ``window``:

        * ``window is None`` — **growing window**: ``t0`` stays at the original
          :meth:`fit` ``t0`` (all measurements from the start are used and the
          original prior anchors ``x̂(t0)``). Use when you keep the whole history.
        * ``window`` a float — **sliding window** of that many days: ``t0 = t1 −
          window``, only measurements in ``[t0, t1]`` are used, and the window
          start is anchored to the network's **own current estimate at ``t0``**
          (a self-consistent hand-off), not the stale initial prior. Use to cap
          compute / memory on a long run.

        As in :meth:`fit`, ``t1`` may run past the last measurement to forecast.

        Args:
            obs_times, obs_values: measurements (the full history for a growing
                window; anything is fine, the window is applied here).
            t1: current window end [days] (also the forecast horizon).
            window: sliding-window length [days], or ``None`` for a growing window.
            epochs, lr: warm-start schedule (few epochs, smaller lr than fit).

        Returns:
            Loss history dict (``loss``, ``data``, ``phys``, ``prior``).
        """
        if self._t0_origin is None or self._opt is None:
            raise RuntimeError("Call fit(...) once before update(...).")

        obs_t = np.asarray(obs_times, dtype=float)
        y_all = self._as_value_array(obs_values, len(obs_t))  # (To, n_ch)

        if window is None:
            t0 = self._t0_origin
            x_prior_anchor = self._x_prior
            sel = obs_t <= float(t1) + 1.0e-9
        else:
            t0 = float(t1) - float(window)
            # Self-anchor: freeze the current estimate at the new window start so
            # the physics has a boundary condition consistent with the past fit.
            with torch.no_grad():
                self.net.eval()
                t0_col = torch.tensor([[t0]], dtype=self.dtype, device=self.device)
                x_prior_anchor = self._forward_state(t0_col)[0].detach()
            sel = (obs_t >= t0 - 1.0e-9) & (obs_t <= float(t1) + 1.0e-9)

        if lr is not None:
            for g in self._opt.param_groups:
                g["lr"] = float(lr)

        return self._run_epochs(
            obs_t[sel],
            y_all[sel],
            t0,
            float(t1),
            x_prior_anchor,
            self._opt,
            n_collocation,
            epochs,
            grad_clip,
            verbose,
        )

    # -- estimate (BatchEstimator contract) -----------------------------
    def estimate(
        self, times: Sequence[float], mc_samples: int = 1
    ) -> TrajectoryEstimate:
        """Query the fitted trajectory (+ MC-Dropout uncertainty).

        Args:
            times: Query times, shape ``(T,)`` [days].
            mc_samples: Number of stochastic forward passes for the MC-Dropout
                uncertainty band. ``> 1`` requires ``dropout > 0``; otherwise
                the estimate is deterministic and ``std`` is zero.
        """
        times = np.asarray(times, dtype=float)
        t = torch.tensor(times.reshape(-1, 1), dtype=self.dtype, device=self.device)

        with torch.no_grad():
            if mc_samples > 1 and self.dropout > 0.0:
                self.net.train()  # keep dropout active for MC sampling
                draws = torch.stack(
                    [self._forward_state(t) for _ in range(mc_samples)], dim=0
                )
                x_hat = draws.mean(dim=0)
                std = draws.std(dim=0)
            else:
                self.net.eval()
                x_hat = self._forward_state(t)
                std = torch.zeros_like(x_hat)

        return TrajectoryEstimate(
            time=times, x_hat=x_hat.cpu().numpy(), std=std.cpu().numpy()
        )

    # -- helpers --------------------------------------------------------
    def _as_value_array(self, obs_values, n_times: int) -> np.ndarray:
        if isinstance(obs_values, dict):
            arr = np.full((n_times, len(self.obs)), np.nan, dtype=float)
            for j, name in enumerate(self.obs.channel_names):
                if name in obs_values:
                    arr[:, j] = np.asarray(obs_values[name], dtype=float)
            return arr
        arr = np.asarray(obs_values, dtype=float)
        if arr.shape != (n_times, len(self.obs)):
            raise ValueError(
                f"obs_values shape {arr.shape} != (n_times={n_times}, n_channels={len(self.obs)})."
            )
        return arr
