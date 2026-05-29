"""Square-Root Unscented Kalman Filter (Wan & Van der Merwe 2001).

Mathematically equivalent to the standard scaled UKF, but propagates the
Cholesky factor ``S`` (with ``P = S @ S.T``) instead of the full covariance
``P``. This buys two robustness wins for high-dimensional, heterogeneous
state vectors (typical case: ADM1da with 41 states spanning many orders of
magnitude in observability):

* **Positive-definiteness by construction.** ``P = SS^T`` cannot become
  numerically indefinite, so the eigenvalue-floor / multiplicative-jitter
  fallbacks of the standard UKF are unnecessary.
* **Halved condition number.** ``κ(S) = √κ(P)``: at ``κ(P) = 1e10`` we lose
  ~5 significant digits in double precision instead of ~10. This is the
  decisive advantage when weakly observable axes coexist with strongly
  observable ones.

Internal state:
    self.x_hat : posterior mean (n,)
    self._S    : posterior Cholesky factor, lower triangular (n, n)
    self.P     : exposed as a @property computing S @ S.T on demand
                 (so callers and EstimationStep diagnostics keep working).

The constructor signature and the predict/update method signatures are
identical to the previous standard UKF — this file is a drop-in
replacement (re-exported as ``UnscentedKalmanFilter`` from
``filters/__init__.py``).
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy.linalg import solve_triangular

from ..base import EstimationStep
from ..observation_model import ObservationModel
from ..process_model import ADM1ProcessModel
from ..state_vector import StateVectorSpec


# ---------------------------------------------------------------------------
# Cholesky rank-1 update / downdate (LINPACK chud/chdd algorithm)
# ---------------------------------------------------------------------------


def _cholupdate(S: np.ndarray, v: np.ndarray, sign: int) -> np.ndarray:
    """Rank-1 update / downdate of a lower-triangular Cholesky factor.

    Given lower-triangular ``S`` with ``P = S @ S.T``, return a new lower
    triangular ``S'`` such that ``S' @ S'.T = P + sign * v v^T``.

    ``sign`` must be ``+1`` (update) or ``-1`` (downdate). For a downdate
    the result exists only if ``P - vv^T`` is still positive definite —
    if not, a ``LinAlgError`` is raised, which is the *correct* failure
    mode: it tells the caller that "we're trying to subtract more
    information than the prior contains" (model error or bug, not a
    numerical glitch).

    Complexity ``O(n^2)``; works in-place on a private copy.
    """
    if sign not in (+1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    n = len(v)
    S = S.copy()
    v = v.astype(float).copy()
    for k in range(n):
        Skk = S[k, k]
        if sign == +1:
            r = np.sqrt(Skk * Skk + v[k] * v[k])
        else:
            r2 = Skk * Skk - v[k] * v[k]
            if r2 <= 0.0:
                raise np.linalg.LinAlgError(
                    f"cholupdate downdate failed at index {k}: "
                    f"S[{k},{k}]^2 - v[{k}]^2 = {r2} ≤ 0 "
                    f"(downdated matrix is not positive definite)."
                )
            r = np.sqrt(r2)
        c = r / Skk
        s = v[k] / Skk
        S[k, k] = r
        if k + 1 < n:
            if sign == +1:
                S[k + 1 :, k] = (S[k + 1 :, k] + s * v[k + 1 :]) / c
            else:
                S[k + 1 :, k] = (S[k + 1 :, k] - s * v[k + 1 :]) / c
            v[k + 1 :] = c * v[k + 1 :] - s * S[k + 1 :, k]
    return S


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------


class UnscentedKalmanFilter:
    """Square-Root UKF with scaled sigma points and gated observations.

    Args:
        process: Wrapped pyadm1 process model. Must call
            ``process.spec`` to the same StateVectorSpec passed below
            (defensive check).
        obs: Observation model carrying channels to compare against
            measurements.
        spec: State-vector specification (provides bounds, initial mean,
            initial covariance, process noise covariance).
        alpha: Spread of the sigma points around the mean. Default
            ``1.0`` (unscaled UKF, Julier–Uhlmann 1997) is robust for
            strongly nonlinear systems like ADM1; the textbook
            ``1e-3`` (Wan/van-der-Merwe) collapses the propagated
            covariance for biological kinetics.
        beta: Distribution-shape parameter. ``2`` is optimal for
            Gaussian priors.
        kappa: Secondary scaling. ``0`` is the canonical choice for
            ``n > 0``.
    """

    def __init__(
        self,
        process: ADM1ProcessModel,
        obs: ObservationModel,
        spec: StateVectorSpec,
        *,
        alpha: float = 1.0,
        beta: float = 2.0,
        kappa: float = 0.0,
    ):
        if process.spec is not spec:
            raise ValueError("process.spec and spec must be the same object.")
        self.process = process
        self.obs = obs
        self.spec = spec
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)

        self.n = len(spec)
        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n
        if self.n + self.lam <= 0:
            raise ValueError(
                f"Invalid UKF scaling: n+lambda = {self.n + self.lam} ≤ 0. "
                f"Try alpha closer to 1 or kappa > -n."
            )
        self.gamma = np.sqrt(self.n + self.lam)
        self._w_m, self._w_c = self._build_weights()

        self.x_hat: np.ndarray = spec.initial_mean()
        # Cholesky factor of the initial covariance. spec.initial_cov() is
        # diagonal (built from per-channel initial_std), so this is just
        # a diagonal Cholesky.
        P0 = spec.initial_cov()
        self._S: np.ndarray = np.linalg.cholesky(P0)

    # ------------------------------------------------------------------
    # Public covariance as a property (computed on demand)
    # ------------------------------------------------------------------
    @property
    def P(self) -> np.ndarray:
        """Full posterior covariance, ``P = S @ S.T``.

        Recomputed from ``self._S`` each access; never stored. Callers
        who need the covariance many times in a row should cache it.
        """
        return self._S @ self._S.T

    @property
    def S(self) -> np.ndarray:
        """Lower-triangular Cholesky factor of P. Read-only view."""
        return self._S

    # ------------------------------------------------------------------
    # Weights & sigma points
    # ------------------------------------------------------------------
    def _build_weights(self):
        n = self.n
        w_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lam)))
        w_c = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lam)))
        w_m[0] = self.lam / (n + self.lam)
        w_c[0] = w_m[0] + (1.0 - self.alpha**2 + self.beta)
        return w_m, w_c

    def _sigma_points(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Build the 2n+1 symmetric sigma points around ``x`` using ``S``.

        No clipping at sigma-point level: bounds-projection only on the
        mean (clipping sigma points would asymmetrise the sample
        covariance and bias the unscented transform).
        """
        n = self.n
        sigma = np.empty((2 * n + 1, n))
        sigma[0] = x
        # Columns of S are the directions of the principal-axes ellipsoid.
        # Adding +/- gamma * S[:, i] places points on the gamma-sigma
        # surface along each axis.
        gS = self.gamma * S
        for i in range(n):
            sigma[i + 1] = x + gS[:, i]
            sigma[n + i + 1] = x - gS[:, i]
        return sigma

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x_hat = np.asarray(x0, dtype=float).copy()
        P0_arr = np.asarray(P0, dtype=float)
        # Symmetrise defensively — callers sometimes pass an unsymmetric
        # outer-product like ``v @ v.T + epsilon*I`` that's floating-point
        # near-symmetric but not bit-equal.
        P0_arr = 0.5 * (P0_arr + P0_arr.T)
        self._S = np.linalg.cholesky(P0_arr)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        # Snapshot full plant state so every sigma-point step starts
        # from the same baseline. The non-estimated portion of the
        # ADM1 state must not drift between sigma evaluations.
        self.process.snapshot()
        sigma = self._sigma_points(self.x_hat, self._S)
        propagated = np.array([self.process.step(s, dt) for s in sigma])

        # Predicted mean (weighted average).
        x_pred = np.sum(self._w_m[:, None] * propagated, axis=0)

        # Square-root predict via QR + cholupdate.
        # diffs[i] = chi_i_propagated - x_pred. We'll build a stacked
        # matrix whose rows are sqrt(w_c[i]) * diff[i] for i = 1..2n,
        # plus rows of sqrt(Q). QR of this matrix gives R such that
        # R^T R = sum_i w_c[i] diff[i] diff[i]^T + Q, i.e. S_pred = R^T.
        # Then we cholupdate the i=0 term, which can have a different
        # (possibly negative) sign in w_c[0].
        diff = propagated - x_pred  # (2n+1, n)

        # w_c[1:] is always strictly positive (= 1/(2(n+lambda)) > 0
        # under our scaling). Take its sqrt elementwise.
        sqrt_wc = np.sqrt(self._w_c[1:])
        weighted_diffs = sqrt_wc[:, None] * diff[1:]  # (2n, n)

        Q = self.spec.process_noise_cov(dt)
        sqrt_Q = np.linalg.cholesky(Q)  # Q is diagonal → trivial

        stacked = np.vstack([weighted_diffs, sqrt_Q.T])  # (2n + n, n)
        # QR of (2n+n, n) gives Q-factor (2n+n, n) and R (n, n).
        _, R = np.linalg.qr(stacked)
        S_pred = R.T  # lower triangular

        # Now fold in the centre point. w_c[0] can be > 0 (our defaults)
        # or < 0 (some parameterisations) — handle both via cholupdate.
        if self._w_c[0] != 0.0:
            u = np.sqrt(abs(self._w_c[0])) * diff[0]
            sign = +1 if self._w_c[0] > 0 else -1
            S_pred = _cholupdate(S_pred, u, sign=sign)

        self.x_hat = self.spec.clip(x_pred)
        self._S = S_pred

        # Restore baseline, propagate mean once more, snapshot — this
        # is what update() will use as its sigma-point baseline.
        self.process.restore()
        self.process.step(self.x_hat, dt)
        self.process.snapshot()

    # ------------------------------------------------------------------
    # Update (with gated observations)
    # ------------------------------------------------------------------
    def update(
        self,
        y: Dict[str, float],
        t: float,
        gate_values: Optional[Dict[str, float]] = None,
        dt_stub: float = 1e-5,
    ) -> EstimationStep:
        gate_values = gate_values or {}
        # Forgiving key matching (case/separator-insensitive, optional Q_
        # prefix). Returns a dict keyed by the exact channel names.
        y = self.obs.match_measurements(y)
        active = self.obs.active_channels(gate_values)
        active = [c for c in active if c.name in y]

        if not active:
            return EstimationStep(
                t=t,
                x_hat=self.x_hat.copy(),
                P=self.P,
                y_pred={},
                innovation={},
                nis=float("nan"),
                active_channels=[],
            )

        # Build sigma points around the prior, evaluate h(·) for each.
        # process.refresh_outputs restores from snapshot on every call,
        # so each sigma point's observation is consistent with the same
        # background plant state.
        self.process.snapshot()
        sigma = self._sigma_points(self.x_hat, self._S)
        m = len(active)
        y_sigma = np.zeros((len(sigma), m))
        for i, s in enumerate(sigma):
            self.process.refresh_outputs(s, dt_stub=dt_stub)
            y_sigma[i] = self.obs.predict(self.process.plant, s, active=active)

        # Drop channels whose model prediction is non-finite at any sigma
        # point: the model cannot reliably predict them this step, so they
        # must not correct the state (same outcome as a missing measurement).
        # A single bad sigma-point evaluation would otherwise poison the
        # whole unscented transform — and individual sigma points cannot be
        # skipped (the UT needs all 2n+1 to reconstruct mean + covariance).
        finite_cols = np.isfinite(y_sigma).all(axis=0)
        if not finite_cols.all():
            active = [c for c, keep in zip(active, finite_cols) if keep]
            if not active:
                return EstimationStep(
                    t=t,
                    x_hat=self.x_hat.copy(),
                    P=self.P,
                    y_pred={},
                    innovation={},
                    nis=float("nan"),
                    active_channels=[],
                )
            y_sigma = y_sigma[:, finite_cols]
            m = len(active)

        y_pred = np.sum(self._w_m[:, None] * y_sigma, axis=0)
        y_diff = y_sigma - y_pred  # (2n+1, m)

        # Innovation Cholesky factor S_y via QR + cholupdate.
        sqrt_wc = np.sqrt(self._w_c[1:])
        weighted_y_diffs = sqrt_wc[:, None] * y_diff[1:]  # (2n, m)
        R_meas = self.obs.R(active=active)
        sqrt_R = np.linalg.cholesky(R_meas)  # R is diagonal → trivial

        stacked = np.vstack([weighted_y_diffs, sqrt_R.T])  # (2n + m, m)
        _, R_qr = np.linalg.qr(stacked)
        S_y = R_qr.T  # (m, m) lower triangular

        if self._w_c[0] != 0.0:
            u_y = np.sqrt(abs(self._w_c[0])) * y_diff[0]
            sign = +1 if self._w_c[0] > 0 else -1
            S_y = _cholupdate(S_y, u_y, sign=sign)

        # Cross-covariance T_xy = sum w_c[i] (chi_i - x_hat) (Y_i - y_pred)^T.
        # This stays in non-square-root form (it's not symmetric in general).
        x_diff = sigma - self.x_hat
        T_xy = np.zeros((self.n, m))
        for i in range(2 * self.n + 1):
            T_xy += self._w_c[i] * np.outer(x_diff[i], y_diff[i])

        # Kalman gain K = T_xy (S_y S_y^T)^{-1} via two triangular solves.
        # Let U = T_xy (S_y^T)^{-1}, then K = U S_y^{-1}.
        #   Step 1: U S_y^T = T_xy  →  solve S_y U^T = T_xy^T (lower-triangular)
        #   Step 2: K S_y = U       →  solve S_y^T K^T = U^T (upper-triangular)
        U_T = solve_triangular(S_y, T_xy.T, lower=True)  # (m, n)
        K_T = solve_triangular(S_y.T, U_T, lower=False)  # (m, n)
        K = K_T.T  # (n, m)

        y_obs = np.array([y[c.name] for c in active], dtype=float)
        innov = y_obs - y_pred
        x_new = self.x_hat + K @ innov

        # Square-root posterior update: downdate S_x with the columns of
        # K @ S_y (m of them), each rank-1.
        # Equivalent to P_new = P - K S_y S_y^T K^T = P - (K S_y)(K S_y)^T.
        KS = K @ S_y  # (n, m)
        S_new = self._S.copy()
        for j in range(m):
            S_new = _cholupdate(S_new, KS[:, j], sign=-1)

        # NIS = ν^T (S_y S_y^T)^{-1} ν. Use the Cholesky factor:
        # solve S_y z = ν, then NIS = ||z||² (since S_y z = ν gives
        # z^T z = ν^T (S_y S_y^T)^{-1} ν).
        z = solve_triangular(S_y, innov, lower=True)
        nis = float(z @ z)

        self.x_hat = self.spec.clip(x_new)
        self._S = S_new

        # Leave the plant at the posterior mean so external observers
        # see a consistent state and the next predict() snapshot starts
        # from the right baseline.
        self.process.refresh_outputs(self.x_hat, dt_stub=dt_stub)
        self.process.snapshot()

        # Per-channel innovation std: sqrt(diag(S)) with S = S_y · S_y^T.
        # Includes both state-driven uncertainty (sigma-point spread of
        # h) and the measurement noise R. Cheap diagonal extraction.
        s_diag = np.sum(S_y * S_y, axis=1)
        y_std_arr = np.sqrt(np.maximum(s_diag, 0.0))

        return EstimationStep(
            t=t,
            x_hat=self.x_hat.copy(),
            P=self.P,
            y_pred={c.name: float(y_pred[i]) for i, c in enumerate(active)},
            y_std={c.name: float(y_std_arr[i]) for i, c in enumerate(active)},
            innovation={c.name: float(innov[i]) for i, c in enumerate(active)},
            nis=nis,
            active_channels=[c.name for c in active],
        )
