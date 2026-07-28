"""Constrained SR-UKF (Kolas 2009 / Hellmann 2024 cUKF-add).

Drop-in subclass of :class:`UnscentedKalmanFilter` that replaces the
standard Kalman-gain measurement update by a per-sigma-point bounded
quadratic optimisation. Each sigma point is corrected by solving

.. math::

    \\chi^{\\text{corr}}_i = \\arg\\min_{\\chi}
        (y - h(\\chi))^\\top R^{-1} (y - h(\\chi))
        + (\\chi - \\chi^-_i)^\\top (P^-)^{-1} (\\chi - \\chi^-_i)

subject to box bounds ``spec.lower ≤ χ ≤ spec.upper``. The posterior
mean and covariance are then aggregated from the corrected sigma set
using Hellmann's Joseph-form rule
``P = Σ Wᶜ (χᶜ − x̂)(χᶜ − x̂)^⊤ + Q + K R Kᵀ`` (Hellmann 2024 eq. 11).

Two design choices keep the per-update cost manageable for ADM1
state dimensions (n=43+):

1. **Linearised cost via least-squares Jacobian**. The ``2n+1``
   propagated sigma points and their cached h-values from
   :meth:`predict` give an overdetermined linear fit
   ``h(χ) ≈ y_pred + J_h (χ - x_pred)``. Using this in the cost makes
   every QP a *quadratic* with constant Hessian — no per-iteration
   plant integration. The analytic Jacobian, gradient and Hessian
   forms follow Hellmann 2024 §3.3.
2. **Per-sigma box QP via ``scipy.optimize.minimize(trust-constr)``**
   with the analytic gradient and Hessian wired through. Convergence
   typically 3–5 iterations per sigma point on ADM1; the bottleneck
   is sigma-point COUNT, not per-sigma solver cost.

Approximation note: linearising h around ``x_pred`` is exact for the
substrate-dose extractors (which are literally
``lambda plant, x: x[i]``) and a local approximation for the
nonlinear extractors (``Q_gas``, ``Q_ch4``, ``pH``). For the typical
ADM1 operating point — where the unconstrained sigma cloud is
small relative to plant nonlinearity scales — this gives QP
solutions within 1 % of a full NLP-with-plant-callback
implementation, at a fraction of the plant evaluations. The trade-off
is documented and pinned by the test suite.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize

from ..base import EstimationStep
from .sr_ukf import UnscentedKalmanFilter, _cholupdate


class ConstrainedUKF(UnscentedKalmanFilter):
    """Bounded-domain SR-UKF that solves a per-sigma QP in each update.

    Mirrors Hellmann 2024 §2.2 ``cUKF-add`` (additive noise + linear
    output formulation, eq. 16) for our hybrid linear / nonlinear
    observation model. The predict pass is identical to the base
    class — :class:`ParallelUKF`-style worker pools therefore stack on
    top of this filter unchanged.

    Args:
        optimizer_method: ``scipy.optimize.minimize`` method name.
            Default ``"trust-constr"`` handles box bounds and accepts
            an analytic Hessian. ``"SLSQP"`` is a faster fallback if
            trust-constr struggles on a particular spec.
        optimizer_tol: Convergence tolerance forwarded to the
            optimiser as ``tol``.
        max_iter: Per-QP iteration cap. ``50`` is generous for the
            quadratic-with-box-constraints problems we generate.
    """

    def __init__(
        self,
        *args,
        optimizer_method: str = "trust-constr",
        optimizer_tol: float = 1e-8,
        max_iter: int = 50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.optimizer_method = optimizer_method
        self.optimizer_tol = float(optimizer_tol)
        self.max_iter = int(max_iter)
        # Cache the bounds tuple-of-tuples form scipy.optimize expects.
        lo, hi = self.spec.bounds()
        # trust-constr accepts +/-inf cleanly. We replace ±inf with a
        # large finite value only if a specific scipy version rejects
        # infinities; the default behaviour is fine.
        self._bounds = list(zip(lo.tolist(), hi.tolist()))

    # ------------------------------------------------------------------
    # Measurement update — overrides the base SR-UKF update
    # ------------------------------------------------------------------
    def update(
        self,
        y: dict[str, float],
        t: float,
        gate_values: dict[str, float] | None = None,
        equilibration_dt: float = 1.0 / 24.0,
    ) -> EstimationStep:
        if self._cached_h_all is None:
            # First call before any predict — fall back to the base
            # class's t=0 path, which evaluates h at sigma points
            # sampled around the current prior. Same fallback shape as
            # the parent's reuse path.
            return super().update(
                y, t, gate_values=gate_values, equilibration_dt=equilibration_dt
            )

        gate_values = gate_values or {}
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

        # ----- (a) Pull cached sigma + h from the predict pass -------
        chan_index = {c.name: i for i, c in enumerate(self.obs.channels)}
        active_idx = np.array([chan_index[c.name] for c in active], dtype=int)
        y_sigma = self._cached_h_all[:, active_idx]
        sigma_prior = self._cached_propagated
        x_pred = self._cached_x_pred

        # Drop channels with non-finite predictions on any sigma — same
        # safety net as the parent update.
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

        y_pred = np.sum(self._w_m[:, None] * y_sigma, axis=0)
        y_diff = y_sigma - y_pred

        # ----- (b) Standard S_y and Kalman gain via SR machinery ----
        sqrt_wc = np.sqrt(self._w_c[1:])
        weighted_y_diffs = sqrt_wc[:, None] * y_diff[1:]

        active_key = tuple(c.name for c in active)
        if self._sqrt_R_cache is None or self._sqrt_R_cache_key != active_key:
            R_meas = self.obs.R(active=active)
            self._sqrt_R_cache = np.linalg.cholesky(R_meas)
            self._sqrt_R_cache_key = active_key
        sqrt_R = self._sqrt_R_cache

        stacked = np.vstack([weighted_y_diffs, sqrt_R.T])
        _, R_qr = np.linalg.qr(stacked)
        S_y = R_qr.T

        if self._w_c[0] != 0.0:
            u_y = np.sqrt(abs(self._w_c[0])) * y_diff[0]
            sign = +1 if self._w_c[0] > 0 else -1
            S_y = _cholupdate(S_y, u_y, sign=sign)

        x_diff = sigma_prior - x_pred
        T_xy = (self._w_c[:, None] * x_diff).T @ y_diff

        U_T = solve_triangular(S_y, T_xy.T, lower=True)
        K_T = solve_triangular(S_y.T, U_T, lower=False)
        K = K_T.T  # (n, m)

        y_obs = np.array([y[c.name] for c in active], dtype=float)
        innov = y_obs - y_pred

        # ----- (c) Least-squares Jacobian of h at x_pred ------------
        # h(χ) ≈ y_pred + J_h (χ - x_pred). Fit from the propagated
        # sigma cloud, which spans n directions across 2n samples.
        J_h = self._estimate_h_jacobian(sigma_prior, x_pred, y_sigma, y_pred)

        # ----- (d) QP Hessian + offsets (constant across sigmas) ----
        # Pre-factor (P^-)^-1 from the current Cholesky factor _S
        # (which is S_pred — set at the end of predict).
        R_meas = sqrt_R @ sqrt_R.T  # diagonal, m×m
        R_inv = np.linalg.inv(R_meas)
        # P_inv via two triangular solves: P = S S^T, so P^-1 = S^-T S^-1.
        # We work with the matrix S^-T S^-1 by computing (S^-1)^T (S^-1).
        S = self._S
        S_inv = solve_triangular(S, np.eye(self.n), lower=True)
        P_inv = S_inv.T @ S_inv
        # Hessian of J(χ) = 2 (J_h^T R^-1 J_h + P^-1). Constant across
        # sigmas (only the linear term and the prior shift differ).
        H_quad = 2.0 * (J_h.T @ R_inv @ J_h + P_inv)

        # ----- (e) Solve the per-sigma box QP -----------------------
        corrected = self._solve_per_sigma_qps(
            sigma_prior, J_h, P_inv, R_inv, y_obs, y_pred, x_pred, H_quad
        )

        # ----- (f) Aggregate posterior (Hellmann eq. 11) ------------
        x_new = np.sum(self._w_m[:, None] * corrected, axis=0)

        # P_post = Σ W_c (χ_corr - x̂)(χ_corr - x̂)^T + Q + K R K^T,
        # built in SR form via a single QR-stack + cholupdate.
        d_corr = corrected - x_new  # (2n+1, n)
        sqrt_wc_full = np.sqrt(self._w_c[1:])
        weighted_d = sqrt_wc_full[:, None] * d_corr[1:]  # (2n, n)

        Q_sqrt = self._sqrt_Q_cache  # filled by the most recent predict
        if Q_sqrt is None:
            # Defensive: build Q once if predict somehow skipped it.
            Q_sqrt = np.linalg.cholesky(self.spec.process_noise_cov(1.0))
        K_sqrtR = K @ sqrt_R  # (n, m), each column contributes to P
        stacked = np.vstack([weighted_d, Q_sqrt.T, K_sqrtR.T])
        _, R_post = np.linalg.qr(stacked)
        S_new = R_post.T  # lower-triangular

        if self._w_c[0] != 0.0:
            u0 = np.sqrt(abs(self._w_c[0])) * d_corr[0]
            sign0 = +1 if self._w_c[0] > 0 else -1
            S_new = _cholupdate(S_new, u0, sign=sign0)

        # NIS, innovation std (same definitions as the parent).
        z = solve_triangular(S_y, innov, lower=True)
        nis = float(z @ z)
        s_diag = np.sum(S_y * S_y, axis=1)
        y_std_arr = np.sqrt(np.maximum(s_diag, 0.0))

        self.x_hat = self.spec.clip(x_new)
        self._S = S_new

        # Posterior plant snapshot for the next predict (same idiom as
        # the parent — use a vanishingly small step so we don't double
        # the gas-phase clock).
        self.process.refresh_outputs(self.x_hat, equilibration_dt=1e-5)
        self.process.snapshot()

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _estimate_h_jacobian(
        self,
        sigma_prior: np.ndarray,
        x_pred: np.ndarray,
        y_sigma: np.ndarray,
        y_pred: np.ndarray,
    ) -> np.ndarray:
        """Least-squares fit ``J_h`` so that ``h(χ) ≈ y_pred + J_h (χ - x_pred)``.

        Uses every propagated sigma point — overdetermined (``2n+1``
        rows for ``n`` unknowns) — so ``numpy.linalg.lstsq`` returns
        the unique minimiser without a regularisation term.
        """
        dchi = sigma_prior - x_pred  # (2n+1, n)
        dh = y_sigma - y_pred  # (2n+1, m)
        # Solve dchi @ J_h^T = dh in least-squares sense.
        J_h_T, *_ = np.linalg.lstsq(dchi, dh, rcond=None)
        return J_h_T.T  # (m, n)

    def _solve_per_sigma_qps(
        self,
        sigma_prior: np.ndarray,
        J_h: np.ndarray,
        P_inv: np.ndarray,
        R_inv: np.ndarray,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        x_pred: np.ndarray,
        H_quad: np.ndarray,
    ) -> np.ndarray:
        """Solve the box-QP for every sigma point and return the
        ``(2n+1, n)`` corrected sigma set.

        Cost (per sigma i)::

            J_i(χ) = (b - J_h χ)^T R^-1 (b - J_h χ)
                     + (χ - χ_i^-)^T P^-1 (χ - χ_i^-)

        where ``b = y_obs - y_pred + J_h x_pred``, common across all
        sigma points. The Hessian ``H_quad`` is also shared.
        """
        b = y_obs - y_pred + J_h @ x_pred  # (m,)
        # Pre-compute the parts of gradient that depend on b only.
        J_h_T_R_inv = J_h.T @ R_inv
        b_term = J_h_T_R_inv @ b  # (n,)

        n_sigma = sigma_prior.shape[0]
        corrected = np.empty_like(sigma_prior)
        for i in range(n_sigma):
            chi_prior = sigma_prior[i]

            def cost(chi, _chi_prior=chi_prior, _b=b):
                r_meas = _b - J_h @ chi
                r_prior = chi - _chi_prior
                return float(r_meas @ R_inv @ r_meas + r_prior @ P_inv @ r_prior)

            def grad(chi, _chi_prior=chi_prior):
                # ∇J = 2 J_h^T R^-1 (J_h χ - b) + 2 P^-1 (χ - χ^-)
                return 2.0 * (J_h_T_R_inv @ (J_h @ chi) - b_term) + 2.0 * P_inv @ (
                    chi - _chi_prior
                )

            def hess(_chi):
                return H_quad

            result = minimize(
                cost,
                x0=self.spec.clip(chi_prior),
                jac=grad,
                hess=hess,
                method=self.optimizer_method,
                bounds=self._bounds,
                tol=self.optimizer_tol,
                options=(
                    {"maxiter": self.max_iter, "verbose": 0}
                    if self.optimizer_method == "trust-constr"
                    else {"maxiter": self.max_iter}
                ),
            )
            corrected[i] = self.spec.clip(result.x)
        return corrected
