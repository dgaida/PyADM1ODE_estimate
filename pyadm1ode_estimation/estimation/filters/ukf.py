"""Unscented Kalman Filter (Wan/van-der-Merwe scaled formulation).

This is the conservative, textbook UKF: symmetric sigma points,
scaled-unscented transform with parameters ``(α, β, κ)``, additive
process and measurement noise. No iteration, no square-root
formulation — the dimensionality (≈ 8 channels for Phase 1) keeps
numerical conditioning well-behaved with the plain Cholesky path.

The filter owns one :class:`pyadm1.BiogasPlant`. Each propagation
sigma point pushes its state into that plant via the process model
and runs ``plant.step(dt)``. After all sigma points have been
propagated the plant is left at the *last sigma point's* state — we
re-apply the posterior mean at the end of :meth:`update` so the
caller always sees the plant at ``x_hat``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ..base import EstimationStep
from ..observation_model import ObservationModel
from ..process_model import ADM1ProcessModel
from ..state_vector import StateVectorSpec

if TYPE_CHECKING:
    pass


class UnscentedKalmanFilter:
    """Standard UKF with scaled sigma points and gated observations.

    Args:
        process: Wrapped pyadm1 process model.
        obs: Observation model carrying the channels to compare
            against measurements.
        spec: State-vector specification (provides bounds, initial,
            process noise covariance).
        alpha: Spread of the sigma points around the mean. ``1e-3``
            is the canonical default for fully nonlinear problems
            and keeps sigma points close to the mean.
        beta: Encodes knowledge of the distribution shape. ``2`` is
            optimal for Gaussian priors.
        kappa: Secondary scaling. ``0`` for ``n > 0`` problems is the
            common choice.
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
        # Default ``alpha=1.0`` is the *un*-scaled UKF (Julier–Uhlmann,
        # 1997). The textbook ``alpha=1e-3`` of Wan/van-der-Merwe gives
        # sigma-point spread γ ≈ √(α²·(n+κ)) ≈ 3e-3 for n=8 — far too
        # small for ADM1's strong nonlinearity, the propagated sample
        # covariance collapses to zero and the filter can't learn. For
        # nearly linear problems α=1e-3 is fine; for biological
        # kinetics the unscaled form is the safe default.
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
        self.gamma = np.sqrt(self.n + self.lam)
        self._w_m, self._w_c = self._build_weights()

        self.x_hat: np.ndarray = spec.initial_mean()
        self.P: np.ndarray = spec.initial_cov()

    # ------------------------------------------------------------------
    # Sigma points + weights
    # ------------------------------------------------------------------
    def _build_weights(self):
        n = self.n
        w_m = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lam)))
        w_c = np.full(2 * n + 1, 1.0 / (2.0 * (n + self.lam)))
        w_m[0] = self.lam / (n + self.lam)
        w_c[0] = w_m[0] + (1.0 - self.alpha**2 + self.beta)
        return w_m, w_c

    def _sigma_points(self, x: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Build the 2n+1 scaled sigma points without clipping.

        Clipping sigma points to the configured bounds *here* would make
        the sample covariance asymmetric and break the unscented
        transform. Bounds are enforced separately on the propagated /
        updated mean — that gives a slight bias toward the interior but
        keeps the filter equations valid.
        """
        n = self.n
        P_sym = 0.5 * (P + P.T)

        # Eigenvalue floor: makes the matrix strictly positive definite
        # before Cholesky is attempted. Cheaper-than-it-sounds for n≈10
        # and far more robust than chasing a Cholesky failure with
        # multiplicative jitter — the latter destroys conditioning when
        # P has rows that differ in scale by many orders of magnitude
        # (typical here: S_ac ~10⁻¹ vs. Q_solid ~10¹).
        floor = 1e-8 * max(1.0, np.trace(P_sym) / n)
        eigs = np.linalg.eigvalsh(P_sym)
        if eigs.min() < floor:
            P_sym = P_sym + (floor - eigs.min()) * np.eye(n)

        try:
            L = np.linalg.cholesky(P_sym)
        except np.linalg.LinAlgError:
            # Final safety net — multiplicative inflation.
            jitter = 1e-6 * np.eye(n)
            for _ in range(10):
                try:
                    L = np.linalg.cholesky(P_sym + jitter)
                    break
                except np.linalg.LinAlgError:
                    jitter *= 10
            else:
                raise np.linalg.LinAlgError(
                    "Posterior covariance is not positive definite even after "
                    "eigenvalue flooring and jitter inflation."
                )

        sigma = np.zeros((2 * n + 1, n))
        sigma[0] = x
        for i in range(n):
            sigma[i + 1] = x + self.gamma * L[:, i]
            sigma[n + i + 1] = x - self.gamma * L[:, i]
        return sigma

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        self.x_hat = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict(self, dt: float) -> None:
        # Snapshot full plant state so every sigma-point step starts
        # from the same baseline (otherwise the 33 non-estimated ADM1
        # variables drift between sigma evaluations and corrupt P).
        self.process.snapshot()
        sigma = self._sigma_points(self.x_hat, self.P)
        propagated = np.array([self.process.step(s, dt) for s in sigma])
        x_pred = np.sum(self._w_m[:, None] * propagated, axis=0)
        diff = propagated - x_pred
        P_pred = (self._w_c[:, None, None] * (diff[:, :, None] @ diff[:, None, :])).sum(
            0
        )
        P_pred = P_pred + self.spec.process_noise_cov(dt)
        P_pred = 0.5 * (P_pred + P_pred.T)
        self.x_hat = self.spec.clip(x_pred)
        self.P = P_pred
        # Restore baseline, propagate mean once more, snapshot the
        # post-step plant — this is what update() will then use as its
        # sigma-point baseline. ``process.step`` already calls restore
        # internally, so the explicit restore() here is the safety
        # belt for callers that bypass step().
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
        active = self.obs.active_channels(gate_values)
        active = [c for c in active if c.name in y and np.isfinite(y[c.name])]

        if not active:
            return EstimationStep(
                t=t,
                x_hat=self.x_hat.copy(),
                P=self.P.copy(),
                y_pred={},
                innovation={},
                nis=float("nan"),
                active_channels=[],
            )

        # Build a fresh sigma set around the prior, evaluate h(·) for
        # each sigma point. The process model restores from its
        # snapshot baseline on every refresh_outputs call, so each
        # sigma point's observation is consistent with the same
        # background plant state.
        self.process.snapshot()
        sigma = self._sigma_points(self.x_hat, self.P)
        y_sigma = np.zeros((len(sigma), len(active)))
        for i, s in enumerate(sigma):
            self.process.refresh_outputs(s, dt_stub=dt_stub)
            y_sigma[i] = self.obs.predict(self.process.plant, s, active=active)

        y_pred = np.sum(self._w_m[:, None] * y_sigma, axis=0)
        y_diff = y_sigma - y_pred
        S = (self._w_c[:, None, None] * (y_diff[:, :, None] @ y_diff[:, None, :])).sum(
            0
        )
        S = S + self.obs.R(active=active)
        S = 0.5 * (S + S.T)

        x_diff = sigma - self.x_hat
        T_xy = (
            self._w_c[:, None, None] * (x_diff[:, :, None] @ y_diff[:, None, :])
        ).sum(0)

        # Kalman gain via solve to avoid explicit inverse.
        K = np.linalg.solve(S.T, T_xy.T).T

        y_obs = np.array([y[c.name] for c in active], dtype=float)
        innov = y_obs - y_pred
        x_new = self.x_hat + K @ innov

        # Joseph-form covariance update — numerically stable, guarantees
        # symmetric positive-semidefinite P even with rank-deficient S.
        #   P⁺ = P − K T_xy^T − T_xy K^T + K S K^T
        # The two cross-terms cancel ``K @ T_xy^T = T_xy @ K^T`` exactly
        # in the analytic gain, but float64 round-off can drift them
        # apart — keeping all four explicit symmetrises the result.
        KT = T_xy @ K.T
        P_new = self.P - KT - KT.T + K @ S @ K.T
        P_new = 0.5 * (P_new + P_new.T)

        # NIS = ν^T S^{-1} ν, scaled by len(active) for χ² interpretation.
        nis = float(innov @ np.linalg.solve(S, innov))

        self.x_hat = self.spec.clip(x_new)
        self.P = P_new

        # Leave the plant at the posterior mean so external observers
        # (e.g. plot scripts) see a consistent state, and so the next
        # predict() snapshot starts from the right baseline.
        self.process.refresh_outputs(self.x_hat, dt_stub=dt_stub)
        self.process.snapshot()

        return EstimationStep(
            t=t,
            x_hat=self.x_hat.copy(),
            P=self.P.copy(),
            y_pred={c.name: float(y_pred[i]) for i, c in enumerate(active)},
            innovation={c.name: float(innov[i]) for i, c in enumerate(active)},
            nis=nis,
            active_channels=[c.name for c in active],
        )
