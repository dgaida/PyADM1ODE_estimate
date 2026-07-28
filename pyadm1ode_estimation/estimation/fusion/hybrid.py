"""Variant C — PINN↔UKF hybrid via covariance-intersection fusion.

Each estimator plays to its strength: the UKF is well-posed and calibrated on
the charge/pH states, the PINN (per-window smoother or amortised observer) is
strong on the biogas-driving states. Covariance Intersection (CI) fuses their
per-state estimates *conservatively* — without assuming the two are
independent — so the combined estimate is at least as informative as either.

This fuses whole trajectories (``TrajectoryEstimate``): at each time step a
single fusion weight ``ω`` is chosen to minimise the fused covariance
determinant (the standard CI criterion), evaluated on the diagonal covariances
built from the per-state standard deviations.

**Caveat:** CI weights by covariance, so it is only as good as the inputs'
*calibration*. An over-confident estimator (e.g. MC-Dropout bands that are too
narrow) will be over-trusted. Calibrating the PINN's uncertainty (the open item
from the twin comparison) is a prerequisite for the fusion to help everywhere.
"""

from __future__ import annotations

import numpy as np

from ..base import TrajectoryEstimate


def fuse_ci_diagonal(
    mean_a: np.ndarray,
    std_a: np.ndarray,
    mean_b: np.ndarray,
    std_b: np.ndarray,
    omega: float | None = None,
    n_grid: int = 21,
    var_floor: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-time-step Covariance Intersection of two diagonal-covariance estimates.

    Args:
        mean_a, std_a, mean_b, std_b: ``(T, n)`` estimate means and per-state
            standard deviations.
        omega: fixed fusion weight in ``[0, 1]``; if ``None`` it is optimised per
            time step (grid search minimising the fused covariance determinant).
        n_grid: grid resolution for the ω search.
        var_floor: added to variances for numerical safety (zero-std inputs).

    Returns:
        ``(mean_fused, std_fused, omega_per_step)`` — the first two shaped
        ``(T, n)``, the last ``(T,)``.
    """
    va = np.asarray(std_a, float) ** 2 + var_floor
    vb = np.asarray(std_b, float) ** 2 + var_floor
    ia, ib = 1.0 / va, 1.0 / vb  # information (inverse variance), (T, n)
    T = mean_a.shape[0]

    if omega is None:
        grid = np.linspace(0.0, 1.0, n_grid)  # (G,)
        # fused information per grid value: (G, T, n); minimise det(cov_fused)
        # == maximise Σ_states log(fused information).
        inv = grid[:, None, None] * ia[None] + (1.0 - grid[:, None, None]) * ib[None]
        logdet_inf = np.log(inv).sum(axis=2)  # (G, T)
        omega_step = grid[np.argmax(logdet_inf, axis=0)]  # (T,)
    else:
        omega_step = np.full(T, float(omega))

    w = omega_step[:, None]  # (T, 1)
    inv_f = w * ia + (1.0 - w) * ib
    v_f = 1.0 / inv_f
    mean_f = v_f * (
        w * ia * np.asarray(mean_a, float) + (1.0 - w) * ib * np.asarray(mean_b, float)
    )
    return mean_f, np.sqrt(v_f), omega_step


def fuse_trajectories_ci(
    traj_a: TrajectoryEstimate,
    traj_b: TrajectoryEstimate,
    omega: float | None = None,
    var_floor: float = 1.0e-12,
) -> TrajectoryEstimate:
    """Covariance-intersection fusion of two trajectory estimates on a shared grid."""
    if traj_a.x_hat.shape != traj_b.x_hat.shape:
        raise ValueError(
            f"trajectory shape mismatch: {traj_a.x_hat.shape} vs {traj_b.x_hat.shape}"
        )
    mean_f, std_f, _ = fuse_ci_diagonal(
        traj_a.x_hat,
        traj_a.std,
        traj_b.x_hat,
        traj_b.std,
        omega=omega,
        var_floor=var_floor,
    )
    return TrajectoryEstimate(
        time=np.asarray(traj_a.time, float), x_hat=mean_f, std=std_f
    )


class HybridEstimator:
    """Variant C: fuses two trajectory estimates (e.g. UKF + PINN) via CI.

    Conforms to the :class:`~pyadm1ode_estimation.estimation.base.BatchEstimator`
    output contract. Build it with the two sub-estimates already computed over
    the same window; :meth:`estimate` returns their CI fusion.
    """

    def __init__(
        self,
        traj_a: TrajectoryEstimate,
        traj_b: TrajectoryEstimate,
        omega: float | None = None,
        var_floor: float = 1.0e-12,
    ):
        self.traj_a = traj_a
        self.traj_b = traj_b
        self.omega = omega
        self.var_floor = var_floor

    def estimate(self, times: np.ndarray | None = None) -> TrajectoryEstimate:
        fused = fuse_trajectories_ci(
            self.traj_a, self.traj_b, omega=self.omega, var_floor=self.var_floor
        )
        if times is not None and not np.array_equal(
            np.asarray(times, float), fused.time
        ):
            raise ValueError(
                "HybridEstimator only supports the sub-estimates' own time grid."
            )
        return fused
