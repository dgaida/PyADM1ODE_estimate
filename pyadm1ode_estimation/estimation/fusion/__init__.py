"""Fusion algorithms (AP 4.4): the PINN↔UKF hybrid via covariance intersection."""

from .hybrid import HybridEstimator, fuse_ci_diagonal, fuse_trajectories_ci

__all__ = [
    "HybridEstimator",
    "fuse_ci_diagonal",
    "fuse_trajectories_ci",
]
