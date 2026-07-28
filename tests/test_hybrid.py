"""Tests for the Variant-C PINN↔UKF hybrid (covariance-intersection fusion)."""

import numpy as np
import pytest

from pyadm1ode_estimation.estimation.base import TrajectoryEstimate
from pyadm1ode_estimation.estimation.fusion import (
    HybridEstimator,
    fuse_ci_diagonal,
    fuse_trajectories_ci,
)


def _traj(mean, std, T=5):
    mean = np.tile(np.asarray(mean, float), (T, 1))
    std = np.tile(np.asarray(std, float), (T, 1))
    return TrajectoryEstimate(time=np.arange(T, dtype=float), x_hat=mean, std=std)


def test_agreement_preserved():
    """Fusing two estimates with the same mean returns that mean."""
    m, _sf, _ = fuse_ci_diagonal(
        np.array([[2.0, 5.0]]),
        np.array([[0.3, 1.0]]),
        np.array([[2.0, 5.0]]),
        np.array([[0.7, 0.4]]),
    )
    assert np.allclose(m, [[2.0, 5.0]])


def test_confident_estimate_dominates():
    """A far more confident estimate pulls the fused mean toward itself."""
    m, sf, _w = fuse_ci_diagonal(
        np.array([[0.0]]),
        np.array([[0.01]]),  # A: tight
        np.array([[10.0]]),
        np.array([[5.0]]),  # B: loose
    )
    assert m[0, 0] < 0.1  # fused ≈ A
    assert sf[0, 0] < 0.02  # fused std ≈ A's


def test_fused_std_between_inputs():
    rng = np.random.default_rng(0)
    ma, sa = rng.normal(size=(4, 6)), np.abs(rng.normal(size=(4, 6))) + 0.1
    mb, sb = rng.normal(size=(4, 6)), np.abs(rng.normal(size=(4, 6))) + 0.1
    _, sf, _ = fuse_ci_diagonal(ma, sa, mb, sb)
    lo, hi = np.minimum(sa, sb), np.maximum(sa, sb)
    assert np.all(sf <= hi + 1e-9) and np.all(sf >= lo - 1e-9)


def test_ci_minimises_determinant():
    """The optimised fused covariance det is ≤ both inputs' det (CI criterion)."""
    rng = np.random.default_rng(1)
    sa = np.abs(rng.normal(size=(3, 5))) + 0.2
    sb = np.abs(rng.normal(size=(3, 5))) + 0.2
    _, sf, _ = fuse_ci_diagonal(
        rng.normal(size=(3, 5)), sa, rng.normal(size=(3, 5)), sb
    )
    det_a = (sa**2).prod(axis=1)
    det_b = (sb**2).prod(axis=1)
    det_f = (sf**2).prod(axis=1)
    assert np.all(det_f <= np.minimum(det_a, det_b) + 1e-9)


def test_fixed_omega():
    m, _sf, w = fuse_ci_diagonal(
        np.array([[0.0]]),
        np.array([[1.0]]),
        np.array([[4.0]]),
        np.array([[1.0]]),
        omega=0.5,
    )
    assert np.allclose(w, 0.5)
    assert np.allclose(m, [[2.0]])  # equal variance + ω=0.5 → midpoint


def test_fuse_trajectories_and_hybrid_estimator():
    a = _traj([1.0, 2.0], [0.5, 2.0])
    b = _traj([1.4, 2.2], [2.0, 0.5])
    fused = fuse_trajectories_ci(a, b)
    assert fused.x_hat.shape == (5, 2)
    # state 0: A tighter → fused nearer A(1.0); state 1: B tighter → nearer B(2.2)
    assert abs(fused.x_hat[0, 0] - 1.0) < abs(fused.x_hat[0, 0] - 1.4)
    assert abs(fused.x_hat[0, 1] - 2.2) < abs(fused.x_hat[0, 1] - 2.0)

    est = HybridEstimator(a, b).estimate()
    assert np.allclose(est.x_hat, fused.x_hat)


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError):
        fuse_trajectories_ci(_traj([1.0], [1.0]), _traj([1.0, 2.0], [1.0, 1.0]))
