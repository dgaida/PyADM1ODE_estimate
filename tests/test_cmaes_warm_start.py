"""Warm start of the CMA-ES noise search (no filter runs, no ``cma`` needed)."""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1ode_estimation.estimation.filter_tuning import cmaes_search as cs

BLOCKS = {"a": [0, 1], "b": [2], "c": [3, 4, 5]}


def _q(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).uniform(1e-6, 1e-2, size=6)


def test_x_from_theta_inverts_theta_from_x():
    q_emp = _q()
    x = np.array([0.7, -1.2, 2.5, 0.3, -0.9])
    theta = cs.theta_from_x(x, q_emp, list(BLOCKS), BLOCKS)
    back = cs.x_from_theta(theta, q_emp, list(BLOCKS), BLOCKS)
    assert back == pytest.approx(x, abs=1e-12)


def test_x_from_theta_carries_block_corrections_to_a_new_q_shape():
    """Against another empirical Q the result is the median log ratio per block."""
    q_old, q_new = _q(0), _q(1)
    x = np.array([1.0, -1.0, 0.5, 0.0, 0.0])
    theta = cs.theta_from_x(x, q_old, list(BLOCKS), BLOCKS)
    back = cs.x_from_theta(theta, q_new, list(BLOCKS), BLOCKS)
    for j, (b, idx) in enumerate(BLOCKS.items()):
        expected = np.median(np.log(theta["q_diag"][idx] / q_new[idx]))
        assert back[j] == pytest.approx(expected)


def test_x_from_theta_defaults_r_and_p0_to_untouched():
    q_emp = _q()
    back = cs.x_from_theta({"q_diag": q_emp * 2.0}, q_emp, list(BLOCKS), BLOCKS)
    assert back[:3] == pytest.approx([np.log(2.0)] * 3)
    assert back[-2:] == pytest.approx([0.0, 0.0])


def test_initial_mean_defaults_to_zero_and_clips_inside_bounds():
    assert cs.initial_mean(None, 4) == [0.0] * 4
    lo, hi = cs.DEFAULT_BOUNDS
    m = cs.initial_mean([lo - 1.0, 0.5, hi, hi + 3.0], 4)
    assert m[1] == 0.5
    assert lo < m[0] < lo + 0.01 and hi - 0.01 < m[2] < hi and m[3] == m[2]


def test_initial_mean_rejects_wrong_length():
    with pytest.raises(ValueError, match="x0 must have 4"):
        cs.initial_mean([0.0, 0.0], 4)
