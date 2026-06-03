"""Bit-stable regression test for the Square-Root UKF.

The linear-Gaussian tests in :mod:`tests.test_ukf` verify the UKF
against the closed-form Kalman filter — strong correctness guarantee,
but they collapse to a linear update where the sigma-point spread
contributes nothing nonlinear. That means linalg refactorings that
*do* change byte-level results (caching, vectorising sums, reordering
QR inputs) can pass those tests while silently shifting trajectories
elsewhere.

This module exercises the unscented transform in nonlinear conditions
with a fixed-seed mock process whose ``step`` mixes an elementwise
quadratic with a small linear coupling. The golden values below were
recorded from commit ``f9baf76`` (before any of the 2026-06 SR-UKF
performance optimisations). Any change that shifts a recorded number
beyond ``atol=1e-12`` is treated as a behavioural regression and must
be justified separately.

The optimisation work documented in :doc:`/changes/ukf-perf-2026-06`
keeps every value in this file unchanged.
"""

from __future__ import annotations

import numpy as np

from pyadm1ode_estimation.estimation import StateChannel, StateVectorSpec
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter
from pyadm1ode_estimation.estimation.observation_model import (
    ObservationChannel,
    ObservationModel,
)


class NonlinearMockProcess:
    """Deterministic nonlinear toy: ``x' = A x + 0.1 * x**2`` per call.

    Mixes an elementwise quadratic (so sigma points do *not* collapse to
    the mean under the unscented transform) with a small linear coupling
    via ``A``. No plant side effects — the snapshot/restore hooks are
    no-ops so the regression is sensitive only to the UKF maths.
    """

    def __init__(self, spec: StateVectorSpec, A: np.ndarray):
        self.spec = spec
        self.A = np.asarray(A, dtype=float)
        self.plant = object()  # ducked

    def step(self, x: np.ndarray, dt: float) -> np.ndarray:  # noqa: ARG002
        x = np.asarray(x, dtype=float)
        return self.A @ x + 0.1 * x * x

    def refresh_outputs(
        self, x: np.ndarray, equilibration_dt: float = 1.0 / 24.0
    ) -> None:  # noqa: ARG002
        return None

    def snapshot(self) -> None:
        return None

    def restore(self) -> None:
        return None


def _build_regression_ukf() -> UnscentedKalmanFilter:
    """Fixed-dimension, fixed-coupling UKF used as a behavioural baseline."""
    n = 4
    channels = [
        StateChannel(
            name=f"x{i}",
            kind="adm1",
            adm1_index=i,
            initial=float(i + 1) * 0.5,
            initial_std=0.4,
            process_noise_std=0.05,
            lower=-1e3,
            upper=1e3,
        )
        for i in range(n)
    ]
    spec = StateVectorSpec(digester_id="d", channels=channels)

    # Diagonal-dominant A so the linearised dynamics stay bounded for
    # this many steps; small off-diagonals couple the states.
    A = 0.9 * np.eye(n) + 0.05 * np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    process = NonlinearMockProcess(spec, A)

    def make_extractor(i):
        # Mildly nonlinear observation: y_i = x_i + 0.05 * x_i**2.
        return lambda plant, x, i=i: float(x[i] + 0.05 * x[i] ** 2)

    obs = ObservationModel(
        channels=[
            ObservationChannel(
                name=f"y{i}", extractor=make_extractor(i), noise_std=0.15
            )
            for i in range(3)  # observe 3 of 4 states
        ]
    )
    return UnscentedKalmanFilter(process, obs, spec)


def _run_regression_trajectory():
    """Step the regression UKF 20 times with a deterministic measurement
    schedule and return the final ``(x_hat, S)``."""
    ukf = _build_regression_ukf()
    rng = np.random.default_rng(2026_06_03)

    n_steps = 20
    for k in range(n_steps):
        ukf.predict(dt=0.5)
        # Synthetic measurements: drift around the prior with bounded noise.
        y = {
            "y0": float(0.6 + 0.02 * k + 0.1 * rng.standard_normal()),
            "y1": float(1.1 - 0.01 * k + 0.1 * rng.standard_normal()),
            "y2": float(1.6 + 0.005 * k + 0.1 * rng.standard_normal()),
        }
        ukf.update(y, t=float(k))
    return ukf.x_hat.copy(), ukf.S.copy()


# ---- Golden reference --------------------------------------------------
# Recorded from the canonical Wan-VdM 2001 SR-UKF (Algorithm 3.1, line 22:
# sigma-point reuse) after the 2026-06 main-path migration. Any update to
# these constants must be paired with a documented change to the SR-UKF
# maths — not a linalg refactor.
#
# Historical note: an earlier set of goldens was recorded from a custom
# "redraw" variant that re-evaluated h at a fresh sigma set around
# (x_pred, S_pred). That redraw form is not the canonical Wan-VdM
# algorithm and was retired during the main-path migration (see
# :doc:`/development/ukf_performance`).
#
# Reproduce with::
#
#     >>> x, S = _run_regression_trajectory()
#     >>> np.set_printoptions(precision=16); print(x); print(S)
GOLDEN_X_HAT = np.array(
    [
        0.8971088747295345,
        1.1143629772331338,
        1.6942747466467123,
        0.0462084324106832,
    ]
)
GOLDEN_S = np.array(
    [
        [0.0828302657340643, 0.0, 0.0, 0.0],
        [0.0052221224317974, 0.0848124610114546, 0.0, 0.0],
        [0.0002674703360287, 0.0045829701418287, 0.0921499863170435, 0.0],
        [
            0.009424849244041,
            -0.0012545587586951,
            0.0107028856329831,
            0.1013075622875361,
        ],
    ]
)


def test_regression_trajectory_is_bit_stable():
    """The full 20-step posterior must match the recorded baseline.

    Tolerance is set at ``atol=1e-12``: well below any numerically
    meaningful filter difference, but loose enough to survive the
    last-bit reordering that vectorised sums or QR re-stackings can
    introduce on different BLAS builds.
    """
    x_hat, S = _run_regression_trajectory()
    np.testing.assert_allclose(x_hat, GOLDEN_X_HAT, atol=1e-12, rtol=0)
    np.testing.assert_allclose(S, GOLDEN_S, atol=1e-12, rtol=0)


# --------------------------------------------------------------------------
# Cache behaviour
# --------------------------------------------------------------------------
# The performance optimisations introduce two single-slot caches:
#
#   * ``_sqrt_Q_cache``  — cholesky(Q(dt)), keyed on dt.
#   * ``_sqrt_R_cache``  — cholesky(R(active)), keyed on the tuple of
#                          active channel names.
#
# The trajectory test above pins behaviour under constant dt and constant
# active set (= permanent cache hits after the first step). These tests
# pin the *invalidation* paths: the cache must recompute when its key
# changes, and must not recompute otherwise.


class _CountingSpec:
    """Wraps a real ``StateVectorSpec`` and counts ``process_noise_cov`` calls."""

    def __init__(self, inner):
        self._inner = inner
        self.process_noise_cov_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def __len__(self):
        return len(self._inner)

    def process_noise_cov(self, dt):
        self.process_noise_cov_calls += 1
        return self._inner.process_noise_cov(dt)


class _CountingObs:
    """Wraps a real ``ObservationModel`` and counts ``R`` calls."""

    def __init__(self, inner):
        self._inner = inner
        self.R_calls = 0

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def R(self, active=None):
        self.R_calls += 1
        return self._inner.R(active=active)


def test_sqrt_Q_cache_hits_under_constant_dt():
    """predict() with the same dt must call process_noise_cov exactly once."""
    ukf = _build_regression_ukf()
    counting = _CountingSpec(ukf.spec)
    ukf.spec = counting

    for _ in range(5):
        ukf.predict(dt=0.5)

    # First predict fills the cache; subsequent calls hit it.
    assert counting.process_noise_cov_calls == 1


def test_sqrt_Q_cache_invalidates_on_dt_change():
    """A different dt must trigger a fresh process_noise_cov computation."""
    ukf = _build_regression_ukf()
    counting = _CountingSpec(ukf.spec)
    ukf.spec = counting

    ukf.predict(dt=0.5)
    ukf.predict(dt=0.5)  # cache hit
    ukf.predict(dt=1.0)  # cache miss — different dt
    ukf.predict(dt=1.0)  # cache hit again

    assert counting.process_noise_cov_calls == 2


def test_sqrt_R_cache_hits_under_constant_active_set():
    """update() with the same active channels must call obs.R exactly once."""
    ukf = _build_regression_ukf()
    counting = _CountingObs(ukf.obs)
    ukf.obs = counting

    # Same three observation channels every step → identical active key.
    y = {"y0": 0.5, "y1": 1.0, "y2": 1.5}
    for k in range(5):
        ukf.predict(dt=0.5)
        ukf.update(y, t=float(k))

    assert counting.R_calls == 1


def test_sqrt_R_cache_invalidates_on_active_set_change():
    """Dropping one of the measured channels must refresh the R cache."""
    ukf = _build_regression_ukf()
    counting = _CountingObs(ukf.obs)
    ukf.obs = counting

    ukf.predict(dt=0.5)
    ukf.update({"y0": 0.5, "y1": 1.0, "y2": 1.5}, t=0.0)  # active = (y0, y1, y2)
    ukf.predict(dt=0.5)
    ukf.update({"y0": 0.5, "y1": 1.0, "y2": 1.5}, t=1.0)  # cache hit
    ukf.predict(dt=0.5)
    ukf.update({"y0": 0.5, "y1": 1.0}, t=2.0)  # active = (y0, y1) — cache miss

    assert counting.R_calls == 2
