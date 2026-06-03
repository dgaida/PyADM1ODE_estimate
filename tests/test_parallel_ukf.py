"""Integration tests for ``ParallelUKF``.

The pool uses ``multiprocessing.get_context("spawn")`` so that behaviour
is identical on Linux and Windows (and so the components-builder must
be picklable — see the module docstring of
``pyadm1ode_estimation.estimation.filters.parallel_ukf``).

These tests use the picklable mock plant from :mod:`_mock_plant`, not
``pyadm1.example_plants.build_simple_plant``. Reason: pip-installed
pyadm1 in CI ships without the substrate YAML catalog, which would
make every test below fail with ``FileNotFoundError``. The
mock satisfies the minimal ``ADM1ProcessModel`` contract — that's
enough to verify the parallelisation infrastructure (the test
subject); the integration with the real pyadm1 dynamics is covered
by the ``_bench_variants.py`` script for local benchmarking.
"""

from __future__ import annotations

import numpy as np
import pytest

# ParallelUKF itself is stdlib + numpy, importable without pyadm1.
# The mock plant builder we use below also avoids pyadm1's substrate
# library, so this test module runs cleanly in CI.
from _mock_plant import build_mock_components  # noqa: E402

from pyadm1ode_estimation.estimation.filters import (  # noqa: E402
    ParallelUKF,
    UnscentedKalmanFilter,
)


def _run_filter(ukf, n_steps=3, dt=1.0 / 24.0, rng_seed=42):
    """Step the filter through a short synthetic measurement sequence.

    Returns the ``(x_hat, S)`` trajectory so callers can diff two runs.
    The measurements themselves don't need to be realistic — they just
    need to be IDENTICAL across the serial and parallel filters so the
    diff is meaningful.
    """
    rng = np.random.default_rng(rng_seed)
    n_obs = len(ukf.obs.channels)
    trajectory = []
    for k in range(n_steps):
        ukf.predict(dt=dt)
        y_pred = np.array(
            [c.extractor(ukf.process.plant, ukf.x_hat) for c in ukf.obs.channels]
        )
        y_noisy = y_pred + 0.05 * rng.standard_normal(n_obs)
        y_dict = {c.name: float(y_noisy[i]) for i, c in enumerate(ukf.obs.channels)}
        ukf.update(y_dict, t=float(k))
        trajectory.append((ukf.x_hat.copy(), ukf.S.copy()))
    return trajectory


@pytest.mark.slow
def test_parallel_matches_serial():
    """The same predict/update sequence run serially and with two workers
    must produce numerically identical trajectories.

    The mock plant dynamics is deterministic; the only realistic source
    of divergence would be the parallel dispatch reordering accumulated
    numpy sums. With ``starmap`` preserving task order this should be
    byte-identical, but we accept ``atol=1e-9`` to cover BLAS reductions
    on the QR path.
    """
    proc_a, obs_a, spec_a = build_mock_components()
    serial = UnscentedKalmanFilter(proc_a, obs_a, spec_a)

    proc_b, obs_b, spec_b = build_mock_components()
    parallel = ParallelUKF(
        proc_b,
        obs_b,
        spec_b,
        n_workers=2,
        components_builder=build_mock_components,
    )
    try:
        traj_serial = _run_filter(serial)
        traj_parallel = _run_filter(parallel)
    finally:
        parallel.shutdown()

    assert len(traj_serial) == len(traj_parallel)
    for k, ((x_s, S_s), (x_p, S_p)) in enumerate(zip(traj_serial, traj_parallel)):
        np.testing.assert_allclose(
            x_p,
            x_s,
            atol=1e-9,
            rtol=0,
            err_msg=f"x_hat mismatch at step {k}",
        )
        np.testing.assert_allclose(
            S_p,
            S_s,
            atol=1e-9,
            rtol=0,
            err_msg=f"S mismatch at step {k}",
        )


def test_n_workers_one_falls_through_to_serial():
    """``n_workers=1`` must NOT spawn a pool — it falls through to the
    serial path with zero startup overhead. The hook override is the
    cheapest way to verify this: ``_pool`` stays ``None``.
    """
    proc, obs, spec = build_mock_components()
    ukf = ParallelUKF(
        proc,
        obs,
        spec,
        n_workers=1,
        components_builder=build_mock_components,
    )
    assert ukf._pool is None
    # The base-class propagation path is reachable through the override.
    ukf.predict(dt=1.0 / 24.0)
    assert ukf._cached_h_all is not None


def test_missing_builder_with_multiple_workers_raises():
    """Asking for >1 workers without a builder is a configuration bug —
    fail loud at construction, not at the first predict()."""
    proc, obs, spec = build_mock_components()
    with pytest.raises(ValueError, match="components_builder"):
        ParallelUKF(proc, obs, spec, n_workers=4, components_builder=None)


def test_shutdown_is_idempotent():
    proc, obs, spec = build_mock_components()
    ukf = ParallelUKF(
        proc,
        obs,
        spec,
        n_workers=2,
        components_builder=build_mock_components,
    )
    ukf.shutdown()
    assert ukf._pool is None
    # Second call must not raise.
    ukf.shutdown()
