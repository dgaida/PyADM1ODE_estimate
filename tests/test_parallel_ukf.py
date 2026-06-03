"""Integration tests for ``ParallelUKF``.

The pool uses ``multiprocessing.get_context("spawn")`` so that behaviour
is identical on Linux and Windows (and so the components-builder must
be picklable — see the module docstring of
``pyadm1ode_estimation.estimation.filters.parallel_ukf``).

These tests start an actual worker pool and therefore (a) take a few
seconds to run, and (b) require the ``pyadm1`` plant builder to be
importable. The whole module is skipped if either prerequisite is
absent.
"""

from __future__ import annotations


import numpy as np
import pytest

# Skip the whole module if pyadm1 / the example plant isn't installed
# in this environment. ParallelUKF itself imports fine without pyadm1
# (its dependencies are stdlib + numpy), but starting workers needs the
# plant builder to succeed in each child.
pytest.importorskip("pyadm1")
pytest.importorskip("pyadm1ode_estimation.example_plants")

from pyadm1ode_estimation.estimation import (
    InputSpec,
    build_filter_components,
)
from pyadm1ode_estimation.estimation.filters import (
    ParallelUKF,
    UnscentedKalmanFilter,
)
from pyadm1ode_estimation.example_plants import build_simple_plant


# Top-level builder so multiprocessing.spawn can pickle a reference to
# it and re-import this module in each worker.
def _build_components_for_test():
    plant = build_simple_plant()
    return build_filter_components(
        plant,
        digester_id="fermenter",
        substrates=[
            InputSpec("maize_silage", substrate_index=0, initial_flow=10.0),
            InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
        ],
        sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
    )


def _run_filter(ukf, n_steps=3, dt=1.0 / 24.0, rng_seed=42):
    rng = np.random.default_rng(rng_seed)
    n_obs = len(ukf.obs.channels)
    trajectory = []
    for k in range(n_steps):
        ukf.predict(dt=dt)
        # Synthetic measurement: tiny perturbation around the predicted h.
        # The exact values don't matter — we just need the same measurements
        # across serial and parallel runs for an apples-to-apples diff.
        y_pred = np.array(
            [c.extractor(ukf.process.plant, ukf.x_hat) for c in ukf.obs.channels]
        )
        y_noisy = y_pred + 0.05 * rng.standard_normal(n_obs)
        y_dict = {c.name: float(y_noisy[i]) for i, c in enumerate(ukf.obs.channels)}
        ukf.update(y_dict, t=float(k))
        trajectory.append((ukf.x_hat.copy(), ukf.S.copy()))
    return trajectory


@pytest.mark.slow
def test_parallel_matches_serial_on_simple_plant():
    """The same predict/update sequence run serially and with two workers
    must produce numerically identical trajectories.

    ODE integration is deterministic given identical inputs; the only
    realistic source of divergence would be the parallel dispatch
    reordering accumulated numpy sums. With ``starmap`` preserving task
    order this should be byte-identical, but we accept ``atol=1e-9`` to
    cover BLAS reductions on the QR path.
    """
    proc_a, obs_a, spec_a = _build_components_for_test()
    serial = UnscentedKalmanFilter(proc_a, obs_a, spec_a)

    proc_b, obs_b, spec_b = _build_components_for_test()
    parallel = ParallelUKF(
        proc_b,
        obs_b,
        spec_b,
        n_workers=2,
        components_builder=_build_components_for_test,
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
    proc, obs, spec = _build_components_for_test()
    ukf = ParallelUKF(
        proc,
        obs,
        spec,
        n_workers=1,
        components_builder=_build_components_for_test,
    )
    assert ukf._pool is None
    # The base-class propagation path is reachable through the override.
    ukf.predict(dt=1.0 / 24.0)
    assert ukf._cached_h_all is not None


def test_missing_builder_with_multiple_workers_raises():
    """Asking for >1 workers without a builder is a configuration bug —
    fail loud at construction, not at the first predict()."""
    proc, obs, spec = _build_components_for_test()
    with pytest.raises(ValueError, match="components_builder"):
        ParallelUKF(proc, obs, spec, n_workers=4, components_builder=None)


def test_shutdown_is_idempotent():
    proc, obs, spec = _build_components_for_test()
    ukf = ParallelUKF(
        proc,
        obs,
        spec,
        n_workers=2,
        components_builder=_build_components_for_test,
    )
    ukf.shutdown()
    assert ukf._pool is None
    # Second call must not raise.
    ukf.shutdown()
