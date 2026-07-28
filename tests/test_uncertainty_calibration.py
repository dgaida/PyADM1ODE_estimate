"""Validate the data-agnostic uncertainty calibrator on a synthetic filter.

A fake estimator produces point estimates with a *known* per-state error std and a
predicted σ that scales with the tunable ``q_scale``. Covariance matching must drive
each state's NEES to ~1 and the ±2σ coverage to ~95 %, regardless of how mis-scaled
the initial σ is — and must give the hard states a larger σ than the easy ones.
"""

import numpy as np
import pytest

from pyadm1ode_estimation.estimation.calibration import (
    TWO_SIGMA_COVERAGE,
    Episode,
    UncertaintyCalibrator,
    compute_report,
    hardest_states,
    sigma_scale_from_report,
)

N_STATE = 6
SIGMA_ACTUAL = np.array([0.05, 0.05, 0.4, 0.4, 2.0, 2.0])  # easy ... hard states
BASE_STD = 0.3  # deliberately mis-calibrated


def _make_episodes(n_ep=3, T=400, seed=0):
    rng = np.random.default_rng(seed)
    eps = []
    t = np.linspace(0, 4 * np.pi, T)
    for e in range(n_ep):
        truth = np.stack([(i + 1) * np.sin(t + i) for i in range(N_STATE)], axis=1)
        err = rng.normal(0.0, 1.0, (T, N_STATE)) * SIGMA_ACTUAL  # fixed per episode
        ep = Episode(obs=None, truth=truth, dt_hours=1.0, name=f"ep{e}")
        ep._err = err  # cache the fixed error
        eps.append(ep)
    return eps


def _run_episode(theta, ep):
    """Fake filter: x_hat = truth + fixed error; σ scales as √(q_scale) (var ∝ Q)."""
    qs = theta["q_scale"]
    std = np.array([BASE_STD * np.sqrt(qs[f"s{i}"]) for i in range(N_STATE)])
    x_hat = ep.truth + ep._err
    return x_hat, np.broadcast_to(std, ep.truth.shape).copy()


def _calibrator(seed=0):
    eps = _make_episodes(seed=seed)
    q_groups = {f"s{i}": [i] for i in range(N_STATE)}  # one group per state
    return UncertaintyCalibrator(eps, _run_episode, q_groups)


def test_covariance_matching_calibrates_nees_and_coverage():
    cal = _calibrator()
    init = cal.evaluate(cal.theta0)
    _theta, rep, history = cal.calibrate(iters=8, damping=1.0, verbose=False)

    # NEES driven to ~1 for every state (informative here) ...
    assert np.all(rep.nees > 0.6) and np.all(rep.nees < 1.6), rep.nees
    # ... so overall 2σ coverage lands near the Gaussian target.
    assert abs(rep.overall_coverage_2sigma - TWO_SIGMA_COVERAGE) < 0.03
    # and calibration improved vs the mis-scaled start.
    assert rep.objective < init.objective
    assert history[-1].mean_log_nees_abs < history[0].mean_log_nees_abs


def test_hard_states_get_larger_sigma():
    cal = _calibrator()
    theta, rep, _ = cal.calibrate(iters=8, damping=1.0, verbose=False)
    # recovered σ must order like the true difficulty (hard states -> larger σ)
    assert rep.mean_std[4] > rep.mean_std[2] > rep.mean_std[0]
    # and roughly recover the actual error magnitudes
    recovered = (
        np.array([theta["q_scale"][f"s{i}"] for i in range(N_STATE)]) ** 0.5 * BASE_STD
    )
    assert np.allclose(recovered, SIGMA_ACTUAL, rtol=0.35)


def test_hardest_states_reporting():
    cal = _calibrator()
    _, rep, _ = cal.calibrate(iters=6, damping=1.0, verbose=False)
    names = [f"x{i}" for i in range(N_STATE)]
    worst = hardest_states(rep, names, k=2)
    assert worst[0][0] in ("x4", "x5")  # the largest-error states


def test_posthoc_sigma_recovers_actual_error_and_coverage():
    cal = _calibrator()
    results = cal.collect(cal.theta0)  # base filter (q_scale = 1)
    rep = compute_report(results)
    scale = sigma_scale_from_report(rep, lo=0.02, hi=100.0)
    # σ' = √(NEES)·σ recovers the true per-state error magnitude ...
    recovered = BASE_STD * scale
    assert np.allclose(recovered, SIGMA_ACTUAL, rtol=0.25), recovered
    # ... and the post-scaled ±2σ band hits the Gaussian coverage target.
    tru = np.concatenate([r[0] for r in results])
    xh = np.concatenate([r[1] for r in results])
    sd = np.concatenate([r[2] for r in results])
    cov = float(np.mean(np.abs(tru - xh) <= 2 * sd * scale[None, :]))
    assert abs(cov - TWO_SIGMA_COVERAGE) < 0.03


def test_parallel_map_matches_serial():
    from multiprocessing.dummy import Pool  # threads: no pickling needed

    serial = _calibrator()
    _, r_serial, _ = serial.calibrate(iters=4, damping=1.0, verbose=False)
    par = _calibrator()
    with Pool(3) as pool:
        par.map_fn = pool.map
        _, r_par, _ = par.calibrate(iters=4, damping=1.0, verbose=False)
    assert abs(r_serial.overall_coverage_2sigma - r_par.overall_coverage_2sigma) < 1e-9


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
