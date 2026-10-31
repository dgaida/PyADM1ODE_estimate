"""The estimator registry: one interface over UKF, PINN v2 and the GRU observer.

The filters are expensive (a 30-day warm-up per series), so the UKF paths are
exercised on construction and wiring only; the observer path trains for a
couple of epochs on a handful of series.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1ode_estimation.estimation.base import TrajectoryEstimate
from pyadm1ode_estimation.estimation.estimators import (
    ESTIMATOR_NAMES,
    ObserverEstimator,
    PinnEstimator,
    build_estimator,
    reference_theta,
    save_runs,
    score,
)
from pyadm1ode_estimation.estimation.filter_tuning.datasets import (
    DEFAULT_BENCHMARK_DIR,
    Series,
    load_benchmark,
)

needs_benchmark = pytest.mark.skipif(
    not (DEFAULT_BENCHMARK_DIR / "train.npz").exists(),
    reason="benchmark dataset not present",
)


def test_score_is_the_median_nrmse_after_burnin():
    rng = np.random.default_rng(0)
    truth = rng.uniform(1.0, 2.0, size=(100, 5))
    pred = truth.copy()
    pred[:, 0] *= 1.1  # one state 10 % off
    pred[:10] = 0.0  # garbage before the burn-in must not count
    s = Series(
        np.zeros((100, 1)), np.zeros((100, 1)), np.arange(100.0), truth, np.zeros(0)
    )
    traj = TrajectoryEstimate(np.arange(100.0), pred, np.zeros_like(pred))
    assert score(traj, s, burnin_steps=10) == pytest.approx(0.0)
    pred[:, :3] *= 1.1
    assert 0.09 < score(traj, s, burnin_steps=10) < 0.11


def test_unknown_name_is_rejected():
    with pytest.raises(KeyError):
        build_estimator("kalman", None)
    assert set(ESTIMATOR_NAMES) == {"ukf", "pinn", "observer"}


def test_reference_theta_falls_back_to_untuned(tmp_path, monkeypatch):
    import pyadm1ode_estimation.estimation.estimators as mod

    monkeypatch.setattr(mod, "REFERENCE_DIR", tmp_path)
    with pytest.warns(UserWarning):
        theta, fe = reference_theta("full")
    assert theta == {} and fe is None


@needs_benchmark
def test_ukf_estimator_is_wired_with_the_reference_theta():
    ds = load_benchmark(splits=("test",))
    est = build_estimator("ukf", ds)
    assert est.name == "ukf" and est.runner.variant == "full"
    assert isinstance(est.theta, dict)
    core = build_estimator("ukf", ds, variant="adcore")
    assert core.runner.core is not None and len(core.runner.core) < 41


@needs_benchmark
def test_pinn_estimator_builds_with_the_study_defaults():
    ds = load_benchmark(splits=("test",))
    est = PinnEstimator(ds)
    assert est.arch.fourier_features == 4 and est.cfg.data_source == "measurements"
    assert est.roll.cadence_hours == 24.0 and est.roll.window_days == 10.0
    assert est.bridge == "hold" and len(est._prior[0]) == 31


@pytest.mark.slow
@needs_benchmark
def test_observer_estimator_train_save_load_run(tmp_path):
    from pyadm1ode_estimation.estimation.filter_tuning.datasets import get_dataset
    from pyadm1ode_estimation.estimation.observer import PinnData

    ds = load_benchmark()
    data = PinnData.build(ds, val_frac=0.2, seed=0, per_group_train=1, per_group_val=1)
    est, res = ObserverEstimator.train(
        data, epochs=2, patience=None, noise_aug=1.0, n_random=2, hidden=8
    )
    assert est.recipe["noise_aug"] == 1.0 and res.best_epoch in (0, 1)
    traj = est.run(ds.test[0])
    assert traj.x_hat.shape == (len(ds.test[0].time), 41)
    assert np.isfinite(score(traj, ds.test[0]))
    path = est.save(tmp_path / "obs.pt")
    again = build_estimator("observer", ds, checkpoint=path)
    np.testing.assert_allclose(again.run(ds.test[0]).x_hat, traj.x_hat)
    # the same estimator scores a variant dataset with the benchmark statistics
    hard = DEFAULT_BENCHMARK_DIR.parent / "benchmark_hard"
    if hard.exists():
        h = get_dataset(hard)
        assert np.isfinite(score(again.run(h.test[0]), h.test[0]))
    out = save_runs(
        tmp_path / "run.npz",
        [{"index": 0, "label": "x", "score": 0.1, "traj": traj}],
        "observer",
    )
    z = np.load(out, allow_pickle=True)
    assert str(z["estimator"]) == "observer" and z["x_hat"][0].shape == traj.x_hat.shape


@needs_benchmark
def test_organic_load_feature_is_substrate_aware():
    from pyadm1ode_estimation.estimation.observer.adapter import (
        ORGANIC_LOAD_GROUPS,
        FeatureSpec,
        organic_load_features,
    )

    ds = load_benchmark(splits=("test",))
    s = ds.test[0]
    spec = FeatureSpec(organic_load=True, use_feed=False)
    x, names = spec.build(s, {"meta": ds.meta, "sensor_names": ds.meta["sensors"]})
    n_sensor = len(ds.meta["sensors"])
    assert x.shape == (len(s.time), n_sensor + len(ORGANIC_LOAD_GROUPS))
    assert names[n_sensor:] == [f"load_{g}" for g in ORGANIC_LOAD_GROUPS]
    assert np.all(np.isfinite(x)) and np.all(x[:, n_sensor:] >= 0.0)
    ccm_dir = DEFAULT_BENCHMARK_DIR.parent / "benchmark_ccm"
    if ccm_dir.exists():
        import json

        ccm_meta = json.loads((ccm_dir / "meta.json").read_text(encoding="utf-8"))
        same_flows = s.feed[:5]
        assert not np.allclose(
            organic_load_features(same_flows, ds.meta),
            organic_load_features(same_flows, ccm_meta),
        )


@needs_benchmark
def test_observer_for_dataset_swaps_only_the_meta():
    from pyadm1ode_estimation.estimation.observer import FeatureSpec

    ds = load_benchmark(splits=("test",))
    est = ObserverEstimator(
        net=None,
        feature_spec=FeatureSpec(organic_load=True),
        feat_mean=np.zeros(3),
        feat_std=np.ones(3),
        ctx={"meta": {"x": 1}, "sensor_names": ["a"], "dt_hours": 1.0},
    )
    other = est.for_dataset(ds)
    assert other.ctx["meta"] is ds.meta and other.ctx["sensor_names"] == ["a"]
    assert est.ctx["meta"] == {"x": 1}


@pytest.mark.slow
@needs_benchmark
def test_kinetics_aware_observer_needs_explicit_kinetics(tmp_path):
    from pyadm1ode_estimation.estimation.observer import FeatureSpec, PinnData

    ds = load_benchmark()
    keys = ("k_m_ac", "k_dis_PS")
    data = PinnData.build(
        ds,
        val_frac=0.2,
        seed=0,
        per_group_train=1,
        per_group_val=1,
        feature_spec=FeatureSpec(kinetic_keys=keys),
    )
    assert data.feature_names[-2:] == ["logk_k_m_ac", "logk_k_dis_PS"]
    est, _ = ObserverEstimator.train(
        data, epochs=2, patience=None, hidden=8, kinetic_noise=0.07
    )
    assert est.kinetic_keys == keys and est.recipe["kinetic_noise"] == 0.07
    s = ds.test[0]
    with pytest.raises(ValueError, match="kinetic inputs"):
        est.run(s)  # the generator truth on the series must never be used silently
    nominal = est.run(s, kinetics={}).x_hat
    slow_ac = est.run(s, kinetics={"k_m_ac": 0.5}).x_hat
    assert nominal.shape == (len(s.time), 41) and not np.allclose(nominal, slow_ac)
    again = ObserverEstimator.load(est.save(tmp_path / "kin.pt"), ds)
    assert again.kinetic_keys == keys
    np.testing.assert_allclose(again.run(s, kinetics={"k_m_ac": 0.5}).x_hat, slow_ac)
    # a kinetics-blind observer ignores the argument
    blind, _ = ObserverEstimator.train(
        PinnData.build(ds, val_frac=0.2, seed=0, per_group_train=1, per_group_val=1),
        epochs=1,
        patience=None,
        hidden=8,
    )
    np.testing.assert_allclose(
        blind.run(s).x_hat, blind.run(s, kinetics={"k_m_ac": 0.5}).x_hat
    )
