"""Tests for the dataset -> PINN adapter (:mod:`...deep_learning.data_adapter`).

Most tests run on a small synthetic :class:`EstimatorDataset` so they stay fast and
independent of the shipped benchmark; the ones that need the real file are marked
and skipped when it is absent. The synthetic dataset deliberately uses a
*different* shape (3 sensors, 2 substrates, 2 labels) than the benchmark — that is
the flexibility contract this module exists for.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1ode_estimation.estimation.deep_learning.data_adapter import (
    FeatureSpec,
    PinnData,
    SplitRecord,
    parse_noise_spec,
)
from pyadm1ode_estimation.estimation.filter_tuning.datasets import (
    EstimatorDataset,
    Series,
)

N_STATE = 7
N_SENSOR = 3
N_FEED = 2
T = 48


def _series(
    rng: np.random.Generator, label: str, i: int, n_state: int = N_STATE
) -> Series:
    time = np.arange(T, dtype=float) / 24.0
    return Series(
        measurements=rng.normal(10.0, 1.0, size=(T, N_SENSOR)),
        feed=rng.uniform(1.0, 5.0, size=(T, N_FEED)),
        time=time,
        truth=rng.uniform(0.5, 2.0, size=(T, n_state)),
        switch_days=np.array([0.5, 1.2]),
        label=label,
        seed=i,
        aux={"ukf_x_hat": rng.uniform(0.5, 2.0, size=(T, n_state))},
    )


def _dataset(n_per_label: int = 10, n_state: int = N_STATE) -> EstimatorDataset:
    rng = np.random.default_rng(0)
    labels = ["calm", "wild"]
    pool = [_series(rng, lb, i, n_state) for lb in labels for i in range(n_per_label)]
    test = [_series(rng, lb, 100 + i, n_state) for lb in labels for i in range(2)]
    meta = {
        "dt_hours": 1.0,
        "state_size": n_state,
        "sensors": ["Q_gas", "pH", "TS"],
        "sensor_noise": {
            "Q_gas": "relative 3 % (flow meter)",
            "pH": "absolute 0.02 (Memosens)",
            "TS": "absolute 0.2 % (Proline)",
        },
        "state_index_map": {str(i): f"S_{i}" for i in range(n_state)},
    }
    return EstimatorDataset(name="synthetic", meta=meta, pool=pool, test=test)


# --------------------------------------------------------------------------
# Noise-spec parsing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("relative 3 % (literature biogas flow meter)", ("relative", 0.03)),
        ("relative 4 %", ("relative", 0.04)),
        ("absolute 0.02 (Memosens CPS16E)", ("absolute", 0.02)),
        # "%" after an absolute value is a unit (percentage points of TS), not a scale.
        ("absolute 0.2 % (Proline Teqwave)", ("absolute", 0.2)),
        ({"kind": "relative", "value": 0.05}, ("relative", 0.05)),
        (0.017, ("absolute", 0.017)),
    ],
)
def test_parse_noise_spec(spec, expected):
    assert parse_noise_spec(spec) == expected


def test_parse_noise_spec_rejects_garbage():
    with pytest.raises(ValueError, match="cannot parse"):
        parse_noise_spec("about a bit noisy")


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def test_split_is_stratified_grouped_and_disjoint():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    assert d.split.labels["train"] == {"calm": 8, "wild": 8}
    assert d.split.labels["val"] == {"calm": 2, "wild": 2}
    assert not set(d.split.train) & set(d.split.val)
    assert len(d.series("train")) == 16 and len(d.series("val")) == 4


def test_split_matches_the_filter_split_exactly():
    """The whole point of routing through ``split_indices``: same seed, same series."""
    ds = _dataset()
    d = PinnData.build(ds, val_frac=0.2, seed=3)
    tr, va, te = ds.split_indices(val_frac=0.2, seed=3)
    assert (d.split.train, d.split.val, d.split.test) == (tr, va, te)


def test_split_is_reproducible_and_seed_dependent():
    ds = _dataset()
    assert PinnData.build(ds, seed=0).split.val == PinnData.build(ds, seed=0).split.val
    assert PinnData.build(ds, seed=0).split.val != PinnData.build(ds, seed=1).split.val


def test_split_roundtrips_through_a_file(tmp_path):
    ds = _dataset()
    a = PinnData.build(ds, val_frac=0.2, seed=5)
    path = a.save_split(tmp_path / "split.json")
    # A different seed must be ignored when a frozen split file is supplied.
    b = PinnData.build(ds, val_frac=0.4, seed=99, split_file=path)
    assert (b.split.train, b.split.val, b.split.test) == (
        a.split.train,
        a.split.val,
        a.split.test,
    )
    assert json.loads(path.read_text(encoding="utf-8"))["seed"] == 5


def test_split_file_out_of_range_is_rejected(tmp_path):
    rec = SplitRecord(
        dataset="synthetic", val_frac=0.2, seed=0, train=[0, 999], val=[1], test=[0]
    )
    path = tmp_path / "bad.json"
    path.write_text(rec.to_json(), encoding="utf-8")
    with pytest.raises(ValueError, match="outside dataset"):
        PinnData.build(_dataset(), split_file=path)


# --------------------------------------------------------------------------
# Train-only statistics (the leakage contract)
# --------------------------------------------------------------------------
def test_statistics_use_train_only():
    ds = _dataset()
    d = PinnData.build(ds, val_frac=0.2, seed=0)

    train_raw = np.concatenate(
        [d.feature_spec.build(s, d._ctx())[0] for s in d.series("train")], axis=0
    )
    assert np.allclose(d.feat_mean, train_raw.mean(axis=0))

    all_raw = np.concatenate(
        [d.feature_spec.build(s, d._ctx())[0] for s in ds.pool], axis=0
    )
    assert not np.allclose(d.feat_mean, all_raw.mean(axis=0))

    train_truth = np.concatenate([s.truth for s in d.series("train")], axis=0)
    assert np.allclose(d.x_scale, np.sqrt(np.mean(train_truth**2, axis=0)) + 1e-12)
    # x_ref must come from the training rows only.
    assert np.any(np.all(np.isclose(train_truth, d.x_ref), axis=1))


def test_reference_states_are_real_states_not_componentwise_medians():
    """x_ref / x_prior are consumed as *states* (h(x), the ODE), so they must be
    reachable. A componentwise median generally is not: on ADM1 it breaks the
    charge balance and the implied pH leaves the physical range entirely."""
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    train = d.series("train")

    all_states = np.concatenate([s.truth for s in train], axis=0)
    x0s = np.stack([s.truth[0] for s in train], axis=0)
    assert np.any(
        np.all(np.isclose(all_states, d.x_ref), axis=1)
    ), "x_ref is not a real state"
    assert np.any(
        np.all(np.isclose(x0s, d.x_prior), axis=1)
    ), "x_prior is not a real initial state"
    # And it is genuinely central, not just the first row.
    assert not np.allclose(d.x_prior, np.median(x0s, axis=0))


def test_medoid_picks_an_actual_row_and_is_central():
    from pyadm1ode_estimation.estimation.deep_learning.data_adapter import _medoid

    rng = np.random.default_rng(0)
    rows = rng.normal(size=(50, 6))
    rows[7] = 0.0  # the most central row by construction
    scale = np.ones(6)
    m = _medoid(rows, scale)
    assert np.allclose(m, rows[7])
    m[:] = 999.0  # must be a copy, not a view into the caller's array
    assert not np.allclose(rows[7], 999.0)


def test_medoid_rejects_empty_input():
    from pyadm1ode_estimation.estimation.deep_learning.data_adapter import _medoid

    with pytest.raises(ValueError, match="non-empty"):
        _medoid(np.zeros((0, 4)), np.ones(4))


def test_relative_and_absolute_sigma_are_resolved():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    meas = np.concatenate([s.measurements for s in d.series("train")], axis=0)
    mag = np.sqrt(np.mean(meas**2, axis=0))
    assert d.sensor_sigma[0] == pytest.approx(0.03 * mag[0])  # relative
    assert d.sensor_sigma[1] == pytest.approx(0.02)  # absolute pH
    assert d.sensor_sigma[2] == pytest.approx(0.2)  # absolute TS


def test_val_and_test_are_normalised_with_train_statistics():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    raw = d.feature_spec.build(d.series("val")[0], d._ctx())[0]
    assert np.allclose(d.features("val")[0], (raw - d.feat_mean) / d.feat_std)


def test_constant_feature_column_does_not_divide_by_zero():
    ds = _dataset()
    for s in ds.pool + ds.test:
        s.feed[:, 1] = 2.0  # a substrate that is never dosed
    d = PinnData.build(ds, val_frac=0.2, seed=0)
    assert np.all(np.isfinite(d.features("train")))


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def test_feature_width_follows_the_data_not_a_constant():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    assert d.n_features == N_SENSOR + N_FEED
    assert d.feature_names[:N_SENSOR] == ["Q_gas", "pH", "TS"]


def test_optional_feature_blocks():
    ds = _dataset()
    spec = FeatureSpec(deltas=True, time_since_switch=True, log_feed=True)
    d = PinnData.build(ds, feature_spec=spec, val_frac=0.2, seed=0)
    assert d.n_features == 2 * (N_SENSOR + N_FEED) + 1
    assert d.feature_names[-1] == "t_since_switch_d"


def test_extra_feature_hook():
    def constant(series, ctx):
        assert ctx["dt_hours"] == 1.0
        return np.full((len(series.time), 2), 3.0)

    d = PinnData.build(
        _dataset(),
        feature_spec=FeatureSpec(extra=constant, extra_names=("a", "b")),
        val_frac=0.2,
        seed=0,
    )
    assert d.feature_names[-2:] == ["a", "b"]
    assert d.n_features == N_SENSOR + N_FEED + 2


def test_extra_hook_length_mismatch_is_rejected():
    d = _dataset()
    spec = FeatureSpec(extra=lambda s, ctx: np.zeros((3, 1)))
    with pytest.raises(ValueError, match="rows"):
        PinnData.build(d, feature_spec=spec, val_frac=0.2, seed=0)


def test_nan_measurements_become_zero_after_normalisation():
    ds = _dataset()
    for s in ds.pool:
        s.measurements[5, 0] = np.nan
    d = PinnData.build(ds, val_frac=0.2, seed=0)
    feats = d.features("train")
    assert np.all(np.isfinite(feats))
    assert feats[0, 5, 0] == 0.0


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------
def test_windows_tile_and_multiply_the_sample_count():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    f, x, t, names = d.windows("train", window=12, stride=12)
    assert f.shape == (16 * 4, 12, N_SENSOR + N_FEED)
    assert x.shape == (16 * 4, 12, N_STATE)
    assert t.shape == (16 * 4, 12)
    assert names[0].endswith("@0") and names[1].endswith("@1")


def test_overlapping_windows():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    f, _, _, _ = d.windows("train", window=24, stride=8)
    assert f.shape[0] == 16 * (1 + (T - 24) // 8)


def test_window_longer_than_series_is_rejected():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    with pytest.raises(ValueError, match="exceeds series length"):
        d.windows("train", window=T + 1)


# --------------------------------------------------------------------------
# Variant B payloads
# --------------------------------------------------------------------------
def test_observer_dataset_shapes_and_raw_measurements():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    od = d.observer_dataset("train", window=16, stride=16)
    assert od.features.shape == (16 * 3, 16, N_SENSOR + N_FEED)
    assert od.states.shape == (16 * 3, 16, N_STATE)
    assert od.feature_names == d.feature_names
    assert np.allclose(od.x_ref, d.x_ref)
    # The raw measurements must invert the normalisation exactly.
    first = d.series("train")[0]
    assert np.allclose(od.measurements[0], first.measurements[:16])


def test_observer_dataset_needs_no_physics():
    """Supervised state regression is physics-free, so a non-ADM1da set still works."""
    d = PinnData.build(_dataset(n_state=7), val_frac=0.2, seed=0)
    assert d.observer_dataset("train", window=16).params is None


def test_measurement_dataset_carries_no_labels():
    d = PinnData.build(_dataset(n_state=41), val_frac=0.2, seed=0, params=object())
    md = d.measurement_dataset("test", window=16, stride=16)
    assert not hasattr(md, "states")
    assert md.dt_days == pytest.approx(1.0 / 24.0)
    assert md.channel_names == d.channel_names
    assert np.allclose(md.feat_mean, d.feat_mean)
    assert np.allclose(md.measurements[0], d.series("test")[0].measurements[:16])


def test_measurement_dataset_requires_physics():
    """Its objective *is* the ADM1 residual, so it must not silently drop it."""
    d = PinnData.build(_dataset(n_state=7), val_frac=0.2, seed=0)
    with pytest.raises(ValueError, match="41-state ADM1da"):
        d.measurement_dataset("test", window=16)


# --------------------------------------------------------------------------
# Variant A payloads
# --------------------------------------------------------------------------
def test_smoother_inputs_carry_the_train_prior():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    items = d.smoother_inputs("val")
    assert len(items) == 4
    it = items[0]
    assert it.obs_values.shape == (T, N_SENSOR)
    assert it.t0 == pytest.approx(0.0) and it.t1 == pytest.approx((T - 1) / 24.0)
    assert np.allclose(it.x_prior, d.x_prior)  # identical for every series
    assert np.allclose(items[1].x_prior, d.x_prior)


def test_smoother_inputs_have_no_feed_params_without_adm1da():
    """A non-ADM1da dataset has no influent model, so params_at stays None."""
    d = PinnData.build(_dataset(n_state=7), val_frac=0.2, seed=0)
    assert all(it.params_at is None for it in d.smoother_inputs("val"))


def test_feed_matching_can_be_switched_off():
    d = PinnData.build(_dataset(n_state=7), val_frac=0.2, seed=0)
    assert d.smoother_inputs("val", feed_matched=False)[0].params_at is None


def test_smoother_inputs_truncation_and_forecast_window():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    it = d.smoother_inputs("val", days=1.0)[0]
    assert len(it.obs_times) == 25
    kw = it.fit_kwargs(forecast_days=0.5)
    assert kw["t1"] == pytest.approx(it.t1 + 0.5)
    assert kw["t0"] == it.t0


# --------------------------------------------------------------------------
# Physics guard (flexibility: a non-ADM1da dataset still works data-side)
# --------------------------------------------------------------------------
def test_non_adm1da_state_vector_still_supports_the_data_paths():
    d = PinnData.build(_dataset(n_state=7), val_frac=0.2, seed=0)
    assert d.n_state == 7
    assert d.windows("train", window=8)[0].shape[-1] == N_SENSOR + N_FEED


def test_non_adm1da_state_vector_rejects_physics_with_a_clear_error():
    d = PinnData.build(_dataset(n_state=7), val_frac=0.2, seed=0)
    with pytest.raises(ValueError, match="41-state ADM1da"):
        d.obs_model()
    with pytest.raises(ValueError, match="41-state ADM1da"):
        d.physics_params()


def test_unsupported_sensor_channel_is_reported():
    ds = _dataset(n_state=41)
    ds.meta["sensors"] = ["Q_gas", "pH", "conductivity"]
    d = PinnData.build(ds, val_frac=0.2, seed=0, params=object())
    with pytest.raises(ValueError, match="no differentiable map"):
        d.obs_model()


# --------------------------------------------------------------------------
# Scoring bridge
# --------------------------------------------------------------------------
def test_scoring_series_matches_the_benchmark_scorer_contract():
    d = PinnData.build(_dataset(), val_frac=0.2, seed=0)
    rows = d.scoring_series("test")
    assert len(rows) == len(d.series("test"))
    for r in rows:
        assert {"states", "time", "switch_days", "ukf_x_hat"} <= set(r)
        assert r["states"].shape == (T, N_STATE)


# --------------------------------------------------------------------------
# The real benchmark
# --------------------------------------------------------------------------
def _benchmark_or_skip():
    pytest.importorskip("pyadm1")
    from pyadm1ode_estimation.estimation.filter_tuning.datasets import (
        DEFAULT_BENCHMARK_DIR,
    )

    if not (DEFAULT_BENCHMARK_DIR / "train.npz").exists():
        pytest.skip("benchmark dataset not installed")
    return PinnData.build("benchmark", val_frac=0.2, seed=0)


@pytest.mark.slow
def test_benchmark_physics_params_are_fed():
    """A freshly built plant has never been fed: a snapshot taken from it carries
    q_ad = 0, i.e. the ODE of a closed batch reactor. The reference plant must be
    warmed before the snapshot."""
    d = _benchmark_or_skip()
    assert float(d.physics_params().q_ad) > 0.0
    assert np.count_nonzero(d.physics_params().s_in) > 0


@pytest.mark.slow
def test_benchmark_feed_params_follow_the_series():
    d = _benchmark_or_skip()
    it = d.smoother_inputs("val", days=5.0)[0]
    assert it.params_at is not None

    t = np.linspace(it.t0, it.t1, 6)
    p = it.params_at(t)
    q = np.asarray(p.q_ad, dtype=float)
    assert q.shape == (6,)
    # The series' own feed, not the plant's nominal operating point.
    actual = it.series.feed[: len(it.obs_times)].sum(axis=1).mean()
    assert abs(q.mean() - actual) / actual < 0.2
    assert abs(q.mean() - float(d.physics_params().q_ad)) > 1.0
    assert all(np.asarray(s, dtype=float).shape == (6,) for s in p.s_in)


@pytest.mark.slow
def test_benchmark_time_varying_feed_reaches_the_rhs():
    """1-D q_ad / s_in must broadcast across the collocation batch, with gradients."""
    torch_ = pytest.importorskip("torch")
    from pyadm1.core.adm1_torch import adm1da_rhs_torch

    d = _benchmark_or_skip()
    it = d.smoother_inputs("val", days=5.0)[0]
    p = it.params_at(np.linspace(it.t0, it.t1, 8))
    x = torch_.tensor(
        np.tile(it.truth[0], (8, 1)), dtype=torch_.float64, requires_grad=True
    )
    f = adm1da_rhs_torch(x, p)
    assert f.shape == (8, 41) and bool(torch_.isfinite(f).all())
    assert not torch_.allclose(f[0], f[-1])  # the feed really varies
    f.sum().backward()
    assert bool(torch_.isfinite(x.grad).all())


@pytest.mark.slow
def test_benchmark_prior_is_physically_consistent():
    """The regression this pins: a componentwise-median prior broke the ADM1 charge
    balance and implied pH ~11.6, starting the fit ~200 sigma off on that channel."""
    torch_ = pytest.importorskip("torch")
    d = _benchmark_or_skip()
    h = d.obs_model(quasi_steady_gas=True)
    with torch_.no_grad():
        y = h.predict(
            torch_.tensor(np.atleast_2d(d.x_prior), dtype=torch_.float64)
        ).numpy()[0]
    ph = y[d.channel_names.index("pH")]
    assert 6.5 < ph < 8.5, f"prior implies a non-physical pH {ph:.2f}"


@pytest.mark.slow
def test_benchmark_split_is_80_20_per_mode():
    d = _benchmark_or_skip()
    assert d.split.labels["train"] == {
        "low_high": 20,
        "oscillating": 20,
        "stable_high": 20,
        "stable_low": 20,
    }
    assert d.split.labels["val"] == {
        "low_high": 5,
        "oscillating": 5,
        "stable_high": 5,
        "stable_low": 5,
    }
    assert d.n_state == 41
    assert d.channel_names == ["Q_gas", "Q_ch4", "Q_co2", "pH", "TS"]
    # 3 % of the ~1800 m3/d gas flow, and the absolute probe specs.
    assert d.sensor_sigma[0] > 10.0
    assert d.sensor_sigma[3] == pytest.approx(0.02)
    assert d.sensor_sigma[4] == pytest.approx(0.2)
