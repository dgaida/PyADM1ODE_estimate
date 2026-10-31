"""The shared benchmark loader and the PINN v2 wrapper built on it."""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1ode_estimation.estimation.filter_tuning.datasets import (
    BENCHMARK_AUX_KEYS,
    DEFAULT_BENCHMARK_DIR,
    Series,
    get_dataset,
    load_benchmark,
)
from pyadm1ode_estimation.estimation.pinn.data import BenchmarkSeries, load_series

needs_benchmark = pytest.mark.skipif(
    not (DEFAULT_BENCHMARK_DIR / "test.npz").exists(), reason="benchmark dataset not present"
)


def test_from_series_requires_the_generator_truth():
    s = Series(np.zeros((3, 1)), np.zeros((3, 2)), np.arange(3.0), np.zeros((3, 41)), np.zeros(0))
    with pytest.raises(KeyError):
        BenchmarkSeries.from_series(s, 0, "test")
    s.aux = {
        "feed_true": np.ones((3, 2)),
        "kinetic_factors": np.ones(2),
        "kinetic_keys": ["k_m_ac", "k_m_pro"],
        "ukf_x_hat": np.zeros((3, 41)),
    }
    bs = BenchmarkSeries.from_series(s, 7, "train")
    assert bs.index == 7 and bs.split == "train" and bs.seed == -1
    assert bs.kinetic_keys == ["k_m_ac", "k_m_pro"] and bs.ukf_x_hat is not None
    assert bs.ukf_std is None


@needs_benchmark
def test_test_split_only_and_aux_contents():
    ds = load_benchmark(splits=("test",))
    assert ds.pool == [] and len(ds.test) == 20 and ds.name == "benchmark"
    aux = ds.test[0].aux
    assert set(BENCHMARK_AUX_KEYS) | {"kinetic_keys"} >= set(aux)
    assert {"feed_true", "kinetic_factors", "kinetic_keys", "ukf_x_hat", "ukf_std"} <= set(aux)
    assert aux["feed_true"].shape == ds.test[0].feed.shape
    assert not np.isnan(aux["ukf_x_hat"]).any()


@needs_benchmark
def test_pinn_loader_matches_the_raw_file():
    z = np.load(DEFAULT_BENCHMARK_DIR / "test.npz", allow_pickle=True)
    (s,) = load_series("test", indices=[3])
    assert s.index == 3 and s.regime == str(z["regime"][3])
    np.testing.assert_array_equal(s.states, np.asarray(z["states"][3], float))
    np.testing.assert_array_equal(s.feed_true, np.asarray(z["feed_true"][3], float))
    np.testing.assert_array_equal(s.measurements, np.asarray(z["measurements"][3], float))
    np.testing.assert_array_equal(s.kinetic_factors, np.asarray(z["kinetic_factors"][3], float))
    with pytest.raises(ValueError):
        load_series("val")


@needs_benchmark
def test_variant_directory_without_train_split_loads():
    hard = DEFAULT_BENCHMARK_DIR.parent / "benchmark_hard"
    if not hard.exists():
        pytest.skip("benchmark_hard not present")
    ds = get_dataset(hard)
    assert ds.name == "benchmark_hard" and ds.pool == [] and len(ds.test) == 20
    # the UKF reference shipped next to test.npz is spliced in
    assert not np.isnan(ds.test[0].aux["ukf_x_hat"]).any()
