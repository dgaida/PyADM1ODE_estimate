"""Pruning the plant to the primary digester changes the run time, not the UKF output."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pyadm1ode_estimation.estimation.filter_tuning import filter_runners as fr
from pyadm1ode_estimation.estimation.filter_tuning.datasets import (
    DEFAULT_BENCHMARK_DIR,
    load_benchmark,
)


@pytest.fixture(scope="module")
def meta():
    return json.loads((DEFAULT_BENCHMARK_DIR / "meta.json").read_text(encoding="utf-8"))


def test_runner_prunes_by_default_but_not_a_foreign_digester(meta):
    assert fr.make_ukf_runner(meta).prune
    assert not fr.make_ukf_runner(meta, prune=False).prune
    assert not fr.make_ukf_runner(meta, digester_id="secondary").prune


def test_cache_key_ignores_pruning(meta):
    """Old cache entries were written unpruned; they must stay valid."""
    assert (
        fr.make_ukf_runner(meta).cache_key()
        == fr.make_ukf_runner(meta, prune=False).cache_key()
    )
    assert (
        fr.make_ukf_runner(meta).cache_key()
        != fr.make_ukf_runner(meta, variant="adcore").cache_key()
    )


def test_pruned_run_is_bit_identical(meta):
    ds = load_benchmark(splits=("test",))
    ep = ds.to_episode(ds.test[10], days=0.25, burnin_days=0.0)
    theta = {"p0_scale": 0.3, "r_scale": 1.4}
    runs = [
        fr.make_ukf_runner(meta, warmup_days=0.5, prune=p)(theta, ep)
        for p in (True, False)
    ]
    assert np.array_equal(runs[0][0], runs[1][0])
    assert np.array_equal(runs[0][1], runs[1][1], equal_nan=True)
