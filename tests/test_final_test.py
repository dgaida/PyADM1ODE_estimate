"""The final-test harness: kinetics handover, plant pruning, artifacts, statistics helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
FINAL = REPO / "experiments" / "final_test"
needs_benchmark = pytest.mark.skipif(
    not (REPO / "datasets" / "benchmark" / "test.npz").exists(), reason="benchmark dataset not present"
)


@pytest.fixture(scope="module")
def meta():
    return json.loads((REPO / "datasets" / "benchmark" / "meta.json").read_text(encoding="utf-8"))


def test_prune_keeps_only_the_sensor_side(meta):
    from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import (
        PRIMARY_COMPONENTS,
        build_plant,
        prune_to_primary,
    )

    plant = prune_to_primary(build_plant(meta))
    assert set(plant.components) == set(PRIMARY_COMPONENTS)
    assert all(c.from_component in PRIMARY_COMPONENTS and c.to_component in PRIMARY_COMPONENTS for c in plant.connections)


def test_kinetic_factors_multiply_the_nominal_constants(meta):
    from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import apply_kinetic_factors, build_plant

    plant = build_plant(meta)
    k = plant.components["primary"].adm1._kinetic
    before = dict(k)
    assert apply_kinetic_factors(plant, {"k_m_ac": 0.5, "k_dis_PS": 2.0}) == ["k_dis_PS", "k_m_ac"]
    assert k["k_m_ac"] == pytest.approx(0.5 * before["k_m_ac"])
    assert k["k_dis_PS"] == pytest.approx(2.0 * before["k_dis_PS"])
    assert k["k_m_pro"] == before["k_m_pro"]
    with pytest.raises(KeyError):
        apply_kinetic_factors(plant, {"k_unknown": 1.1})
    assert apply_kinetic_factors(plant, None) == []


@needs_benchmark
def test_estimators_take_kinetics_and_never_the_truth():
    from pyadm1ode_estimation.estimation.estimators import PinnEstimator, UkfEstimator
    from pyadm1ode_estimation.estimation.filter_tuning.datasets import load_benchmark

    ds = load_benchmark(splits=("test",))
    s = ds.test[0]
    ukf = UkfEstimator(ds, prune=True)
    assert ukf.runner.prune is True
    assert ukf.episode(s, 0, {"k_m_ac": 0.8}).obs["kinetics"] == {"k_m_ac": 0.8}
    assert "kinetics" not in ukf.episode(s, 0).obs
    pinn = PinnEstimator(ds)
    import dataclasses

    pinn.cfg = dataclasses.replace(pinn.cfg, use_series_kinetics=True)
    with pytest.raises(ValueError, match="generator truth"):
        pinn.run(s)


@pytest.fixture(scope="module")
def harness():
    sys.path.insert(0, str(FINAL))
    import common  # noqa: F401
    import score

    return common, score


@needs_benchmark
def test_artifact_round_trip_restores_the_factors(harness, tmp_path, monkeypatch):
    C, _ = harness
    monkeypatch.setattr(C, "artifact_path", lambda cond, ds, i: tmp_path / f"{cond}_{ds}_{i}.yaml")
    p = C.write_artifact("X", "benchmark", 3, {"k_m_ac": 0.8, "k_dis_PS": 1.25}, "test")
    back = C.factors_from_artifact(p, "benchmark")
    assert back == pytest.approx({"k_m_ac": 0.8, "k_dis_PS": 1.25})
    assert "kinetic" in p.read_text(encoding="utf-8")


def test_config_lists_every_condition_and_variant(harness):
    C, _ = harness
    jobs = C.condition_jobs()
    assert ("nominal", "benchmark", "B1") in jobs
    assert ("A_cal", "benchmark_hard", "U2") in jobs
    assert ("B_real", "benchmark", "U2") in jobs and ("B_real", "benchmark_ccm", "U2") not in jobs
    assert all(v != "B1" for c, _, v in jobs if c != "nominal")  # B1 is kinetics-blind


def test_holm_and_hodges_lehmann(harness):
    _, S = harness
    h = S.holm({"a": 0.01, "b": 0.02, "c": 0.04})
    assert h["a"][2] and h["b"][2] and h["c"][2]  # 0.01 <= .05/3, 0.02 <= .05/2, 0.04 <= .05
    h = S.holm({"a": 0.03, "b": 0.001})
    assert h["b"][2] and h["a"][2]  # 0.001 <= .05/2, then 0.03 <= .05
    h = S.holm({"a": 0.06, "b": 0.001, "c": 0.02})
    assert h["b"][2] and h["c"][2] and not h["a"][2]
    h = S.holm({"a": 0.03, "b": 0.04})
    assert not h["a"][2] and not h["b"][2]  # 0.03 > .05/2 stops the sequence
    assert S.hodges_lehmann(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)


def test_gaussian_crps_matches_numeric_integration(harness):
    _, S = harness
    from scipy.stats import norm

    e, s = 0.7, 1.3
    grid = np.linspace(-20, 20, 200001)
    num = np.trapezoid((norm.cdf(grid, loc=e, scale=s) - (grid >= 0)) ** 2, grid)
    assert S.crps_gauss(np.array(e), np.array(s)) == pytest.approx(num, rel=1e-4)


def test_transient_mask_respects_the_window(harness):
    _, S = harness
    t = np.arange(1441) / 24.0
    m = S.transient_mask(t, np.array([10.0, 30.0]))
    assert not m[: S.A].any()  # day 10 lies before the window
    assert m[S.A + 10 * 24 : S.A + 12 * 24].all()  # day 30 to 32
    assert m.sum() == 48
