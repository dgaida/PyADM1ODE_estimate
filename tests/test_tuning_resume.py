"""Resuming an interrupted tuning run: task cache and CMA-ES replay (no real filter runs)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from pyadm1ode_estimation.estimation.filter_tuning import cmaes_search as cs
from pyadm1ode_estimation.estimation.filter_tuning import filter_runners as fr

cma = pytest.importorskip("cma")


class _FakeRunner:
    """Deterministic stand-in for ``UkfRunner`` that counts its calls."""

    def __init__(self):
        self.calls = 0

    def cache_key(self) -> str:
        return "fake-v1"

    def __call__(self, theta, ep):
        self.calls += 1
        if theta.get("fail"):
            raise RuntimeError("stiff")
        x = np.asarray(ep.truth, float) * theta.get("gain", 1.0)
        return x, np.full_like(x, 0.1)


def _episode(seed: int):
    truth = np.random.default_rng(seed).normal(size=(6, 3))
    return SimpleNamespace(
        obs={"time": np.arange(6.0), "label": f"m{seed}"},
        truth=truth,
        dt_hours=1.0,
        burnin=1,
    )


@pytest.fixture
def cache(tmp_path):
    fr.set_task_cache(tmp_path / "cache")
    yield tmp_path / "cache"
    fr.set_task_cache(None)


def test_task_cache_restores_runs_and_failures(cache):
    runner = _FakeRunner()
    eps = [_episode(0), _episode(1)]
    thetas = [{"gain": 2.0}, {"fail": True}]
    first = fr.collect_batch(runner, thetas, eps, jobs=1)
    assert runner.calls == 4
    assert len(list(cache.glob("*.npz"))) == 4

    again = fr.collect_batch(runner, thetas, eps, jobs=1)
    assert runner.calls == 4  # nothing recomputed
    assert again[1] == [None, None]  # a failed run stays failed
    for (t0, x0, s0), (t1, x1, s1) in zip(first[0], again[0]):
        assert np.array_equal(t0, t1) and np.array_equal(x0, x1)
        assert np.array_equal(s0, s1)
        assert len(t1) == 5  # burn-in trimmed

    fr.collect_batch(runner, [{"gain": 3.0}], eps, jobs=1)
    assert runner.calls == 6  # a new theta is a new task


def test_task_cache_key_depends_on_episode_content(cache):
    runner = _FakeRunner()
    fr.collect_batch(runner, [{}], [_episode(0)], jobs=1)
    fr.collect_batch(runner, [{}], [_episode(2)], jobs=1)
    assert runner.calls == 2


BLOCKS = ["methanogenesis", "charge_balance"]


def _fake_eval(runner, thetas, train, val, jobs=1, **_):
    """Score depends smoothly on the candidate, so CMA-ES has something to follow."""
    out = []
    for th in thetas:
        v = float(np.log(np.asarray(th.get("q_diag", [1.0]))).mean()) if th else 0.0
        v += np.log(th.get("r_scale", 1.0)) ** 2 if th else 0.0
        out.append(
            {
                "theta": {**th, "sigma_hi": 1.0, "gamma": 1.0},
                "nrmse": 1.0 + (v + 9.0) ** 2 / 100.0,
                "coverage": 0.9,
                "fostac_band_coverage": 0.9,
                "critical_auc": 0.9,
                "critical_balacc": 0.5,
            }
        )
    return out


def _search(monkeypatch, tmp_path, name, generations, resume=None):
    monkeypatch.setattr(cs.ns, "evaluate_batch", _fake_eval)
    monkeypatch.setattr(cs, "make_ukf_runner", lambda *a, **k: None)
    return cs.run_cmaes(
        SimpleNamespace(meta={}),
        np.full(41, 1e-4),
        blocks=BLOCKS,
        sigma_eps=[1, 2],
        score_eps=[3, 4],
        popsize=4,
        generations=generations,
        sigma0=0.5,
        objective=lambda m, b: b - m["nrmse"],
        out_path=tmp_path / name,
        resume=resume,
    )


def test_resumed_search_matches_an_uninterrupted_one(monkeypatch, tmp_path):
    full = _search(monkeypatch, tmp_path, "full.json", 4)
    _search(monkeypatch, tmp_path, "part.json", 2)
    part = json.loads((tmp_path / "part.json").read_text(encoding="utf-8"))
    resumed = _search(monkeypatch, tmp_path, "part.json", 4, resume=part)

    assert len(resumed["generations"]) == 4
    for g_full, g_res in zip(full["generations"], resumed["generations"]):
        assert [c["x"] for c in g_full["candidates"]] == [
            c["x"] for c in g_res["candidates"]
        ]
    assert resumed["best"]["score"] == full["best"]["score"]
    on_disk = json.loads((tmp_path / "part.json").read_text(encoding="utf-8"))
    assert len(on_disk["generations"]) == 4


def test_resume_refuses_a_trace_with_other_settings(monkeypatch, tmp_path):
    _search(monkeypatch, tmp_path, "part.json", 1)
    part = json.loads((tmp_path / "part.json").read_text(encoding="utf-8"))
    part["config"]["sigma0"] = 1.5
    with pytest.raises(ValueError, match="sigma0"):
        _search(monkeypatch, tmp_path, "part.json", 2, resume=part)
