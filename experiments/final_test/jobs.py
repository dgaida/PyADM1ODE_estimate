"""Units of work of the final test. Each writes exactly one result (or one failure record).

Run in worker processes by :mod:`run`. A job never depends on in-memory state of
another job; everything it needs is read from disk, which is what makes an
interrupted run resumable.
"""

from __future__ import annotations

import os
import time
import warnings
from functools import lru_cache

import numpy as np

import common as C


def init_worker() -> None:
    """One thread per worker process; the pool provides the parallelism."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    warnings.filterwarnings("ignore")
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def _window_score(x: np.ndarray, truth: np.ndarray) -> float:
    a, b = C.window()
    e = x[a:b] - truth[a:b]
    rmse = np.sqrt(np.mean(e**2, axis=0))
    c = np.sqrt(np.mean(truth[a:b] ** 2, axis=0)) + 1e-12
    return float(np.median(rmse / c))


def _meta(**kw) -> str:
    import json

    return json.dumps({"config_hash": C.config_hash(), **kw}, default=C._json_default)


# ---------------------------------------------------------------------------
# Estimator trajectories (UKF, PINN v2, open loop): one series per job
# ---------------------------------------------------------------------------
def kinetics_of(cond: str, ds: str, i: int) -> dict[str, float]:
    """The kinetics handed to the estimators: always read back from the artifact."""
    if cond == "nominal":
        return {}
    return C.factors_from_artifact(C.artifact_path(cond, ds, i), ds)


def run_traj(cond: str, ds: str, variant: str, i: int) -> str:
    path = C.traj_path(cond, ds, variant, i)
    if C.is_done(path):
        return "skip"
    init_worker()
    spec = C.config()["variants"][variant]
    data = C.dataset(ds)
    s = data.test[i]
    t0 = time.perf_counter()
    try:
        kin = kinetics_of(cond, ds, i)
        if spec["kind"] == "ukf":
            from pyadm1ode_estimation.estimation.estimators import UkfEstimator

            tr = UkfEstimator(data, spec["variant"], prune=bool(spec.get("prune", False))).run(s, i, kinetics=kin)
            x, std = tr.x_hat, tr.std
        elif spec["kind"] == "pinn":
            from pyadm1ode_estimation.estimation.estimators import PinnEstimator

            tr = PinnEstimator(data).run(s, i, kinetics=kin)
            x, std = tr.x_hat, np.zeros_like(tr.x_hat)
        elif spec["kind"] == "open_loop":
            from forward import open_loop

            x = open_loop(data.meta, np.asarray(s.feed, float), kin)
            std = np.zeros_like(x)
        else:
            raise ValueError(f"run_traj does not handle {spec['kind']}")
        x = np.asarray(x, float)
        finite = bool(np.all(np.isfinite(x)))
        C.atomic_savez(
            path,
            x_hat=x.astype(np.float32),
            std=np.asarray(std, float).astype(np.float32),
            meta=_meta(cond=cond, ds=ds, variant=variant, i=i, wall_s=time.perf_counter() - t0,
                       kinetics=kin, finite=finite, score=_window_score(x, np.asarray(s.truth, float)) if finite else None),
        )
        return "ok"
    except Exception as exc:  # noqa: BLE001 - a failing series is a result (T4), not a crash
        C.record_failure(path, exc, {"cond": cond, "ds": ds, "variant": variant, "i": i,
                                     "wall_s": time.perf_counter() - t0})
        return "failed"


# ---------------------------------------------------------------------------
# Observer ensembles: one test set per job (fast; members loaded once)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _members(variant: str, ds: str):
    from pyadm1ode_estimation.estimation.estimators import ObserverEstimator

    spec = C.config()["variants"][variant]
    return [
        ObserverEstimator.load(C.REPO / spec["checkpoints"].format(seed=sd), C.dataset(ds))
        for sd in spec["seeds"]
    ]


def run_observer(cond: str, ds: str, variant: str) -> str:
    init_worker()
    members = _members(variant, ds)
    data = C.dataset(ds)
    status = "skip"
    for i, s in enumerate(data.test):
        path = C.traj_path(cond, ds, variant, i)
        if C.is_done(path):
            continue
        t0 = time.perf_counter()
        try:
            kin = kinetics_of(cond, ds, i)
            xs = np.stack([m.run(s, i, kinetics=kin).x_hat for m in members])
            truth = np.asarray(s.truth, float)
            C.atomic_savez(
                path,
                x_hat=xs.mean(axis=0).astype(np.float32),
                std=xs.std(axis=0).astype(np.float32),
                meta=_meta(cond=cond, ds=ds, variant=variant, i=i, wall_s=time.perf_counter() - t0,
                           kinetics=kin, finite=bool(np.all(np.isfinite(xs))),
                           member_scores=[_window_score(x, truth) for x in xs],
                           score=_window_score(xs.mean(axis=0), truth)),
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            C.record_failure(path, exc, {"cond": cond, "ds": ds, "variant": variant, "i": i})
            status = "failed"
    return status


# ---------------------------------------------------------------------------
# Test A calibration: one series per job
# ---------------------------------------------------------------------------
def run_calibration(ds: str, i: int) -> str:
    path = C.calib_path(ds, i)
    if C.is_done(path):
        return "skip"
    init_worker()
    from forward import ForwardModel, calibrate

    cfg = C.config()["calibration"]
    data = C.dataset(ds)
    s = data.test[i]
    h0, h1 = cfg["fit_hours"]
    t0 = time.perf_counter()
    try:
        u1 = C.load_traj("nominal", ds, "U1", i)
        if u1 is not None and np.all(np.isfinite(u1["x_hat"][h0])):
            x_start, start_source = np.asarray(u1["x_hat"][h0], float), f"nominal U1 estimate at hour {h0}"
        else:  # the nominal filter failed on this series: fall back to the warm-up state
            x_start, start_source = ForwardModel.build(data.meta).warmup_state(), "warm-up state (U1 failed)"
        keys = C.calibration_keys()
        res = calibrate(
            data.meta, s.measurements, s.feed, x_start, keys, (h0, h1),
            budget=int(cfg["budget"]), log_bound=float(cfg["log_bound"]),
            xtol=float(cfg["xtol"]), ftol=float(cfg["ftol"]),
        )
        truth = C.true_factors(ds, i)  # diagnostics only, after the fit
        res["log_error"] = {k: float(np.log(res["factors"][k] / truth[k])) for k in keys}
        res["log_error_nominal"] = {k: float(-np.log(truth[k])) for k in keys}
        C.atomic_write_json(path, {"config_hash": C.config_hash(), "ds": ds, "i": i,
                                   "start_source": start_source, "keys": keys, **res})
        return "ok"
    except Exception as exc:  # noqa: BLE001
        C.record_failure(path, exc, {"ds": ds, "i": i, "wall_s": time.perf_counter() - t0})
        return "failed"


# ---------------------------------------------------------------------------
# Validation-split open loop (observability tiers) and fusion scales
# ---------------------------------------------------------------------------
def val_series():
    from pyadm1ode_estimation.estimation.observer import PinnData

    data = PinnData.build(C.dataset("benchmark"), val_frac=0.2, seed=0)
    return data, data.series("val")


def run_val_open_loop(i: int) -> str:
    path = C.results_dir() / "val" / "R1" / f"s{i:02d}.npz"
    if C.is_done(path):
        return "skip"
    init_worker()
    from forward import open_loop

    try:
        _, val = val_series()
        s = val[i]
        x = open_loop(C.dataset("benchmark").meta, np.asarray(s.feed, float), {})
        C.atomic_savez(path, x_hat=x.astype(np.float32), std=np.zeros_like(x, dtype=np.float32),
                       meta=_meta(cond="val", ds="benchmark_val", variant="R1", i=i))
        return "ok"
    except Exception as exc:  # noqa: BLE001
        C.record_failure(path, exc, {"i": i})
        return "failed"
