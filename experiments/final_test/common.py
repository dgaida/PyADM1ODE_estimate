"""Shared infrastructure of the final test: configuration, paths, atomic storage, kinetics.

Every result is one small file, written atomically (temporary file + ``os.replace``),
so an interrupted run never leaves a half-written result behind and a restart
simply skips what already exists. Each file records the SHA-256 of
``config.yaml``; results of another configuration are refused.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import traceback
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CONFIG_PATH = HERE / "config.yaml"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def config_hash() -> str:
    return hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()[:16]


def results_dir() -> Path:
    return REPO / config()["results_dir"]


def dataset_names() -> list[str]:
    return list(config()["datasets"])


def window() -> tuple[int, int]:
    a, b = config()["window_hours"]
    return int(a), int(b)


def condition_jobs() -> list[tuple[str, str, str]]:
    """All ``(condition, dataset, variant)`` triples the config asks for."""
    out = []
    for cond, spec in config()["conditions"].items():
        sets = dataset_names() if spec["datasets"] == "all" else list(spec["datasets"])
        for v in spec["variants"]:
            out += [(cond, d, v) for d in sets]
        for v, extra_sets in (spec.get("extra") or {}).items():
            out += [(cond, d, v) for d in extra_sets]
    return out


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@cache
def dataset(name: str):
    from pyadm1ode_estimation.estimation.filter_tuning.datasets import get_dataset

    return get_dataset(REPO / config()["datasets"][name])


def n_series(name: str) -> int:
    return len(dataset(name).test)


def kinetic_keys_all(name: str) -> list[str]:
    return list(dataset(name).test[0].aux["kinetic_keys"])


@lru_cache(maxsize=1)
def calibration_keys() -> list[str]:
    rep = json.loads(
        (REPO / config()["calibration"]["screening"]).read_text(encoding="utf-8")
    )
    return list(rep["top_k"])


def true_factors(name: str, i: int) -> dict[str, float]:
    s = dataset(name).test[i]
    return {
        str(k): float(f)
        for k, f in zip(s.aux["kinetic_keys"], s.aux["kinetic_factors"])
    }


# ---------------------------------------------------------------------------
# Paths and atomic storage
# ---------------------------------------------------------------------------
def traj_path(cond: str, ds: str, variant: str, i: int) -> Path:
    return results_dir() / "traj" / cond / ds / variant / f"s{i:02d}.npz"


def calib_path(ds: str, i: int) -> Path:
    return results_dir() / "calibration" / ds / f"s{i:02d}.json"


def artifact_path(cond: str, ds: str, i: int) -> Path:
    return results_dir() / "artifacts" / cond / ds / f"s{i:02d}.yaml"


def failed_path(path: Path) -> Path:
    return path.with_name(path.name + ".failed.json")


def sigma_real_path() -> Path:
    return results_dir() / "calibration" / "sigma_real.json"


def fusion_scales_path() -> Path:
    return results_dir() / "fusion_scales.json"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj) -> None:
    atomic_write_text(path, json.dumps(obj, indent=1, default=_json_default))


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_hash(meta_hash: str, path: Path) -> None:
    if meta_hash != config_hash():
        raise RuntimeError(
            f"{path} was produced with config {meta_hash}, current is {config_hash()}. "
            "Results of different configurations must not be mixed."
        )


def load_traj(cond: str, ds: str, variant: str, i: int) -> dict | None:
    """Stored trajectory dict (``x_hat``, ``std``, ``meta``) or ``None`` (missing / failed)."""
    p = traj_path(cond, ds, variant, i)
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    check_hash(meta["config_hash"], p)
    return {"x_hat": z["x_hat"], "std": z["std"], "meta": meta}


def is_done(path: Path) -> bool:
    """A job is done when its result or its failure record exists."""
    return path.exists() or failed_path(path).exists()


def record_failure(path: Path, exc: BaseException, extra: dict | None = None) -> None:
    atomic_write_json(
        failed_path(path),
        {
            "config_hash": config_hash(),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            **(extra or {}),
        },
    )


# ---------------------------------------------------------------------------
# Kinetics per condition
# ---------------------------------------------------------------------------
def kinetics_for(cond: str, ds: str, i: int) -> dict[str, float]:
    """``key -> factor`` (relative to nominal) the estimators get in ``cond``.

    ``nominal``: none. ``A_cal``: the Test A calibration of this series.
    ``B_ident``: the true factors of the calibrated keys. ``B_real``: those times a
    fixed lognormal error of the per-parameter Test A size. ``B_voll``: all true.
    """
    keys = calibration_keys()
    if cond == "nominal":
        return {}
    if cond == "A_cal":
        rec = read_json(calib_path(ds, i))
        check_hash(rec["config_hash"], calib_path(ds, i))
        return {k: float(v) for k, v in rec["factors"].items()}
    truth = true_factors(ds, i)
    if cond == "B_ident":
        return {k: truth[k] for k in keys}
    if cond == "B_voll":
        return dict(truth)
    if cond == "B_real":
        sig = read_json(sigma_real_path())
        check_hash(sig["config_hash"], sigma_real_path())
        rng = np.random.default_rng(
            [int(config()["test_b"]["seed"]), dataset_names().index(ds), i]
        )
        return {
            k: truth[k] * float(np.exp(rng.normal(0.0, float(sig["sigma"][k]))))
            for k in keys
        }
    raise KeyError(cond)


def write_artifact(
    cond: str, ds: str, i: int, factors: dict[str, float], notes: str
) -> Path:
    """The kinetics of one series as a ``CalibrationArtifact`` (absolute constants)."""
    from pyadm1ode_estimation.artifacts.calibration_artifact import (
        CalibrationArtifact,
        CalibrationMetadata,
        save_artifact,
    )

    nominal = nominal_kinetics(ds)
    art = CalibrationArtifact(
        metadata=CalibrationMetadata(
            plant_id=f"{ds}#{i}",
            calibration_run_id=f"final_test/{cond}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            fitted_against=list(
                config()["metrics"].get(
                    "sensors", ["Q_gas", "Q_ch4", "Q_co2", "pH", "TS"]
                )
            ),
            notes=notes,
        ),
        kinetic={"primary": {k: float(nominal[k] * f) for k, f in factors.items()}},
    )
    p = artifact_path(cond, ds, i)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    save_artifact(art, tmp)
    os.replace(tmp, p)
    return p


@cache
def nominal_kinetics(ds: str) -> dict[str, float]:
    from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import build_plant

    k = build_plant(dataset(ds).meta).components["primary"].adm1._kinetic
    return {key: float(v) for key, v in k.items()}


def factors_from_artifact(path: Path, ds: str) -> dict[str, float]:
    """Inverse of :func:`write_artifact`: ``key -> factor`` relative to nominal."""
    from pyadm1ode_estimation.artifacts.calibration_artifact import load_artifact

    art = load_artifact(path)
    nominal = nominal_kinetics(ds)
    return {k: float(v) / nominal[k] for k, v in art.kinetic["primary"].items()}


@dataclass
class Log:
    path: Path

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
