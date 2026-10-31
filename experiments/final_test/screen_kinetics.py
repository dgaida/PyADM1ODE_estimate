"""Rank the 26 kinetic parameters by how strongly they move the sensors — on TRAINING series.

The final test calibrates only the ``k`` most influential kinetics, and the
kinetics-aware observer takes exactly those ``k`` as inputs. The ranking must
come from the training split: the kinetics-calibration study screened its test
series, which would leak into a test that scores those series.

Reuses the Brun et al. (2001) screening of
``PyADM1ODE_calibration/experiments/kinetics_calibration`` (noise-scaled output
Jacobian with respect to the log-factors, one 60-day computation scores every
window length). The Jacobian is linearised at the *nominal* kinetics, which is
what a practitioner can compute without knowing the answer. Importance is pooled
over series by the geometric mean, so a parameter has to matter in most operating
modes to rank high.

    python experiments/final_test/screen_kinetics.py --per-mode 2 --workers 8

Writes ``results_final/screening_train.json`` (per-series reports, pooled ranking
per window, and the ``top_k`` list used downstream).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DEFAULT_STUDY = REPO.parent / "PyADM1ODE_calibration" / "experiments" / "kinetics_calibration"
DEFAULT_DATA = REPO / "datasets" / "benchmark"
OUT = REPO / "results_final" / "screening_train.json"

_MODEL = None


def _init(study: str, data: str) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ["PYADM1ODE_BENCHMARK"] = data
    if study not in sys.path:
        sys.path.insert(0, study)


def _screen(args: tuple) -> dict:
    """Jacobian + Brun report of one training series (runs in a worker)."""
    global _MODEL
    series, study, data, around, n_days = args
    _init(study, data)
    from fastsim import ForwardModel
    from screening import compute_jacobian, screen

    if _MODEL is None:
        _MODEL = ForwardModel.build(prune=True)
    jac = compute_jacobian(series, _MODEL, around=around, n_days=n_days)
    rep = screen(jac)
    rep["pool_index"] = int(series["pool_index"])
    rep["regime"] = str(series["regime"])
    return rep


def training_series(data_dir: Path, per_mode: int) -> list[dict]:
    """``per_mode`` series of each mode from the training part of the shared split."""
    sys.path.insert(0, str(REPO))
    from pyadm1ode_estimation.estimation.filter_tuning.datasets import load_benchmark

    ds = load_benchmark(data_dir, splits=("train",))
    train_idx, _, _ = ds.split_indices(0.2, seed=0)  # the split every estimator uses
    by_mode: dict[str, list[int]] = {}
    for i in train_idx:
        by_mode.setdefault(ds.pool[i].label, []).append(i)
    out = []
    for mode in sorted(by_mode):
        for i in by_mode[mode][:per_mode]:
            s = ds.pool[i]
            out.append(
                {
                    "pool_index": i,
                    "regime": s.label,
                    "feed_noisy": np.asarray(s.feed, float),
                    "states": np.asarray(s.truth, float),
                    "measurements": np.asarray(s.measurements, float),
                    "kinetic_factors": np.asarray(s.aux["kinetic_factors"], float),
                }
            )
    return out


def pool(per_series: list[dict], window: str, keys: list[str]) -> tuple[dict, list[str]]:
    """Geometric mean of ``delta_msqr`` over series at one window length."""
    pooled = {
        k: float(np.exp(np.mean([np.log(max(r["windows"][window]["delta_msqr"][k], 1e-12)) for r in per_series])))
        for k in keys
    }
    return pooled, sorted(pooled, key=lambda k: -pooled[k])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--study", default=str(DEFAULT_STUDY), help="kinetics_calibration folder")
    ap.add_argument("--data", default=str(DEFAULT_DATA))
    ap.add_argument("--per-mode", type=int, default=2)
    ap.add_argument("--around", default="nominal", choices=("nominal", "truth"))
    ap.add_argument("--window", default="20", help="window [d] that defines top_k (the Test A calibration window)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    series = training_series(Path(a.data), a.per_mode)
    print(f"screening {len(series)} training series ({a.around}), {a.workers} workers", flush=True)
    t0 = time.time()
    reports = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_screen, (s, a.study, a.data, a.around, 60.0)): s["pool_index"] for s in series}
        for fut in as_completed(futs):
            rep = fut.result()
            reports.append(rep)
            print(f"  series {rep['pool_index']:3d} ({rep['regime']}) done  [{time.time() - t0:.0f}s]", flush=True)
    reports.sort(key=lambda r: r["pool_index"])
    keys = list(reports[0]["keys"])
    pooled = {}
    for w in reports[0]["windows"]:
        imp, ranking = pool(reports, w, keys)
        pooled[w] = {"delta_msqr_geomean": imp, "ranking": ranking}
    top_k = pooled[a.window]["ranking"][: a.k]
    report = {
        "source": "benchmark train split (shared split, val_frac=0.2, seed=0)",
        "around": a.around,
        "series": [r["pool_index"] for r in reports],
        "per_series": reports,
        "pooled": pooled,
        "window_for_top_k": a.window,
        "top_k": top_k,
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
    for w, p in pooled.items():
        print(f"  {w:>3} d: " + ", ".join(f"{k} {p['delta_msqr_geomean'][k]:.2f}" for k in p["ranking"][:12]))
    print("top_k:", top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
