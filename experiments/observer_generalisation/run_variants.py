"""How well does PINN B generalise, and what makes it generalise better?

The published observer (window 336 h, no overlap, 200 epochs, patience 40,
hidden 64 x 2, no regularisation) scores 15.1 % on the benchmark test set but
18.4 % on ``benchmark_ccm`` (substrate swap) and 20.0 % on ``benchmark_hard``
(swap + 3-day load rhythm), while the UKF degrades from 14.4 % only to 17.1 %
and 17.7 %. This sweep trains regularised variants of the same network on the
same split and scores every one on all three test sets, so the gap between the
in-distribution and the shifted scores (the generalisation gap) can be compared
across recipes. Several seeds expose how much of a difference is noise.

    python experiments/observer_generalisation/run_variants.py
    python experiments/observer_generalisation/run_variants.py --variants baseline noise --seeds 0

Appends one record per (variant, seed) to ``results_observer/generalisation.jsonl``
and stores the checkpoints of seed 0 in ``results_observer/``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyadm1ode_estimation.estimation.estimators import (
    DEFAULT_BURNIN_STEPS,
    ObserverEstimator,
    score,
)
from pyadm1ode_estimation.estimation.filter_tuning.datasets import get_dataset
from pyadm1ode_estimation.estimation.observer import FeatureSpec, PinnData

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results_observer"
TEST_SETS = {
    "benchmark": REPO / "datasets" / "benchmark",
    "benchmark_ccm": REPO / "datasets" / "benchmark_ccm",
    "benchmark_hard": REPO / "datasets" / "benchmark_hard",
}

#: The published PINN B recipe; every variant is a delta against it. Pinned here
#: because ``ObserverEstimator.train`` now defaults to ``dropout=0.1``.
BASE_RECIPE: dict = {"dropout": 0.0}

#: Recipe deltas against the published baseline.
VARIANTS: dict[str, dict] = {
    "baseline": {},
    "noise": {"noise_aug": 1.0},
    "noise2": {"noise_aug": 2.0},
    "wd": {"weight_decay": 1.0e-4},
    "dropout": {"dropout": 0.1},
    "random_windows": {"n_random": 8},
    "overlap": {"stride": 168},
    "small": {"hidden": 32},
    "short_window": {"window": 168},
    "combined": {"noise_aug": 1.0, "weight_decay": 1.0e-4, "n_random": 8},
    "combined_dropout": {
        "noise_aug": 1.0,
        "weight_decay": 1.0e-4,
        "n_random": 8,
        "dropout": 0.1,
    },
    # substrate-agnostic inputs: influent loads instead of / beside per-substrate volumes
    "load": {"features": {"organic_load": True, "use_feed": False}},
    "load_feed": {"features": {"organic_load": True}},
    "load_dropout": {
        "features": {"organic_load": True, "use_feed": False},
        "dropout": 0.1,
    },
}


def recipe_of(variant: str) -> dict:
    """Full training keyword set of a variant (published baseline + its delta)."""
    return {**BASE_RECIPE, **VARIANTS[variant]}


def evaluate(est: ObserverEstimator, datasets: dict, burnin: int) -> dict:
    """Per test set: mean score, per-mode means and the per-series scores."""
    out = {}
    for name, ds in datasets.items():
        e = est.for_dataset(
            ds
        )  # meta-dependent features use the target plant's substrates
        per = [score(e.run(s, i), s, burnin) for i, s in enumerate(ds.test)]
        labels = [s.label for s in ds.test]
        out[name] = {
            "mean": float(np.mean(per)),
            "per_mode": {
                m: float(np.mean([p for p, l in zip(per, labels) if l == m]))
                for m in sorted(set(labels))
            },
            "per_series": per,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--out", default="generalisation.jsonl")
    a = ap.parse_args()

    bad = [v for v in a.variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"unknown variants {bad}; known: {sorted(VARIANTS)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / a.out
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["seed"]))

    datasets = {k: get_dataset(p) for k, p in TEST_SETS.items() if p.exists()}
    data_cache: dict[str, PinnData] = {}

    def data_for(features: dict) -> PinnData:
        key = json.dumps(features, sort_keys=True)
        if key not in data_cache:
            data_cache[key] = PinnData.build(
                datasets["benchmark"],
                val_frac=0.2,
                seed=0,
                feature_spec=FeatureSpec(**features),
            )
            print(data_cache[key].summary(), flush=True)
        return data_cache[key]

    for variant in a.variants:
        for seed in a.seeds:
            if (variant, seed) in done:
                continue
            recipe = recipe_of(variant) | {"epochs": a.epochs, "seed": seed}
            features = recipe.pop("features", {})
            t0 = time.perf_counter()
            est, res = ObserverEstimator.train(data_for(features), **recipe)
            recipe["features"] = features
            secs = time.perf_counter() - t0
            scores = evaluate(est, datasets, DEFAULT_BURNIN_STEPS)
            line = "  ".join(f"{k} {100 * v['mean']:5.1f}%" for k, v in scores.items())
            print(
                f"{variant:<17} seed {seed}  best epoch {res.best_epoch:3d}  val {res.best_val:.3e}  "
                f"{line}  ({secs:.0f}s)",
                flush=True,
            )
            if seed == 0:
                est.save(OUT_DIR / f"observer_{variant}.pt")
            rec = {
                "variant": variant,
                "seed": seed,
                "recipe": recipe,
                "best_epoch": int(res.best_epoch),
                "best_val": float(res.best_val),
                "train_seconds": secs,
                "scores": scores,
            }
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
    print(f"\nrecords in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
