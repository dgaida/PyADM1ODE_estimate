"""Is PINN B's generalisation gap a coverage problem? Train with some shifted series.

The regularisation sweep (``run_variants.py``) leaves the gap between the
benchmark and the substrate-swap test set unchanged, and the error grows on
exactly the states the swap moves (fast particulates, propionate degraders,
charge-balance ions). If the gap is a *coverage* gap, adding a few series of
the shifted plant to the training pool must close most of it on the remaining
shifted series. This script compares two training pools with the same recipe:

``benchmark``      the 100 benchmark training series (as published)
``+ccm``           the same plus every second series of each mode of ``benchmark_ccm`` (12 series)

Both are scored on the other 8 ``benchmark_ccm`` series (never trained on),
on ``benchmark_hard`` (swap + 3-day load rhythm, never trained on) and on the
benchmark test set.

    python experiments/observer_generalisation/coverage.py --seeds 0 1 2
    python experiments/observer_generalisation/coverage.py --features load

Appends to ``results_observer/coverage.jsonl``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_variants import OUT_DIR, TEST_SETS, VARIANTS, recipe_of  # noqa: E402

from pyadm1ode_estimation.estimation.estimators import (  # noqa: E402
    DEFAULT_BURNIN_STEPS,
    ObserverEstimator,
    score,
)
from pyadm1ode_estimation.estimation.filter_tuning.datasets import (  # noqa: E402
    EstimatorDataset,
    get_dataset,
)
from pyadm1ode_estimation.estimation.observer import FeatureSpec, PinnData  # noqa: E402


def mean_score(est, series) -> float:
    return float(np.mean([score(est.run(s, i), s, DEFAULT_BURNIN_STEPS) for i, s in enumerate(series)]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="dropout", choices=sorted(VARIANTS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument(
        "--features",
        default="raw",
        choices=("raw", "load"),
        help="raw per-substrate feed, or substrate-aware organic loads (each series with its plant's meta)",
    )
    a = ap.parse_args()

    bench = get_dataset(TEST_SETS["benchmark"])
    ccm = get_dataset(TEST_SETS["benchmark_ccm"])
    hard = get_dataset(TEST_SETS["benchmark_hard"])
    # alternate *within* each mode (the file cycles through the 4 modes, so a plain
    # [0::2] would put two modes into training and the other two into the held-out half)
    by_mode: dict[str, list] = {}
    for s in ccm.test:
        by_mode.setdefault(s.label, []).append(s)
    ccm_train = [s for m in sorted(by_mode) for s in by_mode[m][0::2]]
    ccm_held = [s for m in sorted(by_mode) for s in by_mode[m][1::2]]
    print(f"ccm: {len(ccm_train)} series added to training, {len(ccm_held)} held out "
          f"(modes {sorted({s.label for s in ccm_held})})", flush=True)

    recipe = recipe_of(a.variant)
    if recipe.pop("features", None):
        raise SystemExit("pick the input with --features, not through the variant")
    spec = FeatureSpec() if a.features == "raw" else FeatureSpec(organic_load=True, use_feed=False)
    # each corn-cob series carries its own plant meta, so its loads use CCM's characterisation
    ccm_train = [dataclasses.replace(s, aux={**s.aux, "meta": ccm.meta}) for s in ccm_train]
    pools = {
        "benchmark": bench,
        "+ccm": EstimatorDataset(name="benchmark+ccm", meta=bench.meta, pool=bench.pool + ccm_train, test=bench.test),
    }
    out = OUT_DIR / "coverage.jsonl"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pool_name, ds in pools.items():
        data = PinnData.build(ds, val_frac=0.2, seed=0, feature_spec=spec)
        for seed in a.seeds:
            t0 = time.perf_counter()
            est, res = ObserverEstimator.train(data, **recipe, epochs=a.epochs, seed=seed)
            scores = {
                "benchmark": mean_score(est.for_dataset(bench), bench.test),
                "ccm_held_out": mean_score(est.for_dataset(ccm), ccm_held),
                "hard": mean_score(est.for_dataset(hard), hard.test),
            }
            line = "  ".join(f"{k} {100 * v:5.1f}%" for k, v in scores.items())
            print(f"{pool_name:<10} seed {seed}  best epoch {res.best_epoch:3d}  {line}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            with out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"pool": pool_name, "features": a.features, "variant": a.variant, "seed": seed,
                                     "best_epoch": int(res.best_epoch), "scores": scores}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
