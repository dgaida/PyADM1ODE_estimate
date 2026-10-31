"""Seed ensemble of the GRU observer: does averaging members close the generalisation gap?

The single-seed spread of PINN B on the shifted test sets (about 1 point) is as
large as the effect of any regularisation recipe, so part of the gap is variance.
This script trains ``--members`` seeds of one recipe from
:data:`run_variants.VARIANTS`, averages their trajectories and scores members and
ensemble on all three test sets.

    python experiments/observer_generalisation/ensemble.py --variant dropout --members 5

Appends one record to ``results_observer/ensemble.jsonl`` and saves the members
as ``results_observer/ensemble_<variant>_<seed>.pt``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_variants import OUT_DIR, TEST_SETS, VARIANTS, recipe_of  # noqa: E402

from pyadm1ode_estimation.estimation.base import TrajectoryEstimate  # noqa: E402
from pyadm1ode_estimation.estimation.estimators import (  # noqa: E402
    DEFAULT_BURNIN_STEPS,
    ObserverEstimator,
    score,
)
from pyadm1ode_estimation.estimation.filter_tuning.datasets import get_dataset  # noqa: E402
from pyadm1ode_estimation.estimation.observer import FeatureSpec, PinnData  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="dropout", choices=sorted(VARIANTS))
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--first-seed", type=int, default=10, help="seeds first-seed .. first-seed+members-1")
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()

    datasets = {k: get_dataset(p) for k, p in TEST_SETS.items() if p.exists()}
    recipe = recipe_of(a.variant)
    features = recipe.pop("features", {})
    data = PinnData.build(datasets["benchmark"], val_frac=0.2, seed=0, feature_spec=FeatureSpec(**features))

    preds: dict[str, list[np.ndarray]] = {k: [] for k in datasets}
    members = []
    for m in range(a.members):
        seed = a.first_seed + m
        t0 = time.perf_counter()
        est, res = ObserverEstimator.train(data, **recipe, epochs=a.epochs, seed=seed)
        est.save(OUT_DIR / f"ensemble_{a.variant}_{seed}.pt")
        row = {"seed": seed, "best_epoch": int(res.best_epoch), "scores": {}}
        for name, ds in datasets.items():
            e = est.for_dataset(ds)
            x = np.stack([e.run(s, i).x_hat for i, s in enumerate(ds.test)])
            preds[name].append(x)
            row["scores"][name] = float(
                np.mean([score(TrajectoryEstimate(s.time, x[i], x[i] * 0), s, DEFAULT_BURNIN_STEPS) for i, s in enumerate(ds.test)])
            )
        members.append(row)
        line = "  ".join(f"{k} {100 * v:5.1f}%" for k, v in row["scores"].items())
        print(f"member seed {seed}  best epoch {res.best_epoch:3d}  {line}  ({time.perf_counter() - t0:.0f}s)", flush=True)

    ens = {}
    for name, ds in datasets.items():
        x = np.mean(preds[name], axis=0)
        per = [score(TrajectoryEstimate(s.time, x[i], x[i] * 0), s, DEFAULT_BURNIN_STEPS) for i, s in enumerate(ds.test)]
        labels = [s.label for s in ds.test]
        ens[name] = {
            "mean": float(np.mean(per)),
            "per_mode": {lb: float(np.mean([p for p, l in zip(per, labels) if l == lb])) for lb in sorted(set(labels))},
            "per_series": per,
            "member_mean": float(np.mean([r["scores"][name] for r in members])),
        }
    print("\nensemble of", a.members, "members:")
    for name, v in ens.items():
        print(f"  {name:<15} ensemble {100 * v['mean']:5.1f} %   mean member {100 * v['member_mean']:5.1f} %")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "ensemble.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"variant": a.variant, "recipe": VARIANTS[a.variant], "members": members, "ensemble": ens}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
