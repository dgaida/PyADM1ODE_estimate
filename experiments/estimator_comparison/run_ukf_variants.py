"""Run both tuned UKF variants over the out-of-distribution test sets.

Nothing is re-tuned. ``theta`` comes from the CMA-ES winner files that produced the
published references, so the only thing that differs from the original evaluation is
the data the filter sees. Re-tuning on the new sets would answer a different question,
namely how well the filter *can* be made to do, rather than how well the one we have
generalises.

The process model does get the new substrate: which substrates are dosed is a known
input in this benchmark, and a filter converting the reported flows through the wrong
characterisation would fail for a reason introduced here rather than for one worth
measuring. See :func:`filter_runners.build_plant`.

    python -m experiments.estimator_comparison.run_ukf_variants --dataset benchmark_ccm

Writes ``ukf_full_reference.npz`` and ``ukf_adcore_reference.npz`` next to the dataset.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyadm1ode_estimation.estimation.estimators import REFERENCE_DIR
from pyadm1ode_estimation.estimation.filter_tuning.cmaes_search import build_reference
from pyadm1ode_estimation.estimation.filter_tuning.datasets import (
    EstimatorDataset,
    Series,
    load_benchmark,
)

REPO = Path(__file__).resolve().parents[2]

#: Tuned configuration per variant, as used for the published references.
WINNERS = {
    "full": REFERENCE_DIR / "cmaes_full_winner.json",
    "adcore": REFERENCE_DIR / "cmaes_adcore_winner.json",
}


def resolve_dataset(name: str) -> EstimatorDataset:
    """The original benchmark, or a variant carrying the original pool."""
    return load_benchmark() if name == "benchmark" else variant_dataset(name)


def variant_dataset(name: str) -> EstimatorDataset:
    """A dataset whose test split is the variant and whose meta describes it.

    The train pool is carried over from the original benchmark. It is not used for a
    test-split reference, but :meth:`EstimatorDataset.make_splits` expects to find
    one, and borrowing the original pool states the intent plainly: the filter was
    tuned there and is being evaluated here.

    Args:
        name: Directory under ``datasets/``.

    Returns:
        EstimatorDataset: Original pool, variant test series, variant meta.
    """
    base = load_benchmark()
    d = REPO / "datasets" / name
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))

    te = np.load(d / "test.npz", allow_pickle=True)
    test = [
        Series(
            measurements=np.asarray(te["measurements"][i], float),
            feed=np.asarray(te["feed_noisy"][i], float),
            time=np.asarray(te["time"][i], float),
            truth=np.asarray(te["states"][i], float),
            switch_days=np.asarray(te["switch_days"][i], float),
            label=str(te["regime"][i]),
            seed=int(te["seed"][i]),
            aux={k: np.asarray(te[k][i], float) for k in ("fostac",) if k in te.files},
        )
        for i in range(len(te["time"]))
    ]
    return EstimatorDataset(name=name, meta=meta, pool=base.pool, test=test)


def main() -> int:
    """Run both UKF variants on one dataset and write their references."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True, help="directory name under datasets/")
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="which split to run. 'val' exists so a fusion weight can be fitted "
        "off the test set; it is only meaningful for the original benchmark, "
        "whose pool a variant merely borrows.",
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument(
        "--variants", nargs="+", default=["full", "adcore"], choices=sorted(WINNERS)
    )
    args = ap.parse_args()

    dataset = resolve_dataset(args.dataset)
    out_dir = REPO / "datasets" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset.test) if args.split == "test" else len(dataset.pool)
    print(
        f"{args.dataset}/{args.split}: {n} Reihen im Pool, Substrat 0 = "
        f"{next(iter(dataset.meta['substrates']))}",
        flush=True,
    )

    for variant in args.variants:
        winner = json.loads(WINNERS[variant].read_text(encoding="utf-8"))
        theta = winner["theta"]
        suffix = "" if args.split == "test" else f"_{args.split}"
        out = out_dir / f"ukf_{variant}{suffix}_reference.npz"
        print(
            f"\n{variant}: theta aus {WINNERS[variant].name} "
            f"(getunt auf '{winner['dataset']}', Stufe-2-Score {winner['stage2_score']:.4f})",
            flush=True,
        )

        t0 = time.time()
        res = build_reference(
            dataset,
            theta,
            variant=variant,
            split=args.split,
            jobs=args.jobs,
            fostac_every_days=winner.get("fostac_every_days"),
            out_npz=out,
        )
        n_ok = len(res["x_hat"]) - len(res.get("failed", []))
        print(
            f"  {n_ok}/{len(res['x_hat'])} Reihen, "
            f"{time.time() - t0:.0f}s -> {out.name}",
            flush=True,
        )
        if res.get("failed"):
            print(f"  FEHLGESCHLAGEN: {res['failed']}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
