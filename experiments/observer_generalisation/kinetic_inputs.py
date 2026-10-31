"""Kinetics-aware PINN B (B2) against the kinetics-blind observer (B1), on VALIDATION series.

B2 is the GRU observer with the log factors of the ``k`` most influential kinetic
constants as extra, constant inputs (``FeatureSpec.kinetic_keys``; the ``k`` come
from the training-split screening ``results_final/screening_train.json``). It is
trained with the generator's true factors of the training series and, in the
``noise`` variants, with a per-sequence perturbation of those inputs that
emulates calibration error. B1 is the current default (dropout 0.1).

Everything is scored on the 20 series of the shared validation split; the test
sets stay untouched for the final test. Each network is a 5-seed ensemble (seeds
10-14, the same seeds as the stored B1 members), scored with the benchmark metric
(median-over-states NRMSE after 48 h). Scenarios for the kinetics B2 is given:

``nominal``   no calibration: every factor 1 (``kinetics={}``)
``true``      the true factors of the ``k`` keys (perfect calibration of the screened set)
``noisy_s``   the true factors times ``exp(N(0, s))``, three fixed draws per series
              (the same draw for every ensemble member, as in operation)

Decision rule, fixed before the run: choose B2 if its ensemble under
``noisy_0.07`` (the calibration study's median log error) is better than B1 with
a paired Wilcoxon p < 0.05 over the 20 series, and B2 under ``nominal`` is not
worse than B1 by more than 0.3 points (it must not need calibration to be safe).

    python experiments/observer_generalisation/kinetic_inputs.py train --name b2_n0
    python experiments/observer_generalisation/kinetic_inputs.py train --name b2_n07 --kinetic-noise 0.07
    python experiments/observer_generalisation/kinetic_inputs.py train --name b2all_n0 --all-keys
    python experiments/observer_generalisation/kinetic_inputs.py eval

Checkpoints go to ``results_observer/kinetic/``, the report to
``results_observer/kinetic_inputs.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_variants import OUT_DIR, recipe_of  # noqa: E402

from pyadm1ode_estimation.estimation.estimators import (  # noqa: E402
    DEFAULT_BURNIN_STEPS,
    ObserverEstimator,
    score,
)
from pyadm1ode_estimation.estimation.base import TrajectoryEstimate  # noqa: E402
from pyadm1ode_estimation.estimation.filter_tuning.datasets import load_benchmark  # noqa: E402
from pyadm1ode_estimation.estimation.observer import FeatureSpec, PinnData  # noqa: E402
from pyadm1ode_estimation.estimation.specs import BLOCK_INDICES  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCREENING = REPO / "results_final" / "screening_train.json"
KIN_DIR = OUT_DIR / "kinetic"
SEEDS = (10, 11, 12, 13, 14)
SIGMAS = (0.07, 0.15)
N_DRAWS = 3


def screened_keys(k: int | None = None) -> list[str]:
    rep = json.loads(SCREENING.read_text(encoding="utf-8"))
    return list(rep["top_k"])[: k or len(rep["top_k"])]


def all_keys() -> list[str]:
    ds = load_benchmark(splits=("train",))
    return list(ds.pool[0].aux["kinetic_keys"])


# ---------------------------------------------------------------------------
def cmd_train(a) -> None:
    keys = all_keys() if a.all_keys else screened_keys(a.k)
    data = PinnData.build("benchmark", val_frac=0.2, seed=0, feature_spec=FeatureSpec(kinetic_keys=tuple(keys)))
    print(f"{a.name}: {len(keys)} kinetic inputs {keys}", flush=True)
    KIN_DIR.mkdir(parents=True, exist_ok=True)
    recipe = recipe_of("dropout")
    for seed in a.seeds:
        path = KIN_DIR / f"{a.name}_{seed}.pt"
        if path.exists():
            print(f"  seed {seed}: exists", flush=True)
            continue
        t0 = time.perf_counter()
        est, res = ObserverEstimator.train(
            data, **recipe, epochs=a.epochs, seed=seed, kinetic_noise=a.kinetic_noise
        )
        est.save(path)
        print(f"  seed {seed}: best epoch {res.best_epoch}, val {res.best_val:.4e} ({time.perf_counter() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------------------
def scenario_kinetics(s, keys: list[str], scenario: str, series_idx: int) -> list[dict]:
    """One or more ``kinetics`` dicts for a scenario (several noise draws)."""
    lookup = {str(k): float(f) for k, f in zip(s.aux["kinetic_keys"], s.aux["kinetic_factors"])}
    true = {k: lookup[k] for k in keys}
    if scenario == "nominal":
        return [{}]
    if scenario == "true":
        return [true]
    sigma = float(scenario.split("_")[1])
    out = []
    for d in range(N_DRAWS):
        rng = np.random.default_rng([series_idx, d, int(round(sigma * 1000))])
        out.append({k: v * float(np.exp(rng.normal(0.0, sigma))) for k, v in true.items()})
    return out


def block_errors(x: np.ndarray, truth: np.ndarray, b: int) -> dict[str, float]:
    p, t = x[b:], truth[b:]
    e = np.sqrt(np.mean((p - t) ** 2, axis=0)) / (np.sqrt(np.mean(t**2, axis=0)) + 1e-12)
    return {name: float(np.median(e[list(idx)])) for name, idx in BLOCK_INDICES.items()}


def ensemble_scores(members: list[ObserverEstimator], val, kinetics_for) -> dict:
    """Per series: score of the member-mean trajectory (averaged over noise draws)."""
    per_series, blocks = [], []
    for i, s in enumerate(val):
        kin_list = kinetics_for(s, i)
        sc, bl = [], []
        for kin in kin_list:
            x = np.mean([m.run(s, i, kinetics=kin).x_hat for m in members], axis=0)
            sc.append(score(TrajectoryEstimate(s.time, x, x * 0), s, DEFAULT_BURNIN_STEPS))
            bl.append(block_errors(x, np.asarray(s.truth, float), DEFAULT_BURNIN_STEPS))
        per_series.append(float(np.mean(sc)))
        blocks.append({k: float(np.mean([b[k] for b in bl])) for k in bl[0]})
    member = [
        float(np.mean([score(m.run(s, i, kinetics=kinetics_for(s, i)[0]), s, DEFAULT_BURNIN_STEPS) for i, s in enumerate(val)]))
        for m in members
    ]
    return {
        "mean": float(np.mean(per_series)),
        "per_series": per_series,
        "labels": [s.label for s in val],
        "member_means": member,
        "blocks": {k: float(np.mean([b[k] for b in blocks])) for k in blocks[0]},
    }


def cmd_eval(a) -> None:
    from scipy.stats import wilcoxon

    data = PinnData.build("benchmark", val_frac=0.2, seed=0)
    val = data.series("val")
    ds = data.dataset
    print(f"validation: {len(val)} series", flush=True)

    b1 = [ObserverEstimator.load(OUT_DIR / f"ensemble_dropout_{sd}.pt", ds) for sd in SEEDS]
    for m in b1:  # the stored members must come from the same split
        rec = json.loads(m.recipe["split"])
        assert rec["val"] == data.split.val, "B1 members were trained on another split"
    report = {"b1": ensemble_scores(b1, val, lambda s, i: [{}])}
    print(f"B1 (kinetics-blind)          {100 * report['b1']['mean']:.2f} %", flush=True)

    names = sorted({p.stem.rsplit("_", 1)[0] for p in KIN_DIR.glob("*.pt")})
    for name in names:
        members = [ObserverEstimator.load(KIN_DIR / f"{name}_{sd}.pt", ds) for sd in SEEDS if (KIN_DIR / f"{name}_{sd}.pt").exists()]
        keys = list(members[0].kinetic_keys)
        report[name] = {"keys": keys, "n_members": len(members), "recipe": members[0].recipe, "scenarios": {}}
        for scen in ("nominal", "true", *[f"noisy_{sg}" for sg in SIGMAS]):
            r = ensemble_scores(members, val, lambda s, i, scen=scen: scenario_kinetics(s, keys, scen, i))
            p = wilcoxon(r["per_series"], report["b1"]["per_series"]).pvalue
            r["delta_vs_b1"] = r["mean"] - report["b1"]["mean"]
            r["p_vs_b1"] = float(p)
            report[name]["scenarios"][scen] = r
            print(f"{name:<12} {scen:<11} {100 * r['mean']:6.2f} %   vs B1 {100 * r['delta_vs_b1']:+.2f} pts  p={p:.3f}", flush=True)

    # decision rule (see module docstring)
    decision = {}
    for name in names:
        sc = report[name]["scenarios"]
        ok_cal = sc["noisy_0.07"]["delta_vs_b1"] < 0 and sc["noisy_0.07"]["p_vs_b1"] < 0.05
        ok_nom = sc["nominal"]["delta_vs_b1"] <= 0.003
        decision[name] = {"better_with_calibration": bool(ok_cal), "safe_without_calibration": bool(ok_nom), "choose": bool(ok_cal and ok_nom)}
    report["decision"] = decision
    out = OUT_DIR / "kinetic_inputs.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\ndecision:", json.dumps(decision, indent=1))
    print("wrote", out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--name", required=True)
    t.add_argument("--k", type=int, default=None, help="number of screened keys (default: all of top_k)")
    t.add_argument("--all-keys", action="store_true", help="all 26 kinetic constants as inputs")
    t.add_argument("--kinetic-noise", type=float, default=0.0)
    t.add_argument("--epochs", type=int, default=200)
    t.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    sub.add_parser("eval")
    a = ap.parse_args()
    if a.cmd == "train":
        cmd_train(a)
    else:
        cmd_eval(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
