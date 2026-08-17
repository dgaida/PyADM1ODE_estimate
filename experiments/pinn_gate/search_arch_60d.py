"""Architecture search for variant A on the FULL 60-day window — on VAL only.

Section D evaluated variant A on a 5-day slice, which was a compromise: the
network was sized for a short, comparatively calm stretch. A 60-day series
carries 5 load switches and a far richer trajectory, so the open question is
whether the net has to grow, and which shape works best.

Measured cost basis (this machine): a 60-day fit at 400 collocation points and
2000 epochs takes ~4.1 min — an earlier ~30-40 min extrapolation was wrong, so a
proper search is affordable.

Deliberately 2 series per mode (8 total), not 1: the earlier 4-series gate
measurement produced a verdict that did not survive the 20-series test set. More
series beats more configs.

    python experiments/pinn_gate/search_arch_60d.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyadm1ode_estimation.estimation.deep_learning import PinnData, PinnSmoother

# (label, hidden_layers, n_collocation)
CONFIGS = [
    ("64x3 (Basis)", (64, 64, 64), 400),
    ("32x2 (kleiner)", (32, 32), 400),
    ("128x3 (breiter)", (128, 128, 128), 400),
    ("64x5 (tiefer)", (64, 64, 64, 64, 64), 400),
    ("256x3 (viel breiter)", (256, 256, 256), 400),
    ("64x3 + 800 coll", (64, 64, 64), 800),
]


def med_nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(truth**2, axis=0)) + 1e-12)))


def pick(items, per_mode: int):
    seen: dict[str, int] = {}
    out = []
    for it in items:
        if seen.get(it.label, 0) < per_mode:
            seen[it.label] = seen.get(it.label, 0) + 1
            out.append(it)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-mode", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--out", default="experiments/output/pinn_gate/search_arch_60d.json"
    )
    args = ap.parse_args()

    data = PinnData.build("benchmark", val_frac=0.2, seed=0)
    obs = data.obs_model(quasi_steady_gas=True)
    params = data.physics_params()
    items = pick(data.smoother_inputs("val", days=60.0), args.per_mode)
    print(
        f"{len(items)} val series x 60 d, {len(CONFIGS)} configs, {args.epochs} epochs\n",
        flush=True,
    )

    priors = {
        it.name: med_nrmse(np.tile(it.x_prior, (len(it.truth), 1)), it.truth)
        for it in items
    }
    print(
        "prior-hold per series:",
        {k: f"{100*v:.1f}%" for k, v in priors.items()},
        flush=True,
    )
    print(f"prior-hold mean: {100*np.mean(list(priors.values())):.1f}%\n", flush=True)

    hdr = f"{'config':<22}{'params':>8}{'medNRMSE':>11}{'data':>9}{'beats prior':>13}{'min':>7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    results = []
    for label, hidden, n_coll in CONFIGS:
        scores, losses, t_start = [], [], time.perf_counter()
        for it in items:
            sm = PinnSmoother(
                params=params,
                obs=obs,
                x_prior=it.x_prior,
                x_scale=it.x_scale,
                hidden_layers=hidden,
                quasi_steady_gas=True,
                solve_cation=True,
                res_clip=3.0,
                rate_floor=0.2,
                params_at=it.params_at,
                seed=0,
            )
            hist = sm.fit(
                **it.fit_kwargs(), n_collocation=n_coll, epochs=args.epochs, lr=args.lr
            )
            scores.append(med_nrmse(sm.estimate(it.obs_times).x_hat, it.truth))
            losses.append(min(hist["data"]))
        n_par = sum(p.numel() for p in sm.net.parameters())
        wins = sum(s < priors[it.name] for s, it in zip(scores, items))
        mins = (time.perf_counter() - t_start) / 60.0
        results.append(
            {
                "label": label,
                "hidden": list(hidden),
                "n_collocation": n_coll,
                "params": n_par,
                "med_nrmse": float(np.mean(scores)),
                "per_series": [float(s) for s in scores],
                "data_loss": float(np.median(losses)),
                "beats_prior": int(wins),
                "minutes": mins,
            }
        )
        print(
            f"{label:<22}{n_par:>8}{100*np.mean(scores):>10.1f}%{np.median(losses):>9.3f}"
            f"{wins:>9}/{len(items)}{mins:>7.1f}",
            flush=True,
        )

    best = min(results, key=lambda r: r["med_nrmse"])
    print(
        f"\nbest by mean medNRMSE: {best['label']} "
        f"({100*best['med_nrmse']:.1f}%, beats prior {best['beats_prior']}/{len(items)})",
        flush=True,
    )
    print(
        f"prior-hold reference : {100*np.mean(list(priors.values())):.1f}%", flush=True
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "priors": priors,
                "results": results,
                "series": [it.name for it in items],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
