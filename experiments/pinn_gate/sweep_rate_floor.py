"""Sweep ``rate_floor`` — the knob that decides whether the physics term does any work.

The physics residual is normalised by ``max(|f_i|, rate_floor * scale_i)``. With the
default ``rate_floor = 1.0`` the floor is the state's full magnitude, so a state whose
rate is slow relative to its size (biomass, VFA — the ones no sensor observes) produces
a residual near zero *whatever* the network does with it. The physics term then exerts
almost no force and those states sit on the prior.

A smaller floor keeps the relative residual near 1 for a wrongly-flat slow state, so the
ODE actually pulls it. This sweep measures where that trade lands, per operating mode,
against the do-nothing baseline.

    python experiments/pinn_gate/sweep_rate_floor.py --per-mode 1 --epochs 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from gate_variant_a import med_nrmse, pick

from pyadm1ode_estimation.estimation.deep_learning import PinnData, PinnSmoother


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="benchmark")
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-mode", type=int, default=1)
    ap.add_argument("--days", type=float, default=5.0)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--collocation", type=int, default=100)
    ap.add_argument("--res-clip", type=float, default=3.0)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument(
        "--rate-floors", type=float, nargs="+", default=[1.0, 0.5, 0.2, 0.05, 0.01]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default="experiments/output/pinn_gate/sweep_rate_floor.json"
    )
    args = ap.parse_args()

    data = PinnData.build(args.dataset, val_frac=0.2, seed=0)
    obs = data.obs_model(quasi_steady_gas=True)
    items = pick(data.smoother_inputs(args.split, days=args.days), args.per_mode)
    print(
        f"{len(items)} series, floors={args.rate_floors}, {args.epochs} epochs\n",
        flush=True,
    )

    hdr = f"{'series':<16}{'mode':<13}{'prior':>8}" + "".join(
        f"{'rf=' + str(f):>10}" for f in args.rate_floors
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    for it in items:
        prior = med_nrmse(np.tile(it.x_prior, (len(it.truth), 1)), it.truth)
        r = {"series": it.name, "mode": it.label, "prior_hold": prior, "by_floor": {}}
        cells = []
        for floor in args.rate_floors:
            sm = PinnSmoother(
                params=data.physics_params(),
                obs=obs,
                x_prior=it.x_prior,
                x_scale=it.x_scale,
                quasi_steady_gas=True,
                solve_cation=True,
                res_clip=args.res_clip,
                lambda_phys=args.lambda_phys,
                rate_floor=floor,
                params_at=it.params_at,
                seed=args.seed,
            )
            hist = sm.fit(
                **it.fit_kwargs(),
                n_collocation=args.collocation,
                epochs=args.epochs,
                lr=args.lr,
            )
            score = med_nrmse(sm.estimate(it.obs_times).x_hat, it.truth)
            r["by_floor"][str(floor)] = {
                "med_nrmse": score,
                "data": min(hist["data"]),
                "phys": min(hist["phys"]),
            }
            cells.append(f"{100*score:9.1f}%")
        rows.append(r)
        print(
            f"{it.name:<16}{it.label:<13}{100*prior:7.1f}%" + "".join(cells), flush=True
        )

    print("\n=== mean over series ===", flush=True)
    prior_mean = float(np.mean([r["prior_hold"] for r in rows]))
    print(f"{'prior-hold':<16}{100*prior_mean:7.1f}%", flush=True)
    best_floor, best_score = None, float("inf")
    for floor in args.rate_floors:
        vals = [r["by_floor"][str(floor)]["med_nrmse"] for r in rows]
        wins = sum(
            r["by_floor"][str(floor)]["med_nrmse"] < r["prior_hold"] for r in rows
        )
        m = float(np.mean(vals))
        if m < best_score:
            best_floor, best_score = floor, m
        print(
            f"rate_floor={floor:<7g}{100*m:7.1f}%   beats prior on {wins}/{len(rows)} series",
            flush=True,
        )
    print(
        f"\nbest floor {best_floor} at {100*best_score:.1f}% vs prior-hold {100*prior_mean:.1f}%"
        f"  ->  {'IMPROVES' if best_score < prior_mean else 'no gain'}",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
