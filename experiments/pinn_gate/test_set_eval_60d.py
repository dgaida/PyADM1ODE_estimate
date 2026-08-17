"""Variant A on the FULL 60-day test windows — the comparison section D could not make.

Section D ran variant A on a 5-day slice, so A and B were not on the same footing.
This closes that gap: A on all 20 test series over the full 60 days, against the
same UKF reference and the same prior-hold baseline that B was scored on.

Configuration comes from the VAL architecture search (`search_arch_60d.py`):
**64x3, 400 collocation points, 2000 epochs**. That search covered a 60x parameter
range (2.3k - 141k) and found the differences to be within series-to-series noise,
so the smallest net with the best fit quality and the highest win rate was kept.

> Test-set discipline: this is the SECOND touch of the test split (the first was
> section D). The architecture was selected on VAL only. Any further iteration on
> these 20 series would start eroding the held-out set — treat this as final.

    python experiments/pinn_gate/test_set_eval_60d.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyadm1ode_estimation.estimation.deep_learning import PinnData, PinnSmoother

BURNIN_STEPS = 48  # same burn-in as variant B, so the two are directly comparable


def med_nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(truth**2, axis=0)) + 1e-12)))


def by_mode(rows: list[dict], key: str) -> dict[str, float]:
    modes = sorted({r["mode"] for r in rows})
    return {m: float(np.mean([r[key] for r in rows if r["mode"] == m])) for m in modes}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    ap.add_argument("--collocation", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--out", default="experiments/output/pinn_gate/test_set_eval_60d.json"
    )
    args = ap.parse_args()

    data = PinnData.build("benchmark", val_frac=0.2, seed=0)
    obs = data.obs_model(quasi_steady_gas=True)
    params = data.physics_params()
    b = BURNIN_STEPS

    print(
        f"variant A on 20 test series x 60 d | hidden={args.hidden} "
        f"coll={args.collocation} epochs={args.epochs}\n",
        flush=True,
    )
    print(
        f"{'series':<17}{'mode':<13}{'prior':>8}{'UKF':>8}{'A':>8}{'data':>8}{'min':>7}",
        flush=True,
    )
    print("-" * 69, flush=True)

    rows = []
    for it in data.smoother_inputs("test", days=60.0):
        sm = PinnSmoother(
            params=params,
            obs=obs,
            x_prior=it.x_prior,
            x_scale=it.x_scale,
            hidden_layers=tuple(args.hidden),
            quasi_steady_gas=True,
            solve_cation=True,
            res_clip=3.0,
            rate_floor=0.2,
            params_at=it.params_at,
            seed=0,
        )
        t0 = time.perf_counter()
        hist = sm.fit(
            **it.fit_kwargs(),
            n_collocation=args.collocation,
            epochs=args.epochs,
            lr=args.lr,
        )
        mins = (time.perf_counter() - t0) / 60.0
        pred = sm.estimate(it.obs_times).x_hat
        ukf = np.asarray(it.series.aux["ukf_x_hat"], dtype=float)

        rows.append(
            {
                "series": it.name,
                "mode": it.label,
                "prior_hold": med_nrmse(
                    np.tile(it.x_prior, (len(it.truth), 1))[b:], it.truth[b:]
                ),
                "pinn_a": med_nrmse(pred[b:], it.truth[b:]),
                "ukf": med_nrmse(ukf[b:], it.truth[b:]),
                "best_data_loss": min(hist["data"]),
                "minutes": mins,
            }
        )
        r = rows[-1]
        print(
            f"{r['series']:<17}{r['mode']:<13}{100*r['prior_hold']:7.1f}%"
            f"{100*r['ukf']:7.1f}%{100*r['pinn_a']:7.1f}%{r['best_data_loss']:8.2f}"
            f"{mins:7.1f}",
            flush=True,
        )

    print(f"\n{'mode':<13}{'prior':>8}{'UKF':>8}{'A':>8}", flush=True)
    for m in sorted({r["mode"] for r in rows}):
        p, u, a = (by_mode(rows, k)[m] for k in ("prior_hold", "ukf", "pinn_a"))
        print(f"{m:<13}{100*p:7.1f}%{100*u:7.1f}%{100*a:7.1f}%", flush=True)

    ov = lambda k: float(np.mean([r[k] for r in rows]))
    print(
        f"\n{'overall':<13}{100*ov('prior_hold'):7.1f}%{100*ov('ukf'):7.1f}%"
        f"{100*ov('pinn_a'):7.1f}%",
        flush=True,
    )
    print(
        f"\nA beats prior on {sum(r['pinn_a'] < r['prior_hold'] for r in rows)}/20, "
        f"beats UKF on {sum(r['pinn_a'] < r['ukf'] for r in rows)}/20",
        flush=True,
    )
    print(
        f"median best data loss: {np.median([r['best_data_loss'] for r in rows]):.2f} "
        f"(noise floor 0.57)",
        flush=True,
    )
    print(f"total wall time: {sum(r['minutes'] for r in rows) / 60:.1f} h", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"args": vars(args), "rows": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
