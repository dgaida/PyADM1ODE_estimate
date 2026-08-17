"""Stage-0 gate for PINN variant A: is the fit worth anything at all?

Before any architecture or hyperparameter search, variant A has to clear a floor:
**beat the trivial baseline of holding the training prior constant**. If it does
not, an architecture comparison measures the physics/optimisation defect, not the
architecture.

The run compares, per operating mode on the validation split:

* ``prior-hold``  — x(t) = x_prior for all t (the do-nothing baseline; the floor).
* ``x0-oracle``   — x(t) = true x(t0) held (context: how much of the error is just
  "the state barely moves over the window", not something a fit can claim credit for).
* ``A/nominal``   — variant A with the reference plant's nominal feed.
* ``A/feed``      — variant A with the series' own feed, time-varying across the
  collocation grid.

Reported per series: median-over-states NRMSE, plus the **best** data loss over the
run (the mean squared standardised residual on the *measured* channels). If that
never drops below its epoch-0 value, training did not improve the fit at all — it
only walked away from the prior, and no architecture choice can be read off it.

    python experiments/pinn_gate/gate_variant_a.py --per-mode 2 --days 5 --epochs 2000
    python experiments/pinn_gate/gate_variant_a.py --res-clip 3 --mask-channel pH
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pyadm1ode_estimation.estimation.deep_learning import PinnData, PinnSmoother


def med_nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    """Median over states of RMSE / RMS(truth) — the benchmark's headline shape."""
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(truth**2, axis=0)) + 1e-12)))


def pick(items, per_mode: int):
    """First ``per_mode`` series of each operating mode, so every mode is covered."""
    seen: dict[str, int] = {}
    out = []
    for it in items:
        if seen.get(it.label, 0) < per_mode:
            seen[it.label] = seen.get(it.label, 0) + 1
            out.append(it)
    return out


def fit_variant_a(
    data: PinnData, it, *, feed_matched: bool, args
) -> tuple[np.ndarray, float, float, float]:
    """One variant-A fit; returns (prediction, best data loss, final data loss, seconds).

    The *best* data loss matters as much as the final one: if it never drops below
    its value at epoch 0, the training did not improve the fit at all — it only
    moved away from the prior.
    """
    sm = PinnSmoother(
        params=data.physics_params(),
        obs=data.obs_model(quasi_steady_gas=True),
        x_prior=it.x_prior,
        x_scale=it.x_scale,
        hidden_layers=tuple(args.hidden),
        quasi_steady_gas=True,
        solve_cation=args.solve_cation,
        lambda_phys=args.lambda_phys,
        lambda_prior=args.lambda_prior,
        rate_floor=args.rate_floor,
        res_clip=args.res_clip,
        params_at=it.params_at if feed_matched else None,
        seed=args.seed,
    )
    kwargs = it.fit_kwargs()
    if args.mask_channel:
        y = np.array(kwargs["obs_values"], dtype=float, copy=True)
        for name in args.mask_channel:
            y[:, it.channel_names.index(name)] = np.nan
        kwargs["obs_values"] = y
    t0 = time.perf_counter()
    hist = sm.fit(
        **kwargs, n_collocation=args.collocation, epochs=args.epochs, lr=args.lr
    )
    secs = time.perf_counter() - t0
    return sm.estimate(it.obs_times).x_hat, min(hist["data"]), hist["data"][-1], secs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="benchmark")
    ap.add_argument("--split", default="val")
    ap.add_argument("--per-mode", type=int, default=2, help="series per operating mode")
    ap.add_argument("--days", type=float, default=5.0, help="window length [d]")
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--collocation", type=int, default=100)
    ap.add_argument("--hidden", type=int, nargs="+", default=[64, 64, 64])
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--lambda-prior", type=float, default=1.0)
    ap.add_argument("--rate-floor", type=float, default=1.0)
    ap.add_argument(
        "--res-clip",
        type=float,
        default=None,
        help="Huber threshold [sigma] on the standardised residuals (see _robust_sq)",
    )
    ap.add_argument(
        "--solve-cation",
        action="store_true",
        help="quasi-steady charge balance: net predicts pH, S_cation is solved (Adjustment 3)",
    )
    ap.add_argument(
        "--mask-channel",
        action="append",
        default=[],
        help="gate a sensor out of the data loss, e.g. --mask-channel pH (repeatable)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--out", default="experiments/output/pinn_gate/gate_variant_a.json")
    ap.add_argument(
        "--skip-nominal", action="store_true", help="only run the feed-matched fit"
    )
    args = ap.parse_args()

    data = PinnData.build(args.dataset, val_frac=args.val_frac, seed=args.split_seed)
    print(data.summary(), flush=True)
    items = pick(data.smoother_inputs(args.split, days=args.days), args.per_mode)
    print(
        f"\n{len(items)} series x {args.days} d, {args.epochs} epochs, hidden={args.hidden}\n",
        flush=True,
    )

    hdr = (
        f"{'series':<18}{'mode':<14}{'prior-hold':>11}{'x0-oracle':>11}"
        f"{'A/nominal':>11}{'A/feed':>9}{'best data':>14}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    rows = []
    for it in items:
        r = {
            "series": it.name,
            "mode": it.label,
            "prior_hold": med_nrmse(np.tile(it.x_prior, (len(it.truth), 1)), it.truth),
            "x0_oracle": med_nrmse(np.tile(it.truth[0], (len(it.truth), 1)), it.truth),
        }
        if not args.skip_nominal:
            pred, dbest, ldata, secs = fit_variant_a(
                data, it, feed_matched=False, args=args
            )
            r["a_nominal"], r["a_nominal_data"], r["a_nominal_s"] = (
                med_nrmse(pred, it.truth),
                ldata,
                secs,
            )
        pred, dbest, ldata, secs = fit_variant_a(data, it, feed_matched=True, args=args)
        r["a_feed"], r["a_feed_data"], r["a_feed_data_best"], r["a_feed_s"] = (
            med_nrmse(pred, it.truth),
            ldata,
            dbest,
            secs,
        )
        rows.append(r)
        print(
            f"{r['series']:<18}{r['mode']:<14}{100*r['prior_hold']:10.1f}%{100*r['x0_oracle']:10.1f}%"
            f"{100*r.get('a_nominal', float('nan')):10.1f}%{100*r['a_feed']:8.1f}%{r['a_feed_data_best']:14.3e}",
            flush=True,
        )

    print("\n=== per mode (mean over series) ===", flush=True)
    print(
        f"{'mode':<14}{'prior-hold':>11}{'x0-oracle':>11}{'A/nominal':>11}{'A/feed':>9}{'verdict':>12}",
        flush=True,
    )
    verdicts = {}
    for mode in sorted({r["mode"] for r in rows}):
        g = [r for r in rows if r["mode"] == mode]
        ph, af = np.mean([r["prior_hold"] for r in g]), np.mean(
            [r["a_feed"] for r in g]
        )
        an = np.mean([r.get("a_nominal", np.nan) for r in g])
        ok = af < ph
        verdicts[mode] = bool(ok)
        print(
            f"{mode:<14}{100*ph:10.1f}%{100*np.mean([r['x0_oracle'] for r in g]):10.1f}%"
            f"{100*an:10.1f}%{100*af:8.1f}%{'PASS' if ok else 'FAIL':>12}",
            flush=True,
        )

    data_ok = float(np.median([r["a_feed_data_best"] for r in rows]))
    print(
        f"\nGATE 1 (beats prior-hold in every mode): {'PASS' if all(verdicts.values()) else 'FAIL'}"
        f"\nGATE 2 (median data loss O(1-10)):       {'PASS' if data_ok < 10 else 'FAIL'}  (median {data_ok:.3e})",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"args": vars(args), "rows": rows, "verdicts": verdicts}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
