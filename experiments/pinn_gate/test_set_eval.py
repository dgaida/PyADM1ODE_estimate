"""The one-time test-set evaluation of PINN variants A and B against the UKF.

Everything upstream (Anpassung 1-3, res_clip, restore_best, the rate_floor sweep,
the variant-B early-stopping recipe) was tuned on train/val only. This script is
the single, final touch of the 20-series test split — run once, numbers go
straight into test_set_results_ukf_vs_pinn.md, no further tuning afterwards.

Variant A: a 5-day representative window per test series (its per-fit cost
~2.5 min rules out a full 60-day fit for 20 series). Variant B: trained once
(early-stopped on the val split) and scored on the FULL 60-day test series --
the actual task.

    python experiments/pinn_gate/test_set_eval.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from pyadm1ode_estimation.estimation.deep_learning import (
    Adm1Observer,
    PinnData,
    PinnSmoother,
)
from pyadm1ode_estimation.estimation.deep_learning.observer_train import (
    pretrain_observer,
)

BURNIN_STEPS = 48  # 48 h, matches the variant-B pilot


def med_nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(truth**2, axis=0)) + 1e-12)))


def by_mode(rows: list[dict], key: str) -> dict[str, float]:
    modes = sorted({r["mode"] for r in rows})
    return {m: float(np.mean([r[key] for r in rows if r["mode"] == m])) for m in modes}


def run_variant_a(data: PinnData, days: float = 5.0) -> list[dict]:
    obs = data.obs_model(quasi_steady_gas=True)
    params = data.physics_params()
    rows = []
    for it in data.smoother_inputs("test", days=days):
        n = len(it.obs_times)
        ukf_win = it.series.aux["ukf_x_hat"][:n]

        sm = PinnSmoother(
            params=params,
            obs=obs,
            x_prior=it.x_prior,
            x_scale=it.x_scale,
            quasi_steady_gas=True,
            solve_cation=True,
            res_clip=3.0,
            rate_floor=0.2,  # best mean on the rate_floor sweep (val, not a proven optimum)
            params_at=it.params_at,
            seed=0,
        )
        t0 = time.perf_counter()
        hist = sm.fit(**it.fit_kwargs(), n_collocation=100, epochs=2000, lr=1e-3)
        secs = time.perf_counter() - t0
        pred = sm.estimate(it.obs_times).x_hat

        rows.append(
            {
                "series": it.name,
                "mode": it.label,
                "prior_hold": med_nrmse(np.tile(it.x_prior, (n, 1)), it.truth),
                "x0_oracle": med_nrmse(np.tile(it.truth[0], (n, 1)), it.truth),
                "pinn_a": med_nrmse(pred, it.truth),
                "ukf": med_nrmse(ukf_win, it.truth),
                "best_data_loss": min(hist["data"]),
                "seconds": secs,
            }
        )
        print(
            f"  A  {it.name:<16} {it.label:<13} "
            f"prior={100*rows[-1]['prior_hold']:5.1f}%  "
            f"UKF={100*rows[-1]['ukf']:5.1f}%  "
            f"A={100*rows[-1]['pinn_a']:5.1f}%  "
            f"data={rows[-1]['best_data_loss']:.3f}  ({secs:.0f}s)",
            flush=True,
        )
    return rows


def run_variant_b(data: PinnData) -> tuple[list[dict], dict]:
    tr = data.observer_dataset("train", window=336, stride=336)
    va = data.observer_dataset("val", window=336, stride=336)

    torch.manual_seed(0)
    net = Adm1Observer(
        tr.params, tr.x_ref, n_features=tr.features.shape[-1], hidden=64, num_layers=2
    )
    t0 = time.perf_counter()
    res = pretrain_observer(
        net,
        tr,
        val_dataset=va,
        epochs=200,
        batch_size=32,
        lr=1e-3,
        burnin=BURNIN_STEPS,
        patience=40,
        restore_best=True,
        seed=0,
    )
    train_secs = time.perf_counter() - t0
    print(
        f"  B  trained: best epoch {res.best_epoch}, best val {res.best_val:.4e}, "
        f"{train_secs:.0f}s",
        flush=True,
    )

    net.eval()
    feats_full = data.features("test")
    rows = []
    for k, (s, name) in enumerate(zip(data.series("test"), data.names("test"))):
        truth = np.asarray(s.truth, dtype=float)
        with torch.no_grad():
            pred = net(
                torch.tensor(feats_full[k : k + 1], dtype=torch.float32)
            ).numpy()[0]
        ukf = np.asarray(s.aux["ukf_x_hat"], dtype=float)
        b = BURNIN_STEPS
        rows.append(
            {
                "series": name,
                "mode": s.label,
                "prior_hold": med_nrmse(
                    np.tile(data.x_prior, (len(truth), 1))[b:], truth[b:]
                ),
                "pinn_b": med_nrmse(pred[b:], truth[b:]),
                "ukf": med_nrmse(ukf[b:], truth[b:]),
            }
        )
        print(
            f"  B  {name:<16} {s.label:<13} "
            f"prior={100*rows[-1]['prior_hold']:5.1f}%  "
            f"UKF={100*rows[-1]['ukf']:5.1f}%  "
            f"B={100*rows[-1]['pinn_b']:5.1f}%",
            flush=True,
        )
    meta = {
        "best_epoch": res.best_epoch,
        "best_val": res.best_val,
        "stopped_early": res.stopped_early,
        "train_seconds": train_secs,
    }
    return rows, meta


def main() -> int:
    data = PinnData.build("benchmark", val_frac=0.2, seed=0)
    print(data.summary(), flush=True)

    print("\n=== Variant A (5-day window per test series) ===", flush=True)
    rows_a = run_variant_a(data)

    print(
        "\n=== Variant B (trained once, scored on full 60-day series) ===", flush=True
    )
    rows_b, meta_b = run_variant_b(data)

    print("\n=== per mode (mean over series) ===", flush=True)
    print(
        f"{'mode':<14}{'prior':>8}{'UKF(A win)':>12}{'A':>8}{'UKF(full)':>11}{'B':>8}",
        flush=True,
    )
    modes = sorted({r["mode"] for r in rows_a})
    pa, ua, aa = (
        by_mode(rows_a, "prior_hold"),
        by_mode(rows_a, "ukf"),
        by_mode(rows_a, "pinn_a"),
    )
    ub, bb = by_mode(rows_b, "ukf"), by_mode(rows_b, "pinn_b")
    for m in modes:
        print(
            f"{m:<14}{100*pa[m]:7.1f}%{100*ua[m]:11.1f}%{100*aa[m]:7.1f}%"
            f"{100*ub[m]:10.1f}%{100*bb[m]:7.1f}%",
            flush=True,
        )

    overall = lambda rows, key: float(np.mean([r[key] for r in rows]))
    print(
        f"\n{'overall':<14}{100*overall(rows_a,'prior_hold'):7.1f}%"
        f"{100*overall(rows_a,'ukf'):11.1f}%{100*overall(rows_a,'pinn_a'):7.1f}%"
        f"{100*overall(rows_b,'ukf'):10.1f}%{100*overall(rows_b,'pinn_b'):7.1f}%",
        flush=True,
    )

    out = Path("experiments/output/pinn_gate/test_set_eval.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"rows_a": rows_a, "rows_b": rows_b, "meta_b": meta_b}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
