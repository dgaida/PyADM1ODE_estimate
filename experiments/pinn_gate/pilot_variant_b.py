"""Pilot for PINN variant B (the amortised GRU observer) — where does it start?

Same question the variant-A gate answered: is the estimator worth anything before
we spend effort tuning it? Variant B differs in that it *does* learn across series,
so the honest reference is not only "hold the prior" but also the reference UKF.

Trains supervised on the training split's windows, selects on the validation split,
and reports the median-over-states NRMSE on **full-length** validation series (the
actual task) next to the do-nothing baseline.

    python experiments/pinn_gate/pilot_variant_b.py --epochs 200 --window 336
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from pyadm1ode_estimation.estimation.deep_learning import Adm1Observer, PinnData


def med_nrmse(pred: np.ndarray, truth: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((pred - truth) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(truth**2, axis=0)) + 1e-12)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument(
        "--window", type=int, default=336, help="training window length [steps]"
    )
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument(
        "--n-random", type=int, default=None, help="random windows per series"
    )
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument(
        "--noise-aug", action="store_true", help="resample sensor noise each batch"
    )
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument(
        "--burnin", type=int, default=48, help="steps dropped from the loss"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--out", default="experiments/output/pinn_gate/pilot_variant_b.json"
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = PinnData.build("benchmark", val_frac=0.2, seed=0)
    print(data.summary(), flush=True)

    tr = data.observer_dataset(
        "train",
        window=args.window,
        stride=args.stride,
        n_random=args.n_random,
        seed=args.seed,
    )
    va = data.observer_dataset("val", window=args.window, stride=args.stride)
    print(
        f"\ntrain windows {tr.features.shape}   val windows {va.features.shape}",
        flush=True,
    )

    # Normalisation from the TRAINING windows only.
    f_tr = torch.tensor(tr.features, dtype=torch.float32)
    s_tr = torch.tensor(tr.states, dtype=torch.float32)
    f_va = torch.tensor(va.features, dtype=torch.float32)
    s_va = torch.tensor(va.states, dtype=torch.float32)
    scale = torch.sqrt((s_tr**2).mean(dim=(0, 1))) + 1e-8

    net = Adm1Observer(
        tr.params,
        tr.x_ref,
        n_features=tr.features.shape[-1],
        hidden=args.hidden,
        num_layers=args.layers,
    )
    print(
        f"observer parameters: {sum(p.numel() for p in net.parameters())}", flush=True
    )

    b = args.burnin

    def loss_of(x, s):
        return (((x[:, b:] - s[:, b:]) / scale) ** 2).mean()

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    best_val, best_state, best_ep = float("inf"), copy.deepcopy(net.state_dict()), -1
    hist = {"train": [], "val": []}

    t_start = time.perf_counter()
    for ep in range(args.epochs):
        net.train()
        order = rng.permutation(len(f_tr))
        losses = []
        for i in range(0, len(order), args.batch):
            idx = torch.as_tensor(order[i : i + args.batch])
            opt.zero_grad()
            lo = loss_of(net(f_tr[idx]), s_tr[idx])
            lo.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(lo.detach()))

        net.eval()
        with torch.no_grad():
            v = float(loss_of(net(f_va), s_va))
        hist["train"].append(float(np.mean(losses)))
        hist["val"].append(v)
        if v < best_val:
            best_val, best_ep = v, ep
            best_state = copy.deepcopy(net.state_dict())
        if ep % max(1, args.epochs // 12) == 0 or ep == args.epochs - 1:
            print(
                f"[{ep:4d}] train {hist['train'][-1]:.4e}  val {v:.4e}"
                f"   ({time.perf_counter() - t_start:.0f}s)",
                flush=True,
            )

    net.load_state_dict(best_state)
    print(f"\nbest val {best_val:.4e} at epoch {best_ep}", flush=True)

    # --- the real task: full-length validation series ---
    net.eval()
    print(f"\n{'series':<16}{'mode':<14}{'prior-hold':>11}{'observer':>11}", flush=True)
    rows = []
    feats_full = data.features("val")
    for k, (s, name) in enumerate(zip(data.series("val"), data.names("val"))):
        truth = np.asarray(s.truth, dtype=float)
        with torch.no_grad():
            pred = net(
                torch.tensor(feats_full[k : k + 1], dtype=torch.float32)
            ).numpy()[0]
        prior = med_nrmse(np.tile(data.x_prior, (len(truth), 1))[b:], truth[b:])
        obs_score = med_nrmse(pred[b:], truth[b:])
        rows.append(
            {
                "series": name,
                "mode": s.label,
                "prior_hold": prior,
                "observer": obs_score,
            }
        )
        print(
            f"{name:<16}{s.label:<14}{100*prior:10.1f}%{100*obs_score:10.1f}%",
            flush=True,
        )

    print("\n=== per mode ===", flush=True)
    print(f"{'mode':<14}{'prior-hold':>11}{'observer':>11}{'verdict':>10}", flush=True)
    verdicts = {}
    for mode in sorted({r["mode"] for r in rows}):
        g = [r for r in rows if r["mode"] == mode]
        ph = float(np.mean([r["prior_hold"] for r in g]))
        ob = float(np.mean([r["observer"] for r in g]))
        verdicts[mode] = bool(ob < ph)
        print(
            f"{mode:<14}{100*ph:10.1f}%{100*ob:10.1f}%{'PASS' if ob < ph else 'FAIL':>10}",
            flush=True,
        )
    overall_p = float(np.mean([r["prior_hold"] for r in rows]))
    overall_o = float(np.mean([r["observer"] for r in rows]))
    print(
        f"\noverall  prior-hold {100*overall_p:.1f}%   observer {100*overall_o:.1f}%"
        f"   -> beats prior in {sum(verdicts.values())}/{len(verdicts)} modes",
        flush=True,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "args": vars(args),
                "history": hist,
                "best_val": best_val,
                "best_epoch": best_ep,
                "rows": rows,
                "verdicts": verdicts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
