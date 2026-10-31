"""Block F2 of the fusion study: fusion rules for the causal PINN and the reference UKF.

Post-processing only: the causal PINN trajectories come from ``rolling_sweep.py``
(block F1), the UKF trajectory and its standard deviation from the dataset.

    python experiments/pinn_v2/fusion_rules.py --scheme W10_C24_S200 --traj hold
    python experiments/pinn_v2/fusion_rules.py --scheme W10_C24_S200 --traj hold \
        --ensemble f1_trajectories f1_trajectories_seed1 f1_trajectories_seed2

Rules
-----
``lin_w`` / ``log_w``   fixed weight ``w`` on the PINN, arithmetic / geometric mean
``block_loso``          one weight per state block, fitted on the other 19 series
``state_loso``          one weight per state, fitted on the other 19 series
``consistency``         ``w_P = c_U / (c_U + c_P)`` with ``c`` the causal 24 h mean of the
                        squared noise-normalised measurement residual of each estimator
``gate_b``              ``w_P = sigmoid(c_U - b)``: the PINN takes over when the UKF stops
                        explaining the measurements
``invvar``              ``w_P = s_U^2 / (s_U^2 + s_P^2)`` with the UKF's own std and the
                        spread of a PINN seed ensemble (needs ``--ensemble``)

All weights are causal (they use measurements up to the current hour only) or static.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import (
    build_plant,
)
from pyadm1ode_estimation.estimation.pinn.data import (
    load_meta,
    load_series,
    scoring_module,
)
from pyadm1ode_estimation.estimation.pinn.measure import MeasurementModel
from pyadm1ode_estimation.estimation.pinn.physics import build_params
from pyadm1ode_estimation.estimation.specs import BLOCK_INDICES

RESULTS = Path(__file__).resolve().parent / "results"
WARM = 72  # evaluation starts after day 3 (hours)
GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


def nrmse_per_state(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """RMSE / RMS(true) per state (the benchmark's definition), rows = time."""
    return np.sqrt(np.mean((pred - true) ** 2, axis=0)) / (
        np.sqrt(np.mean(true**2, axis=0)) + 1e-12
    )


def consistency(
    x41: np.ndarray, y: np.ndarray, meas: MeasurementModel, hours: int = 24
) -> np.ndarray:
    """Causal ``hours``-mean of the squared noise-normalised measurement residual, ``(T,)``."""
    with torch.no_grad():
        r = meas.normalized_residual(
            torch.as_tensor(np.clip(x41[:, :31], 1e-30, None), dtype=torch.float64),
            torch.as_tensor(y, dtype=torch.float64),
        ).numpy()
    c = np.mean(r**2, axis=1)
    csum = np.concatenate([[0.0], np.cumsum(c)])
    k = np.arange(1, len(c) + 1)
    lo = np.maximum(k - hours, 0)
    return (csum[k] - csum[lo]) / (k - lo)


def fuse_lin(p, u, w):
    return w * p + (1.0 - w) * u


def fuse_log(p, u, w):
    return np.exp(
        w * np.log(np.clip(p, 1e-30, None))
        + (1.0 - w) * np.log(np.clip(u, 1e-30, None))
    )


def fit_static_weights(P, U, X, groups: list[list[int]]) -> np.ndarray:
    """Per-group PINN weight on the grid minimising the mean per-state NRMSE over the series."""
    w = np.zeros(X[0].shape[1])
    for idx in groups:
        best, best_w = np.inf, 0.0
        for g in GRID:
            e = np.mean(
                [
                    nrmse_per_state(
                        fuse_lin(p[WARM:, idx], u[WARM:, idx], g), x[WARM:, idx]
                    ).mean()
                    for p, u, x in zip(P, U, X)
                ]
            )
            if e < best:
                best, best_w = e, g
        w[idx] = best_w
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", default="W10_C24_S200")
    ap.add_argument(
        "--traj",
        default="hold",
        help="trajectory key in the npz: hold | adm1 | lag1 | lag3",
    )
    ap.add_argument("--traj-dir", default="f1_trajectories")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument(
        "--ensemble",
        nargs="*",
        default=None,
        help="trajectory dirs of the seed ensemble (incl. the main one)",
    )
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--ukf-traj",
        default=None,
        help="sub-folder of results/ with a re-run UKF (npz keys x, std), e.g. f3_trajectories/periodic_u0.2",
    )
    a = ap.parse_args()

    series = load_series("test", path=a.data_dir)
    meta = load_meta(a.data_dir)
    meas = MeasurementModel.from_meta(build_params(build_plant(meta)), meta)
    sc = scoring_module()

    def load_traj(d):
        return [
            np.load(RESULTS / d / a.scheme / f"series_{s.index:02d}.npz")[
                a.traj
            ].astype(float)
            for s in series
        ]

    P = load_traj(a.traj_dir)
    if a.ukf_traj:
        zs = [
            np.load(RESULTS / a.ukf_traj / f"series_{s.index:02d}.npz") for s in series
        ]
        U = [z["x"].astype(float) for z in zs]
        SU = [z["std"].astype(float) for z in zs]
    else:
        U = [s.ukf_x_hat for s in series]
        SU = [s.ukf_std for s in series]
    X = [s.states for s in series]
    n = len(series)

    ens = None
    if a.ensemble:
        members = [load_traj(d) for d in a.ensemble]
        ens_mean = [np.mean([m[i] for m in members], axis=0) for i in range(n)]
        ens_std = [np.std([m[i] for m in members], axis=0, ddof=1) for i in range(n)]
        ens = (ens_mean, ens_std)

    cU = [consistency(u, s.measurements, meas) for u, s in zip(U, series)]
    cP = [consistency(p, s.measurements, meas) for p, s in zip(P, series)]

    blocks = [list(v) for v in BLOCK_INDICES.values()]
    states = [[i] for i in range(41)]

    rules: dict[str, list[np.ndarray]] = {"ukf": U, "pinn": P}
    for w in (0.25, 0.5, 0.75):
        rules[f"lin_w{w}"] = [fuse_lin(p, u, w) for p, u in zip(P, U)]
        rules[f"log_w{w}"] = [fuse_log(p, u, w) for p, u in zip(P, U)]
    # leave-one-series-out static weights
    for name, groups in (("block_loso", blocks), ("state_loso", states)):
        fused, ws = [], []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            w = fit_static_weights(
                [P[j] for j in others],
                [U[j] for j in others],
                [X[j] for j in others],
                groups,
            )
            ws.append(w)
            fused.append(fuse_lin(P[i], U[i], w))
        rules[name] = fused
        if name == "block_loso":
            block_w_per_series = ws
            wb = np.mean(ws, axis=0)
            block_weights = {b: float(wb[idx[0]]) for b, idx in BLOCK_INDICES.items()}
    # adaptive, causal
    rules["consistency"] = [
        fuse_lin(p, u, (cu / (cu + cp + 1e-12))[:, None])
        for p, u, cu, cp in zip(P, U, cU, cP)
    ]

    def gate(cu, b):
        return (1.0 / (1.0 + np.exp(-(cu - b))))[:, None]

    b_grid = (2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
    for b in b_grid:
        rules[f"gate_b{b:g}"] = [
            fuse_lin(p, u, gate(cu, b)) for p, u, cu in zip(P, U, cU)
        ]
    # threshold chosen on the other 19 series (no in-sample selection)
    fused, chosen = [], []
    for i in range(n):
        others = [j for j in range(n) if j != i]
        errs = [
            np.mean(
                [
                    np.median(
                        nrmse_per_state(
                            fuse_lin(P[j], U[j], gate(cU[j], b))[WARM:], X[j][WARM:]
                        )
                    )
                    for j in others
                ]
            )
            for b in b_grid
        ]
        b_i = b_grid[int(np.argmin(errs))]
        chosen.append(b_i)
        fused.append(fuse_lin(P[i], U[i], gate(cU[i], b_i)))
    rules["gate_loso"] = fused
    gate_b_chosen = float(np.median(chosen))
    print("gate threshold chosen per left-out series:", chosen)
    # block weights when the UKF is consistent, PINN when it is not
    rules["gate_block"] = [
        fuse_lin(
            p, u, w_blk[None, :] + (1.0 - w_blk[None, :]) * gate(cu, gate_b_chosen)
        )
        for p, u, cu, w_blk in zip(P, U, cU, block_w_per_series)
    ]
    if ens is not None:
        em, es = ens
        rules["ens_mean"] = em
        rules["ens_lin_w0.5"] = [fuse_lin(p, u, 0.5) for p, u in zip(em, U)]
        rules["invvar"] = [
            fuse_lin(p, u, np.nan_to_num(su**2 / (su**2 + sp**2 + 1e-30), nan=0.5))
            for p, u, su, sp in zip(em, U, SU, es)
        ]

    # ---- scoring ------------------------------------------------------------
    def score(traj_list, sl):
        out = []
        for t, s in zip(traj_list, series):
            r = sc.score_series(t[sl], s.scoring_dict(sl))
            out.append(
                (
                    r["model"]["nrmse_overall_%"],
                    r["model"]["nrmse_transient_%"],
                    r["vs_ukf"]["rmse_ratio_overall"],
                )
            )
        return np.array(out)

    sl_a, sl_f = slice(WARM, None), slice(0, None)
    table = {}
    base_u = score(U, sl_a)[:, 0]
    base_m = score(rules["lin_w0.5"], sl_a)[:, 0]
    regs = sorted({s.regime for s in series})
    print(
        f"dataset {a.data_dir or 'benchmark'}  scheme {a.scheme}  traj {a.traj}  ensemble {bool(ens)}   window: after day 3 (full in brackets)"
    )
    print(
        f"{'rule':14s} {'NRMSE':>7s} {'(full)':>8s} {'trans':>7s} {'ratio':>6s} {'<UKF':>6s} {'p vs UKF':>9s} {'p vs mean':>10s}  "
        + "  ".join(f"{g[:11]:>11s}" for g in regs)
    )
    for name, tl in rules.items():
        sa, sf = score(tl, sl_a), score(tl, sl_f)
        nr = sa[:, 0]
        p_u = wilcoxon(nr, base_u).pvalue if name != "ukf" else float("nan")
        p_m = wilcoxon(nr, base_m).pvalue if name != "lin_w0.5" else float("nan")
        per = [
            float(np.mean([v for v, s in zip(nr, series) if s.regime == g]))
            for g in regs
        ]
        table[name] = {
            "nrmse_after3d": float(nr.mean()),
            "nrmse_full": float(sf[:, 0].mean()),
            "transient_after3d": float(sa[:, 1].mean()),
            "ratio_vs_ukf": float(sa[:, 2].mean()),
            "better_than_ukf": int((nr < base_u).sum()),
            "p_vs_ukf": float(p_u),
            "p_vs_mean": float(p_m),
            "per_regime": dict(zip(regs, per)),
            "per_series": nr.tolist(),
        }
        print(
            f"{name:14s} {nr.mean():7.2f} {sf[:, 0].mean():8.2f} {sa[:, 1].mean():7.2f} {sa[:, 2].mean():6.2f} "
            f"{int((nr < base_u).sum()):3d}/{n:<2d} {p_u:9.3f} {p_m:10.3f}  "
            + "  ".join(f"{v:11.2f}" for v in per)
        )
    print(
        "\nblock weights on the PINN (LOSO mean):",
        {k: round(v, 2) for k, v in block_weights.items()},
    )
    print(
        "mean consistency (24 h chi2 per channel):  UKF %.2f   PINN %.2f"
        % (
            np.mean([c[WARM:].mean() for c in cU]),
            np.mean([c[WARM:].mean() for c in cP]),
        )
    )
    out = RESULTS / (
        a.out
        or f"f2_fusion_{Path(a.data_dir).name if a.data_dir else 'benchmark'}_{a.scheme}_{a.traj}{'_ens' if ens else ''}{'_' + Path(a.ukf_traj).name if a.ukf_traj else ''}.json"
    )
    out.write_text(
        json.dumps({"rules": table, "block_weights": block_weights}, indent=1),
        encoding="utf-8",
    )
    print("written", out)


if __name__ == "__main__":
    main()
