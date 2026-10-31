"""Score the final test from the stored trajectories and write the report.

Implements the metric set of ``PyADM1ODE_estimate_local/reports/Metriken für den
Schätzervergleich.md`` (H, A1-A4, K1, C1-C5, T1-T6, G1-G3, S1-S8, E1-E4) on the
window ``config.window_hours``. Runs on a partial result set as well; every table
states how many series it covers, and the decisions are only taken once the
inputs are complete.

    python experiments/final_test/score.py            # -> results_final/report/
"""

from __future__ import annotations

import itertools
import sys
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
from scipy.stats import norm, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C

CFG = C.config()
A, B = C.window()
STAT = CFG["statistics"]
MET = CFG["metrics"]
CONDITIONS = list(CFG["conditions"])
DATASETS = C.dataset_names()
SHIFTED = [d for d in DATASETS if d != "benchmark"]
BASE_VARIANTS = ["U1", "U2", "P1", "B1", "B2", "R1"]
FUSIONS = ["F1", "F2"]
VARIANTS = BASE_VARIANTS + FUSIONS + ["R0"]
WITH_SIGMA = ["U1", "U2", "B1", "B2", "F1", "F2"]
RNG_SEED = int(STAT["bootstrap_seed"])
#: T4 failure threshold relative to the prior-hold error over the last days.
FAILURE_FACTOR = 2.0
N_BOOT = int(STAT["bootstrap"])


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
@cache
def state_names() -> list[str]:
    m = C.dataset("benchmark").meta["state_index_map"]
    return [m[str(j)] for j in range(41)]


@cache
def j38() -> np.ndarray:
    ex = set(MET["exclude_from_geometric"])
    return np.array([j for j, n in enumerate(state_names()) if n not in ex])


@cache
def blocks() -> dict[str, list[int]]:
    from pyadm1ode_estimation.estimation.specs import BLOCK_INDICES

    return {k: list(v) for k, v in BLOCK_INDICES.items()}


@cache
def x_prior() -> np.ndarray:
    from pyadm1ode_estimation.estimation.observer import PinnData

    return np.asarray(
        PinnData.build(C.dataset("benchmark"), val_frac=0.2, seed=0).x_prior, float
    )


@cache
def measurement_model(ds: str):
    from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import build_plant
    from pyadm1ode_estimation.estimation.pinn.measure import MeasurementModel
    from pyadm1ode_estimation.estimation.pinn.physics import build_params

    meta = C.dataset(ds).meta
    return MeasurementModel.from_meta(build_params(build_plant(meta)), meta)


def sensors_of(ds: str, x: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return measurement_model(ds).h(torch.as_tensor(np.asarray(x, float))).numpy()


@cache
def series_info(ds: str, i: int) -> dict:
    s = C.dataset(ds).test[i]
    return {
        "truth": np.asarray(s.truth, float),
        "label": s.label,
        "switch_days": np.asarray(s.switch_days, float),
        "time": np.asarray(s.time, float),
    }


def base_traj(cond: str, ds: str, v: str, i: int) -> dict | None:
    """Stored trajectory of a base variant; B1 is the same in every condition."""
    if v == "B1":
        cond = "nominal"
    if cond != "nominal" and not _runs(cond, ds, v):
        return None
    return C.load_traj(cond, ds, v, i)


def _runs(cond: str, ds: str, v: str) -> bool:
    return (cond, ds, v) in set(C.condition_jobs())


def failed(cond: str, ds: str, v: str, i: int) -> bool:
    if v == "B1":
        cond = "nominal"
    return C.failed_path(C.traj_path(cond, ds, v, i)).exists()


@lru_cache(maxsize=1)
def fusion_scales() -> dict:
    """``s_b`` per fusion and state block, fitted on the validation split (cached)."""
    p = C.fusion_scales_path()
    if p.exists():
        rec = C.read_json(p)
        C.check_hash(rec["config_hash"], p)
        return rec["scales"]
    from jobs import _members, val_series

    _, val = val_series()
    z = np.load(
        C.REPO / "datasets/benchmark/ukf_full_val_reference.npz", allow_pickle=True
    )
    w = float(CFG["fusion"]["weight_ukf"])
    rng_keys = C.calibration_keys()
    scales = {}
    for fname, bvar in (("F1", "B1"), ("F2", "B2")):
        members = _members(bvar, "benchmark")
        acc = {b: [] for b in blocks()}
        for i, s in enumerate(val):
            truth = np.asarray(s.truth, float)
            if bvar == "B2":  # emulate a calibration of the usual accuracy (sigma 0.07)
                lookup = {
                    str(k): float(f)
                    for k, f in zip(s.aux["kinetic_keys"], s.aux["kinetic_factors"])
                }
                r = np.random.default_rng([i, 0, 70])
                kin = {
                    k: lookup[k] * float(np.exp(r.normal(0.0, 0.07))) for k in rng_keys
                }
            else:
                kin = {}
            xs = np.stack([m.run(s, i, kinetics=kin).x_hat for m in members])
            xb, sb = xs.mean(0), xs.std(0)
            xu, su = np.asarray(z["ukf_x_hat"][i], float), np.asarray(
                z["ukf_std"][i], float
            )
            xf = w * xu + (1 - w) * xb
            sf = np.sqrt(w**2 * su**2 + (1 - w) ** 2 * sb**2)
            zz = ((xf - truth) / np.maximum(sf, 1e-30))[A:B] ** 2
            for b, idx in blocks().items():
                v = zz[:, idx]
                acc[b].append(v[np.isfinite(v)])
        scales[fname] = {
            b: float(np.sqrt(np.mean(np.concatenate(v)))) for b, v in acc.items()
        }
    C.atomic_write_json(p, {"config_hash": C.config_hash(), "scales": scales})
    return scales


def traj(cond: str, ds: str, v: str, i: int) -> dict | None:
    """Trajectory of any variant (base, fusion, R0) as ``{"x", "std", "wall_s"}``."""
    if v == "R0":
        info = series_info(ds, i)
        x = np.tile(x_prior(), (len(info["time"]), 1))
        return {"x": x, "std": None, "wall_s": 0.0}
    if v in FUSIONS:
        u_name, b_name = CFG["fusion"][v]
        u, b = base_traj(cond, ds, u_name, i), base_traj(cond, ds, b_name, i)
        if u is None or b is None:
            return None
        w = float(CFG["fusion"]["weight_ukf"])
        x = w * u["x_hat"].astype(float) + (1 - w) * b["x_hat"].astype(float)
        sf = np.sqrt(
            w**2 * u["std"].astype(float) ** 2
            + (1 - w) ** 2 * b["std"].astype(float) ** 2
        )
        scale = np.ones(41)
        for blk, idx in blocks().items():
            scale[idx] = fusion_scales()[v][blk]
        return {
            "x": x,
            "std": sf * scale,
            "wall_s": u["meta"]["wall_s"] + b["meta"]["wall_s"],
        }
    t = base_traj(cond, ds, v, i)
    if t is None:
        return None
    std = t["std"].astype(float)
    has_sigma = v in WITH_SIGMA and np.any(np.isfinite(std) & (std > 0))
    return {
        "x": t["x_hat"].astype(float),
        "std": std if has_sigma else None,
        "wall_s": t["meta"]["wall_s"],
    }


def available(cond: str, ds: str, v: str) -> bool:
    if v in ("R0",):
        return True
    if v in FUSIONS:
        u_name, b_name = CFG["fusion"][v]
        return available(cond, ds, u_name) and available(cond, ds, b_name)
    return _runs("nominal" if v == "B1" else cond, ds, v)


# ---------------------------------------------------------------------------
# Per-series metrics
# ---------------------------------------------------------------------------
def transient_mask(time: np.ndarray, switch_days: np.ndarray) -> np.ndarray:
    m = np.zeros(len(time), bool)
    tau = float(MET["transient_hours"]) / 24.0
    for ts in np.atleast_1d(switch_days):
        if np.isfinite(ts):
            m |= (time >= ts) & (time < ts + tau)
    m[:A] = False
    m[B:] = False
    return m


def crps_gauss(e: np.ndarray, s: np.ndarray) -> np.ndarray:
    u = e / s
    return s * (u * (2 * norm.cdf(u) - 1) + 2 * norm.pdf(u) - 1 / np.sqrt(np.pi))


def series_metrics(
    t: dict, ds: str, i: int, r1: dict | None, is_r0: bool = False
) -> dict:
    info = series_info(ds, i)
    truth = info["truth"]
    x = t["x"]
    if not np.all(np.isfinite(x[A:B])):
        return {"failed": True, "reason": "non-finite"}
    e = x[A:B] - truth[A:B]
    c = np.sqrt(np.mean(truth[A:B] ** 2, axis=0)) + 1e-12
    rmse = np.sqrt(np.mean(e**2, axis=0))
    nrmse = rmse / c
    out = {"failed": False, "E": float(np.median(nrmse)), "rmse": rmse, "c": c}
    # failure rule T4: worse than holding the prior over the last days of the window
    n_last = int(MET["failure_last_days"]) * 24
    last = slice(B - A - n_last, B - A)
    r0_last = np.median(
        np.sqrt(np.mean((x_prior()[None] - truth[A:B][last]) ** 2, axis=0)) / c
    )
    # Divergence: clearly worse than doing nothing over the last days. "Worse than
    # the prior" alone flags healthy filters on series that happen to end near the
    # training prior (seen on a stable_low series: 22 % vs 15 %, 17 % over the
    # window), so the threshold is FAILURE_FACTOR x the prior-hold error. R0 is the
    # reference of this rule and cannot fail it.
    if (
        not is_r0
        and float(np.median(np.sqrt(np.mean(e[last] ** 2, axis=0)) / c))
        > FAILURE_FACTOR * r0_last
    ):
        out["failed"], out["reason"] = (
            True,
            f"more than {FAILURE_FACTOR:g}x the prior-hold error at the end",
        )
    out["blocks"] = {b: float(np.median(nrmse[idx])) for b, idx in blocks().items()}
    tm = transient_mask(info["time"], info["switch_days"])[A:B]
    if tm.any():
        out["T1"] = float(np.median(np.sqrt(np.mean(e[tm] ** 2, axis=0)) / c))
        out["T2"] = float(np.median(np.max(np.abs(e[tm]), axis=0) / c))
    if r1 is not None and np.all(np.isfinite(r1["x"][A:B])):
        rmse_r1 = np.sqrt(np.mean((r1["x"][A:B] - truth[A:B]) ** 2, axis=0)) + 1e-30
        ss = 1.0 - rmse / rmse_r1
        out["SS_median"], out["SS_n_gt_0.2"] = float(np.median(ss)), int(
            np.sum(ss > 0.2)
        )
    y_hat, y_true = sensors_of(ds, x[A:B]), sensors_of(ds, truth[A:B])
    out["pH_rmse"] = float(np.sqrt(np.mean((y_hat[:, 3] - y_true[:, 3]) ** 2)))
    out["Qch4_rel"] = float(
        np.sqrt(np.mean((y_hat[:, 1] - y_true[:, 1]) ** 2))
        / (np.mean(np.abs(y_true[:, 1])) + 1e-12)
    )
    # probabilistic scores
    std = t["std"]
    if std is not None:
        s = std[A:B]
        valid = np.isfinite(s) & (s > 0)
        s_safe = np.where(valid, s, 1.0)
        crps = np.where(valid, crps_gauss(e, s_safe), np.abs(e))
        z2 = np.where(valid, (e / s_safe) ** 2, np.nan)
        out["z2"] = z2.astype(np.float32)
        out["sigma2"] = np.where(valid, s**2, np.nan).astype(np.float32)
        out["valid_states"] = valid.all(axis=0)
        nll = np.where(valid, 0.5 * (np.log(2 * np.pi * s_safe**2) + z2), np.nan)
        out["nll"] = np.nanmean(nll, axis=0)
    else:
        crps = np.abs(e)
    out["ncrps"] = np.mean(crps, axis=0) / c
    out["E_crps"] = float(np.median(out["ncrps"]))
    out["crps_mae"] = float(
        np.median(np.mean(crps, axis=0) / (np.mean(np.abs(e), axis=0) + 1e-30))
    )
    out["wall_s"] = float(t["wall_s"])
    return out


# ---------------------------------------------------------------------------
# Aggregation per (condition, dataset, variant)
# ---------------------------------------------------------------------------
@cache
def cell(cond: str, ds: str, v: str) -> dict | None:
    """All per-series metrics of one cell; ``None`` if the variant is not part of it."""
    if not available(cond, ds, v):
        return None
    rows = []
    for i in range(C.n_series(ds)):
        t = traj(cond, ds, v, i)
        r1 = traj(cond, ds, "R1", i) if v != "R1" else None
        if t is None:
            if v in BASE_VARIANTS and failed(cond, ds, v, i):
                rows.append({"failed": True, "reason": "estimator error"})
            else:
                rows.append(None)  # not computed yet
            continue
        rows.append(series_metrics(t, ds, i, r1, is_r0=(v == "R0")))
    return {
        "rows": rows,
        "labels": [series_info(ds, i)["label"] for i in range(C.n_series(ds))],
    }


def primary(cond: str, ds: str, v: str) -> np.ndarray | None:
    """Per-series primary metric with the failure rule (H-Ausfall); ``None`` if incomplete."""
    cl = cell(cond, ds, v)
    if cl is None or any(r is None for r in cl["rows"]):
        return None
    E = np.array([np.nan if r["failed"] else r["E"] for r in cl["rows"]])
    if np.isnan(E).any():
        # a failed series counts as the worst value any variant (or R0) has there
        worst = []
        for i in range(len(E)):
            vals = [primary_raw(cond, ds, w, i) for w in VARIANTS]
            worst.append(np.nanmax([x for x in vals if x is not None] + [0.0]))
        E = np.where(np.isnan(E), np.array(worst), E)
    return E


def primary_raw(cond: str, ds: str, v: str, i: int) -> float | None:
    cl = cell(cond, ds, v)
    if cl is None or cl["rows"][i] is None or cl["rows"][i]["failed"]:
        return None
    return cl["rows"][i]["E"]


def completeness(cond: str, ds: str, v: str) -> tuple[int, int]:
    cl = cell(cond, ds, v)
    if cl is None:
        return 0, 0
    return sum(r is not None for r in cl["rows"]), len(cl["rows"])


def iqm(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    sel = x[(x >= q1) & (x <= q3)]
    return float(np.mean(sel)) if len(sel) else float(np.mean(x))


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def strat_boot_idx(labels: list[str], rng) -> np.ndarray:
    labels = np.asarray(labels)
    idx = []
    for m in sorted(set(labels)):
        pos = np.flatnonzero(labels == m)
        idx.append(rng.choice(pos, size=len(pos), replace=True))
    return np.concatenate(idx)


def hodges_lehmann(d: np.ndarray) -> float:
    w = [(d[a] + d[b]) / 2 for a in range(len(d)) for b in range(a, len(d))]
    return float(np.median(w))


def paired(a: np.ndarray, b: np.ndarray, labels: list[str]) -> dict:
    """``d = a - b`` (negative: a better). Exact Wilcoxon, HL with CI, P(a < b) with CI."""
    d = np.asarray(a, float) - np.asarray(b, float)
    if np.allclose(d, 0):
        p = 1.0
    else:
        nz = d[d != 0]
        exact = len(np.unique(np.abs(nz))) == len(nz) and len(nz) == len(d)
        p = float(
            wilcoxon(
                d, method="exact" if exact else "approx", zero_method="wilcox"
            ).pvalue
        )
    rng = np.random.default_rng(RNG_SEED)
    hl_b, pr_b = [], []
    for _ in range(N_BOOT // 5):
        ix = strat_boot_idx(labels, rng)
        hl_b.append(hodges_lehmann(d[ix]))
        pr_b.append(np.mean(d[ix] < 0))
    return {
        "mean_d": float(np.mean(d)),
        "hl": hodges_lehmann(d),
        "hl_ci": [float(np.percentile(hl_b, 2.5)), float(np.percentile(hl_b, 97.5))],
        "p": p,
        "p_better": float(np.mean(d < 0)),
        "p_better_ci": [
            float(np.percentile(pr_b, 2.5)),
            float(np.percentile(pr_b, 97.5)),
        ],
        "wins": int(np.sum(d < 0)),
        "n": len(d),
    }


def holm(pvals: dict) -> dict:
    """Holm step-down: ``{key: (p, p_adjusted, reject)}``."""
    alpha = float(STAT["alpha"])
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, running, stop = {}, 0.0, False
    for r, k in enumerate(keys):
        adj = min(1.0, (m - r) * pvals[k])
        running = max(running, adj)
        rej = (not stop) and pvals[k] <= alpha / (m - r)
        stop = stop or not rej
        out[k] = (float(pvals[k]), float(running), bool(rej))
    return out


def rho(cond: str, ds: str, va: str, vb: str) -> tuple[float, np.ndarray] | None:
    """A1: geometric RMSE ratio over J_38 (``va`` vs ``vb``), per series and for the set."""
    ca, cb = cell(cond, ds, va), cell(cond, ds, vb)
    if (
        ca is None
        or cb is None
        or any(r is None or r["failed"] for r in ca["rows"] + cb["rows"])
    ):
        return None
    J = j38()
    per = np.array(
        [
            np.exp(np.mean(np.log((ra["rmse"][J] + 1e-30) / (rb["rmse"][J] + 1e-30))))
            for ra, rb in zip(ca["rows"], cb["rows"])
        ]
    )
    return float(np.exp(np.mean(np.log(per)))), per


def tau(cond: str, v: str, s: str) -> dict | None:
    """G3: relative robustness of ``v`` vs U1 on shifted set ``s`` with a stratified bootstrap."""

    def log_ratio(ds):
        a, u = primary(cond, ds, v), primary(cond, ds, "U1")
        if a is None or u is None:
            return None, None
        return np.log(a / u), cell(cond, ds, "U1")["labels"]

    ls, lab_s = log_ratio(s)
    li, lab_i = log_ratio("benchmark")
    if ls is None or li is None:
        return None
    val = float(np.mean(ls) - np.mean(li))
    rng = np.random.default_rng(RNG_SEED + 1)
    boot = np.array(
        [
            np.mean(ls[strat_boot_idx(lab_s, rng)])
            - np.mean(li[strat_boot_idx(lab_i, rng)])
            for _ in range(N_BOOT)
        ]
    )
    margin = float(STAT["margin_log"])
    return {
        "tau": val,
        "ci_upper_95_one_sided": float(np.percentile(boot, 95)),
        "p_noninferior": float(np.mean(boot >= margin)),
    }


# ---------------------------------------------------------------------------
# Uncertainty aggregates C1-C3
# ---------------------------------------------------------------------------
def uncertainty(cond: str, ds: str, v: str) -> dict | None:
    cl = cell(cond, ds, v)
    if cl is None or any(r is None for r in cl["rows"]):
        return None
    rows = [r for r in cl["rows"] if not r["failed"] and "z2" in r]
    if not rows:
        return None
    z2 = np.stack([r["z2"].astype(float) for r in rows])  # (n, T, 41)
    sig2 = np.stack([r["sigma2"].astype(float) for r in rows])
    valid = np.all(np.stack([r["valid_states"] for r in rows]), axis=0)
    cbar = np.mean([r["c"] for r in rows], axis=0)
    out = {"n_states": int(valid.sum())}
    for p in MET["coverage_levels"]:
        q = norm.ppf((1 + p) / 2)
        cov = np.nanmean(z2 <= q**2, axis=(0, 1))
        out[f"cov_{p}"] = {
            "median": float(np.median(cov[valid])),
            "iqr": [
                float(np.percentile(cov[valid], 25)),
                float(np.percentile(cov[valid], 75)),
            ],
        }
    lnees = 10 * np.log10(np.nanmean(z2, axis=(0, 1)))
    out["lnees_db"] = {
        "median": float(np.median(lnees[valid])),
        "iqr": [
            float(np.percentile(lnees[valid], 25)),
            float(np.percentile(lnees[valid], 75)),
        ],
    }
    band = np.nanmean(z2, axis=0)  # (T, 41): mean over series
    lo, hi = 0.480, 1.709
    if len(rows) == 20:
        out["band_fraction"] = float(np.mean(((band >= lo) & (band <= hi))[:, valid]))
    sh = np.sqrt(np.nanmean(sig2, axis=(0, 1))) / cbar
    out["sharpness"] = float(np.median(sh[valid]))
    out["lnees_by_block"] = {
        b: (
            float(np.median(lnees[[j for j in idx if valid[j]]]))
            if any(valid[j] for j in idx)
            else None
        )
        for b, idx in blocks().items()
    }
    return out


# ---------------------------------------------------------------------------
# Observability tiers (validation)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def obs_tiers() -> dict | None:
    from jobs import val_series

    _, val = val_series()
    z = np.load(
        C.REPO / "datasets/benchmark/ukf_full_val_reference.npz", allow_pickle=True
    )
    ratios = []
    for i, s in enumerate(val):
        p = C.results_dir() / "val" / "R1" / f"s{i:02d}.npz"
        if not p.exists():
            return None
        r1 = np.load(p)["x_hat"].astype(float)
        truth = np.asarray(s.truth, float)
        u = np.asarray(z["ukf_x_hat"][i], float)
        ratios.append(
            np.sqrt(np.mean((u - truth)[A:B] ** 2, axis=0))
            / (np.sqrt(np.mean((r1 - truth)[A:B] ** 2, axis=0)) + 1e-30)
        )
    med = np.median(np.stack(ratios), axis=0)
    t1, t2 = MET["observability_tiers"]
    tiers = {
        "beobachtet": [j for j in range(41) if med[j] < t1],
        "schwach": [j for j in range(41) if t1 <= med[j] < t2],
        "praktisch unbeobachtet": [j for j in range(41) if med[j] >= t2],
    }
    return {"ratio": med.tolist(), "tiers": tiers}


def tier_scores(cond: str, ds: str, v: str) -> dict | None:
    tiers = obs_tiers()
    cl = cell(cond, ds, v)
    if tiers is None or cl is None or any(r is None for r in cl["rows"]):
        return None
    out = {}
    for name, idx in tiers["tiers"].items():
        if idx:
            out[name] = float(
                np.mean(
                    [
                        np.median(r["rmse"][idx] / r["c"][idx])
                        for r in cl["rows"]
                        if not r["failed"]
                    ]
                )
            )
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def pct(x) -> str:
    return (
        "–"
        if x is None or (isinstance(x, float) and not np.isfinite(x))
        else f"{100 * x:.1f}".replace(".", ",")
    )


def num(x, d=2) -> str:
    return (
        "–"
        if x is None or (isinstance(x, float) and not np.isfinite(x))
        else f"{x:.{d}f}".replace(".", ",")
    )


def build_report() -> tuple[str, dict]:
    res: dict = {"config_hash": C.config_hash(), "window_hours": [A, B]}
    md = [
        "# Finaler Schätzertest: Ergebnisse",
        "",
        (
            f"Konfiguration `{C.config_hash()}`, Bewertungsfenster Stunde {A} bis {B} (Tag {A // 24} bis {B // 24}). "
            "Primärmaß: Median-NRMSE über die 41 Zustände je Reihe, gemittelt über die 20 Reihen eines Testsatzes, "
            "gescheiterte Reihen als verloren gewertet."
        ),
        "",
    ]

    # completeness
    md += [
        "## Vollständigkeit",
        "",
        "| Bedingung | Testsatz | " + " | ".join(BASE_VARIANTS) + " |",
        "| --- | --- |" + " --- |" * len(BASE_VARIANTS),
    ]
    complete = True
    for cond in CONDITIONS:
        for ds in DATASETS:
            cells = []
            for v in BASE_VARIANTS:
                done, n = completeness(cond, ds, v)
                cells.append("–" if n == 0 else f"{done}/{n}")
                complete &= n == 0 or done == n
            md.append(f"| {cond} | {ds} | " + " | ".join(cells) + " |")
    md.append("")
    res["complete"] = complete

    # calibration quality
    md += ["## Kalibrierung in Test A", ""]
    cal = {}
    for ds in DATASETS:
        recs = [
            C.read_json(C.calib_path(ds, i))
            for i in range(C.n_series(ds))
            if C.calib_path(ds, i).exists()
        ]
        if not recs:
            continue
        keys = recs[0]["keys"]
        cal[ds] = {
            "n": len(recs),
            "chi2_start": float(np.median([r["chi2_start"] for r in recs])),
            "chi2_best": float(np.median([r["chi2_best"] for r in recs])),
            "err": {
                k: float(np.median([abs(r["log_error"][k]) for r in recs]))
                for k in keys
            },
            "err_nominal": {
                k: float(np.median([abs(r["log_error_nominal"][k]) for r in recs]))
                for k in keys
            },
        }
    if cal:
        keys = C.calibration_keys()
        md += [
            "Median des Betrags des log-Fehlers je Parameter, kalibriert gegen nominal (unkalibriert).",
            "",
            "| Parameter | " + " | ".join(cal) + " |",
            "| --- |" + " --- |" * len(cal),
        ]
        for k in keys:
            md.append(
                f"| {k} | "
                + " | ".join(
                    f"{num(cal[d]['err'][k])} ({num(cal[d]['err_nominal'][k])})"
                    for d in cal
                )
                + " |"
            )
        md.append(
            "| χ² Start → Ende, Median | "
            + " | ".join(
                f"{num(cal[d]['chi2_start'], 1)} → {num(cal[d]['chi2_best'], 1)} (n = {cal[d]['n']})"
                for d in cal
            )
            + " |"
        )
        md.append("")
    res["calibration"] = cal

    # primary metric table
    md += [
        "## Primärmaß je Bedingung",
        "",
        "| Variante | Bedingung | " + " | ".join(DATASETS) + " | OOD-Mittel |",
        "| --- | --- |" + " --- |" * (len(DATASETS) + 1),
    ]
    prim = {}
    for v in VARIANTS:
        for cond in CONDITIONS:
            vals = {}
            for ds in DATASETS:
                E = primary(cond, ds, v)
                vals[ds] = None if E is None else float(np.mean(E))
            if all(x is None for x in vals.values()):
                continue
            ood = [vals[d] for d in SHIFTED if vals[d] is not None]
            prim[(v, cond)] = vals
            md.append(
                f"| {v} | {cond} | "
                + " | ".join(pct(vals[d]) for d in DATASETS)
                + f" | {pct(np.mean(ood)) if len(ood) == len(SHIFTED) else '–'} |"
            )
    md.append("")
    res["primary"] = {f"{v}|{c}": d for (v, c), d in prim.items()}

    # E1 best variant
    md += ["## E1 Beste Variante (Test A, `benchmark`, kalibrierte Kinetik)", ""]
    cands = list(STAT["candidates_best"])
    Ec = {v: primary("A_cal", "benchmark", v) for v in cands}
    e1 = {}
    if all(x is not None for x in Ec.values()):
        labels = cell("A_cal", "benchmark", "U1")["labels"]
        pairs = {}
        for va, vb in itertools.combinations(cands, 2):
            pairs[(va, vb)] = paired(Ec[va], Ec[vb], labels)
        h = holm({k: v["p"] for k, v in pairs.items()})
        md += [
            "| A | B | Mittel A − B [Pkt] | HL [Pkt] | 95-%-KI | p | Holm | P(A besser) | ρ A/B |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for (va, vb), r in pairs.items():
            rr = rho("A_cal", "benchmark", va, vb)
            md.append(
                f"| {va} | {vb} | {num(100 * r['mean_d'])} | {num(100 * r['hl'])} | [{num(100 * r['hl_ci'][0])}; {num(100 * r['hl_ci'][1])}] | {num(r['p'], 4)} | {'ja' if h[(va, vb)][2] else 'nein'} | {num(r['p_better'])} | {num(rr[0], 3) if rr else '–'} |"
            )
        best = min(cands, key=lambda v: np.mean(Ec[v]))
        rope = float(STAT["rope_points"]) / 100

        def beats(a, b):
            key = (a, b) if (a, b) in pairs else (b, a)
            r = pairs[key]
            hl = r["hl"] if key == (a, b) else -r["hl"]  # hl of E_a - E_b
            rr = rho("A_cal", "benchmark", a, b)
            return h[key][2] and (-hl >= rope) and (rr is not None and rr[0] < 1)

        others = [v for v in cands if v != best]
        if all(beats(best, b) for b in others):
            e1 = {"winner": best, "top_group": [best]}
            md += [
                "",
                f"**Beste Variante: {best}.** Sie ist gegen jede andere Holm-signifikant, um mindestens {STAT['rope_points']} Punkte und in der Gegenprobe besser.",
            ]
        else:
            group = [best] + [b for b in others if not beats(best, b)]
            ood = {
                v: (
                    np.mean([np.mean(primary("A_cal", d, v)) for d in SHIFTED])
                    if all(primary("A_cal", d, v) is not None for d in SHIFTED)
                    else np.inf
                )
                for v in group
            }
            group = sorted(group, key=lambda v: (ood[v], v))
            e1 = {"winner": None, "top_group": group, "lowest_mean": best}
            md += [
                "",
                f"**Keine einzelne beste Variante.** Spitzengruppe, geordnet nach dem Fehler auf den verschobenen Sätzen: {', '.join(group)}. Niedrigstes Mittel auf `benchmark`: {best}.",
            ]
        res["E1"] = {
            "pairs": {f"{a}|{b}": r for (a, b), r in pairs.items()},
            "holm": {f"{a}|{b}": v for (a, b), v in h.items()},
            **e1,
        }
    else:
        md += ["Noch nicht vollständig."]
    md.append("")

    # E3 calibration gains
    md += [
        "## E3 Kalibriergewinn in Test A (nominal minus kalibriert, positiv = Gewinn)",
        "",
        "| Variante | "
        + " | ".join(f"{d}: Gewinn [Pkt], p, Holm" for d in DATASETS)
        + " | lohnt sich |",
        "| --- |" + " --- |" * (len(DATASETS) + 1),
    ]
    e3 = {}
    for v, (vn, vc) in {
        "U1": ("U1", "U1"),
        "U2": ("U2", "U2"),
        "P1": ("P1", "P1"),
        "R1": ("R1", "R1"),
        "PINN B (B1 → B2)": ("B1", "B2"),
        "Fusion (F1 → F2)": ("F1", "F2"),
    }.items():
        tests = {}
        for ds in DATASETS:
            en, ec = primary("nominal", ds, vn), primary("A_cal", ds, vc)
            if en is None or ec is None:
                continue
            tests[ds] = paired(ec, en, cell("nominal", ds, vn)["labels"])
        if len(tests) < len(DATASETS):
            md.append(f"| {v} | " + " | ".join("–" for _ in DATASETS) + " | offen |")
            continue
        h = holm({d: r["p"] for d, r in tests.items()})
        ok = sum(
            1
            for d in DATASETS
            if h[d][2] and -tests[d]["hl"] >= float(STAT["rope_points"]) / 100
        )
        e3[v] = {"tests": tests, "holm": h, "worth_it": ok >= 2}
        md.append(
            f"| {v} | "
            + " | ".join(
                f"{num(-100 * tests[d]['mean_d'])}, {num(tests[d]['p'], 3)}, {'ja' if h[d][2] else 'nein'}"
                for d in DATASETS
            )
            + f" | {'ja' if ok >= 2 else 'nein'} |"
        )
    res["E3"] = e3
    md.append("")

    # Test B curves
    md += [
        "## Test B: Fehler über Kinetikqualität",
        "",
        "| Variante | Testsatz | nominal | ident | real | voll | η ident | η real |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    tb = {}
    for v in ["U1", "U2", "P1", "B2", "F2", "R1"]:
        for ds in DATASETS:
            vals = {
                c: (
                    None
                    if primary(c, ds, v) is None
                    else float(np.mean(primary(c, ds, v)))
                )
                for c in ("nominal", "B_ident", "B_real", "B_voll")
            }
            if vals["B_voll"] is None and vals["B_ident"] is None:
                continue
            dv = (
                None
                if (vals["nominal"] is None or vals["B_voll"] is None)
                else vals["nominal"] - vals["B_voll"]
            )
            rope = float(STAT["rope_points"]) / 100
            # a share of a gain is only defined when there is a gain to share
            eta = {
                c: (
                    None
                    if dv is None or dv < rope or vals[c] is None
                    else (vals["nominal"] - vals[c]) / dv
                )
                for c in ("B_ident", "B_real")
            }
            tb[f"{v}|{ds}"] = {**vals, "eta": eta}
            md.append(
                f"| {v} | {ds} | "
                + " | ".join(
                    pct(vals[c]) for c in ("nominal", "B_ident", "B_real", "B_voll")
                )
                + f" | {num(eta['B_ident'])} | {num(eta['B_real'])} |"
            )
    res["test_b"] = tb
    md.append("")

    # Generalisation
    md += [
        "## E2 Generalisierung (Test A, kalibrierte Kinetik)",
        "",
        "| Variante | Lücke ccm [Pkt] | Lücke hard [Pkt] | τ ccm | τ hard | obere 95-%-Grenze ccm / hard | generalisiert gut |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    e2 = {}
    gen_p = {}
    for v in ["U1", "U2", "P1", "B2", "F2"]:
        g = {}
        for s in SHIFTED:
            e_s, e_i = primary("A_cal", s, v), primary("A_cal", "benchmark", v)
            g[s] = {
                "gap": (
                    None
                    if e_s is None or e_i is None
                    else float(np.mean(e_s) - np.mean(e_i))
                ),
                "gap_rel": (
                    None
                    if e_s is None or e_i is None
                    else float(np.mean(e_s) / np.mean(e_i) - 1)
                ),
                "tau": tau("A_cal", v, s) if v != "U1" else None,
            }
            if g[s]["tau"] is not None:
                gen_p[(v, s)] = g[s]["tau"]["p_noninferior"]
        e2[v] = g
    hg = holm(gen_p) if gen_p else {}
    for v, g in e2.items():
        ok = v != "U1" and all((v, s) in hg and hg[(v, s)][2] for s in SHIFTED)
        e2[v]["generalises_well"] = ok if v != "U1" else "Referenz"
        taus = [g[s]["tau"] for s in SHIFTED]
        md.append(
            f"| {v} | "
            + " | ".join(
                num(100 * g[s]["gap"]) if g[s]["gap"] is not None else "–"
                for s in SHIFTED
            )
            + " | "
            + " | ".join(num(t["tau"], 3) if t else "–" for t in taus)
            + " | "
            + (
                " / ".join(num(t["ci_upper_95_one_sided"], 3) for t in taus)
                if all(taus)
                else "–"
            )
            + f" | {'Referenz' if v == 'U1' else ('ja' if ok else 'nein')} |"
        )
    res["E2"] = {"per_variant": e2, "holm": {f"{a}|{b}": v for (a, b), v in hg.items()}}
    md.append("")

    # Uncertainty
    md += [
        "## E4 Unsicherheit (C1 bis C4)",
        "",
        "| Variante | Bedingung | Testsatz | Abdeckung 95 % (Median) | LNEES [dB] (Median) | Schärfe | CRPS-Maß | CRPS/MAE | brauchbar |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    e4 = {}
    for v in WITH_SIGMA + ["P1"]:
        for cond in ("nominal", "A_cal"):
            for ds in DATASETS:
                cl = cell(cond, ds, v)
                if cl is None or any(r is None for r in cl["rows"]):
                    continue
                rows = [r for r in cl["rows"] if not r["failed"]]
                ecrps, ratio = float(np.mean([r["E_crps"] for r in rows])), float(
                    np.median([r["crps_mae"] for r in rows])
                )
                u = uncertainty(cond, ds, v) if v in WITH_SIGMA else None
                usable = None
                if u is not None and ds == "benchmark":
                    usable = (
                        0.90 <= u["cov_0.95"]["median"] <= 0.99
                        and abs(u["lnees_db"]["median"]) <= 3.0
                    )
                e4[f"{v}|{cond}|{ds}"] = {
                    "uncertainty": u,
                    "E_crps": ecrps,
                    "crps_mae": ratio,
                    "usable": usable,
                }
                md.append(
                    f"| {v} | {cond} | {ds} | {num(u['cov_0.95']['median']) if u else '–'} | {num(u['lnees_db']['median'], 1) if u else '–'} | "
                    f"{num(u['sharpness']) if u else '–'} | {pct(ecrps)} | {num(ratio)} | {'–' if usable is None else ('ja' if usable else 'nein')} |"
                )
    res["E4"] = e4
    md.append("")

    # Transients, failures, worst case, secondary, cost
    md += [
        "## Transienten, Ausfälle, schlechtester Fall, Nebengrößen",
        "",
        "| Variante | Bedingung | Testsatz | T1 Transient | T2 Spitze | Ausfälle | schlechteste Reihe | schlechtester Modus | ρ zu U1 | Gewinn zu R1 | pH-RMSE | Q_ch4 rel. | Rechenzeit je Schritt [s] |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    sec = {}
    for v in VARIANTS:
        for cond in CONDITIONS:
            for ds in DATASETS:
                cl = cell(cond, ds, v)
                if cl is None or any(r is None for r in cl["rows"]):
                    continue
                E = primary(cond, ds, v)
                ok_rows = [r for r in cl["rows"] if not r["failed"]]
                labels = np.array(cl["labels"])
                worst_mode = max(float(np.mean(E[labels == m])) for m in set(labels))
                rr = rho(cond, ds, v, "U1") if v != "U1" else None
                s = {
                    "T1": (
                        np.mean([r["T1"] for r in ok_rows if "T1" in r])
                        if ok_rows
                        else None
                    ),
                    "T2": (
                        np.mean([r["T2"] for r in ok_rows if "T2" in r])
                        if ok_rows
                        else None
                    ),
                    "failures": sum(r["failed"] for r in cl["rows"]),
                    "worst_series": float(np.max(E)),
                    "worst_mode": worst_mode,
                    "rho_U1": rr[0] if rr else None,
                    "SS": (
                        float(
                            np.mean(
                                [r["SS_median"] for r in ok_rows if "SS_median" in r]
                            )
                        )
                        if any("SS_median" in r for r in ok_rows)
                        else None
                    ),
                    "pH": (
                        float(np.mean([r["pH_rmse"] for r in ok_rows]))
                        if ok_rows
                        else None
                    ),
                    "Qch4": (
                        float(np.mean([r["Qch4_rel"] for r in ok_rows]))
                        if ok_rows
                        else None
                    ),
                    "step_s": (
                        float(np.median([r["wall_s"] for r in ok_rows]) / 1441)
                        if ok_rows
                        else None
                    ),
                    "median_series": float(np.median(E)),
                    "iqm_series": iqm(E),
                    "blocks": (
                        {
                            b: float(np.mean([r["blocks"][b] for r in ok_rows]))
                            for b in blocks()
                        }
                        if ok_rows
                        else None
                    ),
                    "tiers": tier_scores(cond, ds, v),
                }
                sec[f"{v}|{cond}|{ds}"] = s
                md.append(
                    f"| {v} | {cond} | {ds} | {pct(s['T1'])} | {pct(s['T2'])} | {s['failures']} | {pct(s['worst_series'])} | {pct(s['worst_mode'])} | "
                    f"{num(s['rho_U1'], 3) if s['rho_U1'] else '–'} | {num(s['SS'])} | {num(s['pH'], 3)} | {pct(s['Qch4'])} | {num(s['step_s'], 2)} |"
                )
    res["secondary"] = sec
    md.append("")
    tiers = obs_tiers()
    if tiers:
        md += [
            "Beobachtbarkeitsstufen (Validierung, RMSE U1 / RMSE R1): "
            + "; ".join(
                f"{k}: {', '.join(state_names()[j] for j in v)}"
                for k, v in tiers["tiers"].items()
            ),
            "",
        ]
        res["observability_tiers"] = tiers
        md += [
            "## Fehler je Beobachtbarkeitsstufe (`benchmark`)",
            "",
            "| Variante | Bedingung | " + " | ".join(tiers["tiers"]) + " |",
            "| --- | --- |" + " --- |" * len(tiers["tiers"]),
        ]
        for v in ["U1", "U2", "P1", "B1", "B2", "F1", "F2", "R1"]:
            for cond in ("nominal", "A_cal", "B_ident", "B_voll"):
                ts = tier_scores(cond, "benchmark", v)
                if ts:
                    md.append(
                        f"| {v} | {cond} | "
                        + " | ".join(pct(ts.get(k)) for k in tiers["tiers"])
                        + " |"
                    )
        md.append("")

    md += extra_sections(res)
    return "\n".join(md) + "\n", res


def ranking_table(
    cond: str, cands: list[str], title: str, md: list, res: dict, key: str
) -> None:
    """Pairwise comparison and top group of ``cands`` on ``benchmark`` (same rules as E1)."""
    md += [title, ""]
    Ec = {v: primary(cond, "benchmark", v) for v in cands}
    if not all(x is not None for x in Ec.values()):
        md += ["Noch nicht vollständig.", ""]
        return
    labels = cell(cond, "benchmark", "U1")["labels"]
    pairs = {
        (a, b): paired(Ec[a], Ec[b], labels)
        for a, b in itertools.combinations(cands, 2)
    }
    h = holm({k: v["p"] for k, v in pairs.items()})
    md += [
        "| A | B | Mittel A − B [Pkt] | HL [Pkt] | 95-%-KI | p | Holm | P(A besser) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (a, b), r in pairs.items():
        md.append(
            f"| {a} | {b} | {num(100 * r['mean_d'])} | {num(100 * r['hl'])} | [{num(100 * r['hl_ci'][0])}; {num(100 * r['hl_ci'][1])}] | {num(r['p'], 4)} | {'ja' if h[(a, b)][2] else 'nein'} | {num(r['p_better'])} |"
        )
    best = min(cands, key=lambda v: np.mean(Ec[v]))
    rope = float(STAT["rope_points"]) / 100

    def beats(a, b):
        k = (a, b) if (a, b) in pairs else (b, a)
        hl = pairs[k]["hl"] if k == (a, b) else -pairs[k]["hl"]
        rr = rho(cond, "benchmark", a, b)
        return h[k][2] and -hl >= rope and rr is not None and rr[0] < 1

    group = [best] + [b for b in cands if b != best and not beats(best, b)]
    ood = {v: np.mean([np.mean(primary(cond, d, v)) for d in SHIFTED]) for v in group}
    group = sorted(group, key=lambda v: (ood[v], v))
    md += [
        "",
        (
            f"Spitzengruppe, geordnet nach dem Fehler auf den verschobenen Sätzen: {', '.join(f'{v} ({pct(ood[v])} %)' for v in group)}. "
            f"Niedrigstes Mittel auf `benchmark`: {best} ({pct(float(np.mean(Ec[best])))} %)."
        ),
        "",
    ]
    res[key] = {
        "pairs": {f"{a}|{b}": r for (a, b), r in pairs.items()},
        "top_group": group,
        "lowest_mean": best,
    }


def extra_sections(res: dict) -> list[str]:
    md: list[str] = []
    ranking_table(
        "nominal",
        ["U1", "U2", "P1", "B1", "F1"],
        "## Rangfolge ohne Kalibrierung (beschreibend, nicht vorab als Entscheidung festgelegt)",
        md,
        res,
        "ranking_nominal",
    )
    ranking_table(
        "B_ident",
        ["U1", "U2", "P1", "B2", "F2"],
        "## Rangfolge mit wahrer Kinetik der zehn kalibrierbaren Parameter (Test B, beschreibend)",
        md,
        res,
        "ranking_B_ident",
    )

    # PINN B ensemble members (S7)
    md += [
        "## PINN B: einzelne Ensemblemitglieder",
        "",
        "| Variante | Bedingung | Testsatz | Ensemble | Mitglieder, Mittel über Reihen |",
        "| --- | --- | --- | --- | --- |",
    ]
    mem = {}
    for v, conds in (("B1", ["nominal"]), ("B2", ["nominal", "A_cal", "B_ident"])):
        for cond in conds:
            for ds in DATASETS:
                rows = [C.load_traj(cond, ds, v, i) for i in range(C.n_series(ds))]
                if any(r is None for r in rows):
                    continue
                m = np.mean([r["meta"]["member_scores"] for r in rows], axis=0)
                ens = float(np.mean([r["meta"]["score"] for r in rows]))
                mem[f"{v}|{cond}|{ds}"] = {"ensemble": ens, "members": m.tolist()}
                md.append(
                    f"| {v} | {cond} | {ds} | {pct(ens)} | "
                    + ", ".join(pct(float(x)) for x in m)
                    + " |"
                )
    md.append("")
    res["members"] = mem

    # calibration diagnostic outside the test
    diag_dir = C.results_dir() / "diagnostics" / "known_start" / "benchmark"
    diags = sorted(diag_dir.glob("s*.json")) if diag_dir.exists() else []
    if diags:
        d = [C.read_json(f) for f in diags]
        a = [C.read_json(C.calib_path("benchmark", r["i"])) for r in d]
        keys = d[0]["keys"]
        md += [
            "## Diagnose außerhalb des Tests: Kalibrierung mit wahrem Startzustand",
            "",
            (
                f"Dieselbe Kalibrierung auf den Reihen {', '.join(str(r['i']) for r in d)} von `benchmark`, einmal mit dem wahren Zustand zur Stunde 48 als Start. "
                "Median des Betrags des log-Fehlers."
            ),
            "",
            "| Parameter | wahrer Start | Start aus U1 (Test A) | nominal |",
            "| --- | --- | --- | --- |",
        ]
        for k in keys:
            md.append(
                f"| {k} | {num(np.median([abs(r['log_error'][k]) for r in d]))} | {num(np.median([abs(r['log_error'][k]) for r in a]))} | {num(np.median([abs(r['log_error_nominal'][k]) for r in d]))} |"
            )
        md.append("")

    # expectations (descriptive)
    md += ["## Erwartungen aus dem Plan (beschreibend geprüft)", ""]

    def gain(v_nom, v_cal, ds, c_cal="A_cal"):
        en, ec = primary("nominal", ds, v_nom), primary(c_cal, ds, v_cal)
        return None if en is None or ec is None else float(np.mean(en) - np.mean(ec))

    g_b = {v: gain(v, v, "benchmark", "B_ident") for v in ["U1", "U2", "P1", "R1"]}
    g_b["B (B1→B2)"] = gain("B1", "B2", "benchmark", "B_ident")
    g_b["F (F1→F2)"] = gain("F1", "F2", "benchmark", "B_ident")
    md.append(
        "1. **U1 profitiert am stärksten von guter Kinetik.** Gewinn mit wahrer Kinetik der zehn Parameter (Test B, `benchmark`): "
        + ", ".join(
            f"{k} {num(100 * v) if v is not None else '–'} Pkt" for k, v in g_b.items()
        )
        + "."
    )
    gu = [gain("U1", "U1", d) for d in DATASETS]
    gf = [gain("F1", "F2", d) for d in DATASETS]
    md.append(
        "2. **Kalibriergewinn der Fusion zwischen null und dem von U1 (Test A).** U1: "
        + ", ".join(num(100 * x) for x in gu if x is not None)
        + " Pkt; Fusion F1 → F2: "
        + ", ".join(num(100 * x) for x in gf if x is not None)
        + " Pkt."
    )
    etas = {
        k: v["eta"]["B_ident"]
        for k, v in res.get("test_b", {}).items()
        if v["eta"]["B_ident"] is not None
    }
    md.append(
        "3. **B-ident holt nur einen Teil des Gewinns von B-voll.** Anteil η: "
        + ", ".join(f"{k.replace('|', ' ')} {num(v)}" for k, v in etas.items())
        + "."
    )
    tn = [tau("nominal", "B2", s) for s in SHIFTED]
    ta = [tau("A_cal", "B2", s) for s in SHIFTED]
    tb = [tau("B_ident", "B2", s) for s in SHIFTED]
    md.append(
        "4. **τ von B2 sinkt mit kalibrierter Kinetik.** τ (ccm, hard) nominal: "
        + ", ".join(num(x["tau"], 3) for x in tn if x)
        + "; Test A: "
        + ", ".join(num(x["tau"], 3) for x in ta if x)
        + "; wahre Kinetik (B-ident): "
        + ", ".join(num(x["tau"], 3) for x in tb if x)
        + "."
    )
    g5 = [gain("B1", "B2", d) for d in DATASETS]
    md.append(
        "5. **B2 holt in Test A weniger als 1,9 Punkte.** Gewinn B1 → B2 in Test A: "
        + ", ".join(num(100 * x) for x in g5 if x is not None)
        + " Pkt."
    )
    md.append("")
    res["expectations"] = {
        "gain_B_ident": g_b,
        "gain_U1_A": gu,
        "gain_F_A": gf,
        "eta_ident": etas,
        "tau_B2": {"nominal": tn, "A_cal": ta, "B_ident": tb},
        "gain_B_A": g5,
    }
    return md


def main() -> int:
    out_dir = C.results_dir() / "report"
    text, res = build_report()
    C.atomic_write_text(out_dir / "bericht.md", text)
    C.atomic_write_json(out_dir / "scores.json", res)
    print(text)
    print("wrote", out_dir / "bericht.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
