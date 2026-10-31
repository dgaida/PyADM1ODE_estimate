"""Block F3 of the fusion study: the causal PINN corrects the UKF state.

The reference UKF (full 41-state variant, CMA-ES winner ``estimation/reference/cmaes_full_winner.json``)
is re-run with an intervention hook. At the hours where the causal PINN has just been
retrained (every 24 h in scheme ``W10_C24_S200``) its estimate can be fed to the filter:

``update``  exact linear Kalman update with the PINN core state as a pseudo-measurement
            ``z = x_PINN[0:31]``, ``R = diag((sigma_rel * z)^2)`` (or the seed-ensemble spread);
``reset``   overwrite the filter mean with the PINN state and re-initialise the covariance.

The trigger is either periodic (every PINN retraining) or *innovation gated*: only when the
filter's own normalised innovation squared, averaged over the last 24 h and per channel,
exceeds a threshold (NIS ~ 1 for a consistent filter). ``none`` reproduces the shipped
reference and validates the loop.

    python experiments/pinn_v2/ukf_correct.py --variants none periodic_u0.2 gate5_u0.2 --workers 26
    python experiments/pinn_v2/ukf_correct.py --summary
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time as _time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyadm1.core.adm1 import STATE_SIZE

from pyadm1ode_estimation.estimation.estimators import REFERENCE_DIR
from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import (
    _apply_theta,
    _warmup,
    baseline_flows,
    build_plant,
    substrates_from_meta,
)
from pyadm1ode_estimation.estimation.pinn.data import (
    BenchmarkSeries,
    load_meta,
    load_series,
    scoring_module,
)
from pyadm1ode_estimation.estimation.quickstart import (
    build_filter_components,
    build_ukf,
)

RESULTS = Path(__file__).resolve().parent / "results"
N_CORE = 31
WARM = 72  # scoring window starts after day 3

_VARIANT = re.compile(
    r"^(none|periodic|gate(\d+(?:\.\d+)?))(?:_(u|r)(\d+(?:\.\d+)?|ens))?$"
)


@dataclass(frozen=True)
class CorrConfig:
    """One correction variant, e.g. ``gate5_u0.2`` or ``periodic_uens``."""

    trigger: str = "none"  # none | periodic | gate
    threshold: float = 0.0  # NIS per channel (24 h mean) above which the gate fires
    mode: str = "update"  # update (Kalman pseudo-measurement) | reset
    sigma_rel: float = (
        0.2  # pseudo-measurement std relative to |z|; ignored with use_ensemble
    )
    use_ensemble: bool = False
    ensemble_floor: float = 0.05
    cadence_hours: int = 24  # PINN retraining cadence of the trajectory used
    first_hour: int = 24

    @property
    def name(self) -> str:
        if self.trigger == "none":
            return "none"
        trig = "periodic" if self.trigger == "periodic" else f"gate{self.threshold:g}"
        sig = "ens" if self.use_ensemble else f"{self.sigma_rel:g}"
        return f"{trig}_{'u' if self.mode == 'update' else 'r'}{sig}"

    @classmethod
    def parse(cls, s: str) -> CorrConfig:
        m = _VARIANT.match(s)
        if not m:
            raise SystemExit(
                f"bad variant {s!r}; e.g. none, periodic_u0.2, gate5_u0.2, gate5_r0.2, periodic_uens"
            )
        if m.group(1) == "none":
            return cls()
        trig = "periodic" if m.group(1) == "periodic" else "gate"
        thr = float(m.group(2)) if m.group(2) else 0.0
        mode = "reset" if m.group(3) == "r" else "update"
        ens = m.group(4) == "ens"
        return cls(
            trigger=trig,
            threshold=thr,
            mode=mode,
            sigma_rel=0.2 if ens or m.group(4) is None else float(m.group(4)),
            use_ensemble=ens,
        )


def kalman_pseudo_update(x, P, pos, z, r_std):
    """Linear Kalman update with ``z = x[pos] + v``, ``v ~ N(0, diag(r_std^2))`` (Joseph form)."""
    n = len(x)
    H = np.zeros((len(pos), n))
    H[np.arange(len(pos)), pos] = 1.0
    R = np.diag(np.asarray(r_std, float) ** 2)
    S = H @ P @ H.T + R
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    x_new = x + K @ (z - x[pos])
    A = np.eye(n) - K @ H
    P_new = A @ P @ A.T + K @ R @ K.T
    return x_new, 0.5 * (P_new + P_new.T)


def run_corrected_ukf(
    series: BenchmarkSeries,
    meta: dict,
    theta: dict,
    pinn: np.ndarray,
    corr: CorrConfig,
    pinn_std: np.ndarray | None = None,
    max_hours: int | None = None,
) -> dict:
    """Re-run the reference UKF on ``series`` with the PINN correction ``corr``."""
    did = "primary"
    subs = substrates_from_meta(meta)
    sensors = [s.lower() for s in meta["sensors"]]
    channel_names = list(meta["sensors"])
    dt_hours = float(meta.get("dt_hours", 1.0))
    T = series.n_steps if max_hours is None else min(series.n_steps, max_hours + 1)
    meas, feed, time = series.measurements[:T], series.feed_noisy[:T], series.time[:T]

    plant = build_plant(meta, did)
    wp, _, wspec = build_filter_components(
        plant, digester_id=did, substrates=subs, sensors=sensors
    )
    xw = _warmup(
        wp,
        wspec,
        wspec.read_adm1_state(plant),
        float(meta.get("warmup_days", 30.0)),
        baseline_flows(meta, subs),
        dt_hours,
    )
    adm1_vec = np.zeros(STATE_SIZE)
    for i, ch in enumerate(wspec.channels):
        if ch.kind == "adm1":
            adm1_vec[ch.adm1_index] = xw[i]
    plant.components[did].adm1_state = list(adm1_vec)

    ukf = build_ukf(
        plant,
        digester_id=did,
        substrates=(),
        sensors=list(sensors),
        initial_uncertainty_relative=0.05 * float(theta.get("p0_scale", 1.0)),
        adm1_indices=None,
        process_noise_scale=float(theta.get("process_noise_scale", 1.0)),
    )
    spec, obs = ukf.spec, ukf.obs
    _apply_theta(spec, obs, theta)
    adm1_pos = spec.kind_indices("adm1")
    idx_of_pos = [spec.channels[p].adm1_index for p in adm1_pos]
    pos_of_idx = {a: p for p, a in zip(adm1_pos, idx_of_pos)}
    core_pos = np.array([pos_of_idx[i] for i in range(N_CORE)])
    all_pos = np.array([pos_of_idx[i] for i in range(STATE_SIZE)])
    sub_idx = [s.substrate_index for s in subs]

    cols = {
        c.name: meas[:, channel_names.index(c.name)]
        for c in obs.channels
        if c.name in channel_names
    }
    frame = pd.DataFrame(cols, index=time)
    n_ch = len(cols)

    n_state = len(spec)
    x_hat = np.zeros((T, n_state))
    std = np.zeros((T, n_state))
    nis = np.full(T, np.nan)
    fired = []
    dt = dt_hours / 24.0
    t0 = _time.perf_counter()
    for k, t in enumerate(frame.index):
        if k > 0:
            ukf.process.set_known_input(
                {si: float(feed[k - 1, j]) for j, si in enumerate(sub_idx)}
            )
            ukf.predict(dt)
        y = {
            c: float(frame.iloc[k][c])
            for c in frame.columns
            if np.isfinite(frame.iloc[k][c])
        }
        step = ukf.update(y, t=float(t), gate_values={})
        nis[k] = step.nis / max(n_ch, 1) if np.isfinite(step.nis) else np.nan
        x_k, P_k = step.x_hat, step.P

        due = (
            corr.trigger != "none"
            and k >= corr.first_hour
            and (k - corr.first_hour) % corr.cadence_hours == 0
        )
        if due:
            fire = True
            if corr.trigger == "gate":
                fire = bool(np.nanmean(nis[max(0, k - 23) : k + 1]) > corr.threshold)
            if fire:
                z_full = np.clip(pinn[k], 1e-12, None)
                if corr.mode == "update":
                    z = z_full[:N_CORE]
                    if corr.use_ensemble and pinn_std is not None:
                        r_std = np.maximum(
                            pinn_std[k, :N_CORE], corr.ensemble_floor * np.abs(z)
                        )
                    else:
                        r_std = corr.sigma_rel * np.abs(z)
                    x_new, P_new = kalman_pseudo_update(
                        ukf.x_hat.copy(), ukf.P, core_pos, z, np.maximum(r_std, 1e-12)
                    )
                else:  # reset mean to the PINN state, covariance to a diagonal around it
                    x_new = ukf.x_hat.copy()
                    x_new[all_pos] = z_full
                    P_new = ukf.P.copy()
                    P_new[all_pos, :] = 0.0
                    P_new[:, all_pos] = 0.0
                    P_new[all_pos, all_pos] = (corr.sigma_rel * z_full) ** 2
                x_new = spec.clip(x_new)
                P_new = P_new + 1e-18 * np.eye(n_state)
                try:
                    ukf.reset(x_new, P_new)
                    x_k, P_k = ukf.x_hat, ukf.P
                    fired.append(k)
                except np.linalg.LinAlgError:
                    pass  # keep the filter's own posterior if the corrected P is not PD
        x_hat[k] = x_k
        std[k] = np.sqrt(np.clip(np.diag(P_k), 0.0, None))

    seconds = _time.perf_counter() - t0
    x41 = np.zeros((T, STATE_SIZE))
    s41 = np.zeros((T, STATE_SIZE))
    x41[:, idx_of_pos] = x_hat[:, adm1_pos]
    s41[:, idx_of_pos] = std[:, adm1_pos]
    scale = theta.get("std_scale")
    if scale is not None:
        s41 = s41 * np.asarray(scale, float)[None, :]
    return {"x": x41, "std": s41, "nis": nis, "fired": fired, "seconds": seconds}


def _score(x, s, series: BenchmarkSeries, T: int) -> dict:
    sc = scoring_module()
    out = {}
    for name, sl in (("full", slice(0, T)), ("after3d", slice(min(WARM, T - 2), T))):
        d = series.scoring_dict(sl)
        r = sc.score_series(x[sl], d, std=s[sl])
        out[name] = {
            "model": r["model"],
            "ukf": r["ukf_reference"],
            "vs_ukf": r["vs_ukf"],
        }
    return out


def _run(args):
    series, meta, theta, pinn, pinn_std, corr, max_hours, traj_dir = args
    res = run_corrected_ukf(series, meta, theta, pinn, corr, pinn_std, max_hours)
    T = len(res["x"])
    ref_dev = float(
        np.nanmax(
            np.abs(res["x"] - series.ukf_x_hat[:T])
            / (np.abs(series.ukf_x_hat[:T]) + 1e-12)
        )
    )
    d = traj_dir / corr.name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"series_{series.index:02d}.npz",
        x=res["x"].astype(np.float32),
        std=res["std"].astype(np.float32),
        nis=res["nis"],
        fired=np.asarray(res["fired"]),
    )
    return {
        "series_index": series.index,
        "regime": series.regime,
        "corr": asdict(corr),
        "variant": corr.name,
        "n_fired": len(res["fired"]),
        "n_due": max(0, (T - 1 - corr.first_hour) // corr.cadence_hours + 1),
        "nis_mean": float(np.nanmean(res["nis"])),
        "seconds": res["seconds"],
        "max_rel_dev_from_reference": ref_dev,
        "metrics": _score(res["x"], res["std"], series, T),
    }


def summarize(recs: list[dict]) -> str:
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(r["variant"], []).append(r)
    regs = sorted({r["regime"] for r in recs})
    lines = [
        f"{'variant':16s} {'n':>3s} {'NRMSE>3d':>9s} {'ref UKF':>8s} {'diff':>6s} {'<ref':>6s} {'(full)':>7s} {'trans':>7s} {'cov2s%':>7s} {'fired':>8s} {'NIS':>6s} {'h/run':>6s}  "
        + "  ".join(f"{g[:11]:>11s}" for g in regs)
    ]
    for v, rs in sorted(
        by.items(),
        key=lambda kv: np.mean(
            [r["metrics"]["after3d"]["model"]["nrmse_overall_%"] for r in kv[1]]
        ),
    ):
        a = np.array([r["metrics"]["after3d"]["model"]["nrmse_overall_%"] for r in rs])
        u = np.array([r["metrics"]["after3d"]["ukf"]["nrmse_overall_%"] for r in rs])
        f = np.array([r["metrics"]["full"]["model"]["nrmse_overall_%"] for r in rs])
        tr = np.array(
            [r["metrics"]["after3d"]["model"]["nrmse_transient_%"] for r in rs]
        )
        cov = np.array(
            [
                r["metrics"]["after3d"]["model"].get("coverage_2sigma_%", np.nan)
                for r in rs
            ]
        )
        per = [np.mean([x for x, r in zip(a, rs) if r["regime"] == g]) for g in regs]
        lines.append(
            f"{v:16s} {len(rs):3d} {a.mean():9.2f} {u.mean():8.2f} {a.mean() - u.mean():+6.2f} {int((a < u).sum()):3d}/{len(rs):<2d} {f.mean():7.2f} {tr.mean():7.2f} "
            f"{np.nanmean(cov):7.1f} {np.mean([r['n_fired'] for r in rs]):4.1f}/{np.mean([r['n_due'] for r in rs]):<3.0f} {np.mean([r['nis_mean'] for r in rs]):6.2f} {np.mean([r['seconds'] for r in rs]) / 3600:6.2f}  "
            + "  ".join(f"{p:11.2f}" for p in per)
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["none"])
    ap.add_argument("--series", nargs="*", type=int, default=None)
    ap.add_argument("--scheme", default="W10_C24_S200")
    ap.add_argument("--traj-dir", default="f1_trajectories")
    ap.add_argument(
        "--ensemble",
        nargs="*",
        default=["f1_trajectories", "f1_trajectories_seed1", "f1_trajectories_seed2"],
    )
    ap.add_argument(
        "--theta", default=str(REFERENCE_DIR / "cmaes_full_winner.json")
    )
    ap.add_argument(
        "--max-hours",
        type=int,
        default=None,
        help="truncate the run (validation / timing)",
    )
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="f3_ukf_correct.jsonl")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()

    out = RESULTS / a.out
    if a.summary:
        recs = [
            json.loads(line)
            for line in out.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        print(f"{len(recs)} records in {out}")
        print(summarize(recs))
        return

    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(v, "1")
    meta = load_meta()
    theta = json.loads(Path(a.theta).read_text(encoding="utf-8"))["theta"]
    series = load_series("test", indices=a.series)
    variants = [CorrConfig.parse(v) for v in a.variants]

    def traj(d, s):
        return np.load(RESULTS / d / a.scheme / f"series_{s.index:02d}.npz")[
            "hold"
        ].astype(float)

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["variant"], r["series_index"]))
    jobs = []
    for corr in variants:
        for s in series:
            if (corr.name, s.index) in done:
                continue
            pinn = traj(a.traj_dir, s)
            pstd = (
                np.std([traj(d, s) for d in a.ensemble], axis=0, ddof=1)
                if corr.use_ensemble
                else None
            )
            jobs.append(
                (
                    s,
                    meta,
                    theta,
                    pinn,
                    pstd,
                    corr,
                    a.max_hours,
                    RESULTS / "f3_trajectories",
                )
            )
    print(
        f"{len(jobs)} UKF runs ({len(done)} already in {out}); variants={[c.name for c in variants]}",
        flush=True,
    )

    with (
        ProcessPoolExecutor(max_workers=a.workers) as ex,
        out.open("a", encoding="utf-8") as fh,
    ):
        futs = {ex.submit(_run, j): j for j in jobs}
        for n, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            m = rec["metrics"]["after3d"]
            print(
                f"[{n}/{len(jobs)}] {rec['variant']:16s} series {rec['series_index']:2d} {rec['regime']:12s} NRMSE {m['model']['nrmse_overall_%']:6.2f} "
                f"ref {m['ukf']['nrmse_overall_%']:6.2f}  fired {rec['n_fired']}/{rec['n_due']}  dev-from-ref {rec['max_rel_dev_from_reference']:.2e}  {rec['seconds']:.0f}s",
                flush=True,
            )
    recs = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(summarize(recs))


if __name__ == "__main__":
    main()
