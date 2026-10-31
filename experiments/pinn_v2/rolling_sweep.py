"""Block F1 of the fusion study: causal rolling-window PINN schemes on the test series.

    python experiments/pinn_v2/rolling_sweep.py --schemes W10_C24_S200 W20_C24_S500 ... \
        --workers 26 --out f1_rolling.jsonl
    python experiments/pinn_v2/rolling_sweep.py --summary --out f1_rolling.jsonl

Per (scheme, series) one JSON line with the benchmark scores of the four derived
trajectories (hold, adm1, lag1, lag3) and an ``.npz`` with the trajectories for the
fusion post-processing (block F2).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyadm1ode_estimation.estimation.pinn.data import (
    load_meta,
    load_series,
    train_split_log_stats,
    train_split_scale_reference,
)
from pyadm1ode_estimation.estimation.pinn.model import ArchConfig
from pyadm1ode_estimation.estimation.pinn.rolling import (
    RollingConfig,
    run_rolling,
)
from pyadm1ode_estimation.estimation.pinn.train import TrainConfig

RESULTS = Path(__file__).resolve().parent / "results"
_SCHEME = re.compile(r"^W(\d+(?:\.\d+)?)_C(\d+(?:\.\d+)?)_S(\d+)$")


def parse_scheme(name: str) -> RollingConfig:
    m = _SCHEME.match(name)
    if not m:
        raise SystemExit(f"bad scheme {name!r}; expected e.g. W20_C24_S200")
    return RollingConfig(
        window_days=float(m.group(1)),
        cadence_hours=float(m.group(2)),
        steps_warm=int(m.group(3)),
    )


def _run(args):
    series, arch, cfg, roll, meta, threads, prior_stats, scale_ref, traj_dir = args
    import torch

    torch.set_num_threads(threads)
    res = run_rolling(
        series, arch, cfg, roll, meta=meta, prior_stats=prior_stats, scale_ref=scale_ref
    )
    d = traj_dir / roll.name
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        d / f"series_{series.index:02d}.npz",
        time=series.time,
        **{k: v.astype(np.float32) for k, v in res.trajectories.items()},
    )
    return res.to_record()


def summarize(recs: list[dict]) -> str:
    by: dict[str, list[dict]] = {}
    for r in recs:
        by.setdefault(RollingConfig(**r["rolling"]).name, []).append(r)
    lines = [
        f"{'scheme':16s} {'traj':7s} {'n':>3s} {'NRMSE':>7s} {'>3d':>7s} {'UKF':>6s} {'ratio':>6s} {'<1':>5s} {'trans':>7s} {'s/op-h':>7s} {'retrain':>8s}"
    ]
    for name, rs in sorted(by.items()):
        for traj in ("hold", "adm1", "lag1", "lag3"):
            f = np.array(
                [r["metrics"][traj]["full"]["model"]["nrmse_overall_%"] for r in rs]
            )
            a = np.array(
                [r["metrics"][traj]["after3d"]["model"]["nrmse_overall_%"] for r in rs]
            )
            u = np.array(
                [r["metrics"][traj]["full"]["ukf"]["nrmse_overall_%"] for r in rs]
            )
            ratio = np.array(
                [r["metrics"][traj]["full"]["vs_ukf"]["rmse_ratio_overall"] for r in rs]
            )
            tr = np.array(
                [r["metrics"][traj]["full"]["model"]["nrmse_transient_%"] for r in rs]
            )
            lines.append(
                f"{name:16s} {traj:7s} {len(rs):3d} {f.mean():7.2f} {a.mean():7.2f} {u.mean():6.2f} {ratio.mean():6.2f} "
                f"{int((ratio < 1).sum()):2d}/{len(rs):<2d} {tr.mean():7.2f} "
                f"{np.mean([r['seconds_per_operating_hour'] for r in rs]):7.1f} {np.mean([r['n_retrainings'] for r in rs]):8.0f}"
            )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", nargs="*", default=["W20_C24_S200"])
    ap.add_argument("--arch", default="d4_w128_tanh_ff4")
    ap.add_argument("--series", nargs="*", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", default="f1_rolling.jsonl")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--data-dir",
        default=None,
        help="dataset directory with test.npz + meta.json (default: the benchmark); "
        "prior and residual scale always come from the benchmark's training split",
    )
    ap.add_argument(
        "--traj-dir",
        default="f1_trajectories",
        help="sub-folder of results/ for the npz trajectories",
    )
    a = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
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

    cfg = TrainConfig(
        t_obs_days=60.0,
        horizon_days=0.0,
        closure=True,
        data_source="measurements",
        prior="train_split",
        use_true_feed=False,
        use_series_kinetics=False,
        seed=a.seed,
    )
    arch = ArchConfig.parse(a.arch)
    meta = load_meta(a.data_dir)
    series = load_series("test", path=a.data_dir, indices=a.series)
    prior_stats = train_split_log_stats()
    scale_ref = train_split_scale_reference()
    schemes = [parse_scheme(s) for s in a.schemes]
    traj_dir = RESULTS / a.traj_dir

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((RollingConfig(**r["rolling"]).name, r["series_index"]))
    jobs = [
        (s, arch, cfg, roll, meta, a.threads, prior_stats, scale_ref, traj_dir)
        for roll in schemes
        for s in series
        if (roll.name, s.index) not in done
    ]
    print(
        f"{len(jobs)} chains to run ({len(done)} already in {out}); arch={arch.name}; cfg={asdict(cfg)}",
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
            m = rec["metrics"]
            print(
                f"[{n}/{len(jobs)}] {RollingConfig(**rec['rolling']).name:16s} series {rec['series_index']:2d} {rec['regime']:12s} "
                f"hold {m['hold']['full']['model']['nrmse_overall_%']:6.2f}  adm1 {m['adm1']['full']['model']['nrmse_overall_%']:6.2f}  "
                f"lag1 {m['lag1']['full']['model']['nrmse_overall_%']:6.2f}  UKF {m['hold']['full']['ukf']['nrmse_overall_%']:6.2f}  "
                f"{rec['seconds_total']:.0f}s",
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
