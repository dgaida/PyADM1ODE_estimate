"""Architecture sweep for PINN v2, the trajectory PINN (structure study, step 1).

Each architecture is trained separately on every selected benchmark series;
one JSON line per (architecture, series) goes to the results file so a run
can be resumed or extended. ``--summary`` prints the table over all records.

    python experiments/pinn_v2/arch_sweep.py --split test --steps 3000 --workers 10 \
        --archs d2_w64 d4_w64 d8_w64 d2_w256 d4_w256 d8_w256
    python experiments/pinn_v2/arch_sweep.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyadm1ode_estimation.estimation.pinn.data import (
    load_meta,
    load_series,
    train_split_log_stats,
    train_split_scale_reference,
)
from pyadm1ode_estimation.estimation.pinn.model import ArchConfig
from pyadm1ode_estimation.estimation.pinn.train import (
    TrainConfig,
    format_benchmark,
    format_summary,
    summarize,
    train_trajectory,
    ukf_warmup_state,
)

RESULTS = Path(__file__).resolve().parent / "results"


def parse_arch(name: str) -> ArchConfig:
    try:
        return ArchConfig.parse(name)
    except ValueError as e:
        raise SystemExit(str(e)) from e


def expand_archs(names: list[str]) -> list[str]:
    """Expand ``a|b`` alternatives inside tokens into the Cartesian product.

    ``d4_w128_tanh|silu_ff4|rff`` -> ``d4_w128_tanh_ff4``, ``d4_w128_tanh_rff``, ...
    A token ``-`` inside an alternative means "omit this token".
    """
    import itertools

    out = []
    for name in names:
        groups = [tok.split("|") for tok in name.split("_")]
        for combo in itertools.product(*groups):
            out.append("_".join(t for t in combo if t and t != "-"))
    return out


def _run(args):
    series, arch, cfg, meta, threads, prior_stats, ic_state, scale_ref = args
    import torch

    torch.set_num_threads(threads)
    res = train_trajectory(
        series,
        arch,
        cfg,
        meta=meta,
        prior_stats=prior_stats,
        ic_state=ic_state,
        scale_ref=scale_ref,
    )
    return res.to_record()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument(
        "--series",
        nargs="*",
        type=int,
        default=None,
        help="series indices (default: all)",
    )
    ap.add_argument(
        "--archs",
        nargs="*",
        default=[f"d{d}_w{w}" for d in (2, 4, 8) for w in (32, 128, 512)],
        help="architecture names, e.g. d4_w128 d8_w512_silu_ff4",
    )
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--t-obs", type=float, default=30.0)
    ap.add_argument("--horizon", type=float, default=10.0)
    ap.add_argument("--lambda-phys", type=float, default=1.0)
    ap.add_argument("--closure", action="store_true")
    ap.add_argument(
        "--nominal-kinetics",
        action="store_true",
        help="physics with nominal instead of series kinetics",
    )
    ap.add_argument(
        "--data",
        choices=("states", "measurements"),
        default="states",
        help="data loss on the true states (capacity study) or on the 5 sensors (field setting)",
    )
    ap.add_argument(
        "--prior",
        choices=("series", "train_split"),
        default=None,
        help="output normalisation; default: 'series' with --data states, 'train_split' with measurements",
    )
    ap.add_argument(
        "--noisy-feed", action="store_true", help="use feed_noisy instead of feed_true"
    )
    ap.add_argument(
        "--ic-prior",
        action="store_true",
        help="loose prior on x(0) from the plant warm-up state the UKF starts from",
    )
    ap.add_argument(
        "--estimate-kinetics",
        action="store_true",
        help="inverse PINN: learn the perturbed kinetic factors with the generator's prior",
    )
    ap.add_argument("--lr-kinetics", type=float, default=5.0e-3)
    ap.add_argument("--lambda-kinetics-prior", type=float, default=1.0)
    ap.add_argument(
        "--kinetics-keys",
        nargs="*",
        default=None,
        help="subset of kinetic keys to estimate (default: all 26 perturbed keys)",
    )
    ap.add_argument(
        "--causal", type=float, default=0.0, help="causal weighting eps (0 = off)"
    )
    ap.add_argument(
        "--time-march",
        type=float,
        default=0.0,
        help="time-marching window [d] (0 = off)",
    )
    ap.add_argument(
        "--greybox", action="store_true", help="learned rate correction g(x, u)"
    )
    ap.add_argument("--lambda-greybox", type=float, default=1.0)
    ap.add_argument(
        "--ukf-x0",
        action="store_true",
        help="tight prior on x(0) from the series' UKF estimate",
    )
    ap.add_argument("--lbfgs", type=int, default=0, help="L-BFGS iterations after Adam")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="torch threads per worker")
    ap.add_argument("--out", default="sweep.jsonl")
    ap.add_argument(
        "--summary", action="store_true", help="only print the summary of --out"
    )
    ap.add_argument("--seed", type=int, default=0)
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
        s = summarize(recs)
        print(format_summary(s))
        print()
        print(format_benchmark(s))
        return

    prior = a.prior or ("train_split" if a.data == "measurements" else "series")
    cfg = TrainConfig(
        t_obs_days=a.t_obs,
        horizon_days=a.horizon,
        steps=a.steps,
        lambda_phys=a.lambda_phys,
        closure=a.closure,
        use_series_kinetics=not a.nominal_kinetics,
        use_true_feed=not a.noisy_feed,
        data_source=a.data,
        prior=prior,
        ic_prior=a.ic_prior or a.ukf_x0,
        lambda_ic=10.0 if a.ukf_x0 else 1.0,
        causal_eps=a.causal,
        time_march_days=a.time_march,
        greybox=a.greybox,
        lambda_greybox=a.lambda_greybox,
        lbfgs_steps=a.lbfgs,
        estimate_kinetics=a.estimate_kinetics,
        kinetics_keys=tuple(a.kinetics_keys) if a.kinetics_keys else None,
        lr_kinetics=a.lr_kinetics,
        lambda_kinetics_prior=a.lambda_kinetics_prior,
        seed=a.seed,
    )
    meta = load_meta()
    series = load_series(a.split, indices=a.series)
    archs = [parse_arch(n) for n in expand_archs(a.archs)]
    print(f"{len(archs)} architectures: {[x.name for x in archs]}", flush=True)
    prior_stats = train_split_log_stats() if prior == "train_split" else None
    ic_state = ukf_warmup_state(meta) if a.ic_prior else None
    scale_ref = train_split_scale_reference() if a.data == "measurements" else None

    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((ArchConfig(**r["arch"]).name, r["series_index"]))
    jobs = [
        (
            s,
            arch,
            cfg,
            meta,
            a.threads,
            prior_stats,
            (s.ukf_x_hat[0] if a.ukf_x0 else ic_state),
            scale_ref,
        )
        for arch in archs
        for s in series
        if (arch.name, s.index) not in done
    ]
    print(
        f"{len(jobs)} runs to do ({len(done)} already in {out}); cfg={asdict(cfg)}",
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
            if "horizon" not in m:
                m = dict(m, horizon=m["fit"])
            print(
                f"[{n}/{len(jobs)}] {ArchConfig(**rec['arch']).name:20s} series {rec['series_index']:2d} {rec['regime']:12s} "
                f"fit med {100 * m['fit']['rel_rmse_median']:6.2f}%  horizon med {100 * m['horizon']['rel_rmse_median']:6.2f}%  "
                f"pH {m['horizon']['pH_rmse']:.3f}  {rec['seconds']:.0f}s",
                flush=True,
            )
    recs = [
        json.loads(line)
        for line in out.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    s = summarize(recs)
    print(format_summary(s))
    print()
    print(format_benchmark(s))


if __name__ == "__main__":
    main()
