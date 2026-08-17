"""CLI to tune / calibrate a model-based filter on a dataset.

    # compare all methods (baseline σ vs 1.1 Q/R/P0 vs 1.2 differentiable) and pick the
    # best on validation, then report on the held-out test set — for one or both variants:
    python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter compare --variant both --jobs 24

    # or a single method:
    python -m ...filter_tuning.tune_filter sigma --dataset benchmark --variant full
    python -m ...filter_tuning.tune_filter noise --variant adcore --method random --n-iter 20 --jobs 24
    python -m ...filter_tuning.tune_filter diff  --days 8 --epochs 15

All commands load a dataset (name or path), make a stratified train/val/test split, run
the chosen tuner(s), and save a JSON. ``--jobs`` parallelises the UKF base runs across
processes; the differentiable filter (1.2) runs in one process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import metrics as M
from . import noise_search as ns
from . import sigma_calibration as sig
from .datasets import get_dataset
from .filter_runners import make_ukf_runner


def _splits(ds, a):
    return ds.make_splits(
        days=a.days,
        burnin_days=a.burnin,
        val_frac=a.val_frac,
        per_group_train=a.per_mode,
        per_group_val=a.val_per,
        per_group_test=a.test_per,
        seed=a.seed,
    )


def _fmt(m):
    return (
        f"cov {m['coverage']:.3f} | FOS/TAC-band {m['fostac_band_coverage']:.3f} | "
        f"crit bal-acc {m['critical_balacc']:.3f} (TPR {m['critical_tpr']:.2f}/TNR {m['critical_tnr']:.2f})"
    )


def _save(a, obj, tag):
    out = Path(a.out) if a.out else Path("filter_tuning_results")
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{tag}.json"
    with p.open("w", encoding="utf-8") as fh:
        json.dump(
            obj,
            fh,
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
    print(f"wrote {p}")


# --------------------------------------------------------------------------
# single methods
# --------------------------------------------------------------------------
def _baseline(runner, tr, va, te, jobs):
    tr_res = sig.collect(runner, tr, jobs=jobs)
    va_res = sig.collect(runner, va, jobs=jobs)
    best, _ = sig.search_sigma(tr_res, va_res)
    test = sig.apply_to(sig.collect(runner, te, jobs=jobs), best["std_scale"])
    val = {
        k: best[k]
        for k in (
            "coverage",
            "fostac_band_coverage",
            "critical_balacc",
            "critical_tpr",
            "critical_tnr",
        )
    }
    return {
        "method": "baseline_sigma",
        "val": val,
        "test": test,
        "theta": {
            "std_scale": best["std_scale"],
            "sigma_hi": best["sigma_hi"],
            "gamma": best["gamma"],
        },
    }


def _noise(runner, tr, va, te, a):
    best, hist = ns.search_noise(
        runner,
        tr,
        va,
        method=a.method,
        n_iter=a.n_iter,
        seed=a.seed,
        jobs=a.jobs,
        use_skopt=(a.method == "bayes"),
        verbose=True,
    )
    test = sig.apply_to(
        sig.collect(runner, te, theta=best["theta"], jobs=a.jobs),
        best["theta"]["std_scale"],
    )
    val = {
        k: best[k]
        for k in (
            "coverage",
            "fostac_band_coverage",
            "critical_balacc",
            "critical_tpr",
            "critical_tnr",
        )
    }
    return {
        "method": "noise_1.1",
        "val": val,
        "test": test,
        "theta": best["theta"],
        "history": hist[:8],
    }


def _diff_transfer(ds, runner, tr, va, te, a, diff_theta):
    """Apply 1.2's learned diagonal Q,R to THIS SR-UKF variant, + σ on top."""
    tr_res = sig.collect(runner, tr, theta=diff_theta, jobs=a.jobs)
    va_res = sig.collect(runner, va, theta=diff_theta, jobs=a.jobs)
    best, _ = sig.search_sigma(tr_res, va_res)
    theta = {**diff_theta, "std_scale": best["std_scale"]}
    test = sig.apply_to(
        sig.collect(runner, te, theta=theta, jobs=a.jobs), best["std_scale"]
    )
    val = {
        k: best[k]
        for k in (
            "coverage",
            "fostac_band_coverage",
            "critical_balacc",
            "critical_tpr",
            "critical_tnr",
        )
    }
    return {
        "method": "diff_1.2",
        "val": val,
        "test": test,
        "theta": {
            "q_diag": diff_theta.get("q_diag"),
            "r_diag": diff_theta.get("r_diag"),
            "sigma_hi": best["sigma_hi"],
            "gamma": best["gamma"],
        },
    }


# --------------------------------------------------------------------------
def cmd_sigma(a):
    ds = get_dataset(a.dataset)
    tr, va, te = _splits(ds, a)
    runner = make_ukf_runner(ds.meta, variant=a.variant)
    print(
        f"[sigma] {a.variant}: {len(tr)}/{len(va)}/{len(te)} train/val/test (D={a.days}, jobs={a.jobs})"
    )
    r = _baseline(runner, tr, va, te, a.jobs)
    print(
        f"BEST σ_hi={r['theta']['sigma_hi']} γ={r['theta']['gamma']}\nTEST: {_fmt(r['test'])}"
    )
    _save(a, {"variant": a.variant, **r}, f"sigma_{a.variant}")


def cmd_noise(a):
    ds = get_dataset(a.dataset)
    tr, va, te = _splits(ds, a)
    runner = make_ukf_runner(ds.meta, variant=a.variant)
    print(
        f"[noise 1.1] {a.variant}: {len(tr)}/{len(va)} train/val, method={a.method} n_iter={a.n_iter} jobs={a.jobs}"
    )
    r = _noise(runner, tr, va, te, a)
    print(f"TEST: {_fmt(r['test'])}")
    _save(a, {"variant": a.variant, **r}, f"noise_{a.variant}")


def cmd_diff(a):
    ds = get_dataset(a.dataset)
    tr, va, _te = _splits(ds, a)
    from .differentiable import DifferentiableEKF

    print(f"[diff 1.2] {len(tr)}/{len(va)} train/val (D={a.days}, epochs={a.epochs})")
    trainer = DifferentiableEKF(
        ds.meta, n_substeps=a.substeps if hasattr(a, "substeps") else 10
    )
    best, hist = trainer.fit(tr, va, epochs=a.epochs, lr=a.lr)
    th = trainer.as_theta()
    print(
        f"learned Q [{th['q_diag'].min():.1e},{th['q_diag'].max():.1e}] R {np.round(th['r_diag'],4).tolist()}"
    )
    _save(a, {"val_loss": best["val"], "history": hist, **th}, "diff")


# --------------------------------------------------------------------------
# cmaes: the full four-stage pipeline, meant to run unattended for many hours
# --------------------------------------------------------------------------
def _mail(subject: str, body: str, to: str) -> str:
    """Send a plain-text report. Returns a status line; never raises.

    Configuration comes from the environment only, so no credential ever lands in the
    repository:

        TUNING_SMTP_HOST      e.g. smtp.strato.de
        TUNING_SMTP_PORT      default 587 (STARTTLS); use 465 for implicit TLS
        TUNING_SMTP_USER      login, also used as the From address unless overridden
        TUNING_SMTP_PASSWORD  app password, not the account password
        TUNING_MAIL_FROM      optional, overrides the From address

    A missing or broken mail setup must never destroy a finished multi-hour run, so every
    failure is reported and swallowed. The report is on disk regardless.
    """
    import os
    import smtplib
    from email.message import EmailMessage

    host = os.environ.get("TUNING_SMTP_HOST")
    user = os.environ.get("TUNING_SMTP_USER")
    password = os.environ.get("TUNING_SMTP_PASSWORD")
    if not (host and user and password):
        return (
            "mail skipped: set TUNING_SMTP_HOST / TUNING_SMTP_USER / "
            "TUNING_SMTP_PASSWORD to enable it"
        )
    port = int(os.environ.get("TUNING_SMTP_PORT", "587"))
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = os.environ.get("TUNING_MAIL_FROM", user)
    msg["To"] = to
    msg.set_content(body)
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=60) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=60) as s:
                s.starttls()
                s.login(user, password)
                s.send_message(msg)
        return f"mail sent to {to}"
    except Exception as exc:  # noqa: BLE001
        return f"mail FAILED ({type(exc).__name__}: {exc})"


def _cmaes_report(a, res: dict) -> str:
    """Plain-text summary of a finished pipeline run."""
    L = [
        f"CMA-ES tuning finished — variant {a.variant}",
        "=" * 60,
        f"host              {__import__('socket').gethostname()}",
        (
            f"episodes          {res['n_sigma']} sigma-fit + {res['n_score']} scoring "
            f"(both from train), {res['n_val']} validation"
        ),
        f"horizon           {a.days} d",
        (
            f"search            popsize {a.popsize}, {res['gens_run']} of {a.gens} "
            f"generations{' (stopped early)' if res.get('stopped_early') else ''}"
        ),
        f"objective         {a.objective}",
        f"FOS/TAC titration {'off' if not a.fostac_every_days else f'every {a.fostac_every_days:g} d'}",
        f"wall clock        {res['hours']:.2f} h",
        "",
    ]
    L += ["Stage 1 — best per generation (score, best so far):"]
    for r in res["trace_rows"]:
        L.append(
            f"  gen {r['gen']:2d}  {r['best_gen']:+.4f}   {r['best_all']:+.4f}"
            + (f"   {r['minutes']:.0f} min" if r["minutes"] else "")
        )
    L += [
        "",
        "Stage 2 — candidates re-measured on validation at full length:",
        f"  {'candidate':16s}{'NRMSE':>9}{'AUC':>8}{'cov':>8}{'score':>10}",
    ]
    for k, v in res["stage2"].items():
        if v is None:
            L.append(f"  {k:16s}   failed")
            continue
        L.append(
            f"  {k:16s}{v['nrmse']:9.3f}{v['critical_auc']:8.3f}"
            f"{v['coverage']:8.3f}{res['stage2_scores'][k]:+10.3f}"
        )
    L += ["", f"Winner: {res['winner']}", ""]
    if res.get("test"):
        t = res["test"]
        L += [
            "Stage 3 — winner on the held-out test set (20 series, full length):",
            (
                f"  NRMSE {t['nrmse']:.3f} | AUC {t['critical_auc']:.3f} | "
                f"coverage {t['coverage']:.3f}"
            ),
            f"  series ok: {t['n_ok']}/{t['n_total']}",
            "",
        ]
    L += ["Artefacts:"] + [f"  {p}" for p in res["files"]]
    return "\n".join(L)


def cmd_cmaes(a):
    import socket
    import time

    from . import cmaes_search as cs

    t0 = time.time()
    out = Path(a.out) if a.out else Path("filter_tuning_results")
    out.mkdir(parents=True, exist_ok=True)
    tag = f"cmaes_{a.variant}"
    if a.fostac_every_days:
        tag += f"_fostac{a.fostac_every_days:g}d"
    ds = get_dataset(a.dataset)

    print(
        f"[cmaes] host {socket.gethostname()} | variant {a.variant} | D={a.days} | "
        f"jobs {a.jobs}",
        flush=True,
    )

    # -- stage 0 -----------------------------------------------------------------------
    emp_path = Path(a.empirical_q) if a.empirical_q else out / "empirical_noise.json"
    if not emp_path.exists():
        print(f"[stage 0] measuring empirical Q/R -> {emp_path}", flush=True)
        cs.empirical_q(ds, jobs=a.jobs, save_to=emp_path)
    q_emp = cs.load_empirical_q(emp_path)
    print(f"[stage 0] empirical Q from {emp_path}", flush=True)

    # -- episodes ----------------------------------------------------------------------
    # Both the sigma-fit and the scoring set come from the TRAIN pool, so the validation
    # pool stays untouched by the search and is a genuine hold-out for stage 2.
    n_per = a.sigma_per + a.score_per
    train_all, val_eps, _ = ds.make_splits(
        days=a.days,
        burnin_days=a.burnin,
        val_frac=a.val_frac,
        per_group_train=n_per,
        per_group_val=a.val_per,
        seed=a.seed,
    )
    sigma_eps, score_eps = cs.split_per_label(train_all, a.sigma_per)
    print(
        f"[episodes] {len(sigma_eps)} sigma-fit + {len(score_eps)} scoring (train), "
        f"{len(val_eps)} validation",
        flush=True,
    )

    objective = None
    if a.objective == "accuracy":
        objective = lambda m, b: M.objective_accuracy_first(m, b, w_guard=a.w_guard)
    elif a.objective == "combined":
        objective = lambda m, b: M.objective_combined(
            m, b, w_acc=a.w_acc, w_dec=1.0 - a.w_acc
        )

    # -- stage 1 -----------------------------------------------------------------------
    trace_path = out / f"{tag}_stage1.json"

    def progress(e):
        print(
            f"  gen {e['gen']:2d}: best_gen {e['best_gen']:+.4f} | "
            f"best_all {e['best_so_far']:+.4f} | {e['n_ok']}/{a.popsize} ok | "
            f"{e['seconds'] / 60:.0f} min | total {(time.time() - t0) / 3600:.1f} h",
            flush=True,
        )

    print(
        f"[stage 1] CMA-ES, popsize {a.popsize}, up to {a.gens} generations", flush=True
    )
    trace = cs.run_cmaes(
        ds,
        q_emp,
        variant=a.variant,
        sigma_eps=sigma_eps,
        fostac_every_days=a.fostac_every_days,
        score_eps=score_eps,
        popsize=a.popsize,
        generations=a.gens,
        base_nrmse=a.base_nrmse,
        patience=a.patience,
        jobs=a.jobs,
        objective=objective,
        out_path=trace_path,
        progress=progress,
        cma_seed=a.cma_seed,
    )
    base = trace["config"]["base_nrmse"]

    # -- stage 2 -----------------------------------------------------------------------
    blocks = trace["config"]["blocks"]
    cands = sorted(
        (
            c
            for g in trace["generations"]
            for c in g["candidates"]
            if not c.get("failed")
        ),
        key=lambda c: -c["score"],
    )
    if not cands:
        raise RuntimeError(
            f"stage 1 finished {len(trace['generations'])} generations without a single "
            f"evaluable candidate, so there is nothing to validate. Every candidate either "
            f"failed to integrate or scored non-finite. Check the horizon and the episode "
            f"lengths before spending more compute."
        )
    picked, seen = [], []
    for c in cands:  # top-k, skipping near-duplicates
        x = np.asarray(c["x"], float)
        if any(np.allclose(x, s, atol=1e-6) for s in seen):
            continue
        seen.append(x)
        picked.append(x)
        if len(picked) >= a.top_k:
            break
    configs = {"nominal": {}}
    configs.update(
        {
            f"cma_{i + 1}": cs.theta_from_x(x, q_emp, blocks)
            for i, x in enumerate(picked)
        }
    )
    vd = a.verify_days  # None = full series, which is the point of stage 2
    print(
        f"[stage 2] {len(configs)} configs on {len(val_eps)} validation episodes, "
        f"{'full length' if vd is None else f'{vd} d'}",
        flush=True,
    )
    ver = cs.verify(
        ds,
        configs,
        variant=a.variant,
        days=vd,
        burnin_days=a.burnin,
        fostac_every_days=a.fostac_every_days,
        per_group_train=a.sigma_per,
        per_group_val=a.val_per,
        seed=a.seed,
        jobs=a.jobs,
        out_path=out / f"{tag}_stage2.json",
    )
    score_of = objective or (lambda m, b: M.objective_combined(m, b))
    scores = {k: score_of(v, base) for k, v in ver["results"].items() if v is not None}
    scores = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not scores:
        why = "\n  ".join(ns.last_failures()) or "no reason recorded"
        raise RuntimeError(
            f"stage 2 produced no scorable candidate out of {len(configs)}. The search "
            f"itself succeeded, so this is a verification problem, not a tuning one. "
            f"Reported causes:\n  {why}"
        )
    winner = max(scores, key=scores.get)
    print(
        f"[stage 2] winner: {winner} "
        f"({len(scores)}/{len(configs)} configs scorable)",
        flush=True,
    )

    # -- stage 3 -----------------------------------------------------------------------
    test = None
    files = [str(trace_path), str(out / f"{tag}_stage2.json")]
    if not a.no_test:
        theta = dict(configs[winner])
        theta["std_scale"] = np.asarray(ver["results"][winner]["std_scale"], float)
        npz = out / f"{tag}_reference.npz"
        print(f"[stage 3] winner over the test split -> {npz}", flush=True)
        ref = cs.build_reference(
            ds,
            theta,
            variant=a.variant,
            split="test",
            jobs=a.jobs,
            fostac_every_days=a.fostac_every_days,
            out_npz=npz,
        )
        z = (
            np.load(Path(a.dataset_dir) / "test.npz", allow_pickle=True)
            if a.dataset_dir
            else None
        )
        test = {
            "n_ok": len(ref["x_hat"]) - len(ref["failed"]),
            "n_total": len(ref["x_hat"]),
            "minutes": ref["minutes"],
        }
        if z is not None:
            B = 48
            t = np.concatenate([np.asarray(s, float)[B:] for s in z["states"]])
            x = np.concatenate([np.asarray(v, float)[B:] for v in ref["x_hat"]])
            s = np.concatenate([np.asarray(v, float)[B:] for v in ref["std"]])
            test.update(
                {
                    k: v
                    for k, v in M.evaluate(t, x, s).items()
                    if k in ("nrmse", "coverage", "critical_auc")
                }
            )
        files.append(str(npz))

    rows = cs.trace_table(trace)
    summary = {
        "variant": a.variant,
        "host": socket.gethostname(),
        "hours": (time.time() - t0) / 3600.0,
        "objective": a.objective,
        "n_sigma": len(sigma_eps),
        "n_score": len(score_eps),
        "n_val": len(val_eps),
        "gens_run": len(trace["generations"]),
        "stopped_early": trace.get("stopped_early"),
        "base_nrmse": base,
        "trace_rows": rows,
        "stage2": ver["results"],
        "stage2_scores": scores,
        "winner": winner,
        "winner_theta": {k: v for k, v in configs[winner].items() if k != "q_diag"},
        "test": test,
        "files": files,
    }
    sp = out / f"{tag}_summary.json"
    with sp.open("w", encoding="utf-8") as fh:
        json.dump(
            summary,
            fh,
            indent=2,
            default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o),
        )
    files.append(str(sp))

    report = _cmaes_report(a, summary)
    print("\n" + report, flush=True)
    (out / f"{tag}_report.txt").write_text(report, encoding="utf-8")
    if a.email:
        status = _mail(
            f"[{socket.gethostname()}] CMA-ES tuning done — {a.variant} "
            f"({summary['hours']:.1f} h)",
            report,
            a.email,
        )
        print(status, flush=True)


# --------------------------------------------------------------------------
# compare: baseline vs 1.1 vs 1.2 -> val-select -> test, for one/both variants
# --------------------------------------------------------------------------
def cmd_compare(a):
    ds = get_dataset(a.dataset)
    variants = ["full", "adcore"] if a.variant == "both" else [a.variant]

    # 1.2 is trained once (variant-independent full EKF); transferred to each SR-UKF variant
    from .differentiable import DifferentiableEKF

    tr0, va0, _ = _splits(ds, a)
    print(
        f"[compare] training 1.2 (differentiable) once: {len(tr0)}/{len(va0)} train/val, epochs={a.diff_epochs}"
    )
    diff = DifferentiableEKF(ds.meta)
    diff.fit(tr0, va0, epochs=a.diff_epochs, lr=a.lr, verbose=True)
    diff_theta = diff.as_theta()  # {"q_diag","r_diag"}

    all_results = {}
    for v in variants:
        tr, va, te = _splits(ds, a)
        runner = make_ukf_runner(ds.meta, variant=v)
        print(
            f"\n===== variant {v}: {len(tr)}/{len(va)}/{len(te)} train/val/test (D={a.days}, jobs={a.jobs}) ====="
        )
        res = {}
        print("[1/3] baseline (σ only)")
        res["baseline"] = _baseline(runner, tr, va, te, a.jobs)
        print(f"  baseline TEST {_fmt(res['baseline']['test'])}")
        print("[2/3] 1.1 (Q/R/P0 search)")
        res["noise_1.1"] = _noise(runner, tr, va, te, a)
        print("[3/3] 1.2 (differentiable Q,R -> this SR-UKF)")
        res["diff_1.2"] = _diff_transfer(ds, runner, tr, va, te, a, diff_theta)

        winner = max(res, key=lambda k: M.objective(res[k]["val"]))
        print(f"\n  -- variant {v} summary (VAL objective / TEST) --")
        for k in ("baseline", "noise_1.1", "diff_1.2"):
            mark = " <== winner" if k == winner else ""
            print(
                f"    {k:11s} val_obj {M.objective(res[k]['val']):+.3f} | TEST {_fmt(res[k]['test'])}{mark}"
            )
        res["winner"] = winner
        all_results[v] = res
        _save(a, {"variant": v, "winner": winner, **res}, f"compare_{v}")

    _save(
        a,
        {
            "variants": variants,
            "summary": {
                v: {
                    "winner": all_results[v]["winner"],
                    "test": all_results[v][all_results[v]["winner"]]["test"],
                }
                for v in variants
            },
        },
        "compare_summary",
    )
    print("\n=== OVERALL ===")
    for v in variants:
        w = all_results[v]["winner"]
        print(f"  {v}: winner={w} | TEST {_fmt(all_results[v][w]['test'])}")


# --------------------------------------------------------------------------
def main():
    # Docstrings and progress output contain sigma, arrows and multiplication signs. On a
    # Windows console (cp1252) writing those raises UnicodeEncodeError, which used to make
    # even `--help` crash. Force UTF-8 where the stream supports it.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a reconfigurable text stream
            pass

    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sigma", "noise", "diff", "compare", "cmaes"):
        p = sub.add_parser(name)
        p.add_argument("--dataset", default="benchmark")
        p.add_argument(
            "--variant",
            choices=["full", "adcore", "adcore_vfa", "both"],
            default="full",
        )
        p.add_argument("--days", type=float, default=12.0)
        p.add_argument("--burnin", type=float, default=2.0)
        p.add_argument("--val-frac", type=float, default=0.2)
        p.add_argument("--per-mode", type=int, default=2)
        p.add_argument("--val-per", type=int, default=2)
        p.add_argument("--test-per", type=int, default=3)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--jobs", type=int, default=1)
        p.add_argument("--out", default=None)
        if name in ("noise", "compare"):
            p.add_argument(
                "--method", choices=["random", "grid", "bayes"], default="random"
            )
            p.add_argument("--n-iter", type=int, default=12)
        if name in ("diff", "compare"):
            p.add_argument("--lr", type=float, default=0.1)
        if name == "diff":
            p.add_argument("--epochs", type=int, default=15)
            p.add_argument("--substeps", type=int, default=10)
        if name == "compare":
            p.add_argument("--diff-epochs", type=int, default=15)
        if name == "cmaes":
            p.add_argument(
                "--fostac-every-days",
                type=float,
                default=None,
                help="add the FOS/TAC titration every N days (7 = weekly, "
                "1 = daily). Omitted means no lab measurement at all",
            )
            p.add_argument(
                "--empirical-q",
                default=None,
                help="stage-0 JSON to reuse; measured and saved if absent",
            )
            p.add_argument(
                "--sigma-per",
                type=int,
                default=3,
                help="train episodes per mode used to fit sigma",
            )
            p.add_argument(
                "--score-per",
                type=int,
                default=3,
                help="train episodes per mode the score is computed on",
            )
            p.add_argument("--popsize", type=int, default=8)
            p.add_argument("--gens", type=int, default=8)
            p.add_argument(
                "--patience",
                type=int,
                default=3,
                help="stop after N generations without improvement, 0 = never",
            )
            p.add_argument(
                "--top-k",
                type=int,
                default=3,
                help="candidates carried into the validation stage",
            )
            p.add_argument(
                "--objective", choices=["accuracy", "combined"], default="accuracy"
            )
            p.add_argument(
                "--w-guard",
                type=float,
                default=2.0,
                help="accuracy objective: weight of the anti-inversion guard",
            )
            p.add_argument(
                "--w-acc",
                type=float,
                default=0.7,
                help="combined objective: accuracy weight",
            )
            p.add_argument(
                "--base-nrmse",
                type=float,
                default=None,
                help="skip measuring the nominal reference and use this instead",
            )
            p.add_argument("--cma-seed", type=int, default=1)
            p.add_argument(
                "--verify-days",
                type=float,
                default=None,
                help="stage-2 horizon; default is the full series, which is the "
                "whole point of stage 2. Only shorten it for smoke tests",
            )
            p.add_argument(
                "--no-test",
                action="store_true",
                help="stop after validation, do not touch the test set",
            )
            p.add_argument(
                "--dataset-dir",
                default=None,
                help="path to the dataset dir, enables test scoring in the report",
            )
            p.add_argument(
                "--email",
                default=None,
                help="address for the final report; SMTP comes from the env",
            )
    a = ap.parse_args()
    if getattr(a, "patience", None) == 0:
        a.patience = None
    {
        "sigma": cmd_sigma,
        "noise": cmd_noise,
        "diff": cmd_diff,
        "compare": cmd_compare,
        "cmaes": cmd_cmaes,
    }[a.cmd](a)


if __name__ == "__main__":
    main()
