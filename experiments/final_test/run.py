"""Run the final test: resumable job scheduler.

The state of the run is the set of files under ``results_final/``; nothing else is
remembered. Each pass looks at what exists, writes the cheap derived files
(kinetics artifacts, the Test B error size) whose inputs are complete, and keeps
``--workers`` processes busy with the expensive jobs whose inputs exist. Stopping
the process at any point loses at most the jobs that were running; starting it
again continues where it stopped.

    python experiments/final_test/run.py run --workers 28
    python experiments/final_test/run.py status
    python experiments/final_test/run.py run --retry-failed     # retry recorded failures once more

Order of work, by dependency: nominal runs (U1 first, the calibration starts from
its estimate) -> Test A calibration per series -> calibrated artifacts -> Test A
runs; Test B artifacts ``ident`` / ``voll`` need nothing, ``real`` needs every
calibration (its error size comes from them).
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import jobs as J  # noqa: E402

#: Lower runs first. U1 nominal unblocks the calibration; long jobs before short ones.
PRIORITY = {"traj:U1:nominal": 0, "calib": 1, "traj:U1": 2, "traj:U2": 2, "traj:P1": 3,
            "traj:R1": 4, "obs": 5, "val": 6}


# ---------------------------------------------------------------------------
# Job graph
# ---------------------------------------------------------------------------
def all_jobs() -> list[tuple]:
    out = []
    for cond, ds, v in C.condition_jobs():
        kind = C.config()["variants"][v]["kind"]
        if kind == "observer":
            out.append(("obs", cond, ds, v))
        else:
            out += [("traj", cond, ds, v, i) for i in range(C.n_series(ds))]
    for ds in C.dataset_names():
        out += [("calib", ds, i) for i in range(C.n_series(ds))]
    _, val = J.val_series()
    out += [("val", i) for i in range(len(val))]
    return out


def job_done(job: tuple) -> bool:
    if job[0] == "traj":
        return C.is_done(C.traj_path(job[1], job[2], job[3], job[4]))
    if job[0] == "obs":
        return all(C.is_done(C.traj_path(job[1], job[2], job[3], i)) for i in range(C.n_series(job[2])))
    if job[0] == "calib":
        return C.is_done(C.calib_path(job[1], job[2]))
    if job[0] == "val":
        return C.is_done(C.results_dir() / "val" / "R1" / f"s{job[1]:02d}.npz")
    raise KeyError(job)


def job_ready(job: tuple) -> bool:
    if job[0] == "traj":
        cond, ds, _, i = job[1:]
        return cond == "nominal" or C.artifact_path(cond, ds, i).exists()
    if job[0] == "obs":
        cond, ds = job[1], job[2]
        return cond == "nominal" or all(C.artifact_path(cond, ds, i).exists() for i in range(C.n_series(ds)))
    if job[0] == "calib":
        return C.is_done(C.traj_path("nominal", job[1], "U1", job[2]))
    return True


def priority(job: tuple) -> tuple:
    if job[0] == "traj":
        key = "traj:U1:nominal" if (job[3] == "U1" and job[1] == "nominal") else f"traj:{job[3]}"
    else:
        key = job[0]
    return (PRIORITY.get(key, 9), job)


def execute(job: tuple) -> str:
    if job[0] == "traj":
        return J.run_traj(*job[1:])
    if job[0] == "obs":
        return J.run_observer(*job[1:])
    if job[0] == "calib":
        return J.run_calibration(*job[1:])
    if job[0] == "val":
        return J.run_val_open_loop(job[1])
    raise KeyError(job)


# ---------------------------------------------------------------------------
# Cheap derived files, written by the scheduler itself
# ---------------------------------------------------------------------------
def write_derived(log) -> None:
    keys = C.calibration_keys()
    for ds in C.dataset_names():
        for i in range(C.n_series(ds)):
            truth = C.true_factors(ds, i)
            for cond, factors in (("B_ident", {k: truth[k] for k in keys}), ("B_voll", truth)):
                if not C.artifact_path(cond, ds, i).exists():
                    C.write_artifact(cond, ds, i, factors, f"Test B {cond}: generator truth")
            # Test A: the calibration result, or nominal kinetics if the calibration failed
            cp = C.calib_path(ds, i)
            if not C.artifact_path("A_cal", ds, i).exists() and C.is_done(cp):
                if cp.exists():
                    rec = C.read_json(cp)
                    C.check_hash(rec["config_hash"], cp)
                    C.write_artifact("A_cal", ds, i, rec["factors"], f"Test A calibration, {rec['n_evals']} simulations")
                else:
                    C.write_artifact("A_cal", ds, i, {}, "Test A calibration FAILED: nominal kinetics")
    # Test B real: needs every calibration
    all_cal = [(ds, i) for ds in C.dataset_names() for i in range(C.n_series(ds))]
    if not C.sigma_real_path().exists() and all(C.is_done(C.calib_path(ds, i)) for ds, i in all_cal):
        errs = {k: [] for k in keys}
        for ds, i in all_cal:
            cp = C.calib_path(ds, i)
            if cp.exists():
                rec = C.read_json(cp)
                for k in keys:
                    errs[k].append(abs(rec["log_error"][k]))
        sigma = {k: float(np.median(v) / 0.6745) if v else 0.25 for k, v in errs.items()}
        C.atomic_write_json(C.sigma_real_path(), {"config_hash": C.config_hash(), "sigma": sigma,
                                                  "n_calibrations": len(errs[keys[0]])})
        log(f"Test B real: sigma per parameter {sigma}")
    if C.sigma_real_path().exists():
        for ds, i in all_cal:
            if not C.artifact_path("B_real", ds, i).exists():
                C.write_artifact("B_real", ds, i, C.kinetics_for("B_real", ds, i), "Test B real: truth x lognormal error")


# ---------------------------------------------------------------------------
def status(jobs: list[tuple]) -> dict:
    by = {}
    for job in jobs:
        key = job[0] if job[0] in ("calib", "val") else f"{job[0]}:{job[1]}:{job[3]}"
        d = by.setdefault(key, [0, 0, 0])
        d[0] += 1
        if job_done(job):
            d[1] += 1
    failed = sum(1 for _ in C.results_dir().rglob("*.failed.json")) if C.results_dir().exists() else 0
    return {"groups": by, "failed_records": failed}


def print_status(jobs) -> None:
    st = status(jobs)
    total = sum(v[0] for v in st["groups"].values())
    done = sum(v[1] for v in st["groups"].values())
    print(f"config {C.config_hash()}   jobs done {done}/{total}   failure records {st['failed_records']}")
    for key in sorted(st["groups"]):
        n, d, _ = st["groups"][key]
        print(f"  {key:<28} {d:4d}/{n}")


def cmd_run(a) -> int:
    log = C.Log(C.results_dir() / "run.log")
    if a.retry_failed:
        for p in C.results_dir().rglob("*.failed.json"):
            p.unlink()
        log("removed all failure records (--retry-failed)")
    jobs = all_jobs()
    log(f"start: config {C.config_hash()}, {len(jobs)} jobs, {a.workers} workers")
    running: dict = {}
    pool = ProcessPoolExecutor(max_workers=a.workers, initializer=J.init_worker)
    t_start = time.time()
    try:
        while True:
            write_derived(log)
            busy = {f for f in running.values()}
            ready = sorted((j for j in jobs if j not in busy and not job_done(j) and job_ready(j)), key=priority)
            for job in ready[: max(0, a.workers - len(running))]:
                running[pool.submit(execute, job)] = job
            if not running:
                missing = [j for j in jobs if not job_done(j)]
                if missing:
                    log(f"blocked: {len(missing)} jobs wait for inputs that cannot appear, e.g. {missing[:3]}")
                    return 1
                log(f"all {len(jobs)} jobs done in {(time.time() - t_start) / 3600:.1f} h")
                return 0
            finished, _ = wait(list(running), timeout=120, return_when=FIRST_COMPLETED)
            for fut in finished:
                job = running.pop(fut)
                try:
                    res = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception as exc:  # noqa: BLE001 - jobs record their own failures
                    res = f"error {exc!r}"
                log(f"{res:<7} {job}")
    except BrokenProcessPool:
        log("worker pool broke (a worker died); restart run.py to continue")
        return 2
    except KeyboardInterrupt:
        log("interrupted; restart run.py to continue")
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=28)
    r.add_argument("--retry-failed", action="store_true")
    sub.add_parser("status")
    a = ap.parse_args()
    if a.cmd == "status":
        print_status(all_jobs())
        return 0
    return cmd_run(a)


if __name__ == "__main__":
    raise SystemExit(main())
