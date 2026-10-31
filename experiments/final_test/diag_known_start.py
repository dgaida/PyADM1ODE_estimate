"""Diagnostic outside the frozen test: does the Test A calibration fail because of its start state?

Repeats the Test A calibration of a few test series with the TRUE state at hour 48
as the start (everything else identical). Not a legitimate estimator input; it only
separates the two causes of a poor calibration: a wrong start state that the kinetics
compensate, or kinetics the sensors cannot pin down. Results go to
``results_final/diagnostics/known_start/`` and are not part of the scored test.

    python experiments/final_test/diag_known_start.py --series 0 1 2 3 4 5 --workers 3
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import jobs as J  # noqa: E402


def one(args) -> str:
    ds, i = args
    out = C.results_dir() / "diagnostics" / "known_start" / ds / f"s{i:02d}.json"
    if out.exists():
        return f"skip {ds} {i}"
    J.init_worker()
    from forward import calibrate

    cfg = C.config()["calibration"]
    s = C.dataset(ds).test[i]
    h0, h1 = cfg["fit_hours"]
    keys = C.calibration_keys()
    res = calibrate(C.dataset(ds).meta, s.measurements, s.feed, np.asarray(s.truth, float)[h0], keys, (h0, h1),
                    budget=int(cfg["budget"]), log_bound=float(cfg["log_bound"]), xtol=float(cfg["xtol"]), ftol=float(cfg["ftol"]))
    truth = C.true_factors(ds, i)
    res["log_error"] = {k: float(np.log(res["factors"][k] / truth[k])) for k in keys}
    res["log_error_nominal"] = {k: float(-np.log(truth[k])) for k in keys}
    C.atomic_write_json(out, {"diagnostic": "true state at hour 48 as start", "ds": ds, "i": i, "keys": keys, **res})
    return f"ok {ds} {i}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="benchmark")
    ap.add_argument("--series", nargs="*", type=int, default=[0, 1, 2, 3, 4, 5])
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=J.init_worker) as ex:
        for msg in ex.map(one, [(a.dataset, i) for i in a.series]):
            print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
