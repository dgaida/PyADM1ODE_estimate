"""Block F4 of the fusion study: effect of a fixed lag on PINN, UKF and their fusion.

Aggregates the JSON files written by ``fusion_rules.py`` for the PINN trajectories
``hold`` (lag 0), ``lag1`` and ``lag3``. The PINN is a smoother inside its window, so
its estimate of hour ``k`` improves once later measurements are in; the UKF is a filter
and does not change. A fused estimate at lag ``L`` combines the PINN's smoothed value
for hour ``k`` (available at ``k + L``) with the UKF's filtered value for hour ``k``.

    python experiments/pinn_v2/lag_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
SCHEME = "W10_C24_S200"
LAGS = (("hold", "0 d"), ("lag1", "1 d"), ("lag3", "3 d"))
RULES = ("ukf", "pinn", "lin_w0.5", "log_w0.25", "state_loso", "gate_loso")
CASES = (
    ("benchmark, shipped UKF", "benchmark", ""),
    ("benchmark_ccm, shipped UKF", "benchmark_ccm", ""),
    ("benchmark_hard, shipped UKF", "benchmark_hard", ""),
    ("benchmark, re-run UKF (no correction)", "benchmark", "_none"),
    ("benchmark, UKF corrected by the PINN (F3 periodic_u0.2)", "benchmark", "_periodic_u0.2"),
)


def load(dataset: str, traj: str, suffix: str) -> dict | None:
    f = RESULTS / f"f2_fusion_{dataset}_{SCHEME}_{traj}{suffix}.json"
    return json.loads(f.read_text(encoding="utf-8"))["rules"] if f.exists() else None


def main() -> None:
    out = {}
    for title, dataset, suffix in CASES:
        print(f"\n{title}   (NRMSE after day 3 [%]; in brackets: series better than the UKF, Wilcoxon p vs UKF)")
        print(f"{'rule':12s}" + "".join(f"{'lag ' + lab:>24s}" for _, lab in LAGS))
        tabs = {traj: load(dataset, traj, suffix) for traj, _ in LAGS}
        out[title] = {}
        for rule in RULES:
            row = f"{rule:12s}"
            out[title][rule] = {}
            for traj, lab in LAGS:
                t = tabs[traj]
                if t is None or rule not in t:
                    row += f"{'-':>24s}"
                    continue
                r = t[rule]
                out[title][rule][lab] = r["nrmse_after3d"]
                extra = "" if rule == "ukf" else f" ({r['better_than_ukf']:2d}/20, p={r['p_vs_ukf']:.2f})"
                row += f"{r['nrmse_after3d']:8.2f}{extra:>16s}"
            print(row)
    (RESULTS / "f4_lag_summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
