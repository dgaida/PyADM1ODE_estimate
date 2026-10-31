"""Report tables for an architecture sweep (``arch_sweep.py`` results file).

python experiments/pinn_v2/report.py results/closure_sweep.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyadm1ode_estimation.estimation.pinn.model import ArchConfig
from pyadm1ode_estimation.estimation.pinn.train import (
    REPORT_BLOCKS,
    format_benchmark,
    format_summary,
    summarize,
)


def load(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def grid_table(recs: list[dict], key: str, window: str = "horizon") -> str:
    """depth x width grid of the median-over-series value of ``metrics[window][key]``."""
    vals: dict[tuple[int, int], list[float]] = {}
    for r in recs:
        a = ArchConfig(**r["arch"])
        vals.setdefault((a.depth, a.width), []).append(r["metrics"][window][key])
    depths = sorted({d for d, _ in vals})
    widths = sorted({w for _, w in vals})
    scale = 100.0 if "rel" in key else 1.0
    lines = [f"{window}/{key} (median over series{', %' if scale == 100 else ''})"]
    header = "depth/width"
    lines.append(f"{header:>12s}" + "".join(f"{w:>10d}" for w in widths))
    for d in depths:
        row = f"{d:>12d}"
        for w in widths:
            v = vals.get((d, w))
            row += f"{scale * float(np.median(v)):10.2f}" if v else f"{'-':>10s}"
        lines.append(row)
    return "\n".join(lines)


def block_table(summary: dict) -> str:
    blocks = list(REPORT_BLOCKS)
    lines = ["horizon relative RMSE per block (mean over states and series, %)"]
    lines.append(f"{'arch':16s}" + "".join(f"{b[:12]:>13s}" for b in blocks))
    for a, s in sorted(
        summary.items(), key=lambda kv: kv[1]["horizon_rel_rmse_median"][0]
    ):
        lines.append(
            f"{a:16s}"
            + "".join(f"{100 * s['horizon_blocks'][b][0]:13.2f}" for b in blocks)
        )
    return "\n".join(lines)


def regime_table(summary: dict) -> str:
    regimes = sorted(
        {r for s in summary.values() for r in s["per_regime_horizon_rel_rmse_median"]}
    )
    lines = ["horizon relative RMSE median per regime (mean over series, %)"]
    lines.append(f"{'arch':16s}" + "".join(f"{r:>13s}" for r in regimes))
    for a, s in sorted(
        summary.items(), key=lambda kv: kv[1]["horizon_rel_rmse_median"][0]
    ):
        lines.append(
            f"{a:16s}"
            + "".join(
                f"{100 * s['per_regime_horizon_rel_rmse_median'].get(r, float('nan')):13.2f}"
                for r in regimes
            )
        )
    return "\n".join(lines)


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "results" / "closure_sweep.jsonl"
    )
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent / path
    recs = load(path)
    summary = summarize(recs)
    print(f"{len(recs)} records, {len(summary)} architectures, file {path.name}\n")
    print(format_summary(summary), "\n")
    print(format_benchmark(summary), "\n")
    for key in (
        "rel_rmse_median",
        "rel_rmse_mean",
        "pH_rmse",
        "q_ch4_rel_rmse",
        "phys_residual_rms",
    ):
        print(grid_table(recs, key), "\n")
    print(grid_table(recs, "rel_rmse_median", window="fit"), "\n")
    print(grid_table(recs, "rel_rmse_median", window="beyond"), "\n")
    print(block_table(summary), "\n")
    print(regime_table(summary), "\n")
    # timing
    for a, s in summary.items():
        print(
            f"{a:16s} params {s['n_parameters']:8d}  mean seconds/run {s['seconds_mean']:.0f}"
        )


if __name__ == "__main__":
    main()
