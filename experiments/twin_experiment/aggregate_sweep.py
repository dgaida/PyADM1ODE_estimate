"""Aggregate the model-error sweep: read reports/results/<prefix>_s<sigma>.npz
files (each a report_compare run, full vs adcore) and produce the sigma-trend
table + plot reports/figures/<prefix>_trend.png (second-half per-block NRMSE vs
model-error sigma, full-41 vs A+D-core).

Usage:  python aggregate_sweep.py [PREFIX]   (default PREFIX=sweep)
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
RESULTS = _ROOT / "reports" / "results"
FIGURES = _ROOT / "reports" / "figures"

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "sweep"
files = sorted(glob.glob(f"{RESULTS}/{PREFIX}_s*.npz"))
if not files:
    raise SystemExit(f"no {RESULTS}/{PREFIX}_s*.npz files found")

runs = []
for f in files:
    d = np.load(f, allow_pickle=True)
    cands = [str(c) for c in d["candidates"]]
    blocks = [str(b) for b in d["blocks"]]
    runs.append(
        {
            "sigma": float(d["sigma"]),
            "cands": cands,
            "blocks": blocks,
            "summary": d["summary"],
            "nis": d["nis"],
            "cov": d["coverage"],
        }
    )
runs.sort(key=lambda r: r["sigma"])
sigmas = [r["sigma"] for r in runs]
blocks = runs[0]["blocks"]
key_blocks = [
    "weighted",
    "methanogenesis",
    "disintegration_split",
    "hydrolysis_sums",
    "inerts",
]

print(f"=== Model-error sweep (second-half NRMSE), sigmas={sigmas} ===")
for b in key_blocks:
    bi = blocks.index(b)
    print(f"\n{b}:")
    print(f"  {'sigma':>6s}  {'full':>8s}  {'adcore':>8s}  {'ratio f/a':>9s}")
    for r in runs:
        fi, ai = r["cands"].index("full"), r["cands"].index("adcore")
        vf, va = r["summary"][fi, bi], r["summary"][ai, bi]
        print(f"  {r['sigma']:6.2f}  {vf:8.3f}  {va:8.3f}  {vf / max(va, 1e-9):9.2f}")

print("\nCalibration (mean NIS / 2sigma-cov):")
for r in runs:
    fi, ai = r["cands"].index("full"), r["cands"].index("adcore")
    print(
        f"  sigma={r['sigma']:.2f}: full NIS={r['nis'][fi]:.1f} cov={100 * r['cov'][fi]:.0f}%  "
        f"adcore NIS={r['nis'][ai]:.1f} cov={100 * r['cov'][ai]:.0f}%"
    )

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    block_de = {
        "weighted": "Gesamt (Median)",
        "methanogenesis": "Methanogenese",
        "disintegration_split": "Disintegration",
        "hydrolysis_sums": "Hydrolyse-Summen",
        "inerts": "Inerten",
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, b in zip(axes.flat, key_blocks):
        bi = blocks.index(b)
        vf = [r["summary"][r["cands"].index("full"), bi] for r in runs]
        va = [r["summary"][r["cands"].index("adcore"), bi] for r in runs]
        ax.plot(sigmas, vf, "C0-o", label="voll-41")
        ax.plot(sigmas, va, "C1-s", label="A+D-Kern")
        ax.set_title(block_de.get(b, b), fontsize=12)
        ax.set_xlabel("Modellfehler σ")
        ax.set_ylabel("NRMSE (zweite Hälfte)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10)
    axes.flat[-1].axis("off")
    fig.suptitle(
        "Modellfehler-Trend: voll-41 gegen A+D-Kern (Steady-State)", fontsize=14
    )
    fig.tight_layout()
    fig.savefig(f"{FIGURES}/{PREFIX}_trend.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {FIGURES}/{PREFIX}_trend.png")
except Exception as e:  # noqa: BLE001
    print(f"(plot skipped: {e})")
