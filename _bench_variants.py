"""Twin-experiment bench for the current SR-UKF.

Historical: this script originally compared the pre-2026-06 "redraw"
SR-UKF against the SigmaReuseUKF variant. After the main-path migration
both have been unified into the canonical Wan-VdM 2001 form, so the
bench now runs a single variant (the merged ``UnscentedKalmanFilter``).
Compare against the recorded pre-migration numbers in
:doc:`/development/ukf_performance` to verify the speedup persists.

* Plant: ``build_simple_plant`` (single fermenter + storage + CHP,
  n_state = 41 ADM1 + 2 augmented input_flows = 43).
* Truth: propagated under a fixed RNG seed; same noisy observations
  fed to the filter.
* Reported: wall time, RMSE per quality block, mean NIS, 2sigma
  coverage per block.

Run from the repo root::

    python _bench_variants.py --duration-hours 24
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

# Force UTF-8 stdout so block-summary tables with non-ASCII glyphs survive
# Windows' default cp1252 encoding.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import time
from dataclasses import dataclass
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conftest  # noqa: F401, E402

import numpy as np  # noqa: E402

from pyadm1ode_estimation.estimation import (  # noqa: E402
    InputSpec,
    build_filter_components,
)
from pyadm1ode_estimation.estimation.filters import (  # noqa: E402
    ParallelUKF,
    UnscentedKalmanFilter,
)
from pyadm1ode_estimation.estimation.process_model import ADM1ProcessModel  # noqa: E402
from pyadm1ode_estimation.estimation.twin import (  # noqa: E402
    add_measurement_noise,
    coverage_within_2sigma,
    propagate_truth,
    run_filter,
)
from pyadm1ode_estimation.example_plants import build_simple_plant  # noqa: E402

DIGESTER_ID = "fermenter"
SUBSTRATES = [
    InputSpec("maize_silage", substrate_index=0, initial_flow=10.0),
    InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
]
SENSORS = ["q_gas", "q_ch4", "ph", "substrate_dose"]


def make_components_for_workers():
    """Top-level builder so ``multiprocessing.spawn`` can pickle a reference
    to it. Each worker calls this once at pool startup to construct its own
    (process, obs, spec) triple — same setup as the main process."""
    plant = build_simple_plant()
    return build_filter_components(
        plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=SENSORS,
    )


# Same channel grouping as run_twin_experiment for cross-comparable RMSE
# and coverage. Indices < 41 are ADM1 channels; 41+ are augmented.
QUALITY_BLOCKS: Dict[str, List[int]] = {
    "methanogenesis": [6, 7, 8, 9, 27, 28, 37, 38, 39, 40],
    "charge_balance": [29, 30, 31, 32, 33, 34, 35, 36],
    "acidogenesis_substrates": [0, 1, 3, 4, 5],
    "acidogenesis_biomass": [22, 23, 25, 26],
    "hydrolysis_sums": [18, 19, 20],
    "disintegration_split": [12, 13, 14, 15, 16, 17],
    "nitrogen": [10],
    "inerts": [11, 21],
    "fa_block": [2, 24],
    "input_flow": [41, 42],
}


@dataclass
class VariantResult:
    label: str
    elapsed_s: float
    x_hat: np.ndarray
    std: np.ndarray
    nis: np.ndarray
    coverage: np.ndarray
    block_rmse: Dict[str, float]
    block_coverage: Dict[str, float]
    # Optional payload kept for plotting; small enough to ignore otherwise.
    steps: List = None  # type: ignore[assignment]


def _build_ukf_with_class(
    plant,
    cls,
    initial_uncertainty_relative=0.05,
    gamma_override=None,
    n_workers=1,
):
    """Build a UKF of the given class on a given (already-warmed-up) plant.

    Mirrors :func:`build_ukf` but lets the caller choose the filter
    class, the sigma-point scaling, and (for :class:`ParallelUKF`) the
    worker-pool size. Returns ``(ukf, spec, obs, x_truth0)``.
    """
    process, obs, spec = build_filter_components(
        plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=SENSORS,
    )
    ukf_kwargs = dict(alpha=1.0, beta=2.0, kappa=0.0, gamma_override=gamma_override)
    if cls is ParallelUKF:
        ukf = cls(
            process,
            obs,
            spec,
            n_workers=n_workers,
            components_builder=make_components_for_workers,
            **ukf_kwargs,
        )
    else:
        ukf = cls(process, obs, spec, **ukf_kwargs)

    x0 = spec.read_adm1_state(plant)
    aug_starts = 41
    for i, ch in enumerate(spec.channels[aug_starts:], start=aug_starts):
        x0[i] = ch.initial

    sigma_rel = max(initial_uncertainty_relative, 1.0e-3)
    p0_diag = (sigma_rel * (np.abs(x0) + 1.0e-6)) ** 2
    P0 = np.diag(p0_diag)
    ukf.reset(x0, P0)
    return ukf, spec, obs, x0


def _warmup(plant, days, dt_hours, label):
    if days <= 0:
        return
    print(f"  warmup {label}: {days:.0f} d at dt={dt_hours:.0f} h ...")
    plant.simulate(
        duration=float(days), dt=float(dt_hours) / 24.0, save_interval=float(days)
    )


def _build_truth_and_obs(args):
    """Same truth across variants. Fixed seed → reproducible noise."""
    rng = np.random.default_rng(args.seed)
    truth_plant = build_simple_plant()
    _warmup(truth_plant, args.warmup_days, args.warmup_dt_hours, "truth")

    # We need the *spec* once to drive truth propagation; build it via a
    # throw-away UKF (the spec is the same across variants by construction).
    truth_ukf, spec, obs, x_truth0 = _build_ukf_with_class(
        truth_plant, UnscentedKalmanFilter
    )
    del truth_ukf  # not used; we only wanted spec/obs

    truth_process = ADM1ProcessModel(truth_plant, spec)
    truth_process.snapshot()
    n_steps = int(round(args.duration_hours))
    time_arr, truth, obs_clean = propagate_truth(
        spec, truth_process, obs, x_truth0, dt_hours=1.0, n_steps=n_steps
    )
    obs_noisy = add_measurement_noise(obs_clean, obs, rng)

    print(
        f"  truth propagated: {n_steps} steps, "
        f"{len(spec)} state channels, {len(obs.channels)} obs channels"
    )
    return truth_plant, spec, obs, x_truth0, time_arr, truth, obs_clean, obs_noisy


def _run_variant(
    label,
    cls,
    truth_plant,
    spec,
    obs,
    x_truth0,
    time_arr,
    truth,
    obs_noisy,
    initial_perturbation_rel,
    seed,
    gamma_override=None,
    n_workers=1,
):
    """Build a fresh filter plant + UKF of class ``cls`` and step it
    through the same noisy observations."""
    rng = np.random.default_rng(seed + 1)  # different seed slot than truth
    filter_plant = copy.deepcopy(truth_plant)
    ukf, _, _, _ = _build_ukf_with_class(
        filter_plant,
        cls,
        initial_uncertainty_relative=max(initial_perturbation_rel, 1e-3),
        gamma_override=gamma_override,
        n_workers=n_workers,
    )

    if initial_perturbation_rel > 0:
        pert = rng.normal(
            0.0,
            initial_perturbation_rel * (np.abs(x_truth0) + 1e-6),
            size=len(spec),
        )
        ukf.reset(spec.clip(x_truth0 + pert), ukf.P)

    t0 = time.perf_counter()
    try:
        x_hat, std, steps = run_filter(
            ukf, spec, obs, obs_noisy, gate_frame=None, dt_hours=1.0
        )
    finally:
        # Release worker pool if any.
        if hasattr(ukf, "shutdown"):
            ukf.shutdown()
    elapsed = time.perf_counter() - t0

    nis = np.array([s.nis for s in steps])
    coverage = coverage_within_2sigma(truth, x_hat, std)
    block_rmse = {}
    block_cov = {}
    for block, idx in QUALITY_BLOCKS.items():
        rmse = float(np.sqrt(np.mean((truth[:, idx] - x_hat[:, idx]) ** 2)))
        block_rmse[block] = rmse
        block_cov[block] = float(np.mean(coverage[idx]))
    return VariantResult(
        label=label,
        elapsed_s=elapsed,
        x_hat=x_hat,
        std=std,
        nis=nis,
        coverage=coverage,
        block_rmse=block_rmse,
        block_coverage=block_cov,
        steps=steps,
    )


def _print_summary(results: List[VariantResult]):
    base = results[0]
    print("\n" + "=" * 72)
    print(f"{'Variant':<22} {'time [s]':>10} {'speedup':>9} {'mean NIS':>10}")
    print("-" * 72)
    for r in results:
        speed = base.elapsed_s / r.elapsed_s
        nis_mean = float(np.nanmean(r.nis))
        print(f"{r.label:<22} {r.elapsed_s:>10.2f} {speed:>8.2f}x {nis_mean:>10.2f}")
    print("=" * 72)

    print("\nBlock-RMSE (lower is better) — truth vs x_hat:")
    blocks = list(QUALITY_BLOCKS.keys())
    header = f"{'block':<28}" + "".join(f"{r.label:>14}" for r in results)
    print(header)
    print("-" * len(header))
    for block in blocks:
        vals = [r.block_rmse[block] for r in results]
        row = f"{block:<28}"
        for v in vals:
            row += f"{v:>14.4g}"
        print(row)

    print("\nBlock-Coverage (within 2sigma; closer to 95 % is better):")
    header = f"{'block':<28}" + "".join(f"{r.label:>14}" for r in results)
    print(header)
    print("-" * len(header))
    for block in blocks:
        vals = [r.block_coverage[block] for r in results]
        row = f"{block:<28}"
        for v in vals:
            row += f"{100 * v:>13.1f}%"
        print(row)

    # Pairwise mean-trajectory difference baseline vs each variant.
    print("\nMean-trajectory delta vs baseline (max abs diff over time, per channel,")
    print("normalised by baseline std at that channel):")
    for r in results[1:]:
        diff = np.abs(r.x_hat - base.x_hat)
        scale = np.maximum(np.mean(base.std, axis=0), 1e-12)
        rel = diff / scale[None, :]
        max_per_chan = rel.max(axis=0)
        worst = np.argsort(max_per_chan)[-5:][::-1]
        print(f"  {r.label}: max relative delta = {max_per_chan.max():.3g} sigma")
        for idx in worst:
            print(f"    channel {idx:2d}: {max_per_chan[idx]:.3g} sigma")


# --------------------------------------------------------------------------
# Plotting helpers (opt-in via --save-plots)
# --------------------------------------------------------------------------
def _import_matplotlib():
    """Lazy import so the bench runs without matplotlib if no plots wanted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _robust_ylim(*series, percentile=(1.0, 99.0), pad=0.15):
    """Percentile-based Y-limit so the t=0 sigma-point spike on x_hat does
    not flatten the steady-state trajectory."""
    finite_vals = []
    for s in series:
        if s is None:
            continue
        arr = np.asarray(s, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if len(arr):
            finite_vals.append(arr)
    if not finite_vals:
        return None
    pooled = np.concatenate(finite_vals)
    lo, hi = np.percentile(pooled, percentile)
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 1e-6) * 0.1
    return lo - pad * span, hi + pad * span


def _plot_state_trajectories(time_arr, truth, x_hat, std, spec, output_path, title):
    """Per-channel time-series grid: truth vs x_hat with ±2σ band, one
    subplot per state channel."""
    plt = _import_matplotlib()

    n_chan = truth.shape[1]
    n_cols = 4
    n_rows = (n_chan + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.0 * n_cols, 2.0 * n_rows), sharex=True
    )
    flat = axes.flat
    for i in range(n_rows * n_cols):
        ax = flat[i]
        if i >= n_chan:
            ax.set_visible(False)
            continue
        ch = spec.channels[i]
        ax.plot(time_arr, truth[:, i], "k-", lw=1.4, label="truth")
        ax.plot(time_arr, x_hat[:, i], "C0-", lw=1.0, label=r"$\hat{x}$")
        band_lo = x_hat[:, i] - 2.0 * std[:, i]
        band_hi = x_hat[:, i] + 2.0 * std[:, i]
        ax.fill_between(
            time_arr, band_lo, band_hi, color="C0", alpha=0.18, label=r"$\pm 2\sigma$"
        )
        ylim = _robust_ylim(truth[:, i], x_hat[:, i], band_lo, band_hi)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{ch.name} (idx {i})", fontsize=8)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.legend(loc="best", fontsize=7)
    for ax in list(flat)[-n_cols:]:
        if ax.get_visible():
            ax.set_xlabel("time [d]")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_observations(time_arr, obs_clean, obs_noisy, steps, output_path, title):
    """For every observation channel: noise-free truth, noisy sample, ŷ."""
    plt = _import_matplotlib()

    channels = list(obs_clean.columns)
    n = len(channels)
    n_cols = min(3, n)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(5.0 * n_cols, 3.0 * n_rows), squeeze=False
    )
    flat = axes.flat
    for ax, name in zip(flat, channels):
        ax.plot(
            obs_clean.index, obs_clean[name].values, "k-", lw=1.4, label="clean truth"
        )
        ax.plot(
            obs_noisy.index,
            obs_noisy[name].values,
            "rx",
            markersize=4,
            alpha=0.6,
            label="noisy measurement",
        )
        y_pred = np.array([s.y_pred.get(name, np.nan) for s in steps])
        ax.plot(time_arr, y_pred, "C0-", lw=1.0, label=r"$\hat{y}$")
        ylim = _robust_ylim(obs_clean[name].values, obs_noisy[name].values, y_pred)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("time [d]")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=7)
        ax.grid(alpha=0.25)
    for ax in list(flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_nis(time_arr, nis, n_active, output_path, title):
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(time_arr, nis, "C2-o", lw=1.0, markersize=3, label="NIS")
    if n_active > 0:
        ax.axhline(
            float(n_active),
            color="k",
            ls="--",
            alpha=0.6,
            label=f"expected NIS = {n_active}",
        )
    ax.set_xlabel("time [d]")
    ax.set_ylabel("NIS")
    ax.set_title(title)
    if np.nanmax(nis) > 10:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_coverage(block_coverage, output_path, title):
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(11, 4.5))
    names = list(block_coverage.keys())
    values = [100 * block_coverage[n] for n in names]
    bars = ax.bar(names, values, color="C0", alpha=0.75)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1,
            f"{v:.0f} %",
            ha="center",
            fontsize=8,
        )
    ax.axhline(95.0, color="g", ls="--", alpha=0.5, label="95 % (target)")
    ax.axhline(80.0, color="orange", ls="--", alpha=0.5, label="80 % (acceptable)")
    ax.axhline(50.0, color="r", ls="--", alpha=0.5, label="50 % (under-covered)")
    ax.set_ylabel("2σ coverage [%]")
    ax.set_ylim(0, 110)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)


def _save_plots_for_variant(
    result, time_arr, truth, obs_clean, obs_noisy, spec, output_dir
):
    """Emit the full diagnostic plot set for one variant into
    ``output_dir/<label>/``."""
    from pathlib import Path

    out = Path(output_dir) / result.label
    out.mkdir(parents=True, exist_ok=True)
    print(f"  saving plots → {out}")

    title_prefix = f"{result.label} | {len(time_arr) - 1} steps"
    _plot_state_trajectories(
        time_arr,
        truth,
        result.x_hat,
        result.std,
        spec,
        out / "state_trajectories.png",
        title=f"{title_prefix} — truth vs x_hat ± 2σ",
    )
    _plot_observations(
        time_arr,
        obs_clean,
        obs_noisy,
        result.steps,
        out / "observations.png",
        title=f"{title_prefix} — observations",
    )
    n_active = len(result.steps[0].active_channels) if result.steps else 0
    _plot_nis(
        time_arr,
        result.nis,
        n_active,
        out / "nis.png",
        title=f"{title_prefix} — normalised innovation squared",
    )
    _plot_coverage(
        result.block_coverage,
        out / "coverage_per_block.png",
        title=f"{title_prefix} — 2σ coverage per quality block",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-days", type=float, default=5.0)
    parser.add_argument("--warmup-dt-hours", type=float, default=24.0)
    parser.add_argument("--duration-hours", type=int, default=24)
    parser.add_argument("--initial-perturbation-relative", type=float, default=0.05)
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save state-trajectory, observation, NIS and coverage plots "
        "to --output-dir/<variant-label>/.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Plot output directory. Defaults to " "output/twin_<duration>h/.",
    )
    args = parser.parse_args()

    if args.save_plots:
        if args.output_dir is None:
            args.output_dir = f"output/twin_{args.duration_hours}h"
        # pyadm1's plant build chdir's into the data directory during
        # construction, so a relative output dir would otherwise land
        # under PyADM1ODE/output/. Anchor it to this script's directory.
        if not os.path.isabs(args.output_dir):
            args.output_dir = os.path.join(HERE, args.output_dir)

    print("=" * 72)
    print(f"Twin benchmark — simple plant, seed={args.seed}")
    print(f"  Warmup: {args.warmup_days:.0f} d  |  Twin: {args.duration_hours} h")
    print("=" * 72)

    print("\n[1/3] Building truth + noisy observations ...")
    (
        truth_plant,
        spec,
        obs,
        x_truth0,
        time_arr,
        truth,
        obs_clean,
        obs_noisy,
    ) = _build_truth_and_obs(args)

    # Default benchmark: parallel-scaling sweep (Wave 4 — the headline
    # speedup the production filter delivers). Edit this list to compare
    # other variants. Notable past configurations preserved in git history:
    #   * γ-sweep:    UnscentedKalmanFilter with different gamma_override
    #   * cUKF audit: UnscentedKalmanFilter vs ConstrainedUKF (the
    #                 Wave 5 negative-result run — see ukf_performance.md)
    variants = [
        ("production_parallel_8", ParallelUKF, 8),
    ]
    results: List[VariantResult] = []
    for label, cls, n_workers in variants:
        print(f"\n[2/3] Running variant: {label} (n_workers={n_workers}) ...")
        r = _run_variant(
            label,
            cls,
            truth_plant,
            spec,
            obs,
            x_truth0,
            time_arr,
            truth,
            obs_noisy,
            args.initial_perturbation_relative,
            args.seed,
            n_workers=n_workers,
        )
        print(f"  done in {r.elapsed_s:.2f} s, mean NIS = {np.nanmean(r.nis):.2f}")
        results.append(r)
        if args.save_plots:
            _save_plots_for_variant(
                r, time_arr, truth, obs_clean, obs_noisy, spec, args.output_dir
            )

    print("\n[3/3] Summary")
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
