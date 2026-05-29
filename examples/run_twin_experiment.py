"""Twin experiment for the 41-state Square-Root UKF.

Runs two copies of the multi-stage example plant:

* **Truth plant** — propagated with a known initial state and a noisy
  nominal substrate feed; its full trajectory is the ground truth.
* **Filter plant** — a UKF (built via :func:`build_ukf`) that starts from
  a perturbed prior and only sees noisy sensor readings.

Writes diagnostic plots (state trajectories, observations, production
estimate, NIS and 2σ coverage) to ``output/twin_experiment/``.

Usage::

    python examples/run_twin_experiment.py
    python examples/run_twin_experiment.py --duration-days 7 --dt-hours 1
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyadm1ode_estimation.estimation import (  # noqa: E402
    ADM1ProcessModel,
    InputSpec,
    build_ukf,
)
from pyadm1ode_estimation.estimation.sensors import (  # noqa: E402
    SensorAdapter,
    measure_truth_with_sensors,
)
from pyadm1ode_estimation.estimation.twin import (  # noqa: E402
    coverage_within_2sigma,
    run_filter,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "twin_experiment"
DIGESTER_ID = "primary"

# Multi-stage example mix: 40 m³/d split 0.67 / 0.32 / 0.01.
SUBSTRATES = [
    InputSpec("maize_silage", substrate_index=0, initial_flow=26.8),
    InputSpec("slurry", substrate_index=1, initial_flow=12.8),
    InputSpec("cereal_silage", substrate_index=2, initial_flow=0.4),
]
# Phase-1 sensor set: gas + methane + pH + one dosing sensor per substrate.
SENSORS = ["q_gas", "q_ch4", "ph", "substrate_dose"]

# Quality-class index lists for the 41 ADM1 channels (mirrors
# specs.py::_STATE_BLOCKS so plotting can group channels without
# importing private constants).
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup-days", type=float, default=30.0,
        help="Pre-simulate the plant for this many days so the filter starts "
        "from a settled operating point (default 30 d, 0 to skip).",
    )
    parser.add_argument(
        "--duration-days", type=float, default=5.0,
        help="Length of the twin / UKF run after warm-up (default 5 d).",
    )
    parser.add_argument(
        "--dt-hours", type=float, default=1.0,
        help="Filter step interval (default 1 h).",
    )
    parser.add_argument(
        "--warmup-dt-hours", type=float, default=24.0,
        help="Integration step for the warm-up phase (default 24 h).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-perturbation-relative", type=float, default=0.05,
        help="Relative Gaussian noise on the UKF prior (default 0.05; 0 = "
        "perfect initialisation).",
    )
    parser.add_argument(
        "--substrate-noise-relative", type=float, default=0.10,
        help="Relative per-step Gaussian noise on the truth substrate feed "
        "(default 0.10; models dosing/kg-to-m³ uncertainty, 0 = constant).",
    )
    parser.add_argument(
        "--plot-from-day", type=float, default=0.0,
        help="Skip this many days at the start when plotting (burn-in). "
        "Diagnostics are still computed over the full run.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Plot output directory (default {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def build_truth_sensors(obs, seed: int) -> Dict[str, SensorAdapter]:
    """One realistic ``PhysicalSensor`` per observation channel.

    Each sensor's ``measurement_noise`` is taken from the channel's
    ``noise_std`` so the UKF's ``R`` stays consistent with the actual
    instrumentation. Drift / response lag add bias the filter's
    white-noise model does not capture, bringing the twin closer to a
    real plant. The UKF itself never touches these sensors.
    """
    from pyadm1.components.sensors.physical import (  # type: ignore[import-not-found]
        PhysicalSensor,
    )

    sensors: Dict[str, SensorAdapter] = {}
    for off, ch in enumerate(obs.channels):
        is_ph = ch.name == "pH"
        sensor = PhysicalSensor(
            component_id=f"{ch.name}_sensor",
            sensor_type="pH" if is_ph else "flow",
            signal_key=ch.name,
            measurement_noise=float(ch.noise_std),
            drift_rate=0.005 if is_ph else 0.0,  # pH electrodes age slowly
            response_time=(60.0 if is_ph else 30.0) / 86400.0,  # 60 s / 30 s
            sample_interval=0.0,  # one sample per dt
            rng_seed=seed + off,
        )
        sensors[ch.name] = SensorAdapter(sensor, signal_key=ch.name)
    return sensors


def _robust_ylim(*series, percentile=(1.0, 99.0), pad=0.15):
    """Percentile-based Y-limit so the t=0 sigma-point spike on ŷ / x̂ does
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


def plot_trajectory_grid(time, truth, x_hat, std, spec, indices, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = (len(indices) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(13, 2.5 * rows), sharex=True)
    flat = axes.flat if rows > 1 else axes
    for i, ax in enumerate(flat):
        if i >= len(indices):
            ax.set_visible(False)
            continue
        idx = indices[i]
        ch = spec.channels[idx]
        ax.plot(time, truth[:, idx], "k-", lw=1.5, label="truth")
        ax.plot(time, x_hat[:, idx], "C0-", lw=1.1, label=r"$\hat{x}$")
        ax.fill_between(
            time,
            x_hat[:, idx] - 2.0 * std[:, idx],
            x_hat[:, idx] + 2.0 * std[:, idx],
            color="C0", alpha=0.15, label=r"$\pm 2\sigma$",
        )
        ylim = _robust_ylim(
            truth[:, idx], x_hat[:, idx],
            x_hat[:, idx] - 2.0 * std[:, idx], x_hat[:, idx] + 2.0 * std[:, idx],
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{ch.name} (idx {idx})", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(loc="best", fontsize=7)
    flat[-1].set_xlabel("time [d]")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_observations(obs_clean, obs_noisy, steps, time, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = list(obs_clean.columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, name in zip(axes.flat, channels):
        ax.plot(obs_clean.index, obs_clean[name].values, "k-", lw=1.5, label="clean truth")
        ax.plot(
            obs_noisy.index, obs_noisy[name].values, "rx",
            markersize=6, label="noisy measurement", alpha=0.7,
        )
        y_pred = np.array([s.y_pred.get(name, np.nan) for s in steps])
        ax.plot(time, y_pred, "C0-", lw=1.1, label=r"$\hat{y}$")
        ylim = _robust_ylim(obs_clean[name].values, obs_noisy[name].values, y_pred)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("time [d]")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(channels):]:
        ax.set_visible(False)
    fig.suptitle("Observations — truth, noisy measurement, filter prediction", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def evaluate_h_at_xhat(warmed_up_plant, spec, obs, x_hat_history, channel_names,
                       dt_hours: float = 1.0):
    """Deterministic ``h(x̂_k)`` per step — the Jensen-bias-free production
    estimate (a single plant evaluation at the posterior mean, vs the UKF's
    sigma-point-averaged ŷ). Each call runs the plant for ``dt_hours`` so the
    gas phase equilibrates as it does in the truth propagation."""
    plant_copy = copy.deepcopy(warmed_up_plant)
    process = ADM1ProcessModel(plant_copy, spec)
    process.snapshot()

    name_to_chan = {c.name: c for c in obs.channels}
    extractors = [name_to_chan[name].extractor for name in channel_names]

    dt = dt_hours / 24.0
    y_det = np.zeros((len(x_hat_history), len(channel_names)))
    for k in range(len(x_hat_history)):
        # Restore baseline, apply x̂, run one dt, read outputs — without
        # process.step(), so the read-back never overwrites x̂.
        process.restore()
        process._apply_state(spec.clip(x_hat_history[k]))
        plant_copy.step(dt)
        for j, ex in enumerate(extractors):
            y_det[k, j] = float(ex(plant_copy, x_hat_history[k]))
    return y_det


def plot_production_estimate(obs_clean, obs_noisy, steps, time, x_hat_history,
                             warmed_up_plant, spec, obs, output_path):
    """Gas + methane production: truth, raw + smoothed sensor, and the
    deterministic ``h(x̂)`` model estimate, with cumulative production and
    end-of-run error below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = ["Q_gas", "Q_ch4"]
    print("  Computing deterministic h(x_hat) per step ...")
    y_det = evaluate_h_at_xhat(warmed_up_plant, spec, obs, x_hat_history, channels)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, name in enumerate(channels):
        ax_rate = axes[0, col]
        truth_series = obs_clean[name].values
        noisy_series = obs_noisy[name].values
        y_hat = y_det[:, col]

        # Rolling mean of the noisy measurement — the operator's "actual"
        # production reading, free of model bias.
        sensor_series = pd.Series(noisy_series, index=obs_noisy.index)
        window = max(3, len(sensor_series) // 40)
        y_smoothed = sensor_series.rolling(
            window=window, center=True, min_periods=1,
        ).mean().values
        y_sigma = np.array([s.y_std.get(name, np.nan) for s in steps])

        ax_rate.plot(obs_clean.index, truth_series, "k-", lw=1.8, label="truth")
        ax_rate.plot(
            obs_noisy.index, noisy_series, "rx", markersize=4,
            label="sensor (raw)", alpha=0.4,
        )
        ax_rate.plot(
            obs_noisy.index, y_smoothed, "C3-", lw=1.4,
            label=f"sensor (smoothed, {window}-pt)",
        )
        ax_rate.plot(time, y_hat, "C2-", lw=1.2, label=r"$h(\hat{x})$ (model)", alpha=0.85)
        with_band = np.isfinite(y_hat) & np.isfinite(y_sigma)
        if with_band.any():
            ax_rate.fill_between(
                np.asarray(time)[with_band],
                (y_hat - y_sigma)[with_band], (y_hat + y_sigma)[with_band],
                color="C2", alpha=0.12, label=r"$h(\hat{x}) \pm 1\sigma$",
            )
        ylim = _robust_ylim(truth_series, noisy_series, y_hat, y_smoothed)
        if ylim is not None:
            ax_rate.set_ylim(*ylim)
        ax_rate.set_ylabel(f"{name} [m³/d]")
        ax_rate.set_title(f"{name} — instantaneous rate")
        ax_rate.legend(loc="best", fontsize=8)
        ax_rate.grid(alpha=0.3)

        ax_cum = axes[1, col]
        t_arr = np.asarray(obs_clean.index, dtype=float)

        def _cumulative(values, time_axis):
            v = np.asarray(values, dtype=float)
            dt_arr = np.diff(time_axis, prepend=time_axis[0])
            return np.cumsum(np.where(np.isfinite(v), v, 0.0) * dt_arr)

        cum_truth = _cumulative(truth_series, t_arr)
        cum_hxhat = _cumulative(y_hat, np.asarray(time, dtype=float))
        cum_sensor = _cumulative(y_smoothed, t_arr)

        ax_cum.plot(t_arr, cum_truth, "k-", lw=1.8, label="truth")
        ax_cum.plot(t_arr, cum_sensor, "C3-", lw=1.4, label="sensor (smoothed)")
        ax_cum.plot(time, cum_hxhat, "C2-", lw=1.2, label=r"$h(\hat{x})$ (model)", alpha=0.85)
        if len(t_arr) and len(cum_hxhat):
            err_sensor = (cum_sensor[-1] - cum_truth[-1]) / max(cum_truth[-1], 1e-12) * 100.0
            err_model = (cum_hxhat[-1] - cum_truth[-1]) / max(cum_truth[-1], 1e-12) * 100.0
            ax_cum.text(
                0.55, 0.18, f"sensor cum.: {err_sensor:+.1f} %",
                transform=ax_cum.transAxes, fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="C3", alpha=0.8),
            )
            ax_cum.annotate(
                f"model cum.: {err_model:+.1f} %",
                xy=(time[-1], cum_hxhat[-1]), xytext=(0.55, 0.05),
                textcoords="axes fraction", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="C2", alpha=0.8),
            )
        ax_cum.set_xlabel("time [d]")
        ax_cum.set_ylabel(f"cumulative {name} [m³]")
        ax_cum.set_title(f"{name} — cumulative production")
        ax_cum.legend(loc="best", fontsize=9)
        ax_cum.grid(alpha=0.3)

    fig.suptitle("Gas + methane production: truth vs deterministic UKF estimate", fontsize=13)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_nis(steps, time, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nis = np.array([s.nis for s in steps])
    n_active = len(steps[0].active_channels) if steps else 0
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(time, nis, "C2-o", lw=1.2, markersize=4, label="NIS")
    if n_active > 0:
        ax.axhline(float(n_active), color="k", ls="--", alpha=0.6,
                   label=f"expected NIS = {n_active}")
    ax.set_xlabel("time [d]")
    ax.set_ylabel("NIS")
    ax.set_title("Normalized Innovation Squared")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_summary(coverage, spec, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    block_means = {
        block: float(np.mean([coverage[i] for i in indices]))
        for block, indices in QUALITY_BLOCKS.items()
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(block_means.keys())
    values = [block_means[n] * 100.0 for n in names]
    bars = ax.bar(names, values, color="C0", alpha=0.7)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1, f"{v:.0f} %", ha="center", fontsize=9)
    ax.axhline(80.0, color="g", ls="--", alpha=0.5, label="80 % (strong target)")
    ax.axhline(40.0, color="orange", ls="--", alpha=0.5, label="40 % (weak target)")
    ax.axhline(20.0, color="r", ls="--", alpha=0.5, label="20 % (open-loop target)")
    ax.set_ylabel("2σ coverage [%]")
    ax.set_ylim(0, 110)
    ax.set_title("Per-block 2σ coverage (mean over channels in block)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def _warmup(plant, warmup_days, warmup_dt_hours, label):
    """Advance the plant past the t=0 substrate discontinuity to a settled
    operating point (no observations recorded)."""
    if warmup_days <= 0:
        return
    print(f"  Warming up {label} plant: {warmup_days:.0f} d at dt = {warmup_dt_hours:.0f} h ...")
    plant.simulate(
        duration=float(warmup_days),
        dt=float(warmup_dt_hours) / 24.0,
        save_interval=float(warmup_days),
    )


def _propagate_truth_with_substrate_noise(spec, truth_process, obs, x0, dt_hours,
                                          n_steps, rng, substrate_noise_relative,
                                          dt_stub=1e-5):
    """Propagate the truth trajectory with per-step Gaussian noise on the
    augmented substrate-input channels (models the operator's dosing
    uncertainty). Returns ``(time, states, obs_clean)`` like
    :func:`twin.propagate_truth`; the augmented channels in ``states`` hold
    the actually delivered (noisy) substrate values."""
    dt = dt_hours / 24.0
    n_state = len(spec)
    aug_indices = spec.kind_indices("input_flow")
    nominal = np.array([x0[i] for i in aug_indices], dtype=float)

    # Truth substrate trajectory: substrate_truth[k] is fed during step k→k+1.
    if substrate_noise_relative > 0.0:
        noise = rng.normal(
            0.0, substrate_noise_relative * np.abs(nominal),
            size=(n_steps + 1, len(aug_indices)),
        )
    else:
        noise = np.zeros((n_steps + 1, len(aug_indices)))
    substrate_truth = np.clip(nominal[None, :] + noise, a_min=0.0, a_max=None)

    states = np.zeros((n_steps + 1, n_state))
    states[0] = x0.copy()
    for k_aug, i_aug in enumerate(aug_indices):
        states[0, i_aug] = substrate_truth[0, k_aug]

    truth_process.refresh_outputs(states[0], dt_stub=dt_stub)
    obs_rows = [[float(c.extractor(truth_process.plant, states[0])) for c in obs.channels]]

    x = states[0].copy()
    for k in range(n_steps):
        for k_aug, i_aug in enumerate(aug_indices):
            x[i_aug] = substrate_truth[k, k_aug]
        x = truth_process.step(x, dt)
        states[k + 1] = x
        truth_process.refresh_outputs(x, dt_stub=dt_stub)
        obs_rows.append([float(c.extractor(truth_process.plant, x)) for c in obs.channels])

    time_arr = np.arange(n_steps + 1, dtype=float) * dt
    obs_clean = pd.DataFrame(obs_rows, index=time_arr, columns=[c.name for c in obs.channels])
    return time_arr, states, obs_clean


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_steps = int(round(args.duration_days * 24.0 / args.dt_hours))

    print("Twin experiment — multi-stage example, 41-state SR-UKF")
    print(f"  Warm-up: {args.warmup_days:.0f} d (no filter)")
    print(f"  Filter horizon: {args.duration_days:.1f} d  |  "
          f"dt = {args.dt_hours:.1f} h  |  {n_steps} steps")

    # ---- Truth + filter plants -----------------------------------------
    # The filter plant is a deepcopy of the warmed-up truth plant, so the
    # UKF model is bit-identical to the truth at t=0; a third copy is kept
    # pristine for the deterministic h(x̂) production estimate.
    print("\nBuilding + warming up truth plant ...")
    truth_plant = build_multi_stage_plant()
    _warmup(truth_plant, args.warmup_days, args.warmup_dt_hours, "truth")
    filter_plant = copy.deepcopy(truth_plant)
    det_h_plant = copy.deepcopy(truth_plant)

    # ---- UKF: the whole setup in one call ------------------------------
    ukf = build_ukf(
        filter_plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=SENSORS,
        initial_uncertainty_relative=max(args.initial_perturbation_relative, 1e-3),
    )
    spec, obs = ukf.spec, ukf.obs
    x_truth0 = ukf.x_hat.copy()  # build_ukf set this to the (warmed-up) plant state
    print(f"  State vector: {len(spec)} channels  ({len(spec) - 41} augmented)")
    print(f"  Observation channels: {[c.name for c in obs.channels]}")

    # ---- Propagate truth + measure -------------------------------------
    print(f"\nPropagating truth for {args.duration_days:.1f} d "
          f"(substrate noise: {args.substrate_noise_relative * 100:.0f} % per step) ...")
    truth_process = ADM1ProcessModel(truth_plant, spec)
    truth_process.snapshot()
    time, truth, obs_clean = _propagate_truth_with_substrate_noise(
        spec, truth_process, obs, x_truth0,
        dt_hours=args.dt_hours, n_steps=n_steps, rng=rng,
        substrate_noise_relative=args.substrate_noise_relative,
    )

    print("Stepping truth-side sensors (drift + lag + noise) ...")
    obs_noisy = measure_truth_with_sensors(obs_clean, build_truth_sensors(obs, args.seed))

    # ---- Perturb the prior (build_ukf starts at the exact truth) -------
    if args.initial_perturbation_relative > 0.0:
        pert = rng.normal(
            0.0, args.initial_perturbation_relative * (np.abs(x_truth0) + 1e-6),
            size=len(spec),
        )
        ukf.reset(spec.clip(x_truth0 + pert), ukf.P)
        print(f"\nInitial perturbation: ±{args.initial_perturbation_relative * 100:.0f} % "
              "Gaussian noise on the prior.")
    else:
        print("\nInitial perturbation: 0 (perfect initialisation).")

    print(f"Running UKF for {n_steps + 1} steps ...")
    x_hat, std, steps = run_filter(
        ukf, spec, obs, obs_noisy, gate_frame=None, dt_hours=args.dt_hours,
    )

    # ---- Diagnostics ---------------------------------------------------
    coverage = coverage_within_2sigma(truth, x_hat, std)
    nis = np.array([s.nis for s in steps], dtype=float)
    nis_finite = nis[np.isfinite(nis)]
    nis_mean = float(np.mean(nis_finite)) if len(nis_finite) else float("nan")
    n_active = len(steps[0].active_channels) if steps else 0

    print("\n--- Per-quality-block coverage ---")
    for block, indices in QUALITY_BLOCKS.items():
        block_cov = float(np.mean([coverage[i] for i in indices]))
        print(f"  {block:30s}  n={len(indices):2d}  2sigma-cov = {100 * block_cov:5.1f} %")
    print(f"\n  Mean NIS = {nis_mean:.2f}  "
          f"(target {0.5 * n_active:.1f} – {2.0 * n_active:.1f} for {n_active} channels)")

    # ---- Plot slice (visual burn-in; diagnostics use the full run) -----
    plot_mask = time >= args.plot_from_day
    if args.plot_from_day > 0:
        print(f"\n  Plot burn-in: skipping t < {args.plot_from_day:.1f} d "
              f"({int(plot_mask.sum())} of {len(time)} samples kept).")
    time_p, truth_p, x_hat_p, std_p = (
        time[plot_mask], truth[plot_mask], x_hat[plot_mask], std[plot_mask],
    )
    obs_clean_p = obs_clean.loc[obs_clean.index >= args.plot_from_day]
    obs_noisy_p = obs_noisy.loc[obs_noisy.index >= args.plot_from_day]
    steps_p = [s for s in steps if s.t >= args.plot_from_day]

    # ---- Plots ---------------------------------------------------------
    output_dir = args.output_dir
    print(f"\nWriting plots to {output_dir} ...")
    plot_trajectory_grid(
        time_p, truth_p, x_hat_p, std_p, spec,
        indices=[6, 8, 27, 35, 38, 40],
        title="Strong-observable states (A + D fused)",
        output_path=output_dir / "trajectories_strong.png",
    )
    plot_trajectory_grid(
        time_p, truth_p, x_hat_p, std_p, spec,
        indices=[0, 18, 22, 10, 11, 41],  # last is the first input_flow channel
        title="Medium / weak / open-loop states + 1 substrate input",
        output_path=output_dir / "trajectories_weak.png",
    )
    plot_observations(obs_clean_p, obs_noisy_p, steps_p, time_p,
                      output_path=output_dir / "observations.png")
    plot_production_estimate(
        obs_clean_p, obs_noisy_p, steps_p, time_p,
        x_hat_history=x_hat_p, warmed_up_plant=det_h_plant,
        spec=spec, obs=obs, output_path=output_dir / "production_estimate.png",
    )
    plot_nis(steps_p, time_p, output_path=output_dir / "nis.png")
    plot_coverage_summary(coverage, spec, output_path=output_dir / "coverage_summary.png")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
