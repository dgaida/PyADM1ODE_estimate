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

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyadm1ode_estimation.estimation import (
    BLOCK_INDICES,
    STATE_BLOCKS,
    ADM1ProcessModel,
    InputSpec,
    KineticSpec,
    adm1da_full_spec,
    build_filter_components,
    build_ukf,
)
from pyadm1ode_estimation.estimation.realism import (
    MODEL_ERROR_KINETIC_SIGMA,
    build_sensor_noise,
)
from pyadm1ode_estimation.estimation.realism import (
    MODEL_ERROR_PREFIXES as _MODEL_ERROR_PREFIXES,
)
from pyadm1ode_estimation.estimation.sensors import (
    SensorAdapter,
    measure_truth_with_sensors,
)
from pyadm1ode_estimation.estimation.twin import (
    coverage_within_2sigma,
    run_filter,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "twin_experiment"
DIGESTER_ID = "primary"

# Multi-stage example mix: ~23.4 m³/d, manure-based 2024 annual feed log.
# substrate_index must match the plant's Q_substrates slot order
# (see _SUBSTRATE_MIX in example_plants/multi_stage.py).
SUBSTRATES = [
    InputSpec("maize_silage", substrate_index=0, initial_flow=4.74),
    InputSpec("solid_manure", substrate_index=1, initial_flow=13.70),
    InputSpec("chicken_litter", substrate_index=2, initial_flow=1.09),
    InputSpec("slurry", substrate_index=3, initial_flow=3.68),
    InputSpec("cereal_grain", substrate_index=4, initial_flow=0.20),
]
# Phase-1 sensor set: gas + methane + pH + one dosing sensor per substrate.
# q_gas / q_ch4 are now scoped to DIGESTER_ID (the estimated Fermenter), so
# the downstream stages' gas no longer biases the innovation.
SENSORS = ["q_gas", "q_ch4", "ph", "substrate_dose"]

# Real-plant sampling periods [h] per resolved channel name. The NDIR gas
# analyser delivers CH4/CO2 once per day (--daily-gas-analytics); the VFA/FOS
# titration every 12 h (--irregular-sensors adds it). q_gas/pH/doses stay
# hourly. O2/H2S are measured daily in practice but are NOT ADM1da states, so
# the UKF cannot assimilate them (monitoring only).
_DAILY_GAS_CHANNELS = {"Q_ch4": 24.0, "Q_co2": 24.0}
_VFA_INTERVAL_H = {"VFA": 12.0}

# Optional time-varying feed (--feed-change): the substrate composition steps
# at the given day. Flows are in SUBSTRATES order
# [maize_silage, solid_manure, chicken_litter, slurry, cereal_grain].
# Compositions chosen (and verified on the warmed plant) so the primary's
# Q_gas / Q_ch4 differ clearly between phases:
#   P0 ~1924 / 899  ->  P1 ~2407 / 1074 (+25 %/+19 %)  ->  P2 ~1858 / 902.
_FEED_PHASES = [
    (0.0, [4.74, 13.70, 1.09, 3.68, 0.20]),  # days 0-3: nominal manure-based mix
    (3.0, [20.0, 16.0, 2.0, 2.0, 4.0]),  # days 3-6: high-energy load (maize+grain)
    (6.0, [0.5, 6.0, 0.3, 4.0, 0.05]),  # days 6-9: low dilute load (slurry-heavy)
]
# Set by main() when --feed-change is active; read by the plot helpers to draw
# vertical markers at the change times.
_FEED_CHANGE_DAYS: list[float] = []

# Cascade stages for the per-stage gas diagnostic (truth vs. corrected model).
# Only "primary" is estimated by the UKF; "secondary" (Nachgärer) and
# "storage" run open-loop, driven by the corrected primary's effluent.
# (Kinetic-parameter prefixes perturbed by --model-error-std come from
# ``estimation.realism`` so the realism foundation is the single source.)
STAGE_IDS = ["primary", "secondary", "storage"]
STAGE_LABELS = {
    "primary": "Fermenter (primary) — truth vs. UKF estimate",
    "secondary": "Nachgärer (secondary) — truth vs. corrected model",
    "storage": "Storage — truth vs. corrected model",
}

# Quality-class index lists for the 41 ADM1 channels (mirrors
# specs.py::_STATE_BLOCKS so plotting can group channels without
# importing private constants).
QUALITY_BLOCKS: dict[str, list[int]] = {
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


def spec_block_positions(spec) -> dict[str, list[int]]:
    """Map each observability block to the STATE-VECTOR POSITIONS of its
    estimated ADM1 channels.

    Robust to a reduced spec: with the full 41-state vector position == ADM1
    index, but a reduced vector reorders/omits channels, so we look each
    channel's ``adm1_index`` up in :data:`STATE_BLOCKS`. Only blocks with at
    least one estimated channel appear.
    """
    out: dict[str, list[int]] = {}
    for pos, ch in enumerate(spec.channels):
        if ch.kind == "adm1":
            out.setdefault(STATE_BLOCKS[ch.adm1_index], []).append(pos)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup-days",
        type=float,
        default=30.0,
        help="Pre-simulate the plant for this many days so the filter starts "
        "from a settled operating point (default 30 d, 0 to skip).",
    )
    parser.add_argument(
        "--duration-days",
        type=float,
        default=5.0,
        help="Length of the twin / UKF run after warm-up (default 5 d).",
    )
    parser.add_argument(
        "--dt-hours",
        type=float,
        default=1.0,
        help="Filter step interval (default 1 h).",
    )
    parser.add_argument(
        "--warmup-dt-hours",
        type=float,
        default=24.0,
        help="Integration step for the warm-up phase (default 24 h).",
    )
    parser.add_argument(
        "--state-blocks",
        nargs="*",
        default=None,
        metavar="BLOCK",
        help="Estimate only these observability blocks (reduced state vector). "
        f"Choices: {sorted(BLOCK_INDICES)}. Example: "
        "--state-blocks methanogenesis charge_balance (the literature's "
        "provably-observable A+D-fused core). Default: full 41-state vector. "
        "Sensors are unchanged either way.",
    )
    parser.add_argument(
        "--process-noise-scale",
        type=float,
        default=1.0,
        help="Global multiplier on the ADM1 process-noise std (default 1.0). "
        "Raise it (e.g. 10) to make the filter trust the MODEL less and the "
        "SENSOR more — the fix when the model's gas prediction is worse than "
        "the measurement.",
    )
    parser.add_argument(
        "--gas-noise-std",
        type=float,
        default=None,
        help="Override the Q_gas measurement-noise std [m³/d] (default 10). "
        "Q_ch4 is set to half of it (mirrors the 10/5 default ratio). Lower "
        "values make the filter trust the gas sensor more.",
    )
    parser.add_argument(
        "--truth-kinetic-bias",
        type=float,
        default=1.0,
        help="Model-mismatch factor: multiply the TRUTH plant's kinetic "
        "(--truth-kinetic-key) by this, while the FILTER keeps the nominal "
        "value. 1.0 = perfect model (default). E.g. 0.6 makes the truth "
        "produce ~15-20%% less gas than the filter's model predicts, so the "
        "gas SENSOR carries information the MODEL gets wrong — the regime "
        "where trusting the sensor more actually helps.",
    )
    parser.add_argument(
        "--truth-kinetic-key",
        type=str,
        default="k_dis_PS",
        help="Which primary-digester kinetic to bias for the model mismatch "
        "(default k_dis_PS, the gas-limiting slow-disintegration rate).",
    )
    parser.add_argument(
        "--realistic",
        action="store_true",
        help="Apply the literature-grounded realism foundation "
        "(estimation.realism): model error = lognormal(0, "
        f"{MODEL_ERROR_KINETIC_SIGMA}) on the biological kinetics, per-sensor "
        "measurement noise from instrument specs (gas/NDIR/pH), and the "
        "real-plant daily gas analytics. Plant-agnostic, not fitted to any one "
        "dataset. Individual --model-error-std / --gas-noise-std still override.",
    )
    parser.add_argument(
        "--model-error-std",
        type=float,
        default=0.0,
        help="Simulate an imperfect FILTER model: multiply the filter's "
        "biological kinetics (k_dis/k_hyd/k_m/k_dec/K_S) by lognormal(0, STD) "
        "factors, while the TRUTH keeps the real values. The initial state is "
        "still calibrated, but the filter's one-step prediction is slightly "
        "off, so an un-corrected ADM1 state drifts away from the truth over "
        "time. Typical 0.1-0.2 (0 = perfect model).",
    )
    parser.add_argument(
        "--estimate-kinetic",
        nargs="*",
        default=None,
        metavar="KEY",
        help="Co-estimate these primary-digester kinetics as augmented "
        "states (the principled fix for a biased model). E.g. "
        "--estimate-kinetic k_dis_PS lets the filter LEARN the true "
        "disintegration rate from the gas sensor instead of fighting it.",
    )
    parser.add_argument(
        "--kinetic-noise-rel",
        type=float,
        default=0.2,
        help="Random-walk process-noise std of each co-estimated kinetic, "
        "relative to its nominal value (default 0.2). Higher = the gas "
        "innovation can move the kinetic faster (more gain), at the cost of "
        "a noisier estimate. Slow/weakly-observable kinetics (e.g. k_dis_PS) "
        "need a higher value and a longer horizon to converge.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-perturbation-relative",
        type=float,
        default=0.05,
        help="Relative Gaussian noise on the UKF prior (default 0.05; 0 = "
        "perfect initialisation).",
    )
    parser.add_argument(
        "--substrate-noise-relative",
        type=float,
        default=0.10,
        help="Relative per-step Gaussian noise on the truth substrate feed "
        "(default 0.10; models dosing/kg-to-m³ uncertainty, 0 = constant).",
    )
    parser.add_argument(
        "--daily-gas-analytics",
        action="store_true",
        help="Real-plant gas analytics: add a CO2 channel and sample Q_ch4 & "
        "Q_co2 once per day (NDIR), while Q_gas/pH/substrate doses stay hourly. "
        "(O2/H2S are also daily in practice but are not ADM1da states -> not "
        "assimilated.)",
    )
    parser.add_argument(
        "--irregular-sensors",
        action="store_true",
        help="Like --daily-gas-analytics, plus a VFA (FOS) channel sampled "
        "every 12 h. The full multi-rate sensor suite.",
    )
    parser.add_argument(
        "--feed-change",
        action="store_true",
        help="Step the truth substrate composition at days 3 and 6 (see "
        "_FEED_PHASES), so the produced gas/methane changes markedly mid-run. "
        "The filter must track the new operating point from its sensors. "
        "Pair with --duration-days 9.",
    )
    parser.add_argument(
        "--plot-from-day",
        type=float,
        default=0.0,
        help="Skip this many days at the start when plotting (burn-in). "
        "Diagnostics are still computed over the full run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Plot output directory (default {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


def build_truth_sensors(obs, seed: int) -> dict[str, SensorAdapter]:
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

    sensors: dict[str, SensorAdapter] = {}
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


def _apply_sensor_schedule(obs_noisy, obs, dt_hours, interval_h):
    """Multi-rate sampling: NaN out a channel at the steps where it is not
    sampled. ``interval_h`` maps a channel name to its sampling period [h];
    channels not listed (and any with period <= dt) stay sampled every step.
    ``run_filter`` skips non-finite measurements, so a gated channel simply
    contributes no innovation between its sample times."""
    out = obs_noisy.copy()
    n = len(out)
    for c in obs.channels:
        H = interval_h.get(c.name)
        if not H or H <= dt_hours or c.name not in out.columns:
            continue
        every = max(1, round(H / dt_hours))
        mask = (np.arange(n) % every) != 0
        out.loc[out.index[mask], c.name] = np.nan
    return out


def _mark_feed_changes(ax):
    """Draw a dotted vertical line at each substrate-composition change time
    (no-op when the feed is constant)."""
    for d in _FEED_CHANGE_DAYS:
        ax.axvline(d, color="0.35", ls=":", lw=1.0, alpha=0.7, zorder=0)


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


# At most this many subplots per figure; longer index lists are paginated
# into ``<stem>_1.png``, ``<stem>_2.png``, … (2 columns × 3 rows).
_MAX_PANELS = 6


def _page_paths(output_path, n_pages):
    """Output paths for a paginated plot: the bare path for a single page,
    else ``<stem>_<page><suffix>`` for each page."""
    if n_pages <= 1:
        return [output_path]
    return [
        output_path.with_name(f"{output_path.stem}_{p}{output_path.suffix}")
        for p in range(1, n_pages + 1)
    ]


def plot_trajectory_grid(time, truth, x_hat, std, spec, indices, title, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pages = [indices[i : i + _MAX_PANELS] for i in range(0, len(indices), _MAX_PANELS)]
    paths = _page_paths(output_path, len(pages))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for page, (chunk, out) in enumerate(zip(pages, paths), start=1):
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(
            rows, 2, figsize=(13, 2.5 * rows), sharex=True, squeeze=False
        )
        flat = [ax for row in axes for ax in row]
        for j, ax in enumerate(flat):
            if j >= len(chunk):
                ax.set_visible(False)
                continue
            idx = chunk[j]
            ch = spec.channels[idx]
            ax.plot(time, truth[:, idx], "k-", lw=1.5, label="truth")
            ax.plot(time, x_hat[:, idx], "C0-", lw=1.1, label=r"$\hat{x}$")
            ax.fill_between(
                time,
                x_hat[:, idx] - 2.0 * std[:, idx],
                x_hat[:, idx] + 2.0 * std[:, idx],
                color="C0",
                alpha=0.15,
                label=r"$\pm 2\sigma$",
            )
            ylim = _robust_ylim(
                truth[:, idx],
                x_hat[:, idx],
                x_hat[:, idx] - 2.0 * std[:, idx],
                x_hat[:, idx] + 2.0 * std[:, idx],
            )
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(f"{ch.name} (idx {idx})", fontsize=9)
            ax.grid(alpha=0.3)
            _mark_feed_changes(ax)
            if j == 0:
                ax.legend(loc="best", fontsize=7)
        flat[len(chunk) - 1].set_xlabel("time [d]")
        page_title = title if len(pages) == 1 else f"{title}  ({page}/{len(pages)})"
        fig.suptitle(page_title, fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)


def plot_observations(obs_clean, obs_noisy, steps, time, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = list(obs_clean.columns)
    pages = [
        channels[i : i + _MAX_PANELS] for i in range(0, len(channels), _MAX_PANELS)
    ]
    paths = _page_paths(output_path, len(pages))
    base_title = "Observations — truth, noisy measurement (sparse = gated), filter ŷ"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for page, (chunk, out) in enumerate(zip(pages, paths), start=1):
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(
            rows, 2, figsize=(13, 2.6 * rows), sharex=True, squeeze=False
        )
        flat = [ax for row in axes for ax in row]
        for j, ax in enumerate(flat):
            if j >= len(chunk):
                ax.set_visible(False)
                continue
            name = chunk[j]
            ax.plot(
                obs_clean.index,
                obs_clean[name].values,
                "k-",
                lw=1.5,
                label="clean truth",
            )
            ax.plot(
                obs_noisy.index,
                obs_noisy[name].values,
                "rx",
                markersize=6,
                label="noisy measurement",
                alpha=0.7,
            )
            y_pred = np.array([s.y_pred.get(name, np.nan) for s in steps])
            ax.plot(time, y_pred, "C0-", lw=1.1, label=r"$\hat{y}$")
            ylim = _robust_ylim(obs_clean[name].values, obs_noisy[name].values, y_pred)
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_ylabel(name)
            ax.grid(alpha=0.3)
            _mark_feed_changes(ax)
            if j == 0:
                ax.legend(loc="best", fontsize=8)
        flat[len(chunk) - 1].set_xlabel("time [d]")
        page_title = (
            base_title if len(pages) == 1 else f"{base_title}  ({page}/{len(pages)})"
        )
        fig.suptitle(page_title, fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=110, bbox_inches="tight")
        plt.close(fig)


def evaluate_h_at_xhat(
    warmed_up_plant, spec, obs, x_hat_history, channel_names, dt_hours: float = 1.0
):
    """Deterministic ``h(x̂_k)`` per step — the Jensen-bias-free production
    estimate (a single plant evaluation at the posterior mean, vs the UKF's
    sigma-point-averaged ŷ). Each call runs the plant for ``dt_hours`` so the
    gas phase equilibrates as it does in the truth propagation.

    The non-estimated states FREE-RUN (re-snapshot after every step, like
    :func:`evaluate_per_stage_gas`) instead of being pinned to the t=0
    baseline. This matters for a reduced state vector: the un-estimated
    upstream pools (hydrolysis, acidogenesis) that *feed* the gas must keep
    evolving, otherwise ``h(x̂)`` is computed against a frozen substrate
    supply and drifts away from the truth as the estimated states move. With
    the free run, ``h(x̂)`` matches the per-stage gas diagnostic.
    """
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
        # Accumulate: the non-estimated states carry forward to the next step
        # (they are not re-applied from x̂, so this lets them free-run).
        process.snapshot()
        for j, ex in enumerate(extractors):
            y_det[k, j] = float(ex(plant_copy, x_hat_history[k]))
    return y_det


def plot_production_estimate(
    obs_clean,
    obs_noisy,
    steps,
    time,
    x_hat_history,
    warmed_up_plant,
    spec,
    obs,
    output_path,
):
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
        y_smoothed = (
            sensor_series.rolling(
                window=window,
                center=True,
                min_periods=1,
            )
            .mean()
            .values
        )
        y_sigma = np.array([s.y_std.get(name, np.nan) for s in steps])

        ax_rate.plot(obs_clean.index, truth_series, "k-", lw=1.8, label="truth")
        ax_rate.plot(
            obs_noisy.index,
            noisy_series,
            "rx",
            markersize=4,
            label="sensor (raw)",
            alpha=0.4,
        )
        ax_rate.plot(
            obs_noisy.index,
            y_smoothed,
            "C3-",
            lw=1.4,
            label=f"sensor (smoothed, {window}-pt)",
        )
        ax_rate.plot(
            time, y_hat, "C2-", lw=1.2, label=r"$h(\hat{x})$ (model)", alpha=0.85
        )
        with_band = np.isfinite(y_hat) & np.isfinite(y_sigma)
        if with_band.any():
            ax_rate.fill_between(
                np.asarray(time)[with_band],
                (y_hat - y_sigma)[with_band],
                (y_hat + y_sigma)[with_band],
                color="C2",
                alpha=0.12,
                label=r"$h(\hat{x}) \pm 1\sigma$",
            )
        ylim = _robust_ylim(truth_series, noisy_series, y_hat, y_smoothed)
        if ylim is not None:
            ax_rate.set_ylim(*ylim)
        ax_rate.set_ylabel(f"{name} [m³/d]")
        ax_rate.set_title(f"{name} — instantaneous rate")
        ax_rate.legend(loc="best", fontsize=8)
        ax_rate.grid(alpha=0.3)
        _mark_feed_changes(ax_rate)

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
        ax_cum.plot(
            time, cum_hxhat, "C2-", lw=1.2, label=r"$h(\hat{x})$ (model)", alpha=0.85
        )
        if len(t_arr) and len(cum_hxhat):
            err_sensor = (
                (cum_sensor[-1] - cum_truth[-1]) / max(cum_truth[-1], 1e-12) * 100.0
            )
            err_model = (
                (cum_hxhat[-1] - cum_truth[-1]) / max(cum_truth[-1], 1e-12) * 100.0
            )
            ax_cum.text(
                0.55,
                0.18,
                f"sensor cum.: {err_sensor:+.1f} %",
                transform=ax_cum.transAxes,
                fontsize=9,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "edgecolor": "C3",
                    "alpha": 0.8,
                },
            )
            ax_cum.annotate(
                f"model cum.: {err_model:+.1f} %",
                xy=(time[-1], cum_hxhat[-1]),
                xytext=(0.55, 0.05),
                textcoords="axes fraction",
                fontsize=10,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "white",
                    "edgecolor": "C2",
                    "alpha": 0.8,
                },
            )
        ax_cum.set_xlabel("time [d]")
        ax_cum.set_ylabel(f"cumulative {name} [m³]")
        ax_cum.set_title(f"{name} — cumulative production")
        ax_cum.legend(loc="best", fontsize=9)
        ax_cum.grid(alpha=0.3)
        _mark_feed_changes(ax_cum)

    fig.suptitle(
        "Gas + methane production: truth vs deterministic UKF estimate", fontsize=13
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def evaluate_per_stage_gas(
    warmed_up_plant,
    spec,
    primary_state_history,
    stage_ids,
    dt_hours: float = 1.0,
    pre_step=None,
):
    """Replay a primary-state trajectory through a *freely-evolving* cascade,
    recording per-stage ``Q_gas`` / ``Q_ch4``.

    Unlike :func:`evaluate_h_at_xhat`, which restores to a single baseline
    every step (pinning the downstream stages), this re-snapshots after each
    step so the Nachgärer and Storage ACCUMULATE their trajectory. Only the
    primary's ADM1 state and substrate dose are overwritten from
    ``primary_state_history`` each step; the downstream stages free-run,
    driven by the primary's effluent through the liquid cascade.

    Called twice — once with the truth primary states (the "real" downstream
    gas) and once with the UKF posterior x̂ (our corrected model's downstream
    gas). Both replays share the same warmed-up baseline, dt and substrate
    routing, so the per-stage gap is purely the footprint of the primary
    estimation error propagating down the cascade.

    Returns ``(q_gas, q_ch4, primary_adm1)``: the first two are
    ``{stage_id: (T,) array}`` dicts of per-stage gas; ``primary_adm1`` is a
    ``(T, 41)`` array of the primary digester's full ADM1 state per step
    (estimated indices follow ``primary_state_history``, the rest free-run) —
    used to compare the non-estimated states against the real plant.
    """
    plant_copy = copy.deepcopy(warmed_up_plant)
    process = ADM1ProcessModel(plant_copy, spec)
    process.snapshot()
    dt = dt_hours / 24.0
    T = len(primary_state_history)
    q_gas = {sid: np.full(T, np.nan) for sid in stage_ids}
    q_ch4 = {sid: np.full(T, np.nan) for sid in stage_ids}
    primary_adm1 = np.full((T, 41), np.nan)
    for k in range(T):
        if pre_step is not None:
            pre_step(plant_copy, k)  # e.g. re-apply a time-varying model error
        process.step(primary_state_history[k], dt)
        process.snapshot()  # accumulate: the next step continues from here
        for sid in stage_ids:
            od = plant_copy.components[sid].outputs_data
            q_gas[sid][k] = float(od.get("Q_gas", float("nan")))
            q_ch4[sid][k] = float(od.get("Q_ch4", float("nan")))
        adm1 = np.asarray(plant_copy.components[DIGESTER_ID].adm1_state, dtype=float)
        primary_adm1[k, : min(41, adm1.size)] = adm1[:41]
    return q_gas, q_ch4, primary_adm1


def plot_per_stage_gas(
    time, truth_gas, model_gas, truth_ch4, model_ch4, stage_ids, output_path
):
    """One row per cascade stage: truth vs. corrected-model gas + methane.

    The primary row is the Fermenter gas the UKF estimates; the secondary /
    storage rows show how the corrected primary drives the open-loop
    downstream stages relative to the real plant.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(stage_ids)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.0 * n), sharex=True, squeeze=False)
    series_specs = [(truth_gas, model_gas, "Q_gas"), (truth_ch4, model_ch4, "Q_ch4")]
    for row, sid in enumerate(stage_ids):
        model_label = r"UKF $h(\hat{x})$" if sid == "primary" else "corrected model"
        for col, (q_truth, q_model, label) in enumerate(series_specs):
            ax = axes[row][col]
            t_series, m_series = q_truth[sid], q_model[sid]
            ax.plot(time, t_series, "k-", lw=1.6, label="truth")
            ax.plot(time, m_series, "C0-", lw=1.2, label=model_label)
            ylim = _robust_ylim(t_series, m_series)
            if ylim is not None:
                ax.set_ylim(*ylim)
            cum_t, cum_m = np.nansum(t_series), np.nansum(m_series)
            rel = (cum_m - cum_t) / max(abs(cum_t), 1e-12) * 100.0
            ax.text(
                0.02,
                0.88,
                f"cum. err: {rel:+.1f} %",
                transform=ax.transAxes,
                fontsize=8,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.7},
            )
            ax.set_ylabel(f"{label} [m³/d]")
            ax.grid(alpha=0.3)
            _mark_feed_changes(ax)
            if row == 0:
                ax.legend(loc="best", fontsize=8)
        axes[row][0].set_title(STAGE_LABELS.get(sid, sid), fontsize=9, loc="left")
    axes[-1][0].set_xlabel("time [d]")
    axes[-1][1].set_xlabel("time [d]")
    fig.suptitle(
        "Per-stage gas: truth vs. corrected model " "(Fermenter / Nachgärer / Storage)",
        fontsize=13,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_non_estimated_states(
    time, truth_full, filter_full, indices, name_of, output_path
):
    """Per non-estimated ADM1 state: the real plant (truth) vs. the filter's
    OPEN-LOOP propagation.

    These states are NOT in the reduced state vector, so the UKF never
    corrects them — the filter just carries them along its model dynamics,
    driven by the (corrected) estimated states. The gap to the truth shows
    how far an un-estimated state drifts over the run. No uncertainty band:
    the filter has no covariance for them.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pages = [indices[i : i + _MAX_PANELS] for i in range(0, len(indices), _MAX_PANELS)]
    paths = _page_paths(output_path, len(pages))
    base_title = "Non-estimated ADM1 states: real plant vs. filter open-loop"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for page, (chunk, out) in enumerate(zip(pages, paths), start=1):
        rows = (len(chunk) + 1) // 2
        fig, axes = plt.subplots(
            rows, 2, figsize=(13, 2.4 * rows), sharex=True, squeeze=False
        )
        flat = [ax for row in axes for ax in row]
        for j, ax in enumerate(flat):
            if j >= len(chunk):
                ax.set_visible(False)
                continue
            idx = chunk[j]
            ax.plot(time, truth_full[:, idx], "k-", lw=1.5, label="truth (real plant)")
            ax.plot(
                time, filter_full[:, idx], "C1-", lw=1.1, label="filter (open-loop)"
            )
            ylim = _robust_ylim(truth_full[:, idx], filter_full[:, idx])
            if ylim is not None:
                ax.set_ylim(*ylim)
            ax.set_title(f"{name_of(idx)} (idx {idx})", fontsize=9)
            ax.grid(alpha=0.3)
            _mark_feed_changes(ax)
            if j == 0:
                ax.legend(loc="best", fontsize=7)
        flat[len(chunk) - 1].set_xlabel("time [d]")
        page_title = (
            base_title if len(pages) == 1 else f"{base_title}  ({page}/{len(pages)})"
        )
        fig.suptitle(page_title, fontsize=13)
        fig.tight_layout()
        fig.savefig(out, dpi=110, bbox_inches="tight")
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
        ax.axhline(
            float(n_active),
            color="k",
            ls="--",
            alpha=0.6,
            label=f"expected NIS = {n_active}",
        )
    ax.set_xlabel("time [d]")
    ax.set_ylabel("NIS")
    ax.set_title("Normalized Innovation Squared")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    _mark_feed_changes(ax)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_summary(coverage, block_positions, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Canonical display order, restricted to blocks present in this spec.
    ordered = [b for b in QUALITY_BLOCKS if b in block_positions]
    block_means = {
        block: float(np.mean([coverage[i] for i in block_positions[block]]))
        for block in ordered
    }
    fig, ax = plt.subplots(figsize=(11, 5))
    names = list(block_means.keys())
    values = [block_means[n] * 100.0 for n in names]
    bars = ax.bar(names, values, color="C0", alpha=0.7)
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 1,
            f"{v:.0f} %",
            ha="center",
            fontsize=9,
        )
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
    print(
        f"  Warming up {label} plant: {warmup_days:.0f} d at dt = {warmup_dt_hours:.0f} h ..."
    )
    plant.simulate(
        duration=float(warmup_days),
        dt=float(warmup_dt_hours) / 24.0,
        save_interval=float(warmup_days),
    )


def _propagate_truth_with_substrate_noise(
    spec,
    truth_process,
    obs,
    x0,
    dt_hours,
    n_steps,
    rng,
    substrate_noise_relative,
    nominal_schedule=None,
    equilibration_dt=1.0 / 24.0,
):
    """Propagate the truth trajectory with per-step Gaussian noise on the
    augmented substrate-input channels (models the operator's dosing
    uncertainty). Returns ``(time, states, obs_clean)`` like
    :func:`twin.propagate_truth`; the augmented channels in ``states`` hold
    the actually delivered (noisy) substrate values.

    ``nominal_schedule`` (``(n_steps+1, n_inputs)``) gives the nominal feed
    per step for a time-varying composition; ``None`` holds the constant
    ``x0`` doses. The Gaussian noise is relative to the per-step nominal, and
    the standard-normal draw count is unchanged, so a constant schedule
    reproduces the legacy result bit-for-bit."""
    dt = dt_hours / 24.0
    n_state = len(spec)
    aug_indices = spec.kind_indices("input_flow")
    base_nominal = np.array([x0[i] for i in aug_indices], dtype=float)
    if nominal_schedule is None:
        nominal_schedule = np.tile(base_nominal, (n_steps + 1, 1))
    else:
        nominal_schedule = np.asarray(nominal_schedule, dtype=float)

    # Truth substrate trajectory: substrate_truth[k] is fed during step k→k+1.
    if substrate_noise_relative > 0.0:
        noise = rng.normal(
            0.0,
            substrate_noise_relative * np.abs(nominal_schedule),
            size=(n_steps + 1, len(aug_indices)),
        )
    else:
        noise = np.zeros((n_steps + 1, len(aug_indices)))
    substrate_truth = np.clip(nominal_schedule + noise, a_min=0.0, a_max=None)

    states = np.zeros((n_steps + 1, n_state))
    states[0] = x0.copy()
    for k_aug, i_aug in enumerate(aug_indices):
        states[0, i_aug] = substrate_truth[0, k_aug]

    truth_process.refresh_outputs(states[0], equilibration_dt=equilibration_dt)
    obs_rows = [
        [float(c.extractor(truth_process.plant, states[0])) for c in obs.channels]
    ]

    x = states[0].copy()
    for k in range(n_steps):
        for k_aug, i_aug in enumerate(aug_indices):
            x[i_aug] = substrate_truth[k, k_aug]
        x = truth_process.step(x, dt)
        states[k + 1] = x
        truth_process.refresh_outputs(x, equilibration_dt=equilibration_dt)
        obs_rows.append(
            [float(c.extractor(truth_process.plant, x)) for c in obs.channels]
        )

    time_arr = np.arange(n_steps + 1, dtype=float) * dt
    obs_clean = pd.DataFrame(
        obs_rows, index=time_arr, columns=[c.name for c in obs.channels]
    )
    return time_arr, states, obs_clean


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_steps = round(args.duration_days * 24.0 / args.dt_hours)

    # --realistic preset: literature-grounded model error + real-plant daily
    # gas analytics (individual flags still override).
    if args.realistic and args.model_error_std == 0.0:
        args.model_error_std = MODEL_ERROR_KINETIC_SIGMA

    # Sensor set + multi-rate sampling schedule (channel name -> period [h]).
    sensors = list(SENSORS)  # q_gas, q_ch4, ph, substrate_dose (hourly base)
    schedule: dict[str, float] = {}
    if args.daily_gas_analytics or args.irregular_sensors or args.realistic:
        if "q_co2" not in sensors:
            sensors.insert(sensors.index("q_ch4") + 1, "q_co2")
        schedule.update(_DAILY_GAS_CHANNELS)  # Q_ch4, Q_co2 once per day
    if args.irregular_sensors:
        sensors.append("vfa")
        schedule.update(_VFA_INTERVAL_H)  # VFA every 12 h

    print("Twin experiment — multi-stage example, 41-state SR-UKF")
    print(f"  Warm-up: {args.warmup_days:.0f} d (no filter)")
    print(
        f"  Filter horizon: {args.duration_days:.1f} d  |  "
        f"dt = {args.dt_hours:.1f} h  |  {n_steps} steps"
    )

    # ---- Truth + filter plants -----------------------------------------
    # Default (bias = 1.0): the filter plant is a deepcopy of the warmed-up
    # truth plant, so the UKF model is bit-identical to the truth at t=0
    # (perfect model — the filter SHOULD trust it).
    #
    # Model mismatch (bias != 1.0): the TRUTH gets a perturbed kinetic
    # (--truth-kinetic-key) before warm-up, so it settles to a different
    # operating point; the FILTER is reset to the NOMINAL kinetic. Both
    # still start from the truth's biological STATE at t=0 (the filter is a
    # deepcopy with only its kinetic restored), so the divergence over the
    # run is purely the model error. This is the regime the user cares about:
    # the gas sensor (truth) is "better" than the filter's (nominal) model.
    print("\nBuilding + warming up truth plant ...")
    truth_plant = build_multi_stage_plant()

    kinetic_bias = abs(args.truth_kinetic_bias - 1.0) > 1e-12
    if kinetic_bias:
        key = args.truth_kinetic_key
        prim_kin = truth_plant.components[DIGESTER_ID].adm1._kinetic
        if key not in prim_kin:
            raise SystemExit(
                f"--truth-kinetic-key '{key}' not in the primary digester's "
                f"kinetics. Available: {sorted(prim_kin)}."
            )
        nominal_kin = float(prim_kin[key])
        prim_kin[key] = nominal_kin * args.truth_kinetic_bias
        print(
            f"  MODEL MISMATCH: truth {key} = {nominal_kin:.4g} x "
            f"{args.truth_kinetic_bias} = {prim_kin[key]:.4g}; "
            f"filter keeps nominal {nominal_kin:.4g}."
        )

    _warmup(truth_plant, args.warmup_days, args.warmup_dt_hours, "truth")
    filter_plant = copy.deepcopy(truth_plant)  # truth biological state at t=0
    if kinetic_bias:
        # Restore the nominal kinetic in the filter's model only.
        filter_plant.components[DIGESTER_ID].adm1._kinetic[
            args.truth_kinetic_key
        ] = nominal_kin

    # Imperfect filter model: perturb the FILTER's biological kinetics so its
    # one-step prediction is slightly wrong (the truth keeps the real values).
    # The state stays calibrated at t=0; un-corrected ADM1 states then drift.
    if args.model_error_std > 0.0:
        erng = np.random.default_rng(args.seed + 1009)
        fkin = filter_plant.components[DIGESTER_ID].adm1._kinetic
        factors = []
        for key, val in list(fkin.items()):
            if isinstance(val, (int, float)) and key.startswith(_MODEL_ERROR_PREFIXES):
                f = float(np.exp(erng.normal(0.0, args.model_error_std)))
                fkin[key] = float(val) * f
                factors.append(f)
        print(
            f"  MODEL ERROR: perturbed {len(factors)} filter kinetics "
            f"(k_dis/k_hyd/k_m/k_dec/K_S) by lognormal(0, {args.model_error_std}); "
            f"mean |factor-1| = {np.mean([abs(f - 1) for f in factors]):.1%}. "
            "Truth keeps the real kinetics; the filter's prediction drifts where "
            "the UKF does not correct."
        )

    # Per-stage gas bases: truth side keeps the (possibly biased) kinetic;
    # the model side uses the filter's nominal model. With bias = 1.0 both
    # are identical to the legacy single-plant behaviour.
    truth_base = copy.deepcopy(truth_plant)
    det_h_plant = copy.deepcopy(filter_plant)

    # ---- Reduced state vector (optional) -------------------------------
    # Resolve the requested observability blocks into ADM1 indices. None
    # keeps the full 41-state vector; the omitted ADM1 states then propagate
    # open-loop with the plant model (the literature's treatment for
    # structurally non-observable states — see docs/observability).
    adm1_indices = None
    if args.state_blocks:
        unknown = [b for b in args.state_blocks if b not in BLOCK_INDICES]
        if unknown:
            raise SystemExit(
                f"Unknown --state-blocks {unknown}. Choices: {sorted(BLOCK_INDICES)}."
            )
        adm1_indices = sorted({i for b in args.state_blocks for i in BLOCK_INDICES[b]})
        print(
            f"\nReduced state vector: blocks {args.state_blocks} -> "
            f"{len(adm1_indices)} of 41 ADM1 states estimated "
            f"(rest open-loop). Sensors unchanged."
        )

    # ---- Sensor-trust tuning -------------------------------------------
    # Lower gas R and/or higher process-noise scale both raise the Kalman
    # gain on the gas innovation: the filter follows the (good) sensor
    # rather than the (worse) model prediction.
    sensor_noise = None
    if args.realistic:
        # Literature-grounded per-sensor noise, scaled by the warmed plant's
        # own gas magnitudes (plant-agnostic relative specs, not fitted data).
        od = truth_plant.components[DIGESTER_ID].outputs_data
        sensor_noise = build_sensor_noise(
            {
                "q_gas": float(od.get("Q_gas", 1900.0)),
                "q_ch4": float(od.get("Q_ch4", 900.0)),
                "q_co2": float(od.get("Q_co2", 850.0)),
            }
        )
        shown = {
            k: sensor_noise[k]
            for k in ("q_gas", "q_ch4", "q_co2", "ph")
            if k in sensor_noise
        }
        print(
            f"\nRealistic conditions: model error lognormal(0, "
            f"{args.model_error_std:g}); measurement noise "
            + ", ".join(f"{k}={v:.3g}" for k, v in shown.items())
            + " (literature, plant-agnostic)."
        )
    if args.gas_noise_std is not None:
        sensor_noise = dict(sensor_noise or {})
        sensor_noise["q_gas"] = float(args.gas_noise_std)
        sensor_noise["q_ch4"] = float(args.gas_noise_std) / 2.0
        print(
            f"\nSensor-trust: Q_gas noise std = {args.gas_noise_std:.1f}, "
            f"Q_ch4 = {args.gas_noise_std / 2.0:.1f} m³/d (lower = trust sensor more)."
        )
    if args.process_noise_scale != 1.0:
        print(
            f"Sensor-trust: process-noise scale = {args.process_noise_scale:.1f}x "
            "(higher = trust model less)."
        )

    # ---- Kinetic co-estimation (the principled fix for a biased model) -
    # Co-estimating the biasing kinetic lets the filter LEARN the true
    # parameter from the gas innovation, instead of fighting a structural
    # error it can never resolve by adjusting states alone. Prior = nominal
    # (the filter does not know the truth); a wide initial std + random-walk
    # process noise let the gas sensor pull it to the true value.
    kinetic_specs = []
    if args.estimate_kinetic:
        fkin = filter_plant.components[DIGESTER_ID].adm1._kinetic
        for key in args.estimate_kinetic:
            if key not in fkin:
                raise SystemExit(
                    f"--estimate-kinetic '{key}' not a primary kinetic. "
                    f"Available: {sorted(fkin)}."
                )
            nom = abs(float(fkin[key]))
            kinetic_specs.append(
                KineticSpec(
                    name=key,
                    initial=float(fkin[key]),
                    initial_std=max(nom, 1e-6),
                    process_noise_std=args.kinetic_noise_rel * max(nom, 1e-6),
                    lower=0.0,
                    upper=5.0 * nom + 1e-9,
                    drift_model="random_walk",
                )
            )
        print(
            f"\nCo-estimating kinetics {args.estimate_kinetic} as augmented "
            "states (prior = nominal, learned from the gas sensor)."
        )

    # ---- UKF: the whole setup in one call ------------------------------
    ukf = build_ukf(
        filter_plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=sensors,
        kinetic_overrides=kinetic_specs,
        initial_uncertainty_relative=max(args.initial_perturbation_relative, 1e-3),
        adm1_indices=adm1_indices,
        process_noise_scale=args.process_noise_scale,
        sensor_noise=sensor_noise,
    )
    spec, obs = ukf.spec, ukf.obs
    x_filter0 = ukf.x_hat.copy()  # filter prior (perturbed below)
    n_adm1 = len(spec.kind_indices("adm1"))
    n_sigma = 2 * len(spec) + 1
    print(
        f"  State vector: {len(spec)} channels "
        f"({n_adm1} ADM1 + {len(spec) - n_adm1} augmented)  ->  {n_sigma} sigma points"
    )
    print(f"  Observation channels: {[c.name for c in obs.channels]}")

    # ---- Propagate truth (ALWAYS the full 41-state plant) --------------
    # The truth is generated independently of the filter's (possibly reduced)
    # state vector: a full 41-state free-running plant. So the real-plant
    # simulation is IDENTICAL whether the filter estimates 18 or 41 states —
    # only what the filter does with the same measurements changes. The truth
    # keeps the plant's actual kinetics (biased under --truth-kinetic-bias);
    # the filter uses its own model. Same substrate-noise draw order as before
    # (5 input channels either way), so results stay deterministic.
    print(
        f"\nPropagating truth (full 41-state plant) for {args.duration_days:.1f} d "
        f"(substrate noise: {args.substrate_noise_relative * 100:.0f} % per step) ..."
    )
    truth_process, truth_obs, truth_spec = build_filter_components(
        truth_plant, digester_id=DIGESTER_ID, substrates=SUBSTRATES, sensors=sensors
    )
    x_truth0 = truth_spec.read_adm1_state(truth_plant)
    for i, ch in enumerate(truth_spec.channels):
        if ch.kind != "adm1":
            x_truth0[i] = ch.initial

    # Optional time-varying feed: build a per-step nominal-flow schedule that
    # steps the composition at the --feed-change boundaries.
    nominal_schedule = None
    if args.feed_change:
        _FEED_CHANGE_DAYS.clear()
        _FEED_CHANGE_DAYS.extend(d for d, _ in _FEED_PHASES[1:])
        aug_chs = [
            truth_spec.channels[i] for i in truth_spec.kind_indices("input_flow")
        ]
        nominal_schedule = np.zeros((n_steps + 1, len(aug_chs)))
        dt = args.dt_hours / 24.0
        for k in range(n_steps + 1):
            t = k * dt
            flows = _FEED_PHASES[0][1]
            for start_day, f in _FEED_PHASES:
                if t >= start_day:
                    flows = f
            for c_pos, ch in enumerate(aug_chs):
                nominal_schedule[k, c_pos] = flows[ch.input_substrate_index]
        print(
            f"\nTime-varying feed (--feed-change): composition steps at days "
            f"{_FEED_CHANGE_DAYS}; phase totals "
            f"{[round(sum(f), 1) for _, f in _FEED_PHASES]} m³/d. The filter "
            "must track each new operating point from its sensors."
        )

    time, truth_full_traj, obs_clean = _propagate_truth_with_substrate_noise(
        truth_spec,
        truth_process,
        truth_obs,
        x_truth0,
        dt_hours=args.dt_hours,
        n_steps=n_steps,
        rng=rng,
        substrate_noise_relative=args.substrate_noise_relative,
        nominal_schedule=nominal_schedule,
    )

    # Map the full truth onto the filter's state-vector layout for the
    # estimated-state comparison (identity mapping when the filter is full-41).
    adm1_pos_full = {
        ch.adm1_index: i
        for i, ch in enumerate(truth_spec.channels)
        if ch.kind == "adm1"
    }
    input_pos_full = {
        ch.input_substrate_index: i
        for i, ch in enumerate(truth_spec.channels)
        if ch.kind == "input_flow"
    }
    truth = np.zeros((n_steps + 1, len(spec)))
    for p, ch in enumerate(spec.channels):
        if ch.kind == "adm1":
            truth[:, p] = truth_full_traj[:, adm1_pos_full[ch.adm1_index]]
        elif ch.kind == "input_flow":
            truth[:, p] = truth_full_traj[:, input_pos_full[ch.input_substrate_index]]
        elif ch.kind == "kinetic_param":
            truth[:, p] = float(
                truth_plant.components[DIGESTER_ID].adm1._kinetic.get(
                    ch.name, ch.initial
                )
            )

    print("Stepping truth-side sensors (drift + lag + noise) ...")
    obs_noisy = measure_truth_with_sensors(
        obs_clean, build_truth_sensors(truth_obs, args.seed)
    )
    if schedule:
        obs_noisy = _apply_sensor_schedule(
            obs_noisy, truth_obs, args.dt_hours, schedule
        )
        extra = ", VFA every 12 h" if args.irregular_sensors else ""
        print(
            f"  Sensor schedule: Q_ch4 & Q_co2 once per day{extra}; "
            "Q_gas/pH/doses hourly. O2/H2S are daily in practice but are not "
            "ADM1da states -> monitoring only, not assimilated by the UKF."
        )

    # ---- Perturb the prior (build_ukf starts at the exact truth) -------
    # Perturb the FILTER's prior (nominal kinetic), not the truth vector.
    if args.initial_perturbation_relative > 0.0:
        pert = rng.normal(
            0.0,
            args.initial_perturbation_relative * (np.abs(x_filter0) + 1e-6),
            size=len(spec),
        )
        ukf.reset(spec.clip(x_filter0 + pert), ukf.P)
        print(
            f"\nInitial perturbation: ±{args.initial_perturbation_relative * 100:.0f} % "
            "Gaussian noise on the prior."
        )
    else:
        print("\nInitial perturbation: 0 (perfect initialisation).")

    print(f"Running UKF for {n_steps + 1} steps ...")
    x_hat, std, steps = run_filter(
        ukf,
        spec,
        obs,
        obs_noisy,
        gate_frame=None,
        dt_hours=args.dt_hours,
    )

    # ---- Diagnostics ---------------------------------------------------
    coverage = coverage_within_2sigma(truth, x_hat, std)
    nis = np.array([s.nis for s in steps], dtype=float)
    nis_finite = nis[np.isfinite(nis)]
    nis_mean = float(np.mean(nis_finite)) if len(nis_finite) else float("nan")
    n_active = len(steps[0].active_channels) if steps else 0

    # Block → state-vector positions, robust to a reduced spec (only blocks
    # with an estimated channel appear). Canonical display order.
    block_positions = spec_block_positions(spec)
    print("\n--- Per-quality-block coverage (estimated states only) ---")
    for block in [b for b in QUALITY_BLOCKS if b in block_positions]:
        positions = block_positions[block]
        block_cov = float(np.mean([coverage[i] for i in positions]))
        print(
            f"  {block:30s}  n={len(positions):2d}  "
            f"2sigma-cov = {100 * block_cov:5.1f} %"
        )
    print(
        f"\n  Mean NIS = {nis_mean:.2f}  "
        f"(target {0.5 * n_active:.1f} – {2.0 * n_active:.1f} for {n_active} channels)"
    )

    # ---- Co-estimated kinetics: did the filter learn them? -------------
    if kinetic_specs:
        print("\n--- Co-estimated kinetics (learned from the gas sensor) ---")
        for pos in spec.kind_indices("kinetic_param"):
            name = spec.channels[pos].name
            prior = float(spec.channels[pos].initial)
            final = float(x_hat[-1, pos])
            if kinetic_bias and name == args.truth_kinetic_key:
                truth_val = nominal_kin * args.truth_kinetic_bias
                err = (final - truth_val) / max(abs(truth_val), 1e-12) * 100.0
                tag = f"truth={truth_val:.4g}  (err {err:+.1f}%)"
            else:
                tag = "no truth bias"
            print(f"  {name:10s} prior={prior:.4g} -> final={final:.4g}   {tag}")

    # ---- Per-stage gas: truth vs. corrected model ----------------------
    # Both cascades free-run (accumulating) from the same warmed-up baseline;
    # the only difference is the primary state fed in (truth vs. x̂), so the
    # per-stage gap isolates the downstream footprint of the Fermenter
    # estimation error. det_h_plant is the clean warmed-up plant (deepcopied
    # before truth propagation mutated truth_plant).
    stage_ids = [s for s in STAGE_IDS if s in truth_plant.components]
    print("\nComputing per-stage gas (truth vs. corrected model) ...")
    # Truth cascade uses the (possibly biased) truth kinetics; the corrected
    # model uses the filter's nominal model fed the posterior x̂. With a model
    # mismatch the gap on the primary panel is the model error the gas sensor
    # has to overcome.
    # Truth cascade: full 41-state truth through the full spec (so the
    # non-estimated states it produces are the real plant's free run).
    truth_gas, truth_ch4, truth_full = evaluate_per_stage_gas(
        truth_base, truth_spec, truth_full_traj, stage_ids, args.dt_hours
    )
    model_gas, model_ch4, filter_full = evaluate_per_stage_gas(
        det_h_plant, spec, x_hat, stage_ids, args.dt_hours
    )
    print("\n--- Per-stage gas: corrected model vs. truth ---")
    for sid in stage_ids:
        tg, mg = truth_gas[sid], model_gas[sid]
        rmse = float(np.sqrt(np.nanmean((mg - tg) ** 2)))
        mean_truth = float(np.nanmean(tg))
        nrmse = rmse / max(abs(mean_truth), 1e-12) * 100.0
        cum_err = (
            (np.nansum(mg) - np.nansum(tg)) / max(abs(np.nansum(tg)), 1e-12) * 100.0
        )
        role = "estimated" if sid == "primary" else "open-loop"
        print(
            f"  {sid:10s} ({role:9s})  Q_gas mean={mean_truth:8.1f}  "
            f"RMSE={rmse:6.2f}  NRMSE={nrmse:5.1f}%  cum.err={cum_err:+5.1f}%"
        )

    # ---- Plot slice (visual burn-in; diagnostics use the full run) -----
    plot_mask = time >= args.plot_from_day
    if args.plot_from_day > 0:
        print(
            f"\n  Plot burn-in: skipping t < {args.plot_from_day:.1f} d "
            f"({int(plot_mask.sum())} of {len(time)} samples kept)."
        )
    time_p, truth_p, x_hat_p, std_p = (
        time[plot_mask],
        truth[plot_mask],
        x_hat[plot_mask],
        std[plot_mask],
    )
    obs_clean_p = obs_clean.loc[obs_clean.index >= args.plot_from_day]
    obs_noisy_p = obs_noisy.loc[obs_noisy.index >= args.plot_from_day]
    steps_p = [s for s in steps if s.t >= args.plot_from_day]

    # ---- Plots ---------------------------------------------------------
    # Note: the full set of estimated states (with their ±2σ band) is the
    # `states_all_estimated.png` grid below; there is no separate
    # "strong/weak" subset plot — on a reduced spec those subsets collapse
    # onto the same channels and just duplicate panels.
    output_dir = args.output_dir
    print(f"\nWriting plots to {output_dir} ...")
    plot_observations(
        obs_clean_p,
        obs_noisy_p,
        steps_p,
        time_p,
        output_path=output_dir / "observations.png",
    )
    plot_production_estimate(
        obs_clean_p,
        obs_noisy_p,
        steps_p,
        time_p,
        x_hat_history=x_hat_p,
        warmed_up_plant=det_h_plant,
        spec=spec,
        obs=obs,
        output_path=output_dir / "production_estimate.png",
    )
    plot_nis(steps_p, time_p, output_path=output_dir / "nis.png")
    plot_coverage_summary(
        coverage, block_positions, output_path=output_dir / "coverage_summary.png"
    )
    truth_gas_p = {s: truth_gas[s][plot_mask] for s in stage_ids}
    model_gas_p = {s: model_gas[s][plot_mask] for s in stage_ids}
    truth_ch4_p = {s: truth_ch4[s][plot_mask] for s in stage_ids}
    model_ch4_p = {s: model_ch4[s][plot_mask] for s in stage_ids}
    plot_per_stage_gas(
        time_p,
        truth_gas_p,
        model_gas_p,
        truth_ch4_p,
        model_ch4_p,
        stage_ids,
        output_path=output_dir / "per_stage_gas.png",
    )

    # ---- All estimated states with their ±2σ uncertainty band ----------
    plot_trajectory_grid(
        time_p,
        truth_p,
        x_hat_p,
        std_p,
        spec,
        indices=list(range(len(spec))),
        title=f"All {len(spec)} estimated states (truth vs. x̂ ± 2σ)",
        output_path=output_dir / "states_all_estimated.png",
    )

    # ---- Non-estimated ADM1 states: real plant vs. filter open-loop ----
    # These ADM1 indices are NOT in the reduced state vector, so the UKF
    # never corrects them; they just ride along the filter's model dynamics.
    estimated_adm1 = {ch.adm1_index for ch in spec.channels if ch.kind == "adm1"}
    non_est_indices = [i for i in range(41) if i not in estimated_adm1]
    if non_est_indices:
        adm1_names = adm1da_full_spec(DIGESTER_ID).channels
        plot_non_estimated_states(
            time_p,
            truth_full[plot_mask],
            filter_full[plot_mask],
            non_est_indices,
            name_of=lambda idx: adm1_names[idx].name,
            output_path=output_dir / "states_non_estimated.png",
        )
        print(
            f"  Non-estimated states: {len(non_est_indices)} ADM1 indices "
            "plotted (real plant vs. filter open-loop)."
        )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
