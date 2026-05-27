"""Twin experiment for the 41-state Square-Root UKF.

Sets up two independent copies of the multi-stage example plant
(see :mod:`pyadm1ode_estimation.example_plants.multi_stage`):

* **Truth plant** — propagated forward with a known initial state
  and the plant's nominal substrate feed. The full 41-state ADM1
  trajectory plus the three substrate-input rates form the ground
  truth that the filter has to recover.
* **Filter plant** — starts from a perturbed initial state estimate
  and only sees noisy observations of the Phase-1 sensor set
  (``Q_gas`` + ``Q_ch4`` + ``pH`` + ``FOS/TAC``).

The script writes diagnostic plots to ``output/twin_experiment/``:

* ``trajectories_strong.png`` — six representative strong-observable
  states (subset of subsystems A and D),
* ``trajectories_weak.png`` — six representative weak / open-loop
  states (subsystems B / C / E + the augmented substrate inputs),
* ``observations.png`` — clean truth, noisy measurement, filter
  prediction per sensor channel,
* ``nis.png`` — NIS time series with the expected baseline,
* ``coverage_summary.png`` — per-channel ``2σ`` coverage grouped by
  quality class.

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyadm1ode_estimation.estimation import (  # noqa: E402
    ADM1ProcessModel,
    InputSpec,
    ObservationChannel,
    ObservationModel,
    adm1da_full_spec,
)
from pyadm1ode_estimation.estimation.filters import (  # noqa: E402
    UnscentedKalmanFilter,
)
from pyadm1ode_estimation.estimation.observation_model import (  # noqa: E402
    extract_q_ch4_total,
    extract_q_gas_total,
    make_state_extractor,
)
from pyadm1ode_estimation.estimation.sensors import (  # noqa: E402
    SensorAdapter,
    measure_truth_with_sensors,
)
from pyadm1ode_estimation.estimation.twin import (  # noqa: E402
    coverage_within_2sigma,
    propagate_truth,
    run_filter,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant  # noqa: E402


DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "twin_experiment"
DIGESTER_ID = "primary"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--warmup-days",
        type=float,
        default=30.0,
        help="Pre-simulate the plant for this many days before the twin "
        "starts, so the filter sees a settled operating point and "
        "not the initial-transient peak (default 30 d). Set to 0 "
        "to start the filter immediately after build.",
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
        help="Integration step for the warm-up phase (default 24 h). "
        "Coarse is OK — only the steady-state matters.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-perturbation-relative",
        type=float,
        default=0.05,
        help="Initial UKF prior perturbation relative to truth (default "
        "0.05 = 5 %% Gaussian noise on every channel). Set to 0 for "
        "perfect initialisation — useful to verify that the UKF's "
        "first predict step reproduces the truth's first step.",
    )
    parser.add_argument(
        "--plot-from-day",
        type=float,
        default=0.0,
        help="Skip this many days from the start of the trajectory when "
        "plotting (burn-in for filter convergence). Diagnostics "
        "(coverage, NIS) are still computed over the full run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Plot output directory (default {DEFAULT_OUTPUT_DIR}).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sensor set
# ---------------------------------------------------------------------------
#
# Five channels:
#   - Q_gas, Q_ch4  : gas-side, computed from outputs_data
#   - pH            : computed inside pyadm1, NaN-safe wrapper
#   - 3 substrate feed sensors (one per substrate slot) : direct observation
#     of the augmented input-flow channels in the state vector. Closes the
#     identifiability gap exposed by the previous 4-sensor run (filter could
#     not distinguish "doubled substrate + doubled biomass" from "nominal
#     substrate + nominal biomass" — both produce the same gas / pH / FOSTAC).
#
# FOS/TAC has been dropped: under the 30-d-warmup operating point its
# information content was redundant with pH (both proxy the same charge-
# balance state) and its noisy sigma-point evaluation introduced filter
# instability.
#
# Note on PyADM1ODE sensor classes: the upstream package provides
# ``PhysicalSensor`` / ``ChemicalSensor`` / ``GasSensor`` with realistic
# noise, drift, response-time and sampling models. They are designed for
# *forward simulation* (passive observers), not for UKF sigma-point
# evaluation (where calling sensor.step() 89× per filter step would
# corrupt the sensor's drift / sampling state). For this twin we use
# simple, deterministic extractors here and let ``add_measurement_noise``
# apply the channel noise on the truth measurements — equivalent to a
# PhysicalSensor with ``response_time=0``, ``drift_rate=0``,
# ``sample_interval=dt``. The PyADM1ODE sensors can be wired in
# separately (on the truth-side only) to reproduce their richer error
# model — see the comment block below ``build_obs_model``.


def _make_ph_extractor(digester_id: str):
    def extractor(plant, x):  # noqa: ARG001
        val = plant.components[digester_id].outputs_data.get("pH", float("nan"))
        return 7.0 if not np.isfinite(val) else float(val)

    return extractor


def _channel_index(spec, name: str) -> int:
    """Look up the position of a named channel in the state vector."""
    for i, c in enumerate(spec.channels):
        if c.name == name:
            return i
    raise KeyError(f"Channel '{name}' not in spec")


def build_obs_model(spec) -> ObservationModel:
    # Substrate-flow indices in the augmented state vector. The names
    # match the InputSpec names from build_spec().
    i_maize = _channel_index(spec, "maize_silage")
    i_slurry = _channel_index(spec, "slurry")
    i_cereal = _channel_index(spec, "cereal_silage")

    return ObservationModel(
        channels=[
            ObservationChannel(
                name="Q_gas",
                extractor=extract_q_gas_total,
                noise_std=10.0,
            ),
            ObservationChannel(
                name="Q_ch4",
                extractor=extract_q_ch4_total,
                noise_std=5.0,
            ),
            ObservationChannel(
                name="pH",
                extractor=_make_ph_extractor(DIGESTER_ID),
                noise_std=0.05,
            ),
            # Substrate dosing sensors — simulate a hopper / pump-flow meter
            # on each substrate line. Typical industrial dosing equipment
            # has ~5 % relative uncertainty; we set noise_std to 5 % of the
            # nominal feed rate.
            ObservationChannel(
                name="Q_maize_silage",
                extractor=make_state_extractor(i_maize),
                noise_std=2.0,  # 5 % of 40.2 m³/d
            ),
            ObservationChannel(
                name="Q_slurry",
                extractor=make_state_extractor(i_slurry),
                noise_std=1.0,  # 5 % of 19.2 m³/d
            ),
            ObservationChannel(
                name="Q_cereal_silage",
                extractor=make_state_extractor(i_cereal),
                noise_std=0.1,  # ~17 % of 0.6 m³/d (small slot, hard to dose)
            ),
        ]
    )


# ---------------------------------------------------------------------------
# PyADM1ODE truth-side sensors (drift, response lag, sampling, noise)
# ---------------------------------------------------------------------------


def build_truth_sensors(seed: int) -> Dict[str, SensorAdapter]:
    """Construct realistic ``PhysicalSensor`` instances for the truth side.

    Each sensor's ``measurement_noise`` matches the corresponding
    ``ObservationChannel.noise_std`` in :func:`build_obs_model` so the
    UKF's ``R`` matrix is consistent with the actual instrumentation
    noise. Drift and response-time add bias / lag effects that the
    filter's white-noise R model does NOT capture — they make the
    twin closer to real-world plant operation.

    Returns a dict ``{channel_name: SensorAdapter}`` that can be
    passed straight to :func:`measure_truth_with_sensors`.
    """
    from pyadm1.components.sensors.physical import (  # type: ignore[import-not-found]
        PhysicalSensor,
    )

    def _flow_sensor(component_id, signal_key, noise, drift, seed_off):
        # Use sensor_type="flow" for any volume-flow channel. The
        # signal_key tells the sensor which key to read from inputs;
        # we pass that exact key in measure_truth_with_sensors.
        sensor = PhysicalSensor(
            component_id=component_id,
            sensor_type="flow",
            signal_key=signal_key,
            measurement_noise=float(noise),
            drift_rate=float(drift),
            response_time=30.0 / 86400.0,  # 30 s lag
            sample_interval=0.0,  # continuous (1 sample per dt)
            rng_seed=seed + seed_off,
        )
        return SensorAdapter(sensor, signal_key=signal_key)

    return {
        # Gas-side: 30 s response, no drift (NDIR flow meters are stable).
        "Q_gas": _flow_sensor(
            "q_gas_sensor",
            "Q_gas",
            noise=10.0,
            drift=0.0,
            seed_off=0,
        ),
        "Q_ch4": _flow_sensor(
            "q_ch4_sensor",
            "Q_ch4",
            noise=5.0,
            drift=0.0,
            seed_off=1,
        ),
        # pH probe: ~60 s response, small drift (electrodes age slowly).
        "pH": SensorAdapter(
            PhysicalSensor(
                component_id="ph_sensor",
                sensor_type="pH",
                signal_key="pH",
                measurement_noise=0.05,
                drift_rate=0.005,  # ~0.005 pH / day
                response_time=60.0 / 86400.0,  # 60 s
                sample_interval=0.0,
                rng_seed=seed + 2,
            ),
            signal_key="pH",
        ),
        # Substrate dosing scales / pump-flow meters: small drift,
        # 5 s response (mechanical inertia).
        "Q_maize_silage": _flow_sensor(
            "q_maize_sensor",
            "Q_maize_silage",
            noise=2.0,
            drift=0.01,
            seed_off=3,
        ),
        "Q_slurry": _flow_sensor(
            "q_slurry_sensor",
            "Q_slurry",
            noise=1.0,
            drift=0.005,
            seed_off=4,
        ),
        "Q_cereal_silage": _flow_sensor(
            "q_cereal_sensor",
            "Q_cereal_silage",
            noise=0.1,
            drift=0.002,
            seed_off=5,
        ),
    }


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------


def build_spec():
    """Full 41-state spec for the multi-stage plant.

    The substrate-input channels match the ``_SUBSTRATE_MIX`` defined
    in :mod:`pyadm1ode_estimation.example_plants.multi_stage`. If that
    mix changes, the InputSpecs here have to follow.
    """
    return adm1da_full_spec(
        digester_id=DIGESTER_ID,
        substrate_inputs=[
            InputSpec("maize_silage", substrate_index=0, initial_flow=40.2),
            InputSpec("slurry", substrate_index=1, initial_flow=19.2),
            InputSpec("cereal_silage", substrate_index=2, initial_flow=0.6),
        ],
    )


# Quality-class index lists for the 41 ADM1 channels — copied from
# pyadm1ode_estimation/estimation/specs.py::_STATE_BLOCKS so this
# script can group channels for plotting without importing private
# constants.
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _robust_ylim(*series, percentile=(1.0, 99.0), pad=0.15):
    """Compute a Y-axis limit that ignores extreme outliers.

    The UKF's initial sigma-point evaluation at t=0 can produce a
    huge ŷ spike that's an order of magnitude above the steady-state
    signal. Letting matplotlib autoscale onto this spike compresses
    the rest of the data into an invisible band. We use percentile-
    based limits (default 1st–99th) so the spike stays in the plot
    (as a single excursion above the axis) but doesn't dominate it.
    """
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


def plot_trajectory_grid(
    time,
    truth,
    x_hat,
    std,
    spec,
    indices,
    title,
    output_path,
):
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
            color="C0",
            alpha=0.15,
            label=r"$\pm 2\sigma$",
        )
        # Percentile-based Y-limit so the t=0 sigma-point spike on
        # x_hat does not flatten the steady-state trajectory.
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
        ax.plot(
            obs_clean.index, obs_clean[name].values, "k-", lw=1.5, label="clean truth"
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
        # Percentile-based Y-limit so the t=0 ŷ spike does not
        # dominate. clean truth + noisy measurements determine the
        # axis scale; the spike (one or two outlier ŷ points) shows
        # up as an out-of-plot excursion.
        ylim = _robust_ylim(
            obs_clean[name].values,
            obs_noisy[name].values,
            y_pred,
        )
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel("time [d]")
        ax.set_ylabel(name)
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)
    # Hide any unused subplots (if we have fewer channels than panels).
    for ax in axes.flat[len(channels) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Observations — truth, noisy measurement, filter prediction", fontsize=13
    )
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
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_summary(coverage, spec, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Group coverage by quality block.
    block_means = {}
    for block, indices in QUALITY_BLOCKS.items():
        block_means[block] = float(np.mean([coverage[i] for i in indices]))

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _warmup(plant, warmup_days, warmup_dt_hours, label):
    """Run pyadm1's native simulate() to reach a settled operating point.

    No observations recorded — we just want to advance the plant
    state past the substrate-input discontinuity at t = 0 so the
    UKF sees a quasi-stable trajectory rather than the initial
    Q_gas spike.
    """
    if warmup_days <= 0:
        return
    print(
        f"  Warming up {label} plant: {warmup_days:.0f} d "
        f"at dt = {warmup_dt_hours:.0f} h ..."
    )
    plant.simulate(
        duration=float(warmup_days),
        dt=float(warmup_dt_hours) / 24.0,
        save_interval=float(warmup_days),  # only keep the final snapshot
    )


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    n_steps = int(round(args.duration_days * 24.0 / args.dt_hours))

    print("Twin experiment — multi-stage example, 41-state SR-UKF")
    print(f"  Warm-up: {args.warmup_days:.0f} d (no filter)")
    print(
        f"  Filter horizon: {args.duration_days:.1f} d  |  "
        f"dt = {args.dt_hours:.1f} h  |  {n_steps} steps"
    )

    spec = build_spec()
    obs = build_obs_model(spec)
    print(f"  State vector: {len(spec)} channels  ({len(spec) - 41} augmented)")
    print(f"  Observation channels: {[c.name for c in obs.channels]}")

    # ---- Truth ---------------------------------------------------------
    print("\nBuilding truth plant ...")
    truth_plant = build_multi_stage_plant()
    _warmup(truth_plant, args.warmup_days, args.warmup_dt_hours, "truth")

    truth_process = ADM1ProcessModel(truth_plant, spec)
    truth_process.snapshot()
    # Read the warmed-up ADM1 state into the spec channel order. The
    # augmented input-flow channels keep their spec initial values
    # (the plant doesn't expose them as state — we declare them).
    x_truth0 = spec.read_adm1_state(truth_plant)
    aug_starts_at = 41
    for i, c in enumerate(spec.channels[aug_starts_at:], start=aug_starts_at):
        x_truth0[i] = c.initial

    # ---- Filter ---------------------------------------------------------
    # Deep-copy the warmed-up truth plant for the filter. Guarantees that
    # the filter's plant model is bit-identical to the truth at the moment
    # the UKF starts estimating. Side-effect: the UKF's first predict step
    # produces exactly the truth's first ODE step (modulo UKF transformation
    # error from sigma-point spread) when the initial perturbation is 0.
    # Also saves one full warm-up.
    print("\nBuilding filter plant (deepcopy of warmed-up truth plant) ...")
    filter_plant = copy.deepcopy(truth_plant)

    filter_process = ADM1ProcessModel(filter_plant, spec)
    filter_process.snapshot()

    # ---- Propagate truth + measure -------------------------------------
    print(f"\nPropagating truth for {args.duration_days:.1f} d ...")
    time, truth, obs_clean = propagate_truth(
        spec,
        truth_process,
        obs,
        x_truth0,
        dt_hours=args.dt_hours,
        n_steps=n_steps,
    )

    # Truth-side sensor pipeline: each PyADM1ODE sensor sees the clean
    # truth signal and produces a noisy measurement with drift, lag and
    # sampling. This is the *next-state-with-noise* measurement the
    # filter sees. The UKF itself never touches the sensors.
    print("Stepping truth-side sensors (drift + lag + noise) ...")
    truth_sensors = build_truth_sensors(seed=args.seed)
    obs_noisy = measure_truth_with_sensors(obs_clean, truth_sensors)

    ukf = UnscentedKalmanFilter(filter_process, obs, spec)
    if args.initial_perturbation_relative > 0.0:
        perturbation = rng.normal(
            0.0,
            args.initial_perturbation_relative * (np.abs(x_truth0) + 1e-6),
            size=len(spec),
        )
        print(
            f"\nInitial perturbation: ±{args.initial_perturbation_relative * 100:.0f} %"
            " relative Gaussian noise on every channel."
        )
    else:
        perturbation = np.zeros_like(x_truth0)
        print(
            "\nInitial perturbation: 0 (perfect initialisation — UKF's"
            " first predict should match truth's first step)."
        )
    x0 = spec.clip(x_truth0 + perturbation)

    # Initial covariance P0: we use a TIGHT P0 matching the actual
    # perturbation, not the spec's default initial_cov(). The spec
    # default encodes "filter starts without knowing the truth"
    # (factors 0.20–0.80 × magnitude), which is realistic on a real
    # plant but inappropriate here: deepcopy gives us x̂[0] ≈ x_truth[0]
    # exactly, so the actual uncertainty is just the perturbation σ.
    # Using the bloated default would spread sigma points ±130 % of
    # |x| and produce a huge t=0 ŷ spike for nonlinear gas-flow channels.
    sigma_init = max(args.initial_perturbation_relative, 1e-3)
    p0_diag = (sigma_init * (np.abs(x_truth0) + 1e-6)) ** 2
    P0 = np.diag(p0_diag)
    ukf.reset(x0, P0)
    print(
        f"Initial covariance: tight P0 with sigma = {sigma_init * 100:.1f} % "
        "x |x| per channel (matches the perturbation level)."
    )

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

    print("\n--- Per-quality-block coverage ---")
    for block, indices in QUALITY_BLOCKS.items():
        block_cov = float(np.mean([coverage[i] for i in indices]))
        print(
            f"  {block:30s}  n={len(indices):2d}  "
            f"2sigma-cov = {100 * block_cov:5.1f} %"
        )

    print(
        f"\n  Mean NIS = {nis_mean:.2f}  "
        f"(target {0.5 * n_active:.1f} – {2.0 * n_active:.1f} for "
        f"{n_active} channels)"
    )

    # ---- Plot slice (burn-in for filter convergence) -------------------
    # Diagnostics (coverage, NIS) are over the FULL run; only the visual
    # plots get cropped so we don't waste vertical axes on the initial
    # UKF transient.
    plot_mask = time >= args.plot_from_day
    n_kept = int(plot_mask.sum())
    if args.plot_from_day > 0:
        print(
            f"\n  Plot burn-in: skipping first {len(time) - n_kept} steps "
            f"(t < {args.plot_from_day:.1f} d). "
            f"Plotting {n_kept} of {len(time)} samples."
        )
    time_p = time[plot_mask]
    truth_p = truth[plot_mask]
    x_hat_p = x_hat[plot_mask]
    std_p = std[plot_mask]
    obs_clean_p = obs_clean.loc[obs_clean.index >= args.plot_from_day]
    obs_noisy_p = obs_noisy.loc[obs_noisy.index >= args.plot_from_day]
    steps_p = [s for s in steps if s.t >= args.plot_from_day]

    # ---- Plots ---------------------------------------------------------
    output_dir = args.output_dir
    print(f"\nWriting plots to {output_dir} ...")

    # Strong: 6 picks from methanogenesis + charge_balance
    plot_trajectory_grid(
        time_p,
        truth_p,
        x_hat_p,
        std_p,
        spec,
        indices=[6, 8, 27, 35, 38, 40],
        title="Strong-observable states (A + D fused)",
        output_path=output_dir / "trajectories_strong.png",
    )
    # Weak / open-loop: 6 picks
    plot_trajectory_grid(
        time_p,
        truth_p,
        x_hat_p,
        std_p,
        spec,
        indices=[0, 18, 22, 10, 11, 41],  # last is the first input_flow channel
        title="Medium / weak / open-loop states + 1 substrate input",
        output_path=output_dir / "trajectories_weak.png",
    )
    plot_observations(
        obs_clean_p,
        obs_noisy_p,
        steps_p,
        time_p,
        output_path=output_dir / "observations.png",
    )
    plot_nis(steps_p, time_p, output_path=output_dir / "nis.png")
    plot_coverage_summary(
        coverage, spec, output_path=output_dir / "coverage_summary.png"
    )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
