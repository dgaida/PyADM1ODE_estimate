"""Motivational figure for the report's open-loop reference section.

Reconstructs the SAME fixed-seed world as ``report_compare.py`` (truth = nominal
kinetics, identical noisy feed, time-growing lognormal kinetic model error
k(t)=k*exp(sigma*z*t/tau)) and free-runs the imperfect ADM1 model WITHOUT any
filter. Plots truth vs. open-loop model gas/methane for a sweep of model-error
magnitudes sigma, for both the steady and the substrate-change feed.

This is the visual companion to the ``openloop`` candidate of
``report_compare.py`` (same world, same drift); it only adds the per-sigma
truth-vs-model overlay that the per-block report figure does not show.

Run from the repository root::

    python experiments/twin_experiment/openloop_figure.py

Writes:
  * ``reports/figures/openloop.png`` — truth vs. model Q_gas/Q_ch4 overlay.
  * ``reports/figures/openloop_blocks_<feed>.png`` — per-block rolling NRMSE of
    the open-loop model vs. truth over the sigma sweep (same block aggregation
    as the report's per-block figures, e.g. ``A_steady.png``).
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_AP = argparse.ArgumentParser(description=__doc__)
_AP.add_argument(
    "--bias",
    type=float,
    default=0.0,
    help="systematic log-space mean shift of the kinetics (same convention as "
    "report_compare.py --bias); grows with g(t)=t/tau, reaching `bias` at t=tau. "
    "<0 = systematically slower / less gas. Default 0.",
)
_AP.add_argument(
    "--dt",
    type=float,
    default=6.0,
    help="simulation/sampling step [h] (matches report_compare.py --dt). " "Default 6.",
)
_ARGS = _AP.parse_args()

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_twin_experiment import (
    DIGESTER_ID,
    SUBSTRATES,
    _propagate_truth_with_substrate_noise,
)

from pyadm1ode_estimation.estimation import (
    BLOCK_INDICES,
    adm1da_full_spec,
    build_filter_components,
    realism,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

# ---- config (matches report_compare.py defaults) ---------------------------
SEED = 7
DURATION = 60.0  # days
DT_H = _ARGS.dt  # sim step [h]
WARMUP = 30.0
DRIFT_TAU = 30.0  # model-error drift time constant [days]; <=0 -> static
BIAS = _ARGS.bias  # systematic log-space mean shift of the kinetics (grows with
#                      g(t)=t/tau, reaching BIAS at t=tau; <0 = slower / less gas)
SUBNOISE = 0.10
SENSORS = ["q_gas", "q_ch4", "q_co2", "ph", "substrate_dose"]
n_steps = round(DURATION * 24.0 / DT_H)

# Same long-horizon feed phases as report_compare.REPORT_FEED_PHASES.
REPORT_FEED_PHASES = [
    (0.0, [4.74, 13.70, 1.09, 3.68, 0.20]),
    (20.0, [20.0, 16.0, 2.0, 2.0, 4.0]),
    (40.0, [0.5, 6.0, 0.3, 4.0, 0.05]),
]
FEED_DAYS = [d for d, _ in REPORT_FEED_PHASES[1:]]

SIGMAS = [0.10, 0.25, 0.40]  # model-error magnitude sweep
FEEDS = ["none", "change"]

print("Warming up plant ...", flush=True)
WARMED = build_multi_stage_plant()
WARMED.simulate(duration=WARMUP, dt=1.0, save_interval=WARMUP)

_FS = adm1da_full_spec(DIGESTER_ID)
SCALE = np.array(
    [
        max(abs(_FS.channels[i].initial), _FS.channels[i].upper / 100.0, 1e-9)
        for i in range(41)
    ]
)


def feed_schedule(spec, feed):
    if feed != "change":
        return None
    aug = [spec.channels[i] for i in spec.kind_indices("input_flow")]
    sched = np.zeros((n_steps + 1, len(aug)))
    dt = DT_H / 24.0
    for k in range(n_steps + 1):
        t = k * dt
        flows = REPORT_FEED_PHASES[0][1]
        for start_day, f in REPORT_FEED_PHASES:
            if t >= start_day:
                flows = f
        for c, ch in enumerate(aug):
            sched[k, c] = flows[ch.input_substrate_index]
    return sched


def build_world(feed):
    """Truth world: nominal kinetics, noisy feed."""
    truth_plant = copy.deepcopy(WARMED)
    od = truth_plant.components[DIGESTER_ID].outputs_data
    sensor_noise = realism.build_sensor_noise(
        {
            "q_gas": float(od.get("Q_gas", 1900.0)),
            "q_ch4": float(od.get("Q_ch4", 900.0)),
            "q_co2": float(od.get("Q_co2", 850.0)),
        }
    )
    proc, obs, spec = build_filter_components(
        truth_plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=SENSORS,
        sensor_noise=sensor_noise,
    )
    x0 = spec.read_adm1_state(truth_plant)
    for i, ch in enumerate(spec.channels):
        if ch.kind != "adm1":
            x0[i] = ch.initial
    rng = np.random.default_rng(SEED)
    time_d, truth_traj, obs_clean = _propagate_truth_with_substrate_noise(
        spec,
        proc,
        obs,
        x0,
        DT_H,
        n_steps,
        rng,
        SUBNOISE,
        nominal_schedule=feed_schedule(spec, feed),
    )
    feed_idx = spec.kind_indices("input_flow")
    return time_d, truth_traj, obs_clean, truth_traj[:, feed_idx], x0


def model_error_logf(sigma):
    """Seed-fixed log-space mismatch direction, scaled by sigma (== report)."""
    erng = np.random.default_rng(SEED + 1009)
    fk0 = WARMED.components[DIGESTER_ID].adm1._kinetic
    return {
        k: float(sigma * erng.standard_normal())
        for k, v in fk0.items()
        if isinstance(v, (int, float)) and k.startswith(realism.MODEL_ERROR_PREFIXES)
    }


def run_open_loop(sigma, x0, feed_traj):
    """Free-run the imperfect model from the measured substrate dose (passed in
    via ``feed_traj``), same x0 as truth; only the kinetics drift."""
    plant = copy.deepcopy(WARMED)
    proc, obs, mspec = build_filter_components(
        plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=SENSORS,
    )
    logf = model_error_logf(sigma)
    nominal = {k: float(plant.components[DIGESTER_ID].adm1._kinetic[k]) for k in logf}
    feed_pos = mspec.kind_indices("input_flow")

    def apply_err(t_days):
        g = 1.0 if DRIFT_TAU <= 0 else (t_days / DRIFT_TAU)
        fk = plant.components[DIGESTER_ID].adm1._kinetic
        for k, lf in logf.items():
            fk[k] = nominal[k] * float(np.exp((lf + BIAS) * g))

    dt = DT_H / 24.0
    x = x0.copy()
    for j, i in enumerate(feed_pos):
        x[i] = feed_traj[0, j]
    apply_err(0.0)
    proc.refresh_outputs(x, equilibration_dt=1.0 / 24.0)
    rows = [[float(c.extractor(proc.plant, x)) for c in obs.channels]]
    states = np.zeros((n_steps + 1, len(mspec)))
    states[0] = x
    for k in range(n_steps):
        for j, i in enumerate(feed_pos):
            x[i] = feed_traj[k, j]
        apply_err(k * dt)
        x = proc.step(x, dt)
        states[k + 1] = x
        apply_err((k + 1) * dt)
        proc.refresh_outputs(x, equilibration_dt=1.0 / 24.0)
        rows.append([float(c.extractor(proc.plant, x)) for c in obs.channels])
    obs_model = pd.DataFrame(rows, columns=[c.name for c in obs.channels])
    return states, obs_model


def cum_err(truth, model):
    ct, cm = np.nansum(truth), np.nansum(model)
    return (cm - ct) / max(abs(ct), 1e-12) * 100.0


results = {}
for feed in FEEDS:
    time_d, truth_traj, obs_truth, feed_traj, x0 = build_world(feed)
    truth41 = truth_traj[:, :41]
    # MEASURED substrate dose = delivered feed + 3 % dosing-scale noise, the same
    # signal the UKF feeds forward (seed SEED+7, realism substrate_dose 0.03).
    dose_rng = np.random.default_rng(SEED + 7)
    measured_feed = np.clip(
        feed_traj
        + dose_rng.normal(
            0.0, 0.03 * (np.abs(feed_traj[0]) + 1e-9), size=feed_traj.shape
        ),
        0.0,
        None,
    )
    for sigma in SIGMAS:
        states, obs_model = run_open_loop(sigma, x0, measured_feed)
        model41 = states[:, :41]
        h = len(truth41) // 2
        nr = np.sqrt(np.mean((model41[h:] - truth41[h:]) ** 2, axis=0)) / SCALE
        block_nrmse = {b: float(np.median(nr[idx])) for b, idx in BLOCK_INDICES.items()}
        weighted = float(
            np.median([block_nrmse[b] for b in BLOCK_INDICES if b != "charge_balance"])
        )
        results[(feed, sigma)] = {
            "weighted": weighted,
            "methano": block_nrmse["methanogenesis"],
            "gas_cum": cum_err(obs_truth["Q_gas"], obs_model["Q_gas"]),
            "ch4_cum": cum_err(obs_truth["Q_ch4"], obs_model["Q_ch4"]),
            "obs_truth": obs_truth,
            "obs_model": obs_model,
            "time": time_d,
            "truth41": truth41,
            "model41": model41,
        }
        print(
            f"[{feed:6s} sigma={sigma:.2f}] weighted-NRMSE={weighted:.3f} "
            f"methano={block_nrmse['methanogenesis']:.3f} | "
            f"Q_gas cum {results[(feed,sigma)]['gas_cum']:+.1f}% | "
            f"Q_ch4 cum {results[(feed,sigma)]['ch4_cum']:+.1f}%",
            flush=True,
        )

print("\n=== Open-loop model error (NO UKF), second-half vs truth ===")
print(
    f"{'feed':8s}{'sigma':>7s}{'wNRMSE':>9s}{'methano':>9s}"
    f"{'Qgas_cum%':>11s}{'Qch4_cum%':>11s}"
)
for feed in FEEDS:
    for sigma in SIGMAS:
        r = results[(feed, sigma)]
        print(
            f"{feed:8s}{sigma:>7.2f}{r['weighted']:>9.3f}{r['methano']:>9.3f}"
            f"{r['gas_cum']:>+11.1f}{r['ch4_cum']:>+11.1f}"
        )

# ---- plot: Q_gas / Q_ch4 truth vs open-loop model --------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(
    len(FEEDS), 2, figsize=(13, 4.2 * len(FEEDS)), sharex=True, squeeze=False
)
for row, feed in enumerate(FEEDS):
    for col, ch in enumerate(["Q_gas", "Q_ch4"]):
        ax = axes[row][col]
        r0 = results[(feed, SIGMAS[0])]
        t = r0["time"]
        ax.plot(t, r0["obs_truth"][ch], "k-", lw=2.0, label="truth (no model error)")
        for sigma in SIGMAS:
            r = results[(feed, sigma)]
            ax.plot(
                t,
                r["obs_model"][ch],
                lw=1.2,
                label=f"model error σ={sigma} (~±{sigma * 100:.0f}%)",
            )
        if feed == "change":
            for d in FEED_DAYS:
                ax.axvline(d, color="0.6", ls=":", lw=0.9)
        feed_label = "steady" if feed == "none" else feed
        ax.set_title(f"{ch} (feed={feed_label})", fontsize=10)
        ax.set_ylabel(f"{ch} [m3/d]")
        ax.grid(alpha=0.3)
        if row == 0 and col == 0:
            ax.legend(loc="best", fontsize=8)
for ax in axes[-1]:
    ax.set_xlabel("time [d]")
fig.suptitle(
    "ADM1 with vs without model error, NO filter "
    f"(seed {SEED}, drift_tau={DRIFT_TAU:g}d, bias={BIAS:g}, {DURATION:g}d)",
    fontsize=13,
)
fig.tight_layout()
out = _ROOT / "reports" / "figures" / "openloop.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=110, bbox_inches="tight")
plt.close(fig)
print(f"\nWrote {out}")

# ---- plot: per-BLOCK rolling NRMSE, open-loop model vs truth ----------------
# Same aggregation as report_compare.block_series (the report's per-block
# figures): each block is summarized into one curve = rolling 1-day median of
# the per-state NRMSE |model - truth| / SCALE. One 3x3 grid per feed, the sigma
# sweep as lines. This is the no-filter analogue of A_steady.png / B_feed.png.
WIN = max(1, round(24.0 / DT_H))  # ~1-day rolling window


def block_rolling_nrmse(model41, truth41):
    e = np.abs(model41 - truth41) / SCALE
    out = {}
    for b, idx in BLOCK_INDICES.items():
        med = np.median(e[:, idx], axis=1)
        out[b] = np.convolve(med, np.ones(WIN) / WIN, mode="same")
    # weighted overall = median over blocks except the ill-conditioned charge_balance
    out["weighted"] = np.median(
        np.stack([out[b] for b in BLOCK_INDICES if b != "charge_balance"]), axis=0
    )
    return out


PLOT_BLOCKS = ["weighted"] + list(BLOCK_INDICES)  # 1 + 9 = 10 panels (4x3 grid)

for feed in FEEDS:
    truth41 = results[(feed, SIGMAS[0])]["truth41"]
    t = results[(feed, SIGMAS[0])]["time"]
    feed_label = "steady" if feed == "none" else feed
    series_by_sigma = {
        sigma: block_rolling_nrmse(results[(feed, sigma)]["model41"], truth41)
        for sigma in SIGMAS
    }
    ncol = 3
    rows = (len(PLOT_BLOCKS) + ncol - 1) // ncol
    fig, axes = plt.subplots(
        rows, ncol, figsize=(4.8 * ncol, 2.7 * rows), sharex=True, squeeze=False
    )
    flat = [ax for row in axes for ax in row]
    for j, ax in enumerate(flat):
        if j >= len(PLOT_BLOCKS):
            ax.set_visible(False)
            continue
        b = PLOT_BLOCKS[j]
        for sigma in SIGMAS:
            ax.plot(
                t,
                series_by_sigma[sigma][b],
                lw=1.3,
                label=f"σ={sigma} (~±{sigma * 100:.0f}%)",
            )
        if feed == "change":
            for d in FEED_DAYS:
                ax.axvline(d, color="0.6", ls=":", lw=0.8)
        title = b + ("  (overall)" if b == "weighted" else "")
        ax.set_title(title, fontsize=9, fontweight="bold" if b == "weighted" else None)
        ax.set_ylabel("rolling NRMSE")
        ax.grid(alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("time [d]")
    fig.suptitle(
        f"Open-loop model error per ADM1 block (feed={feed_label}, NO "
        f"filter; seed {SEED}, drift_tau={DRIFT_TAU:g}d, bias={BIAS:g})",
        fontsize=12,
    )
    fig.tight_layout()
    sout = _ROOT / "reports" / "figures" / f"openloop_blocks_{feed_label}.png"
    fig.savefig(sout, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {sout}")
