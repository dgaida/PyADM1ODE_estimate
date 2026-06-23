"""Report driver: full-41 vs A+D-core (+ ConstrainedUKF, +known-input) on the
FINAL UKF implementation, one consistent code path for every scenario.

A single fixed-seed "world" (truth with real kinetics + realistic sensor noise
+ daily CH4/CO2 gating) is shared by every candidate, so differences are purely
the estimator. The motion model of every UKF is realistically perturbed by the
SAME lognormal kinetic model error (estimation.realism, magnitude --sigma); the
perturbation DIRECTION is seed-fixed so raising --sigma only scales the same
mismatch (clean model-error trend).

Candidates:
  full      - SR-UKF, full 41-state vector.
  adcore    - SR-UKF, reduced A+D core (methanogenesis+charge_balance, 18/41);
              un-estimated states run open-loop through the imperfect model.
  cukf      - ConstrainedUKF (box-bounded update, Kolas 2009 / Hellmann 2024),
              full 41-state vector. Tests whether bounding tames the full UKF.
  adcore_ki - A+D core with KNOWN INPUT: feed fed forward from the measured
              dose instead of estimated (only meaningful with feed change).

Writes, per --tag, into reports/results/: an array dump <tag>.npz (for the
plot scripts), a raw time-series plot <tag>.png and provenance <tag>_meta.txt;
with --save-traj also <tag>_traj.npz (for obs_plots.py).

Usage:
  python report_compare.py --tag A_steady --feed none  --sigma 0.25 \
                           --candidates full,adcore,cukf --duration 60 --dt 6
  python report_compare.py --tag B_feed   --feed change --sigma 0.25 \
                           --candidates full,adcore,adcore_ki,cukf
"""

from __future__ import annotations

import argparse
import copy
import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = (
    Path(__file__).resolve().parents[2]
)  # experiments/twin_experiment/X.py -> repo root
RESULTS = _ROOT / "reports" / "results"
sys.path.insert(0, str(_ROOT))
sys.path.insert(
    0, str(Path(__file__).resolve().parent)
)  # co-located run_twin_experiment
from run_twin_experiment import (  # noqa: E402
    DIGESTER_ID,
    SUBSTRATES,
    _DAILY_GAS_CHANNELS,
    _apply_sensor_schedule,
    _propagate_truth_with_substrate_noise,
    evaluate_per_stage_gas,
)

from pyadm1ode_estimation.estimation import (  # noqa: E402
    ADM1ProcessModel,
    BLOCK_INDICES,
    adm1da_full_spec,
    build_filter_components,
    build_ukf,
    realism,
)
from pyadm1ode_estimation.estimation.filters.constrained_ukf import (  # noqa: E402
    ConstrainedUKF,
)
from pyadm1ode_estimation.estimation.twin import (  # noqa: E402
    add_measurement_noise,
    coverage_within_2sigma,
    run_filter,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant  # noqa: E402

# ---- CLI -------------------------------------------------------------------
p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
p.add_argument("--seed", type=int, default=7)
p.add_argument("--duration", type=float, default=60.0)
p.add_argument("--dt", type=float, default=6.0)
p.add_argument("--sigma", type=float, default=realism.MODEL_ERROR_KINETIC_SIGMA)
p.add_argument("--feed", choices=["none", "change"], default="none")
p.add_argument("--candidates", default="full,adcore")
p.add_argument("--warmup", type=float, default=30.0)
p.add_argument(
    "--save-traj",
    action="store_true",
    help="also dump observation/state trajectories to <tag>_traj.npz "
    "(truth vs estimate of gas/methane/pH/VFA) for the real-vs-estimated plots",
)
A = p.parse_args()
SAVE_TRAJ = A.save_traj
VFA_IDX = [3, 4, 5, 6]  # S_va, S_bu, S_pro, S_ac -> total VFA

DT_H, DURATION, SEED = A.dt, A.duration, A.seed
PERT, SUBNOISE = 0.05, 0.10
SENSORS = ["q_gas", "q_ch4", "q_co2", "ph", "substrate_dose"]  # daily CH4/CO2
AD_CORE = sorted(BLOCK_INDICES["methanogenesis"] + BLOCK_INDICES["charge_balance"])
n_steps = int(round(DURATION * 24.0 / DT_H))
CANDS = A.candidates.split(",")
_FS = adm1da_full_spec(DIGESTER_ID)
SCALE = np.array(
    [
        max(abs(_FS.channels[i].initial), _FS.channels[i].upper / 100.0, 1e-9)
        for i in range(41)
    ]
)
# Long-horizon feed schedule (composition steps, m³/d), reusing the physical
# mixes of the twin's _FEED_PHASES but spread across the 60-day run so the
# operating point moves twice over the horizon.
REPORT_FEED_PHASES = [
    (0.0, [4.74, 13.70, 1.09, 3.68, 0.20]),  # nominal manure-based (~23 m³/d)
    (20.0, [20.0, 16.0, 2.0, 2.0, 4.0]),  # high-energy load (~44 m³/d)
    (40.0, [0.5, 6.0, 0.3, 4.0, 0.05]),  # low dilute load (~11 m³/d)
]
FEED_DAYS = [d for d, _ in REPORT_FEED_PHASES[1:]]


def _git(path):
    try:
        c = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        d = subprocess.run(
            ["git", "-C", path, "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return c + ("+dirty" if d else "")
    except Exception:
        return "?"


def _feed_schedule(spec):
    if A.feed != "change":
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


print(
    f"[{A.tag}] seed={SEED} {DURATION:g}d dt={DT_H:g}h sigma={A.sigma} feed={A.feed} "
    f"cands={CANDS}. Warming up ...",
    flush=True,
)
WARMED = build_multi_stage_plant()
WARMED.simulate(duration=A.warmup, dt=1.0, save_interval=A.warmup)


def apply_model_error(plant, kin):
    fk = plant.components[DIGESTER_ID].adm1._kinetic
    for k, f in kin.items():
        fk[k] = float(fk[k]) * f


# ---- shared world (fixed seed) ---------------------------------------------
rng = np.random.default_rng(SEED)
erng = np.random.default_rng(SEED + 1009)
# z fixed by seed; magnitude scales with --sigma => same mismatch direction.
KIN = {
    k: float(np.exp(A.sigma * erng.standard_normal()))
    for k, v in WARMED.components[DIGESTER_ID].adm1._kinetic.items()
    if isinstance(v, (int, float)) and k.startswith(realism.MODEL_ERROR_PREFIXES)
}

truth_plant = copy.deepcopy(WARMED)
od = truth_plant.components[DIGESTER_ID].outputs_data
SENSOR_NOISE = realism.build_sensor_noise(
    {
        "q_gas": float(od.get("Q_gas", 1900.0)),
        "q_ch4": float(od.get("Q_ch4", 900.0)),
        "q_co2": float(od.get("Q_co2", 850.0)),
    }
)
tproc, tobs, tspec = build_filter_components(
    truth_plant,
    digester_id=DIGESTER_ID,
    substrates=SUBSTRATES,
    sensors=SENSORS,
    sensor_noise=SENSOR_NOISE,
)
x0 = tspec.read_adm1_state(truth_plant)
for i, ch in enumerate(tspec.channels):
    if ch.kind != "adm1":
        x0[i] = ch.initial
time_d, truth_traj, obs_clean = _propagate_truth_with_substrate_noise(
    tspec,
    tproc,
    tobs,
    x0,
    DT_H,
    n_steps,
    rng,
    SUBNOISE,
    nominal_schedule=_feed_schedule(tspec),
)
obs_noisy = _apply_sensor_schedule(
    add_measurement_noise(obs_clean, tobs, rng), tobs, DT_H, _DAILY_GAS_CHANNELS
)
TRUTH41 = truth_traj[:, :41]
TRUTH_FEED = truth_traj[:, tspec.kind_indices("input_flow")]
PERT_VEC = rng.normal(0.0, PERT * (np.abs(x0) + 1e-6), size=len(tspec))
_ADP = {c.adm1_index: i for i, c in enumerate(tspec.channels) if c.kind == "adm1"}
_INP = {
    c.input_substrate_index: i
    for i, c in enumerate(tspec.channels)
    if c.kind == "input_flow"
}


def prior_in_layout(spec):
    x = np.zeros(len(spec))
    for pos, ch in enumerate(spec.channels):
        if ch.kind == "adm1":
            x[pos] = TRUTH41[0, ch.adm1_index] + PERT_VEC[_ADP[ch.adm1_index]]
        elif ch.kind == "input_flow":
            j = _INP[ch.input_substrate_index]
            x[pos] = truth_traj[0, j] + PERT_VEC[j]
    return x


def est_ukf(adm1_indices, constrained=False):
    plant = copy.deepcopy(WARMED)
    apply_model_error(plant, KIN)  # SAME realistic motion-model error for all
    base = build_ukf(
        plant,
        digester_id=DIGESTER_ID,
        substrates=SUBSTRATES,
        sensors=SENSORS,
        sensor_noise=SENSOR_NOISE,
        initial_uncertainty_relative=PERT,
        adm1_indices=adm1_indices,
    )
    spec = base.spec
    if constrained:
        ukf = ConstrainedUKF(
            base.process, base.obs, spec, alpha=1.0, beta=2.0, kappa=0.0
        )
    else:
        ukf = base
    ukf.reset(spec.clip(prior_in_layout(spec)), base.P)
    x_hat, std, steps = run_filter(ukf, spec, ukf.obs, obs_noisy, None, DT_H)
    _qg, _qc, est41 = evaluate_per_stage_gas(
        copy.deepcopy(plant), spec, x_hat, ["primary"], DT_H
    )
    est_qgas, est_qch4 = _qg["primary"], _qc["primary"]
    truth_layout = np.zeros((n_steps + 1, len(spec)))
    for pos, ch in enumerate(spec.channels):
        if ch.kind == "adm1":
            est41[:, ch.adm1_index] = x_hat[:, pos]
            truth_layout[:, pos] = TRUTH41[:, ch.adm1_index]
    nis = np.array([s.nis for s in steps], dtype=float)
    cov = coverage_within_2sigma(truth_layout, x_hat, std)
    adm1_pos = [pos for pos, ch in enumerate(spec.channels) if ch.kind == "adm1"]
    traj = None
    if SAVE_TRAJ:
        ph_name = next(
            (c.name for c in ukf.obs.channels if "ph" in c.name.lower()), None
        )
        est_ph = np.array(
            [
                steps[k].y_pred.get(ph_name, np.nan) if ph_name else np.nan
                for k in range(len(steps))
            ]
        )
        traj = dict(
            qgas=np.asarray(est_qgas, dtype=float),
            qch4=np.asarray(est_qch4, dtype=float),
            ph=est_ph,
            vfa=est41[:, VFA_IDX].sum(axis=1),
        )
    return (
        est41,
        float(np.nanmean(nis[np.isfinite(nis)])),
        float(np.mean(cov[adm1_pos])),
        traj,
    )


# ---- known-input A+D (feed fed forward, not estimated) ---------------------
def _set_feed(dig, feed_row):
    qv = list(dig.Q_substrates)
    for j in range(min(len(qv), len(feed_row))):
        qv[j] = float(feed_row[j])
    dig.Q_substrates = qv
    try:
        dig.adm1.create_influent(qv, 0)
    except Exception:
        pass


def est_ukf_ki():
    plant = copy.deepcopy(WARMED)
    apply_model_error(plant, KIN)
    ukf = build_ukf(
        plant,
        digester_id=DIGESTER_ID,
        substrates=[],
        sensors=["q_gas", "q_ch4", "q_co2", "ph"],
        sensor_noise=SENSOR_NOISE,
        initial_uncertainty_relative=PERT,
        adm1_indices=AD_CORE,
    )
    spec = ukf.spec
    x0i = ukf.x_hat.copy()
    for pos, ch in enumerate(spec.channels):
        x0i[pos] = TRUTH41[0, ch.adm1_index] + PERT_VEC[_ADP[ch.adm1_index]]
    ukf.reset(spec.clip(x0i), ukf.P)
    krng = np.random.default_rng(SEED + 7)
    nom = np.abs(TRUTH_FEED[0]) + 1e-9
    known = np.clip(
        TRUTH_FEED + krng.normal(0.0, 0.03 * nom, size=TRUTH_FEED.shape), 0.0, None
    )
    dt = DT_H / 24.0
    dig = ukf.process.plant.components[DIGESTER_ID]
    T = len(obs_noisy)
    x_hat = np.zeros((T, len(spec)))
    std = np.zeros((T, len(spec)))
    steps = []
    for k, t in enumerate(obs_noisy.index):
        _set_feed(dig, known[k])
        if k > 0:
            ukf.predict(dt)
        y = {
            c.name: float(obs_noisy.iloc[k][c.name])
            for c in ukf.obs.channels
            if c.name in obs_noisy.columns and np.isfinite(obs_noisy.iloc[k][c.name])
        }
        step = ukf.update(y, t=float(t))
        x_hat[k] = step.x_hat
        std[k] = np.sqrt(np.diag(step.P))
        steps.append(step)
    # open-loop replay (full 41) driven by x_hat + known feed
    proc = ADM1ProcessModel(copy.deepcopy(plant), spec)
    rdig = proc.plant.components[DIGESTER_ID]
    est41 = np.full((T, 41), np.nan)
    for k in range(T):
        _set_feed(rdig, known[k])
        proc.snapshot()
        proc.step(x_hat[k], dt)
        ad = np.asarray(rdig.adm1_state, dtype=float)
        est41[k, : min(41, ad.size)] = ad[:41]
    truth_layout = np.zeros((T, len(spec)))
    for pos, ch in enumerate(spec.channels):
        est41[:, ch.adm1_index] = x_hat[:, pos]
        truth_layout[:, pos] = TRUTH41[:, ch.adm1_index]
    nis = np.array([s.nis for s in steps], dtype=float)
    cov = coverage_within_2sigma(truth_layout, x_hat, std)
    traj = None
    if SAVE_TRAJ:

        def _name(*keys, excl=()):
            return next(
                (
                    c.name
                    for c in ukf.obs.channels
                    if all(k in c.name.lower() for k in keys)
                    and not any(e in c.name.lower() for e in excl)
                ),
                None,
            )

        def _series(nm):
            a = np.array(
                [
                    steps[k].y_pred.get(nm, np.nan) if nm else np.nan
                    for k in range(len(steps))
                ],
                dtype=float,
            )
            last = np.nan  # forward-fill daily-gated channels (e.g. q_ch4)
            for i in range(len(a)):
                if np.isfinite(a[i]):
                    last = a[i]
                elif np.isfinite(last):
                    a[i] = last
            return a

        traj = dict(
            qgas=_series(_name("gas", excl=("ch4", "co2"))),
            qch4=_series(_name("ch4")),
            ph=_series(_name("ph")),
            vfa=est41[:, VFA_IDX].sum(axis=1),
        )
    return est41, float(np.nanmean(nis[np.isfinite(nis)])), float(np.mean(cov)), traj


RUNNERS = {
    "full": lambda: est_ukf(None, constrained=False),
    "adcore": lambda: est_ukf(AD_CORE, constrained=False),
    "cukf": lambda: est_ukf(None, constrained=True),
    "adcore_ki": est_ukf_ki,
}

# ---- run -------------------------------------------------------------------
import time as _time  # noqa: E402

BLOCKS = list(BLOCK_INDICES) + ["weighted"]
WIN = max(1, int(round(24.0 / DT_H)))  # ~1-day rolling mean


def block_series(est41):
    e = np.abs(est41 - TRUTH41) / SCALE
    out = {b: np.median(e[:, idx], axis=1) for b, idx in BLOCK_INDICES.items()}
    out["weighted"] = np.median(
        np.stack([out[b] for b in BLOCK_INDICES if b != "charge_balance"]), axis=0
    )
    for k in out:
        out[k] = np.convolve(out[k], np.ones(WIN) / WIN, mode="same")
    return out


def second_half(est41):
    """RMSE-NRMSE per block over the converged second half (matches MC metric)."""
    h = len(TRUTH41) // 2
    nr = np.sqrt(np.mean((est41[h:] - TRUTH41[h:]) ** 2, axis=0)) / SCALE
    out = {b: float(np.median(nr[idx])) for b, idx in BLOCK_INDICES.items()}
    out["weighted"] = float(
        np.median([out[b] for b in BLOCK_INDICES if b != "charge_balance"])
    )
    return out


series = np.zeros((len(CANDS), len(BLOCKS), n_steps + 1))
nis_arr, cov_arr = {}, {}
summary = {}
trajs = {}
for ci, name in enumerate(CANDS):
    _t = _time.time()
    est41, nis, cov, traj = RUNNERS[name]()
    print(
        f"  {name}: {_time.time() - _t:.1f}s  NIS={nis:.1f} cov={100 * cov:.0f}%",
        flush=True,
    )
    s = block_series(est41)
    for bi, b in enumerate(BLOCKS):
        series[ci, bi] = s[b]
    nis_arr[name], cov_arr[name] = nis, cov
    summary[name] = second_half(est41)
    if traj is not None:
        trajs[name] = traj

# ---- checkpoint table ------------------------------------------------------
cps = [d for d in (10, 20, 30, 45, 60) if d <= DURATION]
print(f"\n=== [{A.tag}] rolling-NRMSE checkpoints (cols = {'/'.join(CANDS)}) ===")
print(f"{'block':24s} " + "  ".join(f"d{d:>2}" for d in cps))
for bi, b in enumerate(BLOCKS):
    line = f"{b:24s} "
    for d in cps:
        k = min(n_steps, int(round(d * 24.0 / DT_H)))
        line += "  " + "/".join(f"{series[ci, bi, k]:.2f}" for ci in range(len(CANDS)))
    print(line)
print("\nsecond-half NRMSE (cols = " + "/".join(CANDS) + "):")
for b in BLOCKS:
    print(f"  {b:24s} " + " / ".join(f"{summary[c][b]:.3f}" for c in CANDS))
print(
    "\nNIS / coverage: "
    + "  ".join(f"{c}={nis_arr[c]:.1f}/{100 * cov_arr[c]:.0f}%" for c in CANDS)
)

# ---- plot ------------------------------------------------------------------
RESULTS.mkdir(parents=True, exist_ok=True)
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    to_plot = [
        "weighted",
        "methanogenesis",
        "acidogenesis_substrates",
        "hydrolysis_sums",
        "disintegration_split",
        "inerts",
    ]
    colors = {"full": "C0", "adcore": "C1", "cukf": "C2", "adcore_ki": "C3"}
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    for ax, b in zip(axes.flat, to_plot):
        bi = BLOCKS.index(b)
        for ci, name in enumerate(CANDS):
            ax.plot(
                time_d, series[ci, bi], colors.get(name, f"C{ci}"), lw=1.3, label=name
            )
        if A.feed == "change":
            for d in FEED_DAYS:
                ax.axvline(d, color="0.6", ls=":", lw=0.9)
        role = (
            "estimated by both"
            if b in ("methanogenesis",)
            else ("decision headline" if b == "weighted" else "A+D: open-loop")
        )
        ax.set_title(f"{b}  ({role})", fontsize=10)
        ax.set_ylabel("rolling NRMSE")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time [d]")
    fig.suptitle(
        f"[{A.tag}] full-41 vs A+D (seed {SEED}, feed={A.feed}, model-error sigma={A.sigma})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/{A.tag}.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nWrote {RESULTS}/{A.tag}.png")
except Exception as e:  # noqa: BLE001
    print(f"(plot skipped: {e})")

# ---- dump + provenance -----------------------------------------------------
np.savez(
    f"{RESULTS}/{A.tag}.npz",
    time=time_d,
    candidates=np.array(CANDS),
    blocks=np.array(BLOCKS),
    series=series,
    nis=np.array([nis_arr[c] for c in CANDS]),
    coverage=np.array([cov_arr[c] for c in CANDS]),
    summary=np.array([[summary[c][b] for b in BLOCKS] for c in CANDS]),
    sigma=A.sigma,
    feed=A.feed,
    seed=SEED,
    duration=DURATION,
    dt=DT_H,
)
if SAVE_TRAJ and trajs:
    traj_cands = list(trajs)
    np.savez(
        f"{RESULTS}/{A.tag}_traj.npz",
        time=time_d,
        feed=A.feed,
        feed_days=np.array(FEED_DAYS if A.feed == "change" else [], dtype=float),
        obs_cols=np.array(list(obs_clean.columns)),
        obs_clean=obs_clean.to_numpy(dtype=float),
        obs_noisy=obs_noisy.to_numpy(dtype=float),
        truth_vfa=TRUTH41[:, VFA_IDX].sum(axis=1),
        traj_cands=np.array(traj_cands),
        est_qgas=np.stack([trajs[c]["qgas"] for c in traj_cands]),
        est_qch4=np.stack([trajs[c]["qch4"] for c in traj_cands]),
        est_ph=np.stack([trajs[c]["ph"] for c in traj_cands]),
        est_vfa=np.stack([trajs[c]["vfa"] for c in traj_cands]),
    )
    print(f"Wrote {RESULTS}/{A.tag}_traj.npz (real-vs-estimated trajectories)")
with open(f"{RESULTS}/{A.tag}_meta.txt", "w", encoding="utf-8") as fh:
    fh.write(
        f"timestamp     : {datetime.datetime.now().isoformat(timespec='seconds')}\n"
    )
    fh.write(f"estimate_git  : {_git(str(_ROOT))}\n")
    fh.write(
        "filter_class  : estimation.filters.sr_ukf.UnscentedKalmanFilter (SR-UKF)\n"
    )
    fh.write(
        "              + estimation.filters.constrained_ukf.ConstrainedUKF (cukf)\n"
    )
    fh.write(
        "ukf_params    : alpha=1.0 beta=2.0 kappa=0.0 gamma=canonical sqrt(n+lambda)\n"
    )
    fh.write(
        f"config        : tag={A.tag} seed={SEED} duration={DURATION}d dt={DT_H}h "
        f"sigma={A.sigma} feed={A.feed} candidates={'/'.join(CANDS)}\n"
    )
    fh.write(f"sensors       : {'/'.join(SENSORS)} (CH4/CO2 daily, rest at dt)\n")
    if A.feed == "change":
        fh.write(f"feed_phases   : {REPORT_FEED_PHASES} (steps at days {FEED_DAYS})\n")
print(f"Wrote {RESULTS}/{A.tag}.npz + {RESULTS}/{A.tag}_meta.txt")
