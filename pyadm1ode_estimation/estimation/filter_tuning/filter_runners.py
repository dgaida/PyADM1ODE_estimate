"""Runners for the model-based **filters** (dataset-driven) + parallel collection.

`make_ukf_runner(meta, variant)` returns a **picklable** ``UkfRunner`` — callable as
``runner(theta, episode) -> (x_hat, std)`` — the single interface every tuner consumes.
The plant / operating point is built **from the dataset's ``meta``** (k_dec_ac, substrates,
sensors, baseline load), so the same runner works for any ADM1-style dataset. Variants:

* ``"full"``   — 41-state known-input UKF (the shipped reference).
* ``"adcore"`` — observable A+D core (methanogenesis + charge_balance, 18 states); the
  non-core states are reconstructed open-loop → output is still (T, 41), std ``NaN`` there.
* ``"adcore_vfa"`` — the A+D core plus S_va/S_bu/S_pro (21 states). Only makes sense with
  ``fostac_every_days`` set, because those three are what a FOS titration observes.

Passing ``fostac_every_days`` adds two gated lab channels (FOS and TAC in mg/L) on top of
the online sensors. They are Nordmann titrations, not an online signal, so they are only
present every N days; the rest of the axis is NaN and the filter skips it automatically.

``theta`` (all optional) parametrises Q/R/P0 and the post-hoc σ:
    p0_scale, process_noise_scale (global Q), q_groups+q_scale (per-block Q scale),
    q_diag (per-state absolute process σ² — e.g. learned by the differentiable filter),
    r_scale (global R), r_diag (per-sensor absolute R), std_scale (post-hoc output-σ).

``collect_parallel`` runs a runner over many episodes across processes (Enabler B); the
``UkfRunner`` is picklable so this works with ``multiprocessing.Pool`` on Windows (spawn).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
from pyadm1.core.adm1 import STATE_SIZE

from ...example_plants import build_multi_stage_plant
from ...example_plants.multi_stage import _SUBSTRATE_MIX as _DEFAULT_MIX
from ..observation_model import ObservationChannel
from ..quickstart import build_filter_components, build_ukf
from ..specs import BLOCK_INDICES, InputSpec
from ..twin import run_filter
from . import metrics as _metrics

AD_CORE = sorted(
    set(BLOCK_INDICES["methanogenesis"]) | set(BLOCK_INDICES["charge_balance"])
)

#: The three volatile fatty acids that make up the FOS numerator but are *not* in the A+D
#: core, because they are not observable from gas + pH + TS alone. A FOS titration observes
#: exactly their weighted sum, so it is the one measurement that justifies estimating them.
#: Deliberately not the whole ``acidogenesis_substrates`` block: S_SU and S_AA do not enter
#: FOS, so adding them would only hand the filter more states it still cannot see.
#: Indices come from the FOS/TAC model itself, so the two cannot drift apart.
VFA_STATES = sorted(_metrics._I[_n] for _n in ("S_va", "S_bu", "S_pro"))

#: A+D core extended by the VFA substrates. Only sensible together with a FOS measurement.
AD_CORE_VFA = sorted(set(AD_CORE) | set(VFA_STATES))

#: State indices the FOS/TAC forward model reads, for documentation and sanity checks.
VARIANT_STATES = {"full": None, "adcore": AD_CORE, "adcore_vfa": AD_CORE_VFA}

#: Titration channel names. They must not collide with the online sensor names in ``meta``.
FOS_CHANNEL, TAC_CHANNEL = "FOS", "TAC"

#: Measurement noise of one Nordmann titration, as standard deviation in mg/L. Derived from
#: the dataset's own error model (2 % sample error, 0.15 mL endpoint error, 0.02 mL burette
#: error per leg): the endpoint error shifts titrant between the two legs, so it dominates
#: FOS and is negligible for TAC. Measured on the shipped test set, the residual FOS/TAC
#: spread is about 12 % relative.
FOS_NOISE_STD = 125.0
TAC_NOISE_STD = 200.0


def make_fos_tac_extractors(digester_id: str):
    """``(h_FOS, h_TAC)`` predicting a Nordmann titration in mg/L from a sigma point.

    Both read the **full 41-state vector out of the plant**, not the filter's own state
    vector. The process model writes each sigma point into the plant before the observation
    model is evaluated, so this sees the complete state including the open-loop ones. That
    matters for the reduced variants: the A+D core does not estimate S_va/S_bu/S_pro, yet
    they enter FOS, and reading them from the plant keeps the prediction physically correct
    instead of silently dropping two thirds of the numerator.

    The formula is :func:`metrics.fos_tac_mg_l`, verified bit-identical to the titration
    the dataset stores, so the filter is not fitting a biased forward model.
    """

    def _state(plant) -> np.ndarray:
        return np.asarray(plant.components[digester_id].adm1_state, float)

    def h_fos(plant, x) -> float:
        return float(_metrics.fos_tac_mg_l(_state(plant))[0])

    def h_tac(plant, x) -> float:
        return float(_metrics.fos_tac_mg_l(_state(plant))[1])

    return h_fos, h_tac


# --------------------------------------------------------------------------
# meta -> plant / substrates / sensors
# --------------------------------------------------------------------------
def substrates_from_meta(meta: dict) -> list[InputSpec]:
    subs = sorted(meta["substrates"].items(), key=lambda kv: kv[1]["index"])
    return [
        InputSpec(
            name,
            substrate_index=v["index"],
            initial_flow=float(v["nominal_flow_m3_per_d"]),
        )
        for name, v in subs
    ]


def baseline_flows(meta: dict, substrates: list[InputSpec]) -> np.ndarray:
    nominal = np.array([s.initial_flow for s in substrates], float)
    total = float(
        meta.get("operating_point", {}).get("baseline_total_m3_per_d", nominal.sum())
    )
    return nominal * (total / max(nominal.sum(), 1e-9))


#: Influent composition a dataset's truth obeys; see :func:`influent_mode`.
NOMINAL_MIX = "nominal_mix"
BLEND = "blend"
INFLUENT_MODES = (NOMINAL_MIX, BLEND)


def influent_mode(meta: dict) -> str:
    """How the influent composition of a dataset's truth follows the feed.

    ``"blend"``: the composition is the flow-weighted blend of the substrates fed
    at each step. This is what pyadm1 computes today.

    ``"nominal_mix"``: the composition is that of the plant's nominal mix at every
    step and the reported flows only set the total flow. pyadm1 used to cache the
    blend when a digester was initialised, so every dataset generated before that
    was fixed obeys this mode, whatever its feed record says.

    The mode is read from ``meta["influent_mode"]``. Datasets without the field
    predate the fix and are ``"nominal_mix"``; a generator running on the fixed
    pyadm1 has to write ``"blend"``.

    Args:
        meta: The dataset metadata.

    Returns:
        ``"nominal_mix"`` or ``"blend"``.

    Raises:
        ValueError: The field holds another value.
    """
    mode = str(meta.get("influent_mode", NOMINAL_MIX))
    if mode not in INFLUENT_MODES:
        raise ValueError(
            f"meta['influent_mode'] must be one of {INFLUENT_MODES}, got {mode!r}"
        )
    return mode


def freeze_nominal_mix(plant) -> None:
    """Pin every digester's influent composition to its nominal substrate mix.

    Reproduces the process model the ``"nominal_mix"`` datasets were generated
    with: the composition is the blend at the digester's configured
    ``Q_substrates`` and later feed rates only change the flow. pyadm1 honours an
    explicitly set influent DataFrame over the feedstock blend, so this is the
    documented way to ask for that behaviour.

    Args:
        plant: A built ``BiogasPlant``; modified in place.
    """
    for component in plant.components.values():
        adm1 = getattr(component, "adm1", None)
        feedstock = getattr(adm1, "feedstock", None)
        if feedstock is None or not hasattr(feedstock, "get_influent_dataframe"):
            continue
        adm1.set_influent_dataframe(
            feedstock.get_influent_dataframe(Q=list(component.Q_substrates))
        )


def build_plant(
    meta: dict, digester_id: str = "primary", *, apply_operating_point: bool = True
):
    """Example plant with the dataset's operating point and substrates applied.

    The substrate mix comes from ``meta`` rather than from the plant's default. A
    dataset generated on a different feed would otherwise be filtered with a process
    model that converts the reported flows through the wrong characterisation, and the
    resulting error would look like an estimator weakness instead of a mismatch that
    was introduced here. Which substrates are dosed is a known input in this benchmark,
    so telling the filter is the honest setup; what is *not* re-fitted for a new
    dataset is Q, R and the kinetics.

    Falls back to the plant's own mix when ``meta`` predates the ``library_id`` field.

    The influent composition follows :func:`influent_mode`: for a ``"nominal_mix"``
    dataset the digesters are pinned to the nominal mix with
    :func:`freeze_nominal_mix`, so filter and physics run on the model that
    generated the truth.

    Args:
        meta: The dataset metadata.
        digester_id: Component whose kinetics carry the operating point.
        apply_operating_point: Write ``k_dec_ac`` from ``meta``. The PINN adapter
            passes ``False``: its published results were produced against the plant's
            own decay constant, and correcting that here would silently change numbers
            that are already in the report. It is a real model error, worth fixing on
            purpose rather than as a side effect of a substrate swap.
    """
    flows = {
        v["library_id"]: float(v["nominal_flow_m3_per_d"])
        for v in sorted(meta.get("substrates", {}).values(), key=lambda v: v["index"])
        if v.get("library_id")
    }
    total = sum(flows.values())
    mix = {k: v / total for k, v in flows.items()} if total > 0 else None

    # Only override when the dataset really uses a different feed. The plant's own
    # fractions sum to 1.0001 and are multiplied by 23.4 rather than by the 23.41 the
    # meta flows add up to, so re-deriving them shifts the seeded inoculum by ~3e-3
    # even when the substrates are the same -- enough to move published results for a
    # change that was meant to leave them alone.
    if (
        mix is not None
        and set(mix) == set(_DEFAULT_MIX)
        and all(abs(mix[k] - _DEFAULT_MIX[k]) < 1e-3 for k in mix)
    ):
        mix = None

    plant = build_multi_stage_plant(substrate_mix=mix)
    kdec = meta.get("operating_point", {}).get("k_dec_ac")
    if apply_operating_point and kdec is not None:
        plant.components[digester_id].adm1._kinetic["k_dec_ac"] = float(kdec)
    if influent_mode(meta) == NOMINAL_MIX:
        freeze_nominal_mix(plant)
    return plant


#: Components the benchmark sensors depend on. The cascade is one-directional
#: (primary -> secondary -> storage, gas -> CHP -> heating), so everything else
#: cannot influence an observed quantity; see :func:`prune_to_primary`.
PRIMARY_COMPONENTS = frozenset({"primary", "primary_storage"})


def prune_to_primary(plant, keep=PRIMARY_COMPONENTS):
    """Drop every component that cannot influence the primary digester's sensors.

    ``ADM1ProcessModel.step`` advances the whole plant for every sigma point, and
    the downstream digesters dominate the stiff solver. The kinetics-calibration
    study verified that the pruned plant gives bit-identical primary observables
    (``PyADM1ODE_calibration/experiments/kinetics_calibration/test_fastsim.py``);
    the final test checks the same for the filter against its stored references.

    Args:
        plant: A built ``BiogasPlant``; modified in place and returned.
        keep: Component ids to keep.
    """
    for cid in list(plant.components):
        if cid not in keep:
            del plant.components[cid]
    plant.connections = [
        c
        for c in plant.connections
        if c.from_component in keep and c.to_component in keep
    ]
    for comp in plant.components.values():
        comp.inputs = [i for i in getattr(comp, "inputs", []) if i in keep]
        comp.outputs = [o for o in getattr(comp, "outputs", []) if o in keep]
    return plant


def apply_kinetic_factors(
    plant, factors: dict | None, digester_id: str = "primary"
) -> list[str]:
    """Multiply the digester's kinetic constants by ``factors`` (``key -> factor``).

    The factors are relative to the plant's nominal values, the convention of the
    benchmark's ``kinetic_factors`` and of the calibration (log-factor search).

    Returns:
        The keys that were applied.

    Raises:
        KeyError: for a key the digester does not have.
    """
    if not factors:
        return []
    kinetic = plant.components[digester_id].adm1._kinetic
    for key, f in factors.items():
        if key not in kinetic:
            raise KeyError(f"unknown kinetic constant {key!r}")
        kinetic[key] = float(kinetic[key]) * float(f)
    return sorted(factors)


def _apply_theta(spec, obs, theta: dict) -> None:
    """Set Q (per-block scale and/or per-state absolute) and R (global scale and/or
    per-sensor absolute) on the built filter (mutates in place)."""
    if not theta:
        return
    pos = {spec.channels[p].adm1_index: p for p in spec.kind_indices("adm1")}
    # per-block Q scale
    groups = theta.get("q_groups") or BLOCK_INDICES
    for name, idxs in groups.items():
        s = float(theta.get("q_scale", {}).get(name, 1.0))
        if s != 1.0:
            for a in idxs:
                if int(a) in pos:
                    spec.channels[pos[int(a)]].process_noise_std *= s
    # per-state absolute Q (variance) override — e.g. from the differentiable filter
    qd = theta.get("q_diag")
    if qd is not None:
        qd = np.asarray(qd, float)
        for a, p in pos.items():
            spec.channels[p].process_noise_std = float(
                np.sqrt(max(float(qd[a]), 1e-30))
            )
    # R
    rs = float(theta.get("r_scale", 1.0))
    if rs != 1.0:
        for c in obs.channels:
            if getattr(c, "noise_std", None):
                c.noise_std = float(c.noise_std) * rs
    rd = theta.get("r_diag")
    if rd is not None:
        rd = np.asarray(rd, float)
        for j, c in enumerate(obs.channels):
            if j < len(rd) and getattr(c, "noise_std", None) is not None:
                c.noise_std = float(np.sqrt(max(float(rd[j]), 1e-30)))


def _propagate_open_loop(spec, process, obs, x0, feed, dt_hours, n_steps):
    """Impose the known feed each step, no correction; return full (T, n_state) states."""
    dt = dt_hours / 24.0
    aug = spec.kind_indices("input_flow")
    x = x0.copy()
    states = np.zeros((n_steps + 1, len(spec)))
    for k in range(n_steps + 1):
        for j, i_aug in enumerate(aug):
            x[i_aug] = feed[min(k, n_steps), j]
        states[k] = x
        if k < n_steps:
            x = process.step(x, dt)
    return states


def _warmup(process, spec, x0, warmup_days, feed_flows, dt_hours):
    dt = dt_hours / 24.0
    x = x0.copy()
    for i, i_aug in enumerate(spec.kind_indices("input_flow")):
        x[i_aug] = feed_flows[i]
    for _ in range(round(warmup_days * 24.0 / dt_hours)):
        x = process.step(x, dt)
    return x


# --------------------------------------------------------------------------
# Picklable runner
# --------------------------------------------------------------------------
class UkfRunner:
    """Picklable ``runner(theta, episode) -> (x_hat (T,41), std (T,41))`` for a variant.

    ``prune`` (default on) simulates only the estimated digester and its gas storage.
    Every sigma point otherwise drags the two downstream digesters through the stiff
    solver as well, which costs more than half of the run time, and nothing they compute
    reaches the primary digester's states or sensors: the output is bit-identical (checked
    over a full 60-day benchmark series). Pruning applies only to the digester ids in
    :data:`PRIMARY_COMPONENTS`; another ``digester_id`` keeps the whole plant.
    """

    def __init__(
        self,
        meta: dict,
        variant: str = "full",
        digester_id: str = "primary",
        warmup_days: float | None = None,
        fostac_every_days: float | None = None,
        fostac_noise: tuple[float, float] | None = None,
        prune: bool = True,
    ):
        self.meta = meta
        self.variant = variant
        # Simulate only the components the sensors depend on (see prune_to_primary).
        self.prune = bool(prune) and digester_id in PRIMARY_COMPONENTS
        # Lab titration: None disables it entirely, a number is the sampling interval in
        # days (7 = weekly, 1 = daily, dt_hours/24 = every step).
        self.fostac_every_days = fostac_every_days
        self.fostac_noise = (
            tuple(fostac_noise) if fostac_noise else (FOS_NOISE_STD, TAC_NOISE_STD)
        )
        self.did = digester_id
        self.substrates = substrates_from_meta(meta)
        self.sensors = [s.lower() for s in meta["sensors"]]
        self.channel_names = list(meta["sensors"])
        self.base = baseline_flows(meta, self.substrates)
        self.dt_hours = float(meta.get("dt_hours", 1.0))
        self.wdays = float(
            warmup_days if warmup_days is not None else meta.get("warmup_days", 30.0)
        )
        self.sub_idx = [s.substrate_index for s in self.substrates]
        if variant not in VARIANT_STATES:
            raise ValueError(
                f"unknown variant {variant!r}, expected one of "
                f"{sorted(VARIANT_STATES)}"
            )
        self.core = VARIANT_STATES[variant]

    def cache_key(self) -> str:
        """Digest of everything that shapes this runner's output, for the task cache.

        ``prune`` changes the run time, not the output, so its slot is fixed to the
        value older keys were written with: cached unpruned runs stay valid for a pruned
        runner and vice versa.
        """
        return _digest(
            [
                type(self).__name__,
                self.meta,
                self.variant,
                self.did,
                False,  # former prune slot, see above
                self.fostac_every_days,
                self.fostac_noise,
                self.wdays,
            ]
        )

    def __call__(self, theta: dict, ep) -> tuple[np.ndarray, np.ndarray]:
        theta = theta or {}
        # Kinetics of this plant (``key -> factor``), e.g. from a calibration artifact.
        # Applied before the warm-up so warm-up, open loop and filter share them.
        kinetics = ep.obs.get("kinetics")
        meas = np.asarray(ep.obs["measurements"], float)
        feed = np.asarray(ep.obs["feed_noisy"], float)
        time = np.asarray(ep.obs["time"], float)
        n_steps = len(time) - 1

        plant = build_plant(self.meta, self.did)
        if self.prune:
            prune_to_primary(plant)
        apply_kinetic_factors(plant, kinetics, self.did)
        wp, wobs, wspec = build_filter_components(
            plant,
            digester_id=self.did,
            substrates=self.substrates,
            sensors=self.sensors,
        )
        xw = _warmup(
            wp,
            wspec,
            wspec.read_adm1_state(plant),
            self.wdays,
            self.base,
            self.dt_hours,
        )
        adm1_vec = np.zeros(STATE_SIZE)
        for i, ch in enumerate(wspec.channels):
            if ch.kind == "adm1":
                adm1_vec[ch.adm1_index] = xw[i]
        plant.components[self.did].adm1_state = list(adm1_vec)

        ol41 = None
        if self.core is not None:
            ol_full = _propagate_open_loop(
                wspec, wp, wobs, xw.copy(), feed, self.dt_hours, n_steps
            )
            ol41 = ol_full[:, wspec.kind_indices("adm1")]

        sensors = list(self.sensors)
        if self.fostac_every_days:
            h_fos, h_tac = make_fos_tac_extractors(self.did)
            s_fos = float(theta.get("fos_noise", self.fostac_noise[0]))
            s_tac = float(theta.get("tac_noise", self.fostac_noise[1]))
            sensors += [
                ObservationChannel(FOS_CHANNEL, h_fos, s_fos),
                ObservationChannel(TAC_CHANNEL, h_tac, s_tac),
            ]

        p0 = 0.05 * float(theta.get("p0_scale", 1.0))
        ukf = build_ukf(
            plant,
            digester_id=self.did,
            substrates=(),
            sensors=sensors,
            initial_uncertainty_relative=p0,
            adm1_indices=self.core,
            process_noise_scale=float(theta.get("process_noise_scale", 1.0)),
        )
        spec, obs = ukf.spec, ukf.obs
        _apply_theta(spec, obs, theta)
        adm1_pos = spec.kind_indices("adm1")
        core_idx = [spec.channels[p].adm1_index for p in adm1_pos]

        cols = {
            c.name: meas[:, self.channel_names.index(c.name)]
            for c in obs.channels
            if c.name in self.channel_names
        }
        if self.fostac_every_days:
            # NaN everywhere except on sampling days. run_filter only feeds a channel when
            # its value is finite, so this *is* the gating — no separate gate frame needed.
            # The dataset stores an independent titration for every hour, so picking rows
            # 0, N, 2N, ... is a statistically correct weekly series, not an approximation.
            ft = ep.obs.get("fostac")
            if ft is None:
                raise KeyError(
                    "fostac_every_days is set but the episode carries no 'fostac' array. "
                    "The dataset must provide the hourly titration."
                )
            step = max(1, round(self.fostac_every_days * 24.0 / self.dt_hours))
            masked = np.full((len(time), 2), np.nan)
            masked[::step] = np.asarray(ft, float)[: len(time)][::step]
            cols[FOS_CHANNEL] = masked[:, 0]
            cols[TAC_CHANNEL] = masked[:, 1]
        obs_noisy = pd.DataFrame(cols, index=time)

        def pre_step(k, _t):
            if k > 0:
                ukf.process.set_known_input(
                    {si: float(feed[k - 1, j]) for j, si in enumerate(self.sub_idx)}
                )

        xh, sd, _ = run_filter(
            ukf,
            spec,
            obs,
            obs_noisy,
            gate_frame=None,
            dt_hours=self.dt_hours,
            pre_step=pre_step,
        )

        if self.core is None:
            x_out, s_out = xh[:, adm1_pos], sd[:, adm1_pos]
        else:
            x_out = ol41.copy()
            s_out = np.full_like(ol41, np.nan)
            x_out[:, core_idx] = xh[:, adm1_pos]
            s_out[:, core_idx] = sd[:, adm1_pos]

        scale = theta.get("std_scale")
        if scale is not None:
            s_out = s_out * np.asarray(scale, float)[None, :]
        return x_out, s_out


def make_ukf_runner(
    meta: dict,
    variant: str = "full",
    digester_id: str = "primary",
    warmup_days: float | None = None,
    fostac_every_days: float | None = None,
    fostac_noise: tuple[float, float] | None = None,
    prune: bool = True,
) -> UkfRunner:
    """Build the picklable :class:`UkfRunner`; see there for ``prune``."""
    return UkfRunner(
        meta,
        variant,
        digester_id,
        warmup_days,
        fostac_every_days=fostac_every_days,
        fostac_noise=fostac_noise,
        prune=prune,
    )


# --------------------------------------------------------------------------
# Parallel collection (Enabler B)
# --------------------------------------------------------------------------
def _collect_worker(args):
    """Run one (theta, episode) task. Returns ``None`` if the filter blew up.

    Failures are contained here on purpose: a stiff Q draw can make the ADM1 integration
    fail, and letting that exception travel out of a pool worker tears down the whole
    batch (one bad candidate would kill an entire generation). The caller treats a ``None``
    as "this candidate is unusable" and moves on.
    """
    runner, theta, ep = args
    try:
        xh, sd = runner(theta, ep)
    except Exception:  # noqa: BLE001
        return None
    b = int(getattr(ep, "burnin", 0) or 0)
    return (np.asarray(ep.truth, float)[b:], xh[b:], sd[b:])


class WorkerPool:
    """A **reusable** process pool for filter runs.

    Creating a fresh ``mp.Pool`` for every batch is the fragile part on Windows: each new
    pool re-spawns workers that must duplicate the parent's pipe handle, and under load
    that can fail with ``PermissionError: [WinError 5]``, taking the run down at an
    arbitrary batch boundary. Holding one pool for the whole session avoids the repeated
    spawn entirely — and saves the (substantial) worker start-up cost as well.

        with WorkerPool(32) as pool:
            for theta in candidates:
                collect_batch(runner, [theta], episodes, pool=pool)
    """

    def __init__(self, jobs: int = 1):
        self.jobs = int(jobs)
        self._pool = None

    def __enter__(self):
        if self.jobs > 1:
            import multiprocessing as mp

            self._pool = mp.Pool(self.jobs)
        return self

    def __exit__(self, *exc):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None
        return False

    def map(self, args):
        if self._pool is None:
            return [_collect_worker(a) for a in args]
        return self._pool.map(_collect_worker, args, chunksize=1)


# --------------------------------------------------------------------- task cache ---
#: Directory that persists every finished (theta, episode) filter run, or None.
_TASK_CACHE: Path | None = None


def set_task_cache(path: str | Path | None) -> None:
    """Persist every finished ``(theta, episode)`` filter run under ``path``.

    A 60-day UKF run takes hours, and a multi-day tuning run evaluates hundreds of them. With
    the cache set, each result is written the moment its worker returns, and any later batch
    that contains the same task loads it instead of recomputing it. A run killed by a reboot
    therefore loses only the tasks that were in flight, whatever stage it was in.

    The key covers the runner configuration (dataset meta included), the theta and the
    episode's arrays, but not the code. Use one cache directory per run and delete it with
    the run, not across code changes. ``None`` switches the cache off. Only runners with a
    ``cache_key()`` method are cached.

    Args:
        path: Cache directory, created if missing, or ``None``.
    """
    global _TASK_CACHE
    _TASK_CACHE = None if path is None else Path(path)
    if _TASK_CACHE is not None:
        _TASK_CACHE.mkdir(parents=True, exist_ok=True)


def _digest(obj, h=None) -> str:
    """Stable SHA-256 over nested dicts, sequences, arrays and scalars."""
    top = h is None
    h = hashlib.sha256() if top else h
    if isinstance(obj, np.ndarray):
        a = np.ascontiguousarray(obj)
        h.update(f"nd{a.dtype.str}{a.shape}".encode())
        h.update(a.tobytes())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            h.update(repr(k).encode())
            _digest(obj[k], h)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            _digest(v, h)
        h.update(b"]")
    elif isinstance(obj, np.generic):
        _digest(obj.item(), h)
    else:
        h.update(f"{type(obj).__name__}:{obj!r};".encode())
    return h.hexdigest() if top else ""


def _task_key(runner, theta, ep) -> str | None:
    if not hasattr(runner, "cache_key"):
        return None
    return _digest(
        [
            runner.cache_key(),
            theta or {},
            ep.obs,
            ep.truth,
            ep.dt_hours,
            int(getattr(ep, "burnin", 0) or 0),
        ]
    )


_MISS = object()


def _cache_load(key: str, ep):
    f = _TASK_CACHE / f"{key}.npz"
    if not f.exists():
        return _MISS
    try:
        with np.load(f) as z:
            if bool(z["failed"]):
                return None
            b = int(getattr(ep, "burnin", 0) or 0)
            return (np.asarray(ep.truth, float)[b:], z["x_hat"], z["std"])
    except Exception:  # noqa: BLE001 - a truncated file is just a miss
        return _MISS


def _cache_save(key: str, res) -> None:
    f = _TASK_CACHE / f"{key}.npz"
    tmp = f.with_name(f"{key}.{os.getpid()}.tmp.npz")
    if res is None:
        np.savez(tmp, failed=True)
    else:
        np.savez(tmp, failed=False, x_hat=res[1], std=res[2])
    os.replace(tmp, f)  # atomic, so a reboot never leaves a half-written entry


def _indexed_worker(item):
    i, args = item
    return i, _collect_worker(args)


def _run_tasks(args, jobs, pool: WorkerPool | None = None):
    if _TASK_CACHE is None:
        return _run_tasks_uncached(args, jobs, pool)
    keys = [_task_key(*a) for a in args]
    out: list = [None] * len(args)
    todo = []
    for i, (a, k) in enumerate(zip(args, keys)):
        hit = _MISS if k is None else _cache_load(k, a[2])
        if hit is _MISS:
            todo.append(i)
        else:
            out[i] = hit
    if len(todo) < len(args):
        print(
            f"[task cache] {len(args) - len(todo)}/{len(args)} runs restored "
            f"from {_TASK_CACHE}",
            flush=True,
        )
    items = [(i, args[i]) for i in todo]
    if pool is not None and pool._pool is not None:
        stream = pool._pool.imap_unordered(_indexed_worker, items, chunksize=1)
        done = _drain(stream, keys, out)
    elif pool is None and jobs and jobs > 1 and len(items) > 1:
        import multiprocessing as mp

        with mp.Pool(min(jobs, len(items))) as p:
            done = _drain(
                p.imap_unordered(_indexed_worker, items, chunksize=1), keys, out
            )
    else:
        done = _drain(map(_indexed_worker, items), keys, out)
    assert done == len(items)
    return out


def _drain(stream, keys, out) -> int:
    """Store results as they arrive, writing each to the cache at once."""
    n = 0
    for i, res in stream:
        out[i] = res
        if keys[i] is not None:
            _cache_save(keys[i], res)
        n += 1
    return n


def _run_tasks_uncached(args, jobs, pool: WorkerPool | None = None):
    if pool is not None:
        return pool.map(args)
    if jobs and jobs > 1 and len(args) > 1:
        import multiprocessing as mp

        with mp.Pool(min(jobs, len(args))) as p:
            return p.map(_collect_worker, args, chunksize=1)
    return [_collect_worker(a) for a in args]


def collect_parallel(
    runner: UkfRunner,
    episodes: list,
    theta: dict | None = None,
    jobs: int = 1,
    pool: WorkerPool | None = None,
):
    """Run ``runner`` over episodes; return burn-in-trimmed (truth, x̂, σ) per episode.
    Pass a :class:`WorkerPool` to reuse workers across calls, else ``jobs>1`` spawns a
    throw-away pool."""
    return _run_tasks([(runner, theta or {}, ep) for ep in episodes], jobs, pool)


def collect_batch(
    runner: UkfRunner,
    thetas: list,
    episodes: list,
    jobs: int = 1,
    pool: WorkerPool | None = None,
):
    """Evaluate MANY parameter sets in **one** parallel wave.

    Flattens every ``(theta, episode)`` pair into a single pool of ``len(thetas) ×
    len(episodes)`` tasks, so all cores stay busy. Evaluating candidates one after the
    other (each parallelising only over its episodes) wastes most of the machine whenever
    ``len(episodes) < n_cores`` — e.g. 4 episodes on 32 cores would use 4 workers and
    serialise the candidates, ~8x slower for a population of 8.

    Returns one residual list per theta, in input order.
    """
    n = len(episodes)
    flat = _run_tasks(
        [(runner, th or {}, ep) for th in thetas for ep in episodes], jobs, pool
    )
    return [flat[i * n : (i + 1) * n] for i in range(len(thetas))]
