"""Simulator-driven dataset generator for the amortised PINN observer.

The amortised observer (unlike the per-window ``PinnSmoother``) is *pre-trained*
on many simulated scenarios so it generalises across operating points, then
fine-tuned online. This module produces the offline pre-training data: it runs
the plant simulator over randomised scenarios (varied substrate feed + initial
state) and records, per scenario,

* the **noisy sensor measurements** (Q_gas, Q_ch4, pH) + the known feed — the
  observer's *inputs*, and
* the **true 41-state ADM1 trajectory** — the supervised *target*.

Reuses the twin machinery (:func:`propagate_truth`, :func:`add_measurement_noise`)
and the standard filter components, so the scenarios match the twin experiment.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from pyadm1.core.adm1_torch import Adm1TorchParams

from ..quickstart import build_filter_components
from ..specs import InputSpec
from ..twin import add_measurement_noise, propagate_truth

_STATE_SIZE = 41


@dataclass
class ObserverDataset:
    """Pre-training data for the amortised observer.

    Attributes:
        features: ``(N, T, n_feat)`` normalised inputs (measurements + feed).
        states: ``(N, T, 41)`` true ADM1 state trajectory (the target).
        time: ``(T,)`` window times [days].
        feature_names: names of the ``n_feat`` input channels.
        feat_mean, feat_std: per-feature normalisation stats (so the online
            phase can apply the same scaling).
        x_ref: the nominal warmed 41-state — the observer's log-transform base.
        params: ADM1 parameter snapshot for the observer's gas solve / physics.
        measurements: ``(N, T, n_channels)`` *raw* sensor values (the first
            ``n_channels`` features before normalisation) — kept so the same
            windows can also drive the self-supervised objective (measurement +
            physics loss) without ground truth.
        channel_names: the ``n_channels`` sensor channels in ``measurements``.
    """

    features: np.ndarray
    states: np.ndarray
    time: np.ndarray
    feature_names: list[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    x_ref: np.ndarray
    params: Adm1TorchParams
    measurements: np.ndarray | None = None
    channel_names: list[str] | None = None

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """Apply the stored normalisation to raw ``(..., n_feat)`` inputs."""
        return (raw - self.feat_mean) / self.feat_std


def generate_observer_dataset(
    n_scenarios: int,
    substrates: Sequence[InputSpec],
    *,
    sensors: Sequence[str] = ("q_gas", "q_ch4", "ph"),
    digester_id: str = "primary",
    plant_builder: Callable | None = None,
    warmup_days: float = 30.0,
    duration_days: float = 4.0,
    dt_hours: float = 1.0,
    init_sigma: float = 0.08,
    feed_range: tuple = (0.6, 1.4),
    seed: int = 0,
) -> ObserverDataset:
    """Generate ``n_scenarios`` simulated (measurement, true-state) windows.

    Args:
        substrates: the plant's substrate inputs (their nominal flows are the
            feed baseline; each scenario scales them by a random factor).
        sensors: state-derived sensor channels the observer sees.
        plant_builder: ``() -> BiogasPlant``; defaults to the multi-stage example.
        warmup_days: settle the plant before generating scenarios.
        duration_days, dt_hours: window length and step.
        init_sigma: lognormal std of the per-scenario initial-state perturbation.
        feed_range: (min, max) multiplicative factor on the nominal feed.
        seed: RNG seed.
    """
    if plant_builder is None:
        from ...example_plants import build_multi_stage_plant

        plant_builder = build_multi_stage_plant

    rng = np.random.default_rng(seed)
    n_steps = round(duration_days * 24.0 / dt_hours)

    plant = plant_builder()
    if warmup_days > 0:
        plant.simulate(
            duration=float(warmup_days), dt=1.0, save_interval=float(warmup_days)
        )

    process, obs, spec = build_filter_components(
        plant, digester_id=digester_id, substrates=substrates, sensors=sensors
    )
    channel_names = [c.name for c in obs.channels]
    adm1_pos = spec.kind_indices("adm1")
    input_pos = spec.kind_indices("input_flow")

    # Nominal state + feed baseline (the warmed operating point).
    x_base = spec.read_adm1_state(plant)
    for i, ch in enumerate(spec.channels):
        if ch.kind == "input_flow":
            x_base[i] = ch.initial

    feat_list: list[np.ndarray] = []
    state_list: list[np.ndarray] = []
    meas_list: list[np.ndarray] = []
    time = None
    for _ in range(n_scenarios):
        x0 = x_base.copy()
        # initial-state variety (lognormal on the ADM1 channels)
        for p in adm1_pos:
            x0[p] = max(x0[p] * float(np.exp(rng.normal(0.0, init_sigma))), 1e-12)
        # operating-point variety: scale the whole feed by one random factor
        feed_factor = float(rng.uniform(*feed_range))
        for p in input_pos:
            x0[p] = max(x_base[p] * feed_factor, 0.0)

        time, states, obs_clean = propagate_truth(
            spec, process, obs, x0, dt_hours=dt_hours, n_steps=n_steps
        )
        obs_noisy = add_measurement_noise(obs_clean, obs, rng)

        meas = obs_noisy[channel_names].to_numpy()  # (T+1, n_ch)
        feed = np.array([x0[p] for p in input_pos], dtype=float)  # (n_sub,)
        feed_seq = np.tile(feed, (meas.shape[0], 1))  # (T+1, n_sub)
        feat_list.append(np.concatenate([meas, feed_seq], axis=1))
        state_list.append(states[:, adm1_pos])  # (T+1, 41)
        meas_list.append(meas)  # (T+1, n_ch)

    features = np.stack(feat_list, axis=0)  # (N, T+1, n_feat)
    states_arr = np.stack(state_list, axis=0)  # (N, T+1, 41)
    meas_arr = np.stack(meas_list, axis=0)  # (N, T+1, n_ch)

    feat_mean = features.reshape(-1, features.shape[-1]).mean(axis=0)
    feat_std = features.reshape(-1, features.shape[-1]).std(axis=0) + 1e-8
    features_norm = (features - feat_mean) / feat_std

    feature_names = list(channel_names) + [f"feed_{s.name}" for s in substrates]
    x_ref = np.array([x_base[p] for p in adm1_pos], dtype=float)
    params = Adm1TorchParams.from_adm1(plant.components[digester_id].adm1)
    return ObserverDataset(
        features=features_norm,
        states=states_arr,
        time=np.asarray(time, dtype=float),
        feature_names=feature_names,
        feat_mean=feat_mean,
        feat_std=feat_std,
        x_ref=x_ref,
        params=params,
        measurements=meas_arr,
        channel_names=list(channel_names),
    )


@dataclass
class MeasurementDataset:
    """Measurement-only windows for **self-supervised** pre-training.

    The supervised :class:`ObserverDataset` needs the true state (only the
    simulator can provide it). Real plant history has no ground truth, so it is
    trained with the same self-supervised objective as the online fine-tuning
    (measurement fit + physics residual). This container carries what that needs:
    the normalised observer inputs, the raw measured sensor values, and the ADM1
    parameters for the physics term.

    Build it from real logs with :meth:`from_real`, or from a simulator dataset
    with :meth:`from_observer_dataset` (to self-supervise on simulated windows,
    e.g. for a sim-vs-real ablation).

    Attributes:
        features: ``(N, T, n_feat)`` **normalised** inputs (measurements + feed).
        measurements: ``(N, T, n_channels)`` **raw** sensor values (``NaN`` =
            missing), in ``channel_names`` order.
        dt_days: measurement step [days] for the discrete physics residual.
        channel_names: sensor channels in ``measurements``.
        feat_mean, feat_std: the normalisation stats behind ``features`` (share
            these with the simulator dataset for a consistent sim→real scale).
        x_ref: nominal warmed 41-state (the observer's log-transform base).
        params: ADM1 parameters for the physics term (pass a
            :meth:`~pyadm1.core.adm1_torch.Adm1TorchParams.with_q_ad` snapshot to
            match the real operating point / feed).
        time: optional ``(T,)`` window times [days].
    """

    features: np.ndarray
    measurements: np.ndarray
    dt_days: float
    channel_names: list[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    x_ref: np.ndarray
    params: Adm1TorchParams
    time: np.ndarray | None = None

    @property
    def n_sequences(self) -> int:
        return int(self.features.shape[0])

    def normalize(self, raw: np.ndarray) -> np.ndarray:
        """Apply the stored normalisation to raw ``(..., n_feat)`` inputs."""
        return (raw - self.feat_mean) / self.feat_std

    @classmethod
    def from_real(
        cls,
        measurements: np.ndarray,
        feed: np.ndarray,
        *,
        dt_hours: float,
        params: Adm1TorchParams,
        x_ref: np.ndarray,
        channel_names: Sequence[str],
        feat_mean: np.ndarray | None = None,
        feat_std: np.ndarray | None = None,
        time: np.ndarray | None = None,
    ) -> MeasurementDataset:
        """Build from **real** logged plant data.

        Args:
            measurements: raw sensor values, ``(T, n_ch)`` for a single history
                or ``(N, T, n_ch)`` for several segments. ``NaN`` marks a gated /
                offline sensor and is masked out of the loss.
            feed: the known substrate feed, broadcast to the measurements' shape:
                a constant ``(n_feed,)``, a per-step ``(T, n_feed)``, or a full
                ``(N, T, n_feed)``.
            dt_hours: measurement interval [h].
            params: ADM1 parameters for the physics term (ideally feed-matched via
                ``params.with_q_ad(...)``).
            x_ref: nominal warmed 41-state.
            channel_names: sensor channels, matching ``measurements``' last axis.
            feat_mean, feat_std: normalisation stats. If omitted they are computed
                from this data (NaN-robust). **Pass the simulator dataset's stats
                here to keep sim and real on the same scale** for sim→real
                pre-training.
        """
        meas = np.asarray(measurements, dtype=float)
        if meas.ndim == 2:
            meas = meas[None, ...]
        if meas.ndim != 3:
            raise ValueError(
                f"measurements must be (T, n_ch) or (N, T, n_ch), got {meas.shape}."
            )
        n, t, n_ch = meas.shape
        if len(channel_names) != n_ch:
            raise ValueError(
                f"channel_names ({len(channel_names)}) != n_channels ({n_ch})."
            )

        feed = np.asarray(feed, dtype=float)
        if feed.ndim == 1:
            feed_seq = np.broadcast_to(feed, (n, t, feed.shape[0]))
        elif feed.ndim == 2:
            feed_seq = np.broadcast_to(feed[None, ...], (n, t, feed.shape[-1]))
        elif feed.ndim == 3:
            feed_seq = feed
        else:
            raise ValueError(
                f"feed must be (n_feed,), (T, n_feed) or (N, T, n_feed), got {feed.shape}."
            )

        raw = np.concatenate(
            [meas, feed_seq], axis=-1
        )  # (N, T, n_feat), may hold NaN in meas
        flat = raw.reshape(-1, raw.shape[-1])
        if feat_mean is None:
            feat_mean = np.nanmean(flat, axis=0)
        if feat_std is None:
            feat_std = np.nanstd(flat, axis=0) + 1e-8
        feat_mean = np.asarray(feat_mean, dtype=float)
        feat_std = np.asarray(feat_std, dtype=float)
        # NaN sensors cannot enter the recurrent input; map them to the
        # normalised mean (0). The loss still masks them on the target side.
        features = np.nan_to_num((raw - feat_mean) / feat_std, nan=0.0)

        return cls(
            features=features,
            measurements=meas,
            dt_days=float(dt_hours) / 24.0,
            channel_names=list(channel_names),
            feat_mean=feat_mean,
            feat_std=feat_std,
            x_ref=np.asarray(x_ref, dtype=float),
            params=params,
            time=None if time is None else np.asarray(time, dtype=float),
        )

    @classmethod
    def from_observer_dataset(
        cls,
        dataset: ObserverDataset,
        *,
        params: Adm1TorchParams | None = None,
    ) -> MeasurementDataset:
        """Derive a self-supervised dataset from a supervised simulator dataset.

        Uses the same windows but the raw ``measurements`` as targets instead of
        the (here discarded) true state — for a sim-vs-real ablation of the two
        pre-training objectives on identical scenarios.
        """
        if dataset.measurements is None or dataset.channel_names is None:
            raise ValueError(
                "ObserverDataset carries no raw measurements; regenerate it with the "
                "current generate_observer_dataset (it now stores them)."
            )
        dt_days = (
            float(np.diff(dataset.time).mean())
            if dataset.time is not None
            else 1.0 / 24.0
        )
        return cls(
            features=dataset.features,
            measurements=dataset.measurements,
            dt_days=dt_days,
            channel_names=list(dataset.channel_names),
            feat_mean=dataset.feat_mean,
            feat_std=dataset.feat_std,
            x_ref=dataset.x_ref,
            params=params if params is not None else dataset.params,
            time=dataset.time,
        )
