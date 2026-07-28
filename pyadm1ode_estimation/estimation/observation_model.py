"""Observation model - extracts predicted measurements from the plant.

A channel is a triple

  - ``extractor`` - pure function ``h_i(plant, x)`` returning the
    prediction for one measurement.
  - ``noise_std`` - measurement standard deviation.
  - ``gate`` - optional time-varying flag deciding whether the
    channel is observable at the current step.

Built-in extractors cover the common SCADA signals of an agricultural
biogas plant (total Q_gas, CHP electrical / thermal output, biogas
consumption, gas-storage fill). Custom extractors can be passed for
one-off sensors.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant


Extractor = Callable[["BiogasPlant", np.ndarray], float]

_SEP_RE = re.compile(r"[\s_\-]+")


def _normalize_key(key: str) -> str:
    """Case- and separator-insensitive form of a channel/measurement key.

    ``"Q_gas"``, ``"q gas"`` and ``"qgas"`` all collapse to ``"qgas"``.
    """
    return _SEP_RE.sub("", str(key).strip().lower())


def _loose_key(key: str) -> str:
    """Like :func:`_normalize_key` but also drops a leading ``q`` flow
    prefix, so ``"Q_maize_silage"`` matches the bare substrate name
    ``"maize_silage"`` and ``"Q_gas"`` matches ``"gas"``."""
    norm = _normalize_key(key)
    return norm[1:] if norm.startswith("q") and len(norm) > 1 else norm


@dataclass
class ObservationChannel:
    """A single measurement channel.

    Attributes:
        name: Channel identifier. Should match the column in the
            measurement DataFrame fed to the filter.
        extractor: ``h(plant, x) -> float`` predicting this channel
            from the current plant outputs and the augmented state.
            Built-in shortcuts in :func:`built_in_extractor`.
        noise_std: Measurement standard deviation (same units as the
            extractor output).
        gate_column: Optional column name in the measurement frame.
            When set, the channel is only used at steps where
            ``gate_predicate(value)`` is true.
        gate_predicate: How to interpret the gate column.
            ``"truthy"`` accepts non-zero / non-NaN, ``"finite"``
            accepts non-NaN values only.
    """

    name: str
    extractor: Extractor
    noise_std: float
    gate_column: str | None = None
    gate_predicate: str = "truthy"

    def is_active(self, gate_value: float | None) -> bool:
        if self.gate_column is None:
            return True
        if gate_value is None:
            return False
        if isinstance(gate_value, float) and np.isnan(gate_value):
            return False
        if self.gate_predicate == "truthy":
            return bool(gate_value)
        if self.gate_predicate == "finite":
            return bool(np.isfinite(gate_value))
        raise ValueError(
            f"Unknown gate_predicate '{self.gate_predicate}' for channel "
            f"'{self.name}'."
        )


class ObservationModel:
    """A bag of :class:`ObservationChannel` instances.

    The filter calls :meth:`predict_active` with the current plant
    and gate values. The method returns the channel names plus the
    predicted vector + measurement-noise diagonal in matching order.
    """

    def __init__(self, channels: list[ObservationChannel]):
        names = [c.name for c in channels]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate channel names in ObservationModel.")
        self.channels = list(channels)

    def __len__(self) -> int:
        return len(self.channels)

    def match_measurements(self, y: dict[str, float]) -> dict[str, float]:
        """Resolve a user measurement dict onto canonical channel names.

        Matching is forgiving: case- and separator-insensitive, and the
        ``Q_`` flow prefix is optional. So ``"q_gas"``, ``"Q_gas"`` and
        ``"gas"`` all map to channel ``"Q_gas"``, and ``"maize_silage"``
        maps to ``"Q_maize_silage"``. A key is only assigned when the
        match is *unambiguous*, ambiguous keys (e.g. one value that could
        belong to two channels) and keys matching no channel are ignored,
        as are non-finite values.

        Returns a new dict keyed by the exact ``ObservationChannel.name``.
        """
        norm_to_keys: dict[str, list[str]] = {}
        loose_to_keys: dict[str, list[str]] = {}
        for k in y:
            norm_to_keys.setdefault(_normalize_key(k), []).append(k)
            loose_to_keys.setdefault(_loose_key(k), []).append(k)

        # A loose channel form is only usable if no other channel shares it.
        chan_loose: dict[str, list[str]] = {}
        for c in self.channels:
            chan_loose.setdefault(_loose_key(c.name), []).append(c.name)

        resolved: dict[str, float] = {}
        for c in self.channels:
            cand = norm_to_keys.get(_normalize_key(c.name))
            if cand is None:
                loose = _loose_key(c.name)
                if len(chan_loose[loose]) == 1:
                    cand = loose_to_keys.get(loose)
            if not cand or len(cand) != 1:
                continue  # unmatched or ambiguous → skip this channel
            val = y[cand[0]]
            if np.isfinite(val):
                resolved[c.name] = float(val)
        return resolved

    def active_channels(
        self, gate_values: dict[str, float]
    ) -> list[ObservationChannel]:
        return [
            c
            for c in self.channels
            if c.is_active(gate_values.get(c.gate_column) if c.gate_column else None)
        ]

    def predict(
        self,
        plant: BiogasPlant,
        x: np.ndarray,
        active: list[ObservationChannel] | None = None,
    ) -> np.ndarray:
        """Evaluate ``h_i(plant, x)`` for every active channel.

        Returns a ``(len(active),)`` vector in the order of ``active``.
        """
        chans = active if active is not None else self.channels
        return np.array([float(c.extractor(plant, x)) for c in chans], dtype=float)

    def R(self, active: list[ObservationChannel] | None = None) -> np.ndarray:
        chans = active if active is not None else self.channels
        return np.diag([c.noise_std**2 for c in chans])


# --------------------------------------------------------------------------
# Built-in extractors for the standard channels of an agricultural biogas plant.
# --------------------------------------------------------------------------


def _digester_iter(plant: BiogasPlant):
    for comp in plant.components.values():
        if comp.component_type.value == "digester":
            yield comp


def _chp_iter(plant: BiogasPlant):
    for comp in plant.components.values():
        if comp.component_type.value == "chp":
            yield comp


def _heating_iter(plant: BiogasPlant):
    for comp in plant.components.values():
        if comp.component_type.value == "heating":
            yield comp


def extract_q_gas_total(plant: BiogasPlant, x: np.ndarray) -> float:
    return float(sum(d.outputs_data.get("Q_gas", 0.0) for d in _digester_iter(plant)))


def extract_q_ch4_total(plant: BiogasPlant, x: np.ndarray) -> float:
    return float(sum(d.outputs_data.get("Q_ch4", 0.0) for d in _digester_iter(plant)))


def extract_q_gas_consumed_total(plant: BiogasPlant, x: np.ndarray) -> float:
    return float(
        sum(c.outputs_data.get("Q_gas_consumed", 0.0) for c in _chp_iter(plant))
    )


def extract_p_el_total(plant: BiogasPlant, x: np.ndarray) -> float:
    return float(sum(c.outputs_data.get("P_el", 0.0) for c in _chp_iter(plant)))


def extract_p_th_used_total(plant: BiogasPlant, x: np.ndarray) -> float:
    return float(
        sum(h.outputs_data.get("P_th_used", 0.0) for h in _heating_iter(plant))
    )


def make_q_gas_extractor(digester_id: str) -> Extractor:
    """Biogas volumetric flow of a *single* named digester.

    Unlike :func:`extract_q_gas_total`, which sums ``Q_gas`` over *every*
    digester in the plant, this reads only ``digester_id``'s own
    production. Use it when the filter estimates one stage of a
    multi-stage cascade: the whole-plant total mixes in gas from the
    downstream stages (post-fermenter, digestate storage) that the
    filter does not estimate, so feeding the total into the gas
    innovation biases the estimated stage. A stage-scoped meter keeps
    the measurement consistent with the estimated state.

    Returns ``NaN`` when the digester has no ``Q_gas`` output yet, so the
    UKF drops the channel for that step instead of correcting with a
    fudged number (see :meth:`UnscentedKalmanFilter.update`).
    """

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        comp = plant.components[digester_id]
        return float(comp.outputs_data.get("Q_gas", float("nan")))

    return extractor


def make_q_ch4_extractor(digester_id: str) -> Extractor:
    """Methane volumetric flow of a *single* named digester.

    Per-stage counterpart of :func:`extract_q_ch4_total`; see
    :func:`make_q_gas_extractor` for the multi-stage rationale.
    """

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        comp = plant.components[digester_id]
        return float(comp.outputs_data.get("Q_ch4", float("nan")))

    return extractor


def make_q_co2_extractor(digester_id: str) -> Extractor:
    """CO2 volumetric flow of a *single* named digester (``Q_co2``).

    Per-stage counterpart to the gas channels; pairs with the NDIR gas
    analyser's CO2 reading. Returns ``NaN`` when unavailable so the UKF
    drops the channel for that step.
    """

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        comp = plant.components[digester_id]
        return float(comp.outputs_data.get("Q_co2", float("nan")))

    return extractor


def make_ts_extractor(digester_id: str) -> Extractor:
    """Total solids [% mass] of a single named digester (``TS``).

    Reads the digester's ``TS`` indicator — the online microwave total-solids
    reading of a Proline Teqwave MW 300. Returns ``NaN`` when unavailable so the
    UKF drops the channel for that step.
    """

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        comp = plant.components[digester_id]
        return float(comp.outputs_data.get("TS", float("nan")))

    return extractor


def make_vfa_extractor(digester_id: str) -> Extractor:
    """Total VFA concentration [g HAc-eq/L] of a single named digester.

    Reads the digester's ``VFA`` indicator (sum of S_va/S_bu/S_pro/S_ac as
    acetate equivalent — the FOS value of a Nordmann titration). Typically a
    sparse lab/titration measurement (e.g. every 12 h), so it is usually
    gated rather than fed every step.
    """

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        comp = plant.components[digester_id]
        return float(comp.outputs_data.get("VFA", float("nan")))

    return extractor


def make_state_extractor(channel_index: int) -> Extractor:
    """Identity extractor - returns ``x[channel_index]``.

    Used for direct observations of augmented input states (e.g.
    ``hopper_dose`` when the load-cell sees a negative ΔW).
    """

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        return float(x[channel_index])

    return extractor


def make_stored_volume_extractor(digester_id: str) -> Extractor:
    """Return the fractional fill of the named digester's gas dome."""

    def extractor(plant: BiogasPlant, x: np.ndarray) -> float:
        comp = plant.components[digester_id]
        gs = comp.outputs_data.get("gas_storage", {}) or {}
        vol = gs.get("stored_volume_m3", float("nan"))
        cap = float(getattr(comp, "V_gas", float("nan")))
        if not cap or cap <= 0 or not np.isfinite(vol):
            return float("nan")
        return float(vol / cap)

    return extractor


BUILT_IN_EXTRACTORS: dict[str, Extractor] = {
    "Q_gas": extract_q_gas_total,
    "Q_ch4": extract_q_ch4_total,
    "Q_gas_consumed": extract_q_gas_consumed_total,
    "P_el": extract_p_el_total,
    "P_th_used": extract_p_th_used_total,
}
