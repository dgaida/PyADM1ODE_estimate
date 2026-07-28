"""Factories for fully populated ADM1da state-vector specifications.

The function :func:`adm1da_full_spec` builds a complete 41-state
StateVectorSpec with sensible defaults baked in from two sources:

* **Literature steady-state values** for the dissolved, particulate,
  biomass, charge-balance and gas-phase channels (Schlattmann 2011,
  Henze et al. / Batstone 2002, adapted to typical mesophilic
  agricultural-AD operating points).
* **Per-channel observability classification** from the subsystem
  analysis (see ``observability_experiment/results.md``): each state
  belongs to one of nine "blocks" with a default quality level, and
  the quality maps to a process-noise scaling, initial uncertainty,
  and drift-model choice. Plants with stronger or weaker sensor sets
  override the default via :class:`SensorQualityProfile`.

The factory also accepts optional ``substrate_inputs`` (augmented
input-flow channels) and ``kinetic_overrides`` (augmented kinetic
parameters), making the resulting spec ready for production UKF use.

Typical call::

    from pyadm1ode_estimation.estimation.specs import (
        adm1da_full_spec, InputSpec, SensorQualityProfile, Quality,
    )

    spec = adm1da_full_spec(
        digester_id="main_fermenter",
        substrate_inputs=[
            InputSpec("maize", substrate_index=0, initial_flow=5.0),
            InputSpec("slurry", substrate_index=1, initial_flow=12.0),
        ],
        sensor_quality=SensorQualityProfile(
            acidogenesis_biomass=Quality.MEDIUM,  # we have GC-FID
        ),
    )
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .state_vector import StateChannel, StateVectorSpec

# ---------------------------------------------------------------------------
# Observability quality classes
# ---------------------------------------------------------------------------


class Quality(str, Enum):
    """Sensor / observability quality class for a block of states.

    Quality drives per-channel process noise, initial uncertainty, and
    drift-model choice. Values are ordered roughly by decreasing
    Filter-correctability:
    ``STRONG > MEDIUM > WEAK > PSPF > OPEN_LOOP``.
    """

    STRONG = "strong"
    """Direct sensor innovation, every step. Filter posterior σ stays
    near sensor-σ. Used for the A+D fused subsystem (Variante II)."""

    MEDIUM = "medium"
    """Indirect via 1–2 Lie-derivative steps; observable in B substrates
    with FOS/TAC. Posterior σ ≈ 30 % of typical magnitude."""

    WEAK = "weak"
    """Observable only via deeper model coupling. Biomass states in B,
    hydrolysis sums in C. Posterior σ wide, but bounded by mass balance."""

    PSPF = "pspf"
    """Slow/fast disintegration split - structurally non-separable from
    aggregated lab measurements. Needs substrate characterization
    prior. OU drift to the literature mean acts as that prior."""

    OPEN_LOOP = "open_loop"
    """No direct innovation channel under the Phase-1 sensor set.
    Purely model-driven, OU-anchored to literature mean. Posterior σ
    grows along OU stationary distribution."""


# Per-quality scaling factors. The first column scales the process
# noise σ_w per √dt. The second scales the initial std. The third
# selects the drift model. The fourth gives an OU reversion rate
# (in 1/d) when drift is "ou".
_QUALITY_NOISE_FACTOR: dict[Quality, float] = {
    Quality.STRONG: 0.05,
    Quality.MEDIUM: 0.15,
    Quality.WEAK: 0.30,
    Quality.PSPF: 0.40,
    Quality.OPEN_LOOP: 0.40,
}

_QUALITY_INITIAL_STD_FACTOR: dict[Quality, float] = {
    Quality.STRONG: 0.20,
    Quality.MEDIUM: 0.30,
    Quality.WEAK: 0.50,
    Quality.PSPF: 0.60,
    Quality.OPEN_LOOP: 0.80,
}

_QUALITY_DRIFT_MODEL: dict[Quality, str] = {
    Quality.STRONG: "random_walk",
    Quality.MEDIUM: "random_walk",
    Quality.WEAK: "random_walk",
    Quality.PSPF: "ou",
    Quality.OPEN_LOOP: "ou",
}

_QUALITY_OU_THETA: dict[Quality, float] = {
    Quality.PSPF: 0.05,  # ~20-day reversion to substrate-prior mean
    Quality.OPEN_LOOP: 0.10,  # ~10-day reversion to literature mean
}


# ---------------------------------------------------------------------------
# ADM1da defaults - order matches the 41-state index layout in
# pyadm1/core/adm1.py.
# ---------------------------------------------------------------------------

# Each entry: (name, initial, lower, upper)
# initial values are Schlattmann-2011-style mesophilic agricultural-AD
# steady-state; per-channel initial_std and process_noise_std are
# derived from these times the Quality factors above.
_ADM1DA_DEFAULTS: dict[int, tuple] = {
    # ---- dissolved (kg COD/m³ for organics; kmol/m³ for S_co2, S_nh4)
    0: ("S_su", 0.012, 0.0, 10.0),
    1: ("S_aa", 0.005, 0.0, 10.0),
    2: ("S_fa", 0.10, 0.0, 10.0),
    # VFA upper bounds widened to cover severe overload / acidification: the
    # reachable-set ensemble (sigma_truth 0.30, dose up to 1.8x) reaches
    # S_bu≈8.5, S_pro≈20, S_ac≈38 kg COD/m³. The estimator must represent
    # extreme overload, not exclude it.
    3: ("S_va", 0.012, 0.0, 5.0),
    4: ("S_bu", 0.013, 0.0, 10.0),
    5: ("S_pro", 0.016, 0.0, 20.0),
    6: ("S_ac", 0.20, 0.0, 40.0),
    7: ("S_h2", 2.4e-7, 0.0, 1e-2),
    8: ("S_ch4", 0.055, 0.0, 2.0),
    9: ("S_co2", 0.0099, 0.0, 1.0),
    10: ("S_nh4", 0.13, 0.0, 1.0),
    11: ("S_I", 0.328, 0.0, 10.0),
    # ---- particulate (kg COD/m³)
    12: ("X_PS_ch", 5.0, 0.0, 100.0),
    13: ("X_PS_pr", 3.0, 0.0, 100.0),
    14: ("X_PS_li", 1.0, 0.0, 50.0),
    15: ("X_PF_ch", 1.0, 0.0, 50.0),
    16: ("X_PF_pr", 0.5, 0.0, 50.0),
    17: ("X_PF_li", 0.2, 0.0, 20.0),
    18: ("X_S_ch", 0.5, 0.0, 20.0),
    19: ("X_S_pr", 0.3, 0.0, 20.0),
    20: ("X_S_li", 0.1, 0.0, 10.0),
    21: ("X_I", 25.0, 0.0, 100.0),
    # ---- biomass (kg COD/m³)
    22: ("X_su", 0.42, 0.0, 5.0),
    23: ("X_aa", 1.18, 0.0, 5.0),
    24: ("X_fa", 0.24, 0.0, 5.0),
    25: ("X_c4", 0.43, 0.0, 5.0),
    26: ("X_pro", 0.14, 0.0, 5.0),
    27: ("X_ac", 0.76, 0.0, 5.0),
    28: ("X_h2", 0.32, 0.0, 5.0),
    # ---- charge balance: S_cation/S_anion/S_hco3/S_nh3 in kmol/m³, but the VFA
    # ions S_va/bu/pro/ac_ion are in kg COD/m³ (same units as the total VFAs in
    # pyadm1 core/adm1.py) with 0 ≤ S_xx_ion ≤ S_xx. Their bounds therefore equal
    # the corresponding total-VFA bounds; initials track the (near-fully ionized)
    # totals. (Earlier kmol-style ~0.1 bounds/1e-5 initials were a units bug that
    # clipped the acetate ion ~40× even at steady state.)
    29: ("S_cation", 0.04, 0.0, 1.0),
    30: ("S_anion", 0.02, 0.0, 1.0),
    31: ("S_va_ion", 0.012, 0.0, 5.0),
    32: ("S_bu_ion", 0.013, 0.0, 10.0),
    33: ("S_pro_ion", 0.016, 0.0, 20.0),
    34: ("S_ac_ion", 0.20, 0.0, 40.0),
    35: ("S_hco3", 0.014, 0.0, 0.5),
    36: ("S_nh3", 4.1e-3, 0.0, 0.1),
    # ---- gas-phase partial pressures (bar)
    37: ("p_gas_h2", 1.6e-5, 0.0, 0.01),
    38: ("p_gas_ch4", 0.65, 0.0, 5.0),
    39: ("p_gas_co2", 0.33, 0.0, 5.0),
    40: ("pTOTAL", 1.013, 0.5, 5.0),
}

assert len(_ADM1DA_DEFAULTS) == 41, "ADM1da has exactly 41 states."


# ---------------------------------------------------------------------------
# Block assignment per state index - from the subsystem analysis.
# ---------------------------------------------------------------------------

_STATE_BLOCKS: dict[int, str] = {
    # methanogenesis (A subsystem core + gas-phase pressures)
    6: "methanogenesis",
    7: "methanogenesis",
    8: "methanogenesis",
    9: "methanogenesis",
    27: "methanogenesis",
    28: "methanogenesis",
    37: "methanogenesis",
    38: "methanogenesis",
    39: "methanogenesis",
    40: "methanogenesis",
    # charge balance / pH (D subsystem)
    29: "charge_balance",
    30: "charge_balance",
    31: "charge_balance",
    32: "charge_balance",
    33: "charge_balance",
    34: "charge_balance",
    35: "charge_balance",
    36: "charge_balance",
    # acidogenesis substrates (B subsystem, observable subset)
    0: "acidogenesis_substrates",
    1: "acidogenesis_substrates",
    3: "acidogenesis_substrates",
    4: "acidogenesis_substrates",
    5: "acidogenesis_substrates",
    # acidogenesis biomass (B subsystem, weakly observable remainder)
    22: "acidogenesis_biomass",
    23: "acidogenesis_biomass",
    25: "acidogenesis_biomass",
    26: "acidogenesis_biomass",
    # hydrolysis sums (C subsystem hydrolyzable pools)
    18: "hydrolysis_sums",
    19: "hydrolysis_sums",
    20: "hydrolysis_sums",
    # disintegration PS/PF split (structurally non-separable)
    12: "disintegration_split",
    13: "disintegration_split",
    14: "disintegration_split",
    15: "disintegration_split",
    16: "disintegration_split",
    17: "disintegration_split",
    # nitrogen (E subsystem)
    10: "nitrogen",
    # inerts (no direct sensor)
    11: "inerts",
    21: "inerts",
    # FA block (decoupled from VFA-sum; trackable open-loop via A only)
    2: "fa_block",
    24: "fa_block",
}

assert set(_STATE_BLOCKS.keys()) == set(
    range(41)
), "Every ADM1 index must belong to exactly one block."


#: Public read-only view of the per-index block assignment (index → block).
STATE_BLOCKS: dict[int, str] = dict(_STATE_BLOCKS)

#: Public block → sorted ADM1 indices map. Use to select whole observability
#: blocks for a reduced state vector, e.g.
#: ``BLOCK_INDICES["methanogenesis"] + BLOCK_INDICES["charge_balance"]``
#: for the literature's provably-observable "A+D fused" core.
BLOCK_INDICES: dict[str, list[int]] = {}
for _i in range(41):
    BLOCK_INDICES.setdefault(_STATE_BLOCKS[_i], []).append(_i)


# ---------------------------------------------------------------------------
# Sensor quality profile (per-block defaults)
# ---------------------------------------------------------------------------


@dataclass
class SensorQualityProfile:
    """Quality level for each ADM1da block.

    Defaults reflect the Phase-1 sensor set
    (``Q_gas`` + CH4/CO2 NDIR + pH online + FOS/TAC) as verified by the
    subsystem analysis. Override individual blocks to reflect a plant
    with stronger or weaker sensors.

    Example: a plant with online GC-FID for individual VFAs (which
    closes subsystem B fully - see ``results.md``) lifts
    ``acidogenesis_biomass`` from ``WEAK`` to ``MEDIUM``.
    """

    methanogenesis: Quality = Quality.STRONG
    charge_balance: Quality = Quality.STRONG
    acidogenesis_substrates: Quality = Quality.MEDIUM
    acidogenesis_biomass: Quality = Quality.WEAK
    hydrolysis_sums: Quality = Quality.WEAK
    disintegration_split: Quality = Quality.PSPF
    nitrogen: Quality = Quality.OPEN_LOOP
    inerts: Quality = Quality.OPEN_LOOP
    fa_block: Quality = Quality.OPEN_LOOP

    def get(self, block: str) -> Quality:
        return getattr(self, block)


# ---------------------------------------------------------------------------
# Augmentation specs
# ---------------------------------------------------------------------------


@dataclass
class InputSpec:
    """Substrate-feed channel to estimate as an augmented state.

    Appended to the spec after the 41 ADM1 channels. Use one
    InputSpec per substrate the UKF should track (typically 2-4 for
    agricultural plants).

    Attributes:
        name: Identifier (e.g. ``"maize"``).
        substrate_index: Position in the plant's
            ``Q_substrates`` vector (the same convention as
            ``StateChannel.input_substrate_index``).
        initial_flow: Initial estimate of the feed rate [m³/d].
        initial_std: Prior standard deviation [m³/d]. Defaults to
            half the initial flow.
        process_noise_std: Per-day random-walk noise std [m³/d].
            Defaults to 20 % of the initial flow.
        lower, upper: Hard bounds [m³/d].
        drift_model: ``"random_walk"`` (default) or ``"ou"``. OU is
            recommended when the substrate feed is expected to be
            stationary over long sensor-blind intervals.
        ou_mean, ou_theta: Required when ``drift_model="ou"``.
    """

    name: str
    substrate_index: int
    initial_flow: float
    initial_std: float | None = None
    process_noise_std: float | None = None
    lower: float = 0.0
    upper: float = 100.0
    drift_model: str = "random_walk"
    ou_mean: float | None = None
    ou_theta: float = 0.1


@dataclass
class KineticSpec:
    """Kinetic-parameter override to estimate as an augmented state.

    Phase-3 feature: lets the UKF adapt selected kinetic rates online.
    The plant must expose ``digester.adm1._kinetic[name]`` for this
    channel to be writable.

    Attributes:
        name: Kinetic parameter name (must match ``_kinetic`` key).
        initial: Initial estimate.
        initial_std: Prior std.
        process_noise_std: Per-day noise std (small - kinetic params
            change slowly).
        lower, upper: Hard bounds.
        drift_model: Default ``"ou"`` because kinetic parameters
            should not drift unbounded; reverts to literature value.
        ou_mean: Long-term mean (default: ``initial``).
        ou_theta: Reversion rate [1/d]. Default 0.01 (~100-day decay).
    """

    name: str
    initial: float
    initial_std: float
    process_noise_std: float
    lower: float = 0.0
    upper: float = 1.0e6
    drift_model: str = "ou"
    ou_mean: float | None = None
    ou_theta: float = 0.01


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _adm1_channel(
    idx: int, quality: Quality, process_noise_scale: float = 1.0
) -> StateChannel:
    """Build one ADM1 channel from its index and quality assignment.

    ``process_noise_scale`` multiplies the per-channel process-noise std
    (default ``1.0`` keeps the quality-derived value). Values ``> 1``
    make the filter trust the *model* less and the *measurements* more:
    a larger ``σ_w`` widens the predicted covariance, so the sigma-point
    spread — and therefore the Kalman gain on every innovation — grows.
    Use it when the model's prediction (e.g. gas production) is known to
    be worse than the sensor.
    """
    name, initial, lower, upper = _ADM1DA_DEFAULTS[idx]

    # Scale noise / std by a characteristic magnitude. We don't want
    # to use ``abs(initial)`` directly because some states have
    # initial ≈ 0 (e.g. dissociated ions): in that case we'd collapse
    # the noise to zero. Use a floor of (upper / 100), which gives a
    # sensible reference even for near-zero initials.
    magnitude = max(abs(initial), upper / 100.0, 1e-9)
    initial_std = _QUALITY_INITIAL_STD_FACTOR[quality] * magnitude
    process_noise_std = _QUALITY_NOISE_FACTOR[quality] * magnitude * process_noise_scale
    drift_model = _QUALITY_DRIFT_MODEL[quality]

    kwargs = {
        "name": name,
        "kind": "adm1",
        "adm1_index": idx,
        "initial": initial,
        "initial_std": initial_std,
        "process_noise_std": process_noise_std,
        "lower": lower,
        "upper": upper,
        "drift_model": drift_model,
    }
    if drift_model == "ou":
        kwargs["ou_mean"] = initial
        kwargs["ou_theta"] = _QUALITY_OU_THETA[quality]
    return StateChannel(**kwargs)


def _input_channel(spec: InputSpec) -> StateChannel:
    initial_std = (
        spec.initial_std
        if spec.initial_std is not None
        else max(0.5 * abs(spec.initial_flow), 0.1)
    )
    process_noise_std = (
        spec.process_noise_std
        if spec.process_noise_std is not None
        else max(0.2 * abs(spec.initial_flow), 0.05)
    )
    kwargs = {
        "name": spec.name,
        "kind": "input_flow",
        "input_substrate_index": spec.substrate_index,
        "initial": spec.initial_flow,
        "initial_std": initial_std,
        "process_noise_std": process_noise_std,
        "lower": spec.lower,
        "upper": spec.upper,
        "drift_model": spec.drift_model,
    }
    if spec.drift_model == "ou":
        kwargs["ou_mean"] = (
            spec.ou_mean if spec.ou_mean is not None else spec.initial_flow
        )
        kwargs["ou_theta"] = spec.ou_theta
    return StateChannel(**kwargs)


def _kinetic_channel(spec: KineticSpec) -> StateChannel:
    kwargs = {
        "name": spec.name,
        "kind": "kinetic_param",
        "initial": spec.initial,
        "initial_std": spec.initial_std,
        "process_noise_std": spec.process_noise_std,
        "lower": spec.lower,
        "upper": spec.upper,
        "drift_model": spec.drift_model,
    }
    if spec.drift_model == "ou":
        kwargs["ou_mean"] = spec.ou_mean if spec.ou_mean is not None else spec.initial
        kwargs["ou_theta"] = spec.ou_theta
    return StateChannel(**kwargs)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def adm1da_full_spec(
    digester_id: str,
    *,
    substrate_inputs: Sequence[InputSpec] = (),
    kinetic_overrides: Sequence[KineticSpec] = (),
    sensor_quality: SensorQualityProfile | None = None,
    process_noise_scale: float = 1.0,
) -> StateVectorSpec:
    """Build a complete 41-state ADM1da StateVectorSpec.

    Always includes all 41 ADM1 channels. Augments with optional
    substrate-input channels and kinetic-parameter channels (in that
    order, after the ADM1 block).

    Args:
        digester_id: Component ID of the main digester in the
            ``BiogasPlant`` the filter will be attached to.
        substrate_inputs: Substrate channels to estimate as
            augmented states. Empty by default.
        kinetic_overrides: Kinetic parameters to estimate. Empty by
            default.
        sensor_quality: Per-block quality profile. If ``None``, the
            Phase-1-sensor-set defaults from
            :class:`SensorQualityProfile` apply.
        process_noise_scale: Global multiplier on every ADM1 channel's
            process-noise std (default ``1.0``). ``> 1`` makes the
            filter trust the model less and the measurements more (see
            :func:`_adm1_channel`). Only the ADM1 states are scaled; the
            augmented input/kinetic channels keep their own tuning.

    Returns:
        A :class:`StateVectorSpec` with at least 41 channels
        (more if augmentations are provided), ready to construct a
        :class:`UnscentedKalmanFilter`.
    """
    profile = sensor_quality or SensorQualityProfile()
    channels: list[StateChannel] = []

    # 41 ADM1 channels, in index order.
    for idx in range(41):
        block = _STATE_BLOCKS[idx]
        quality = profile.get(block)
        channels.append(_adm1_channel(idx, quality, process_noise_scale))

    # Substrate inputs.
    for inp in substrate_inputs:
        channels.append(_input_channel(inp))

    # Kinetic overrides.
    for kin in kinetic_overrides:
        channels.append(_kinetic_channel(kin))

    return StateVectorSpec(digester_id=digester_id, channels=channels)


def adm1da_reduced_spec(
    digester_id: str,
    adm1_indices: Sequence[int],
    *,
    substrate_inputs: Sequence[InputSpec] = (),
    kinetic_overrides: Sequence[KineticSpec] = (),
    sensor_quality: SensorQualityProfile | None = None,
    process_noise_scale: float = 1.0,
) -> StateVectorSpec:
    """Build a StateVectorSpec that estimates only a SUBSET of the 41 ADM1
    states (plus the usual augmented inputs / kinetics).

    ADM1 indices NOT listed are dropped from the state vector entirely:
    the filter neither corrects them nor tracks their covariance — they
    keep whatever value the plant model propagates. This is the
    literature's "propagate open-loop" treatment for structurally
    non-observable states (see ``docs/observability``); the prime use
    case is the provably-observable "A+D fused" core (``methanogenesis``
    + ``charge_balance``, 18 states).

    The kept channels reuse the exact per-block quality, initials, bounds,
    and drift model of :func:`adm1da_full_spec` — only the membership of
    the state vector changes, not the per-channel tuning.

    Args:
        adm1_indices: The ADM1 state indices (0..40) to estimate.
            Normalised to ascending + de-duplicated so the covariance
            layout is deterministic regardless of call order. Select
            whole observability blocks via :data:`BLOCK_INDICES`.
        substrate_inputs, kinetic_overrides, sensor_quality,
        process_noise_scale: identical to :func:`adm1da_full_spec`.

    Raises:
        ValueError: if ``adm1_indices`` is empty or any index is outside
            ``[0, 40]``.
    """
    idx_set = sorted({int(i) for i in adm1_indices})
    if not idx_set:
        raise ValueError("adm1_indices must contain at least one ADM1 index.")
    if idx_set[0] < 0 or idx_set[-1] > 40:
        raise ValueError(
            f"adm1_indices must lie in [0, 40]; got min={idx_set[0]}, "
            f"max={idx_set[-1]}."
        )

    profile = sensor_quality or SensorQualityProfile()
    channels: list[StateChannel] = []
    for idx in idx_set:
        channels.append(
            _adm1_channel(idx, profile.get(_STATE_BLOCKS[idx]), process_noise_scale)
        )
    for inp in substrate_inputs:
        channels.append(_input_channel(inp))
    for kin in kinetic_overrides:
        channels.append(_kinetic_channel(kin))

    return StateVectorSpec(digester_id=digester_id, channels=channels)


__all__ = [
    "BLOCK_INDICES",
    "STATE_BLOCKS",
    "InputSpec",
    "KineticSpec",
    "Quality",
    "SensorQualityProfile",
    "adm1da_full_spec",
    "adm1da_reduced_spec",
]
