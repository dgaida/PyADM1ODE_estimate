"""High-level convenience API for the most common UKF setup.

The :func:`build_ukf` factory bundles the typical 6-step setup
(``adm1da_full_spec`` → ``ObservationModel`` → ``ADM1ProcessModel`` →
``UnscentedKalmanFilter`` → snapshot → reset) into a single call.
Sensors are picked from a small built-in catalog by name:

* ``"q_gas"``  - total biogas volumetric flow rate
* ``"q_ch4"``  - total methane volumetric flow rate
* ``"ph"``     - pH of the named digester
* ``"substrate_dose"`` - direct observation of every augmented
  substrate-input channel in the spec

A typical setup is then::

    from pyadm1ode_estimation.estimation import InputSpec, build_ukf
    from pyadm1ode_estimation.example_plants import build_multi_stage_plant

    plant = build_multi_stage_plant()

    ukf = build_ukf(
        plant,
        digester_id="primary",
        substrates=[
            InputSpec("maize_silage",   substrate_index=0, initial_flow=4.74),
            InputSpec("solid_manure",   substrate_index=1, initial_flow=13.70),
            InputSpec("chicken_litter", substrate_index=2, initial_flow=1.09),
            InputSpec("slurry",         substrate_index=3, initial_flow=3.68),
            InputSpec("cereal_grain",   substrate_index=4, initial_flow=0.20),
        ],
        sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
    )

    for t, y in measurement_stream:
        ukf.predict(dt=1.0 / 24.0)
        step = ukf.update(y, t=t)

For non-standard sensors or custom noise, pass ``ObservationChannel``
instances directly in the ``sensors`` list instead of name strings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Union

import numpy as np

from .observation_model import (
    ObservationChannel,
    ObservationModel,
    extract_q_ch4_total,
    extract_q_gas_total,
    make_state_extractor,
)
from .process_model import ADM1ProcessModel
from .specs import (
    InputSpec,
    KineticSpec,
    SensorQualityProfile,
    adm1da_full_spec,
)
from .state_vector import StateVectorSpec

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant  # type: ignore[import-not-found]

    from .filters.sr_ukf import UnscentedKalmanFilter


# Default per-sensor noise std. Calibrated for a typical agricultural-AD
# plant with online NDIR + pH probe + dosing scale.
_DEFAULT_NOISE_STD = {
    "q_gas": 10.0,  # m³/d biogas
    "q_ch4": 5.0,  # m³/d methane
    "ph": 0.05,  # pH units
    "substrate_dose": 0.05,  # 5 % relative on the augmented channel
}


def _make_ph_extractor(digester_id: str):
    """pH extractor for the named digester.

    Returns the raw pH. A non-finite value propagates so the filter drops
    the channel for that step instead of correcting with a fudged number
    (see :meth:`UnscentedKalmanFilter.update`).
    """

    def extractor(plant, x):
        return float(plant.components[digester_id].outputs_data.get("pH", float("nan")))

    return extractor


def _build_substrate_channels(
    spec: StateVectorSpec, substrate_dose_noise_relative: float
) -> List[ObservationChannel]:
    """One ``ObservationChannel`` per augmented input-flow channel in ``spec``.

    Each channel observes the augmented channel directly (``x[idx]``)
    and uses ``substrate_dose_noise_relative * initial_flow`` as the
    measurement std.
    """
    channels: List[ObservationChannel] = []
    for i, ch in enumerate(spec.channels):
        if ch.kind != "input_flow":
            continue
        nominal = max(abs(ch.initial), 1e-3)
        noise_std = max(substrate_dose_noise_relative * nominal, 1e-3)
        channels.append(
            ObservationChannel(
                name=f"Q_{ch.name}",
                extractor=make_state_extractor(i),
                noise_std=noise_std,
            )
        )
    return channels


def _resolve_sensors(
    sensors: Sequence[Union[str, ObservationChannel]],
    digester_id: str,
    spec: StateVectorSpec,
    noise_overrides: Optional[dict] = None,
) -> List[ObservationChannel]:
    """Translate sensor name strings into ``ObservationChannel`` instances.

    Items that are already ``ObservationChannel`` instances pass through
    unchanged, so users can mix the convenience catalog with custom
    channels for non-standard signals.
    """
    overrides = noise_overrides or {}
    out: List[ObservationChannel] = []
    for item in sensors:
        if isinstance(item, ObservationChannel):
            out.append(item)
            continue
        if not isinstance(item, str):
            raise TypeError(
                f"Sensor entries must be str or ObservationChannel, got "
                f"{type(item).__name__}"
            )
        key = item.lower()
        noise = overrides.get(key, _DEFAULT_NOISE_STD.get(key))
        if key == "q_gas":
            out.append(
                ObservationChannel(
                    name="Q_gas",
                    extractor=extract_q_gas_total,
                    noise_std=float(noise),
                )
            )
        elif key == "q_ch4":
            out.append(
                ObservationChannel(
                    name="Q_ch4",
                    extractor=extract_q_ch4_total,
                    noise_std=float(noise),
                )
            )
        elif key == "ph":
            out.append(
                ObservationChannel(
                    name="pH",
                    extractor=_make_ph_extractor(digester_id),
                    noise_std=float(noise),
                )
            )
        elif key == "substrate_dose":
            out.extend(_build_substrate_channels(spec, float(noise)))
        else:
            raise ValueError(
                f"Unknown sensor name '{item}'. Built-in names: "
                f"{sorted(_DEFAULT_NOISE_STD)}. Pass an ObservationChannel "
                f"instance for custom sensors."
            )
    return out


def build_ukf(
    plant: "BiogasPlant",
    digester_id: str,
    substrates: Sequence[InputSpec] = (),
    sensors: Sequence[Union[str, ObservationChannel]] = (
        "q_gas",
        "q_ch4",
        "ph",
        "substrate_dose",
    ),
    *,
    kinetic_overrides: Sequence[KineticSpec] = (),
    sensor_quality: Optional[SensorQualityProfile] = None,
    sensor_noise: Optional[dict] = None,
    initial_uncertainty_relative: float = 0.05,
    alpha: float = 1.0,
    beta: float = 2.0,
    kappa: float = 0.0,
    gamma_override: Optional[float] = None,
) -> "UnscentedKalmanFilter":
    """Bundle the full 41-state SR-UKF setup into a single call.

    The factory does, in order:

    1. Build the 41-channel spec via :func:`adm1da_full_spec` with the
       declared substrate inputs and (optional) kinetic overrides.
    2. Translate the ``sensors`` list into an ``ObservationModel`` via
       the built-in catalog (or accept ``ObservationChannel`` instances
       directly).
    3. Wrap the plant in an ``ADM1ProcessModel`` and snapshot it at the
       current state.
    4. Construct an ``UnscentedKalmanFilter``.
    5. Read the plant's current ADM1 state and the spec's input
       initials as ``x0``. Build a tight diagonal ``P0`` matching
       ``initial_uncertainty_relative``.
    6. ``ukf.reset(x0, P0)`` to set up the prior.

    The returned filter is ready for the online ``predict()`` /
    ``update()`` loop.

    Args:
        plant: A built :class:`pyadm1.BiogasPlant`.
        digester_id: Identifier of the main digester to estimate.
        substrates: List of :class:`InputSpec` declaring each substrate
            channel and its nominal flow.
        sensors: List of sensor name strings (from the built-in catalog)
            and/or :class:`ObservationChannel` instances for custom
            signals. Default is the Phase-1 sensor set: gas + methane
            + pH + every declared substrate dose.
        kinetic_overrides: Optional list of :class:`KineticSpec` to
            augment kinetic parameters into the state vector.
        sensor_quality: Optional :class:`SensorQualityProfile` for
            non-default per-block estimation quality.
        sensor_noise: Optional dict overriding per-name noise std for
            the built-in sensors, e.g. ``{"q_gas": 25.0}``.
        initial_uncertainty_relative: Relative ``σ`` per channel for
            the diagonal initial covariance ``P0``. Default 5 %.
        alpha, beta, kappa: UKF scaling parameters. The defaults
            ``(1.0, 2.0, 0.0)`` are the unscaled (Julier-Uhlmann)
            settings recommended for the strongly nonlinear ADM1
            kinetics.
        gamma_override: Override the canonical sigma-point radius
            ``γ = √(n + λ)``. ``gamma_override=1.0`` implements the
            Hellmann et al. 2024 (ECC) reduced-scaling trick that
            yielded the largest accuracy improvement on ADM1-R4-Core in
            their UKF benchmark (NRMSE_x 0.85 → 0.37). Default ``None``
            keeps the canonical Wan–Van der Merwe scaling.

    Returns:
        A ready-to-run :class:`UnscentedKalmanFilter`.
    """
    # Local import to avoid a circular dependency between
    # estimation/__init__ and the filter submodule at module load.
    from .filters.sr_ukf import UnscentedKalmanFilter

    if digester_id not in plant.components:
        raise KeyError(
            f"digester_id='{digester_id}' is not a component of the plant. "
            f"Available: {sorted(plant.components)}"
        )

    # ---- 1. Spec --------------------------------------------------------
    spec = adm1da_full_spec(
        digester_id=digester_id,
        substrate_inputs=substrates,
        kinetic_overrides=kinetic_overrides,
        sensor_quality=sensor_quality,
    )

    # ---- 2. Observation model -------------------------------------------
    obs_channels = _resolve_sensors(
        sensors=sensors,
        digester_id=digester_id,
        spec=spec,
        noise_overrides=sensor_noise,
    )
    if not obs_channels:
        raise ValueError(
            "No observation channels constructed - pass at least one sensor."
        )
    obs = ObservationModel(channels=obs_channels)

    # ---- 3. Process model -----------------------------------------------
    process = ADM1ProcessModel(plant, spec)
    process.snapshot()

    # ---- 4. UKF ---------------------------------------------------------
    ukf = UnscentedKalmanFilter(
        process,
        obs,
        spec,
        alpha=alpha,
        beta=beta,
        kappa=kappa,
        gamma_override=gamma_override,
    )

    # ---- 5. Initial state + covariance ----------------------------------
    x0 = spec.read_adm1_state(plant)
    aug_starts = 41
    for i, ch in enumerate(spec.channels[aug_starts:], start=aug_starts):
        x0[i] = ch.initial

    sigma = max(float(initial_uncertainty_relative), 1.0e-3)
    p0_diag = (sigma * (np.abs(x0) + 1.0e-6)) ** 2
    P0 = np.diag(p0_diag)

    # ---- 6. Reset -------------------------------------------------------
    ukf.reset(x0, P0)
    return ukf


def build_filter_components(
    plant: "BiogasPlant",
    digester_id: str,
    substrates: Sequence[InputSpec] = (),
    sensors: Sequence[Union[str, ObservationChannel]] = (
        "q_gas",
        "q_ch4",
        "ph",
        "substrate_dose",
    ),
    *,
    kinetic_overrides: Sequence[KineticSpec] = (),
    sensor_quality: Optional[SensorQualityProfile] = None,
    sensor_noise: Optional[dict] = None,
):
    """Build ``(process, obs, spec)`` without constructing a UKF.

    Mirrors steps 1–3 of :func:`build_ukf` (spec → obs → process model
    + initial snapshot) without going on to instantiate the filter
    itself. Useful when a caller needs the three building blocks for
    something other than the standard ``UnscentedKalmanFilter`` —
    notably :class:`ParallelUKF` workers, which rebuild their local
    process model and observation closures from scratch on each
    worker process.

    Returns:
        A 3-tuple ``(process, obs, spec)``. ``process`` is already
        snapshotted at the plant's current state.
    """
    if digester_id not in plant.components:
        raise KeyError(
            f"digester_id='{digester_id}' is not a component of the plant. "
            f"Available: {sorted(plant.components)}"
        )

    spec = adm1da_full_spec(
        digester_id=digester_id,
        substrate_inputs=substrates,
        kinetic_overrides=kinetic_overrides,
        sensor_quality=sensor_quality,
    )

    obs_channels = _resolve_sensors(
        sensors=sensors,
        digester_id=digester_id,
        spec=spec,
        noise_overrides=sensor_noise,
    )
    if not obs_channels:
        raise ValueError(
            "No observation channels constructed - pass at least one sensor."
        )
    obs = ObservationModel(channels=obs_channels)

    process = ADM1ProcessModel(plant, spec)
    process.snapshot()
    return process, obs, spec


__all__ = ["build_ukf", "build_filter_components"]
