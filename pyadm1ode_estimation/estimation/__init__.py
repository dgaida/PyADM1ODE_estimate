"""State-estimation subpackage.

Provides a filter-agnostic structure for estimating biogas plant
states (ADM1da biological pools + augmented substrate inputs) from
the measurements declared in a :class:`PlantSchema`.

The underlying process model is **ADM1da** (Schlattmann 2011) as
implemented in ``pyadm1.core.adm1`` — a 41-state ODE with sub-fraction
disintegration (X_PS slow + X_PF fast, each in ch/pr/li), hydrolysable
intermediates (X_S_ch/pr/li), temperature-corrected kinetics, and
modified inhibition.

Layers
------
* :mod:`.base` — :class:`StateEstimator` protocol + :class:`EstimationStep`.
* :mod:`.state_vector` — :class:`StateChannel` / :class:`StateVectorSpec`
  declare *what* is being estimated (ADM1 indices 0-40, augmented
  inputs, augmented kinetic parameters).
* :mod:`.process_model` — :class:`ADM1ProcessModel` propagates a
  state vector through one pyadm1 time step.
* :mod:`.observation_model` — :class:`ObservationModel` maps the
  current plant state to the measurement channels declared in the
  schema, with optional gating for sparse observations. Built-in
  extractors target ``Q_gas``, ``Q_ch4``, ``Q_gas_consumed``, ``P_el``,
  ``P_th_used``; the pyadm1 model additionally exposes ``pH_l``,
  ``VFA``, ``TAC``, ``FOSTAC`` and ``AcvsPro`` which can be wired in
  via custom extractors once lab data becomes available.
* :mod:`.filters` — concrete implementations (UKF for Phase 1).
* :mod:`.twin` — helpers for synthetic-truth twin experiments.
"""

from .base import EstimationStep, StateEstimator
from .measurement_calendar import MeasurementCalendar, SampleRate
from .observation_model import ObservationChannel, ObservationModel
from .process_model import ADM1ProcessModel
from .specs import (
    InputSpec,
    KineticSpec,
    Quality,
    SensorQualityProfile,
    adm1da_full_spec,
)
from .state_vector import StateChannel, StateVectorSpec

__all__ = [
    "ADM1ProcessModel",
    "EstimationStep",
    "InputSpec",
    "KineticSpec",
    "MeasurementCalendar",
    "ObservationChannel",
    "ObservationModel",
    "Quality",
    "SampleRate",
    "SensorQualityProfile",
    "StateChannel",
    "StateEstimator",
    "StateVectorSpec",
    "adm1da_full_spec",
]
