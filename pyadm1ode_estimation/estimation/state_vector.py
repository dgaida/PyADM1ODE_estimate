"""State-vector specification: what the filter actually estimates.

Three kinds of channels are supported:

* ``"adm1"`` - a slot of the 41-element ADM1da state vector (see
  ``pyadm1.core.adm1`` module-level docstring for the index map).
  Pushed into ``digester.adm1_state`` before each simulation step,
  read back afterwards.
* ``"input_flow"`` - an augmented substrate-input rate in m³/d FM.
  Propagated by a random-walk (or OU) drift inside the filter. No
  direct interaction with the ADM1 state. Read by the process
  model to set ``digester.Q_substrates``.
* ``"kinetic_param"`` - an augmented kinetic-rate override (Phase 3).
  Written into ``adm1._kinetic`` before each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant


ChannelKind = Literal["adm1", "input_flow", "kinetic_param"]
DriftModel = Literal["random_walk", "ou", "constant"]


@dataclass
class StateChannel:
    """One slot in the augmented state vector.

    Attributes:
        name: Human-readable identifier (used in plots / logs).
        kind: Where the value is consumed by the process model.
            ``"adm1"`` requires ``adm1_index``. ``"input_flow"`` is
            applied via ``Q_substrates``. ``"kinetic_param"`` is
            applied via ``adm1._kinetic[name]``.
        adm1_index: Index in the 41-element ADM1da state vector,
            required when ``kind="adm1"``.
        input_substrate_index: Position in the plant's
            ``Q_substrates`` vector for ``kind="input_flow"``. The
            n-slot vector is laid out as
            ``[Q_total_substrate_1, …, Q_total_substrate_n]``, where
            ``n`` matches the schema substrate count.
        initial: Initial state estimate.
        initial_std: Initial standard deviation (used to build
            :math:`P_0`).
        process_noise_std: Standard deviation of the per-step process
            noise. The filter scales this by ``sqrt(dt)`` for the
            ``random_walk`` drift model.
        drift_model: Augmented dynamics. ``"random_walk"`` is the
            default; ``"ou"`` is an Ornstein-Uhlenbeck pull-back to
            ``ou_mean`` with rate ``ou_theta`` and is recommended for
            input flows that should not drift unbounded during long
            sensor-blind intervals; ``"constant"`` adds zero noise
            (used for diagnostics).
        ou_mean: Long-term mean for the OU process.
        ou_theta: Mean-reversion rate for the OU process [1/d].
        lower, upper: Hard bounds enforced after every predict /
            update step.
    """

    name: str
    kind: ChannelKind = "adm1"
    adm1_index: Optional[int] = None
    input_substrate_index: Optional[int] = None
    initial: float = 0.0
    initial_std: float = 0.1
    process_noise_std: float = 0.01
    drift_model: DriftModel = "random_walk"
    ou_mean: Optional[float] = None
    ou_theta: float = 1.0
    lower: float = 0.0
    upper: float = float("inf")

    def __post_init__(self) -> None:
        if self.kind == "adm1" and self.adm1_index is None:
            raise ValueError(
                f"Channel '{self.name}': adm1_index is required when kind='adm1'."
            )
        if self.kind == "input_flow" and self.input_substrate_index is None:
            raise ValueError(
                f"Channel '{self.name}': input_substrate_index is required when "
                f"kind='input_flow'."
            )
        if self.drift_model == "ou" and self.ou_mean is None:
            raise ValueError(
                f"Channel '{self.name}': ou_mean is required when " f"drift_model='ou'."
            )
        if self.initial_std <= 0:
            raise ValueError(f"Channel '{self.name}': initial_std must be > 0.")
        if self.process_noise_std < 0:
            raise ValueError(f"Channel '{self.name}': process_noise_std must be >= 0.")
        if self.lower > self.upper:
            raise ValueError(
                f"Channel '{self.name}': lower {self.lower} > upper {self.upper}."
            )


@dataclass
class StateVectorSpec:
    """A list of channels, ordered as they appear in the state vector.

    The order matters because the filter's covariance matrix and
    sigma-point construction use it. Reordering channels after the
    fact will silently break ``P``.
    """

    digester_id: str
    channels: List[StateChannel] = field(default_factory=list)

    # ---------------------------------------------------------------- meta
    def __len__(self) -> int:
        return len(self.channels)

    @property
    def names(self) -> List[str]:
        return [c.name for c in self.channels]

    def kind_indices(self, kind: ChannelKind) -> List[int]:
        return [i for i, c in enumerate(self.channels) if c.kind == kind]

    # ---------------------------------------------------------------- vectors
    def initial_mean(self) -> np.ndarray:
        return np.array([c.initial for c in self.channels], dtype=float)

    def initial_cov(self) -> np.ndarray:
        stds = np.array([c.initial_std for c in self.channels], dtype=float)
        return np.diag(stds**2)

    def process_noise_cov(self, dt: float) -> np.ndarray:
        """Discrete process-noise covariance for one ``dt``- step.

        Random-walk / constant channels contribute ``(σ_w · sqrt(dt))^2``.
        OU channels contribute the discrete equivalent
        ``σ²(1 - e^{-2θ·dt}) / (2θ)`` (Doob-Itō). The mean-reversion
        term itself is applied inside the process model, not here.
        """
        q_diag = np.zeros(len(self.channels))
        for i, c in enumerate(self.channels):
            if c.drift_model == "constant":
                q_diag[i] = 0.0
            elif c.drift_model == "ou":
                theta = max(c.ou_theta, 1e-9)
                q_diag[i] = (
                    (c.process_noise_std**2)
                    * (1.0 - np.exp(-2.0 * theta * dt))
                    / (2.0 * theta)
                )
            else:
                q_diag[i] = (c.process_noise_std**2) * dt
        return np.diag(q_diag)

    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lower = np.array([c.lower for c in self.channels], dtype=float)
        upper = np.array([c.upper for c in self.channels], dtype=float)
        return lower, upper

    def clip(self, x: np.ndarray) -> np.ndarray:
        """Hard-clip a state vector to the configured channel bounds."""
        lower, upper = self.bounds()
        return np.minimum(np.maximum(x, lower), upper)

    # ---------------------------------------------------------------- plant I/O
    def push_adm1_state(self, x: np.ndarray, plant: "BiogasPlant") -> None:
        """Write the ``"adm1"`` channels of ``x`` into the digester."""
        digester = plant.components[self.digester_id]
        state = list(digester.adm1_state)
        for i in self.kind_indices("adm1"):
            ch = self.channels[i]
            state[ch.adm1_index] = float(x[i])
        digester.adm1_state = state

    def read_adm1_state(self, plant: "BiogasPlant") -> np.ndarray:
        """Read the ``"adm1"`` channels back from the digester state."""
        digester = plant.components[self.digester_id]
        out = np.zeros(len(self.channels))
        # Preserve non-ADM1 channel values from the input.
        state = digester.adm1_state
        for i, ch in enumerate(self.channels):
            if ch.kind == "adm1":
                out[i] = float(state[ch.adm1_index])
        return out
