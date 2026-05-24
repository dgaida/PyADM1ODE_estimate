"""ADM1 process model — propagates one state vector through one step.

Wraps a :class:`pyadm1.BiogasPlant` so the filter can treat
state propagation as a deterministic function ``x' = f(x, dt)``.

Responsibilities
----------------
* Push the ``"adm1"`` channels of the state into the digester's
  41-element ADM1 state.
* Push the ``"input_flow"`` channels into the digester's
  ``Q_substrates`` slot at the position declared by
  ``input_substrate_index``.
* Push the ``"kinetic_param"`` channels into ``adm1._kinetic`` (the
  same channel pyadm1 actually reads at every ODE step).
* Run ``plant.step(dt)``.
* Read the ``"adm1"`` channels back from the digester; pass the
  augmented channels through unchanged so the filter's random-walk
  noise term handles their drift.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np

from .state_vector import StateVectorSpec

if TYPE_CHECKING:
    from pyadm1 import BiogasPlant


class ADM1ProcessModel:
    """Deterministic propagator ``x(t+dt) = f(x(t), dt)``.

    Args:
        plant: A built :class:`pyadm1.BiogasPlant` instance. The
            estimator owns the simulation state of this plant: the
            process model writes the state in, steps it, reads the
            state out, and never restores. If you want twin-style
            sweeps with shared initial conditions, use deepcopied
            plants.
        spec: Declares which slots of the state vector map to ADM1
            indices, ``Q_substrates`` positions and kinetic params.
    """

    def __init__(self, plant: "BiogasPlant", spec: StateVectorSpec):
        if spec.digester_id not in plant.components:
            raise KeyError(
                f"StateVectorSpec.digester_id='{spec.digester_id}' is not a "
                f"component of the plant ({sorted(plant.components)})."
            )
        self.plant = plant
        self.spec = spec
        self._baseline: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Baseline snapshot / restore
    # ------------------------------------------------------------------
    # The UKF needs every sigma point to start from the *same* full
    # plant state. Without that, propagating sigma_1 inherits whatever
    # the previous sigma_0 step left in the 33 ADM1 components we are
    # *not* estimating (and in the gas-storage volume, the CHP load
    # factor, etc.), and the sample covariance becomes nonsense.
    #
    # ``snapshot`` records every mutable bit of plant state that
    # ``step`` touches; ``restore`` writes it back. Both calls take
    # microseconds — much cheaper than ``copy.deepcopy(plant)``, which
    # walks every numpy buffer in pyadm1.
    def snapshot(self) -> None:
        digester = self.plant.components[self.spec.digester_id]
        snap: Dict[str, Any] = {
            "adm1_state": list(digester.adm1_state),
            "Q_substrates": list(getattr(digester, "Q_substrates", []) or []),
            "sim_time": float(self.plant.simulation_time),
        }
        # Gas storage state across every digester, if present.
        storage_snap: Dict[str, Dict[str, Any]] = {}
        for cid, comp in self.plant.components.items():
            gs = getattr(comp, "gas_storage", None)
            if gs is None:
                continue
            storage_snap[cid] = dict(getattr(gs, "outputs_data", {}) or {})
            for attr in ("stored_volume_m3", "_stored_volume_m3"):
                if hasattr(gs, attr):
                    storage_snap[cid][f"__attr__{attr}"] = getattr(gs, attr)
        snap["gas_storage"] = storage_snap
        self._baseline = snap

    def restore(self) -> None:
        if not self._baseline:
            return
        digester = self.plant.components[self.spec.digester_id]
        digester.adm1_state = list(self._baseline["adm1_state"])
        if self._baseline["Q_substrates"]:
            digester.Q_substrates = list(self._baseline["Q_substrates"])
        self.plant.simulation_time = float(self._baseline["sim_time"])
        for cid, snap in self._baseline.get("gas_storage", {}).items():
            comp = self.plant.components.get(cid)
            gs = getattr(comp, "gas_storage", None) if comp else None
            if gs is None:
                continue
            attr_updates = {k: v for k, v in snap.items() if k.startswith("__attr__")}
            outputs_part = {
                k: v for k, v in snap.items() if not k.startswith("__attr__")
            }
            try:
                gs.outputs_data = dict(outputs_part)
            except Exception:  # noqa: BLE001
                pass
            for key, val in attr_updates.items():
                attr_name = key.removeprefix("__attr__")
                try:
                    setattr(gs, attr_name, val)
                except Exception:  # noqa: BLE001
                    pass

    # ------------------------------------------------------------------
    # State application
    # ------------------------------------------------------------------
    def _apply_state(self, x: np.ndarray) -> None:
        """Push every channel of ``x`` to its declared sink."""
        self.spec.push_adm1_state(x, self.plant)
        self._apply_input_flows(x)
        self._apply_kinetic_params(x)

    def _apply_input_flows(self, x: np.ndarray) -> None:
        digester = self.plant.components[self.spec.digester_id]
        q_vec = list(getattr(digester, "Q_substrates", []) or [])
        for i in self.spec.kind_indices("input_flow"):
            ch = self.spec.channels[i]
            idx = int(ch.input_substrate_index)
            while len(q_vec) <= idx:
                q_vec.append(0.0)
            q_vec[idx] = float(max(0.0, x[i]))
        if q_vec:
            digester.Q_substrates = q_vec
            # pyadm1 also caches the influent state derived from
            # Q_substrates — refresh it so the next step uses the
            # current per-step flows.
            try:
                digester.adm1.create_influent(q_vec, 0)
            except Exception:  # noqa: BLE001
                pass

    def _apply_kinetic_params(self, x: np.ndarray) -> None:
        digester = self.plant.components[self.spec.digester_id]
        kinetic = getattr(digester.adm1, "_kinetic", None)
        if kinetic is None:
            return
        for i in self.spec.kind_indices("kinetic_param"):
            ch = self.spec.channels[i]
            if ch.name in kinetic:
                kinetic[ch.name] = float(max(0.0, x[i]))

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------
    def step(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Apply ``x``, simulate ``dt`` days, return the new state.

        The state vector is clipped to the configured channel bounds
        *before* being pushed to the plant — ADM1 fails to integrate
        for negative concentrations, which the unclipped sigma points
        of the UKF easily produce when the prior is wide.

        ``step`` automatically calls :meth:`restore` so every invocation
        starts from the most recent :meth:`snapshot`, guaranteeing
        independent sigma-point propagation.
        """
        self.restore()
        x_safe = self.spec.clip(x)
        self._apply_state(x_safe)
        self.plant.step(dt)
        x_new = np.copy(x_safe)
        adm1_part = self.spec.read_adm1_state(self.plant)
        for i in self.spec.kind_indices("adm1"):
            x_new[i] = adm1_part[i]
        # Apply OU mean-reversion to input/kinetic augmented channels;
        # filter noise covariance handles the diffusive term.
        for i, ch in enumerate(self.spec.channels):
            if ch.kind == "adm1" or ch.drift_model != "ou":
                continue
            mean = float(ch.ou_mean)
            theta = max(ch.ou_theta, 1e-9)
            decay = np.exp(-theta * dt)
            x_new[i] = mean + (x_new[i] - mean) * decay
        return self.spec.clip(x_new)

    def refresh_outputs(self, x: np.ndarray, dt_stub: float = 1e-5) -> None:
        """Apply ``x`` and run a vanishingly small step.

        Used by filters that need to evaluate ``h(x)`` for additional
        sigma points without advancing the dynamics. The dt is small
        enough that the ADM1 right-hand side leaves the state
        essentially unchanged, but the side outputs (``Q_gas``,
        ``Q_ch4``, CHP draw, etc.) get refreshed. The plant is
        restored to the latest snapshot first so the side outputs are
        consistent with the calling filter's baseline.
        """
        self.restore()
        self._apply_state(self.spec.clip(x))
        self.plant.step(dt_stub)
