"""Picklable plant + components builder for tests that don't need the
real pyadm1 substrate library.

The integration tests for :class:`ParallelUKF` and :class:`ConstrainedUKF`
historically depended on :func:`pyadm1ode_estimation.example_plants.build_simple_plant`,
which in turn reads YAML substrate files. Pip-installed pyadm1 wheels in
CI environments ship without that catalog — every test that called
``build_simple_plant()`` then failed with ``FileNotFoundError``.

The mock here satisfies the minimal interface
:class:`pyadm1ode_estimation.estimation.process_model.ADM1ProcessModel`
consumes:

* ``plant.components`` — dict of components, at least one with
  ``adm1_state`` and ``Q_substrates``
* ``plant.simulation_time`` — float, advanced by ``step``
* ``plant.step(dt)`` — advances the plant for ``dt`` days

The "dynamics" is a deterministic linear decay plus drift — enough to
make the UKF actually have something to track without requiring
pyadm1's full ODE machinery. All attributes are pickleable (no closures,
no .NET objects) so ``multiprocessing.spawn`` can ship the components
builder to :class:`ParallelUKF` workers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class _MockADM1:
    """Stand-in for pyadm1's ADM1 model. ``ADM1ProcessModel._apply_input_flows``
    calls ``digester.adm1.create_influent`` inside a try/except, so a no-op
    implementation is enough. ``_kinetic`` is consulted via ``getattr`` with
    a ``None`` default, so ``None`` here means "no kinetic params"."""

    _kinetic: dict | None = None

    def create_influent(self, q_vec, idx):
        return None


@dataclass
class _MockDigester:
    """ADM1ProcessModel-compatible digester component.

    The default ``adm1_state`` of length 41 matches ``adm1da_full_spec``
    so the standard spec works against this mock unchanged. Q_substrates
    is sized for the two-substrate ``simple plant`` topology (maize +
    cattle slurry) but ADM1ProcessModel handles arbitrary sizes via
    ``input_substrate_index``.
    """

    adm1_state: list[float] = field(default_factory=lambda: [0.5] * 41)
    Q_substrates: list[float] = field(default_factory=lambda: [10.0, 5.0])
    adm1: _MockADM1 = field(default_factory=_MockADM1)


@dataclass
class _MockPlant:
    """Picklable BiogasPlant stand-in.

    The ``step`` dynamics is a trivial deterministic linear decay with
    small drift — chosen so that:

    * the UKF actually sees state evolution (otherwise the unscented
      transform would collapse trivially);
    * the result is deterministic and bounded (so repeated calls give
      identical answers, which the parallel-vs-serial test relies on);
    * the operation involves no random number generators (would break
      pickling and reproducibility).
    """

    simulation_time: float = 0.0
    components: dict[str, Any] = field(
        default_factory=lambda: {"fermenter": _MockDigester()}
    )

    def step(self, dt: float) -> None:
        digester = self.components["fermenter"]
        decay = 1.0 - 0.05 * dt
        drift = 0.001 * dt
        digester.adm1_state = [max(0.0, x * decay + drift) for x in digester.adm1_state]
        self.simulation_time += dt


@dataclass
class _MockGasStorage:
    """Stand-in for a digester's gas-dome storage sub-object.

    ``ADM1ProcessModel.snapshot`` captures, for every component exposing a
    ``gas_storage`` attribute, that sub-object's ``outputs_data`` dict plus
    its ``stored_volume_m3`` attribute; :meth:`ADM1ProcessModel.restore`
    writes both back. This mock carries exactly those two pieces of state,
    so the multi-component snapshot path is exercised end to end.
    """

    outputs_data: dict[str, Any] = field(default_factory=lambda: {"Q_gas_stored": 0.0})
    stored_volume_m3: float = 100.0


@dataclass
class _MockDigesterWithStorage:
    """Digester mock that additionally owns an *evolving* gas storage.

    Unlike :class:`_MockDigester`, the gas-dome fill accumulates across
    ``step`` calls AND feeds back into the ADM1 decay rate (see
    :meth:`_MockMultiStagePlant.step`). That coupling is what makes the
    parallel-vs-serial test meaningful: if ``snapshot``/``restore`` failed
    to capture the gas-storage state, sigma points would inherit each
    other's fill level and the serial and parallel propagations would
    diverge.
    """

    adm1_state: list[float] = field(default_factory=lambda: [0.5] * 41)
    Q_substrates: list[float] = field(default_factory=lambda: [10.0, 5.0])
    adm1: _MockADM1 = field(default_factory=_MockADM1)
    gas_storage: _MockGasStorage = field(default_factory=_MockGasStorage)
    V_gas: float = 200.0


@dataclass
class _MockMultiStagePlant:
    """Picklable multi-component plant whose gas storage carries state.

    ``step`` couples the ADM1 decay to the current gas-dome fill and then
    advances the fill from the (new) ADM1 inventory. Both quantities are
    deterministic and bounded, so two identical runs agree bit-for-bit —
    the property the parallel-vs-serial test relies on — while still
    forcing the snapshot/restore machinery to reset the gas storage
    between sigma points. Drop the gas-storage capture from
    ``ADM1ProcessModel.snapshot`` and the parallel run diverges from the
    serial one, which is exactly the regression the test guards against.
    """

    simulation_time: float = 0.0
    components: dict[str, Any] = field(
        default_factory=lambda: {"fermenter": _MockDigesterWithStorage()}
    )

    def step(self, dt: float) -> None:
        digester = self.components["fermenter"]
        gs = digester.gas_storage
        # Fill level (0..~1) modulates the decay: a fuller dome slows the
        # apparent ADM1 turnover. This is the coupling that leaks between
        # sigma points if the gas storage is not part of the snapshot.
        fill = gs.stored_volume_m3 / digester.V_gas
        decay = 1.0 - 0.05 * dt * (1.0 + 0.1 * fill)
        drift = 0.001 * dt
        digester.adm1_state = [max(0.0, x * decay + drift) for x in digester.adm1_state]
        # Gas production feeds the dome; a constant draw empties it.
        production = 0.01 * dt * sum(digester.adm1_state)
        gs.stored_volume_m3 = max(0.0, gs.stored_volume_m3 + production - 0.5 * dt)
        gs.outputs_data = {"Q_gas_stored": production}
        self.simulation_time += dt


def build_mock_components():
    """Top-level (importable) factory: returns ``(process, obs, spec)``
    built around the :class:`_MockPlant`.

    Top-level so ``multiprocessing.spawn`` can pickle a reference to it
    and re-call it inside :class:`ParallelUKF` workers. Mirrors the
    return contract of
    :func:`pyadm1ode_estimation.estimation.build_filter_components`,
    so it drops into the same call sites as the real builder.

    Sensor set is restricted to ``substrate_dose`` — the only built-in
    extractor that doesn't depend on plant-side gas outputs (which the
    mock doesn't provide). Two substrate channels give the UKF a 2-dim
    observation, enough to test the per-channel cache, parallel
    propagation, and constrained update paths.
    """
    from pyadm1ode_estimation.estimation import (
        InputSpec,
        adm1da_full_spec,
    )
    from pyadm1ode_estimation.estimation.observation_model import (
        ObservationModel,
    )
    from pyadm1ode_estimation.estimation.process_model import (
        ADM1ProcessModel,
    )
    from pyadm1ode_estimation.estimation.quickstart import _resolve_sensors

    plant = _MockPlant()
    spec = adm1da_full_spec(
        digester_id="fermenter",
        substrate_inputs=[
            InputSpec("maize_silage", substrate_index=0, initial_flow=10.0),
            InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
        ],
    )
    obs_channels = _resolve_sensors(
        sensors=["substrate_dose"],
        digester_id="fermenter",
        spec=spec,
    )
    obs = ObservationModel(channels=obs_channels)
    process = ADM1ProcessModel(plant, spec)
    process.snapshot()
    return process, obs, spec


def build_multistage_mock_components():
    """Top-level factory for a mock plant whose gas storage carries state.

    Same return contract and sensor set as :func:`build_mock_components`,
    but built around :class:`_MockMultiStagePlant`. It exercises the
    *multi-component* snapshot path added in the 2026-06 SR-UKF work: the
    gas-dome fill evolves during ``step`` and couples back into the ADM1
    dynamics, so a parallel-vs-serial comparison fails the moment
    ``ADM1ProcessModel.snapshot``/``restore`` stops capturing the
    gas-storage state.

    Like :func:`build_mock_components`, it depends only on
    ``pyadm1ode_estimation`` itself — no ``pyadm1`` simulator, no
    substrate YAML catalog, no data files — so it runs in any CI
    environment. Top-level (no closures) so ``multiprocessing.spawn`` can
    pickle a reference and re-build the plant inside
    :class:`ParallelUKF` workers.
    """
    from pyadm1ode_estimation.estimation import (
        InputSpec,
        adm1da_full_spec,
    )
    from pyadm1ode_estimation.estimation.observation_model import (
        ObservationModel,
    )
    from pyadm1ode_estimation.estimation.process_model import (
        ADM1ProcessModel,
    )
    from pyadm1ode_estimation.estimation.quickstart import _resolve_sensors

    plant = _MockMultiStagePlant()
    spec = adm1da_full_spec(
        digester_id="fermenter",
        substrate_inputs=[
            InputSpec("maize_silage", substrate_index=0, initial_flow=10.0),
            InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
        ],
    )
    obs_channels = _resolve_sensors(
        sensors=["substrate_dose"],
        digester_id="fermenter",
        spec=spec,
    )
    obs = ObservationModel(channels=obs_channels)
    process = ADM1ProcessModel(plant, spec)
    process.snapshot()
    return process, obs, spec
