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
from typing import Any, Dict, List, Optional


@dataclass
class _MockADM1:
    """Stand-in for pyadm1's ADM1 model. ``ADM1ProcessModel._apply_input_flows``
    calls ``digester.adm1.create_influent`` inside a try/except, so a no-op
    implementation is enough. ``_kinetic`` is consulted via ``getattr`` with
    a ``None`` default, so ``None`` here means "no kinetic params"."""

    _kinetic: Optional[dict] = None

    def create_influent(self, q_vec, idx):  # noqa: ARG002
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

    adm1_state: List[float] = field(default_factory=lambda: [0.5] * 41)
    Q_substrates: List[float] = field(default_factory=lambda: [10.0, 5.0])
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
    components: Dict[str, Any] = field(
        default_factory=lambda: {"fermenter": _MockDigester()}
    )

    def step(self, dt: float) -> None:
        digester = self.components["fermenter"]
        decay = 1.0 - 0.05 * dt
        drift = 0.001 * dt
        digester.adm1_state = [max(0.0, x * decay + drift) for x in digester.adm1_state]
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
