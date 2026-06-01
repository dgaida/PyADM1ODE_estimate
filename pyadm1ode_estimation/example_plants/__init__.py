"""Reference plant builders for state-estimation tutorials and tests.

These builders use the :mod:`pyadm1` plant configurator to assemble
small, fully functional :class:`BiogasPlant` instances that the
estimator can run against. Two builders are provided, in increasing
complexity:

* :func:`build_simple_plant` — single fermenter + digestate storage +
  one CHP.
* :func:`build_multi_stage_plant` — three-stage cascade
  (primary → secondary → storage) with two CHPs and per-stage
  heating circuits.

In addition, :func:`build_example_sensor_defs` provides a realistic
SCADA-tag → UKF-channel mapping for the multi-stage plant. It is the
reference for writing custom :class:`SensorChannelDef` lists against
the :mod:`pyadm1ode_estimation.io` data pipeline.

The plant builders return ready-to-simulate plant instances. The estimator
setup (StateVectorSpec, ObservationModel, MeasurementCalendar) is
constructed separately by the caller using
:func:`pyadm1ode_estimation.estimation.adm1da_full_spec`.
"""

from .multi_stage import build_multi_stage_plant
from .sensor_defs import build_example_sensor_defs, example_scada_columns
from .simple import build_simple_plant

__all__ = [
    "build_simple_plant",
    "build_multi_stage_plant",
    "build_example_sensor_defs",
    "example_scada_columns",
]
