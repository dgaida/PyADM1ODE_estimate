"""Reference plant builders for state-estimation tutorials and tests.

These builders use the :mod:`pyadm1` plant configurator to assemble
small, fully functional :class:`BiogasPlant` instances that the
estimator can run against. Two builders are provided, in increasing
complexity:

* :func:`build_simple_plant` — single fermenter + digestate storage +
  one CHP. The smallest mass-balance-closed plant; recommended as an
  entry point.
* :func:`build_multi_stage_plant` — three-stage cascade
  (primary → secondary → storage) with two CHPs and per-stage
  heating circuits. A realistic agricultural-AD topology used for
  end-to-end estimator validation.

The builders return ready-to-simulate plant instances; estimator
setup (StateVectorSpec, ObservationModel, MeasurementCalendar) is
constructed separately by the caller using
:func:`pyadm1ode_estimation.estimation.adm1da_full_spec`.
"""

from .multi_stage import build_multi_stage_plant
from .simple import build_simple_plant

__all__ = ["build_simple_plant", "build_multi_stage_plant"]
