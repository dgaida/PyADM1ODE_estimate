"""Data-pipeline adapters bridging external sensor data to the UKF.

The subpackage separates DB-side concerns (unit conversion,
calibration, range validation, quality flags) from the UKF-side
mathematics (extractor, noise model). Mapping is captured per channel
in :class:`SensorChannelDef`; concrete data sources implement the
:class:`SensorSource` protocol.

The reference adapter is :class:`DataFrameSensorSource`, which reads
from an in-memory ``pandas.DataFrame``. Database-specific adapters
(PostgreSQL, InfluxDB, OPC-UA, Kafka, …) live in downstream packages
and only need to implement the protocol.
"""

from .sensor_defs import SensorChannelDef, UNIT_CONVERSIONS, lookup_conversion
from .sensor_source import DataFrameSensorSource, SensorSource

__all__ = [
    "SensorChannelDef",
    "UNIT_CONVERSIONS",
    "lookup_conversion",
    "SensorSource",
    "DataFrameSensorSource",
]
