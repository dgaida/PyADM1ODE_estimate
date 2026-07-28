"""Abstract sensor-data source for the UKF measurement loop.

The :class:`SensorSource` protocol decouples the filter loop from
*where* the measurements come from (DB, CSV, message broker, in-memory
DataFrame). All adapters return the same shape:
``(t_days, y_dict)`` per yield, where ``y_dict`` contains only the
channels that produced a valid reading on that step. Invalid /
missing channels are simply absent - the UKF skips them.

This module ships one concrete implementation:
:class:`DataFrameSensorSource`. It reads from an in-memory
``pandas.DataFrame`` and is the canonical reference / test adapter.
Real DB adapters live in downstream packages and only need to
implement the protocol.

Typical filter loop::

    from pyadm1ode_estimation.io import (
        DataFrameSensorSource, SensorChannelDef,
    )

    defs = [SensorChannelDef(...), ...]
    source = DataFrameSensorSource(measurements_df, defs)

    prev_t = None
    for t, y in source.stream():
        dt = (t - prev_t) if prev_t is not None else 1.0 / 24.0
        ukf.predict(dt=dt)
        ukf.update(y, t=t)
        prev_t = t
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Protocol,
    runtime_checkable,
)

from .sensor_defs import SensorChannelDef

if TYPE_CHECKING:
    import pandas as pd


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SensorSource(Protocol):
    """Abstract chronological stream of sensor measurements.

    All adapters yield ``(t_days, y_dict)`` pairs in monotonically
    non-decreasing time order. ``y_dict`` maps
    ``ObservationChannel.name`` → measured value in **model units**
    (i.e. already through unit conversion + calibration). Channels
    without a valid reading at time ``t`` are simply absent from
    ``y_dict``; the UKF's ``update`` skips them automatically.

    Adapters are responsible for:
    * mapping their native column / tag names to UKF channel names,
    * converting engineering units to model units,
    * applying calibration (scale + offset),
    * dropping out-of-range and bad-quality values.

    All of that is captured by a list of :class:`SensorChannelDef`.
    """

    def stream(
        self,
        start_t: float | None = None,
        end_t: float | None = None,
    ) -> Iterator[tuple[float, dict[str, float]]]:
        """Yield ``(t, y)`` pairs in chronological order.

        Args:
            start_t: Optional minimum time (inclusive). Rows with
                ``t < start_t`` are skipped.
            end_t: Optional maximum time (exclusive). Iteration stops
                when ``t >= end_t``.

        Yields:
            Tuple ``(t_days, y_dict)``. ``y_dict`` contains one entry
            per *valid* channel reading at that step. Rows where no
            channel produced a valid reading are still yielded as
            ``(t, {})`` so the filter can run a pure ``predict()``
            with the correct ``dt``.
        """
        ...


# ---------------------------------------------------------------------------
# DataFrameSensorSource
# ---------------------------------------------------------------------------


class DataFrameSensorSource:
    """``SensorSource`` backed by a ``pandas.DataFrame``.

    The DataFrame must:

    * be indexed by **time in days** (monotonically non-decreasing),
    * carry one column per ``db_column`` referenced in the channel
      defs (extra columns are ignored),
    * additionally carry any ``quality_column``-referenced columns.

    Missing values must be encoded as ``NaN`` (the standard pandas
    convention).

    The class is intentionally minimal. It is the canonical reference
    for the :class:`SensorSource` protocol and the cleanest path for
    backtesting against historical exports. Live database adapters
    should mimic the same I/O contract.

    Args:
        df: Time-indexed DataFrame with the raw measurements.
        sensor_defs: List of :class:`SensorChannelDef` describing the
            DB-to-UKF mapping for each channel.

    Raises:
        ValueError: If a referenced ``db_column`` or ``quality_column``
            is not present in ``df``.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sensor_defs: Sequence[SensorChannelDef],
    ):
        self._df = df
        self._defs = list(sensor_defs)
        self._validate_columns()

    def _validate_columns(self) -> None:
        cols = set(self._df.columns)
        for d in self._defs:
            if d.db_column not in cols:
                raise ValueError(
                    f"SensorChannelDef references db_column={d.db_column!r} "
                    f"which is not in the DataFrame. Available columns: "
                    f"{sorted(cols)}"
                )
            if d.quality_column is not None and d.quality_column not in cols:
                raise ValueError(
                    f"SensorChannelDef for {d.ukf_channel!r} references "
                    f"quality_column={d.quality_column!r} which is not in "
                    f"the DataFrame."
                )

    # ------------------------------------------------------------------
    # SensorSource implementation
    # ------------------------------------------------------------------

    def stream(
        self,
        start_t: float | None = None,
        end_t: float | None = None,
    ) -> Iterator[tuple[float, dict[str, float]]]:
        df = self._df
        if start_t is not None:
            df = df.loc[df.index >= start_t]
        if end_t is not None:
            df = df.loc[df.index < end_t]

        for t, row in df.iterrows():
            y: dict[str, float] = {}
            for d in self._defs:
                raw = row.get(d.db_column)
                # pandas returns NaN for missing scalar entries; let
                # SensorChannelDef.transform deal with NaN.
                quality = (
                    row.get(d.quality_column) if d.quality_column is not None else None
                )
                val = d.transform(raw, quality)
                if val is not None:
                    y[d.ukf_channel] = val
            yield float(t), y


__all__ = [
    "DataFrameSensorSource",
    "SensorSource",
]
