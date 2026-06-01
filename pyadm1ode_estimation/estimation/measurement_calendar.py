"""Sensor-rate-aware gating: which channels are 'live' at time *t*?

In a real plant, measurements arrive at very different cadences:

* online sensors every few minutes (Q_gas, pH, NDIR fractions),
* daily titrations (FOS/TAC),
* weekly lab samples (TS, VS, COD-total),
* sporadic ad-hoc measurements (one-off NH4-N or LCFA samples).

The UKF accepts a ``gate_values`` dict on every ``update()`` call. The
calendar produces this dict (plus the matching ``y`` dict) from a single
measurement DataFrame, applying per-channel validity windows:

* an **online** measurement remains valid for ``2 ×`` its sampling period
  (tolerates clock skew and missed samples),
* a **daily** measurement remains valid for 24 h after the sample,
* a **weekly** measurement for 7 d,
* a **sporadic** measurement only in a narrow window around its timestamp.

Within the validity window, the most recent finite value in the
channel's column is used. Outside the window, or if no finite value
exists in the column near *t*, the channel is gated *off* and the
filter skips it for that step.

The DataFrame is expected to be indexed by time in **days** (the unit
the rest of the estimation package uses). Columns are channel names,
matching either the ``ObservationChannel.name`` (the default) or the
explicit ``gate_column`` if one is configured.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from .observation_model import ObservationModel

# ---------------------------------------------------------------------------
# Sample-rate spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleRate:
    """Validity window for one measurement channel.

    A channel's most recent finite measurement remains valid for
    ``validity_window_d`` days. Outside that window (or with no
    measurement in the column at all) the channel is gated *off*.

    Use the classmethod constructors for the common cases. The raw
    constructor accepts an explicit window in days for custom rates.
    """

    validity_window_d: float

    def __post_init__(self):
        if self.validity_window_d < 0:
            raise ValueError(
                f"validity_window_d must be ≥ 0, got {self.validity_window_d}"
            )

    @classmethod
    def online(cls, period_min: float = 5.0) -> "SampleRate":
        """Online sensor sampled every ``period_min`` minutes.

        Validity window is ``2 × period`` to tolerate clock skew and
        occasional missed samples. For a 5-minute sampling rate the
        window is 10 min ≈ 6.9 × 10⁻³ d.
        """
        if period_min <= 0:
            raise ValueError(f"period_min must be > 0, got {period_min}")
        return cls(validity_window_d=2.0 * period_min / 1440.0)

    @classmethod
    def periodic(cls, period_h: float) -> "SampleRate":
        """Periodic sampling every ``period_h`` hours, one period of
        validity after each measurement."""
        if period_h <= 0:
            raise ValueError(f"period_h must be > 0, got {period_h}")
        return cls(validity_window_d=period_h / 24.0)

    @classmethod
    def daily(cls) -> "SampleRate":
        """Daily measurement, valid for 24 h."""
        return cls.periodic(period_h=24.0)

    @classmethod
    def weekly(cls) -> "SampleRate":
        """Weekly measurement, valid for 7 d."""
        return cls.periodic(period_h=24.0 * 7.0)

    @classmethod
    def sporadic(cls, tolerance_min: float = 5.0) -> "SampleRate":
        """One-off lab measurement, valid only in a ``tolerance_min``
        window around the actual sample timestamp.

        Default ``5 min`` is generous enough that the filter sees the
        measurement on the step containing the sample but doesn't
        leak it into the next step.
        """
        if tolerance_min <= 0:
            raise ValueError(f"tolerance_min must be > 0, got {tolerance_min}")
        return cls(validity_window_d=tolerance_min / 1440.0)


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------


class MeasurementCalendar:
    """Maps a measurement DataFrame to filter gate dicts at any time *t*.

    Construct with a per-channel mapping of :class:`SampleRate`. Use
    :meth:`gate_values_at` and :meth:`measurements_at` (or the combined
    :meth:`values_for_filter`) per filter step.

    Example::

        calendar = MeasurementCalendar({
            "Q_gas":   SampleRate.online(period_min=5),
            "pH":      SampleRate.online(period_min=5),
            "FOS_TAC": SampleRate.daily(),
            "NH4_N":   SampleRate.sporadic(),
        })

        for t in time_grid:
            y, gates = calendar.values_for_filter(t=t, df=measurements_df)
            ukf.predict(dt=dt)
            ukf.update(y=y, t=t, gate_values=gates)
    """

    def __init__(self, rates: Mapping[str, SampleRate]):
        if not rates:
            raise ValueError("MeasurementCalendar needs at least one channel rate.")
        self.rates: Dict[str, SampleRate] = dict(rates)

    @classmethod
    def from_obs_model(
        cls,
        obs_model: ObservationModel,
        default_rates: Optional[Mapping[str, SampleRate]] = None,
        fallback: Optional[SampleRate] = None,
    ) -> "MeasurementCalendar":
        """Build a calendar covering every channel of ``obs_model``.

        For each channel:
        * use ``default_rates[gate_key]`` if specified (where
          ``gate_key`` is ``channel.gate_column`` if set, else
          ``channel.name``),
        * else use ``fallback`` (defaults to
          :meth:`SampleRate.sporadic`).
        """
        defaults = dict(default_rates or {})
        fb = fallback if fallback is not None else SampleRate.sporadic()
        rates: Dict[str, SampleRate] = {}
        for c in obs_model.channels:
            gate_key = c.gate_column if c.gate_column else c.name
            rates[gate_key] = defaults.get(gate_key, fb)
        return cls(rates)

    # ------------------------------------------------------------------
    # Per-step API
    # ------------------------------------------------------------------
    def gate_values_at(self, t: float, df: pd.DataFrame) -> Dict[str, float]:
        """Return ``{name: 1.0 if active else 0.0}`` at time ``t``.

        A channel is *active* if its column in ``df`` has at least one
        finite value within the validity window
        ``[t - validity_window_d, t]``. The window is half-open on the
        upper side. Measurements exactly at ``t`` are included, future
        measurements are not.
        """
        gates: Dict[str, float] = {}
        for name, rate in self.rates.items():
            gates[name] = (
                1.0 if self._latest_valid(name, rate, t, df) is not None else 0.0
            )
        return gates

    def measurements_at(self, t: float, df: pd.DataFrame) -> Dict[str, float]:
        """Return ``{name: most_recent_value_in_window or NaN}``.

        The value is the **most recent finite** entry in the channel's
        column within the validity window, or ``NaN`` if no such entry
        exists. The UKF skips channels with non-finite ``y`` values, so
        the NaN sentinel is safe to pass through to ``update()``.
        """
        out: Dict[str, float] = {}
        for name, rate in self.rates.items():
            val = self._latest_valid(name, rate, t, df)
            out[name] = float(val) if val is not None else float("nan")
        return out

    def values_for_filter(
        self, t: float, df: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Convenience: returns ``(y_dict, gate_values_dict)`` together.

        Equivalent to::

            (self.measurements_at(t, df), self.gate_values_at(t, df))

        but iterates the DataFrame only once.
        """
        y: Dict[str, float] = {}
        gates: Dict[str, float] = {}
        for name, rate in self.rates.items():
            val = self._latest_valid(name, rate, t, df)
            if val is None:
                y[name] = float("nan")
                gates[name] = 0.0
            else:
                y[name] = float(val)
                gates[name] = 1.0
        return y, gates

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _latest_valid(
        name: str, rate: SampleRate, t: float, df: pd.DataFrame
    ) -> Optional[float]:
        """Return the most recent finite value of column ``name`` in the
        validity window ending at ``t``, or ``None`` if none exists."""
        if name not in df.columns:
            return None
        lower = t - rate.validity_window_d
        # Index expected to be numeric (days). Inclusive on both ends:
        # a measurement taken exactly at t is valid, and at t - window
        # is the oldest still-valid sample.
        col = df.loc[(df.index >= lower) & (df.index <= t), name]
        col = col.dropna()
        if len(col) == 0:
            return None
        # Filter out non-finite values (np.inf etc.) the dropna missed.
        finite = col[np.isfinite(col.values)]
        if len(finite) == 0:
            return None
        return finite.iloc[-1]


__all__ = ["SampleRate", "MeasurementCalendar"]
