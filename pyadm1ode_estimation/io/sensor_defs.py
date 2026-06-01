"""Sensor channel definitions for the DB-to-UKF data pipeline.

The :class:`SensorChannelDef` dataclass binds one DB column (or any
named data source) to one UKF observation channel and captures all
the data-side transformations a real plant SCADA needs:

* Engineering-unit conversion (e.g. ``Nm³/h`` → ``m³/d``)
* Linear calibration (``scale``, ``offset``)
* Range validation (negative Q_gas, pH > 14 ...)
* Quality-flag handling (vendor status codes that mean "ignore me")

It does **not** carry any UKF mathematics, the noise model and
extractor live on the :class:`ObservationChannel`. The split lets you
swap SCADA implementations (PostgreSQL, InfluxDB, OPC-UA, Kafka, …)
without touching the filter setup.

Typical usage::

    from pyadm1ode_estimation.io import SensorChannelDef

    defs = [
        SensorChannelDef(
            db_column="fic101_q_gas_nm3h",
            ukf_channel="Q_gas",
            unit_in="Nm3/h",
            unit_out="m3/d",
            valid_range=(0.0, 50_000.0),
            quality_column="fic101_quality",
        ),
        SensorChannelDef(
            db_column="ait201_ph",
            ukf_channel="pH",
            unit_in="-",
            unit_out="-",
            valid_range=(3.0, 11.0),
        ),
        SensorChannelDef(
            db_column="wt301_maize_kg_h",
            ukf_channel="Q_maize_silage",
            unit_in="kg/h",
            unit_out="m3/d",
            # Maize silage density ≈ 700 kg/m³.
            converter=lambda kg_h: kg_h * 24.0 / 700.0,
        ),
    ]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

# ---------------------------------------------------------------------------
# Built-in unit conversion table
# ---------------------------------------------------------------------------
#
# Maps ``(unit_in, unit_out) -> (scale, offset)`` where
# ``value_out = scale * value_in + offset``.
#
# Conversions that need extra context (densities, COD factors,
# molar masses, gas norm conditions) are NOT in this table. The
# caller must provide an explicit ``converter`` callable for those.
#
# The table is intentionally small (~20 entries) and focused on the
# unambiguous, unitful conversions that appear in agricultural biogas
# SCADA. Easy to extend.


UNIT_CONVERSIONS: dict[Tuple[str, str], Tuple[float, float]] = {
    # ---- Volumetric flow ----------------------------------------
    ("m3/d", "m3/d"): (1.0, 0.0),
    ("m3/h", "m3/d"): (24.0, 0.0),
    ("Nm3/h", "m3/d"): (24.0, 0.0),
    ("Nm3/d", "m3/d"): (1.0, 0.0),
    ("L/h", "m3/d"): (0.024, 0.0),
    ("L/min", "m3/d"): (1.44, 0.0),
    ("L/s", "m3/d"): (86.4, 0.0),
    # ---- Mass flow ----------------------------------------------
    ("kg/h", "kg/d"): (24.0, 0.0),
    ("kg/d", "kg/d"): (1.0, 0.0),
    ("t/h", "kg/d"): (24_000.0, 0.0),
    ("t/d", "kg/d"): (1_000.0, 0.0),
    # ---- Concentration ------------------------------------------
    ("g/L", "kg/m3"): (1.0, 0.0),
    ("mg/L", "kg/m3"): (1.0e-3, 0.0),
    ("kg/m3", "kg/m3"): (1.0, 0.0),
    # ---- Temperature --------------------------------------------
    ("°C", "K"): (1.0, 273.15),
    ("C", "K"): (1.0, 273.15),
    ("K", "K"): (1.0, 0.0),
    ("K", "°C"): (1.0, -273.15),
    # ---- Pressure -----------------------------------------------
    ("bar", "Pa"): (1.0e5, 0.0),
    ("bar", "bar"): (1.0, 0.0),
    ("mbar", "Pa"): (100.0, 0.0),
    ("Pa", "bar"): (1.0e-5, 0.0),
    # ---- Time (occasionally needed for sample intervals) ---------
    ("s", "d"): (1.0 / 86_400.0, 0.0),
    ("min", "d"): (1.0 / 1_440.0, 0.0),
    ("h", "d"): (1.0 / 24.0, 0.0),
    ("d", "d"): (1.0, 0.0),
    # ---- Dimensionless ------------------------------------------
    ("-", "-"): (1.0, 0.0),
    ("%", "-"): (0.01, 0.0),
    ("ppm", "-"): (1.0e-6, 0.0),
}


def lookup_conversion(unit_in: str, unit_out: str) -> Optional[Tuple[float, float]]:
    """Return ``(scale, offset)`` for ``unit_in → unit_out`` or ``None``.

    A return of ``None`` means the conversion needs context (density,
    molar mass, ...) and the caller must supply a custom ``converter``
    callable. Identity (``unit_in == unit_out``) always succeeds with
    ``(1.0, 0.0)`` even if the unit string is not in the table.
    """
    if unit_in == unit_out:
        return (1.0, 0.0)
    return UNIT_CONVERSIONS.get((unit_in, unit_out))


# ---------------------------------------------------------------------------
# SensorChannelDef
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorChannelDef:
    """Binds one DB column to one UKF observation channel.

    The dataclass is frozen so it can be safely used as a dict key
    and so config files (YAML / JSON) round-trip cleanly.

    Attributes:
        db_column: Column name / tag in the data source (DB, CSV, …).
        ukf_channel: Channel name as declared in the ``ObservationModel``
            (= the key in the ``y_dict`` that ``ukf.update`` expects).
        unit_in: Engineering unit of the raw reading (e.g. ``"Nm3/h"``,
            ``"kg/h"``, ``"mg/L"``, ``"°C"``). Use ``"-"`` for
            dimensionless quantities (pH, ratios).
        unit_out: Target unit in the model coordinate system. Matches
            ``ObservationChannel.noise_std``'s implicit unit. Typically
            ``"m3/d"``, ``"kg/m3"``, ``"-"`` etc.
        converter: Optional callable ``raw -> model_value``. Used when
            the conversion needs context the unit table can't express
            (densities, COD factors, gas norm conditions). When ``None``
            the unit table is consulted; if no entry matches, the
            transformation raises at runtime.
        valid_range: Optional ``(low, high)`` bounds in ``unit_out``.
            Values outside the range are dropped (the channel becomes
            inactive for that step) rather than passed through to the
            filter.
        scale: Linear calibration multiplier applied AFTER unit
            conversion. Typical: instrument-specific calibration
            corrections.
        offset: Calibration offset applied AFTER scale.
        quality_column: Optional column carrying a per-row quality
            flag. When set, rows where the column value falls into
            ``bad_status_values`` are dropped.
        bad_status_values: Tuple of values that mean "ignore this
            row" in ``quality_column``. Defaults to ``(False, 0, None)``
            so vendor-specific truthy/falsy flags work out of the box.

    Transformation pipeline applied to each raw value::

        if raw is None:                  skip
        if quality is bad:               skip
        if converter is set:
            val = converter(raw)
        else:
            scale_lut, off_lut = lookup_conversion(unit_in, unit_out)
            val = raw * scale_lut + off_lut
        val = val * scale + offset
        if val not in valid_range:        skip
        return val
    """

    db_column: str
    ukf_channel: str

    unit_in: str = "-"
    unit_out: str = "-"

    converter: Optional[Callable[[float], float]] = None

    valid_range: Optional[Tuple[float, float]] = None

    scale: float = 1.0
    offset: float = 0.0

    quality_column: Optional[str] = None
    bad_status_values: Tuple = field(default_factory=lambda: (False, 0, None))

    # ------------------------------------------------------------------
    # Transformation
    # ------------------------------------------------------------------

    def transform(
        self, raw: Optional[float], quality: Optional[object] = None
    ) -> Optional[float]:
        """Apply the full transformation pipeline. Returns ``None`` if
        the value should be dropped (NULL, bad quality, out of range).

        ``quality`` is the value read from ``quality_column``. When
        ``quality_column`` is ``None``, this argument is ignored.
        """
        if raw is None:
            return None

        # ---- quality flag -----------------------------------------
        if self.quality_column is not None:
            if quality in self.bad_status_values:
                return None

        # ---- conversion -------------------------------------------
        try:
            raw_f = float(raw)
        except (TypeError, ValueError):
            return None
        if not _is_finite(raw_f):
            return None

        if self.converter is not None:
            val = float(self.converter(raw_f))
        else:
            lut = lookup_conversion(self.unit_in, self.unit_out)
            if lut is None:
                raise ValueError(
                    f"No built-in unit conversion {self.unit_in!r} → "
                    f"{self.unit_out!r} for channel {self.ukf_channel!r}. "
                    f"Provide a custom converter=..."
                )
            scale_lut, off_lut = lut
            val = raw_f * scale_lut + off_lut

        # ---- calibration ------------------------------------------
        val = val * self.scale + self.offset

        if not _is_finite(val):
            return None

        # ---- range check ------------------------------------------
        if self.valid_range is not None:
            lo, hi = self.valid_range
            if val < lo or val > hi:
                return None

        return val


def _is_finite(x: float) -> bool:
    """Local equivalent of math.isfinite without importing math at module top."""
    return x == x and x != float("inf") and x != float("-inf")


__all__ = [
    "SensorChannelDef",
    "UNIT_CONVERSIONS",
    "lookup_conversion",
]
