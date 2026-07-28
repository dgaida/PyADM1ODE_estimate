"""Unit tests for the io/ subpackage (SensorChannelDef + SensorSource)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyadm1ode_estimation.io import (
    UNIT_CONVERSIONS,
    DataFrameSensorSource,
    SensorChannelDef,
    SensorSource,
    lookup_conversion,
)

# ---------------------------------------------------------------------------
# Unit-conversion table
# ---------------------------------------------------------------------------


class TestUnitConversionTable:
    def test_identity_is_always_available(self):
        # Even units that are not in the table return (1, 0) for self.
        assert lookup_conversion("custom_unit", "custom_unit") == (1.0, 0.0)
        assert lookup_conversion("-", "-") == (1.0, 0.0)

    def test_nm3h_to_m3d_is_24(self):
        scale, off = lookup_conversion("Nm3/h", "m3/d")
        assert scale == 24.0
        assert off == 0.0

    def test_lmin_to_m3d(self):
        # 1 L/min = 0.001 m3/min = 1.44 m3/d
        scale, _off = lookup_conversion("L/min", "m3/d")
        assert scale == pytest.approx(1.44)

    def test_mgL_to_kgm3(self):
        scale, off = lookup_conversion("mg/L", "kg/m3")
        assert scale == pytest.approx(1e-3)
        assert off == 0.0

    def test_celsius_to_kelvin_is_affine(self):
        scale, off = lookup_conversion("°C", "K")
        assert scale == 1.0
        assert off == pytest.approx(273.15)

    def test_unknown_conversion_returns_none(self):
        # No density information available → not in table.
        assert lookup_conversion("kg/h", "m3/d") is None

    def test_table_has_no_unexpected_entries(self):
        # Sanity check: every entry has the right shape.
        for (u_in, u_out), val in UNIT_CONVERSIONS.items():
            assert isinstance(u_in, str)
            assert isinstance(u_out, str)
            assert isinstance(val, tuple) and len(val) == 2
            scale, off = val
            assert isinstance(scale, float)
            assert isinstance(off, float)


# ---------------------------------------------------------------------------
# SensorChannelDef transformations
# ---------------------------------------------------------------------------


class TestSensorChannelDefTransform:
    def test_basic_nm3h_to_m3d(self):
        d = SensorChannelDef(
            db_column="q_gas",
            ukf_channel="Q_gas",
            unit_in="Nm3/h",
            unit_out="m3/d",
        )
        assert d.transform(250.0) == pytest.approx(6000.0)

    def test_null_input_returns_none(self):
        d = SensorChannelDef(db_column="x", ukf_channel="X", unit_in="-", unit_out="-")
        assert d.transform(None) is None

    def test_nan_input_returns_none(self):
        d = SensorChannelDef(db_column="x", ukf_channel="X")
        assert d.transform(float("nan")) is None

    def test_inf_input_returns_none(self):
        d = SensorChannelDef(db_column="x", ukf_channel="X")
        assert d.transform(float("inf")) is None

    def test_non_numeric_raw_is_skipped(self):
        d = SensorChannelDef(db_column="x", ukf_channel="X")
        assert d.transform("not_a_number") is None

    def test_out_of_range_returns_none(self):
        d = SensorChannelDef(
            db_column="ph",
            ukf_channel="pH",
            valid_range=(3.0, 11.0),
        )
        assert d.transform(2.5) is None  # below lower
        assert d.transform(13.0) is None  # above upper
        assert d.transform(7.0) == 7.0  # in range

    def test_range_boundaries_inclusive(self):
        d = SensorChannelDef(
            db_column="x",
            ukf_channel="X",
            valid_range=(0.0, 10.0),
        )
        assert d.transform(0.0) == 0.0
        assert d.transform(10.0) == 10.0

    def test_calibration_scale_and_offset(self):
        d = SensorChannelDef(
            db_column="x",
            ukf_channel="X",
            scale=2.0,
            offset=5.0,
        )
        # val = (10 * 1) + 0 [unit] then * 2 + 5 = 25
        assert d.transform(10.0) == pytest.approx(25.0)

    def test_custom_converter_takes_precedence(self):
        # If a converter is set, unit table is NOT consulted.
        d = SensorChannelDef(
            db_column="x",
            ukf_channel="X",
            unit_in="kg/h",
            unit_out="m3/d",
            # Maize-silage density 700 kg/m³.
            converter=lambda kg_h: kg_h * 24.0 / 700.0,
        )
        # 700 kg/h * 24 / 700 = 24 m3/d
        assert d.transform(700.0) == pytest.approx(24.0)

    def test_unknown_unit_raises(self):
        d = SensorChannelDef(
            db_column="x",
            ukf_channel="X",
            unit_in="kg/h",
            unit_out="m3/d",
            # NO converter → must rely on table → no entry for kg/h to m3/d.
        )
        with pytest.raises(ValueError, match="No built-in unit conversion"):
            d.transform(100.0)

    def test_quality_column_drops_bad_rows(self):
        d = SensorChannelDef(
            db_column="x",
            ukf_channel="X",
            quality_column="x_quality",
            bad_status_values=(False, 0, None, "BAD"),
        )
        assert d.transform(100.0, quality=True) == 100.0
        assert d.transform(100.0, quality=False) is None
        assert d.transform(100.0, quality=0) is None
        assert d.transform(100.0, quality="BAD") is None
        assert d.transform(100.0, quality="OK") == 100.0

    def test_celsius_to_kelvin_offset(self):
        d = SensorChannelDef(
            db_column="t",
            ukf_channel="T",
            unit_in="°C",
            unit_out="K",
        )
        assert d.transform(25.0) == pytest.approx(298.15)


# ---------------------------------------------------------------------------
# DataFrameSensorSource
# ---------------------------------------------------------------------------


class TestDataFrameSensorSource:
    def _sample_df(self):
        return pd.DataFrame(
            {
                "fic101_q_gas_nm3h": [250.0, 252.0, np.nan, 248.0],
                "ait201_ph": [7.4, 7.3, 7.5, 14.5],  # last one out of range
                "fic101_quality": [True, True, True, True],
            },
            index=[0.0, 0.0417, 0.0833, 0.125],  # 1 h, 2 h, 3 h in days
        )

    def _sample_defs(self):
        return [
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
                valid_range=(3.0, 11.0),
            ),
        ]

    def test_implements_protocol(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        assert isinstance(source, SensorSource)

    def test_full_stream_yields_all_steps(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        rows = list(source.stream())
        assert len(rows) == 4

    def test_chronological_order(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        times = [t for t, _ in source.stream()]
        assert times == sorted(times)

    def test_nan_channel_is_omitted(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        rows = list(source.stream())
        # Third row has NaN for q_gas → Q_gas should not be in y.
        _, y3 = rows[2]
        assert "Q_gas" not in y3
        assert "pH" in y3
        assert y3["pH"] == pytest.approx(7.5)

    def test_out_of_range_channel_is_omitted(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        rows = list(source.stream())
        # Fourth row has pH=14.5 → out of range → should not be in y.
        _, y4 = rows[3]
        assert "pH" not in y4
        assert "Q_gas" in y4

    def test_unit_conversion_applied(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        rows = list(source.stream())
        # 250 Nm3/h → 6000 m3/d
        _, y0 = rows[0]
        assert y0["Q_gas"] == pytest.approx(6000.0)

    def test_start_t_filter(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        rows = list(source.stream(start_t=0.05))
        assert all(t >= 0.05 for t, _ in rows)
        assert len(rows) == 2  # 0.0833 and 0.125

    def test_end_t_filter_is_exclusive(self):
        source = DataFrameSensorSource(self._sample_df(), self._sample_defs())
        rows = list(source.stream(end_t=0.0833))
        assert all(t < 0.0833 for t, _ in rows)
        assert len(rows) == 2  # 0.0 and 0.0417

    def test_missing_column_raises(self):
        defs = [
            SensorChannelDef(
                db_column="this_column_does_not_exist",
                ukf_channel="X",
            ),
        ]
        with pytest.raises(ValueError, match="not in the DataFrame"):
            DataFrameSensorSource(self._sample_df(), defs)

    def test_missing_quality_column_raises(self):
        defs = [
            SensorChannelDef(
                db_column="fic101_q_gas_nm3h",
                ukf_channel="Q_gas",
                unit_in="Nm3/h",
                unit_out="m3/d",
                quality_column="not_a_real_column",
            ),
        ]
        with pytest.raises(ValueError, match="quality_column"):
            DataFrameSensorSource(self._sample_df(), defs)
