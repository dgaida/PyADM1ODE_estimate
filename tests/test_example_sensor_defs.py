"""Smoke tests for the example SCADA sensor-def builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyadm1ode_estimation.example_plants import (
    build_example_sensor_defs,
    example_scada_columns,
)
from pyadm1ode_estimation.io import DataFrameSensorSource, SensorChannelDef

# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------


class TestBuildExampleSensorDefs:
    def test_returns_list_of_sensor_channel_def(self):
        defs = build_example_sensor_defs()
        assert isinstance(defs, list)
        assert len(defs) >= 6
        assert all(isinstance(d, SensorChannelDef) for d in defs)

    def test_channel_names_match_build_ukf_catalog(self):
        # The built-in catalog produces these UKF channels for the
        # example plant's substrates. The example defs must hit the same
        # names so build_ukf + DataFrameSensorSource line up.
        defs = build_example_sensor_defs()
        names = {d.ukf_channel for d in defs}
        assert {"Q_gas", "Q_ch4", "pH"}.issubset(names)
        assert {
            "Q_maize_silage",
            "Q_solid_manure",
            "Q_chicken_litter",
            "Q_slurry",
            "Q_cereal_grain",
        }.issubset(names)

    def test_quality_columns_are_declared_for_every_channel(self):
        # Realistic SCADA always has a quality flag — verify the
        # example matches the protocol's recommended pattern.
        defs = build_example_sensor_defs()
        for d in defs:
            assert d.quality_column is not None
            assert d.quality_column.endswith("_quality")

    def test_density_override_changes_converter(self):
        # Doubling the density halves the m³/d output for the same kg/h.
        defs_default = build_example_sensor_defs()
        defs_dense = build_example_sensor_defs(rho_maize_kg_m3=1400.0)

        maize_default = next(
            d for d in defs_default if d.ukf_channel == "Q_maize_silage"
        )
        maize_dense = next(d for d in defs_dense if d.ukf_channel == "Q_maize_silage")

        # quality_column is set, so transform() needs a non-bad quality
        # value — the example's bad-status set includes None.
        v_default = maize_default.transform(700.0, quality=True)
        v_dense = maize_dense.transform(700.0, quality=True)
        assert v_default == pytest.approx(24.0)  # 700 kg/h / 700 kg/m³ * 24 h/d
        assert v_dense == pytest.approx(12.0)


class TestExampleScadaColumns:
    def test_every_def_db_column_is_documented(self):
        defs = build_example_sensor_defs()
        columns = example_scada_columns()
        for d in defs:
            assert d.db_column in columns
            assert d.quality_column in columns


# ---------------------------------------------------------------------------
# End-to-end through DataFrameSensorSource
# ---------------------------------------------------------------------------


class TestExampleDefsWithDataFrameSource:
    def _sample_df(self):
        # Two timestamps, 1-hour apart, all channels Good.
        return pd.DataFrame(
            {
                # gas line
                "fic101_q_gas_nm3h": [250.0, 252.0],
                "fic101_quality": ["Good", "Good"],
                "fic102_q_ch4_nm3h": [130.0, 132.0],
                "fic102_quality": ["Good", "Good"],
                # pH
                "ait201_ph": [7.4, 7.3],
                "ait201_quality": ["Good", "Good"],
                # substrate dosing
                "wit301_maize_kg_h": [700.0, 700.0],
                "wit301_quality": [True, True],
                "fic401_slurry_m3h": [0.5, 0.5],
                "fic401_quality": [True, True],
                "wit302_cereal_kg_h": [60.0, 60.0],
                "wit302_quality": [True, True],
                "wit303_manure_kg_h": [500.0, 500.0],
                "wit303_quality": [True, True],
                "wit304_htk_kg_h": [40.0, 40.0],
                "wit304_quality": [True, True],
            },
            index=[0.0, 1.0 / 24.0],
        )

    def test_full_pipeline_yields_all_channels(self):
        source = DataFrameSensorSource(self._sample_df(), build_example_sensor_defs())
        rows = list(source.stream())
        assert len(rows) == 2

        _, y0 = rows[0]
        # All 8 UKF channels should be populated.
        assert set(y0) == {
            "Q_gas",
            "Q_ch4",
            "pH",
            "Q_maize_silage",
            "Q_solid_manure",
            "Q_chicken_litter",
            "Q_slurry",
            "Q_cereal_grain",
        }

    def test_unit_conversions_round_trip(self):
        source = DataFrameSensorSource(self._sample_df(), build_example_sensor_defs())
        _, y0 = next(iter(source.stream()))

        # 250 Nm³/h → 6000 m³/d
        assert y0["Q_gas"] == pytest.approx(6000.0)
        # 130 Nm³/h → 3120 m³/d
        assert y0["Q_ch4"] == pytest.approx(3120.0)
        # 700 kg/h ÷ 700 kg/m³ × 24 = 24 m³/d
        assert y0["Q_maize_silage"] == pytest.approx(24.0)
        # 0.5 m³/h × 24 = 12 m³/d
        assert y0["Q_slurry"] == pytest.approx(12.0)
        # 60 kg/h ÷ 600 kg/m³ × 24 = 2.4 m³/d
        assert y0["Q_cereal_grain"] == pytest.approx(2.4)
        # 500 kg/h ÷ 1000 kg/m³ × 24 = 12 m³/d
        assert y0["Q_solid_manure"] == pytest.approx(12.0)
        # 40 kg/h ÷ 750 kg/m³ × 24 = 1.28 m³/d
        assert y0["Q_chicken_litter"] == pytest.approx(1.28)

    def test_bad_status_drops_channel(self):
        df = self._sample_df()
        df.at[0.0, "fic101_quality"] = "Bad"
        df.at[0.0, "wit301_quality"] = False  # PLC diagnostic bit

        source = DataFrameSensorSource(df, build_example_sensor_defs())
        _, y0 = next(iter(source.stream()))

        assert "Q_gas" not in y0
        assert "Q_maize_silage" not in y0
        # Unrelated channels still pass through.
        assert "Q_ch4" in y0
        assert "Q_slurry" in y0
        assert "pH" in y0

    def test_nan_reading_drops_channel(self):
        df = self._sample_df()
        df.at[0.0, "wit301_maize_kg_h"] = np.nan

        source = DataFrameSensorSource(df, build_example_sensor_defs())
        _, y0 = next(iter(source.stream()))

        assert "Q_maize_silage" not in y0
        assert "Q_slurry" in y0  # other doses unaffected

    def test_out_of_range_ph_drops_channel(self):
        df = self._sample_df()
        df.at[0.0, "ait201_ph"] = 13.5  # probe coating / wraparound

        source = DataFrameSensorSource(df, build_example_sensor_defs())
        _, y0 = next(iter(source.stream()))

        assert "pH" not in y0
