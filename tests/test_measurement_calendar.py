"""Unit tests for MeasurementCalendar and SampleRate.

Covers:
* SampleRate constructor parameter validation
* Validity-window arithmetic for online / daily / weekly / sporadic
* gate_values / measurements semantics at arbitrary t
* Forward-fill behaviour within the window
* Channels missing from the DataFrame stay gated off
* NaN / inf handling
* from_obs_model factory
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyadm1ode_estimation.estimation import (
    MeasurementCalendar,
    ObservationChannel,
    ObservationModel,
    SampleRate,
)

# ---------------------------------------------------------------------------
# SampleRate
# ---------------------------------------------------------------------------


class TestSampleRate:
    def test_online_5min_window_is_10min(self):
        rate = SampleRate.online(period_min=5.0)
        # 2 × 5 min / 1440 min/d = 10 / 1440
        assert rate.validity_window_d == pytest.approx(10.0 / 1440.0)

    def test_daily_window_is_one_day(self):
        rate = SampleRate.daily()
        assert rate.validity_window_d == pytest.approx(1.0)

    def test_weekly_window_is_seven_days(self):
        rate = SampleRate.weekly()
        assert rate.validity_window_d == pytest.approx(7.0)

    def test_sporadic_default_is_5_minutes(self):
        rate = SampleRate.sporadic()
        assert rate.validity_window_d == pytest.approx(5.0 / 1440.0)

    def test_negative_window_rejected(self):
        with pytest.raises(ValueError, match="must be ≥ 0"):
            SampleRate(validity_window_d=-1.0)

    def test_zero_period_rejected(self):
        with pytest.raises(ValueError, match="must be > 0"):
            SampleRate.online(period_min=0.0)
        with pytest.raises(ValueError, match="must be > 0"):
            SampleRate.periodic(period_h=0.0)


# ---------------------------------------------------------------------------
# Calendar core behaviour
# ---------------------------------------------------------------------------


class TestCalendarCore:
    def _df(self, values: dict) -> pd.DataFrame:
        """Build a sparse DataFrame: each column may have NaN gaps."""
        return pd.DataFrame(values)

    def test_active_when_value_inside_window(self):
        cal = MeasurementCalendar({"Q_gas": SampleRate.online(period_min=5.0)})
        # Measurement at t=10.0 d, window is 10 min ≈ 0.00694 d.
        df = pd.DataFrame({"Q_gas": [42.0]}, index=[10.0])
        # Query right at t=10.0 d → active.
        gates = cal.gate_values_at(t=10.0, df=df)
        assert gates["Q_gas"] == 1.0

    def test_inactive_when_no_value_in_window(self):
        cal = MeasurementCalendar({"Q_gas": SampleRate.online(period_min=5.0)})
        # Last measurement at t=9.0 d; window only 10 min.
        df = pd.DataFrame({"Q_gas": [42.0]}, index=[9.0])
        # Query at t=10.0 d → 1 day later → outside window.
        gates = cal.gate_values_at(t=10.0, df=df)
        assert gates["Q_gas"] == 0.0

    def test_forward_fill_within_window(self):
        # Daily measurement at t=8.0 should still be valid at t=8.5 d.
        cal = MeasurementCalendar({"FOS_TAC": SampleRate.daily()})
        df = pd.DataFrame({"FOS_TAC": [0.3]}, index=[8.0])
        gates = cal.gate_values_at(t=8.5, df=df)
        assert gates["FOS_TAC"] == 1.0

    def test_picks_most_recent_value(self):
        # Two daily measurements; calendar should return the newer one.
        cal = MeasurementCalendar({"FOS_TAC": SampleRate.daily()})
        df = pd.DataFrame({"FOS_TAC": [0.3, 0.5]}, index=[8.0, 8.5])
        meas = cal.measurements_at(t=8.6, df=df)
        assert meas["FOS_TAC"] == pytest.approx(0.5)

    def test_ignores_future_measurements(self):
        cal = MeasurementCalendar({"FOS_TAC": SampleRate.daily()})
        df = pd.DataFrame({"FOS_TAC": [0.5]}, index=[10.0])
        # Query at t=9.0 d — measurement is in the future, must not be picked.
        gates = cal.gate_values_at(t=9.0, df=df)
        meas = cal.measurements_at(t=9.0, df=df)
        assert gates["FOS_TAC"] == 0.0
        assert np.isnan(meas["FOS_TAC"])

    def test_nan_values_in_column_are_skipped(self):
        cal = MeasurementCalendar({"NH4_N": SampleRate.sporadic()})
        # Most-recent finite value before t=10.0 is at t=10.0.
        df = pd.DataFrame({"NH4_N": [120.0, np.nan]}, index=[10.0, 10.003])
        # tol_default ≈ 5 min ≈ 0.00347 d.
        meas = cal.measurements_at(t=10.003, df=df)
        # At t=10.003, both entries are inside the window (which ends at
        # 10.003). The NaN at index=10.003 is dropped, so the finite
        # value at index=10.0 is returned.
        assert meas["NH4_N"] == pytest.approx(120.0)

    def test_inf_values_are_treated_as_missing(self):
        cal = MeasurementCalendar({"X": SampleRate.daily()})
        df = pd.DataFrame({"X": [np.inf, 5.0]}, index=[8.0, 8.5])
        meas = cal.measurements_at(t=8.6, df=df)
        # Most recent finite value: 5.0 at t=8.5.
        assert meas["X"] == pytest.approx(5.0)

    def test_inf_only_yields_no_measurement(self):
        cal = MeasurementCalendar({"X": SampleRate.daily()})
        df = pd.DataFrame({"X": [np.inf]}, index=[8.0])
        gates = cal.gate_values_at(t=8.5, df=df)
        assert gates["X"] == 0.0

    def test_channel_missing_from_dataframe_stays_off(self):
        cal = MeasurementCalendar(
            {
                "Q_gas": SampleRate.online(period_min=5.0),
                "NH4_N": SampleRate.sporadic(),
            }
        )
        df = pd.DataFrame({"Q_gas": [40.0]}, index=[10.0])
        gates = cal.gate_values_at(t=10.0, df=df)
        assert gates["Q_gas"] == 1.0
        assert gates["NH4_N"] == 0.0


# ---------------------------------------------------------------------------
# values_for_filter — combined access
# ---------------------------------------------------------------------------


class TestValuesForFilter:
    def test_active_channel_returns_value_and_gate_one(self):
        cal = MeasurementCalendar({"Q_gas": SampleRate.online(period_min=5.0)})
        df = pd.DataFrame({"Q_gas": [42.0]}, index=[10.0])
        y, gates = cal.values_for_filter(t=10.0, df=df)
        assert gates["Q_gas"] == 1.0
        assert y["Q_gas"] == pytest.approx(42.0)

    def test_inactive_channel_returns_nan_and_gate_zero(self):
        cal = MeasurementCalendar({"Q_gas": SampleRate.online(period_min=5.0)})
        df = pd.DataFrame({"Q_gas": [42.0]}, index=[8.0])  # 2 days ago
        y, gates = cal.values_for_filter(t=10.0, df=df)
        assert gates["Q_gas"] == 0.0
        assert np.isnan(y["Q_gas"])


# ---------------------------------------------------------------------------
# from_obs_model factory
# ---------------------------------------------------------------------------


class TestFromObsModel:
    def _obs(self):
        def ex(plant, x):
            return 0.0

        return ObservationModel(
            channels=[
                ObservationChannel(
                    name="Q_gas", extractor=ex, noise_std=1.0, gate_column="Q_gas"
                ),
                ObservationChannel(
                    name="FOS_TAC", extractor=ex, noise_std=0.1, gate_column="FOS_TAC"
                ),
                ObservationChannel(
                    name="NH4_N", extractor=ex, noise_std=10.0, gate_column="NH4_N"
                ),
            ]
        )

    def test_factory_assigns_default_rates(self):
        cal = MeasurementCalendar.from_obs_model(
            self._obs(),
            default_rates={
                "Q_gas": SampleRate.online(period_min=5.0),
                "FOS_TAC": SampleRate.daily(),
                "NH4_N": SampleRate.sporadic(),
            },
        )
        assert cal.rates["Q_gas"].validity_window_d == pytest.approx(10.0 / 1440.0)
        assert cal.rates["FOS_TAC"].validity_window_d == pytest.approx(1.0)
        assert cal.rates["NH4_N"].validity_window_d == pytest.approx(5.0 / 1440.0)

    def test_factory_falls_back_to_sporadic(self):
        # No defaults supplied → every channel gets the fallback.
        cal = MeasurementCalendar.from_obs_model(self._obs())
        for name in ("Q_gas", "FOS_TAC", "NH4_N"):
            assert cal.rates[name].validity_window_d == pytest.approx(5.0 / 1440.0)

    def test_factory_uses_channel_name_when_gate_column_missing(self):
        # Channel without gate_column → keyed by .name
        def ex(plant, x):
            return 0.0

        obs = ObservationModel(
            channels=[
                ObservationChannel(name="Q_gas", extractor=ex, noise_std=1.0),
            ]
        )
        cal = MeasurementCalendar.from_obs_model(
            obs,
            default_rates={"Q_gas": SampleRate.online(period_min=5.0)},
        )
        assert "Q_gas" in cal.rates
        assert cal.rates["Q_gas"].validity_window_d == pytest.approx(10.0 / 1440.0)
