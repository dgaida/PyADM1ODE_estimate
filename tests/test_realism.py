"""Unit tests for the literature-grounded realism foundation."""

from __future__ import annotations

import pytest

from pyadm1ode_estimation.estimation import realism


def test_relative_sensor_std_scales_with_magnitude():
    # q_gas is a 3 % relative spec.
    assert realism.absolute_sensor_std("q_gas", 2000.0) == pytest.approx(0.03 * 2000)
    assert realism.absolute_sensor_std("q_gas", 1000.0) == pytest.approx(0.03 * 1000)


def test_absolute_sensor_std_ignores_magnitude():
    # pH is an absolute 0.05 pH-unit spec, independent of the reading.
    assert realism.absolute_sensor_std("ph", 7.0) == pytest.approx(0.05)
    assert realism.absolute_sensor_std("ph", 8.5) == pytest.approx(0.05)


def test_unknown_sensor_raises():
    with pytest.raises(KeyError):
        realism.absolute_sensor_std("not_a_sensor", 1.0)


def test_build_sensor_noise_mixes_relative_absolute_and_factor():
    d = realism.build_sensor_noise({"q_gas": 1900.0, "q_ch4": 900.0, "q_co2": 850.0})
    assert d["q_gas"] == pytest.approx(0.03 * 1900.0)
    assert d["q_ch4"] == pytest.approx(0.04 * 900.0)
    assert d["ph"] == pytest.approx(0.05)  # absolute
    # substrate_dose is returned as the RELATIVE factor (build_ukf scales it).
    assert d["substrate_dose"] == pytest.approx(0.03)


def test_model_error_constants_are_in_literature_range():
    # ~10 % (sensitivity) to ~30 % (plant-model mismatch); default 0.25.
    assert 0.1 <= realism.MODEL_ERROR_KINETIC_SIGMA <= 0.35
    assert realism.MODEL_ERROR_PREFIXES == ("k_dis", "k_hyd", "k_m_", "k_dec", "K_S")
