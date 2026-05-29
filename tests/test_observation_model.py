"""Unit tests for ObservationChannel / ObservationModel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyadm1ode_estimation.estimation import ObservationChannel, ObservationModel
from pyadm1ode_estimation.estimation.observation_model import (
    extract_q_gas_total,
    make_state_extractor,
)


class _FakeDigester:
    component_type = type("CT", (), {"value": "digester"})()

    def __init__(self, outputs):
        self.outputs_data = outputs


class _FakePlant:
    def __init__(self, outputs):
        self.components = {"d": _FakeDigester(outputs)}


class TestExtractorNonFinitePropagates:
    # Extractors return the raw prediction, including non-finite values, so
    # the UKF can drop an unreliable channel for that step rather than
    # correcting the state with a fudged number.
    def test_q_gas_nan_propagates(self):
        plant = _FakePlant({"Q_gas": float("nan")})
        assert math.isnan(extract_q_gas_total(plant, np.zeros(1)))

    def test_q_gas_finite_passes_through(self):
        plant = _FakePlant({"Q_gas": 412.0})
        assert extract_q_gas_total(plant, np.zeros(1)) == 412.0

    def test_state_extractor_nan_propagates(self):
        ex = make_state_extractor(2)
        assert math.isnan(ex(None, np.array([1.0, 2.0, float("nan")])))


def trivial_extractor(plant, x):
    """Returns the first state element regardless of plant."""
    return float(x[0])


class TestObservationChannel:
    def test_active_when_no_gate(self):
        c = ObservationChannel(name="y", extractor=trivial_extractor, noise_std=1.0)
        assert c.is_active(None) is True

    def test_inactive_when_gate_value_missing(self):
        c = ObservationChannel(
            name="y", extractor=trivial_extractor, noise_std=1.0, gate_column="gate"
        )
        assert c.is_active(None) is False

    def test_truthy_gate(self):
        c = ObservationChannel(
            name="y",
            extractor=trivial_extractor,
            noise_std=1.0,
            gate_column="gate",
            gate_predicate="truthy",
        )
        assert c.is_active(1.0) is True
        assert c.is_active(0.0) is False

    def test_finite_gate(self):
        c = ObservationChannel(
            name="y",
            extractor=trivial_extractor,
            noise_std=1.0,
            gate_column="gate",
            gate_predicate="finite",
        )
        assert c.is_active(1.0) is True
        assert c.is_active(0.0) is True
        assert c.is_active(float("nan")) is False

    def test_nan_gate_value_rejected_regardless_of_predicate(self):
        c = ObservationChannel(
            name="y", extractor=trivial_extractor, noise_std=1.0, gate_column="gate"
        )
        assert c.is_active(math.nan) is False

    def test_unknown_predicate_raises(self):
        c = ObservationChannel(
            name="y",
            extractor=trivial_extractor,
            noise_std=1.0,
            gate_column="gate",
            gate_predicate="unknown_mode",
        )
        with pytest.raises(ValueError, match="Unknown gate_predicate"):
            c.is_active(1.0)


class TestObservationModel:
    @pytest.fixture
    def model(self):
        return ObservationModel(
            channels=[
                ObservationChannel(
                    name="y1", extractor=trivial_extractor, noise_std=1.0
                ),
                ObservationChannel(
                    name="y2",
                    extractor=lambda p, x: float(x[1]),
                    noise_std=2.0,
                    gate_column="g2",
                ),
            ]
        )

    def test_duplicate_channel_names_rejected(self):
        c1 = ObservationChannel(name="y", extractor=trivial_extractor, noise_std=1.0)
        c2 = ObservationChannel(name="y", extractor=trivial_extractor, noise_std=1.0)
        with pytest.raises(ValueError, match="Duplicate channel names"):
            ObservationModel(channels=[c1, c2])

    def test_active_channels_filters_gated(self, model):
        active = model.active_channels({"g2": 0.0})
        names = [c.name for c in active]
        assert "y1" in names and "y2" not in names

    def test_active_channels_includes_when_gate_truthy(self, model):
        active = model.active_channels({"g2": 1.0})
        names = [c.name for c in active]
        assert names == ["y1", "y2"]

    def test_predict_returns_active_order(self, model):
        active = model.active_channels({"g2": 1.0})
        y = model.predict(plant=None, x=np.array([7.0, 13.0]), active=active)
        np.testing.assert_allclose(y, [7.0, 13.0])

    def test_R_diagonal_matches_noise(self, model):
        active = model.active_channels({"g2": 1.0})
        R = model.R(active=active)
        np.testing.assert_allclose(np.diag(R), [1.0, 4.0])


class TestMatchMeasurements:
    @pytest.fixture
    def model(self):
        ex = lambda p, x: 0.0  # noqa: E731
        names = ["Q_gas", "Q_ch4", "pH", "Q_maize_silage", "Q_slurry"]
        return ObservationModel(
            channels=[ObservationChannel(n, ex, 1.0) for n in names]
        )

    def test_exact_names_pass_through(self, model):
        assert model.match_measurements({"Q_gas": 1.0, "pH": 7.0}) == {
            "Q_gas": 1.0,
            "pH": 7.0,
        }

    def test_case_and_separator_insensitive(self, model):
        out = model.match_measurements({"q gas": 1.0, "PH": 7.0, "Q-ch4": 2.0})
        assert out == {"Q_gas": 1.0, "pH": 7.0, "Q_ch4": 2.0}

    def test_optional_q_prefix(self, model):
        # build_ukf-style bare names also resolve to the Q_ channels.
        out = model.match_measurements({"gas": 1.0, "maize_silage": 3.0})
        assert out == {"Q_gas": 1.0, "Q_maize_silage": 3.0}

    def test_unknown_and_nan_keys_ignored(self, model):
        out = model.match_measurements(
            {"q_gas": 1.0, "not_a_sensor": 9.0, "pH": float("nan")}
        )
        assert out == {"Q_gas": 1.0}

    def test_ambiguous_user_key_prefers_exact(self, model):
        # Both "Q_gas" and bare "gas" given → exact match wins, no clash.
        out = model.match_measurements({"Q_gas": 1.0, "gas": 99.0})
        assert out["Q_gas"] == 1.0
