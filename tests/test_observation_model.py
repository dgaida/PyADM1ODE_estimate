"""Unit tests for ObservationChannel / ObservationModel."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyadm1ode_estimation.estimation import ObservationChannel, ObservationModel


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
