"""Unit tests for the high-level ``build_ukf`` convenience factory.

The factory itself is plant-agnostic (it only needs a ``BiogasPlant``-
like object with a ``components`` dict). To keep the test suite
independent of the example_plants module and the pyadm1 base package,
we use a tiny stub plant just for the construction tests, and skip
the simulation-roundtrip test when pyadm1 is missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1ode_estimation.estimation import (
    InputSpec,
    ObservationChannel,
    build_ukf,
)

# ---------------------------------------------------------------------------
# Stub plant for construction-only tests
# ---------------------------------------------------------------------------


class _StubDigester:
    """Mimics the minimal attributes ``ADM1ProcessModel`` reads."""

    def __init__(self):
        # 41-element ADM1 state with plausible non-zero values so the
        # tight-P0 finite-difference doesn't degenerate to all-zeros.
        self.adm1_state = list(np.linspace(0.01, 1.0, 41))
        self.Q_substrates = [0.0] * 10

        class _ADM1:
            _kinetic = {}

            def create_influent(self, q, k):
                pass

        self.adm1 = _ADM1()
        self.outputs_data = {"Q_gas": 0.0, "Q_ch4": 0.0, "pH": 7.0}
        self.gas_storage = None


class _StubPlant:
    def __init__(self):
        self.components = {"primary": _StubDigester()}
        self.simulation_time = 0.0

    def step(self, dt):  # pragma: no cover — never called by build_ukf
        pass


# ---------------------------------------------------------------------------
# Construction tests
# ---------------------------------------------------------------------------


class TestBuildUKFConstruction:
    """Verify the factory builds a ready-to-use UKF with sensible defaults."""

    def test_default_setup_with_two_substrates(self):
        plant = _StubPlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[
                InputSpec("maize_silage", substrate_index=0, initial_flow=10.0),
                InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
            ],
        )
        # 41 ADM1 channels + 2 substrate channels.
        assert len(ukf.spec) == 43
        # Default sensor set yields Q_gas + Q_ch4 + pH + 2 substrate_dose = 5.
        assert len(ukf.obs.channels) == 5
        sensor_names = [c.name for c in ukf.obs.channels]
        assert "Q_gas" in sensor_names
        assert "Q_ch4" in sensor_names
        assert "pH" in sensor_names
        assert "Q_maize_silage" in sensor_names
        assert "Q_cattle_slurry" in sensor_names

    def test_x_hat_initialised_to_plant_state(self):
        plant = _StubPlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 8.0)],
        )
        # ADM1 part of x_hat must match the plant's adm1_state.
        expected_adm1 = np.array(plant.components["primary"].adm1_state)
        np.testing.assert_allclose(ukf.x_hat[:41], expected_adm1, atol=1e-12)
        # Augmented substrate channel gets the spec's initial flow.
        assert ukf.x_hat[41] == pytest.approx(8.0)

    def test_p0_tight_matches_initial_uncertainty(self):
        plant = _StubPlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 10.0)],
            initial_uncertainty_relative=0.02,
        )
        # P0 must be diagonal with sigma = 2 % * |x|.
        std = np.sqrt(np.diag(ukf.P))
        expected_std = 0.02 * (np.abs(ukf.x_hat) + 1e-6)
        np.testing.assert_allclose(std, expected_std, rtol=1e-10)
        # Off-diagonal entries must all be zero.
        off_diag = ukf.P - np.diag(np.diag(ukf.P))
        np.testing.assert_array_equal(off_diag, 0.0)

    def test_unknown_digester_id_raises(self):
        plant = _StubPlant()
        with pytest.raises(KeyError, match="not a component"):
            build_ukf(plant, digester_id="nonexistent_id")

    def test_unknown_sensor_name_raises(self):
        plant = _StubPlant()
        with pytest.raises(ValueError, match="Unknown sensor name"):
            build_ukf(
                plant,
                digester_id="primary",
                sensors=["q_gas", "magic_sensor_that_doesnt_exist"],
            )

    def test_no_sensors_raises(self):
        plant = _StubPlant()
        # No default sensors AND empty list → no channels constructed.
        with pytest.raises(ValueError, match="at least one sensor"):
            build_ukf(
                plant,
                digester_id="primary",
                sensors=[],
            )

    def test_custom_observation_channel_passes_through(self):
        plant = _StubPlant()

        def extractor(plant, x):  # noqa: ARG001
            return 42.0

        custom = ObservationChannel(
            name="custom_signal",
            extractor=extractor,
            noise_std=1.0,
        )
        ukf = build_ukf(
            plant,
            digester_id="primary",
            sensors=["q_gas", custom],
        )
        names = [c.name for c in ukf.obs.channels]
        assert "Q_gas" in names
        assert "custom_signal" in names

    def test_sensor_noise_override(self):
        plant = _StubPlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            sensors=["q_gas"],
            sensor_noise={"q_gas": 25.0},
        )
        q_gas_channel = next(c for c in ukf.obs.channels if c.name == "Q_gas")
        assert q_gas_channel.noise_std == 25.0

    def test_substrate_dose_noise_is_relative_to_flow(self):
        plant = _StubPlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[
                InputSpec("big_substrate", 0, 20.0),
                InputSpec("small_substrate", 1, 1.0),
            ],
            sensors=["substrate_dose"],
            sensor_noise={"substrate_dose": 0.05},  # 5 %
        )
        big = next(c for c in ukf.obs.channels if c.name == "Q_big_substrate")
        small = next(c for c in ukf.obs.channels if c.name == "Q_small_substrate")
        assert big.noise_std == pytest.approx(1.0)  # 5 % of 20
        assert small.noise_std == pytest.approx(0.05)  # 5 % of 1
