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


class _GasDigester(_StubDigester):
    """Stub digester with a fixed per-stage gas/methane output.

    ``component_type.value == "digester"`` so ``extract_q_*_total`` (which
    iterates every digester) counts it; ``outputs_data`` carries the
    per-stage ``Q_gas`` / ``Q_ch4`` that the stage-scoped extractors read.
    """

    component_type = type("CT", (), {"value": "digester"})()

    def __init__(self, q_gas: float, q_ch4: float):
        super().__init__()
        self.outputs_data = {"Q_gas": q_gas, "Q_ch4": q_ch4, "pH": 7.0}


class _TwoStagePlant:
    """Primary + secondary cascade so per-stage vs. whole-plant gas
    extractors give visibly different numbers."""

    def __init__(self):
        self.components = {
            "primary": _GasDigester(1900.0, 1000.0),
            "secondary": _GasDigester(500.0, 260.0),
        }
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

    def test_reduced_state_vector_via_adm1_indices(self):
        """``adm1_indices`` shrinks the ADM1 block to the requested subset;
        the augmented substrate channels still follow, and x0 is initialised
        correctly despite the augmented block no longer starting at 41."""
        from pyadm1ode_estimation.estimation import BLOCK_INDICES

        plant = _StubPlant()
        ad_core = sorted(
            BLOCK_INDICES["methanogenesis"] + BLOCK_INDICES["charge_balance"]
        )
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 8.0)],
            sensors=["q_gas"],
            adm1_indices=ad_core,
        )
        # 18 ADM1 (A+D core) + 1 substrate channel.
        assert len(ukf.spec) == len(ad_core) + 1
        adm1_chans = [c for c in ukf.spec.channels if c.kind == "adm1"]
        assert [c.adm1_index for c in adm1_chans] == ad_core
        # The single augmented channel sits at position 18 (not 41) and is
        # initialised to its spec initial flow, not left at 0.
        assert ukf.spec.channels[-1].kind == "input_flow"
        assert ukf.x_hat[len(ad_core)] == pytest.approx(8.0)
        # The estimated ADM1 part matches the plant's state at those indices.
        plant_state = np.array(plant.components["primary"].adm1_state)
        np.testing.assert_allclose(
            ukf.x_hat[: len(ad_core)], plant_state[ad_core], atol=1e-12
        )

    def test_reduced_spec_rejects_out_of_range_index(self):
        from pyadm1ode_estimation.estimation import adm1da_reduced_spec

        with pytest.raises(ValueError, match=r"\[0, 40\]"):
            adm1da_reduced_spec("primary", [6, 41])

    def test_process_noise_scale_raises_model_distrust(self):
        """``process_noise_scale`` multiplies the ADM1 process-noise std, so
        the process-noise *variance* scales with its square. Augmented input
        channels are left untouched."""
        plant = _StubPlant()
        base = build_ukf(
            plant, digester_id="primary", substrates=[InputSpec("maize", 0, 8.0)]
        )
        tuned = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 8.0)],
            process_noise_scale=10.0,
        )
        dt = 1.0 / 24.0
        q_base = np.diag(base.spec.process_noise_cov(dt))
        q_tuned = np.diag(tuned.spec.process_noise_cov(dt))
        # ADM1 channels (first 41): variance up by 10² = 100x.
        adm1 = base.spec.kind_indices("adm1")
        np.testing.assert_allclose(q_tuned[adm1], 100.0 * q_base[adm1], rtol=1e-9)
        # Augmented input channel: unchanged.
        inp = base.spec.kind_indices("input_flow")
        np.testing.assert_allclose(q_tuned[inp], q_base[inp], rtol=1e-9)

    def test_co2_and_vfa_sensors_resolve(self):
        """The ``q_co2`` and ``vfa`` catalog names add Q_co2 / VFA channels
        scoped to the estimated digester."""
        plant = _StubPlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 8.0)],
            sensors=["q_co2", "vfa"],
        )
        names = [c.name for c in ukf.obs.channels]
        assert "Q_co2" in names
        assert "VFA" in names

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

    def test_q_gas_q_ch4_are_scoped_to_estimated_digester(self):
        """``q_gas``/``q_ch4`` must read ONLY the estimated digester's own
        production, not the plant total. In a cascade the downstream
        stages produce gas the filter does not estimate; mixing it into
        the innovation would bias the estimated stage."""
        plant = _TwoStagePlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 8.0)],
            sensors=["q_gas", "q_ch4"],
        )
        q_gas = next(c for c in ukf.obs.channels if c.name == "Q_gas")
        q_ch4 = next(c for c in ukf.obs.channels if c.name == "Q_ch4")
        # primary alone: 1900 / 1000, NOT the 2400 / 1260 plant totals.
        assert q_gas.extractor(plant, ukf.x_hat) == pytest.approx(1900.0)
        assert q_ch4.extractor(plant, ukf.x_hat) == pytest.approx(1000.0)

    def test_q_gas_total_still_sums_every_digester(self):
        """The explicit ``q_gas_total`` / ``q_ch4_total`` catalog names
        preserve the whole-plant sum for meters that sit downstream of
        all stages."""
        plant = _TwoStagePlant()
        ukf = build_ukf(
            plant,
            digester_id="primary",
            substrates=[InputSpec("maize", 0, 8.0)],
            sensors=["q_gas_total", "q_ch4_total"],
        )
        q_gas = next(c for c in ukf.obs.channels if c.name == "Q_gas")
        q_ch4 = next(c for c in ukf.obs.channels if c.name == "Q_ch4")
        assert q_gas.extractor(plant, ukf.x_hat) == pytest.approx(2400.0)
        assert q_ch4.extractor(plant, ukf.x_hat) == pytest.approx(1260.0)

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
