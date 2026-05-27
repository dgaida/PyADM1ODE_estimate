"""Unit tests for the 41-state ADM1da spec factory.

Covers:
* Default factory yields exactly 41 ADM1 channels in index order
* Per-channel quality classification matches the observability findings
* OU channels carry valid ``ou_mean`` / ``ou_theta``
* Augmentations (InputSpec, KineticSpec) append correctly after the
  ADM1 block, preserving index ordering
* SensorQualityProfile overrides propagate to the per-channel noise scale
"""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1ode_estimation.estimation import (
    InputSpec,
    KineticSpec,
    Quality,
    SensorQualityProfile,
    adm1da_full_spec,
)


# ---------------------------------------------------------------------------
# Default spec
# ---------------------------------------------------------------------------


class TestDefaultSpec:
    @pytest.fixture
    def spec(self):
        return adm1da_full_spec(digester_id="D1")

    def test_has_exactly_41_adm1_channels(self, spec):
        adm1 = [c for c in spec.channels if c.kind == "adm1"]
        assert len(adm1) == 41

    def test_adm1_indices_are_0_to_40_in_order(self, spec):
        adm1 = [c for c in spec.channels if c.kind == "adm1"]
        indices = [c.adm1_index for c in adm1]
        assert indices == list(range(41))

    def test_no_augmentations_by_default(self, spec):
        # Only ADM1 channels — no input_flow, no kinetic_param
        kinds = {c.kind for c in spec.channels}
        assert kinds == {"adm1"}

    def test_strong_states_have_random_walk_drift(self, spec):
        # All A+D (methanogenesis + charge_balance) members
        strong_idx = {
            6,
            7,
            8,
            9,
            27,
            28,  # methanogenesis: S_ac, S_h2, S_ch4, S_co2, X_ac, X_h2
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,  # charge balance: S_cation..S_nh3
            37,
            38,
            39,
            40,  # gas pressures: p_gas_*, pTOTAL
        }
        for c in spec.channels:
            if c.kind == "adm1" and c.adm1_index in strong_idx:
                assert c.drift_model == "random_walk", (
                    f"Channel {c.name} (idx {c.adm1_index}) "
                    f"should be random_walk but is {c.drift_model}"
                )

    def test_open_loop_states_use_ou_drift(self, spec):
        # FA block + nitrogen + inerts → OU
        open_loop_idx = {2, 10, 11, 21, 24}  # S_fa, S_nh4, S_I, X_I, X_fa
        for c in spec.channels:
            if c.kind == "adm1" and c.adm1_index in open_loop_idx:
                assert c.drift_model == "ou"
                assert c.ou_mean is not None
                assert c.ou_theta > 0

    def test_pspf_states_use_ou_drift(self, spec):
        # PS/PF split (6 channels) → OU
        pspf_idx = {12, 13, 14, 15, 16, 17}
        for c in spec.channels:
            if c.kind == "adm1" and c.adm1_index in pspf_idx:
                assert c.drift_model == "ou"
                assert c.ou_mean == c.initial

    def test_bounds_are_physically_sensible(self, spec):
        for c in spec.channels:
            if c.kind == "adm1":
                assert c.lower >= 0.0, f"{c.name}: lower bound must be non-negative"
                assert c.upper > c.lower, f"{c.name}: upper must exceed lower"
                # initial must lie strictly inside the bounds
                assert (
                    c.lower <= c.initial <= c.upper
                ), f"{c.name}: initial {c.initial} outside [{c.lower}, {c.upper}]"

    def test_initial_std_and_process_noise_positive(self, spec):
        for c in spec.channels:
            assert c.initial_std > 0
            assert c.process_noise_std >= 0

    def test_initial_mean_has_shape_41(self, spec):
        x0 = spec.initial_mean()
        assert x0.shape == (41,)
        assert np.all(np.isfinite(x0))

    def test_initial_cov_is_diagonal_and_positive(self, spec):
        P0 = spec.initial_cov()
        assert P0.shape == (41, 41)
        # Diagonal: off-diagonal entries are zero
        np.testing.assert_array_equal(P0 - np.diag(np.diag(P0)), 0.0)
        # All diagonal entries strictly positive
        assert np.all(np.diag(P0) > 0)


# ---------------------------------------------------------------------------
# Quality-driven noise scaling
# ---------------------------------------------------------------------------


class TestQualityScaling:
    def test_strong_states_get_smaller_noise_than_weak(self):
        spec = adm1da_full_spec(digester_id="D1")
        s_ac = next(c for c in spec.channels if c.name == "S_ac")  # STRONG
        x_su = next(c for c in spec.channels if c.name == "X_su")  # WEAK

        # Normalise by initial magnitude to compare across states with
        # different scales: strong / weak factors are 0.05 / 0.30.
        ratio_strong = s_ac.process_noise_std / abs(s_ac.initial)
        ratio_weak = x_su.process_noise_std / abs(x_su.initial)
        assert ratio_strong < ratio_weak

    def test_override_lifts_acidogenesis_biomass_to_medium(self):
        # Plant with GC-FID → lift acidogenesis_biomass from WEAK to MEDIUM
        default_spec = adm1da_full_spec(digester_id="D1")
        gc_spec = adm1da_full_spec(
            digester_id="D1",
            sensor_quality=SensorQualityProfile(
                acidogenesis_biomass=Quality.MEDIUM,
            ),
        )
        # X_su is in acidogenesis_biomass — its noise should drop
        x_su_default = next(c for c in default_spec.channels if c.name == "X_su")
        x_su_gc = next(c for c in gc_spec.channels if c.name == "X_su")
        assert x_su_gc.process_noise_std < x_su_default.process_noise_std

    def test_override_does_not_affect_other_blocks(self):
        default_spec = adm1da_full_spec(digester_id="D1")
        gc_spec = adm1da_full_spec(
            digester_id="D1",
            sensor_quality=SensorQualityProfile(
                acidogenesis_biomass=Quality.MEDIUM,
            ),
        )
        s_ac_default = next(c for c in default_spec.channels if c.name == "S_ac")
        s_ac_gc = next(c for c in gc_spec.channels if c.name == "S_ac")
        assert s_ac_gc.process_noise_std == s_ac_default.process_noise_std


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------


class TestAugmentations:
    def test_input_specs_appended_after_adm1_block(self):
        spec = adm1da_full_spec(
            digester_id="D1",
            substrate_inputs=[
                InputSpec("maize", substrate_index=0, initial_flow=5.0),
                InputSpec("slurry", substrate_index=1, initial_flow=12.0),
            ],
        )
        assert len(spec.channels) == 43
        # First 41 are ADM1, last 2 are input_flow
        assert all(c.kind == "adm1" for c in spec.channels[:41])
        assert all(c.kind == "input_flow" for c in spec.channels[41:])
        assert spec.channels[41].name == "maize"
        assert spec.channels[42].name == "slurry"

    def test_input_spec_defaults_initial_std_from_flow(self):
        spec = adm1da_full_spec(
            digester_id="D1",
            substrate_inputs=[
                InputSpec("maize", substrate_index=0, initial_flow=10.0),
            ],
        )
        maize = spec.channels[-1]
        # initial_std default = 0.5 * abs(initial_flow) = 5.0
        assert maize.initial_std == 5.0

    def test_kinetic_overrides_appended_last(self):
        spec = adm1da_full_spec(
            digester_id="D1",
            substrate_inputs=[
                InputSpec("maize", substrate_index=0, initial_flow=5.0),
            ],
            kinetic_overrides=[
                KineticSpec(
                    name="k_dis_PS",
                    initial=0.04,
                    initial_std=0.01,
                    process_noise_std=0.001,
                ),
            ],
        )
        assert len(spec.channels) == 43
        assert spec.channels[41].kind == "input_flow"
        assert spec.channels[42].kind == "kinetic_param"
        assert spec.channels[42].name == "k_dis_PS"

    def test_kinetic_spec_defaults_to_ou_drift(self):
        spec = adm1da_full_spec(
            digester_id="D1",
            kinetic_overrides=[
                KineticSpec(
                    name="k_dis_PS",
                    initial=0.04,
                    initial_std=0.01,
                    process_noise_std=0.001,
                ),
            ],
        )
        kin = spec.channels[-1]
        assert kin.drift_model == "ou"
        assert kin.ou_mean == 0.04
