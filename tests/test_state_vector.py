"""Unit tests for StateChannel / StateVectorSpec."""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1ode_estimation.estimation import StateChannel, StateVectorSpec


class TestStateChannel:
    def test_adm1_channel_requires_index(self):
        with pytest.raises(ValueError, match="adm1_index is required"):
            StateChannel(name="X_ac", kind="adm1")

    def test_input_flow_requires_substrate_index(self):
        with pytest.raises(ValueError, match="input_substrate_index is required"):
            StateChannel(name="Q_solid", kind="input_flow", initial=30.0)

    def test_ou_requires_mean(self):
        with pytest.raises(ValueError, match="ou_mean is required"):
            StateChannel(
                name="Q_solid",
                kind="input_flow",
                input_substrate_index=0,
                drift_model="ou",
                initial=30.0,
            )

    def test_initial_std_must_be_positive(self):
        with pytest.raises(ValueError, match="initial_std must be > 0"):
            StateChannel(name="X", kind="adm1", adm1_index=0, initial_std=0.0)

    def test_bounds_must_be_valid(self):
        with pytest.raises(ValueError, match="lower .* > upper"):
            StateChannel(
                name="X",
                kind="adm1",
                adm1_index=0,
                initial_std=1.0,
                lower=5.0,
                upper=1.0,
            )


class TestStateVectorSpec:
    @pytest.fixture
    def spec(self) -> StateVectorSpec:
        return StateVectorSpec(
            digester_id="primary",
            channels=[
                StateChannel(
                    name="X_ac",
                    kind="adm1",
                    adm1_index=27,
                    initial=0.8,
                    initial_std=0.2,
                    process_noise_std=0.05,
                ),
                StateChannel(
                    name="S_ac",
                    kind="adm1",
                    adm1_index=6,
                    initial=0.2,
                    initial_std=0.1,
                    process_noise_std=0.03,
                ),
                StateChannel(
                    name="Q_solid",
                    kind="input_flow",
                    input_substrate_index=0,
                    initial=36.0,
                    initial_std=5.0,
                    process_noise_std=0.5,
                    lower=0.0,
                    upper=80.0,
                ),
            ],
        )

    def test_initial_mean(self, spec):
        x0 = spec.initial_mean()
        np.testing.assert_allclose(x0, [0.8, 0.2, 36.0])

    def test_initial_cov_is_diagonal(self, spec):
        P0 = spec.initial_cov()
        assert P0.shape == (3, 3)
        np.testing.assert_allclose(np.diag(P0), [0.04, 0.01, 25.0])
        # Off-diagonal must be zero
        assert P0[0, 1] == 0.0

    def test_process_noise_random_walk_scales_with_dt(self, spec):
        Q1 = spec.process_noise_cov(dt=1.0)
        Q2 = spec.process_noise_cov(dt=2.0)
        np.testing.assert_allclose(np.diag(Q2), 2 * np.diag(Q1))

    def test_clip_to_bounds(self, spec):
        x = np.array([-1.0, 100.0, 200.0])
        x_clipped = spec.clip(x)
        np.testing.assert_allclose(x_clipped, [0.0, 100.0, 80.0])

    def test_kind_indices(self, spec):
        assert spec.kind_indices("adm1") == [0, 1]
        assert spec.kind_indices("input_flow") == [2]
        assert spec.kind_indices("kinetic_param") == []

    def test_ou_process_noise_smaller_than_random_walk(self):
        rw = StateChannel(
            name="rw",
            kind="input_flow",
            input_substrate_index=0,
            initial=30.0,
            initial_std=5.0,
            process_noise_std=1.0,
            drift_model="random_walk",
        )
        ou = StateChannel(
            name="ou",
            kind="input_flow",
            input_substrate_index=1,
            initial=30.0,
            initial_std=5.0,
            process_noise_std=1.0,
            drift_model="ou",
            ou_mean=30.0,
            ou_theta=1.0,
        )
        spec = StateVectorSpec(digester_id="d", channels=[rw, ou])
        # OU saturates at σ² / (2θ); for σ=1, θ=1 → 0.5. RW at dt=1 → 1.
        Q = spec.process_noise_cov(dt=1.0)
        assert Q[0, 0] > Q[1, 1]
