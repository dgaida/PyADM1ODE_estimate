"""Linear-Gaussian UKF reference tests.

For a perfectly linear, Gaussian process and observation, the UKF
must reproduce the closed-form Kalman filter to within numerical
tolerance. That's the most direct sanity test for the sigma-point
machinery, independent of pyadm1.

The fake process / observation here implement the duck-typed
interface that :class:`UnscentedKalmanFilter` consumes (``.step``,
``.refresh_outputs``, ``.plant``, ``.spec`` on the process; ``.channels``,
``.predict``, ``.R``, ``.active_channels`` on the observation model).
"""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1ode_estimation.estimation import StateChannel, StateVectorSpec
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter
from pyadm1ode_estimation.estimation.observation_model import (
    ObservationChannel,
    ObservationModel,
)


class LinearProcess:
    """Constant transition matrix; no plant side effects.

    Implements the duck-typed interface ``UnscentedKalmanFilter`` calls:
    ``step``, ``refresh_outputs``, ``snapshot``, ``restore``, plus the
    ``plant`` / ``spec`` attributes. The latter two are inert because
    no plant integration happens in the linear-Gaussian test.
    """

    def __init__(self, spec, F):
        self.spec = spec
        self.F = np.asarray(F, dtype=float)
        self.plant = object()  # ducked — UKF only touches .plant via obs

    def step(self, x, dt):
        return self.F @ x

    def refresh_outputs(self, x, dt_stub=1e-5):  # noqa: ARG002
        return None

    def snapshot(self):
        return None

    def restore(self):
        return None


def make_linear_spec(
    n: int, initial: float, initial_std: float, process_std: float
) -> StateVectorSpec:
    channels = [
        StateChannel(
            name=f"x{i}",
            kind="adm1",
            adm1_index=i,
            initial=initial,
            initial_std=initial_std,
            process_noise_std=process_std,
            lower=-1e6,
            upper=1e6,
        )
        for i in range(n)
    ]
    return StateVectorSpec(digester_id="d", channels=channels)


def closed_form_kf(F, Q, H, R, x0, P0, y_seq):
    """Run a classical KF as a reference."""
    x, P = np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)
    xs, Ps = [], []
    for y in y_seq:
        x = F @ x
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        innov = y - H @ x
        x = x + K @ innov
        P = (np.eye(len(x)) - K @ H) @ P
        xs.append(x.copy())
        Ps.append(P.copy())
    return np.array(xs), np.array(Ps)


class TestUKFLinear:
    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(0)
        n = 3
        spec = make_linear_spec(
            n,
            initial=0.0,
            initial_std=1.0,
            process_std=0.1,
        )
        F = np.eye(n)  # random walk
        process = LinearProcess(spec, F)

        # Identity observations on the first two dims
        H = np.eye(n)[:2]
        noise_std = 0.2

        def make_extractor(i):
            return lambda plant, x: float(x[i])

        obs = ObservationModel(
            channels=[
                ObservationChannel(
                    name=f"y{i}", extractor=make_extractor(i), noise_std=noise_std
                )
                for i in range(2)
            ]
        )

        ukf = UnscentedKalmanFilter(process, obs, spec)

        # Build a measurement sequence: 10 random observations
        T = 10
        truth = rng.normal(0, 1, size=(T, n))
        y_seq = (H @ truth.T).T + rng.normal(0, noise_std, size=(T, 2))

        return ukf, spec, F, H, noise_std, y_seq

    def test_ukf_matches_classical_kf(self, setup):
        ukf, spec, F, H, noise_std, y_seq = setup

        # Reference KF with the exact same noise statistics
        Q = spec.process_noise_cov(dt=1.0)
        R = noise_std**2 * np.eye(2)
        x0 = spec.initial_mean()
        P0 = spec.initial_cov()
        xs_ref, Ps_ref = closed_form_kf(F, Q, H, R, x0, P0, y_seq)

        # UKF run with dt = 1.0 so Q matches
        for k, y in enumerate(y_seq):
            ukf.predict(dt=1.0)
            ukf.update({"y0": float(y[0]), "y1": float(y[1])}, t=float(k))

            # UKF posterior must agree with KF to ~1e-6
            np.testing.assert_allclose(ukf.x_hat, xs_ref[k], atol=1e-6)
            np.testing.assert_allclose(ukf.P, Ps_ref[k], atol=1e-6)

    def test_predict_only_inflates_covariance(self, setup):
        ukf, *_ = setup
        P_before = ukf.P.copy()
        ukf.predict(dt=1.0)
        # Random-walk dynamics ⇒ trace must grow strictly
        assert np.trace(ukf.P) > np.trace(P_before)

    def test_gated_channel_skipped_in_update(self, setup):
        # Build a fresh UKF with one gated channel and check it is
        # silently skipped when the gate is off.
        ukf, spec, _F, _H, noise_std, _y_seq = setup

        def ex(plant, x):  # noqa: ARG001
            return float(x[0])

        obs = ObservationModel(
            channels=[
                ObservationChannel(
                    name="y0", extractor=ex, noise_std=noise_std, gate_column="gate"
                ),
            ]
        )
        ukf2 = UnscentedKalmanFilter(ukf.process, obs, spec)
        # gate closed → no update, x_hat / P unchanged
        x_prev = ukf2.x_hat.copy()
        P_prev = ukf2.P.copy()
        step = ukf2.update({"y0": 5.0}, t=0.0, gate_values={"gate": 0.0})
        assert step.active_channels == []
        np.testing.assert_allclose(ukf2.x_hat, x_prev)
        np.testing.assert_allclose(ukf2.P, P_prev)

        # gate open → update happens, x_hat moves toward observation
        step = ukf2.update({"y0": 5.0}, t=0.0, gate_values={"gate": 1.0})
        assert step.active_channels == ["y0"]
        assert ukf2.x_hat[0] > x_prev[0]
