"""Integration tests for ``ConstrainedUKF`` (Hellmann 2024 cUKF-add).

Covers three properties that together define correctness:

1. With wide bounds and a linear h, the cUKF posterior approaches the
   plain SR-UKF posterior to within the linearisation tolerance
   (``J_h`` fitted from the propagated sigma cloud).
2. With tight bounds the cUKF posterior stays inside the box at every
   step — clipping alone (the parent's ``spec.clip()``) couldn't make
   that guarantee on the *sigma points* themselves.
3. End-to-end on the actual ADM1 plant: the constrained variant
   completes a short twin without crashing and produces a finite,
   bounded posterior.
"""

from __future__ import annotations


import numpy as np
import pytest

from pyadm1ode_estimation.estimation import StateChannel, StateVectorSpec
from pyadm1ode_estimation.estimation.filters import (
    ConstrainedUKF,
    UnscentedKalmanFilter,
)
from pyadm1ode_estimation.estimation.observation_model import (
    ObservationChannel,
    ObservationModel,
)


# --------------------------------------------------------------------------
# Linear toy fixtures (no plant) so we can compare against the standard
# Kalman filter / SR-UKF behaviour.
# --------------------------------------------------------------------------
class LinearProcess:
    def __init__(self, spec, F):
        self.spec = spec
        self.F = np.asarray(F, dtype=float)
        self.plant = object()

    def step(self, x, dt):  # noqa: ARG002
        return self.F @ x

    def refresh_outputs(self, x, equilibration_dt=1.0 / 24.0):  # noqa: ARG002
        return None

    def snapshot(self):
        return None

    def restore(self):
        return None


def _make_linear_spec(n, *, lower, upper, process_std=1e-6):
    return StateVectorSpec(
        digester_id="d",
        channels=[
            StateChannel(
                name=f"x{i}",
                kind="adm1",
                adm1_index=i,
                initial=0.5,
                initial_std=1.0,
                process_noise_std=process_std,
                lower=lower,
                upper=upper,
            )
            for i in range(n)
        ],
    )


def _make_obs(n_meas):
    def ex(i):
        return lambda plant, x: float(x[i])

    return ObservationModel(
        channels=[
            ObservationChannel(name=f"y{i}", extractor=ex(i), noise_std=0.2)
            for i in range(n_meas)
        ]
    )


def _run_cycle(ukf, y_seq):
    """Step the filter through a sequence of measurements, returning
    ``(x_hat, S)`` history."""
    xs, Ss = [], []
    for k, y in enumerate(y_seq):
        ukf.predict(dt=1.0)
        ukf.update({f"y{i}": float(y[i]) for i in range(len(y))}, t=float(k))
        xs.append(ukf.x_hat.copy())
        Ss.append(ukf.S.copy())
    return np.array(xs), np.array(Ss)


# --------------------------------------------------------------------------
# Wide-bounds: cUKF should reproduce the SR-UKF up to QP tolerance
# --------------------------------------------------------------------------
def test_cukf_matches_ukf_with_wide_bounds():
    """With bounds well outside the operating range, no constraint is
    ever active. The QP per sigma reduces to an unconstrained
    least-squares solve and must agree with the standard SR-UKF
    posterior to within the linearised-h tolerance.

    The linearisation IS exact for the identity observation model used
    here (``h(x) = x[i]`` is linear), so the match is tight: ``atol=
    1e-6``.
    """
    n = 3
    rng = np.random.default_rng(0)
    spec = _make_linear_spec(n, lower=-1e6, upper=1e6)
    F = np.eye(n)
    obs = _make_obs(2)

    base = UnscentedKalmanFilter(LinearProcess(spec, F), obs, spec)
    cons = ConstrainedUKF(LinearProcess(spec, F), obs, spec)

    T = 5
    truth = rng.normal(0, 0.5, size=(T, n))
    y_seq = truth[:, :2] + 0.1 * rng.standard_normal(size=(T, 2))

    x_base, S_base = _run_cycle(base, y_seq)
    x_cons, S_cons = _run_cycle(cons, y_seq)

    np.testing.assert_allclose(x_cons, x_base, atol=1e-6)
    # Cholesky factor: only the lower triangle is meaningful, and only
    # up to a sign convention on the diagonal. Compare via P = S Sᵀ.
    P_base = np.einsum("kij,klj->kil", S_base, S_base)
    P_cons = np.einsum("kij,klj->kil", S_cons, S_cons)
    np.testing.assert_allclose(P_cons, P_base, atol=1e-6)


# --------------------------------------------------------------------------
# Tight bounds: posterior must stay inside the box
# --------------------------------------------------------------------------
def test_cukf_respects_lower_bound_when_measurement_pulls_negative():
    """Drive the prior near zero, then feed measurements that would
    pull the unconstrained posterior negative. The cUKF must keep
    ``x_hat[i] ≥ 0`` for every observed channel."""
    n = 3
    spec = _make_linear_spec(n, lower=0.0, upper=10.0)
    F = np.eye(n)
    obs = _make_obs(2)

    cons = ConstrainedUKF(LinearProcess(spec, F), obs, spec)
    # Strongly negative measurements relative to the prior at x=0.5.
    y_seq = np.tile(np.array([-5.0, -5.0]), (10, 1))
    x_hist, _ = _run_cycle(cons, y_seq)

    # Posterior mean of OBSERVED channels never dips below 0.
    assert np.all(x_hist[:, 0] >= 0.0), x_hist[:, 0]
    assert np.all(x_hist[:, 1] >= 0.0), x_hist[:, 1]


def test_cukf_respects_upper_bound_when_measurement_pulls_high():
    n = 3
    spec = _make_linear_spec(n, lower=-1e6, upper=1.0)
    F = np.eye(n)
    obs = _make_obs(2)
    cons = ConstrainedUKF(LinearProcess(spec, F), obs, spec)
    y_seq = np.tile(np.array([100.0, 100.0]), (5, 1))
    x_hist, _ = _run_cycle(cons, y_seq)
    assert np.all(x_hist[:, 0] <= 1.0 + 1e-8), x_hist[:, 0]
    assert np.all(x_hist[:, 1] <= 1.0 + 1e-8), x_hist[:, 1]


# --------------------------------------------------------------------------
# End-to-end through ADM1ProcessModel — uses the picklable mock plant so the
# tests run without pyadm1's substrate YAML catalog (which CI environments
# typically lack). Real-plant smoke testing is the bench script's job.
# --------------------------------------------------------------------------
from _mock_plant import build_mock_components  # noqa: E402


@pytest.mark.slow
def test_cukf_runs_through_adm1_process_model_without_crashing():
    """Smoke test: three predict + update cycles on the ADM1ProcessModel
    + ConstrainedUKF stack, using the mock plant for the underlying ODE.

    Verifies the integration between ConstrainedUKF, ADM1ProcessModel
    snapshot/restore, the spec's 41-channel layout, and the QP solver —
    everything except the pyadm1 ODE itself. Must complete without
    raising, must produce a finite posterior inside the spec bounds.
    """
    process, obs, spec = build_mock_components()
    cukf = ConstrainedUKF(process, obs, spec)
    rng = np.random.default_rng(0)
    lo, hi = cukf.spec.bounds()
    n_obs = len(cukf.obs.channels)

    for k in range(3):
        cukf.predict(dt=1.0 / 24.0)
        y_pred = np.array(
            [c.extractor(cukf.process.plant, cukf.x_hat) for c in cukf.obs.channels]
        )
        y_noisy = y_pred + 0.05 * rng.standard_normal(n_obs)
        y_dict = {c.name: float(y_noisy[i]) for i, c in enumerate(cukf.obs.channels)}
        cukf.update(y_dict, t=float(k))

    assert np.all(np.isfinite(cukf.x_hat))
    assert np.all(cukf.x_hat >= lo - 1e-8), "posterior breached lower bound"
    assert np.all(cukf.x_hat <= hi + 1e-8), "posterior breached upper bound"
