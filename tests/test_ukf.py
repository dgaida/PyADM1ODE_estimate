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

    def refresh_outputs(self, x, equilibration_dt=1.0 / 24.0):  # noqa: ARG002
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

    def test_ukf_matches_classical_kf_with_negligible_Q(self):
        """With ``Q`` numerically negligible the canonical Wan-VdM SR-UKF
        reproduces the closed-form Kalman filter to ``atol=1e-6``.

        For non-negligible ``Q`` the sigma-reuse measurement update (see
        :mod:`pyadm1ode_estimation.estimation.filters.sr_ukf`) drops the
        ``H Q H^T`` contribution from the innovation covariance — a known
        property of the canonical SR-UKF (Wan & Van der Merwe 2001,
        Algorithm 3.1, line 22) discussed against the augmented form in
        Wu et al. 2005. The bound-Q test below pins the magnitude of that
        deviation.
        """
        rng = np.random.default_rng(0)
        n = 3
        # process_std = 1e-8 → Q ≈ 1e-16, ten orders of magnitude below P0.
        # Cholesky of Q stays well-defined; the H Q H^T contribution to S_y
        # is below the 1e-6 test tolerance.
        spec = make_linear_spec(n, initial=0.0, initial_std=1.0, process_std=1e-8)
        F = np.eye(n)
        process = LinearProcess(spec, F)

        H = np.eye(n)[:2]
        noise_std = 0.2

        def make_ex(i):
            return lambda plant, x: float(x[i])

        obs = ObservationModel(
            channels=[
                ObservationChannel(
                    name=f"y{i}", extractor=make_ex(i), noise_std=noise_std
                )
                for i in range(2)
            ]
        )
        ukf = UnscentedKalmanFilter(process, obs, spec)

        T = 10
        truth = rng.normal(0, 1, size=(T, n))
        y_seq = (H @ truth.T).T + rng.normal(0, noise_std, size=(T, 2))

        Q = spec.process_noise_cov(dt=1.0)
        R = noise_std**2 * np.eye(2)
        x0 = spec.initial_mean()
        P0 = spec.initial_cov()
        xs_ref, Ps_ref = closed_form_kf(F, Q, H, R, x0, P0, y_seq)

        for k, y in enumerate(y_seq):
            ukf.predict(dt=1.0)
            ukf.update({"y0": float(y[0]), "y1": float(y[1])}, t=float(k))
            np.testing.assert_allclose(ukf.x_hat, xs_ref[k], atol=1e-6)
            np.testing.assert_allclose(ukf.P, Ps_ref[k], atol=1e-6)

    def test_ukf_approximates_classical_kf_with_random_walk_Q(self, setup):
        """With ``Q/P ≈ 1 %`` the canonical SR-UKF tracks the closed-form
        KF to a few percent, not bit-exactly.

        The reuse formulation skips ``H Q H^T`` in ``S_y``; over 10 random
        walk steps that compounds into a small but measurable bias. The
        loose tolerance here is the upper bound for "still numerically
        sane" behaviour; if it tightens or relaxes, something deeper
        shifted.
        """
        ukf, spec, F, H, noise_std, y_seq = setup

        Q = spec.process_noise_cov(dt=1.0)
        R = noise_std**2 * np.eye(2)
        x0 = spec.initial_mean()
        P0 = spec.initial_cov()
        xs_ref, Ps_ref = closed_form_kf(F, Q, H, R, x0, P0, y_seq)

        for k, y in enumerate(y_seq):
            ukf.predict(dt=1.0)
            ukf.update({"y0": float(y[0]), "y1": float(y[1])}, t=float(k))
            np.testing.assert_allclose(ukf.x_hat, xs_ref[k], atol=0.05)
            np.testing.assert_allclose(ukf.P, Ps_ref[k], atol=0.05)

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

    def test_channel_with_nonfinite_prediction_is_dropped(self, setup):
        # A channel whose extractor (model prediction) returns NaN must be
        # dropped for that step — the good channel still updates.
        ukf, spec, _F, _H, noise_std, _y_seq = setup

        obs = ObservationModel(
            channels=[
                ObservationChannel("y0", lambda p, x: float(x[0]), noise_std),
                ObservationChannel("ybad", lambda p, x: float("nan"), noise_std),
            ]
        )
        ukf2 = UnscentedKalmanFilter(ukf.process, obs, spec)
        x_prev = ukf2.x_hat.copy()
        step = ukf2.update({"y0": 5.0, "ybad": 5.0}, t=0.0)
        assert step.active_channels == ["y0"]
        assert ukf2.x_hat[0] > x_prev[0]

    def test_all_predictions_nonfinite_means_no_correction(self, setup):
        # If every active channel predicts NaN, the state is left at the
        # prior (no correction), exactly like a fully missing measurement.
        ukf, spec, _F, _H, noise_std, _y_seq = setup

        obs = ObservationModel(
            channels=[ObservationChannel("ybad", lambda p, x: float("nan"), noise_std)]
        )
        ukf2 = UnscentedKalmanFilter(ukf.process, obs, spec)
        x_prev = ukf2.x_hat.copy()
        P_prev = ukf2.P.copy()
        step = ukf2.update({"ybad": 5.0}, t=0.0)
        assert step.active_channels == []
        np.testing.assert_allclose(ukf2.x_hat, x_prev)
        np.testing.assert_allclose(ukf2.P, P_prev)


class TestReducedSigmaScaling:
    """Hellmann et al. 2024 (ECC) reduced-scaling trick: replace the
    canonical sigma-point radius ``γ = √(n + λ)`` with a user-supplied
    constant (typically ``γ = 1``) while keeping the canonical mean /
    covariance weights. Pins the three behaviours that matter:

    * default ``gamma_override=None`` keeps the canonical scaling and
      cannot drift over future refactors;
    * passing ``gamma_override=1.0`` actually changes ``self.gamma`` and
      the resulting sigma-point spread;
    * the override goes through end-to-end (predict + update) without
      crashing on the linear-Gaussian setup.
    """

    def _build(self, n=4, gamma_override=None):
        spec = make_linear_spec(n, initial=0.0, initial_std=1.0, process_std=0.1)
        F = np.eye(n)
        process = LinearProcess(spec, F)

        def make_ex(i):
            return lambda plant, x: float(x[i])

        obs = ObservationModel(
            channels=[
                ObservationChannel(name=f"y{i}", extractor=make_ex(i), noise_std=0.2)
                for i in range(2)
            ]
        )
        return UnscentedKalmanFilter(process, obs, spec, gamma_override=gamma_override)

    def test_default_keeps_canonical_scaling(self):
        ukf = self._build(n=4)
        expected = np.sqrt(4 + ukf.lam)
        assert ukf.gamma == pytest.approx(expected)
        # With default α=1, β=2, κ=0: λ=0 → γ=√n=2.
        assert ukf.gamma == pytest.approx(2.0)

    def test_override_replaces_sigma_radius(self):
        ukf = self._build(n=4, gamma_override=1.0)
        assert ukf.gamma == pytest.approx(1.0)
        # Weights stay canonical (built from λ, not γ): w_m[1..] == 1/(2n)
        # at default α, β, κ.
        np.testing.assert_allclose(ukf._w_m[1:], 1.0 / (2 * 4))

    def test_override_changes_sigma_spread(self):
        canonical = self._build(n=4)
        reduced = self._build(n=4, gamma_override=1.0)
        sigma_can = canonical._sigma_points(canonical.x_hat, canonical._S)
        sigma_red = reduced._sigma_points(reduced.x_hat, reduced._S)
        # Centre point identical, but the ± gamma·S offsets must shrink
        # by exactly the gamma ratio.
        ratio = reduced.gamma / canonical.gamma
        offset_can = sigma_can[1] - sigma_can[0]
        offset_red = sigma_red[1] - sigma_red[0]
        np.testing.assert_allclose(offset_red, ratio * offset_can)

    def test_override_rejects_non_positive(self):
        with pytest.raises(ValueError, match="gamma_override must be > 0"):
            self._build(n=4, gamma_override=0.0)
        with pytest.raises(ValueError, match="gamma_override must be > 0"):
            self._build(n=4, gamma_override=-1.0)

    def test_filter_runs_end_to_end_with_override(self):
        # Ten predict + update cycles must complete without raising and
        # leave (x_hat, P) finite. We don't pin the trajectory — that's
        # the canonical KF test's job; here we just verify the override
        # path is not a runtime trap.
        ukf = self._build(n=4, gamma_override=1.0)
        rng = np.random.default_rng(1)
        for k in range(10):
            ukf.predict(dt=1.0)
            y = {f"y{i}": float(rng.normal()) for i in range(2)}
            ukf.update(y, t=float(k))
        assert np.all(np.isfinite(ukf.x_hat))
        assert np.all(np.isfinite(ukf.P))


class TestSquareRootProperties:
    """SR-UKF specific tests: covariance is always exposed as ``P = SS^T``,
    the Cholesky factor ``S`` is lower triangular at all times, and
    ``reset`` reconstructs it correctly from a target ``P``.
    """

    @pytest.fixture
    def ukf(self):
        # Reuse the same setup as TestUKFLinear.setup but without
        # generating measurements (we only need a working filter instance).
        n = 3
        spec = make_linear_spec(
            n,
            initial=0.0,
            initial_std=1.0,
            process_std=0.1,
        )
        F = np.eye(n)
        process = LinearProcess(spec, F)

        def make_extractor(i):
            return lambda plant, x: float(x[i])

        obs = ObservationModel(
            channels=[
                ObservationChannel(
                    name=f"y{i}", extractor=make_extractor(i), noise_std=0.2
                )
                for i in range(2)
            ]
        )
        return UnscentedKalmanFilter(process, obs, spec)

    def test_S_is_lower_triangular(self, ukf):
        S = ukf.S
        # Upper triangle (strictly above diagonal) must be zero.
        upper = np.triu(S, k=1)
        np.testing.assert_allclose(upper, 0.0, atol=1e-12)

    def test_P_equals_S_S_T(self, ukf):
        P = ukf.P
        SST = ukf.S @ ukf.S.T
        np.testing.assert_allclose(P, SST, atol=1e-12)

    def test_S_stays_lower_triangular_after_predict_update_cycles(self, ukf):
        # After several predict/update cycles, the Cholesky factor must
        # still be lower triangular. The QR + cholupdate path should
        # produce a clean triangular S, not just an "almost triangular"
        # one with rounding noise above the diagonal.
        for k in range(5):
            ukf.predict(dt=1.0)
            ukf.update({"y0": 0.1 * k, "y1": -0.1 * k}, t=float(k))
            S = ukf.S
            np.testing.assert_allclose(np.triu(S, k=1), 0.0, atol=1e-10)
            # And P = SS^T must still be symmetric and PSD by construction.
            P = ukf.P
            np.testing.assert_allclose(P, P.T, atol=1e-12)
            assert np.linalg.eigvalsh(P).min() > -1e-12

    def test_reset_recomputes_S_from_P(self, ukf):
        # Reset to a custom P0 and verify that S = chol(P0) afterwards.
        P0 = np.diag([4.0, 1.0, 9.0])  # eigenvalues 1, 4, 9 → cond=9
        x0 = np.array([1.0, 2.0, 3.0])
        ukf.reset(x0, P0)
        np.testing.assert_allclose(ukf.x_hat, x0)
        np.testing.assert_allclose(ukf.P, P0, atol=1e-12)
        # S should be the diagonal Cholesky: diag(2, 1, 3).
        expected_S = np.diag([2.0, 1.0, 3.0])
        np.testing.assert_allclose(ukf.S, expected_S, atol=1e-12)


class TestCholupdate:
    """Direct tests of the internal _cholupdate primitive — verifies that
    the rank-1 update/downdate produce mathematically correct results
    and that downdate failure is reported as ``LinAlgError``.
    """

    def test_rank_one_update_matches_chol_of_full_matrix(self):
        from pyadm1ode_estimation.estimation.filters.sr_ukf import _cholupdate

        rng = np.random.default_rng(7)
        # Build a random positive-definite P, take its Cholesky, then
        # update by a random vector v. Compare to direct chol(P + vv^T).
        A = rng.normal(size=(5, 5))
        P = A @ A.T + np.eye(5)
        S = np.linalg.cholesky(P)
        v = rng.normal(size=5)

        S_new = _cholupdate(S, v, sign=+1)

        P_expected = P + np.outer(v, v)
        np.testing.assert_allclose(S_new @ S_new.T, P_expected, atol=1e-10)
        np.testing.assert_allclose(np.triu(S_new, k=1), 0.0, atol=1e-12)

    def test_rank_one_downdate_matches_chol_of_full_matrix(self):
        from pyadm1ode_estimation.estimation.filters.sr_ukf import _cholupdate

        rng = np.random.default_rng(13)
        A = rng.normal(size=(5, 5))
        P = A @ A.T + 10.0 * np.eye(5)  # generous PSD margin
        S = np.linalg.cholesky(P)
        v = 0.1 * rng.normal(size=5)  # small enough that P - vv^T stays PSD

        S_new = _cholupdate(S, v, sign=-1)

        P_expected = P - np.outer(v, v)
        np.testing.assert_allclose(S_new @ S_new.T, P_expected, atol=1e-10)

    def test_downdate_fails_when_matrix_becomes_indefinite(self):
        from pyadm1ode_estimation.estimation.filters.sr_ukf import _cholupdate

        # Tiny P, large v — P - vv^T must go negative definite.
        S = np.eye(3) * 0.1  # P = 0.01 * I
        v = np.array([2.0, 0.0, 0.0])  # vv^T has 4.0 on (0,0) → downdate fails
        with pytest.raises(np.linalg.LinAlgError, match="downdate failed"):
            _cholupdate(S, v, sign=-1)
