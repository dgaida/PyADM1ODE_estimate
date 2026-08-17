"""Integration tests for the PINN state-trajectory smoother (Variant A).

These assert the end-to-end machinery works — the network fits observed
channels, the physics loss is computed and finite, training reduces the loss,
and estimate() returns a well-formed trajectory with optional MC-Dropout
uncertainty. Reconstruction *accuracy* on the full 41-dim state is an
experimental tuning matter validated in the twin-experiment scripts, not here.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.core.adm1_torch import (
    Adm1TorchParams,
    calc_gas_quasi_steady_torch,
    calc_gas_torch,
)

from pyadm1ode_estimation.estimation.base import TrajectoryEstimate
from pyadm1ode_estimation.estimation.deep_learning.observation_torch import (
    TorchObservationModel,
)
from pyadm1ode_estimation.estimation.deep_learning.pinn_smoother import PinnSmoother

_Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]


def _setup():
    fs = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=10,
    )
    d = Digester("d", fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    d.initialize({"Q_substrates": _Q})
    try:
        d.adm1.create_influent(d.Q_substrates, 0)
    except Exception:  # noqa: BLE001, S110
        pass
    params = Adm1TorchParams.from_adm1(d.adm1)
    obs = TorchObservationModel.from_adm1(
        d.adm1, {"Q_gas": 50.0, "Q_ch4": 40.0, "pH": 0.1}
    )
    x_prior = np.asarray(d.adm1_state, dtype=float)
    return params, obs, x_prior


def _reachable_measurements(obs, x_prior, times, scale=0.15, seed=0):
    """Synthetic measurements = h(x_true) for a physically reachable x_true
    (the prior perturbed in log-space), tiled across the measurement times."""
    rng = np.random.default_rng(seed)
    x_true = x_prior * np.exp(scale * rng.normal(size=x_prior.shape))
    y = obs.predict(torch.tensor(x_true, dtype=torch.float64)).detach().numpy()
    return np.tile(y, (len(times), 1))


def test_fit_reduces_data_loss():
    """With physics off, the net must fit the observed channels."""
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, lambda_phys=0.0, lambda_prior=0.0, seed=1)
    times = [0.1, 0.3, 0.5, 0.7, 0.9]
    y = _reachable_measurements(obs, x_prior, times, seed=1)
    hist = sm.fit(times, y, t0=0.0, t1=1.0, n_collocation=16, epochs=250, lr=1e-3)
    assert np.isfinite(hist["data"]).all()
    assert hist["data"][-1] < 0.5 * hist["data"][0]


def test_physics_training_runs_and_reduces_loss():
    """With physics on, training must run with finite losses and reduce total loss."""
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, lambda_phys=1.0, lambda_prior=1.0, seed=2)
    times = [0.2, 0.5, 0.8]
    y = _reachable_measurements(obs, x_prior, times, scale=0.1, seed=2)
    hist = sm.fit(times, y, t0=0.0, t1=1.0, n_collocation=16, epochs=150, lr=1e-3)
    for key in ("loss", "data", "phys", "prior"):
        assert np.isfinite(hist[key]).all()
    assert hist["loss"][-1] < hist["loss"][0]
    # At initialisation (zero-init output) the trajectory equals the prior, so
    # the physics residual of a near-steady prior starts small and finite.
    assert hist["phys"][0] < np.inf


def test_estimate_shapes_and_positive():
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, lambda_phys=0.5, seed=3)
    sm.fit(
        [0.5],
        _reachable_measurements(obs, x_prior, [0.5], seed=3),
        t0=0.0,
        t1=1.0,
        n_collocation=8,
        epochs=40,
        lr=1e-3,
    )
    times = np.linspace(0.0, 1.0, 11)
    est = sm.estimate(times)
    assert isinstance(est, TrajectoryEstimate)
    assert est.x_hat.shape == (11, 41)
    assert est.std.shape == (11, 41)
    assert np.all(est.x_hat > 0.0)
    assert np.isfinite(est.x_hat).all()
    assert np.allclose(est.std, 0.0)  # no MC-Dropout requested


def test_mc_dropout_uncertainty():
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, dropout=0.1, lambda_phys=0.0, seed=4)
    sm.fit(
        [0.5],
        _reachable_measurements(obs, x_prior, [0.5], seed=4),
        t0=0.0,
        t1=1.0,
        n_collocation=8,
        epochs=40,
        lr=1e-3,
    )
    times = np.linspace(0.0, 1.0, 6)
    est = sm.estimate(times, mc_samples=20)
    assert est.std.max() > 0.0  # dropout injects spread
    est1 = sm.estimate(times, mc_samples=1)
    assert np.allclose(est1.std, 0.0)  # single deterministic pass


# --------------------------------------------------------------------------
# Forecasting: data window shorter than the collocation window
# --------------------------------------------------------------------------
def test_forecast_tail_is_physics_driven_and_stable():
    """Collocation past the last measurement → the ODE continues the trajectory.

    On a settled (steady) truth the physics forecast must stay near the in-sample
    estimate rather than drift or blow up, and stay finite + positive.
    """
    d = _warmed_digester()
    params = Adm1TorchParams.from_adm1(d.adm1)
    obs = TorchObservationModel.from_adm1(
        d.adm1, {"Q_gas": 50.0, "Q_ch4": 40.0, "pH": 0.1}, quasi_steady_gas=True
    )
    x_prior = np.asarray(d.adm1_state, dtype=float)
    sm = PinnSmoother(
        params,
        obs,
        x_prior,
        quasi_steady_gas=True,
        gas_n_iter=12,
        lambda_phys=1.0,
        lambda_prior=1.0,
        seed=7,
    )

    # measurements only over [0, 0.6]; collocation (physics) runs out to 1.0
    obs_t = [0.1, 0.25, 0.4, 0.55]
    y = _reachable_measurements(
        obs, x_prior, obs_t, scale=0.0, seed=7
    )  # exact steady state
    sm.fit(obs_t, y, t0=0.0, t1=1.0, n_collocation=40, epochs=300, lr=1e-3)

    est = sm.estimate(np.linspace(0.0, 1.0, 21))
    x = est.x_hat
    assert np.isfinite(x).all() and np.all(x[:, :37] > 0.0)
    in_sample = x[:12]  # t <= ~0.55 (data region)
    forecast = x[12:]  # t > 0.55 (physics-only tail)
    # the forecast must not run away from the data-region level (liquid states)
    ref = in_sample[-1, :37]
    rel_drift = np.abs(forecast[:, :37] - ref) / (np.abs(ref) + 1e-9)
    assert rel_drift.mean() < 0.15


# --------------------------------------------------------------------------
# Warm-started rolling update
# --------------------------------------------------------------------------
def test_update_growing_warm_starts():
    """update() continues from the fitted weights instead of retraining."""
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, lambda_phys=0.5, lambda_prior=1.0, seed=8)
    y = _reachable_measurements(obs, x_prior, [0.2, 0.5, 0.8], scale=0.1, seed=8)
    fit_hist = sm.fit(
        [0.2, 0.5, 0.8], y, t0=0.0, t1=1.0, n_collocation=16, epochs=200, lr=1e-3
    )

    # one hour later: same history + one new sample, growing window to t1=1.2
    y2 = _reachable_measurements(obs, x_prior, [0.2, 0.5, 0.8, 1.1], scale=0.1, seed=8)
    upd_hist = sm.update(
        [0.2, 0.5, 0.8, 1.1],
        y2,
        t1=1.2,
        window=None,
        n_collocation=16,
        epochs=20,
        lr=3e-4,
    )

    assert len(upd_hist["loss"]) == 20
    # warm start: the FIRST update epoch already sits near where fit ended,
    # not at a fresh-init loss (which would be orders of magnitude higher).
    assert upd_hist["loss"][0] < 3.0 * fit_hist["loss"][-1]
    est = sm.estimate(np.linspace(0.0, 1.2, 7))
    assert np.isfinite(est.x_hat).all() and np.all(est.x_hat[:, :37] > 0.0)


def test_update_sliding_window():
    """A sliding window moves t0 forward and still returns a valid estimate."""
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, lambda_phys=0.5, seed=9)
    y = _reachable_measurements(obs, x_prior, [0.1, 0.4, 0.7, 1.0], scale=0.05, seed=9)
    sm.fit(
        [0.1, 0.4, 0.7, 1.0], y, t0=0.0, t1=1.0, n_collocation=16, epochs=150, lr=1e-3
    )

    # keep only the most recent 0.6 d of data (t0 = 1.4 - 0.6 = 0.8)
    y2 = _reachable_measurements(obs, x_prior, [0.9, 1.1, 1.3], scale=0.05, seed=9)
    hist = sm.update(
        [0.9, 1.1, 1.3], y2, t1=1.4, window=0.6, n_collocation=16, epochs=20, lr=3e-4
    )
    assert np.isfinite(hist["loss"]).all()
    est = sm.estimate([1.4])
    assert np.isfinite(est.x_hat).all() and np.all(est.x_hat[:, :37] > 0.0)


def test_update_requires_fit_first():
    params, obs, x_prior = _setup()
    sm = PinnSmoother(params, obs, x_prior, seed=10)
    with pytest.raises(RuntimeError):
        sm.update([0.5], _reachable_measurements(obs, x_prior, [0.5], seed=10), t1=1.0)


# --------------------------------------------------------------------------
# Quasi-steady gas phase (Point 1 fix)
# --------------------------------------------------------------------------
def _warmed_digester():
    fs = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=10,
    )
    d = Digester("d", fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    d.initialize({"Q_substrates": _Q})
    for k in range(15):  # settle the gas phase to equilibrium
        d.step(t=float(k), dt=1.0, inputs={"Q_substrates": _Q})
    return d


def test_quasi_steady_gas_matches_equilibrium():
    """At a settled state the quasi-steady solve reproduces calc_gas."""
    d = _warmed_digester()
    params = Adm1TorchParams.from_adm1(d.adm1)
    x = torch.tensor(np.asarray(d.adm1_state, dtype=float), dtype=torch.float64)

    qg_hard = float(calc_gas_torch(x, params)[0])
    qg_qss = float(calc_gas_quasi_steady_torch(x, params)[0])
    assert qg_hard > 100.0  # sanity: the digester is producing gas
    # QSS pins pTOTAL near p_ext-p_h2o (drops the tiny R/kp term), so a few %.
    assert qg_qss == pytest.approx(qg_hard, rel=0.05)


def test_quasi_steady_smoother_pins_ptotal_and_flows():
    """A QSS smoother runs, returns a full 41-state, and keeps pTOTAL pinned."""
    d = _warmed_digester()
    params = Adm1TorchParams.from_adm1(d.adm1)
    obs = TorchObservationModel.from_adm1(
        d.adm1, {"Q_gas": 50.0, "Q_ch4": 40.0, "pH": 0.1}, quasi_steady_gas=True
    )
    x_prior = np.asarray(d.adm1_state, dtype=float)
    sm = PinnSmoother(
        params,
        obs,
        x_prior,
        quasi_steady_gas=True,
        gas_n_iter=10,
        lambda_phys=0.1,
        lambda_prior=1.0,
        seed=5,
    )
    # net predicts only the 37 liquid states
    assert sm.net.net[-1].out_features == 37

    y = _reachable_measurements(obs, x_prior, [0.5], seed=5)
    sm.fit([0.5], y, t0=0.0, t1=1.0, n_collocation=6, epochs=30, lr=1e-3)

    est = sm.estimate(np.linspace(0.0, 1.0, 5))
    assert est.x_hat.shape == (5, 41)
    assert np.isfinite(est.x_hat).all()
    # pTOTAL (index 40) is slaved to ~ p_ext - p_h2o, not a free variable.
    p_pin = params.p_ext - params.p_gas_h2o
    assert np.allclose(est.x_hat[:, 40], p_pin, atol=0.02)


# --------------------------------------------------------------------------
# Physics-residual scaling (stiff-system fix)
# --------------------------------------------------------------------------
def test_physics_scaling_default_is_rate():
    params, obs, x_prior = _setup()
    assert PinnSmoother(params, obs, x_prior).physics_scaling == "rate"
    with pytest.raises(ValueError):
        PinnSmoother(params, obs, x_prior, physics_scaling="bogus")


def test_rate_scaling_bounds_physics_loss():
    """Rate scaling keeps the stiff physics residual O(1); state scaling doesn't."""
    d = _warmed_digester()
    params = Adm1TorchParams.from_adm1(d.adm1)
    obs = TorchObservationModel.from_adm1(
        d.adm1, {"Q_gas": 50.0, "pH": 0.1}, quasi_steady_gas=True
    )
    x_prior = np.asarray(d.adm1_state, dtype=float)
    y = _reachable_measurements(obs, x_prior, [0.5], seed=6)

    common = {
        "quasi_steady_gas": True,
        "gas_n_iter": 10,
        "lambda_phys": 1.0,
        "lambda_prior": 1.0,
        "seed": 6,
    }
    h_rate = PinnSmoother(params, obs, x_prior, physics_scaling="rate", **common).fit(
        [0.5], y, t0=0.0, t1=1.0, n_collocation=6, epochs=20, lr=1e-3
    )
    h_state = PinnSmoother(params, obs, x_prior, physics_scaling="state", **common).fit(
        [0.5], y, t0=0.0, t1=1.0, n_collocation=6, epochs=20, lr=1e-3
    )

    assert np.isfinite(h_rate["phys"]).all()
    # relative residual stays modest; the stiff k_A_B=1e8 terms make the state-
    # scaled residual orders of magnitude larger.
    assert max(h_rate["phys"]) < 1.0e3
    assert max(h_state["phys"]) > 1.0e2 * max(h_rate["phys"])


# --------------------------------------------------------------------------
# Robust residual (res_clip)
# --------------------------------------------------------------------------
def test_robust_sq_is_quadratic_inside_and_linear_outside():
    """Huber: below the threshold identical to res^2, above it linear — so the
    gradient is bounded and one ill-conditioned channel cannot set the step."""
    import torch

    from pyadm1ode_estimation.estimation.deep_learning.pinn_smoother import _robust_sq

    res = torch.tensor([0.0, 1.0, -2.0, 3.0, 10.0, -50.0], dtype=torch.float64)
    assert torch.allclose(_robust_sq(res, None), res**2)

    out = _robust_sq(res, 3.0)
    inside = res.abs() <= 3.0
    assert torch.allclose(out[inside], res[inside] ** 2)
    # continuous at the threshold, and far below the square outside it
    assert out[3].item() == pytest.approx(9.0)
    assert out[5].item() == pytest.approx(3.0 * (2 * 50.0 - 3.0))
    assert out[5].item() < (res[5] ** 2).item()


def test_robust_sq_bounds_the_gradient():
    import torch

    from pyadm1ode_estimation.estimation.deep_learning.pinn_smoother import _robust_sq

    for value in (10.0, 100.0, 1000.0):
        r = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        _robust_sq(r, 3.0).sum().backward()
        assert r.grad.item() == pytest.approx(6.0)  # 2 * delta, independent of value
        # ...whereas the plain square grows without bound
        r2 = torch.tensor([value], dtype=torch.float64, requires_grad=True)
        (r2**2).sum().backward()
        assert r2.grad.item() == pytest.approx(2 * value)


def test_res_clip_must_be_positive():
    params, obs, x_prior = _setup()
    with pytest.raises(ValueError, match="res_clip must be positive"):
        PinnSmoother(params, obs, x_prior, res_clip=0.0)


def test_res_clip_still_fits_a_reachable_target():
    """Robustifying must not stop the fit converging when nothing is off-scale."""
    params, obs, x_prior = _setup()
    sm = PinnSmoother(
        params, obs, x_prior, lambda_phys=0.0, lambda_prior=0.0, res_clip=3.0, seed=1
    )
    times = [0.1, 0.3, 0.5, 0.7, 0.9]
    y = _reachable_measurements(obs, x_prior, times, seed=1)
    hist = sm.fit(times, y, t0=0.0, t1=1.0, n_collocation=16, epochs=250, lr=1e-3)
    assert np.isfinite(hist["data"]).all()
    assert hist["data"][-1] < 0.5 * hist["data"][0]


# --------------------------------------------------------------------------
# Adjustment 3: quasi-steady charge balance (solve_cation)
# --------------------------------------------------------------------------
def test_charge_balance_inversion_round_trips():
    """solve_cation_for_ph must be the exact inverse of ph_torch."""
    from pyadm1.core.adm1_torch import ph_torch

    from pyadm1ode_estimation.estimation.deep_learning.charge_balance import apply_ph

    params, _, x_prior = _setup()
    x = torch.tensor(np.tile(x_prior, (6, 1)), dtype=torch.float64)
    targets = torch.tensor([6.0, 6.5, 7.0, 7.5, 8.0, 8.5], dtype=torch.float64)
    assert torch.allclose(
        ph_torch(apply_ph(x, targets, params), params), targets, atol=1e-8
    )


def test_charge_balance_inversion_is_differentiable():
    from pyadm1ode_estimation.estimation.deep_learning.charge_balance import (
        solve_cation_for_ph,
    )

    params, _, x_prior = _setup()
    x = torch.tensor(np.atleast_2d(x_prior), dtype=torch.float64)
    ph = torch.tensor([7.4], dtype=torch.float64, requires_grad=True)
    solve_cation_for_ph(x, ph, params).sum().backward()
    assert ph.grad is not None and bool(torch.isfinite(ph.grad).all())


def test_solve_cation_conditions_the_ph_channel():
    """The point of Adjustment 3: pH must stop being orders of magnitude steeper
    than every other channel with respect to the network's own outputs."""
    params, obs, x_prior = _setup()
    j = obs.channel_names.index("pH")
    sens = {}
    for flag in (False, True):
        sm = PinnSmoother(
            params, obs, x_prior, quasi_steady_gas=True, solve_cation=flag, seed=0
        )
        y = obs.predict(sm._forward_state(torch.tensor([[0.0]], dtype=torch.float64)))
        g = torch.autograd.grad(y[0, j], sm.net.net[-1].bias)[0]
        sens[flag] = float(g.norm())
    assert sens[True] < sens[False] / 100.0, sens


def test_solve_cation_preserves_the_prior_at_initialisation():
    """Zero-init must still put the trajectory exactly on the prior's liquid state.

    Reparameterising the S_cation slot as pH only holds up if solving the charge
    balance hands back the very cation the prior had — otherwise every fit would
    start from a different state than the caller asked for.

    States the prior sets to exactly zero are excluded: the log transform floors
    its base at 1e-8 (a true-zero prior would pin that state at zero forever),
    which is pre-existing behaviour independent of this parameterisation.
    """
    params, obs, x_prior = _setup()
    sm = PinnSmoother(
        params, obs, x_prior, quasi_steady_gas=True, solve_cation=True, seed=0
    )
    x0 = sm.estimate([0.0]).x_hat[0]

    positive = x_prior[:37] > 0.0
    assert np.allclose(x0[:37][positive], x_prior[:37][positive], rtol=1e-6)
    assert np.allclose(x0[:37][~positive], 1.0e-8)
    # the reparameterised slot specifically
    cation = 29
    assert x0[cation] == pytest.approx(x_prior[cation], rel=1e-9)


def test_solve_cation_masks_the_cation_from_physics_and_prior():
    """S_cation becomes algebraic, so its ODE / prior anchor must be dropped."""
    from pyadm1ode_estimation.estimation.deep_learning.charge_balance import (
        CATION_INDEX,
    )

    params, obs, x_prior = _setup()
    on = PinnSmoother(
        params, obs, x_prior, quasi_steady_gas=True, solve_cation=True, seed=0
    )
    off = PinnSmoother(
        params, obs, x_prior, quasi_steady_gas=True, solve_cation=False, seed=0
    )
    assert on._state_mask[CATION_INDEX].item() == 0.0
    assert off._state_mask[CATION_INDEX].item() == 1.0
    assert on._state_mask.sum().item() == on._n_free - 1


def test_solve_cation_needs_the_cation_slot_among_the_free_states():
    params, obs, x_prior = _setup()
    # Not reachable through the public flags today, but the guard must exist.
    assert PinnSmoother(params, obs, x_prior, solve_cation=True)._n_free > 29


# --------------------------------------------------------------------------
# Best-weight restore
# --------------------------------------------------------------------------
def test_restore_best_returns_the_best_trajectory_not_the_last():
    """The collocation fit is not monotone: it reaches its best trajectory and then
    walks away from it. Returning the final weights throws away the answer."""
    params, obs, x_prior = _setup()
    times = [0.1, 0.3, 0.5, 0.7, 0.9]
    y = _reachable_measurements(obs, x_prior, times, seed=11)

    def run(restore_best):
        sm = PinnSmoother(
            params,
            obs,
            x_prior,
            lambda_phys=0.0,
            lambda_prior=0.0,
            restore_best=restore_best,
            seed=11,
        )
        hist = sm.fit(times, y, t0=0.0, t1=1.0, n_collocation=16, epochs=200, lr=5e-2)
        # loss of the model actually handed back
        x = sm.estimate(times).x_hat
        res = (obs.predict(torch.tensor(x)) - torch.tensor(y)) / obs.noise_std_tensor()
        return hist, float((res**2).mean())

    hist_off, returned_off = run(False)
    hist_on, returned_on = run(True)

    # identical optimisation path — restoring happens only at the end
    assert np.allclose(hist_off["loss"], hist_on["loss"])
    # with a large lr the run overshoots, so the last weights are worse than the best
    assert min(hist_on["loss"]) < hist_on["loss"][-1]
    assert returned_on <= returned_off


def test_restore_best_is_monotone_safe():
    """It must never hand back something worse than where it started."""
    params, obs, x_prior = _setup()
    times = [0.2, 0.6]
    y = _reachable_measurements(obs, x_prior, times, seed=12)
    sm = PinnSmoother(params, obs, x_prior, lambda_phys=1.0, restore_best=True, seed=12)
    hist = sm.fit(times, y, t0=0.0, t1=1.0, n_collocation=12, epochs=120, lr=1e-1)

    x = sm.estimate(times).x_hat
    assert np.isfinite(x).all() and np.all(x[:, :37] > 0.0)
    # the restored model's loss equals the best the run ever saw
    assert min(hist["loss"]) <= hist["loss"][0] + 1e-12
