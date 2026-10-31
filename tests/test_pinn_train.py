"""Smoke tests of the trajectory-PINN network and trainer (PINN v2, step 1)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1ode_estimation.estimation.filter_tuning.datasets import DEFAULT_BENCHMARK_DIR
from pyadm1ode_estimation.estimation.pinn.data import feed_lookup, load_series
from pyadm1ode_estimation.estimation.pinn.model import ArchConfig, TrajectoryNet
from pyadm1ode_estimation.estimation.pinn.train import (
    TrainConfig,
    summarize,
    train_trajectory,
)


def test_feed_lookup_is_piecewise_constant_on_the_hour():
    time = np.arange(0.0, 1.0, 1.0 / 24.0)
    feed = np.arange(24.0)[:, None]
    np.testing.assert_array_equal(
        feed_lookup(feed, time, [0.0, 0.5 / 24, 1.0 / 24, 5.99 / 24, 10.0])[:, 0],
        [0, 0, 1, 5, 23],
    )


def test_trajectory_net_output_transform_and_rates():
    arch = ArchConfig(depth=2, width=16, fourier_features=2)
    net = TrajectoryNet(
        arch,
        n_out=3,
        n_feed=2,
        t_scale=10.0,
        feed_scale=np.ones(2),
        log_mu=np.zeros(3),
        log_sigma=np.ones(3),
    ).double()
    t = torch.rand(5, 1, dtype=torch.float64) * 10
    u = torch.rand(5, 2, dtype=torch.float64)
    x = net(t, u)
    assert x.shape == (5, 3) and torch.all(x > 0)
    torch.testing.assert_close(net.to_latent(x), net.latent(t, u))
    x2, dxdt = net.with_rates(t, u)
    torch.testing.assert_close(x2, x)
    eps = 1e-6
    fd = (net(t + eps, u) - net(t - eps, u)) / (2 * eps)
    torch.testing.assert_close(dxdt, fd, rtol=1e-5, atol=1e-8)
    # gradients of a loss on the rates reach the parameters (forward-over-reverse)
    (dxdt**2).mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters()
    )
    assert arch.name == "d2_w16_tanh_ff2"
    assert ArchConfig.parse(arch.name) == arch


@pytest.mark.parametrize(
    "name",
    [
        "d2_w16_silu_rff8_hist7_blockheads",
        "d2_w16_sin_multiscale_ln",
        "d3_w16_gelu_res_softplus_modmlp",
        "d2_w16_tanh_hist30_direct",
        "d2_w16_tanh_nofeed",
    ],
)
def test_structure_variants_run_and_roundtrip(name):
    from pyadm1ode_estimation.estimation.pinn.data import feed_features, n_feed_features

    arch = ArchConfig.parse(name)
    assert arch.name == name
    n_sub = 5
    n_feat = n_feed_features(n_sub, arch.feed_mode)
    time = np.arange(0.0, 60.0, 1.0 / 24.0)
    feed = np.abs(np.random.default_rng(1).normal(5.0, 1.0, (len(time), n_sub)))
    t = np.array([0.0, 3.5, 12.0, 59.9])
    u = feed_features(feed, time, t, arch.feed_mode)
    assert u.shape == (4, n_feat)
    if arch.feed_mode == "hist7":  # bin means lie within the feed range
        assert np.all(u >= feed.min()) and np.all(u <= feed.max())
    net = TrajectoryNet(
        arch,
        n_out=31,
        n_feed=n_feat,
        t_scale=60.0,
        feed_scale=np.full(n_feat, 10.0),
        log_mu=np.zeros(31),
        log_sigma=np.ones(31),
    ).double()
    tt = torch.as_tensor(t[:, None], dtype=torch.float64)
    uu = torch.as_tensor(u, dtype=torch.float64)
    x, dxdt = net.with_rates(tt, uu)
    assert x.shape == (4, 31) and torch.all(x > 0) and torch.isfinite(dxdt).all()
    torch.testing.assert_close(
        net.to_latent(x), net.latent(tt, uu), rtol=1e-6, atol=1e-6
    )
    (dxdt**2).mean().backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )


@pytest.mark.slow
def test_measurement_map_reproduces_the_sensors_at_noise_level():
    if not (DEFAULT_BENCHMARK_DIR / "test.npz").exists():
        pytest.skip("benchmark dataset not available")
    from pyadm1ode_estimation.estimation.pinn.data import load_meta
    from pyadm1ode_estimation.estimation.pinn.measure import CHANNELS, MeasurementModel
    from pyadm1ode_estimation.estimation.pinn.train import build_physics

    series = load_series("test", indices=[0])[0]
    meta = load_meta()
    phys = build_physics(series, TrainConfig(closure=True), meta)
    meas = MeasurementModel.from_meta(phys.params, meta)
    x = torch.as_tensor(series.states[::12, :31], dtype=torch.float64)
    y = torch.as_tensor(series.measurements[::12], dtype=torch.float64)
    r = meas.normalized_residual(x, y)
    assert r.shape == (x.shape[0], len(CHANNELS))
    # noise-normalised residual of the truth ~ N(0, 1) per channel (gas bias -0.6 % from the lag)
    assert torch.all(r.std(dim=0) < 1.5) and torch.all(r.abs().mean(dim=0) < 1.5)
    assert meas.noise.kind == (
        "relative",
        "relative",
        "relative",
        "absolute",
        "absolute",
    )


@pytest.mark.slow
def test_train_on_measurements_with_train_split_prior_smoke():
    if not (DEFAULT_BENCHMARK_DIR / "test.npz").exists():
        pytest.skip("benchmark dataset not available")
    from pyadm1ode_estimation.estimation.pinn.data import (
        train_split_log_stats,
        train_split_scale_reference,
    )

    series = load_series("test", indices=[5])[0]
    prior = train_split_log_stats()
    assert prior[0].shape == (31,) and np.all(prior[1] >= 0.05)
    scale_ref = train_split_scale_reference(every=24)
    assert (
        scale_ref[0].shape[1] == 41 and scale_ref[0].shape[0] == scale_ref[1].shape[0]
    )
    cfg = TrainConfig(
        steps=10,
        n_collocation=16,
        n_data_batch=16,
        t_obs_days=3.0,
        horizon_days=1.0,
        eval_days=5.0,
        closure=True,
        data_source="measurements",
        prior="train_split",
        use_true_feed=False,
        use_series_kinetics=False,
        ic_prior=True,
        estimate_kinetics=True,
        log_every=5,
    )
    with pytest.raises(
        ValueError
    ):  # the field setting must not fall back to the series' truth
        train_trajectory(series, ArchConfig(depth=2, width=16), cfg, prior_stats=prior)
    ic_state = series.ukf_x_hat[0].copy()
    ic_state[7] = (
        0.0  # a zero entry (S_h2 after the warm-up) must be masked, not log(0)
    )
    res = train_trajectory(
        series,
        ArchConfig(depth=2, width=16),
        cfg,
        prior_stats=prior,
        ic_state=ic_state,
        scale_ref=scale_ref,
    )
    assert np.isfinite(res.loss_curve[-1]["loss"])
    assert np.isfinite(res.metrics["kinetics"]["log_rmse_estimated"])
    assert len(res.metrics["kinetics"]["per_key"]) == 26
    bm = res.metrics["benchmark"]
    assert set(bm) == {"fit", "horizon", "full"}
    assert np.isfinite(bm["fit"]["model"]["nrmse_overall_%"]) and np.isfinite(
        bm["fit"]["ukf"]["nrmse_overall_%"]
    )
    assert np.isfinite(bm["full"]["vs_ukf"]["rmse_ratio_overall"])


def test_kinetic_factors_enter_the_physics_and_get_gradients():
    from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import build_plant
    from pyadm1ode_estimation.estimation.pinn.data import load_meta
    from pyadm1ode_estimation.estimation.pinn.influent import InfluentModel
    from pyadm1ode_estimation.estimation.pinn.kinetics import (
        KineticFactors,
        perturbed_keys,
    )
    from pyadm1ode_estimation.estimation.pinn.physics import (
        ADM1Physics,
        PhysicsConfig,
        build_params,
    )

    if not (DEFAULT_BENCHMARK_DIR / "meta.json").exists():
        pytest.skip("benchmark dataset not available")
    plant = build_plant(load_meta())
    adm1 = plant.components["primary"].adm1
    adm1.create_influent([4.0, 2.0, 0.5, 0.0, 0.0], 0)
    base = build_params(plant)
    kin = KineticFactors(perturbed_keys(base), sigma_prior=0.25)
    assert len(kin.keys) == 26 and all(k in base.kinetic for k in kin.keys)
    phys = ADM1Physics(base, InfluentModel.from_adm1(adm1), PhysicsConfig(closure=True))
    x = torch.tensor(
        np.abs(np.random.default_rng(0).normal(0.1, 0.05, (3, 31))), dtype=torch.float64
    )
    feed = torch.tensor([[4.0, 2.0, 0.5, 0.0, 0.0]] * 3, dtype=torch.float64)
    f0 = phys.residual(x, torch.zeros_like(x), feed)
    with torch.no_grad():
        kin.theta[kin.keys.index("k_hyd_ch")] = np.log(2.0)
    phys.params = kin.apply(base)
    assert float(phys.params.kinetic["k_hyd_ch"]) == pytest.approx(
        2.0 * base.kinetic["k_hyd_ch"]
    )
    f1 = phys.residual(x, torch.zeros_like(x), feed)
    assert not torch.allclose(f0, f1)
    (f1**2).mean().backward()
    assert kin.theta.grad is not None and torch.isfinite(kin.theta.grad).all()
    assert kin.prior_loss().item() == pytest.approx((np.log(2.0) / 0.25) ** 2 / 26)
    rep = kin.report(kin.keys, np.full(26, 2.0))
    assert rep["per_key"]["k_hyd_ch"]["log_error"] == pytest.approx(0.0)
    assert rep["log_rmse_nominal"] == pytest.approx(np.log(2.0))


@pytest.mark.slow
def test_training_variants_smoke():
    """Causal weighting, time-marching, grey-box and L-BFGS run end to end (60-day task)."""
    if not (DEFAULT_BENCHMARK_DIR / "test.npz").exists():
        pytest.skip("benchmark dataset not available")
    from pyadm1ode_estimation.estimation.pinn.data import (
        train_split_log_stats,
        train_split_scale_reference,
    )

    series = load_series("test", indices=[2])[0]
    prior = train_split_log_stats()
    scale_ref = train_split_scale_reference(every=24)
    cfg = TrainConfig(
        steps=12,
        n_collocation=16,
        n_data_batch=16,
        t_obs_days=4.0,
        horizon_days=0.0,
        eval_days=4.0,
        closure=True,
        data_source="measurements",
        prior="train_split",
        use_true_feed=False,
        use_series_kinetics=False,
        causal_eps=10.0,
        time_march_days=2.0,
        greybox=True,
        greybox_width=8,
        lbfgs_steps=3,
        ic_prior=True,
        lambda_ic=10.0,
        log_every=6,
    )
    res = train_trajectory(
        series,
        ArchConfig(depth=2, width=16, output="blockheads"),
        cfg,
        prior_stats=prior,
        ic_state=series.ukf_x_hat[0],
        scale_ref=scale_ref,
    )
    assert "horizon" not in res.metrics and set(res.metrics["benchmark"]) == {
        "fit",
        "full",
    }
    assert np.isfinite(res.metrics["fit"]["phys_residual_rms"])
    assert res.loss_curve[-1].get("lbfgs") is True and np.isfinite(
        res.loss_curve[-1]["loss"]
    )
    assert np.isfinite(res.metrics["greybox"]["g_rms"])
    s = summarize([res.to_record()])
    assert next(iter(s.values()))["main_window"] == "fit"


@pytest.mark.slow
def test_rolling_causal_chain_smoke():
    """Rolling-window warm-start chain: bridging, fixed-lag estimates and scoring (block F1)."""
    if not (DEFAULT_BENCHMARK_DIR / "test.npz").exists():
        pytest.skip("benchmark dataset not available")
    import dataclasses

    from pyadm1ode_estimation.estimation.pinn.data import (
        train_split_log_stats,
        train_split_scale_reference,
    )
    from pyadm1ode_estimation.estimation.pinn.rolling import RollingConfig, run_rolling

    s = load_series("test", indices=[0])[0]
    n = 2 * 24 + 1
    s2 = dataclasses.replace(
        s,
        time=s.time[:n],
        states=s.states[:n],
        feed_true=s.feed_true[:n],
        feed_noisy=s.feed_noisy[:n],
        measurements=s.measurements[:n],
        ukf_x_hat=s.ukf_x_hat[:n],
        ukf_std=s.ukf_std[:n],
    )
    cfg = TrainConfig(
        t_obs_days=60.0,
        horizon_days=0.0,
        closure=True,
        data_source="measurements",
        prior="train_split",
        use_true_feed=False,
        use_series_kinetics=False,
        n_collocation=16,
        n_data_batch=16,
    )
    roll = RollingConfig(
        window_days=1.0,
        cadence_hours=12.0,
        steps_first=8,
        steps_warm=4,
        lags_days=(0.5,),
    )
    res = run_rolling(
        s2,
        ArchConfig.parse("d2_w16_tanh_ff2"),
        cfg,
        roll,
        prior_stats=train_split_log_stats(),
        scale_ref=train_split_scale_reference(every=24),
    )
    assert res.n_retrainings == 3 and roll.name == "W1_C12_S4"
    for k in ("hold", "adm1", "lag0.5"):
        assert (
            res.trajectories[k].shape == (n, 41)
            and np.isfinite(res.trajectories[k]).all()
        )
        assert np.isfinite(res.metrics[k]["full"]["model"]["nrmse_overall_%"])
    assert not np.allclose(res.trajectories["adm1"], res.trajectories["hold"])
    assert res.seconds_per_operating_hour > 0


@pytest.mark.slow
def test_train_trajectory_smoke():
    if not (DEFAULT_BENCHMARK_DIR / "test.npz").exists():
        pytest.skip("benchmark dataset not available")
    series = load_series("test", indices=[0])[0]
    cfg = TrainConfig(
        steps=20,
        n_collocation=32,
        t_obs_days=5.0,
        horizon_days=2.0,
        eval_days=8.0,
        log_every=10,
    )
    res = train_trajectory(series, ArchConfig(depth=2, width=32), cfg)
    m = res.metrics
    assert set(m) >= {"fit", "horizon", "beyond"}
    assert np.isfinite(m["fit"]["rel_rmse_mean"]) and np.isfinite(
        m["horizon"]["phys_residual_rms"]
    )
    assert len(m["fit"]["rel_rmse_per_state"]) == 41
    assert res.loss_curve[-1]["step"] == 20
    s = summarize([res.to_record()])
    assert "d2_w32_tanh" in s and s["d2_w32_tanh"]["n_runs"] == 1
