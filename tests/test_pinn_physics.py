"""PINN v2 physics: influent map, batched RHS, closures, residual, truth verification.

The physics side of a PINN loss is only worth anything if it reproduces the
simulator that generated the data. These tests pin that down at three levels:
(1) the influent map equals what the digester feeds the ODE, (2) the batched
RHS equals the per-row reference, (3) the finite-difference residual of the
benchmark truth is at the discretisation-error level for the ODE rows.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1.core.adm1 import STATE_SIZE
from pyadm1.core.adm1_torch import adm1da_rhs_torch

from pyadm1ode_estimation.estimation.filter_tuning.datasets import DEFAULT_BENCHMARK_DIR
from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import (
    build_plant,
    freeze_nominal_mix,
    influent_mode,
)
from pyadm1ode_estimation.estimation.pinn import (
    CLOSURE_ODE_STATES,
    FULL_ODE_STATES,
    N_CORE,
    ADM1Physics,
    InfluentModel,
    PhysicsConfig,
    build_params,
    close_state,
    close_state_with_rates,
    ion_equilibrium_residual,
    solve_charge_balance,
    trajectory_residual,
    with_influent,
)
from pyadm1ode_estimation.estimation.pinn.physics import _charge_balance
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

DIGESTER = "primary"

# Order-of-magnitude state of a healthy digester (ADM1da index map).
_BASE_STATE = np.array(
    [
        0.012,
        0.005,
        0.10,
        0.012,
        0.013,
        0.016,
        0.20,
        2.4e-7,
        0.05,
        0.15,
        0.13,
        0.02,
    ]  # 0-11
    + [2.0, 2.0, 2.0, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, 5.0]  # 12-21
    + [0.5, 0.3, 0.2, 0.3, 0.4, 0.6, 0.3]  # 22-28
    + [0.04, 0.02, 0.011, 0.013, 0.016, 0.20, 0.12, 0.004]  # 29-36
    + [1.0e-5, 0.55, 0.45, 1.05],  # 37-40
    dtype=float,
)


@pytest.fixture(scope="module")
def plant():
    """The benchmark's process model: digesters pinned to the nominal mix.

    pyadm1 blends the influent from the current feed. The benchmark truth was
    generated when it still cached the blend of the initialisation, so the
    physics under test has to be pinned the same way.
    """
    p = build_multi_stage_plant()
    p.components[DIGESTER].adm1._kinetic["k_dec_ac"] = 0.02
    freeze_nominal_mix(p)
    return p


@pytest.fixture(scope="module")
def unpinned_adm1():
    """A digester as pyadm1 builds it today: the influent follows the feed."""
    return build_multi_stage_plant().components[DIGESTER].adm1


@pytest.fixture(scope="module")
def adm1(plant):
    return plant.components[DIGESTER].adm1


@pytest.fixture(scope="module")
def n_sub(adm1):
    return len(adm1.feedstock.q_conversion_factors)


def _random_feed(rng, n, n_sub):
    feed = rng.uniform(0.0, 12.0, size=(n, n_sub))
    feed[0, 1:] = 0.0  # a single-substrate row
    return feed


def _random_states(rng, n):
    x = _BASE_STATE * np.exp(rng.normal(0.0, 0.3, size=(n, STATE_SIZE)))
    x[:, 40] = np.maximum(x[:, 40], 0.5)
    return x


# --------------------------------------------------------------------------
# Influent
# --------------------------------------------------------------------------
def test_nominal_mix_matches_what_the_digester_feeds_the_ode(adm1, n_sub):
    rng = np.random.default_rng(0)
    feeds = _random_feed(rng, 4, n_sub)
    model = InfluentModel.from_adm1(adm1, mode="nominal_mix")
    for feed in feeds:
        adm1.create_influent(list(feed), 0)
        q_ref = float(np.sum(adm1._Q))
        s_ref = np.asarray(adm1._state_input, float)
        q, s = model(feed)
        assert q == pytest.approx(q_ref, rel=1e-12)
        np.testing.assert_allclose(s, s_ref, rtol=1e-12, atol=1e-15)


def test_blend_mode_matches_the_feedstock_blend(adm1, n_sub):
    rng = np.random.default_rng(1)
    feeds = _random_feed(rng, 4, n_sub)
    model = InfluentModel.from_adm1(adm1, mode="blend")
    fs = adm1.feedstock
    for feed in feeds:
        df = fs.get_influent_dataframe(list(feed))
        row = df.iloc[0].to_numpy(float)
        q, s = model(feed)
        assert q == pytest.approx(row[-1], rel=1e-12)
        np.testing.assert_allclose(s, row[:-1], rtol=1e-10, atol=1e-14)


def test_blend_mode_matches_what_an_unpinned_digester_feeds_the_ode(
    unpinned_adm1, n_sub
):
    rng = np.random.default_rng(1)
    feeds = _random_feed(rng, 4, n_sub)
    model = InfluentModel.from_adm1(unpinned_adm1, mode="blend")
    for feed in feeds:
        unpinned_adm1.create_influent(list(feed), 0)
        q, s = model(feed)
        assert q == pytest.approx(float(np.sum(unpinned_adm1._Q)), rel=1e-12)
        np.testing.assert_allclose(
            s, np.asarray(unpinned_adm1._state_input, float), rtol=1e-10, atol=1e-14
        )


def test_nominal_mix_needs_a_pinned_digester(unpinned_adm1, n_sub):
    """An unpinned digester has no nominal composition; the last blend is not one."""
    unpinned_adm1.create_influent([1.0] + [0.0] * (n_sub - 1), 0)
    with pytest.raises(ValueError, match="freeze_nominal_mix"):
        InfluentModel.from_adm1(unpinned_adm1, mode="nominal_mix")


def test_pinning_fixes_the_composition_at_the_nominal_feed(n_sub):
    p = build_multi_stage_plant()
    digester = p.components[DIGESTER]
    nominal = digester.adm1.feedstock.blended_concentrations(digester.Q_substrates)
    freeze_nominal_mix(p)
    digester.adm1.create_influent([1.0] + [0.0] * (n_sub - 1), 0)
    np.testing.assert_allclose(
        digester.adm1._state_input, list(nominal.values()), rtol=1e-12, atol=1e-15
    )


def test_build_plant_pins_only_datasets_generated_with_the_nominal_mix():
    """Datasets without the field predate the pyadm1 fix; new ones state "blend"."""
    legacy, blended = {}, {"influent_mode": "blend"}
    assert influent_mode(legacy) == "nominal_mix" and influent_mode(blended) == "blend"
    assert build_plant(legacy).components[DIGESTER].adm1._influent_df is not None
    assert build_plant(blended).components[DIGESTER].adm1._influent_df is None
    with pytest.raises(ValueError, match="influent_mode"):
        influent_mode({"influent_mode": "frozen"})


def test_influent_is_batched_and_torch_compatible(adm1, n_sub):
    rng = np.random.default_rng(2)
    feed = _random_feed(rng, 7, n_sub)
    for mode in ("nominal_mix", "blend"):
        model = InfluentModel.from_adm1(adm1, mode=mode)
        q_np, s_np = model(feed)
        q_t, s_t = model(torch.as_tensor(feed, dtype=torch.float64))
        assert q_t.shape == (7,) and s_t.shape == (7, 37)
        np.testing.assert_allclose(q_t.numpy(), q_np, rtol=1e-12)
        np.testing.assert_allclose(s_t.numpy(), s_np, rtol=1e-12)


def test_influent_zero_feed_gives_zero_flow_and_finite_composition(adm1, n_sub):
    model = InfluentModel.from_adm1(adm1, mode="blend")
    q, s = model(np.zeros((2, n_sub)))
    assert np.all(q == 0.0) and np.all(np.isfinite(s)) and np.all(s == 0.0)


# --------------------------------------------------------------------------
# Batched RHS
# --------------------------------------------------------------------------
def test_batched_rhs_equals_the_per_row_reference(adm1, n_sub):
    rng = np.random.default_rng(3)
    x = _random_states(rng, 6)
    feed = _random_feed(rng, 6, n_sub)
    params = build_params(adm1)
    influent = InfluentModel.from_adm1(adm1, mode="blend")
    q, s = influent(torch.as_tensor(feed, dtype=torch.float64))
    got = adm1da_rhs_torch(
        torch.as_tensor(x, dtype=torch.float64), with_influent(params, q, s)
    ).numpy()
    for k in range(6):
        import dataclasses

        p_k = dataclasses.replace(
            params, q_ad=float(q[k]), s_in=[float(v) for v in s[k]]
        )
        ref = adm1da_rhs_torch(torch.as_tensor(x[k], dtype=torch.float64), p_k).numpy()
        np.testing.assert_allclose(got[k], ref, rtol=1e-12, atol=1e-18)


def test_kinetic_factors_scale_the_rates(adm1):
    p0 = build_params(adm1)
    p1 = build_params(adm1, kinetic_factors=[2.0], kinetic_keys=["k_hyd_ch"])
    assert p1.kinetic["k_hyd_ch"] == pytest.approx(2.0 * p0.kinetic["k_hyd_ch"])
    assert p1.kinetic["k_hyd_pr"] == p0.kinetic["k_hyd_pr"]
    with pytest.raises(KeyError):
        build_params(adm1, kinetic_factors=[1.0], kinetic_keys=["no_such_key"])


# --------------------------------------------------------------------------
# Closures
# --------------------------------------------------------------------------
def test_charge_balance_root_and_ion_equilibrium(adm1):
    rng = np.random.default_rng(4)
    core = torch.as_tensor(_random_states(rng, 5)[:, :N_CORE], dtype=torch.float64)
    params = build_params(adm1)
    S_H = solve_charge_balance(core, params)
    F = _charge_balance(S_H, core, params.inhib)
    assert torch.all(F.abs() < 1e-12 * (core[:, 29] + 1.0))
    pH = -torch.log10(S_H)
    assert torch.all((pH > 5.0) & (pH < 9.0))
    x = close_state(core, params)
    ip = params.inhib
    # S_hco3 (K_a + S_H) == K_a S_co2  and  S_nh3 (K_a + S_H) == K_a S_nh4
    np.testing.assert_allclose(
        (x[:, 35] * (ip["K_a_co2"] + S_H)).numpy(),
        (ip["K_a_co2"] * x[:, 9]).numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        (x[:, 36] * (ip["K_a_IN"] + S_H)).numpy(),
        (ip["K_a_IN"] * x[:, 10]).numpy(),
        rtol=1e-12,
    )
    # quasi-steady headspace: partial pressures sum to the total, pinned near p_ext - p_h2o
    np.testing.assert_allclose(x[:, 37:40].sum(1).numpy(), x[:, 40].numpy(), rtol=1e-10)
    np.testing.assert_allclose(
        x[:, 40].numpy(), params.p_ext - params.p_gas_h2o, rtol=1e-6
    )


def test_closure_gradients_flow_and_are_finite(adm1):
    rng = np.random.default_rng(5)
    core = torch.tensor(
        _random_states(rng, 3)[:, :N_CORE], dtype=torch.float64, requires_grad=True
    )
    params = build_params(adm1)
    x = close_state(core, params)
    x[:, 31:41].sum().backward()
    assert torch.isfinite(core.grad).all()
    assert (core.grad.abs() > 0).any()


def test_closure_rates_match_finite_differences(adm1):
    rng = np.random.default_rng(6)
    core = torch.as_tensor(_random_states(rng, 3)[:, :N_CORE], dtype=torch.float64)
    dcore = torch.as_tensor(
        rng.normal(size=core.shape) * 1e-3 * core.numpy(), dtype=torch.float64
    )
    params = build_params(adm1)
    x_full, dx_full = close_state_with_rates(core, dcore, params)
    np.testing.assert_allclose(
        x_full.numpy(), close_state(core, params).numpy(), rtol=1e-12
    )
    eps = 1e-6
    fd = (
        close_state(core + eps * dcore, params)
        - close_state(core - eps * dcore, params)
    ) / (2 * eps)
    # ions (31-36): exact implicit gradient; gas rows (37-40) are not used as residuals
    np.testing.assert_allclose(
        dx_full[:, :37].numpy(), fd[:, :37].numpy(), rtol=1e-4, atol=1e-14
    )


# --------------------------------------------------------------------------
# Residual
# --------------------------------------------------------------------------
def test_residual_shapes_and_scaling(adm1, n_sub):
    rng = np.random.default_rng(7)
    x41 = _random_states(rng, 8)
    feed = torch.as_tensor(_random_feed(rng, 8, n_sub), dtype=torch.float64)
    params = build_params(adm1)
    influent = InfluentModel.from_adm1(adm1)

    phys = ADM1Physics(params, influent, PhysicsConfig(closure=True))
    assert (
        phys.config.ode_indices == CLOSURE_ODE_STATES
        and 7 not in CLOSURE_ODE_STATES
        and 8 not in CLOSURE_ODE_STATES
    )
    x = torch.as_tensor(x41[:, :N_CORE], dtype=torch.float64)
    dx = torch.zeros_like(x)
    r = phys.residual(x, dx, feed)
    assert r.shape == (8, len(CLOSURE_ODE_STATES)) and phys.n_residual == len(
        CLOSURE_ODE_STATES
    )
    # dxdt = 0  ->  residual is -f (with the conserved-sum rows)
    f = phys.model_rates(phys.full_state(x), feed)
    np.testing.assert_allclose(r.numpy(), -f.numpy())

    scale = phys.fit_scale(x41, feed.numpy())
    assert scale.shape == (len(CLOSURE_ODE_STATES),) and np.all(scale > 0)
    r_s = phys.residual(x, dx, feed)
    np.testing.assert_allclose(r_s.numpy(), (-f / torch.as_tensor(scale)).numpy())
    np.testing.assert_allclose(
        phys.residual(x, dx, feed, scaled=False).numpy(), -f.numpy()
    )

    # DAE mode: 41 outputs, core + gas ODE rows, plus 6 equilibrium residuals
    dae = ADM1Physics(params, influent, PhysicsConfig(closure=False))
    assert (
        dae.config.ode_indices == FULL_ODE_STATES
        and dae.n_residual == len(FULL_ODE_STATES) + 6
    )
    x_t = torch.as_tensor(x41, dtype=torch.float64)
    r_dae = dae.residual(x_t, torch.zeros_like(x_t), feed)
    assert r_dae.shape == (8, len(FULL_ODE_STATES) + 6)
    np.testing.assert_allclose(
        r_dae[:, -6:].numpy(), ion_equilibrium_residual(x_t, params).numpy()
    )
    x_eq = close_state(
        x_t[:, :N_CORE], params
    )  # closed ions satisfy the equilibrium exactly
    assert torch.all(ion_equilibrium_residual(x_eq, params).abs() < 1e-10)
    with pytest.raises(ValueError):
        PhysicsConfig(closure=False, ode_indices=(0, 35))


def test_total_rows_carry_no_acid_base_term(adm1, n_sub):
    """Rows 9 / 10 are totals: an off-equilibrium ion moves its own row by the
    k_A_B = 1e8 relaxation term, the row of the total only by the smooth free-CO2 /
    pH dependence."""
    rng = np.random.default_rng(8)
    params = build_params(adm1)
    # start exactly at equilibrium (closed ions), then push one ion off by 0.1 %
    x41 = close_state(
        torch.as_tensor(_random_states(rng, 4)[:, :N_CORE], dtype=torch.float64), params
    )
    feed = torch.as_tensor(_random_feed(rng, 4, n_sub), dtype=torch.float64)
    phys = ADM1Physics(
        params,
        InfluentModel.from_adm1(adm1),
        PhysicsConfig(closure=False, ode_indices=(9, 10)),
    )
    x_off = x41.clone()
    x_off[:, 35] *= 1.001  # off-equilibrium bicarbonate
    total = (phys.model_rates(x_off, feed) - phys.model_rates(x41, feed))[:, 0].abs()
    ion = (phys.rhs(x_off, feed) - phys.rhs(x41, feed))[:, 35].abs()
    assert torch.all(total < 1e-2 * ion), (total, ion)


def test_residual_is_differentiable_for_training(adm1, n_sub):
    """A tiny 'network' x(t) = a + b t: gradients of the scaled residual w.r.t. its
    parameters must be finite and non-zero in closure mode."""
    rng = np.random.default_rng(9)
    base = torch.as_tensor(_random_states(rng, 1)[0, :N_CORE], dtype=torch.float64)
    feed = torch.as_tensor(_random_feed(rng, 5, n_sub), dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64, requires_grad=True).unsqueeze(
        1
    )
    a = base.clone().requires_grad_(True)
    b = (0.01 * base).clone().requires_grad_(True)
    x = a + b * t
    (dxdt,) = torch.autograd.grad(
        x, t, grad_outputs=torch.ones_like(x), create_graph=True
    )
    dxdt = dxdt.expand_as(x) * 0 + b  # d(a + b t)/dt = b for every row
    params = build_params(adm1)
    phys = ADM1Physics(params, InfluentModel.from_adm1(adm1))
    loss = (phys.residual(x, dxdt, feed) ** 2).mean()
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()
    assert (a.grad.abs() > 0).any() and (b.grad.abs() > 0).any()


# --------------------------------------------------------------------------
# Truth verification on the benchmark (skipped if the dataset is not present)
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def benchmark_series():
    d = DEFAULT_BENCHMARK_DIR
    if not (d / "train.npz").exists() or not (d / "meta.json").exists():
        pytest.skip("benchmark dataset not available")
    z = np.load(d / "train.npz", allow_pickle=True)
    i = 0
    return {
        "states": z["states"][i],
        "feed": z["feed_true"][i],
        "time": z["time"][i],
        "kinetic_keys": [str(k) for k in z["kinetic_keys"]],
        "kinetic_factors": z["kinetic_factors"][i],
        "meta": json.loads((d / "meta.json").read_text(encoding="utf-8")),
    }


def _benchmark_physics(series, closure: bool, mode: str | None = None):
    plant = build_plant(series["meta"])
    adm1 = plant.components[DIGESTER].adm1
    adm1.create_influent(list(series["feed"][0]), 0)
    params = build_params(
        plant,
        kinetic_factors=series["kinetic_factors"],
        kinetic_keys=series["kinetic_keys"],
    )
    influent = InfluentModel.from_adm1(adm1, mode=mode or influent_mode(series["meta"]))
    return ADM1Physics(params, influent, PhysicsConfig(closure=closure))


def _relative_rms(out):
    rms = lambda a: np.sqrt(np.mean(a * a, axis=0))
    return rms(out["r"]) / np.maximum(rms(out["model_rates"]), 1e-30)


@pytest.mark.slow
def test_truth_trajectory_residual_is_at_finite_difference_level(benchmark_series):
    """DAE mode: all 31 core rows + 4 gas rows, and the ion equilibrium residuals."""
    s = benchmark_series
    phys = _benchmark_physics(s, closure=False)
    out = trajectory_residual(phys, s["states"], s["feed"], s["time"])
    ratio = _relative_rms(out)
    assert phys.config.ode_indices == FULL_ODE_STATES
    # Hourly central differences on a stiff-but-smooth trajectory: ~1e-3 typical,
    # the fast-disintegration pools (X_PF, tau = 2.5 d) reach ~2e-2; gas rows < 5e-3.
    assert np.all(ratio < 0.05), ratio
    assert np.all(ratio[-4:] < 0.02), ratio[-4:]
    # the truth's ions sit ~1e-3 off exact equilibrium (finite k_A_B)
    assert np.all(np.median(np.abs(out["ion_eq"]), axis=0) < 1e-2)


#: Closure-mode rows whose residual is amplified by the ~1e-3 ion-closure error
#: (small net of large gross terms through the steep NH3 / pH inhibitions).
_AMPLIFIED_ROWS = {0, 1, 6, 9}


@pytest.mark.slow
def test_truth_residual_with_closure_is_documented(benchmark_series):
    s = benchmark_series
    phys = _benchmark_physics(s, closure=True)
    assert phys.config.ode_indices == CLOSURE_ODE_STATES
    closed = _relative_rms(trajectory_residual(phys, s["states"], s["feed"], s["time"]))
    for j, i in enumerate(phys.config.ode_indices):
        if i in _AMPLIFIED_ROWS:
            assert closed[j] < 0.5, (i, closed[j])
        else:
            assert closed[j] < 0.05, (i, closed[j])


@pytest.mark.slow
def test_blend_is_visibly_wrong_on_the_nominal_mix_truth(benchmark_series):
    """Documents the dataset: its truth saw the nominal composition at every step.

    The feed record varies the substrate mix per phase, but the generator ran on a
    pyadm1 that cached the blend of the initialisation. With the flow-weighted
    blend the particulate rows are off by about 100 % on the simulator's own
    output; with the nominal mix they sit at the finite-difference level.
    """
    s = benchmark_series
    if influent_mode(s["meta"]) != "nominal_mix":
        pytest.skip("dataset was generated with the flow-weighted blend")
    phys = _benchmark_physics(s, closure=False, mode="blend")
    ratio = _relative_rms(trajectory_residual(phys, s["states"], s["feed"], s["time"]))
    rows = [
        list(phys.config.ode_indices).index(i) for i in range(12, 18)
    ]  # X_PS_*, X_PF_*
    assert np.all(ratio[rows] > 0.5), ratio[rows]
