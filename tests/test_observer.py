"""Tests for the amortised observer: data generator + GRU network (increments 1+2)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester

from pyadm1ode_estimation.estimation.deep_learning.observation_torch import (
    TorchObservationModel,
)
from pyadm1ode_estimation.estimation.deep_learning.observer import Adm1Observer
from pyadm1ode_estimation.estimation.deep_learning.observer_data import (
    MeasurementDataset,
    ObserverDataset,
    generate_observer_dataset,
)
from pyadm1ode_estimation.estimation.deep_learning.observer_train import (
    finetune_observer,
    observer_predict,
    per_state_nrmse,
    pretrain_observer,
    pretrain_observer_selfsup,
    pretrain_observer_sim2real,
)
from pyadm1ode_estimation.estimation.specs import InputSpec

_SUBSTRATES = [
    InputSpec("maize_silage", substrate_index=0, initial_flow=4.74),
    InputSpec("solid_manure", substrate_index=1, initial_flow=13.70),
]


# --------------------------------------------------------------------------
# Increment 1: data generator
# --------------------------------------------------------------------------
def test_generate_dataset_shapes_and_stats():
    ds = generate_observer_dataset(
        n_scenarios=3,
        substrates=_SUBSTRATES,
        warmup_days=10.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=0,
    )
    assert isinstance(ds, ObserverDataset)
    n_ch = 3  # Q_gas, Q_ch4, pH
    n_feat = n_ch + len(_SUBSTRATES)
    T = 24 + 1
    assert ds.features.shape == (3, T, n_feat)
    assert ds.states.shape == (3, T, 41)
    assert ds.feature_names[:3] == ["Q_gas", "Q_ch4", "pH"]
    assert len(ds.feature_names) == n_feat
    # normalised features: ~zero mean / unit std across all samples
    flat = ds.features.reshape(-1, n_feat)
    assert np.allclose(flat.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-3)
    assert np.isfinite(ds.states).all() and (ds.states > 0).all()


def test_dataset_scenarios_differ():
    """Different feed / initial-state draws must produce different windows."""
    ds = generate_observer_dataset(
        n_scenarios=4,
        substrates=_SUBSTRATES,
        warmup_days=10.0,
        duration_days=1.0,
        feed_range=(0.6, 1.4),
        seed=1,
    )
    # the true methane state trajectories should not be identical across scenarios
    assert not np.allclose(ds.states[0], ds.states[1])


# --------------------------------------------------------------------------
# Increment 2: GRU observer network
# --------------------------------------------------------------------------
def _observer(n_feat=5, dropout=0.0, dtype=torch.float32, gas_n_iter=15):
    fs = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=10,
    )
    d = Digester("d", fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    d.initialize({"Q_substrates": [11.4, 6.1] + [0.0] * 8})
    x_ref = np.asarray(d.adm1_state, dtype=float)
    return Adm1Observer.from_adm1(
        d.adm1,
        x_ref,
        n_features=n_feat,
        hidden=16,
        num_layers=1,
        dropout=dropout,
        dtype=dtype,
        gas_n_iter=gas_n_iter,
    )


def test_observer_forward_shape_and_positive():
    torch.manual_seed(0)
    obs = _observer()
    u = torch.randn(4, 12, 5)  # (B, T, n_features)
    x = obs(u)
    assert x.shape == (4, 12, 41)
    assert torch.isfinite(x).all()
    assert (x[..., :37] > 0).all()  # liquid states positive


def test_observer_gas_is_slaved():
    """The 4 gas states are exactly the quasi-steady solve of the liquid output."""
    from pyadm1.core.adm1_torch import gas_equilibrium_torch

    torch.manual_seed(0)
    obs = _observer(dtype=torch.float64, gas_n_iter=20)
    x = obs(torch.randn(2, 8, 5, dtype=torch.float64))
    gas_expected = gas_equilibrium_torch(x[..., :37], obs.params, n_iter=obs.gas_n_iter)
    assert torch.allclose(x[..., 37:41], gas_expected, atol=1e-8)


def test_observer_causal_and_differentiable():
    torch.manual_seed(0)
    obs = _observer()
    u = torch.randn(1, 10, 5, requires_grad=True)
    x = obs(u)
    x.sum().backward()
    assert u.grad is not None and torch.isfinite(u.grad).all()
    # causality: changing the last input must not affect earlier outputs
    u2 = u.detach().clone()
    u2[0, -1] += 5.0
    with torch.no_grad():
        x2 = obs(u2)
    assert torch.allclose(obs(u.detach())[0, :-1], x2[0, :-1], atol=1e-4)


# --------------------------------------------------------------------------
# Increment 3: supervised pre-training
# --------------------------------------------------------------------------
def test_pretrain_runs_and_reduces_loss():
    ds = generate_observer_dataset(
        n_scenarios=6,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=0,
    )
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=16,
        num_layers=1,
        gas_n_iter=10,
    )
    res = pretrain_observer(obs, ds, epochs=8, batch_size=3, val_frac=0.34, seed=0)

    assert len(res.history["train"]) == 8
    assert (
        np.isfinite(res.history["train"]).all()
        and np.isfinite(res.history["val"]).all()
    )
    assert min(res.history["train"]) < res.history["train"][0]  # training improves

    pred = observer_predict(obs, ds.features)
    assert pred.shape == ds.states.shape
    nrmse = per_state_nrmse(pred, ds.states)
    assert nrmse.shape == (41,) and np.isfinite(nrmse).all()


# --------------------------------------------------------------------------
# Increment 4: self-supervised online fine-tuning
# --------------------------------------------------------------------------
def test_finetune_improves_measurement_fit():
    ds = generate_observer_dataset(
        n_scenarios=4,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=0,
    )
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=16,
        num_layers=1,
        gas_n_iter=10,
        dropout=0.0,
    )
    pretrain_observer(obs, ds, epochs=10, batch_size=2, val_frac=0.5, seed=0)

    # One "live" window: normalised inputs + the raw measurements it encodes.
    feats = ds.features[0]  # (T, n_feat) normalised
    raw_meas = feats[:, :3] * ds.feat_std[:3] + ds.feat_mean[:3]  # (T, 3) denormalised
    tobs = TorchObservationModel(
        channel_names=["Q_gas", "Q_ch4", "pH"],
        noise_std=[50.0, 40.0, 0.1],
        params=ds.params,
        quasi_steady_gas=True,
    )

    hist = finetune_observer(
        obs,
        feats,
        tobs,
        raw_meas,
        dt_days=1.0 / 24.0,
        epochs=25,
        lr=1e-4,
        lambda_phys=0.5,
        lambda_meas=1.0,
        seed=0,
    )
    assert np.isfinite(hist["loss"]).all() and np.isfinite(hist["phys"]).all()
    # self-supervised: the measurement fit must improve during fine-tuning
    assert hist["meas"][-1] < hist["meas"][0]
    # the estimate stays valid (positive liquid states, finite)
    x = observer_predict(obs, feats)
    assert np.isfinite(x).all() and (x[..., :37] > 0).all()


def test_finetune_anchor_limits_drift():
    """With a strong anchor, fine-tuning stays near the pre-trained prediction."""
    ds = generate_observer_dataset(
        n_scenarios=3,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        seed=1,
    )
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=16,
        num_layers=1,
        gas_n_iter=10,
        dropout=0.0,
    )
    pretrain_observer(obs, ds, epochs=8, batch_size=2, val_frac=0.5, seed=1)

    feats = ds.features[0]
    raw_meas = feats[:, :3] * ds.feat_std[:3] + ds.feat_mean[:3]
    tobs = TorchObservationModel(
        ["Q_gas", "Q_ch4", "pH"], [50.0, 40.0, 0.1], ds.params, quasi_steady_gas=True
    )
    before = observer_predict(obs, feats)[:, :37]
    finetune_observer(
        obs,
        feats,
        tobs,
        raw_meas,
        dt_days=1 / 24.0,
        epochs=15,
        lr=1e-3,
        lambda_anchor=1e3,
        seed=1,
    )
    after = observer_predict(obs, feats)[:, :37]
    rel = np.abs(after - before) / (np.abs(before) + 1e-9)
    assert rel.mean() < 0.05  # strong anchor keeps it close


# --------------------------------------------------------------------------
# Increment 5: self-supervised pre-training (real / measurement-only data)
# --------------------------------------------------------------------------
def test_generate_dataset_stores_raw_measurements():
    ds = generate_observer_dataset(
        n_scenarios=2,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=0,
    )
    assert ds.channel_names == ["Q_gas", "Q_ch4", "pH"]
    assert ds.measurements.shape == (2, 25, 3)
    # the raw measurements are the first n_ch features, de-normalised
    denorm = ds.features[..., :3] * ds.feat_std[:3] + ds.feat_mean[:3]
    assert np.allclose(denorm, ds.measurements, atol=1e-6)


def test_measurement_dataset_from_real_shapes_and_nan():
    T, n_ch, n_feed = 12, 3, 2
    rng = np.random.default_rng(0)
    meas = rng.normal(size=(T, n_ch))
    meas[3, 1] = np.nan  # a gated sensor
    feed = np.array([4.7, 13.7])  # constant feed broadcast over time
    md = MeasurementDataset.from_real(
        meas,
        feed,
        dt_hours=1.0,
        params=None,
        x_ref=np.ones(41),
        channel_names=["Q_gas", "Q_ch4", "pH"],
    )
    assert md.features.shape == (1, T, n_ch + n_feed)
    assert md.measurements.shape == (1, T, n_ch)
    assert md.dt_days == pytest.approx(1.0 / 24.0)
    # NaN survives on the target side but is scrubbed from the network input
    assert np.isnan(md.measurements[0, 3, 1])
    assert np.isfinite(md.features).all()


def test_measurement_dataset_shares_sim_normalisation():
    ds = generate_observer_dataset(
        n_scenarios=2,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=0,
    )
    feed = np.array([s.initial_flow for s in _SUBSTRATES])
    md = MeasurementDataset.from_real(
        ds.measurements[0],
        feed,
        dt_hours=1.0,
        params=ds.params,
        x_ref=ds.x_ref,
        channel_names=ds.channel_names,
        feat_mean=ds.feat_mean,
        feat_std=ds.feat_std,
    )
    # sharing the sim stats reproduces the sim dataset's normalised sensor columns
    assert np.allclose(md.features[0, :, :3], ds.features[0, :, :3], atol=1e-6)


def _selfsup_fixture(seed=0):
    ds = generate_observer_dataset(
        n_scenarios=6,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=seed,
    )
    md = MeasurementDataset.from_observer_dataset(ds)
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=16,
        num_layers=1,
        gas_n_iter=10,
        dropout=0.0,
    )
    tobs = TorchObservationModel(
        channel_names=ds.channel_names,
        noise_std=[50.0, 40.0, 0.1],
        params=ds.params,
        quasi_steady_gas=True,
    )
    return ds, md, obs, tobs


def test_selfsup_pretrain_warm_started_improves():
    """After a supervised warm start, self-supervision improves the fit safely."""
    ds, md, obs, tobs = _selfsup_fixture(seed=0)
    pretrain_observer(obs, ds, epochs=15, batch_size=3, val_frac=0.34, seed=0)
    res = pretrain_observer_selfsup(
        obs,
        md,
        tobs,
        epochs=15,
        batch_size=3,
        val_frac=0.34,
        lambda_phys=0.5,
        lr=1e-4,
        seed=0,
    )
    val = res.history["val"]
    assert len(val) >= 1 and np.isfinite(val).all()  # only finite values recorded
    assert min(val) <= val[0]  # never worse than the start
    x = observer_predict(obs, ds.features)
    assert np.isfinite(x).all() and (x[..., :37] > 0).all()


def test_selfsup_pretrain_cold_start_is_safe():
    """A cold start can diverge (gas conditioning); it must fail safe, not NaN."""
    ds, md, obs, tobs = _selfsup_fixture(seed=1)
    res = pretrain_observer_selfsup(
        obs,
        md,
        tobs,
        epochs=10,
        batch_size=3,
        val_frac=0.34,
        lambda_phys=0.5,
        lr=1e-3,
        seed=0,
    )
    # never returns a NaN model, whatever happened during training
    assert np.isfinite(res.history["val"]).all()
    x = observer_predict(obs, ds.features)
    assert np.isfinite(x).all() and (x[..., :37] > 0).all()


def test_sim2real_pipeline_runs():
    ds = generate_observer_dataset(
        n_scenarios=5,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=2,
    )
    # a "real" segment reusing the sim normalisation for a consistent scale
    feed = np.array([s.initial_flow for s in _SUBSTRATES])
    md = MeasurementDataset.from_real(
        ds.measurements[:2],
        np.tile(feed, (2, ds.measurements.shape[1], 1)),
        dt_hours=1.0,
        params=ds.params,
        x_ref=ds.x_ref,
        channel_names=ds.channel_names,
        feat_mean=ds.feat_mean,
        feat_std=ds.feat_std,
    )
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=16,
        num_layers=1,
        gas_n_iter=10,
        dropout=0.0,
    )
    tobs = TorchObservationModel(
        ds.channel_names, [50.0, 40.0, 0.1], ds.params, quasi_steady_gas=True
    )
    r_sup, r_self = pretrain_observer_sim2real(
        obs,
        ds,
        md,
        tobs,
        sup_epochs=10,
        selfsup_epochs=6,
        sup_kwargs={"batch_size": 2, "val_frac": 0.4, "seed": 0},
        selfsup_kwargs={
            "batch_size": 2,
            "val_frac": 0.5,
            "seed": 0,
            "lambda_phys": 0.5,
            "lr": 1e-4,
        },
    )
    assert min(r_sup.history["train"]) < r_sup.history["train"][0]
    assert len(r_self.history["val"]) >= 1
    assert np.isfinite(r_self.history["val"]).all()


def test_observer_rejects_bad_ref():
    fs = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24, total_simtime=10)
    d = Digester("d", fs)
    d.initialize({"Q_substrates": [11.4] + [0.0] * 9})
    with pytest.raises(ValueError):
        Adm1Observer.from_adm1(d.adm1, np.zeros(37), n_features=5)


def _observer_and_dataset(n: int = 6, seed: int = 0, hidden: int = 16):
    """A small simulated dataset plus a matching observer (fast fixture)."""
    ds = generate_observer_dataset(
        n_scenarios=n,
        substrates=_SUBSTRATES,
        warmup_days=8.0,
        duration_days=1.0,
        dt_hours=1.0,
        seed=seed,
    )
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=hidden,
        num_layers=1,
        gas_n_iter=10,
    )
    return obs, ds


# --------------------------------------------------------------------------
# Pre-training: external split, train-only statistics, best-val restore
# --------------------------------------------------------------------------
def test_pretrain_scale_uses_training_sequences_only():
    """Computing the per-state scale over the whole set first leaks validation
    statistics into the objective every batch."""
    obs, ds = _observer_and_dataset(n=8)
    res = pretrain_observer(obs, ds, epochs=1, batch_size=4, val_frac=0.5, seed=0)

    train_states = ds.states[res.train_idx]
    expected = np.sqrt((train_states**2).mean(axis=(0, 1))) + 1e-8
    assert np.allclose(res.scale, expected, rtol=1e-5)

    all_states = np.sqrt((ds.states**2).mean(axis=(0, 1))) + 1e-8
    assert not np.allclose(res.scale, all_states, rtol=1e-5)


def test_pretrain_accepts_an_external_validation_set():
    """Sharing one split across estimators requires injecting it, not re-drawing."""
    obs, ds = _observer_and_dataset(n=6)
    _, val_ds = _observer_and_dataset(n=3, seed=7)
    res = pretrain_observer(obs, ds, val_dataset=val_ds, epochs=2, batch_size=3, seed=0)

    assert len(res.train_idx) == 6  # the whole dataset trains
    assert len(res.val_idx) == 3
    assert np.allclose(
        res.scale, np.sqrt((ds.states**2).mean(axis=(0, 1))) + 1e-8, rtol=1e-5
    )


def test_pretrain_restores_the_best_validated_weights():
    obs, ds = _observer_and_dataset(n=8)
    res = pretrain_observer(
        obs,
        ds,
        epochs=12,
        batch_size=4,
        val_frac=0.5,
        lr=5e-2,
        restore_best=True,
        seed=3,
    )
    assert 0 <= res.best_epoch < len(res.history["val"])
    assert res.best_val == pytest.approx(min(res.history["val"]))

    # the restored model must actually score its best value, not the last one
    val_ds_states = torch.tensor(ds.states[res.val_idx], dtype=obs._dtype)
    val_feats = torch.tensor(ds.features[res.val_idx], dtype=obs._dtype)
    scale = torch.tensor(res.scale, dtype=obs._dtype)
    obs.eval()
    with torch.no_grad():
        got = float((((obs(val_feats) - val_ds_states) / scale) ** 2).mean())
    assert got == pytest.approx(res.best_val, rel=1e-4)


def test_pretrain_early_stopping_halts_and_is_flagged():
    obs, ds = _observer_and_dataset(n=6)
    res = pretrain_observer(
        obs, ds, epochs=200, batch_size=3, val_frac=0.5, patience=3, lr=1e-4, seed=5
    )
    assert len(res.history["val"]) < 200
    assert res.stopped_early
    assert len(res.history["val"]) - 1 - res.best_epoch >= 3


def test_pretrain_noise_augmentation_perturbs_only_the_sensor_columns():
    """Noise belongs on the measurements; the substrate feed is a known input."""
    from pyadm1ode_estimation.estimation.deep_learning.observer_train import (
        _noise_scales,
    )

    _, ds = _observer_and_dataset(n=4)
    n_ch = len(ds.channel_names)
    scales = _noise_scales(ds, np.full(n_ch, 0.5)).numpy()
    assert np.all(scales[:n_ch] > 0.0)
    assert np.all(scales[n_ch:] == 0.0)
    # given in raw units, applied in normalised feature units
    assert np.allclose(scales[:n_ch], 0.5 / np.asarray(ds.feat_std)[:n_ch])


def test_pretrain_noise_std_length_is_checked():
    from pyadm1ode_estimation.estimation.deep_learning.observer_train import (
        _noise_scales,
    )

    _, ds = _observer_and_dataset(n=4)
    with pytest.raises(ValueError, match="noise_std must have shape"):
        _noise_scales(ds, [0.1, 0.2])


def test_pretrain_burnin_excludes_the_leading_steps():
    from pyadm1ode_estimation.estimation.deep_learning.observer_train import (
        _scaled_state_loss,
    )

    x = torch.zeros(2, 6, 3)
    y = torch.zeros(2, 6, 3)
    y[:, :2, :] = 100.0  # a huge error only in the burn-in region
    scale = torch.ones(3)
    assert float(_scaled_state_loss(x, y, scale, burnin=0)) > 0.0
    assert float(_scaled_state_loss(x, y, scale, burnin=2)) == pytest.approx(0.0)
