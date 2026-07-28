"""Tests for the sliding-window operational driver."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1ode_estimation.estimation.deep_learning.observation_torch import (
    TorchObservationModel,
)
from pyadm1ode_estimation.estimation.deep_learning.observer import Adm1Observer
from pyadm1ode_estimation.estimation.deep_learning.observer_data import (
    generate_observer_dataset,
)
from pyadm1ode_estimation.estimation.deep_learning.observer_train import (
    pretrain_observer,
)
from pyadm1ode_estimation.estimation.deep_learning.online_observer import (
    OnlineEstimate,
    SlidingWindowObserver,
)
from pyadm1ode_estimation.estimation.specs import InputSpec

_SUBS = [
    InputSpec("maize_silage", substrate_index=0, initial_flow=4.74),
    InputSpec("solid_manure", substrate_index=1, initial_flow=13.70),
]


def _setup(dropout=0.0):
    ds = generate_observer_dataset(
        n_scenarios=4, substrates=_SUBS, warmup_days=8.0, duration_days=1.0, seed=0
    )
    obs = Adm1Observer(
        ds.params,
        ds.x_ref,
        n_features=ds.features.shape[-1],
        hidden=16,
        num_layers=1,
        gas_n_iter=10,
        dropout=dropout,
    )
    pretrain_observer(obs, ds, epochs=8, batch_size=2, val_frac=0.5, seed=0)
    tobs = TorchObservationModel(
        ["Q_gas", "Q_ch4", "pH"], [50.0, 40.0, 0.1], ds.params, quasi_steady_gas=True
    )
    # reconstruct one scenario's raw measurement + feed stream
    feats = ds.features[0]
    n_ch = 3
    raw = feats * ds.feat_std + ds.feat_mean
    meas_stream = raw[:, :n_ch]
    feed_stream = raw[:, n_ch:]
    return obs, tobs, ds, meas_stream, feed_stream


def test_stream_produces_valid_estimates():
    obs, tobs, ds, meas, feed = _setup()
    swo = SlidingWindowObserver(
        obs,
        tobs,
        ds.feat_mean,
        ds.feat_std,
        window_hours=12,
        dt_hours=1.0,
        finetune_every=0,
    )
    ests = swo.run(meas, feed)
    assert len(ests) == len(meas)
    for k, e in enumerate(ests):
        assert isinstance(e, OnlineEstimate)
        assert e.state.shape == (41,)
        assert np.isfinite(e.state).all() and (e.state[:37] > 0).all()
        assert e.t == pytest.approx(k / 24.0)
        assert not e.finetuned  # disabled


def test_window_slides_and_is_bounded():
    obs, tobs, ds, meas, feed = _setup()
    swo = SlidingWindowObserver(
        obs, tobs, ds.feat_mean, ds.feat_std, window_hours=6, dt_hours=1.0
    )
    swo.run(meas, feed)
    assert len(swo._meas) == swo.window_steps == 6  # buffer capped at the window
    assert swo._count == len(meas)  # but total seen keeps growing


def test_finetune_triggers_on_schedule():
    obs, tobs, ds, meas, feed = _setup()
    swo = SlidingWindowObserver(
        obs,
        tobs,
        ds.feat_mean,
        ds.feat_std,
        window_hours=24,
        dt_hours=1.0,
        finetune_every=6,
        finetune_epochs=3,
        min_steps=6,
        finetune_kwargs={"lambda_phys": 0.2, "lambda_anchor": 1.0},
    )
    ests = swo.run(meas, feed)
    fine_steps = [k for k, e in enumerate(ests) if e.finetuned]
    # fires at counts 6, 12, 18, ... (1-indexed count → 0-indexed k = count-1)
    assert fine_steps and all((k + 1) % 6 == 0 for k in fine_steps)


def test_gated_sensor_nan_input_is_handled():
    obs, tobs, ds, meas, feed = _setup()
    meas = meas.copy()
    meas[3, 1] = np.nan  # Q_ch4 offline at one step
    swo = SlidingWindowObserver(
        obs, tobs, ds.feat_mean, ds.feat_std, window_hours=12, dt_hours=1.0
    )
    ests = swo.run(meas, feed)  # must not crash
    assert all(np.isfinite(e.state).all() for e in ests)


def test_mc_dropout_uncertainty_online():
    obs, tobs, ds, meas, feed = _setup(dropout=0.1)
    swo = SlidingWindowObserver(
        obs,
        tobs,
        ds.feat_mean,
        ds.feat_std,
        window_hours=12,
        dt_hours=1.0,
        mc_samples=15,
    )
    e = swo.run(meas, feed)[-1]
    assert e.std.shape == (41,)
    assert e.std.max() > 0.0
