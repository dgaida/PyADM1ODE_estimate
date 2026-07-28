"""Tests for the differentiable torch observation model."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1.core.adm1 import ADM1, calc_total_solids
from pyadm1.core.adm1_torch import Adm1TorchParams, calc_gas_torch, ph_torch, ts_torch

from pyadm1ode_estimation.estimation.deep_learning.observation_torch import (
    SUPPORTED_CHANNELS,
    TorchObservationModel,
)

_BASE = np.array(
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
        2.0,
        2.0,
        2.0,
        0.5,
        0.5,
        0.5,
        2.0,
        2.0,
        2.0,
        5.0,
        0.5,
        0.3,
        0.2,
        0.3,
        0.4,
        0.6,
        0.3,
        0.04,
        0.02,
        0.011,
        0.013,
        0.016,
        0.20,
        0.12,
        0.004,
        1.0e-5,
        0.55,
        0.45,
        1.05,
    ],
    dtype=np.float64,
)


def _adm1() -> ADM1:
    return ADM1(feedstock=None, V_liq=1200.0, V_gas=216.0, T_ad=315.15)


def test_from_adm1_and_shapes():
    obs = TorchObservationModel.from_adm1(_adm1(), {"Q_gas": 50.0, "pH": 0.1})
    assert obs.channel_names == ["Q_gas", "pH"]
    assert obs.noise_std == [50.0, 0.1]

    x = torch.tensor(_BASE, dtype=torch.float64)
    single = obs.predict(x)
    assert single.shape == (2,)

    batch = torch.tensor(np.stack([_BASE, _BASE * 1.01]), dtype=torch.float64)
    assert obs.predict(batch).shape == (2, 2)


def test_values_match_underlying_maps():
    adm1 = _adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    obs = TorchObservationModel.from_adm1(adm1, ["Q_gas", "Q_ch4", "pH"])
    x = torch.tensor(_BASE, dtype=torch.float64)

    out = obs.predict(x).detach().numpy()
    g, c, _ = calc_gas_torch(x, params)
    assert out[0] == pytest.approx(g.item())
    assert out[1] == pytest.approx(c.item())
    assert out[2] == pytest.approx(ph_torch(x, params).item())


def test_ts_channel_matches_numpy_and_is_supported():
    adm1 = _adm1()
    assert "TS" in SUPPORTED_CHANNELS
    obs = TorchObservationModel.from_adm1(adm1, {"TS": 0.2})
    x = torch.tensor(_BASE, dtype=torch.float64)
    out = float(obs.predict(x).item())
    # torch channel == standalone ts_torch == numpy calc_total_solids (exact parity)
    assert out == pytest.approx(float(ts_torch(x).item()), rel=1e-12, abs=1e-12)
    assert out == pytest.approx(calc_total_solids(_BASE), rel=1e-12, abs=1e-12)


def test_channel_spec_forms():
    adm1 = _adm1()
    from_list = TorchObservationModel.from_adm1(adm1, ["Q_gas", "Q_ch4"])
    assert from_list.noise_std == [1.0, 1.0]
    from_pairs = TorchObservationModel.from_adm1(adm1, [("Q_gas", 10.0), ("pH", 0.2)])
    assert from_pairs.channel_names == ["Q_gas", "pH"]
    assert from_pairs.noise_std == [10.0, 0.2]


def test_unsupported_channel_rejected():
    with pytest.raises(ValueError):
        TorchObservationModel.from_adm1(_adm1(), ["Q_gas", "temperature"])


def test_supported_channels_exposed():
    for ch in ("Q_gas", "Q_ch4", "Q_co2", "pH", "VFA", "TAC"):
        assert ch in SUPPORTED_CHANNELS


def test_differentiable():
    adm1 = _adm1()
    obs = TorchObservationModel.from_adm1(adm1, ["Q_gas", "pH"])
    x = torch.tensor(_BASE, dtype=torch.float64, requires_grad=True)
    obs.predict(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
