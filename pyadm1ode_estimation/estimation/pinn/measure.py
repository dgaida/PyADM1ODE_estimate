"""Measurement map ``h(x)`` of the benchmark's online sensors, in torch.

The five hourly sensors of the benchmark - Q_gas, Q_ch4, Q_co2 [m3/d], pH, TS [%] -
are computed from the *core* state through the closure (quasi-steady gas phase,
charge-balance pH), so the same map serves closure-mode and DAE-mode networks.
Against the truth it reproduces the stored measurements at the sensor-noise
level (relative residual std 3 / 4 / 4 %, pH 0.02, TS 0.2 %; gas bias -0.6 %
from the headspace lag).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from pyadm1.core.adm1_torch import (
    Adm1TorchParams,
    calc_gas_quasi_steady_torch,
    ph_torch,
    ts_torch,
)

from .physics import N_CORE, close_state

CHANNELS: tuple[str, ...] = ("Q_gas", "Q_ch4", "Q_co2", "pH", "TS")

_NOISE_RE = re.compile(r"(relative|absolute)\s+([\d.]+)")


@dataclass(frozen=True)
class SensorNoise:
    """Per-channel noise: ``kind`` is ``"relative"`` (fraction) or ``"absolute"``."""

    kind: tuple[str, ...]
    value: tuple[float, ...]

    @classmethod
    def from_meta(cls, meta: dict) -> SensorNoise:
        kinds, values = [], []
        for ch in CHANNELS:
            m = _NOISE_RE.search(str(meta["sensor_noise"][ch]))
            if not m:
                raise ValueError(
                    f"cannot parse sensor noise for {ch}: {meta['sensor_noise'][ch]!r}"
                )
            kinds.append(m.group(1))
            v = float(m.group(2))
            values.append(v / 100.0 if m.group(1) == "relative" else v)
        return cls(tuple(kinds), tuple(values))

    def sigma(self, y: torch.Tensor) -> torch.Tensor:
        """Noise std per channel for measurements ``y (..., 5)``."""
        cols = []
        for j, (k, v) in enumerate(zip(self.kind, self.value)):
            cols.append(
                v * y[..., j].abs()
                if k == "relative"
                else torch.full_like(y[..., j], v)
            )
        return torch.stack(cols, dim=-1)


class MeasurementModel:
    """``h(x_core) -> (Q_gas, Q_ch4, Q_co2, pH, TS)`` and the noise-normalised residual."""

    def __init__(self, params: Adm1TorchParams, noise: SensorNoise):
        self.params = params
        self.noise = noise

    @classmethod
    def from_meta(cls, params: Adm1TorchParams, meta: dict) -> MeasurementModel:
        if list(meta["sensors"]) != list(CHANNELS):
            raise ValueError(
                f"benchmark sensors {meta['sensors']} differ from {CHANNELS}"
            )
        return cls(params, SensorNoise.from_meta(meta))

    def h(self, x: torch.Tensor) -> torch.Tensor:
        """Sensor values ``(..., 5)`` from a core ``(..., 31)`` or full ``(..., 41)`` state."""
        x_full = close_state(x[..., :N_CORE], self.params)
        q_gas, q_ch4, q_co2 = calc_gas_quasi_steady_torch(x_full, self.params)
        return torch.stack(
            [q_gas, q_ch4, q_co2, ph_torch(x_full, self.params), ts_torch(x_full)],
            dim=-1,
        )

    def normalized_residual(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """``(h(x) - y) / sigma(y)``, dimensionless, ``(..., 5)``."""
        return (self.h(x) - y) / self.noise.sigma(y)
