"""Amortised physics-informed observer for ADM1 state estimation.

Unlike :class:`~pyadm1ode_estimation.estimation.deep_learning.pinn_smoother.PinnSmoother`
(which fits ``t -> x(t)`` from scratch per window), this network is *conditioned
on the measurement stream*: a recurrent encoder (GRU) maps the sequence of
sensor readings + known feed to the state trajectory. It is pre-trained on many
simulated scenarios (see :mod:`observer_data`) and then fine-tuned online.

It reuses the same physical building blocks as the smoother: the network
predicts only the **37 liquid states** (as a log-deviation from a reference, for
positivity), and the 4 gas-phase pressures are slaved via the quasi-steady solve
(:func:`gas_equilibrium_torch`). Physics / measurement losses can then be built
with :func:`adm1da_rhs_torch` and the differentiable observation map exactly as
for the smoother.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from pyadm1.core.adm1_torch import Adm1TorchParams, gas_equilibrium_torch
from torch import nn

_STATE_SIZE = 41
_N_LIQUID = 37


class Adm1Observer(nn.Module):
    """GRU observer: measurement stream -> ADM1 state trajectory.

    Args:
        params: ADM1 parameter snapshot (for the quasi-steady gas solve).
        x_ref: reference state ``(41,)`` — the log-transform base
            (``x = x_ref * exp(raw)``); use the calibrated warmed operating point.
        n_features: input feature dimension (measurements + feed).
        hidden: GRU hidden size.
        num_layers: stacked GRU layers.
        dropout: dropout between GRU layers / for MC-Dropout uncertainty.
        gas_n_iter: Newton iterations for the gas-phase solve.
    """

    def __init__(
        self,
        params: Adm1TorchParams,
        x_ref: Sequence[float],
        n_features: int,
        hidden: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        gas_n_iter: int = 15,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.params = params
        self.gas_n_iter = int(gas_n_iter)
        self.dropout_p = float(dropout)

        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.shape != (_STATE_SIZE,):
            raise ValueError(
                f"x_ref must have shape ({_STATE_SIZE},), got {x_ref.shape}."
            )
        base = np.clip(x_ref[:_N_LIQUID], 1.0e-8, None)
        self.register_buffer("_base", torch.tensor(base, dtype=dtype))

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden, _N_LIQUID)
        self._drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.to(dtype=dtype)
        self._dtype = dtype

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """``u (B, T, n_features) -> x (B, T, 41)``.

        The GRU is causal (each step sees only past + current inputs), so the
        observer works as a streaming filter; the 4 gas states are solved from
        the predicted liquid state at every step.
        """
        h, _ = self.gru(u)
        raw = self.head(self._drop(h))
        x_liq = self._base * torch.exp(torch.clamp(raw, -10.0, 10.0))
        gas = gas_equilibrium_torch(x_liq, self.params, n_iter=self.gas_n_iter)
        return torch.cat([x_liq, gas], dim=-1)

    @classmethod
    def from_adm1(cls, adm1, x_ref, n_features, **kwargs) -> Adm1Observer:
        """Build with parameters snapshotted from a (calibrated) ADM1 instance."""
        return cls(Adm1TorchParams.from_adm1(adm1), x_ref, n_features, **kwargs)
