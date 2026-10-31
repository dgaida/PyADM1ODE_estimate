"""Differentiable observation model  h(x)  for learned/PINN estimators.

The recursive filters use :class:`~pyadm1ode_estimation.estimation.observation_model.ObservationModel`,
whose extractors read ``plant.outputs_data`` *after* a numpy simulation step.
A PINN (or any gradient-based estimator) instead needs the sensor signals as a
*differentiable function of the state*, so it can backpropagate the data loss.

This wrapper selects which sensor channels are observed and evaluates them via
the differentiable maps in :mod:`pyadm1.core.adm1_torch`. It operates on the
full 41-state ADM1da vector (batched: ``x`` shaped ``(41,)`` or ``(B, 41)``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from pyadm1.core.adm1_torch import (
    Adm1TorchParams,
    calc_gas_quasi_steady_torch,
    calc_gas_torch,
    ph_torch,
    tac_torch,
    ts_torch,
    vfa_torch,
)

_GAS_CHANNELS = ("Q_gas", "Q_ch4", "Q_co2")

if TYPE_CHECKING:
    from pyadm1.core.adm1 import ADM1

# Channel name -> differentiable map ``f(x, params, soft) -> tensor[...]``.
# ``soft`` only affects the gas channels (leaky vs. hard flow floor).
_CHANNEL_FNS: dict[
    str, Callable[[torch.Tensor, Adm1TorchParams, bool], torch.Tensor]
] = {
    "Q_gas": lambda x, p, soft: calc_gas_torch(x, p, soft=soft)[0],
    "Q_ch4": lambda x, p, soft: calc_gas_torch(x, p, soft=soft)[1],
    "Q_co2": lambda x, p, soft: calc_gas_torch(x, p, soft=soft)[2],
    "pH": lambda x, p, soft: ph_torch(x, p),
    "VFA": lambda x, p, soft: vfa_torch(x),
    "TAC": lambda x, p, soft: tac_torch(x, p),
    "TS": lambda x, p, soft: ts_torch(x),
}

#: Channel names this wrapper can evaluate differentiably.
SUPPORTED_CHANNELS: tuple[str, ...] = tuple(_CHANNEL_FNS)

ChannelSpec = Sequence[str] | dict[str, float] | Sequence[tuple[str, float]]


@dataclass
class TorchObservationModel:
    """A selected set of differentiable sensor channels ``h(x)``.

    Attributes:
        channel_names: Ordered channel identifiers (subset of
            :data:`SUPPORTED_CHANNELS`).
        noise_std: Measurement standard deviation per channel, same order.
        params: Parameter snapshot the maps need (gas constants, Henry
            coefficients, acid-base constants). Build from a calibrated model.
    """

    channel_names: list[str]
    noise_std: list[float]
    params: Adm1TorchParams
    soft_gas: bool = False
    quasi_steady_gas: bool = False

    def __post_init__(self) -> None:
        unknown = [c for c in self.channel_names if c not in _CHANNEL_FNS]
        if unknown:
            raise ValueError(
                f"Unsupported observation channel(s) {unknown}; "
                f"supported: {list(SUPPORTED_CHANNELS)}."
            )
        if len(self.channel_names) != len(self.noise_std):
            raise ValueError("channel_names and noise_std length mismatch.")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("Duplicate channel names.")

    def __len__(self) -> int:
        return len(self.channel_names)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate every channel: ``x (..., 41) -> (..., n_channels)``.

        With ``quasi_steady_gas``, the gas channels are computed from the liquid
        state via the well-conditioned quasi-steady solve
        (:func:`calc_gas_quasi_steady_torch`) instead of ``calc_gas`` on a
        (free, knife-edge) pTOTAL.
        """
        gas_map = None
        if self.quasi_steady_gas and any(
            n in _GAS_CHANNELS for n in self.channel_names
        ):
            qg, qc, qco = calc_gas_quasi_steady_torch(x, self.params)
            gas_map = {"Q_gas": qg, "Q_ch4": qc, "Q_co2": qco}

        cols = []
        for name in self.channel_names:
            if gas_map is not None and name in gas_map:
                cols.append(gas_map[name])
            else:
                cols.append(_CHANNEL_FNS[name](x, self.params, self.soft_gas))
        return torch.stack(cols, dim=-1)

    def noise_std_tensor(self, dtype=torch.float64, device="cpu") -> torch.Tensor:
        """Per-channel ``σ`` as a tensor, for weighting the data loss by 1/σ²."""
        return torch.tensor(self.noise_std, dtype=dtype, device=device)

    @classmethod
    def from_adm1(
        cls,
        adm1: ADM1,
        channels: ChannelSpec,
        soft_gas: bool = False,
        quasi_steady_gas: bool = False,
    ) -> TorchObservationModel:
        """Build from a (calibrated) :class:`ADM1` instance.

        Args:
            adm1: The model whose parameters/influent define ``h`` and ``f``.
            channels: Either a list of channel names (unit noise), a mapping
                ``{name: noise_std}``, or a list of ``(name, noise_std)`` pairs.
            soft_gas: Use the leaky gas-flow floor (see :func:`calc_gas_torch`).
            quasi_steady_gas: Compute the gas channels from the liquid state via
                the quasi-steady solve — the well-conditioned fix for the
                knife-edge biogas map. Recommended for PINN training.
        """
        names, stds = _parse_channels(channels)
        return cls(
            channel_names=names,
            noise_std=stds,
            params=Adm1TorchParams.from_adm1(adm1),
            soft_gas=soft_gas,
            quasi_steady_gas=quasi_steady_gas,
        )


def _parse_channels(channels: ChannelSpec) -> tuple[list[str], list[float]]:
    if isinstance(channels, dict):
        return list(channels.keys()), [float(v) for v in channels.values()]
    names: list[str] = []
    stds: list[float] = []
    for item in channels:
        if isinstance(item, str):
            names.append(item)
            stds.append(1.0)
        else:
            name, std = item
            names.append(str(name))
            stds.append(float(std))
    return names, stds
