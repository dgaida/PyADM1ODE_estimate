"""Trainable kinetic factors for the inverse PINN.

The benchmark perturbs every kinetic constant whose name starts with one of
``k_dis, k_hyd, k_m_, k_dec, K_S`` by a lognormal factor ``exp(N(0, 0.25))``
(``meta["kinetic_perturbation"]``). A field PINN does not know those factors,
so here they become parameters of the loss: ``k_i = k_i,nominal * exp(theta_i)``
with the generator's own lognormal prior on ``theta``. The factors enter the
physics through the parameter snapshot's kinetic dict, so
:func:`pyadm1.core.adm1_torch.adm1da_rhs_torch` needs no change - it multiplies
whatever sits in the dict, and autograd carries the gradient back to ``theta``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch
from pyadm1.core.adm1_torch import Adm1TorchParams
from torch import nn

DEFAULT_PREFIXES: tuple[str, ...] = ("k_dis", "k_hyd", "k_m_", "k_dec", "K_S")


def perturbed_keys(params: Adm1TorchParams, prefixes=DEFAULT_PREFIXES) -> list[str]:
    """Kinetic keys the benchmark perturbs (dict order of the snapshot)."""
    return [k for k in params.kinetic if any(k.startswith(p) for p in prefixes)]


class KineticFactors(nn.Module):
    """``theta`` (n_keys,) log-factors with a Gaussian prior ``N(0, sigma_prior^2)``."""

    def __init__(
        self,
        keys: list[str],
        sigma_prior: float = 0.25,
        dtype: torch.dtype = torch.float64,
    ):
        super().__init__()
        self.keys = [str(k) for k in keys]
        self.sigma_prior = float(sigma_prior)
        self.theta = nn.Parameter(torch.zeros(len(self.keys), dtype=dtype))

    def apply(self, base: Adm1TorchParams) -> Adm1TorchParams:
        """Copy of ``base`` whose perturbed kinetic constants are ``k * exp(theta)`` tensors."""
        kinetic = dict(base.kinetic)
        for i, k in enumerate(self.keys):
            kinetic[k] = float(base.kinetic[k]) * torch.exp(self.theta[i])
        return dataclasses.replace(base, kinetic=kinetic)

    def prior_loss(self) -> torch.Tensor:
        """Mean squared ``theta / sigma_prior`` (the generator's lognormal prior)."""
        return ((self.theta / self.sigma_prior) ** 2).mean()

    def factors(self) -> np.ndarray:
        return torch.exp(self.theta.detach()).cpu().numpy()

    def report(self, true_keys, true_factors) -> dict:
        """Estimated vs. true factors: log-RMSE of the estimate and of the nominal model (theta = 0)."""
        true = {str(k): float(f) for k, f in zip(true_keys, true_factors)}
        est = self.factors()
        log_err, log_nom, per_key = [], [], {}
        for k, f_est in zip(self.keys, est):
            if k not in true:
                continue
            e, n = float(np.log(f_est) - np.log(true[k])), float(np.log(true[k]))
            log_err.append(e)
            log_nom.append(n)
            per_key[k] = {"estimated": float(f_est), "true": true[k], "log_error": e}
        log_err, log_nom = np.asarray(log_err), np.asarray(log_nom)
        return {
            "log_rmse_estimated": (
                float(np.sqrt(np.mean(log_err**2))) if len(log_err) else float("nan")
            ),
            "log_rmse_nominal": (
                float(np.sqrt(np.mean(log_nom**2))) if len(log_nom) else float("nan")
            ),
            "corr_log": (
                float(np.corrcoef(log_nom, log_nom - log_err)[0, 1])
                if len(log_nom) > 2
                else float("nan")
            ),
            "per_key": per_key,
        }
