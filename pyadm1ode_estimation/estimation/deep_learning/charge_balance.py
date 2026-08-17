"""Quasi-steady charge balance — the pH counterpart of the quasi-steady gas solve.

**The problem.** ADM1 gets pH from electroneutrality. With

    fixed = S_cation - S_anion + (S_nh4 - S_nh3) - S_hco3 - vfa_anions
    S_H   = 0.5 * (-fixed + sqrt(fixed^2 + 4*K_w))

``fixed`` is a difference of terms of order 0.1-0.2 kmol/m³ that cancels down to
~1e-6 at a normal operating point — **5.3 decades of cancellation**, measured on
the benchmark's prior state. Since ``S_H ≈ K_w / fixed`` once ``fixed >>
sqrt(K_w)``, pH is essentially ``-log10(K_w) + log10(fixed)``: perturb any single
ion state by 1 % and ``fixed`` moves by orders of magnitude.

That is the same knife-edge as the biogas map, where ``Q_gas ≈ 1.1e7 * (p_total +
p_H2O - p_ext)`` hangs on a bracket of +0.00017 — and it has the same
consequence for a network that predicts the ion states freely: pH responds with
~50 000 sigma to a 1 % state change, so its gradient dictates every optimiser
step and the fit diverges instead of converging.

**The fix.** Do not let the network guess the quantity the cancellation hangs on.
The gas solve slaves the 4 gas pressures to the liquid state; here we slave
``S_cation``: the network predicts **pH** (a well-scaled ~7.5 output whose
sensitivity is 1:1) and :func:`solve_cation_for_ph` inverts the charge balance
for the ``S_cation`` that produces exactly that pH.

**What this trades away.** ``S_cation`` stops being a differential state and
becomes an algebraic one — the model turns from an ODE into a DAE, with the
trivial dilution equation ``dS_cation/dt = D_in*s_in - D_out*S_cation`` replaced
by the electroneutrality constraint. Callers must therefore exclude ``S_cation``
from the ODE residual (:class:`~.pinn_smoother.PinnSmoother` does this
automatically). This is a deliberate, documented trade: ``S_cation`` is a
bookkeeping charge rather than a measured species, and enforcing its dilution
equation is exactly what creates the ill-conditioning.

For the same reason the solved value may come out slightly **negative** — it then
represents a net anion excess, which is well within how ADM1 uses these two
slots (this dataset's slurry influent already carries a negative ``S_anion``).
"""

from __future__ import annotations

import torch
from pyadm1.core.adm1 import (
    _IDX_S_AC_ION,
    _IDX_S_ANION,
    _IDX_S_BU_ION,
    _IDX_S_CATION,
    _IDX_S_HCO3,
    _IDX_S_NH3,
    _IDX_S_NH4,
    _IDX_S_PRO_ION,
    _IDX_S_VA_ION,
)
from pyadm1.core.adm1_torch import Adm1TorchParams

#: Index of the state this module slaves to the charge balance.
CATION_INDEX = _IDX_S_CATION

# COD-to-charge conversion per VFA, matching ``adm1_torch._calc_ph``.
_COD_AC, _COD_PRO, _COD_BU, _COD_VA = 64.0, 112.0, 160.0, 208.0

#: pH outside this band is treated as a numerical excursion and clamped before
#: the inversion, so a wild network output cannot produce a non-finite state.
PH_LIMITS = (3.0, 12.0)


def charge_without_cation(x: torch.Tensor) -> torch.Tensor:
    """The charge-balance sum ``fixed`` **excluding** the ``S_cation`` term.

    ``fixed = S_cation + charge_without_cation(x)``, so adding the solved cation
    reproduces the quantity :func:`pyadm1.core.adm1_torch.ph_torch` uses.

    Args:
        x: state tensor ``(..., >=37)``.
    """
    vfa_anions = (
        x[..., _IDX_S_AC_ION] / _COD_AC
        + x[..., _IDX_S_PRO_ION] / _COD_PRO
        + x[..., _IDX_S_BU_ION] / _COD_BU
        + x[..., _IDX_S_VA_ION] / _COD_VA
    )
    return (
        -x[..., _IDX_S_ANION]
        + (x[..., _IDX_S_NH4] - x[..., _IDX_S_NH3])
        - x[..., _IDX_S_HCO3]
        - vfa_anions
    )


def solve_cation_for_ph(
    x: torch.Tensor, ph: torch.Tensor, params: Adm1TorchParams
) -> torch.Tensor:
    """The ``S_cation`` that makes :func:`ph_torch` return ``ph`` for state ``x``.

    Closed-form and differentiable — the charge balance is a quadratic in
    ``S_H``, so inverting it for the fixed-charge term is exact:

        S_H   = 10^-pH
        fixed = (K_w - S_H^2) / S_H
        S_cat = fixed - charge_without_cation(x)

    Args:
        x: state tensor ``(..., >=37)``; its own ``S_cation`` slot is ignored.
        ph: target pH, broadcastable to ``x[..., 0]``.
        params: supplies ``K_w``.

    Returns:
        ``S_cation`` shaped like ``x[..., 0]``.
    """
    k_w = float(params.inhib["K_w"])
    ph = torch.clamp(ph, *PH_LIMITS)
    s_h = torch.pow(torch.as_tensor(10.0, dtype=x.dtype, device=x.device), -ph)
    fixed = (k_w - s_h * s_h) / s_h
    return fixed - charge_without_cation(x)


def apply_ph(
    x: torch.Tensor, ph: torch.Tensor, params: Adm1TorchParams
) -> torch.Tensor:
    """``x`` with its ``S_cation`` slot replaced so the state realises ``ph``.

    Out-of-place, so it stays autograd-safe inside a forward pass.
    """
    cation = solve_cation_for_ph(x, ph, params)
    return torch.cat(
        [x[..., :CATION_INDEX], cation.unsqueeze(-1), x[..., CATION_INDEX + 1 :]],
        dim=-1,
    )


__all__ = [
    "CATION_INDEX",
    "PH_LIMITS",
    "apply_ph",
    "charge_without_cation",
    "solve_cation_for_ph",
]
