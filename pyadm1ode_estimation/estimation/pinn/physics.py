"""ADM1da physics for the PINN v2: batched RHS, algebraic closures, scaled residuals.

Built from scratch for the depth/width study. The *dynamics* ``f(x, u)`` are
NOT re-implemented: they come from the tested, autograd-differentiable
:func:`pyadm1.core.adm1_torch.adm1da_rhs_torch`. This module adds what a
physics loss needs on top of it (see ``private_docs/pinn_v2_physics_equations.md``):

1. **Batched influent.** ``adm1da_rhs_torch`` reads a scalar ``q_ad`` and a
   float list ``s_in``; :func:`with_influent` swaps in per-row tensors so one
   call evaluates a whole batch of collocation points with different feeds.
2. **Algebraic closures.** The six ion states (31-36) relax with
   ``k_A_B = 1e8 /d`` and the four gas states (37-40) with ``k_p, k_L_a``; as
   ODE residuals they would dominate the loss. :func:`close_state` computes
   them from the 31 "core" states (0-30) by solving the charge balance with
   the ions at acid-base equilibrium and the quasi-steady gas phase.
3. **Conserved sums for S_IC / S_IN.** The core model carries ``+rho_A_co2`` on
   the S_co2 row and ``-rho_A_co2`` on the S_hco3 row (same for N). With
   ``k_A_B = 1e8`` the ion tracks its equilibrium and ``rho_A`` is *whatever
   that tracking needs* - not zero. The clean index-1 reduction adds the two
   rows: ``d(S_co2 + S_hco3)/dt = f_9 + f_35`` has no ``rho_A`` at all. That
   is the residual used for rows 9 and 10 (``acid_base="sum"``). In closure
   mode the time derivative of the closed ion is obtained by forward-mode AD
   through the closure, so the network still predicts only the 31 core states.
4. **Residual selection and scaling.** :class:`ADM1Physics.residual` returns
   ``r_i = dx_i/dt - f_i(x, u)`` for the configured rows, divided by a
   per-row rate scale ``sigma_i`` (RMS of ``|f_i|`` over reference data) so a
   slow, high-level state (X_PS) cannot drown a fast, low-level one (S_ac).

Two verified formulations (finite-difference residual of the benchmark truth,
relative to the RMS rate of each row; see ``private_docs/pinn_v2_physics_equations.md`` §6):

* ``closure=True`` (network -> 31 core states). ODE rows
  :data:`CLOSURE_ODE_STATES` = core minus S_h2 / S_ch4. Those two rows are a
  small net of large opposing terms (production vs. transfer) and the
  quasi-steady gas closure lags the true headspace composition (residence
  ~2.5 h) enough to make them meaningless (residual ~100 % of the rate).
  The truth's ions also sit ~1e-3 off exact equilibrium (finite ``k_A_B``),
  which the steep NH3 / pH inhibitions amplify: S_ac 5-40 %, S_co2 10-14 %,
  S_su / S_aa 2-13 %; every other row < 2 %.
* ``closure=False`` (network -> all 41 states, "DAE mode"). ODE rows
  :data:`FULL_ODE_STATES` = core + the four gas rows (all < 2 %, gas rows
  < 0.5 %), plus six *algebraic* ion-equilibrium residuals
  (:func:`ion_equilibrium_residual`) in place of the ``k_A_B``-stiff ion ODEs.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
from pyadm1.core.adm1 import (
    _IDX_S_AC,
    _IDX_S_AC_ION,
    _IDX_S_ANION,
    _IDX_S_BU,
    _IDX_S_BU_ION,
    _IDX_S_CATION,
    _IDX_S_CH4,
    _IDX_S_CO2,
    _IDX_S_H2,
    _IDX_S_HCO3,
    _IDX_S_NH3,
    _IDX_S_NH4,
    _IDX_S_PRO,
    _IDX_S_PRO_ION,
    _IDX_S_VA,
    _IDX_S_VA_ION,
    STATE_SIZE,
)
from pyadm1.core.adm1_torch import (
    Adm1TorchParams,
    _calc_ph,
    adm1da_rhs_torch,
    gas_equilibrium_torch,
)

from .influent import N_INFLUENT, InfluentModel

#: States carried as ODE residuals (dissolved, particulate, biomass, cation/anion).
CORE_STATES: tuple[int, ...] = tuple(range(31))
#: Ion states closed algebraically (acid-base equilibrium).
ION_STATES: tuple[int, ...] = tuple(range(31, 37))
#: Gas states closed algebraically (quasi-steady headspace).
GAS_STATES: tuple[int, ...] = tuple(range(37, 41))
N_CORE = len(CORE_STATES)

#: Default ODE rows with closure: core minus the gas-slaved S_h2 / S_ch4 rows.
CLOSURE_ODE_STATES: tuple[int, ...] = tuple(
    i for i in CORE_STATES if i not in (_IDX_S_H2, _IDX_S_CH4)
)
#: Default ODE rows without closure ("DAE mode"): core + the four gas rows.
FULL_ODE_STATES: tuple[int, ...] = CORE_STATES + GAS_STATES

#: Rows whose residual is formed on the conserved sum ``x_i + x_j`` (``acid_base="sum"``).
CONSERVED_PAIRS: dict[int, int] = {_IDX_S_CO2: _IDX_S_HCO3, _IDX_S_NH4: _IDX_S_NH3}

#: (ion row, total row, K_a key) of the six acid-base pairs.
ION_PAIRS: tuple[tuple[int, int, str], ...] = (
    (_IDX_S_VA_ION, _IDX_S_VA, "K_a_va"),
    (_IDX_S_BU_ION, _IDX_S_BU, "K_a_bu"),
    (_IDX_S_PRO_ION, _IDX_S_PRO, "K_a_pro"),
    (_IDX_S_AC_ION, _IDX_S_AC, "K_a_ac"),
    (_IDX_S_HCO3, _IDX_S_CO2, "K_a_co2"),
    (_IDX_S_NH3, _IDX_S_NH4, "K_a_IN"),
)

ACID_BASE_MODES = ("sum", "drop", "keep")

# COD-to-mole conversions of the VFA anions in the charge balance [kg COD / kmol].
_COD_VA, _COD_BU, _COD_PRO, _COD_AC = 208.0, 160.0, 112.0, 64.0


# ---------------------------------------------------------------------------
# Parameters and batched influent
# ---------------------------------------------------------------------------
def build_params(
    plant_or_adm1,
    *,
    kinetic_factors=None,
    kinetic_keys=None,
    digester_id: str = "primary",
) -> Adm1TorchParams:
    """Snapshot the ADM1da parameters, optionally with per-series kinetic factors.

    The benchmark stores a multiplicative lognormal factor per kinetic key
    (``kinetic_keys`` / ``kinetic_factors``) that the truth was generated with.
    Applying them here makes the physics residual of the truth trajectory
    (near) zero; leaving them out evaluates the *nominal* model, which is what a
    PINN in the field would have.
    """
    adm1 = (
        plant_or_adm1.components[digester_id].adm1
        if hasattr(plant_or_adm1, "components")
        else plant_or_adm1
    )
    p = Adm1TorchParams.from_adm1(adm1)
    if kinetic_factors is not None:
        if kinetic_keys is None:
            raise ValueError("kinetic_factors needs kinetic_keys")
        for key, f in zip(kinetic_keys, kinetic_factors):
            key = str(key)
            if key not in p.kinetic:
                raise KeyError(f"unknown kinetic key {key!r}")
            p.kinetic[key] = p.kinetic[key] * float(f)
    return p


def with_influent(
    params: Adm1TorchParams, q_ad: torch.Tensor, s_in: torch.Tensor
) -> Adm1TorchParams:
    """Copy of ``params`` whose influent is a batch: ``q_ad (B,)``, ``s_in (B, 37)``.

    ``adm1da_rhs_torch`` only ever multiplies ``q_ad`` and indexes ``s_in[i]``,
    so per-row tensors broadcast through it unchanged (``D_in * s_in[i]`` with
    both shaped ``(B,)``). Any outflow override is dropped: it is a scalar of
    the snapshot and cannot follow a batch.
    """
    if s_in.shape[-1] != N_INFLUENT:
        raise ValueError(f"s_in must have {N_INFLUENT} columns, got {s_in.shape}")
    cols = [s_in[..., i] for i in range(N_INFLUENT)]
    return dataclasses.replace(params, q_ad=q_ad, s_in=cols, Q_out_override=None)


def rhs(
    x: torch.Tensor, params: Adm1TorchParams, q_ad: torch.Tensor, s_in: torch.Tensor
) -> torch.Tensor:
    """``dx/dt`` for a batch ``x (B, 41)`` with per-row influent."""
    return adm1da_rhs_torch(x, with_influent(params, q_ad, s_in))


# ---------------------------------------------------------------------------
# Closures: ions from the charge balance, gas from the quasi-steady headspace
# ---------------------------------------------------------------------------
def _speciation(
    S_H: torch.Tensor, core: torch.Tensor, ip: dict
) -> tuple[torch.Tensor, ...]:
    """Ion concentrations at acid-base equilibrium for proton concentration ``S_H``."""

    def ion(K_a: float, total: torch.Tensor) -> torch.Tensor:
        return K_a * total / (K_a + S_H)

    return (
        ion(ip["K_a_va"], core[..., _IDX_S_VA]),
        ion(ip["K_a_bu"], core[..., _IDX_S_BU]),
        ion(ip["K_a_pro"], core[..., _IDX_S_PRO]),
        ion(ip["K_a_ac"], core[..., _IDX_S_AC]),
        ion(ip["K_a_co2"], core[..., _IDX_S_CO2]),
        ion(ip["K_a_IN"], core[..., _IDX_S_NH4]),
    )


def _charge_balance(S_H: torch.Tensor, core: torch.Tensor, ip: dict) -> torch.Tensor:
    """Net charge ``F(S_H)`` with the ions at equilibrium; monotone increasing in ``S_H``."""
    va, bu, pro, ac, hco3, nh3 = _speciation(S_H, core, ip)
    return (
        core[..., _IDX_S_CATION]
        - core[..., _IDX_S_ANION]
        + (core[..., _IDX_S_NH4] - nh3)
        - hco3
        - ac / _COD_AC
        - pro / _COD_PRO
        - bu / _COD_BU
        - va / _COD_VA
        + S_H
        - ip["K_w"] / S_H
    )


def solve_charge_balance(
    core: torch.Tensor, params: Adm1TorchParams, n_iter: int = 60
) -> torch.Tensor:
    """Proton concentration ``S_H`` of the core state with all ions at equilibrium.

    Bisection on pH in ``[0, 14]`` on a *detached* copy (``F`` is monotone, so
    the bracket is safe and 60 halvings reach float64 resolution), followed by
    one differentiable Newton step at the root: the value is unchanged, the
    gradient (reverse or forward mode) flows via the implicit function theorem
    ``dS_H/dcore = -F_core / F_S_H``.
    """
    ip = params.inhib
    c0 = core.detach()
    lo = torch.zeros_like(c0[..., 0])  # pH 0
    hi = torch.full_like(c0[..., 0], 14.0)  # pH 14
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _charge_balance(10.0 ** (-mid), c0, ip)
        # F increases with S_H, i.e. decreases with pH: F(mid) > 0 -> root at higher pH.
        lo = torch.where(f_mid > 0.0, mid, lo)
        hi = torch.where(f_mid > 0.0, hi, mid)
    S_H = 10.0 ** (-0.5 * (lo + hi))
    # One differentiable Newton step on the live ``core`` (implicit gradient).
    va, bu, pro, ac, hco3, nh3 = _speciation(S_H, core, ip)
    f = _charge_balance(S_H, core, ip)
    # dF/dS_H = 1 + K_w/S_H^2 + sum_i x_i / (K_a_i + S_H)   (each ion term is decreasing in S_H)
    dF = (
        1.0
        + ip["K_w"] / (S_H * S_H)
        + nh3 / (ip["K_a_IN"] + S_H)
        + hco3 / (ip["K_a_co2"] + S_H)
        + (ac / _COD_AC) / (ip["K_a_ac"] + S_H)
        + (pro / _COD_PRO) / (ip["K_a_pro"] + S_H)
        + (bu / _COD_BU) / (ip["K_a_bu"] + S_H)
        + (va / _COD_VA) / (ip["K_a_va"] + S_H)
    )
    return torch.clamp(S_H - f / dF, min=1.0e-14)


def close_state(
    core: torch.Tensor, params: Adm1TorchParams, n_gas_iter: int = 25
) -> torch.Tensor:
    """Extend ``core (B, 31)`` to the full ``(B, 41)`` state with closed ions and gas.

    Ions: acid-base equilibrium at the charge-balance pH. Gas: quasi-steady
    headspace from :func:`pyadm1.core.adm1_torch.gas_equilibrium_torch`.
    """
    if core.shape[-1] != N_CORE:
        raise ValueError(f"core must have {N_CORE} columns, got {core.shape}")
    S_H = solve_charge_balance(core, params)
    ions = torch.stack(_speciation(S_H, core, params.inhib), dim=-1)
    zeros = torch.zeros(
        *core.shape[:-1], len(GAS_STATES), dtype=core.dtype, device=core.device
    )
    gas = gas_equilibrium_torch(
        torch.cat([core, ions, zeros], dim=-1), params, n_iter=n_gas_iter
    )
    return torch.cat([core, ions, gas], dim=-1)


def close_state_with_rates(
    core: torch.Tensor, dcore: torch.Tensor, params: Adm1TorchParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(x_full, dx_full/dt)`` from the core state and its time derivative.

    The closed states are functions of the core, so their rate is the
    Jacobian-vector product ``(d close/d core) · dcore``; forward-mode AD gives
    it in one pass and keeps the result differentiable for training.
    """
    return torch.func.jvp(lambda c: close_state(c, params), (core,), (dcore,))


def proton_concentration(x: torch.Tensor, params: Adm1TorchParams) -> torch.Tensor:
    """``S_H`` of a full 41-state vector from its charge balance (as the core RHS does)."""
    return _calc_ph(
        x[..., _IDX_S_NH4],
        x[..., _IDX_S_NH3],
        x[..., _IDX_S_HCO3],
        x[..., _IDX_S_AC_ION],
        x[..., _IDX_S_PRO_ION],
        x[..., _IDX_S_BU_ION],
        x[..., _IDX_S_VA_ION],
        x[..., _IDX_S_CATION],
        x[..., _IDX_S_ANION],
        params.inhib["K_w"],
    )


def ion_equilibrium_residual(x: torch.Tensor, params: Adm1TorchParams) -> torch.Tensor:
    """Dimensionless acid-base equilibrium residuals of a full state, ``(..., 6)``.

    ``(S_ion (K_a + S_H) - K_a S_tot) / (K_a S_tot)`` for the six pairs in
    :data:`ION_PAIRS` - the ``k_A_B``-free form of the ion ODEs' steady state.
    Zero when the ions are at equilibrium with the state's own pH; used instead
    of the stiff ion ODE rows when the network predicts all 41 states.
    """
    ip = params.inhib
    S_H = proton_concentration(x, params)
    out = []
    for i_ion, i_tot, key in ION_PAIRS:
        K_a = ip[key]
        tot = x[..., i_tot]
        out.append((x[..., i_ion] * (K_a + S_H) - K_a * tot) / (K_a * tot + 1.0e-30))
    return torch.stack(out, dim=-1)


def acid_base_rates(
    x: torch.Tensor, params: Adm1TorchParams
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rho_A_co2, rho_A_IN)`` exactly as the core RHS computes them."""
    ip = params.inhib
    S_H = _calc_ph(
        x[..., _IDX_S_NH4],
        x[..., _IDX_S_NH3],
        x[..., _IDX_S_HCO3],
        x[..., _IDX_S_AC_ION],
        x[..., _IDX_S_PRO_ION],
        x[..., _IDX_S_BU_ION],
        x[..., _IDX_S_VA_ION],
        x[..., _IDX_S_CATION],
        x[..., _IDX_S_ANION],
        ip["K_w"],
    )
    k_AB = ip["k_A_B"]
    S_co2, S_hco3 = x[..., _IDX_S_CO2], x[..., _IDX_S_HCO3]
    S_nh4, S_nh3 = x[..., _IDX_S_NH4], x[..., _IDX_S_NH3]
    rho_co2 = k_AB * (S_hco3 * S_H - ip["K_a_co2"] * (S_co2 - S_hco3))
    rho_in = k_AB * (S_nh3 * S_H - ip["K_a_IN"] * (S_nh4 - S_nh3))
    return rho_co2, rho_in


# ---------------------------------------------------------------------------
# Residual
# ---------------------------------------------------------------------------
@dataclass
class PhysicsConfig:
    """What the physics residual contains.

    Attributes:
        closure: ``True`` -> the network predicts the 31 core states and ions/gas
            are closed algebraically; ``False`` -> the network predicts all 41
            ("DAE mode").
        ode_indices: State rows carried as ODE residuals. ``None`` (default)
            resolves to :data:`CLOSURE_ODE_STATES` or :data:`FULL_ODE_STATES`.
        acid_base: How rows 9 (S_co2) and 10 (S_nh4) are formed.
            ``"sum"`` (default) - residual of the conserved sums
            ``S_co2 + S_hco3`` / ``S_nh4 + S_nh3`` (exact reduction, no ``rho_A``);
            ``"drop"`` - raw rows with the ``rho_A`` terms removed (approximate);
            ``"keep"`` - raw rows as the core model writes them.
        ion_equilibrium: In DAE mode append the six dimensionless
            acid-base equilibrium residuals (:func:`ion_equilibrium_residual`)
            after the ODE rows. Ignored with closure.
        scale: Per-ODE-row rate scale ``sigma_i`` (same order as ``ode_indices``);
            ``None`` = unscaled. Set by :meth:`ADM1Physics.fit_scale`. The
            equilibrium residuals are already dimensionless and are not scaled.
    """

    closure: bool = True
    ode_indices: tuple[int, ...] | None = None
    acid_base: str = "sum"
    ion_equilibrium: bool = True
    scale: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.ode_indices is None:
            self.ode_indices = CLOSURE_ODE_STATES if self.closure else FULL_ODE_STATES
        self.ode_indices = tuple(int(i) for i in self.ode_indices)
        if self.closure and any(i >= N_CORE for i in self.ode_indices):
            raise ValueError(
                "with closure=True only core rows (0-30) can be ODE residuals"
            )
        if any(i in ION_STATES for i in self.ode_indices):
            raise ValueError(
                "ion rows (31-36) are k_A_B-stiff; use ion_equilibrium instead of ODE residuals"
            )
        if self.acid_base not in ACID_BASE_MODES:
            raise ValueError(
                f"acid_base must be one of {ACID_BASE_MODES}, got {self.acid_base!r}"
            )
        if self.scale is not None:
            self.scale = np.asarray(self.scale, dtype=float)
            if self.scale.shape != (len(self.ode_indices),):
                raise ValueError("scale must have one entry per ode_index")

    @property
    def n_ode(self) -> int:
        return len(self.ode_indices)

    @property
    def n_algebraic(self) -> int:
        return len(ION_PAIRS) if (self.ion_equilibrium and not self.closure) else 0


class ADM1Physics:
    """Physics residual of the ADM1da for a PINN.

    Args:
        params: Parameter snapshot (see :func:`build_params`).
        influent: Feed -> influent map (see :class:`InfluentModel`).
        config: Residual configuration.

    All tensor inputs are batched along the leading dimensions; ``x`` and
    ``dxdt`` are ``(..., 31)`` with closure or ``(..., 41)`` without.
    """

    def __init__(
        self,
        params: Adm1TorchParams,
        influent: InfluentModel,
        config: PhysicsConfig | None = None,
    ):
        self.params = params
        self.influent = influent
        self.config = config or PhysicsConfig()

    @property
    def n_state(self) -> int:
        return N_CORE if self.config.closure else STATE_SIZE

    @property
    def n_ode(self) -> int:
        return self.config.n_ode

    @property
    def n_residual(self) -> int:
        """Columns returned by :meth:`residual`: ODE rows + algebraic rows."""
        return self.config.n_ode + self.config.n_algebraic

    # -- state ------------------------------------------------------------
    def full_state(self, x: torch.Tensor) -> torch.Tensor:
        """``(..., 41)`` state: closure applied when configured, else ``x`` itself."""
        if self.config.closure:
            return close_state(x, self.params)
        if x.shape[-1] != STATE_SIZE:
            raise ValueError(f"expected {STATE_SIZE} state columns, got {x.shape}")
        return x

    def full_state_with_rates(
        self, x: torch.Tensor, dxdt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x_full, dx_full/dt)``; the closed rows' rates come from forward-mode AD."""
        if dxdt.shape != x.shape:
            raise ValueError("dxdt must have the same shape as x")
        if self.config.closure:
            return close_state_with_rates(x, dxdt, self.params)
        return self.full_state(x), dxdt

    # -- dynamics ---------------------------------------------------------
    def rhs(self, x_full: torch.Tensor, feed: torch.Tensor) -> torch.Tensor:
        """Raw ``f(x, u)``, all 41 rows, for the full state and feed ``(..., n_sub)``."""
        q_ad, s_in = self.influent(feed)
        return rhs(x_full, self.params, q_ad, s_in)

    def _rows(self, v41: torch.Tensor, model: bool) -> torch.Tensor:
        """Select the ODE rows from a 41-column rate tensor, applying the acid-base mode.

        ``model=True`` for ``f`` (the ``"drop"`` mode subtracts ``rho_A`` here);
        ``model=False`` for a state rate ``dx/dt``.
        """
        mode = self.config.acid_base
        idx = list(self.config.ode_indices)
        out = v41[..., idx]
        if mode == "sum":
            for j, i in enumerate(idx):
                partner = CONSERVED_PAIRS.get(i)
                if partner is not None:
                    out = out.clone() if j == 0 else out
                    out[..., j] = out[..., j] + v41[..., partner]
        return out

    def model_rates(self, x_full: torch.Tensor, feed: torch.Tensor) -> torch.Tensor:
        """Effective ``f_i`` per configured row, ``(..., n_ode)``."""
        f = self.rhs(x_full, feed)
        if self.config.acid_base == "drop":
            rho_co2, rho_in = acid_base_rates(x_full, self.params)
            corr = torch.zeros_like(f)
            corr[..., _IDX_S_CO2] = rho_co2
            corr[..., _IDX_S_NH4] = rho_in
            f = f - corr
        return self._rows(f, model=True)

    def state_rates(self, dx_full: torch.Tensor) -> torch.Tensor:
        """Effective ``dx_i/dt`` per configured row, ``(..., n_ode)``."""
        return self._rows(dx_full, model=False)

    # -- residual ---------------------------------------------------------
    def residual(
        self,
        x: torch.Tensor,
        dxdt: torch.Tensor,
        feed: torch.Tensor,
        scaled: bool = True,
    ) -> torch.Tensor:
        """Physics residual ``(..., n_ode [+ 6])``.

        Columns ``0 .. n_ode-1``: ``r_i = dx_i/dt - f_i(x, u)`` on the configured
        rows (scaled by ``sigma_i`` when set). In DAE mode with
        ``ion_equilibrium`` the six dimensionless equilibrium residuals follow.
        ``dxdt`` is the time derivative of the network output ``x`` (same shape).
        """
        x_full, dx_full = self.full_state_with_rates(x, dxdt)
        r = self.state_rates(dx_full) - self.model_rates(x_full, feed)
        if scaled and self.config.scale is not None:
            r = r / torch.as_tensor(self.config.scale, dtype=r.dtype, device=r.device)
        if self.config.n_algebraic:
            r = torch.cat([r, ion_equilibrium_residual(x_full, self.params)], dim=-1)
        return r

    def fit_scale(
        self,
        states,
        feed,
        *,
        mode: str = "jacobian",
        n_jacobian: int = 64,
        chunk: int = 4096,
        floor: float = 1.0e-12,
    ) -> np.ndarray:
        """Set the per-row residual scale ``sigma_i`` from reference data.

        ``states (N, 41|31)`` / ``feed (N, n_sub)`` are the truth of the observed
        window. Two modes:

        ``"rate"``
            ``sigma_i = RMS(f_i)``, the net rate. Right for a network that already
            sits on the slow manifold, but for rows whose net rate is a small
            difference of large gross terms (gas rows, S_h2, S_ch4) a network
            that is a few percent off produces residuals of 1e6-1e11 times the
            net rate, and the physics loss then ignores everything else.
        ``"jacobian"`` (default)
            ``sigma_i = sqrt(RMS(f_i)^2 + RMS_t(sum_j (J_ij s_j)^2))`` with the
            RHS Jacobian ``J = df/dx`` at ``n_jacobian`` reference points and the
            per-state variation ``s_j = std(x_j)`` over the window. The second
            term is the rate error a *typical-variation-sized* state error
            causes in row ``i``, so ``r_i / sigma_i ~ 1`` means "as wrong as a
            state that is off by its own spread" in fast and slow rows alike.
        """
        if mode not in ("rate", "jacobian"):
            raise ValueError("mode must be 'rate' or 'jacobian'")
        states = np.asarray(states, dtype=float)
        feed = np.asarray(feed, dtype=float)
        if states.ndim != 2 or feed.ndim != 2 or len(states) != len(feed):
            raise ValueError("states and feed must be (N, ·) with equal N")
        sq = torch.zeros(self.n_ode, dtype=torch.float64)
        n = 0
        with torch.no_grad():
            for a in range(0, len(states), chunk):
                xs = torch.as_tensor(
                    states[a : a + chunk, : self.n_state], dtype=torch.float64
                )
                fs = torch.as_tensor(feed[a : a + chunk], dtype=torch.float64)
                f = self.model_rates(self.full_state(xs), fs)
                sq += (f * f).sum(dim=0)
                n += len(xs)
        scale_sq = sq / max(n, 1)

        if mode == "jacobian":
            n_state = self.n_state
            spread = torch.as_tensor(
                states[:, :n_state].std(axis=0), dtype=torch.float64
            )
            pick = np.linspace(0, len(states) - 1, min(n_jacobian, len(states))).astype(
                int
            )
            jac_sq = torch.zeros(self.n_ode, dtype=torch.float64)
            for k in pick:
                xk = torch.as_tensor(states[k, :n_state], dtype=torch.float64)
                fk = torch.as_tensor(feed[k], dtype=torch.float64)

                def _rates(xv, fk=fk):
                    xf = self.full_state(xv.unsqueeze(0))
                    return self.model_rates(xf, fk.unsqueeze(0))[0]

                J = torch.autograd.functional.jacobian(_rates, xk, vectorize=True)
                jac_sq += ((J * spread) ** 2).sum(dim=1)
            scale_sq = scale_sq + jac_sq / len(pick)

        scale = torch.sqrt(scale_sq).clamp(min=floor).numpy()
        self.config.scale = scale
        return scale


# ---------------------------------------------------------------------------
# Verification helper: residual of a stored trajectory by finite differences
# ---------------------------------------------------------------------------
def trajectory_residual(
    physics: ADM1Physics,
    states: np.ndarray,
    feed: np.ndarray,
    time: np.ndarray,
    *,
    dtype: torch.dtype = torch.float64,
) -> dict[str, np.ndarray | None]:
    """Physics residual of a trajectory with ``dx/dt`` from central differences.

    Used to verify the physics against simulator output (the truth of a
    dataset): with the right parameters and influent the scaled residual of a
    smooth, slow state is small, bounded by the finite-difference error. Returns
    ``model_rates`` (``f_i``), ``state_rates`` (finite difference), ``r``
    (unscaled), ``r_scaled`` (``None`` if no scale is set) and ``ion_eq`` (the
    equilibrium residuals, ``None`` with closure), all on the interior points
    ``1 .. T-2``; the rate arrays are restricted to the configured ODE rows.
    """
    states = np.asarray(states, dtype=float)
    feed = np.asarray(feed, dtype=float)
    time = np.asarray(time, dtype=float)
    x = states[:, : physics.n_state]
    dxdt = (x[2:] - x[:-2]) / (time[2:] - time[:-2])[:, None]
    with torch.no_grad():
        xt = torch.as_tensor(x[1:-1], dtype=dtype)
        dt = torch.as_tensor(dxdt, dtype=dtype)
        ft = torch.as_tensor(feed[1:-1], dtype=dtype)
        x_full, dx_full = physics.full_state_with_rates(xt, dt)
        f = physics.model_rates(x_full, ft).numpy()
        d = physics.state_rates(dx_full).numpy()
        ion_eq = (
            None
            if physics.config.closure
            else ion_equilibrium_residual(x_full, physics.params).numpy()
        )
    r = d - f
    out: dict[str, np.ndarray | None] = {
        "model_rates": f,
        "state_rates": d,
        "r": r,
        "r_scaled": None,
        "ion_eq": ion_eq,
    }
    if physics.config.scale is not None:
        out["r_scaled"] = r / physics.config.scale
    return out
