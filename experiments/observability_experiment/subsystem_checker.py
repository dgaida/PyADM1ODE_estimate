"""STRIKE-GOLDD-equivalent observability checker for ADM1da subsystems.

Implements the same Sedoglavic-2002 numerical-rank observability test
that STRIKE-GOLDD's FISPO/ORC-DF perform internally:

    1. Build symbolic f(x, u, θ) and h(x, u, θ).
    2. Iteratively compute Lie derivatives  L_f^k h.
    3. Lambdify each row's Jacobian w.r.t. x.
    4. Evaluate at a random sample point of (x, u).
    5. Stack into the observability matrix and check numerical rank.
    6. Stop when rank == n_states or a time budget is exceeded.

Run all six subsystems (A, B, C, D, E, AD_combined) sequentially:

    python subsystem_checker.py

Output: a result table on stdout + a markdown report in results.md.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np
import psutil
import sympy as sp

# ---------------------------------------------------------------------------
# Profiling utilities
# ---------------------------------------------------------------------------


@dataclass
class SubsystemSpec:
    name: str
    label: str
    x_names: List[str]
    u_names: List[str]
    f_builder: Callable
    h_builder: Callable
    boundary_note: str = ""


@dataclass
class ProfileResult:
    name: str
    label: str
    n_states: int
    n_outputs: int
    n_iters_done: int
    rank: int
    fully_observable: bool
    wall_s: float
    peak_mem_mib: float
    note: str = ""


def _mem_mib() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


# ---------------------------------------------------------------------------
# Core: iterated Lie derivatives + numerical rank
# ---------------------------------------------------------------------------


def _lie(h: sp.Matrix, f: sp.Matrix, x: List[sp.Symbol]) -> sp.Matrix:
    """L_f h = (∂h/∂x) · f."""
    return h.jacobian(x) * f


def _safe_lambdify_eval(
    expr: sp.Matrix, vars_: List[sp.Symbol], values: List[float]
) -> np.ndarray:
    """Lambdify expr w.r.t. vars_ and evaluate at values. Returns 2D ndarray."""
    fn = sp.lambdify(vars_, expr, "numpy")
    out = np.asarray(fn(*values), dtype=float)
    # Ensure 2D shape
    if out.ndim == 0:
        out = out.reshape(1, 1)
    elif out.ndim == 1:
        out = out.reshape(1, -1)
    return out


def check_observability(
    spec: SubsystemSpec,
    time_budget_s: float = 1800.0,
    max_iters: int | None = None,
    verbose: bool = True,
) -> ProfileResult:
    """Run the Sedoglavic test on one subsystem."""
    gc.collect()
    mem_start = _mem_mib()
    t0 = time.perf_counter()

    x = [sp.Symbol(n, positive=True) for n in spec.x_names]
    u = [sp.Symbol(n) for n in spec.u_names]

    f_list = spec.f_builder(x, u)
    h_list = spec.h_builder(x, u)
    f = sp.Matrix(f_list)
    h = sp.Matrix(h_list)
    n_states = len(x)
    n_outputs = len(h_list)

    if max_iters is None:
        # Heuristic upper bound: n_states + 2 iterations is enough for any
        # nonlinear observability question that has a finite answer.
        max_iters = n_states + 2

    # Random sample point: positive rationals away from zero, away from one
    # to avoid accidental cancellations.
    rng = np.random.default_rng(42)
    sample = {}
    for v in x + u:
        sample[v] = float(0.13 + 0.7 * rng.random())
    all_vars = x + u
    sample_values = [sample[v] for v in all_vars]

    rows_numeric: List[np.ndarray] = []

    # Iteration 0: ∂h/∂x
    try:
        J0 = h.jacobian(x)
        rows_numeric.append(_safe_lambdify_eval(J0, all_vars, sample_values))
    except Exception as e:
        return ProfileResult(
            name=spec.name,
            label=spec.label,
            n_states=n_states,
            n_outputs=n_outputs,
            n_iters_done=0,
            rank=0,
            fully_observable=False,
            wall_s=time.perf_counter() - t0,
            peak_mem_mib=_mem_mib() - mem_start,
            note=f"FAIL at iter 0: {type(e).__name__}: {e}",
        )

    obs_matrix = np.vstack(rows_numeric)
    rank = int(np.linalg.matrix_rank(obs_matrix, tol=1e-9))
    if verbose:
        print(f"  iter 0: rank {rank}/{n_states}", flush=True)

    if rank >= n_states:
        wall = time.perf_counter() - t0
        return ProfileResult(
            name=spec.name,
            label=spec.label,
            n_states=n_states,
            n_outputs=n_outputs,
            n_iters_done=0,
            rank=rank,
            fully_observable=True,
            wall_s=wall,
            peak_mem_mib=_mem_mib() - mem_start,
            note="Rank reached at output Jacobian (iter 0).",
        )

    Lkh = h
    iters_done = 0
    for k in range(1, max_iters + 1):
        iters_done = k
        try:
            Lkh = _lie(Lkh, f, x)
            Jk = Lkh.jacobian(x)
            rows_numeric.append(_safe_lambdify_eval(Jk, all_vars, sample_values))
        except Exception as e:
            return ProfileResult(
                name=spec.name,
                label=spec.label,
                n_states=n_states,
                n_outputs=n_outputs,
                n_iters_done=k - 1,
                rank=rank,
                fully_observable=False,
                wall_s=time.perf_counter() - t0,
                peak_mem_mib=_mem_mib() - mem_start,
                note=f"FAIL at iter {k}: {type(e).__name__}: {e}",
            )

        obs_matrix = np.vstack(rows_numeric)
        rank = int(np.linalg.matrix_rank(obs_matrix, tol=1e-9))
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"  iter {k}: rank {rank}/{n_states}  (cumul. {elapsed:.1f}s)",
                flush=True,
            )

        if rank >= n_states:
            break

        if time.perf_counter() - t0 > time_budget_s:
            return ProfileResult(
                name=spec.name,
                label=spec.label,
                n_states=n_states,
                n_outputs=n_outputs,
                n_iters_done=k,
                rank=rank,
                fully_observable=False,
                wall_s=time.perf_counter() - t0,
                peak_mem_mib=_mem_mib() - mem_start,
                note=f"Time-budget exceeded after {k} iters.",
            )

    wall = time.perf_counter() - t0
    return ProfileResult(
        name=spec.name,
        label=spec.label,
        n_states=n_states,
        n_outputs=n_outputs,
        n_iters_done=iters_done,
        rank=rank,
        fully_observable=(rank == n_states),
        wall_s=wall,
        peak_mem_mib=_mem_mib() - mem_start,
    )


# ---------------------------------------------------------------------------
# Concrete numerical parameter values
# ---------------------------------------------------------------------------
# Substituting numerical values for parameters keeps Lie derivatives small.
# These match the order of magnitude of typical ADM1da operating values;
# structural observability holds for almost all parameter values, so the
# specific numbers do not change the answer.

V_LIQ, V_GAS, RT = 1200.0, 216.0, 2.6e-2
K_LA = 200.0
K_H_CH4, K_H_CO2, K_H_H2 = 1.8, 27.0, 7.4e-4
K_P, P_GAS_H2O, P_EXT = 1.0e4, 0.06, 1.013
K_M_AC, K_S_AC, Y_AC, K_DEC_AC = 8.0, 0.15, 0.05, 0.04
K_M_H2, K_S_H2, Y_H2, K_DEC_H2 = 35.0, 7e-6, 0.06, 0.02
K_M_SU, K_S_SU, Y_SU, K_DEC_SU = 30.0, 0.5, 0.10, 0.02
K_M_AA, K_S_AA, Y_AA, K_DEC_AA = 50.0, 0.3, 0.08, 0.02
K_M_FA, K_S_FA, Y_FA, K_DEC_FA = 6.0, 0.4, 0.06, 0.02
K_M_C4, K_S_C4, Y_C4, K_DEC_C4 = 20.0, 0.2, 0.06, 0.02
K_M_PRO, K_S_PRO, Y_PRO, K_DEC_PRO = 13.0, 0.1, 0.04, 0.02
K_DIS_PS, K_DIS_PF = 0.04, 0.4
K_HYD_CH, K_HYD_PR, K_HYD_LI = 4.0, 4.0, 4.0
FXI_PS, FXI_PF, FSI = 0.0, 0.0, 0.0
F_CH_BAC, F_PR_BAC, F_LI_BAC, F_P_BAC = 0.196, 0.567, 0.036, 0.20
F_FA_LI = 0.95
F_H2_SU, F_BU_SU, F_PRO_SU, F_AC_SU = 0.19, 0.13, 0.27, 0.41
F_H2_AA, F_VA_AA, F_BU_AA, F_PRO_AA, F_AC_AA = 0.06, 0.23, 0.26, 0.05, 0.40
N_BAC, N_AA, N_I = 0.005353, 0.0076, 0.06 / 14
K_AB = 1.0e8
K_A_VA, K_A_BU, K_A_PRO, K_A_AC = 10**-4.86, 10**-4.82, 10**-4.88, 10**-4.76
K_A_CO2, K_A_IN = 10**-6.35, 10**-9.25
K_W = 1e-14


# ---------------------------------------------------------------------------
# Subsystem A — Gas + methanogenesis (11 states, 3 outputs)
# ---------------------------------------------------------------------------


def _A_f(x, u):
    S_ac, S_h2, S_ch4, S_co2, X_ac, X_h2, S_hco3, p_h2, p_ch4, p_co2, p_total = x
    Q, I_ac, I_h2 = u

    rho_ac = K_M_AC * S_ac / (K_S_AC + S_ac) * X_ac * I_ac
    rho_h2 = K_M_H2 * S_h2 / (K_S_H2 + S_h2) * X_h2 * I_h2
    rho_dec_ac = K_DEC_AC * X_ac
    rho_dec_h2 = K_DEC_H2 * X_h2

    S_co2_free = S_co2 - S_hco3
    rho_T_h2 = K_LA * (S_h2 - 16 * p_h2 / (RT * K_H_H2)) * (V_LIQ / V_GAS)
    rho_T_ch4 = K_LA * (S_ch4 - 64 * p_ch4 / (RT * K_H_CH4)) * (V_LIQ / V_GAS)
    rho_T_co2 = K_LA * (S_co2_free - p_co2 / (RT * K_H_CO2)) * (V_LIQ / V_GAS)
    rho_T_11 = K_P * (p_total + P_GAS_H2O - P_EXT) * (V_LIQ / V_GAS)

    D = Q / V_LIQ
    return [
        -D * S_ac - rho_ac,
        -D * S_h2 - rho_h2 - V_GAS / V_LIQ * rho_T_h2,
        -D * S_ch4
        + (1 - Y_AC) * rho_ac
        + (1 - Y_H2) * rho_h2
        - V_GAS / V_LIQ * rho_T_ch4,
        -D * S_co2 - V_GAS / V_LIQ * rho_T_co2,
        -D * X_ac + Y_AC * rho_ac - rho_dec_ac,
        -D * X_h2 + Y_H2 * rho_h2 - rho_dec_h2,
        -D * S_hco3,
        rho_T_h2 * RT / 16 - p_h2 / p_total * rho_T_11,
        rho_T_ch4 * RT / 64 - p_ch4 / p_total * rho_T_11,
        rho_T_co2 * RT - p_co2 / p_total * rho_T_11,
        RT / 16 * rho_T_h2 + RT / 64 * rho_T_ch4 + RT * rho_T_co2 - rho_T_11,
    ]


def _A_h(x, u):
    _, _, _, _, _, _, _, _, p_ch4, p_co2, p_total = x
    return [p_total, p_ch4, p_co2]


SPEC_A = SubsystemSpec(
    name="A_gas_methanogenesis",
    label="A — Gas + methanogenesis",
    x_names=[
        "S_ac",
        "S_h2",
        "S_ch4",
        "S_co2",
        "X_ac",
        "X_h2",
        "S_hco3",
        "p_h2",
        "p_ch4",
        "p_co2",
        "p_total",
    ],
    u_names=["Q", "I_ac", "I_h2"],
    f_builder=_A_f,
    h_builder=_A_h,
    boundary_note="I_ac, I_h2 as known inputs (from D and E).",
)


# ---------------------------------------------------------------------------
# Subsystem B — Acidogenesis (11 states, 1 output: VFA-sum)
# ---------------------------------------------------------------------------


def _B_f(x, u):
    # NOTE: S_fa, X_fa are analytically decoupled from VFA-sum in ADM1da
    # (rho_fa flows into S_ac and S_h2, both of which are in A, not B). So
    # they form a closed 2-state subsystem that the FOS sensor cannot see.
    # We drop them from B's state vector — they are tracked open-loop via A.
    S_su, S_aa, S_va, S_bu, S_pro, X_su, X_aa, X_c4, X_pro = x
    Q, I_su, I_aa, I_c4, I_pro, Rho_hyd_ch, Rho_hyd_pr, Rho_hyd_li, S_ac_in_B = u

    rho_su = K_M_SU * S_su / (K_S_SU + S_su) * X_su * I_su
    rho_aa = K_M_AA * S_aa / (K_S_AA + S_aa) * X_aa * I_aa
    S_vbu = S_va + S_bu
    rho_c4_va = K_M_C4 * S_va / (K_S_C4 + S_va) * X_c4 * (S_va / S_vbu) * I_c4
    rho_c4_bu = K_M_C4 * S_bu / (K_S_C4 + S_bu) * X_c4 * (S_bu / S_vbu) * I_c4
    rho_pro = K_M_PRO * S_pro / (K_S_PRO + S_pro) * X_pro * I_pro

    rho_dec_su = K_DEC_SU * X_su
    rho_dec_aa = K_DEC_AA * X_aa
    rho_dec_c4 = K_DEC_C4 * X_c4
    rho_dec_pro = K_DEC_PRO * X_pro

    D = Q / V_LIQ
    return [
        -D * S_su
        + (1 - FSI) * Rho_hyd_ch
        + (1 - FSI) * (1 - F_FA_LI) * Rho_hyd_li
        - rho_su,
        -D * S_aa + (1 - FSI) * Rho_hyd_pr - rho_aa,
        -D * S_va + (1 - Y_AA) * F_VA_AA * rho_aa - rho_c4_va,
        -D * S_bu
        + (1 - Y_SU) * F_BU_SU * rho_su
        + (1 - Y_AA) * F_BU_AA * rho_aa
        - rho_c4_bu,
        -D * S_pro
        + (1 - Y_SU) * F_PRO_SU * rho_su
        + (1 - Y_AA) * F_PRO_AA * rho_aa
        + (1 - Y_C4) * 0.54 * rho_c4_va
        - rho_pro,
        -D * X_su + Y_SU * rho_su - rho_dec_su,
        -D * X_aa + Y_AA * rho_aa - rho_dec_aa,
        -D * X_c4 + Y_C4 * (rho_c4_va + rho_c4_bu) - rho_dec_c4,
        -D * X_pro + Y_PRO * rho_pro - rho_dec_pro,
    ]


def _B_h(x, u):
    S_su, S_aa, S_va, S_bu, S_pro, *_ = x
    # VFA-sum (= FOS titration); S_ac taken as boundary input is added by the
    # operator but not differentiated. Sum gives essentially S_va + S_bu/2.x +
    # S_pro/1.x + S_ac scaled. Without S_ac the output is sum of state coords.
    return [60.0 * (S_va / 208.0 + S_bu / 160.0 + S_pro / 112.0)]


SPEC_B = SubsystemSpec(
    name="B_acidogenesis",
    label="B — Acidogenesis",
    x_names=["S_su", "S_aa", "S_va", "S_bu", "S_pro", "X_su", "X_aa", "X_c4", "X_pro"],
    u_names=[
        "Q",
        "I_su",
        "I_aa",
        "I_c4",
        "I_pro",
        "Rho_hyd_ch",
        "Rho_hyd_pr",
        "Rho_hyd_li",
        "S_ac_in_B",
    ],
    f_builder=_B_f,
    h_builder=_B_h,
    boundary_note="Inhibitions I_* from D, hydrolysis fluxes Rho_hyd_* from C. "
    "S_fa, X_fa decoupled (tracked open-loop via A).",
)


# ---------------------------------------------------------------------------
# Subsystem C — Disintegration / hydrolysis (10 states, 3 outputs)
# ---------------------------------------------------------------------------


def _C_f(x, u):
    (
        X_PS_ch,
        X_PS_pr,
        X_PS_li,
        X_PF_ch,
        X_PF_pr,
        X_PF_li,
        X_S_ch,
        X_S_pr,
        X_S_li,
        X_I,
    ) = x
    Q, sum_decay = u

    rho_dis_PS_ch = K_DIS_PS * X_PS_ch
    rho_dis_PS_pr = K_DIS_PS * X_PS_pr
    rho_dis_PS_li = K_DIS_PS * X_PS_li
    rho_dis_PF_ch = K_DIS_PF * X_PF_ch
    rho_dis_PF_pr = K_DIS_PF * X_PF_pr
    rho_dis_PF_li = K_DIS_PF * X_PF_li
    rho_hyd_ch = K_HYD_CH * X_S_ch
    rho_hyd_pr = K_HYD_PR * X_S_pr
    rho_hyd_li = K_HYD_LI * X_S_li

    D = Q / V_LIQ
    return [
        -D * X_PS_ch - rho_dis_PS_ch,
        -D * X_PS_pr - rho_dis_PS_pr,
        -D * X_PS_li - rho_dis_PS_li,
        -D * X_PF_ch - rho_dis_PF_ch,
        -D * X_PF_pr - rho_dis_PF_pr,
        -D * X_PF_li - rho_dis_PF_li,
        -D * X_S_ch
        + (1 - FXI_PS) * rho_dis_PS_ch
        + (1 - FXI_PF) * rho_dis_PF_ch
        + F_CH_BAC * sum_decay
        - rho_hyd_ch,
        -D * X_S_pr
        + (1 - FXI_PS) * rho_dis_PS_pr
        + (1 - FXI_PF) * rho_dis_PF_pr
        + F_PR_BAC * sum_decay
        - rho_hyd_pr,
        -D * X_S_li
        + (1 - FXI_PS) * rho_dis_PS_li
        + (1 - FXI_PF) * rho_dis_PF_li
        + F_LI_BAC * sum_decay
        - rho_hyd_li,
        -D * X_I
        + FXI_PS * (rho_dis_PS_ch + rho_dis_PS_pr + rho_dis_PS_li)
        + FXI_PF * (rho_dis_PF_ch + rho_dis_PF_pr + rho_dis_PF_li)
        + F_P_BAC * sum_decay,
    ]


def _C_h(x, u):
    (
        X_PS_ch,
        X_PS_pr,
        X_PS_li,
        X_PF_ch,
        X_PF_pr,
        X_PF_li,
        X_S_ch,
        X_S_pr,
        X_S_li,
        X_I,
    ) = x
    TS = (
        X_PS_ch
        + X_PS_pr
        + X_PS_li
        + X_PF_ch
        + X_PF_pr
        + X_PF_li
        + X_S_ch
        + X_S_pr
        + X_S_li
        + X_I
    )
    VS = TS - X_I
    # COD-equivalents per gVS differ by category (Henze et al., Batstone 2002):
    # carbohydrates ~1.03, proteins ~1.5, lipids ~2.9 gCOD/gVS.
    # This weighted sum is what distinguishes the ch/pr/li axes from TS/VS,
    # which is what makes the disintegration analysis nontrivial.
    COD_part = (
        1.03 * (X_PS_ch + X_PF_ch + X_S_ch)
        + 1.50 * (X_PS_pr + X_PF_pr + X_S_pr)
        + 2.90 * (X_PS_li + X_PF_li + X_S_li)
    )
    return [TS, VS, COD_part]


SPEC_C = SubsystemSpec(
    name="C_disintegration",
    label="C — Disintegration",
    x_names=[
        "X_PS_ch",
        "X_PS_pr",
        "X_PS_li",
        "X_PF_ch",
        "X_PF_pr",
        "X_PF_li",
        "X_S_ch",
        "X_S_pr",
        "X_S_li",
        "X_I",
    ],
    u_names=["Q", "sum_decay"],
    f_builder=_C_f,
    h_builder=_C_h,
    boundary_note="sum_decay from A+B biomass decay (known input).",
)


# ---------------------------------------------------------------------------
# Subsystem D — Charge balance / pH (8 states, 2 outputs)
# ---------------------------------------------------------------------------


def _D_f(x, u):
    S_cation, S_anion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3, S_nh3 = x
    Q, S_va_B, S_bu_B, S_pro_B, S_ac_A, S_co2_A, S_nh4_E = u

    vfa_anions = S_ac_ion / 64 + S_pro_ion / 112 + S_bu_ion / 160 + S_va_ion / 208
    fixed = S_cation - S_anion + (S_nh4_E - S_nh3) - S_hco3 - vfa_anions
    S_H = (-fixed + sp.sqrt(fixed**2 + 4 * K_W)) / 2

    rho_A_va = K_AB * (S_va_ion * S_H - K_A_VA * (S_va_B - S_va_ion))
    rho_A_bu = K_AB * (S_bu_ion * S_H - K_A_BU * (S_bu_B - S_bu_ion))
    rho_A_pro = K_AB * (S_pro_ion * S_H - K_A_PRO * (S_pro_B - S_pro_ion))
    rho_A_ac = K_AB * (S_ac_ion * S_H - K_A_AC * (S_ac_A - S_ac_ion))
    rho_A_co2 = K_AB * (S_hco3 * S_H - K_A_CO2 * (S_co2_A - S_hco3))
    rho_A_IN = K_AB * (S_nh3 * S_H - K_A_IN * (S_nh4_E - S_nh3))

    D = Q / V_LIQ
    return [
        -D * S_cation,
        -D * S_anion,
        -D * S_va_ion - rho_A_va,
        -D * S_bu_ion - rho_A_bu,
        -D * S_pro_ion - rho_A_pro,
        -D * S_ac_ion - rho_A_ac,
        -D * S_hco3 - rho_A_co2,
        -D * S_nh3 - rho_A_IN,
    ]


def _D_h(x, u):
    S_cation, S_anion, S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion, S_hco3, S_nh3 = x
    _Q, S_va_B, S_bu_B, S_pro_B, S_ac_A, S_co2_A, S_nh4_E = u

    vfa_anions = S_ac_ion / 64 + S_pro_ion / 112 + S_bu_ion / 160 + S_va_ion / 208
    fixed = S_cation - S_anion + (S_nh4_E - S_nh3) - S_hco3 - vfa_anions
    S_H = (-fixed + sp.sqrt(fixed**2 + 4 * K_W)) / 2

    # pH proxy: use S_H itself (monotone-invertible).
    pH_signal = S_H

    # TAC at pH 5
    SH_end = 1.0e-5
    a_nh4 = K_A_IN / (SH_end + K_A_IN)
    a_co2 = K_A_CO2 / (SH_end + K_A_CO2)
    a_ac = K_A_AC / (SH_end + K_A_AC)
    a_pro = K_A_PRO / (SH_end + K_A_PRO)
    a_bu = K_A_BU / (SH_end + K_A_BU)
    a_va = K_A_VA / (SH_end + K_A_VA)
    tac_signal = (
        (S_nh3 - a_nh4 * (S_nh4_E + S_nh3))
        + (S_hco3 - a_co2 * S_co2_A)
        + (S_ac_ion / 64 - a_ac * S_ac_A / 64)
        + (S_pro_ion / 112 - a_pro * S_pro_B / 112)
        + (S_bu_ion / 160 - a_bu * S_bu_B / 160)
        + (S_va_ion / 208 - a_va * S_va_B / 208)
        + S_anion
        - S_cation
    )
    return [pH_signal, tac_signal]


SPEC_D = SubsystemSpec(
    name="D_charge_balance",
    label="D — Charge balance / pH",
    x_names=[
        "S_cation",
        "S_anion",
        "S_va_ion",
        "S_bu_ion",
        "S_pro_ion",
        "S_ac_ion",
        "S_hco3",
        "S_nh3",
    ],
    u_names=["Q", "S_va_B", "S_bu_B", "S_pro_B", "S_ac_A", "S_co2_A", "S_nh4_E"],
    f_builder=_D_f,
    h_builder=_D_h,
    boundary_note="Bulk dissolved species from A, B, E as known inputs.",
)


# ---------------------------------------------------------------------------
# Subsystem E — Nitrogen + soluble inert (2 states, 2 pseudo-outputs)
# ---------------------------------------------------------------------------


def _E_f(x, u):
    S_nh4, S_I = x
    (
        Q,
        Rho_su,
        Rho_aa,
        Rho_fa,
        Rho_c4_va,
        Rho_c4_bu,
        Rho_pro,
        Rho_ac,
        Rho_h2,
        Rho_hyd_ch,
        Rho_hyd_pr,
        Rho_hyd_li,
        sum_decay,
        S_nh3_D,
    ) = u

    D = Q / V_LIQ
    diff_S_nh4 = (
        -D * S_nh4
        - Y_SU * N_BAC * Rho_su
        + (N_AA - Y_AA * N_BAC) * Rho_aa
        - Y_FA * N_BAC * Rho_fa
        - Y_C4 * N_BAC * Rho_c4_va
        - Y_C4 * N_BAC * Rho_c4_bu
        - Y_PRO * N_BAC * Rho_pro
        - Y_AC * N_BAC * Rho_ac
        - Y_H2 * N_BAC * Rho_h2
        + (N_BAC - F_PR_BAC * N_AA - F_P_BAC * N_I) * sum_decay
    )
    diff_S_I = -D * S_I + FSI * (Rho_hyd_ch + Rho_hyd_pr + Rho_hyd_li)
    return [diff_S_nh4, diff_S_I]


def _E_h(x, u):
    # Pseudo-outputs: right-hand sides themselves (open-loop observability).
    return list(_E_f(x, u))


SPEC_E = SubsystemSpec(
    name="E_nitrogen",
    label="E — Nitrogen + S_I",
    x_names=["S_nh4", "S_I"],
    u_names=[
        "Q",
        "Rho_su",
        "Rho_aa",
        "Rho_fa",
        "Rho_c4_va",
        "Rho_c4_bu",
        "Rho_pro",
        "Rho_ac",
        "Rho_h2",
        "Rho_hyd_ch",
        "Rho_hyd_pr",
        "Rho_hyd_li",
        "sum_decay",
        "S_nh3_D",
    ],
    f_builder=_E_f,
    h_builder=_E_h,
    boundary_note="No direct sensor in Phase-1 set; pseudo-output via dynamics.",
)


# ---------------------------------------------------------------------------
# Super-subsystem A+D — Variante II (18 states, 5 outputs, internal pH)
# ---------------------------------------------------------------------------


def _AD_f(x, u):
    (
        S_ac,
        S_h2,
        S_ch4,
        S_co2,
        X_ac,
        X_h2,
        S_cation,
        S_anion,
        S_va_ion,
        S_bu_ion,
        S_pro_ion,
        S_ac_ion,
        S_hco3,
        S_nh3,
        p_h2,
        p_ch4,
        p_co2,
        p_total,
    ) = x
    Q, S_va_B, S_bu_B, S_pro_B, S_nh4_E = u

    # pH algebra (internal)
    vfa_anions = S_ac_ion / 64 + S_pro_ion / 112 + S_bu_ion / 160 + S_va_ion / 208
    fixed = S_cation - S_anion + (S_nh4_E - S_nh3) - S_hco3 - vfa_anions
    S_H = (-fixed + sp.sqrt(fixed**2 + 4 * K_W)) / 2

    # Inhibition factors (Hill-type, simplified to the most important ones)
    # K_pH constants:
    K_pH_ac_n3 = (10 ** (-(6.0 + 7.0) / 2)) ** 3
    K_pH_h2_n3 = (10 ** (-(5.0 + 6.0) / 2)) ** 3
    K_I_NH3, K_S_IN, K_S_CO2_H2, K_IH_AC = 0.0018, 1.0e-4, 5.0e-5, 2.417e-3

    S_IN_total = S_nh4_E + S_nh3
    I_IN = S_IN_total / (K_S_IN + S_IN_total)
    I_pH_ac = K_pH_ac_n3 / (K_pH_ac_n3 + S_H**3)
    I_pH_h2 = K_pH_h2_n3 / (K_pH_h2_n3 + S_H**3)
    I_nh3 = (K_I_NH3**2) / (K_I_NH3**2 + S_nh3**2)
    I_co2_h2 = S_co2**2 / (K_S_CO2_H2**2 + S_co2**2)
    S_HAc = (S_ac / 64) * S_H / (S_H + K_A_AC)
    I_HAc = K_IH_AC / (K_IH_AC + S_HAc)

    I_ac = I_pH_ac * I_IN * I_nh3 * I_HAc
    I_h2 = I_pH_h2 * I_IN * I_co2_h2

    rho_ac = K_M_AC * S_ac / (K_S_AC + S_ac) * X_ac * I_ac
    rho_h2 = K_M_H2 * S_h2 / (K_S_H2 + S_h2) * X_h2 * I_h2
    rho_dec_ac = K_DEC_AC * X_ac
    rho_dec_h2 = K_DEC_H2 * X_h2

    # Acid-base rates (k_AB = 1e8, very fast)
    rho_A_va = K_AB * (S_va_ion * S_H - K_A_VA * (S_va_B - S_va_ion))
    rho_A_bu = K_AB * (S_bu_ion * S_H - K_A_BU * (S_bu_B - S_bu_ion))
    rho_A_pro = K_AB * (S_pro_ion * S_H - K_A_PRO * (S_pro_B - S_pro_ion))
    rho_A_ac = K_AB * (S_ac_ion * S_H - K_A_AC * (S_ac - S_ac_ion))
    rho_A_co2 = K_AB * (S_hco3 * S_H - K_A_CO2 * (S_co2 - S_hco3))
    rho_A_IN = K_AB * (S_nh3 * S_H - K_A_IN * (S_nh4_E - S_nh3))

    S_co2_free = S_co2 - S_hco3
    rho_T_h2 = K_LA * (S_h2 - 16 * p_h2 / (RT * K_H_H2)) * (V_LIQ / V_GAS)
    rho_T_ch4 = K_LA * (S_ch4 - 64 * p_ch4 / (RT * K_H_CH4)) * (V_LIQ / V_GAS)
    rho_T_co2 = K_LA * (S_co2_free - p_co2 / (RT * K_H_CO2)) * (V_LIQ / V_GAS)
    rho_T_11 = K_P * (p_total + P_GAS_H2O - P_EXT) * (V_LIQ / V_GAS)

    D = Q / V_LIQ
    return [
        -D * S_ac - rho_ac,
        -D * S_h2 - rho_h2 - V_GAS / V_LIQ * rho_T_h2,
        -D * S_ch4
        + (1 - Y_AC) * rho_ac
        + (1 - Y_H2) * rho_h2
        - V_GAS / V_LIQ * rho_T_ch4,
        -D * S_co2 - V_GAS / V_LIQ * rho_T_co2 + rho_A_co2,
        -D * X_ac + Y_AC * rho_ac - rho_dec_ac,
        -D * X_h2 + Y_H2 * rho_h2 - rho_dec_h2,
        -D * S_cation,
        -D * S_anion,
        -D * S_va_ion - rho_A_va,
        -D * S_bu_ion - rho_A_bu,
        -D * S_pro_ion - rho_A_pro,
        -D * S_ac_ion - rho_A_ac,
        -D * S_hco3 - rho_A_co2,
        -D * S_nh3 - rho_A_IN,
        rho_T_h2 * RT / 16 - p_h2 / p_total * rho_T_11,
        rho_T_ch4 * RT / 64 - p_ch4 / p_total * rho_T_11,
        rho_T_co2 * RT - p_co2 / p_total * rho_T_11,
        RT / 16 * rho_T_h2 + RT / 64 * rho_T_ch4 + RT * rho_T_co2 - rho_T_11,
    ]


def _AD_h(x, u):
    (
        S_ac,
        _S_h2,
        _S_ch4,
        S_co2,
        _X_ac,
        _X_h2,
        S_cation,
        S_anion,
        S_va_ion,
        S_bu_ion,
        S_pro_ion,
        S_ac_ion,
        S_hco3,
        S_nh3,
        _p_h2,
        p_ch4,
        p_co2,
        p_total,
    ) = x
    _Q, S_va_B, S_bu_B, S_pro_B, S_nh4_E = u

    vfa_anions = S_ac_ion / 64 + S_pro_ion / 112 + S_bu_ion / 160 + S_va_ion / 208
    fixed = S_cation - S_anion + (S_nh4_E - S_nh3) - S_hco3 - vfa_anions
    S_H = (-fixed + sp.sqrt(fixed**2 + 4 * K_W)) / 2

    SH_end = 1.0e-5
    a_nh4 = K_A_IN / (SH_end + K_A_IN)
    a_co2 = K_A_CO2 / (SH_end + K_A_CO2)
    a_ac = K_A_AC / (SH_end + K_A_AC)
    a_pro = K_A_PRO / (SH_end + K_A_PRO)
    a_bu = K_A_BU / (SH_end + K_A_BU)
    a_va = K_A_VA / (SH_end + K_A_VA)
    tac_signal = (
        (S_nh3 - a_nh4 * (S_nh4_E + S_nh3))
        + (S_hco3 - a_co2 * S_co2)
        + (S_ac_ion / 64 - a_ac * S_ac / 64)
        + (S_pro_ion / 112 - a_pro * S_pro_B / 112)
        + (S_bu_ion / 160 - a_bu * S_bu_B / 160)
        + (S_va_ion / 208 - a_va * S_va_B / 208)
        + S_anion
        - S_cation
    )
    return [p_total, p_ch4, p_co2, S_H, tac_signal]


SPEC_AD = SubsystemSpec(
    name="AD_combined",
    label="A+D — Gas + meth + charge (fused, Variante II)",
    x_names=[
        "S_ac",
        "S_h2",
        "S_ch4",
        "S_co2",
        "X_ac",
        "X_h2",
        "S_cation",
        "S_anion",
        "S_va_ion",
        "S_bu_ion",
        "S_pro_ion",
        "S_ac_ion",
        "S_hco3",
        "S_nh3",
        "p_h2",
        "p_ch4",
        "p_co2",
        "p_total",
    ],
    u_names=["Q", "S_va_B", "S_bu_B", "S_pro_B", "S_nh4_E"],
    f_builder=_AD_f,
    h_builder=_AD_h,
    boundary_note="pH-algebra and inhibitions internal; boundaries only to B and E.",
)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


SPECS_DEFAULT = [SPEC_A, SPEC_B, SPEC_C, SPEC_D, SPEC_E, SPEC_AD]
TIME_BUDGETS_DEFAULT = {
    "A_gas_methanogenesis": 600,
    "B_acidogenesis": 600,
    "C_disintegration": 300,
    "D_charge_balance": 300,
    "E_nitrogen": 60,
    "AD_combined": 3600,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", nargs="*", help="Only run subsystems with these names"
    )
    parser.add_argument(
        "--skip-ad", action="store_true", help="Skip the expensive AD_combined run"
    )
    parser.add_argument("--out", default="results.md", help="Markdown output file")
    args = parser.parse_args()

    specs = SPECS_DEFAULT
    if args.only:
        specs = [s for s in specs if s.name in args.only]
    if args.skip_ad:
        specs = [s for s in specs if s.name != "AD_combined"]

    print("Plan B -- Python/sympy subsystem checker for ADM1da", flush=True)
    print("=" * 60, flush=True)
    results: List[ProfileResult] = []
    total_t0 = time.perf_counter()

    for spec in specs:
        budget = TIME_BUDGETS_DEFAULT.get(spec.name, 600)
        print(f"\n>>> {spec.label}", flush=True)
        print(
            f"    n_states = {len(spec.x_names)}, "
            f"n_outputs = {len(spec.h_builder(*_dummy_args(spec)))}, "
            f"budget = {budget}s",
            flush=True,
        )
        r = check_observability(spec, time_budget_s=budget)
        results.append(r)
        status = "OBSERVABLE" if r.fully_observable else "rank deficit"
        print(
            f"    => {status}: rank {r.rank}/{r.n_states}, "
            f"wall {r.wall_s:.1f}s, dRAM {r.peak_mem_mib:.1f} MiB",
            flush=True,
        )
        if r.note:
            print(f"    note: {r.note}", flush=True)

    total_wall = time.perf_counter() - total_t0
    print("\n" + "=" * 60, flush=True)
    print(f"TOTAL: {total_wall:.1f}s wall", flush=True)

    _write_report(results, total_wall, args.out)


def _dummy_args(spec: SubsystemSpec):
    """Return dummy symbols just to count outputs in main()'s preview line."""
    x = [sp.Symbol(n, positive=True) for n in spec.x_names]
    u = [sp.Symbol(n) for n in spec.u_names]
    return x, u


def _write_report(results: List[ProfileResult], total_wall: float, path: str):
    p = Path(path)
    lines = [
        "# ADM1da subsystem observability",
        "",
        "Run via `python subsystem_checker.py`. Method: symbolic Lie",
        "derivatives + numerical rank check at random sample point",
        "(Sedoglavic 2002).",
        "",
        "## Results",
        "",
        "| Subsystem | n | n_out | iters | rank | observable? | wall [s] | dRAM [MiB] |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.label} | {r.n_states} | {r.n_outputs} | "
            f"{r.n_iters_done} | {r.rank} | "
            f"{'**yes**' if r.fully_observable else 'no'} | "
            f"{r.wall_s:.1f} | {r.peak_mem_mib:.1f} |"
        )
    lines.append("")
    lines.append(f"**Total wall:** {total_wall:.1f}s")
    lines.append("")
    lines.append("## Notes per subsystem")
    lines.append("")
    for r in results:
        if r.note:
            lines.append(f"* **{r.label}**: {r.note}")
    p.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {p.resolve()}", flush=True)


if __name__ == "__main__":
    main()
