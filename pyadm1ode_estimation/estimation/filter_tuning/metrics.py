"""Shared metrics for estimator training / calibration.

Self-contained numpy — no dependency on the dataset directory. Covers the two goals we
optimise for: honest **uncertainty** (2σ coverage, NEES) and the **FOS/TAC critical-state
decision** (VFA/TAC band coverage + a critical alarm). FOS/TAC = VFA/TAC is a fixed
function of the ADM1 state (a numpy port of pyadm1's vfa/tac, with a physical TAC floor
so an unphysical near-zero denominator cannot blow the ratio up).
"""

from __future__ import annotations

import numpy as np

# --- FOS/TAC constants (fixed physical/acid constants; not the tuned kinetics) ---
_M_HAC = 60.0
_COD = {"ac": 64.0, "pro": 112.0, "bu": 160.0, "va": 208.0}
_KA = {
    "IN": 1.7422628075231628e-09,
    "co2": 5.275470941922822e-07,
    "ac": 1.7378008287493764e-05,
    "pro": 1.3182567385564074e-05,
    "bu": 1.5135612484362071e-05,
    "va": 1.3803842646028839e-05,
}
_I = {
    "S_ac": 6,
    "S_pro": 5,
    "S_bu": 4,
    "S_va": 3,
    "S_nh4": 10,
    "S_nh3": 36,
    "S_co2": 9,
    "S_hco3": 35,
    "S_ac_ion": 34,
    "S_pro_ion": 33,
    "S_bu_ion": 32,
    "S_va_ion": 31,
    "S_anion": 30,
    "S_cation": 29,
}
FT_IDX = sorted(_I.values())
TAC_FLOOR = 1.0  # kg CaCO3/m3 — see project_ukf_fostac_diagnosis
CRIT = 0.6  # Nordmann critical FOS/TAC
TARGET_COV = 0.9545  # 2σ coverage of a Gaussian


def _vfa_tac(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """VFA and TAC in kg/m3 from a (..., 41) ADM1 state array.

    Verified bit-identical to the dataset's own ``fostac_true`` (generated with
    pyadm1's ``vfa_torch`` / ``tac_torch``) over all 28820 test cells, which is what
    makes it usable as the filter's observation model for a titration measurement.
    """
    x = np.asarray(states, float)
    g = lambda n: x[..., _I[n]]
    vfa = _M_HAC * (
        g("S_ac") / _COD["ac"]
        + g("S_pro") / _COD["pro"]
        + g("S_bu") / _COD["bu"]
        + g("S_va") / _COD["va"]
    )
    h = 1.0e-5
    a = {k: _KA[k] / (h + _KA[k]) for k in _KA}
    tac_mol = (
        (g("S_nh3") - a["IN"] * (g("S_nh4") + g("S_nh3")))
        + (g("S_hco3") - a["co2"] * g("S_co2"))
        + (g("S_ac_ion") / _COD["ac"] - a["ac"] * g("S_ac") / _COD["ac"])
        + (g("S_pro_ion") / _COD["pro"] - a["pro"] * g("S_pro") / _COD["pro"])
        + (g("S_bu_ion") / _COD["bu"] - a["bu"] * g("S_bu") / _COD["bu"])
        + (g("S_va_ion") / _COD["va"] - a["va"] * g("S_va") / _COD["va"])
        + g("S_anion")
        - g("S_cation")
    )
    return vfa, 50.0 * tac_mol


def fos_tac_mg_l(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """FOS and TAC in **mg/L**, the units the dataset stores its titrations in.

    This is the forward model ``h(x)`` of a Nordmann titration: FOS as mg acetic-acid
    equivalent per litre, TAC as mg CaCO3 per litre. Unlike :func:`fostac` it does not
    floor the denominator, because a measurement channel must stay linearisable.
    """
    vfa, tac = _vfa_tac(states)
    return vfa * 1000.0, tac * 1000.0


def fostac(states: np.ndarray) -> np.ndarray:
    """FOS/TAC (VFA/TAC) from a (..., 41) ADM1 state array; TAC floored at TAC_FLOOR."""
    vfa, tac = _vfa_tac(states)
    return vfa / np.clip(tac, TAC_FLOOR, None)


def fostac_sigma(x_hat: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Linearised FOS/TAC std from the diagonal per-state std (NaN std treated as 0)."""
    sd = np.nan_to_num(np.asarray(std, float), nan=0.0)
    f0 = fostac(x_hat)
    var = np.zeros(len(x_hat))
    for j in FT_IDX:
        d = np.maximum(sd[:, j], 1e-12)
        xp = x_hat.copy()
        xp[:, j] = x_hat[:, j] + d
        var += (fostac(xp) - f0) ** 2
    return np.sqrt(var)


def nees_per_state(truth, x_hat, std, mask=None):
    """NEES per state = mean((truth-x̂)²/σ²); NaN std (unestimated) -> NaN NEES."""
    sd = np.asarray(std, float)
    z2 = ((truth - x_hat) / (sd + 1e-12)) ** 2
    with np.errstate(invalid="ignore"):
        out = np.nanmean(np.where(np.isfinite(sd), z2, np.nan), axis=0)
    return out


def coverage(truth, x_hat, std, state_mask=None):
    """Fraction of state-time cells within ±2σ (over ``state_mask`` states if given)."""
    sd = np.asarray(std, float)
    cov = np.abs(truth - x_hat) <= 2.0 * sd
    cols = np.arange(truth.shape[1]) if state_mask is None else np.asarray(state_mask)
    finite = np.isfinite(sd[:, cols])
    return float(np.mean(cov[:, cols][finite]))


def fostac_band_coverage(truth, x_hat, std):
    ft_t, ft_h = fostac(truth), fostac(x_hat)
    ft_s = fostac_sigma(x_hat, std)
    return float(np.mean(np.abs(ft_t - ft_h) <= 2.0 * ft_s))


def critical_decision(truth, x_hat, std, crit=CRIT):
    """Alarm = 'critical plausible' (upper 2σ FOS/TAC band > crit) vs true FOS/TAC>crit.
    Returns (balanced_accuracy, tpr, tnr).

    NOTE: this is a *fixed-threshold* rule and therefore gameable — inflating σ pushes the
    upper band above ``crit`` everywhere, giving TPR 1 / TNR 0 (bal-acc 0.5) while looking
    perfectly "covered". Use :func:`critical_auc` for selection; keep this for reporting.
    """
    ft_t, ft_h = fostac(truth), fostac(x_hat)
    ft_s = fostac_sigma(x_hat, std)
    pred = (ft_h + 2.0 * ft_s) > crit
    tc = ft_t > crit
    tpr = float(pred[tc].mean()) if tc.any() else np.nan
    tnr = float((~pred)[~tc].mean()) if (~tc).any() else np.nan
    return float(np.nanmean([tpr, tnr])), tpr, tnr


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    """Rank-based ROC-AUC (Mann-Whitney U), tie-safe. 0.5 = no discrimination."""
    from scipy.stats import rankdata

    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def critical_auc(truth, x_hat, std, crit=CRIT):
    """Threshold-FREE quality of the critical-state decision (ROC-AUC).

    Scores each cell by the filter's own belief ``P(FOS/TAC > crit)`` under its Gaussian
    (point estimate + propagated σ) and measures how well that *ranks* truly-critical
    cells above healthy ones.

    Key property: being rank-based, it is **invariant to a uniform σ inflation** (scaling
    every σ by the same factor is a monotone transform of the score, so the ranking — and
    hence the AUC — does not move; verified: σ×1000 leaves the AUC unchanged). Inflating σ
    therefore **buys nothing** on the decision term, whereas the fixed-threshold bal-acc
    could be pushed to TPR 1 / TNR 0 that way. What AUC measures is the genuine
    discriminative power of the point estimate; over-coverage is handled separately by the
    penalty in :func:`objective`. This is the metric to select on.
    """
    from scipy.stats import norm

    ft_t, ft_h = fostac(truth), fostac(x_hat)
    ft_s = np.maximum(fostac_sigma(x_hat, std), 1e-12)
    p_crit = norm.sf((crit - ft_h) / ft_s)  # P(FOS/TAC > crit)
    return _auc(p_crit, ft_t > crit)


def nrmse(truth, x_hat, state_mask=None) -> float:
    """Median over states of RMSE / RMS(truth) — the general accuracy metric.

    Median (not mean) because a few states are numerically ~zero, so their *relative*
    error explodes while the absolute error is irrelevant.
    """
    cols = np.arange(truth.shape[1]) if state_mask is None else np.asarray(state_mask)
    t, x = truth[:, cols], x_hat[:, cols]
    rmse = np.sqrt(np.mean((x - t) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(t**2, axis=0)) + 1e-12)))


def critical_decision_prob(truth, x_hat, std, crit=CRIT, p_thresh=0.5):
    """Critical alarm from the filter's own probability ``P(FOS/TAC > crit) > p_thresh``.

    The principled alternative to :func:`critical_decision`'s ad-hoc "upper 2σ band exceeds
    crit" rule, which is equivalent to a ~2.3 % probability threshold and therefore alarms
    almost always. ``p_thresh=0.5`` is the Bayes rule for equal error costs; lower it to
    trade false alarms for sensitivity. Measured on the shipped reference this lifts
    bal-acc 0.68 → 0.79 and, per series, the healthy-cleared rate from 3/15 to 15/15 —
    without touching the filter. Returns ``(balanced_accuracy, tpr, tnr)``.
    """
    from scipy.stats import norm

    ft_t, ft_h = fostac(truth), fostac(x_hat)
    ft_s = np.maximum(fostac_sigma(x_hat, std), 1e-12)
    pred = norm.sf((crit - ft_h) / ft_s) > p_thresh
    tc = ft_t > crit
    tpr = float(pred[tc].mean()) if tc.any() else np.nan
    tnr = float((~pred)[~tc].mean()) if (~tc).any() else np.nan
    return float(np.nanmean([tpr, tnr])), tpr, tnr


def evaluate(truth, x_hat, std, state_mask=None):
    """All headline metrics for one (stacked) set of cells."""
    bal, tpr, tnr = critical_decision(truth, x_hat, std)
    balp, tprp, tnrp = critical_decision_prob(truth, x_hat, std)
    return {
        "nrmse": nrmse(truth, x_hat, state_mask),  # accuracy (point estimate)
        "coverage": coverage(truth, x_hat, std, state_mask),
        "fostac_band_coverage": fostac_band_coverage(truth, x_hat, std),
        "critical_auc": critical_auc(truth, x_hat, std),  # threshold-free (selection)
        "critical_balacc": bal,
        "critical_tpr": tpr,
        "critical_tnr": tnr,
        # Bayes rule P(FOS/TAC>crit)>0.5 — the usable alarm; the band rule above is kept
        # for continuity with earlier reports.
        "critical_balacc_p50": balp,
        "critical_tpr_p50": tprp,
        "critical_tnr_p50": tnrp,
    }


def objective(m: dict, w_cov=0.5, w_ftcov=0.5, over_penalty=2.0) -> float:
    """Decision-focused selection score (higher is better).

    Driven by the **threshold-free** :func:`critical_auc` (falls back to bal-acc if the
    AUC is undefined, e.g. one class absent), with coverage as a *soft* constraint toward
    the 2σ target. Over-coverage is penalised extra (``over_penalty``).

    Why AUC and not bal-acc: the fixed-threshold rule is gameable — inflating σ makes the
    upper band exceed the critical level everywhere (TPR 1 / TNR 0, coverage → 1.0) and
    used to win the search. AUC is rank-based and invariant to uniform σ scaling, so that
    trick gains nothing; the decision term then reflects the point estimate's real
    discriminative power, and the over-coverage penalty handles the inflation itself.
    """

    def pen(x, w):
        d = x - TARGET_COV
        return w * (over_penalty if d > 0 else 1.0) * abs(d)

    decision = m.get("critical_auc")
    if decision is None or not np.isfinite(decision):
        decision = m["critical_balacc"]
    return (
        decision - pen(m["coverage"], w_cov) - pen(m["fostac_band_coverage"], w_ftcov)
    )


def objective_accuracy_first(
    m: dict,
    nrmse_baseline: float,
    w_guard: float = 2.0,
    w_cov: float = 0.5,
    over_penalty: float = 2.0,
) -> float:
    """Accuracy-only score with a one-sided guard against an *inverted* FOS/TAC call.

    Differs from :func:`objective_combined` in three ways:

    * no reward for AUC above chance, so the decision cannot pull the search away from the
      most accurate candidates (measured: accuracy and decision correlate -0.50 for the
      full UKF and -0.32 for the A+D core),
    * no separate FOS/TAC band-coverage penalty, which is largely redundant with the AUC
      (they correlate +0.43 / +0.69 over the candidates),
    * a penalty that only bites when the ranking is *worse than a coin flip*.

    The guard is not optional: a pure accuracy objective walks straight into the inverted
    region, where the filter says the opposite of the truth. On the reference search trace
    **76 of 96 candidates had AUC < 0.5**, and the winner under this objective is inverted
    for ``w_guard`` up to 1.0 (AUC 0.000 at 0, 0.323 at 0.5, 0.473 at 1.0). ``w_guard=2.0``
    is the calibrated smallest value that reliably selects a non-inverted candidate.
    """
    acc = 0.0
    nr = m.get("nrmse")
    if nr is not None and np.isfinite(nr) and nrmse_baseline:
        acc = (nrmse_baseline - nr) / nrmse_baseline
    d = m["coverage"] - TARGET_COV
    cov_pen = w_cov * (over_penalty if d > 0 else 1.0) * abs(d)
    auc = m.get("critical_auc")
    guard = 0.0 if auc is None or not np.isfinite(auc) else max(0.0, 0.5 - auc)
    return acc - cov_pen - w_guard * guard


def objective_combined(
    m: dict,
    nrmse_baseline: float,
    w_acc=0.5,
    w_dec=0.5,
    w_cov=0.5,
    w_ftcov=0.5,
    over_penalty=2.0,
) -> float:
    """Selection score covering **both** goals: accuracy and the critical decision.

    Both terms are made dimensionless and centred so the weights are meaningful:

        acc_gain = (NRMSE_baseline − NRMSE) / NRMSE_baseline   # >0 : better than baseline
        dec_gain = (AUC − 0.5) / 0.5                           #  0 : chance, 1 : perfect

    Coverage stays a soft constraint toward the 2σ target, with over-coverage penalised
    extra. Log every component alongside so the accuracy↔decision Pareto trade-off can be
    reconstructed from the search trace without extra runs.
    """

    def pen(x, w):
        d = x - TARGET_COV
        return w * (over_penalty if d > 0 else 1.0) * abs(d)

    dec = m.get("critical_auc")
    if dec is None or not np.isfinite(dec):
        dec = m.get("critical_balacc", 0.5)
    dec_gain = (dec - 0.5) / 0.5
    nr = m.get("nrmse")
    acc_gain = (
        0.0
        if (nr is None or not np.isfinite(nr) or not nrmse_baseline)
        else (nrmse_baseline - nr) / nrmse_baseline
    )
    return (
        w_acc * acc_gain
        + w_dec * dec_gain
        - pen(m["coverage"], w_cov)
        - pen(m["fostac_band_coverage"], w_ftcov)
    )
