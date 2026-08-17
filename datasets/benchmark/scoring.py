"""Scoring for the ADM1 state-estimation benchmark (numpy only).

The relevant skill is tracking the state through **substrate changes**, so the
headline metric is the **transient NRMSE**: per-state error restricted to the
windows right after each feed switch. Steady-state and overall NRMSE plus the
2-sigma coverage are reported alongside, and every score is shown next to the
reference UKF so you can gauge your model without running a filter yourself.

NRMSE (= RMSE / RMS(true)) is normalised by a state's magnitude, so it flatters
states with a large offset and explodes for near-zero states. Two additions avoid
that: the ``vs_ukf`` block reports the median per-state **RMSE ratio** model/UKF
(``<1`` beats the UKF; pairing both errors on the same state cancels its scale and
offset entirely — the fairest single number), and :func:`per_state_report` gives the
per-state **RMSE in physical units** next to the state's own mean/range for context.

    from loader import load_test
    from scoring import score_series, score_dataset, per_state_report

    test = load_test()
    my_pred = my_model(test[0])                 # (T, 41) state estimate
    print(score_series(my_pred, test[0]))       # NRMSE + coverage + vs_ukf ratio
    rep = per_state_report(my_pred, test[0])    # per-state RMSE + context (which states)
"""

from __future__ import annotations

import numpy as np

TAU_HOURS = 48.0  # transient window length after each switch
DT_HOURS = 1.0


def _rmse_per_state(pred: np.ndarray, true: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Plain RMSE per state (physical units) over the steps selected by ``mask``."""
    if mask.sum() == 0:  # no steps selected (e.g. a series with no switches)
        return np.full(true.shape[1], np.nan)
    p, t = pred[mask], true[mask]
    return np.sqrt(np.mean((p - t) ** 2, axis=0))


def _nrmse_per_state(
    pred: np.ndarray, true: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """RMSE / RMS(true) per state over the time steps selected by ``mask``."""
    if mask.sum() == 0:  # no steps selected (e.g. a series with no switches)
        return np.full(true.shape[1], np.nan)
    denom = np.sqrt(np.mean(true[mask] ** 2, axis=0)) + 1e-12
    return _rmse_per_state(pred, true, mask) / denom


def _transient_mask(
    time: np.ndarray, switch_days: np.ndarray, tau_hours: float
) -> np.ndarray:
    """True for steps within ``tau_hours`` after any switch."""
    tau = tau_hours / 24.0
    m = np.zeros(len(time), dtype=bool)
    for sd in np.atleast_1d(switch_days):
        m |= (time >= sd) & (time < sd + tau)
    return m


def per_state_nrmse(
    pred: np.ndarray,
    series: dict[str, np.ndarray],
    window: str = "overall",
    tau_hours: float = TAU_HOURS,
) -> np.ndarray:
    """Per-state NRMSE as a ``(41,)`` array, to see *which* states are weak.

    ``window`` is ``"overall"``, ``"transient"`` (only after feed switches) or
    ``"steady"``. Note that a few states are numerically ~zero, so their relative
    error is not meaningful (see :func:`score_series`).
    """
    true, time = series["states"], series["time"]
    tr = _transient_mask(time, series["switch_days"], tau_hours)
    mask = {"overall": np.ones(len(time), dtype=bool), "transient": tr, "steady": ~tr}[
        window
    ]
    return _nrmse_per_state(pred, true, mask)


def per_state_rmse(
    pred: np.ndarray,
    series: dict[str, np.ndarray],
    window: str = "overall",
    tau_hours: float = TAU_HOURS,
) -> np.ndarray:
    """Per-state **RMSE in physical units** as a ``(41,)`` array.

    Unlike :func:`per_state_nrmse` this is not normalised, so it is not distorted by
    a state's magnitude/offset — read it next to the state's own mean/range (see
    :func:`per_state_report`) to judge whether an error is large. ``window`` is
    ``"overall"``, ``"transient"`` or ``"steady"``.
    """
    true, time = series["states"], series["time"]
    tr = _transient_mask(time, series["switch_days"], tau_hours)
    mask = {"overall": np.ones(len(time), dtype=bool), "transient": tr, "steady": ~tr}[
        window
    ]
    return _rmse_per_state(pred, true, mask)


def per_state_report(
    pred: np.ndarray,
    series: dict[str, np.ndarray],
    window: str = "overall",
    tau_hours: float = TAU_HOURS,
) -> dict[str, np.ndarray]:
    """Per-state error **with context**, as a dict of ``(41,)`` arrays — the honest
    way to see *which* states are missed and *how large* the error is physically.

    Keys: ``rmse`` (model, physical units), ``ukf_rmse`` (reference, ``nan`` if the
    UKF is pending), ``rmse_ratio`` (``rmse / ukf_rmse``; ``<1`` beats the UKF, and
    since both errors are on the same state this cancels the state's scale/offset
    entirely), ``true_mean``, ``true_range`` and ``true_rms`` (the state's own
    magnitude/variation for context), and ``nrmse`` (the RMS-normalised value, for
    continuity). Pair ``rmse`` with ``true_range`` to gauge an error, and use
    ``rmse_ratio`` to rank yourself against the UKF free of normalisation artefacts.
    """
    true, time = series["states"], series["time"]
    tr = _transient_mask(time, series["switch_days"], tau_hours)
    mask = {"overall": np.ones(len(time), dtype=bool), "transient": tr, "steady": ~tr}[
        window
    ]
    t = true[mask]
    rmse = _rmse_per_state(pred, true, mask)
    ukf = series.get("ukf_x_hat")
    if ukf is None or np.all(np.isnan(np.asarray(ukf, dtype=float))):
        ukf_rmse = np.full(true.shape[1], np.nan)
    else:
        ukf_rmse = _rmse_per_state(np.asarray(ukf, dtype=float), true, mask)
    return {
        "rmse": rmse,
        "ukf_rmse": ukf_rmse,
        "rmse_ratio": rmse / (ukf_rmse + 1e-12),
        "true_mean": t.mean(axis=0),
        "true_range": t.max(axis=0) - t.min(axis=0),
        "true_rms": np.sqrt(np.mean(t**2, axis=0)),
        "nrmse": _nrmse_per_state(pred, true, mask),
    }


def coverage_2sigma(pred: np.ndarray, true: np.ndarray, std: np.ndarray) -> float:
    """Fraction of state-time cells with ``|true - pred| <= 2 std`` (needs a std)."""
    if std is None:
        return float("nan")
    return float(np.mean(np.abs(true - pred) <= 2.0 * std))


def score_series(
    pred: np.ndarray,
    series: dict[str, np.ndarray],
    std: np.ndarray | None = None,
    tau_hours: float = TAU_HOURS,
) -> dict[str, float]:
    """Score one prediction ``(T, 41)`` against a test series (incl. the UKF).

    Returns transient / steady / overall mean NRMSE [%], and — if ``std`` is given —
    the 2-sigma coverage; each paired with the reference UKF's value. Also returns a
    ``vs_ukf`` block: the median per-state **RMSE ratio** ``model / UKF`` per window
    (``<1`` beats the UKF). The ratio is the fairest headline — pairing the two errors
    on the same state cancels that state's magnitude/offset/variation, so it is free of
    the normalisation artefacts NRMSE suffers from (see :func:`per_state_report`).
    """
    true = series["states"]
    time = series["time"]
    tr = _transient_mask(time, series["switch_days"], tau_hours)
    st = ~tr
    allm = np.ones(len(time), dtype=bool)

    def block(p, s=None):
        # Median over the 41 states, not the mean: a few states are numerically
        # ~zero (S_cation, S_h2, p_gas_h2 have an RMS around 1e-6), so their
        # *relative* error explodes to five-digit percentages while the absolute
        # error stays irrelevant. The mean is dominated by exactly those and would
        # reward optimising noise. It is reported alongside for transparency.
        out = {
            "nrmse_transient_%": 100 * float(np.median(_nrmse_per_state(p, true, tr))),
            "nrmse_steady_%": 100 * float(np.median(_nrmse_per_state(p, true, st))),
            "nrmse_overall_%": 100 * float(np.median(_nrmse_per_state(p, true, allm))),
            "nrmse_overall_mean_%": 100
            * float(np.mean(_nrmse_per_state(p, true, allm))),
        }
        if s is not None:
            out["coverage_2sigma_%"] = 100 * coverage_2sigma(p, true, s)
        return out

    res = {"model": block(pred, std)}
    ukf = series.get("ukf_x_hat")
    if ukf is None or np.all(np.isnan(np.asarray(ukf, dtype=float))):
        # UKF reference not computed yet (see loader: ``ukf_pending``).
        res["ukf_reference"] = {k: float("nan") for k in res["model"]}
        res["vs_ukf"] = {
            k: float("nan")
            for k in ("rmse_ratio_transient", "rmse_ratio_steady", "rmse_ratio_overall")
        }
    else:
        ukf = np.asarray(ukf, dtype=float)
        res["ukf_reference"] = block(ukf, series.get("ukf_std"))

        def ratio(mask):  # median over states of RMSE_model / RMSE_ukf
            return float(
                np.median(
                    _rmse_per_state(pred, true, mask)
                    / (_rmse_per_state(ukf, true, mask) + 1e-12)
                )
            )

        res["vs_ukf"] = {
            "rmse_ratio_transient": ratio(tr),
            "rmse_ratio_steady": ratio(st),
            "rmse_ratio_overall": ratio(allm),
        }
    return res


def score_dataset(
    predictions: list[np.ndarray],
    test: list[dict[str, np.ndarray]],
    stds: list[np.ndarray] | None = None,
) -> dict[str, float]:
    """Average the scoring metrics over all test series.

    ``predictions[i]`` must be the ``(T, 41)`` estimate for ``test[i]``.
    """
    rows_model, rows_ukf, rows_ratio = [], [], []
    for i, series in enumerate(test):
        std = stds[i] if stds is not None else None
        r = score_series(predictions[i], series, std=std)
        rows_model.append(r["model"])
        rows_ukf.append(r["ukf_reference"])
        rows_ratio.append(r["vs_ukf"])

    def _avg(rows):
        keys = rows[0].keys()
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}

    return {
        "model_mean": _avg(rows_model),
        "ukf_reference_mean": _avg(rows_ukf),
        "vs_ukf_mean": _avg(
            rows_ratio
        ),  # median-per-state RMSE ratio, mean over series
        "n_series": len(rows_model),
    }
