"""Stage 0 — Q and R estimated **directly from ground truth**, without any search.

The benchmark gives us the true state trajectory, which most noise-covariance literature
does not assume. With it, Q and R are simply what they are defined to be:

    Q_i = Var_k( x_true(k+1) − f(x_true(k)) )_i     one-step model error per state
    R_j = Var_k( z_measured(k) − h(x_true(k)) )_j   measurement error per sensor

Cost: one plant propagation per time step (≈ the cost of a *single* UKF episode for the
whole training set, since the UKF pays 2n+1 = 83 propagations per step). That makes this
~100x cheaper than one pass of a black-box search over Q, and it cannot overfit — there
is no selection step.

Caveat: this is the *true* model-error covariance. It is optimal for the filter only if
the filter were exact; a UKF must additionally absorb linearisation error and the fact
that it propagates x̂ ≠ x_true. So use it as a **physically grounded starting point**
(and as the per-state *shape* of Q), then search only a few correction factors around it
— rather than sampling an 11-dimensional box blindly.
"""

from __future__ import annotations

import numpy as np
from pyadm1.core.adm1 import STATE_SIZE

from ..quickstart import build_filter_components
from .filter_runners import build_plant, substrates_from_meta

__all__ = ["empirical_noise", "q_scales_vs_nominal"]


def _series_residuals(args):
    """Worker: h-step model residuals + observation residuals for ONE series."""
    meta, digester_id, truth, feed, meas, stride, obs_stride, horizon = args
    horizon = max(int(horizon), 1)
    substrates = substrates_from_meta(meta)
    sensors = [s.lower() for s in meta["sensors"]]
    channel_names = list(meta["sensors"])
    dt = float(meta.get("dt_hours", 1.0)) / 24.0

    plant = build_plant(meta, digester_id)
    process, obs, spec = build_filter_components(
        plant, digester_id=digester_id, substrates=substrates, sensors=sensors
    )
    adm1_pos = spec.kind_indices("adm1")
    adm1_idx = [spec.channels[p].adm1_index for p in adm1_pos]
    aug = spec.kind_indices("input_flow")

    x = np.zeros(len(spec))
    n = len(truth) - horizon
    q_sum = np.zeros(STATE_SIZE)
    q_cnt = 0
    r_sum = np.zeros(len(channel_names))
    r_cnt = 0

    for k in range(0, n, stride):
        # ---- Q: propagate the TRUE state `horizon` steps under the known feed ----
        for p, a in zip(adm1_pos, adm1_idx):
            x[p] = truth[k, a]
        try:
            xn = x.copy()
            for h in range(horizon):
                for j, i_aug in enumerate(aug):
                    xn[i_aug] = feed[k + h, j]
                xn = process.step(xn, dt)
        except Exception:  # noqa: BLE001, S112
            # The stiff ADM1 refuses to integrate from some true states. Skipping the
            # sample is correct here: this is a *statistic* over many steps, and a step
            # that cannot be propagated contributes no residual. Counted via q_cnt.
            continue
        resid = np.array(
            [truth[k + horizon, a] - xn[p] for p, a in zip(adm1_pos, adm1_idx)]
        )
        if np.all(np.isfinite(resid)):
            q_sum += resid**2
            q_cnt += 1

        # ---- R: predicted sensors from the TRUE state vs the measurement ----
        if obs_stride and k % obs_stride == 0:
            try:
                process.refresh_outputs(x, equilibration_dt=dt)
                pred = {
                    c.name: float(c.extractor(process.plant, x)) for c in obs.channels
                }
                rr = np.array(
                    [
                        meas[k, i] - pred.get(nm, np.nan)
                        for i, nm in enumerate(channel_names)
                    ]
                )
                if np.all(np.isfinite(rr)):
                    r_sum += rr**2
                    r_cnt += 1
            except Exception:  # noqa: BLE001, S110
                # Same reasoning as above, for the measurement residual. r_cnt records
                # how many samples actually contributed.
                pass
    return q_sum, q_cnt, r_sum, r_cnt


def empirical_noise(
    dataset,
    series=None,
    digester_id: str = "primary",
    jobs: int = 1,
    stride: int = 1,
    obs_stride: int = 4,
    max_series: int | None = None,
    horizon: int = 1,
):
    """Estimate Q (per ADM1 state) and R (per sensor) from ground truth.

    Args:
        dataset: an :class:`~.datasets.EstimatorDataset`.
        series: which :class:`~.datasets.Series` to use (default: the whole train pool).
        stride: use every ``stride``-th time step for Q (1 = all).
        obs_stride: use every ``obs_stride``-th step for R (R needs far fewer samples,
            and ``refresh_outputs`` is the expensive part).
        jobs: process pool size (parallel over series).

    Returns:
        ``{"q_diag": (41,), "r_diag": (n_sensor,), "q_std": (41,), "r_std": (...),
           "n_q": int, "n_r": int}`` — variances plus their square roots (std form,
        which is what the spec's ``process_noise_std`` / ``noise_std`` expect).
    """
    ser = list(series if series is not None else dataset.pool)
    if max_series:
        ser = ser[:max_series]
    meta = dataset.meta
    args = [
        (
            meta,
            digester_id,
            np.asarray(s.truth, float),
            np.asarray(s.feed, float),
            np.asarray(s.measurements, float),
            int(stride),
            int(obs_stride),
            int(horizon),
        )
        for s in ser
    ]

    if jobs and jobs > 1 and len(args) > 1:
        import multiprocessing as mp

        with mp.Pool(min(jobs, len(args))) as pool:
            parts = pool.map(_series_residuals, args, chunksize=1)
    else:
        parts = [_series_residuals(a) for a in args]

    q_sum = np.zeros(STATE_SIZE)
    q_cnt = 0
    r_sum = np.zeros(len(meta["sensors"]))
    r_cnt = 0
    for qs, qc, rs, rc in parts:
        q_sum += qs
        q_cnt += qc
        r_sum += rs
        r_cnt += rc
    # Per-step Q: divide the h-step error variance by h. For a *white* model error this
    # is horizon-independent; if it grows with h the error is correlated (a bias), and the
    # one-step estimate would leave the filter over-confident.
    q_diag = q_sum / max(q_cnt, 1) / max(int(horizon), 1)
    r_diag = r_sum / max(r_cnt, 1)
    return {
        "q_diag": q_diag,
        "r_diag": r_diag,
        "q_std": np.sqrt(q_diag),
        "r_std": np.sqrt(r_diag),
        "n_q": int(q_cnt),
        "n_r": int(r_cnt),
        "n_series": len(ser),
        "horizon": int(horizon),
    }


def q_scales_vs_nominal(meta: dict, q_std: np.ndarray, digester_id: str = "primary"):
    """How far the empirical Q is from the filter's nominal Q, per state and per block.

    Returns ``(per_state_ratio (41,), {block: median ratio})`` — the factors the nominal
    process-noise std would need in order to match the measured model error. A block
    ratio far from 1 says the nominal Q is mis-scaled there; these are exactly the
    correction factors a search would otherwise have to discover on its own.
    """
    from ..specs import BLOCK_INDICES

    plant = build_plant(meta, digester_id)
    _, _, spec = build_filter_components(
        plant,
        digester_id=digester_id,
        substrates=substrates_from_meta(meta),
        sensors=[s.lower() for s in meta["sensors"]],
    )
    nominal = np.full(STATE_SIZE, np.nan)
    for p in spec.kind_indices("adm1"):
        nominal[spec.channels[p].adm1_index] = float(spec.channels[p].process_noise_std)
    ratio = np.asarray(q_std, float) / np.where(nominal > 0, nominal, np.nan)
    per_block = {
        b: float(np.nanmedian(ratio[list(idx)])) for b, idx in BLOCK_INDICES.items()
    }
    return ratio, per_block, nominal
