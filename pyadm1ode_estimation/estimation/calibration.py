"""Data-structure-agnostic calibration of a state estimator's *uncertainty*.

The goal is a filter whose predicted covariance is **well calibrated**: its ±2σ
band should cover the truth ~95 % of the time, and states it cannot estimate well
should carry a correspondingly large σ. The main lever is the process-noise
covariance ``Q`` (per state / per state-group); measurement noise ``R`` and the
initial covariance ``P0`` are optional secondary knobs.

Design — nothing here is tied to the UKF, ADM1 or any file format:

* :class:`Episode` — one labelled trajectory: a measurement frame + the ground
  truth. Build these from *any* data source (a benchmark ``.npz``, a CSV, a live
  log, a different plant ...).
* ``run_episode(theta, episode) -> (x_hat, std)`` — an opaque callable you supply
  that builds and runs *your* estimator with parameters ``theta`` on one episode.
  This is the single integration point; the calibrator never constructs a filter.
* ``q_groups`` — maps a Q-group name to the state indices it controls, and
  ``theta["q_scale"][group]`` is that group's multiplicative process-noise scale.
  Covariance matching adjusts these scales so each group's NEES → 1.
* ``map_fn`` — inject parallelism (e.g. ``multiprocessing.Pool.map``); defaults to
  serial :func:`map`.

The optimiser is **covariance matching**: run the filter, measure the per-state
normalised error ``z² = (truth - x̂)² / σ²`` (its time-mean is the NEES), and scale
each group's ``Q`` by ``√(NEES)`` (damped) until the NEES sits at 1. This drives
both goals at once — a NEES of 1 gives ~95 % 2σ coverage, and a state the filter
tracks poorly ends up with a large ``Q`` and hence an honestly large σ.
"""

from __future__ import annotations

import copy
import functools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

try:  # pandas is only needed by callers that use DataFrame measurement frames
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None  # type: ignore

#: 2σ coverage of a Gaussian: P(|z| ≤ 2) = erf(2/√2).
TWO_SIGMA_COVERAGE = math.erf(2.0 / math.sqrt(2.0))  # ≈ 0.9545

# A ``run_episode`` maps (theta, episode) -> (x_hat (T, n), std (T, n)).
RunEpisode = Callable[[Mapping, "Episode"], tuple[np.ndarray, np.ndarray]]
MapFn = Callable[[Callable, Sequence], Sequence]


@dataclass
class Episode:
    """One calibration trajectory.

    Attributes:
        obs: the measurement frame the estimator consumes (e.g. a
            ``pandas.DataFrame`` indexed by time). Passed through untouched to
            ``run_episode``; the calibrator never inspects it.
        truth: ground-truth state trajectory, shape ``(T, n_state)``, aligned
            row-for-row with the estimator's output.
        dt_hours: sampling interval [h] (passed through to ``run_episode``).
        name: optional label for reporting.
        weight: relative weight in the aggregate metrics (default 1).
        burnin: number of leading steps excluded from the metrics (lets the filter
            converge from its prior before the calibration measures its error).
    """

    obs: object
    truth: np.ndarray
    dt_hours: float = 1.0
    name: str = ""
    weight: float = 1.0
    burnin: int = 0


@dataclass
class CalibrationReport:
    """Per-state and aggregate calibration metrics for a parameter set."""

    nees: np.ndarray  # (n_state,) time/episode-mean z² per state
    coverage_2sigma: np.ndarray  # (n_state,) fraction within ±2σ
    rmse: np.ndarray  # (n_state,) RMSE per state
    mean_std: np.ndarray  # (n_state,) mean predicted σ per state
    overall_coverage_2sigma: float  # over all state-time cells
    median_nees: float
    mean_log_nees_abs: float  # mean |log NEES| — 0 is perfectly calibrated
    sharpness: float  # mean predicted σ (lower is sharper)
    objective: float  # scalar to minimise (calibration first)
    n_cells: int = 0
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _run_and_align(run_episode: RunEpisode, theta: dict, ep: Episode):
    """Top-level (picklable) helper: run one episode, drop its burn-in, pair w/ truth."""
    x_hat, std = run_episode(theta, ep)
    b = int(getattr(ep, "burnin", 0) or 0)
    return (
        np.asarray(ep.truth, float)[b:],
        np.asarray(x_hat, float)[b:],
        np.asarray(std, float)[b:],
    )


def _stack_episode(results, weights):
    """Concatenate per-episode (truth, x_hat, std, w) into flat (cells, n) arrays."""
    tru = np.concatenate([r[0] for r in results], axis=0)
    est = np.concatenate([r[1] for r in results], axis=0)
    std = np.concatenate([r[2] for r in results], axis=0)
    w = np.concatenate([np.full(len(r[0]), wt) for r, wt in zip(results, weights)])
    return tru, est, std, w


def compute_report(
    results: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray]],
    weights: Sequence[float] | None = None,
    var_floor: float = 1e-12,
    coverage_target: float = TWO_SIGMA_COVERAGE,
) -> CalibrationReport:
    """Aggregate ``(truth, x_hat, std)`` episode results into a report.

    ``z² = (truth - x̂)² / (σ² + floor)``; its weighted mean per state is the NEES,
    which should be 1 when the filter is calibrated. ``objective`` puts calibration
    first (mean ``|log NEES|`` over informative states) and adds a light coverage
    and sharpness term so ties break toward well-covered, sharp estimates.
    """
    weights = list(weights) if weights is not None else [1.0] * len(results)
    truth, est, std, w = _stack_episode(results, weights)
    var = std**2 + var_floor
    err2 = (truth - est) ** 2
    z2 = err2 / var
    wsum = float(np.sum(w))

    def wmean(a, axis=0):
        return np.sum(a * w[:, None], axis=axis) / max(wsum, 1e-12)

    nees = wmean(z2)  # (n,)
    within = (np.abs(truth - est) <= 2.0 * std).astype(float)
    coverage = wmean(within)  # (n,)
    rmse = np.sqrt(wmean(err2))
    mean_std = wmean(std)

    overall_cov = float(np.sum(within * w[:, None]) / max(wsum * truth.shape[1], 1e-12))
    # informative states: non-degenerate truth (avoid the numerically ~zero states
    # dominating the calibration objective with meaningless ratios).
    rms_true = np.sqrt(wmean(truth**2))
    informative = rms_true > (1e-6 * (np.median(rms_true) + 1e-12))
    safe_nees = np.clip(nees[informative], 1e-6, 1e6)
    mean_log_nees_abs = (
        float(np.mean(np.abs(np.log(safe_nees)))) if safe_nees.size else 0.0
    )
    sharpness = float(np.mean(mean_std))

    objective = (
        mean_log_nees_abs
        + 2.0 * abs(overall_cov - coverage_target)
        + 0.02 * math.log1p(sharpness)
    )
    return CalibrationReport(
        nees=nees,
        coverage_2sigma=coverage,
        rmse=rmse,
        mean_std=mean_std,
        overall_coverage_2sigma=overall_cov,
        median_nees=(
            float(np.median(nees[informative]))
            if informative.any()
            else float(np.median(nees))
        ),
        mean_log_nees_abs=mean_log_nees_abs,
        sharpness=sharpness,
        objective=objective,
        n_cells=int(truth.shape[0]),
        extra={"informative": informative},
    )


# --------------------------------------------------------------------------
# Calibrator
# --------------------------------------------------------------------------
class UncertaintyCalibrator:
    """Covariance-matching calibrator for an estimator's process noise.

    Args:
        episodes: the calibration set.
        run_episode: ``(theta, episode) -> (x_hat, std)``; builds + runs your filter.
        q_groups: ``{group_name: [state indices]}``; each group has one process-noise
            scale in ``theta["q_scale"]``. Use one group per state for maximal freedom
            or per state-block for a robust, low-variance fit.
        theta0: initial parameters. Must contain ``"q_scale": {group: float}``; may
            also carry any keys your ``run_episode`` understands (``"r_scale"`` ...).
        map_fn: parallel map for the per-episode evaluations (default serial).
        q_bounds: clip range for each group's cumulative Q-scale.
    """

    def __init__(
        self,
        episodes: Sequence[Episode],
        run_episode: RunEpisode,
        q_groups: Mapping[str, Sequence[int]],
        theta0: dict | None = None,
        map_fn: MapFn | None = None,
        q_bounds: tuple[float, float] = (1e-3, 1e3),
    ):
        if not episodes:
            raise ValueError("need at least one calibration episode")
        self.episodes = list(episodes)
        self.run_episode = run_episode
        self.q_groups = {k: list(v) for k, v in q_groups.items()}
        self.map_fn = map_fn or (lambda f, it: list(map(f, it)))
        self.q_lo, self.q_hi = q_bounds
        self.n_state = self.episodes[0].truth.shape[1]
        self.theta0 = copy.deepcopy(theta0) if theta0 else {}
        self.theta0.setdefault("q_scale", {g: 1.0 for g in self.q_groups})
        for g in self.q_groups:
            self.theta0["q_scale"].setdefault(g, 1.0)

    # -- evaluation ------------------------------------------------------
    def collect(self, theta: dict):
        """Run every episode with ``theta``; return the raw per-episode
        ``(truth, x_hat, std)`` (burn-in already dropped). Uses
        :func:`functools.partial` (not a closure) so ``map_fn`` may be a real process
        pool — provided ``run_episode`` is top-level and the episodes are picklable."""
        fn = functools.partial(_run_and_align, self.run_episode, dict(theta))
        return list(self.map_fn(fn, self.episodes))

    def evaluate(self, theta: dict) -> CalibrationReport:
        """Run every episode with ``theta`` and aggregate the calibration metrics."""
        return compute_report(
            self.collect(theta), weights=[ep.weight for ep in self.episodes]
        )

    # -- covariance matching --------------------------------------------
    def _match_step(
        self, theta: dict, report: CalibrationReport, damping: float
    ) -> dict:
        """One covariance-matching update: scale each group's Q by √(group NEES)."""
        theta = copy.deepcopy(theta)
        for g, idx in self.q_groups.items():
            g_nees = float(np.nanmedian(report.nees[idx])) if idx else 1.0
            if not np.isfinite(g_nees) or g_nees <= 0:
                continue
            factor = g_nees ** (0.5 * damping)  # more error -> more Q
            theta["q_scale"][g] = float(
                np.clip(theta["q_scale"][g] * factor, self.q_lo, self.q_hi)
            )
        return theta

    def calibrate(
        self, iters: int = 4, damping: float = 0.8, verbose: bool = True
    ) -> tuple[dict, CalibrationReport, list[CalibrationReport]]:
        """Iterate covariance matching. Returns ``(theta*, final report, history)``.

        Keeps the parameters with the lowest objective seen (covariance matching is
        a damped fixed point, usually monotone but not guaranteed)."""
        theta = copy.deepcopy(self.theta0)
        history: list[CalibrationReport] = []
        best_theta, best_rep = theta, None
        for it in range(iters + 1):
            rep = self.evaluate(theta)
            history.append(rep)
            if best_rep is None or rep.objective < best_rep.objective:
                best_theta, best_rep = copy.deepcopy(theta), rep
            if verbose:
                print(
                    f"  iter {it}: cov2σ={rep.overall_coverage_2sigma:.3f} "
                    f"med_NEES={rep.median_nees:.2f} |log NEES|={rep.mean_log_nees_abs:.3f} "
                    f"sharp={rep.sharpness:.3g} obj={rep.objective:.4f}",
                    flush=True,
                )
            if it < iters:
                theta = self._match_step(theta, rep, damping)
        return best_theta, best_rep, history


# --------------------------------------------------------------------------
# Reporting helpers (data-agnostic)
# --------------------------------------------------------------------------
def sigma_scale_from_report(
    report: CalibrationReport, lo: float = 0.3, hi: float = 30.0
):
    """Per-state OUTPUT-σ multiplier that makes each state's NEES → 1: ``σ' = √(NEES)·σ``.

    This is **post-hoc variance recalibration** — it rescales the reported uncertainty
    to match the observed error WITHOUT changing the filter dynamics or the point
    estimate. Unlike inflating the process noise ``Q`` (which for weakly-observable
    states corrupts the prediction), it is a one-shot fit that always improves
    coverage and cannot destabilise the filter. Clipped to ``[lo, hi]``."""
    return np.clip(np.sqrt(np.maximum(report.nees, 1e-12)), lo, hi)


def hardest_states(report: CalibrationReport, state_names: Sequence[str], k: int = 8):
    """The ``k`` states with the largest RMSE-to-signal — i.e. worst estimated."""
    order = np.argsort(-report.rmse)
    return [
        (
            state_names[i] if i < len(state_names) else f"state_{i}",
            float(report.rmse[i]),
            float(report.mean_std[i]),
            float(report.nees[i]),
            float(report.coverage_2sigma[i]),
        )
        for i in order[:k]
    ]
