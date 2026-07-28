"""Protocols and shared dataclasses for the estimation subpackage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class EstimationStep:
    """One filter step's output, suitable for time-series logging.

    Attributes:
        t: Wall-clock time of the step (days since experiment start
            or any consistent monotonic reference).
        x_hat: Posterior state estimate, shape ``(n_state,)``.
        P: Posterior covariance, shape ``(n_state, n_state)``.
        y_pred: Predicted measurement value for each channel that
            had an observation in this step. Empty dict on a pure
            predict-only step.
        y_std: Standard deviation of the predicted measurement,
            ``sqrt(diag(S))`` where ``S = Cov(h(sigma)) + R`` is the
            full innovation covariance. Includes both state-driven
            uncertainty (state covariance propagated through ``h``)
            and the sensor noise ``R``. Useful for plotting predicted
            measurement bands.
        innovation: ``y_obs - y_pred`` per active channel.
        nis: Normalised innovation squared
            ``ν^T S^{-1} ν`` summed across active channels. ``nan``
            when no channel was observed.
        active_channels: Names of channels that contributed to the
            update. Useful for diagnostics with gated channels.
    """

    t: float
    x_hat: np.ndarray
    P: np.ndarray
    y_pred: dict[str, float] = field(default_factory=dict)
    y_std: dict[str, float] = field(default_factory=dict)
    innovation: dict[str, float] = field(default_factory=dict)
    nis: float = float("nan")
    active_channels: list = field(default_factory=list)


@dataclass
class TrajectoryEstimate:
    """Output of a batch/offline estimator over a whole window.

    The shared currency between recursive filters and batch smoothers: a
    posterior trajectory with a per-state uncertainty band, so the twin-
    experiment harness (coverage, plots, error metrics) can evaluate both
    estimator families the same way.

    Attributes:
        time: Query times, shape ``(T,)`` [days].
        x_hat: Posterior state estimate, shape ``(T, n_state)``.
        std: Per-state posterior standard deviation, shape ``(T, n_state)``.
            Zeros when the estimator provides no uncertainty.
    """

    time: np.ndarray
    x_hat: np.ndarray
    std: np.ndarray


class BatchEstimator(Protocol):
    """Common interface for offline/batch estimators (e.g. a PINN smoother).

    Unlike :class:`StateEstimator`, which runs a recursive predict/update
    loop, a batch estimator is *fitted* to a whole window of (sparse)
    measurements at once and then queried for the full state trajectory:

    .. code-block:: text

        estimator.fit(...)                       # train over the window
        traj = estimator.estimate(query_times)   # (T, n_state) + std

    ``fit`` is estimator-specific (training schedule, collocation, priors);
    :meth:`estimate` is the shared output contract used for evaluation.
    """

    def estimate(self, times: np.ndarray) -> TrajectoryEstimate:
        """Return the posterior trajectory + uncertainty at ``times``."""


class StateEstimator(Protocol):
    """Common interface for all filters in this subpackage.

    The lifecycle is the usual two-step Bayesian filter loop:

    .. code-block:: text

        for each measurement time t:
            estimator.predict(dt)            # propagate prior
            step = estimator.update(y_dict)  # incorporate measurement
            log(step)

    Augmented-input states (e.g. substrate flows) live inside the
    state vector and are picked up by ``predict`` from the latest
    ``x_hat``. Sparse / gated channels are passed in ``y_dict`` with
    only the keys that are currently observable.
    """

    def predict(self, dt: float) -> None:
        """Propagate the state distribution forward by ``dt`` days."""

    def update(self, y: dict[str, float], t: float) -> EstimationStep:
        """Fuse observations ``y`` at time ``t`` into the estimate."""

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Re-initialise the filter with a new prior."""
