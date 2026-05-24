"""Protocols and shared dataclasses for the estimation subpackage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Protocol

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
    y_pred: Dict[str, float] = field(default_factory=dict)
    innovation: Dict[str, float] = field(default_factory=dict)
    nis: float = float("nan")
    active_channels: list = field(default_factory=list)


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

    def update(self, y: Dict[str, float], t: float) -> EstimationStep:
        """Fuse observations ``y`` at time ``t`` into the estimate."""

    def reset(self, x0: np.ndarray, P0: np.ndarray) -> None:
        """Re-initialise the filter with a new prior."""
