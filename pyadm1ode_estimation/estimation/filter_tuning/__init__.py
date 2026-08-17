"""Filter tuning & calibration (dataset-agnostic).

Tunes the noise / uncertainty parameters (Q, R, σ) of the model-based **filters** (UKF
variants) — distinct from ``estimation.deep_learning``, which *trains* neural estimators.
Separates three concerns so any filter can be tuned on any dataset:

* :mod:`datasets`       — load ANY dataset into ``Episode`` objects + train/val/test splits.
* :mod:`filter_runners` — build & run a filter (UKF variants) on one episode with a
  parameter set ``theta``; exposes the ``run_episode(theta, episode) -> (x_hat, std)``
  callable the tuners consume, plus the persistent ``WorkerPool``.
* :mod:`metrics`        — coverage, FOS/TAC-band coverage, NEES, critical-state decision.

Tuners (all consume episodes + a ``run_episode`` callable, so they are filter- and
dataset-agnostic):

* :mod:`sigma_calibration` — post-hoc per-state σ recalibration (does not change x̂).
* :mod:`empirical_noise`   — Q and R computed from ground truth instead of searched.
* :mod:`noise_search`      — approach 1.1: search Q/R (+P0) block-scales on validation.
* :mod:`differentiable`    — approach 1.2: differentiable filter, learn Q/R by gradient.
* :mod:`cmaes_search`      — approach 1.3: empirical Q + CMA-ES; the pipeline behind the
  shipped benchmark reference. See this package's README for a worked example.

CLI orchestrator: :mod:`tune_filter`.
"""

from .datasets import EstimatorDataset, Series, get_dataset, load_dataset

__all__ = ["EstimatorDataset", "Series", "get_dataset", "load_dataset"]
