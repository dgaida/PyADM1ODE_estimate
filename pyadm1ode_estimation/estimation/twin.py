"""Twin-experiment helpers for validating a filter against a known truth.

The standard recipe:

  1. Build a *truth* plant + spec and propagate it through a chosen
     horizon, recording the full state trajectory and the noise-free
     observations.
  2. Add white Gaussian noise (and optional gate masks) to the
     observations.
  3. Build a *filter* plant + spec — same topology, but the filter
     does not know the truth.
  4. Step the filter through the same horizon, feeding it the noisy
     gated observations.
  5. Compare the filter's posterior trajectory + uncertainty band to
     the truth.

This module owns the propagation/measurement loop and a few
plot/diagnostic helpers; the actual scenario (initial states, input
trajectories, gate masks) belongs to the caller's plant-specific
twin script.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import EstimationStep, StateEstimator
from .observation_model import ObservationModel
from .process_model import ADM1ProcessModel
from .state_vector import StateVectorSpec


@dataclass
class TwinTrajectory:
    """Output of one twin-experiment run."""

    time: np.ndarray  # (T,) days
    truth: np.ndarray  # (T, n_state)
    x_hat: np.ndarray  # (T, n_state)
    std: np.ndarray  # (T, n_state) — sqrt(diag(P))
    obs_clean: pd.DataFrame  # noise-free observations h(truth)
    obs_noisy: pd.DataFrame  # measurements actually fed to filter
    innovations: pd.DataFrame  # filter innovation per active channel
    nis: np.ndarray  # (T,)

    @property
    def channel_names(self) -> list[str]:
        return list(self.x_hat.shape[1] * [""])  # filled in by caller


def propagate_truth(
    spec: StateVectorSpec,
    process: ADM1ProcessModel,
    obs: ObservationModel,
    x0: np.ndarray,
    dt_hours: float,
    n_steps: int,
    equilibration_dt: float = 1.0 / 24.0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Run the noise-free reference trajectory.

    Returns:
        time (n_steps + 1,): in days; ``time[0] = 0``.
        states (n_steps + 1, n_state): truth trajectory including
            the initial state at index 0.
        obs_clean: hourly observations ``h(truth)`` for every
            channel of ``obs`` (no noise, no gating).
    """
    dt = dt_hours / 24.0
    n_state = len(spec)
    states = np.zeros((n_steps + 1, n_state))
    states[0] = x0
    obs_rows: list[list[float]] = []

    process.refresh_outputs(x0, equilibration_dt=equilibration_dt)
    obs_rows.append([float(c.extractor(process.plant, x0)) for c in obs.channels])

    x = x0.copy()
    for k in range(n_steps):
        x = process.step(x, dt)
        states[k + 1] = x
        process.refresh_outputs(x, equilibration_dt=equilibration_dt)
        obs_rows.append([float(c.extractor(process.plant, x)) for c in obs.channels])

    time = np.arange(n_steps + 1, dtype=float) * dt
    obs_clean = pd.DataFrame(
        obs_rows,
        index=time,
        columns=[c.name for c in obs.channels],
    )
    return time, states, obs_clean


def add_measurement_noise(
    obs_clean: pd.DataFrame,
    obs_model: ObservationModel,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add ``N(0, σ_i)`` noise to each column according to the channel."""
    out = obs_clean.copy()
    for c in obs_model.channels:
        if c.name in out.columns:
            noise = rng.normal(0.0, c.noise_std, size=len(out))
            out[c.name] = out[c.name].values + noise
    return out


def apply_gate_masks(
    obs_noisy: pd.DataFrame,
    obs_model: ObservationModel,
    gate_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Set non-observable cells to NaN according to the gates.

    ``gate_frame`` columns are matched to the channels' ``gate_column``.
    Cells where the gate is falsy become NaN in the returned frame.
    """
    out = obs_noisy.copy()
    if gate_frame is None:
        return out
    for c in obs_model.channels:
        if c.gate_column and c.gate_column in gate_frame.columns:
            mask = ~gate_frame[c.gate_column].astype(bool).values
            out.loc[out.index[mask], c.name] = np.nan
    return out


def run_filter(
    estimator: StateEstimator,
    spec: StateVectorSpec,
    obs: ObservationModel,
    obs_noisy: pd.DataFrame,
    gate_frame: pd.DataFrame | None,
    dt_hours: float,
    pre_step: Callable[[int, float], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[EstimationStep]]:
    """Step the filter through the measurement frame.

    Args:
        pre_step: optional callback ``f(k, t)`` invoked at the top of each
            iteration (before ``predict``). Lets the caller mutate the
            estimator's process model as a function of step/time, e.g. to
            inject a time-growing model error. Default ``None`` is a no-op.

    Returns:
        x_hat (T, n_state), std (T, n_state), steps list.
    """
    dt = dt_hours / 24.0
    n_state = len(spec)
    T = len(obs_noisy)
    x_hat = np.zeros((T, n_state))
    std = np.zeros((T, n_state))
    steps: list[EstimationStep] = []

    for k, t in enumerate(obs_noisy.index):
        if pre_step is not None:
            pre_step(k, float(t))
        if k > 0:
            estimator.predict(dt)

        y_dict: dict[str, float] = {}
        for c in obs.channels:
            if c.name in obs_noisy.columns:
                val = float(obs_noisy.iloc[k][c.name])
                if np.isfinite(val):
                    y_dict[c.name] = val

        gate_dict: dict[str, float] = {}
        if gate_frame is not None:
            for col in gate_frame.columns:
                gate_dict[col] = float(gate_frame.iloc[k][col])

        step = estimator.update(y_dict, t=float(t), gate_values=gate_dict)
        x_hat[k] = step.x_hat
        std[k] = np.sqrt(np.diag(step.P))
        steps.append(step)

    return x_hat, std, steps


def coverage_within_2sigma(
    truth: np.ndarray,
    x_hat: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """Per-channel fraction of time steps where truth ∈ x_hat ± 2σ."""
    inside = np.abs(truth - x_hat) <= 2.0 * std
    return inside.mean(axis=0)
