"""Sliding-window operational driver for the amortised observer.

Turns the pre-trained :class:`Adm1Observer` + self-supervised fine-tuning into a
**continuous, online estimator**. It keeps a moving window of the most recent
measurements; on every new sensor reading it returns the current state estimate,
and — on a configurable schedule — self-supervised fine-tunes on the recent
window so it adapts to the live plant.

Because the observer's GRU is *causal* (each step sees only past + present), the
estimate at the newest step is a proper online filtered estimate. This is the
"engine" (:func:`finetune_observer`) wrapped in the "car that drives it
continuously".

Typical use::

    swo = SlidingWindowObserver(observer, obs_model, feat_mean, feat_std,
                                window_hours=48, finetune_every=24)
    for meas, feed in live_sensor_stream:        # meas = [Q_gas, Q_ch4, pH]
        est = swo.step(meas, feed)               # est.state = current 41-state
        log(est.t, est.state, est.std)
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .observation_torch import TorchObservationModel
from .observer import Adm1Observer
from .observer_train import finetune_observer

_STATE_SIZE = 41


@dataclass
class OnlineEstimate:
    """One online step's output."""

    t: float  # current time [days]
    state: np.ndarray  # (41,) current state estimate
    std: np.ndarray  # (41,) current per-state uncertainty (0 without MC-Dropout)
    finetuned: bool  # whether a fine-tune ran on this step


class SlidingWindowObserver:
    """Continuous online estimator over a sliding measurement window.

    Args:
        observer: a **pre-trained** :class:`Adm1Observer`.
        obs_model: the differentiable observation map (for the fine-tune
            measurement loss); its ``channel_names`` order must match the
            ``measurement`` vectors pushed in.
        feat_mean, feat_std: the pre-training feature normalisation stats
            (measurements first, then feed), shape ``(n_features,)``.
        window_hours: length of the sliding window [h].
        dt_hours: measurement interval [h].
        finetune_every: fine-tune every this many steps (``0`` disables it).
        finetune_epochs, finetune_lr: fine-tune schedule per trigger.
        finetune_kwargs: extra keyword args forwarded to :func:`finetune_observer`
            (e.g. ``lambda_phys``, ``lambda_anchor``, ``params``).
        min_steps: minimum buffered steps before fine-tuning starts
            (default: half the window).
        mc_samples: MC-Dropout forward passes for the uncertainty band
            (``1`` → deterministic, ``std = 0``).
    """

    def __init__(
        self,
        observer: Adm1Observer,
        obs_model: TorchObservationModel,
        feat_mean: Sequence[float],
        feat_std: Sequence[float],
        *,
        window_hours: float = 48.0,
        dt_hours: float = 1.0,
        finetune_every: int = 0,
        finetune_epochs: int = 30,
        finetune_lr: float = 1.0e-4,
        finetune_kwargs: dict | None = None,
        min_steps: int | None = None,
        mc_samples: int = 1,
        feed_aware: bool = True,
        nominal_feed_sum: float | None = None,
    ):
        self.observer = observer
        self.obs_model = obs_model
        self.feat_mean = np.asarray(feat_mean, dtype=float)
        self.feat_std = np.asarray(feat_std, dtype=float)
        self.dt_hours = float(dt_hours)
        self.dt_days = self.dt_hours / 24.0
        self.window_steps = max(1, round(window_hours / dt_hours))
        self.finetune_every = int(finetune_every)
        self.finetune_epochs = int(finetune_epochs)
        self.finetune_lr = float(finetune_lr)
        self.finetune_kwargs = dict(finetune_kwargs or {})
        self.min_steps = (
            int(min_steps) if min_steps is not None else max(4, self.window_steps // 2)
        )
        self.mc_samples = int(mc_samples)
        # Feed-awareness: make the fine-tune physics use the live feed. Off the
        # nominal operating point (off-distribution) the pre-trained params'
        # nominal q_ad is wrong, so we rescale it to the window's actual feed.
        self.feed_aware = bool(feed_aware)
        self.nominal_feed_sum = nominal_feed_sum
        self._base_q_ad = float(observer.params.q_ad)

        self._meas: deque = deque(maxlen=self.window_steps)  # raw measurements (n_ch,)
        self._feed: deque = deque(maxlen=self.window_steps)  # feed (n_feed,)
        self._count = 0  # total steps ever seen

    # -- public ---------------------------------------------------------
    def step(
        self, measurement: Sequence[float], feed: Sequence[float]
    ) -> OnlineEstimate:
        """Ingest one reading, (optionally) fine-tune, and return the current estimate."""
        self._meas.append(np.asarray(measurement, dtype=float))
        self._feed.append(np.asarray(feed, dtype=float))
        self._count += 1

        feats_norm = self._features()  # (W, n_feat)
        finetuned = False
        if (
            self.finetune_every > 0
            and len(self._meas) >= self.min_steps
            and self._count % self.finetune_every == 0
        ):
            ft_kwargs = dict(self.finetune_kwargs)
            if self.feed_aware:
                ft_kwargs["params"] = self._feed_aware_params()
            finetune_observer(
                self.observer,
                feats_norm,
                self.obs_model,
                np.stack(self._meas),
                dt_days=self.dt_days,
                epochs=self.finetune_epochs,
                lr=self.finetune_lr,
                **ft_kwargs,
            )
            finetuned = True

        state, std = self._estimate(feats_norm)  # (W, 41) each
        return OnlineEstimate(
            t=(self._count - 1) * self.dt_days,
            state=state[-1],
            std=std[-1],
            finetuned=finetuned,
        )

    def run(self, measurements: np.ndarray, feeds: np.ndarray) -> list[OnlineEstimate]:
        """Replay a recorded stream ``(T, n_ch)`` / ``(T, n_feed)`` step by step."""
        return [self.step(measurements[k], feeds[k]) for k in range(len(measurements))]

    # -- internals ------------------------------------------------------
    def _feed_aware_params(self):
        """Params snapshot with q_ad rescaled to the window's mean total feed."""
        feed_sum = float(np.stack(self._feed).sum(axis=1).mean())
        if self.nominal_feed_sum:
            q_ad = self._base_q_ad * (feed_sum / self.nominal_feed_sum)
        else:
            q_ad = feed_sum
        return self.observer.params.with_q_ad(q_ad)

    def _features(self) -> np.ndarray:
        raw = np.concatenate(
            [np.stack(self._meas), np.stack(self._feed)], axis=1
        )  # (W, n_feat)
        norm = (raw - self.feat_mean) / self.feat_std
        # A gated / offline sensor arrives as NaN; the recurrent net cannot take
        # NaN as input, so map it to the normalised mean (0). The fine-tune loss
        # still masks the NaN on the *target* side.
        return np.nan_to_num(norm, nan=0.0)

    def _estimate(self, feats_norm: np.ndarray):
        u = torch.tensor(feats_norm[None, ...], dtype=self.observer._dtype)
        if self.mc_samples > 1 and self.observer.dropout_p > 0:
            self.observer.train()  # MC-Dropout active
            with torch.no_grad():
                draws = np.stack(
                    [self.observer(u).cpu().numpy()[0] for _ in range(self.mc_samples)]
                )
            return draws.mean(axis=0), draws.std(axis=0)
        self.observer.eval()
        with torch.no_grad():
            x = self.observer(u).cpu().numpy()[0]
        return x, np.zeros_like(x)
