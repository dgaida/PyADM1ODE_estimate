"""Benchmark series for the PINN v2 experiments.

One :class:`BenchmarkSeries` per stored series with the fields the trajectory
PINN needs - hourly time, the 41 true states, the true and the noisy feed,
the per-series kinetic factors and the operating mode. The arrays come from the
shared loader :func:`~..filter_tuning.datasets.load_benchmark` (one reader for
every estimator); this module only adds the PINN-side conveniences (feed lookup,
feed-history features, training-split priors).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..filter_tuning.datasets import DEFAULT_BENCHMARK_DIR, Series, load_benchmark


@dataclass(frozen=True)
class BenchmarkSeries:
    """One benchmark series (hourly, 60 days)."""

    index: int
    split: str  # "train" | "test"
    regime: str
    seed: int
    time: np.ndarray  # (T,) days
    states: np.ndarray  # (T, 41) truth
    feed_true: np.ndarray  # (T, n_sub) m3/d
    feed_noisy: np.ndarray  # (T, n_sub)
    switch_days: np.ndarray  # (n_switch,)
    kinetic_keys: list[str]
    kinetic_factors: np.ndarray  # (n_kin,)
    measurements: np.ndarray | None = None  # (T, n_sensor) noisy online sensors
    ukf_x_hat: np.ndarray | None = None  # (T, 41) reference UKF estimate (test split)
    ukf_std: np.ndarray | None = None  # (T, 41)

    @classmethod
    def from_series(cls, s: Series, index: int, split: str) -> BenchmarkSeries:
        """Wrap a shared :class:`~..filter_tuning.datasets.Series`.

        Needs the generator truth ``feed_true`` / ``kinetic_factors`` /
        ``kinetic_keys`` in ``s.aux`` (any benchmark split has them).
        """
        missing = [
            k
            for k in ("feed_true", "kinetic_factors", "kinetic_keys")
            if k not in s.aux
        ]
        if missing:
            raise KeyError(
                f"series lacks generator truth {missing}; not a benchmark split?"
            )
        return cls(
            index=index,
            split=split,
            regime=s.label,
            seed=int(s.seed) if s.seed is not None else -1,
            time=np.asarray(s.time, dtype=float),
            states=np.asarray(s.truth, dtype=float),
            feed_true=np.asarray(s.aux["feed_true"], dtype=float),
            feed_noisy=np.asarray(s.feed, dtype=float),
            switch_days=np.asarray(s.switch_days, dtype=float),
            kinetic_keys=list(s.aux["kinetic_keys"]),
            kinetic_factors=np.asarray(s.aux["kinetic_factors"], dtype=float),
            measurements=np.asarray(s.measurements, dtype=float),
            ukf_x_hat=s.aux.get("ukf_x_hat"),
            ukf_std=s.aux.get("ukf_std"),
        )

    @property
    def n_steps(self) -> int:
        return len(self.time)

    def scoring_dict(self, sl: slice = slice(None)) -> dict[str, np.ndarray]:
        """Series as the dict ``datasets/benchmark/scoring.py`` expects, restricted to ``sl``."""
        d = {
            "states": self.states[sl],
            "time": self.time[sl],
            "switch_days": self.switch_days,
        }
        if self.ukf_x_hat is not None:
            d["ukf_x_hat"] = self.ukf_x_hat[sl]
        if self.ukf_std is not None:
            d["ukf_std"] = self.ukf_std[sl]
        return d

    @property
    def dt_days(self) -> float:
        return float(self.time[1] - self.time[0])

    def feed_at(self, t, *, true: bool = True) -> np.ndarray:
        """Piecewise-constant feed at day(s) ``t`` (hourly grid lookup), ``(..., n_sub)``."""
        return feed_lookup(self.feed_true if true else self.feed_noisy, self.time, t)


def feed_lookup(feed: np.ndarray, time: np.ndarray, t) -> np.ndarray:
    """Feed row of the grid step containing ``t`` (``t`` in days, any shape)."""
    t = np.asarray(t, dtype=float)
    dt = float(time[1] - time[0])
    k = np.clip(np.floor((t - time[0]) / dt + 1e-9).astype(int), 0, len(time) - 1)
    return feed[k]


#: Feed-history bins [days back] per feed mode: (from, to) means the mean feed over
#: ``[t - to, t - from]``. ``hist7`` = daily means of the last week, ``hist30`` = weekly
#: means of the last month.
FEED_HISTORY_BINS: dict[str, list[tuple[float, float]]] = {
    "current": [],
    "none": [],
    "hist7": [(float(k), float(k + 1)) for k in range(7)],
    "hist30": [(0.0, 7.0), (7.0, 14.0), (14.0, 21.0), (21.0, 30.0)],
}


def n_feed_features(n_sub: int, mode: str) -> int:
    if mode == "none":
        return 0
    return n_sub * (1 + len(FEED_HISTORY_BINS[mode]))


def feed_features(
    feed: np.ndarray, time: np.ndarray, t, mode: str = "current"
) -> np.ndarray:
    """Feed input of the network at day(s) ``t``: ``(..., n_feed_features)``.

    ``current``: the feed of the hour containing ``t``. ``hist7`` / ``hist30``: the
    current feed followed by the mean feed of each history bin (a bin that reaches
    before the record start is averaged over the available part; before the first
    hour it falls back to the first feed row). Uses only the dosing log, no
    switch markers.
    """
    t = np.asarray(t, dtype=float)
    cur = feed_lookup(feed, time, t)
    if mode == "none":
        return np.zeros(t.shape + (0,))
    if mode == "current":
        return cur
    dt = float(time[1] - time[0])
    csum = np.concatenate(
        [np.zeros((1, feed.shape[1])), np.cumsum(feed, axis=0)], axis=0
    )

    def idx(tt):
        return np.clip(np.floor((tt - time[0]) / dt + 1e-9).astype(int), 0, len(time))

    parts = [cur]
    for a, b in FEED_HISTORY_BINS[mode]:
        i1, i0 = idx(t - a), idx(t - b)
        n = np.maximum(i1 - i0, 1)[..., None]
        mean = (csum[i1] - csum[i0]) / n
        parts.append(np.where((i1 - i0)[..., None] > 0, mean, feed[0]))
    return np.concatenate(parts, axis=-1)


def load_meta(path: str | Path | None = None) -> dict:
    d = Path(path) if path else DEFAULT_BENCHMARK_DIR
    return json.loads((d / "meta.json").read_text(encoding="utf-8"))


def load_series(
    split: str = "test", path: str | Path | None = None, indices=None
) -> list[BenchmarkSeries]:
    """Load the ``train`` or ``test`` split through the shared benchmark loader."""
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    ds = load_benchmark(path, splits=(split,))
    rows = ds.pool if split == "train" else ds.test
    sel = range(len(rows)) if indices is None else [int(i) for i in indices]
    return [BenchmarkSeries.from_series(rows[i], i, split) for i in sel]


def train_split_log_stats(
    n_out: int = 31, path: str | Path | None = None, floor: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state mean / std of ``log x`` over the whole *training* split.

    The network's output normalisation ``x = exp(mu + sigma z)`` must not be fitted
    on the series it is evaluated on (that would leak the true states); the
    training split is the legitimate prior.
    """
    d = Path(path) if path else DEFAULT_BENCHMARK_DIR
    z = np.load(d / "train.npz", allow_pickle=True)
    states = np.asarray(z["states"], dtype=float)[..., :n_out]
    logx = np.log(np.clip(states, 1.0e-30, None)).reshape(-1, n_out)
    return logx.mean(axis=0), np.maximum(logx.std(axis=0), floor)


def train_split_scale_reference(
    n_per_regime: int = 1,
    every: int = 6,
    path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(states, feed_noisy)`` rows from the training split for the residual scale.

    Takes ``n_per_regime`` series of each operating mode, subsampled every
    ``every`` steps, so the physics residual scale of a field-setting run is fitted
    on training data rather than on the evaluated series' own truth.
    """
    d = Path(path) if path else DEFAULT_BENCHMARK_DIR
    z = np.load(d / "train.npz", allow_pickle=True)
    regimes = np.asarray(z["regime"]).astype(str)
    states, feed = [], []
    for reg in sorted(set(regimes)):
        for i in np.flatnonzero(regimes == reg)[:n_per_regime]:
            states.append(np.asarray(z["states"][i], dtype=float)[::every])
            feed.append(np.asarray(z["feed_noisy"][i], dtype=float)[::every])
    return np.concatenate(states), np.concatenate(feed)


def scoring_module():
    """The benchmark's own ``scoring.py`` (not a package), imported by path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "benchmark_scoring", DEFAULT_BENCHMARK_DIR / "scoring.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
