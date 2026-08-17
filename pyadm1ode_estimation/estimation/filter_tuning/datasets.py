"""Dataset-agnostic loading for estimator training / calibration.

The tuners never touch a raw file format — they consume :class:`~..calibration.Episode`
objects (a measurement frame + ground truth) plus a ``meta`` dict. Any dataset that can
produce per-series ``(measurements, feed, time, truth-states)`` plugs in through a loader
registered in :data:`LOADERS`; :func:`get_dataset` resolves a name or a path.

    from pyadm1ode_estimation.estimation.filter_tuning import get_dataset
    ds = get_dataset("benchmark")                 # or a path to a dataset directory
    train, val, test = ds.make_splits(days=30, burnin_days=2, per_group_train=2, per_group_val=2)
    #  -> three lists of Episode; feed them to any tuner

Design: the split is **stratified** by a per-series ``label`` (the operating mode for the
benchmark) so every split covers every regime. ``make_splits`` truncates each series to
``days`` and sets the metric ``burnin``; the full untruncated series stay available via
:attr:`EstimatorDataset.pool` / :attr:`~EstimatorDataset.test`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..calibration import Episode

# Default install location of the shipped benchmark dataset (../../datasets/benchmark
# relative to the repo root — this file lives at estimation/filter_tuning/).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCHMARK_DIR = _REPO_ROOT / "datasets" / "benchmark"


# --------------------------------------------------------------------------
@dataclass
class Series:
    """One labelled trajectory in dataset-native units (before truncation)."""

    measurements: np.ndarray  # (T, n_sensor) noisy online sensors
    feed: np.ndarray  # (T, n_sub) noisy substrate feed (control input)
    time: np.ndarray  # (T,) days
    truth: np.ndarray  # (T, n_state) ground-truth state
    switch_days: np.ndarray  # feed-change times [d]
    label: str = ""  # stratification label (e.g. operating mode)
    seed: int | None = None
    aux: dict = field(default_factory=dict)  # e.g. stored ukf_x_hat/ukf_std for reuse


@dataclass
class EstimatorDataset:
    """A dataset as a train/val **pool** + a held-out **test** set, plus ``meta``.

    ``pool`` is split into train/val by :meth:`make_splits`; ``test`` is never touched
    until the final evaluation.
    """

    name: str
    meta: dict
    pool: list[Series]  # split into train + val
    test: list[Series]

    # -- introspection ---------------------------------------------------
    @property
    def dt_hours(self) -> float:
        return float(self.meta.get("dt_hours", 1.0))

    @property
    def state_names(self) -> list[str]:
        imap = self.meta.get("state_index_map", {})
        n = int(self.meta.get("state_size", self.pool[0].truth.shape[1]))
        return [str(imap.get(str(i), f"state_{i}")) for i in range(n)]

    @property
    def sensor_names(self) -> list[str]:
        return list(self.meta.get("sensors", []))

    def labels(self) -> list[str]:
        return sorted({s.label for s in self.pool})

    # -- episode construction -------------------------------------------
    def _to_episode(
        self, s: Series, days: float | None, burnin_days: float, name: str
    ) -> Episode:
        T = (
            len(s.time)
            if days is None
            else min(len(s.time), round(days * 24.0 / self.dt_hours) + 1)
        )
        obs = {
            "measurements": np.asarray(s.measurements[:T], float),
            "feed_noisy": np.asarray(s.feed[:T], float),
            "time": np.asarray(s.time[:T], float),
            "switch_days": np.asarray(s.switch_days, float),
            "label": s.label,
        }
        # Hourly Nordmann titration (T, 2) = [FOS, TAC] in mg/L, if the dataset has it.
        # The runner subsamples it to the interval an experiment wants; it is NOT an
        # online sensor and is only used when explicitly switched on.
        if "fostac" in s.aux:
            obs["fostac"] = np.asarray(s.aux["fostac"][:T], float)
        burnin = round(burnin_days * 24.0 / self.dt_hours)
        return Episode(
            obs=obs,
            truth=np.asarray(s.truth[:T], float),
            dt_hours=self.dt_hours,
            name=name,
            burnin=burnin,
        )

    def _stratified_pick(
        self, series: list[Series], per_group: int | None, rng
    ) -> list[int]:
        """Indices picked evenly per label (all if ``per_group`` is None)."""
        by_label: dict[str, list[int]] = {}
        for i, s in enumerate(series):
            by_label.setdefault(s.label, []).append(i)
        out: list[int] = []
        for lbl in sorted(by_label):
            idx = list(by_label[lbl])
            rng.shuffle(idx)
            out += idx if per_group is None else idx[:per_group]
        return sorted(out)

    def split_indices(
        self,
        val_frac: float = 0.2,
        per_group_train: int | None = None,
        per_group_val: int | None = None,
        per_group_test: int | None = None,
        seed: int = 0,
    ) -> tuple[list[int], list[int], list[int]]:
        """Stratified train/val indices into ``pool`` + test indices into ``test``.

        The single source of truth for *how this dataset is split*: :meth:`make_splits`
        (filters) and the deep-learning adapter both go through it, so a filter and a
        network configured with the same ``(val_frac, seed)`` see exactly the same
        series — a prerequisite for comparing them.

        ``val_frac`` of each label goes to val, the rest to train; ``per_group_*`` then
        subsamples how many series per label to actually emit (None = all).
        """
        rng = np.random.default_rng(seed)
        by_label: dict[str, list[int]] = {}
        for i, s in enumerate(self.pool):
            by_label.setdefault(s.label, []).append(i)

        train_idx, val_idx = [], []
        for lbl in sorted(by_label):
            idx = list(by_label[lbl])
            rng.shuffle(idx)
            n_val = max(1, round(val_frac * len(idx)))
            vpool, tpool = idx[:n_val], idx[n_val:]
            val_idx += vpool[:per_group_val] if per_group_val else vpool
            train_idx += tpool[:per_group_train] if per_group_train else tpool

        test_idx = self._stratified_pick(
            self.test, per_group_test, np.random.default_rng(seed + 1)
        )
        return sorted(train_idx), sorted(val_idx), test_idx

    def make_splits(
        self,
        days: float | None = None,
        burnin_days: float = 0.0,
        val_frac: float = 0.2,
        per_group_train: int | None = None,
        per_group_val: int | None = None,
        per_group_test: int | None = None,
        seed: int = 0,
    ) -> tuple[list[Episode], list[Episode], list[Episode]]:
        """Stratified train/val split of ``pool`` + the held-out ``test`` set.

        Splits via :meth:`split_indices` and wraps the result as :class:`Episode`
        objects truncated to ``days``. Returns ``(train, val, test)``.
        """
        train_idx, val_idx, test_idx = self.split_indices(
            val_frac, per_group_train, per_group_val, per_group_test, seed
        )
        train = [
            self._to_episode(
                self.pool[i], days, burnin_days, f"{self.pool[i].label}#{i}"
            )
            for i in train_idx
        ]
        val = [
            self._to_episode(
                self.pool[i], days, burnin_days, f"{self.pool[i].label}#{i}"
            )
            for i in val_idx
        ]
        test = [
            self._to_episode(
                self.test[i], days, burnin_days, f"{self.test[i].label}#t{i}"
            )
            for i in test_idx
        ]
        return train, val, test


# --------------------------------------------------------------------------
# Loaders (one per dataset format).  Register new datasets here.
# --------------------------------------------------------------------------
def load_benchmark(path: str | Path | None = None) -> EstimatorDataset:
    """Load the ADM1 benchmark (``train.npz`` + ``test.npz`` + ``meta.json``)."""
    d = Path(path) if path else DEFAULT_BENCHMARK_DIR
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    tr = np.load(d / "train.npz", allow_pickle=True)
    te = np.load(d / "test.npz", allow_pickle=True)

    # Read every array from the (compressed) npz EXACTLY ONCE. Indexing `tr["states"][i]`
    # inside a loop re-decompresses the whole ~47 MB array on every iteration and, worse,
    # the returned row is a *view* that keeps its freshly decompressed base alive — 100
    # rows then retain 100 separate full arrays (~5.8 GB observed). Pulling the arrays out
    # first keeps one base per field and makes loading ~100x faster.
    def _grab(z, keys):
        return {k: z[k] for k in keys if k in z.files}

    trd = _grab(
        tr,
        (
            "measurements",
            "feed_noisy",
            "time",
            "states",
            "switch_days",
            "regime",
            "seed",
            "fostac",
        ),
    )
    ted = _grab(
        te,
        (
            "measurements",
            "feed_noisy",
            "time",
            "states",
            "switch_days",
            "regime",
            "seed",
            "ukf_x_hat",
            "ukf_std",
            "fostac",
        ),
    )

    def _reg(d, i):
        return str(d["regime"][i]) if "regime" in d else ""

    def _seed(d, i):
        return int(d["seed"][i]) if "seed" in d else None

    pool = [
        Series(
            measurements=trd["measurements"][i],
            feed=trd["feed_noisy"][i],
            time=trd["time"][i],
            truth=trd["states"][i],
            switch_days=trd["switch_days"][i],
            label=_reg(trd, i),
            seed=_seed(trd, i),
            aux=(
                {"fostac": np.asarray(trd["fostac"][i], float)}
                if "fostac" in trd
                else {}
            ),
        )
        for i in range(len(trd["time"]))
    ]
    test = [
        Series(
            measurements=np.asarray(ted["measurements"][i], float),
            feed=np.asarray(ted["feed_noisy"][i], float),
            time=np.asarray(ted["time"][i], float),
            truth=np.asarray(ted["states"][i], float),
            switch_days=np.asarray(ted["switch_days"][i], float),
            label=_reg(ted, i),
            seed=_seed(ted, i),
            aux={
                k: np.asarray(ted[k][i], float)
                for k in ("ukf_x_hat", "ukf_std", "fostac")
                if k in ted
            },
        )
        for i in range(len(ted["time"]))
    ]
    return EstimatorDataset(name="benchmark", meta=meta, pool=pool, test=test)


#: name -> loader.  Extend with other datasets that yield the same Series structure.
LOADERS: dict[str, Callable[..., EstimatorDataset]] = {"benchmark": load_benchmark}


def load_dataset(name: str, path: str | Path | None = None) -> EstimatorDataset:
    if name not in LOADERS:
        raise KeyError(f"unknown dataset '{name}'; known: {sorted(LOADERS)}")
    return LOADERS[name](path)


def get_dataset(name_or_path: str | Path = "benchmark") -> EstimatorDataset:
    """Resolve a registered name (e.g. ``"benchmark"``) OR a path to a dataset dir."""
    p = Path(name_or_path)
    if p.exists() and p.is_dir():
        # infer format: a benchmark dir has train.npz + meta.json
        if (p / "train.npz").exists() and (p / "meta.json").exists():
            return load_benchmark(p)
        raise ValueError(f"cannot infer dataset format for directory {p}")
    return load_dataset(str(name_or_path))
