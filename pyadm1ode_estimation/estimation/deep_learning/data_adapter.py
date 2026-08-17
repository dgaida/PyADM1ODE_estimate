"""Dataset → PINN adapter: one wrapper, both variants, any dataset.

The filters already consume data through :mod:`..filter_tuning.datasets`: a
registry of loaders that all yield the same :class:`~..filter_tuning.datasets.Series`
structure (measurements, feed, time, truth, switch days, a stratification label).
This module puts the *neural* estimators on **that same layer**, so a new or
changed dataset format is absorbed once — in its loader — and the UKF, the
per-window smoother and the amortised observer all see it.

    from pyadm1ode_estimation.estimation.deep_learning import PinnData

    data = PinnData.build("benchmark", val_frac=0.2, seed=0)
    obs_ds = data.observer_dataset("train", window=336)   # variant B input
    inputs = data.smoother_inputs("val")                  # variant A input

What the wrapper guarantees, and why each matters:

* **The same split as the filters.** Both go through
  :meth:`~..filter_tuning.datasets.EstimatorDataset.split_indices`, stratified by
  the series label (the operating mode), grouped by series (never within one).
  Same ``(val_frac, seed)`` ⇒ same series, so a network and a filter are
  comparable. Freeze it with :meth:`PinnData.save_split` / ``split_file=``.
* **Train-only statistics.** Feature normalisation, the log-transform base
  ``x_ref``, the prior ``x_prior`` and the per-state scale are all estimated on
  the *training* series alone. Fitting them on val or test leaks.
* **Nothing hard-coded.** Sensor count, substrate count, state size, ``dt`` and
  the sensor noise are read from the data and its ``meta``. The ADM1da-specific
  parts (physics residual, ``h(x)``) are requested lazily and raise a clear error
  if the dataset's state vector is not the 41-slot ADM1da one — the purely
  data-driven paths keep working.

Adding a new dataset means registering a loader in
:data:`..filter_tuning.datasets.LOADERS`; nothing here changes. Changing an
existing format (extra sensors, different noise spec, a new field) is absorbed by
that loader plus, at most, a :class:`FeatureSpec` flag.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..filter_tuning.datasets import EstimatorDataset, Series, get_dataset
from .observation_torch import SUPPORTED_CHANNELS, TorchObservationModel
from .observer_data import MeasurementDataset, ObserverDataset

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyadm1.core.adm1_torch import Adm1TorchParams

#: State size of the ADM1da vector the physics / observation maps are written for.
ADM1DA_STATE_SIZE = 41

#: Length of the ADM1da influent vector ``s_in``.
_N_INFLUENT = 37

#: The three splits this wrapper emits.
SPLITS = ("train", "val", "test")

# Canonical differentiable channel name per normalised alias ("q_gas" -> "Q_gas",
# "ph" -> "pH"). Dataset ``meta["sensors"]`` entries are matched case- and
# separator-insensitively so a loader may spell them either way.
_CANONICAL_CHANNEL = {
    re.sub(r"[^a-z0-9]", "", c.lower()): c for c in SUPPORTED_CHANNELS
}

# "relative 3 % (literature biogas flow meter)" / "absolute 0.02 (Memosens)".
_NOISE_RE = re.compile(r"(relative|absolute)\s*([0-9.eE+-]+)\s*(%?)", re.IGNORECASE)


# --------------------------------------------------------------------------
# Sensor noise
# --------------------------------------------------------------------------
def parse_noise_spec(spec: Any) -> tuple[str, float]:
    """Normalise one ``meta["sensor_noise"]`` entry to ``(kind, value)``.

    Accepts the three forms a dataset may plausibly use:

    * a **string** ``"relative 3 %"`` / ``"absolute 0.02"`` (trailing prose is
      ignored). A ``%`` after a *relative* value divides by 100; after an
      *absolute* value it is only a unit marker (``"absolute 0.2 %"`` for a TS
      probe means 0.2 percentage points) and the value is kept as is.
    * a **mapping** ``{"kind": "relative", "value": 0.03}``.
    * a bare **number**, read as an absolute standard deviation.

    Returns:
        ``(kind, value)`` with ``kind`` in ``{"relative", "absolute"}``.

    Raises:
        ValueError: if the entry cannot be interpreted.
    """
    if isinstance(spec, (int, float, np.floating)):
        return "absolute", float(spec)
    if isinstance(spec, dict):
        kind = str(spec.get("kind", "absolute")).lower()
        if kind not in ("relative", "absolute"):
            raise ValueError(f"noise kind must be relative/absolute, got {kind!r}.")
        return kind, float(spec["value"])
    m = _NOISE_RE.search(str(spec))
    if m is None:
        raise ValueError(f"cannot parse sensor-noise spec {spec!r}.")
    kind, value, pct = m.group(1).lower(), float(m.group(2)), m.group(3)
    if kind == "relative" and pct:
        value /= 100.0
    return kind, value


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FeatureSpec:
    """Declares how one :class:`Series` becomes the network's input features.

    The two mandatory blocks are the sensors and the (known) substrate feed; the
    rest are opt-in channels that cost nothing when unused. Width and names are
    derived from the data, so a dataset with a different number of sensors or
    substrates needs no change here.

    Attributes:
        use_measurements: include the sensor columns.
        use_feed: include the substrate-feed columns (a known control input).
        deltas: append the first difference of every included column. Gives a
            causal network the local trend without having to integrate it out of
            the recurrent state.
        time_since_switch: append the time since the last feed change [days]. The
            headline benchmark metric is the *transient* error right after a
            switch, and this makes "how far into a transient am I" explicit.
        log_feed: pass ``log1p(feed)`` instead of the raw flows. Substrates differ
            by ~11x in COD per m³, so the raw flows are heavy-tailed.
        extra: ``f(series, ctx) -> (T, k)`` for dataset-specific channels; ``ctx``
            carries ``dt_hours`` and ``meta``. Name them via ``extra_names``.
        extra_names: names of the ``k`` columns ``extra`` returns.
    """

    use_measurements: bool = True
    use_feed: bool = True
    deltas: bool = False
    time_since_switch: bool = False
    log_feed: bool = False
    extra: Callable[[Series, dict], np.ndarray] | None = None
    extra_names: tuple[str, ...] = ()

    def build(self, s: Series, ctx: dict) -> tuple[np.ndarray, list[str]]:
        """Raw (un-normalised) features ``(T, n_feat)`` plus their names."""
        blocks: list[np.ndarray] = []
        names: list[str] = []
        base_n = 0  # width of the block the deltas are taken over

        if self.use_measurements:
            meas = np.asarray(s.measurements, dtype=float)
            blocks.append(meas)
            names += [str(c) for c in ctx["sensor_names"]]
            base_n += meas.shape[1]
        if self.use_feed:
            feed = np.asarray(s.feed, dtype=float)
            if self.log_feed:
                feed = np.log1p(np.clip(feed, 0.0, None))
            blocks.append(feed)
            names += [f"feed_{i}" for i in range(feed.shape[1])]
            base_n += feed.shape[1]
        if not blocks:
            raise ValueError(
                "FeatureSpec produces no columns; enable at least one block."
            )

        base = np.concatenate(blocks, axis=1)
        out, out_names = [base], list(names)

        if self.deltas:
            d = np.diff(base, axis=0, prepend=base[:1])
            out.append(d)
            out_names += [f"d_{n}" for n in names[:base_n]]
        if self.time_since_switch:
            out.append(_time_since_switch(s)[:, None])
            out_names.append("t_since_switch_d")
        if self.extra is not None:
            e = np.asarray(self.extra(s, ctx), dtype=float)
            if e.ndim == 1:
                e = e[:, None]
            if e.shape[0] != base.shape[0]:
                raise ValueError(
                    f"FeatureSpec.extra returned {e.shape[0]} rows, expected {base.shape[0]}."
                )
            if self.extra_names and len(self.extra_names) != e.shape[1]:
                raise ValueError(
                    f"extra_names has {len(self.extra_names)} entries but extra returned "
                    f"{e.shape[1]} columns."
                )
            out.append(e)
            out_names += list(self.extra_names) or [
                f"extra_{i}" for i in range(e.shape[1])
            ]

        return np.concatenate(out, axis=1), out_names


def _time_since_switch(s: Series) -> np.ndarray:
    """Days since the most recent feed switch (0 before the first one)."""
    t = np.asarray(s.time, dtype=float)
    out = t - t[0]
    for sd in np.atleast_1d(np.asarray(s.switch_days, dtype=float)):
        if not np.isfinite(sd):
            continue
        after = t >= sd
        out[after] = t[after] - sd
    return out


# --------------------------------------------------------------------------
# Variant A payload
# --------------------------------------------------------------------------
@dataclass
class SmootherInputs:
    """Everything :meth:`~.pinn_smoother.PinnSmoother.fit` needs for one series.

    Variant A has no weights shared across series — one fit per series — so this
    is the per-series call payload, not a training set. ``x_prior`` / ``x_scale``
    are the only quantities estimated from the training split.

    Attributes:
        name: series identifier (``"<label>#<index>"``).
        label: the operating mode / stratification label.
        obs_times: ``(T,)`` measurement times [days].
        obs_values: ``(T, n_channels)`` measured values (``NaN`` = missing).
        channel_names: the channels of ``obs_values``, in order.
        t0, t1: collocation window bounds [days]. ``t1`` past the last measurement
            turns the tail into a physics-driven forecast.
        x_prior: prior state at ``t0``, from the training split.
        x_scale: per-state magnitude used to normalise prior / physics residuals.
        truth: ``(T, n_state)`` ground truth — for scoring only, never for fitting.
        series: the underlying :class:`Series` (feed, switch days, ``aux``).
        params_at: ``t -> Adm1TorchParams`` following this series' actual feed;
            pass it to :class:`~.pinn_smoother.PinnSmoother` as ``params_at``.
            ``None`` when the dataset has no ADM1 physics.
    """

    name: str
    label: str
    obs_times: np.ndarray
    obs_values: np.ndarray
    channel_names: list[str]
    t0: float
    t1: float
    x_prior: np.ndarray
    x_scale: np.ndarray
    truth: np.ndarray
    series: Series
    params_at: Callable[[np.ndarray], Adm1TorchParams] | None = None

    def fit_kwargs(self, forecast_days: float = 0.0) -> dict:
        """Keyword arguments for ``PinnSmoother.fit`` (add ``epochs``/``lr`` yourself).

        Args:
            forecast_days: extend ``t1`` past the last measurement by this many
                days, so the data-free tail is carried by the ODE alone.
        """
        return {
            "obs_times": self.obs_times,
            "obs_values": self.obs_values,
            "t0": self.t0,
            "t1": self.t1 + float(forecast_days),
        }


# --------------------------------------------------------------------------
# Split record
# --------------------------------------------------------------------------
@dataclass
class SplitRecord:
    """A frozen, inspectable record of which series went where.

    Written next to the experiment so a run can be reproduced even if the
    dataset's loader or shuffling changes later; reload it with ``split_file=``.
    """

    dataset: str
    val_frac: float
    seed: int
    train: list[int]
    val: list[int]
    test: list[int]
    labels: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "dataset": self.dataset,
                "val_frac": self.val_frac,
                "seed": self.seed,
                "train": list(map(int, self.train)),
                "val": list(map(int, self.val)),
                "test": list(map(int, self.test)),
                "labels": self.labels,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> SplitRecord:
        d = json.loads(text)
        return cls(
            dataset=str(d.get("dataset", "")),
            val_frac=float(d.get("val_frac", float("nan"))),
            seed=int(d.get("seed", -1)),
            train=[int(i) for i in d["train"]],
            val=[int(i) for i in d["val"]],
            test=[int(i) for i in d["test"]],
            labels=d.get("labels", {}),
        )


# --------------------------------------------------------------------------
# The wrapper
# --------------------------------------------------------------------------
@dataclass
class PinnData:
    """A dataset prepared for both PINN variants; build it with :meth:`build`.

    Attributes:
        dataset: the underlying dataset-agnostic container.
        split: which series are train / val / test.
        feature_spec: how a series becomes input features.
        feature_names: names of the ``n_feat`` input columns.
        feat_mean, feat_std: feature normalisation, fitted on **train** only.
        x_ref: ``(n_state,)`` central train state — the observer's log-transform
            base. A **medoid**, i.e. a real observed state (see :func:`_medoid`).
        x_prior: ``(n_state,)`` central *initial* train state — the smoother's
            anchor at ``t0`` (the only legitimate prior for an unseen series).
            Also a medoid, so it is physically consistent.
        x_scale: ``(n_state,)`` per-state RMS over train — the residual scale.
            Componentwise, because it is a magnitude rather than a state.
        sensor_sigma: absolute 1-sigma per sensor channel, with the relative
            entries resolved against the train-split magnitude of that channel.
    """

    dataset: EstimatorDataset
    split: SplitRecord
    feature_spec: FeatureSpec
    feature_names: list[str]
    feat_mean: np.ndarray
    feat_std: np.ndarray
    x_ref: np.ndarray
    x_prior: np.ndarray
    x_scale: np.ndarray
    sensor_sigma: np.ndarray
    burnin_steps: int = 0
    _params: Adm1TorchParams | None = None
    _plant_cache: object | None = None
    _influent_cache: dict = field(default_factory=dict)

    # -- construction ---------------------------------------------------
    @classmethod
    def build(
        cls,
        dataset: str | Path | EstimatorDataset = "benchmark",
        *,
        val_frac: float = 0.2,
        seed: int = 0,
        per_group_train: int | None = None,
        per_group_val: int | None = None,
        per_group_test: int | None = None,
        feature_spec: FeatureSpec | None = None,
        split_file: str | Path | None = None,
        burnin_days: float = 0.0,
        params: Adm1TorchParams | None = None,
    ) -> PinnData:
        """Load a dataset, split it, and fit the train-only statistics.

        Args:
            dataset: a registered name (``"benchmark"``), a path to a dataset
                directory, or an already-loaded :class:`EstimatorDataset`.
            val_frac, seed, per_group_*: passed to
                :meth:`~..filter_tuning.datasets.EstimatorDataset.split_indices`,
                so the split matches a filter configured the same way.
            feature_spec: input-feature layout (default: sensors + feed).
            split_file: reuse a frozen :class:`SplitRecord` instead of
                re-splitting. The file wins over ``val_frac``/``seed``.
            burnin_days: leading days each consumer should ignore (the observer
                cannot know the initial state). Stored, not applied here.
            params: ADM1 parameter snapshot for the physics / observation maps.
                Defaults to a lazily-built snapshot of the example plant.

        Raises:
            ValueError: if a loaded ``split_file`` indexes outside the dataset.
        """
        ds = dataset if isinstance(dataset, EstimatorDataset) else get_dataset(dataset)
        spec = feature_spec or FeatureSpec()

        if split_file is not None and Path(split_file).exists():
            rec = SplitRecord.from_json(Path(split_file).read_text(encoding="utf-8"))
            n_pool, n_test = len(ds.pool), len(ds.test)
            bad = [i for i in rec.train + rec.val if not 0 <= i < n_pool]
            bad += [i for i in rec.test if not 0 <= i < n_test]
            if bad:
                raise ValueError(
                    f"split file {split_file} indexes series {sorted(set(bad))} outside "
                    f"dataset '{ds.name}' (pool={n_pool}, test={n_test})."
                )
        else:
            tr, va, te = ds.split_indices(
                val_frac, per_group_train, per_group_val, per_group_test, seed
            )
            rec = SplitRecord(
                dataset=ds.name, val_frac=val_frac, seed=seed, train=tr, val=va, test=te
            )
            rec.labels = {
                "train": _label_counts([ds.pool[i] for i in tr]),
                "val": _label_counts([ds.pool[i] for i in va]),
                "test": _label_counts([ds.test[i] for i in te]),
            }

        obj = cls(
            dataset=ds,
            split=rec,
            feature_spec=spec,
            feature_names=[],
            feat_mean=np.zeros(0),
            feat_std=np.ones(0),
            x_ref=np.zeros(0),
            x_prior=np.zeros(0),
            x_scale=np.ones(0),
            sensor_sigma=np.zeros(0),
            burnin_steps=round(burnin_days * 24.0 / ds.dt_hours),
            _params=params,
        )
        obj._fit_train_statistics()
        return obj

    def _fit_train_statistics(self) -> None:
        """Estimate every shared quantity on the **training** series only."""
        train = self.series("train")
        if not train:
            raise ValueError(
                "the training split is empty — check val_frac / per_group_*."
            )

        raw = [self.feature_spec.build(s, self._ctx())[0] for s in train]
        self.feature_names = self.feature_spec.build(train[0], self._ctx())[1]
        flat = np.concatenate(raw, axis=0)
        self.feat_mean = np.nanmean(flat, axis=0)
        std = np.nanstd(flat, axis=0)
        # A constant column (e.g. an unused substrate) would divide by ~0.
        self.feat_std = np.where(std > 1e-12, std, 1.0)

        truth = np.concatenate([np.asarray(s.truth, float) for s in train], axis=0)
        # A per-state magnitude, not a state: componentwise is what "scale" means.
        self.x_scale = np.sqrt(np.mean(truth**2, axis=0)) + 1e-12
        # x_ref and x_prior are *states* and are consumed as such (h(x), the gas
        # solve, the ODE). A componentwise median is NOT a state: taking each
        # component's median independently breaks the identities that tie them
        # together — on ADM1 the charge balance, which fixes pH. Measured on the
        # benchmark: the componentwise median implies pH 11.6 while every real
        # training state lies in 7.1-7.8, and the resulting prior starts the fit
        # ~200 sigma off on the pH channel. The medoid is the real training state
        # closest to that median, so every physical identity holds by construction.
        self.x_ref = _medoid(truth, self.x_scale)
        b = min(self.burnin_steps, min(len(s.truth) for s in train) - 1)
        self.x_prior = _medoid(
            np.stack([np.asarray(s.truth, float)[b] for s in train], axis=0),
            self.x_scale,
        )
        self.sensor_sigma = self._fit_sensor_sigma(train)

    def _fit_sensor_sigma(self, train: list[Series]) -> np.ndarray:
        """Absolute per-channel sigma; relative specs resolved on the train split."""
        noise = self.dataset.meta.get("sensor_noise", {})
        meas = np.concatenate(
            [np.asarray(s.measurements, float) for s in train], axis=0
        )
        mag = np.sqrt(np.nanmean(meas**2, axis=0))  # per-channel magnitude
        out = np.ones(meas.shape[1], dtype=float)
        for j, name in enumerate(self.sensor_names):
            spec = noise.get(name)
            if spec is None:
                # Unknown channel: a relative 3 % default keeps the data loss
                # finite and roughly balanced rather than silently weighting by 1.
                out[j] = 0.03 * mag[j]
                continue
            kind, value = parse_noise_spec(spec)
            out[j] = value * mag[j] if kind == "relative" else value
        return np.clip(out, 1e-12, None)

    def _ctx(self) -> dict:
        return {
            "dt_hours": self.dataset.dt_hours,
            "meta": self.dataset.meta,
            "sensor_names": self.sensor_names,
        }

    # -- introspection --------------------------------------------------
    @property
    def dt_hours(self) -> float:
        return float(self.dataset.dt_hours)

    @property
    def dt_days(self) -> float:
        return float(self.dataset.dt_hours) / 24.0

    @property
    def n_state(self) -> int:
        return int(self.dataset.pool[0].truth.shape[1])

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    @property
    def state_names(self) -> list[str]:
        return self.dataset.state_names

    @property
    def sensor_names(self) -> list[str]:
        """Sensor channels, from ``meta`` or positional if the meta omits them."""
        names = list(self.dataset.meta.get("sensors", []))
        n = int(self.dataset.pool[0].measurements.shape[1])
        if len(names) != n:
            names = [f"sensor_{i}" for i in range(n)]
        return names

    @property
    def channel_names(self) -> list[str]:
        """Sensor names mapped onto the differentiable ``h(x)`` channel names."""
        return [
            _CANONICAL_CHANNEL.get(re.sub(r"[^a-z0-9]", "", n.lower()), n)
            for n in self.sensor_names
        ]

    def series(self, split: str) -> list[Series]:
        """The :class:`Series` objects of one split, in split order."""
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}.")
        if split == "test":
            return [self.dataset.test[i] for i in self.split.test]
        idx = self.split.train if split == "train" else self.split.val
        return [self.dataset.pool[i] for i in idx]

    def names(self, split: str) -> list[str]:
        """Stable per-series identifiers, aligned with :meth:`series`."""
        suffix = "t" if split == "test" else ""
        idx = getattr(self.split, split)
        pool = self.dataset.test if split == "test" else self.dataset.pool
        return [f"{pool[i].label}#{suffix}{i}" for i in idx]

    def summary(self) -> str:
        """One-line-per-split overview — print it at the top of a run's log."""
        lines = [
            (
                f"dataset={self.dataset.name} states={self.n_state} "
                f"sensors={self.sensor_names} features={self.n_features} dt={self.dt_hours} h"
            )
        ]
        for sp in SPLITS:
            ser = self.series(sp)
            lines.append(f"  {sp:<5} n={len(ser):<4} {_label_counts(ser)}")
        return "\n".join(lines)

    def save_split(self, path: str | Path) -> Path:
        """Freeze the split to JSON so later runs reuse it verbatim."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.split.to_json(), encoding="utf-8")
        return p

    # -- features -------------------------------------------------------
    def features(self, split: str) -> np.ndarray:
        """Normalised features ``(N, T, n_feat)`` for a split (train statistics).

        Series of unequal length are not stacked — use :meth:`windows` for those.
        """
        raw = [self.feature_spec.build(s, self._ctx())[0] for s in self.series(split)]
        lens = {len(r) for r in raw}
        if len(lens) != 1:
            raise ValueError(
                f"split '{split}' has series of differing length {sorted(lens)}; "
                "use windows(...) to cut them to a common length."
            )
        return self._normalize(np.stack(raw, axis=0))

    def _normalize(self, raw: np.ndarray) -> np.ndarray:
        # NaN sensors cannot enter a recurrent input; map them to the normalised
        # mean (0). Losses still mask them on the target side.
        return np.nan_to_num((raw - self.feat_mean) / self.feat_std, nan=0.0)

    def windows(
        self,
        split: str,
        window: int | None = None,
        stride: int | None = None,
        *,
        n_random: int | None = None,
        seed: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Cut a split into equal-length windows.

        Training the recurrent observer on whole 60-day series wastes the only
        real defence against overfitting a ~80-series set; tiling into shorter
        windows multiplies the effective sample count and teaches it to lock on
        from an unknown start.

        Args:
            split: which split to cut.
            window: window length in steps (``None`` = whole series).
            stride: step between window starts (default: ``window``, i.e. no overlap).
                A stride below ``window`` overlaps them, which is the cheapest way
                to enlarge a small training set.
            n_random: instead of tiling, draw this many windows per series at
                random start points (overrides ``stride``). Use for the training
                split; keep the deterministic tiling for val/test so the score is
                reproducible.
            seed: RNG seed for ``n_random``.

        Returns:
            ``(features (N,W,n_feat), states (N,W,n_state), time (N,W), names)``.
        """
        if n_random is not None and n_random < 1:
            raise ValueError(f"n_random must be >= 1 or None, got {n_random}.")
        rng = np.random.default_rng(seed)
        feats, states, times, names = [], [], [], []
        for s, name in zip(self.series(split), self.names(split)):
            raw, _ = self.feature_spec.build(s, self._ctx())
            truth = np.asarray(s.truth, dtype=float)
            t = np.asarray(s.time, dtype=float)
            w = len(raw) if window is None else int(window)
            if w > len(raw):
                raise ValueError(
                    f"window {w} exceeds series length {len(raw)} ({name})."
                )
            if n_random is not None:
                starts = rng.integers(0, len(raw) - w + 1, size=int(n_random))
            else:
                st = w if stride is None else int(stride)
                starts = range(0, len(raw) - w + 1, st)
            for k, start in enumerate(starts):
                sl = slice(int(start), int(start) + w)
                feats.append(raw[sl])
                states.append(truth[sl])
                times.append(t[sl])
                names.append(f"{name}@{k}")
        if not feats:
            raise ValueError(f"split '{split}' produced no windows.")
        return (
            self._normalize(np.stack(feats, axis=0)),
            np.stack(states, axis=0),
            np.stack(times, axis=0),
            names,
        )

    # -- variant B: amortised observer ----------------------------------
    def observer_dataset(
        self,
        split: str = "train",
        *,
        window: int | None = None,
        stride: int | None = None,
        n_random: int | None = None,
        seed: int = 0,
    ) -> ObserverDataset:
        """Supervised training data for :class:`~.observer.Adm1Observer`.

        Feeds :func:`~.observer_train.pretrain_observer`. The normalisation and
        ``x_ref`` come from the training split, so the same call on ``"val"`` /
        ``"test"`` yields consistently scaled inputs.

        The supervised objective is pure state regression and needs no physics, so
        for a non-ADM1da dataset ``params`` is left ``None`` and only a net that
        actually evaluates the ADM1 maps will complain.
        """
        feats, states, times, _ = self.windows(
            split, window, stride, n_random=n_random, seed=seed
        )
        n_ch = len(self.sensor_names)
        # Raw measurements for the self-supervised objective: undo the
        # normalisation on the sensor block rather than re-slicing the series,
        # so any FeatureSpec that reorders or transforms stays consistent.
        meas = None
        if self.feature_spec.use_measurements and not self.feature_spec.deltas:
            meas = feats[..., :n_ch] * self.feat_std[:n_ch] + self.feat_mean[:n_ch]
        return ObserverDataset(
            features=feats,
            states=states,
            time=times[0],
            feature_names=list(self.feature_names),
            feat_mean=self.feat_mean.copy(),
            feat_std=self.feat_std.copy(),
            x_ref=self.x_ref.copy(),
            params=self._optional_params(),
            measurements=meas,
            channel_names=list(self.channel_names) if meas is not None else None,
        )

    def measurement_dataset(
        self,
        split: str = "test",
        *,
        window: int | None = None,
        stride: int | None = None,
    ) -> MeasurementDataset:
        """Measurement-only windows for the self-supervised objective.

        Drives :func:`~.observer_train.pretrain_observer_selfsup` and the online
        :func:`~.observer_train.finetune_observer`. Ground truth is deliberately
        dropped: this is the objective that also works on a real plant, and on the
        test split it is legitimate — it never sees a label.

        Unlike :meth:`observer_dataset` this *does* require the physics: its whole
        objective is the ADM1 residual plus ``h(x)``.
        """
        feats, _, times, _ = self.windows(split, window, stride)
        n_ch = len(self.sensor_names)
        raw_meas = feats[..., :n_ch] * self.feat_std[:n_ch] + self.feat_mean[:n_ch]
        return MeasurementDataset(
            features=feats,
            measurements=raw_meas,
            dt_days=self.dt_days,
            channel_names=list(self.channel_names),
            feat_mean=self.feat_mean.copy(),
            feat_std=self.feat_std.copy(),
            x_ref=self.x_ref.copy(),
            params=self.physics_params(),
            time=times[0],
        )

    # -- variant A: per-window smoother ---------------------------------
    def smoother_inputs(
        self,
        split: str = "val",
        *,
        days: float | None = None,
        feed_matched: bool = True,
    ) -> list[SmootherInputs]:
        """Per-series call payloads for :class:`~.pinn_smoother.PinnSmoother`.

        Each payload carries a ``params_at`` closure over that series' own feed,
        so the physics residual is evaluated against the plant that produced the
        data rather than the reference plant's nominal operating point.

        Args:
            split: which split to emit.
            days: truncate each series to this many days (``None`` = full length).
                Variant A costs minutes per series, so a short window is the
                practical unit for a hyperparameter search.
            feed_matched: build the ``params_at`` closure (default). Set ``False``
                for the ablation that shows what the nominal-feed physics costs.
        """
        matched = feed_matched and self.n_state == ADM1DA_STATE_SIZE
        out: list[SmootherInputs] = []
        for s, name in zip(self.series(split), self.names(split)):
            t = np.asarray(s.time, dtype=float)
            n = (
                len(t)
                if days is None
                else min(len(t), round(days * 24.0 / self.dt_hours) + 1)
            )
            out.append(
                SmootherInputs(
                    name=name,
                    label=s.label,
                    obs_times=t[:n],
                    obs_values=np.asarray(s.measurements, dtype=float)[:n],
                    channel_names=list(self.channel_names),
                    t0=float(t[0]),
                    t1=float(t[n - 1]),
                    x_prior=self.x_prior.copy(),
                    x_scale=self.x_scale.copy(),
                    truth=np.asarray(s.truth, dtype=float)[:n],
                    series=s,
                    params_at=self.feed_params_at(s) if matched else None,
                )
            )
        return out

    # -- physics / observation ------------------------------------------
    def _optional_params(self) -> Adm1TorchParams | None:
        """Physics parameters if this dataset can have them, else ``None``."""
        if self._params is None and self.n_state != ADM1DA_STATE_SIZE:
            return None
        return self.physics_params()

    def _plant(self):
        """The (warmed) reference plant behind the physics — built once.

        A freshly built plant has never been fed: ``adm1._Q`` and
        ``adm1._state_input`` are ``None``, so a parameter snapshot taken from it
        carries ``q_ad = 0`` and ``s_in = 0`` — the ODE of a *closed batch
        reactor*. Warming it first gives a real operating point; the per-window
        feed is then applied on top by :meth:`feed_params`.
        """
        if self._plant_cache is None:
            from ...example_plants import build_multi_stage_plant

            plant = build_multi_stage_plant()
            warmup = float(self.dataset.meta.get("warmup_days", 30.0))
            if warmup > 0:
                plant.simulate(duration=warmup, dt=1.0, save_interval=warmup)
            self._plant_cache = plant
        return self._plant_cache

    def physics_params(self) -> Adm1TorchParams:
        """The ADM1 parameter snapshot for the physics residual and ``h(x)``.

        Built lazily from the **warmed** example plant unless one was injected at
        :meth:`build` — pass your own (e.g. from a calibration artifact) when the
        dataset is not the shipped example plant.

        This carries the plant's *nominal* feed. For a fit on a specific series
        use :meth:`feed_params`, which replaces the feed by the one that series
        actually ran: the influent term ``D_in·s_in`` dominates the ADM1 right-hand
        side, so a nominal feed against another series' data is a large, silent
        model error.

        Raises:
            ValueError: if the dataset's state vector is not the 41-slot ADM1da
                one, which the differentiable maps are written for.
        """
        if self._params is None:
            self._require_adm1da("the ADM1 physics parameters")
            from pyadm1.core.adm1_torch import Adm1TorchParams

            p = Adm1TorchParams.from_adm1(self._plant().components["primary"].adm1)
            # Validate before caching, so a raise cannot leave a poisoned snapshot
            # that a retry would then hand out silently.
            if float(p.q_ad) <= 0.0:
                raise ValueError(
                    "the reference plant reports q_ad = 0 (never fed) — the physics "
                    "residual would model a closed batch reactor. Inject a fed "
                    "parameter snapshot via PinnData.build(params=...)."
                )
            self._params = p
        return self._params

    def influent(self, flows: np.ndarray) -> np.ndarray:
        """The 37-slot ADM1 influent ``s_in`` for a substrate flow vector.

        Delegates to the plant's feedstock, which is the authority: mixing is
        **not** linear in the flows (blending the per-substrate vectors from
        ``meta`` instead is off by up to ~5 % on the dominant particulate slots
        ``X_PS_ch`` / ``X_PS_pr`` / ``X_PF_ch``). Concentrations are invariant to
        the overall scale, so only the flow *ratios* matter here; the magnitude
        enters through ``q_ad``.

        Args:
            flows: ``(n_sub,)`` volumetric flows [m³/d] in the dataset's order.
        """
        key = tuple(np.round(np.asarray(flows, dtype=float), 6))
        hit = self._influent_cache.get(key)
        if hit is None:
            fs = self._plant().components["primary"].adm1._feedstock
            df = fs.get_influent_dataframe(list(key))
            hit = np.asarray(df.iloc[0].tolist()[:_N_INFLUENT], dtype=float)
            self._influent_cache[key] = hit
        return hit

    def feed_params(
        self, flows: np.ndarray, *, base: Adm1TorchParams | None = None
    ) -> Adm1TorchParams:
        """Physics parameters carrying an actual feed instead of the nominal one.

        Args:
            flows: either ``(n_sub,)`` for a constant feed — the result then holds
                scalar ``q_ad`` / ``s_in`` — or ``(K, n_sub)`` for a feed that
                varies across ``K`` evaluation points, which yields ``q_ad`` and
                each ``s_in`` slot as length-``K`` tensors. ``adm1da_rhs_torch``
                broadcasts those across its batch, so a single call then evaluates
                every point under its own feed.
            base: parameters to copy the non-feed entries from (default:
                :meth:`physics_params`).

        Returns:
            A parameter snapshot whose ``q_ad`` / ``s_in`` match ``flows``.
        """
        import dataclasses

        import torch

        p = base if base is not None else self.physics_params()
        f = np.asarray(flows, dtype=float)
        if f.ndim == 1:
            return dataclasses.replace(
                p, q_ad=float(f.sum()), s_in=[float(v) for v in self.influent(f)]
            )
        if f.ndim != 2:
            raise ValueError(f"flows must be (n_sub,) or (K, n_sub), got {f.shape}.")
        s = np.stack([self.influent(row) for row in f], axis=0)  # (K, 37)
        dt = torch.float64
        return dataclasses.replace(
            p,
            q_ad=torch.tensor(f.sum(axis=1), dtype=dt),
            s_in=[torch.tensor(s[:, j], dtype=dt) for j in range(s.shape[1])],
        )

    def feed_params_at(
        self, series: Series, *, base: Adm1TorchParams | None = None
    ) -> Callable[[np.ndarray], Adm1TorchParams]:
        """A ``t -> params`` callable following one series' feed over time.

        Hand this to :class:`~.pinn_smoother.PinnSmoother` as ``params_at`` so the
        physics residual at each collocation time uses the feed that was actually
        dosed then. Without it a window spanning a feed switch is fitted against
        the wrong influent — and the 48 h after a switch is exactly what the
        benchmark scores.

        The feed is held piecewise-constant (zero-order hold, as dosed), matching
        how the series records it.
        """
        t_feed = np.asarray(series.time, dtype=float)
        flows = np.asarray(series.feed, dtype=float)

        def at(t_query: np.ndarray) -> Adm1TorchParams:
            idx = np.clip(
                np.searchsorted(t_feed, np.asarray(t_query, float), side="right") - 1,
                0,
                len(t_feed) - 1,
            )
            return self.feed_params(flows[idx], base=base)

        return at

    def obs_model(
        self, *, quasi_steady_gas: bool = True, soft_gas: bool = False
    ) -> TorchObservationModel:
        """The differentiable ``h(x)`` for this dataset's sensors.

        Channel noise is the fitted :attr:`sensor_sigma` (relative specs resolved
        against the train magnitude), so the data loss ``((h(x)-y)/σ)²`` weights
        the channels as the dataset actually defines them.

        Args:
            quasi_steady_gas: solve the gas channels from the liquid state instead
                of a free pTOTAL. Recommended — the direct map is the knife-edge
                cancellation that collapses ``Q_gas`` to zero during training.

        Raises:
            ValueError: if a sensor has no differentiable map, or the state vector
                is not ADM1da.
        """
        self._require_adm1da("the differentiable observation model")
        unknown = [n for n in self.channel_names if n not in SUPPORTED_CHANNELS]
        if unknown:
            raise ValueError(
                f"no differentiable map for sensor(s) {unknown}; supported: "
                f"{list(SUPPORTED_CHANNELS)}. Drop them from the dataset's sensors "
                f"or add a map in observation_torch._CHANNEL_FNS."
            )
        return TorchObservationModel(
            channel_names=list(self.channel_names),
            noise_std=[float(v) for v in self.sensor_sigma],
            params=self.physics_params(),
            soft_gas=soft_gas,
            quasi_steady_gas=quasi_steady_gas,
        )

    def _require_adm1da(self, what: str) -> None:
        if self.n_state != ADM1DA_STATE_SIZE:
            raise ValueError(
                f"{what} needs the {ADM1DA_STATE_SIZE}-state ADM1da vector, but dataset "
                f"'{self.dataset.name}' has {self.n_state} states. The data-driven paths "
                f"(features, windows, observer_dataset) still work."
            )

    # -- scoring bridge --------------------------------------------------
    def scoring_series(self, split: str = "test") -> list[dict]:
        """Series as the benchmark's ``scoring.py`` expects them.

        Keeps this module independent of the benchmark directory: it returns the
        plain dicts (``states``, ``time``, ``switch_days``, and ``ukf_x_hat`` /
        ``ukf_std`` when the loader carried them), so any scorer with that
        contract can consume a split — including val, once you have generated
        UKF references for it.
        """
        out = []
        for s in self.series(split):
            d = {
                "states": np.asarray(s.truth, dtype=float),
                "time": np.asarray(s.time, dtype=float),
                "switch_days": np.asarray(s.switch_days, dtype=float),
                "measurements": np.asarray(s.measurements, dtype=float),
                "regime": s.label,
            }
            for k in ("ukf_x_hat", "ukf_std"):
                if k in (s.aux or {}):
                    d[k] = np.asarray(s.aux[k], dtype=float)
            out.append(d)
        return out


def _medoid(
    states: np.ndarray, scale: np.ndarray, max_samples: int = 20000
) -> np.ndarray:
    """The row of ``states`` closest to their componentwise median.

    Returns an **actual observed state**, so anything the physics ties together
    (charge balance → pH, gas equilibrium) still holds — unlike the median itself,
    which is a per-component statistic and generally not a reachable state.

    Args:
        states: ``(N, n_state)`` candidate states.
        scale: ``(n_state,)`` per-state magnitude, so the distance is not
            dominated by the largest-magnitude states.
        max_samples: subsample above this many rows (the medoid is a robust
            central pick; it does not need every row).
    """
    s = np.asarray(states, dtype=float)
    if s.ndim != 2 or s.shape[0] == 0:
        raise ValueError(
            f"states must be a non-empty (N, n_state) array, got {s.shape}."
        )
    if s.shape[0] > max_samples:
        idx = np.linspace(0, s.shape[0] - 1, max_samples).astype(int)
        s = s[idx]
    med = np.median(s, axis=0)
    d = np.linalg.norm((s - med) / np.clip(scale, 1e-12, None), axis=1)
    return s[int(np.argmin(d))].copy()


def _label_counts(series: Sequence[Series]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for s in series:
        counts[s.label] = counts.get(s.label, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "ADM1DA_STATE_SIZE",
    "FeatureSpec",
    "PinnData",
    "SmootherInputs",
    "SplitRecord",
    "parse_noise_spec",
]
