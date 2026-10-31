"""One entry point for every estimator: build it by name, run it on a series, score it.

The three estimator families (recursive filter, collocation PINN, amortised
observer) live in their own subpackages with their own training code. This
module gives them one face so an experiment can swap them without touching the
data layer::

    from pyadm1ode_estimation.estimation.estimators import build_estimator, score
    from pyadm1ode_estimation.estimation.filter_tuning import get_dataset

    ds = get_dataset("benchmark")
    est = build_estimator("ukf", ds)          # "ukf" | "pinn" | "observer"
    traj = est.run(ds.test[0])                # TrajectoryEstimate: time, x_hat, std
    score(traj, ds.test[0], burnin_steps=48)  # median-over-states NRMSE

=============  ===============================================================
``ukf``        41-state SR-UKF with the CMA-ES reference ``theta`` (or A+D core)

``pinn``       PINN v2, causal rolling-window training (:mod:`.pinn.rolling`)
``observer``   PINN B, the GRU observer (:mod:`.observer`); ``checkpoint=...``
=============  ===============================================================

Every estimator consumes the shared :class:`~.filter_tuning.datasets.Series`
and returns a :class:`~.base.TrajectoryEstimate`, so the benchmark metric and
the fusion code see one currency. The CLI runs one estimator on a few series::

    python -m pyadm1ode_estimation.estimation.estimators ukf --series 0 1 --out run.npz
"""

from __future__ import annotations

import argparse
import json
import time as _time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .base import TrajectoryEstimate
from .filter_tuning.datasets import (
    EstimatorDataset,
    Series,
    get_dataset,
)
from .filter_tuning.filter_runners import (
    UkfRunner,
    collect_parallel,
    make_ukf_runner,
)

ESTIMATOR_NAMES: tuple[str, ...] = ("ukf", "pinn", "observer")

#: Shipped model parameters: the CMA-ES winners of the reference filters
#: (``theta`` per variant) and the trained observer.
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
#: Default checkpoint location of the trained observer.
DEFAULT_OBSERVER_CHECKPOINT = REFERENCE_DIR / "observer.pt"

#: Steps dropped before scoring (48 h): the estimators cannot know the initial state.
DEFAULT_BURNIN_STEPS = 48


@runtime_checkable
class Estimator(Protocol):
    """What :func:`build_estimator` returns."""

    name: str

    def run(self, series: Series, index: int = 0) -> TrajectoryEstimate:
        """Estimate the whole series (causally where the method is causal)."""


# ---------------------------------------------------------------------------
# Metric
# ---------------------------------------------------------------------------
def score(
    traj: TrajectoryEstimate, series: Series, burnin_steps: int = DEFAULT_BURNIN_STEPS
) -> float:
    """Median-over-states NRMSE after the burn-in (the benchmark headline number)."""
    p = np.asarray(traj.x_hat, float)[burnin_steps:]
    t = np.asarray(series.truth, float)[burnin_steps:]
    rmse = np.sqrt(np.mean((p - t) ** 2, axis=0))
    return float(np.median(rmse / (np.sqrt(np.mean(t**2, axis=0)) + 1e-12)))


# ---------------------------------------------------------------------------
# UKF
# ---------------------------------------------------------------------------
def reference_theta(variant: str = "full") -> tuple[dict, float | None]:
    """``(theta, fostac_every_days)`` of the CMA-ES reference run, or ``({}, None)``."""
    path = REFERENCE_DIR / f"cmaes_{variant}_winner.json"
    if not path.exists():
        warnings.warn(
            f"no reference theta at {path}; using the untuned filter", stacklevel=2
        )
        return {}, None
    winner = json.loads(path.read_text(encoding="utf-8"))
    return dict(winner.get("theta", {})), winner.get("fostac_every_days")


@dataclass
class UkfEstimator:
    """The SR-UKF of :mod:`.filter_tuning.filter_runners` as an :class:`Estimator`.

    Args:
        dataset: Supplies ``meta`` (plant, substrates, sensors, burn-in).
        variant: ``"full"`` (41 states) or ``"adcore"`` (acid + digester core).
        theta: Q/R/P0 parameters; default: the CMA-ES reference of ``variant``.
        fostac_every_days: Lab titration interval (``None`` = no FOS/TAC).
        prune: Simulate only the primary digester and its gas storage, the parts
            the sensors depend on (about 12x faster, same primary trajectories).
        name: Registry name.
    """

    dataset: EstimatorDataset
    variant: str = "full"
    theta: dict | None = None
    fostac_every_days: float | None = None
    prune: bool = False
    name: str = "ukf"
    _runner: UkfRunner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.theta is None:
            self.theta, fe = reference_theta(self.variant)
            if self.fostac_every_days is None:
                self.fostac_every_days = fe
        self._runner = make_ukf_runner(
            self.dataset.meta,
            self.variant,
            fostac_every_days=self.fostac_every_days,
            prune=self.prune,
        )

    @property
    def runner(self) -> UkfRunner:
        return self._runner

    def episode(self, series: Series, index: int = 0, kinetics: dict | None = None):
        """The filter episode of ``series``, carrying the plant ``kinetics`` if given."""
        ep = self.dataset.to_episode(
            series, burnin_days=0.0, name=f"{series.label}#{index}"
        )
        if kinetics:
            ep.obs["kinetics"] = dict(kinetics)
        return ep

    def run(
        self, series: Series, index: int = 0, kinetics: dict[str, float] | None = None
    ) -> TrajectoryEstimate:
        """Filter one series; ``kinetics`` (``key -> factor``) replaces the nominal constants."""
        ep = self.episode(series, index, kinetics)
        xh, sd = self._runner(self.theta, ep)
        return TrajectoryEstimate(time=np.asarray(series.time, float), x_hat=xh, std=sd)

    def run_many(
        self, series: list[Series], jobs: int = 1
    ) -> list[TrajectoryEstimate | None]:
        """All series, ``jobs`` processes; ``None`` where the filter diverged."""
        eps = [
            self.dataset.to_episode(s, burnin_days=0.0, name=f"{s.label}#{i}")
            for i, s in enumerate(series)
        ]
        out = collect_parallel(self._runner, eps, self.theta, jobs=jobs)
        return [
            (
                None
                if r is None
                else TrajectoryEstimate(np.asarray(s.time, float), r[1], r[2])
            )
            for s, r in zip(series, out)
        ]


# ---------------------------------------------------------------------------
# PINN v2 (causal rolling window)
# ---------------------------------------------------------------------------
@dataclass
class PinnEstimator:
    """PINN v2 as a causal estimator: :func:`~.pinn.rolling.run_rolling` per series.

    The defaults are the study's recommendation: the best structure (depth 4,
    width 128, four Fourier time features; measurements only, train-split
    prior), retrained daily with 200 warm-start steps on a 10-day window, the
    last estimate held between retrainings. The output normalisation and the
    residual scale are fitted on the *benchmark* training split, so a variant
    dataset is estimated with benchmark priors.

    Args:
        bridge: Which chain trajectory to return: ``"hold"`` (default; the last
            estimate between retrainings), ``"adm1"`` (nominal ADM1
            continuation, 0.8 points worse in the study) or a lag key such as
            ``"lag1"`` (fixed-lag smoother, one day late).
        split: Split tag written into the :class:`~.pinn.data.BenchmarkSeries`.
    """

    dataset: EstimatorDataset
    arch: object | None = None
    cfg: object | None = None
    roll: object | None = None
    bridge: str = "hold"
    split: str = "test"
    verbose: bool = False
    name: str = "pinn"
    _prior: tuple = field(init=False, repr=False)
    _scale_ref: tuple = field(init=False, repr=False)

    def __post_init__(self) -> None:
        from .pinn.data import train_split_log_stats, train_split_scale_reference
        from .pinn.model import ArchConfig
        from .pinn.rolling import RollingConfig
        from .pinn.train import TrainConfig

        self.arch = self.arch or ArchConfig(fourier_features=4)
        self.cfg = self.cfg or TrainConfig(
            t_obs_days=60.0,
            horizon_days=0.0,
            closure=True,
            data_source="measurements",
            prior="train_split",
            use_true_feed=False,
            use_series_kinetics=False,
        )
        self.roll = self.roll or RollingConfig(window_days=10.0)
        self._prior = train_split_log_stats()
        self._scale_ref = train_split_scale_reference()

    def run(
        self, series: Series, index: int = 0, kinetics: dict[str, float] | None = None
    ) -> TrajectoryEstimate:
        """Causal PINN chain on one series.

        ``kinetics`` (``key -> factor``, keys not given are nominal) replaces the
        nominal constants in the physics residual. Without it the physics is the
        nominal model; the generator truth on the series is never used.
        """
        import dataclasses

        from .pinn.data import BenchmarkSeries
        from .pinn.rolling import run_rolling

        bs = BenchmarkSeries.from_series(series, index, self.split)
        cfg = self.cfg
        if kinetics:
            factors = np.array([float(kinetics.get(k, 1.0)) for k in bs.kinetic_keys])
            bs = dataclasses.replace(bs, kinetic_factors=factors)
            cfg = dataclasses.replace(cfg, use_series_kinetics=True)
        elif cfg.use_series_kinetics:
            raise ValueError(
                "use_series_kinetics would read the generator truth; pass kinetics"
            )
        roll = self.roll
        if self.bridge != "adm1" and getattr(roll, "adm1_bridge", True):
            roll = dataclasses.replace(
                roll, adm1_bridge=False
            )  # not needed, saves time
        res = run_rolling(
            bs,
            self.arch,
            cfg,
            roll,
            meta=self.dataset.meta,
            prior_stats=self._prior,
            scale_ref=self._scale_ref,
            verbose=self.verbose,
        )
        traj = np.asarray(res.trajectories[self.bridge], float)
        return TrajectoryEstimate(
            np.asarray(series.time, float), traj, np.zeros_like(traj)
        )


# ---------------------------------------------------------------------------
# PINN B (GRU observer)
# ---------------------------------------------------------------------------
@dataclass
class ObserverEstimator:
    """PINN B: a trained :class:`~.observer.model.Adm1Observer` on one series.

    Carries the feature layout and normalisation of the training data, so the
    same object scores another dataset of the same plant with the benchmark
    statistics.
    Build with :meth:`train` (from a :class:`~.observer.adapter.PinnData`) or
    :meth:`load`; persist with :meth:`save`.
    """

    net: object
    feature_spec: object
    feat_mean: np.ndarray
    feat_std: np.ndarray
    ctx: dict
    recipe: dict = field(default_factory=dict)
    name: str = "observer"

    def for_dataset(self, dataset: EstimatorDataset) -> ObserverEstimator:
        """Same network and normalisation, feature context of ``dataset``.

        Only the meta-dependent features (``FeatureSpec.organic_load``) change:
        they are computed with the target plant's substrate characterisation.
        """
        import dataclasses

        return dataclasses.replace(self, ctx={**self.ctx, "meta": dataset.meta})

    @property
    def kinetic_keys(self) -> tuple[str, ...]:
        """Kinetic constants the network takes as inputs (empty: kinetics-blind)."""
        return tuple(getattr(self.feature_spec, "kinetic_keys", ()) or ())

    def features(self, series: Series) -> np.ndarray:
        raw, _ = self.feature_spec.build(series, self.ctx)
        z = (np.asarray(raw, float) - self.feat_mean) / self.feat_std
        return np.nan_to_num(z, nan=0.0)

    def run(
        self, series: Series, index: int = 0, kinetics: dict[str, float] | None = None
    ) -> TrajectoryEstimate:
        """Estimate one series.

        Args:
            series: The series to estimate.
            index: Position in its split (labels only).
            kinetics: ``key -> factor`` relative to nominal, for a
                kinetics-aware network (``FeatureSpec.kinetic_keys``); keys it
                lacks count as nominal, ``{}`` means "nominal kinetics". Required
                for such a network, so the generator truth a benchmark series
                carries can never slip in. Ignored by a kinetics-blind network.
        """
        import dataclasses

        import torch

        if self.kinetic_keys:
            if kinetics is None:
                raise ValueError(
                    "this observer takes kinetic inputs; pass kinetics={...} "
                    "(an empty dict means nominal kinetics)"
                )
            series = dataclasses.replace(
                series, aux={**series.aux, "kinetics": dict(kinetics)}
            )
        self.net.eval()
        with torch.no_grad():
            u = torch.tensor(self.features(series)[None], dtype=self.net._dtype)
            x = self.net(u).cpu().numpy()[0]
        return TrajectoryEstimate(np.asarray(series.time, float), x, np.zeros_like(x))

    # -- training -------------------------------------------------------
    @classmethod
    def train(
        cls,
        data,
        *,
        window: int = 336,
        stride: int | None = None,
        n_random: int | None = None,
        epochs: int = 200,
        patience: int | None = 40,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
        lr: float = 1.0e-3,
        batch_size: int = 32,
        burnin: int = DEFAULT_BURNIN_STEPS,
        noise_aug: float = 0.0,
        kinetic_noise: float = 0.0,
        weight_decay: float = 0.0,
        seed: int = 0,
        verbose: bool = False,
    ):
        """Train the observer on ``data`` (a :class:`~.observer.adapter.PinnData`).

        The defaults are the published PINN B recipe (window 336 h without
        overlap, 200 epochs, patience 40, hidden 64 x 2, burn-in 48 h) plus
        dropout 0.1 between the GRU layers, the only change of the generalisation
        sweep that improved all three test sets (``dropout=0.0`` reproduces the
        published network). ``noise_aug`` resamples sensor noise of ``noise_aug``
        x the dataset's sensor sigma onto the inputs every batch; ``n_random``
        draws that many random windows per series instead of tiling.
        ``kinetic_noise`` (log units) perturbs the kinetic input columns of a
        kinetics-aware feature layout by one draw per training sequence, so the
        network learns to use kinetics that are only known to calibration
        accuracy.

        Returns:
            ``(estimator, PretrainResult)``.
        """
        import torch

        from .observer import Adm1Observer, pretrain_observer

        torch.manual_seed(seed)
        tr = data.observer_dataset(
            "train", window=window, stride=stride, n_random=n_random, seed=seed
        )
        va = data.observer_dataset("val", window=window, stride=stride)
        net = Adm1Observer(
            tr.params,
            tr.x_ref,
            n_features=tr.features.shape[-1],
            hidden=hidden,
            num_layers=layers,
            dropout=dropout,
        )
        noise_std = None
        if noise_aug > 0.0:
            noise_std = (np.asarray(data.sensor_sigma, float) * noise_aug).tolist()
        seq_noise = None
        if kinetic_noise > 0.0:
            kin_cols = [
                j for j, n in enumerate(data.feature_names) if n.startswith("logk_")
            ]
            if not kin_cols:
                raise ValueError(
                    "kinetic_noise needs a feature layout with kinetic_keys"
                )
            seq_noise = np.zeros(len(data.feature_names))
            seq_noise[kin_cols] = (
                kinetic_noise / np.asarray(data.feat_std, float)[kin_cols]
            )
        res = pretrain_observer(
            net,
            tr,
            val_dataset=va,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            burnin=burnin,
            noise_std=noise_std,
            sequence_noise_std=seq_noise,
            weight_decay=weight_decay,
            restore_best=True,
            patience=patience,
            seed=seed,
            verbose=verbose,
        )
        recipe = {
            "window": window,
            "stride": stride,
            "n_random": n_random,
            "epochs": epochs,
            "patience": patience,
            "hidden": hidden,
            "layers": layers,
            "dropout": dropout,
            "lr": lr,
            "batch_size": batch_size,
            "burnin": burnin,
            "noise_aug": noise_aug,
            "kinetic_noise": kinetic_noise,
            "kinetic_keys": list(getattr(data.feature_spec, "kinetic_keys", ())),
            "weight_decay": weight_decay,
            "seed": seed,
            "best_epoch": int(res.best_epoch),
            "best_val": float(res.best_val),
            "split": data.split.to_json(),
        }
        est = cls(
            net=net,
            feature_spec=data.feature_spec,
            feat_mean=np.asarray(data.feat_mean, float),
            feat_std=np.asarray(data.feat_std, float),
            ctx=data._ctx(),
            recipe=recipe,
        )
        return est, res

    # -- persistence ----------------------------------------------------
    def save(self, path: str | Path) -> Path:
        import torch

        spec = {
            k: v
            for k, v in asdict(self.feature_spec).items()
            if k not in ("extra", "extra_names")
        }
        if getattr(self.feature_spec, "extra", None) is not None:
            raise ValueError(
                "a FeatureSpec with a custom 'extra' callable cannot be saved"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "params": self.net.params,
                "x_ref": (self.net._base.detach().cpu().numpy().tolist() + [0.0] * 4),
                "hidden": int(self.net.gru.hidden_size),
                "layers": int(self.net.gru.num_layers),
                "dropout": float(self.net.dropout_p),
                "n_features": int(self.net.gru.input_size),
                "feature_spec": spec,
                "feat_mean": self.feat_mean.tolist(),
                "feat_std": self.feat_std.tolist(),
                "ctx": {k: v for k, v in self.ctx.items() if k != "meta"},
                "recipe": self.recipe,
            },
            path,
        )
        return path

    @classmethod
    def load(
        cls, path: str | Path, dataset: EstimatorDataset | None = None
    ) -> ObserverEstimator:
        """Rebuild from :meth:`save`; ``dataset`` supplies the ``meta`` of the feature context."""
        import torch

        from .observer import Adm1Observer, FeatureSpec

        blob = torch.load(Path(path), map_location="cpu", weights_only=False)
        net = Adm1Observer(
            blob["params"],
            np.asarray(blob["x_ref"], float),
            n_features=blob["n_features"],
            hidden=blob["hidden"],
            num_layers=blob["layers"],
            dropout=blob["dropout"],
        )
        net.load_state_dict(blob["state_dict"])
        net.eval()
        ctx = dict(blob["ctx"])
        ctx["meta"] = dataset.meta if dataset is not None else {}
        return cls(
            net=net,
            feature_spec=FeatureSpec(
                **{
                    **blob["feature_spec"],
                    "kinetic_keys": tuple(blob["feature_spec"].get("kinetic_keys", ())),
                }
            ),
            feat_mean=np.asarray(blob["feat_mean"], float),
            feat_std=np.asarray(blob["feat_std"], float),
            ctx=ctx,
            recipe=dict(blob.get("recipe", {})),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
def build_estimator(name: str, dataset: EstimatorDataset, **options) -> Estimator:
    """Build an estimator by registry name.

    Options per name: ``ukf`` -> :class:`UkfEstimator` fields; ``pinn`` ->
    :class:`PinnEstimator` fields; ``observer`` -> ``checkpoint`` (default
    :data:`DEFAULT_OBSERVER_CHECKPOINT`).
    """
    if name == "ukf":
        return UkfEstimator(dataset, **options)

    if name == "pinn":
        return PinnEstimator(dataset, **options)
    if name == "observer":
        ckpt = options.pop("checkpoint", DEFAULT_OBSERVER_CHECKPOINT)
        if options:
            raise TypeError(f"observer takes only 'checkpoint', got {sorted(options)}")
        return ObserverEstimator.load(ckpt, dataset)
    raise KeyError(f"unknown estimator {name!r}; known: {ESTIMATOR_NAMES}")


def run_estimator(
    name: str,
    dataset: EstimatorDataset | str | Path = "benchmark",
    split: str = "test",
    indices=None,
    burnin_steps: int = DEFAULT_BURNIN_STEPS,
    jobs: int = 1,
    **options,
) -> list[dict]:
    """Run one estimator over a split and score every series.

    Returns one record per series: ``index``, ``label``, ``score``, ``seconds``
    and the :class:`~.base.TrajectoryEstimate` under ``traj`` (``None`` on failure).
    """
    ds = dataset if isinstance(dataset, EstimatorDataset) else get_dataset(dataset)
    rows = ds.test if split == "test" else ds.pool
    sel = list(range(len(rows))) if indices is None else [int(i) for i in indices]
    est = build_estimator(name, ds, **options)
    out = []
    if jobs > 1 and isinstance(est, UkfEstimator):
        t0 = _time.perf_counter()
        trajs = est.run_many([rows[i] for i in sel], jobs=jobs)
        secs = (_time.perf_counter() - t0) / max(len(sel), 1)
        for i, tr in zip(sel, trajs):
            out.append(
                {
                    "index": i,
                    "label": rows[i].label,
                    "score": None if tr is None else score(tr, rows[i], burnin_steps),
                    "seconds": secs,
                    "traj": tr,
                }
            )
        return out
    for i in sel:
        t0 = _time.perf_counter()
        try:
            tr = est.run(rows[i], i)
        # one divergent series must not end the sweep
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"{name} failed on series {i}: {exc}", stacklevel=2)
            tr = None
        out.append(
            {
                "index": i,
                "label": rows[i].label,
                "score": None if tr is None else score(tr, rows[i], burnin_steps),
                "seconds": _time.perf_counter() - t0,
                "traj": tr,
            }
        )
    return out


def save_runs(path: str | Path, records: list[dict], name: str) -> Path:
    """Store the trajectories of :func:`run_estimator` as an npz (object arrays)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = [r for r in records if r["traj"] is not None]
    np.savez_compressed(
        path,
        estimator=name,
        index=np.array([r["index"] for r in ok]),
        label=np.array([r["label"] for r in ok]),
        score=np.array([r["score"] for r in ok], float),
        x_hat=np.array([r["traj"].x_hat for r in ok], dtype=object),
        std=np.array([r["traj"].std for r in ok], dtype=object),
        time=np.array([r["traj"].time for r in ok], dtype=object),
    )
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("estimator", choices=ESTIMATOR_NAMES)
    ap.add_argument(
        "--dataset", default="benchmark", help="registered name or dataset directory"
    )
    ap.add_argument("--split", default="test", choices=("test", "pool"))
    ap.add_argument("--series", nargs="*", type=int, default=None)
    ap.add_argument("--variant", default="full", help="UKF variant (full | adcore)")
    ap.add_argument("--checkpoint", default=None, help="observer checkpoint")
    ap.add_argument("--jobs", type=int, default=1, help="processes for the UKF")
    ap.add_argument("--burnin", type=int, default=DEFAULT_BURNIN_STEPS)
    ap.add_argument("--out", default=None, help="npz with the trajectories")
    a = ap.parse_args(argv)

    options: dict = {}
    if a.estimator == "ukf":
        options["variant"] = a.variant
    if a.checkpoint and a.estimator == "observer":
        options["checkpoint"] = a.checkpoint
    recs = run_estimator(
        a.estimator,
        a.dataset,
        a.split,
        a.series,
        burnin_steps=a.burnin,
        jobs=a.jobs,
        **options,
    )
    print(f"{'series':>6}  {'label':<14}{'score':>8}  {'s':>7}")
    for r in recs:
        s = "  fail" if r["score"] is None else f"{100 * r['score']:6.1f}%"
        print(f"{r['index']:>6}  {r['label']:<14}{s:>8}  {r['seconds']:7.0f}")
    ok = [r["score"] for r in recs if r["score"] is not None]
    if ok:
        print(
            f"\n{a.estimator}: mean {100 * float(np.mean(ok)):.1f} % over {len(ok)} series"
        )
    if a.out:
        print("wrote", save_runs(a.out, recs, a.estimator))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
