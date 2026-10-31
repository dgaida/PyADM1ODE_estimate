"""PINN v2: trajectory training on one benchmark series (structure study, step 1).

One network per series: a data loss on the true states of the observed window
``[0, t_obs]`` and the physics residual on ``[0, t_obs + horizon]`` with the
known feed. The horizon is a forecast the network has to produce from the
physics alone. Everything the architecture study varies is in
:class:`~.model.ArchConfig`; everything about the task is in :class:`TrainConfig`.

    series = load_series("test")[0]
    result = train_trajectory(series, ArchConfig(depth=4, width=128), TrainConfig(steps=3000))
    result.metrics["horizon"]["rel_rmse_mean"]
"""

from __future__ import annotations

import dataclasses
import math
import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from pyadm1.core.adm1 import STATE_SIZE
from pyadm1.core.adm1_torch import calc_gas_quasi_steady_torch, ph_torch

from ..filter_tuning.filter_runners import build_plant
from ..specs import BLOCK_INDICES
from .data import (
    BenchmarkSeries,
    feed_features,
    feed_lookup,
    load_meta,
    n_feed_features,
    scoring_module,
)
from .influent import InfluentModel
from .kinetics import KineticFactors, perturbed_keys
from .measure import MeasurementModel
from .model import ArchConfig, TrajectoryNet
from .physics import N_CORE, ADM1Physics, PhysicsConfig, build_params, close_state

#: Blocks reported in the metrics (name -> state indices), from the observability spec.
REPORT_BLOCKS: dict[str, list[int]] = dict(BLOCK_INDICES)


@dataclass(frozen=True)
class TrainConfig:
    """Task and optimisation settings of one trajectory training."""

    t_obs_days: float = 30.0  # data window [0, t_obs]
    horizon_days: float = 10.0  # physics-only forecast window (t_obs, t_obs + horizon]
    steps: int = 3000
    n_collocation: int = 512
    lr: float = 2.0e-3
    lr_final_factor: float = 0.01  # cosine decay target
    lambda_phys: float = 1.0
    lambda_warmup_frac: float = (
        0.2  # linear ramp of lambda_phys over this fraction of the steps
    )
    scale_mode: str = "jacobian"  # residual scaling, see ADM1Physics.fit_scale
    data_weight_gas: float = (
        1.0  # extra weight on the four gas rows of the data loss (DAE mode)
    )
    closure: bool = False  # False = DAE mode (41 outputs)
    use_series_kinetics: bool = True  # physics with the series' true kinetic factors
    use_true_feed: bool = True
    data_source: str = "states"  # "states" (true states) | "measurements" (5 sensors)
    n_data_batch: int = 256  # measurement points per step (measurements mode)
    ic_prior: bool = (
        False  # loose prior on x(0) from the plant warm-up (what the UKF starts from)
    )
    lambda_ic: float = 1.0
    estimate_kinetics: bool = False  # inverse PINN: learn the perturbed kinetic factors
    kinetics_keys: tuple[str, ...] | None = (
        None  # subset of keys to estimate (None = all 26 perturbed keys)
    )
    lr_kinetics: float = 5.0e-3
    lambda_kinetics_prior: float = (
        1.0  # weight of the lognormal prior on the log-factors
    )
    kinetics_sigma_prior: float = (
        0.25  # meta["kinetic_perturbation"]["lognormal_sigma"]
    )
    causal_eps: float = 0.0  # causal collocation weighting (Wang et al. 2022); 0 = off
    time_march_days: float = 0.0  # cumulative time-marching windows [d]; 0 = off
    greybox: bool = False  # learned rate correction g(x, u) added to the physics
    greybox_width: int = 32
    lambda_greybox: float = 1.0  # L2 penalty on the correction (scaled-residual units)
    lbfgs_steps: int = 0  # L-BFGS iterations after Adam (0 = off)
    prior: str = (
        "series"  # output normalisation: "series" (its own truth) | "train_split"
    )
    log_sigma_floor: float = 0.05
    seed: int = 0
    dtype: str = "float64"
    log_every: int = 100
    eval_days: float = (
        60.0  # evaluate the network on [0, eval_days] (beyond the physics range)
    )


@dataclass
class RunResult:
    series_index: int
    regime: str
    arch: dict
    train: dict
    n_parameters: int
    seconds: float
    metrics: dict
    loss_curve: list[dict] = field(default_factory=list)
    prediction: np.ndarray | None = None  # (T, 41) on the series grid (optional)

    def to_record(self) -> dict:
        d = asdict(self)
        d.pop("prediction", None)
        return d


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------
def _dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float64": torch.float64}[name]


def build_physics(
    series: BenchmarkSeries,
    cfg: TrainConfig,
    meta: dict | None = None,
    scale_ref: tuple[np.ndarray, np.ndarray] | None = None,
) -> ADM1Physics:
    """Physics of ``series``; the residual scale is fitted on ``scale_ref``
    ``(states, feed)`` when given (training-split data, no truth of the series),
    else on the series' own observed window."""
    meta = meta or load_meta()
    plant = build_plant(meta)
    adm1 = plant.components["primary"].adm1
    adm1.create_influent(list(series.feed_true[0]), 0)
    kw = {}
    if cfg.use_series_kinetics:
        kw = {
            "kinetic_factors": series.kinetic_factors,
            "kinetic_keys": series.kinetic_keys,
        }
    params = build_params(plant, **kw)
    influent = InfluentModel.from_adm1(adm1, mode="nominal_mix")
    phys = ADM1Physics(params, influent, PhysicsConfig(closure=cfg.closure))
    if scale_ref is not None:
        phys.fit_scale(scale_ref[0], scale_ref[1], mode=cfg.scale_mode)
    else:
        n_obs = round(cfg.t_obs_days / series.dt_days) + 1
        feed = series.feed_true if cfg.use_true_feed else series.feed_noisy
        phys.fit_scale(series.states[:n_obs], feed[:n_obs], mode=cfg.scale_mode)
    return phys


def ukf_warmup_state(meta: dict, digester_id: str = "primary") -> np.ndarray:
    """The ``(41,)`` state the reference UKF starts every series from.

    Mirrors ``filter_tuning.filter_runners.UkfRunner``: the example plant with the
    dataset's operating point, warmed up for ``meta["warmup_days"]`` at the
    baseline feed with *nominal* kinetics. Series-independent, so compute it once.
    """
    from ..filter_tuning.filter_runners import (
        _warmup,
        baseline_flows,
        substrates_from_meta,
    )
    from ..quickstart import build_filter_components

    plant = build_plant(meta, digester_id)
    subs = substrates_from_meta(meta)
    wp, _, wspec = build_filter_components(
        plant,
        digester_id=digester_id,
        substrates=subs,
        sensors=[s.lower() for s in meta["sensors"]],
    )
    xw = _warmup(
        wp,
        wspec,
        wspec.read_adm1_state(plant),
        float(meta.get("warmup_days", 30.0)),
        baseline_flows(meta, subs),
        float(meta.get("dt_hours", 1.0)),
    )
    x0 = np.zeros(STATE_SIZE)
    for i, ch in enumerate(wspec.channels):
        if ch.kind == "adm1":
            x0[ch.adm1_index] = xw[i]
    return x0


def build_net(
    series: BenchmarkSeries,
    arch: ArchConfig,
    cfg: TrainConfig,
    prior_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> TrajectoryNet:
    n_out = N_CORE if cfg.closure else STATE_SIZE
    n_obs = round(cfg.t_obs_days / series.dt_days) + 1
    if cfg.prior == "train_split":
        if prior_stats is None:
            raise ValueError(
                "prior='train_split' needs prior_stats=(mu, sigma) from train_split_log_stats()"
            )
        mu, sigma = (np.asarray(v, dtype=float)[:n_out] for v in prior_stats)
    elif cfg.prior == "series":
        logx = np.log(np.clip(series.states[:n_obs, :n_out], 1.0e-30, None))
        mu = logx.mean(axis=0)
        sigma = np.maximum(logx.std(axis=0), cfg.log_sigma_floor)
    else:
        raise ValueError(f"unknown prior {cfg.prior!r}")
    feed = series.feed_true if cfg.use_true_feed else series.feed_noisy
    n_sub = feed.shape[1]
    n_feat = n_feed_features(n_sub, arch.feed_mode)
    per_sub = np.maximum(feed.max(axis=0), 1.0e-6)
    feed_scale = np.tile(per_sub, n_feat // n_sub) if n_feat else np.zeros(0)
    return TrajectoryNet(
        arch,
        n_out=n_out,
        n_feed=n_feat,
        t_scale=cfg.t_obs_days + cfg.horizon_days,
        feed_scale=feed_scale,
        log_mu=mu,
        log_sigma=sigma,
        seed=cfg.seed,
    )


class RateCorrection(torch.nn.Module):
    """Grey-box rate correction ``g(x, u)`` in scaled-residual units.

    A small tanh MLP on the normalised log-state and the feed features whose
    output is subtracted from the scaled physics residual, i.e. the effective
    dynamics become ``f_ADM1 + sigma * g``. Zero-initialised last layer, so the
    model starts as the pure physics; ``lambda_greybox`` keeps it small.
    """

    def __init__(self, net: TrajectoryNet, n_out: int, width: int):
        super().__init__()
        self._net = [net]  # not a submodule (avoid double registration)
        n_in = net.n_out + net.n_feed
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_in, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, width),
            torch.nn.Tanh(),
            torch.nn.Linear(width, n_out),
        )
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        net = self._net[0]
        z = net.to_latent(x[..., : net.n_out])
        parts = [z] + ([u / net.feed_scale] if net.n_feed else [])
        return self.mlp(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_trajectory(
    series: BenchmarkSeries,
    arch: ArchConfig,
    cfg: TrainConfig,
    *,
    meta: dict | None = None,
    prior_stats: tuple[np.ndarray, np.ndarray] | None = None,
    ic_state: np.ndarray | None = None,
    scale_ref: tuple[np.ndarray, np.ndarray] | None = None,
    keep_prediction: bool = False,
    verbose: bool = False,
) -> RunResult:
    """Train one network on one series and evaluate fit + forecast.

    ``data_source="states"`` fits the true states (capacity study);
    ``data_source="measurements"`` fits only the five noisy sensors through the
    measurement map (field setting). ``prior_stats`` is required with
    ``prior="train_split"`` (see :func:`~.data.train_split_log_stats`);
    ``ic_state`` (41,) is required with ``ic_prior`` (see :func:`ukf_warmup_state`).
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dt = _dtype(cfg.dtype)
    t0 = _time.perf_counter()
    meta = meta or load_meta()
    if cfg.data_source == "measurements" and scale_ref is None:
        raise ValueError(
            "data_source='measurements' needs scale_ref=(states, feed) from the training split "
            "(train_split_scale_reference()) so no truth of the series is used"
        )

    phys = build_physics(series, cfg, meta, scale_ref)
    net = build_net(series, arch, cfg, prior_stats).to(dt)
    n_out = net.n_out
    feed_np = series.feed_true if cfg.use_true_feed else series.feed_noisy

    # Data window (states: full batch, 721 hourly points is cheap; measurements:
    # a random minibatch per step, the closure inside h(x) is the expensive part).
    n_obs = round(cfg.t_obs_days / series.dt_days) + 1
    t_data = torch.as_tensor(series.time[:n_obs, None], dtype=dt)
    u_data = torch.as_tensor(
        feed_features(feed_np, series.time, series.time[:n_obs], arch.feed_mode),
        dtype=dt,
    )
    if cfg.data_source == "states":
        z_target = net.to_latent(
            torch.as_tensor(series.states[:n_obs, :n_out], dtype=dt)
        )
        w_data = torch.ones(n_out, dtype=dt)
        if not cfg.closure and cfg.data_weight_gas != 1.0:
            w_data[37:41] = cfg.data_weight_gas
        meas = None
    elif cfg.data_source == "measurements":
        if series.measurements is None:
            raise ValueError("series has no measurements")
        meas = MeasurementModel.from_meta(phys.params, meta)
        y_data = torch.as_tensor(series.measurements[:n_obs], dtype=dt)
    else:
        raise ValueError(f"unknown data_source {cfg.data_source!r}")
    # Initial-condition prior: log-space Gaussian around the warm-up state with the
    # network's own per-state log-std (loose: one sigma = the state's typical spread).
    if cfg.ic_prior:
        if ic_state is None:
            raise ValueError("ic_prior=True needs ic_state (see ukf_warmup_state())")
        ic = np.asarray(ic_state, dtype=float)[:n_out]
        # states the warm-up leaves at exactly zero (S_h2) carry no prior
        ic_mask = torch.as_tensor(ic > 0.0, dtype=dt)
        z_ic = net.to_latent(
            torch.as_tensor(np.where(ic > 0.0, ic, 1.0)[None, :], dtype=dt)
        )

    t_max = cfg.t_obs_days + cfg.horizon_days
    gen = torch.Generator().manual_seed(cfg.seed)

    # Inverse PINN: the perturbed kinetic constants become log-factor parameters.
    kin = None
    base_params = phys.params
    param_groups = [{"params": net.parameters(), "lr": cfg.lr}]
    if cfg.estimate_kinetics:
        sigma_prior = float(
            meta.get("kinetic_perturbation", {}).get(
                "lognormal_sigma", cfg.kinetics_sigma_prior
            )
        )
        keys = (
            list(cfg.kinetics_keys)
            if cfg.kinetics_keys
            else perturbed_keys(base_params)
        )
        kin = KineticFactors(keys, sigma_prior=sigma_prior, dtype=dt)
        param_groups.append({"params": kin.parameters(), "lr": cfg.lr_kinetics})
    # Grey-box: learned rate correction in scaled-residual units.
    gbox = None
    if cfg.greybox:
        gbox = RateCorrection(net, phys.n_residual, cfg.greybox_width).to(dt)
        param_groups.append({"params": gbox.parameters(), "lr": cfg.lr})
    opt = torch.optim.Adam(param_groups)
    # Cosine decay to lr_final_factor * lr, applied relatively to every group.
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: cfg.lr_final_factor
        + (1.0 - cfg.lr_final_factor)
        * 0.5
        * (1.0 + math.cos(math.pi * min(s, cfg.steps) / cfg.steps)),
    )
    curve = []

    # Time-marching: cumulative windows [0, (k+1)·W], each with an equal share of
    # the steps; stage 0 sees only the first W days. Off (one stage) by default.
    n_stage = (
        max(1, math.ceil(t_max / cfg.time_march_days)) if cfg.time_march_days > 0 else 1
    )
    steps_per_stage = max(1, cfg.steps // n_stage)

    def losses(step: int, t_fixed: torch.Tensor | None = None, full_data: bool = False):
        """Loss terms of one step: ``(data, phys_weighted, phys_plain, extra)``.

        ``phys_weighted`` is what is optimised (causal weights, grey-box
        correction), ``phys_plain`` the unweighted mean squared residual for
        logging, ``extra`` the prior / penalty terms.
        """
        stage = min(n_stage - 1, (step - 1) // steps_per_stage)
        t_hi = t_max if n_stage == 1 else min(t_max, (stage + 1) * cfg.time_march_days)
        n_hi = min(n_obs, round(t_hi / series.dt_days) + 1)
        if kin is not None:
            phys.params = kin.apply(base_params)
        # data
        if meas is None:
            z_hat = net.latent(t_data[:n_hi], u_data[:n_hi])
            loss_data = ((z_hat - z_target[:n_hi]) ** 2 * w_data).mean()
        else:
            if full_data:
                idx = torch.arange(n_hi)
            else:
                idx = torch.randint(n_hi, (min(cfg.n_data_batch, n_hi),), generator=gen)
            x_d = net(t_data[idx], u_data[idx])
            loss_data = (meas.normalized_residual(x_d, y_data[idx]) ** 2).mean()
        if cfg.ic_prior:
            d_ic = (net.latent(t_data[:1], u_data[:1]) - z_ic) ** 2 * ic_mask
            loss_data = loss_data + cfg.lambda_ic * d_ic.sum() / ic_mask.sum()
        # physics at random collocation points in [0, t_hi]
        if t_fixed is None:
            t_c = torch.rand(cfg.n_collocation, 1, generator=gen, dtype=dt) * t_hi
        else:
            t_c = t_fixed
        t_c_np = t_c.numpy()[:, 0]
        u_c = torch.as_tensor(
            feed_features(feed_np, series.time, t_c_np, arch.feed_mode), dtype=dt
        )
        feed_c = torch.as_tensor(feed_lookup(feed_np, series.time, t_c_np), dtype=dt)
        x_c, dx_c = net.with_rates(t_c, u_c)
        r = phys.residual(x_c, dx_c, feed_c)
        extra = torch.zeros((), dtype=dt)
        if gbox is not None:
            g = gbox(x_c, u_c)
            r = r - g
            extra = extra + cfg.lambda_greybox * (g**2).mean()
        if kin is not None:
            extra = extra + cfg.lambda_kinetics_prior * kin.prior_loss()
        per_point = (r**2).mean(dim=-1)
        phys_plain = per_point.mean()
        if cfg.causal_eps > 0:
            order = torch.argsort(t_c[:, 0])
            lp = per_point[order]
            cum = torch.cumsum(lp.detach(), 0) - lp.detach()
            w = torch.exp(-cfg.causal_eps * cum / len(lp))
            phys_weighted = (w * lp).sum() / w.sum()
        else:
            phys_weighted = phys_plain
        return loss_data, phys_weighted, phys_plain, extra

    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        loss_data, loss_phys, phys_plain, extra = losses(step)
        warm = (
            min(1.0, step / max(1.0, cfg.lambda_warmup_frac * cfg.steps))
            if cfg.lambda_warmup_frac > 0
            else 1.0
        )
        loss = loss_data + warm * cfg.lambda_phys * loss_phys + extra
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        if step % cfg.log_every == 0 or step == 1 or step == cfg.steps:
            rec = {
                "step": step,
                "loss": loss.item(),
                "data": loss_data.item(),
                "phys": phys_plain.item(),
            }
            curve.append(rec)
            if verbose:
                print(
                    f"  step {step:5d}  loss {rec['loss']:.3e}  data {rec['data']:.3e}  phys {rec['phys']:.3e}"
                )

    # Optional L-BFGS polish on a fixed collocation set and the full data window.
    if cfg.lbfgs_steps > 0:
        t_fix = torch.rand(2 * cfg.n_collocation, 1, generator=gen, dtype=dt) * t_max
        all_params = [p for g in opt.param_groups for p in g["params"]]
        lb = torch.optim.LBFGS(
            all_params,
            lr=1.0,
            max_iter=cfg.lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-10,
            tolerance_change=1e-12,
        )

        def closure():
            lb.zero_grad(set_to_none=True)
            ld, lp, _, ex = losses(cfg.steps, t_fixed=t_fix, full_data=True)
            total = ld + cfg.lambda_phys * lp + ex
            total.backward()
            return total

        lb.step(closure)
        with torch.no_grad():
            ld, _, pp, _ = losses(cfg.steps, t_fixed=t_fix, full_data=True)
        curve.append(
            {
                "step": cfg.steps + cfg.lbfgs_steps,
                "loss": float(ld + cfg.lambda_phys * pp),
                "data": ld.item(),
                "phys": pp.item(),
                "lbfgs": True,
            }
        )

    seconds = _time.perf_counter() - t0
    if kin is not None:
        # Freeze the estimate as plain floats for evaluation (detached snapshot).
        phys.params = dataclasses.replace(
            base_params,
            kinetic={
                k: (float(v) if not torch.is_tensor(v) else float(v.detach()))
                for k, v in kin.apply(base_params).kinetic.items()
            },
        )
    metrics, pred = evaluate(net, phys, series, cfg)
    if gbox is not None:
        with torch.no_grad():
            n_ev = len(pred)
            u_ev = torch.as_tensor(
                feed_features(feed_np, series.time, series.time[:n_ev], arch.feed_mode),
                dtype=dt,
            )
            g_ev = gbox(torch.as_tensor(pred[:, :n_out], dtype=dt), u_ev)
        metrics["greybox"] = {
            "g_rms": float(torch.sqrt((g_ev**2).mean())),
            "g_rms_per_row": torch.sqrt((g_ev**2).mean(dim=0)).tolist(),
        }
    if kin is not None:
        metrics["kinetics"] = kin.report(series.kinetic_keys, series.kinetic_factors)
    return RunResult(
        series_index=series.index,
        regime=series.regime,
        arch=asdict(arch),
        train=asdict(cfg),
        n_parameters=net.n_parameters,
        seconds=seconds,
        metrics=metrics,
        loss_curve=curve,
        prediction=pred if keep_prediction else None,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def _rel_rmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-state relative RMSE ``sqrt(mean(((p - x)/x)^2))`` over the rows."""
    rel = (pred - truth) / np.maximum(np.abs(truth), 1.0e-30)
    return np.sqrt(np.mean(rel * rel, axis=0))


def _window_metrics(
    pred41: np.ndarray, truth41: np.ndarray, phys: ADM1Physics, dt: torch.dtype
) -> dict:
    rel = _rel_rmse(pred41, truth41)
    out = {
        "rel_rmse_per_state": rel.tolist(),
        "rel_rmse_mean": float(rel.mean()),
        "rel_rmse_median": float(np.median(rel)),
        "blocks": {name: float(rel[idx].mean()) for name, idx in REPORT_BLOCKS.items()},
    }
    # pH and methane flow are evaluated from the *closed* core state in both modes:
    # the charge balance and the pressure-driven gas flow are knife-edge differences
    # of large terms, so the raw ion / gas outputs of a DAE-mode network give a
    # meaningless pH even at percent-level state accuracy. The closure gives the
    # pH / q_ch4 the predicted core implies, for prediction and truth alike.
    with torch.no_grad():
        xp = close_state(torch.as_tensor(pred41[:, :N_CORE], dtype=dt), phys.params)
        xt = close_state(torch.as_tensor(truth41[:, :N_CORE], dtype=dt), phys.params)
        ph_p, ph_t = ph_torch(xp, phys.params), ph_torch(xt, phys.params)
        _, q_p, _ = calc_gas_quasi_steady_torch(xp, phys.params)
        _, q_t, _ = calc_gas_quasi_steady_torch(xt, phys.params)
        ph_raw = ph_torch(torch.as_tensor(pred41, dtype=dt), phys.params)
    out["pH_rmse"] = float(torch.sqrt(((ph_p - ph_t) ** 2).mean()))
    out["pH_rmse_raw_ions"] = float(torch.sqrt(((ph_raw - ph_t) ** 2).mean()))
    out["q_ch4_rel_rmse"] = float(
        torch.sqrt((((q_p - q_t) / torch.clamp(q_t.abs(), min=1e-9)) ** 2).mean())
    )
    return out


def evaluate(
    net: TrajectoryNet, phys: ADM1Physics, series: BenchmarkSeries, cfg: TrainConfig
) -> tuple[dict, np.ndarray]:
    """Fit / horizon / beyond-horizon metrics on the series grid; returns ``(metrics, pred (T, 41))``."""
    dt = _dtype(cfg.dtype)
    n_eval = min(round(cfg.eval_days / series.dt_days) + 1, series.n_steps)
    feed_np = series.feed_true if cfg.use_true_feed else series.feed_noisy
    with torch.no_grad():
        t = torch.as_tensor(series.time[:n_eval, None], dtype=dt)
        u = torch.as_tensor(
            feed_features(
                feed_np, series.time, series.time[:n_eval], net.arch.feed_mode
            ),
            dtype=dt,
        )
        x = net(t, u)
        if cfg.closure:
            x = close_state(x, phys.params)
        pred = x.numpy()
    truth = series.states[:n_eval]
    n_obs = min(round(cfg.t_obs_days / series.dt_days) + 1, n_eval)
    n_hor = min(round((cfg.t_obs_days + cfg.horizon_days) / series.dt_days) + 1, n_eval)
    has_horizon = n_hor > n_obs
    metrics = {"fit": _window_metrics(pred[:n_obs], truth[:n_obs], phys, dt)}
    if has_horizon:
        metrics["horizon"] = _window_metrics(
            pred[n_obs:n_hor], truth[n_obs:n_hor], phys, dt
        )
    if n_eval > n_hor + 1:
        metrics["beyond"] = _window_metrics(pred[n_hor:], truth[n_hor:], phys, dt)
    # physics residual level of the *network* (scaled, RMS over rows): on the horizon
    # when there is one, else on the fit window (sub-sampled).
    sl = slice(n_obs, n_hor) if has_horizon else slice(0, n_obs, max(1, n_obs // 240))
    tt = torch.as_tensor(series.time[sl, None], dtype=dt)
    uu = torch.as_tensor(
        feed_features(feed_np, series.time, series.time[sl], net.arch.feed_mode),
        dtype=dt,
    )
    ff = torch.as_tensor(feed_lookup(feed_np, series.time, series.time[sl]), dtype=dt)
    x_h, dx_h = net.with_rates(tt, uu)
    with torch.no_grad():
        r = phys.residual(x_h.detach(), dx_h.detach(), ff)
    (metrics["horizon"] if has_horizon else metrics["fit"])["phys_residual_rms"] = (
        float(torch.sqrt((r**2).mean()))
    )
    # Benchmark scoring (the UKF's own NRMSE / RMSE-ratio metrics) per window.
    windows = {"fit": slice(0, n_obs), "full": slice(0, n_hor)}
    if has_horizon:
        windows["horizon"] = slice(n_obs, n_hor)
    metrics["benchmark"] = benchmark_scores(pred, series, windows)
    return metrics, pred


def benchmark_scores(
    pred: np.ndarray, series: BenchmarkSeries, windows: dict[str, slice]
) -> dict:
    """``datasets/benchmark/scoring.score_series`` on each window: model vs. reference UKF.

    Keys per window: ``model`` / ``ukf`` (transient / steady / overall NRMSE in %,
    median over the 41 states) and ``vs_ukf`` (median per-state RMSE ratio, < 1
    beats the UKF). The UKF is a *filter* that sees the measurements of every
    window, the PINN sees none in the horizon - read the horizon comparison as
    "forecast vs. filter", not as equals.
    """
    sc = scoring_module()
    out = {}
    for name, sl in windows.items():
        r = sc.score_series(pred[sl], series.scoring_dict(sl))
        out[name] = {
            "model": r["model"],
            "ukf": r["ukf_reference"],
            "vs_ukf": r["vs_ukf"],
        }
    return out


def summarize(records: list[dict]) -> dict:
    """Mean / std over series per architecture for the headline numbers.

    The "main" window is the horizon when the runs have one, else the fit window
    (60-day estimation task). Missing metrics are reported as ``nan``.
    """
    by_arch: dict[str, list[dict]] = {}
    for rec in records:
        a = ArchConfig(**rec["arch"]).name
        by_arch.setdefault(a, []).append(rec)
    out = {}
    for a, recs in by_arch.items():

        def col(path, recs=recs):
            vals = []
            for r in recs:
                v = r["metrics"]
                try:
                    for p in path:
                        v = v[p]
                    vals.append(float(v))
                except (KeyError, TypeError):
                    vals.append(float("nan"))
            return (
                float(np.nanmean(vals)),
                float(np.nanstd(vals)),
                int(np.sum(np.isfinite(vals))),
            )

        main = "horizon" if all("horizon" in r["metrics"] for r in recs) else "fit"
        fit_med = [r["metrics"]["fit"]["rel_rmse_median"] for r in recs]
        out[a] = {
            "n_runs": len(recs),
            "main_window": main,
            "n_parameters": recs[0]["n_parameters"],
            "seconds_mean": float(np.mean([r["seconds"] for r in recs])),
            "diverged": int(sum(v > 0.5 for v in fit_med)),
            "fit_rel_rmse_mean": col(("fit", "rel_rmse_mean")),
            "fit_rel_rmse_median": col(("fit", "rel_rmse_median")),
            "main_rel_rmse_mean": col((main, "rel_rmse_mean")),
            "main_rel_rmse_median": col((main, "rel_rmse_median")),
            "main_pH_rmse": col((main, "pH_rmse")),
            "main_q_ch4_rel_rmse": col((main, "q_ch4_rel_rmse")),
            "main_phys_residual_rms": col((main, "phys_residual_rms")),
            "main_blocks": {b: col((main, "blocks", b)) for b in REPORT_BLOCKS},
            "per_regime_main_rel_rmse_median": {
                reg: float(
                    np.mean(
                        [
                            r["metrics"][main]["rel_rmse_median"]
                            for r in recs
                            if r["regime"] == reg
                        ]
                    )
                )
                for reg in sorted({r["regime"] for r in recs})
            },
        }
        # backwards-compatible aliases used by report.py
        for k in (
            "rel_rmse_mean",
            "rel_rmse_median",
            "pH_rmse",
            "q_ch4_rel_rmse",
            "phys_residual_rms",
            "blocks",
        ):
            out[a][f"horizon_{k}"] = out[a][f"main_{k}"]
        out[a]["per_regime_horizon_rel_rmse_median"] = out[a][
            "per_regime_main_rel_rmse_median"
        ]
        if all("beyond" in r["metrics"] for r in recs):
            out[a]["beyond_rel_rmse_median"] = col(("beyond", "rel_rmse_median"))
        if all("kinetics" in r["metrics"] for r in recs):
            out[a]["kinetics"] = {
                "log_rmse_estimated": col(("kinetics", "log_rmse_estimated")),
                "log_rmse_nominal": col(("kinetics", "log_rmse_nominal")),
                "corr_log": col(("kinetics", "corr_log")),
            }
        if all("benchmark" in r["metrics"] for r in recs):
            bm = {}
            for w in ("fit", "horizon", "full"):
                if not all(w in r["metrics"]["benchmark"] for r in recs):
                    continue
                bm[w] = {
                    "model_nrmse_overall": col(
                        ("benchmark", w, "model", "nrmse_overall_%")
                    ),
                    "model_nrmse_transient": col(
                        ("benchmark", w, "model", "nrmse_transient_%")
                    ),
                    "model_nrmse_steady": col(
                        ("benchmark", w, "model", "nrmse_steady_%")
                    ),
                    "ukf_nrmse_overall": col(
                        ("benchmark", w, "ukf", "nrmse_overall_%")
                    ),
                    "ukf_nrmse_transient": col(
                        ("benchmark", w, "ukf", "nrmse_transient_%")
                    ),
                    "ukf_nrmse_steady": col(("benchmark", w, "ukf", "nrmse_steady_%")),
                    "rmse_ratio_overall": col(
                        ("benchmark", w, "vs_ukf", "rmse_ratio_overall")
                    ),
                    "rmse_ratio_transient": col(
                        ("benchmark", w, "vs_ukf", "rmse_ratio_transient")
                    ),
                    "beats_ukf": int(
                        sum(
                            r["metrics"]["benchmark"][w]["vs_ukf"]["rmse_ratio_overall"]
                            < 1.0
                            for r in recs
                        )
                    ),
                }
            out[a]["benchmark"] = bm
    return out


def format_benchmark(summary: dict) -> str:
    """Model vs. UKF table (benchmark NRMSE, median over states, mean over series)."""
    lines = [
        (
            f"{'arch':34s} {'window':8s} {'PINN ovr%':>10s} {'UKF ovr%':>9s} {'ratio':>6s} {'<1':>6s} "
            f"{'PINN trans%':>12s} {'UKF trans%':>11s} {'ratio':>6s} {'div':>4s}"
        )
    ]

    def key(kv):
        b = kv[1].get("benchmark", {})
        w = "full" if "full" in b else "fit"
        return b.get(w, {}).get("model_nrmse_overall", (math.inf,))[0]

    for a, s in sorted(summary.items(), key=key):
        if "benchmark" not in s:
            continue
        for w, b in s["benchmark"].items():
            lines.append(
                f"{a:34s} {w:8s} {b['model_nrmse_overall'][0]:10.2f} {b['ukf_nrmse_overall'][0]:9.2f} "
                f"{b['rmse_ratio_overall'][0]:6.2f} {b['beats_ukf']:3d}/{s['n_runs']:<2d} "
                f"{b['model_nrmse_transient'][0]:12.2f} {b['ukf_nrmse_transient'][0]:11.2f} "
                f"{b['rmse_ratio_transient'][0]:6.2f} {s['diverged']:4d}"
            )
        if "kinetics" in s:
            k = s["kinetics"]
            lines.append(
                f"{a:34s} kinetics  log-RMSE estimated {k['log_rmse_estimated'][0]:.3f} "
                f"(nominal {k['log_rmse_nominal'][0]:.3f}), corr(log true, log est) {k['corr_log'][0]:.2f}"
            )
    return "\n".join(lines)


def format_summary(summary: dict) -> str:
    any_s = next(iter(summary.values()), {})
    main = any_s.get("main_window", "horizon")
    lines = [
        (
            f"{'arch':34s} {'params':>8s} {'s/run':>6s} {'div':>4s} {'fit med%':>9s} {main[:3] + ' med%':>9s} "
            f"{main[:3] + ' mean%':>10s} {'pH':>7s} {'qCH4%':>7s} {'phys':>7s} {'beyond med%':>12s}"
        )
    ]
    for a, s in sorted(
        summary.items(), key=lambda kv: kv[1]["main_rel_rmse_median"][0]
    ):
        b = s.get("beyond_rel_rmse_median", (math.nan, 0, 0))
        lines.append(
            f"{a:34s} {s['n_parameters']:8d} {s['seconds_mean']:6.0f} {s['diverged']:4d} "
            f"{100 * s['fit_rel_rmse_median'][0]:9.2f} {100 * s['main_rel_rmse_median'][0]:9.2f} "
            f"{100 * s['main_rel_rmse_mean'][0]:10.2f} {s['main_pH_rmse'][0]:7.3f} "
            f"{100 * s['main_q_ch4_rel_rmse'][0]:7.2f} {s['main_phys_residual_rms'][0]:7.3f} {100 * b[0]:12.2f}"
        )
    return "\n".join(lines)
