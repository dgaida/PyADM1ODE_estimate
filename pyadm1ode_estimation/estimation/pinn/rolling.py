"""Causal PINN: rolling-window training with warm start (fusion study, block F1).

In operation the network cannot see the whole series. Every ``cadence_hours``
it is retrained on the window ``[t_r - window_days, t_r]`` of measurements
available at time ``t_r``, warm-started from the previous weights. Between two
retrainings the estimate at the current hour is *bridged*:

``hold``
    keep the last estimate ``x(t_r)``;
``adm1``
    integrate the ADM1 (nominal kinetics, known feed) from ``x(t_r)`` forward.

Both bridgings are derived from one chain of retrainings. The chain also yields
fixed-lag estimates: for hour ``k`` the estimate of the first retraining whose
window end is at least ``lag`` days later (the network is a smoother inside its
window). Every trajectory is scored with the benchmark metric against the truth
and the reference UKF, so causal and offline estimates are directly comparable.
"""

from __future__ import annotations

import dataclasses
import math
import time as _time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from pyadm1.core.adm1 import STATE_SIZE
from pyadm1.core.adm1_torch import adm1da_rhs_torch
from scipy.integrate import solve_ivp

from .data import BenchmarkSeries, feed_features, feed_lookup, load_meta
from .measure import MeasurementModel
from .model import ArchConfig, TrajectoryNet
from .physics import N_CORE, ADM1Physics, close_state
from .train import TrainConfig, benchmark_scores, build_net, build_physics


@dataclass(frozen=True)
class RollingConfig:
    """One causal scheme of block F1."""

    window_days: float = 20.0
    cadence_hours: float = 24.0
    steps_first: int = 3000  # first training (from scratch)
    steps_warm: int = 200  # every later retraining (warm start)
    lr_warm: float = 5.0e-4
    lags_days: tuple[float, ...] = (1.0, 3.0)
    min_window_days: float = 1.0  # before that, no retraining (prior mean is used)
    adm1_bridge: bool = True  # also integrate the "adm1" bridging (costly; "hold" does not need it)

    @property
    def name(self) -> str:
        return f"W{self.window_days:g}_C{self.cadence_hours:g}_S{self.steps_warm}"


@dataclass
class RollingResult:
    series_index: int
    regime: str
    arch: dict
    train: dict
    rolling: dict
    n_retrainings: int
    seconds_total: float
    seconds_per_operating_hour: float
    metrics: dict
    trajectories: dict = field(default_factory=dict)  # name -> (T, 41)

    def to_record(self) -> dict:
        d = asdict(self)
        d.pop("trajectories", None)
        return d


# ---------------------------------------------------------------------------
# Window training (warm start)
# ---------------------------------------------------------------------------
def fit_window(
    net: TrajectoryNet,
    phys: ADM1Physics,
    meas: MeasurementModel,
    series: BenchmarkSeries,
    cfg: TrainConfig,
    t0: float,
    t1: float,
    steps: int,
    lr: float,
    gen: torch.Generator,
    feed_np: np.ndarray,
    y_all: torch.Tensor,
    t_all: torch.Tensor,
    u_all: torch.Tensor,
) -> dict:
    """Train ``net`` in place on measurements in ``[t0, t1]`` (physics on the same window)."""
    dt = t_all.dtype
    i0 = int(np.searchsorted(series.time, t0 - 1e-9))
    i1 = int(np.searchsorted(series.time, t1 + 1e-9))  # exclusive
    n_win = max(i1 - i0, 1)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: cfg.lr_final_factor
        + (1.0 - cfg.lr_final_factor)
        * 0.5
        * (1.0 + math.cos(math.pi * min(s, steps) / steps)),
    )
    last = {}
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        idx = i0 + torch.randint(n_win, (min(cfg.n_data_batch, n_win),), generator=gen)
        x_d = net(t_all[idx], u_all[idx])
        loss_data = (meas.normalized_residual(x_d, y_all[idx]) ** 2).mean()
        t_c = t0 + torch.rand(cfg.n_collocation, 1, generator=gen, dtype=dt) * (t1 - t0)
        t_c_np = t_c.numpy()[:, 0]
        u_c = torch.as_tensor(
            feed_features(feed_np, series.time, t_c_np, net.arch.feed_mode), dtype=dt
        )
        feed_c = torch.as_tensor(feed_lookup(feed_np, series.time, t_c_np), dtype=dt)
        x_c, dx_c = net.with_rates(t_c, u_c)
        r = phys.residual(x_c, dx_c, feed_c)
        loss_phys = (r**2).mean()
        warm = (
            min(1.0, step / max(1.0, cfg.lambda_warmup_frac * steps))
            if steps >= 1000
            else 1.0
        )
        loss = loss_data + warm * cfg.lambda_phys * loss_phys
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        sched.step()
        last = {"data": loss_data.item(), "phys": loss_phys.item()}
    return last


# ---------------------------------------------------------------------------
# ADM1 continuation (bridging) from a closed 41-state
# ---------------------------------------------------------------------------
def continue_adm1(
    x41: np.ndarray,
    t_from: float,
    hours: int,
    series: BenchmarkSeries,
    phys: ADM1Physics,
    feed_np: np.ndarray,
) -> np.ndarray:
    """Integrate the nominal ADM1 hour by hour from ``x41`` at ``t_from``; returns ``(hours, 41)``."""
    out = np.zeros((hours, STATE_SIZE))
    x = np.asarray(x41, dtype=float).copy()
    dt_h = series.dt_days
    for h in range(hours):
        t_a = t_from + h * dt_h
        feed_row = feed_lookup(feed_np, series.time, np.array([t_a + 0.5 * dt_h]))
        q_ad, s_in = phys.influent(feed_row)
        p = dataclasses.replace(
            phys.params,
            q_ad=float(q_ad[0]),
            s_in=[float(v) for v in s_in[0]],
            Q_out_override=None,
        )

        def fun(t, y, p=p):
            return (
                adm1da_rhs_torch(torch.as_tensor(y, dtype=torch.float64), p)
                .detach()
                .numpy()
            )

        sol = solve_ivp(
            fun,
            (0.0, dt_h),
            np.clip(x, 1e-12, None),
            method="BDF",
            rtol=1e-6,
            atol=1e-9,
        )
        x = sol.y[:, -1] if sol.success else x
        out[h] = x
    return out


# ---------------------------------------------------------------------------
# The causal chain
# ---------------------------------------------------------------------------
def run_rolling(
    series: BenchmarkSeries,
    arch: ArchConfig,
    cfg: TrainConfig,
    roll: RollingConfig,
    *,
    meta: dict | None = None,
    prior_stats=None,
    scale_ref=None,
    verbose: bool = False,
) -> RollingResult:
    """Run one causal scheme on one series and score all derived trajectories."""
    torch.manual_seed(cfg.seed)
    meta = meta or load_meta()
    dt = torch.float64
    t_start = _time.perf_counter()

    phys = build_physics(series, cfg, meta, scale_ref)
    meas = MeasurementModel.from_meta(phys.params, meta)
    net = build_net(series, arch, cfg, prior_stats).to(dt)  # t_scale = 60 d, fixed
    feed_np = series.feed_true if cfg.use_true_feed else series.feed_noisy
    T = series.n_steps
    t_all = torch.as_tensor(series.time[:, None], dtype=dt)
    u_all = torch.as_tensor(
        feed_features(feed_np, series.time, series.time, arch.feed_mode), dtype=dt
    )
    y_all = torch.as_tensor(series.measurements, dtype=dt)
    gen = torch.Generator().manual_seed(cfg.seed)

    def predict(idx: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x = net(t_all[idx], u_all[idx])
            return close_state(x, phys.params).numpy()

    # trajectories on the hourly grid
    hold = np.full((T, STATE_SIZE), np.nan)
    adm1 = np.full((T, STATE_SIZE), np.nan)
    lags = {f"lag{L:g}": np.full((T, STATE_SIZE), np.nan) for L in roll.lags_days}
    lag_filled = {k: 0 for k in lags}  # hours already assigned per lag

    # before the first retraining: prior mean (network output at initialisation)
    x0 = predict(np.array([0]))[0]
    hold[:] = x0
    adm1[:] = x0

    cad_steps = max(1, round(roll.cadence_hours / 24.0 / series.dt_days))
    first_idx = max(cad_steps, round(roll.min_window_days / series.dt_days))
    retrain_idx = list(range(first_idx, T, cad_steps))
    n_retrain = 0
    train_seconds = 0.0
    losses = []
    for n, k_r in enumerate(retrain_idx):
        t_r = float(series.time[k_r])
        t_lo = max(0.0, t_r - roll.window_days)
        steps = roll.steps_first if n == 0 else roll.steps_warm
        lr = cfg.lr if n == 0 else roll.lr_warm
        t0 = _time.perf_counter()
        last = fit_window(
            net,
            phys,
            meas,
            series,
            cfg,
            t_lo,
            t_r,
            steps,
            lr,
            gen,
            feed_np,
            y_all,
            t_all,
            u_all,
        )
        train_seconds += _time.perf_counter() - t0
        n_retrain += 1
        losses.append({"t": t_r, **last})
        # current estimate and bridging until the next retraining (or the end)
        k_next = retrain_idx[n + 1] if n + 1 < len(retrain_idx) else T
        x_now = predict(np.array([k_r]))[0]
        hold[k_r:k_next] = x_now
        adm1[k_r] = x_now
        if k_next - k_r > 1:
            if roll.adm1_bridge:
                adm1[k_r + 1 : k_next] = continue_adm1(
                    x_now, t_r, k_next - k_r - 1, series, phys, feed_np
                )
            else:
                adm1[k_r + 1 : k_next] = x_now
        # fixed-lag estimates from this window (smoother inside the window)
        k_lo = int(np.searchsorted(series.time, t_lo - 1e-9))
        for L in roll.lags_days:
            key = f"lag{L:g}"
            k_hi = k_r - round(L / series.dt_days)  # hours with t <= t_r - L
            a, b = max(lag_filled[key], k_lo), k_hi + 1
            if b > a:
                lags[key][a:b] = predict(np.arange(a, b))
                lag_filled[key] = b
        if verbose:
            print(
                f"  retrain {n:4d} at t={t_r:6.2f} d  window [{t_lo:.1f}, {t_r:.1f}]  data {last['data']:.2f} phys {last['phys']:.3f}",
                flush=True,
            )

    # hours never reached by a lag (end of series): fall back to the causal hold estimate
    for key, arr in lags.items():
        m = np.isnan(arr[:, 0])
        arr[m] = hold[m]

    seconds_total = _time.perf_counter() - t_start
    trajectories = {"hold": hold, "adm1": adm1, **lags}
    metrics = {}
    for name, traj in trajectories.items():
        metrics[name] = _score(traj, series)
    metrics["losses"] = losses[-5:]
    return RollingResult(
        series_index=series.index,
        regime=series.regime,
        arch=asdict(arch),
        train=asdict(cfg),
        rolling=asdict(roll),
        n_retrainings=n_retrain,
        seconds_total=seconds_total,
        seconds_per_operating_hour=train_seconds / max(T - 1, 1),
        metrics=metrics,
        trajectories=trajectories,
    )


def _score(traj: np.ndarray, series: BenchmarkSeries) -> dict:
    """Benchmark score of a hourly trajectory over the full series and after day 3."""
    T = len(traj)
    warm = round(3.0 / series.dt_days)
    out = benchmark_scores(
        traj, series, {"full": slice(0, T), "after3d": slice(warm, T)}
    )
    rel = np.abs(traj[:, :N_CORE] - series.states[:, :N_CORE]) / np.maximum(
        np.abs(series.states[:, :N_CORE]), 1e-30
    )
    out["rel_rmse_median_core"] = float(np.median(np.sqrt(np.mean(rel**2, axis=0))))
    return out
