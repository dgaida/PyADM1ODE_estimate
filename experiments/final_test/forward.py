"""Pruned forward model of a test set's plant: kinetics calibration and ADM1 open loop.

Mirrors ``fastsim.ForwardModel`` of the kinetics-calibration study
(``PyADM1ODE_calibration/experiments/kinetics_calibration``): the plant is pruned to
the primary digester and its gas storage, the feed schedule is imposed hour by
hour, and the observables are read before each step. Unlike the study it builds
the plant from the test set's ``meta.json`` (so the substrate swap of
``benchmark_ccm`` / ``benchmark_hard`` is honoured) and never starts from the true
initial state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import (
    _warmup,
    apply_kinetic_factors,
    baseline_flows,
    build_plant,
    prune_to_primary,
    substrates_from_meta,
)
from pyadm1ode_estimation.estimation.quickstart import build_filter_components

DIGESTER = "primary"


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class ForwardModel:
    meta: dict
    plant: object
    process: object
    obs: object
    spec: object
    nominal: dict

    @classmethod
    def build(cls, meta: dict) -> ForwardModel:
        plant = prune_to_primary(build_plant(meta, DIGESTER))
        process, obs, spec = build_filter_components(
            plant,
            digester_id=DIGESTER,
            substrates=substrates_from_meta(meta),
            sensors=[s.lower() for s in meta["sensors"]],
        )
        nominal = {k: float(v) for k, v in plant.components[DIGESTER].adm1._kinetic.items()}
        return cls(meta, plant, process, obs, spec, nominal)

    def set_factors(self, factors: dict[str, float] | None) -> None:
        kinetic = self.plant.components[DIGESTER].adm1._kinetic
        kinetic.update(self.nominal)
        apply_kinetic_factors(self.plant, factors or {}, DIGESTER)

    def simulate(self, x0_adm1: np.ndarray, feed: np.ndarray, n_steps: int, dt_hours: float = 1.0):
        """``(states (n+1, 41), observations (n+1, n_sensor))`` under the imposed feed."""
        dt = dt_hours / 24.0
        adm1_pos = self.spec.kind_indices("adm1")
        aug = self.spec.kind_indices("input_flow")
        x = np.zeros(len(self.spec))
        x[adm1_pos] = np.asarray(x0_adm1, dtype=float)
        states = np.zeros((n_steps + 1, len(adm1_pos)))
        obs_rows = np.zeros((n_steps + 1, len(self.obs.channels)))
        for k in range(n_steps + 1):
            for j, i_aug in enumerate(aug):
                x[i_aug] = feed[min(k, len(feed) - 1), j]
            states[k] = x[adm1_pos]
            self.process.refresh_outputs(x, equilibration_dt=dt)
            obs_rows[k] = [float(c.extractor(self.process.plant, x)) for c in self.obs.channels]
            if k < n_steps:
                x = self.process.step(x, dt)
        return states, obs_rows

    def warmup_state(self) -> np.ndarray:
        """The 41-state after the dataset's warm-up at the baseline feed (as the UKF does)."""
        subs = substrates_from_meta(self.meta)
        x = _warmup(
            self.process,
            self.spec,
            self.spec.read_adm1_state(self.plant),
            float(self.meta.get("warmup_days", 30.0)),
            baseline_flows(self.meta, subs),
            float(self.meta.get("dt_hours", 1.0)),
        )
        return x[self.spec.kind_indices("adm1")]


def sensor_sigma(meta: dict, meas: np.ndarray) -> np.ndarray:
    """Per-cell 1-sigma of the measurements from ``meta["sensor_noise"]``."""
    from pyadm1ode_estimation.estimation.observer.adapter import parse_noise_spec

    sig = np.zeros_like(meas, dtype=float)
    for j, name in enumerate(meta["sensors"]):
        kind, value = parse_noise_spec(meta["sensor_noise"][name])
        sig[:, j] = value * np.abs(meas[:, j]) if kind == "relative" else value
    return np.maximum(sig, 1e-9)


def open_loop(meta: dict, feed: np.ndarray, factors: dict[str, float] | None) -> np.ndarray:
    """R1: the ADM1 with the given kinetics, warm-up start, recorded feed, no correction."""
    fm = ForwardModel.build(meta)
    fm.set_factors(factors)
    x0 = fm.warmup_state()
    states, _ = fm.simulate(x0, feed, len(feed) - 1)
    return states


def calibrate(
    meta: dict,
    measurements: np.ndarray,
    feed: np.ndarray,
    x_start: np.ndarray,
    keys: list[str],
    fit_hours: tuple[int, int],
    budget: int,
    log_bound: float,
    xtol: float = 1e-3,
    ftol: float = 1e-4,
) -> dict:
    """Powell search of the log-factors of ``keys`` on the measurements of ``fit_hours``.

    The simulation starts at hour ``fit_hours[0]`` in ``x_start`` (an estimate, never
    the truth) and is scored as the mean squared noise-normalised residual of all
    sensors over ``[fit_hours[0], fit_hours[1]]``. Inactive kinetics stay nominal.
    """
    h0, h1 = int(fit_hours[0]), int(fit_hours[1])
    fm = ForwardModel.build(meta)
    meas = np.asarray(measurements, float)[h0 : h1 + 1]
    sig = sensor_sigma(meta, meas)
    feed_w = np.asarray(feed, float)[h0 : h1 + 1]
    n_steps = h1 - h0
    state = {"n": 0, "best": np.inf, "best_theta": None, "trace": []}

    def objective(theta: np.ndarray) -> float:
        if state["n"] >= budget:
            raise BudgetExhausted
        state["n"] += 1
        try:
            fm.set_factors({k: float(np.exp(t)) for k, t in zip(keys, theta)})
            _, obs = fm.simulate(x_start, feed_w, n_steps)
            val = float(np.mean(((obs - meas) / sig) ** 2))
            if not np.isfinite(val):
                val = 1e12
        except Exception:  # noqa: BLE001 - an unsolvable parameter set scores as invalid
            val = 1e12
        if val < state["best"]:
            state["best"], state["best_theta"] = val, np.array(theta, float)
            state["trace"].append((state["n"], val))
        return val

    t0 = time.perf_counter()
    theta0 = np.zeros(len(keys))
    chi2_start = objective(theta0)
    message = ""
    try:
        res = minimize(
            objective,
            theta0,
            method="Powell",
            bounds=[(-log_bound, log_bound)] * len(keys),
            options={"maxfev": budget, "xtol": xtol, "ftol": ftol},
        )
        message = str(res.message)
    except BudgetExhausted:
        message = f"budget of {budget} simulations used"
    theta = state["best_theta"] if state["best_theta"] is not None else theta0
    return {
        "factors": {k: float(np.exp(t)) for k, t in zip(keys, theta)},
        "log_factors": {k: float(t) for k, t in zip(keys, theta)},
        "chi2_start": float(chi2_start),
        "chi2_best": float(state["best"]),
        "n_evals": int(state["n"]),
        "trace": state["trace"],
        "message": message,
        "wall_s": time.perf_counter() - t0,
    }
