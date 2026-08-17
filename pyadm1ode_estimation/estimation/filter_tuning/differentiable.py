"""Approach 1.2 — differentiable filter: learn Q and R by gradient descent.

A differentiable **linearised** Kalman filter that learns the diagonal process- and
measurement-noise covariances ``Q, R`` by back-propagating an innovation negative-log-
likelihood (+ an optional supervised state term — we have ground truth) through the whole
filter recursion (Barratt–Boyd 2020 / BackpropKF 2016).

Why linearised: the ADM1 ODE is **stiff**, so an explicit differentiable integrator is
unstable (blows up unless you take hundreds of sub-steps). Instead we linearise around the
**stable** numpy (BDF) open-loop reference trajectory ``x̄_k`` and build a stiffness-robust
discrete transition Jacobian from the *matrix exponential* of the continuous ADM1 Jacobian,
``F_k = expm(A_k · dt)`` with ``A_k = ∂/∂x adm1da_rhs_torch(x̄_k)``. The observation Jacobian
``H_k`` comes from the differentiable torch sensor model. ``F_k, H_k, x̄_k`` are constants
(no grad); only ``Q, R`` are learnable, so the recursion is cheap linear algebra and stable.

    trainer = DifferentiableEKF(ds.meta)
    trainer.fit(train, val, epochs=15, days=8, lr=0.1)
    theta = trainer.as_theta()          # {"q_diag":..., "r_diag":...}
"""

from __future__ import annotations

import dataclasses

import numpy as np
from pyadm1.core.adm1 import STATE_SIZE
from pyadm1.core.adm1_torch import Adm1TorchParams, adm1da_rhs_torch

from ..deep_learning.observation_torch import TorchObservationModel
from ..quickstart import build_filter_components
from .filter_runners import (
    _propagate_open_loop,
    _warmup,
    build_plant,
    substrates_from_meta,
)


def _torch():
    import torch

    return torch


class DifferentiableEKF:
    """Differentiable linearised KF whose diagonal Q, R are learned by gradient."""

    def __init__(
        self, meta: dict, digester_id: str = "primary", supervised_weight: float = 1.0
    ):
        torch = _torch()
        self.meta = meta
        self.did = digester_id
        self.dt_days = float(meta.get("dt_hours", 1.0)) / 24.0
        self.w_sup = float(supervised_weight)
        self.substrates = substrates_from_meta(meta)
        self.sub_idx = [s.substrate_index for s in self.substrates]
        self.sensors = [s.lower() for s in meta["sensors"]]
        self.channel_names = list(meta["sensors"])

        plant = build_plant(meta, digester_id)
        self.adm1 = plant.components[digester_id].adm1
        self.feedstock = self.adm1._feedstock
        chan = {name: _noise_of(name) for name in meta["sensors"]}
        self.obs = TorchObservationModel.from_adm1(
            self.adm1, chan, quasi_steady_gas=True
        )
        self.p_base = Adm1TorchParams.from_adm1(self.adm1)

        self.log_q = torch.nn.Parameter(
            torch.full((STATE_SIZE,), float(np.log(1e-4)), dtype=torch.float64)
        )
        self.log_r = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    [max(s, 1e-6) ** 2 for s in self.obs.noise_std], dtype=torch.float64
                )
            )
        )

    # -- per-step feed -> ADM1 params -----------------------------------
    def _params_at(self, feed_row: np.ndarray) -> Adm1TorchParams:
        df = self.feedstock.get_influent_dataframe(list(feed_row))
        s_in = [float(v) for v in df.iloc[0].tolist()[: len(self.p_base.s_in)]]
        return dataclasses.replace(self.p_base, q_ad=float(feed_row.sum()), s_in=s_in)

    # -- stable reference + linearisation (no grad, cached per episode) --
    def linearize(self, episode):
        torch = _torch()
        feed = np.asarray(episode.obs["feed_noisy"], float)
        n_steps = len(feed) - 1
        plant = build_plant(self.meta, self.did)
        wp, wobs, wspec = build_filter_components(
            plant,
            digester_id=self.did,
            substrates=self.substrates,
            sensors=self.sensors,
        )
        base = np.array([s.initial_flow for s in self.substrates], float)
        base = base * (
            float(
                self.meta.get("operating_point", {}).get(
                    "baseline_total_m3_per_d", base.sum()
                )
            )
            / max(base.sum(), 1e-9)
        )
        xw = _warmup(
            wp,
            wspec,
            wspec.read_adm1_state(plant),
            float(self.meta.get("warmup_days", 30.0)),
            base,
            self.meta.get("dt_hours", 1.0),
        )
        ol = _propagate_open_loop(
            wspec, wp, wobs, xw.copy(), feed, self.meta.get("dt_hours", 1.0), n_steps
        )
        xbar = ol[:, wspec.kind_indices("adm1")]  # (T, 41) stable reference

        Fs, Hs, hbars = [], [], []
        with torch.no_grad():
            for k in range(len(feed)):
                xk = torch.tensor(xbar[k], dtype=torch.float64)
                p = self._params_at(feed[k])
                A = torch.autograd.functional.jacobian(
                    lambda z, p=p: adm1da_rhs_torch(z, p), xk, vectorize=True
                )
                Fs.append(torch.linalg.matrix_exp(A * self.dt_days))
                obs_k = dataclasses.replace(self.obs, params=p)
                Hs.append(
                    torch.autograd.functional.jacobian(
                        obs_k.predict, xk, vectorize=True
                    )
                )
                hbars.append(obs_k.predict(xk))
        return {
            "xbar": torch.tensor(xbar, dtype=torch.float64),
            "F": Fs,
            "H": Hs,
            "hbar": torch.stack(hbars),
            "meas": torch.tensor(np.asarray(episode.obs["measurements"], float)),
            "truth": torch.tensor(np.asarray(episode.truth, float)),
            "burnin": int(getattr(episode, "burnin", 0) or 0),
        }

    # -- differentiable linear KF over deviations δx = x - x̄ ------------
    def filter_episode(self, episode, lin=None, accumulate_loss=True):
        torch = _torch()
        L = lin or self.linearize(episode)
        xbar, F, H, hbar, meas, truth, b = (
            L["xbar"],
            L["F"],
            L["H"],
            L["hbar"],
            L["meas"],
            L["truth"],
            L["burnin"],
        )
        Q = torch.diag(torch.exp(self.log_q))
        R = torch.diag(torch.exp(self.log_r))
        I = torch.eye(STATE_SIZE, dtype=torch.float64)
        dx = torch.zeros(STATE_SIZE, dtype=torch.float64)
        P = I * 1e-3
        xs, loss = [], torch.zeros((), dtype=torch.float64)
        T = len(F)
        for k in range(T):
            if k > 0:
                dx = F[k - 1] @ dx
                P = F[k - 1] @ P @ F[k - 1].T + Q
            nu = meas[k] - hbar[k] - H[k] @ dx
            S = H[k] @ P @ H[k].T + R
            Sinv = torch.linalg.inv(S)
            K = P @ H[k].T @ Sinv
            dx = dx + K @ nu
            P = (I - K @ H[k]) @ P
            xk = xbar[k] + dx
            xs.append(xk)
            if accumulate_loss and k >= b:
                nll = 0.5 * (nu @ Sinv @ nu + torch.logdet(S))
                sup = self.w_sup * torch.mean(
                    ((xk - truth[k]) / (truth[k].abs() + 1e-6)) ** 2
                )
                loss = loss + nll + sup
        return torch.stack(xs), loss / max(T - b, 1)

    # -- training --------------------------------------------------------
    def fit(
        self, train_eps, val_eps=None, epochs=15, lr=0.1, grad_clip=5.0, verbose=True
    ):
        torch = _torch()
        lin_tr = [self.linearize(e) for e in train_eps]  # stable linearisation, once
        lin_va = [self.linearize(e) for e in (val_eps or [])]
        opt = torch.optim.Adam([self.log_q, self.log_r], lr=lr)
        best = {
            "val": float("inf"),
            "log_q": self.log_q.detach().clone(),
            "log_r": self.log_r.detach().clone(),
        }
        history = []
        for ep in range(epochs):
            opt.zero_grad()
            total = torch.stack(
                [self.filter_episode(None, L)[1] for L in lin_tr]
            ).mean()
            total.backward()
            torch.nn.utils.clip_grad_norm_([self.log_q, self.log_r], grad_clip)
            opt.step()
            if lin_va:
                with torch.no_grad():
                    vloss = float(
                        torch.stack(
                            [self.filter_episode(None, L)[1] for L in lin_va]
                        ).mean()
                    )
            else:
                vloss = float(total.detach())
            history.append({"epoch": ep, "train": float(total.detach()), "val": vloss})
            if verbose:
                print(
                    f"  epoch {ep}: train {float(total.detach()):.3f} | val {vloss:.3f}",
                    flush=True,
                )
            if vloss < best["val"]:
                best = {
                    "val": vloss,
                    "log_q": self.log_q.detach().clone(),
                    "log_r": self.log_r.detach().clone(),
                }
        with torch.no_grad():
            self.log_q.copy_(best["log_q"])
            self.log_r.copy_(best["log_r"])
        return best, history

    def as_theta(self) -> dict:
        torch = _torch()
        return {
            "q_diag": torch.exp(self.log_q).detach().numpy(),
            "r_diag": torch.exp(self.log_r).detach().numpy(),
        }


def _noise_of(name: str) -> float:
    return float(
        {"Q_gas": 60.0, "Q_ch4": 40.0, "Q_co2": 30.0, "pH": 0.02, "TS": 0.2}.get(
            name, 1.0
        )
    )
