"""Substrate feed -> ADM1da influent ``(q_ad, s_in)``, vectorised over time / batch.

The ADM1da right-hand side needs two influent quantities per collocation point:
the total volumetric inflow ``q_ad`` [m3/d] and the 37-dim influent
concentration vector ``s_in``. A PINN evaluates the physics residual on a
*batch* of time points with *different* feeds, so this module turns a feed
array ``(..., n_sub)`` into ``q_ad (...,)`` and ``s_in (..., 37)`` with plain
array algebra (numpy or torch, no per-row object calls).

Two composition modes exist because the benchmark truth was **not** generated
with a flow-weighted blend:

``"nominal_mix"``
    ``pyadm1.Digester.set_feedstock`` attaches an influent DataFrame built from
    the *nominal* substrate mix, and ``ADM1.create_influent`` then reads the
    composition from that DataFrame regardless of the per-step ``Q``. Only the
    total flow ``q_ad = sum(Q_j * factor_j)`` follows the feed. This is what the
    benchmark dataset's truth trajectories obey, so it is the default: a physics
    residual with the "correct" blend would be non-zero on the truth.

``"blend"``
    Flow-weighted blend of the per-substrate concentrations,
    ``s_in = sum_j q_j c_j / sum_j q_j`` — what ``Feedstock.get_influent_dataframe``
    returns for a given ``Q``. Physically right; use it for datasets or plants
    whose composition really varies.

``factor_j`` is the ADM1da mass-to-volume conversion (``1000 / rho_j`` under the
SIMBA Q convention, else 1), read from ``Feedstock.q_conversion_factors``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pyadm1.core.adm1 import INFLUENT_COLUMNS

#: Length of the ADM1da influent vector (indices 0-36; no gas-phase states).
N_INFLUENT = 37

MODES = ("nominal_mix", "blend")

_INFLUENT_KEYS = INFLUENT_COLUMNS[:-1]  # drop the trailing "Q" column


def _is_torch(a: Any) -> bool:
    return type(a).__module__.split(".")[0] == "torch"


@dataclass(frozen=True)
class InfluentModel:
    """Map substrate flows to the ADM1da influent, batch-wise.

    Attributes:
        conc: ``(n_sub, 37)`` influent concentrations of each substrate alone.
        q_factors: ``(n_sub,)`` volume-conversion factors of the substrates.
        nominal_s_in: ``(37,)`` composition the digester actually sees in
            ``"nominal_mix"`` mode.
        mode: ``"nominal_mix"`` or ``"blend"`` (see module docstring).
    """

    conc: np.ndarray
    q_factors: np.ndarray
    nominal_s_in: np.ndarray
    mode: str = "nominal_mix"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.conc.shape != (len(self.q_factors), N_INFLUENT):
            raise ValueError(
                f"conc must be (n_sub, {N_INFLUENT}), got {self.conc.shape}"
            )
        if self.nominal_s_in.shape != (N_INFLUENT,):
            raise ValueError(
                f"nominal_s_in must be ({N_INFLUENT},), got {self.nominal_s_in.shape}"
            )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def from_adm1(cls, adm1, mode: str = "nominal_mix") -> InfluentModel:
        """Read the substrate characterisation from an :class:`pyadm1.core.adm1.ADM1`.

        The nominal composition is taken from the influent DataFrame the digester
        attached (row 0), falling back to the last ``create_influent`` result.
        """
        fs = adm1.feedstock
        if fs is None:
            raise ValueError("ADM1 instance has no feedstock attached")
        conc = np.array(
            [
                [float(c.get(k, 0.0)) for k in _INFLUENT_KEYS]
                for c in fs.concentrations_list
            ],
            dtype=float,
        )
        q_factors = np.asarray(fs.q_conversion_factors, dtype=float)

        if getattr(adm1, "_influent_df", None) is not None:
            nominal = adm1._influent_df.iloc[0][_INFLUENT_KEYS].to_numpy(dtype=float)
        elif getattr(adm1, "_state_input", None) is not None:
            nominal = np.asarray(adm1._state_input, dtype=float)
        elif mode == "blend":
            nominal = np.zeros(N_INFLUENT)
        else:
            raise ValueError(
                "'nominal_mix' mode needs a digester with an influent DataFrame or a created influent"
            )
        return cls(conc=conc, q_factors=q_factors, nominal_s_in=nominal, mode=mode)

    @classmethod
    def from_plant(
        cls, plant, digester_id: str = "primary", mode: str = "nominal_mix"
    ) -> InfluentModel:
        """Convenience: :meth:`from_adm1` on ``plant.components[digester_id].adm1``."""
        return cls.from_adm1(plant.components[digester_id].adm1, mode=mode)

    # ------------------------------------------------------------------
    @property
    def n_substrates(self) -> int:
        return len(self.q_factors)

    def __call__(self, feed):
        """Return ``(q_ad, s_in)`` for ``feed`` of shape ``(..., n)`` with ``n <= n_sub``.

        Accepts a numpy array or a torch tensor and returns the same kind
        (torch results share dtype / device with ``feed`` and stay differentiable
        w.r.t. it). Columns beyond the feed's width are treated as zero flow.
        """
        n = feed.shape[-1]
        if n > self.n_substrates:
            raise ValueError(
                f"feed has {n} columns but the model knows {self.n_substrates} substrates"
            )

        if _is_torch(feed):
            import torch

            factors = torch.as_tensor(
                self.q_factors[:n], dtype=feed.dtype, device=feed.device
            )
            conc = torch.as_tensor(self.conc[:n], dtype=feed.dtype, device=feed.device)
            nominal = torch.as_tensor(
                self.nominal_s_in, dtype=feed.dtype, device=feed.device
            )
            q_conv = feed * factors
            q_ad = q_conv.sum(dim=-1)
            if self.mode == "blend":
                denom = torch.clamp(q_ad, min=1.0e-12).unsqueeze(-1)
                s_in = (q_conv @ conc) / denom
                s_in = torch.where(
                    q_ad.unsqueeze(-1) > 0.0, s_in, torch.zeros_like(s_in)
                )
            else:
                s_in = nominal.expand(*feed.shape[:-1], N_INFLUENT)
            return q_ad, s_in

        feed = np.asarray(feed, dtype=float)
        q_conv = feed * self.q_factors[:n]
        q_ad = q_conv.sum(axis=-1)
        if self.mode == "blend":
            denom = np.maximum(q_ad, 1.0e-12)[..., None]
            s_in = (q_conv @ self.conc[:n]) / denom
            s_in = np.where(q_ad[..., None] > 0.0, s_in, 0.0)
        else:
            s_in = np.broadcast_to(
                self.nominal_s_in, feed.shape[:-1] + (N_INFLUENT,)
            ).copy()
        return q_ad, s_in
