"""Trajectory network ``x(t, u)`` for the PINN v2 structure study.

Every structural choice is a field of :class:`ArchConfig`, so a sweep can name a
structure as a string (``d4_w128_silu_ff4_hist7_blockheads_ln``) and rebuild it
with :meth:`ArchConfig.parse`.

Inputs are the normalised time, optional Fourier features of it, and the feed
features (current feed, or a feed history). The output is mapped to positive
concentrations through a per-state transform (log-affine by default) so the
network works in a zero-mean, unit-variance space regardless of the five
orders of magnitude between S_h2 and X_PS.

Families
--------
* activation: ``tanh`` | ``silu`` | ``gelu`` | ``sin`` (SIREN initialisation)
* time encoding: ``fourier_features`` (deterministic pairs) and/or
  ``rff_features`` (random Gaussian Fourier features, Tancik et al. 2020)
* feed input: ``feed_mode`` ``current`` | ``hist7`` | ``hist30`` | ``none``
  (features are built by :func:`~.data.feed_features`)
* output structure: ``shared`` | ``blockheads`` (one head per state block) |
  ``multiscale`` (a slow net for the particulate/biomass pools without time
  encoding plus a fast net for the dissolved states with it)
* stability: ``layer_norm``, ``residual`` blocks
* output transform: ``logaffine`` | ``softplus`` | ``direct``
* ``modified_mlp``: the two-encoder gated MLP of Wang, Teng & Perdikaris (2021)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTIVATIONS = ("tanh", "silu", "gelu", "sin")
FEED_MODES = ("current", "hist7", "hist30", "none")
OUTPUTS = ("shared", "blockheads", "multiscale")
TRANSFORMS = ("logaffine", "softplus", "direct")

#: State blocks of the 31-state core (observability spec, restricted to < 31).
CORE_BLOCKS: dict[str, tuple[int, ...]] = {
    "acidogenesis_substrates": (0, 1, 3, 4, 5),
    "fa_block": (2, 24),
    "methanogenesis": (6, 7, 8, 9, 27, 28),
    "nitrogen": (10,),
    "inerts": (11, 21),
    "disintegration_split": (12, 13, 14, 15, 16, 17),
    "hydrolysis_sums": (18, 19, 20),
    "acidogenesis_biomass": (22, 23, 25, 26),
    "charge_balance": (29, 30),
}
#: Multi-scale split of the core: slow pools vs. fast dissolved states.
SLOW_STATES: tuple[int, ...] = tuple(range(11, 31))
FAST_STATES: tuple[int, ...] = tuple(range(11))

_TOKEN = re.compile(
    r"^(d\d+|w\d+|tanh|silu|gelu|sin|ff\d+|rff\d*|hist7|hist30|nofeed|shared|blockheads|multiscale|ln|res|logaffine|softplus|direct|modmlp)$"
)


@dataclass(frozen=True)
class ArchConfig:
    """Architecture knobs of :class:`TrajectoryNet`."""

    depth: int = 4  # hidden layers
    width: int = 128
    activation: str = "tanh"
    fourier_features: int = 0  # K deterministic sin/cos pairs of the normalised time
    rff_features: int = 0  # K random Gaussian Fourier pairs of the normalised time
    rff_sigma: float = 2.0
    feed_mode: str = "current"
    output: str = "shared"
    layer_norm: bool = False
    residual: bool = False
    transform: str = "logaffine"
    modified_mlp: bool = False

    def __post_init__(self) -> None:
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {ACTIVATIONS}")
        if self.feed_mode not in FEED_MODES:
            raise ValueError(f"feed_mode must be one of {FEED_MODES}")
        if self.output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}")
        if self.transform not in TRANSFORMS:
            raise ValueError(f"transform must be one of {TRANSFORMS}")

    @property
    def name(self) -> str:
        s = f"d{self.depth}_w{self.width}_{self.activation}"
        if self.fourier_features:
            s += f"_ff{self.fourier_features}"
        if self.rff_features:
            s += f"_rff{self.rff_features}"
        if self.feed_mode == "none":
            s += "_nofeed"
        elif self.feed_mode != "current":
            s += f"_{self.feed_mode}"
        if self.output != "shared":
            s += f"_{self.output}"
        if self.layer_norm:
            s += "_ln"
        if self.residual:
            s += "_res"
        if self.transform != "logaffine":
            s += f"_{self.transform}"
        if self.modified_mlp:
            s += "_modmlp"
        return s

    @classmethod
    def parse(cls, name: str) -> ArchConfig:
        """Inverse of :attr:`name` (tokens in any order, e.g. ``d4_w128_silu_ff4_blockheads``)."""
        kw: dict = {}
        for tok in name.split("_"):
            if not _TOKEN.match(tok):
                raise ValueError(f"unknown architecture token {tok!r} in {name!r}")
            if tok.startswith("d") and tok[1:].isdigit():
                kw["depth"] = int(tok[1:])
            elif tok.startswith("w") and tok[1:].isdigit():
                kw["width"] = int(tok[1:])
            elif tok in ACTIVATIONS:
                kw["activation"] = tok
            elif tok.startswith("ff"):
                kw["fourier_features"] = int(tok[2:])
            elif tok.startswith("rff"):
                kw["rff_features"] = int(tok[3:]) if tok[3:] else 16
            elif tok in ("hist7", "hist30"):
                kw["feed_mode"] = tok
            elif tok == "nofeed":
                kw["feed_mode"] = "none"
            elif tok in OUTPUTS:
                kw["output"] = tok
            elif tok == "ln":
                kw["layer_norm"] = True
            elif tok == "res":
                kw["residual"] = True
            elif tok in TRANSFORMS:
                kw["transform"] = tok
            elif tok == "modmlp":
                kw["modified_mlp"] = True
        return cls(**kw)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class Sine(nn.Module):
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


def _activation(name: str, first: bool = False) -> nn.Module:
    if name == "sin":
        return Sine(30.0 if first else 1.0)
    return {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}[name]()


def _siren_init(layer: nn.Linear, first: bool, w0: float = 30.0) -> None:
    n = layer.in_features
    bound = 1.0 / n if first else math.sqrt(6.0 / n) / w0
    with torch.no_grad():
        layer.weight.uniform_(-bound, bound)
        layer.bias.uniform_(-bound, bound)


class MLP(nn.Module):
    """Hidden stack with optional LayerNorm, residual blocks and the modified-MLP gating."""

    def __init__(self, n_in: int, arch: ArchConfig, width: int | None = None):
        super().__init__()
        w = width or arch.width
        self.arch = arch
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = n_in
        for i in range(arch.depth):
            lin = nn.Linear(d, w)
            if arch.activation == "sin":
                _siren_init(lin, first=(i == 0))
            self.layers.append(lin)
            self.acts.append(_activation(arch.activation, first=(i == 0)))
            self.norms.append(nn.LayerNorm(w) if arch.layer_norm else nn.Identity())
            d = w
        if arch.modified_mlp:
            self.enc_u = nn.Linear(n_in, w)
            self.enc_v = nn.Linear(n_in, w)
            self.enc_act = _activation(arch.activation, first=True)
        self.out_features = w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch.modified_mlp:
            u = self.enc_act(self.enc_u(x))
            v = self.enc_act(self.enc_v(x))
        h = x
        for i, (lin, act, norm) in enumerate(zip(self.layers, self.acts, self.norms)):
            z = norm(act(lin(h)))
            if self.arch.modified_mlp:
                z = (1.0 - z) * u + z * v
            h = h + z if (self.arch.residual and i > 0) else z
        return h


class Heads(nn.Module):
    """Output heads: one linear layer (``shared``) or one small head per state block."""

    def __init__(
        self,
        n_hidden: int,
        n_out: int,
        arch: ArchConfig,
        indices: tuple[int, ...] | None = None,
    ):
        super().__init__()
        self.n_out = n_out
        self.indices = tuple(range(n_out)) if indices is None else tuple(indices)
        self.arch = arch
        if arch.output == "blockheads":
            self.blocks = [
                tuple(i for i in idx if i in self.indices)
                for idx in CORE_BLOCKS.values()
            ]
            self.blocks = [b for b in self.blocks if b]
            self.heads = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(n_hidden, n_hidden // 2),
                    _activation(arch.activation),
                    nn.Linear(n_hidden // 2, len(b)),
                )
                for b in self.blocks
            )
            for h in self.heads:
                with torch.no_grad():
                    h[-1].weight.mul_(0.1)
                    h[-1].bias.zero_()
        else:
            self.head = nn.Linear(n_hidden, len(self.indices))
            with torch.no_grad():
                self.head.weight.mul_(0.1)
                self.head.bias.zero_()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns ``(..., n_out)`` with zeros in rows this head does not own."""
        out = h.new_zeros(*h.shape[:-1], self.n_out)
        if self.arch.output == "blockheads":
            for b, head in zip(self.blocks, self.heads):
                out[..., list(b)] = head(h)
        else:
            out[..., list(self.indices)] = self.head(h)
        return out


# ---------------------------------------------------------------------------
# The trajectory network
# ---------------------------------------------------------------------------
class TrajectoryNet(nn.Module):
    """``(t, feed features) -> x`` with a positive output transform.

    Args:
        arch: Architecture.
        n_out: Number of state outputs (31 in closure mode, 41 in DAE mode).
        n_feed: Number of feed feature columns (see :func:`~.data.feed_features`).
        t_scale: Time normalisation ``t_norm = t / t_scale`` [days].
        feed_scale: ``(n_feed,)`` feed feature normalisation.
        log_mu, log_sigma: ``(n_out,)`` per-state mean / std of ``log x`` (prior).
    """

    def __init__(
        self,
        arch: ArchConfig,
        n_out: int,
        n_feed: int,
        t_scale: float,
        feed_scale: np.ndarray,
        log_mu: np.ndarray,
        log_sigma: np.ndarray,
        seed: int = 0,
    ):
        super().__init__()
        self.arch = arch
        self.n_out = int(n_out)
        self.n_feed = int(n_feed)
        self.register_buffer("t_scale", torch.tensor(float(t_scale)))
        self.register_buffer(
            "feed_scale", torch.as_tensor(np.asarray(feed_scale, float))
        )
        mu = np.asarray(log_mu, float)
        sigma = np.asarray(log_sigma, float)
        self.register_buffer("log_mu", torch.as_tensor(mu))
        self.register_buffer("log_sigma", torch.as_tensor(sigma))
        # linear-space prior for the softplus / direct transforms
        mean_x = np.exp(mu + 0.5 * sigma**2)
        std_x = mean_x * np.sqrt(np.expm1(sigma**2))
        self.register_buffer("lin_a", torch.as_tensor(mean_x))
        self.register_buffer("lin_b", torch.as_tensor(np.maximum(std_x, 1e-30)))
        if arch.fourier_features:
            k = torch.arange(1, arch.fourier_features + 1, dtype=torch.float64)
            self.register_buffer("fourier_k", 2.0 * math.pi * k)
        if arch.rff_features:
            g = torch.Generator().manual_seed(seed)
            B = (
                torch.randn(arch.rff_features, generator=g, dtype=torch.float64)
                * arch.rff_sigma
            )
            self.register_buffer("rff_B", 2.0 * math.pi * B)

        n_time = 1 + 2 * arch.fourier_features + 2 * arch.rff_features
        n_in = n_time + self.n_feed
        if arch.output == "multiscale":
            self.slow = MLP(1 + self.n_feed, arch)  # plain time + feed, no encoding
            self.fast = MLP(n_in, arch)
            self.slow_head = Heads(
                self.slow.out_features,
                self.n_out,
                arch,
                indices=tuple(i for i in SLOW_STATES if i < self.n_out)
                + tuple(range(31, self.n_out)),
            )
            self.fast_head = Heads(
                self.fast.out_features,
                self.n_out,
                arch,
                indices=tuple(i for i in FAST_STATES if i < self.n_out),
            )
        else:
            self.trunk = MLP(n_in, arch)
            self.head = Heads(self.trunk.out_features, self.n_out, arch)

    # -- features ----------------------------------------------------------
    def time_features(self, t: torch.Tensor, encoded: bool = True) -> torch.Tensor:
        tn = t / self.t_scale
        parts = [tn]
        if encoded and self.arch.fourier_features:
            ang = tn * self.fourier_k
            parts += [torch.sin(ang), torch.cos(ang)]
        if encoded and self.arch.rff_features:
            ang = tn * self.rff_B
            parts += [torch.sin(ang), torch.cos(ang)]
        return torch.cat(parts, dim=-1)

    def features(
        self, t: torch.Tensor, u: torch.Tensor, encoded: bool = True
    ) -> torch.Tensor:
        parts = [self.time_features(t, encoded)]
        if self.n_feed:
            parts.append(u / self.feed_scale)
        return torch.cat(parts, dim=-1)

    # -- latent and output -------------------------------------------------
    def latent(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Normalised state ``z`` (the network's own output space)."""
        if self.arch.output == "multiscale":
            return self.slow_head(
                self.slow(self.features(t, u, encoded=False))
            ) + self.fast_head(self.fast(self.features(t, u)))
        return self.head(self.trunk(self.features(t, u)))

    def from_latent(self, z: torch.Tensor) -> torch.Tensor:
        tr = self.arch.transform
        if tr == "logaffine":
            return torch.exp(self.log_mu + self.log_sigma * z)
        if tr == "softplus":
            return F.softplus(self.lin_a + self.lin_b * z) + 1e-30
        return torch.clamp(self.lin_a + self.lin_b * z, min=1e-30)

    def to_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse of the output transform (for state targets and priors)."""
        tr = self.arch.transform
        x = torch.clamp(x, min=1e-30)
        if tr == "logaffine":
            return (torch.log(x) - self.log_mu) / self.log_sigma
        if tr == "softplus":
            inv = x + torch.log(-torch.expm1(-x))  # softplus^-1
            return (inv - self.lin_a) / self.lin_b
        return (x - self.lin_a) / self.lin_b

    def forward(self, t: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """States ``(B, n_out)`` for ``t (B, 1)`` and feed features ``u (B, n_feed)``."""
        return self.from_latent(self.latent(t, u))

    def with_rates(
        self, t: torch.Tensor, u: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, dx/dt)`` by forward-mode AD over ``t`` (``u`` is piecewise constant in ``t``)."""
        return torch.func.jvp(
            lambda tt: self.forward(tt, u), (t,), (torch.ones_like(t),)
        )

    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
