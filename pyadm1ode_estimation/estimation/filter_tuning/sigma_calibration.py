"""Post-hoc per-state σ recalibration (method=sigma) — generic over estimator & dataset.

Runs the estimator once per episode (the expensive step), then fits per-state output-σ
multipliers ``std_scale = clip(√NEES, 0.3, σ_hi)·γ`` on TRAIN residuals and selects
``(σ_hi, γ)`` on VAL. Only the reported σ changes — the point estimate x̂ is untouched,
so the search over ``(σ_hi, γ)`` is free (no re-runs). NaN std (states the estimator does
not estimate, e.g. the A+D non-core) keep scale 1.0 and are excluded from coverage.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import metrics as M
from .filter_runners import collect_parallel

Residual = tuple[
    np.ndarray, np.ndarray, np.ndarray
]  # (truth, x_hat, std), burn-in removed


def collect(
    runner,
    episodes: list,
    theta: dict | None = None,
    jobs: int = 1,
    drop_failed: bool = True,
) -> list[Residual]:
    """Run the (picklable) runner on each episode; return burn-in-trimmed (truth, x̂, σ).

    ``jobs>1`` runs the episodes across processes. Episodes whose filter run failed come
    back as ``None`` from the worker and are dropped (``drop_failed``) so one unstable
    series cannot invalidate the whole collection; set ``drop_failed=False`` to see them.
    """
    res = collect_parallel(runner, episodes, theta or {}, jobs)
    if not drop_failed:
        return res
    ok = [r for r in res if r is not None]
    if len(ok) < len(res):
        print(
            f"    warn: {len(res) - len(ok)}/{len(res)} Episoden fehlgeschlagen",
            flush=True,
        )
    return ok


def _stack(res: list[Residual]):
    return (
        np.concatenate([r[0] for r in res]),
        np.concatenate([r[1] for r in res]),
        np.concatenate([r[2] for r in res]),
    )


def estimated_mask(res: list[Residual]) -> np.ndarray:
    """Indices of states the estimator actually estimates (finite σ)."""
    sd = res[0][2]
    return np.where(np.isfinite(sd).all(axis=0))[0]


def fit_sigma_scale(
    train_res: list[Residual], sigma_hi: float, gamma: float
) -> np.ndarray:
    """Per-state std multiplier from TRAIN NEES; NaN-σ states keep scale 1.0."""
    tru, xh, sd = _stack(train_res)
    nees = M.nees_per_state(tru, xh, sd)
    scale = np.clip(np.sqrt(np.maximum(nees, 1e-12)), 0.3, sigma_hi) * gamma
    return np.where(np.isfinite(nees), scale, 1.0)


DEFAULT_HI = (10.0, 20.0, 30.0, 50.0)
DEFAULT_GAMMA = (1.0, 1.3, 1.7, 2.2)


def search_sigma(train_res, val_res, hi_grid=DEFAULT_HI, gamma_grid=DEFAULT_GAMMA):
    """Grid over (σ_hi, γ): fit on train, score on val. Returns (best, all_sorted)."""
    mask = estimated_mask(train_res)
    tru_v, xh_v, sd_v = _stack(val_res)
    out = []
    for hi, g in itertools.product(hi_grid, gamma_grid):
        scale = fit_sigma_scale(train_res, hi, g)
        m = M.evaluate(tru_v, xh_v, sd_v * scale[None, :], state_mask=mask)
        out.append(
            {
                "sigma_hi": hi,
                "gamma": g,
                "std_scale": scale,
                "score": M.objective(m),
                **m,
            }
        )
    out.sort(key=lambda r: -r["score"])
    return out[0], out


def apply_to(res: list[Residual], std_scale: np.ndarray) -> dict:
    """Evaluate a fitted std_scale on a residual set (e.g. TEST)."""
    tru, xh, sd = _stack(res)
    mask = estimated_mask(res)
    return M.evaluate(
        tru, xh, sd * np.asarray(std_scale, float)[None, :], state_mask=mask
    )
