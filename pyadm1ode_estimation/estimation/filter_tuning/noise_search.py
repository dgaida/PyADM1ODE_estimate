"""Approach 1.1 — search the process/measurement noise (Q, R) and initial covariance (P0).

Unlike the σ-recalibration, changing Q/R/P0 changes the Kalman gain → the point estimate
x̂, so each candidate needs a fresh estimator run. We parametrise Q as **per-block scales**
(``estimation.specs.BLOCK_INDICES``) plus a global R-scale and P0-scale, search them on a
held-out validation set (grid / random / Bayesian optimisation), and for each candidate
still apply the free σ-recalibration on top. Objective = :func:`metrics.objective`
(coverage → 0.955 + FOS/TAC-band coverage + critical balanced accuracy).

    from .filter_runners import make_ukf_runner
    from .noise_search import search_noise, default_space
    run = make_ukf_runner(ds.meta, variant="full")
    best, hist = search_noise(run, train, val, default_space(), method="random", n_iter=20)
"""

from __future__ import annotations

import itertools

import numpy as np

from . import sigma_calibration as sig


def default_space(blocks=None, q_range=(0.3, 10.0)) -> dict:
    """Log-uniform search ranges: a Q scale for **every** observability block, plus a
    global R and P0 scale.

    Covers all 9 blocks of :data:`BLOCK_INDICES` by default. (An earlier version searched
    only 4 of them, which froze 15 of the 41 states at their nominal Q — harmless for the
    18-state A+D core, whose states are exactly methanogenesis + charge_balance, but a
    real handicap for the full 41-state filter.) Pass ``blocks`` to restrict the space,
    e.g. ``list(AD_CORE_BLOCKS)`` when tuning the reduced filter.
    """
    from ..specs import BLOCK_INDICES

    names = list(BLOCK_INDICES) if blocks is None else list(blocks)
    return {
        "q_blocks": {b: tuple(q_range) for b in names},  # scale on each block's Q σ
        "r_scale": (0.5, 3.0),
        "p0_scale": (0.3, 3.0),
    }


#: The two blocks the A+D core actually estimates (for a matched, smaller search space).
AD_CORE_BLOCKS = ("methanogenesis", "charge_balance")


def _sample(space: dict, rng) -> dict:
    lg = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    return {
        "q_scale": {b: lg(*r) for b, r in space["q_blocks"].items()},
        "r_scale": lg(*space["r_scale"]),
        "p0_scale": lg(*space["p0_scale"]),
    }


def _grid(space: dict, points: int = 3) -> list[dict]:
    dims, ranges = [], []
    for b, r in space["q_blocks"].items():
        dims.append(("q", b))
        ranges.append(np.geomspace(r[0], r[1], points))
    dims.append(("r", None))
    ranges.append(np.geomspace(*space["r_scale"], points))
    dims.append(("p0", None))
    ranges.append(np.geomspace(*space["p0_scale"], points))
    out = []
    for combo in itertools.product(*ranges):
        th = {"q_scale": {}, "r_scale": 1.0, "p0_scale": 1.0}
        for (kind, b), v in zip(dims, combo):
            if kind == "q":
                th["q_scale"][b] = float(v)
            elif kind == "r":
                th["r_scale"] = float(v)
            else:
                th["p0_scale"] = float(v)
        out.append(th)
    return out


def _lhs(space: dict, n_iter: int, seed: int) -> list[dict]:
    """Latin-Hypercube samples (log-uniform per dimension).

    With every block in the space the search is ~11-dimensional; plain random sampling
    leaves large holes at a small budget, so we stratify each dimension instead. Needs
    only scipy (no scikit-optimize).
    """
    from scipy.stats import qmc

    blocks = list(space["q_blocks"])
    lo = np.log(
        [
            *(space["q_blocks"][b][0] for b in blocks),
            space["r_scale"][0],
            space["p0_scale"][0],
        ]
    )
    hi = np.log(
        [
            *(space["q_blocks"][b][1] for b in blocks),
            space["r_scale"][1],
            space["p0_scale"][1],
        ]
    )
    pts = np.exp(
        qmc.scale(qmc.LatinHypercube(d=len(lo), seed=seed).random(n_iter), lo, hi)
    )
    return [
        {
            "q_scale": {b: float(p[i]) for i, b in enumerate(blocks)},
            "r_scale": float(p[-2]),
            "p0_scale": float(p[-1]),
        }
        for p in pts
    ]


def generate_candidates(space, method="random", n_iter=20, seed=0) -> list[dict]:
    if method == "grid":
        return _grid(space)
    if method == "lhs":
        return _lhs(space, n_iter, seed)
    rng = np.random.default_rng(seed)
    return [_sample(space, rng) for _ in range(n_iter)]


def evaluate_candidate(
    runner,
    theta_qrp0,
    train_eps,
    val_eps,
    jobs=1,
    hi_grid=sig.DEFAULT_HI,
    gamma_grid=sig.DEFAULT_GAMMA,
) -> dict:
    """Run the filter with a Q/R/P0 candidate, then σ-recalibrate; score on val.

    Train and val episodes are collected in a **single** parallel wave (the pool sizes
    itself to the episode count, so two separate calls would serialise into two waves and
    roughly double the wall time per candidate).
    """
    both = sig.collect(
        runner, list(train_eps) + list(val_eps), theta=theta_qrp0, jobs=jobs
    )
    tr, va = both[: len(train_eps)], both[len(train_eps) :]
    best_sigma, _ = sig.search_sigma(tr, va, hi_grid, gamma_grid)
    theta = dict(theta_qrp0)
    theta["std_scale"] = best_sigma["std_scale"]
    theta["sigma_hi"] = best_sigma["sigma_hi"]
    theta["gamma"] = best_sigma["gamma"]
    return {
        "theta": theta,
        "score": best_sigma["score"],
        **{
            k: best_sigma[k]
            for k in (
                "nrmse",
                "coverage",
                "fostac_band_coverage",
                "critical_auc",
                "critical_balacc",
                "critical_tpr",
                "critical_tnr",
            )
        },
    }


#: Why the most recent :func:`evaluate_batch` call dropped candidates. Cleared on entry.
#: Read it when a batch comes back all-``None`` — without it the cause is invisible.
_LAST_FAILURES: list[str] = []


def last_failures() -> list[str]:
    """Reasons the candidates of the most recent :func:`evaluate_batch` were dropped."""
    return list(_LAST_FAILURES)


def evaluate_batch(
    runner,
    thetas,
    train_eps,
    val_eps,
    jobs=1,
    pool=None,
    hi_grid=sig.DEFAULT_HI,
    gamma_grid=sig.DEFAULT_GAMMA,
) -> list[dict]:
    """Evaluate a whole **population** of Q/R/P0 candidates in ONE parallel wave.

    The batch equivalent of :func:`evaluate_candidate` — use this for population-based
    optimisers (CMA-ES etc.), where evaluating candidates sequentially would leave most
    cores idle. Failed candidates (e.g. the stiff ODE refusing to integrate) come back as
    ``None`` so the caller can assign them the worst fitness.
    """
    from .filter_runners import collect_batch

    _LAST_FAILURES.clear()
    eps = list(train_eps) + list(val_eps)
    n_tr = len(train_eps)
    out: list[dict] = []
    for th, res in zip(
        thetas, collect_batch(runner, thetas, eps, jobs=jobs, pool=pool)
    ):
        if any(r is None for r in res):  # one episode run failed
            _LAST_FAILURES.append(
                f"{sum(r is None for r in res)}/{len(res)} episode runs failed"
            )
            out.append(None)
            continue
        try:
            best_sigma, _ = sig.search_sigma(
                res[:n_tr], res[n_tr:], hi_grid, gamma_grid
            )
        except Exception as exc:  # noqa: BLE001
            # Swallowing this silently makes an unattended multi-day run undiagnosable,
            # so keep the reason around for the caller to report.
            _LAST_FAILURES.append(f"search_sigma: {type(exc).__name__}: {exc}")
            out.append(None)
            continue
        theta = dict(th)
        theta["std_scale"] = best_sigma["std_scale"]
        theta["sigma_hi"] = best_sigma["sigma_hi"]
        theta["gamma"] = best_sigma["gamma"]
        out.append(
            {
                "theta": theta,
                "score": best_sigma["score"],
                **{
                    k: best_sigma[k]
                    for k in (
                        "nrmse",
                        "coverage",
                        "fostac_band_coverage",
                        "critical_auc",
                        "critical_balacc",
                        "critical_tpr",
                        "critical_tnr",
                    )
                },
            }
        )
    return out


def search_noise(
    runner,
    train_eps,
    val_eps,
    space=None,
    method="random",
    n_iter=20,
    seed=0,
    jobs=1,
    use_skopt=False,
    verbose=True,
):
    """Search Q/R/P0 (+σ) on validation. Returns (best_result, history)."""
    space = space or default_space()

    if use_skopt and method == "bayes":
        return _search_skopt(
            runner, train_eps, val_eps, space, n_iter, seed, jobs, verbose
        )

    cands = generate_candidates(space, method, n_iter, seed)
    # All candidates are known upfront here (grid/random/lhs), so evaluate them in ONE
    # flattened wave — keeps every core busy instead of one candidate at a time.
    results = evaluate_batch(runner, cands, train_eps, val_eps, jobs=jobs)
    history = []
    for i, r in enumerate(results):
        if r is None:
            if verbose:
                print(
                    f"  [{i+1}/{len(cands)}] FAILED (integration / sigma fit)",
                    flush=True,
                )
            continue
        history.append(r)
        if verbose:
            print(
                f"  [{i+1}/{len(cands)}] score={r['score']:.3f} cov={r['coverage']:.3f} "
                f"ftcov={r['fostac_band_coverage']:.3f} balacc={r['critical_balacc']:.3f}",
                flush=True,
            )
    if not history:
        raise RuntimeError("all candidates failed to evaluate")
    history.sort(key=lambda r: -r["score"])
    return history[0], history


def _search_skopt(runner, train_eps, val_eps, space, n_iter, seed, jobs, verbose):
    """Bayesian optimisation via scikit-optimize (if installed)."""
    from skopt import gp_minimize
    from skopt.space import Real

    blocks = list(space["q_blocks"])
    dims = [Real(*space["q_blocks"][b], prior="log-uniform", name=b) for b in blocks]
    dims += [
        Real(*space["r_scale"], prior="log-uniform", name="r_scale"),
        Real(*space["p0_scale"], prior="log-uniform", name="p0_scale"),
    ]
    history = []

    def to_theta(x):
        return {
            "q_scale": {b: float(x[i]) for i, b in enumerate(blocks)},
            "r_scale": float(x[-2]),
            "p0_scale": float(x[-1]),
        }

    def obj(x):
        r = evaluate_candidate(runner, to_theta(x), train_eps, val_eps, jobs)
        history.append(r)
        if verbose:
            print(
                f"  score={r['score']:.3f} cov={r['coverage']:.3f} "
                f"balacc={r['critical_balacc']:.3f}",
                flush=True,
            )
        return -r["score"]

    gp_minimize(obj, dims, n_calls=n_iter, random_state=seed)
    history.sort(key=lambda r: -r["score"])
    return history[0], history
