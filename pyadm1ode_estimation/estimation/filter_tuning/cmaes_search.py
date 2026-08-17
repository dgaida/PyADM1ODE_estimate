"""CMA-ES tuning of a filter's Q/R/P0, warm-started at the empirical Q (approach 1.3).

This is the pipeline that produced the shipped UKF reference of the benchmark dataset. It
is deliberately split into the four stages that were run separately, because they have very
different costs and only stage 1 is an actual search:

===== ======================================================= ===================
Stage Purpose                                                  Typical cost
===== ======================================================= ===================
0     Q and R computed from ground truth, no search at all     ~1 filter episode
1     CMA-ES over block corrections to that Q, plus R and P0   hours (the search)
2     Re-measure the top candidates at the FULL series length  minutes per candidate
3     Winner applied to the test split -> shipped reference    one pass over test
===== ======================================================= ===================

Why the warm start matters: stage 0 fixes the *shape* of Q per state from data, so CMA-ES
only has to learn one correction factor per observability block. ``theta = 0`` is exactly
the empirical Q, and the hand-set nominal Q sits around ``exp(2.7)`` inside the bounds, so
the optimiser can travel between both regimes on its own.

Why stage 2 is not a formality: the rank correlation between a 20-day tuning window and the
full 60-day quality is only about 0.64, so the stage-1 ranking does not transfer reliably.

Parallelism: every generation is evaluated as ONE flattened wave of ``popsize x episodes``
tasks over a persistent :class:`~.filter_runners.WorkerPool`, and all candidates of a
generation see the SAME episodes (common random numbers) so their ranking is paired.

Windows note: all parallel work goes through objects defined in this package, never through
callables defined by the caller, so the pipeline is safe to drive from a notebook (spawn
cannot pickle notebook-local functions).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from . import metrics as M
from . import noise_search as ns
from .filter_runners import WorkerPool, collect_parallel, make_ukf_runner

#: Blocks tuned for the A+D core. The other blocks are not estimated by that variant, so
#: giving them their own search dimension would only add noise.
AD_CORE_BLOCKS = ["methanogenesis", "charge_balance"]

#: Search bounds on the log correction per block: exp(-2) = 0.14x to exp(4) = 55x the
#: empirical Q. Wide enough to reach the nominal Q (~15x) from the empirical warm start.
DEFAULT_BOUNDS = (-2.0, 4.0)

#: Fitness handed to CMA-ES for a candidate whose filter run failed (stiff ODE, etc.).
FAILED_FITNESS = 1e3

_METRIC_KEYS = (
    "nrmse",
    "coverage",
    "fostac_band_coverage",
    "critical_auc",
    "critical_balacc",
)


# --------------------------------------------------------------------------- stage 0 ---
def load_empirical_q(path: str | Path) -> np.ndarray:
    """Read a per-state empirical Q **variance** vector from a stage-0 JSON.

    Accepts both artefact shapes produced by :mod:`empirical_noise`: a flat
    ``{"q_std": [...]}`` and the horizon study's ``{"1": {"q_std": [...]}, ...}``, whose
    ``"1"`` entry is the one-step (dt) residual and therefore the Q we want. Non-finite or
    non-positive entries fall back to a tiny variance rather than to NaN, so the filter
    stays integrable.
    """
    raw = json.loads(Path(path).read_text())
    q_std = raw.get("q_std") if "q_std" in raw else raw["1"]["q_std"]
    a = np.array([x if x is not None else np.nan for x in q_std], float)
    return np.where(np.isfinite(a) & (a > 0), a, 1e-9) ** 2


def empirical_q(
    dataset,
    *,
    jobs: int = 8,
    stride: int = 1,
    obs_stride: int = 8,
    max_series: int | None = None,
    save_to: str | Path | None = None,
):
    """Compute Q and R from ground truth (stage 0). Thin wrapper over `empirical_noise`.

    Returns the raw result dict; ``result["q_diag"]`` is the variance vector that
    :func:`theta_from_x` expects. Costs roughly one filter episode for the whole training
    set because it never runs the filter, it only differences the true trajectory.
    """
    from .empirical_noise import empirical_noise

    res = empirical_noise(
        dataset, jobs=jobs, stride=stride, obs_stride=obs_stride, max_series=max_series
    )
    if save_to:
        Path(save_to).write_text(json.dumps(res, indent=2, default=_json_default))
    return res


# ---------------------------------------------------------------- parametrisation ---
def theta_from_x(
    x: Sequence[float],
    q_emp: np.ndarray,
    blocks: Sequence[str],
    block_indices: dict | None = None,
) -> dict:
    """Map a CMA-ES vector to a filter parameter set.

    ``x = [theta_block_0 ... theta_block_n, theta_r, theta_p0]``, all in log space::

        Q_i      = Q_empirical_i * exp(theta_block(i))
        r_scale  = exp(theta_r)
        p0_scale = exp(theta_p0)

    so ``x = 0`` reproduces the pure empirical Q with untouched R and P0.
    """
    if block_indices is None:
        from ..specs import BLOCK_INDICES

        block_indices = BLOCK_INDICES
    q = np.asarray(q_emp, float).copy()
    for b, t in zip(blocks, x[: len(blocks)]):
        for i in block_indices[b]:
            q[int(i)] *= float(np.exp(t))
    return {
        "q_diag": q,
        "r_scale": float(np.exp(x[-2])),
        "p0_scale": float(np.exp(x[-1])),
    }


def q_scales_of(x: Sequence[float], blocks: Sequence[str]) -> dict[str, float]:
    """Human-readable ``{block: factor}`` of a CMA-ES vector, relative to the empirical Q."""
    return {b: round(float(np.exp(t)), 3) for b, t in zip(blocks, x[: len(blocks)])}


# --------------------------------------------------------------------------- stage 1 ---
def split_per_label(episodes: Sequence, n_first: int) -> tuple[list, list]:
    """Split an episode list per operating mode into (first ``n_first``, the rest).

    Used to carve a sigma-fit set and a scoring set out of the **same** pool while keeping
    both stratified. Episode names are ``"<label>#<index>"``.
    """
    by_label: dict[str, list] = {}
    for e in episodes:
        by_label.setdefault(str(e.name).split("#")[0], []).append(e)
    first, rest = [], []
    for lbl in sorted(by_label):
        first += by_label[lbl][:n_first]
        rest += by_label[lbl][n_first:]
    return first, rest


def run_cmaes(
    dataset,
    q_emp: np.ndarray,
    *,
    variant: str = "full",
    blocks: Sequence[str] | None = None,
    days: float = 20.0,
    burnin_days: float = 2.0,
    per_group_train: int = 2,
    per_group_val: int = 2,
    seed: int = 0,
    sigma_eps: Sequence | None = None,
    score_eps: Sequence | None = None,
    popsize: int = 8,
    generations: int = 12,
    sigma0: float = 1.5,
    bounds: tuple[float, float] = DEFAULT_BOUNDS,
    cma_seed: int = 1,
    w_acc: float = 0.7,
    w_dec: float = 0.3,
    base_nrmse: float | None = None,
    patience: int | None = None,
    fostac_every_days: float | None = None,
    jobs: int = 8,
    out_path: str | Path | None = None,
    objective: Callable[[dict, float], float] | None = None,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Stage 1: CMA-ES over per-block Q corrections plus R and P0.

    Evaluates each generation in one parallel wave on a fixed set of train and val
    episodes. Scoring uses :func:`metrics.objective_combined` unless ``objective`` is
    given, which lets a caller swap in an accuracy-first objective without touching this
    function. Two reference points are measured first: the untouched nominal parameters
    (whose NRMSE becomes the baseline the accuracy term is normalised against) and the pure
    empirical Q.

    Writes the full search trace to ``out_path`` after every generation, so a crashed run
    keeps everything computed so far. Returns that same dict.
    """
    import cma  # optional dependency, only needed here

    if blocks is None:
        from ..specs import BLOCK_INDICES

        blocks = AD_CORE_BLOCKS if variant == "adcore" else list(BLOCK_INDICES)
    blocks = list(blocks)
    lo, hi = bounds

    # Either the caller hands in the two episode sets (which is how the train-only search
    # gets both of them out of the *train* pool), or they are split off here as usual.
    if sigma_eps is not None and score_eps is not None:
        train, val = list(sigma_eps), list(score_eps)
    else:
        train, val, _ = dataset.make_splits(
            days=days,
            burnin_days=burnin_days,
            val_frac=0.2,
            per_group_train=per_group_train,
            per_group_val=per_group_val,
            seed=seed,
        )
    runner = make_ukf_runner(
        dataset.meta, variant=variant, fostac_every_days=fostac_every_days
    )

    # Reference points. evaluate_batch returns None for a candidate that failed to run.
    # A caller that already knows the nominal NRMSE for this horizon can pass it in and
    # save 2 x len(episodes) filter runs — the nominal one is the most expensive candidate
    # there is, because its large Q makes the ODE stiffer.
    if base_nrmse is None:
        ref = ns.evaluate_batch(runner, [{}, {"q_diag": q_emp}], train, val, jobs=jobs)
        base_nrmse = ref[0]["nrmse"] if ref[0] else 1.0
    else:
        ref = [None, None]

    # A non-finite baseline makes every score NaN, and a NaN score never beats the
    # incumbent, so the search would silently run to completion and return its first
    # candidate. Refuse to start instead. The usual cause is empty episodes, i.e.
    # burnin_days >= days.
    if not np.isfinite(base_nrmse) or base_nrmse <= 0:
        steps = [len(e.obs["time"]) for e in val[:3]]
        raise ValueError(
            f"base_nrmse is {base_nrmse!r}, so no candidate can be scored. "
            f"{len(train)} sigma-fit and {len(val)} scoring episodes, the first scoring "
            f"ones have {steps} time steps. Empty episodes mean the burn-in is as long as "
            f"the episode itself, so raise `days` above `burnin_days`."
        )

    score_of = objective or (
        lambda m, b: M.objective_combined(m, b, w_acc=w_acc, w_dec=w_dec)
    )
    res = {
        "config": {
            "variant": variant,
            "days": days,
            "popsize": popsize,
            "generations": generations,
            "jobs": jobs,
            "blocks": blocks,
            "w_acc": w_acc,
            "w_dec": w_dec,
            "n_train": len(train),
            "n_val": len(val),
            "base_nrmse": base_nrmse,
            "bounds": [lo, hi],
            "sigma0": sigma0,
            "cma_seed": cma_seed,
            "patience": patience,
            "fostac_every_days": fostac_every_days,
        },
        "reference": {"nominal": ref[0], "empirical": ref[1]},
        "generations": [],
    }
    _dump(res, out_path)

    es = cma.CMAEvolutionStrategy(
        [0.0] * (len(blocks) + 2),
        sigma0,
        {
            "popsize": popsize,
            "bounds": [lo, hi],
            "seed": cma_seed,
            "verbose": -9,
            "maxiter": generations,
        },
    )
    best = None
    for g in range(generations):
        t0 = time.time()
        X = es.ask()
        out = ns.evaluate_batch(
            runner, [theta_from_x(x, q_emp, blocks) for x in X], train, val, jobs=jobs
        )
        fits, recs = [], []
        for x, r in zip(X, out):
            if r is None:
                fits.append(FAILED_FITNESS)
                recs.append({"x": list(map(float, x)), "failed": True})
                continue
            sc = score_of(r, base_nrmse)
            if not np.isfinite(sc):  # treat like a failed run, never rank it
                fits.append(FAILED_FITNESS)
                recs.append(
                    {
                        "x": list(map(float, x)),
                        "failed": True,
                        "reason": "non-finite score",
                    }
                )
                continue
            fits.append(-sc)  # CMA-ES minimises
            recs.append(
                {
                    "x": list(map(float, x)),
                    "score": sc,
                    **{k: r[k] for k in _METRIC_KEYS},
                    "sigma": {
                        "sigma_hi": r["theta"]["sigma_hi"],
                        "gamma": r["theta"]["gamma"],
                    },
                }
            )
            if best is None or sc > best["score"]:
                best = {
                    "score": sc,
                    "x": list(map(float, x)),
                    "gen": g,
                    "theta": r["theta"],
                    "q_scales": q_scales_of(x, blocks),
                    "metrics": {k: r[k] for k in _METRIC_KEYS},
                }
        es.tell(X, fits)
        ok = [r for r in recs if not r.get("failed")]
        entry = {
            "gen": g,
            "candidates": recs,
            "n_ok": len(ok),
            "best_gen": max((r["score"] for r in ok), default=float("nan")),
            "best_so_far": best["score"] if best else None,
            "seconds": time.time() - t0,
        }
        res["generations"].append(entry)
        res["best"] = best
        _dump(res, out_path)
        if progress:
            progress(entry)

        # Stagnation stop. Lets a run be configured with a generous generation count
        # without paying for the tail when the search has clearly converged.
        if patience and best is not None and g - best["gen"] >= patience:
            res["stopped_early"] = {
                "at_gen": g,
                "best_gen": best["gen"],
                "patience": patience,
            }
            break
    return res


# --------------------------------------------------------------------------- stage 2 ---
def verify(
    dataset,
    candidates: dict[str, dict],
    *,
    variant: str = "full",
    days: float | None = None,
    burnin_days: float = 2.0,
    per_group_train: int = 1,
    per_group_val: int = 5,
    seed: int = 0,
    jobs: int = 8,
    fostac_every_days: float | None = None,
    out_path: str | Path | None = None,
) -> dict:
    """Stage 2: re-measure named candidates at the full series length on validation.

    ``candidates`` maps a label to a theta dict (as returned by :func:`theta_from_x`, or
    ``{}`` for the untouched nominal parameters). ``days=None`` means the full series.
    Necessary because a 20-day ranking transfers to 60 days only weakly.

    ``per_group_train`` must stay small: the train episodes are only there to fit the σ
    recalibration, but they cost a full filter run each. Leaving it at ``None`` would pull
    in the entire training pool.
    """
    train, val, _ = dataset.make_splits(
        days=days,
        burnin_days=burnin_days,
        val_frac=0.2,
        per_group_train=per_group_train,
        per_group_val=per_group_val,
        seed=seed,
    )
    runner = make_ukf_runner(
        dataset.meta, variant=variant, fostac_every_days=fostac_every_days
    )
    labels = list(candidates)
    out = ns.evaluate_batch(
        runner, [candidates[k] for k in labels], train, val, jobs=jobs
    )
    res = {
        "days": days,
        "variant": variant,
        "fostac_every_days": fostac_every_days,
        "n_train": len(train),
        "n_val": len(val),
        "results": {
            k: (
                None
                if r is None
                else {
                    **{m: r[m] for m in _METRIC_KEYS},
                    "std_scale": r["theta"]["std_scale"],
                }
            )
            for k, r in zip(labels, out)
        },
        "theta": {k: candidates[k] for k in labels},
    }
    _dump(res, out_path)
    return res


def select_best(verified: dict, key: str = "nrmse", maximise: bool = False) -> str:
    """Label of the best verified candidate. Default picks the lowest NRMSE."""
    items = [(k, v) for k, v in verified["results"].items() if v is not None]
    if not items:
        raise ValueError("no candidate survived verification")
    return (max if maximise else min)(items, key=lambda kv: kv[1][key])[0]


# --------------------------------------------------------------------------- stage 3 ---
def build_reference(
    dataset,
    theta: dict,
    *,
    variant: str = "full",
    split: str = "test",
    jobs: int = 8,
    fostac_every_days: float | None = None,
    out_npz: str | Path | None = None,
) -> dict:
    """Stage 3: run one configuration over a whole split at full length.

    Produces the ``ukf_x_hat`` / ``ukf_std`` arrays in the shipped format (full series, no
    burn-in trim). Deliberately does NOT overwrite the dataset: replacing the shipped
    reference stays a separate, explicit step.
    """
    train, val, test = dataset.make_splits(
        days=None, burnin_days=0.0, val_frac=0.2, seed=0
    )
    eps = {"train": train, "val": val, "test": test}[split]
    runner = make_ukf_runner(
        dataset.meta, variant=variant, fostac_every_days=fostac_every_days
    )
    t0 = time.time()
    with WorkerPool(min(jobs, len(eps))) as pool:
        res = collect_parallel(runner, eps, theta=theta, pool=pool)

    n_steps = [len(e.obs["time"]) for e in eps]
    x_hat = np.empty(len(res), dtype=object)
    std = np.empty(len(res), dtype=object)
    failed = []
    for i, r in enumerate(res):
        if r is None:  # keep the slot shaped, mark it NaN
            failed.append(i)
            x_hat[i] = np.full((n_steps[i], len(dataset.state_names)), np.nan)
            std[i] = np.full((n_steps[i], len(dataset.state_names)), np.nan)
        else:
            x_hat[i] = np.asarray(r[1], float)
            std[i] = np.asarray(r[2], float)
    if out_npz:
        np.savez_compressed(out_npz, ukf_x_hat=x_hat, ukf_std=std)
    return {
        "x_hat": x_hat,
        "std": std,
        "failed": failed,
        "split": split,
        "minutes": (time.time() - t0) / 60.0,
    }


# ------------------------------------------------------------------------------ io ---
def _json_default(o):
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o.tolist() if hasattr(o, "tolist") else str(o)


def _dump(obj: dict, path: str | Path | None) -> None:
    if path:
        Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


# -------------------------------------------------------------------------- report ---
def trace_table(trace: dict) -> list[dict]:
    """Flatten a stage-1 trace into one row per generation, for quick inspection.

    Tolerates traces written before this module existed, which stored only the candidate
    list, by deriving the per-generation summary from the candidates themselves.
    """
    rows = []
    for g in trace["generations"]:
        ok = [c for c in g["candidates"] if not c.get("failed")]
        rows.append(
            {
                "gen": g["gen"],
                "best_gen": g.get(
                    "best_gen", max((c["score"] for c in ok), default=float("nan"))
                ),
                "best_all": g.get("best_so_far"),
                "n_ok": g.get("n_ok", len(ok)),
                "minutes": (g["seconds"] / 60.0) if "seconds" in g else None,
            }
        )
    return rows


def term_breakdown(
    trace: dict,
    target_cov: float = M.TARGET_COV,
    over_penalty: float = 2.0,
    w_cov: float = 0.5,
    w_ftcov: float = 0.5,
) -> dict:
    """Split every candidate's score into its four terms and report their influence.

    Returns the weighted terms per candidate plus, per term, the correlation with the total
    score and with the sum of the *other* terms. The second one is the informative figure:
    negative means the term is in conflict with the rest, positive means it is redundant
    with it. (Correlation with the total score is biased upward because each term is part
    of that sum.)
    """
    cfg = trace["config"]
    base, w_acc, w_dec = cfg["base_nrmse"], cfg["w_acc"], cfg["w_dec"]

    def pen(x, w):
        d = x - target_cov
        return w * (over_penalty if d > 0 else 1.0) * abs(d)

    cands = [
        c for g in trace["generations"] for c in g["candidates"] if not c.get("failed")
    ]
    T = np.array(
        [
            [
                w_acc * (base - c["nrmse"]) / base,
                w_dec * (c["critical_auc"] - 0.5) / 0.5,
                -pen(c["coverage"], w_cov),
                -pen(c["fostac_band_coverage"], w_ftcov),
            ]
            for c in cands
        ]
    )
    names = ["accuracy", "decision", "coverage_penalty", "fostac_band_penalty"]
    total = T.sum(1)
    return {
        "n": len(cands),
        "terms": dict(zip(names, T.T)),
        "score": total,
        "corr_to_score": {
            n: float(np.corrcoef(T[:, i], total)[0, 1]) for i, n in enumerate(names)
        },
        "corr_to_rest": {
            n: float(np.corrcoef(T[:, i], np.delete(T, i, axis=1).sum(1))[0, 1])
            for i, n in enumerate(names)
        },
    }
