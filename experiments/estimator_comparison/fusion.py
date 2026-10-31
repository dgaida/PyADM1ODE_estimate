"""Fuse the tuned full UKF with the amortised PINN observer (variant B).

The two best estimators fail differently. The UKF carries the ADM1 forward and stays
tight wherever the model is right, but it inherits the model's bias and loses the acid
chain when the influent leaves the characterisation it was given. The observer network
never integrates anything; it maps the measurement window straight onto states, so it
degrades gently rather than drifting, but it has no mechanism that would let it be
exact. A convex blend is the cheapest way to ask whether those two error patterns are
independent enough to cancel.

    python -m experiments.estimator_comparison.fusion

Fitting the weights
-------------------
Anything fitted on the test set would not be comparable against the four estimators it
is being ranked beside, all of whose settings were frozen before the test split was
touched. So every weight here is fitted on the **validation split of the original
benchmark** -- 20 series the UKF was not tuned on and the observer never took a
gradient step on -- and then applied unchanged to all three test sets, the two
out-of-distribution ones included. The observer's early stopping did read that split,
which is model selection rather than fitting, but it is the reason ``val`` is the
weakest link in this design and worth saying out loud.

Three fusions, in order of how much they assume:

``mean``
    A fixed half and half. No parameters at all, so nothing can be fitted wrongly.

``state``
    One weight per state, the least-squares optimum on the validation split. Cheap and
    it captures the obvious structure: gas channels the UKF owns, acid channels it
    does not.

``unc``
    Inverse-variance weighting from the UKF's own reported standard deviation against
    a constant per-state error scale for the observer. The only variant that can react
    within a series -- if the filter's covariance opens up where it is struggling, the
    blend leans on the network there. The UKF is known to be overconfident, so a single
    global scale on its sigma is fitted on the validation split too.

Two in-sample rows are printed alongside, neither of them usable, both there to bound
how much a better weighting rule could still win:

``LS-Optimum``
    Per-state weights fitted on the very test set they are scored on. Careful with
    this one: it minimises pooled squared error, while the reported number is a median
    over states of a normalised RMSE. Those are different objectives, so this row can
    and does land *above* a fusion that was not fitted at all. It is not an upper bound.

``bestes globales w``
    The single fixed weight that minimises the reported metric on that test set. Within
    the one-parameter family this one *is* a genuine in-sample bound.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from .comparison import BURNIN_STEPS, REPO, load

#: Validation-split trajectories the weights are fitted on.
VAL_UKF = REPO / "datasets/benchmark/ukf_full_val_reference.npz"
VAL_PINN = REPO / "experiments/output/pinn_gate/test_trajectories_val.npz"

#: Where each dataset keeps the full UKF's reported standard deviation.
STD_FILES = {"benchmark": REPO / "results_ukf/cmaes_full_reference.npz"}

#: The datasets the fusion is scored on, original first.
DATASETS = ("benchmark", "benchmark_ccm", "benchmark_hard")

#: Grid the sigma scale of the ``unc`` fusion is searched over, log-spaced.
SIGMA_SCALES = np.geomspace(0.05, 20.0, 61)


def std_file(dataset: str) -> Path:
    """Path to the full UKF's ``ukf_std`` for one dataset."""
    return STD_FILES.get(
        dataset, REPO / "datasets" / dataset / "ukf_full_reference.npz"
    )


def _stack(path: Path, key: str) -> np.ndarray:
    """Load an object array of per-series matrices as one ``(N, T, 41)`` block."""
    raw = np.load(path, allow_pickle=True)[key]
    return np.stack([np.asarray(x, dtype=float) for x in raw])


# ------------------------------------------------------------------------ metric ---
def med_nrmse(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per series, the median over states of RMSE normalised by the truth's RMS.

    The convention the PINN evaluations and the overview figure use, burn-in already
    assumed to be trimmed off both arguments.

    Args:
        pred: Estimate, ``(n_series, n_steps, 41)``.
        truth: Ground truth, same shape.

    Returns:
        np.ndarray: One value per series, shape ``(n_series,)``.
    """
    rmse = np.sqrt(np.nanmean((pred - truth) ** 2, axis=1))
    return np.nanmedian(rmse / (np.sqrt(np.nanmean(truth**2, axis=1)) + 1e-12), axis=1)


def nrmse_per_state(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Mean over series of the per-state normalised RMSE, ``(41,)``."""
    rmse = np.sqrt(np.nanmean((pred - truth) ** 2, axis=1))
    return np.nanmean(rmse / (np.sqrt(np.nanmean(truth**2, axis=1)) + 1e-12), axis=0)


# ------------------------------------------------------------------------ weights ---
def optimal_state_weights(
    ukf: np.ndarray, pinn: np.ndarray, truth: np.ndarray
) -> np.ndarray:
    """The convex weight per state that minimises squared error on the given data.

    With ``d = ukf - pinn`` and ``r = pinn - truth`` the blend's error is ``w*d + r``,
    quadratic in ``w``, so the unconstrained optimum is ``-<d,r>/<d,d>``. Clipping it
    into ``[0, 1]`` keeps the result a genuine interpolation: a weight outside the
    interval would be extrapolating past one of the two estimators, which is where a
    fit on 20 series stops being trustworthy.

    Normalising each state by the RMS of its truth, as the metric does, multiplies that
    state's objective by a positive constant and so leaves the optimum untouched.

    Args:
        ukf: First estimate, ``(n_series, n_steps, 41)``, burn-in trimmed.
        pinn: Second estimate, same shape.
        truth: Ground truth, same shape.

    Returns:
        np.ndarray: Weight on the UKF per state, shape ``(41,)``.
    """
    d = ukf - pinn
    r = pinn - truth
    num = np.nansum(d * r, axis=(0, 1))
    den = np.nansum(d * d, axis=(0, 1))
    w = np.where(den > 1e-30, -num / np.where(den > 1e-30, den, 1.0), 0.5)
    return np.clip(w, 0.0, 1.0)


def uncertainty_weights(
    ukf_std: np.ndarray, sigma_pinn: np.ndarray, scale: float
) -> np.ndarray:
    """Inverse-variance weight on the UKF, per series, step and state.

    Args:
        ukf_std: The filter's reported standard deviation, ``(n_series, n_steps, 41)``.
        sigma_pinn: Constant error scale of the observer per state, ``(41,)``.
        scale: Global correction on the filter's sigma, fitted on the validation split.

    Returns:
        np.ndarray: Weights in ``[0, 1]``, same shape as ``ukf_std``.
    """
    su = np.nan_to_num(np.abs(ukf_std) * scale, nan=np.inf, posinf=np.inf) ** 2
    sp = np.maximum(sigma_pinn, 1e-30) ** 2
    w = sp / (su + sp)
    return np.clip(np.nan_to_num(w, nan=0.5), 0.0, 1.0)


@dataclass
class Weights:
    """Everything fitted on the validation split, ready to apply to a test set.

    Attributes:
        state: Per-state UKF weight of the ``state`` fusion, ``(41,)``.
        sigma_pinn: Per-state RMSE of the observer on the validation split, ``(41,)``.
        sigma_scale: Multiplier on the filter's reported sigma for the ``unc`` fusion.
        val_scores: Validation score of each fusion and of the two inputs.
    """

    state: np.ndarray
    sigma_pinn: np.ndarray
    sigma_scale: float
    val_scores: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialise for the record, so a later run can be checked against it."""
        return json.dumps(
            {
                "state": [round(float(v), 6) for v in self.state],
                "sigma_pinn": [float(v) for v in self.sigma_pinn],
                "sigma_scale": float(self.sigma_scale),
                "val_scores": {k: round(v, 6) for k, v in self.val_scores.items()},
            },
            indent=2,
        )


def fit_weights(verbose: bool = True) -> Weights:
    """Fit every fusion parameter on the validation split of the original benchmark.

    Returns:
        Weights: The frozen parameters, plus what they scored where they were fitted.

    Raises:
        FileNotFoundError: If a validation trajectory file has not been produced yet.
    """
    for p in (VAL_UKF, VAL_PINN):
        if not p.exists():
            raise FileNotFoundError(
                f"{p} missing -- produce it with\n"
                "  python -m experiments.estimator_comparison.run_ukf_variants "
                "--dataset benchmark --split val --variants full\n"
                "  python experiments/pinn_gate/export_test_trajectories.py "
                "--dataset benchmark --split val --only b"
            )

    b = BURNIN_STEPS
    _, val, _ = _val_episodes()
    truth = np.stack([np.asarray(e.truth, dtype=float) for e in val])[:, b:]
    ukf = _stack(VAL_UKF, "ukf_x_hat")[:, b:]
    std = _stack(VAL_UKF, "ukf_std")[:, b:]
    pinn = _stack(VAL_PINN, "pinn_b_x_hat")[:, b:]

    w_state = optimal_state_weights(ukf, pinn, truth)
    sigma_pinn = np.sqrt(np.nanmean((pinn - truth) ** 2, axis=(0, 1)))

    def unc_score(scale: float) -> float:
        w = uncertainty_weights(std, sigma_pinn, scale)
        return float(np.mean(med_nrmse(w * ukf + (1 - w) * pinn, truth)))

    scores = [unc_score(s) for s in SIGMA_SCALES]
    scale = float(SIGMA_SCALES[int(np.argmin(scores))])

    w_unc = uncertainty_weights(std, sigma_pinn, scale)
    val_scores = {
        "UKF voll": float(np.mean(med_nrmse(ukf, truth))),
        "PINN B": float(np.mean(med_nrmse(pinn, truth))),
        "mean": float(np.mean(med_nrmse(0.5 * (ukf + pinn), truth))),
        "state": float(np.mean(med_nrmse(w_state * ukf + (1 - w_state) * pinn, truth))),
        "unc": float(np.mean(med_nrmse(w_unc * ukf + (1 - w_unc) * pinn, truth))),
    }

    if verbose:
        print(f"Gewichte auf dem Validierungssplit ({len(val)} Reihen) angepasst")
        print(f"  sigma-Skalierung des UKF: {scale:.3g}")
        print(f"  mittleres Zustandsgewicht auf den UKF: {w_state.mean():.3f}")
        print("  Validierungsscore:")
        for k, v in val_scores.items():
            print(f"    {k:<12}{100 * v:6.1f}%")
        print()

    return Weights(w_state, sigma_pinn, scale, val_scores)


def _val_episodes():
    """The train/val/test episodes of the original benchmark, split as everywhere."""
    from pyadm1ode_estimation.estimation.filter_tuning.datasets import load_benchmark

    return load_benchmark().make_splits(
        days=None, burnin_days=0.0, val_frac=0.2, seed=0
    )


# --------------------------------------------------------------------- evaluation ---
def fuse(
    kind: str,
    ukf: np.ndarray,
    pinn: np.ndarray,
    weights: Weights,
    ukf_std: np.ndarray | None = None,
) -> np.ndarray:
    """Blend two estimates with one of the fitted rules.

    Args:
        kind: ``mean``, ``state`` or ``unc``.
        ukf: The filter's estimate, ``(n_series, n_steps, 41)``.
        pinn: The observer's estimate, same shape.
        weights: Parameters from :func:`fit_weights`.
        ukf_std: Required for ``unc``, the filter's reported standard deviation.

    Returns:
        np.ndarray: The fused estimate, same shape as the inputs.

    Raises:
        ValueError: On an unknown ``kind``, or ``unc`` without a standard deviation.
    """
    if kind == "mean":
        return 0.5 * (ukf + pinn)
    if kind == "state":
        return weights.state * ukf + (1 - weights.state) * pinn
    if kind == "unc":
        if ukf_std is None:
            raise ValueError("the 'unc' fusion needs the filter's reported sigma.")
        w = uncertainty_weights(ukf_std, weights.sigma_pinn, weights.sigma_scale)
        return w * ukf + (1 - w) * pinn
    raise ValueError(f"unknown fusion {kind!r}.")


def evaluate(dataset: str, weights: Weights) -> dict:
    """Score both inputs, the three fusions and the oracle on one test set.

    Args:
        dataset: Directory under ``datasets/``.
        weights: Parameters fitted on the validation split.

    Returns:
        dict: Per-variant scores, per-state scores and the per-series values.
    """
    b = BURNIN_STEPS
    c = load(dataset=dataset)
    truth = c.truth[:, b:]
    ukf = c.estimates["UKF voll"][:, b:]
    pinn = c.estimates["PINN B"][:, b:]
    std = _stack(std_file(dataset), "ukf_std")[:, b:]

    preds = {
        "UKF voll": ukf,
        "PINN B": pinn,
        "Fusion mean": fuse("mean", ukf, pinn, weights),
        "Fusion state": fuse("state", ukf, pinn, weights),
        "Fusion unc": fuse("unc", ukf, pinn, weights, ukf_std=std),
    }
    w_ls = optimal_state_weights(ukf, pinn, truth)
    preds["LS-Optimum*"] = w_ls * ukf + (1 - w_ls) * pinn

    grid = np.linspace(0.0, 1.0, 101)
    w_glob = float(
        grid[
            int(
                np.argmin(
                    [np.mean(med_nrmse(w * ukf + (1 - w) * pinn, truth)) for w in grid]
                )
            )
        ]
    )
    preds["bestes glob. w*"] = w_glob * ukf + (1 - w_glob) * pinn

    per_series = {k: med_nrmse(v, truth) for k, v in preds.items()}
    return {
        "dataset": dataset,
        "state_names": c.state_names,
        "series_names": c.series_names,
        "per_series": per_series,
        "overall": {k: float(np.mean(v)) for k, v in per_series.items()},
        "per_state": {k: nrmse_per_state(v, truth) for k, v in preds.items()},
        "ls_weights": w_ls,
        "global_w": w_glob,
    }


def report(results: list[dict], weights: Weights) -> str:
    """Format the whole comparison as plain text.

    Args:
        results: One :func:`evaluate` result per dataset.
        weights: The frozen parameters, for the weight listing.

    Returns:
        str: The report.
    """
    variants = list(results[0]["overall"])
    out: list[str] = []

    out.append(
        "Gesamtgenauigkeit (Median ueber 41 Zustaende je Reihe, Mittel ueber 20 Reihen)"
    )
    out.append("")
    head = f"{'Variante':<16}" + "".join(f"{r['dataset']:>17}" for r in results)
    out.append(head)
    out.append("-" * len(head))
    for v in variants:
        row = f"{v:<16}"
        for r in results:
            base = r["overall"]["UKF voll"]
            cur = r["overall"][v]
            rel = 100 * (cur / base - 1)
            row += f"{100 * cur:11.1f}% {rel:+5.1f}"
        out.append(row)
    out.append("")
    out.append(
        "Die zweite Spalte je Datensatz ist die relative Aenderung gegen UKF voll, in Prozent.\n"
        "Mit * markierte Zeilen sind auf dem Testsatz selbst angepasst und nicht verwendbar."
    )
    out.append("")

    out.append("Paarweiser Vorzeichen-Rangtest gegen UKF voll (20 Reihen, zweiseitig)")
    out.append("")
    for r in results:
        cells = []
        for v in variants:
            if not v.startswith("Fusion"):
                continue
            stat = wilcoxon(r["per_series"][v], r["per_series"]["UKF voll"])
            cells.append(f"{v.split()[-1]} p={stat.pvalue:.3f}")
        out.append(f"  {r['dataset']:<16}" + "   ".join(cells))
    out.append("")
    out.append(
        "Bestes festes Gewicht je Datensatz (in-sample): "
        + ", ".join(f"{r['dataset']} {r['global_w']:.2f}" for r in results)
    )
    out.append("")

    out.append(
        "Auf wie vielen der 20 Reihen die Fusion den besseren der beiden Eingaenge schlaegt"
    )
    out.append("")
    for r in results:
        best_in = np.minimum(r["per_series"]["UKF voll"], r["per_series"]["PINN B"])
        cells = [
            f"{v.split()[-1]} {int((r['per_series'][v] < best_in).sum())}/20"
            for v in variants
            if v.startswith("Fusion")
        ]
        out.append(f"  {r['dataset']:<16}" + "   ".join(cells))
    out.append("")

    n_states = len(results[0]["state_names"])
    out.append(f"Auf wie vielen der {n_states} Zustaende jede Variante die beste ist")
    out.append("")
    for r in results:
        stack = np.stack([r["per_state"][v] for v in variants])
        wins = np.bincount(np.argmin(stack, axis=0), minlength=len(variants))
        out.append(
            f"  {r['dataset']:<16}"
            + "  ".join(f"{v}: {n}" for v, n in zip(variants, wins) if n)
        )
    out.append("")

    out.append(
        "Zustandsgewichte der 'state'-Fusion, angepasst auf dem Validierungssplit"
    )
    out.append(
        "(1.0 = ganz UKF, 0.0 = ganz PINN B); daneben das In-sample-LS-Optimum je Testsatz"
    )
    out.append("")
    names = results[0]["state_names"]
    order = np.argsort(-np.abs(weights.state - 0.5))
    hdr = f"  {'Zustand':<14}{'val':>7}" + "".join(
        f"{r['dataset'].replace('benchmark', 'bm'):>10}" for r in results
    )
    out.append(hdr)
    out.append("  " + "-" * (len(hdr) - 2))
    for j in order[:12]:
        row = f"  {names[j]:<14}{weights.state[j]:7.2f}"
        row += "".join(f"{r['ls_weights'][j]:10.2f}" for r in results)
        out.append(row)
    out.append(f"  ... {n_states - 12} weitere Zustaende")
    return "\n".join(out)


def main() -> int:
    """Fit on validation, score on every test set, print and optionally save."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out", default=None, help="write the report and weights here")
    args = ap.parse_args()

    weights = fit_weights()
    results = [evaluate(d, weights) for d in args.datasets]
    text = report(results, weights)
    print(text)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        (out.parent / "fusion_weights.json").write_text(
            weights.to_json(), encoding="utf-8"
        )
        print(f"\n-> {out} und fusion_weights.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
