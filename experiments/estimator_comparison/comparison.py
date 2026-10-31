"""Load the four estimator runs on the test set and score them per state.

The estimators compared here all ran on the same 20 test series over 60 days,
without a FOS/TAC measurement:

===============  ==========================================================
UKF voll         41-state SR-UKF, ``results_ukf/cmaes_full_reference.npz``
UKF (A+D)        18-state acid/digester core, ``cmaes_adcore_reference.npz``
PINN v2          collocation PINN, causal: 10-day window, daily retraining, hold
PINN B           one observer network, trained once, applied to every series
===============  ==========================================================

PINN B's trajectories are the stored test-set export of the published observer,
PINN v2's come from ``experiments/pinn_v2/rolling_sweep.py`` (one ``series_NN.npz``
per series, scheme ``W10_C24_S200``, key ``hold``).

Burn-in
-------
The PINN evaluations drop the first 48 hours before scoring, so the same window is
dropped for the UKF here. Comparing a filter that starts from a prior against an
estimator that saw the whole window would otherwise flatter the latter on exactly
the stretch where the filter has not converged yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

#: Steps dropped at the start of every series, matching the PINN evaluations.
BURNIN_STEPS = 48

#: Where the original run keeps its trajectories. The variants were produced later and
#: keep theirs beside their own data, see :func:`sources_for`.
SOURCES: dict[str, tuple[Path, str]] = {
    "UKF voll": (REPO / "results_ukf/cmaes_full_reference.npz", "ukf_x_hat"),
    "UKF (A+D)": (REPO / "results_ukf/cmaes_adcore_reference.npz", "ukf_x_hat"),
    "PINN v2": (REPO / "experiments/pinn_v2/results/f1_trajectories/W10_C24_S200", "hold"),
    "PINN B": (
        REPO / "experiments/output/pinn_gate/test_trajectories_b.npz",
        "pinn_b_x_hat",
    ),
}

#: File name per estimator inside a variant dataset directory.
VARIANT_FILES: dict[str, tuple[str, str]] = {
    "UKF voll": ("ukf_full_reference.npz", "ukf_x_hat"),
    "UKF (A+D)": ("ukf_adcore_reference.npz", "ukf_x_hat"),
    "PINN v2": ("", "hold"),  # per-series directory, see sources_for
    "PINN B": ("pinn_trajectories.npz", "pinn_b_x_hat"),
}


def sources_for(dataset: str) -> dict[str, tuple[Path, str]]:
    """Where to find each estimator's trajectories for one dataset.

    Args:
        dataset: Directory name under ``datasets/``.

    Returns:
        dict: Estimator name to ``(path, npz key)``, in display order.
    """
    if dataset == "benchmark":
        return SOURCES
    d = REPO / "datasets" / dataset
    out = {name: (d / f, k) for name, (f, k) in VARIANT_FILES.items()}
    out["PINN v2"] = (
        REPO / "experiments/pinn_v2/results" / f"f1_trajectories_{dataset.removeprefix('benchmark_')}"
        / "W10_C24_S200",
        "hold",
    )
    return out


#: One colour per estimator, kept stable across every plot. Run through the palette
#: validator: lightness band, chroma floor and colour-vision separation all pass, the
#: worst adjacent pair being orange against red at dE 16.8 under deuteranopia. Cyan and
#: orange fall below 3:1 contrast on a light surface, which the table view answers.
COLORS = {
    "UKF voll": "tab:blue",
    "UKF (A+D)": "tab:cyan",
    "PINN v2": "tab:red",
    "PINN B": "tab:orange",
}

#: A second channel beside colour, so the lines stay separable in print, in greyscale,
#: and for a reader who cannot tell the hues apart.
STYLES = {"UKF voll": "-", "UKF (A+D)": "--", "PINN v2": "-.", "PINN B": (0, (1, 1))}

#: The same idea for the dot plot, where a line style has nothing to draw on.
MARKERS = {"UKF voll": "o", "UKF (A+D)": "s", "PINN v2": "^", "PINN B": "D"}

#: Physical unit of every state, transcribed from the state-vector table in the module
#: docstring of ``pyadm1/core/adm1.py`` (lines 13-66), which is the only place either
#: repo states all 41. ``meta.json`` carries the names but no units.
#:
#: Written out in full rather than derived from the ``S_``/``X_``/``P_`` prefix,
#: because a prefix rule gets the charge-balance block wrong: ``S_CATION`` and
#: ``S_ANION`` are kmol/m3 while the four VFA ions beside them are kg COD/m3, the same
#: units as the totals they are a part of. ``specs.py`` records that assuming kmol
#: there was once a real bug, clipping the acetate ion by roughly 40x.
STATE_UNITS: dict[str, str] = {
    # dissolved (0-11)
    "S_SU": "kg COD/m³",
    "S_AA": "kg COD/m³",
    "S_FA": "kg COD/m³",
    "S_VA": "kg COD/m³",
    "S_BU": "kg COD/m³",
    "S_PRO": "kg COD/m³",
    "S_AC": "kg COD/m³",
    "S_H2": "kg COD/m³",
    "S_CH4": "kg COD/m³",
    "S_CO2": "kmol C/m³",
    "S_NH4": "kmol N/m³",
    "S_I": "kg COD/m³",
    # particulate sub-fractions (12-21)
    "X_PS_CH": "kg COD/m³",
    "X_PS_PR": "kg COD/m³",
    "X_PS_LI": "kg COD/m³",
    "X_PF_CH": "kg COD/m³",
    "X_PF_PR": "kg COD/m³",
    "X_PF_LI": "kg COD/m³",
    "X_S_CH": "kg COD/m³",
    "X_S_PR": "kg COD/m³",
    "X_S_LI": "kg COD/m³",
    "X_I": "kg COD/m³",
    # biomass (22-28)
    "X_SU": "kg COD/m³",
    "X_AA": "kg COD/m³",
    "X_FA": "kg COD/m³",
    "X_C4": "kg COD/m³",
    "X_PRO": "kg COD/m³",
    "X_AC": "kg COD/m³",
    "X_H2": "kg COD/m³",
    # charge balance (29-36)
    "S_CATION": "kmol/m³",
    "S_ANION": "kmol/m³",
    "S_VA_ION": "kg COD/m³",
    "S_BU_ION": "kg COD/m³",
    "S_PRO_ION": "kg COD/m³",
    "S_AC_ION": "kg COD/m³",
    "S_HCO3": "kmol C/m³",
    "S_NH3": "kmol N/m³",
    # gas phase (37-40), absolute pressure
    "P_H2": "bar",
    "P_CH4": "bar",
    "P_CO2": "bar",
    "P_TOTAL": "bar",
}


@dataclass
class Comparison:
    """Everything the viewer needs, already aligned and scored.

    Attributes:
        estimators: Names of the estimators that were found, in display order.
        series_names: One label per test series, ``mode#index``.
        regimes: The operating mode of each series.
        state_names: The 41 ADM1 state names.
        time: Time axis in days, shared by every series.
        truth: Ground truth, ``(n_series, n_steps, 41)``.
        estimates: Per estimator, ``(n_series, n_steps, 41)``.
        nrmse: Per estimator, ``(n_series, 41)``, normalised RMSE after burn-in.
        missing: Estimators that were expected but whose file is not there yet.
    """

    estimators: list[str]
    series_names: list[str]
    regimes: list[str]
    state_names: list[str]
    time: np.ndarray
    truth: np.ndarray
    estimates: dict[str, np.ndarray]
    nrmse: dict[str, np.ndarray]
    missing: list[str]

    @property
    def n_series(self) -> int:
        """How many test series are loaded."""
        return len(self.series_names)

    @property
    def n_states(self) -> int:
        """How many states each series carries."""
        return len(self.state_names)

    def unit(self, state: int) -> str:
        """The physical unit of one state, from :data:`STATE_UNITS`."""
        return STATE_UNITS[self.state_names[state]]

    def state_label(self, state: int) -> str:
        """State name with its unit, for an axis label or a header."""
        return f"{self.state_names[state]} [{self.unit(state)}]"

    def overall(self, estimator: str) -> np.ndarray:
        """Per-state accuracy over the whole test set.

        The median across series rather than the mean: a single series that an
        estimator diverges on would otherwise decide the ranking of a state.

        Args:
            estimator: Which estimator to summarise.

        Returns:
            np.ndarray: One NRMSE per state, shape ``(41,)``.
        """
        return np.nanmedian(self.nrmse[estimator], axis=0)

    def error_band(
        self, estimator: str, state: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """How one state's error develops over the 60 days, across all series.

        Args:
            estimator: Which estimator to summarise.
            state: Index of the state.

        Returns:
            tuple: Median, 25th and 75th percentile of the signed relative error at
            each time step, each shape ``(n_steps,)``.
        """
        err = self.relative_error(estimator, state)
        return (
            np.nanmedian(err, axis=0),
            np.nanpercentile(err, 25, axis=0),
            np.nanpercentile(err, 75, axis=0),
        )

    def relative_error(self, estimator: str, state: int) -> np.ndarray:
        """Signed error of one state, scaled by that state's typical magnitude.

        Dividing by the RMS of the truth rather than by the truth itself keeps the
        curve readable when a state passes through zero, which several of the ADM1
        concentrations do.

        Args:
            estimator: Which estimator to evaluate.
            state: Index of the state.

        Returns:
            np.ndarray: ``(n_series, n_steps)``, in units of the state's RMS.
        """
        truth = self.truth[:, :, state]
        scale = np.sqrt(np.nanmean(truth[:, BURNIN_STEPS:] ** 2, axis=1, keepdims=True))
        return (self.estimates[estimator][:, :, state] - truth) / (scale + 1e-12)


def _state_names() -> list[str]:
    """The 41 state names, read from the dataset metadata.

    Raises:
        KeyError: If the state vector has a name :data:`STATE_UNITS` does not cover.
            Better to fail here than to draw an axis whose unit is quietly missing.
    """
    meta = json.loads(
        (REPO / "datasets/benchmark/meta.json").read_text(encoding="utf-8")
    )
    index_map = meta["state_index_map"]
    names = [index_map[str(i)] for i in range(len(index_map))]
    unknown = [n for n in names if n not in STATE_UNITS]
    if unknown:
        raise KeyError(
            f"no unit recorded for {unknown}; add it to STATE_UNITS from the state "
            "table in the module docstring of pyadm1/core/adm1.py."
        )
    return names


def _nrmse_per_state(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """RMSE per state, normalised by the RMS of the truth of that state.

    The per-state form of the ``med_nrmse`` used in the PINN evaluations, which takes
    the median of exactly these numbers.

    Args:
        pred: Estimate, ``(n_steps, 41)``.
        truth: Ground truth, same shape.

    Returns:
        np.ndarray: One value per state, shape ``(41,)``.
    """
    rmse = np.sqrt(np.nanmean((pred - truth) ** 2, axis=0))
    return rmse / (np.sqrt(np.nanmean(truth**2, axis=0)) + 1e-12)


def load(
    sources: dict[str, tuple[Path, str]] | None = None, dataset: str = "benchmark"
) -> Comparison:
    """Load the truth and every estimator run that is present.

    An estimator whose file is missing is reported in :attr:`Comparison.missing`
    rather than raising, so the viewer stays usable while a long export is still
    running.

    Args:
        sources: Override of the per-dataset defaults, for tests or an ad-hoc run.
        dataset: Directory under ``datasets/``. ``benchmark_ccm`` and
            ``benchmark_hard`` hold the out-of-distribution sets, scored with the very
            same tuned configurations as the original.

    Returns:
        Comparison: Aligned trajectories and per-state scores.

    Raises:
        FileNotFoundError: If the test dataset itself is missing.
    """
    sources = sources_for(dataset) if sources is None else sources

    test = np.load(REPO / "datasets" / dataset / "test.npz", allow_pickle=True)
    truth = np.stack([np.asarray(s, dtype=float) for s in test["states"]])
    time = np.asarray(test["time"][0], dtype=float)
    regimes = [str(r) for r in test["regime"]]
    names = [f"{r}#t{i}" for i, r in enumerate(regimes)]

    estimates: dict[str, np.ndarray] = {}
    missing: list[str] = []
    for label, (path, key) in sources.items():
        if not path.exists():
            missing.append(label)
            continue
        if path.is_dir():
            # one file per series (PINN v2 rolling chains); all must be present
            files = [path / f"series_{i:02d}.npz" for i in range(len(truth))]
            if not all(f.exists() for f in files):
                missing.append(label)
                continue
            raw = [np.load(f)[key] for f in files]
        else:
            raw = np.load(path, allow_pickle=True)[key]
        estimates[label] = np.stack([np.asarray(x, dtype=float) for x in raw])

    b = BURNIN_STEPS
    nrmse = {
        label: np.stack(
            [_nrmse_per_state(est[i, b:], truth[i, b:]) for i in range(len(truth))]
        )
        for label, est in estimates.items()
    }

    return Comparison(
        estimators=[k for k in sources if k in estimates],
        series_names=names,
        regimes=regimes,
        state_names=_state_names(),
        time=time,
        truth=truth,
        estimates=estimates,
        nrmse=nrmse,
        missing=missing,
    )
