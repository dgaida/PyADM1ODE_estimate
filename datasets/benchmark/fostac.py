"""FOS/TAC as a realistic laboratory measurement (Nordmann titration).

The FOS/TAC value is the standard acidification indicator of an agricultural
biogas plant. It is **not** an online sensor: an operator draws a sample and
titrates it, typically once a week.

Stored hourly, sampled on demand
--------------------------------
The dataset holds an **independent titration for every hour**, even though no
plant measures that often. That is deliberate: an experiment picks its own
frequency with :func:`subsample`, so "how often would we have to measure?" is a
question the data can answer instead of one that was decided when the files were
written. Because each hour carries its own independent draw, any subset of rows
is a genuine measurement series at that frequency — weekly sampling is rows
0, 168, 336, ..., which is exactly 9 independent titrations.

Why FOS and TAC are two values but not two measurements
-------------------------------------------------------
Nordmann (1977) titrates **one** sample (20 mL of digester filtrate) with 0.1 N
H2SO4 through **two consecutive endpoints**::

    TAC [mg CaCO3/L] = V1 [mL] * 250                     (start  -> pH 5.0)
    FOS [mg HAc/L]   = (V2 [mL] * 1.66 - 0.15) * 500     (pH 5.0 -> pH 4.4)

The FOS leg *starts where the TAC leg stops*. Two consequences follow, and both
are the reason this module models the titration rather than simply adding two
independent Gaussians:

* A sample-volume or dilution error scales V1 and V2 **together** — the errors
  are positively correlated.
* An error in locating the pH 5.0 endpoint moves titrant **out of one leg and
  into the other** — that part is anti-correlated.

The same mechanism explains the asymmetry reported in the literature (~1.5 %
for alkalinity, ~6.7 % for VFA): at TAC ~10 g/L the first leg is V1 ~40 mL,
while at FOS ~2 g/L the second is only V2 ~2.5 mL. An endpoint error of 0.15 mL
is 0.4 % of V1 but 6 % of V2.

Literature
----------
* Nordmann, W. (1977). Die Überwachung der Schlammfaulung.
* Standard deviation 1.45 % (alkalinity) / 6.7 % (VFA):
  Rapid, Simple, and Accurate Method for Measurement of VFA and Carbonate
  Alkalinity in Anaerobic Reactors, Environ. Sci. Technol. (2002).
* Validity limits of the empirical Nordmann formulas (errors explode above
  ~20 g/L): Appl. Sci. 11(24):11843 (2021); ChemEngineering 9(3):53 (2025).
* Inter-laboratory scatter (no normed sample preparation): KTBL/VDLUFA biogas
  ring test.

Known idealisation
------------------
``fos_true`` here is the model's *true* VFA sum, whereas a real Nordmann FOS is
an empirical titration proxy that deviates systematically from it (and the
deviation grows with total solids). Only the measurement noise is modelled, not
that method bias — mixing the two would make the dataset untestable, since the
bias is exactly what a mechanistic model cannot reproduce.
"""

from __future__ import annotations

import numpy as np

#: Sampling interval a real plant would use, and the default of every experiment
#: that does not say otherwise. The dataset itself stores the titration **hourly**
#: (see :func:`build_series`) so any interval can be selected per experiment;
#: this is only the default, never a property of the data.
SAMPLE_EVERY_DAYS = 7.0

#: Nordmann calibration constants (20 mL sample, 0.1 N H2SO4).
TAC_PER_ML = 250.0  # mg CaCO3/L per mL to pH 5.0
FOS_SLOPE = 1.66  # mL -> mg HAc/L, empirical
FOS_OFFSET = 0.15
FOS_PER_ML = 500.0

#: Common multiplicative error on the sample (pipetting, dilution, filtration).
SAMPLE_REL_SIGMA = 0.02
#: Absolute error in locating the pH 5.0 endpoint [mL]. Shifts titrant between
#: the two legs, so it is the dominant error for FOS and negligible for TAC.
ENDPOINT_ML_SIGMA = 0.15
#: Burette/readout error per leg [mL], independent between legs.
BURETTE_ML_SIGMA = 0.02


def fostac_true(states: np.ndarray, params) -> tuple[np.ndarray, np.ndarray]:
    """True FOS and TAC in mg/L from the 41-state ADM1 trajectory.

    Args:
        states: ``(..., 41)`` ADM1 states.
        params: ``Adm1TorchParams`` for the acid/base constants.

    Returns:
        ``(fos_mg_l, tac_mg_l)``, both shaped like ``states[..., 0]``.
        FOS as mg acetic-acid equivalent per litre, TAC as mg CaCO3 per litre.
    """
    import torch
    from pyadm1.core.adm1_torch import tac_torch, vfa_torch

    x = torch.as_tensor(np.asarray(states, dtype=float))
    fos = vfa_torch(x).numpy() * 1000.0  # kg/m^3 -> mg/L
    tac = tac_torch(x, params).numpy() * 1000.0
    return fos, tac


def titrate(
    fos_mg_l: np.ndarray,
    tac_mg_l: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the Nordmann titration error model to true FOS/TAC values.

    Converts back to titrant volumes, perturbs those, and recomputes — so the
    correlation between the two channels, and the strong asymmetry in their
    relative errors, both fall out of the geometry instead of being imposed.

    Args:
        fos_mg_l: True FOS [mg HAc/L].
        tac_mg_l: True TAC [mg CaCO3/L].
        rng: Random generator.

    Returns:
        ``(fos_meas, tac_meas)`` in the same units, clipped at zero.
    """
    fos_mg_l = np.asarray(fos_mg_l, dtype=float)
    tac_mg_l = np.asarray(tac_mg_l, dtype=float)

    v1 = tac_mg_l / TAC_PER_ML
    v2 = (fos_mg_l / FOS_PER_ML + FOS_OFFSET) / FOS_SLOPE

    shape = fos_mg_l.shape
    eps_sample = rng.normal(0.0, SAMPLE_REL_SIGMA, size=shape)
    d_endpoint = rng.normal(0.0, ENDPOINT_ML_SIGMA, size=shape)

    v1n = (
        v1 * (1.0 + eps_sample) + d_endpoint + rng.normal(0.0, BURETTE_ML_SIGMA, shape)
    )
    v2n = (
        v2 * (1.0 + eps_sample) - d_endpoint + rng.normal(0.0, BURETTE_ML_SIGMA, shape)
    )

    tac_meas = np.maximum(v1n * TAC_PER_ML, 0.0)
    fos_meas = np.maximum((v2n * FOS_SLOPE - FOS_OFFSET) * FOS_PER_ML, 0.0)
    return fos_meas, tac_meas


def sample_indices(
    n_steps: int, dt_hours: float = 1.0, every_days: float = SAMPLE_EVERY_DAYS
) -> np.ndarray:
    """Time indices of a sampling schedule (day 0, then every ``every_days``).

    Args:
        n_steps: Length of the time axis.
        dt_hours: Step size of the time axis.
        every_days: Interval between laboratory samples. ``dt_hours/24`` gives
            every step.

    Returns:
        Indices into the hourly axis.
    """
    step = max(1, round(every_days * 24.0 / dt_hours))
    return np.arange(0, n_steps, step, dtype=int)


def subsample(
    fostac_hourly: np.ndarray,
    every_days: float = SAMPLE_EVERY_DAYS,
    dt_hours: float = 1.0,
    as_mask: bool = False,
) -> np.ndarray:
    """Pick a sampling frequency out of the stored hourly titrations.

    The stored array holds an **independent** titration for every hour, so any
    subset of rows is a valid measurement series at that frequency: a real
    operator sampling weekly performs 9 independent titrations, and rows 0, 168,
    336, ... are exactly 9 independent titrations. Subsampling is therefore
    statistically correct, not an approximation — which is the whole point of
    storing it hourly.

    Args:
        fostac_hourly: ``(T, 2)`` hourly measurements.
        every_days: Desired interval, e.g. ``7`` weekly, ``1`` daily,
            ``1/24`` hourly.
        dt_hours: Step size of the time axis.
        as_mask: Return a ``(T, 2)`` copy with non-sampled rows set to ``NaN``
            instead of the compact ``(n_samples, 2)`` array. Useful for filters
            that walk the full time axis and skip gaps.

    Returns:
        ``(n_samples, 2)`` values, or a ``(T, 2)`` NaN-masked copy.
    """
    arr = np.asarray(fostac_hourly, dtype=float)
    idx = sample_indices(len(arr), dt_hours, every_days)
    if not as_mask:
        return arr[idx]
    out = np.full_like(arr, np.nan)
    out[idx] = arr[idx]
    return out


def build_series(
    states: np.ndarray, params, seed: int, dt_hours: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Hourly FOS/TAC for one series: ``(measured, true)``, each ``(T, 2)``.

    A separate, independent titration is drawn for **every** hour, so an
    experiment can pick any sampling frequency afterwards with
    :func:`subsample` — including asking how often a plant would have to
    measure. Column 0 is FOS [mg HAc/L], column 1 TAC [mg CaCO3/L].

    The noise-free values are returned as well and stored alongside: deriving
    them from ``states`` needs pyadm1 and torch, and the dataset's loader is
    deliberately numpy-only.
    """
    fos, tac = fostac_true(states, params)
    rng = np.random.default_rng(seed + 777_000)
    fos_m, tac_m = titrate(fos, tac, rng)
    return (
        np.stack([fos_m, tac_m], axis=-1),
        np.stack([fos, tac], axis=-1),
    )
