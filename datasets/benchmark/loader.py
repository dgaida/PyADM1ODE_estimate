"""Load the ADM1 state-estimation benchmark (numpy only — no PyADM1ODE needed).

from loader import load_train, load_test, load_meta
train = load_train()            # dict of stacked arrays
test  = load_test()             # list of per-series dicts
meta  = load_meta()             # index map, sensor noise, substrate info
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent

# Per-series field names.
_TRAIN_KEYS = [
    "time",
    "measurements",
    "feed_noisy",
    "feed_true",
    "states",
    "switch_days",
]
_TEST_KEYS = _TRAIN_KEYS + ["ukf_x_hat", "ukf_std", "kinetic_factors"]
#: Fields added after the first release; absent from older copies of the files,
#: so they are attached only when present instead of being required.
_OPTIONAL_KEYS = ["fostac", "fostac_true"]


def load_meta() -> dict:
    """The metadata: 41-state index map + units, sensor noise, substrate
    characterisation, kinetic ranges."""
    return json.loads((_DIR / "meta.json").read_text(encoding="utf-8"))


def load_train() -> dict[str, np.ndarray]:
    """Training set as stacked arrays: ``measurements (N,T,5)``, ``feed_noisy``,
    ``states (N,T,41)``, ``time (N,T)``, ``switch_days``, ``kinetic_factors``,
    ``seed`` and ``regime`` (the operating mode of each series)."""
    d = np.load(_DIR / "train.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_test() -> list[dict[str, np.ndarray]]:
    """Test set as a list of per-series dicts (all the same length as train).

    Each dict holds ``measurements (T,5)``, ``feed_noisy (T,5)``,
    ``states (T,41)`` (the truth to score against), ``ukf_x_hat`` / ``ukf_std``
    (the reference UKF estimate), ``time``, ``switch_days``, plus the flags
    ``regime`` (the operating mode), ``seed`` and ``ukf_pending`` (True while the
    UKF reference has not been computed yet).

    Also ``fostac (T, 2)`` / ``fostac_true (T, 2)`` when present: the Nordmann
    titration (column 0 FOS [mg HAc/L], column 1 TAC [mg CaCO3/L]), measured and
    noise-free. Stored **hourly** with an independent titration per step, so an
    experiment picks its own sampling frequency::

        from fostac import subsample
        weekly = subsample(series["fostac"], every_days=7)     # (9, 2)
        daily  = subsample(series["fostac"], every_days=1)     # (61, 2)

    See ``fostac.py`` for the measurement model.
    """
    d = np.load(_DIR / "test.npz", allow_pickle=True)
    n = len(d["time"])
    scalars = {k: d[k] for k in ("regime", "seed", "ukf_pending") if k in d.files}
    out = []
    for i in range(n):
        s = {k: d[k][i] for k in _TEST_KEYS}
        for k in _OPTIONAL_KEYS:
            if k in d.files:
                s[k] = d[k][i]
        if "regime" in scalars:
            s["regime"] = str(scalars["regime"][i])
        if "seed" in scalars:
            s["seed"] = int(scalars["seed"][i])
        if "ukf_pending" in scalars:
            s["ukf_pending"] = bool(scalars["ukf_pending"][i])
        out.append(s)
    return out


# Channel order in ``measurements`` (and the observed columns of any prediction).
CHANNELS = ["Q_gas", "Q_ch4", "Q_co2", "pH", "TS"]
