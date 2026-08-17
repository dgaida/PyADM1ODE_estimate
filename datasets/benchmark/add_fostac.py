"""Retrofit the FOS/TAC laboratory measurement into an existing dataset.

**Not needed for a freshly generated dataset**: ``generate_benchmark.py`` writes
``fostac`` / ``fostac_true`` in the same pass as everything else, so a rebuilt
dataset is complete and reproducible from the seeds alone. This script exists for
dataset files that predate that, and to regenerate the titration after a change
to the noise model in :mod:`fostac` without re-simulating 120 series.

It computes FOS and TAC from the *stored* 41-state trajectories, so no
re-simulation is needed (seconds instead of hours), applies the Nordmann
titration error model from :mod:`fostac`, and writes the result back as two
extra arrays per file::

    fostac       (N, T, 2)   measured, column 0 FOS [mg HAc/L], 1 TAC [mg CaCO3/L]
    fostac_true  (N, T, 2)   the same without measurement noise

Both are **hourly**: an independent titration is drawn for every step, so an
experiment selects its own sampling frequency with ``fostac.subsample`` rather
than inheriting one that was fixed when these files were written. The noise-free
copy is stored because deriving it from ``states`` needs pyadm1 and torch, and
the loader is deliberately numpy-only.

Usage::

    python add_fostac.py            # rewrites train.npz and test.npz in place
    python add_fostac.py --dry-run  # only report what would change
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from fostac import build_series, sample_indices

#: Digester the benchmark's sensors and lab samples refer to.
DIGESTER_ID = "primary"

#: Acetate-methanogen decay reverted to the ADM1 standard, exactly as in
#: ``generate_benchmark.py``. It does not affect the acid/base constants read
#: below, but keeping the plant identical to the generator's means this script
#: cannot drift away from it.
K_DEC_AC = 0.02


def _params():
    """Acid/base constants of the benchmark plant (identical for every series:
    the kinetic perturbation does not touch them).

    Builds the plant here rather than importing a forward model from an
    experiment: this script belongs to the dataset and must stay runnable on its
    own, independently of which repository a given study lives in.
    """
    from pyadm1.core.adm1_torch import Adm1TorchParams

    from pyadm1ode_estimation.example_plants import build_multi_stage_plant

    plant = build_multi_stage_plant()
    plant.components[DIGESTER_ID].adm1._kinetic["k_dec_ac"] = K_DEC_AC
    return Adm1TorchParams.from_adm1(plant.components[DIGESTER_ID].adm1)


def process(path: Path, params, dry_run: bool = False) -> None:
    """Add the ``fostac`` array to one npz file."""
    data = np.load(path, allow_pickle=True)
    arrays = {k: data[k] for k in data.files}

    if "fostac" in arrays:
        print(f"  {path.name}: 'fostac' existiert bereits, wird neu berechnet")
    arrays.pop("fostac_true", None)

    states = arrays["states"]  # (N, T, 41), in test.npz an object array
    seeds = arrays["seed"]
    n_series = len(states)

    built = [build_series(states[i], params, int(seeds[i])) for i in range(n_series)]
    meas_list = [m for m, _ in built]
    true_list = [t for _, t in built]

    # test.npz stores its per-series arrays as an object array; matching that
    # keeps np.load(...)["fostac"][i] working exactly like every other field.
    def pack(items: list[np.ndarray]) -> np.ndarray:
        if states.dtype != object:
            return np.stack(items, axis=0)
        boxed = np.empty(n_series, dtype=object)
        for i, arr in enumerate(items):
            boxed[i] = arr
        return boxed

    arrays["fostac"] = pack(meas_list)
    arrays["fostac_true"] = pack(true_list)

    fostac_flat = np.concatenate(meas_list, axis=0)
    n_hours = len(states[0])
    n_weekly = len(sample_indices(n_hours))
    n_zero = int((fostac_flat[:, 0] <= 0.0).sum())
    print(
        f"  {path.name}: {n_series} Reihen x {n_hours} Stunden "
        f"({fostac_flat.shape[0]} Titrationen, davon {n_zero} auf 0 geklippt)"
    )
    print(f"    -> bei woechentlicher Abtastung {n_weekly} Messungen je Reihe")
    print(
        f"    FOS [mg/L] {np.nanmin(fostac_flat[:,0]):8.0f} .. {np.nanmax(fostac_flat[:,0]):8.0f}"
    )
    print(
        f"    TAC [mg/L] {np.nanmin(fostac_flat[:,1]):8.0f} .. {np.nanmax(fostac_flat[:,1]):8.0f}"
    )
    ratio = fostac_flat[:, 0] / np.maximum(fostac_flat[:, 1], 1e-9)
    print(f"    FOS/TAC    {np.nanmin(ratio):8.3f} .. {np.nanmax(ratio):8.3f}")

    if dry_run:
        print("    (dry-run, nicht geschrieben)")
        return
    np.savez_compressed(path, **arrays)
    print(f"    geschrieben ({path.stat().st_size/1e6:.1f} MB)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    params = _params()
    print("FOS/TAC (Nordmann-Titration, woechentlich) wird ergaenzt:")
    for name in ("train.npz", "test.npz"):
        process(_DIR / name, params, args.dry_run)


if __name__ == "__main__":
    main()
