# Troubleshooting

Common issues and how to resolve them.

## Installation

### `ImportError: No module named 'pyadm1'`

The base package must be installed separately:

```bash
pip install git+https://github.com/dgaida/PyADM1ODE.git
```

### `pythonnet` build fails

`pythonnet` needs a .NET runtime. On Windows this is usually present; on
Linux:

```bash
sudo apt install mono-complete
```

## Filter behaviour

### NIS values chronically ≫ n_obs

The filter is underestimating uncertainty. Possible causes:

* `process_noise_std` too small in individual `StateChannel`s.
* `noise_std` of the `ObservationChannel`s too small (sensor noise
  underestimated).
* Model mismatch — e.g. an unmodelled inhibitor is active in reality.

First step: increase `process_noise_std` of each biological channel by a
factor of 2–3.

### Filter does not converge to sensible values

* **Check the initial state**: with a large initial error, convergence
  typically takes 10–15 days (see Haugen et al. 2014).
* **Check observability**: is the state in question estimable at all with
  the available sensors? See
  [Observability literature review](observability/literature_review.md).
* **Check bounds**: overly tight `lower`/`upper` bounds clip aggressively
  and suppress information.

### `LinAlgError` during Cholesky decomposition

Posterior covariance is no longer SPD. The UKF in `filters/ukf.py` has
eigenvalue flooring and jitter fallback built in — if that still fails,
the `Q`/`R` scales are likely numerically inconsistent.

## Calibration artifact

### `ValueError: schema_version=999`

The artifact has a newer schema version than this runtime supports. Either
upgrade the estimation package or roll back the calibration to a compatible
version.

### `apply_to_plant`: many "skipped" warnings

The artifact references components or kinetic parameters that do not exist
in the current plant build. If this happens during a production startup,
**always call with `strict=True`** — fail-fast rather than silently
half-applied.

## Building docs locally

### `mkdocs serve` fails

```bash
pip install -e ".[docs]"
```

ensures all documentation dependencies (Material theme, i18n plugin,
mkdocstrings, mermaid2) are present.
