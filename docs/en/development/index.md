# Development — overview

Notes for developers working on the repository itself or on the ecosystem.

## Contents

* [Ecosystem integration](ecosystem-integration.md) — checklist for hooking a
  new extension into the PyADM1ODE ecosystem. Also useful when bringing the
  two sibling repos (`PyADM1ODE`, `PyADM1ODE_calibration`) up to the same
  documentation standard as `PyADM1ODE_estimate`.

## Development workflow

```bash
# Setup
pip install -e ".[dev,docs]"

# Tests
python -m pytest

# Docs locally
mkdocs serve

# Linting
ruff check pyadm1ode_estimation/
black pyadm1ode_estimation/
```

## CI/CD

* `.github/workflows/tests.yml` — pytest on push and PR.
* `.github/workflows/lint.yml` — ruff + black.
* `.github/workflows/docs.yml` — MkDocs build and deploy via `mike` to
  `gh-pages` on push to `main` or tag `v*`.
* `.github/workflows/auto-version-badges.yml` — automatic version badges.
