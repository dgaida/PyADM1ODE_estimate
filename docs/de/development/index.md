# Entwicklung — Übersicht

Hinweise für Entwickler:innen, die am Repo selbst oder am Ökosystem arbeiten.

## Inhalt

* [Ecosystem-Integration](ecosystem-integration.md) — Checkliste, wie eine  
  neue Erweiterung im PyADM1ODE-Ökosystem dokumentiert und mit den anderen
  Repos verlinkt wird. Auch nützlich, wenn die zwei bestehenden Repos
  (`PyADM1ODE`, `PyADM1ODE_calibration`) auf den gleichen Stand wie
  `PyADM1ODE_estimate` gebracht werden sollen.
* [SR-UKF Performance](ukf_performance.md) — verlustfreie Optimierungen
  am Square-Root-UKF (Vektorisierung + Cholesky-Caching) und die
  Regressionstest-Strategie, die ihre Bit-Stabilität absichert.

## Entwicklungs-Workflow

```bash
# Setup
pip install -e ".[dev,docs]"

# Tests
python -m pytest

# Doku lokal
mkdocs serve

# Linting
ruff check pyadm1ode_estimation/
black pyadm1ode_estimation/
```

## CI/CD

* `.github/workflows/tests.yml` — pytest auf Push und PR.  
* `.github/workflows/lint.yml` — ruff + black.  
* `.github/workflows/docs.yml` — MkDocs Build und Deploy via `mike` zu  
  `gh-pages` bei Push auf `main` oder Tag `v*`.  
* `.github/workflows/auto-version-badges.yml` — Versions-Badges automatisch  
  pflegen.
