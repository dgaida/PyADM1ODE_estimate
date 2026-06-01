# PyADM1ODE_estimation

State estimation framework for PyADM1ODE biogas plant models.

[![Version](https://img.shields.io/github/v/tag/dgaida/PyADM1ODE_estimate?label=version)](https://github.com/dgaida/PyADM1ODE_estimate/tags)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Quality](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/PyADM1ODE_estimate/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/PyADM1ODE_estimate)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/PyADM1ODE_estimate/)


This repository implements advanced state estimation algorithms for the Anaerobic Digestion Model No. 1 (ADM1), focusing on agricultural biogas plants.

## Project Goals

This project is part of a research initiative (AP 4.2 - AP 4.4) to develop and compare different state estimation approaches:

* **AP 4.2: Unscented Kalman Filter (UKF)**: Implementation of a UKF that estimates the plant state as a probability distribution using the mechanistic ADM1 model.  
* **AP 4.3: Deep Learning Ensemble**: Training an ensemble of deep neural networks to predict the state based on historical measurement data and substrate mixtures. Each network represents a possible reality, providing a distribution of predictions.  
* **AP 4.4: Comparison and Fusion**: Benchmarking both approaches regarding speed and accuracy. Implementation of a fusion algorithm using **Covariance Intersection** to combine both estimates.  

## Project Structure

```text
PyADM1ODE_estimation/
├── pyadm1ode_estimation/       # Main package
│   ├── estimation/             # Estimation algorithms
│   │   ├── __init__.py
│   │   ├── base.py             # StateEstimator protocol + EstimationStep
│   │   ├── state_vector.py     # StateChannel / StateVectorSpec
│   │   ├── process_model.py    # ADM1ProcessModel (pyadm1 propagator)
│   │   ├── observation_model.py# ObservationChannel / ObservationModel
│   │   ├── twin.py             # Twin-experiment helpers
│   │   ├── filters/            # Filter implementations
│   │   │   └── ukf.py          # Unscented Kalman Filter (scaled, gated obs)
│   │   ├── deep_learning/      # Deep Learning models (Ensembles, PINN)
│   │   └── fusion/             # Fusion algorithms (Covariance Intersection)
│   ├── artifacts/              # Handoff artifacts (calibration → estimation)
│   │   └── calibration_artifact.py  # YAML format for calibrated parameters
│   ├── utils/                  # Utility functions
│   └── ...
├── docs/                       # Documentation (MkDocs, bilingual)
├── examples/                   # Usage examples (UKF, twin experiment)
├── tests/                      # Unit and integration tests
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/dgaida/PyADM1ODE_estimation.git
cd PyADM1ODE_estimation

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

Note: This package requires [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) to be installed.

## Usage

### Unscented Kalman Filter

```python
from pyadm1ode_estimation.estimation import (
    StateChannel, StateVectorSpec,
    ADM1ProcessModel, ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter

# 1. Declare what to estimate (indices refer to the 41-state ADM1da vector).
spec = StateVectorSpec(digester_id="primary", channels=[
    StateChannel("S_ac",  kind="adm1", adm1_index=6,  initial=0.1,
                 initial_std=0.3, process_noise_std=0.5),
    StateChannel("X_ac",  kind="adm1", adm1_index=27, initial=1.2,
                 initial_std=0.3, process_noise_std=0.1),
    StateChannel("Q_solid", kind="input_flow", input_substrate_index=0,
                 initial=35.0, initial_std=5.0, process_noise_std=0.5,
                 drift_model="ou", ou_mean=35.0, ou_theta=0.1,
                 lower=0.0, upper=80.0),
])

# 2. Wrap a pyadm1 BiogasPlant.
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[...])
ukf = UnscentedKalmanFilter(process, obs, spec)

# 3. Run the predict/update loop.
for t, y in measurements:
    ukf.predict(dt=1/24)
    step = ukf.update(y, t=t)
```

### Calibration handoff

```python
from pyadm1ode_estimation.artifacts import load_artifact, apply_to_plant
artifact = load_artifact("calibrated/plant_2026-05-14.yaml")
plant = build_plant(schema)            # your own plant builder
apply_to_plant(artifact, plant, strict=True)
```

## Documentation

Full documentation is built with MkDocs Material and published via GitHub Pages with a German/English language switcher and versioning via `mike`:

[**dgaida.github.io/PyADM1ODE_estimate**](https://dgaida.github.io/PyADM1ODE_estimate/)

Local sources live under [`docs/de/`](docs/de/) (default) and [`docs/en/`](docs/en/). Build locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

Key pages:

* [Home (DE)](docs/de/index.md) · [Home (EN)](docs/en/index.md)  
* [Getting started (DE)](docs/de/getting-started.md) · [Getting started (EN)](docs/en/getting-started.md)  
* [UKF in practice](docs/de/usage/ukf.md) · [Calibration artifact](docs/de/usage/calibration_artifact.md)  
* [Observability — Literature review (DE)](docs/de/observability/literature_review.md) · [(EN)](docs/en/observability/literature_review.md)  
* [Examples — methodology](docs/de/examples/index.md)  
* [ADM1da model overview](docs/de/theory/adm1.md)  

## License

This project is licensed under the MIT License.
