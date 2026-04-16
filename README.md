# PyADM1ODE_estimation

State estimation framework for PyADM1ODE biogas plant models.

![Version](https://img.shields.io/badge/version-0.1.1-blue)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code Quality](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/PyADM1ODE_estimate/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/PyADM1ODE_estimate)


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
│   │   ├── ukf.py              # Unscented Kalman Filter
│   │   ├── deep_learning/      # Deep Learning models
│   │   └── fusion/             # Fusion algorithms (Covariance Intersection)
│   ├── utils/                  # Utility functions
│   └── ...
├── data/                       # Training and validation data
├── examples/                   # Usage examples
├── tests/                      # Unit and integration tests
├── scripts/                    # Data collection and training scripts
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
from pyadm1ode_estimation.estimation.ukf import ADM1UKF
# ... initialization and update loop
```

## License

This project is licensed under the MIT License.
