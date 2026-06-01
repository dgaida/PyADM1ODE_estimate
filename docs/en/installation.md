# Installation

## Prerequisites

* **Python** ≥ 3.10  
* **PyADM1ODE** (base package) — installed transitively. If not automatically  
  available:

  ```bash
  pip install git+https://github.com/dgaida/PyADM1ODE.git
  ```

* **Optional**: PyADM1ODE_calibration for the plant builder and historical  
  data access.

## Standard installation

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
```

This installs the core dependencies: `numpy`, `scipy`, `pandas`,
`pythonnet`, `pyyaml`, `pyadm1`.

## Optional components

### Deep learning (PINN, ensemble)

```bash
pip install -e ".[deep_learning]"
```

Adds `torch` and `scikit-learn`.

### Development tools

```bash
pip install -e ".[dev]"
```

Adds `pytest`, `pytest-cov`, `black`, `ruff`.

### Building the documentation locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

The documentation is then served at `http://localhost:8000`.

## Verification

```bash
python -m pytest tests/test_ukf.py tests/test_observation_model.py \
                 tests/test_state_vector.py tests/test_calibration_artifact.py
```

All tests should pass.

## Troubleshooting

See [Troubleshooting](troubleshooting.md) for common issues.
