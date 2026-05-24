# Getting started

This guide brings up a local UKF — from installation to a first state estimate.

## Prerequisites

* Python ≥ 3.10
* [`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE) (base package with the
  ADM1da mechanistic model)
* Optional: [`PyADM1ODE_calibration`](https://github.com/dgaida/PyADM1ODE_calibration)
  to produce calibrated models and to access plant-specific plant builders.

## Installation

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
```

Details: [Installation](installation.md).

## Minimal example

The simplest working setup uses the production UKF on a synthetic plant:

```python
import numpy as np
from pyadm1ode_estimation.estimation import (
    StateChannel, StateVectorSpec,
    ADM1ProcessModel, ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter

# 1. Declare the state vector
spec = StateVectorSpec(digester_id="primary", channels=[
    StateChannel("X_ac", kind="adm1", adm1_index=27,
                 initial=1.2, initial_std=0.3, process_noise_std=0.1),
    StateChannel("Q_solid", kind="input_flow", input_substrate_index=0,
                 initial=35.0, initial_std=5.0, process_noise_std=0.5,
                 drift_model="ou", ou_mean=35.0, ou_theta=0.1,
                 lower=0.0, upper=80.0),
])

# 2. Process and observation model (plant built by the caller)
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[...])
ukf = UnscentedKalmanFilter(process, obs, spec)

# 3. Online loop
dt = 1.0 / 24.0  # one-hour step in days
for t, y in measurements:
    ukf.predict(dt=dt)
    step = ukf.update(y, t=t)
    print(f"t={t:.2f}d  x_hat={step.x_hat}  NIS={step.nis:.2f}")
```

## Where to go next

* [Usage → UKF in practice](usage/ukf.md) — detailed plant setup.
* [Observability → Literature review](observability/literature_review.md) —
  which states are actually estimable with which sensors?
* [Examples](examples/index.md) — twin experiment, methodology.
