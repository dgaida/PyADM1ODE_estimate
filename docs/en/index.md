# PyADM1ODE Estimate

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml)

**State estimation for agricultural biogas plants** based on the **ADM1da**
mechanistic model (Schlattmann 2011, agricultural extension of ADM1),
implemented in [`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE).

## What this repository does

In live operation, the state of a real plant is estimated from incoming
sensor data and a calibrated simulation model. The estimated states feed
controllers that issue feeding recommendations to the plant operator.

Data flow:

```text
Historical measurements ──► PyADM1ODE_calibration ──► calibrated model
                                                            │
Live sensors ──────────────────────────────────────► PyADM1ODE_estimate
                                                            │
                                                            ▼
                                                  State estimate x_hat
                                                            │
                                                            ▼
                                                     Controller / UI
```

## Key features

* 🎯 **Mechanistic and data-driven** — UKF on the full 41-state ADM1da plus
  planned deep-learning and fusion layers.
* 🔌 **Sensor-tolerant** — channel gating for sparse lab measurements,
  OU drift for augmented input estimation.
* 📊 **Diagnostic** — per-step NIS monitoring, per-channel innovation,
  Joseph-form covariance update for long runs.

## Three estimation approaches

| Approach | Status | Work package |
|---|---|---|
| **UKF** — mechanistic | production | AP 4.2 |
| **Deep Learning Ensemble** | skeleton | AP 4.3 |
| **Fusion** (Covariance Intersection) | placeholder | AP 4.4 |

## Table of contents

* [Getting started](getting-started.md) — quick onboarding.
* [Installation](installation.md) — setup for different environments.
* [Usage](usage/index.md) — configure the UKF, load the calibration artifact.
* [Observability](observability/index.md) — which states can be estimated from which sensors?
* [Theory](theory/index.md) — ADM1da model, PINN concept.
* [Examples](examples/index.md) — end-to-end examples (twin, real plant).
* [API reference](api/index.md) — auto-generated via mkdocstrings.

## Quickstart

```python
from pyadm1ode_estimation.estimation import (
    StateChannel, StateVectorSpec,
    ADM1ProcessModel, ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter
from pyadm1ode_estimation.artifacts import load_artifact, apply_to_plant

# 1. Load the calibrated model
artifact = load_artifact("calibrated/plant_2026-05-14.yaml")
plant = build_plant(schema)            # your own plant builder
apply_to_plant(artifact, plant, strict=True)

# 2. Declare the state vector
spec = StateVectorSpec(digester_id="primary", channels=[
    StateChannel("X_ac", kind="adm1", adm1_index=27,
                 initial=1.2, initial_std=0.3, process_noise_std=0.1),
    StateChannel("Q_solid", kind="input_flow", input_substrate_index=0,
                 initial=35.0, initial_std=5.0, process_noise_std=0.5,
                 drift_model="ou", ou_mean=35.0, ou_theta=0.1),
])

# 3. Set up the UKF and run it
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[...])
ukf = UnscentedKalmanFilter(process, obs, spec)

for t, y in measurements:
    ukf.predict(dt=1/24)
    step = ukf.update(y, t=t)
```

## Related projects

* [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) — base package with the ADM1da model.
* [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) —
  parameter calibration against historical data, writes the calibration artifact.
