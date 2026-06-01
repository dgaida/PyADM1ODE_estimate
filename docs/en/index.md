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
    ADM1ProcessModel,
    InputSpec,
    ObservationChannel,
    ObservationModel,
    adm1da_full_spec,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter
from pyadm1ode_estimation.estimation.observation_model import (
    extract_q_gas_total, extract_q_ch4_total,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

# 1. Build the plant (example or your own PyADM1ODE builder)
plant = build_multi_stage_plant()

# 2. Full 41-state spec with substrate augmentation
spec = adm1da_full_spec(
    digester_id="primary",
    substrate_inputs=[
        InputSpec("maize_silage",   substrate_index=0, initial_flow=4.74),
        InputSpec("solid_manure",   substrate_index=1, initial_flow=13.70),
        InputSpec("chicken_litter", substrate_index=2, initial_flow=1.09),
        InputSpec("slurry",         substrate_index=3, initial_flow=3.68),
        InputSpec("cereal_grain",   substrate_index=4, initial_flow=0.20),
    ],
)

# 3. Process + observation models
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[
    ObservationChannel("Q_gas", extract_q_gas_total, noise_std=10.0),
    ObservationChannel("Q_ch4", extract_q_ch4_total, noise_std=5.0),
])

# 4. Set up the SR-UKF and run it
ukf = UnscentedKalmanFilter(process, obs, spec)
for t, y in measurements:
    ukf.predict(dt=1/24)
    step = ukf.update(y, t=t)
```

More details: [UKF in practice](usage/ukf.md) and
[twin experiments](usage/twin_experiments.md).

## Related projects

* [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) — base package with the ADM1da model.  
* [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) —  
  parameter calibration against historical data, writes the calibration artifact.
