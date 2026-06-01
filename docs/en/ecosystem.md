# Ecosystem

`PyADM1ODE_estimate` is an extension of the base package **PyADM1ODE** and
works together with the extension PyADM1ODE_calibration.

## The three components

### [PyADM1ODE](https://dgaida.github.io/PyADM1ODE/latest/) — Base

The mechanistic process model. Implements **ADM1da** (Schlattmann 2011), a
41-state ODE system for anaerobic digestion in agricultural biogas plants.
Contains:

* `pyadm1.core` — ODE and parameters  
* `pyadm1.components` — modular plant building blocks (digester, CHP, heating, …)  
* `pyadm1.configurator` — plant builder API  
* `pyadm1.substrates` — substrate library  
* `pyadm1.simulation` — simulator and parallel simulator  

→ [Documentation](https://dgaida.github.io/PyADM1ODE/latest/)
→ [GitHub](https://github.com/dgaida/PyADM1ODE)

### [PyADM1ODE_calibration](https://dgaida.github.io/PyADM1ODE_calibration/latest/) — Offline calibration

Fits ADM1 parameters to historical plant data. Contains:

* IO pipeline for SCADA exports (CSV)  
* Local + global optimizers  
* Sensitivity and identifiability analysis  
* SQLAlchemy persistence for calibration runs  
* Plant builders for real plants  

→ [Documentation](https://dgaida.github.io/PyADM1ODE_calibration/latest/)
→ [GitHub](https://github.com/dgaida/PyADM1ODE_calibration)

### PyADM1ODE_estimate — Online estimation

UKF and (planned) Deep Learning + Fusion for real-time estimation of the
plant state. Reads the [calibration artifact](usage/calibration_artifact.md)
produced by calibration, estimates the full state vector from current sensor
values, and feeds it to controllers.

→ [GitHub](https://github.com/dgaida/PyADM1ODE_estimate)

## Data flow between the repos

```text
                       PyADM1ODE
                  (ADM1da model, plant API)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
   PyADM1ODE_calibration         PyADM1ODE_estimate
            │                           │
            │  writes YAML              │  reads YAML +
            ▼                           ▼  live sensors
   ┌────────────────────┐         ┌───────────────┐
   │ CalibrationArtifact│ ──────► │ UKF / filter  │
   └────────────────────┘         └───────┬───────┘
                                          ▼
                                    State estimate
                                          ▼
                                  Controller / UI
```

## Which packages to install for what

| Use case | Packages |
|---|---|
| Pure plant simulation | `pyadm1` only |
| Fit parameters from historical data | `pyadm1` + `pyadm1ode_calibration` |
| Live filter on a production plant | `pyadm1` + `pyadm1ode_estimation` |
| End-to-end (calibration + live) | all three |

In live operation, `pyadm1ode_calibration` is used only **periodically**
(typically every few weeks or months when the operator triggers a
recalibration); `pyadm1ode_estimation` runs continuously.

## Versioning across repos

Each repo's docs has a `mike` version selector (top right). Recommendation:
use the same version number across all three repos for coordinated releases
so users find the same state in every version dropdown.

## Contributing docs

The repos are released independently. Contributions to ecosystem-wide
consistency are welcome — see
[Ecosystem integration](development/ecosystem-integration.md) for the
checklist on how a new extension hooks into the others.
