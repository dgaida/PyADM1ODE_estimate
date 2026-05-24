# API reference — overview

Auto-generated API documentation via `mkdocstrings`.

## Modules

* **`pyadm1ode_estimation.estimation`** — core of state estimation
    * `base` — `EstimationStep`, `StateEstimator` protocol
    * `state_vector` — `StateChannel`, `StateVectorSpec`
    * `process_model` — `ADM1ProcessModel`
    * `observation_model` — `ObservationChannel`, `ObservationModel`
    * `twin` — twin experiment helpers
    * `filters.ukf` — `UnscentedKalmanFilter`
* **`pyadm1ode_estimation.artifacts`** — interface to calibration
    * `calibration_artifact` — `CalibrationArtifact`, `load_artifact`,
      `save_artifact`, `apply_to_plant`

## Auto-doc preview

::: pyadm1ode_estimation.estimation.state_vector
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.base
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.artifacts.calibration_artifact
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
