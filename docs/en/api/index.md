# API reference — overview

Auto-generated API documentation via `mkdocstrings`.

## Modules

* **`pyadm1ode_estimation.estimation`** — core of state estimation  
    * `base` — `EstimationStep`, `StateEstimator` protocol  
    * `state_vector` — `StateChannel`, `StateVectorSpec`  
    * `specs` — `adm1da_full_spec()` factory, `InputSpec`,  
      `KineticSpec`, `SensorQualityProfile`, `Quality` enum  
    * `process_model` — `ADM1ProcessModel`  
    * `observation_model` — `ObservationChannel`, `ObservationModel`,  
      built-in extractor functions  
    * `measurement_calendar` — `MeasurementCalendar`, `SampleRate`  
    * `sensors` — `SensorAdapter`, `measure_truth_with_sensors`  
      (adapter for PyADM1ODE sensor classes)  
    * `twin` — twin experiment helpers (`propagate_truth`, `run_filter`,  
      `coverage_within_2sigma`)  
    * `filters.sr_ukf` — `UnscentedKalmanFilter` (Square-Root UKF)  
* **`pyadm1ode_estimation.example_plants`** — reference plants  
    * `simple` — `build_simple_plant()`  
      (1 fermenter + storage + CHP)  
    * `multi_stage` — `build_multi_stage_plant()`  
      (3-fermenter cascade with 2 CHPs)  
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
