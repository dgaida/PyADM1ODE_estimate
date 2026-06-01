# API-Referenz — Übersicht

Automatisch generierte API-Dokumentation via `mkdocstrings`.

## Module

* **`pyadm1ode_estimation.estimation`** — Kern der Zustandsschätzung  
    * `base` — `EstimationStep`, `StateEstimator`-Protocol  
    * `state_vector` — `StateChannel`, `StateVectorSpec`  
    * `specs` — `adm1da_full_spec()` Factory, `InputSpec`,  
      `KineticSpec`, `SensorQualityProfile`, `Quality`-Enum  
    * `process_model` — `ADM1ProcessModel`  
    * `observation_model` — `ObservationChannel`, `ObservationModel`,  
      eingebaute Extractor-Funktionen  
    * `measurement_calendar` — `MeasurementCalendar`, `SampleRate`  
    * `sensors` — `SensorAdapter`, `measure_truth_with_sensors`  
      (Adapter zu PyADM1ODE-Sensor-Klassen)  
    * `twin` — Twin-Experiment-Helfer (`propagate_truth`, `run_filter`,  
      `coverage_within_2sigma`)  
    * `filters.sr_ukf` — `UnscentedKalmanFilter` (Square-Root UKF)  
* **`pyadm1ode_estimation.example_plants`** — Referenz-Anlagen  
    * `simple` — `build_simple_plant()` (1 Fermenter + Storage + BHKW)  
    * `multi_stage` — `build_multi_stage_plant()`  
      (3 Fermenter-Kaskade mit 2 BHKW)  
* **`pyadm1ode_estimation.artifacts`** — Schnittstelle zur Kalibrierung  
    * `calibration_artifact` — `CalibrationArtifact`, `load_artifact`,  
      `save_artifact`, `apply_to_plant`

## Auto-Doc-Beispiel

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
