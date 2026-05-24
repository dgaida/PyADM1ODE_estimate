# API-Referenz — Übersicht

Automatisch generierte API-Dokumentation via `mkdocstrings`.

## Module

* **`pyadm1ode_estimation.estimation`** — Kern der Zustandsschätzung
    * `base` — `EstimationStep`, `StateEstimator`-Protocol
    * `state_vector` — `StateChannel`, `StateVectorSpec`
    * `process_model` — `ADM1ProcessModel`
    * `observation_model` — `ObservationChannel`, `ObservationModel`
    * `twin` — Twin-Experiment-Helfer
    * `filters.ukf` — `UnscentedKalmanFilter`
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
