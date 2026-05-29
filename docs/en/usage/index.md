# Usage — overview

Practical guides on how to use the repository in real operation.

## Contents

* [UKF in practice](ukf.md) — how to configure the Square-Root UKF for
  a concrete plant: the `adm1da_full_spec()` factory, `ObservationModel`
  channels, `MeasurementCalendar` for sparse lab samples, online loop.
* [Twin experiments](twin_experiments.md) — end-to-end validation
  against a known truth; the bundled `run_twin_experiment.py` and
  interpretation of the diagnostic plots.
* [Calibration artifact](calibration_artifact.md) — the YAML handoff
  format between calibration and estimation, and how the filter applies
  it at startup.

## Big picture

In live operation, the data flow is:

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

Steps for a new setup:

1. Build the **plant** via PyADM1ODE — either via the example
   builders `build_simple_plant()` / `build_multi_stage_plant()` in
   `example_plants/` or via your own `PlantConfigurator`-based
   builder.
2. Load the **calibrated artifact** via `load_artifact(...)` and apply
   it with `apply_to_plant(...)`.
3. Build the **state-vector spec** via the `adm1da_full_spec()`
   factory — it produces all 41 ADM1 states with observability-driven
   defaults. Attach substrate inputs as an `InputSpec` list. With
   non-Phase-1 sensors, upgrade individual blocks via
   `SensorQualityProfile`.
4. Wire the **observation model** to the available sensor channels —
   the built-in extractors cover Q_gas, Q_ch4, P_el, P_th_used,
   stored_volume and direct state read-out.
5. Attach a **MeasurementCalendar** for sample-rate management if
   measurements arrive at different rates.
6. Instantiate the **SR-UKF** (= the current `UnscentedKalmanFilter`),
   call `ukf.reset(x0, P0)` with realistic initial covariance, and
   alternate `predict()` and `update()` in the online loop.
7. Before production: run a **twin experiment**, check mean NIS and
   coverage (see [twin experiments](twin_experiments.md)).
