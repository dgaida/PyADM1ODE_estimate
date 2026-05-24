# Usage — overview

Practical guides on how to use the repository in real operation.

## Contents

* [UKF in practice](ukf.md) — how to configure an `UnscentedKalmanFilter` for
  a concrete plant: set up `StateVectorSpec`, define `ObservationModel`
  channels, write the online loop.
* [Calibration artifact](calibration_artifact.md) — the YAML handoff format
  between calibration and estimation, and how the filter applies it at
  startup.

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

1. Get the **plant topology** from `pyadm1ode_calibration.plants` or your
   own builder.
2. Load the **calibrated artifact** via `load_artifact(...)` and apply it
   with `apply_to_plant(...)`.
3. Set up the **state-vector spec** — which ADM1 indices and which
   augmented input rates do you want to estimate?
   (See [Observability literature review](../observability/literature_review.md)
   for the sensor-to-state logic.)
4. Wire the **observation model** to the available sensor channels —
   `BUILT_IN_EXTRACTORS` covers the common Q_gas/P_el signals.
5. Instantiate the **UKF**, alternate `predict()` and `update()` calls in
   the online loop.
