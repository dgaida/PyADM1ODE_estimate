# Examples — overview

Methodological end-to-end examples for using the UKF on an agricultural
biogas plant. The examples show the methodology without plant-specific
details — a concrete plant is documented separately and privately.

## In the repo (scripts)



## Planned

* **Validation report** — a summary of how the methods (UKF, later
  Deep Learning Ensemble, later Fusion) were tested against reference data.
  Contents: accuracy, robustness, consistency diagnostics (NIS),
  convergence behaviour.

## Procedure for your own plant

1. **Plant topology** built via the `pyadm1.configurator` API, or imported
   from a private plant-builder repo.
2. **Calibrated artifact** loaded with `load_artifact(...)` and applied with
   `apply_to_plant(...)`.
3. **State-vector spec** tailored — not all 41 ADM1 states are estimable
   with typical SCADA sensors; see
   [Observability → Literature review](../observability/literature_review.md).
4. **Observation model** wired to the available sensor channels.
5. **UKF** instantiated; in the online loop, alternate `predict()` and
   `update()` calls, and monitor NIS.
