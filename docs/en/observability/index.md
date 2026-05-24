# Observability — overview

Observability answers the central question:

> Which internal states can an observer (UKF, EKF, asymptotic observer)
> *in principle* reconstruct from the available measurements, and which
> ones cannot — regardless of the chosen filter architecture?

This is a **model property**, not an algorithm property. A better filter
cannot make unobservable states estimable.

## Why this matters here

The ADM1da model has **41 states**. A real biogas plant typically has 3–6
informative online sensors plus possibly weekly lab values. The mismatch
is large. Before any filter setup, it must be clear which subset of the
41 states is actually estimable given the concrete sensor suite.

## What is documented here

* **Step 1 — [Literature review](literature_review.md)** — what do
  Hellmann et al. 2023, Gaida et al. 2012 and Haugen et al. 2014 say?
  Which measurement sets unlock which model classes? Where are the
  structural levers (pH, CH₄/CO₂ separation, lab values)?
* **Step 2 — [Sensor–state dependencies](sensor_state_dependencies.md)**
  — concrete reconciliation of the literature findings with our ADM1da
  implementation. Per measurement channel, with code citations from
  `pyadm1/core/adm1.py`: which states appear directly in the measurement
  model, and which become exposed 1-step indirectly through the ODE
  coupling? Master table + STRIKE-GOLDD applicability.

## Planned extensions

* **Sensitivity analysis** — numerical check which channels are actually
  separated by the filter under what `Q`/`R` tuning.
