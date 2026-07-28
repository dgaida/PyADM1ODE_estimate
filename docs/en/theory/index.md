# Theory — overview

Background on the models and methods used.

## Contents

* [ADM1da model](adm1.md) — description of the 41-state model used here  
  (Schlattmann 2011, agricultural extension of ADM1) and its key state
  indices.  
* [PINN](pinn.md) — the per-window Physics-Informed Neural Network smoother: from
  the intuition for a newcomer through the loss maths and the ADM1-specific
  engineering to the implementation (`pinn.py`, `pinn_smoother.py`).
* [Pre-trained observer](observer.md) — the amortised GRU observer: pre-trained
  offline on many simulated scenarios, then near-instant online inference with
  self-supervised fine-tuning.
* [UKF ↔ PINN fusion](fusion.md) — combining a UKF and a PINN estimate via
  covariance intersection.

## Background reading

The formal observability analysis of ADM1-based models is documented in
[Observability → Literature review](../observability/literature_review.md).
