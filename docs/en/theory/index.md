# Theory — overview

Background on the models and methods used.

## Contents

* [ADM1da model](adm1.md) — description of the 41-state model used here  
  (Schlattmann 2011, agricultural extension of ADM1) and its key state
  indices.  
* [PINN v2](pinn.md) — the collocation trajectory PINN: network, closures,  
  physics residual, causal rolling-window operation and what the structure study
  found (`estimation/pinn/`).  
* [GRU observer (PINN B)](observer.md) — the amortised observer: pre-trained  
  offline on many simulated scenarios, then near-instant online inference with
  self-supervised fine-tuning; how well it generalises to shifted datasets.

* [UKF ↔ PINN fusion](fusion.md) — combining a UKF and a PINN estimate via  
  covariance intersection.

## Background reading

The formal observability analysis of ADM1-based models is documented in
[Observability → Literature review](../observability/literature_review.md).
