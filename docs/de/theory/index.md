# Theorie — Übersicht

Hintergrund zu den verwendeten Modellen und Methoden.

## Inhalte

* [ADM1da-Modell](adm1.md) — Beschreibung des verwendeten 41-State-Modells  
  (Schlattmann 2011, agrar-erweitertes ADM1) und der zentralen Zustandsindizes.  
* [PINN](pinn.md) — der per-Window-PINN-Smoother: von der Intuition für
  Einsteiger über die Loss-Mathematik und die ADM1-spezifischen Kniffe bis zur
  Implementierung (`pinn.py`, `pinn_smoother.py`).
* [Vortrainierter Observer](observer.md) — der amortisierte GRU-Observer: offline
  auf vielen simulierten Szenarien vortrainiert, dann nahezu sofortige
  Online-Inferenz mit self-supervised Feinabstimmung.
* [UKF ↔ PINN-Fusion](fusion.md) — Kombination einer UKF- und einer
  PINN-Schätzung per Kovarianzschnitt.

## Hintergrund-Lektüre

Die formale Observability-Analyse ADM1-basierter Modelle ist in
[Observability → Literaturüberblick](../observability/literature_review.md)
dokumentiert.
