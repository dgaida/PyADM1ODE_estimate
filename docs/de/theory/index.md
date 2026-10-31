# Theorie — Übersicht

Hintergrund zu den verwendeten Modellen und Methoden.

## Inhalte

* [ADM1da-Modell](adm1.md) — Beschreibung des verwendeten 41-State-Modells  
  (Schlattmann 2011, agrar-erweitertes ADM1) und der zentralen Zustandsindizes.  
* [PINN v2](pinn.md) — das Kollokations-Trajektorien-PINN: Netz, Closures,  
  Physik-Residuum, kausaler Rolling-Window-Betrieb und die Ergebnisse der
  Strukturstudie (`estimation/pinn/`).  
* [GRU-Observer (PINN B)](observer.md) — der amortisierte Observer: offline  
  auf vielen simulierten Szenarien vortrainiert, dann nahezu sofortige
  Online-Inferenz mit self-supervised Feinabstimmung; wie gut er auf
  verschobene Datensätze generalisiert.

* [UKF ↔ PINN-Fusion](fusion.md) — Kombination einer UKF- und einer  
  PINN-Schätzung per Kovarianzschnitt.

## Hintergrund-Lektüre

Die formale Observability-Analyse ADM1-basierter Modelle ist in
[Observability → Literaturüberblick](../observability/literature_review.md)
dokumentiert.
