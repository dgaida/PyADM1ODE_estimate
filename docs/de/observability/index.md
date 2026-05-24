# Observability — Übersicht

Observability beantwortet die zentrale Frage:

> Welche internen Zustände kann ein Beobachter (UKF, EKF, Asymptotic Observer)
> aus den verfügbaren Messungen *prinzipiell* rekonstruieren, und welche
> nicht — unabhängig von der gewählten Filter-Architektur?

Das ist eine **Modell-Eigenschaft**, keine Algorithmus-Eigenschaft. Ein
besserer Filter kann unobservable Zustände nicht schätzbar machen.

## Warum das hier wichtig ist

Das ADM1da-Modell hat **41 Zustände**. Eine reale Biogasanlage hat typischerweise
3-6 informative Online-Sensoren plus eventuell wöchentliche Laborwerte. Die
Diskrepanz ist groß. Vor jedem Filter-Setup muss klar sein, welche
Untermenge der 41 Zustände aus der konkreten Sensorik wirklich schätzbar ist.

## Was hier dokumentiert ist

* **Schritt 1 — [Literaturüberblick](literature_review.md)** — Was sagen
  Hellmann et al. 2023, Gaida et al. 2012 und Haugen et al. 2014 dazu?
  Welche Mess-Sets erschließen welche Modellklassen? Wo sind die
  strukturellen Hebel (pH, CH₄/CO₂-Trennung, Lab-Werte)?
* **Schritt 2 — [Sensor-Zustand-Abhängigkeiten](sensor_state_dependencies.md)**
  — Konkreter Abgleich der Literatur-Erkenntnisse mit unserer ADM1da-
  Implementierung. Pro Mess-Kanal mit Code-Zitat aus
  `pyadm1/core/adm1.py`: welche Zustände sind direkt im Messmodell,
  welche werden 1-Schritt-indirekt über die ODE-Kopplung erschlossen?
  Master-Tabelle + STRIKE-GOLDD-Anwendbarkeit.

## Geplante Erweiterungen

* **Sensitivitätsanalyse**: numerischer Test, welche Channels bei welchem
  `Q`/`R`-Tuning tatsächlich vom Filter getrennt werden.
