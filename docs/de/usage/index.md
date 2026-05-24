# Nutzung — Übersicht

Praktische Anleitungen, wie das Repo im realen Betrieb genutzt wird.

## Inhalt

* [UKF im Einsatz](ukf.md) — Wie man einen `UnscentedKalmanFilter` für eine
  konkrete Anlage konfiguriert: `StateVectorSpec` aufsetzen,
  `ObservationModel`-Kanäle definieren, Online-Schleife schreiben.
* [Kalibrierungs-Artefakt](calibration_artifact.md) — Wie das YAML-Format zwischen
  Kalibrierung und Schätzung aussieht und wie der Filter es beim Hochfahren
  aufträgt.

## Gesamtbild

Im Live-Betrieb sieht der Datenfluss so aus:

```text
Historische Messdaten ──► PyADM1ODE_calibration ──► kalibriertes Modell
                                                            │
Live-Sensoren ────────────────────────────────────► PyADM1ODE_estimate
                                                            │
                                                            ▼
                                                 Zustandsschätzung x_hat
                                                            │
                                                            ▼
                                                       Regler / UI
```

Schritte für ein neues Setup:

1. **Plant-Topologie** aus `pyadm1ode_calibration.plants` oder eigener
   Implementierung holen.
2. **Kalibriertes Artefakt** via `load_artifact(...)` einlesen und mit
   `apply_to_plant(...)` auftragen.
3. **State-Vektor-Spec** aufsetzen — welche ADM1-Indizes und welche
   augmentierten Eingangsraten sollen geschätzt werden?
   (Siehe [Observability-Literaturüberblick](../observability/literature_review.md)
   für die Sensor-→-Zustands-Logik.)
4. **Observation-Model** mit den verfügbaren Sensorkanälen verdrahten —
   `BUILT_IN_EXTRACTORS` deckt die gängigen Q_gas/P_el-Signale ab.
5. **UKF** instanziieren, in der Online-Schleife abwechselnd `predict()`
   und `update()` aufrufen.
