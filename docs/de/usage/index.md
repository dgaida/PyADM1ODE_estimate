# Nutzung — Übersicht

Praktische Anleitungen, wie das Repo im realen Betrieb genutzt wird.

## Inhalt

* [UKF im Einsatz](ukf.md) — Wie man den UKF für eine
  konkrete Anlage konfiguriert: `adm1da_full_spec()`-Factory,
  `ObservationModel`-Kanäle, `MeasurementCalendar` für sparse
  Lab-Messungen, Online-Schleife.
* [Twin-Experimente](twin_experiments.md) — End-to-End-Validierung gegen
  eine bekannte Wahrheit. Das vorgefertigte `run_twin_experiment.py`
  und Interpretation der Diagnose-Plots.
* [Kalibrierungs-Artefakt](calibration_artifact.md) — Wie das YAML-Format
  zwischen Kalibrierung und Schätzung aussieht und wie der Filter es
  beim Hochfahren aufträgt.

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

1. **Plant** über PyADM1ODE bauen — entweder die Beispiel-Builder
   `build_simple_plant()` / `build_multi_stage_plant()` aus
   `example_plants/` oder einen eigenen über die
   PyADM1ODE-`PlantConfigurator`-API.
2. **Kalibriertes Artefakt** via `load_artifact(...)` einlesen und mit
   `apply_to_plant(...)` auf die Plant auftragen.
3. **State-Vektor-Spec** über die `adm1da_full_spec()`-Factory bauen,
   liefert alle 41 ADM1-States mit Observability-basierten Defaults.
   Substrat-Eingänge als `InputSpec`-Liste anhängen. Bei besonderer
   Sensorik per `SensorQualityProfile` einzelne Blöcke aufwerten.
4. **Observation-Model** mit den verfügbaren Sensorkanälen verdrahten.
   Eingebaute Extractor-Funktionen decken Q_gas, Q_ch4, P_el, P_th_used,
   stored_volume und State-Auslesungen ab.
5. **MeasurementCalendar** für die Sample-Rate-Verwaltung anhängen,
   wenn Messungen mit unterschiedlichen Raten kommen.
6. **SR-UKF** instanziieren (= aktueller `UnscentedKalmanFilter`),
   `ukf.reset(x0, P0)` mit realistischer Initial-Kovarianz, in der
   Online-Schleife abwechselnd `predict()` und `update()` aufrufen.
7. Vor Produktiv-Einsatz: **Twin-Experiment** laufen lassen, NIS-Mean
   und Coverage prüfen (siehe [Twin-Experimente](twin_experiments.md)).
