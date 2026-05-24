# Beispiele — Übersicht

Methodische End-to-End-Beispiele für den Einsatz des UKF auf einer
landwirtschaftlichen Biogasanlage. Die Beispiele zeigen das Vorgehen
ohne anlagenspezifische Details — eine konkrete Anlage wird separat
und privat dokumentiert.

## Im Repo (Skripte)



## Geplant

* **Validierungs-Bericht** — eine zusammenfassende Auswertung, wie die
  Methoden (UKF, später Deep-Learning-Ensemble, später Fusion) gegen
  Referenz-Daten getestet wurden. Inhalt: Genauigkeit, Robustheit,
  Konsistenz-Diagnostik (NIS), Konvergenzverhalten.

## Vorgehen für eine eigene Anlage

1. **Plant-Topologie** über die `pyadm1.configurator`-API bauen oder
   einen Plant-Builder aus einem privaten Repo importieren.
2. **Kalibriertes Artefakt** mit `load_artifact(...)` einlesen und mit
   `apply_to_plant(...)` auftragen.
3. **State-Vektor-Spec** zuschneiden — nicht alle 41 ADM1-Zustände
   sind aus typischer SCADA-Sensorik schätzbar; siehe
   [Observability → Literaturüberblick](../observability/literature_review.md).
4. **Observation-Model** mit den verfügbaren Sensor-Kanälen verdrahten.
5. **UKF** instanziieren, in der Online-Schleife abwechselnd `predict()`
   und `update()` aufrufen, NIS überwachen.
