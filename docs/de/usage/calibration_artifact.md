# Kalibrierungs-Artefakt

Das `CalibrationArtifact` ist die **Schnittstelle zwischen Kalibrierung und Schätzung**.
Es ist eine versionierte YAML-Datei, die Calibration schreibt und Estimation
beim Hochfahren des Live-Filters einliest.

## Workflow

```text
PyADM1ODE_calibration                     PyADM1ODE_estimate
  │                                            │
  │  optimiert kinetische Parameter,           │
  │  fittet gegen historische Messdaten        │
  │                                            │
  ▼                                            │
[ artifact.yaml ] ───────── handoff ──────────►│
                                               │
                                               │  load_artifact(...)
                                               │  apply_to_plant(...)
                                               │
                                               ▼
                                        kalibrierte BiogasPlant
                                               │
                                               ▼
                                        UKF startet von hier
```

## Format-Übersicht

Das YAML hat fünf Hauptsektionen (jede außer `metadata` ist optional):

```yaml
schema_version: 1

metadata:                # Provenienz: welche Anlage, welcher Datenraum
  plant_id: plant_id_xyz
  calibration_run_id: 2026-05-14-3w
  timestamp: 2026-05-14T12:30:00Z
  data_window_start: 2026-04-01
  data_window_end: 2026-04-21
  adm1_version: "ADM1da-0.3.4"
  fitted_against: [Q_gas, P_el]

kinetic:                 # adm1._kinetic-Overrides pro Fermenter
  primary:
    k_dis_PS: 0.045
    k_hyd_ch: 4.2
    k_m_ac: 8.5

substrates:              # Substrat-Disintegrations-Fraktionen
  maize_silage:
    f_ch_xc: 0.55

initial_state:           # ADM1da-Zustand am Ende des Cal-Fensters
  primary:
    _values: [...41 Werte in kanonischer ADM1da-Reihenfolge...]

residuals:               # Fit-Qualität pro Channel (informativ)
  Q_gas:
    rmse: 145.0
    units: m3/d
```

Vollständiges Beispiel:
[calibration_artifact_example.yaml](../../assets/calibration_artifact_example.yaml).

## Schreiben (Calibration-Seite)

```python
from pyadm1ode_estimation.artifacts import (
    CalibrationArtifact, CalibrationMetadata, save_artifact,
)

artifact = CalibrationArtifact(
    metadata=CalibrationMetadata(
        plant_id="plant_id_xyz",
        calibration_run_id="2026-05-14-3w",
        timestamp="2026-05-14T12:30:00Z",
        fitted_against=["Q_gas", "P_el"],
    ),
    kinetic={"primary": {"k_dis_PS": 0.045, "k_hyd_ch": 4.2}},
    initial_state={"primary": {"_values": list_of_41_floats}},
)
save_artifact(artifact, "calibrated/plant_2026-05-14.yaml")
```

## Lesen (Estimation-Seite)

```python
from pyadm1ode_estimation.artifacts import load_artifact, apply_to_plant

artifact = load_artifact("calibrated/plant_2026-05-14.yaml")
plant = build_plant(schema)            # anlagenspezifischer Plant-Builder

# strict=True: bei Topologie-Mismatch fail-fast (empfohlen im Produktiv-Start)
# strict=False: nur Warnungen, graceful Degradation (gut für Tests, alte Artefakte)
diag = apply_to_plant(artifact, plant, strict=True)

print(f"Angewendet: {len(diag['applied'])} Parameter")
print(f"Übersprungen: {len(diag['skipped'])} Parameter")
```

## Designentscheidungen

| Entscheidung | Warum |
|---|---|
| `schema_version` Pflicht | Unbekannte Versionen → `ValueError`, kein Silent-Mis-Read |
| `plant_id` + `run_id` Pflicht | Audit-Trail bei mehreren parallelen Kalibrierungen |
| Sektionen optional | Partielle Calibrations (nur kinetic) erlaubt |
| `strict=False` Default | Graceful Degradation bei evolvierter Topologie |
| `strict=True` für Produktion | Live-Filter-Startup soll bei Inkompatibilität fail-fast |
| Dict-Strukturen statt feste Felder | Plant-agnostisch — neue Anlage erfordert kein Schema-Update |
| `_values: [41 Werte]` Array-Form | Pragmatisch ohne pyadm1-Index-Map-Introspektion |
