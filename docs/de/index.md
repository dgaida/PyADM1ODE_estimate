# PyADM1ODE Estimate

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE_estimate/actions/workflows/tests.yml)

**Zustandsschätzung für landwirtschaftliche Biogasanlagen** auf Basis des
mechanistischen Modells **ADM1da** (Schlattmann 2011, agrar-erweitertes ADM1),
implementiert in [`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE).

## Was dieses Repo macht

Im Live-Betrieb wird der Zustand einer realen Anlage aus laufenden
Sensordaten und einem kalibrierten Simulationsmodell geschätzt. Die
geschätzten Zustände sind die Eingabe für Regler, die Fütterempfehlungen
an den Anlagenbetreiber geben.

Datenfluss:

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

## Hauptmerkmale

* 🎯 **Mechanistisch und datengetrieben** — UKF auf dem vollen 41-State ADM1da
  plus geplante Deep-Learning- und Fusion-Schichten.
* 🔌 **Sensor-tolerant** — Channel-Gating für sparsame Lab-Messungen,
  OU-Drift für augmentierte Eingangsschätzung.
* 📊 **Diagnostisch** — NIS-Monitoring je Schritt, Innovation pro Channel,
  Joseph-Form Kovarianz-Update für lange Läufe.

## Drei Schätzansätze

| Ansatz | Status | Arbeitspaket |
|---|---|---|
| **UKF** — mechanistisch | produktiv | AP 4.2 |
| **Deep Learning Ensemble** | Skelett | AP 4.3 |
| **Fusion** (Covariance Intersection) | Platzhalter | AP 4.4 |

## Inhaltsverzeichnis

* [Erste Schritte](getting-started.md) — Schneller Einstieg.
* [Installation](installation.md) — Setup für verschiedene Umgebungen.
* [Nutzung](usage/index.md) — UKF konfigurieren, Kalibrierungs-Artefakt einlesen.
* [Observability](observability/index.md) — Welche Zustände sind aus welchen Sensoren schätzbar?
* [Theorie](theory/index.md) — ADM1da-Modell, PINN-Konzept.
* [Beispiele](examples/index.md) — End-to-End-Beispiele (Twin, echte Anlage).
* [API-Referenz](api/index.md) — Auto-generiert via mkdocstrings.

## Quickstart

```python
from pyadm1ode_estimation.estimation import (
    StateChannel, StateVectorSpec,
    ADM1ProcessModel, ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter
from pyadm1ode_estimation.artifacts import load_artifact, apply_to_plant

# 1. Kalibriertes Modell laden
artifact = load_artifact("calibrated/plant_2026-05-14.yaml")
plant = build_plant(schema)            # eigener Plant-Builder
apply_to_plant(artifact, plant, strict=True)

# 2. State-Vektor deklarieren
spec = StateVectorSpec(digester_id="primary", channels=[
    StateChannel("X_ac", kind="adm1", adm1_index=27,
                 initial=1.2, initial_std=0.3, process_noise_std=0.1),
    StateChannel("Q_solid", kind="input_flow", input_substrate_index=0,
                 initial=35.0, initial_std=5.0, process_noise_std=0.5,
                 drift_model="ou", ou_mean=35.0, ou_theta=0.1),
])

# 3. UKF aufsetzen und laufen lassen
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[...])
ukf = UnscentedKalmanFilter(process, obs, spec)

for t, y in measurements:
    ukf.predict(dt=1/24)
    step = ukf.update(y, t=t)
```

## Verwandte Projekte

* [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) — Basis-Paket mit ADM1da-Modell.
* [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) —
  Parameter-Kalibrierung gegen historische Daten, schreibt das Calibration-Artefakt.
