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
    ADM1ProcessModel,
    InputSpec,
    ObservationChannel,
    ObservationModel,
    adm1da_full_spec,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter
from pyadm1ode_estimation.estimation.observation_model import (
    extract_q_gas_total, extract_q_ch4_total,
)
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

# 1. Anlage bauen (Beispiel oder eigener PyADM1ODE-Builder)
plant = build_multi_stage_plant()

# 2. Voller 41-State-Spec mit Substrat-Augmentation
spec = adm1da_full_spec(
    digester_id="primary",
    substrate_inputs=[
        InputSpec("maize_silage",   substrate_index=0, initial_flow=4.74),
        InputSpec("solid_manure",   substrate_index=1, initial_flow=13.70),
        InputSpec("chicken_litter", substrate_index=2, initial_flow=1.09),
        InputSpec("slurry",         substrate_index=3, initial_flow=3.68),
        InputSpec("cereal_grain",   substrate_index=4, initial_flow=0.20),
    ],
)

# 3. Process- und Observation-Model
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[
    ObservationChannel("Q_gas", extract_q_gas_total, noise_std=10.0),
    ObservationChannel("Q_ch4", extract_q_ch4_total, noise_std=5.0),
])

# 4. SR-UKF aufsetzen und laufen lassen
ukf = UnscentedKalmanFilter(process, obs, spec)
for t, y in measurements:
    ukf.predict(dt=1/24)
    step = ukf.update(y, t=t)
```

Mehr Details: [UKF im Einsatz](usage/ukf.md) und
[Twin-Experimente](usage/twin_experiments.md).

## Verwandte Projekte

* [PyADM1ODE](https://github.com/dgaida/PyADM1ODE) — Basis-Paket mit ADM1da-Modell.  
* [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) —  
  Parameter-Kalibrierung gegen historische Daten, schreibt das Calibration-Artefakt.
