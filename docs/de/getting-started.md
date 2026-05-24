# Erste Schritte

Diese Anleitung bringt einen lokalen UKF auf die Beine — von der Installation
bis zur ersten Zustandsschätzung.

## Voraussetzungen

* Python ≥ 3.10
* [`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE) (Basis-Paket mit dem
  mechanistischen ADM1da-Modell)
* Optional: [`PyADM1ODE_calibration`](https://github.com/dgaida/PyADM1ODE_calibration)
  zur Erzeugung kalibrierter Modelle und für anlagenspezifische Plant-Builder

## Installation

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
```

Details siehe [Installation](installation.md).

## Minimal-Beispiel

Das einfachste lauffähige Setup nutzt das produktive UKF auf einer
synthetischen Anlage:

```python
import numpy as np
from pyadm1ode_estimation.estimation import (
    StateChannel, StateVectorSpec,
    ADM1ProcessModel, ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter

# 1. Zustandsvektor deklarieren
spec = StateVectorSpec(digester_id="primary", channels=[
    StateChannel("X_ac", kind="adm1", adm1_index=27,
                 initial=1.2, initial_std=0.3, process_noise_std=0.1),
    StateChannel("Q_solid", kind="input_flow", input_substrate_index=0,
                 initial=35.0, initial_std=5.0, process_noise_std=0.5,
                 drift_model="ou", ou_mean=35.0, ou_theta=0.1,
                 lower=0.0, upper=80.0),
])

# 2. Prozess- und Messmodell (Plant wird vom Caller gebaut)
process = ADM1ProcessModel(plant, spec)
obs = ObservationModel(channels=[...])
ukf = UnscentedKalmanFilter(process, obs, spec)

# 3. Online-Schleife
dt = 1.0 / 24.0  # 1-Stunden-Schritt in Tagen
for t, y in measurements:
    ukf.predict(dt=dt)
    step = ukf.update(y, t=t)
    print(f"t={t:.2f}d  x_hat={step.x_hat}  NIS={step.nis:.2f}")
```

## Wo geht's weiter

* [Nutzung → UKF im Einsatz](usage/ukf.md) — Detail-Setup für eine Anlage.
* [Observability → Literaturüberblick](observability/literature_review.md) —
  Welche Zustände sind mit welchen Sensoren überhaupt schätzbar?
* [Beispiele](examples/index.md) — Twin-Experiment, Methodik.
