# UKF im Einsatz

Konkretes Setup des produktiven UKF auf einer Biogasanlage.

## State-Vektor deklarieren

Der `StateVectorSpec` legt fest, *was* der Filter schätzt. Drei Sorten Kanäle:

| `kind` | Bedeutung | Pflichtfeld |
|---|---|---|
| `"adm1"` | Direkt ein Slot im 41-State ADM1da-Vektor | `adm1_index` |
| `"input_flow"` | Augmentierte Substrat-Zufuhr (m³/d) | `input_substrate_index` |
| `"kinetic_param"` | Augmentierter kinetischer Parameter | (Name muss in `adm1._kinetic` existieren) |

Beispiel-Konfiguration für eine landwirtschaftliche Biogasanlage
(6 biologische + 2 Inputs = 8 Kanäle):

```python
from pyadm1ode_estimation.estimation import StateChannel, StateVectorSpec

ADM1_CHANNELS = [
    # (name,     idx, initial, init_std, proc_std)
    ("S_ac",       6,   0.10,    0.30,    0.50),
    ("X_PS_ch",   12,  19.16,    5.00,    1.00),
    ("X_PF_ch",   15,   2.28,    1.00,    0.50),
    ("X_su",      22,   0.50,    0.30,    0.30),
    ("X_ac",      27,   1.20,    0.30,    0.10),
    ("X_h2",      28,   0.30,    0.10,    0.05),
]

INPUT_CHANNELS = [
    # (name,    sub_idx, init, init_std, proc_std, lower, upper, ou_mean)
    ("Q_solid",  0, 35.0, 5.0, 0.5,  0.0,  80.0, 35.0),
    ("Q_liquid", 2, 28.0, 5.0, 1.0,  0.0, 100.0, 28.0),
]

channels = []
for name, idx, init, init_std, proc_std in ADM1_CHANNELS:
    channels.append(StateChannel(
        name=name, kind="adm1", adm1_index=idx,
        initial=init, initial_std=init_std,
        process_noise_std=proc_std,
        lower=1e-6, upper=100.0,
    ))
for name, sub_idx, init, init_std, proc_std, lo, hi, ou_mean in INPUT_CHANNELS:
    channels.append(StateChannel(
        name=name, kind="input_flow", input_substrate_index=sub_idx,
        initial=init, initial_std=init_std,
        process_noise_std=proc_std,
        drift_model="ou", ou_mean=ou_mean, ou_theta=0.1,
        lower=lo, upper=hi,
    ))

spec = StateVectorSpec(digester_id="primary", channels=channels)
```

## Prozess-Modell

```python
from pyadm1ode_estimation.estimation import ADM1ProcessModel

plant = build_plant(schema)               # anlagenspezifischer Plant-Builder
process = ADM1ProcessModel(plant, spec)
```

`ADM1ProcessModel.step(x, dt)` ist der eigentliche Propagator. Snapshot/Restore
sorgt dafür, dass jeder Sigma-Punkt aus identischem Plant-Zustand startet.

## Observation-Model

`ObservationModel` aggregiert `ObservationChannel`s. Eingebaute Extractoren
decken Q_gas, Q_ch4, Q_gas_consumed, P_el und P_th_used ab.

```python
from pyadm1ode_estimation.estimation import (
    ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.observation_model import (
    BUILT_IN_EXTRACTORS, make_state_extractor,
)

q_solid_idx = next(i for i, c in enumerate(spec.channels) if c.name == "Q_solid")

obs = ObservationModel(channels=[
    ObservationChannel(
        name="Q_gas",
        extractor=BUILT_IN_EXTRACTORS["Q_gas"],
        noise_std=150.0,
    ),
    ObservationChannel(
        name="P_el",
        extractor=BUILT_IN_EXTRACTORS["P_el"],
        noise_std=5.0,
    ),
    ObservationChannel(
        name="hopper_dose",
        extractor=make_state_extractor(q_solid_idx),
        noise_std=2.0,
        gate_column="hopper_observable",     # nur aktiv wenn ΔW < 0
    ),
])
```

## Filter aufsetzen und betreiben

```python
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(
    process, obs, spec,
    alpha=1.0,    # unskalierte UKF — robust für stark nichtlineare ADM1-Kinetik
    beta=2.0,
    kappa=0.0,
)

dt = 1.0 / 24.0  # 1-Stunden-Schritt in Tagen

for t, y_dict, gate_dict in measurement_stream:
    ukf.predict(dt=dt)
    step = ukf.update(y_dict, t=t, gate_values=gate_dict)

    # step.x_hat       — Posterior-Mittelwert
    # step.P           — Posterior-Kovarianz
    # step.innovation  — y_obs - y_pred pro aktivem Kanal
    # step.nis         — Normalised Innovation Squared (Konsistenz-Indikator)
    # step.active_channels — welche Channels haben Update geliefert
```

## NIS-Monitoring

Der zentrale Konsistenz-Indikator ist `step.nis`:

* **NIS ≈ Anzahl aktiver Channels** → Filter gut kalibriert.
* **NIS ≫ n_obs** → Filter unterschätzt die Unsicherheit (`Q` oder `R` zu klein).
* **NIS ≪ n_obs** → Filter überschätzt die Unsicherheit; konservativ aber wenig informativ.

Plausible Grenzen für Alarmierung: NIS > 3 × n_obs über mehrere Stunden = Filter
divergiert wahrscheinlich (z.B. nicht modellierter Inhibitor wirkt).
