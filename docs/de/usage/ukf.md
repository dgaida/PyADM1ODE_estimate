# UKF im Einsatz

Praktisches Setup für den Square-Root-UKF auf einer Biogasanlage. Diese
Seite zeigt das **aktuelle** API (ab Refactor 2026). Die einzelnen
Bausteine sind:

* :func:`adm1da_full_spec` — Factory für den vollen 41-State ADM1da-Vektor
* :class:`UnscentedKalmanFilter` — Square-Root-UKF (Wan & Van der Merwe 2001)
* :class:`MeasurementCalendar` — Sample-Rate-Verwaltung pro Sensor
* :class:`SensorAdapter` — Anschluss an die PyADM1ODE-Sensor-Klassen

## State-Vektor deklarieren

Statt manuell ausgewählte Channels zusammenzustellen, baut die Factory
:func:`adm1da_full_spec` den **kompletten 41-State-Vektor** mit
sinnvollen Defaults aus der Observability-Analyse:

```python
from pyadm1ode_estimation.estimation import (
    adm1da_full_spec, InputSpec, SensorQualityProfile, Quality,
)

spec = adm1da_full_spec(
    digester_id="primary",
    substrate_inputs=[
        InputSpec("maize_silage",   substrate_index=0, initial_flow=26.8),
        InputSpec("slurry",         substrate_index=1, initial_flow=12.8),
        InputSpec("cereal_silage",  substrate_index=2, initial_flow=0.4),
    ],
)
# spec hat 44 Channels: 41 ADM1-Slots + 3 Substrat-Inputs.
```

Pro Channel wird intern eine `process_noise_std`, `initial_std`, ein
`drift_model` und ein OU-Mittelwert (bei `drift_model="ou"`) gesetzt,
abhängig von der Observability-Klasse des Channels:

| Block                              | Default-Quality | Drift-Modell |
| ---------------------------------- | --------------- | ------------ |
| methanogenesis (A)                 | STRONG          | random_walk  |
| charge_balance (D)                 | STRONG          | random_walk  |
| acidogenesis_substrates (B subset) | MEDIUM          | random_walk  |
| acidogenesis_biomass               | WEAK            | random_walk  |
| hydrolysis_sums (C)                | WEAK            | random_walk  |
| disintegration_split (PS/PF)       | PSPF            | OU           |
| nitrogen (E)                       | OPEN_LOOP       | OU           |
| inerts                             | OPEN_LOOP       | OU           |
| fa_block                           | OPEN_LOOP       | OU           |

**Per-Plant Overrides** über `SensorQualityProfile`, wenn z.B. dein Plant
ein GC-FID für Einzel-VFAs hat:

```python
spec = adm1da_full_spec(
    digester_id="primary",
    substrate_inputs=[...],
    sensor_quality=SensorQualityProfile(
        acidogenesis_biomass=Quality.MEDIUM,  # GC-FID erhöht die Sensorqualität
    ),
)
```

## Plant-Modell

Die Anlage selbst wird über das PyADM1ODE-Paket gebaut. Für Tutorials
und Tests stehen zwei Referenzanlagen bereit:

```python
from pyadm1ode_estimation.example_plants import (
    build_simple_plant,         # 1 Fermenter + 1 Storage + 1 BHKW
    build_multi_stage_plant,    # 3 Fermenter (Kaskade) + 2 BHKW
)

plant = build_multi_stage_plant()
```

Für eine echte Anlage baust du den `BiogasPlant` über die
PyADM1ODE-Konfigurator-API (`PlantConfigurator.add_digester`, `.add_chp`,
`.connect`, …). Die Builder in `example_plants/` sind dafür die
Vorlage.

## Process-Model

```python
from pyadm1ode_estimation.estimation import ADM1ProcessModel

process = ADM1ProcessModel(plant, spec)
process.snapshot()   # baseline für restore/sigma-points
```

`ADM1ProcessModel.step(x, dt)` ist der eigentliche Propagator. Snapshot
/Restore sorgt dafür, dass jeder Sigma-Punkt aus identischem
Plant-Zustand startet. Bei der vollen Spec werden zwar alle 41 ADM1-States
jeden Schritt neu gesetzt, doch die `BiogasPlant` trägt weiteren
gekoppelten Zustand, der *nicht* im Spec steht. Allen voran
`simulation_time` (die Fütterung ist zeitabhängig), den
Gasspeicher-Füllstand sowie BHKW-/Heizkreis-Zustand. Ohne Restore würden
diese von Sigma-Punkt zu Sigma-Punkt mitdriften und die Sample-Kovarianz
wäre unbrauchbar. (Bei einer Teil-Spec gilt das zusätzlich für die nicht
deklarierten ADM1-States.)

## Observation-Model

```python
from pyadm1ode_estimation.estimation import (
    ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.observation_model import (
    extract_q_gas_total, extract_q_ch4_total, make_state_extractor,
)

def make_ph_extractor(digester_id: str):
    def extractor(plant, x):
        val = plant.components[digester_id].outputs_data.get("pH", float("nan"))
        return 7.0 if not (val == val) else float(val)  # NaN-safe
    return extractor

# Substrat-Sensor-Indizes im State-Vektor nachschlagen.
def _idx(name):
    return next(i for i, c in enumerate(spec.channels) if c.name == name)

obs = ObservationModel(channels=[
    ObservationChannel("Q_gas",           extract_q_gas_total,        noise_std=10.0),
    ObservationChannel("Q_ch4",           extract_q_ch4_total,        noise_std=5.0),
    ObservationChannel("pH",              make_ph_extractor("primary"), noise_std=0.05),
    ObservationChannel("Q_maize_silage",  make_state_extractor(_idx("maize_silage")), noise_std=1.4),
    ObservationChannel("Q_slurry",        make_state_extractor(_idx("slurry")),       noise_std=0.7),
    ObservationChannel("Q_cereal_silage", make_state_extractor(_idx("cereal_silage")), noise_std=0.07),
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

# Initial-Schätzung aus Calibration-Artefakt oder pre-inoculated state:
x0 = spec.read_adm1_state(plant)
# augmentierte Substrat-Channels aus dem Operator-Setpoint:
for i, ch in enumerate(spec.channels):
    if ch.kind == "input_flow":
        x0[i] = ch.initial

# P0 muss zur tatsächlichen Initial-Unsicherheit passen.
# Bei 5 % relativer Unsicherheit auf jedem Channel:
import numpy as np
sigma_init = 0.05
P0 = np.diag((sigma_init * (np.abs(x0) + 1e-6)) ** 2)
ukf.reset(x0, P0)

# Im Live-Betrieb:
dt = 1.0 / 24.0  # 1-Stunden-Schritt in Tagen
for t, y_dict, gate_dict in measurement_stream:
    ukf.predict(dt=dt)
    step = ukf.update(y_dict, t=t, gate_values=gate_dict)

    # step.x_hat       — Posterior-Mittelwert
    # step.P           — Posterior-Kovarianz
    # step.y_pred      — h(x̂) pro aktivem Channel (UKF-internes ŷ, Jensen-biased)
    # step.y_std       — sqrt(diag(S)) für jeden Channel
    # step.innovation  — y_obs − y_pred
    # step.nis         — Normalised Innovation Squared
    # step.active_channels — welche Channels haben Update geliefert
```

## Square-Root-UKF

Der `UnscentedKalmanFilter` propagiert den Cholesky-Faktor `S` (mit
`P = S·Sᵀ`) statt der vollen Kovarianz `P`. 

Effekte:

* `κ(S) = √κ(P)` → halbierte Konditionszahl → bei n=44 strukturell
  robuster als der frühere Cholesky-UKF.
* **Positiv-Definitheit per Konstruktion**: `P = SSᵀ` kann nie indefinit
  werden.
* Bei `_cholupdate`-Downdate-Fehler weißt du strukturell, dass dein
  `Q`/`R`-Tuning oder dein Mess-Modell ein echtes Problem hat, kein
  numerischer Glitch (siehe [Troubleshooting](../troubleshooting.md)).

## Sample-Rate-Verwaltung mit MeasurementCalendar

In der Praxis kommen Messungen mit sehr unterschiedlichen Raten:

| Sensor                   | Rate       |
| ------------------------ | ---------- |
| Q_gas (Gas-Volumenstrom) | alle 5 min |
| pH online                | alle 5 min |
| FOS/TAC (Titration)      | täglich   |
| NH4-N (Lab)              | sporadisch |

Der `MeasurementCalendar` baut pro Filter-Schritt den `gate_values`-Dict
aus einer Mess-DataFrame:

```python
from pyadm1ode_estimation.estimation import MeasurementCalendar, SampleRate

calendar = MeasurementCalendar({
    "Q_gas":   SampleRate.online(period_min=5),
    "pH":      SampleRate.online(period_min=5),
    "FOS_TAC": SampleRate.daily(),
    "NH4_N":   SampleRate.sporadic(),
})

# Im Filter-Loop:
for t in time_grid:
    y, gates = calendar.values_for_filter(t=t, df=measurements_df)
    ukf.predict(dt=dt)
    ukf.update(y=y, t=t, gate_values=gates)
```

Sporadische Lab-Messungen werden so genau in einem Schritt aktiv. Der
Filter braucht *keine* Restrukturierung, wenn z.B. drei Monate später
einmal eine NH4-N-Lab-Probe kommt. Channel-Slot bleibt im State-Vektor
ohne Messung läuft er als OU-Drift.

## NIS-Monitoring

`step.nis` ist der zentrale Konsistenz-Indikator:

| Bereich                      | Diagnose                                              |
| ---------------------------- | ----------------------------------------------------- |
| `NIS ≈ n_active_channels` | Filter gut kalibriert                                 |
| `NIS ≫ n_active_channels` | Filter unterschätzt Unsicherheit (Q oder R zu klein) |
| `NIS ≪ n_active_channels` | Filter überschätzt Unsicherheit (Q oder R zu groß) |

Faustregel für gut-kalibrierten Filter: `NIS ∈ [0.5·n, 2.0·n]`.

Bei 6 Mess-Channels heißt das `NIS ∈ [3, 12]`. Wenn der **mittlere NIS
über mehrere Tage** außerhalb dieses Fensters liegt:

* `NIS > 3·n` über Stunden → Filter divergiert wahrscheinlich (z.B.
  nicht modellierter Inhibitor, ungedoste Substrat-Schwankung).
* `NIS < 0.3·n` → Sensoren-`R` ist deutlich zu groß angegeben oder das
  Process-Noise `Q` ist überdimensioniert.

Siehe auch [Twin-Experimente](twin_experiments.md) für ein End-to-End Beispiel mit Plot-Diagnose.
