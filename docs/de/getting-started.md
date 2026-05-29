# Erste Schritte

Diese Anleitung bringt einen lokalen UKF auf die Beine. Der UKF wird mit der Funktion `build_ukf()` eingerichtet und erstellt.

## Voraussetzungen

* Python ≥ 3.10
* [`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE) — Basis-Paket mit
  dem ADM1da-Modell
* Optional: [`PyADM1ODE_calibration`](https://github.com/dgaida/PyADM1ODE_calibration)
  für kalibrierte Modelle aus historischen Anlagendaten

## Installation

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
```

Details siehe [Installation](installation.md).

## Minimal-Setup

Das komplette Setup für die Multi-Stage-Beispielanlage:

```python
from pyadm1ode_estimation.estimation import InputSpec, build_ukf
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

# 1. Anlage bauen
plant = build_multi_stage_plant()

# 2. UKF aufsetzen — eine einzige Funktion
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[
        InputSpec("maize_silage",  substrate_index=0, initial_flow=26.8),
        InputSpec("slurry",        substrate_index=1, initial_flow=12.8),
        InputSpec("cereal_silage", substrate_index=2, initial_flow=0.4),
    ],
    sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
)

# 3. Messdaten-Stream: Iterable von (t [d], {sensor: wert}).
#    Hier aus einer CSV mit je einer Spalte pro Sensor (Q_gas, Q_ch4, pH, Q_maize_silage, ...):
import pandas as pd
df = pd.read_csv("messwerte.csv", index_col="t")
measurement_stream = ((t, row.dropna().to_dict()) for t, row in df.iterrows())

# 4. Online-Schleife — zwei Zeilen pro Schritt
for t, measurements in measurement_stream:
    ukf.predict(dt=1.0 / 24.0)   # 1-Stunden-Schritt
    step = ukf.update(measurements, t=t)
    print(f"t={t:.2f}d  S_ac={step.x_hat[6]:.3f}  NIS={step.nis:.2f}")
```

Das war's. Der Filter schätzt jetzt alle **41 ADM1-States + 3
Substrat-Inputs** aus den vier deklarierten Sensoren.

### Woher kommt `measurement_stream`?

`measurement_stream` ist ein Iterable, das pro
Zeitschritt ein Tupel `(t, measurements)` liefert:

* `t` — Zeit in Tagen (`float`),
* `measurements` — ein `dict` `{sensor: wert}`. Die Zuordnung erfolgt
  **über den Namen, nicht über die Reihenfolge**, und ist tolerant:
  Groß-/Kleinschreibung sowie Trenner (`_`, `-`, Leerzeichen) spielen
  keine Rolle und das `Q_`-Präfix ist optional. `"Q_gas"`, `"q_gas"` und
  `"gas"` treffen also denselben Kanal, `"maize_silage"` trifft
  `"Q_maize_silage"`. Mehrdeutige, unbekannte oder `NaN`-Keys werden in
  diesem Schritt einfach übersprungen.

Hast du noch keine echten Anlagendaten, erzeugt das Skript
[`run_twin_experiment.py`](usage/twin_experiments.md) genau so einen Stream
aus einer simulierten Wahrheitsanlage. Ideal zum Validieren, bevor echte
Sensoren angeschlossen werden.

## Was `build_ukf()` macht

In einem einzigen Aufruf passieren intern:

| Schritt | Was passiert |
| --- | --- |
| 1 | `adm1da_full_spec()` baut den 41-State-Vektor mit Observability-Defaults |
| 2 | Sensor-Strings werden in `ObservationChannel`s übersetzt |
| 3 | `ADM1ProcessModel` wrappt die Plant + Snapshot |
| 4 | `UnscentedKalmanFilter` wird instanziiert |
| 5 | Initial-State aus Plant lesen, Initial-Kovarianz setzen |
| 6 | `ukf.reset(x0, P0)` — Filter ist bereit |

Sensor-Strings aus dem Built-in-Katalog:

| Name | Bedeutung | Default-Rauschen |
| --- | --- | --- |
| `"q_gas"` | totaler Biogas-Volumenstrom | 10 m³/d |
| `"q_ch4"` | totaler Methan-Volumenstrom | 5 m³/d |
| `"ph"` | pH des primären Fermenters | 0.05 |
| `"substrate_dose"` | direkter Sensor pro Substrat-Slot | 5 % relativ |

Robust gegen NaN: Liefert das Modell für einen Kanal eine nicht-finite
Vorhersage `h(x)` (z. B. an einem extremen Sigma-Punkt), wird dieser Kanal
in dem Schritt **übersprungen**. Der Zustand wird also nicht mit einer
unzuverlässigen Vorhersage korrigiert, genau wie bei einer fehlenden
Messung. Sind alle Kanäle betroffen, läuft nur `predict()` (keine
Korrektur).

**Option 1 — Rauschen pro Sensor überschreiben:**

```python
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[...],
    sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
    sensor_noise={"q_gas": 25.0, "ph": 0.1},   # m³/d bzw. pH-Einheiten
)
```

**Option 2 — eigenes `ObservationChannel` direkt übergeben** (beliebig mit
Katalog-Strings mischbar):

```python
from pyadm1ode_estimation.estimation import ObservationChannel

# Nicht-Standard-Sensor: CO₂-Anteil im Biogas des primären Fermenters.
co2 = ObservationChannel(
    name="x_co2",
    extractor=lambda plant, x: plant.components["primary"].outputs_data.get("x_co2", 0.0),
    noise_std=0.02,
)
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[...],
    sensors=["q_gas", "q_ch4", co2],   # Katalog + Custom gemischt
)
# Im Messdaten-dict heißt der Kanal dann "x_co2".
```

### Was bedeutet der Sensor `substrate_dose`?

Wichtig: `substrates=[...]` und der Sensor `"substrate_dose"` sind **zwei
verschiedene Dinge**:

* `substrates=[InputSpec(...)]` **deklariert** die Substrat-Zuflüsse als
  zusätzliche Zustände im State-Vektor. Der Filter schätzt also den
  *tatsächlich gefütterten* Volumenstrom.
* `"substrate_dose"` fügt **Sensoren** hinzu, die genau diese Zustände
  *messen*: pro deklariertem Substrat einen Kanal. Aus dem einen String
  `"substrate_dose"` werden also mehrere Kanäle, hier `Q_maize_silage`,
  `Q_slurry`, `Q_cereal_silage`.

`"substrate_dose"` ist somit kein einzelner Kanal, sondern eine Abkürzung
für „je ein Dosier-Sensor pro Substrat". Im Messdaten-`dict` gibst du
entsprechend **pro Substrat** einen Wert an, nicht unter dem Key
`"substrate_dose"`:

```python
measurements = {
    "q_gas": 410.0,
    "q_ch4": 228.0,
    "ph": 7.42,
    "maize_silage": 26.8,   # = Kanal Q_maize_silage
    "slurry": 12.8,
    "cereal_silage": 0.4,
}
```

Ein Key `"substrate_dose"` im Mess-`dict` würde zu **keinem** Kanal passen
und daher ignoriert. Lässt du den Sensor `"substrate_dose"` ganz weg,
bleiben die Substrat-Zuflüsse trotzdem geschätzte Zustände, nur eben
*ohne* direkte Messung (schwächer beobachtbar, nur indirekt über Gas/pH).

## Eigene Anlage statt Beispiel

Wenn deine Anlage anders aufgebaut ist als die Beispiel-Builder, baust
du die Plant direkt mit der PyADM1ODE-API:

```python
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

feedstock = Feedstock(["maize_silage", "slurry"], feeding_freq=24, total_simtime=365)
plant = BiogasPlant("Meine Anlage")
cfg = PlantConfigurator(plant, feedstock)

cfg.add_digester(digester_id="primary", V_liq=1200.0, V_gas=216.0,
                 T_ad=315.15, Q_substrates=[20.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0])
cfg.add_chp(chp_id="chp1", P_el_nom=250.0, eta_el=0.40, eta_th=0.45)
cfg.auto_connect_digester_to_chp("primary", "chp1")
plant.initialize()

# Danach genauso:
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[InputSpec("maize_silage", 0, 20.0), InputSpec("slurry", 1, 10.0)],
)
```

Sieh die [Beispiel-Plant-Builder](https://github.com/dgaida/PyADM1ODE_estimate/blob/main/pyadm1ode_estimation/example_plants/multi_stage.py)
als Vorlage für komplexere Topologien (Mehrstufige Kaskade, mehrere
BHKWs, Heizkreisläufe).

## Vor dem Produktiv-Einsatz: Twin-Test

Es ist gute Praxis, den Filter erst gegen eine bekannte Wahrheit zu
validieren. Das vorgefertigte Twin-Skript macht das in einem Lauf:

```bash
python examples/run_twin_experiment.py --warmup-days 30 --duration-days 5
```

Erzeugt 6 Diagnose-Plots in `output/twin_experiment/`. Erwartete
Ergebnisse: 2σ-Coverage deutlich über dem 80 %-Ziel auf strong-observable
Blöcken, NIS-Mean um den Erwartungswert (= Anzahl der Mess-Channels).
Mehr dazu:
[Twin-Experimente](usage/twin_experiments.md).

## Wo geht's weiter

* [Nutzung → UKF im Einsatz](usage/ukf.md) — Detaillierte Erklärung
  der einzelnen Bausteine + `MeasurementCalendar` für sporadische
  Lab-Messungen.
* [Nutzung → Twin-Experimente](usage/twin_experiments.md) —
  End-to-End-Validierung gegen eine bekannte Wahrheit.
* [Observability → Sensor-Zustand-Abhängigkeiten](observability/sensor_state_dependencies.md) —
  Welche Zustände sind mit welchen Sensoren prinzipiell schätzbar?
