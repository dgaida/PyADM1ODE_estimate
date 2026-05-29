# Getting started

This guide brings up a local UKF. The UKF is set up and created with
the `build_ukf()` function.

## Prerequisites

* Python ≥ 3.10
* [`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE) — base package
  with the ADM1da model
* Optional: [`PyADM1ODE_calibration`](https://github.com/dgaida/PyADM1ODE_calibration)
  for calibrated models from historical plant data

## Installation

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
```

See [Installation](installation.md) for details.

## Minimal setup

Complete setup for the multi-stage reference plant:

```python
from pyadm1ode_estimation.estimation import InputSpec, build_ukf
from pyadm1ode_estimation.example_plants import build_multi_stage_plant

# 1. Build the plant
plant = build_multi_stage_plant()

# 2. Set up the UKF — a single function
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

# 3. Measurement stream: an iterable of (t [d], {sensor: value}).
#    Here from a CSV with one column per sensor (Q_gas, Q_ch4, pH, Q_maize_silage, ...):
import pandas as pd
df = pd.read_csv("measurements.csv", index_col="t")
measurement_stream = ((t, row.dropna().to_dict()) for t, row in df.iterrows())

# 4. Online loop — two lines per step
for t, measurements in measurement_stream:
    ukf.predict(dt=1.0 / 24.0)   # 1-hour step
    step = ukf.update(measurements, t=t)
    print(f"t={t:.2f}d  S_ac={step.x_hat[6]:.3f}  NIS={step.nis:.2f}")
```

That's it. The filter now estimates all **41 ADM1 states + 3
substrate inputs** from the four declared sensors.

### Where does `measurement_stream` come from?

`measurement_stream` is an iterable that yields one
`(t, measurements)` tuple per time step:

* `t` — time in days (`float`),
* `measurements` — a `dict` `{sensor: value}`. Channels are matched **by
  name, not by position**, and matching is forgiving: case and separators
  (`_`, `-`, spaces) are ignored and the `Q_` prefix is optional. So
  `"Q_gas"`, `"q_gas"` and `"gas"` all hit the same channel, and
  `"maize_silage"` hits `"Q_maize_silage"`. Ambiguous, unknown or `NaN`
  keys are simply skipped for that step.

If you don't have real plant data yet,
[`run_twin_experiment.py`](usage/twin_experiments.md) generates exactly
such a stream from a simulated truth plant. Ideal for validating the
filter before wiring up real sensors.

## What `build_ukf()` does

In a single call, internally:

| Step | What happens |
| --- | --- |
| 1 | `adm1da_full_spec()` builds the 41-state vector with observability defaults |
| 2 | Sensor strings are translated into `ObservationChannel` instances |
| 3 | `ADM1ProcessModel` wraps the plant + snapshot |
| 4 | `UnscentedKalmanFilter` is instantiated |
| 5 | Initial state read from the plant, initial covariance set |
| 6 | `ukf.reset(x0, P0)` — filter is ready |

Sensor strings from the built-in catalog:

| Name | Meaning | Default noise |
| --- | --- | --- |
| `"q_gas"` | total biogas flow | 10 m³/d |
| `"q_ch4"` | total methane flow | 5 m³/d |
| `"ph"` | pH of the primary fermenter | 0.05 |
| `"substrate_dose"` | one direct sensor per substrate slot | 5 % relative |

Robust to NaN: if the model returns a non-finite prediction `h(x)` for a
channel (e.g. at an extreme sigma point), that channel is **skipped** for
the step. The state is not corrected with an unreliable prediction, just
like a missing measurement. If every channel is affected, only `predict()`
runs (no correction).

**Option 1 — override per-sensor noise:**

```python
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[...],
    sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
    sensor_noise={"q_gas": 25.0, "ph": 0.1},   # m³/d / pH units
)
```

**Option 2 — pass your own `ObservationChannel`** (freely mixable with
catalog strings):

```python
from pyadm1ode_estimation.estimation import ObservationChannel

# Non-standard sensor: CO₂ fraction in the primary fermenter's biogas.
co2 = ObservationChannel(
    name="x_co2",
    extractor=lambda plant, x: plant.components["primary"].outputs_data.get("x_co2", 0.0),
    noise_std=0.02,
)
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[...],
    sensors=["q_gas", "q_ch4", co2],   # catalog + custom mixed
)
# In the measurement dict the channel is then called "x_co2".
```

### What does the `substrate_dose` sensor mean?

Important: `substrates=[...]` and the `"substrate_dose"` sensor are **two
different things**:

* `substrates=[InputSpec(...)]` **declares** the substrate feeds as extra
  states in the state vector. The filter therefore estimates the
  *actually delivered* volumetric flow.
* `"substrate_dose"` adds **sensors** that *measure* those very states:
  one channel per declared substrate. The single string `"substrate_dose"`
  thus expands into several channels — here `Q_maize_silage`, `Q_slurry`,
  `Q_cereal_silage`.

So `"substrate_dose"` is not a single channel but shorthand for "one
dosing sensor per substrate". In the measurement `dict` you accordingly
provide one value **per substrate**, not under a `"substrate_dose"` key:

```python
measurements = {
    "q_gas": 410.0,
    "q_ch4": 228.0,
    "ph": 7.42,
    "maize_silage": 26.8,   # = channel Q_maize_silage
    "slurry": 12.8,
    "cereal_silage": 0.4,
}
```

A `"substrate_dose"` key in the measurement `dict` would match **no**
channel and is ignored. If you drop the `"substrate_dose"` sensor
entirely, the substrate feeds remain estimated states — just *without* a
direct measurement (weakly observable, only indirectly via gas / pH).

## A custom plant instead of the example

If your plant differs from the bundled example builders, you build the
plant directly via the PyADM1ODE API:

```python
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

feedstock = Feedstock(["maize_silage", "slurry"], feeding_freq=24, total_simtime=365)
plant = BiogasPlant("My plant")
cfg = PlantConfigurator(plant, feedstock)

cfg.add_digester(digester_id="primary", V_liq=1200.0, V_gas=216.0,
                 T_ad=315.15, Q_substrates=[20.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0])
cfg.add_chp(chp_id="chp1", P_el_nom=250.0, eta_el=0.40, eta_th=0.45)
cfg.auto_connect_digester_to_chp("primary", "chp1")
plant.initialize()

# Then exactly the same:
ukf = build_ukf(
    plant,
    digester_id="primary",
    substrates=[InputSpec("maize_silage", 0, 20.0), InputSpec("slurry", 1, 10.0)],
)
```

See the [example plant builders](https://github.com/dgaida/PyADM1ODE_estimate/blob/main/pyadm1ode_estimation/example_plants/multi_stage.py)
as templates for more complex topologies (multi-stage cascades, multiple
CHPs, heating circuits).

## Before production: twin test

Good practice is to validate the filter against a known truth first.
The bundled twin script does that in a single run:

```bash
python examples/run_twin_experiment.py --warmup-days 30 --duration-days 5
```

This produces 6 diagnostic plots in `output/twin_experiment/`. Expected
results: 2σ coverage well above the 80 % target on strong-observable
blocks, mean NIS near its expected value (= number of measurement
channels). More on this:
[twin experiments](usage/twin_experiments.md).

## Where to go next

* [Usage → UKF in practice](usage/ukf.md) — detailed walkthrough of
  the individual building blocks + `MeasurementCalendar` for sporadic
  lab measurements.
* [Usage → twin experiments](usage/twin_experiments.md) — end-to-end
  validation against a known truth.
* [Observability → sensor-state dependencies](observability/sensor_state_dependencies.md) —
  which states are at all estimable with which sensors?
