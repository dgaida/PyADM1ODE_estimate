# UKF in practice

Hands-on setup for the Square-Root UKF on a biogas plant. This page
covers the **current** API (refactor 2026); the building blocks are:

* :func:`adm1da_full_spec` — factory for the full 41-state ADM1da vector  
* :class:`UnscentedKalmanFilter` — Square-Root UKF (Wan & Van der Merwe 2001)  
* :class:`MeasurementCalendar` — per-sensor sample-rate management  
* :class:`SensorAdapter` — bridge to the PyADM1ODE sensor classes  

## Declaring the state vector

Instead of hand-selecting channels, the :func:`adm1da_full_spec`
factory builds the **complete 41-state vector** with defaults derived
from the observability analysis:

```python
from pyadm1ode_estimation.estimation import (
    adm1da_full_spec, InputSpec, SensorQualityProfile, Quality,
)

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
# spec has 44 channels: 41 ADM1 slots + 3 substrate inputs.
```

Per channel, the factory sets `process_noise_std`, `initial_std`, a
`drift_model`, and an OU mean (when `drift_model="ou"`) based on the
channel's observability class:

| Block | Default quality | Drift model |
|---|---|---|
| methanogenesis (A) | STRONG | random_walk |
| charge_balance (D) | STRONG | random_walk |
| acidogenesis_substrates (B subset) | MEDIUM | random_walk |
| acidogenesis_biomass | WEAK | random_walk |
| hydrolysis_sums (C) | WEAK | random_walk |
| disintegration_split (PS/PF) | PSPF | OU |
| nitrogen (E) | OPEN_LOOP | OU |
| inerts | OPEN_LOOP | OU |
| fa_block | OPEN_LOOP | OU |

**Per-plant overrides** via `SensorQualityProfile` — e.g. if your plant
runs an online GC-FID for individual VFAs:

```python
spec = adm1da_full_spec(
    digester_id="primary",
    substrate_inputs=[...],
    sensor_quality=SensorQualityProfile(
        acidogenesis_biomass=Quality.MEDIUM,  # GC-FID lifts sensor quality
    ),
)
```

## Plant model

The plant itself is built via the PyADM1ODE package. Two reference
plants are bundled for tutorials and tests:

```python
from pyadm1ode_estimation.example_plants import (
    build_simple_plant,         # 1 fermenter + 1 storage + 1 CHP
    build_multi_stage_plant,    # 3-fermenter cascade + 2 CHPs
)

plant = build_multi_stage_plant()
```

For a real plant you build the `BiogasPlant` via PyADM1ODE's
configurator API (`PlantConfigurator.add_digester`, `.add_chp`,
`.connect`, …). The builders in `example_plants/` are the template
for that.

## Process model

```python
from pyadm1ode_estimation.estimation import ADM1ProcessModel

process = ADM1ProcessModel(plant, spec)
process.snapshot()   # baseline for restore / sigma points
```

`ADM1ProcessModel.step(x, dt)` is the actual propagator. Snapshot/
restore guarantees that every sigma point starts from an identical
plant state. With the full spec all 41 ADM1 states are re-applied every
step, but the `BiogasPlant` carries further coupled state that is *not*
in the spec. Above all `simulation_time` (feeding is time-dependent),
the gas-storage fill level, and CHP / heating-circuit state. Without
restore these would drift between sigma points and the sample covariance
would be unusable. (For a partial spec this also applies to the ADM1
states you don't estimate.)

## Observation model

```python
from pyadm1ode_estimation.estimation import (
    ObservationChannel, ObservationModel,
)
from pyadm1ode_estimation.estimation.observation_model import (
    make_q_gas_extractor, make_q_ch4_extractor, make_state_extractor,
)

def make_ph_extractor(digester_id: str):
    def extractor(plant, x):
        val = plant.components[digester_id].outputs_data.get("pH", float("nan"))
        return 7.0 if not (val == val) else float(val)  # NaN-safe
    return extractor

# Look up substrate-sensor indices in the state vector.
def _idx(name):
    return next(i for i, c in enumerate(spec.channels) if c.name == name)

obs = ObservationModel(channels=[
    ObservationChannel("Q_gas",            make_q_gas_extractor("primary"), noise_std=10.0),
    ObservationChannel("Q_ch4",            make_q_ch4_extractor("primary"), noise_std=5.0),
    ObservationChannel("pH",               make_ph_extractor("primary"), noise_std=0.05),
    ObservationChannel("Q_maize_silage",   make_state_extractor(_idx("maize_silage")),   noise_std=0.24),
    ObservationChannel("Q_solid_manure",   make_state_extractor(_idx("solid_manure")),   noise_std=0.69),
    ObservationChannel("Q_chicken_litter", make_state_extractor(_idx("chicken_litter")), noise_std=0.06),
    ObservationChannel("Q_slurry",         make_state_extractor(_idx("slurry")),         noise_std=0.18),
    ObservationChannel("Q_cereal_grain",   make_state_extractor(_idx("cereal_grain")),   noise_std=0.01),
])
```

!!! note "Stage-scoped vs. whole-plant gas"
    `make_q_gas_extractor("primary")` / `make_q_ch4_extractor("primary")` read
    only the **estimated** digester's own production. In a multi-stage cascade
    the downstream stages (post-fermenter, digestate storage) also produce gas
    that the filter does not estimate — feeding the plant total into the
    innovation would bias the estimated stage. Use the stage-scoped extractors
    whenever a per-stage gas meter exists. Only when the physical meter sits
    downstream of every stage (e.g. a single flow meter in front of the CHP)
    use the whole-plant sums `extract_q_gas_total` / `extract_q_ch4_total`
    (catalog names `q_gas_total` / `q_ch4_total` in `build_ukf`).

## Setting up and running the filter

```python
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(
    process, obs, spec,
    alpha=1.0,    # unscaled UKF — robust for strongly nonlinear ADM1 kinetics
    beta=2.0,
    kappa=0.0,
)

# Initial estimate from a calibration artifact or the pre-inoculated state:
x0 = spec.read_adm1_state(plant)
# augmented substrate channels from the operator's setpoint:
for i, ch in enumerate(spec.channels):
    if ch.kind == "input_flow":
        x0[i] = ch.initial

# P0 must match the actual initial uncertainty.
# At 5 % relative uncertainty per channel:
import numpy as np
sigma_init = 0.05
P0 = np.diag((sigma_init * (np.abs(x0) + 1e-6)) ** 2)
ukf.reset(x0, P0)

# Live operation:
dt = 1.0 / 24.0  # 1-hour step in days
for t, y_dict, gate_dict in measurement_stream:
    ukf.predict(dt=dt)
    step = ukf.update(y_dict, t=t, gate_values=gate_dict)

    # step.x_hat       — posterior mean
    # step.P           — posterior covariance
    # step.y_pred      — h(x̂) per active channel (UKF internal ŷ, Jensen-biased)
    # step.y_std       — sqrt(diag(S)) per channel
    # step.innovation  — y_obs − y_pred
    # step.nis         — Normalised Innovation Squared
    # step.active_channels — which channels contributed an update
```

## Square-Root UKF

The `UnscentedKalmanFilter` propagates the Cholesky factor `S` (with
`P = S·Sᵀ`) rather than the full covariance `P`.

Effects:

* `κ(S) = √κ(P)` → half the condition number → at n=44 structurally  
  more robust than the earlier Cholesky-UKF.  
* **Positive-definiteness by construction**: `P = SSᵀ` can never go  
  indefinite.  
* On a `_cholupdate` downdate failure, you know structurally that your  
  `Q`/`R` tuning or your measurement model has a real problem — not
  a numerical glitch (see [troubleshooting](../troubleshooting.md)).

## Sample-rate management with MeasurementCalendar

In practice, measurements arrive at very different rates:

| Sensor | Rate |
|---|---|
| Q_gas (gas flow) | every 5 min |
| pH online | every 5 min |
| FOS/TAC (titration) | daily |
| NH4-N (lab) | sporadic |

`MeasurementCalendar` builds the per-step `gate_values` dict from a
measurement DataFrame:

```python
from pyadm1ode_estimation.estimation import MeasurementCalendar, SampleRate

calendar = MeasurementCalendar({
    "Q_gas":   SampleRate.online(period_min=5),
    "pH":      SampleRate.online(period_min=5),
    "FOS_TAC": SampleRate.daily(),
    "NH4_N":   SampleRate.sporadic(),
})

# In the filter loop:
for t in time_grid:
    y, gates = calendar.values_for_filter(t=t, df=measurements_df)
    ukf.predict(dt=dt)
    ukf.update(y=y, t=t, gate_values=gates)
```

A sporadic lab sample lands as an active channel in exactly one step —
the filter needs *no* restructuring when, three months later, an NH4-N
sample arrives. The channel slot stays in the state vector; without a
measurement it just drifts as OU.

## NIS monitoring

`step.nis` is the central consistency indicator:

| Range | Diagnosis |
|---|---|
| `NIS ≈ n_active_channels` | filter well calibrated |
| `NIS ≫ n_active_channels` | filter underestimates uncertainty (Q or R too small) |
| `NIS ≪ n_active_channels` | filter overestimates uncertainty (Q or R too large) |

Rule of thumb for a well-tuned filter: `NIS ∈ [0.5·n, 2.0·n]`.

At 6 measurement channels that means `NIS ∈ [3, 12]`. If the **mean
NIS over multiple days** falls outside this window:

* `NIS > 3·n` for hours → filter is likely diverging (e.g. unmodelled  
  inhibitor, undosed substrate spike).  
* `NIS < 0.3·n` → sensor `R` is set noticeably too large or the  
  process noise `Q` is over-dimensioned.

See also [twin experiments](twin_experiments.md) for an end-to-end
example with plot diagnostics.
