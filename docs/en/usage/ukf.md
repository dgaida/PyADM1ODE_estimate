# UKF in practice

Concrete setup of the production UKF on a biogas plant.

## Declare the state vector

The `StateVectorSpec` defines *what* the filter estimates. Three channel kinds:

| `kind` | Meaning | Mandatory field |
|---|---|---|
| `"adm1"` | Direct slot in the 41-state ADM1da vector | `adm1_index` |
| `"input_flow"` | Augmented substrate feed (m³/d) | `input_substrate_index` |
| `"kinetic_param"` | Augmented kinetic parameter | (name must exist in `adm1._kinetic`) |

Example configuration for an agricultural biogas plant
(6 biological + 2 inputs = 8 channels):

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

!!! warning "Realistic channel count"
    Eight channels with only standard SCADA instrumentation (Q_gas, P_el,
    substrate flow) is *over-specified*. See the
    [observability literature review](../observability/literature_review.md)
    — rule of thumb: 1 independent sensor ⇒ approximately 1 separable
    state dimension.

## Process model

```python
from pyadm1ode_estimation.estimation import ADM1ProcessModel

plant = build_plant(schema)               # plant-specific plant builder
process = ADM1ProcessModel(plant, spec)
```

`ADM1ProcessModel.step(x, dt)` is the actual propagator. Snapshot/restore
ensures that every sigma point starts from an identical plant state.

## Observation model

`ObservationModel` aggregates `ObservationChannel`s. Built-in extractors
cover Q_gas, Q_ch4, Q_gas_consumed, P_el and P_th_used.

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
        gate_column="hopper_observable",     # active only when ΔW < 0
    ),
])
```

## Set up and run the filter

```python
from pyadm1ode_estimation.estimation.filters import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(
    process, obs, spec,
    alpha=1.0,    # unscaled UKF — robust for strongly nonlinear ADM1 kinetics
    beta=2.0,
    kappa=0.0,
)

dt = 1.0 / 24.0  # one-hour step in days

for t, y_dict, gate_dict in measurement_stream:
    ukf.predict(dt=dt)
    step = ukf.update(y_dict, t=t, gate_values=gate_dict)

    # step.x_hat       — posterior mean
    # step.P           — posterior covariance
    # step.innovation  — y_obs - y_pred per active channel
    # step.nis         — Normalised Innovation Squared (consistency indicator)
    # step.active_channels — which channels contributed an update
```

## NIS monitoring

The central consistency indicator is `step.nis`:

* **NIS ≈ number of active channels** → filter well calibrated.
* **NIS ≫ n_obs** → filter underestimates uncertainty (`Q` or `R` too small).
* **NIS ≪ n_obs** → filter overestimates uncertainty; conservative but
  little informative.

Plausible alarm thresholds: NIS > 3 × n_obs over several hours = filter is
likely diverging (e.g. an unmodelled inhibitor is active).
