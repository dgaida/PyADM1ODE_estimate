# Twin experiments

End-to-end validation of the UKF against a **known truth**: simulate
reality (truth plant) and feed the filter only noisy measurements
drawn from it. The comparison between filter estimate and truth tells
you whether the filter is well calibrated *before* you let it loose on
real plant data.

## Principle

```text
                ┌─────────────────────────────────────────┐
                │   TRUTH SIDE (= "the real plant")       │
                │                                         │
   build →      │ truth_plant (ADM1 ODE)                  │
   warm-up →    │   │                                     │
                │   ↓ propagate_truth: ODE step dt=1h     │
                │   x_truth[k+1]                          │
                │   │                                     │
                │   ↓ h(x): clean truth observation       │
                │   obs_clean[k+1]                        │
                │   │                                     │
                │   ↓ truth_sensors[name].read(t)         │
                │     ↓ drift + lag + noise + sampling    │
                │   obs_noisy[k+1]  ──────────────────────│─┐
                └─────────────────────────────────────────┘ │
                                                            │
                ┌─────────────────────────────────────────┐ │
                │   FILTER SIDE (UKF)                     │ │
                │                                         │ │
   deepcopy →   │ filter_plant (= truth_plant at t=0)     │ │
                │   │                                     │ │
                │   ↓ ukf.predict(dt=1h):                 │ │
                │     - 89 sigma points                   │ │
                │     - each: filter_plant.step()         │ │
                │     - weighted mean                     │ │
                │   x̂[k+1] (predicted), P_pred            │ │
                │   │                                     │ │
                │   ↓ ukf.update(y=obs_noisy[k+1])  ←─────│─┘
                │     - 89 sigma points through h(x)      │
                │     - innovation y − ŷ                  │
                │     - Kalman gain → posterior           │
                │   x̂[k+1] (posterior), P_post            │
                └─────────────────────────────────────────┘
```

In **reality there is no truth_plant** — the sensors provide the
measurements directly. The filter_plant in the twin corresponds to the
*internal model* the estimator uses.

## The bundled script

The repo ships `examples/run_twin_experiment.py`, which runs the
complete workflow on the multi-stage reference plant:

```bash
python examples/run_twin_experiment.py \
    --warmup-days 30 \
    --duration-days 5 \
    --initial-perturbation-relative 0.05 \
    --substrate-noise-relative 0.10
```

CLI parameters:

| Parameter                           | Default | Meaning                                                                                               |
| ----------------------------------- | ------- | ----------------------------------------------------------------------------------------------------- |
| `--warmup-days`                   | 30      | Pre-simulation before the filter starts, so the plant has reached a quasi-steady state                |
| `--duration-days`                 | 5       | Length of the actual UKF run                                                                          |
| `--dt-hours`                      | 1.0     | Filter step size                                                                                      |
| `--initial-perturbation-relative` | 0.05    | Relative Gaussian perturbation of the filter's initial state                                          |
| `--substrate-noise-relative`      | 0.10    | Per-step noise on the substrate dosing (models kg-to-m³ conversion + seepage-water level correction) |
| `--plot-from-day`                 | 0.0     | Burn-in for plots (diagnostics are still computed over the full run)                                  |

## What happens inside

1. **Build truth plant** (`build_multi_stage_plant`) and warm up for  
   30 days (`plant.simulate()`). The ODE reaches a quasi-stationary
   operating point.  
2. **Deepcopy** the warmed plant for the filter — guarantees a  
   bit-identical model between truth and filter at `t=0`.  
3. **Propagate truth** (`_propagate_truth_with_substrate_noise`) with  
   per-step noise on the substrate inputs (operator delivery is never
   exact).  
4. **Truth sensors** (`build_truth_sensors`) generate the noisy  
   measurement signal using PyADM1ODE's `PhysicalSensor` classes for
   realistic drift, response lag and sampling (see
   `pyadm1ode_estimation.estimation.sensors`).  
5. **UKF** initialised with `ukf.reset(x_truth0 + perturbation, P0)`.  
6. **Filter loop** over all measurement timestamps.  
7. **Write plots** to `output/twin_experiment/`.  

## Generated plots

Each run produces 6 plots in `output/twin_experiment/`:

* **`trajectories_strong.png`** — 6 strong-observable states  
  (S_ac, S_ch4, X_ac, S_hco3, p_gas_ch4, pTOTAL) with truth, `x̂`
  and the ±2σ band.  
* **`trajectories_weak.png`** — 6 weak / open-loop states + 1  
  substrate input.  
* **`observations.png`** — all 6 measurement channels with clean  
  truth, noisy measurement and filter prediction ŷ.  
* **`production_estimate.png`** — the production-facing plot:  
  * Truth Q_gas / Q_ch4 (black)  
  * Raw sensor (red ×) — what the measurement points deliver  
  * Sensor-smoothed (red, solid) — rolling mean  
  * h(x̂) (green) — deterministic model re-evaluation at the  
    filter posterior (Jensen-bias-free, a single plant-step
    evaluation rather than 89 sigma points)  
  * `±1σ` band from the UKF-internal `y_std`  
  * Cumulative production with end-error annotated  
* **`nis.png`** — NIS time series on log scale with the expected  
  value as reference line.  
* **`coverage_summary.png`** — per-quality-block 2σ coverage as bar  
  chart with the target lines (80 % / 40 % / 20 %).

## What the results mean

From a typical 30+5 day run (10 % substrate noise, 5 % initial
perturbation):

| Block                   | Coverage | Status               |
| ----------------------- | -------- | -------------------- |
| methanogenesis          | 86.7 %   | strong ✓            |
| charge_balance          | 94.3 %   | strong ✓            |
| acidogenesis_substrates | 99.3 %   | medium ✓            |
| acidogenesis_biomass    | 99.8 %   | weak ✓              |
| hydrolysis_sums         | 100 %    | weak ✓              |
| disintegration_split    | 85.7 %   | structurally limited |
| nitrogen                | 99.2 %   | open-loop ✓         |
| inerts                  | 100 %    | open-loop ✓         |
| fa_block                | 100 %    | open-loop ✓         |

`disintegration_split` structurally stays below 100 % because the
PS/PF splits are fundamentally not separable from process measurements
(see [observability docs](../observability/sensor_state_dependencies.md)).

**Mean NIS ≈ 8.9** at 6 channels sits in the ideal window `[3, 12]`.
The filter is well calibrated.

## Production plot: sensor vs. model

The production plot reveals an important practical finding:

| Source                       | Cumulative end-error |
| ---------------------------- | -------------------- |
| Sensor (smoothed)            | ≈ −0.5 %           |
| h(x̂) (model re-evaluation) | ≈ −3.5 %           |

**The sensor beats the model-based estimate for directly measured
quantities.** Q_gas has 0.16 % relative noise — no model-based
estimator can do better, because the model aggregates over 44 states
and accumulates small bias contributions.

**Practical hierarchy for operator reporting:**

| Quantity                                                        | Best source                        |
| --------------------------------------------------------------- | ---------------------------------- |
| Q_gas / Q_ch4 (directly measured)                               | sensor + smoothing                 |
| pH (directly measured)                                          | sensor + smoothing                 |
| Substrate dosing (directly measured)                            | sensor + smoothing or operator log |
| **S_ac, X_ac, biomass, acid-base species** (not measured) | **UKF x̂**                  |

The UKF does not give you "a better Q_gas measurement"; it gives you
**the unmeasured 35-40 states from the measured 5-6**. That is the
actual value.

## Acceptance criteria

Rule-of-thumb thresholds for a production UKF:

* **Strong-observable blocks** (methanogenesis, charge_balance,  
  acidogenesis_substrates): 2σ coverage ≥ 80 %  
* **Weak / OU blocks**: 2σ coverage ≥ 40 %  
* **Open-loop**: 2σ coverage ≥ 20 %  
* **Mean NIS** over multiple days in `[0.5·n, 2.0·n]`  

For an acceptance run all of them should hold. If e.g. the mean NIS
falls outside the band, check:

* Are the `noise_std` values realistic for the actual sensors?  
* Is the warm-up long enough for the plant to reach a quasi-steady  
  state?  
* Is the initial perturbation `--initial-perturbation-relative`  
  realistic relative to the initial covariance `P0`?  
* Is `dt_hours` small enough for ADM1's nonlinearity?  
