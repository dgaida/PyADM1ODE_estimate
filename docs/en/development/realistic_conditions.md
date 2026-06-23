# Realistic test conditions — the "realism foundation"

To judge whether a state estimator is *good enough for the project goal* (state
estimation on a real agricultural biogas plant, ultimately to support operating
recommendations), the twin experiment must run under conditions close to a real
plant. This page defines those conditions and — crucially — **how they are
chosen**.

## Guiding principle: plant-agnostic and literature-based, not fitted

The sensor suite of the real test plant is **not final** (sensors are still
being added), and the goal is a method that works for **biogas plants in
general**, not one dataset. Therefore the noise and model-error settings are
**not fitted to the current plant data**. Two consequences:

* **Measurement noise** is attached to *sensor types* as generic  
  instrument-accuracy specs, so adding or removing a sensor does not change the
  others, and the conditions transfer across plants.  
* **Model error** — the gap between ADM1 and reality — **cannot be identified  
  from operating data** (it is confounded with input and measurement error). It
  is therefore set from the **kinetic-parameter uncertainty reported in the
  ADM1 literature**, applied as a multiplicative perturbation.

The values live in [`estimation/realism.py`](../api/index.md) and are applied
by the twin's `--realistic` preset.

## Model error: kinetic-parameter uncertainty

The model error is a **multiplicative lognormal** perturbation of the
filter's biological kinetics, while the truth keeps the "real" values:

$$ k_\text{filter} = k_\text{nominal}\cdot e^{\mathcal N(0,\sigma)},\qquad
   \sigma = 0.25 $$

applied to the rate/affinity prefixes `k_dis`, `k_hyd`, `k_m_`, `k_dec`,
`K_S`. Physical equilibrium constants (`K_a`, `K_w`, `K_H`) and stoichiometry
(`Y_*`, `f_*`) are left untouched.

Why this form and magnitude:

| Evidence | Value |
|---|---|
| ADM1 local/global sensitivity studies perturb kinetic/stoichiometric parameters | ~10 % |
| The gas-relevant *sensitive* parameters are exactly decay / disintegration / hydrolysis / `k_m` / `K_S` | — |
| UKF-for-AD studies inject a deliberate plant-model mismatch on the rate constants | ~28–30 % |
| Monte-Carlo uncertainty studies treat kinetic parameters as **lognormal**; hydrolysis is the most uncertain, yet biogas flow stays comparatively robust | lognormal, CoV up to ~50 % |

`σ = 0.25` (CoV ≈ 25 %) sits between the sensitivity range (10 %) and the
plant-model-mismatch end (30 %). It is the *transferable* stand-in for "the
model is not 100 % reality": the **observed** quantities (gas) stay trackable,
while **un-corrected** states drift — the realistic behaviour the estimator
must cope with.

## Measurement noise (1-σ, per sensor type)

Generic instrument-accuracy specs, **relative** unless noted absolute:

| Sensor | Noise (1-σ) | Basis |
|---|---|---|
| Biogas flow `q_gas` | 3 % rel. | biogas-volume metrology, ~3 % expanded uncertainty, drift < 0.15 %/24 h |
| Methane flow `q_ch4` | 4 % rel. | `q_gas` × NDIR CH₄ (≈ 0.7–1 % abs.) combined |
| CO₂ flow `q_co2` | 4 % rel. | `q_gas` × NDIR CO₂ |
| pH | 0.05 (absolute, pH units) | glass-electrode probe accuracy |
| VFA / FOS | 8 % rel. | FOS/TAC titration |
| TS / VS | 3 % rel. | gravimetric |
| Substrate dose | 3 % rel. | dosing scale |

Some UKF-for-AD studies inflate `R` by ~1.5× for robustness; that factor is
exposed as `R_INFLATION` (default 1.0).

## Sampling (real-plant cadence)

* **Online (hourly in the twin, seconds on the plant):** `q_gas`, pH, substrate  
  doses, levels, CHP, temperatures.  
* **Daily:** gas composition CH₄ / CO₂ (NDIR). O₂ and H₂S are also measured  
  daily in practice but are **not ADM1da states** → monitoring only, *not
  assimilated*.  
* **Lab cadence:** TS, VFA (gated, e.g. every 12 h for VFA).  

## Deliberately *not* included (and why)

* **Plant-specific gas-derivation chain.** On the real plant, `Q_gas` is not a  
  flow meter reading but is reconstructed from gas-dome level changes (ΔV,
  differentiated → noise-amplified) + CHP consumption + flare. That error
  structure is plant-specific; the foundation uses the generic flow-meter spec
  instead so the conditions stay transferable. (A plant-specific noise model can
  be layered on top when evaluating *that* plant.)  
* **O₂ / H₂S.** Not ADM1da states — cannot inform the UKF.  
* **Fitting σ or R to the current dataset** — would bias toward the incomplete  
  sensor suite.

## Usage

```bash
# Realism preset (model error + per-sensor noise + daily gas analytics):
python examples/run_twin_experiment.py --realistic --warmup-days 30 --duration-days 14
# Reduced state vector under realistic conditions:
python examples/run_twin_experiment.py --realistic --state-blocks methanogenesis charge_balance
```

Individual flags (`--model-error-std`, `--gas-noise-std`) still override the
preset for sensitivity sweeps.

## Evaluation harness & provenance

The estimator is evaluated by a **paired Monte-Carlo ensemble**
(`monte_carlo_eval.py`): each seed draws an independent model-error realisation
(this foundation) plus noise and prior perturbation, and **all candidates see
the same world** (open-loop model, raw-sensor floor, UKF full-41, UKF A+D
core, and **A+D core with known input** `adcore_ki` — the substrate feed is fed
forward from the measured dosing instead of estimated, so the filter is not
"surprised" by feed changes; this collapses the feed-change NIS from ~10³ to the
consistency band). Metric: decision-weighted per-block NRMSE (block **median**, **converged
second half** to discount the UKF's initial transient; `charge_balance` reported
separately because its near-zero ions are ill-conditioned), plus calibration
(mean NIS, 2σ coverage) and paired win-rates.

**Which estimator/version was tested** is recorded per run in
`output/mc_eval_meta.txt` — keep it with any archived result, since `output/`
is git-ignored. It captures the timestamp, the PyADM1ODE_estimate and pyadm1
git commits (with a `+dirty` flag if the working tree had uncommitted changes),
the pyadm1 version, the filter class + parameters, the model-error σ and the
sensor schedule. The estimator under test is the **Square-Root UKF**
`estimation.filters.sr_ukf.UnscentedKalmanFilter` (α = 1.0, β = 2.0, κ = 0.0,
γ = canonical √(n+λ)); the reduced **"A+D core"** is the same filter restricted
to `--state-blocks methanogenesis charge_balance` (18 of 41 ADM1 states).

## Sources

* ADM1 sensitivity analysis — [WIT Transactions (ADM1 local SA)](https://www.witpress.com/elibrary/wit-transactions-on-ecology-and-the-environment/258/38278);  
  [Surrogate-based global SA of ADM1 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0301479720313815).  
* UKF / plant-model mismatch for AD — [Comparison of UKF designs for AD (arXiv:2310.15958)](https://arxiv.org/html/2310.15958);  
  Haugen et al. 2014, *State Estimation … Pilot AD Reactor* ([DOI:10.1155/2014/572621](https://doi.org/10.1155/2014/572621)).  
* ADM1 Monte-Carlo / lognormal uncertainty — [Uncertainty analysis of a simplified AD model (IWA WST 92(4):610)](https://iwaponline.com/wst/article/92/4/610/108810/Uncertainty-analysis-of-a-simplified-anaerobic);  
  [Probabilistic ADM1 simulation of biogas (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1369703X23000050).  
* Sensor accuracy — NDIR biogas analysers ([Olythe](https://www.olythe.io/analyzers/biogas-analyzer/), [Dynament](https://dynament.com/application/biogas-monitoring/));  
  biogas-volume metrology ([PMC12693810](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12693810/)).
