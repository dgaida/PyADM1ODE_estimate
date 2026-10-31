# The GRU observer (PINN B)

[PINN v2](pinn.md) fits a fresh network to every series and retrains it every
day, paying a full optimisation each time. The **amortised observer** (PINN B)
moves that cost *offline*: it is pre-trained once on many simulated scenarios and
then reads out the state in a single forward pass. Where PINN v2 *optimises* per
window, the observer *learns a reusable mapping* from the measurement stream to
the state trajectory.

It shares the physics scaffolding of PINN v2 (positivity via a log-transform,
quasi-steady gas), only the network and the training differ.

---

## 1. Architecture: a GRU filter

The observer is a **recurrent** network (a GRU). Its input at each step is the
current sensor reading plus the known feed; it consumes the sequence and emits a
state per step:

$$
\underbrace{(y_t,\ u_t)}_{\text{measurements + feed}}\ \longrightarrow\
\text{GRU}\ \longrightarrow\ \hat{x}(t)\ \ (41\text{ states}).
$$

Two design choices it shares with [PINN v2](pinn.md):

* **Positivity / scaling** — the head predicts only the **37 liquid states** as a  
  log-deviation from a reference, $\hat{x}_\text{liq} = x_\text{ref}\odot
  \exp(\text{raw})$, and the 4 gas pressures are slaved by the quasi-steady
  equilibrium solve.  
* **Causality** — the GRU at step $t$ sees only past and present inputs, so the  
  newest step is a proper **online filtered** estimate: the same network runs as
  a streaming filter (section 3).

The key difference from PINN v2: the observer is *conditioned on the
measurements*. PINN v2's network is a function of time and feed only (the data
enters through the loss), whereas the observer takes the measurement stream as
its **input**. This is what lets a single trained net serve every series
instead of being refit each window. How far that carries beyond the training
distribution is the subject of section 5.

---

## 2. Training, phase 1

Pre-training can use either of two objectives (or both):

**Supervised** (`pretrain_observer`) — on a **simulator** dataset where the true
41-state is known. The loss is a per-state-scaled state MSE
$\lVert (\hat{x}-x_\text{true})/s \rVert^2$ (each state normalised by its RMS
magnitude), which teaches the *full* state.

!!! tip "Overfitting is the binding constraint, not capacity"
    With ~80 training series the network memorises them quickly: measured on the
    benchmark, the validation loss bottoms out around epoch 60 of 200 while the
    training loss keeps falling. `restore_best=True` (the default) plus
    `patience=N` returns the best-validated weights and halves the runtime at an
    identical result.

    What did **not** help, measured: enlarging the window set. Overlapping or
    randomly-placed windows lower the validation loss but make the test score
    *worse* (section 5). They add gradient steps, not information -- the 80
    series are the bottleneck, not how they are cut. Of the regularisers, only
    `dropout=0.1` between the GRU layers improved the test score; a smaller
    `hidden`, weight decay and input-noise augmentation did not.

The scale `s` is computed from the **training** sequences only; deriving it over
the whole set first would leak validation statistics into the objective. Options
worth knowing:

| Option | What it does |
| --- | --- |
| `val_dataset` | an **externally split** validation set. Pass it to share one split across estimators -- `PinnData.observer_dataset` emits train and val from the same stratified split the filters use, which is what makes a filter and a network comparable. Splitting internally instead gives each model its own random split. |
| `burnin` | leading steps excluded from the loss. A causal observer cannot know the initial state, so that error measures the unknowable rather than the model. |
| `noise_std` | per-channel sensor noise in raw units, resampled onto the measurement features every batch. Only the measurement block is perturbed; the feed is a known control input. |
| `restore_best` / `patience` | best-validated weights / early stopping. |
| `weight_decay` | L2 regularisation. |

**Self-supervised** (`pretrain_observer_selfsup`) — on **measurement-only**
windows with *no* ground truth (real plant history, or simulated windows for an
ablation). It uses the same objective as the online fine-tuning: a measurement
fit $\big((h(\hat{x})-y)/\sigma\big)^2$ plus a rate-scaled physics residual. This
lets the observer be primed directly on the real plant it will run on.

**Sim→real** (`pretrain_observer_sim2real`) — the recommended recipe: supervised
on the simulator first (learn the full-state structure), then self-supervised on
real history (close the sim-to-real gap). Share the input normalisation between
the two stages.

---

## 3. Training, phase 2

`finetune_observer` adapts the pre-trained observer to the live plant on the
recent window, warm-started from its weights with a small learning rate. It is
**self-supervised** (only measurements are known online): measurement fit +
discrete rate-scaled physics residual, with an optional anchor (`lambda_anchor`)
that keeps the trajectory near the frozen pre-trained prediction (a trust region
against noise). Like pre-training it is monotone-safe.

`SlidingWindowObserver` wraps this into a **continuous online estimator**:

```python
swo = SlidingWindowObserver(
    observer, obs_model, feat_mean, feat_std,
    window_hours=48, finetune_every=24,
)
for meas, feed in live_sensor_stream:   # meas = [Q_gas, Q_ch4, pH]
    est = swo.step(meas, feed)          # est.state = current 41-state, est.std
```

It keeps a moving window of recent readings. On every new sample it returns the
current estimate and, on a schedule, self-supervised fine-tunes on the window.
Two operational details:

* **Feed-aware physics** — off the nominal operating point the pre-trained  
  parameters' nominal feed is wrong, so the fine-tune rescales `q_ad` (the total
  influent flow into the digester [m³/d]) to the window's actual mean feed.  
* **Missing sensors** — a gated/offline reading arrives as `NaN`. The recurrent  
  net cannot take `NaN`, so it is mapped to the normalised mean (0) on the input
  side and masked out of the loss on the target side.

**Uncertainty** is optional MC-Dropout (`mc_samples > 1` with a dropout-trained net).

---

## 4. Strengths and limits

**Strengths.** Near-instant inference after the one-time offline cost (well under
a second per 60-day series, against minutes for the UKF and tens of minutes for
PINN v2); learns the *full* state structure from the simulator, so it is strong
even on unobserved states; one network serves every series. On the benchmark
test set it is the only learned estimator close to the UKF (20 series, 60 days,
median-over-states NRMSE after a 48-h burn-in):

| Estimator | Error |
| --- | --- |
| UKF, 41 states (reference) | 14.4 % |
| PINN B as published (no dropout) | 15.1 % |
| PINN B with `dropout=0.1` (new default, mean of 3 seeds) | 14.7 % |
| PINN v2, causal, 10-day window | 18.7 % |

PINN B beats the UKF in the `stable_high` mode (12.9 % against 19.1 %), the
chronically overloaded regime where the UKF's nominal kinetics are furthest off.

**Limits.** It is bounded by its training distribution: on shifted plants it
loses more than the UKF does (section 5). It carries a **sim-to-real gap** that
only self-supervised adaptation closes, and the ill-conditioned biogas map still
limits self-supervised cold starts.

For combining it with the UKF, see the [UKF ↔ PINN fusion](fusion.md).

---

## 5. Generalisation to shifted plants

Two test sets shift the plant, and neither is used in training: one replaces
maize silage by corn-cob mix at the same organic load, the other adds a 3-day
load rhythm on top. Both were generated for this study only and are not part of
the repository. Scores are the median NRMSE
over the states after a 48-h burn-in; PINN B rows are the mean of three seeds
(`experiments/observer_generalisation/`).

| Estimator | Benchmark | Substrate swap | Swap + rhythm |
| --- | --- | --- | --- |
| UKF, 41 states | 14.4 % | 17.1 % | 17.7 % |
| UKF, acid + digester core | 17.5 % | 20.1 % | 19.6 % |
| PINN v2, causal | 18.7 % | 23.0 % | 21.9 % |
| PINN B, published recipe | 15.2 % | 19.3 % | 20.2 % |
| PINN B, `dropout=0.1` (default) | 14.7 % | 18.9 % | 19.5 % |
| PINN B, 5-seed ensemble with dropout | **14.1 %** | 18.5 % | 19.2 % |

PINN B loses 4 to 5 points on a shifted plant, the UKF about 3. The 18.4 %
published for the swap set was a favourable seed; three seeds of the same recipe
average 19.3 %, with a spread of about ±0.5 points per seed.

**What does not close the gap.** Input-noise augmentation, weight decay, a
smaller network, overlapping, random or shorter training windows and their
combinations all leave the gap at 3.5 to 5 points. Dropout and the ensemble
lower all three scores together; the gap stays.

**Substrate-aware input.** `FeatureSpec(organic_load=True, use_feed=False)`
replaces the per-substrate volumes by influent loads per component group,
computed with the plant's own substrate characterisation. Inside the training
distribution it is the best observer (14.0 % with dropout, below the UKF), but
it generalises worse: 19.2 % and 19.4 % on the shifted sets.

**Shifted data in training.** Adding 12 of the 20 corn-cob series to the
training pool and scoring on the other 8 (`coverage.py`, three seeds each):

| Input | Training pool | Benchmark | Held-out corn-cob | Swap + rhythm |
| --- | --- | --- | --- | --- |
| raw feed | benchmark | 14.7 % | 17.8 % | 19.5 % |
| raw feed | + 12 corn-cob series | 15.1 % | 17.8 % | 19.6 % |
| organic load | benchmark | 14.0 % | 19.6 % | 19.4 % |
| organic load | + 12 corn-cob series | 15.5 % | 18.4 % | 20.0 % |

With the raw feed the extra series change nothing: column 0 means maize in 100
series and corn-cob mix in 12, so the network cannot tell the plants apart. With
the load input they help on the held-out corn-cob series, but less than the raw
input reaches without them, and they cost accuracy on the original plant.

**Where the error grows.** On the shifted sets PINN B's error rises most on the
fast particulate fractions, the propionate degraders and the anion pool — states
the substrate swap moves and the five sensors barely see. The UKF's error rises
mostly on acetate. The UKF is told the substrate mix through its model, while the
observer has to infer the composition from the sensor history and maps an
unfamiliar history back toward the states it was trained on.

**Consequence.** PINN B is a strong estimator inside its training distribution,
and the cheap gains there are dropout and a seed ensemble. Outside it the UKF is
more robust. For a new plant the observer has to be trained on simulations of
that plant; where the kinetics differ, the UKF with a calibrated model is the
more robust route.

---

## 6. Kinetics as input

PINN B never sees which kinetics a series has, so calibrated kinetics cannot
help it. `FeatureSpec(kinetic_keys=...)` changes that: it appends one input
column per named kinetic constant, holding the log of its factor relative to the
nominal value, constant over the whole series.

* **Training** uses the generator's true factors of each simulated series.  
  `ObserverEstimator.train(..., kinetic_noise=0.07)` perturbs them by one draw
  per training sequence, so the network learns to use kinetics that are known
  only to calibration accuracy.  
* **Inference** takes the kinetics as given:  
  `est.run(series, kinetics={"k_m_ac": 0.8, ...})`. Keys that are not given count
  as nominal, and `{}` means nominal kinetics. A kinetics-aware observer refuses
  to run without explicit kinetics, so the generator truth a benchmark series
  carries can never slip in.

The inputs are the ten constants that move the sensors most over a 20-day
window, screened on eight training series
(`experiments/final_test/screen_kinetics.py`): `k_dis_PS`, `k_m_ac`, `k_m_pro`, `k_dis_PF`, `k_hyd_ch`, `k_dec_ac`, `k_dec_su`, `K_S_ac`, `k_dec_pro`, `K_S_pro`.

Comparison on the 20 validation series, both as 5-seed ensembles with the same
recipe and seeds, median NRMSE after 48 h (the test sets stay untouched for the
final test; `experiments/observer_generalisation/kinetic_inputs.py`):

| Kinetics given | Kinetics-blind | 10 kinetic inputs |
| --- | --- | --- |
| none, nominal values | 14.7 % | 17.2 % |
| true factors | 14.7 % | 12.4 % |
| true × exp(N(0, 0.07)) | 14.7 % | 12.9 % |
| true × exp(N(0, 0.15)) | 14.7 % | 14.0 % |

For orientation: the UKF with nominal kinetics reaches 16.3 % on the same series.
Five inputs instead of ten give 13.2 % at σ = 0.07. With all 26 constants known
exactly, which no calibration achieves, the observer reaches 7.7 %.

* **Most of the observer's error is unknown kinetics.** Knowing all of them  
  halves it.  
* **With calibrated kinetics it is clearly better.** At the accuracy the kinetics  
  calibration study reaches, σ ≈ 0.07, the kinetics-aware observer is 1.9 points
  better (p = 0.005). At σ = 0.15 the gain shrinks to 0.8 points and is no longer
  significant.  
* **Without calibration it is worse.** Told "nominal", it trusts that input  
  instead of inferring the kinetics from the sensor history as the kinetics-blind
  observer does, and loses 2.5 points.

The decision rule was fixed before the run: better with calibration at σ = 0.07,
and not more than 0.3 points worse without. The second part fails, so the
kinetics-aware observer does not replace the kinetics-blind one. Use it, with ten
inputs and trained with `kinetic_noise=0.07`, whenever calibrated kinetics are
available; use the kinetics-blind observer otherwise. The checkpoints are not
shipped; `kinetic_inputs.py` retrains them.

---

## Source files

* `pyadm1ode_estimation/estimation/observer/model.py` — `Adm1Observer`  
* `pyadm1ode_estimation/estimation/observer/datasets.py` — `generate_observer_dataset`, `ObserverDataset`, `MeasurementDataset`  
* `pyadm1ode_estimation/estimation/observer/train.py` — `pretrain_observer`, `pretrain_observer_selfsup`, `pretrain_observer_sim2real`, `finetune_observer`  
* `pyadm1ode_estimation/estimation/observer/online.py` — `SlidingWindowObserver`  
* `pyadm1ode_estimation/estimation/observer/adapter.py` — `PinnData`, `FeatureSpec` (incl. `organic_load`, `kinetic_keys`), `kinetic_log_factors`  
* `pyadm1ode_estimation/estimation/estimators.py` — `ObserverEstimator` (train, save, load, `for_dataset`)  
* `experiments/observer_generalisation/` — recipe sweep, seed ensemble, coverage test, kinetic inputs  

## API reference

::: pyadm1ode_estimation.estimation.observer.model.Adm1Observer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.observer.online.SlidingWindowObserver
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
