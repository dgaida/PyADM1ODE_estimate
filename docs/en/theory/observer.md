# The pre-trained observer

The [per-window PINN](pinn.md) fits a fresh network to every window, but it pays a
full optimisation (seconds to minutes) each time. The **amortised observer** moves
that cost *offline*: it is pre-trained once on many simulated scenarios and then
reads out the state in a single forward pass. Where the smoother *optimises* per
window, the observer *learns a reusable mapping* from the measurement stream to
the state trajectory.

It shares the physics scaffolding of the smoother (positivity via a log-transform,
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

Two things carry over unchanged from the [smoother](pinn.md):

* **Positivity / scaling** — the head predicts only the **37 liquid states** as a  
  log-deviation from a reference, $\hat{x}_\text{liq} = x_\text{ref}\odot
  \exp(\text{raw})$, and the 4 gas pressures are slaved by the quasi-steady
  equilibrium solve.  
* **Causality** — the GRU at step $t$ sees only past and present inputs, so the  
  newest step is a proper **online filtered** estimate: the same network runs as
  a streaming filter (section 3).

The key difference from the smoother: the observer is *conditioned on the
measurements*. The smoother's network is a function of time only (the data enters
through the loss), whereas the observer takes the measurement stream as its
**input**. This is exactly what lets a single trained net generalise across
operating points instead of being refit each window.

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
    randomly-placed windows (320 -> 1120 -> 1600) lower the validation loss
    slightly but make the full-series score *worse* (15.6 % -> 16.7 %). They add
    gradient steps, not information -- the 80 series are the bottleneck, not how
    they are cut. Prefer capacity reduction and regularisation (`weight_decay`,
    `dropout`, a smaller `hidden`).

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

**Uncertainty** is optional MC-Dropout, exactly as for the smoother.

---

## 4. Strengths and limits vs. the smoother

**Strengths.** Near-instant inference after the one-time offline cost; learns the
*full* state structure from the simulator (so it is strong even on unobserved
states); generalises across operating points without a per-window refit.

Measured on the benchmark's validation split, a plain 200-epoch run with no
tuning already **beats the do-nothing baseline in all four operating modes**
(15.6 % vs. 32.6 % median NRMSE), where the per-window
[smoother](pinn.md) manages two of four after four rounds of fixes. It also costs
~1.7 s per epoch against ~2.5 min for a *single* smoother window -- roughly 70x
cheaper per step, which is what puts a real hyperparameter search within reach.

**Limits.** Needs a **representative** pre-training distribution; carries a
**sim-to-real gap** that only self-supervised adaptation closes; and the
ill-conditioned biogas map still limits self-supervised cold starts.

For where both PINNs sit relative to the UKF, and how to combine them, see the
[UKF ↔ PINN fusion](fusion.md).

---

## Source files

* `pyadm1ode_estimation/estimation/deep_learning/observer.py` — `Adm1Observer`  
* `pyadm1ode_estimation/estimation/deep_learning/observer_data.py` — `generate_observer_dataset`, `ObserverDataset`, `MeasurementDataset`  
* `pyadm1ode_estimation/estimation/deep_learning/observer_train.py` — `pretrain_observer`, `pretrain_observer_selfsup`, `pretrain_observer_sim2real`, `finetune_observer`  
* `pyadm1ode_estimation/estimation/deep_learning/online_observer.py` — `SlidingWindowObserver`  

## API reference

::: pyadm1ode_estimation.estimation.deep_learning.observer.Adm1Observer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.deep_learning.online_observer.SlidingWindowObserver
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
