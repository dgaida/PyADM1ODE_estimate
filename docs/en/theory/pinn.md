# Physics-Informed Neural Networks (PINN) for ADM1

This page explains the PINN estimator from the ground up. It starts with the
idea, then builds up the exact mathematics, explains the ADM1-specific tricks
that make it work at all, and finally how it is implemented in this repository
(`pinn.py`, `pinn_smoother.py`).

!!! abstract "In one sentence"
    A PINN is a neural network that reconstructs the digester's hidden state
    trajectory $\hat{x}(t)$ by simultaneously **fitting the few available sensor
    measurements** and **obeying the ADM1 differential equations**.

Sections 1–3 are the intuition (no maths), section 4 is the formal loss
function, section 5 explains why the stiff ADM1 needs special care, and
section 6 maps every idea onto the actual code.

---

## 1. The problem this solves

A biogas digester is described by **41 internal ADM1 states**: dissolved sugars
and acids, particulate fractions, microbial biomass, ion/charge species and
gas-phase pressures (see the [ADM1da model](adm1.md)). Online we can only measure
a handful of them indirectly: biogas flow $Q_\text{gas}$, methane $Q_\text{ch4}$,
pH, and a few more. **State estimation** is the task of reconstructing all 41
states over time from those few noisy signals.

There are two classic routes, each with a weakness:

* **Pure mechanistic model (ADM1).** Trustworthy structure, but it needs the  
  exact initial state and exact kinetic parameters. Small errors accumulate and
  the simulation drifts away from the real plant.  
* **Pure machine learning.** Flexible, but it needs a large labelled dataset of  
  *true* states to learn from, which on a real plant we never have, because the
  true states are exactly what we cannot measure.

A PINN combines the two: a neural network flexible enough to fit the sparse data,
**constrained by the ADM1 physics** so that it needs far less data and cannot
wander into physically impossible trajectories.

---

## 2. A neural network

* A neural network is a flexible mathematical function $y = \text{NN}_\theta(x)$  
  with many tunable numbers $\theta$ (its *weights*).  
* **Training** means adjusting $\theta$ so that a *loss* is minimised by gradient  
  descent.  
* The **input is time $t$** and the **output is the state vector $\hat{x}(t)$**.  
  So the network *is* the estimated trajectory itself: ask it at any time $t$ and
  it returns the estimated plant state at that instant.

Concretely it is a multilayer perceptron (MLP): a few `Linear` layers with a
smooth `tanh` nonlinearity in between. `tanh` is preferred in PINNs because its
derivatives are smooth.

---

## 3. What "physics-informed" means

A network trained on data alone would fit the measurement points but be free to
do anything between and after them. A PINN adds a second requirement: the
trajectory must satisfy the known governing equation

$$
\frac{dx}{dt} = f(x, u),
$$

where $f$ is the ADM1 right-hand side and $u$ is the (known) substrate feed.

The trick that makes this checkable is **automatic differentiation**: because the
network output $\hat{x}(t)$ is an explicit function of the input $t$, we can ask
the deep-learning framework for its exact time derivative $\tfrac{d\hat{x}}{dt}$.
ADM1 gives us $f(\hat{x}, u)$. Their mismatch,

$$
r(t) \;=\; \frac{d\hat{x}}{dt} - f(\hat{x}, u),
$$

is the **physics residual** &rarr; zero when the trajectory obeys the model.

Training then minimises a sum of three demands:

$$
\boxed{\text{network prediction } \hat{x}(t)}\ \Longrightarrow\
\left\{
\begin{array}{l}
\text{compare with the sensors} \ \longrightarrow\ \boxed{\text{data loss}} \\[8pt]
\text{compare with the ADM1 ODE} \ \longrightarrow\ \boxed{\text{physics loss}} \\[8pt]
\text{compare with the prior at the start} \ \longrightarrow\ \boxed{\text{prior loss}}
\end{array}
\right\}
\ \Longrightarrow\ \boxed{\text{total loss } L}
$$

Gradient descent on $\theta$ minimises $L$ and feeds the updated weights back
into the network.

The payoff of the physics term: it **interpolates through gaps** in the data,
**regularises** against sensor noise, and — because the ODE is defined at every
instant — lets the model **forecast** past the last measurement.

---

## 4. The mathematics

Let the network be $\hat{x}_\theta(t)$. The estimator minimises

$$
L(\theta) \;=\; L_\text{data} \;+\; \lambda_\text{phys}\, L_\text{phys}
             \;+\; \lambda_\text{prior}\, L_\text{prior}.
$$

The weights $\lambda_\text{phys}, \lambda_\text{prior}$ balance the three demands
and are the main tuning knobs.

**Data loss:** fit the measurements. With the differentiable measurement map $h$
(state $\to$ sensor channels) and per-channel noise std $\sigma$, over the
observed cells $\mathcal{O}$ (missing/gated readings are masked out):

$$
L_\text{data} \;=\; \frac{1}{|\mathcal{O}|}
  \sum_{i}\left\lVert \frac{h(\hat{x}_\theta(t_i)) - y_i}{\sigma} \right\rVert^2 .
$$

**Physics loss:** obey the ODE. It measures how strongly the predicted
trajectory violates the ODE. This is checked at $N_c$ *collocation points*
$\{\tau_j\}$, self-chosen times placed across the window. ADM1 mixes very fast
processes (acid–base reactions) with very slow ones (biomass). Without a fix the
fast ones would dominate the loss, so the residual is divided per state by a
typical scale $s$ so that all count equally (details in section 5):

$$
L_\text{phys} \;=\; \frac{1}{N_c}\sum_{j}
  \left\lVert \frac{\tfrac{d\hat{x}_\theta}{dt}(\tau_j)  
    - f(\hat{x}_\theta(\tau_j), u)}{s} \right\rVert^2 .  
$$

**Prior loss:** anchor the start. A weak boundary condition tying the trajectory
to a physical prior state $x_\text{prior}$ at $t_0$:

$$
L_\text{prior} \;=\;
  \left\lVert \frac{\hat{x}_\theta(t_0) - x_\text{prior}}{s} \right\rVert^2 .
$$

!!! tip "Why collocation points matter"
    Because $L_\text{data}$ lives only at the measurement times but
    $L_\text{phys}$ can be evaluated at any collocation point, the two time
    ranges are decoupled. Put collocation points past the last measurement and
    the ODE alone carries the state forward: that is how the same fit
    **forecasts**.

---

## 5. Why ADM1 is hard

A textbook PINN fails on ADM1: the system is **stiff** (rates spanning many
orders of magnitude) and its states span from $\sim 10^{-7}$ to tens, all of
which must stay non-negative. Four engineering choices in `PinnSmoother` make it
converge (it settles stably on a good solution instead of derailing).

### 1. Positive, well-scaled outputs

So far we said the network outputs $\hat{x}(t)$. More precisely, its last layer
first produces a **raw**, unbounded vector $\text{raw}_\theta(t)$. The physical
state is then formed from it as a **log-deviation from a prior**:

$$
\hat{x}(t) \;=\; x_\text{prior}\,\odot\, \exp\!\big(\text{raw}_\theta(t)\big).
$$

The final layer is **zero-initialised**, so at the start of training
$\text{raw}_\theta \equiv 0$ and $\hat{x}(t) = x_\text{prior}$ exactly. The
exponential guarantees $\hat{x}(t) > 0$ for all time (`raw` is clamped to
$[-10, 10]$ for numerical safety).

### 2. Relative physics residual

Acid–base reactions in ADM1 run at rates $\sim 10^{8}$, while biomass changes
over days. Normalising the residual by state magnitude would let the fast terms
swamp the loss. Instead the scale $s$ from section 4 is not fixed: each equation
is divided by its **own current rate** $\lvert f_i \rvert$ (floored so
near-equilibrium states with $f\approx 0$ don't blow up). The result is a
*relative* residual, so no single stiff equation dominates by sheer magnitude.

### 3. Quasi-steady gas

The four gas-phase pressures (especially the total pressure) are numerically
delicate: tiny changes can tip the solution over. Rather than let the network
predict them, it predicts **only the 37 liquid states**. The gas pressures are
computed from gas–liquid equilibrium at every evaluation (`gas_equilibrium_torch`).
The four gas ODEs are then satisfied automatically, so the physics loss only has
to check the 37 liquid states.

### 4. Prior anchoring and per-state weighting

The prior term from section 4 pulls **all** states toward their physical prior at
$t_0$. The data loss pulls the well-observed ones back off it, while the
weakly-observed ones stay stuck there. Which states this affects therefore
follows automatically from observability. To make the optimiser move such
sluggish states instead of parking them at the prior, they can be **up-weighted**
individually in $L_\text{phys}$: a hand-set weight vector, **not a learned
parameter**. Which states to up-weight is your choice (domain knowledge /
diagnosis), not something the training decides.

---

## 6. How it is implemented here

The implementation has **two layers**: a textbook template and the production
estimator built on top of it.

| Layer | File | Role |
| --- | --- | --- |
| `ADM1PINN` + `PINNLoss` | `deep_learning/pinn.py` | The building blocks: the MLP $t \to x$ (`tanh`, configurable hidden layers, optional dropout) and a generic *data + physics* loss template. Readable reference implementation. |
| `PinnSmoother` | `deep_learning/pinn_smoother.py` | The production estimator. Builds on `ADM1PINN` and adds the log-transform, the three-term loss, rate scaling, quasi-steady gas, forecasting, rolling updates and MC-Dropout uncertainty. |

The differentiable physics pieces it wires together all come from the base
`pyadm1` package:

| Symbol | Provided by |
| --- | --- |
| $f$ — ADM1 right-hand side | `pyadm1.core.adm1_torch.adm1da_rhs_torch` |
| gas–liquid equilibrium | `pyadm1.core.adm1_torch.gas_equilibrium_torch` |
| $h$ — measurement map | `deep_learning.observation_torch.TorchObservationModel` |
| feed / parameters | `Adm1TorchParams` — the substrate feed is baked in, so the network input is **time only** |

### Batch, not recursive

Unlike the UKF (a recursive `StateEstimator` doing predict/update each step), the
PINN is a **`BatchEstimator`**: fit the whole window once, then query it.

```python
from pyadm1ode_estimation.estimation.deep_learning import PinnSmoother

smoother = PinnSmoother(params, obs, x_prior, quasi_steady_gas=True)
smoother.fit(obs_times, obs_values, t0=0.0, t1=30.0)   # train over [0, 30] days
traj = smoother.estimate(query_times)                  # (T, 41) states + std
```

`estimate` returns a `TrajectoryEstimate` (`time`, `x_hat`, `std`), the shared
output currency with the UKF, so the twin-experiment harness scores both
estimator families the same way.

### Forecasting

Because the collocation window `[t0, t1]` is independent of the measurement
times, setting `t1` **past the last measurement** makes the ADM1 ODE carry the
state into the data-free tail — a physics-driven forecast from the same fit.

### Online operation

`update(...)` warm-starts from the current weights **and** optimiser state for
cheap incremental re-fits as new samples arrive:

* **growing window:** keep the whole history, anchor stays at the original  
  `t0`; or  
* **sliding window:** a fixed horizon `t0 = t1 − window`, self-anchored at the  
  network's own current estimate at the window start (bounds compute on long
  runs).

### Uncertainty

Optional **MC-Dropout**: set `dropout > 0` and call `estimate(..., mc_samples=k)`
to get the mean and standard deviation over `k` stochastic forward passes.

---

## 7. Strengths, limits, and where the UKF fits

**Strengths.** Fits the biogas-driving states tightly. The soft-physics coupling
is flexible and handles sparse, irregular sampling.

**Limits.** The pH / charge-balance map is ill-conditioned (both estimators
struggle here, the PINN more so). The MC-Dropout uncertainty is not yet
calibrated, and a from-scratch fit costs seconds to minutes per window.

**Where it sits.** The UKF (see [UKF in practice](../usage/ukf.md) and
[SR-UKF performance](../development/ukf_performance.md)) is recursive, cheap per
step, and better calibrated — especially on pH. The PINN is stronger on the
biogas channels and on forecasting. A covariance-intersection
**[hybrid](fusion.md)** can fuse the two, keeping each estimator's strengths.

---

## Source files

* `pyadm1ode_estimation/estimation/deep_learning/pinn.py` — `ADM1PINN`, `PINNLoss`  
* `pyadm1ode_estimation/estimation/deep_learning/pinn_smoother.py` — `PinnSmoother`  
* `pyadm1ode_estimation/estimation/deep_learning/observation_torch.py` — `TorchObservationModel`  

## References

* Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed  
  neural networks.* Journal of Computational Physics 378:686–707.  
* ADM1da model and state indices: [ADM1da model](adm1.md).  

## API reference

::: pyadm1ode_estimation.estimation.deep_learning.pinn_smoother.PinnSmoother
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.deep_learning.pinn.ADM1PINN
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
