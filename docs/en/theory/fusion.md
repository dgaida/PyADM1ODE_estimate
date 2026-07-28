# UKF ↔ PINN fusion (covariance intersection)

The [UKF](../usage/ukf.md) and the PINN family have **complementary** strengths:
the UKF is well-posed and well-calibrated on the pH / charge-balance states, the
PINN ([smoother](pinn.md) or [amortised observer](observer.md)) is strong on the
biogas-driving states and forecasts naturally. The fusion combines their two
state trajectories so the result keeps the best of each.

---

## 1. The idea: Covariance Intersection

We have two estimates of the same state, but we do **not** know how correlated
their errors are (they see the same plant and the same noise). Covariance
Intersection (CI) fuses them *conservatively* without assuming independence, so
the fused estimate stays consistent (never over-confident) and is at least as
informative as either input.

Per state, CI weights by **information** (inverse variance): the more confident
estimate contributes more. A single weight $\omega \in [0,1]$ per time step trades
the two off.

---

## 2. The mathematics

Take two diagonal-covariance estimates with means $m_a, m_b$ and per-state
variances $v_a = \sigma_a^2,\ v_b = \sigma_b^2$, and write the information
(inverse variance) as $I = 1/v$. CI fuses them as

$$
I_f = \omega\, I_a + (1-\omega)\, I_b, \qquad
m_f = I_f^{-1}\big(\omega\, I_a\, m_a + (1-\omega)\, I_b\, m_b\big),
$$

with fused variance $v_f = 1/I_f$. The weight $\omega$ is either fixed or, per
time step, **optimised** by a grid search that minimises the fused covariance
determinant. Equivalently, it maximises $\sum_\text{states}\log I_f$.

Because $\omega$ is chosen per step and the covariances are diagonal (built from
the per-state standard deviations), the fusion is cheap and applies to a whole
trajectory at once.

---

## 3. Implementation

`fuse_ci_diagonal(mean_a, std_a, mean_b, std_b, omega=None)` is the core: a
per-time-step CI of two `(T, n)` estimates, returning the fused mean, fused std,
and the chosen $\omega$ per step.

`HybridEstimator` wraps it in the [`BatchEstimator`](pinn.md) contract. Build it
from the two sub-estimates, and `estimate()` returns their CI fusion as a
`TrajectoryEstimate`:

```python
from pyadm1ode_estimation.estimation.fusion import HybridEstimator

hybrid = HybridEstimator(traj_ukf, traj_pinn)   # two TrajectoryEstimates
fused = hybrid.estimate()                        # CI fusion on the shared grid
```

Both inputs must live on the same time grid, `fuse_trajectories_ci` enforces
matching shapes.

---

## 4. Caveat: calibration is a prerequisite

CI weights purely by covariance, so **it is only as good as the inputs'
calibration**. An over-confident estimator, e.g. MC-Dropout bands that are too
narrow, is over-trusted and drags the fusion toward it. In the twin comparison
the fusion *never hurts*, but it cannot yet blend the UKF's pH strength into the
state estimate, because the PINN's uncertainty is not yet calibrated.
Calibrating the PINN band (and fixing the pH conditioning) is the open item that
unlocks the fusion.

---

## Source files

* `pyadm1ode_estimation/estimation/fusion/hybrid.py` — `HybridEstimator`, `fuse_ci_diagonal`, `fuse_trajectories_ci`

## API reference

::: pyadm1ode_estimation.estimation.fusion.hybrid.HybridEstimator
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.fusion.hybrid.fuse_ci_diagonal
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
