# SR-UKF Performance — Architecture and Optimisations

The [`UnscentedKalmanFilter`](../../api/index.md) implements the
canonical Square-Root UKF after **Wan & van der Merwe 2001
(Algorithm 3.1)**, in the variant that re-uses sigma points between
predict and measurement update. This page documents which optimisations
brought the code to that state and how their correctness is verified.

## Current algorithm

Per filter step:

1. **Predict** draws `2n+1` sigma points around `(x_hat, S)`,  
   propagates them through `process.step(σ_i, dt)`, and *in the same
   loop* reads `h(plant, σ_i_propagated)` for every observation
   channel. Predicted mean + Cholesky factor via QR-stack with `√Q`.  
2. **Update** consumes the predict-side h-cache directly (column  
   selection over the active channels). No second sigma-point draw, no
   second plant pass. Cross-covariance `T_xy` from
   `(propagated − x_pred)` differences.

ODE load per filter step: **`2n+1` plant integrations** (predict only).
The earlier "redraw" variant did `2 · (2n+1) = 4n+2` — twice the load
for no mathematical gain.

## Optimisation waves (2026-06)

### Wave 1 — lossless linalg refactors

All four changes are bit-stable to last-bit roundoff (`atol=1e-12`):

1. **`_sigma_points` vectorised** — Python loop → two broadcasts.  
2. **`T_xy` vectorised** — `for i: outer(...)` → one matmul.  
3. **`cholesky(Q)` cached** — keyed on `dt`, hit when `dt` is constant.  
4. **`cholesky(R)` cached** — keyed on the active-channel tuple.  

### Wave 2 — sigma-point reuse

Structural change saving ~50 % of ODE calls. The earlier code drew a
fresh sigma set around `(x_pred, S_pred)` inside `update()` and stepped
the plant once per sigma — an extension that is **not** part of the
canonical Wan-VdM 2001 form. In a 24-hour ADM1 twin (fixed seed, simple
plant), the migration delivered **1.84× speedup** (429 s → 234 s) with
comparable per-block accuracy.

The mathematical approximation it accepts: the `H · Q · H^T`
contribution to the innovation covariance `S_y` is dropped (propagated
sigma points cover `sample_cov`, not `sample_cov + Q`). Wu et al. 2005
show this is inferior to the state-augmented UKF — by an amount that
scales with `||Q|| / ||P||`. For ADM1 at `dt = 1 h` with the spec's
typical `Q/P` ratio (≈ 1/250), the effect is two orders of magnitude
below the measurement-noise floor. Hellmann et al. 2024 (ECC, ADM1
specifically) find identical trajectories between their `UKF-SR`
variant (= our current form) and `UKF-add` without augmentation.

## Literature

| Source | Statement |
|---|---|
| Wan & van der Merwe (2001) "The Square-Root Unscented Kalman Filter for State and Parameter-Estimation" | Algorithm 3.1, line 22: `Y_{k|k-1} = H[X_{k|k-1}]` — h is evaluated directly on the propagated sigma points, no redraw. |
| Wu, Hu, Wu & Hu (2005) "Unscented Kalman filtering for additive noise case: Augmented vs. non-augmented" | The augmented form (Q-dimensions in the sigma set) is theoretically superior because it propagates Q's odd-moment information through the measurement path. Effect scales with `||Q|| / ||P||`. |
| Hellmann, Wilms, Streif & Weinrich (2024, ECC) "Comparison of Unscented Kalman Filter Design for Agricultural Anaerobic Digestion Model" | Direct ADM1 study, 9 UKF variants. UKF-SR (= our form) and UKF-aug deliver practically identical NRMSE under reduced sigma scaling, with lower runtime for UKF-SR. |

## How verified

### Bit-stable regression — nonlinear mock process

[`tests/test_ukf_regression.py`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/tests/test_ukf_regression.py)
runs a 20-step twin with a nonlinear mock process (elementwise
quadratic + linear coupling) and compares the final posterior against
hardcoded golden values (`atol=1e-12, rtol=0`). Goldens were regenerated
after the main-path migration.

**Regenerate goldens** (after deliberate algorithm changes):

```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")
import conftest  # Windows DLL paths
sys.path.insert(0, "tests")
import test_ukf_regression as m
import numpy as np
np.set_printoptions(precision=16, floatmode="maxprec")
x, S = m._run_regression_trajectory()
print("X =", repr(x))
print("S =", repr(S))
PY
```

### Linear-KF consistency

[`tests/test_ukf.py::TestUKFLinear`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/tests/test_ukf.py)
has two tests against the closed-form Kalman filter:

* `test_ukf_matches_classical_kf_with_negligible_Q` — with `Q ≈ 1e-16`  
  (ten orders of magnitude below `P`) the SR-UKF trajectory matches the
  KF to `atol=1e-6`. Pins the algebraic correctness of the reuse form.  
* `test_ukf_approximates_classical_kf_with_random_walk_Q` — with  
  `Q/P ≈ 1 %` the trajectory diverges from the KF by a few percent,
  consistent with the dropped `H Q H^T` term. `atol=0.05`. Pins the
  magnitude of the approximation; a clear violation would signal an
  unintended algorithmic shift.

### Cache invalidation

Four further tests in `test_ukf_regression.py` use counting wrappers:

* `test_sqrt_Q_cache_hits_under_constant_dt`  
* `test_sqrt_Q_cache_invalidates_on_dt_change`  
* `test_sqrt_R_cache_hits_under_constant_active_set`  
* `test_sqrt_R_cache_invalidates_on_active_set_change`  

### Wave 3 — Reduced sigma scaling (opt-in)

Since 2026-06, ``UnscentedKalmanFilter`` accepts a ``gamma_override``
parameter that replaces the canonical sigma-point radius
``γ = √(n + λ)`` with a user-supplied value. The weights ``w_m``,
``w_c`` remain canonical (built from ``α, β, κ``) — only the position
of the sigma points on the covariance ellipsoid changes.

Empirical motivation from Hellmann et al. 2024 §5.1.2: on
ADM1-R4-Core, ``γ = 1`` (instead of canonical ``√(n+λ) ≈ 2.45``) cuts
NRMSE_x from 0.85 to 0.37 — the single largest gain in the paper. The
intuition: for high-dimensional, weakly nonlinear systems with
Gaussian-ish measurement noise the tighter sigma cloud smooths better
than the canonical Julier–Uhlmann scaling.

Usage:

```python
ukf = build_ukf(plant, digester_id="primary", substrates=...,
                gamma_override=1.0)
```

Default ``None`` leaves the sigma-point scaling unchanged — all
existing setups continue to behave exactly as before.

#### Empirical γ sweep on the simple plant (n=43)

24-hour twin, fixed seed (42), identical truth and measurements,
only γ varies. Wall time per variant 229-242 s — γ is a pure quality
and calibration knob, not a speed knob.

| γ | Mean NIS (target ≈ 5) | Avg. coverage | Worst block | Verdict |
|---|---|---|---|---|
| canonical (≈ 6.56) | 6.75 | **96.5 %** | charge_balance 70 % | well calibrated |
| 5.0 | 9.79 | 91.2 % | charge_balance 60 % | mildly overconfident |
| 4.0 | **17.23** | 84.8 % | **nitrogen 16 %** | calibration collapse |
| **3.0** | **44.80** | 81.2 % | nitrogen 20 % | **catastrophic** |
| 2.0 | 17.80 | 84.0 % | charge_balance 50 % | bad |
| 1.5 | 9.06 | 87.4 % | charge_balance 48 % | mediocre |
| 1.0 | 6.32 | 88.4 % | charge_balance 50 % | best mean, overconfident |

RMSE: on well-observable blocks the point-estimate error drops by
40-73 % at γ=1; on the weakly-observable `input_flow` channels it
rises by 25 %. Intermediate γ values (2-5) are strictly worse than
either endpoint on **both** metrics.

#### Why non-monotonic

The weights ``w_m``, ``w_c`` are derived from the canonical scaling
``γ = √(n+λ)`` such that the unscented transform recovers mean and
covariance to third order — *only* when paired with that exact γ.
Hellmann's trick decouples γ from the weights: tight sigma cloud +
weights built for a wider cloud → the reconstructed posterior
covariance is systematically too small, NIS explodes.

At γ=1 a different dynamic (effective low-pass smoothing of the
measurement) takes over and provides acceptable NIS — for formally
"wrong" reasons. In the transition zone (γ ≈ 2-5) neither effect
applies cleanly, so filter consistency collapses.

#### Recommendation

* **Default ``gamma_override=None``** for UQ-relevant applications  
  (coverage calibration is the priority there).  
* **``gamma_override=1.0``** as opt-in for modes that only consume  
  the posterior mean (e.g. MPC targets, control signals), where
  coverage is secondary.  
* **Avoid values between 1 and 6** — no sweet spot, both metrics  
  degrade simultaneously.  
* For *simultaneously* good mean and calibration: state-augmented  
  form (Wave 4 on the roadmap) fixes the `Q` inconsistency in the
  algorithm itself instead of compensating via γ.

### Wave 4 — Process-pool parallelisation of sigma propagation

New class [`ParallelUKF`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/pyadm1ode_estimation/estimation/filters/parallel_ukf.py)
dispatches the ``2n+1`` plant integrations of a ``predict()`` step to
a ``multiprocessing.spawn`` worker pool. The linalg path (QR,
cholupdate, weights) stays on the main process; only the embarrassingly
parallel sigma-point propagation moves to workers.

#### Architecture

* Each worker builds **its own** ``(process, obs, spec)`` triple at  
  pool startup via a user-provided top-level builder. This sidesteps
  the pickle issue with ``obs`` closures referring to plant objects.  
* Per task only the tiny snapshot dict (~1 kB) crosses the IPC  
  boundary. The plant objects themselves cross only once at pool spawn.  
* ``process.step`` calls ``restore()`` internally — we extended  
  [``process_model.snapshot/restore``](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/pyadm1ode_estimation/estimation/process_model.py)
  so that *every* component with ``adm1_state`` is reset, not just the
  primary digester. This fixes a latent drift bug in the serial code
  (storage tank accumulated state across sigma points within a predict)
  and makes the UKF theoretically cleaner: every sigma point now sees
  an independent baseline.

#### Benchmark (simple plant, 24-h twin, fixed seed)

| Variant | Wall time | Speedup | s per filter step |
|---|---|---|---|
| serial | 238.6 s | 1.00× | 9.94 |
| parallel_2 | 207.7 s | 1.15× | 8.66 |
| parallel_4 | 111.9 s | 2.13× | 4.66 |
| parallel_8 | 80.6 s | **2.96×** | 3.36 |

**Block-RMSE and 2σ coverage are bit-identical across worker counts.**
Mean-trajectory delta vs. serial: exactly 0 σ on every channel. Mean
NIS = 7.08 for all variants.

Sub-linear scaling expected (Amdahl):  
* Pool-spawn overhead (~2-3 s per worker on Windows spawn)  
* Linalg path stays sequential and grows in relative weight with more workers  
* Beyond ~4 physical cores (likely hyperthreads on this setup) the gain plateaus  

For a typical twin workflow (5 days × 24 h = 120 steps), that means
serial ≈ 20 min → parallel_8 ≈ 7 min.

#### Usage

```python
from pyadm1ode_estimation.estimation import (
    InputSpec, build_filter_components,
)
from pyadm1ode_estimation.estimation.filters import ParallelUKF
from pyadm1ode_estimation.example_plants import build_simple_plant

# Top-level (importable) — multiprocessing.spawn pickles a reference,
# so NO lambdas or closure-bound methods here.
def make_components():
    plant = build_simple_plant()
    return build_filter_components(
        plant,
        digester_id="fermenter",
        substrates=[
            InputSpec("maize_silage",  substrate_index=0, initial_flow=10.0),
            InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
        ],
        sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
    )

process, obs, spec = make_components()
ukf = ParallelUKF(
    process, obs, spec,
    n_workers=4,
    components_builder=make_components,
)
# ... normal predict/update loop ...
ukf.shutdown()  # release the worker pool
```

``n_workers=1`` falls back to the serial main path — no pool
overhead, identical behaviour to the base class.

### Wave 5 — Constrained UKF: implemented, negative result on ADM1

[`ConstrainedUKF`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/pyadm1ode_estimation/estimation/filters/constrained_ukf.py)
implements Hellmann 2024's `cUKF-add`: every propagated sigma point is
corrected through a **per-sigma box-constrained QP** instead of the
standard Kalman gain.

```math
\chi^{\text{corr}}_i = \arg\min_\chi \;
  \|y - h(\chi)\|^2_{R^{-1}}
  + \|\chi - \chi^-_i\|^2_{(P^-)^{-1}}
\quad \text{s.t.} \quad x_{\text{lo}} \le \chi \le x_{\text{hi}}
```

Implementation choices:  
* `h` linearised by least-squares fit from the cached propagated sigma  
  cloud (no extra plant evaluations)  
* Per-sigma QP via `scipy.optimize.minimize(method="trust-constr")`  
  with analytic gradient and Hessian  
* Posterior in square-root form per Hellmann eq. 11:  
  `P = Σ Wᶜ (χᶜ - x̂)(χᶜ - x̂)^⊤ + Q + K R Kᵀ`  
* Four unit tests pin correctness: with wide bounds and linear h,  
  `cUKF ≡ UKF` to `atol=1e-6`; with tight bounds the box constraint
  is respected; smoke test on the ADM1 plant runs end-to-end.

#### Empirical result (24-hour twin, simple plant, n=43)

| Variant | Wall time | Mean NIS | Avg. block RMSE |
|---|---|---|---|
| UKF (baseline) | 232.1 s | 7.08 | reference |
| cUKF | **571.9 s** (2.5× slower) | 9.32 (worse) | **2-5× worse** on well-observable blocks |

Methanogenesis RMSE: 0.015 → 0.077 (5× worse). Disintegration: 0.20 →
0.95. Coverage uniformly down or flat. Per-channel mean trajectories
diverge by up to **2390 σ** from the UKF baseline.

#### Structural cause: sigma-spread collapse on a multi-scale state

ADM1 has 43 concentration channels spanning **six orders of
magnitude** — from substrate dosing (10 m³/d) to trace gases
(≈ 10⁻⁸ mol/L). The spec sets `lower = 0` on all concentrations.

For small-magnitude channels (methanogenesis traces, acidogenesis
intermediates) the propagated sigma points sit close to the
lower bound. Per-sigma bounding pulls a substantial fraction of the
sigmas exactly onto the bound → the corrected sigma set clusters →
`Σ Wᶜ (χᶜ - x̂)²` collapses along that axis → posterior covariance
under-estimates dramatically → next predict starts with too tight a
sigma cloud → the filter goes "blind". The result is the ~10²-σ mean
drifts on the weakest channels.

Hellmann (2024) doesn't see this because their n=6 model keeps every
state in the kg/m³ range (`x_0 = [4.09, 10.52, 11.04, 2.57, 0.96,
2.02]`). The zero bound is effectively inactive there. ADM1's
multi-scale concentrations are a qualitatively different situation.

**Finding**: this is not an implementation or linearisation defect
but a structural limitation of per-sigma-point bounding on
bound-rich state spaces. For the doctoral work it stands on its own
as a negative result: Hellmann's `cUKF-add` — the best variant in
their n=6 benchmark — **does not transfer directly** to a realistic
multi-stage ADM1 with concentration channels across six orders of
magnitude.

The class stays in the repo as opt-in for use cases where the bounds
are practically inactive (simpler AD models per Hellmann 2024, or
bioreactor models with moderate state ranges).

## What's next on the list

| Idea | Expected impact | Effort |
|---|---|---|
| **Log-scaling of small channels** | Parametrise `x_i = exp(z_i)` for trace-gas channels. The `x_i > 0` bound becomes implicit in the transform, no per-sigma bounding needed. Likely route to recover Hellmann's cUKF gains on ADM1. | Medium — spec extension (`channel.log_scale`) plumbed transparently through process/obs/spec. |
| **State-augmented form** (Wu 2005, Hellmann's `UKF-aug`) | Correct treatment of `Q` in the measurement path. Empirically small for ADM1 at small `Q/P`, but theoretically cleaner; could algorithmically resolve the Wave 3 γ-sweep disaster region. | Medium — `2(n+m)+1` sigma points instead of `2n+1`, with the Q-block efficiently groupable (Wu 2005 §IV). |
| **Spherical-simplex sigma points** (Julier 2003) | `n+2` instead of `2n+1` sigma points → another ~2×, stacks with the parallelisation. | Medium — recursive simplex construction, its own validation. |
| **NLP cUKF with plant callback** | Hellmann's true variant (evaluate `h(χ)` per QP iteration via plant.step instead of linearising). Removes the linearisation approximation but does NOT fix the sigma-spread collapse — so unlikely to beat the current Wave 5 result. | High — ~10 plant equilibrations per QP iteration, parallelisable via the Wave 4 pool. |
