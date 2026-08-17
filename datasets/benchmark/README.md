# ADM1 State-Estimation Benchmark

Estimate the **full 41-dimensional ADM1 digester state** from a handful of noisy
online sensors plus the (noisy) substrate feed — the classic soft-/state-sensor
task for an agricultural biogas plant. Same plant as the internal UKF comparison
(multi-stage example, digester `primary`, 5 substrates).

## The task

Per hourly time step you receive:

- **`measurements` (T, 5)** — noisy sensors: `Q_gas`, `Q_ch4`, `Q_co2` [m³/d],  
  `pH`, `TS` [% total solids].  
- **`feed_noisy` (T, 5)** — the substrate feed with 5 % dosing-report noise  
  (maize silage, solid manure, chicken litter, slurry, cereal grain).  
- **`fostac` (T, 2)** / **`fostac_true` (T, 2)** — the FOS/TAC laboratory  
  titration, measured and noise-free: column 0 FOS [mg HAc/L], column 1
  TAC [mg CaCO3/L]. Wet chemistry, not an online sensor, but stored **hourly**
  so you pick your own sampling frequency. See
  [Lab measurement](#lab-measurement-fostac).

Predict:

- **`states` (T, 41)** — the ADM1 state trajectory (the index → name → unit map  
  is in `meta.json`).

Each series spans **60 days** as **6 phases of ~10 days** (5 substrate load changes),
with smooth rate-limited spline transitions. The load drives the digester health via
**FOS/TAC** (VFA/TAC, the Nordmann acidification indicator): a low sustained load
keeps it healthy (~0.1–0.3), an overload drives it up toward critical (~1.0+).

## Operating modes

Every series belongs to one of **4 operating modes** (field `regime`), and both the
train and test sets contain all four:

| Mode | Scenario | FOS/TAC |
|---|---|---|
| `low_high` | healthy start, load ramped up | ~0.1 → ~0.8–1.0 (acidification onset) |
| `stable_low` | low load throughout | ~0.05–0.3 (healthy) |
| `stable_high` | chronic overload (pre-acidified) | ~0.4–1.0 (stays elevated) |
| `oscillating` | load swings up and down | sawtooth, net up (fast rise, slow fall) |

> **No recovery mode.** A FOS/TAC *decline* (high→low) is physically impossible with
> this plant: once past the acidification tipping point the acidified state is a
> stable attractor (verified to stay high even after 240 d at near-zero feed) — the
> irreversible-acidification that real digesters famously suffer. The asymmetry
> (fast acidification, no quick recovery) is itself part of what the modes show.

Each series also perturbs the ADM1 **kinetics** (lognormal, σ=0.25) — the hidden
model error. For the modes the kinetics are rejection-sampled so every series
actually exhibits its mode's FOS/TAC signature despite that spread.

The **substrate mix varies too**: each phase draws its own composition (2–5 of the
5 substrates active, biased to 4–5; a load change may also change *which* substrates
are fed), and it differs per series. Because the substrates differ ~11× in
degradable COD per m³ (slurry weakest, maize strongest), the load is steered as an
**organic (COD) load**, not a volume: for every phase the volumetric flows are scaled
so the COD load hits the target, which keeps the mode's FOS/TAC signature invariant
to the mix. `feed_true` / `feed_noisy` hold the actual volumetric flows [m³/d]; the
per-substrate characterisation and the scheme are in `meta.json → feed_composition`.

## Files

| File | Content |
|---|---|
| `train.npz` | **100** labelled series, 25 per mode (stacked): `measurements (100,T,5)`, `feed_noisy`, `feed_true`, `states (100,T,41)`, `time`, `switch_days`, `kinetic_factors`, `seed`, `regime` |
| `test.npz`  | **20** series, 5 per mode (per-series), each **with the true `states`** and the reference `ukf_x_hat` / `ukf_std`, plus `regime`, `seed`, `ukf_pending` |
| `meta.json` | 41-state index map + units, per-channel sensor noise, **substrate characterisation**, operating point (`k_dec_ac`, baseline load, load band), the `modes`, kinetic-perturbation ranges |
| `loader.py` | `load_train()`, `load_test()`, `load_meta()` (numpy only) |
| `scoring.py`| transient / steady / overall NRMSE + 2σ coverage, paired with the UKF |
| `viewer.py` | interactive browser (numpy + matplotlib) |

Browse the data with `python viewer.py` (`--dataset test` for the test set). Keys:
`←/→` switch series, `t` toggle train/test, **`v` toggle the view** — *overview*
(sensors + feed + key states) vs *states* (a paged grid of **all 41 states** with
the UKF ±2σ; page with `↑/↓`). Render one to file, e.g.
`python viewer.py --dataset test --view states --state-page 1 --save p1.png`.

The test labels are provided so you can **self-score locally**. Test series use
different (hidden) kinetics + seeds than the training set, so the score measures
**generalisation** — use the labels only for scoring, not for training.

## Sensors & noise

| Channel | Noise | Source |
|---|---|---|
| Q_gas | 3 % relative | literature biogas flow meter |
| Q_ch4 | 4 % relative | literature NDIR |
| Q_co2 | 4 % relative | literature NDIR |
| pH | ±0.02 absolute | Memosens CPS16E |
| TS | ±0.2 % absolute | Proline Teqwave MW 300 |
| feed | 5 % relative | dosing report |

The substrate composition (each substrate's ADM1 influent `s_in` at nominal
flow) is in `meta.json → substrates`, so you do not need the plant repo.

## Lab measurement: FOS/TAC

The acidification indicator every operator actually tracks. Unlike the five
channels above it is **wet chemistry**, so treat it as a sparse, high-value
observation rather than another sensor.

**Stored hourly, sampled on demand.** An independent titration is drawn for every
hour, even though no plant measures that often, so *you* choose the frequency
rather than inheriting one fixed when the files were written — which makes
"how often would we have to measure?" a question the data can answer:

```python
from fostac import subsample
weekly = subsample(series["fostac"], every_days=7)      # (9, 2)  — realistic
daily  = subsample(series["fostac"], every_days=1)      # (61, 2)
masked = subsample(series["fostac"], every_days=7, as_mask=True)   # (T, 2), NaN in the gaps
```

Subsampling is exact, not an approximation: each hour carries its own
independent draw, so rows 0, 168, 336, … are precisely the 9 independent
titrations a weekly operator would perform. `fostac_true` holds the same values
without measurement noise, so scoring needs no pyadm1 or torch.

**FOS and TAC are two numbers but one measurement.** Nordmann (1977) titrates
*one* 20 mL sample with 0.1 N H₂SO₄ through two consecutive endpoints:

| Value | Titration leg | Formula |
| --- | --- | --- |
| TAC | start → pH 5.0 | `TAC [mg CaCO₃/L] = V₁ · 250` |
| FOS | pH 5.0 → pH 4.4 | `FOS [mg HAc/L] = (V₂ · 1.66 − 0.15) · 500` |

The FOS leg starts where the TAC leg stops, so their errors are **coupled**: a
sample/dilution error scales both together, while an error in finding the pH 5.0
endpoint moves titrant out of one leg into the other. `fostac.py` therefore
models the titration itself (perturb the volumes, recompute) instead of adding
two independent Gaussians.

That geometry also produces the asymmetry seen in the literature (≈1.45 % for
alkalinity, ≈6.7 % for VFA): at TAC ≈10 g/L the first leg is ≈40 mL, at
FOS ≈2 g/L the second is only ≈2.5 mL, so the same 0.15 mL endpoint error is
0.4 % of one and 6 % of the other. The resulting FOS noise is
concentration-dependent — ≈10 % at 1200 mg/L, ≈6.7 % at 2000 mg/L, ≈3 % at
5400 mg/L — and the FOS/TAC correlation rises from ≈0.05 (healthy) to ≈0.5
(acidified).

**Detection limit.** A titration whose true FOS falls below the formula's offset
gives a non-positive value and is clipped to `0` (0.1 % of hours, concentrated in
the healthiest series). Treat `FOS == 0` as *below detection* and drop it rather
than fitting to it — its sigma is meaningless.

**Known idealisation.** The stored value is the model's *true* VFA sum, not the
empirical Nordmann proxy. The method bias of the Nordmann formula (it grows with
total solids and at high VFA) is deliberately not modelled, so measurement noise
stays separable from method error. Regenerate with `python add_fostac.py`.

## Scoring

The skill being tested is tracking the state **through the feed changes**, so the
headline metric is the **transient NRMSE** — the per-state error in the 48 h
window after each switch. Steady-state and overall NRMSE plus 2σ coverage are
reported too, always next to the reference UKF.

**The reported NRMSE is a median over the 41 states, not a mean.** A few states are
numerically ~zero (`S_cation`, `S_h2` and `p_gas_h2` have an RMS around 1e-6), so
their *relative* error reaches five-digit percentages while the absolute error is
irrelevant. A mean is dominated by exactly those states (it explodes into the
thousands of percent while the median stays at a few tens of percent) and would
reward optimising noise. The mean is still reported as `nrmse_overall_mean_%` for
transparency.

**NRMSE is normalised by a state's magnitude**, so it flatters high-offset states
(pH ~7.4) and inflates near-zero ones — it mixes *how big* a state is with *how well*
it is estimated. Two metrics avoid that:

- **`vs_ukf` — the RMSE ratio model/UKF** (median over states, per window). `<1` beats  
  the UKF. Because both errors are measured on the *same* state, its scale, offset and
  variation cancel completely — this is the fairest single number, and beating the UKF
  is the goal, so **this is the metric to optimise**.  
- **`per_state_report(pred, series, window)` — per-state RMSE in physical units** next  
  to each state's own `true_mean` / `true_range` (plus the UKF's RMSE and the ratio).
  The honest way to see *which* states you miss and whether an error is actually large
  (e.g. the UKF's `S_ac` RMSE ≈ 4 is huge against its range ≈ 6 — it barely tracks it,
  which the magnitude-normalised NRMSE hides). `per_state_nrmse` remains for continuity.

```python
from loader import load_test
from scoring import score_series, score_dataset, per_state_report

test = load_test()
preds = [my_model(s) for s in test]           # each (T, 41)
print(score_dataset(preds, test))             # model vs UKF + vs_ukf_mean ratio
print(score_series(preds[0], test[0]))        # one series: NRMSE + coverage + vs_ukf
rep = per_state_report(preds[0], test[0])     # per-state RMSE + context (which states)
```

Beating the UKF's **transient NRMSE** on the test series is the goal; matching its
calibration is the bonus.

## The UKF reference

`ukf_x_hat` / `ukf_std` come from the full 41-state Unscented Kalman Filter run on
the same 5 sensors, in the **known-input** variant: the reported substrate feed is a
known control input (not an estimated/observed state). It does **not** know each
series' true kinetics — a realistic baseline that carries the model mismatch.
You never run it yourself; it is provided for comparison.

The reference to beat (median over states, mean over the scoring series):

| Metric | UKF reference |
|---|---|
| transient NRMSE | **18 %** |
| steady NRMSE | 21 % |
| overall NRMSE | 20 % |
| 2σ coverage | **91 %** |

Per mode (overall NRMSE): low_high 14 % · oscillating 15 % · stable_low 25 % ·
**stable_high 28 %** — chronic acidification from a cold healthy start stays the hardest
case. The acidified VFA states (`S_ac`, `S_pro` and their ions) remain the weak spot, but
the filter now tracks them well enough to be a demanding baseline.

> **The reference was re-tuned (2026-08) and is now ~2.5x more accurate** than the
> previously shipped one (overall NRMSE 52 % → 20 %). Its process/measurement noise was
> optimised with CMA-ES, warm-started from the *empirically measured* model error rather
> than a hand-set Q. The tuning pipeline is
> `pyadm1ode_estimation/estimation/filter_tuning/cmaes_search.py`. The previous reference and
> the full write-up of the tuning are kept outside this repository.

Its **uncertainty is calibrated** on the training set (see
`estimation/filter_tuning/sigma_calibration.py`, built on the data-agnostic
`estimation/calibration.py`) by **post-hoc per-state σ
recalibration** — the reported σ of each state is scaled by √(NEES) fitted on the
training residuals, so the ±2σ band is well covered and the states the filter cannot
track carry an honestly large σ, **without changing the point estimate**. (Inflating
the process noise `Q` instead was tried and rejected: with the known-input variant's
fewer observations it corrupts the prediction.) The shipped `ukf_calibration.json`
uses the **cross-validated** best config (train/val/test, `σ_hi=10, γ=2.2`). Against the
CMA-ES-retuned reference the shipped test coverage is **0.91** (0.92 before the retune —
σ was fitted on 4 training series at 60 days). To regenerate the reference from
`ukf_calibration.json`, use `cmaes_search.build_reference`.

**FOS/TAC critical decision.** Estimating FOS/TAC (VFA/TAC) — "is the plant critical?"
— is hard from these 5 sensors (the VFA states are barely observable; even *true*
kinetics do not fix the point estimate).

The CMA-ES retune improved this too: the shipped reference now **ranks** critical from
healthy almost perfectly (**AUC 0.95**), but the fixed alarm rule ("upper 2σ FOS/TAC band
> 0.6") sits badly on that ranking, so balanced accuracy is only **0.54**. A calibrated
threshold is cheap headroom and has nothing to do with filter tuning.

An **A+D-core UKF** that estimates only the 18 provably observable states (methanogenesis
plus charge balance) and propagates the rest open-loop keeps the TAC denominator physical
instead of letting it collapse. Untuned, that structural advantage made it the better
critical detector by a wide margin.

> **It does not survive a fair comparison.** Both variants were retuned with *identical*
> settings (CMA-ES, same episodes, 12 generations, 8 candidates, weighting 0.7 / 0.3,
> 60-day verification, selection on validation). On the same 20 test series, alarm rule
> `P(FOS/TAC > 0.6) > 0.5`:
>
> | Filter (identically tuned) | NRMSE | AUC | bal-acc | TPR | TNR |
> | --- | --- | --- | --- | --- | --- |
> | full UKF | **0.278** | **0.948** | **0.789** | **0.581** | 0.997 |
> | A+D core | 0.286 | 0.923 | 0.678 | 0.356 | 1.000 |
>
> The full UKF wins on every metric, and the gap survives a free choice of alarm threshold
> (best achievable balanced accuracy 0.874 against 0.836). Two side findings: the A+D core
> looks *better* on short windows and collapses at full length (AUC 1.000 at 20 days to
> 0.853 at 60 days), and tuning buys it almost nothing at 60 days (+0.1 % NRMSE against
> +67 % for the full filter), because 23 of its 41 states run open-loop where Q cannot act.
