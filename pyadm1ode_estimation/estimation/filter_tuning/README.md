# estimation/filter_tuning — tune & calibrate the model-based filters

Dataset-agnostic tooling to tune the noise / uncertainty parameters (**Q, R, σ**) of our
model-based **filters** (UKF variants, and a differentiable filter). This is distinct from
`estimation.deep_learning`, which *trains* neural estimators — here we **calibrate/tune
filters**. Everything is split into three concerns so any filter can be tuned on any dataset:

| File | Responsibility |
|---|---|
| `datasets.py` | **Load ANY dataset** → `Episode` objects + stratified **train/val/test** splits. The single place that knows a file format (`EstimatorDataset`, `get_dataset`). |
| `filter_runners.py` | Build & run a model-based **filter** on one episode from the dataset's `meta` (plant/operating point). Exposes `make_ukf_runner(meta, variant) -> run_episode(theta, ep)`. Variants: `full` (41-state), `adcore` (18-state observable core). |
| `metrics.py` | Shared metrics: 2σ coverage, NEES, **FOS/TAC-band coverage**, **critical-state decision** (TPR/TNR/balacc), and the selection `objective`. |
| `sigma_calibration.py` | Post-hoc per-state σ recalibration (`√NEES` clip·γ). Only rescales the reported σ — **does not change x̂**; the (σ_hi, γ) search is free. |
| `noise_search.py` | **Approach 1.1** — search Q (per block) + R + P0 on validation (grid / random / Bayesian opt), with σ-recalibration on top. Changes x̂ → needs re-runs. Also `evaluate_batch`, which scores a whole population in ONE parallel wave. |
| `differentiable.py` | **Approach 1.2** — differentiable EKF over `adm1_torch`; learns diagonal Q, R by back-propagating an NLL + supervised loss (Barratt–Boyd / BackpropKF). |
| `empirical_noise.py` | Q and R computed **from ground truth** instead of searched: `Q = Var(x_true(k+1) − f(x_true(k)))`. Costs ~one filter episode. |
| `cmaes_search.py` | **Approach 1.3 — the pipeline that produced the shipped reference.** Stages 0–3: empirical Q → CMA-ES over per-block corrections + R + P0 → full-length verification → build the reference. Plus `term_breakdown` for diagnosing the objective. |
| `tune_filter.py` | CLI orchestrator: split → tune → val-select → report on the held-out test set → save JSON. |

## Design: one loader hands data to every tuner

```python
from pyadm1ode_estimation.estimation.filter_tuning import get_dataset
from pyadm1ode_estimation.estimation.filter_tuning.filter_runners import make_ukf_runner
from pyadm1ode_estimation.estimation.filter_tuning import sigma_calibration as sig

ds = get_dataset("benchmark")                       # name OR path to a dataset directory
train, val, test = ds.make_splits(days=30, burnin_days=2,
                                  per_group_train=2, per_group_val=2)   # stratified by mode
run = make_ukf_runner(ds.meta, variant="full")       # run_episode(theta, episode) -> (x_hat, std)

tr, va = sig.collect(run, train), sig.collect(run, val)
best, _ = sig.search_sigma(tr, va)                   # (σ_hi, γ) selected on validation
```

A new dataset only needs a loader that yields `Series` (measurements, feed, time, truth,
label) — register it in `datasets.LOADERS`; all tuners then work unchanged.

## CLI

```bash
# post-hoc σ calibration (both variants)
python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter sigma --dataset benchmark --variant full
python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter sigma --dataset benchmark --variant adcore

# 1.1 — Q/R/P0 search on validation
python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter noise --variant full --method random --n-iter 20
python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter noise --variant adcore --method bayes --n-iter 30   # needs scikit-optimize

# 1.2 — differentiable EKF, learn Q,R by gradient
python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter diff --days 8 --epochs 10 --lr 0.1
```

Results are written to `filter_tuning_results/` (or `--out`).

## CMA-ES pipeline (1.3)

Not exposed through `tune_filter.py`, because stage 1 runs for hours and belongs in a
detached script rather than a CLI call. Import it directly:

```python
from pyadm1ode_estimation.estimation.filter_tuning import get_dataset
from pyadm1ode_estimation.estimation.filter_tuning import cmaes_search as cs

ds = get_dataset("benchmark")

# Stage 0 — Q and R measured from ground truth. Roughly one filter episode; do it once
# and keep the JSON, `load_empirical_q()` reads it back.
emp = cs.empirical_q(ds, jobs=8, save_to="empirical_noise.json")
q_emp = cs.load_empirical_q("empirical_noise.json")

trace = cs.run_cmaes(ds, q_emp, variant="full", days=20, popsize=8, generations=12,
                     jobs=30, out_path="trace.json")          # hours — run detached
ver = cs.verify(ds, {"nominal": {}, "cma": cs.theta_from_x(trace["best"]["x"], q_emp, blocks)},
                days=None, jobs=30)                            # full length, on val
ref = cs.build_reference(ds, theta, split="test", jobs=30, out_npz="reference.npz")
```

Requires `cma`, which is not in `requirements.txt` because only this module needs it. The
trace is written after every generation, so a crash costs at most one generation.

## Notes

- **Objective** = 2σ coverage → 0.955 **+** FOS/TAC-band coverage → 0.955 **+** critical  
  balanced accuracy (see `metrics.objective`). Tunes both honest uncertainty and the
  "is the plant critical?" decision.  
- **Cost:** the UKF base runs dominate 1.1; `tune_filter.py` is serial — for large sweeps run  
  candidates/episodes as parallel processes (as the archived `old/benchmark` scripts did).
  1.2 is heaviest (autograd Jacobians of the stiff RHS) → use short `--days`, few episodes.  
- Only 1.1/1.2 change the point estimate x̂; σ-calibration only makes the error bars honest.  
  The FOS/TAC point accuracy is capped by observability, so no amount of tuning fixes it.
