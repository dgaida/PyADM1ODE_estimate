# The estimator registry

Three estimator families live in the package, each with its own training code:
the recursive filters (`filters/`, tuned in `filter_tuning/`), the collocation
PINN (`pinn/`) and the amortised GRU observer (`observer/`). The registry in
`estimation/estimators.py` gives them **one interface**, so an experiment can
swap the estimator without touching the data layer.

```python
from pyadm1ode_estimation.estimation.estimators import build_estimator, score
from pyadm1ode_estimation.estimation.filter_tuning import get_dataset

ds = get_dataset("benchmark")                 # or a dataset directory
est = build_estimator("ukf", ds)              # "ukf" | "pinn" | "observer"
traj = est.run(ds.test[0])                    # TrajectoryEstimate: time, x_hat, std
score(traj, ds.test[0], burnin_steps=48)      # median-over-states NRMSE
```

Every estimator consumes the shared `Series` of
`filter_tuning.datasets` and returns a `TrajectoryEstimate`
(`time (T,)`, `x_hat (T, 41)`, `std (T, 41)`; zeros when the method has no
uncertainty). The benchmark metric, the fusion code and the comparison plots see
one currency.

## The estimators

| Name | Class | What it runs | Options |
| --- | --- | --- | --- |
| `ukf` | `UkfEstimator` | 41-state SR-UKF (`variant="full"`) or the acid + digester core (`"adcore"`) with the shipped CMA-ES reference `theta` from `estimation/reference/` | `variant`, `theta`, `fostac_every_days` |
| `pinn` | `PinnEstimator` | [PINN v2](../theory/pinn.md) in the causal rolling window, best structure of the study | `arch`, `cfg`, `roll`, `bridge="adm1" \| "hold"` |
| `observer` | `ObserverEstimator` | the trained [GRU observer](../theory/observer.md) (PINN B) in one forward pass | `checkpoint` (default: the shipped `estimation/reference/observer.pt`) |

Cost per 60-day series on a CPU: the UKF needs minutes (a 30-day warm-up plus
1440 sigma-point steps), the PINN tens of minutes (60 retrainings), the
observer well under a second.

## Training the learned estimators

The observer is trained from a `PinnData` wrapper, which fixes the stratified
split and the train-only normalisation:

```python
from pyadm1ode_estimation.estimation.estimators import ObserverEstimator
from pyadm1ode_estimation.estimation.observer import PinnData

data = PinnData.build("benchmark", val_frac=0.2, seed=0)
est, result = ObserverEstimator.train(data, window=336, epochs=200, patience=40,
                                      noise_aug=1.0, weight_decay=1e-4)
est.save("results_observer/observer.pt")
```

The checkpoint carries the network, the feature layout and the normalisation,
so `build_estimator("observer", other_dataset, checkpoint=...)` scores another
dataset of the same plant with the benchmark statistics. An observer trained with
`FeatureSpec(kinetic_keys=...)` takes the kinetics of the plant as input and is
run as `est.run(series, kinetics={...})`; see the
[observer page](../theory/observer.md), section 6.

## Command line

```bash
python -m pyadm1ode_estimation.estimation.estimators ukf --series 0 1 --jobs 2 --out run_ukf.npz
python -m pyadm1ode_estimation.estimation.estimators observer --dataset datasets/benchmark \
    --checkpoint results_observer/observer.pt
```

`--split test` (default) or `--split pool`, `--series` for a subset, `--burnin`
for the scoring burn-in (48 steps). `--out` stores the trajectories as object
arrays (`x_hat`, `std`, `time`, `score`, `index`, `label`).

## Parallel runs

`UkfEstimator.run_many(series, jobs=n)` runs the filter in `n` processes
through the persistent worker pool of `filter_tuning.filter_runners`.

## Final comparison

`experiments/final_test/` runs every estimator on the three test sets under five
kinetics conditions. It consists of a resumable scheduler `run.py`, a frozen
`config.yaml` and the scorer `score.py`; results go to `results_final/`. The table
gives the median-over-states NRMSE on days 20 to 60, averaged over 20 series, with
nominal kinetics:

| Estimator | `benchmark` | substrate swap | swap + load rhythm |
| --- | --- | --- | --- |
| UKF, 41 states | 17.3 % | 18.1 % | 18.8 % |
| Fusion UKF + PINN B | 18.1 % | 22.1 % | 22.8 % |
| UKF, A+D core | 18.2 % | 21.1 % | 20.3 % |
| Open loop (model without measurements) | 19.6 % | 21.9 % | 21.6 % |
| PINN B, kinetics-blind | 21.5 % | 29.8 % | 29.7 % |
| PINN v2 | 25.5 % | 30.6 % | 29.8 % |

On `benchmark` the two UKF variants and the fusion are not separated statistically;
on the two shifted sets the UKF with 41 states has the lowest error. With the true
factors of the ten calibratable kinetics the UKF reaches 12.1, 11.9 and 11.4 %. A
calibration of those kinetics from the first 20 days of measurements does not
recover them and makes every estimator worse (UKF: 20.4, 23.4 and 21.5 %), because
it compensates the error of the estimated start state.

!!! note "Dataset of these numbers"
    The table was measured on 2026-11-17 on the benchmark regenerated on
    2026-11-15 with the revised ADM1da model (see `datasets/benchmark/README.md`).
    The UKF was not re-tuned for it; its settings are those found on the benchmark
    published before 2026-11-06. PINN B was retrained. The run of
    2026-10-20 on that earlier benchmark gave 13.5 % for the UKF with 41 states,
    13.1 % for the fusion and 13.8 % for PINN B on `benchmark`. Most of the
    difference comes from the dataset revision of 2026-11-06/08 (varying substrate
    mix, unclipped truth) and not from the model revision: trained on the dataset
    of 2026-11-08, PINN B is as far from its earlier value as on the present one.
