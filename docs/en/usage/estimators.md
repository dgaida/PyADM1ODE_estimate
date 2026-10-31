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
| Fusion UKF + PINN B | 13.1 % | 15.2 % | 15.5 % |
| UKF, 41 states | 13.5 % | 16.1 % | 16.2 % |
| PINN B, kinetics-blind | 13.8 % | 17.5 % | 18.5 % |
| UKF, A+D core | 16.5 % | 18.2 % | 18.0 % |
| PINN v2 | 18.4 % | 21.7 % | 20.5 % |

With the true factors of the ten calibratable kinetics the UKF reaches 6.4, 7.8 and
8.4 %. A calibration of those kinetics from the first 20 days of measurements does
not recover them and makes every estimator worse, because it compensates the error
of the estimated start state.
