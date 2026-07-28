# Reproduce the UKF report

Minimal script set to reproduce the quantitative results of the UKF report
(`reports/ukf_comparison.qd`): the full-41 vs. A+D-core comparison (Tests A/B)
and the model-error sweep. Plot-only scripts live one level up in
`experiments/` and are **not** part of this folder.

## Contents

| Script | Role |
| --- | --- |
| `report_compare.py` | Main driver. One code path for every reported number; writes `reports/results/<tag>.npz` (+ `<tag>_meta.txt` provenance). Candidates: `full`, `adcore`, `cukf`, `adcore_ki`, `openloop`. |
| `run_twin_experiment.py` | Dependency of `report_compare.py` (sensor schedule, truth propagation with substrate noise, per-stage gas evaluation). |
| `aggregate_sweep.py` | Aggregates the model-error sweep `.npz` into the σ-trend table (Table 6.1). |
| `openloop_figure.py` | No-filter reference figures: truth-vs-model gas/methane overlay (`reports/figures/openloop.png`) plus one per-block ADM1 state-trajectory figure per feed (`openloop_states_<feed>_<block>.png`). |

The `openloop` candidate is the **no-filter baseline**: the imperfect ADM1 model
free-runs from the same perturbed prior, driven by the same known feed and the
same time-growing kinetic model error, but never assimilates a sensor. Its gap to
the truth is the raw model drift the filters have to correct (NIS / coverage are
undefined for it).

All scripts resolve paths relative to the repository root, so run them from
there. Results land in `reports/results/`, figures in `reports/figures/`
(both git-ignored).

## Run

```bash
# Test A (no substrate change) and Test B (with substrate change)
python experiments/twin_experiment/report_compare.py --tag A_steady --feed none   --sigma 0.25 \
    --candidates full,adcore,cukf --duration 60 --dt 6
python experiments/twin_experiment/report_compare.py --tag B_feed   --feed change  --sigma 0.25 \
    --candidates full,adcore,adcore_ki --duration 60 --dt 6

# Real-vs-estimated trajectories (add --save-traj)
python experiments/twin_experiment/report_compare.py --tag A_obs      --feed none   --sigma 0.25 --candidates full,adcore --duration 60 --dt 6 --save-traj
python experiments/twin_experiment/report_compare.py --tag A_obs_cukf --feed none   --sigma 0.25 --candidates cukf        --duration 60 --dt 6 --save-traj
python experiments/twin_experiment/report_compare.py --tag B_obs      --feed change  --sigma 0.25 --candidates full,adcore --duration 60 --dt 6 --save-traj
python experiments/twin_experiment/report_compare.py --tag B_obs_ki   --feed change  --sigma 0.25 --candidates adcore_ki   --duration 60 --dt 6 --save-traj

# Model-error sweep + aggregation (Table 6.1)
for s in 0.10 0.25 0.40 0.55; do \
  python experiments/twin_experiment/report_compare.py --tag sweep_s$s --feed none --sigma $s \
      --candidates full,adcore --duration 30 --dt 6; done
python experiments/twin_experiment/aggregate_sweep.py sweep

# No-filter reference: open-loop ADM1 vs. A+D/full, plus the truth-vs-model figure
python experiments/twin_experiment/report_compare.py --tag OL_A --feed none   --sigma 0.25 \
    --candidates openloop,adcore,full --duration 60 --dt 6
python experiments/twin_experiment/report_compare.py --tag OL_B --feed change --sigma 0.25 \
    --candidates openloop,adcore,full --duration 60 --dt 6
python experiments/twin_experiment/openloop_figure.py   # -> reports/figures/openloop.png

# Truth vs. A+D-UKF vs. open-loop trajectory comparison (report figs 7.2 / 7.3)
python experiments/twin_experiment/report_compare.py --tag CMP_A --feed none   --sigma 0.25 \
    --candidates adcore,openloop --duration 60 --dt 6 --save-traj
python experiments/twin_experiment/report_compare.py --tag CMP_B --feed change --sigma 0.25 \
    --candidates adcore,openloop --duration 60 --dt 6 --save-traj
python reports/obs_plots.py CMP_A CMP_B
```

## Figures (not in this folder)

The plot scripts in `experiments/` consume the `.npz` produced above:

```bash
python experiments/replot_blocks.py A_steady B_feed          # per-block NRMSE (Fig. 4.1, 5.1)
python experiments/obs_plots.py A_obs,A_obs_cukf B_obs,B_obs_ki  # real-vs-estimated (Fig. 4.2, 5.2)
python experiments/make_method_figure.py                     # predict/update cycle (Fig. 2.1)
```
