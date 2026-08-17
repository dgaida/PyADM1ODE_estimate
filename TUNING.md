# Running the UKF tuning

Step-by-step guide to tune the noise parameters (Q, R, P0, σ) of the two UKF variants on a
dedicated machine. Each run takes one to three days, produces a reference trajectory for the
benchmark, and can e-mail its report when it finishes.

Run **one variant per machine**. The two searches are independent, and 30 workers already
saturate a 16-core CPU, so running both on one machine just halves the speed of each.

---

## 1. Prepare the machine

```bash
git clone https://github.com/dgaida/PyADM1ODE_estimate.git
cd PyADM1ODE_estimate
pip install -r requirements.txt
pip install -e .
pip install cma                 # only the CMA-ES search needs it
```

Also needs [PyADM1ODE](https://github.com/dgaida/PyADM1ODE).

Check that it works:

```bash
python -c "import cma, pyadm1ode_estimation.estimation.filter_tuning.cmaes_search as cs; print('ok', cma.__version__)"
```

**On Windows, set `PYTHONUTF8=1`.** The progress output contains sigma and arrow characters
that a cp1252 console cannot encode.

**Pin `numpy`, `scipy` and `cma` to the same versions on every machine.** The stiff ODE
solver is version-sensitive enough that results are otherwise not strictly comparable, which
defeats the point of comparing the two variants.

---

## 2. Choose the number of workers

Set `--jobs` to the number of **logical** processors minus two. SMT helps here, because each
worker is single-threaded.

Scaling is strongly sublinear, so a smaller machine costs less than proportionally:

| Workers | 8 | 12 | 16 | 20 | 24 | 30 |
| --- | --- | --- | --- | --- | --- | --- |
| Time factor | 2.6x | 1.9x | 1.5x | 1.3x | 1.15x | 1.0x |

---

## 3. Optional: enable the e-mail report

Configuration is read from the environment only, so no credential ever enters the repository.
Use an **app password**, not your account password.

```powershell
# Windows PowerShell
$env:TUNING_SMTP_HOST     = "smtp.your-provider.tld"
$env:TUNING_SMTP_PORT     = "587"          # 465 for implicit TLS
$env:TUNING_SMTP_USER     = "you@example.org"
$env:TUNING_SMTP_PASSWORD = "<app password>"
```

```bash
# Linux / macOS
export TUNING_SMTP_HOST=smtp.your-provider.tld
export TUNING_SMTP_PORT=587
export TUNING_SMTP_USER=you@example.org
export TUNING_SMTP_PASSWORD='<app password>'
```

If these are unset the run prints `mail skipped` and finishes normally. A failing mail server
can never abort a finished run: every send error is caught, and the report is written to disk
either way.

---

## 4. The four runs

Four configurations are tuned: the two filter variants **without** a laboratory
measurement, and two **with** a weekly FOS/TAC titration. One run per machine.

| # | Variant | Titration | States estimated | Question it answers |
| --- | --- | --- | --- | --- |
| 1 | `full` | off | 41 | the shipped reference, best achievable from the 5 online sensors |
| 2 | `adcore` | off | 18 | does restricting to the observable core beat the full filter |
| 3 | `full` | weekly | 41 | how much does a lab titration buy the full filter |
| 4 | `adcore_vfa` | weekly | 21 | the core plus the three VFAs the titration actually observes |

The common part of every command:

```bash
COMMON="--days 60 --sigma-per 1 --score-per 1         --popsize 8 --gens 8 --patience 3         --val-per 5 --top-k 5 --objective accuracy         --jobs 30 --dataset-dir datasets/benchmark         --out filter_tuning_results --email you@example.org"

RUN="python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter cmaes"
```

### Without a laboratory measurement

```bash
$RUN --variant full   $COMMON          # run 1
$RUN --variant adcore $COMMON          # run 2
```

### With a weekly FOS/TAC titration

```bash
$RUN --variant full       --fostac-every-days 7 $COMMON     # run 3
$RUN --variant adcore_vfa --fostac-every-days 7 $COMMON     # run 4
```

`--fostac-every-days` takes any interval: `7` weekly, `1` daily, `0.0417` hourly. The
dataset stores an **independent** titration for every hour, so every interval is a
statistically valid measurement series rather than an approximation. It is a wet-chemistry
Nordmann titration and not an online sensor, so it only reaches the filter on sampling days.
The rest of the time axis is skipped automatically.

Results never overwrite each other: a run with a titration writes to
`cmaes_<variant>_fostac7d_*` instead of `cmaes_<variant>_*`.

### Why run 4 uses `adcore_vfa` and not `adcore`

FOS is the weighted sum of S_ac, S_pro, S_bu and S_va. The plain A+D core estimates **only
S_ac** of those four, so feeding it a FOS value would push the entire correction of a
four-substance measurement into a single state and bias it. `adcore_vfa` adds S_va, S_bu and
S_pro, which is exactly what the titration observes and precisely what gas, pH and TS cannot
see. That is why those three were excluded in the first place.

S_nh4 stays out although it enters TAC: estimating it was measured to be harmful, because
its process noise leaks into pH through the charge balance. The ammonia part of TAC is
already represented by S_nh3, which is in the core.

Pairing `adcore` with the titration is possible but not recommended, and comparing it against
run 4 is the cleanest way to demonstrate the point empirically.

### What the options mean

| Option | Meaning |
| --- | --- |
| `--days 60` | Search at the **full** horizon. A 20-day window ranks candidates differently (rank correlation only 0.64) and badly misleads the A+D core. |
| `--sigma-per 1 --score-per 1` | Per operating mode: 1 episode to fit σ, 1 to score on. With 4 modes that is **8 episodes per candidate**, all from the train pool. |
| `--popsize 8 --gens 8` | 8 candidates per generation, up to 8 generations, so at most 512 filter runs in the search. |
| `--patience 3` | Stop after 3 generations without improvement, so the tail is only paid for when it earns its keep. |
| `--val-per 5 --top-k 5` | The 5 best candidates are re-measured on **20 validation episodes** that the search never saw. The winner is picked there. |
| `--objective accuracy` | Score is the NRMSE gain, minus a coverage penalty, minus a one-sided guard `2.0 · max(0, 0.5 − AUC)` that only bites if the FOS/TAC ranking is worse than a coin flip. |
| `--dataset-dir` | Lets the final report score the new reference against the true states. |
| `--fostac-every-days` | Interval of the FOS/TAC titration in days. Omitted means no lab measurement. |

Available variants:

| `--variant` | States estimated | Use with a titration? |
| --- | --- | --- |
| `full` | all 41 | optional |
| `adcore` | 18, methanogenesis + charge balance | no, it cannot use FOS properly |
| `adcore_vfa` | 21, the above plus S_va / S_bu / S_pro | **yes, this is what it is for** |

The validation pool stays untouched during the search, so stage 2 is a genuine hold-out and
not, as in earlier runs, the very set being optimised against.

---

## 5. Run it detached

The process must survive a closed terminal.

```powershell
# Windows: pythonw.exe has no console at all
Start-Process -FilePath "C:\path\to\env\pythonw.exe" `
  -ArgumentList "-m","pyadm1ode_estimation.estimation.filter_tuning.tune_filter","cmaes","--variant","full","..." `
  -WorkingDirectory "C:\path\to\PyADM1ODE_estimate" `
  -RedirectStandardOutput "tuning_full.log" -RedirectStandardError "tuning_full.err"
```

```bash
# Linux / macOS
nohup python -m pyadm1ode_estimation.estimation.filter_tuning.tune_filter cmaes \
      --variant full ... > tuning_full.log 2>&1 &
```

Watch progress with `tail -f tuning_full.log`. One line per generation.

---

## 6. What you get

Everything lands in `--out` (default `filter_tuning_results/`):

| File | Contents |
| --- | --- |
| `empirical_noise.json` | Stage 0, Q and R measured from ground truth. Reused automatically on the next run. |
| `cmaes_<variant>_stage1.json` | Every candidate of every generation with all its metrics. Written after each generation, so a crash costs at most one generation. |
| `cmaes_<variant>_stage2.json` | The top-k plus nominal, re-measured on validation at full length. |
| `cmaes_<variant>_reference.npz` | `ukf_x_hat` / `ukf_std` of the winner over the 20 test series. |
| `cmaes_<variant>_summary.json` | Machine-readable summary of the whole run. |
| `cmaes_<variant>_report.txt` | The same as plain text. This is what gets e-mailed. |

**Nothing overwrites the shipped dataset.** Replacing `test.npz` with a new reference stays a
separate, deliberate step.

---

## 7. How long it takes

At 30 workers, with the 25 % buffer that stiff candidates need:

| Stage | full UKF | A+D core |
| --- | --- | --- |
| 1 — search (512 runs) | 39.5 h | 31.5 h |
| 2 — validation (144 runs) | 11.1 h | 8.9 h |
| 3 — test (20 runs) | 1.5 h | 1.2 h |
| **Total per run** | **≈ 52 h** | **≈ 42 h** |

`adcore_vfa` sits between the two, closer to `adcore`: 21 estimated states means 43 sigma
points against 37 for the core and 83 for the full filter. The titration itself costs almost
nothing, it adds two scalar evaluations on sampling days only.

All four runs on two machines, one variant per machine and the two runs of a machine in
sequence: roughly **four days**. At 16 workers about 1.5 times that. Those figures assume a
CPU of similar single-core speed to a Ryzen 9 7950X3D; an older chip adds 30 to 50 % on top,
independent of core count. `--patience 3` typically saves a few hours per run.

## 8. If something goes wrong

**The run dies without a message on Windows.** Almost always a `multiprocessing` problem. Any
script of your own that creates a pool must be guarded by `if __name__ == "__main__":`, and
the workers must not receive functions defined in a notebook. Tell a deadlock from a slow
candidate by looking at worker CPU time: near zero means deadlock.

**More workers make it slower.** The `OMP_NUM_THREADS=1` family of variables is set
automatically inside the workers. If you set them yourself to something larger, the workers
oversubscribe the CPU.

**`UnicodeEncodeError` on Windows.** Set `PYTHONUTF8=1`.

**A candidate fails.** Expected and harmless. The stiff ADM1 cannot be integrated for every Q.
Such candidates come back as `None`, receive the worst fitness and the search continues. The
empirical-Q warm start keeps this rare, in the reference runs all 8 of 8 candidates were
evaluable in every generation.

**Resuming.** There is no built-in resume. `cmaes_<variant>_stage1.json` holds every candidate
evaluated so far, so a crashed search can be continued by hand, and stage 2 and 3 can be run
from it without repeating the search.
