# Die Schätzer-Registry

Im Paket leben drei Schätzerfamilien, jede mit eigenem Trainingscode: die
rekursiven Filter (`filters/`, getunt in `filter_tuning/`), das
Kollokations-PINN (`pinn/`) und der amortisierte GRU-Observer (`observer/`).
Die Registry in `estimation/estimators.py` gibt ihnen **eine Schnittstelle**,
sodass ein Experiment den Schätzer tauschen kann, ohne die Datenschicht
anzufassen.

```python
from pyadm1ode_estimation.estimation.estimators import build_estimator, score
from pyadm1ode_estimation.estimation.filter_tuning import get_dataset

ds = get_dataset("benchmark")                 # oder ein Datensatz-Verzeichnis
est = build_estimator("ukf", ds)              # "ukf" | "pinn" | "observer"
traj = est.run(ds.test[0])                    # TrajectoryEstimate: time, x_hat, std
score(traj, ds.test[0], burnin_steps=48)      # Median-NRMSE über die Zustände
```

Jeder Schätzer nimmt das gemeinsame `Series` aus `filter_tuning.datasets` und
gibt ein `TrajectoryEstimate` zurück (`time (T,)`, `x_hat (T, 41)`,
`std (T, 41)`; Nullen, wenn die Methode keine Unsicherheit liefert). Die
Benchmark-Metrik, der Fusionscode und die Vergleichsplots sehen damit eine
gemeinsame Währung.

## Die Schätzer

| Name | Klasse | Was läuft | Optionen |
| --- | --- | --- | --- |
| `ukf` | `UkfEstimator` | 41-Zustands-SR-UKF (`variant="full"`) oder der Säure- und Fermenterkern (`"adcore"`) mit dem mitgelieferten CMA-ES-Referenz-`theta` aus `estimation/reference/` | `variant`, `theta`, `fostac_every_days` |
| `pinn` | `PinnEstimator` | [PINN v2](../theory/pinn.md) im kausalen gleitenden Fenster, beste Struktur der Studie | `arch`, `cfg`, `roll`, `bridge="adm1" \| "hold"` |
| `observer` | `ObserverEstimator` | der trainierte [GRU-Observer](../theory/observer.md) (PINN B) in einem Vorwärtspass | `checkpoint` (Standard: der mitgelieferte `estimation/reference/observer.pt`) |

Kosten je 60-Tage-Reihe auf der CPU: Der UKF braucht Minuten (30 Tage Warm-up
plus 1440 Sigma-Punkt-Schritte), das PINN einige zehn Minuten (60
Nachtrainings), der Observer deutlich unter einer Sekunde.

## Training der gelernten Schätzer

Der Observer wird aus einem `PinnData`-Wrapper trainiert, der den
stratifizierten Split und die nur auf dem Training gefittete Normierung festlegt:

```python
from pyadm1ode_estimation.estimation.estimators import ObserverEstimator
from pyadm1ode_estimation.estimation.observer import PinnData

data = PinnData.build("benchmark", val_frac=0.2, seed=0)
est, result = ObserverEstimator.train(data, window=336, epochs=200, patience=40,
                                      noise_aug=1.0, weight_decay=1e-4)
est.save("results_observer/observer.pt")
```

Der Checkpoint enthält Netz, Merkmalslayout und Normierung, sodass
`build_estimator("observer", anderer_datensatz, checkpoint=...)` einen anderen
Datensatz derselben Anlage mit den Benchmark-Statistiken bewertet. Ein mit
`FeatureSpec(kinetic_keys=...)` trainierter Observer nimmt die Kinetik der Anlage
als Eingang und läuft als `est.run(series, kinetics={...})`; siehe die
[Observer-Seite](../theory/observer.md), Abschnitt 6.

## Kommandozeile

```bash
python -m pyadm1ode_estimation.estimation.estimators ukf --series 0 1 --jobs 2 --out run_ukf.npz
python -m pyadm1ode_estimation.estimation.estimators observer --dataset datasets/benchmark \
    --checkpoint results_observer/observer.pt
```

`--split test` (Standard) oder `--split pool`, `--series` für eine Teilmenge,
`--burnin` für das Burn-in der Bewertung (48 Schritte). `--out` speichert die
Trajektorien als Objekt-Arrays (`x_hat`, `std`, `time`, `score`, `index`,
`label`).

## Parallele Läufe

`UkfEstimator.run_many(series, jobs=n)` lässt den Filter in `n` Prozessen über
den persistenten Worker-Pool aus `filter_tuning.filter_runners` laufen.

## Finaler Vergleich

`experiments/final_test/` lässt jeden Schätzer auf den drei Testsätzen unter fünf
Kinetik-Bedingungen laufen. Dazu gehören ein wiederaufnehmbarer Scheduler `run.py`,
eine eingefrorene `config.yaml` und die Auswertung `score.py`; die Ergebnisse liegen
in `results_final/`. Die Tabelle zeigt den Median-NRMSE über die Zustände auf Tag 20
bis 60, gemittelt über 20 Reihen, mit nominaler Kinetik:

| Schätzer | `benchmark` | Substratwechsel | Wechsel + Lastrhythmus |
| --- | --- | --- | --- |
| Fusion UKF + PINN B | 13,1 % | 15,2 % | 15,5 % |
| UKF, 41 Zustände | 13,5 % | 16,1 % | 16,2 % |
| PINN B, kinetikblind | 13,8 % | 17,5 % | 18,5 % |
| UKF, A+D-Kern | 16,5 % | 18,2 % | 18,0 % |
| PINN v2 | 18,4 % | 21,7 % | 20,5 % |

Mit den wahren Faktoren der zehn kalibrierbaren Kinetiken erreicht der UKF 6,4, 7,8
und 8,4 %. Eine Kalibrierung dieser Kinetiken aus den ersten 20 Tagen Messungen
findet sie nicht wieder und macht jeden Schätzer schlechter, weil sie den Fehler des
geschätzten Startzustands ausgleicht.
