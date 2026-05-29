# Twin-Experimente

End-to-End-Validierung des UKF gegen eine **bekannte Wahrheit**: man
simuliert die Realität (truth plant) und gibt dem Filter nur verrauschte
Messungen daraus. Der Vergleich zwischen Filter-Schätzung und Wahrheit
sagt, ob der Filter gut kalibriert ist *bevor* er auf echte Anlagendaten
losgelassen wird.

## Grundprinzip

```text
                ┌─────────────────────────────────────────┐
                │   TRUTH SIDE (= "die echte Anlage")     │
                │                                         │
   build →      │ truth_plant (ADM1 ODE)                  │
   warm-up →    │   │                                     │
                │   ↓ propagate_truth: ODE-Schritt dt=1h  │
                │   x_truth[k+1]                          │
                │   │                                     │
                │   ↓ h(x): clean truth observation       │
                │   obs_clean[k+1]                        │
                │   │                                     │
                │   ↓ truth_sensors[name].read(t)         │
                │     ↓ drift + lag + noise + sampling    │
                │   obs_noisy[k+1]  ──────────────────────│─┐
                └─────────────────────────────────────────┘ │
                                                            │
                ┌─────────────────────────────────────────┐ │
                │   FILTER SIDE (UKF)                     │ │
                │                                         │ │
   deepcopy →   │ filter_plant (= truth_plant zu t=0)     │ │
                │   │                                     │ │
                │   ↓ ukf.predict(dt=1h):                 │ │
                │     - 89 Sigma-Punkte                   │ │
                │     - jeder Punkt: filter_plant.step()  │ │
                │     - gewichteter Mittelwert            │ │
                │   x̂[k+1] (predicted), P_pred            │ │
                │   │                                     │ │
                │   ↓ ukf.update(y=obs_noisy[k+1])  ←─────│─┘
                │     - 89 Sigma-Punkte durch h(x)        │
                │     - Innovation y − ŷ                  │
                │     - Kalman-Gain → Posterior           │
                │   x̂[k+1] (posterior), P_post            │
                └─────────────────────────────────────────┘
```

In der **Realität existiert kein truth_plant**, die Sensoren liefern
direkt die Messungen. Der filter_plant im Twin entspricht dem
*internen Modell* des Schätzers.

## Das vorgefertigte Skript

Im Repo liegt `examples/run_twin_experiment.py`, das den vollen Workflow
für die Multi-Stage-Beispielanlage durchspielt:

```bash
python examples/run_twin_experiment.py \
    --warmup-days 30 \
    --duration-days 5 \
    --initial-perturbation-relative 0.05 \
    --substrate-noise-relative 0.10
```

CLI-Parameter:

| Parameter | Default | Bedeutung |
|---|---|---|
| `--warmup-days` | 30 | Vor-Simulation, bevor der Filter startet, damit die Anlage in einem Quasi-Steady-State ist |
| `--duration-days` | 5 | Länge des eigentlichen UKF-Laufs |
| `--dt-hours` | 1.0 | Filter-Schrittweite |
| `--initial-perturbation-relative` | 0.05 | Relative Gauß-Perturbation des Filter-Initial-State |
| `--substrate-noise-relative` | 0.10 | Per-Schritt-Rauschen auf der Substrat-Dosierung |
| `--plot-from-day` | 0.0 | Burn-in für die Plots (Diagnostik wird trotzdem über vollen Lauf berechnet) |

## Was passiert intern

1. **Truth-Plant bauen** (`build_multi_stage_plant`) und für 30 Tage
   warmlaufen lassen (`plant.simulate()`). Dadurch erreicht die ODE
   einen quasi-stationären Operating Point.
2. **Deepcopy** der gewärmten Plant für den Filter. Garantiert
   bit-identisches Modell zwischen Truth und Filter zu `t=0`.
3. **Truth-Trajektorie** propagieren (`_propagate_truth_with_substrate_noise`)
   mit Per-Schritt-Rauschen auf den Substrat-Inputs (Operator-Lieferung
   ist nie exakt).
4. **Truth-Sensoren** (`build_truth_sensors`) erzeugen das verrauschte
   Mess-Signal. Diese benutzen die `PhysicalSensor`-Klassen aus
   PyADM1ODE für realistische Drift, Response-Lag und Sampling
   (siehe `pyadm1ode_estimation.estimation.sensors`).
5. **UKF** mit `ukf.reset(x_truth0 + perturbation, P0)` initialisieren.
6. **Filter-Loop** über alle Mess-Zeitstempel.
7. **Plots schreiben** in `output/twin_experiment/`.

## Erzeugte Plots

Pro Run werden 6 Plots in `output/twin_experiment/` abgelegt:

* **`trajectories_strong.png`** — 6 strong-observable States
  (S_ac, S_ch4, X_ac, S_hco3, p_gas_ch4, pTOTAL) mit truth, `x̂`
  und ±2σ-Band.
* **`trajectories_weak.png`** — 6 weak / open-loop States + 1
  Substrat-Input.
* **`observations.png`** — alle 6 Mess-Channels mit clean truth,
  noisy measurement und Filter-Vorhersage ŷ.
* **`production_estimate.png`** — der eigentliche
  Production-Plot:
    * Truth Q_gas / Q_ch4 (schwarz)
    * Raw sensor (rote X) — was die Messpunkte liefern
    * Sensor-smoothed (rot, durchgezogen) — gleitender Mittelwert
    * h(x̂) (grün) — deterministische Modell-Re-evaluation am
      Filter-Posterior (Jensen-bias-frei, eine einzige Plant-Step-
      Auswertung statt 89 Sigma-Punkte)
    * `±1σ`-Band aus der UKF-internen `y_std`
    * Kumulative Produktion mit End-Error annotiert
* **`nis.png`** — NIS-Zeitreihe in log-Skala mit Erwartungswert
  als Referenz-Linie.
* **`coverage_summary.png`** — Per-Quality-Block 2σ-Coverage als
  Bar-Chart mit Sollwerten (80 % / 40 % / 20 %).

## Was die Resultate bedeuten

Aus einem typischen 30+5-Tage-Lauf (10 % Substrat-Rauschen,
5 % Initial-Perturbation):

| Block | Coverage | Status |
|---|---|---|
| methanogenesis | 86.7 % | strong ✓ |
| charge_balance | 94.3 % | strong ✓ |
| acidogenesis_substrates | 99.3 % | medium ✓ |
| acidogenesis_biomass | 99.8 % | weak ✓ |
| hydrolysis_sums | 100 % | weak ✓ |
| disintegration_split | 85.7 % | strukturell limitiert |
| nitrogen | 99.2 % | open-loop ✓ |
| inerts | 100 % | open-loop ✓ |
| fa_block | 100 % | open-loop ✓ |

`disintegration_split` bleibt strukturell unter 100 %, weil die
PS/PF-Splits aus Prozessmessungen prinzipiell nicht trennbar sind
(siehe [Observability-Doku](../observability/sensor_state_dependencies.md)).

**NIS-Mean ≈ 8.9** bei 6 Channels liegt im Idealfenster `[3, 12]`. Der
Filter ist gut kalibriert.

## Production-Plot: Sensor vs. Modell

Im Production-Plot zeigt sich ein wichtiger praktischer Befund:

| Quelle | Cumulative End-Error |
|---|---|
| Sensor (smoothed) | ≈ −0.5 % |
| h(x̂) (Modell-Re-evaluation) | ≈ −3.5 % |

**Der Sensor schlägt die modell-basierte Schätzung für direkt gemessene
Größen.** Q_gas hat 0.16 % relatives Rauschen — kein modell-basierter
Schätzer kann das unterbieten, weil das Modell über 44 States aggregiert
und kleine Bias-Beiträge sammelt.

**Praktische Hierarchie für Operator-Reporting:**

| Größe | Beste Quelle |
|---|---|
| Q_gas / Q_ch4 (direkt gemessen) | Sensor + Glättung |
| pH (direkt gemessen) | Sensor + Glättung |
| Substrat-Dosierung (direkt gemessen) | Sensor + Glättung oder Operator-Logbuch |
| **S_ac, X_ac, Biomasse, Säure-Base-Spezies** (nicht gemessen) | **UKF x̂** |

Der UKF gibt also nicht "bessere Q_gas-Messung", er gibt dir **die
nicht-gemessenen 35-40 Zustände aus den gemessenen 5-6**. Das ist der
eigentliche Nutzen.

## Acceptance-Kriterien

Für einen produktiven UKF gelten als Faustregel:

* **Strong-observable Blöcke** (methanogenesis, charge_balance,
  acidogenesis_substrates): 2σ-Coverage ≥ 80 %
* **Weak / OU-Blöcke**: 2σ-Coverage ≥ 40 %
* **Open-Loop**: 2σ-Coverage ≥ 20 %
* **NIS-Mean** über mehrere Tage in `[0.5·n, 2.0·n]`

Bei einem Acceptance-Test sollten alle erfüllt sein. Wenn z.B. der
NIS-Mean außerhalb des Bereichs liegt, prüfe:

* Sind die `noise_std`-Werte realistisch zu den Sensoren?
* Reicht der Warm-up, damit die Plant im Quasi-Steady-State ist?
* Ist die Initial-Perturbation `--initial-perturbation-relative`
  realistisch zur Initial-Kovarianz `P0`?
* Ist `dt_hours` klein genug für die ADM1-Nichtlinearität?
