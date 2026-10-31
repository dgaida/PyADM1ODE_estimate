# Der GRU-Observer (PINN B)

[PINN v2](pinn.md) fittet für jede Reihe ein frisches Netz und trainiert es
täglich nach; es zahlt jedes Mal eine volle Optimierung. Der **amortisierte
Observer** (PINN B) verlagert diese Kosten *offline*: Er wird einmal auf vielen
simulierten Szenarien vortrainiert und liest den Zustand danach in einem einzigen
Vorwärtspass aus. Wo PINN v2 pro Fenster *optimiert*, *lernt* der Observer eine
wiederverwendbare Abbildung vom Messstrom auf die Zustandstrajektorie.

Er teilt das Physik-Gerüst von PINN v2 (Positivität über die
Log-Transformation, quasi-stationäre Gasphase), nur Netz und Training
unterscheiden sich.

---

## 1. Architektur: ein GRU-Filter

Der Observer ist ein **rekurrentes** Netz (ein GRU). Seine Eingabe je Schritt ist
die aktuelle Sensormessung plus die bekannte Zufuhr; er verarbeitet die Sequenz
und gibt je Schritt einen Zustand aus:

$$
\underbrace{(y_t,\ u_t)}_{\text{Messungen + Zufuhr}}\ \longrightarrow\
\text{GRU}\ \longrightarrow\ \hat{x}(t)\ \ (41\text{ Zustände}).
$$

Zwei Entwurfsentscheidungen teilt er mit [PINN v2](pinn.md):

* **Positivität / Skalierung:** der Kopf sagt nur die **37 Flüssigzustände** als  
  Log-Abweichung von einer Referenz vorher, $\hat{x}_\text{liq} = x_\text{ref}
  \odot \exp(\text{raw})$,
  und die 4 Gasdrücke folgen aus dem quasi-stationären Gleichgewicht.  
* **Kausalität:** das GRU sieht in Schritt $t$ nur Vergangenheit und Gegenwart,  
  der neueste Schritt ist damit eine echte **online gefilterte** Schätzung: das
  Netz läuft als Streaming-Filter (Abschnitt 3).

Der zentrale Unterschied zu PINN v2: Der Observer ist *auf die Messungen
konditioniert*. Das Netz von PINN v2 ist nur eine Funktion von Zeit und
Fütterung (die Daten kommen über den Loss herein), der Observer nimmt den
Messstrom als **Eingabe**. Genau das lässt ein einziges trainiertes Netz alle
Reihen bedienen, statt es je Fenster neu zu fitten. Wie weit das über die
Trainingsverteilung hinaus trägt, behandelt Abschnitt 5.

---

## 2. Training, Phase 1

Das Vortraining kann eines von zwei Zielen nutzen (oder beide):

**Supervised** (`pretrain_observer`)**:** auf einem **Simulator**-Datensatz, in dem
der wahre 41-Zustand bekannt ist. Der Loss ist ein zustandsweise skalierter
Zustands-MSE $\lVert (\hat{x}-x_\text{true})/s \rVert^2$ (jeder Zustand mit seiner
RMS-Größe normiert), der den *vollen* Zustand lehrt.

!!! tip "Der Engpass ist Overfitting, nicht Kapazitaet"
    Bei ~80 Trainingsserien lernt das Netz sie schnell auswendig: auf dem
    Benchmark gemessen erreicht der Validierungs-Loss sein Minimum um Epoche 60
    von 200, waehrend der Trainings-Loss weiter faellt. `restore_best=True`
    (Standard) zusammen mit `patience=N` gibt die bestvalidierten Gewichte
    zurueck und halbiert die Laufzeit bei identischem Ergebnis.

    Was **nicht** geholfen hat, ebenfalls gemessen: die Fenstermenge zu
    vergroessern. Ueberlappende oder zufaellig platzierte Fenster senken den
    Validierungs-Loss, verschlechtern aber die Testbewertung (Abschnitt 5). Sie
    liefern mehr Gradientenschritte, keine zusaetzliche Information -- der
    Engpass sind die 80 Serien, nicht ihre Zerlegung. Von den Regularisierern
    hat nur `dropout=0.1` zwischen den GRU-Schichten die Testbewertung
    verbessert; kleineres `hidden`, Weight Decay und Eingangsrauschen nicht.

Die Skala `s` wird ausschliesslich aus den **Trainings**-Sequenzen berechnet; sie
vorher ueber den gesamten Satz zu bilden wuerde Validierungsstatistik in das Ziel
einsickern lassen. Wichtige Optionen:

| Option | Wirkung |
| --- | --- |
| `val_dataset` | ein **extern gesplitteter** Validierungssatz. Damit teilen sich mehrere Schaetzer einen Split -- `PinnData.observer_dataset` liefert Train und Val aus demselben stratifizierten Split, den auch die Filter nutzen, und genau das macht Filter und Netz vergleichbar. Intern zu splitten gaebe jedem Modell seinen eigenen Zufallssplit. |
| `burnin` | fuehrende Schritte, die aus dem Loss fallen. Ein kausaler Observer kann den Anfangszustand nicht kennen, dieser Fehler misst also das Unkennbare statt das Modell. |
| `noise_std` | Sensorrauschen je Kanal in Roh-Einheiten, pro Batch neu auf die Messfeatures gezogen. Nur der Messblock wird gestoert, die Zufuhr ist ein bekannter Stelleingang. |
| `restore_best` / `patience` | bestvalidierte Gewichte / Early Stopping. |
| `weight_decay` | L2-Regularisierung. |

**Self-supervised** (`pretrain_observer_selfsup`)**:** auf **reinen Mess**-Fenstern
*ohne* Grundwahrheit (reale Anlagenhistorie oder simulierte Fenster für eine
Ablation). Es nutzt dasselbe Ziel wie die Online-Feinabstimmung: einen Mess-Fit
$\big((h(\hat{x})-y)/\sigma\big)^2$ plus ein raten-skaliertes Physik-Residuum. So
lässt sich der Observer direkt auf der realen Anlage vorbereiten, auf der er
laufen wird.

**Sim→real** (`pretrain_observer_sim2real`)**:** das empfohlene Rezept: erst
supervised auf dem Simulator (die Vollzustands-Struktur lernen), dann
self-supervised auf realer Historie (die Sim-zu-Real-Lücke schließen). Die
Eingabe-Normierung zwischen beiden Stufen teilen.

---

## 3. Training, Phase 2

`finetune_observer` passt den vortrainierten Observer auf dem jüngsten Fenster an
die laufende Anlage an, warm gestartet aus seinen Gewichten mit kleiner Lernrate.
Es ist **self-supervised** (online sind nur die Messungen bekannt): Mess-Fit +
diskretes raten-skaliertes Physik-Residuum, mit einem optionalen Anker
(`lambda_anchor`), der die Trajektorie nahe der eingefrorenen vortrainierten
Vorhersage hält (eine Vertrauensregion gegen Rauschen). Wie das Vortraining ist
es monoton-sicher.

`SlidingWindowObserver` verpackt das zu einem **kontinuierlichen Online-Schätzer**:

```python
swo = SlidingWindowObserver(
    observer, obs_model, feat_mean, feat_std,
    window_hours=48, finetune_every=24,
)
for meas, feed in live_sensor_stream:   # meas = [Q_gas, Q_ch4, pH]
    est = swo.step(meas, feed)          # est.state = aktueller 41-Zustand, est.std
```

Er hält ein gleitendes Fenster jüngster Messwerte. Bei jedem neuen Sample gibt er
die aktuelle Schätzung zurück und feintunt nach Zeitplan self-supervised auf dem
Fenster. Zwei Betriebsdetails:

* **Zufuhr-bewusste Physik:** abseits des nominalen Betriebspunkts stimmt die  
  nominale Zufuhr der vortrainierten Parameter nicht, daher skaliert die
  Feinabstimmung `q_ad` (die gesamte Zulaufrate in den Fermenter [m³/d])
  auf die tatsächliche mittlere Zufuhr des Fensters.  
* **Fehlende Sensoren:** ein geblockter/offline Messwert kommt als `NaN`. Das  
  rekurrente Netz kann `NaN` nicht verarbeiten, daher wird er eingabeseitig auf
  den normierten Mittelwert (0) abgebildet und zielseitig aus dem Loss maskiert.

**Unsicherheit** ist optionaler MC-Dropout (`mc_samples > 1` mit einem Dropout-trainierten Netz).

---

## 4. Stärken und Grenzen

**Stärken:** Nahezu sofortige Inferenz nach den einmaligen Offline-Kosten
(deutlich unter einer Sekunde je 60-Tage-Reihe, gegen Minuten für den UKF und
einige zehn Minuten für PINN v2); lernt die *volle* Zustandsstruktur aus dem
Simulator, ist also auch bei unbeobachteten Zuständen stark; ein Netz bedient
alle Reihen. Auf dem Benchmark-Testsatz ist er der einzige gelernte Schätzer
nahe am UKF (20 Reihen, 60 Tage, Median-NRMSE über die Zustände nach 48 h
Burn-in):

| Schätzer | Fehler |
| --- | --- |
| UKF, 41 Zustände (Referenz) | 14,4 % |
| PINN B wie veröffentlicht (ohne Dropout) | 15,1 % |
| PINN B mit `dropout=0.1` (neuer Standard, Mittel über 3 Seeds) | 14,7 % |
| PINN v2, kausal, 10-Tage-Fenster | 18,7 % |

PINN B schlägt den UKF im Modus `stable_high` (12,9 % gegen 19,1 %), dem
chronisch überlasteten Regime, in dem die Nominalkinetik des UKF am weitesten
danebenliegt.

**Grenzen:** Er ist an seine Trainingsverteilung gebunden: Auf verschobenen
Anlagen verliert er mehr als der UKF (Abschnitt 5). Er trägt eine
**Sim-zu-Real-Lücke**, die nur die self-supervised Anpassung schließt, und die
schlecht konditionierte Biogas-Abbildung begrenzt weiterhin self-supervised
Kaltstarts.

Zur Kombination mit dem UKF siehe die [UKF ↔ PINN-Fusion](fusion.md).

---

## 5. Generalisierung auf verschobene Anlagen

Zwei Testsätze verschieben die Anlage, keiner wird im Training genutzt: Einer
ersetzt Maissilage bei gleicher organischer Last durch Corn-Cob-Mix, der andere
legt zusätzlich einen 3-Tage-Lastrhythmus darüber. Beide wurden nur für diese
Studie erzeugt und liegen nicht im Repository. Bewertet wird der Median-NRMSE über die Zustände nach 48 h Burn-in; die Zeilen
für PINN B sind Mittel über drei Seeds (`experiments/observer_generalisation/`).

| Schätzer | Benchmark | Substratwechsel | Wechsel + Rhythmus |
| --- | --- | --- | --- |
| UKF, 41 Zustände | 14,4 % | 17,1 % | 17,7 % |
| UKF, Säure- und Fermenterkern | 17,5 % | 20,1 % | 19,6 % |
| PINN v2, kausal | 18,7 % | 23,0 % | 21,9 % |
| PINN B, veröffentlichtes Rezept | 15,2 % | 19,3 % | 20,2 % |
| PINN B, `dropout=0.1` (Standard) | 14,7 % | 18,9 % | 19,5 % |
| PINN B, Ensemble aus 5 Seeds mit Dropout | **14,1 %** | 18,5 % | 19,2 % |

PINN B verliert auf einer verschobenen Anlage 4 bis 5 Punkte, der UKF etwa 3. Die
für den Wechselsatz veröffentlichten 18,4 % waren ein günstiger Seed; drei Seeds
desselben Rezepts liegen im Mittel bei 19,3 %, bei einer Streuung von etwa
±0,5 Punkten je Seed.

**Was die Lücke nicht schließt.** Eingangsrauschen, Weight Decay, ein kleineres
Netz, überlappende, zufällige oder kürzere Trainingsfenster und ihre
Kombinationen lassen die Lücke bei 3,5 bis 5 Punkten. Dropout und das Ensemble
senken alle drei Werte gemeinsam; die Lücke bleibt.

**Substratbewusste Eingabe.** `FeatureSpec(organic_load=True, use_feed=False)`
ersetzt die Volumina je Substrat durch die Zulauffrachten je Komponentengruppe,
berechnet mit der Substratcharakterisierung der jeweiligen Anlage. Innerhalb der
Trainingsverteilung ist das der beste Observer (14,0 % mit Dropout, unter dem
UKF), er generalisiert aber schlechter: 19,2 % und 19,4 % auf den verschobenen
Sätzen.

**Verschobene Daten im Training.** 12 der 20 Corn-Cob-Reihen kommen in den
Trainingspool, bewertet wird auf den anderen 8 (`coverage.py`, je drei Seeds):

| Eingabe | Trainingspool | Benchmark | Zurückgehaltene Corn-Cob-Reihen | Wechsel + Rhythmus |
| --- | --- | --- | --- | --- |
| Rohfütterung | Benchmark | 14,7 % | 17,8 % | 19,5 % |
| Rohfütterung | + 12 Corn-Cob-Reihen | 15,1 % | 17,8 % | 19,6 % |
| organische Fracht | Benchmark | 14,0 % | 19,6 % | 19,4 % |
| organische Fracht | + 12 Corn-Cob-Reihen | 15,5 % | 18,4 % | 20,0 % |

Mit der Rohfütterung ändern die zusätzlichen Reihen nichts: Spalte 0 bedeutet in
100 Reihen Mais und in 12 Corn-Cob-Mix, das Netz kann die Anlagen nicht
unterscheiden. Mit der Frachteingabe helfen sie auf den zurückgehaltenen
Corn-Cob-Reihen, aber weniger, als die Rohfütterung ohne sie erreicht, und sie
kosten Genauigkeit auf der ursprünglichen Anlage.

**Wo der Fehler wächst.** Auf den verschobenen Sätzen steigt der Fehler von
PINN B am stärksten bei den schnellen Partikulärfraktionen, den
Propionatabbauern und dem Anionenpool — Zuständen, die der Substratwechsel
verschiebt und die fünf Sensoren kaum sehen. Beim UKF steigt vor allem der
Acetatfehler. Der UKF bekommt die Substratmischung über sein Modell mitgeteilt,
der Observer muss die Zusammensetzung aus dem Sensorverlauf erschließen und
bildet einen ungewohnten Verlauf auf die Zustände ab, die er aus dem Training
kennt.

**Folgerung.** PINN B ist innerhalb seiner Trainingsverteilung ein starker
Schätzer, und die günstigen Gewinne dort sind Dropout und ein Seed-Ensemble.
Außerhalb ist der UKF robuster. Für eine neue Anlage muss der Observer auf
Simulationen genau dieser Anlage trainiert werden; bei abweichender Kinetik ist
der UKF mit kalibriertem Modell der robustere Weg.

---

## 6. Kinetik als Eingang

PINN B sieht nie, welche Kinetik eine Reihe hat, kalibrierte Kinetik kann ihm
also nicht helfen. `FeatureSpec(kinetic_keys=...)` ändert das: Je genannter
Kinetikkonstante kommt eine Eingangsspalte mit dem Logarithmus ihres Faktors
gegenüber dem Nominalwert hinzu, konstant über die ganze Reihe.

* **Im Training** stehen dort die wahren Faktoren des Generators für jede  
  simulierte Reihe. `ObserverEstimator.train(..., kinetic_noise=0.07)` stört sie
  mit einem Zufallswert je Trainingssequenz, damit das Netz lernt, Kinetik zu
  nutzen, die nur auf Kalibriergenauigkeit bekannt ist.  
* **In der Anwendung** wird die Kinetik vorgegeben:  
  `est.run(series, kinetics={"k_m_ac": 0.8, ...})`. Nicht genannte Schlüssel
  gelten als nominal, `{}` bedeutet nominale Kinetik. Ein kinetikbewusster
  Observer verweigert den Lauf ohne ausdrücklich übergebene Kinetik, damit die
  Generatorwahrheit einer Benchmark-Reihe nie unbemerkt hineinrutscht.

Eingänge sind die zehn Konstanten, die die Sensoren über ein 20-Tage-Fenster am
stärksten bewegen, gescreent auf acht Trainingsreihen
(`experiments/final_test/screen_kinetics.py`): `k_dis_PS`, `k_m_ac`, `k_m_pro`, `k_dis_PF`, `k_hyd_ch`, `k_dec_ac`, `k_dec_su`, `K_S_ac`, `k_dec_pro`, `K_S_pro`.

Vergleich auf den 20 Validierungsreihen, beide als Ensemble aus 5 Seeds mit
gleichem Rezept und gleichen Seeds, Median-NRMSE nach 48 h (die Testsätze bleiben
für den finalen Test unberührt;
`experiments/observer_generalisation/kinetic_inputs.py`):

| Vorgegebene Kinetik | Kinetikblind | 10 Kinetik-Eingänge |
| --- | --- | --- |
| keine, Nominalwerte | 14,7 % | 17,2 % |
| wahre Faktoren | 14,7 % | 12,4 % |
| wahr × exp(N(0; 0,07)) | 14,7 % | 12,9 % |
| wahr × exp(N(0; 0,15)) | 14,7 % | 14,0 % |

Zur Einordnung: Der UKF mit nominaler Kinetik erreicht auf denselben Reihen
16,3 %. Fünf statt zehn Eingänge ergeben 13,2 % bei σ = 0,07. Mit allen 26
Konstanten exakt bekannt, was keine Kalibrierung schafft, erreicht der Observer
7,7 %.

* **Der größte Teil des Observer-Fehlers ist unbekannte Kinetik.** Kennt er sie  
  ganz, halbiert sich der Fehler.  
* **Mit kalibrierter Kinetik ist er deutlich besser.** Bei der Genauigkeit, die  
  die Kinetikkalibrierungsstudie erreicht, σ ≈ 0,07, ist der kinetikbewusste
  Observer 1,9 Punkte besser (p = 0,005). Bei σ = 0,15 schrumpft der Gewinn auf
  0,8 Punkte und ist nicht mehr signifikant.  
* **Ohne Kalibrierung ist er schlechter.** Bekommt er „nominal“, vertraut er  
  diesem Eingang, statt die Kinetik wie der kinetikblinde Observer aus dem
  Sensorverlauf zu erschließen, und verliert 2,5 Punkte.

Die Entscheidungsregel stand vor dem Lauf fest: besser mit Kalibrierung bei
σ = 0,07 und ohne Kalibrierung höchstens 0,3 Punkte schlechter. Der zweite Teil
ist verletzt, der kinetikbewusste Observer ersetzt den kinetikblinden also
nicht. Er wird mit zehn Eingängen und `kinetic_noise=0.07` eingesetzt, sobald
kalibrierte Kinetik vorliegt; sonst der kinetikblinde. Die Checkpoints werden
nicht mitgeliefert; `kinetic_inputs.py` trainiert sie neu.

---

## Quelldateien

* `pyadm1ode_estimation/estimation/observer/model.py` — `Adm1Observer`  
* `pyadm1ode_estimation/estimation/observer/datasets.py` — `generate_observer_dataset`, `ObserverDataset`, `MeasurementDataset`  
* `pyadm1ode_estimation/estimation/observer/train.py` — `pretrain_observer`, `pretrain_observer_selfsup`, `pretrain_observer_sim2real`, `finetune_observer`  
* `pyadm1ode_estimation/estimation/observer/online.py` — `SlidingWindowObserver`  
* `pyadm1ode_estimation/estimation/observer/adapter.py` — `PinnData`, `FeatureSpec` (inkl. `organic_load`, `kinetic_keys`), `kinetic_log_factors`  
* `pyadm1ode_estimation/estimation/estimators.py` — `ObserverEstimator` (trainieren, speichern, laden, `for_dataset`)  
* `experiments/observer_generalisation/` — Rezept-Sweep, Seed-Ensemble, Abdeckungstest, Kinetik-Eingänge  

## API-Referenz

::: pyadm1ode_estimation.estimation.observer.model.Adm1Observer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.observer.online.SlidingWindowObserver
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
