# Der vortrainierte Observer

Das [per-Window-PINN](pinn.md) fittet für jedes Fenster ein frisches Netz, aber
es zahlt jedes Mal eine volle Optimierung (Sekunden bis Minuten). Der
**amortisierte Observer** verlagert diese Kosten *offline*: Er wird einmal auf
vielen simulierten Szenarien vortrainiert und liest den Zustand danach in einem
einzigen Vorwärtspass aus. Wo der Smoother pro Fenster *optimiert*, *lernt* der
Observer eine wiederverwendbare Abbildung vom Messstrom auf die
Zustandstrajektorie.

Er teilt das Physik-Gerüst des Smoothers (Positivität über die Log-Transformation,
quasi-stationäre Gasphase), nur Netz und Training unterscheiden sich.

---

## 1. Architektur: ein GRU-Filter

Der Observer ist ein **rekurrentes** Netz (ein GRU). Seine Eingabe je Schritt ist
die aktuelle Sensormessung plus die bekannte Zufuhr; er verarbeitet die Sequenz
und gibt je Schritt einen Zustand aus:

$$
\underbrace{(y_t,\ u_t)}_{\text{Messungen + Zufuhr}}\ \longrightarrow\
\text{GRU}\ \longrightarrow\ \hat{x}(t)\ \ (41\text{ Zustände}).
$$

Zwei Dinge werden unverändert vom [Smoother](pinn.md) übernommen:

* **Positivität / Skalierung:** der Kopf sagt nur die **37 Flüssigzustände** als  
  Log-Abweichung von einer Referenz vorher, $\hat{x}_\text{liq} = x_\text{ref}
  \odot \exp(\text{raw})$,
  und die 4 Gasdrücke folgen aus dem quasi-stationären Gleichgewicht.  
* **Kausalität:** das GRU sieht in Schritt $t$ nur Vergangenheit und Gegenwart,  
  der neueste Schritt ist damit eine echte **online gefilterte** Schätzung: das
  Netz läuft als Streaming-Filter (Abschnitt 3).

Der zentrale Unterschied zum Smoother: Der Observer ist *auf die Messungen
konditioniert*. Das Netz des Smoothers ist nur eine Funktion der Zeit (die Daten
kommen über den Loss herein), der Observer nimmt den Messstrom als **Eingabe**.
Genau das lässt ein einziges trainiertes Netz über Betriebspunkte hinweg
generalisieren, statt es je Fenster neu zu fitten.

---

## 2. Training, Phase 1

Das Vortraining kann eines von zwei Zielen nutzen (oder beide):

**Supervised** (`pretrain_observer`)**:** auf einem **Simulator**-Datensatz, in dem
der wahre 41-Zustand bekannt ist. Der Loss ist ein zustandsweise skalierter
Zustands-MSE $\lVert (\hat{x}-x_\text{true})/s \rVert^2$ (jeder Zustand mit seiner
RMS-Größe normiert), der den *vollen* Zustand lehrt.

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

**Unsicherheit** ist optionaler MC-Dropout, genau wie beim Smoother.

---

## 4. Stärken und Grenzen gegenüber dem Smoother

**Stärken:** Nahezu sofortige Inferenz nach den einmaligen Offline-Kosten, lernt
die *volle* Zustands-Struktur aus dem Simulator (also stark selbst bei
unbeobachteten Zuständen). Generalisiert über Betriebspunkte ohne per-Window-Refit.

**Grenzen:** Braucht eine **repräsentative** Vortrainings-Verteilung. Trägt eine
**Sim-zu-Real-Lücke**, die nur die self-supervised Anpassung schließt und die
schlecht konditionierte Biogas-Abbildung begrenzt weiterhin self-supervised
Kaltstarts.

Wie beide PINNs relativ zum UKF stehen und wie man sie kombiniert, siehe die
[UKF ↔ PINN-Fusion](fusion.md).

---

## Quelldateien

* `pyadm1ode_estimation/estimation/deep_learning/observer.py` — `Adm1Observer`  
* `pyadm1ode_estimation/estimation/deep_learning/observer_data.py` — `generate_observer_dataset`, `ObserverDataset`, `MeasurementDataset`  
* `pyadm1ode_estimation/estimation/deep_learning/observer_train.py` — `pretrain_observer`, `pretrain_observer_selfsup`, `pretrain_observer_sim2real`, `finetune_observer`  
* `pyadm1ode_estimation/estimation/deep_learning/online_observer.py` — `SlidingWindowObserver`  

## API-Referenz

::: pyadm1ode_estimation.estimation.deep_learning.observer.Adm1Observer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.deep_learning.online_observer.SlidingWindowObserver
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
