# Physics-Informed Neural Networks (PINN) für ADM1

Diese Seite erklärt den PINN-Schätzer von Grund auf. Sie beginnt mit der Idee, baut dann die genaue
Mathematik auf, erklärt die ADM1-spezifischen Kniffe, die es überhaupt zum
Funktionieren bringen und schließlich, wie es in diesem Repository umgesetzt ist
(`pinn.py`, `pinn_smoother.py`).

!!! abstract "In einem Satz"
    Ein PINN ist ein neuronales Netz, das die verborgene Zustandstrajektorie
    $\hat{x}(t)$ des Fermenters rekonstruiert, indem es gleichzeitig **die wenigen
    verfügbaren Sensormessungen anpasst** und **die ADM1-Differenzialgleichungen
    erfüllt**.

Abschnitte 1–3 sind die Intuition (ohne Mathematik), Abschnitt 4 ist
die formale Verlustfunktion, Abschnitt 5 erklärt, warum das steife ADM1 besondere
Sorgfalt braucht und Abschnitt 6 bildet jede Idee auf den echten Code ab.

---

## 1. Das gelöste Problem

Ein Biogasfermenter wird durch **41 interne ADM1-Zustände** beschrieben: gelöste
Zucker und Säuren, partikuläre Fraktionen, mikrobielle Biomasse, Ionen-/
Ladungsspezies und Gasphasendrücke (siehe [ADM1da-Modell](adm1.md)). Online lässt
sich davon nur eine Handvoll indirekt messen: Biogasstrom $Q_\text{gas}$, Methan
$Q_\text{ch4}$, pH und wenige weitere. **Zustandsschätzung** ist die Aufgabe, aus
diesen wenigen verrauschten Signalen alle 41 Zustände über die Zeit zu
rekonstruieren.

Es gibt zwei klassische Wege, jeder mit einer Schwäche:

* **Rein mechanistisches Modell (ADM1).** Vertrauenswürdige Struktur, aber es  
  braucht den exakten Anfangszustand und exakte kinetische Parameter. Kleine
  Fehler summieren sich und die Simulation driftet von der realen Anlage weg.  
* **Rein maschinelles Lernen.** Flexibel, aber es braucht einen großen  
  gelabelten Datensatz *echter* Zustände zum Lernen, den wir an einer realen
  Anlage nie haben, denn die echten Zustände sind genau das, was wir nicht messen
  können.

Ein PINN kombiniert beides: ein neuronales Netz, flexibel genug, um die spärlichen
Daten zu treffen, **eingeschränkt durch die ADM1-Physik**, sodass es weit weniger
Daten braucht und nicht in physikalisch unmögliche Trajektorien abdriften kann.

---

## 2. Ein neuronales Netz

* Ein neuronales Netz ist eine flexible mathematische Funktion  
  $y = \text{NN}_\theta(x)$ mit vielen einstellbaren Zahlen $\theta$ (den
  *Gewichten*).  
* **Training** heißt, $\theta$ so anzupassen, dass ein *Loss* per Gradientenabstieg minimiert  
  wird.  
* Die **Eingabe ist die Zeit $t$** und die **Ausgabe ist der Zustandsvektor $\hat{x}(t)$**.  
 Das Netz *ist* also die geschätzte Trajektorie selbst. Fragt man es zu
  einem beliebigen Zeitpunkt $t$, liefert es den geschätzten Anlagenzustand in
  diesem Moment.

Konkret ist es ein Multilayer-Perceptron (MLP): einige `Linear`-Schichten
mit einer glatten `tanh`-Nichtlinearität dazwischen. `tanh` wird in PINNs
bevorzugt, weil seine Ableitungen glatt sind.

---

## 3. Was „physics-informed" bedeutet

Ein nur auf Daten trainiertes Netz würde die Messpunkte treffen, wäre aber frei,
dazwischen und danach beliebig zu verlaufen. Ein PINN fügt eine zweite Forderung
hinzu: die Trajektorie muss die bekannte Bewegungsgleichung erfüllen

$$
\frac{dx}{dt} = f(x, u),
$$

wobei $f$ die rechte Seite von ADM1 und $u$ die (bekannte) Substratzufuhr ist.

Der Kniff, der das überprüfbar macht, ist die **automatische Differenzierung**:
Da die Netzausgabe $\hat{x}(t)$ eine explizite Funktion der Eingabe $t$ ist,
können wir das Framework nach ihrer exakten Zeitableitung
$\tfrac{d\hat{x}}{dt}$ fragen. ADM1 liefert $f(\hat{x}, u)$. Ihre Abweichung,

$$
r(t) \;=\; \frac{d\hat{x}}{dt} - f(\hat{x}, u),
$$

ist das **Physik-Residuum** &rarr;  Null, wenn die Trajektorie das Modell erfüllt.

Das Training minimiert dann eine Summe aus drei Forderungen:

$$
\boxed{\text{Netz-Vorhersage } \hat{x}(t)}\ \Longrightarrow\
\left\{
\begin{array}{l}
\text{vergleiche mit den Sensoren} \ \longrightarrow\ \boxed{\text{Daten-Loss}} \\[8pt]
\text{vergleiche mit der ADM1-ODE} \ \longrightarrow\ \boxed{\text{Physik-Loss}} \\[8pt]
\text{vergleiche mit dem Prior am Start} \ \longrightarrow\ \boxed{\text{Prior-Loss}}
\end{array}
\right\}
\ \Longrightarrow\ \boxed{\text{Gesamt-Loss } L}
$$

Der Gradientenabstieg auf $\theta$ minimiert $L$ und speist die aktualisierten
Gewichte zurück ins Netz.

Der Nutzen des Physik-Terms: er **interpoliert über Lücken** in den Daten,
**regularisiert** gegen Sensorrauschen und lässt das Modell, da die ODE zu jedem
Zeitpunkt definiert ist, über die letzte Messung hinaus **prognostizieren**.

---

## 4. Die Mathematik

Sei das Netz $\hat{x}_\theta(t)$. Der Schätzer minimiert

$$
L(\theta) \;=\; L_\text{data} \;+\; \lambda_\text{phys}\, L_\text{phys}
             \;+\; \lambda_\text{prior}\, L_\text{prior}.
$$

Die Gewichte $\lambda_\text{phys}, \lambda_\text{prior}$ balancieren die drei
Forderungen und sind die wichtigsten Stellschrauben.

**Daten-Loss:** die Messungen treffen. Mit der differenzierbaren Messabbildung
$h$ (Zustand $\to$ Sensorkanäle) und der Rausch-Standardabweichung $\sigma$ je
Kanal, über die beobachteten Zellen $\mathcal{O}$ (fehlende/geblockte Messungen
werden maskiert):

$$
L_\text{data} \;=\; \frac{1}{|\mathcal{O}|}
  \sum_{i}\left\lVert \frac{h(\hat{x}_\theta(t_i)) - y_i}{\sigma} \right\rVert^2 .
$$

**Physik-Loss:** die ADM1-Gleichung einhalten. Er misst, wie stark die
vorhergesagte Trajektorie die ODE verletzt. Geprüft wird das an $N_c$ *Kollokationspunkten* $\{\tau_j\}$,
selbst gewählte Zeitpunkte, die man über das Fenster legt. ADM1 mischt sehr
schnelle Vorgänge (Säure-Base-Reaktionen) mit sehr langsamen (Biomasse). 
Ohne Ausgleich würden die schnellen den Loss dominieren.  Daher wird das Residuum je Zustand 
durch eine typische Skala $s$ geteilt, sodass alle gleich zählen (Details in Abschnitt 5):

$$
L_\text{phys} \;=\; \frac{1}{N_c}\sum_{j}
  \left\lVert \frac{\tfrac{d\hat{x}_\theta}{dt}(\tau_j)  
    - f(\hat{x}_\theta(\tau_j), u)}{s} \right\rVert^2 .  
$$

**Prior-Loss:** den Start verankern. Eine schwache Randbedingung, die die
Trajektorie bei $t_0$ an einen physikalischen Prior-Zustand $x_\text{prior}$
bindet:

$$
L_\text{prior} \;=\;
  \left\lVert \frac{\hat{x}_\theta(t_0) - x_\text{prior}}{s} \right\rVert^2 .
$$

!!! tip "Warum Kollokationspunkte wichtig sind"
    Da $L_\text{data}$ nur an den Messzeiten lebt, $L_\text{phys}$ aber an jedem
    beliebigen Kollokationspunkt ausgewertet werden kann, sind die beiden
    Zeitbereiche entkoppelt. Legt man Kollokationspunkte hinter die letzte
    Messung, trägt die ODE allein den Zustand weiter: so **prognostiziert**
    derselbe Fit.


---

## 5. Warum ADM1 schwierig ist

Ein Lehrbuch-PINN scheitert an ADM1: das System ist
**steif** (Raten über viele Größenordnungen) und seine Zustände reichen von
$\sim 10^{-7}$ bis in die Zehner, wobei alle nicht-negativ bleiben müssen. Sechs
Konstruktionsentscheidungen in `PinnSmoother` bringen es zur Konvergenz
(es läuft stabil auf eine gute Lösung zu, statt zu entgleisen).

!!! warning "Das ist kein optionaler Feinschliff"
    Auf dem mitgelieferten Benchmark gemessen: ohne die Ladungsbilanz-Auflösung
    aus §5.5 verbessert der Fit seinen eigenen Startwert **nie** — bei keiner
    Lernrate, keiner Loss-Gewichtung und keiner Netzgröße. Die beste je erreichte
    Data-Loss war ihr Wert bei Epoche 0; das Training hat den Fit nur
    verschlechtert. Zahlen in §7.

### 1. Positive, gut skalierte Ausgaben

Bisher hieß es vereinfacht: das Netz gibt $\hat{x}(t)$ aus. Genauer liefert die
letzte Schicht erst eine **rohe**, unbeschränkte Zahlenreihe $\text{raw}_\theta(t)$.
Den physikalischen Zustand bildet daraus erst eine **Log-Abweichung von einem Prior**:

$$
\hat{x}(t) \;=\; x_\text{prior}\,\odot\, \exp\!\big(\text{raw}_\theta(t)\big).
$$

Die letzte Schicht ist **null-initialisiert**, sodass zu Trainingsbeginn
$\text{raw}_\theta \equiv 0$ und $\hat{x}(t) = x_\text{prior}$ exakt gilt. Die Exponentialfunktion garantiert
$\hat{x}(t) > 0$ für alle Zeiten (`raw` wird für die numerische Sicherheit auf
$[-10, 10]$ begrenzt).

### 2. Relatives Physik-Residuum

Säure-Base-Reaktionen in ADM1 laufen mit Raten $\sim 10^{8}$, während sich
Biomasse über Tage ändert. Eine Normierung des Residuums nach Zustandsgröße
würde die schnellen Terme den Loss überfluten lassen. Stattdessen ist die Skala
$s$ aus Abschnitt 4 nicht fest, sondern jede Gleichung wird durch ihre **eigene
aktuelle Rate** $\lvert f_i \rvert$ geteilt (mit einer Untergrenze, damit Zustände nahe dem
Gleichgewicht mit $f\approx 0$ nicht explodieren). Ergebnis ist ein *relatives* Residuum,
sodass keine einzelne steife Gleichung allein durch ihre Größe dominiert.

### 3. Quasi-stationäre Gasphase

Die vier Gasphasendrücke (besonders der Gesamtdruck) sind numerisch heikel: schon
winzige Änderungen können die Lösung kippen lassen. Statt das Netz sie vorhersagen
zu lassen, sagt es **nur die 37 Flüssigzustände** vorher. Die Gasdrücke werden bei jeder
Auswertung aus dem Gas-Flüssig-Gleichgewicht berechnet (`gas_equilibrium_torch`). 
Die vier Gas-ODEs sind dann automatisch erfüllt, sodass der Physik-Loss nur noch
die 37 Flüssigzustände prüfen muss.

### 4. Prior-Verankerung und zustandsweise Gewichtung

Der Prior-Term aus §4 zieht **alle** Zustände bei $t_0$ zu ihrem physikalischen
Prior. Gut beobachtete zieht der Daten-Loss wieder weg, schwach beobachtete
bleiben dort hängen. Welche das sind, ergibt sich also von selbst aus der
Beobachtbarkeit. Damit der Optimierer solche trägen Zustände trotzdem bewegt,
statt sie am Prior zu parken, kann man sie in $L_\text{phys}$ einzeln **höher gewichten**:
ein von Hand gesetzter Gewichtsvektor, **kein gelernter Parameter**. Welche man
hochgewichtet, entscheidet man selbst per Fachwissen/Diagnose, nicht das Training.

Der Prior selbst muss ein **physikalisch erreichbarer Zustand** sein. Eine
komponentenweise Statistik (etwa der Median je Zustand über einen Trainingssatz)
ist das im Allgemeinen nicht: nimmt man jede Komponente für sich, zerbrechen die
Identitäten, die sie verbinden. Bei ADM1 ist das die Ladungsbilanz, und der so
entstandene „Durchschnittszustand" impliziert einen pH weit außerhalb des
physikalischen Bereichs. Stattdessen den **Medoid** nehmen — einen echten
beobachteten Zustand; `PinnData` macht das automatisch (§6).

### 5. Quasi-stationäre Ladungsbilanz (`solve_cation=True`)

ADM1 gewinnt den pH aus der Elektroneutralität:

$$
	ext{fixed} = S_	ext{cat} - S_	ext{an} + (S_{\mathrm{NH_4}} - S_{\mathrm{NH_3}})  
              - S_{\mathrm{HCO_3}} - \sum 	ext{VFA}^- ,  
\qquad
S_{\mathrm H} = 	frac12\!\left(-	ext{fixed} + \sqrt{	ext{fixed}^2 + 4K_w}
ight).
$$

An einem normalen Arbeitspunkt ist `fixed` eine Differenz von Termen der
Größenordnung $0{,}1\!-\!0{,}2$ kmol m⁻³, die sich auf $\sim\!10^{-6}$ auslöscht —
**5,3 Dekaden Auslöschung**. Wegen $S_\mathrm{H} pprox K_w/	ext{fixed}$ gilt
faktisch $	ext{pH} pprox -\log_{10} K_w + \log_{10}(	ext{fixed})$: stört man
irgendeinen Ionenzustand um 1 %, bewegt sich `fixed` um Größenordnungen.

Das ist dieselbe Messerschneide wie die Biogas-Abbildung in §5.3, mit derselben
Folge — der pH reagiert mit $\sim\!5\cdot10^{4}\,\sigma$ auf eine 1-%-Zustands-
änderung, sein Gradient diktiert jeden Optimierungsschritt und der Fit divergiert.

Die Lösung spiegelt den Gas-Solver: **das Netz darf nicht raten, woran die
Auslöschung hängt.** Das Netz sagt direkt den **pH** vorher (eine gut skalierte
Ausgabe um $pprox 7{,}5$ mit Empfindlichkeit 1:1), und $S_	ext{cat}$ wird
geschlossen aus der Ladungsbilanz gelöst:

$$
S_\mathrm{H} = 10^{-	ext{pH}},\qquad
	ext{fixed} = rac{K_w - S_\mathrm{H}^2}{S_\mathrm{H}},\qquad
S_	ext{cat} = 	ext{fixed} - (	ext{übrige Ladungsterme}).
$$

!!! note "Was das kostet"
    $S_	ext{cat}$ ist damit kein differentieller, sondern ein algebraischer
    Zustand — aus der ODE wird eine **DAE**, die triviale Verdünnungsgleichung
    wird durch die Elektroneutralitäts-Nebenbedingung ersetzt. `PinnSmoother`
    maskiert ihn deshalb aus dem Physik-Residuum **und** dem Prior-Anker. Das ist
    Absicht: $S_	ext{cat}$ ist eine Buchhaltungsladung, keine gemessene Spezies,
    und genau die Erzwingung seiner Verdünnungsgleichung erzeugt die schlechte
    Kondition. Aus demselben Grund kann der gelöste Wert leicht negativ werden —
    das ist ein Anionenüberschuss und liegt im Rahmen dessen, wie ADM1 diese
    beiden Slots verwendet.

### 6. Begrenzte Residuen (`res_clip`)

Ein quadriertes Residuum hat einen Gradienten proportional zum Residuum, also
diktiert ein schlecht konditionierter Kanal den ganzen Schritt. `res_clip` stellt
das Residuum auf **Huber** um: quadratisch unterhalb der Schwelle $\delta$, linear
darüber, sodass der Gradient bei $2\delta$ gedeckelt bleibt, ohne null zu werden.

Das ist bewusst weicher als ein hartes Abschneiden (das den Gradienten jenseits
der Schwelle auf null setzt und einem bereits entglittenen Kanal jede Rückholkraft
nimmt). Es begrenzt den Schaden eines schlecht konditionierten Kanals, ersetzt
aber **nicht** §5.5 — es deckelt, wie weit ein schlechter Schritt geht, nicht die
Kondition, die ihn verursacht.

---

## 6. Umsetzung in diesem Repository

Die Umsetzung hat **zwei Ebenen**: eine Lehrbuch-Vorlage und den darauf
aufbauenden Produktivschätzer.

| Ebene | Datei | Rolle |
| --- | --- | --- |
| `ADM1PINN` + `PINNLoss` | `deep_learning/pinn.py` | Die Bausteine: das MLP $t \to x$ (`tanh`, konfigurierbare Hidden-Layer, optionaler Dropout) und eine generische *Daten + Physik*-Loss-Vorlage. Lesbare Referenzimplementierung. |
| `PinnSmoother` | `deep_learning/pinn_smoother.py` | Der Produktivschätzer. Baut auf `ADM1PINN` auf und ergänzt die Log-Transformation, den Drei-Term-Loss, Rate-Scaling, quasi-stationäre Gasphase, Prognose, rollierende Updates und MC-Dropout-Unsicherheit. |

Die differenzierbaren Physik-Bausteine, die er zusammenschaltet, kommen alle aus
dem Basispaket `pyadm1`:

| Symbol | Bereitgestellt von |
| --- | --- |
| $f$ — rechte Seite von ADM1 | `pyadm1.core.adm1_torch.adm1da_rhs_torch` |
| Gas-Flüssig-Gleichgewicht | `pyadm1.core.adm1_torch.gas_equilibrium_torch` |
| $h$ — Messabbildung | `deep_learning.observation_torch.TorchObservationModel` |
| Ladungsbilanz-Inversion | `deep_learning.charge_balance.solve_cation_for_ph` |
| Zufuhr / Parameter | `Adm1TorchParams` — die Netzeingabe bleibt **nur die Zeit**; eine im Fenster *veränderliche* Zufuhr kommt über `params_at` |

### Die Datenanbindung: der Adapter

`PinnData` (`deep_learning/data_adapter.py`) bringt jeden in
`filter_tuning.datasets` registrierten Datensatz in die Form, die beide
PINN-Varianten brauchen. Drei Zusagen sind für die Korrektheit entscheidend:

* **Derselbe Split wie bei den Filtern.** Beide gehen durch  
  `EstimatorDataset.split_indices` — stratifiziert nach dem Serien-Label,
  gruppiert nach Serie. Gleiches `(val_frac, seed)` ⇒ gleiche Serien, und genau
  das macht Netz und Filter vergleichbar. Mit `save_split()` / `split_file=`
  einfrieren.  
* **Statistik nur aus Train.** Feature-Normierung, `x_ref`, `x_prior` (als  
  Medoid) und die Zustandsskala werden ausschließlich aus den Trainingsserien
  geschätzt.  
* **Feed-passende Physik.** `smoother_inputs()` hängt jeder Serie eine  
  `params_at`-Closure über ihre *eigene* Zufuhr an, damit das ODE-Residuum gegen
  die Anlage ausgewertet wird, die die Daten erzeugt hat. Das wiegt schwerer als
  es klingt: ein ungefütterter Parametersatz (`q_ad = 0`) modelliert einen
  *geschlossenen Batch-Reaktor*, und der Zulaufterm $D_	ext{in}\,s_	ext{in}$
  dominiert die rechte Seite von ADM1.

```python
from pyadm1ode_estimation.estimation.deep_learning import PinnData

data = PinnData.build("benchmark", val_frac=0.2, seed=0)
inputs = data.smoother_inputs("val", days=5.0)     # ein Payload je Serie
```

### Batch, nicht rekursiv

Anders als der UKF (ein rekursiver `StateEstimator` mit Prädiktion/Korrektur je
Schritt) ist das PINN ein **`BatchEstimator`**: einmal über das Fenster fitten,
dann abfragen.

```python
from pyadm1ode_estimation.estimation.deep_learning import PinnSmoother

it = inputs[0]
smoother = PinnSmoother(
    data.physics_params(), data.obs_model(), it.x_prior, it.x_scale,
    quasi_steady_gas=True,   # §5.3 — Gasdrücke werden gelöst
    solve_cation=True,       # §5.5 — S_cation folgt dem vorhergesagten pH
    res_clip=3.0,            # §5.6 — Huber-Schwelle in Sigma
    params_at=it.params_at,  # die eigene, zeitvariable Zufuhr der Serie
)
smoother.fit(**it.fit_kwargs(), epochs=2000, lr=1e-3)
traj = smoother.estimate(it.obs_times)                 # (T, 41) Zustände + std
```

`estimate` liefert einen `TrajectoryEstimate` (`time`, `x_hat`, `std`), die
gemeinsame Ausgabewährung mit dem UKF, sodass das Twin-Experiment-Harness beide
Schätzerfamilien gleich bewertet.

`restore_best=True` (Standard) gibt die Gewichte der besten Epoche zurück statt
der letzten. Der Kollokations-Fit ist **nicht monoton** — er erreicht regelmäßig
seine beste Trajektorie und läuft dann wieder von ihr weg, sodass die
Endgewichte die bereits gefundene Antwort wegwerfen. Zugleich wird der Fit damit
monoton-sicher: er kann nie etwas Schlechteres als den Prior zurückgeben.

### Prognose

Da das Kollokationsfenster `[t0, t1]` unabhängig von den Messzeiten ist, trägt
das Setzen von `t1` **hinter die letzte Messung** den Zustand über die ADM1-ODE
in den datenfreien Bereich, eine physikgetriebene Prognose aus demselben Fit.

### Online-Betrieb

`update(...)` startet warm aus den aktuellen Gewichten **und** dem
Optimierer-Zustand für günstige inkrementelle Nachfits, wenn neue Messwerte
eintreffen:

* **wachsendes Fenster:** die ganze Historie behalten, Anker bleibt beim  
  ursprünglichen `t0`; oder  
* **gleitendes Fenster:** ein fester Horizont `t0 = t1 − window`, selbst  
  verankert am eigenen aktuellen Schätzwert des Netzes am Fensteranfang
  (begrenzt den Rechenaufwand bei langen Läufen).

### Unsicherheit

Optionaler **MC-Dropout**: `dropout > 0` setzen und `estimate(..., mc_samples=k)`
aufrufen, um Mittelwert und Standardabweichung über `k` stochastische
Vorwärtspässe zu erhalten.

---

## 7. Gemessener Stand

Die Zahlen stammen vom Validierungs-Split des mitgelieferten Benchmarks (4
Betriebsmodi), 5-Tage-Fenster, reproduzierbar über `experiments/pinn_gate/`.

**Die Optimierung ist gelöst.** Data-Loss = mittleres quadriertes standardisiertes
Residuum auf den Messkanälen; der Rauschboden (was der *wahre* Zustand erreicht)
liegt bei 0,57.

| Konfiguration | Median der besten Data-Loss |
| --- | --- |
| Ausgangspunkt | 3520 |
| + `res_clip` | 11,7 |
| + `solve_cation` (§5.5) | **0,562** |

Der Messfit erreicht damit sein theoretisches Optimum. Vor der
Ladungsbilanz-Auflösung hat das Training den Fit nie verbessert — der beste je
erreichte Wert war der Wert bei Epoche 0.

**Die Zustandsgenauigkeit ist es nicht.** Median-NRMSE über 41 Zustände, gegen
die triviale Referenz „Prior konstant halten":

| | Median-NRMSE |
| --- | --- |
| `x_prior` halten (nichts tun) | 25,1 % |
| PINN A (bestes `rate_floor`) | 22,0 % — aber nur 2 von 4 Modi |
| wahres $x(t_0)$ halten (Referenz) | 1,2 – 3,7 % |

Ein `rate_floor`-Sweep zeigt kein belastbares Optimum. **Fünf Sensoren legen 41
Zustände nicht fest**, und der ADM1-Term kann die Lücke nicht schließen, weil ein
Fit pro Fenster die wahre Kinetik nicht lernen kann (der Benchmark stört sie je
Serie lognormal, $\sigma = 0{,}25$). Die verbleibende Lücke ist
**Beobachtbarkeit, nicht Architektur** — deshalb sind Breite, Tiefe und
Aktivierung des Netzes bislang nicht optimiert: sie sind nachweislich nicht der
begrenzende Faktor.

**Grenzen.** Die MC-Dropout-Unsicherheit ist nicht kalibriert. Ein Fit von Grund
auf kostet ~2,5 min pro 5-Tage-Fenster, was den bezahlbaren Umfang einer
Hyperparametersuche deckelt.

**Einordnung.** Der UKF (siehe [UKF im Einsatz](../usage/ukf.md) und
[SR-UKF-Performance](../development/ukf_performance.md)) ist rekursiv und günstig
pro Schritt. Auf dieser Metrik ist er allerdings selbst **in 3 von 4 Modi
schlechter als Nichtstun** (50,5 % gegen 34,2 % gesamt auf dem Test-Split) — also
beide Referenzen berichten, sonst sieht ein Ergebnis besser aus als es ist. Der
**[vortrainierte Observer](observer.md)** nimmt dieselbe Hürde in allen vier
Modi, weil er über Serien hinweg lernt, was ein Einzelfenster-Fit nicht kann. Ein
**[Hybrid per Kovarianzschnitt](fusion.md)** kann Schätzer fusionieren.

---

## Quelldateien

* `pyadm1ode_estimation/estimation/deep_learning/pinn.py` — `ADM1PINN`, `PINNLoss`  
* `pyadm1ode_estimation/estimation/deep_learning/charge_balance.py` — `solve_cation_for_ph`, `apply_ph`  
* `pyadm1ode_estimation/estimation/deep_learning/data_adapter.py` — `PinnData`, `FeatureSpec`  
* `experiments/pinn_gate/` — Gate-Messung und `rate_floor`-Sweep  
* `pyadm1ode_estimation/estimation/deep_learning/pinn_smoother.py` — `PinnSmoother`  
* `pyadm1ode_estimation/estimation/deep_learning/observation_torch.py` — `TorchObservationModel`  

## Literatur

* Raissi, M., Perdikaris, P. & Karniadakis, G. E. (2019). *Physics-informed  
  neural networks.* Journal of Computational Physics 378:686–707.  
* ADM1da-Modell und Zustandsindizes: [ADM1da-Modell](adm1.md).  

## API-Referenz

::: pyadm1ode_estimation.estimation.deep_learning.pinn_smoother.PinnSmoother
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: pyadm1ode_estimation.estimation.deep_learning.pinn.ADM1PINN
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
