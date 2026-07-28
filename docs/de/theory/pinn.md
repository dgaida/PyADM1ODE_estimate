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
$\sim 10^{-7}$ bis in die Zehner, wobei alle nicht-negativ bleiben müssen. Vier
Konstruktionsentscheidungen in `PinnSmoother` bringen es zur Konvergenz 
(es läuft stabil auf eine gute Lösung zu, statt zu entgleisen).

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
| Zufuhr / Parameter | `Adm1TorchParams` — die Substratzufuhr ist eingebacken, daher ist die Netzeingabe **nur die Zeit** |

### Batch, nicht rekursiv

Anders als der UKF (ein rekursiver `StateEstimator` mit Prädiktion/Korrektur je
Schritt) ist das PINN ein **`BatchEstimator`**: einmal über das Fenster fitten,
dann abfragen.

```python
from pyadm1ode_estimation.estimation.deep_learning import PinnSmoother

smoother = PinnSmoother(params, obs, x_prior, quasi_steady_gas=True)
smoother.fit(obs_times, obs_values, t0=0.0, t1=30.0)   # Training über [0, 30] Tage
traj = smoother.estimate(query_times)                  # (T, 41) Zustände + std
```

`estimate` liefert einen `TrajectoryEstimate` (`time`, `x_hat`, `std`), die
gemeinsame Ausgabewährung mit dem UKF, sodass das Twin-Experiment-Harness beide
Schätzerfamilien gleich bewertet.

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

## 7. Stärken, Grenzen und die Einordnung zum UKF

**Stärken.** Trifft die biogastreibenden Zustände genau. Die weiche Physik-Kopplung ist flexibel,
kommt mit spärlicher, unregelmäßiger Abtastung zurecht.

**Grenzen.** Die pH-/Ladungsbilanz-Abbildung ist schlecht konditioniert (beide
Schätzer haben hier Mühe, das PINN mehr). Die MC-Dropout-Unsicherheit ist noch
nicht kalibriert, ein Fit von Grund auf kostet Sekunden bis Minuten pro Fenster.

**Einordnung.** Der UKF (siehe [UKF im Einsatz](../usage/ukf.md) und
[SR-UKF-Performance](../development/ukf_performance.md)) ist rekursiv, günstig pro
Schritt und besser kalibriert, vor allem beim pH. Das PINN ist stärker auf den
Biogaskanälen und bei der Prognose. Ein **[Hybrid per Kovarianzschnitt](fusion.md)**
kann beide fusionieren und die Stärken jedes Schätzers
bewahren.

---

## Quelldateien

* `pyadm1ode_estimation/estimation/deep_learning/pinn.py` — `ADM1PINN`, `PINNLoss`  
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
