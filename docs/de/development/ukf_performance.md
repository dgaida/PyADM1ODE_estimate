# SR-UKF Performance — Architektur und Optimierungen

Der [`UnscentedKalmanFilter`](../../api/index.md) implementiert die
kanonische Square-Root-UKF nach **Wan & van der Merwe 2001
(Algorithmus 3.1)**, in der Variante mit Sigma-Punkt-Wiederverwendung
zwischen Predict und Measurement Update. Diese Seite dokumentiert,
welche Optimierungen den Code dorthin gebracht haben und wie ihre
Korrektheit verifiziert wird.

## Aktueller Algorithmus

Pro Filterschritt:

1. **Predict** zieht `2n+1` Sigma-Punkte um `(x_hat, S)`, propagiert sie  
   durch `process.step(σ_i, dt)` und liest **gleichzeitig**
   `h(plant, σ_i_propagated)` für *alle* Beobachtungskanäle aus.
   Predicted Mean + Cholesky-Faktor via QR-Stack mit `√Q`.  
2. **Update** verbraucht die im Predict gecachten `h`-Werte direkt  
   (Spaltenauswahl über die aktiven Kanäle). Keine zweite Sigma-Punkt-
   Auswertung, kein zweiter Plant-Durchlauf. Cross-Kovarianz
   `T_xy` aus `(propagated − x_pred)`-Differenzen.

ODE-Last pro Filterschritt: **`2n+1` Plant-Integrationen** (nur im
Predict). Die frühere "Redraw"-Variante hatte `2 · (2n+1) = 4n+2` —
die doppelte Last für mathematisch keinen Gewinn.

## Optimierungs-Wellen (2026-06)

### Welle 1 — verlustfreie Linalg-Optimierungen

Alle vier Änderungen ergebnisidentisch bis auf letzte Float-Bits
(`atol=1e-12`):

1. **`_sigma_points` vektorisiert** — Python-Schleife → zwei Broadcasts.  
2. **`T_xy` vektorisiert** — Schleife `for i: outer(...)` → ein  
   Matrix-Produkt.  
3. **`cholesky(Q)` gecacht** — keyed auf `dt`, Hit bei konstantem `dt`.  
4. **`cholesky(R)` gecacht** — keyed auf das aktive Kanal-Tupel.  

### Welle 2 — Sigma-Punkt-Wiederverwendung

Strukturelle Änderung, die ~50% der ODE-Aufrufe einspart. Der frühere
Code zog im `update()` einen frischen Sigma-Satz um `(x_pred, S_pred)`
und stoppte den Plant für jeden noch einmal — eine Erweiterung, die
nicht zur kanonischen Wan-VdM-2001-Form gehört. Im 24-h-ADM1-Twin
(festem Seed, simple-Plant) brachte die Umstellung **1.84× Speedup**
(429 s → 234 s) bei vergleichbarer Genauigkeit pro Quality-Block.

Mathematische Approximation, die dabei akzeptiert wurde: der
`H · Q · H^T`-Beitrag zur Innovationskovarianz `S_y` wird weggelassen
(die propagierten Sigma-Punkte decken `sample_cov`, nicht `sample_cov +
Q`). Wu et al. 2005 zeigen, dass das gegenüber der state-augmentierten
UKF einen Genauigkeitsverlust bedeutet — der für ADM1 mit kleinem
`Q/P` (≈ 1/250 bei `dt = 1h`) jedoch zwei Größenordnungen unter dem
Messrauschen liegt. Hellmann et al. 2024 (ECC, AD-spezifisch)
finden auf einem ADM1-R4-Modell die gleichen Trajektorien zwischen
ihrer `UKF-SR`-Variante (= jetzt unsere Form) und `UKF-add` ohne
Augmentierung.

## Literaturgrundlage

| Quelle | Aussage |
|---|---|
| Wan & van der Merwe (2001) "The Square-Root Unscented Kalman Filter for State and Parameter-Estimation" | Algorithmus 3.1, Zeile 22: `Y_{k|k-1} = H[X_{k|k-1}]` — h wird direkt auf den propagierten Sigma-Punkten ausgewertet, kein Redraw. |
| Wu, Hu, Wu & Hu (2005) "Unscented Kalman filtering for additive noise case: Augmented vs. non-augmented" | Augmentierte Form (Q-Dimensionen im Sigma-Satz) ist theoretisch überlegen, weil sie Odd-Moment-Information aus Q im Messpfad propagiert. Effekt skaliert mit `||Q|| / ||P||`. |
| Hellmann, Wilms, Streif & Weinrich (2024, ECC) "Comparison of Unscented Kalman Filter Design for Agricultural Anaerobic Digestion Model" | Direkte ADM1-Studie, 9 UKF-Varianten. UKF-SR (= unsere Form) und UKF-aug ergeben bei reduzierter Sigma-Skalierung praktisch identische NRMSE bei niedrigerer Laufzeit für UKF-SR. |

## Wie verifiziert

### Bit-stabile Regression — nichtlinearer Mock-Prozess

[`tests/test_ukf_regression.py`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/tests/test_ukf_regression.py)
fährt einen 20-Schritt-Twin mit nichtlinearem Mock-Prozess
(elementweise Quadratik + lineare Kopplung) und vergleicht den
finalen Posterior gegen einen hartcodierten Goldwert (`atol=1e-12,
rtol=0`). Goldwerte wurden nach der Hauptpfad-Migration neu erzeugt.

**Goldwerte regenerieren** (nach absichtlichen Algorithmus-Änderungen):

```bash
python - <<'PY'
import sys
sys.path.insert(0, ".")
import conftest  # DLL-Pfade auf Windows
sys.path.insert(0, "tests")
import test_ukf_regression as m
import numpy as np
np.set_printoptions(precision=16, floatmode="maxprec")
x, S = m._run_regression_trajectory()
print("X =", repr(x))
print("S =", repr(S))
PY
```

### Lineare-KF-Konsistenz

[`tests/test_ukf.py::TestUKFLinear`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/tests/test_ukf.py)
hat zwei Tests zum Vergleich mit dem geschlossenen Kalman-Filter:

* `test_ukf_matches_classical_kf_with_negligible_Q` — bei `Q ≈ 1e-16`  
  (zehn Größenordnungen unter `P`) stimmt die SR-UKF-Trajektorie mit
  dem KF auf `atol=1e-6` überein. Pinnt die Algebra-Korrektheit der
  Reuse-Form fest.  
* `test_ukf_approximates_classical_kf_with_random_walk_Q` — bei  
  `Q/P ≈ 1%` weicht die Trajektorie um wenige Prozent vom KF ab,
  konsistent mit dem weggelassenen `H Q H^T`-Term. `atol=0.05`.
  Pinnt die Größenordnung der Approximation fest; deutliche
  Abweichungen würden auf eine ungewollte Algorithmusverschiebung
  hinweisen.

### Cache-Invalidierung

Vier weitere Tests in `test_ukf_regression.py` mit zählenden Wrappern:

* `test_sqrt_Q_cache_hits_under_constant_dt`  
* `test_sqrt_Q_cache_invalidates_on_dt_change`  
* `test_sqrt_R_cache_hits_under_constant_active_set`  
* `test_sqrt_R_cache_invalidates_on_active_set_change`  

### Welle 3 — Reduced Sigma Scaling (opt-in)

Der `UnscentedKalmanFilter` akzeptiert seit 2026-06 einen
`gamma_override`-Parameter, der die kanonische Sigma-Punkt-Radius
`γ = √(n + λ)` durch einen frei wählbaren Wert ersetzt. Die Gewichte
`w_m`, `w_c` bleiben kanonisch (aus `α, β, κ` gebaut) — nur die
Position der Sigma-Punkte auf dem Kovarianz-Ellipsoid ändert sich.

Empirische Motivation aus Hellmann et al. 2024 §5.1.2: auf
ADM1-R4-Core sinkt NRMSE_x bei `γ = 1` (statt kanonisch `√(n+λ) ≈ 2.45`)
von 0.85 auf 0.37 — der größte Einzelgewinn im Paper. Begründung:
für hochdimensionale, schwach-nichtlineare Systeme mit Gauß-ähnlichem
Messrauschen liefert die engere Sigma-Wolke ein besseres Smoothing als
die kanonische Julier–Uhlmann-Skalierung.

Anwendung:

```python
ukf = build_ukf(plant, digester_id="primary", substrates=...,
                gamma_override=1.0)
```

Default `None` lässt die Sigma-Skalierung unverändert — alle bestehenden
Setups laufen weiter exakt wie vorher.

#### Empirischer γ-Sweep auf der simple plant (n=43)

24-h-Twin, fester Seed (42), identische Truth und Messungen, nur γ
variiert. Wandzeit pro Variante 229-242 s — γ ist ein reiner Qualitäts-
und Kalibrierungs-Knopf, kein Speed-Knopf.

| γ | Mean NIS (Ziel ≈ 5) | Avg. Coverage | schlimmster Block | Bewertung |
|---|---|---|---|---|
| canonical (≈ 6.56) | 6.75 | **96.5 %** | charge_balance 70 % | sauber kalibriert |
| 5.0 | 9.79 | 91.2 % | charge_balance 60 % | leicht überzuversichtlich |
| 4.0 | **17.23** | 84.8 % | **nitrogen 16 %** | Kalibrierung kollabiert |
| **3.0** | **44.80** | 81.2 % | nitrogen 20 % | **katastrophal** |
| 2.0 | 17.80 | 84.0 % | charge_balance 50 % | schlecht |
| 1.5 | 9.06 | 87.4 % | charge_balance 48 % | mäßig |
| 1.0 | 6.32 | 88.4 % | charge_balance 50 % | bestes Mean, überzuversichtlich |

RMSE: auf gut beobachtbaren Blöcken sinkt der Fehler bei γ=1 um
40-73 %, auf den schwach beobachtbaren `input_flow`-Kanälen steigt er
um 25 %. Mittlere γ-Werte (2-5) sind in **beiden** Metriken strikt
schlechter als die Endpunkte canonical und γ=1.

#### Warum nicht-monoton

Die Gewichte `w_m`, `w_c` sind aus der kanonischen Skalierung
`γ = √(n+λ)` so gebaut, dass die unscented Transform Mittelwert und
Kovarianz bis zur dritten Ordnung exakt rekonstruiert — *nur in
Kombination* mit genau diesem γ. Hellmann's Trick entkoppelt γ von
den Gewichten: tighte Sigma-Wolke + Gewichte einer weiten Wolke →
die rekonstruierte Posterior-Kovarianz ist systematisch zu klein,
NIS explodiert.

Bei γ=1 wirkt eine andere Dynamik (effektives Tiefpass-Smoothing der
Messung) wieder regularisierend und liefert akzeptables NIS aus
formal "falschem" Grund. Im Übergangsbereich (γ ≈ 2-5) wirkt keiner
der beiden Effekte sauber → die Filter-Konsistenz kollabiert.

#### Empfehlung

* **Default `gamma_override=None`** für UQ-relevante Anwendungen  
  (Coverage-Kalibrierung ist hier Priorität).  
* **`gamma_override=1.0`** opt-in für Modi, wo nur der posterior Mean  
  verbraucht wird (z.B. MPC-Targets, Steuerungssignale) und Coverage
  zweitrangig ist.  
* **Werte zwischen 1 und 6 vermeiden** — kein Sweet Spot, beide  
  Metriken werden gleichzeitig schlechter.  
* Für *gleichzeitig* gute Mean und Calibration: state-augmentierte  
  Form (Welle 4 in der Roadmap), die die `Q`-Inkonsistenz direkt im
  Algorithmus behebt, statt sie über γ zu kompensieren.

### Welle 4 — Prozess-Parallelisierung der Sigma-Auswertung

Neue Klasse [`ParallelUKF`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/pyadm1ode_estimation/estimation/filters/parallel_ukf.py)
verteilt die `2n+1` Plant-Integrationen einer `predict()`-Iteration
auf einen `multiprocessing.spawn`-Worker-Pool. Linalg-Pfad (QR,
Cholupdate, Gewichte) bleibt auf dem Hauptprozess; nur die
embarrassingly parallel Sigma-Propagation wandert in Worker.

#### Architektur

* Jeder Worker baut sich beim Pool-Start **seine eigene** `(process,  
  obs, spec)`-Tupel über einen user-bereitgestellten Top-Level-Builder
  auf. Damit umgehen wir das Pickle-Problem der `obs`-Closures, die auf
  Plant-Referenzen verweisen würden.  
* Pro Task wird nur das winzige Snapshot-Dict (~1 KB) zwischen  
  Hauptprozess und Workern ausgetauscht. Die Plant-Objekte selbst
  überqueren die IPC-Grenze nur einmal beim Pool-Spawn.  
* `process.step` ruft intern `restore()` auf — wir mussten  
  [`process_model.snapshot/restore`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/pyadm1ode_estimation/estimation/process_model.py)
  erweitern, sodass *alle* Komponenten mit `adm1_state` zurückgesetzt
  werden, nicht nur das primäre Digester-Modul. Das schließt einen
  schon im seriellen Code latent vorhandenen Drift-Bug (Storage-Tank
  akkumulierte Zustand über Sigma-Punkte hinweg) und macht den UKF
  theoretisch sauberer: jede Sigma-Punkt-Auswertung sieht jetzt
  unabhängig dieselbe Baseline.

#### Benchmark (simple plant, 24-h-Twin, fester Seed)

| Variante | Wandzeit | Speedup | s pro Filterschritt |
|---|---|---|---|
| serial | 238.6 s | 1.00× | 9.94 |
| parallel_2 | 207.7 s | 1.15× | 8.66 |
| parallel_4 | 111.9 s | 2.13× | 4.66 |
| parallel_8 | 80.6 s | **2.96×** | 3.36 |

**Block-RMSE und 2σ-Coverage sind über alle Worker-Zahlen
bit-identisch.** Mean-Trajektorien-Differenz vs. seriell: exakt
0 σ auf jedem Kanal. Mean NIS = 7.08 für alle Varianten.

Sub-lineare Skalierung erwartbar (Amdahl):  
* Pool-Spawn-Overhead (~2-3 s pro Worker bei Windows-spawn)  
* Linalg-Pfad bleibt sequenziell und wird relativ größer mit mehr Workern  
* Über 4 physische Cores hinaus (vermutlich Hyperthreads in diesem Setup)  
  geht der Gewinn deutlich zurück

Für deinen typischen Twin-Workflow (5 Tage × 24 h = 120 Schritte)
heißt das: serial ≈ 20 min → parallel_8 ≈ 7 min.

#### Anwendung

```python
from pyadm1ode_estimation.estimation import (
    InputSpec, build_filter_components,
)
from pyadm1ode_estimation.estimation.filters import ParallelUKF
from pyadm1ode_estimation.example_plants import build_simple_plant

# Top-level (importable) — multiprocessing.spawn pickelt eine Referenz
# darauf, KEINE Lambdas oder closure-bound Methoden hier.
def make_components():
    plant = build_simple_plant()
    return build_filter_components(
        plant,
        digester_id="fermenter",
        substrates=[
            InputSpec("maize_silage",  substrate_index=0, initial_flow=10.0),
            InputSpec("cattle_slurry", substrate_index=1, initial_flow=5.0),
        ],
        sensors=["q_gas", "q_ch4", "ph", "substrate_dose"],
    )

process, obs, spec = make_components()
ukf = ParallelUKF(
    process, obs, spec,
    n_workers=4,
    components_builder=make_components,
)
# ... normale predict/update-Schleife ...
ukf.shutdown()  # Worker-Pool freigeben
```

`n_workers=1` fällt explizit auf den seriellen Hauptpfad zurück —
kein Pool-Overhead, identisches Verhalten zur Basisklasse.

### Welle 5 — Constrained UKF: implementiert, negativ-Resultat auf ADM1

[`ConstrainedUKF`](https://github.com/dgaida/PyADM1ODE_estimate/blob/master/pyadm1ode_estimation/estimation/filters/constrained_ukf.py)
implementiert Hellmann 2024's `cUKF-add`: jeder propagierte Sigma-Punkt
wird im Update statt über die Kalman-Verstärkung durch ein
**box-constrained QP** pro Sigma korrigiert.

```math
\chi^{\text{corr}}_i = \arg\min_\chi \;
  \|y - h(\chi)\|^2_{R^{-1}}
  + \|\chi - \chi^-_i\|^2_{(P^-)^{-1}}
\quad \text{s.t.} \quad x_{\text{lo}} \le \chi \le x_{\text{hi}}
```

Implementierungs-Choices:  
* `h` linearisiert über LS-Fit von den `2n+1` propagierten Sigma-Punkten  
  und ihren bereits gecachten h-Werten (keine extra Plant-Auswertung)  
* Per-Sigma-QP über `scipy.optimize.minimize(method="trust-constr")`  
  mit analytischem Gradient + Hessian  
* Posterior in Square-Root-Form nach Hellmann eq. 11:  
  `P = Σ Wᶜ (χᶜ - x̂)(χᶜ - x̂)^⊤ + Q + K R Kᵀ`  
* 4 Unit-Tests pinnen Korrektheit: bei weiten Bounds & linearem h  
  reproduziert `cUKF ≡ UKF` zu `atol=1e-6`; bei engen Bounds wird die
  Schranke respektiert; smoke test auf der ADM1-Plant läuft durch.

#### Empirisches Ergebnis (24h-Twin, simple plant, n=43)

| Variante | Wandzeit | Mean NIS | Avg. Block-RMSE |
|---|---|---|---|
| UKF (Baseline) | 232.1 s | 7.08 | Referenz |
| cUKF | **571.9 s** (2.5× langsamer) | 9.32 (schlechter) | **2-5× schlechter** auf gut beobachtbaren Blöcken |

Methanogenese-RMSE: 0.015 → 0.077 (5× schlechter). Disintegration:
0.20 → 0.95. Coverage durchweg gesunken oder gleich. Mean-Trajektorien
auf einzelnen Kanälen bis zu **2390 σ** abweichend von der UKF-Baseline.

#### Strukturelle Ursache: Sigma-Spread-Kollaps am multi-skalierten Zustandsraum

ADM1 hat 43 Konzentrationskanäle, die **über sechs Größenordnungen
spannen** — von Substratdosis (10 m³/d) bis Spurengasen (≈ 10⁻⁸ mol/L).
Die Spec definiert `lower = 0` für alle Konzentrationen.

Für kleine Kanäle (Methanogenese-Spuren, Acidogenese-Zwischenprodukte)
liegen die propagierten Sigma-Punkte teilweise sehr nahe an der
Null-Schranke. Per-Sigma-Bounding zieht dort einen substantiellen
Anteil der Sigmas exakt auf die Schranke → die korrigierte
Sigma-Menge clustert → `Σ Wᶜ (χᶜ - x̂)²` kollabiert auf der
betroffenen Achse → Posterior-Kovarianz untertreibt drastisch →
nächster Predict startet mit zu enger Sigma-Wolke → Filter
"blendet sich aus". Die Folge sind die ~10²-σ-Mean-Drifts auf den
schwächsten Kanälen.

Hellmann (2024) berichtet das nicht, weil ihr Modell mit n=6
Zuständen alle im kg/m³-Bereich liegt (`x_0 = [4.09, 10.52, 11.04,
2.57, 0.96, 2.02]`). Die Null-Schranke ist dort effektiv inaktiv. Die
multi-skalierten Konzentrationen von ADM1 sind eine qualitativ andere
Situation.

**Befund**: Das ist keine Implementierungs- oder
Linearisierungs-Schwäche, sondern eine strukturelle Limitation der
per-Sigma-Bound-Strategie auf bound-reichen Zustandsräumen. Für die
Dissertation ist das ein eigenständiger negativer Befund: Hellmanns
`cUKF-add` — die beste Variante in ihrem n=6-Benchmark —
**überträgt sich nicht direkt** auf ein realistisches multi-stage
ADM1 mit Konzentrationskanälen über sechs Größenordnungen.

Die Klasse bleibt im Repo als opt-in für Anwendungsfälle, wo die
Bounds praktisch inaktiv sind (z.B. einfachere AD-Modelle nach
Hellmann 2024 oder Bioreaktor-Modelle mit moderaten
Zustandsbereichen).

## Was als nächstes auf der Liste steht

| Idee | Erwarteter Effekt | Aufwand |
|---|---|---|
| **Log-Skalierung kleiner Kanäle** | Parametrisiere `x_i = exp(z_i)` für Spurengas-Kanäle. Schranke `x_i > 0` wird durch die Transformation impliziert, kein per-Sigma-Bounding nötig. Würde voraussichtlich Hellmanns cUKF-Gewinne auch auf ADM1 freischalten. | Mittel — Spec-Erweiterung (`channel.log_scale`), transparente Durchreichung durch process/obs/spec. |
| **State-augmentierte Form** (Wu 2005, Hellmann's `UKF-aug`) | Korrekte Behandlung von `Q` im Messpfad. Bei ADM1 mit kleinem `Q/P` empirisch klein, aber theoretisch sauberer; könnte die γ-Sweep-Disaster-Region (Welle 3) algorithmisch auflösen. | Mittel — `2(n+m)+1` Sigma-Punkte statt `2n+1`, mit Q-Sigma-Punkten effizient eingruppierbar (Wu 2005 §IV). |
| **Spherical-Simplex-Sigma-Punkte** (Julier 2003) | `n+2` statt `2n+1` Sigma-Punkte → ~2× zusätzlich, stackt mit der Parallelisierung. | Mittel — rekursive Simplex-Konstruktion, eigene Validierung. |
| **NLP-cUKF mit Plant-Callback** | Hellmanns echte Variante (h(χ) per QP-Iteration aus dem Plant lesen statt linearisieren). Würde die Linearisierungs-Approximation entfernen, löst aber das Sigma-Spread-Kollaps-Problem nicht — daher voraussichtlich keine Verbesserung gegenüber dem jetzigen Welle-5-Ergebnis. | Hoch — ~10 Plant-Equilibrationen pro QP-Iteration, parallelisierbar über Welle-4-Pool. |
