# Realistische Testbedingungen — das „Realismus-Fundament"

Um zu beurteilen, ob ein Zustandsschätzer *für das Projektziel gut genug* ist
(Zustandsschätzung an einer realen landwirtschaftlichen Biogasanlage, letztlich
für Bedienempfehlungen), muss das Twin-Experiment unter Bedingungen nahe einer
realen Anlage laufen. Diese Seite definiert diese Bedingungen — und vor allem,
**wie sie gewählt werden**.

## Leitprinzip: plant-agnostisch und literatur-basiert, nicht gefittet

Die Sensorik der realen Testanlage ist **nicht final** (es werden noch Sensoren
ergänzt), und das Ziel ist eine Methode, die **allgemein für Biogasanlagen**
funktioniert, nicht für einen Datensatz. Deshalb werden Rauschen und
Modellfehler **nicht an die aktuelle Datenlage angepasst**:

* **Messrauschen** hängt an *Sensortypen* als generische Instrumenten-Genauigkeit
  — ergänzt/entfernt man einen Sensor, ändern sich die anderen nicht, und die
  Bedingungen übertragen sich auf andere Anlagen.
* **Modellfehler** — die Lücke zwischen ADM1 und Realität — **lässt sich nicht
  aus Betriebsdaten bestimmen** (er ist mit Eingangs- und Messfehler vermengt).
  Er wird daher aus der **Kinetik-Parameter-Unsicherheit der ADM1-Literatur**
  gesetzt, als multiplikative Störung.

Die Werte liegen in [`estimation/realism.py`](../api/index.md) und werden vom
`--realistic`-Preset des Twins angewendet.

## Modellfehler: Kinetik-Parameter-Unsicherheit

Der Modellfehler ist eine **multiplikative lognormale** Störung der biologischen
Kinetiken des Filters, während die Truth die „echten" Werte behält:

$$ k_\text{Filter} = k_\text{nominal}\cdot e^{\mathcal N(0,\sigma)},\qquad
   \sigma = 0.25 $$

auf die Raten-/Affinitäts-Prefixe `k_dis`, `k_hyd`, `k_m_`, `k_dec`, `K_S`.
Physikalische Gleichgewichtskonstanten (`K_a`, `K_w`, `K_H`) und Stöchiometrie
(`Y_*`, `f_*`) bleiben unangetastet.

Begründung von Form und Größe:

| Evidenz | Wert |
|---|---|
| ADM1-Sensitivitätsanalysen stören Kinetik-/Stöchiometrie-Parameter | ~10 % |
| Die gas-relevanten *sensiblen* Parameter sind genau Zerfall / Disintegration / Hydrolyse / `k_m` / `K_S` | — |
| UKF-für-AD-Studien injizieren bewusst einen Plant-Model-Mismatch auf die Ratenkonstanten | ~28–30 % |
| Monte-Carlo-Unsicherheitsstudien behandeln Kinetik-Parameter als **lognormal**; Hydrolyse am unsichersten, Gasstrom bleibt vergleichsweise robust | lognormal, CoV bis ~50 % |

`σ = 0.25` (CoV ≈ 25 %) liegt zwischen Sensitivitätsbereich (10 %) und
Plant-Model-Mismatch-Ende (30 %). Es ist der *übertragbare* Stellvertreter für
„das Modell ist nicht zu 100 % die Realität": die **beobachteten** Größen (Gas)
bleiben verfolgbar, während **unkorrigierte** Zustände wegdriften — genau das
realistische Verhalten, das der Schätzer aushalten muss.

## Messrauschen (1-σ, pro Sensortyp)

Generische Instrumenten-Genauigkeit, **relativ** außer wo absolut vermerkt:

| Sensor | Rauschen (1-σ) | Basis |
|---|---|---|
| Biogas-Strom `q_gas` | 3 % rel. | Biogas-Volumen-Metrologie, ~3 % erweiterte Unsicherheit, Drift < 0.15 %/24 h |
| Methan-Strom `q_ch4` | 4 % rel. | `q_gas` × NDIR CH₄ (≈ 0.7–1 % abs.) kombiniert |
| CO₂-Strom `q_co2` | 4 % rel. | `q_gas` × NDIR CO₂ |
| pH | 0.05 (absolut, pH-Einheiten) | Glaselektroden-Genauigkeit |
| VFA / FOS | 8 % rel. | FOS/TAC-Titration |
| TS / VS | 3 % rel. | gravimetrisch |
| Substratdosis | 3 % rel. | Dosierwaage |

Manche UKF-für-AD-Studien blähen `R` zur Robustheit um ~1.5× auf; dieser Faktor
ist als `R_INFLATION` verfügbar (Default 1.0).

## Abtastraten (reale Anlagen-Kadenz)

* **Online (im Twin stündlich, an der Anlage Sekunden):** `q_gas`, pH,
  Substratdosen, Füllstände, BHKW, Temperaturen.
* **Täglich:** Gaszusammensetzung CH₄ / CO₂ (NDIR). O₂ und H₂S werden praktisch
  ebenfalls täglich gemessen, sind aber **keine ADM1da-Zustände** → nur
  Monitoring, *nicht assimiliert*.
* **Laborkadenz:** TS, VFA (gegated, z. B. VFA alle 12 h).

## Bewusst *nicht* enthalten (und warum)

* **Anlagenspezifische Gas-Ableitungskette.** An der realen Anlage ist `Q_gas`
  kein Flowmeter-Wert, sondern wird aus Gasblasen-Füllstandsänderungen (ΔV,
  differenziert → rauschverstärkt) + BHKW-Verbrauch + Fackel rekonstruiert.
  Diese Fehlerstruktur ist anlagenspezifisch; das Fundament nutzt stattdessen
  die generische Flowmeter-Spezifikation, damit die Bedingungen übertragbar
  bleiben. (Ein anlagenspezifisches Rauschmodell kann beim Bewerten *dieser*
  Anlage darübergelegt werden.)
* **O₂ / H₂S.** Keine ADM1da-Zustände — können den UKF nicht informieren.
* **σ oder R an die aktuelle Datenlage fitten** — würde zur unvollständigen
  Sensorik hin verzerren.

## Verwendung

```bash
# Realismus-Preset (Modellfehler + Sensorrauschen + tägliche Gasanalytik):
python examples/run_twin_experiment.py --realistic --warmup-days 30 --duration-days 14
# Reduzierter Zustandsvektor unter realistischen Bedingungen:
python examples/run_twin_experiment.py --realistic --state-blocks methanogenesis charge_balance
```

Einzel-Flags (`--model-error-std`, `--gas-noise-std`) überschreiben das Preset
weiterhin für Sensitivitäts-Sweeps.

## Bewertungs-Harness & Provenance

Der Schätzer wird über ein **gepaartes Monte-Carlo-Ensemble** bewertet
(`monte_carlo_eval.py`): pro Seed eine unabhängige Modellfehler-Realisierung
(dieses Fundament) plus Rauschen und Prior-Störung, und **alle Kandidaten sehen
dieselbe Welt** (Open-Loop-Modell, Roh-Sensor-Floor, UKF voll-41, UKF A+D-Kern,
und **A+D-Kern mit Known-Input** `adcore_ki` — die Substratdosierung wird aus der
gemessenen Dosis vorgesteuert statt geschätzt, sodass der Filter von
Futterwechseln nicht „überrascht" wird; das senkt die Feed-Change-NIS von ~10³
auf das Konsistenzband).
Metrik: entscheidungs-gewichtete per-Block-NRMSE (Block-**Median**, **konvergierte
zweite Hälfte**, um den UKF-Einschwingvorgang nicht zu bestrafen; `charge_balance`
separat berichtet, da seine nahe-null Ionen schlecht konditioniert sind), dazu
Kalibrierung (mittlere NIS, 2σ-Coverage) und gepaarte Gewinnraten.

**Welcher Schätzer / welche Version** getestet wurde, wird pro Lauf in
`output/mc_eval_meta.txt` festgehalten — bei archivierten Ergebnissen mitnehmen,
da `output/` gitignored ist. Erfasst werden: Zeitstempel, die git-Commits von
PyADM1ODE_estimate und pyadm1 (mit `+dirty`-Flag bei uncommitteten Änderungen),
die pyadm1-Version, die Filter-Klasse + Parameter, das Modellfehler-σ und der
Sensor-Zeitplan. Der getestete Schätzer ist der **Square-Root-UKF**
`estimation.filters.sr_ukf.UnscentedKalmanFilter` (α = 1.0, β = 2.0, κ = 0.0,
γ = canonical √(n+λ)); der reduzierte **„A+D-Kern"** ist derselbe Filter,
eingeschränkt auf `--state-blocks methanogenesis charge_balance` (18 von 41
ADM1-Zuständen).

## Quellen

* ADM1-Sensitivitätsanalyse — [WIT Transactions (ADM1 local SA)](https://www.witpress.com/elibrary/wit-transactions-on-ecology-and-the-environment/258/38278);
  [Surrogate-based global SA of ADM1 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0301479720313815).
* UKF / Plant-Model-Mismatch für AD — [Comparison of UKF designs for AD (arXiv:2310.15958)](https://arxiv.org/html/2310.15958);
  Haugen et al. 2014, *State Estimation … Pilot AD Reactor* ([DOI:10.1155/2014/572621](https://doi.org/10.1155/2014/572621)).
* ADM1 Monte-Carlo / lognormale Unsicherheit — [Uncertainty analysis of a simplified AD model (IWA WST 92(4):610)](https://iwaponline.com/wst/article/92/4/610/108810/Uncertainty-analysis-of-a-simplified-anaerobic);
  [Probabilistic ADM1 simulation of biogas (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S1369703X23000050).
* Sensor-Genauigkeit — NDIR-Biogasanalysatoren ([Olythe](https://www.olythe.io/analyzers/biogas-analyzer/), [Dynament](https://dynament.com/application/biogas-monitoring/));
  Biogas-Volumen-Metrologie ([PMC12693810](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12693810/)).
