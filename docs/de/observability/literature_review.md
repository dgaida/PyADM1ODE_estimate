# Observability of ADM1 State Estimators — Literaturüberblick

> **Schritt 1** des Observability-Konzepts in diesem Repo: was sagt die
> publizierte Literatur darüber, welche ADM1-Zustände aus welchen
> Sensormessungen rekonstruierbar sind?
>
> Schritt 2 (Abgleich mit unserer ADM1da-Implementierung) folgt separat.

## Zentrale Frage

Ein UKF (oder ein beliebiger anderer Beobachter) kann nur diejenigen Zustände
rekonstruieren, die durch die verfügbaren Messungen **observable** sind. Die
Frage lautet daher konkret:

> Welche ADM1-Zustände sind aus welcher
> Teilmenge typischer Anlagenmessungen schätzbar? Welche bleiben prinzipiell
> unbestimmt, egal wie gut der Filter ist?

Drei Arbeiten geben darauf strukturierte Antworten:

| # | Quelle | Modellklasse | Ansatz |
|---|---|---|---|
| 1 | Hellmann et al. 2023 ([arXiv:2301.05068](https://arxiv.org/abs/2301.05068)) | ADM1-R4, ADM1-R3, ADM1-R2 (vereinfachte ADM1-Varianten) | Formale Observability-/Identifiability-Analyse (algebraisch + geometrisch) |
| 2 | Gaida et al. 2012 ([PMID:22797239](https://pubmed.ncbi.nlm.nih.gov/22797239/)) | Voller ADM1 (37 States) | Pattern Recognition / Maschinelles Lernen (statt klassischem Observer) |
| 3 | Haugen et al. 2014 ([10.1155/2014/572621](https://doi.org/10.1155/2014/572621)) | Modified Hill (4 States + 1 augmentiert) | UKF mit nur einer einzigen Messung |

Diese drei Arbeiten spannen das praktische Spektrum auf: von der einfachsten
Modellklasse mit minimaler Messung (Haugen) über die analytisch handhabbaren
Mittelklassen (Hellmann) bis zur vollen Komplexität, die nur noch
statistisch handhabbar ist (Gaida).

---

## 1. Hellmann et al. 2023 — Formale Observability für ADM1-Varianten

Die analytisch sauberste Arbeit. Untersucht systematisch, welche Modellvarianten
unter welchen Mess-Sets *strukturell* observable sind.

### Setup

**Modelle (vereinfachte ADM1-Varianten nach Weinrich & Nelles 2021):**

| Modell | States | Eigenschaften |
|---|---|---|
| ADM1-R4 | 11 | First-order Hydrolyse + Methanogenese als Summenreaktion |
| ADM1-R3 | 17 | Acetoclastische Methanogenese explizit, pH-Inhibition, NH3-Inhibition |
| ADM1-R2 | mehr | VFA-Spektrum (Acetat, Butyrat, Propionat, Valerat) explizit |

Plus zusätzliche Submodelle, in denen einzelne Modellbestandteile A bis E
(Biomasse-Decay, Gaslöslichkeit, N-Limitation, pH/NH3-Inhibition, pH-Berechnung)
isoliert weggelassen werden — zur systematischen Bewertung, welcher Teil die
Observability bricht.

**Angenommene Messungen:**

| Modellklasse | Online | Offline |
|---|---|---|
| ADM1-R4 | CH₄, CO₂ (Partialdrücke) | TS, VS, IN (Total Solids, Volatile Solids, Inorganic Nitrogen) |
| ADM1-R3 | CH₄, CO₂, pH | TS, VS, IN, (Sac nur für Algebra) |
| ADM1-R2 | CH₄, CO₂, pH, VFA (utopisch) | TS, VS, IN |

Wichtig: TS, VS, IN sind **Laborwerte** im realen Betrieb, mit Sample-and-Hold
modelliert.

### Methodik

Zwei unabhängige Ansätze, beide symbolisch:

1. **Algebraischer Ansatz** (Mathematica): Bilde Lie-Ableitungen
   $y, \dot y, \ddot y, \dots$ der Messungen, baue daraus ein
   Gleichungssystem und löse symbolisch nach den Zuständen auf. Eindeutige
   Lösung → global observable; mehrere Lösungen → lokal observable.

2. **Geometrischer Ansatz** (STRIKE_GOLDD-Toolbox in Matlab, Algorithmen
   FISPO und ORC-DF): Prüfe Rang einer Observability-Matrix $\mathcal{O}(x)$
   aus Lie-Ableitungen. Rang $n$ → lokal observable.

### Hauptergebnisse

#### ADM1-R4 (11 States): global observable mit CH₄, CO₂, TS, VS, IN

Die fünf Messungen reichen aus, alle 11 Zustände werden rekonstruiert.
Wichtig: TS, VS und IN sind **nicht ersetzbar**. Hellmann begründet das so:

> *„This follows directly from the model equations: all three states only
> appear in their corresponding differential equation. Therefore, if they
> were not available as measurements, they could not be observable because
> they would not be introduced into the system of equations via other
> measurements, regardless of the degree of time derivatives."* (S. 10)

Das ist der **Single-Channel-State-Effekt**: ein Zustand, der nur in seiner
eigenen Differentialgleichung auftaucht und sonst nirgends, ist nur durch
direkte Messung observable. Keine Zeit-Ableitungs-Magie hilft.

#### ADM1-R3 (17 States): lokal observable mit CH₄, CO₂, pH, TS, VS, IN (+ Sac)

- Mit dem algebraischen Ansatz scheitert die volle ADM1-R3, weil das Gleichungssystem
  zu komplex wird (Mathematica-Kernel stirbt).
- Mit dem geometrischen Ansatz (STRIKE_GOLDD) erfolgreich gezeigt: 17 von 17
  States lokal observable.
- Für die Algebra-Lösung der **Subvarianten** ist eine Online-Messung von
  Sac (Acetat) nötig — was im Feld nicht realistisch ist. Geometrisch zeigt
  sich die volle ADM1-R3 auch ohne Sac als observable.

#### ADM1-R2 (mehr States): **nicht observable** ohne Online-VFA

> *„In an agricultural setting, these acid measurements are not available
> online. Even when assuming them to be available online, both algorithms
> of the geometric approach failed to evaluate the respective observability
> rank condition, and thus did not allow to draw conclusive statements."* (S. 12)

Das ist eine **harte praktische Schranke**: für Modelle mit individuellem
VFA-Spektrum reicht die Sensorik einer Agraranlage nicht aus.

#### Identifizierbarkeit

Strukturell identifizierbar gezeigt:

| Modell | Identifizierbare Parameter |
|---|---|
| ADM1-R4 | Hydrolyse- und Decay-Raten (zeitvariant) |
| ADM1-R3 | $\mu_{m,ac}$, $K_{S,ac}$, $K_{I,nh3}$ |

D.h. der UKF kann nicht nur Zustände, sondern auch ausgewählte kinetische
Parameter online mitschätzen, vorausgesetzt die Messmenge bleibt unverändert.

### Strukturelle Erkenntnisse für den Filter-Bau

1. **Manche Messungen sind Pflicht, nicht „nice to have"**: jeder Zustand, der
   nur in einer einzigen DGL und sonst nirgendwo auftritt, *muss* gemessen
   werden. Im ADM1-R4 sind das genau IN, TS, VS.

2. **pH-Messung ist ein struktureller Hebel**: pH-Messung erlaubt direkte
   Berechnung von $S_{H^+}$ (über $\mathrm{pH} = -\log_{10} S_{H^+}$). Damit
   werden die drei Ionen-Zustände $S_{ion}$, $S_{ac^-}$, $S_{hco3^-}$
   **redundant** und können aus dem Zustandsvektor entfernt werden:

   > *„Measuring the pH allows to infer $S_{H^+}$ directly because these two
   > variables are linked via the negative common logarithm. […] However, as
   > $S_{H^+}$ can be directly determined from pH measurements, the states
   > $S_{ion}$, $S_{ac^-}$ and $S_{hco3^-}$ become redundant. Their respective
   > differential equations can be cut out of the system of equations."*
   > (S. 30, *„Neglecting model part E"*)

   Eine einzige pH-Sonde reduziert effektiv die Zustandsdimension um 3 und
   eliminiert ein algebraisch nichttriviales Subsystem (die Ladungsbilanz $\Phi$).

3. **Komplexität skaliert schlecht**: Rechenzeit der geometrischen Analyse für
   ADM1-R3 (17 States, FISPO): ~12.000 s; nach Weglassen einzelner
   Modellteile auf BMR3+ABC (13 States): ~12 s. Faktor 1000. Für ADM1-R2
   und höher reicht keine der beiden Methoden mehr aus.

---

## 2. Gaida et al. 2012 — Pattern Recognition statt Observer

Ein anderes Paradigma. Statt einen mathematischen Beobachter zu konstruieren,
wird die Abbildung *Messung → Zustand* als statistisches Klassifikationsproblem
behandelt.

### Setup

- **Modell:** volles ADM1 (Batstone et al. 2002), 37 States
- **Anlage:** Full-Scale Agraranlage (Simulationsstudie)
- **Messungen:** Biogasstrom, CH₄- und CO₂-Konzentrationen im Biogas, pH-Wert,
  Substratmenge je Substrattyp (Mais, Gras, Gülle, Mistfeststoffe)
- **Methodik:** Diskriminanzanalyse / Maschinelles Lernen — statische
  Mapping-Funktion Messung → Operating State

### Hauptaussage

> *„The operating state vector of the modelled anaerobic digestion process can
> be predicted with an overall accuracy of about 90%."*

D.h. **90 %** Klassifikations-Genauigkeit über *alle* ADM1-Zustände hinweg,
allein mit den genannten Standard-SCADA-Messungen.

### Interpretation und Einordnung

Was Gaida zeigt, ist nicht klassische Observability im Sinne Hellmanns. Es ist
eher: *„unter typischen Betriebsbedingungen liegt der ADM1-Zustand auf einer
niedrigdimensionalen Mannigfaltigkeit, die statistisch gut vorhersagbar ist,
obwohl er formal nicht aus den Messungen rekonstruierbar wäre."*

Drei Implikationen:

1. **Praktische Observability ≠ strukturelle Observability**: Ein System kann
   formal nicht observable sein, der Betrieb dennoch in einem Bereich liegen,
   in dem statistische Methoden ausreichen.

2. **Lernverfahren brauchen Trainingsdaten** mit verlässlichem *„Wahrheits"-
   Zustand. Das ist im Feld der knappste Faktor.

3. **Vorbild für Hybrid-Ansätze**: Klassischer Observer (UKF) für die
   strukturell observablen Zustände + statistisches Modell für den Rest.

---

## 3. Haugen et al. 2014 — UKF mit einer einzigen Messung

Die andere Extremposition: kein vollständiges ADM1, sondern ein drastisch
vereinfachtes mechanistisches Modell, und nur eine einzige Online-Messung.
Funktioniert trotzdem.

### Setup

- **Modell:** Modified Hill Model — 4 States plus eine augmentierte Größe:

  | State | Bedeutung |
  |---|---|
  | $S_{bvs}$ | Biodegradable Volatile Solids (Substratspeicher) |
  | $S_{vfa}$ | Volatile Fatty Acids (Acetat-Surrogat) |
  | $X_{acid}$ | Acidogene Biomasse |
  | $X_{meth}$ | Methanogene Biomasse |
  | $S_{vs_{in}}$ | (augmentiert) Volatile Solids im Zulauf — als Random Walk modelliert |

- **Messung:** Genau eine Online-Größe — $F_{meth}$ (Methanstrom).

- **Reaktor:** UASB-Pilotreaktor (250 L), Substrat Rindergülle.

### Methodik

Standard-UKF nach Wan/van-der-Merwe; Tuning der Diagonalelemente von $Q$
proportional zu den Zustandsmagnituden (mit Skalierungsfaktoren $m_i$ für
Feinabstimmung).

### Hauptergebnis

Trotz nur einer Messung schätzt der UKF alle 5 augmentierten Zustände
erfolgreich. Insbesondere konvergiert $S_{vs_{in}}$ aus einem absichtlich
großen initialen Fehler (20 % des wahren Werts) innerhalb von ca. 15 Tagen.

> *„The linearized reactor model, augmented with $S_{vs_{in}}$, is found
> observable at a number of typical operating points using the obsv function
> of the Matlab Control System Toolbox."* (S. 6f)

D.h. die strukturelle Observability ist im linearisierten Sinn an typischen
Betriebspunkten gegeben.

### Lehre

- Bei drastischer Modellreduktion (5 vs 17 vs 37 States) reichen wenige
  Messungen. Der Preis: das Modell trifft viele biologische Mechanismen nicht,
  und Modellfehler erscheinen als Drift in den Schätzungen (Haugen
  beobachtet das im Plot für $S_{vfa}$, Fig. 4 — *„from $t=150\,d$, there is
  a noticeable difference between the estimate and the laboratory analysis
  of $S_{vfa}$"*, S. 7f).
- Augmentierte Zustände wie ein unbekannter Zulauf $S_{vs_{in}}$ können
  schon mit einer einzigen, indirekten Messung mitgeschätzt werden, wenn das
  System linearisiert observable ist.

---

## Synthese — Wer sagt was zur Mess-Zustands-Abhängigkeit?

### Kanal-für-Kanal: welche Messung erschließt welche Zustände?

Aus den drei Quellen zusammengeführt. Erklärung der Spalten:
**direkt** = Messung tritt direkt im Output-Vektor auf, der Zustand erscheint
nur durch diese Messung im Gleichungssystem.
**indirekt** = Messung koppelt nichtlinear mit dem Zustand über ein anderes
gemessenes Signal (z.B. CH₄-Strom hängt über die Kinetik von $X_{ac}$ ab).
**konstruiert** = Messung erlaubt algebraische Reduktion (z.B. pH → $S_{H^+}$
→ Ionen-Zustände eliminiert).

| Messung | Erschließt direkt | Erschließt indirekt | Konstruiert |
|---|---|---|---|
| $p_{CH_4}$ (CH₄-Partialdruck) | $p_{CH_4}$, $S_{ch4,gas}$ | $X_{ac}$, $S_{ac}$ (über Methanogenese-Kinetik) | — |
| $p_{CO_2}$ (CO₂-Partialdruck) | $S_{co2,gas}$ | $S_{IC}$, $X_{ac}$ + $X_{h2}$ (Verhältnis CH₄/CO₂) | — |
| $\mathrm{pH}$ | — | Inhibition $I_{ac}$ (über alle X-Pools) | $S_{H^+}$ → $S_{ion}$, $S_{ac^-}$, $S_{hco3^-}$ eliminierbar |
| $S_{IN}$ (NH₄-N, Lab) | $S_{IN}$ direkt | NH₃-Inhibition (über $S_{nh3}$) | — |
| TS (Total Solids, Lab) | $X_{ash}$ direkt (via $S_{h2o}$) | partikuläre Pools $X_{ch/pr/li}$ summarisch | — |
| VS (Volatile Solids, Lab) | $X_I$ direkt | summarisch alle partikulären biologischen Pools | — |
| $S_{ac}$ (Acetat, Lab oder online) | $S_{ac}$ direkt | $X_{ac}$ (über Acetoclasten-Kinetik) | — |
| $F_{meth}$ (Methanstrom) | gewichtet alle methanogenen Beiträge | $X_{meth}$, $S_{bvs}$ (im Hill-Modell alles über Kinetik gekoppelt) | — |
| $Q_{feed}$ (Substratstrom) | augmentierte Input-Channels direkt | Verdünnungsrate $D$ als Faktor in *jeder* DGL | — |

### Mess-Sets vs. Modellgröße: Faustregel

| Modell | States | Mindest-Mess-Set für volle Observability |
|---|---|---|
| Modified Hill (Haugen) | 5 | $F_{meth}$ allein (linear lokal) |
| ADM1-R4 (Hellmann) | 11 | CH₄, CO₂, TS, VS, IN (5 Größen) |
| ADM1-R3 (Hellmann) | 17 | CH₄, CO₂, pH, TS, VS, IN (6 Größen) — strukturell, formal nachgewiesen via geometrisch |
| ADM1-R2 (Hellmann) | >17 | nicht observable bei Agrar-typischen Sensoren |
| ADM1 voll (Gaida) | 37 | klassisch nicht handhabbar — nur über ML mit ≈90 % Klassifikations-Genauigkeit |

Empirische Beobachtung: **Pro zusätzlichem unabhängigen Sensorkanal kommt
etwa eine Zustandsdimension hinzu, die der Filter wirklich trennen kann.**

Die drei Quellen liefern zusammen ein konsistentes Bild, warum viele
Zustände in der Praxis unobservable bleiben:

1. **Single-Channel-States ohne Messung sind formal unobservable.**
   Hellmann zeigt das exakt für TS, VS, IN: kein noch so trickreicher
   Filter kann sie schätzen, wenn sie nicht direkt gemessen werden.

2. **Mehrere Zustände wirken ähnlich auf wenige Messungen, der Filter
   verteilt das Innovationssignal über den Prior**, nicht über echte
   Information. Im Hill-Modell sind die Bio-Pools dadurch effektiv auf 2-3
   trennbare Dimensionen reduziert. Bei ADM1 sind es bei Standard-Sensorik
   ähnlich viele. Dies ist die *praktische* (nicht strukturelle) Schranke,
   die Gaidas Empirie reflektiert: 90 % Genauigkeit über *alle* States
   bedeutet, dass die effektive Mannigfaltigkeit niedrigdimensional ist.

### Strukturelle Hebel — was die Literatur als entscheidend identifiziert

Aus den drei Arbeiten lassen sich drei Hebel ableiten, die die Observability
qualitativ verbessern:

| Hebel | Wirkung | Quelle |
|---|---|---|
| **pH-Sonde** ergänzen | Eliminiert 3 Ionen-Zustände aus dem Filter, macht $I_{ac}$ verifizierbar | Hellmann (Part E redundant), Gaida (pH ist eine ihrer 5 Messungen) |
| **CH₄/CO₂-Trennung** (statt nur Gesamt-Q_gas) | Trennt acetoklastische ($X_{ac}$) von hydrogenotropher ($X_{h2}$) Methanogenese | Hellmann (CH₄ und CO₂ als getrennte Outputs), Gaida (beides Messungen) |
| **Lab-Werte für TS/VS/IN** | Macht die jeweiligen Zustände überhaupt erst observable | Hellmann (Single-Channel-Argument) |
| **VFA-Lab-Werte** (FOS/TAC) | Macht $S_{ac}$ direkt observable, erschließt das VFA-Spektrum | Hellmann (ADM1-R3 mit Sac), Haugen (impliziert über Frühwarn-Diskussion) |

---

## Bedeutung für die IMplementierung des UKF

- **ADM1da ist näher an ADM1-R3 als an ADM1-R4** in Komplexität (Sub-Fraktion-
  Disintegration, Inhibitionen, Ladungsbilanz), aber mit deutlich mehr States
  als ADM1-R3 (41 statt 17). Hellmanns ADM1-R2-Befund ist die Warnung:
  jenseits von ~17 Zuständen ohne VFA-Online-Sensor ist die Lage prekär.
- **Reale Agraranlagen haben oft noch weniger Sensoren als Hellmanns Szenario**
  (kein pH, kein FOS/TAC, kein NH₄-N). Die Faustregel aus der Literatur ist
  damit klar: aus ADM1da lässt sich realistisch nur **eine Untermenge** der
  Zustände schätzen.
- **Hebel für die Zukunft** sind in Reihenfolge des Nutzens:
  pH-Sonde > CH₄/CO₂-Anteilssensor > tägliche FOS/TAC-Lab-Werte.
  Jeder davon ist in der Literatur als struktureller Sprung markiert, nicht
  als gradueller Zuwachs.

---

## Quellen

1. Hellmann, S., Hempel, A.-J., Streif, S., Weinrich, S.
   *Observability and Identifiability Analyses of Process Models for
   Agricultural Anaerobic Digestion Plants.* 24th Intl. Conference on
   Process Control, 2023. arXiv:2301.05068v3.
2. Gaida, D., Wolf, C., Meyer, C., et al.
   *State estimation for anaerobic digesters using the ADM1.*
   Water Science and Technology, 66(5):1088–1095, 2012.
   [PMID: 22797239](https://pubmed.ncbi.nlm.nih.gov/22797239/).
3. Haugen, F., Bakke, R., Lie, B.
   *State Estimation and Model-Based Control of a Pilot Anaerobic
   Digestion Reactor.* Journal of Control Science and Engineering, 2014,
   Article ID 572621. [DOI: 10.1155/2014/572621](https://doi.org/10.1155/2014/572621).

Weiterführend zitiert:

- Weinrich, S., Nelles, M. *Systematic simplification of the anaerobic
  digestion model no. 1 (ADM1) — model development and stoichiometric
  analysis.* Bioresource Technology, 333:125124, 2021.
- Villaverde, A. F., Barreiro, A., Papachristodoulou, A.
  *Structural identifiability of dynamic systems biology models.*
  PLoS Computational Biology, 12(10):e1005153, 2016.
  (Hintergrund zur STRIKE_GOLDD-Toolbox.)
