# Sensor-Zustand-Abhängigkeiten in ADM1da

Ziel: pro Mess-Kanal belegen, welche der 41 ADM1da-Zustände direkt im
Messmodell auftauchen und welche nur indirekt über die Dynamik des
Modells beobachtbar werden. Daraus folgt die Begründung, welche Zustände
ein UKF realistisch schätzen kann, wenn nur bestimmte Sensoren verfügbar
sind.

Die Messungen sind nach **Verfügbarkeit** getrennt, denn ein Online-
Sensor liefert mehrmals am Tag Messungen, eine Lab-Analytik nur alle paar
Tage und ein Forschungs-Verfahren vielleicht ein einziges Mal pro Jahr.

## Methodik

Ein Zustand $x_i$ ist aus einer Messung $y = h(x)$ überhaupt nur dann
beobachtbar, wenn entweder

1. **direkt** $x_i$ explizit in $h(x)$ auftritt (Ableitung ungleich Null im  
   Messmodell), oder  
2. **indirekt** $x_i$ in der rechten Seite $\dot x_j = f_j(x)$ eines  
   direkt beobachtbaren $x_j$ auftritt, dann liefert $\dot y$ Information
   über $x_i$. Dieser Mechanismus wird iteriert: nach $k$ Schritten taucht
   $x_i$ in $y^{(k)}$ auf, falls eine Kette der Länge $k$ durch die
   Jacobi-Matrix existiert (Lie-Ableitungen, siehe Hellmann et al. 2023).

Praktisch heißt das:

* **D = direkt** — sofortige Beobachtbarkeit in einem einzigen  
  Update-Schritt.  
* **I₁ = 1-Schritt-indirekt** — Beobachtbarkeit nach mindestens einem  
  Zeitschritt mit Dynamik dazwischen.  
* **I₂+ = k-Schritt-indirekt** — nur über sehr genaue Modelldynamik und  
  lange Beobachtungsdauer rekonstruierbar. In der Praxis meist nicht
  trennbar von Modellfehlern.  
* **C = korrelativ** — kein direkter Modellausdruck, aber empirisch  
  gekoppelt (z.B. Leitfähigkeit ↔ ionische Stärke).

Die folgenden Tabellen zitieren explizit die Code-Stellen in
`pyadm1/core/adm1.py` (Datei-Zeilennummern beziehen sich auf den Stand
v0.3.4).

## Notation und Symbole

Glossar der wichtigsten Größen, damit die Gleichungen und Tabellen weiter
unten lesbar bleiben. Detail-Indizes für die 41 Zustände stehen in
[Theorie → ADM1da-Modell](../theory/adm1.md).

### Zustandsklassen

| Präfix | Bedeutung | Typische Einheit |
|---|---|---|
| $S_\bullet$ | gelöste Spezies (Solute) im Liquid | kg COD/m³ oder kmol/m³ |
| $X_\bullet$ | partikuläre Spezies (Biomasse, Feststoff-Pools) | kg COD/m³ |
| $p_{\text{gas},\bullet}$ | Partialdruck einer Gasphase-Spezies | bar |
| $p_{\text{total}}$ (= pTOTAL) | Gesamt-Gasdruck im Headspace, State-Index 40 | bar |

### Subskripte (Stoff-Klassen)

| Kürzel | Bedeutung |
|---|---|
| su | Zucker / monosaccharides (sugars) |
| aa | Aminosäuren (amino acids) |
| fa | langkettige Fettsäuren / LCFA |
| va | Valeriansäure / Valerat (C5) |
| bu | Buttersäure / Butyrat (C4) |
| pro | Propionsäure / Propionat (C3) |
| ac | Essigsäure / Acetat (C2) |
| h2 | Wasserstoff |
| ch4 | Methan |
| co2 | anorganischer Kohlenstoff (= S_IC) |
| nh4 | Ammonium-Stickstoff (= S_IN) |
| nh3 | freies Ammoniak |
| hco3 | Bikarbonat HCO₃⁻ |
| _ion (Suffix) | dissoziierte Ionenform einer VFA (z.B. $S_{ac\_ion}$ = Acetat-Anion) |
| cation, anion | Summe aller starken Kationen bzw. Anionen [kmol/m³] |
| I | Inert (gelöst $S_I$ oder partikulär $X_I$) |
| ch / pr / li | Kohlenhydrate / Proteine / Lipide |
| PS / PF / S | Pool-Klasse: slow-disintegrierend / fast-disintegrierend / hydrolysierbar |
| c4 | C4-Säure-Abbauer (Valerat/Butyrat-Verwerter) |
| bac | Bakterien-Biomasse (allgemeiner Sammelbegriff) |

### Beobachtbare Größen (Outputs / Mess-Modelle)

| Symbol | Bedeutung |
|---|---|
| $Q_{\text{gas}}$ | gesamter Biogas-Volumenstrom [m³/d] |
| $Q_{\text{CH4}}$, $Q_{\text{CO2}}$ | Methan- bzw. CO₂-Volumenstrom [m³/d] |
| $P_{\text{gas}}$ | Summe der trockenen Partialdrücke $p_{\text{gas,h2}} + p_{\text{gas,ch4}} + p_{\text{gas,co2}}$ [bar] |
| $p_{\text{total}}$ | Gesamt-Gasdruck als integrierter Modell-Zustand (Idx 40) |
| $p_{\text{gas,h2o}}$ | Wasserdampf-Sättigungsdruck — temperaturabhängiger Parameter, **kein** Zustand |
| pH | $-\log_{10}(S_H)$ mit $S_H$ aus Newton-Lösung der Ladungsbilanz |
| VFA | Summe flüchtige Fettsäuren als HAc-Äquivalent [g HAc/L] |
| TAC | totale Alkalinität (Titration bis pH 5) [g CaCO₃/L] |
| FOS/TAC | Verhältnis VFA zu TAC — Nordmann-Frühwarn-Indikator |
| $Q_{\text{solid}}$, $Q_{\text{liquid}}$ | augmentierte Substrat-Zufuhrraten [m³/d] — keine ADM1-Zustände, sondern erweiterter State-Vector |

### Reaktionsraten $\rho$ (im Code `Rho_*`)

| Symbol | Bedeutung |
|---|---|
| $\rho_{\text{dis,PS}}$, $\rho_{\text{dis,PF}}$ | Disintegrations-Raten ($X_{PS} \to X_S$, $X_{PF} \to X_S$) |
| $\rho_{\text{hyd,ch/pr/li}}$ | Hydrolyse-Raten ($X_S \to S$) |
| $\rho_{\text{su}}, \rho_{\text{aa}}, \rho_{\text{fa}}, \rho_{\text{c4}}, \rho_{\text{pro}}, \rho_{\text{ac}}, \rho_{\text{h2}}$ | Substrat-Aufnahme-Raten (Monod-Kinetik × Biomasse × Inhibition) |
| $\rho_{\text{dec,}\bullet}$ | Biomasse-Decay-Raten |
| $\rho_{A,\bullet}$ | Säure-Base-Reaktionsraten (sehr schnell, $k_{AB} = 10^8$) |
| $\rho_{T,\text{h2}}, \rho_{T,\text{ch4}}, \rho_{T,\text{co2}}$ | Gas-Liquid-Transferraten (Henry-Gesetz) |
| $\rho_{T,11}$ | Bulk-Gas-Ausstrom aus dem Headspace |

### Inhibitions-Faktoren $I$ (Werte in [0,1])

| Symbol | Bedeutung |
|---|---|
| $I_{\text{pH,aa}}, I_{\text{pH,ac}}, I_{\text{pH,h2}}$ | pH-Inhibition (Hill-Form, Exponent 1-3 je Pool) |
| $I_{\text{IN}}$ | N-Limitation: $S_{IN}/(K_{S,IN}+S_{IN})$ |
| $I_{\text{h2,fa/c4/pro}}$ | H₂-Inhibition für LCFA-, C4-, Propionat-Abbauer |
| $I_{\text{nh3}}, I_{\text{nh3,pro}}$ | NH₃-Inhibition für Acetoclasten und Propionat-Abbauer |
| $I_{\text{HAc}}, I_{\text{HPr}}$ | Inhibition durch undissoziierte Essig-/Propionsäure |
| $I_{\text{co2,h2}}$ | CO₂-Hemmung der H₂-produzierenden Pfade |

### Wichtige Parameter

| Symbol | Bedeutung |
|---|---|
| $k_{m,\bullet}$, $K_{S,\bullet}$ | maximale Monod-Rate und Halbsättigungs-Konstante |
| $k_{\text{dec,}\bullet}$ | Decay-Rate je Biomasse-Pool [d⁻¹] |
| $k_{\text{dis,PS/PF}}$, $k_{\text{hyd,ch/pr/li}}$ | Disintegrations- / Hydrolyse-Raten [d⁻¹] |
| $Y_\bullet$ | Biomasse-Ausbeute pro Substrat-COD [-] |
| $K_{a,\bullet}$ | Säure-Dissoziationskonstanten |
| $K_{H,\text{ch4/co2/h2}}$ | Henry-Konstanten (Gas-Löslichkeit) |
| $K_w$ | Ionenprodukt des Wassers |
| $k_{L}a$ | Gas-Liquid-Transferkoeffizient [d⁻¹] |
| $k_p$ | Gas-Ausstrom-Koeffizient (pipe-flow) |
| $V_{\text{liq}}$, $V_{\text{gas}}$ | Liquid- bzw. Gasphase-Volumen [m³] |
| $T_{\text{ad}}$ | Betriebstemperatur (adiabatisch) [K] |
| $R$, $RT$ | Gaskonstante / Produkt $R \cdot T_{\text{ad}}$ |
| $M_{\text{HAc}}$ | 60 g/mol (Acetat-Molmasse) für VFA-Umrechnung |
| 64, 112, 160, 208 | COD-Äquivalente für Acetat / Propionat / Butyrat / Valerat [g COD/mol] |
| $f_{\bullet}$ | Stöchiometrische Produkt-Fraktionen (z.B. $f_{\text{ac,su}} = 0.41$) |
| $N_{\text{bac}}, N_{\text{aa}}, N_I$ | Stickstoff-Anteile pro Biomasse-, Aminosäure-, Inert-Pool [kmol N/kg COD] |
| $D_{\text{in}}, D_{\text{out}}$ | Verdünnungsraten $Q/V_{\text{liq}}$ [d⁻¹] |
| $S_H$ | $H^+$-Konzentration aus Newton-Lösung der Ladungsbilanz |

## Messmodell-Gleichungen (aus dem Code)

### Gas-Volumenströme

Aus [`ADM1.calc_gas`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py#L431-L467):

```python
# Z. 451-454
q_gas = max(
    k_p * (p_total_wet - p_ext) / (RT/1000 * NQ) * V_liq,
    0.0,
)
# Z. 456-460
p_gas    = pi_Sh2 + pi_Sch4 + pi_Sco2          # = p_gas_h2 + p_gas_ch4 + p_gas_co2
q_ch4    = q_gas * pi_Sch4 / (p_gas + p_gas_h2o)
q_co2    = q_gas * pi_Sco2 / (p_gas + p_gas_h2o)
```

| Observable | Direkt abhängig von Zustand(en) | Idx |
|---|---|---|
| $Q_{\text{gas}}$ | $p_{\text{total}}$ | 40 |
| $P_{\text{gas}}$ | $p_{\text{gas,h2}}$, $p_{\text{gas,ch4}}$, $p_{\text{gas,co2}}$ | 37, 38, 39 |
| $Q_{\text{CH4}}$ | $p_{\text{total}}$, $p_{\text{gas,h2}}$, $p_{\text{gas,ch4}}$, $p_{\text{gas,co2}}$ | 37, 38, 39, 40 |
| $Q_{\text{CO2}}$ | $p_{\text{total}}$, $p_{\text{gas,h2}}$, $p_{\text{gas,ch4}}$, $p_{\text{gas,co2}}$ | 37, 38, 39, 40 |

### pH

Aus [`ADM1._calc_ph`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py#L1031-L1056):

```python
# Z. 1046-1048
vfa_anions = S_ac_ion/64 + S_pro_ion/112 + S_bu_ion/160 + S_va_ion/208
fixed      = S_cation - S_anion + (S_nh4 - S_nh3) - S_hco3 - vfa_anions
# Newton: f(S_H) = fixed + S_H − K_w/S_H = 0
pH = -log10(S_H)
```

| Observable | Direkt abhängig von Zustand(en) | Idx |
|---|---|---|
| pH | $S_{nh4}, S_{nh3}, S_{hco3}, S_{ac\_ion}, S_{pro\_ion}, S_{bu\_ion}, S_{va\_ion}, S_{cation}, S_{anion}$ | 10, 36, 35, 34, 33, 32, 31, 29, 30 |

### VFA und TAC

Aus
[`Digester._compute_indicators`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/components/biological/digester.py#L338-L400):

```python
# Z. 362-367 VFA in g HAc-äq / m³
vfa = M_HAc * (S_ac/64 + S_pro/112 + S_bu/160 + S_va/208)

# Z. 389-399 TAC in g CaCO3 / m³ (Endpunkt pH 5)
# kombiniert: ungebundenes NH3/NH4, HCO3/CO2, alle VFA-Ionen, S_cation/S_anion.
```

| Observable | Direkt abhängig von Zustand(en) | Idx |
|---|---|---|
| VFA-Summe | $S_{va}, S_{bu}, S_{pro}, S_{ac}$ | 3, 4, 5, 6 |
| VFA-einzeln (HPLC) | je nach Säure einzeln | je 3, 4, 5 oder 6 |
| TAC | $S_{va}, S_{bu}, S_{pro}, S_{ac}, S_{co2}, S_{nh4}, S_{cation}, S_{anion}, S_{va\_ion}, S_{bu\_ion}, S_{pro\_ion}, S_{ac\_ion}, S_{hco3}, S_{nh3}$ | 3-6, 9, 10, 29-36 |
| FOS/TAC | Vereinigung von VFA und TAC | dito |

## A — Online-Messungen

Kontinuierlich verfügbar, sekündliche bis minütliche Innovation. Diese
Liste deckt typische SCADA-Sensorik einer landwirtschaftlichen
Biogasanlage plus optionale Erweiterungen ab.

### A.1 Standard-SCADA

| Sensor | Was gemessen | direkter Bezug zu ADM1da-States | Klasse |
|---|---|---|---|
| Q_gas (Gasstrom-Sensor, thermisch/Turbine) | Biogas-Volumenstrom [m³/d] | pTOTAL (40) | D |
| Gasspeicher-Level (% Füllung) | Speicher-Δ über Zeit → Beitrag zu Q_gas | wie Q_gas | rekonstruktiv |
| Flare on/off | Status; Boundary für Q_gas-Bilanz | — (Boundary) | — |
| p_head (Kopfdruck) | Headspace-Druck [mbar/bar] | pTOTAL (40) | D |
| T_reactor (PT100 im Fermenter) | Fermenter-Temperatur | — (Boundary, skaliert Kinetik via Arrhenius) | — |
| T_ambient, T_inlet | Heat-Balance-Boundary | — | — |
| Substrat-Wägezelle / Pumpenzähler | $-\Delta W$ → Q_feed (Feststoff oder flüssig) | $Q_{solid}$ / $Q_{liquid}$ (augmentiert) | D |
| BHKW Q_ch4-Verbrauch | Methanverbrauch pro CHP [m³/h] | pTOTAL + p_gas_ch4 (40, 38) via $\eta$ | I₁ |
| BHKW P_el (Wirkleistung) | Elektrische Leistung [kW] | wie Q_ch4, abgeleitet via $\eta_{el}$ | I₁ |
| Stirrer-Strom/-Leistung | Rührwerk — Viskositätsproxy | — (keine ADM1-Kopplung) | — |
| Heating power | Heizleistung [kW] | — (Boundary) | — |

### A.2 Online-Gas-Analytik (NDIR-Multi-Gas-Sensor)

| Sensor | Was gemessen | direkter Bezug zu ADM1da-States | Klasse |
|---|---|---|---|
| CH₄-Anteil im Biogas (NDIR) | $p_{gas,ch4}/p_{gas}$ [%] | $p_{gas,ch4}$ (38) | D |
| CO₂-Anteil im Biogas (NDIR) | $p_{gas,co2}/p_{gas}$ [%] | $p_{gas,co2}$ (39) | D |
| H₂-Anteil im Biogas (TCD/elektrochemisch) | $p_{gas,h2}/p_{gas}$ [ppm-%] | $p_{gas,h2}$ (37) | D |
| O₂-Anteil im Biogas | Qualitätskontrolle (Lufteintrag) | — (kein ADM1-State) | — |
| H₂S-Anteil im Biogas | Schwefelwasserstoff [ppm] | — (kein ADM1-State; entschwefelt) | — |

### A.3 Online-Liquid-Phase (Premium-Ausstattung)

| Sensor | Was gemessen | direkter Bezug zu ADM1da-States | Klasse |
|---|---|---|---|
| pH-Sonde (Glaselektrode, ISFET) | liquid pH | 9 Ladungsbilanz-Zustände (10, 29-36) | D |
| ORP / Redox-Potential | Reduktionspotential [mV] | $S_{h2}$ (7), Methanogen-Aktivität | C |
| Conductivity (EC) | elektrische Leitfähigkeit [µS/cm] | $S_{cation} + S_{anion}$ + alle Ionen-Spezies | C |
| Online-TAC (FOS-Titrator) | Alkalität [g CaCO₃/L] | wie Lab-TAC, siehe unten | D |
| Online-VFA (HPLC online) | Einzelne Säuren [g/L] | $S_{va}, S_{bu}, S_{pro}, S_{ac}$ einzeln (3-6) | D |
| Online-NH₄⁺ (ISE) | Ammonium [g/L] | $S_{nh4}$ (10) | D |
| Level / V_liq | Füllstand | — (im aktuellen Modell konstant) | — |

## B — Lab-Analytik

Periodisch (täglich, wöchentlich, monatlich). Liefert Stand-und-Halt-
Werte. UKF nutzt sie als **gated observations** zu sporadischen Zeitpunkten.

### B.1 Standard-AD-Routine (typ. wöchentlich)

| Lab-Größe | Methode | direkter Bezug zu ADM1da-States | Klasse |
|---|---|---|---|
| pH (Lab) | Glaselektrode | wie Online-pH | D |
| VFA (FOS-Titration Nordmann) | titrimetrisch | VFA-Summe: $S_{va}, S_{bu}, S_{pro}, S_{ac}$ als Summe | D-Summen |
| TAC (Alkalitäts-Titration) | titrimetrisch | wie oben (TAC-Eintrag) | D |
| FOS/TAC | abgeleitet | Verhältnis | D |
| TS (Trockensubstanz) | Trockenschrank | $X_I$ direkt, partikuläre Pools summarisch | I₁(Σ) |
| VS / oTS (Glühverlust) | Glühen 550 °C | analog TS, ohne mineralische Asche | I₁(Σ) |
| NH₄-N | Indophenol oder IC | $S_{nh4}$ (10) | D |
| Leitfähigkeit | LF-Messgerät | $S_{cation} + S_{anion}$ Summe | C |

### B.2 Erweiterte Analytik (typ. wöchentlich bis monatlich)

| Lab-Größe | Methode | direkter Bezug zu ADM1da-States | Klasse |
|---|---|---|---|
| Einzel-VFA (Acetat, Propionat, n-/iso-Butyrat, n-/iso-Valerat) | HPLC oder GC | $S_{ac}$ (6), $S_{pro}$ (5), $S_{bu}$ (4), $S_{va}$ (3) jeweils einzeln | D je Säure |
| TKN (Total Kjeldahl Nitrogen) | Kjeldahl-Aufschluss | $S_{nh4} + S_{nh3} + N$-Anteil aller Biomasse- und Protein-Pools | D-Summe über N |
| CSB (COD) gesamt | Photometrisch | Summe aller COD-Spezies | I₁(Σ) |
| CSB gelöst (filtriert 0.45 µm) | gleich | nur gelöste COD: $S_{su, aa, fa, va, bu, pro, ac, h2, ch4, I}$ Summe | I₁(Σ-gelöst) |
| CSB partikulär | CSB_total − CSB_gelöst | partikuläre Pools Summe | I₁(Σ-part.) |
| Kationen-Inventur (Na⁺, K⁺, Ca²⁺, Mg²⁺) | IC oder ICP-OES | $S_{cation}$ (29) | D |
| Anionen-Inventur (Cl⁻, SO₄²⁻, PO₄³⁻, NO₃⁻) | IC | $S_{anion}$ (30) | D |
| BSB5 (BOD5) | 5-Tage-Atmung | biologisch abbaubare COD-Fraktion | I₂(Σ) |

### B.3 Substrat-Analytik (für s_in der Influent-Bilanz)

| Lab-Größe | Was es liefert | direkter Bezug zu ADM1da-Inputs |
|---|---|---|
| Substrat-TS, -oTS, -CSB | Charakterisierung pro Substrat | s_in[i] für partikuläre / inerte Pools, je Substrat-Slot |
| Substrat-NH₄-N, -TKN | Stickstoff-Beitrag | s_in[10], s_in[11] |
| Substrat-Disintegrations-Fraktionen | aus Profilen oder Lab-BMP-Test | f_ch_xc, f_pr_xc, f_li_xc, f_xi_xc, f_si_xc — fließt in Calibration-Artefakt, nicht State |

Substrat-Analytik beeinflusst nicht den Filter-State direkt, sondern den
**Influent-Vektor** $s_{in}$. Das heißt sie schiebt die Anfangsbedingung der
Mass-Balance, nicht den Zustand selbst. Falsch parametrisierter Influent
führt zu chronischen Filter-Biases.

## C — Forschungs- / Spezialverfahren (für Vollständigkeit)

| Methode | Was gemessen | direkter Bezug zu ADM1da-States | Anmerkung |
|---|---|---|---|
| qPCR / 16S-rRNA-Sequenzierung | Biomasse-Populationen quantitativ | $X_{su}, X_{aa}, X_{fa}, X_{c4}, X_{pro}, X_{ac}, X_{h2}$ (22-28) jeweils einzeln | aufwändig (~100 €/Probe), Wochen Vorlauf — produktiv selten |
| GC-MS für LCFA / Spuren-VFA | Einzel-LCFA-Spezies, Spuren-Säuren | $S_{fa}$ (2) und feinere Subskala | Forschung |
| In-situ-H₂-Sonde (Headspace) | Online-$p_{gas,h2}$ | $p_{gas,h2}$ (37) | siehe A.2 |
| Trace-Element-Analytik (Fe, Co, Ni, Mo, Se) | für Kinetik wichtig | — (kein ADM1-State, beeinflusst $k_{m,*}$) | als Kalibrierungs-Anhaltspunkt |
| H₂S gelöst, Sulfat, Sulfid | Schwefel-Inhibition | — (nicht in ADM1da) | für Hill-erweiterte Modelle |
| BMP-Test (Batch Methane Potential) | spezifische Methanausbeute eines Substrats | $f_{*}$-Fraktionen, k_dis je Substrat | Eingang ins Calibration-Artefakt |

## ODE-Kopplungsstruktur — wer hängt von wem ab

Die rechte Seite jeder Differentialgleichung enthält weitere Zustände.
Die folgenden Abhängigkeiten sind aus
[`ADM_ODE`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py#L540-L962) extrahiert:

### Gas-Phase-DGLs (Z. 914-918)

```python
diff_p_h2  = Rho_T_h2  * RT/16 − p_gas_h2/pTOTAL  * Rho_T_11
diff_p_ch4 = Rho_T_ch4 * RT/64 − p_gas_ch4/pTOTAL * Rho_T_11
diff_p_co2 = Rho_T_co2 * RT    − p_gas_co2/pTOTAL * Rho_T_11
diff_pTOT  = RT/16 * Rho_T_h2 + RT/64 * Rho_T_ch4 + RT * Rho_T_co2 − Rho_T_11
```

mit (Z. 712-723):

```python
Rho_T_h2  = k_L_a * (S_h2  - 16*p_gas_h2 /(RT*K_H_h2 )) * V_liq/V_gas
Rho_T_ch4 = k_L_a * (S_ch4 - 64*p_gas_ch4/(RT*K_H_ch4)) * V_liq/V_gas
Rho_T_co2 = k_L_a * (S_co2_free - p_gas_co2/(RT*K_H_co2)) * V_liq/V_gas
S_co2_free = max(S_co2 - S_hco3, 0)
Rho_T_11  = k_p * (pTOTAL + p_gas_h2o − p_ext) * V_liq/V_gas
```

**→ Eine Gasstrom-Messung erschließt in einem Zeitschritt:**
pTOTAL (D), $p_{gas,h2/ch4/co2}$ (1-Schritt-indirekt aus $\dot p_{TOT}$),
$S_{h2}, S_{ch4}, S_{co2}, S_{hco3}$ (alle via $\rho_T$-Terme).

### Methanstrom-Kette (S_ch4-DGL, Z. 828-834)

```python
diff_S_ch4 = D_in·s_in[8] - D_out·S_ch4 + (1-Y_ac)·Rho_ac + (1-Y_h2)·Rho_h2
             - V_gas/V_liq · Rho_T_ch4
```

mit $\rho_{ac} = k_{m,ac}\cdot S_{ac}/(K_{S,ac}+S_{ac})\cdot X_{ac}\cdot I_{ac}$
und $\rho_{h2} = k_{m,h2}\cdot S_{h2}/(K_{S,h2}+S_{h2})\cdot X_{h2}\cdot I_{h2}$
(Z. 686-687).

**→ Aus Q_CH4 zusätzlich erreichbar nach 2 Schritten:** $X_{ac}, X_{h2},
S_{ac}$, und alle Inhibitions-Faktoren $I_{ac}$ (= alle Ionen + $S_{nh3}$).

### Acetat-Kette (S_ac-DGL, Z. 805-815)

```python
diff_S_ac = D_in·s_in[6] - D_out·S_ac
            + (1-Y_su)·f_ac_su  · Rho_su      # X_su, S_su
            + (1-Y_aa)·f_ac_aa  · Rho_aa      # X_aa, S_aa
            + (1-Y_fa)·0.7      · Rho_fa      # X_fa, S_fa
            + (1-Y_c4)·0.31     · Rho_c4_va   # X_c4, S_va
            + (1-Y_c4)·0.8      · Rho_c4_bu   # X_c4, S_bu
            + (1-Y_pro)·0.57    · Rho_pro     # X_pro, S_pro
            - Rho_ac
```

**→ Aus VFA-Messung (sieht $S_{ac}$ direkt) wird nach 1 Schritt sichtbar:**
$S_{su}, S_{aa}, S_{fa}, S_{va}, S_{bu}, S_{pro}$ und $X_{su}, X_{aa},
X_{fa}, X_{c4}, X_{pro}, X_{ac}$.

### Stickstoff-Kette (S_nh4-DGL, Z. 840-853)

```python
diff_S_nh4 = D_in·s_in[10] - D_out·S_nh4
             - Y_su·N_bac·Rho_su + (N_aa - Y_aa·N_bac)·Rho_aa
             - Y_fa·N_bac·Rho_fa - ...
             + (N_bac - f_pr_bac·N_aa - f_p_bac·N_I)·sum_decay
             + Rho_A_IN
```

**→ NH4-N-Messung erschließt zusätzlich:** alle Biomasse-Decay-Pools,
$S_{nh3}$ via $\rho_{A,IN}$, und über $S_H$ die gesamte pH-Algebra.

### Partikuläre Pools (Z. 856-893)

Disintegration X_PS_* → X_S_* und Hydrolyse X_S_* → S_su/aa/fa wirken auf
**Summen**, die ch/pr/li-Aufteilung erscheint überall identisch und ist
deshalb mit aggregierten Messungen (TS, VS, CSB) **nicht trennbar**.

## Master-Tabelle: Sensor → erschlossene Zustände

Synthese der direkten + 1-Schritt-indirekten Abhängigkeiten, getrennt
nach Verfügbarkeit. Symbole: **D** direkt · **I₁** 1-Schritt · **I₂+**
mehrere Schritte · **C** korrelativ · **Σ** nur als Summe trennbar ·
*(leer)* kein realistischer Pfad.

### Online-Sensoren

#### Gelöste Komponenten (0-11)

| Sensor | S_su | S_aa | S_fa | S_va | S_bu | S_pro | S_ac | S_h2 | S_ch4 | S_co2 | S_nh4 | S_I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q_gas | | | | | | | | I₁ | I₁ | I₁ | | |
| Q_ch4 (NDIR) | | | | | | | I₂ | I₁ | I₁ | I₂ | I₂ | |
| Q_co2 (NDIR) | | | | | | | I₂ | | | I₁ | I₂ | |
| H₂-Gas (NDIR) | | | | | | | | I₁ | | I₂ | | |
| pH (online) | | | | | | | | | | | D | |
| Conductivity | | | | | | | | | | | C | |
| ORP | | | | | | | | C | | | | |
| Online-VFA (HPLC) | | | | D | D | D | D | | | | | |
| Online-NH₄⁺ (ISE) | | | | | | | | | | | D | |
| BHKW Q_ch4 | | | | | | | I₂ | I₁ | I₁ | I₂ | | |

#### Biomasse + augmentierte Inputs

| Sensor | X_su | X_aa | X_fa | X_c4 | X_pro | X_ac | X_h2 | Q_solid | Q_liquid |
|---|---|---|---|---|---|---|---|---|---|
| Q_gas | I₂+ | I₂+ | I₂+ | I₂+ | I₂+ | I₂ | I₂ | | |
| Q_ch4 | I₂+ | I₂+ | I₂+ | I₂+ | I₂ | I₁ | I₁ | | |
| Online-VFA | I₁ | I₁ | I₁ | I₁ | I₁ | I₁ | I₂+ | | |
| Hopper-Wägezelle | | | | | | | | D | |
| Pre-Pit-Level / Pumpenzähler | | | | | | | | | D |

#### Ladungsbilanz + Gasphase

| Sensor | S_cation | S_anion | S_va_ion | S_bu_ion | S_pro_ion | S_ac_ion | S_hco3 | S_nh3 | p_gas_h2 | p_gas_ch4 | p_gas_co2 | pTOTAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q_gas | | | | | | | I₁ | | I₁ | I₁ | I₁ | D |
| Q_ch4 | | | | | | | I₂ | | D | D | I₁ | D |
| Q_co2 | | | | | | | I₁ | | D | I₁ | D | D |
| H₂-Gas | | | | | | | | | D | | | I₁ |
| p_head | | | | | | | | | | | | D |
| pH (online) | D | D | D | D | D | D | D | D | | | | |
| Conductivity | C | C | | | | | | | | | | |
| Online-VFA | | | | | | | | | | | | |

### Lab-Analytik

#### Gelöste Komponenten (0-11)

| Lab-Größe | S_su | S_aa | S_fa | S_va | S_bu | S_pro | S_ac | S_h2 | S_ch4 | S_co2 | S_nh4 | S_I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VFA-Summe (FOS) | | | | D-Σ | D-Σ | D-Σ | D-Σ | | | | | |
| TAC | | | | D | D | D | D | | | D | D | |
| FOS/TAC | | | | D | D | D | D | | | D | D | |
| pH (Lab) | | | | | | | | | | | D | |
| NH₄-N (Lab) | | | | | | | | | | | D | |
| Einzel-VFA (HPLC) | | | | D | D | D | D | | | | | |
| TKN | | | | | | | | | | | D-Σ | |
| TS / VS (Lab) | | | | | | | | | | | | I₂ |
| CSB total | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | | | I₁(Σ) |
| CSB gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | D-Σ-gelöst | | | D-Σ-gelöst |
| Anionen-Inv. | | | | | | | | | | | | |
| Kationen-Inv. | | | | | | | | | | | | |
| Substrat-TS/oTS/CSB | (s_in[*] — Influent, kein State direkt) | | | | | | | | | | | |

#### Biomasse + Ladungsbilanz (Lab)

| Lab-Größe | X_PS_* / X_PF_* / X_S_* | X_I | X_su-h2 | S_cation | S_anion | S_*_ion | S_hco3 | S_nh3 |
|---|---|---|---|---|---|---|---|---|
| TS / VS | I₁(Σ) | D | I₂+ | | | | | |
| CSB partikulär | D-Σ-part. | D-Σ-part. | D-Σ-part. | | | | | |
| TAC | | | | D | D | D | D | D |
| TKN | | | | | | | | I₁ |
| Anionen-Inv. | | | | | D | D? | D | |
| Kationen-Inv. | | | | D | | | | I₁ |

> Hinweis zu Anionen-/Kationen-Inventur: $S_{cation}$ und $S_{anion}$
> sind im Modell **gepoolt** als kmol/m³. Eine spezies-aufgelöste
> Inventur (Na⁺, K⁺, …) gibt die Pool-Summe direkt; einzelne Spezies
> existieren im State-Vektor nicht.

## Implikationen für den UKF

### Praktische Faustregeln

1. **Nur Standard-SCADA (Q_gas, BHKW, T, Hopper, Pre-Pit)**  
   * direkt schätzbar: pTOTAL, $Q_{solid}, Q_{liquid}$ (augmentiert)  
   * 1-Schritt: $S_{h2/ch4/co2}, S_{hco3}, p_{gas,*}$  
   * 2-Schritt: $X_{ac}, X_{h2}, S_{ac}$  
   * Effektiv 6-8 trennbare Dimensionen, $X_{ac}$/$X_{h2}$ teilweise entartet.  

2. **+ NDIR-Gasanalytik (CH₄/CO₂-Anteil online)**  
   * $X_{ac}$ vs $X_{h2}$ trennbar (durch CH₄/CO₂-Ratio).  
   * $S_{co2}$ vs $S_{hco3}$ sauberer trennbar.  
   * H₂-Sensor optional: macht $p_{gas,h2}$ direkt observable.  

3. **+ pH-Sonde**  
   * 9 Ladungsbilanz-Zustände direkt observable.  
   * Newton-Iteration im Update entfällt, algebraisch sauber.  

4. **+ FOS/TAC Routine-Lab (täglich/wöchentlich)**  
   * VFA-Summe + Alkalitäts-Summe direkt.  
   * Acetat-Methanogen-Kette robust observable.  
   * Frühwarn-Indikatoren quantitativ.  

5. **+ Einzel-VFA (HPLC, monatlich)**  
   * $S_{ac}, S_{pro}, S_{bu}, S_{va}$ einzeln statt nur Summe.  
   * Trennt acetoclasten- vs propionat-getriebene Inhibitionen.  

6. **+ NH4-N + TKN**  
   * $S_{nh4}$ direkt, NH3-Inhibition $I_{nh3}$ quantifizierbar.  
   * Bei Stickstoff-reichen Substraten (Hühnermist, Schlachtabfälle)  
     entscheidend.

7. **+ Kationen-/Anionen-Inventur (monatlich)**  
   * $S_{cation}, S_{anion}$ direkt — sonst sehr schwer observable.  

### Was der UKF nicht kann, egal mit welcher Sensorik

* **Einzelne ch/pr/li-Fraktionen trennen.** Disintegrations- und  
  Hydrolyseraten wirken auf Summen, die Aufteilung bleibt durch den
  Prior bestimmt. Lösung nur über qPCR oder Substrat-Profile.  
* **$X_I$/$S_I$-Akkumulation in Stunden.** Inert-Pools haben  
  Zeitkonstanten von Wochen bis Monaten.  
* **Substrat-spezifische Disintegration online.** Selbst mit Lab-VFA  
  und pH bleibt die Aufteilung zwischen X_PS und X_PF (slow/fast)
  schwach observable. Lösung: Calibration-Artefakt setzt diese
  Parameter, Filter schätzt sie nicht mit.  
* **Einzelne Biomasse-Populationen** ohne qPCR, gemessen werden nur  
  Summen-Effekte über die Gasstrom-Antwort.

### Latenz-Bewertung

| Mess-Set | Update-Frequenz | Reaktionszeit auf Anlagenstörung |
|---|---|---|
| Nur Q_gas | s-Bereich | minütlich — sehr schnell |
| + NDIR + pH | s-Bereich | bei pH-Veränderung sofort |
| + FOS/TAC Routine-Lab | täglich | 1-2 Tage Verzögerung — UKF muss mit Prior überbrücken |
| + Einzel-VFA / NH4-N / Kationen | wöchentlich-monatlich | sehr langsame Korrektur; eher Bias-Anker als Reaktion |
| + qPCR | jährlich oder selten | Validierung, keine Reaktion |

## STRIKE-GOLDD-Anwendbarkeit für ADM1da

Kurz: **nicht direkt für das volle 41-State-Modell.**

### Vergleichsdaten aus der Literatur

Hellmann et al. (2023) zeigen empirisch für die STRIKE_GOLDD-Toolbox (Matlab):

| Modellklasse | States | FISPO-Zeit | ORC-DF-Zeit |
|---|---|---|---|
| ADM1-R4 | 11 | 3 s | 7 s |
| BMR3+ABC | 13 | 12 s | 5 s |
| ADM1-R3 | 17 | 11.959 s (≈ 3.3 h) | 811 s |
| ADM1-R2 | >17 | **kein Ergebnis** (Toolbox bricht ab) | **kein Ergebnis** |

### Eigenes Python/sympy-Benchmark

Wir haben den STRIKE-GOLDD Algorithmus direkt in Python/sympy
nachgebaut (`observability_experiment/subsystem_checker.py`):
symbolische Lie-Ableitungen + numerische Rangprüfung am Sample-Punkt
(Sedoglavic 2002). Exakt das, was ORC-DF intern auch macht, nur ohne
die Matlab↔Octave↔Python-Brücke.

Eine Direktanalyse aller 41 Zustände ist nicht praktikabel: die
Lie-Ableitungs-Komplexität wächst exponentiell (ein Polynom-Fit aus
synthetischen Testläufen mit n = 4, 6, 8 ergibt $t \propto 4.04^{n}$
Sekunden bzw. $m \propto 3.39^{n}$ MiB; bereits ab n = 16 überschreitet
der RAM-Bedarf 256 GiB). Hellmann (2023) stoppt aus dem gleichen Grund
bei n = 17.

**Lösung: Divide-and-Conquer über fünf Subsysteme + ein Super-Subsystem.**
Das volle Modell wird entlang seiner topologischen Struktur (Gas / Säure /
Hydrolyse / Ladungsbilanz / Stickstoff) in Blöcke zerlegt, jeder Block
wird isoliert analysiert, Kopplungen über *known inputs* aufgelöst.

### Ergebnisse (Phase-1-Sensorik: $Q_{gas}$ + CH4/CO2 NDIR + pH online + FOS/TAC)

| Subsystem | n | n_out | iters | Rang | Verdict | Wall |
|---|---|---|---|---|---|---|
| A — Gas + Methanogenese        | 11 | 3 | 3 | 11 | **observable**            | 0.7 s |
| B — Acidogenese                |  9 | 1 | 5 |  6 | partial (sympy-Toolchain-Limit; Decke ≈ 9) | 51 min |
| C — Disintegration / Hydrolyse | 10 | 3 | 12 |  7 | partial (PS/PF-Split echt nicht trennbar) | 1.4 s |
| D — Ladungsbilanz / pH         |  8 | 2 | 3 |  8 | **observable**            | 5.6 min |
| E — Stickstoff + $S_I$         |  2 | 2 | 0 |  2 | **observable** (open-loop) | < 0.1 s |
| **A+D fusioniert (Variante II)** | **18** | **5** | **3** | **18** | **observable, ohne Handshake-Annahmen** | **17.8 min** |

Total: ≈ 80 min Rechenzeit, alle Runs im 1-Stunden-Budget pro Subsystem.

**Kernaussagen:**

* **A+D fusioniert** ist der stärkste Composite-Beweis: 18 von 41  
  Zuständen, inkl. der vollen pH-Algebra mit $S_H = (-fixed +
  \sqrt{fixed^2 + 4 K_w})/2$ und der Inhibitionsfaktoren
  $I_{ac}, I_{h2}, I_{HAc}, I_{nh3}$ sind ohne irgendwelche
  „opaken" Boundary-Inputs strukturell observabel. Damit ist die
  Gas-Phase, die Methanogenese und die komplette Säure-Base-Chemie
  bewiesenermaßen schätzbar.

* **C zeigt das echte strukturelle Defizit**: TS und VS allein können  
  Kohlenhydrate, Proteine und Lipide nicht trennen (Rang 4/10). Mit
  COD-gewichteter Messung (1.03 / 1.5 / 2.9 gCOD/gVS für ch/pr/li)
  steigt der Rang auf 7/10, die verbleibenden drei Defekte sind die
  PS/PF-Splits (langsam vs. schnell abbaubar), die aus
  Prozessmessungen prinzipiell nicht auflösbar sind und durch
  Substrat-Charakterisierung im Labor festgelegt werden müssen.

* **B ist toolchain-limitiert, nicht physikalisch limitiert**:  
  $S_{fa}, X_{fa}$ wurden als analytisch entkoppelt entfernt (sie
  fließen via $\rho_{fa}$ nur in $S_{ac}, S_{h2}$ und damit nach A,
  nicht in die VFA-Summe der FOS-Messung). Die Rang-Steigerung
  $1 \to 2 \to 3 \to 4 \to 5 \to 6$ läuft konstant +1/Iteration; bei
  Iter 6 trifft SymPy einen CPython-Buffer-Limit
  (`bytesobject.c:3219`). Das Muster legt strukturell Rang 9 nahe,
  aber beweisbar ist im aktuellen Toolchain nur Rang 6.

* **E** (S_nh4, S_I) ist *open-loop observable*: jedes RHS-Element  
  steht über A+B+C+D zur Verfügung, der Ausgangs-Jacobian erreicht
  Rang 2 ohne Lie-Iteration. Aber ohne NH4-N-Messung kein
  Innovations-Kanal, Drift im UKF muss über OU-Prior gefangen werden.

**Komposite-Bilanz:**

| Block | Provably observabel | Plausibel observabel |
|---|---|---|
| A+D fusioniert | 18 / 18 | 18 / 18 |
| B (substrat-seitig) | 5 / 9 | 9 / 9 (sympy-limit) |
| C | 7 / 10 | 7 / 10 (PS/PF-Defizit ist real) |
| E | 2 / 2 (open-loop) | 2 / 2 |
| **ADM1da gesamt** | **32 / 41** | **36 / 41** |

Die verbleibenden 5-9 Zustände sind: drei PS/PF-Splits in C (Lab-
Charakterisierung pflichtig), zwei Stickstoff-Zustände in E
(Korrekturkanal nur mit NH4-N), und 0-4 Biomasse-Zustände in B
(strukturell vermutlich observabel, im aktuellen Toolchain
unbeweisbar).

Vollständige Details und Reproduktionsanleitung:
`observability_experiment/results.md`.

### Konsequenz für die UKF-Auslegung

* **Direkter Innovationskanal** für 30 Zustände: 18 A+D + 5  
  Acidogenese-Substrate + 7 Hydrolyse-Modi.  
* **Open-Loop-Propagation** für 11 Zustände: 4 Acidogenese-Biomassen  
  (X_su, X_aa, X_c4, X_pro), 3 PS/PF-Splits in C, 2
  Stickstoff-Zustände in E, 2 FA-Zustände (S_fa, X_fa über A
  langsam observabel). Modellieren als OU-Drift-Kanäle in
  `StateVectorSpec`.

## Quellen

* `pyadm1/core/adm1.py` (v0.3.4) — alle Zeilen-Zitate beziehen sich hierauf.  
* `pyadm1/components/biological/digester.py` — VFA-/TAC-Berechnung.  
* Hellmann, S. et al. (2023). *Observability and Identifiability  
  Analyses of Process Models for Agricultural Anaerobic Digestion
  Plants.* arXiv:2301.05068v3.  
* Haugen, F. et al. (2014). *State Estimation and Model-Based Control of  
  a Pilot Anaerobic Digestion Reactor.* J. Control Sci. Eng., 572621.  
* Villaverde, A. F. (2022). *STRIKE_GOLDD 4.0.* arXiv:2207.07346.  
* Wolf, C., Gaida, D., Bongards, M. (2014). *Online-measurement systems  
  for agricultural and industrial AD plants — a review and practice
  test.* Kompendium :metabolon.  
* [Observability-Literatur-Überblick](literature_review.md)  
