# Sensor–state dependencies in ADM1da

This document is **step 2** of the observability analysis:
the concrete reconciliation of the published literature
([step 1](literature_review.md)) with *our* ADM1da implementation in
[`pyadm1.core.adm1`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py).

Goal: for each measurement channel, document which of the 41 ADM1da
states appear directly in the measurement equation and which become
observable only indirectly through the model's dynamics. From this
follows the rationale for which states a UKF can realistically estimate
when only certain sensors are available.

Measurements are split by **availability** — an online sensor delivers
innovation every second, lab analytics every few days, research-grade
methods maybe once per year.

## Methodology

A state $x_i$ is observable from a measurement $y = h(x)$ only if either

1. **directly** $x_i$ appears explicitly in $h(x)$ (nonzero derivative in  
   the measurement model), or  
2. **indirectly** $x_i$ appears in the right-hand side $\dot x_j = f_j(x)$  
   of some directly observed $x_j$ — then $\dot y$ carries information
   about $x_i$. The mechanism iterates: after $k$ steps, $x_i$ appears in
   $y^{(k)}$ if a chain of length $k$ exists through the Jacobian
   (Lie derivatives, see Hellmann et al. 2023).

In practice:

* **D = direct** — immediate observability in a single update step.  
* **I₁ = 1-step indirect** — observability after at least one time step  
  with dynamics in between.  
* **I₂+ = k-step indirect** — reconstructible only with very accurate  
  model dynamics and a long observation window; in practice often
  inseparable from model errors.  
* **C = correlative** — no direct model expression, but empirically  
  coupled (e.g. conductivity ↔ ionic strength).

The tables below cite code locations in `pyadm1/core/adm1.py` (line
numbers refer to v0.3.4).

## Notation and symbols

Glossary of the most important quantities so the equations and tables
below stay readable. The full 41-state index map is in
[Theory → ADM1da model](../theory/adm1.md).

### State classes

| Prefix | Meaning | Typical unit |
|---|---|---|
| $S_\bullet$ | dissolved species (solute) in the liquid | kg COD/m³ or kmol/m³ |
| $X_\bullet$ | particulate species (biomass, solid pools) | kg COD/m³ |
| $p_{\text{gas},\bullet}$ | partial pressure of a gas-phase species | bar |
| $p_{\text{total}}$ (= pTOTAL) | total gas pressure in the headspace, state index 40 | bar |

### Subscripts (compound classes)

| Code | Meaning |
|---|---|
| su | sugars (monosaccharides) |
| aa | amino acids |
| fa | long-chain fatty acids (LCFA) |
| va | valeric acid / valerate (C5) |
| bu | butyric acid / butyrate (C4) |
| pro | propionic acid / propionate (C3) |
| ac | acetic acid / acetate (C2) |
| h2 | hydrogen |
| ch4 | methane |
| co2 | inorganic carbon (= S_IC) |
| nh4 | ammonium nitrogen (= S_IN) |
| nh3 | free ammonia |
| hco3 | bicarbonate HCO₃⁻ |
| _ion (suffix) | dissociated ionic form of a VFA (e.g. $S_{ac\_ion}$ = acetate anion) |
| cation, anion | sum of all strong cations / anions [kmol/m³] |
| I | inert (dissolved $S_I$ or particulate $X_I$) |
| ch / pr / li | carbohydrates / proteins / lipids |
| PS / PF / S | pool class: slow-disintegrating / fast-disintegrating / hydrolysable |
| c4 | C4-acid degraders (valerate / butyrate utilizers) |
| bac | bacterial biomass (generic) |

### Observable quantities (outputs / measurement models)

| Symbol | Meaning |
|---|---|
| $Q_{\text{gas}}$ | total biogas volume flow [m³/d] |
| $Q_{\text{CH4}}$, $Q_{\text{CO2}}$ | methane / CO₂ volume flow [m³/d] |
| $P_{\text{gas}}$ | sum of dry partial pressures $p_{\text{gas,h2}} + p_{\text{gas,ch4}} + p_{\text{gas,co2}}$ [bar] |
| $p_{\text{total}}$ | total gas pressure as an integrated model state (index 40) |
| $p_{\text{gas,h2o}}$ | water-vapour saturation pressure — a temperature-dependent parameter, **not** a state |
| pH | $-\log_{10}(S_H)$ with $S_H$ from the Newton solution of the charge balance |
| VFA | sum of volatile fatty acids as HAc equivalent [g HAc/L] |
| TAC | total alkalinity (titrated to pH 5) [g CaCO₃/L] |
| FOS/TAC | ratio of VFA to TAC — Nordmann early-warning indicator |
| $Q_{\text{solid}}$, $Q_{\text{liquid}}$ | augmented substrate feed rates [m³/d] — not ADM1 states but part of the extended state vector |

### Reaction rates $\rho$ (in the code: `Rho_*`)

| Symbol | Meaning |
|---|---|
| $\rho_{\text{dis,PS}}$, $\rho_{\text{dis,PF}}$ | disintegration rates ($X_{PS} \to X_S$, $X_{PF} \to X_S$) |
| $\rho_{\text{hyd,ch/pr/li}}$ | hydrolysis rates ($X_S \to S$) |
| $\rho_{\text{su}}, \rho_{\text{aa}}, \rho_{\text{fa}}, \rho_{\text{c4}}, \rho_{\text{pro}}, \rho_{\text{ac}}, \rho_{\text{h2}}$ | substrate uptake rates (Monod kinetics × biomass × inhibition) |
| $\rho_{\text{dec,}\bullet}$ | biomass decay rates |
| $\rho_{A,\bullet}$ | acid-base reaction rates (very fast, $k_{AB} = 10^8$) |
| $\rho_{T,\text{h2}}, \rho_{T,\text{ch4}}, \rho_{T,\text{co2}}$ | gas-liquid transfer rates (Henry's law) |
| $\rho_{T,11}$ | bulk gas outflow from the headspace |

### Inhibition factors $I$ (values in [0, 1])

| Symbol | Meaning |
|---|---|
| $I_{\text{pH,aa}}, I_{\text{pH,ac}}, I_{\text{pH,h2}}$ | pH inhibition (Hill type, exponent 1-3 per pool) |
| $I_{\text{IN}}$ | N limitation: $S_{IN}/(K_{S,IN}+S_{IN})$ |
| $I_{\text{h2,fa/c4/pro}}$ | H₂ inhibition for LCFA, C4, propionate degraders |
| $I_{\text{nh3}}, I_{\text{nh3,pro}}$ | NH₃ inhibition for acetoclasts and propionate degraders |
| $I_{\text{HAc}}, I_{\text{HPr}}$ | inhibition by undissociated acetic / propionic acid |
| $I_{\text{co2,h2}}$ | CO₂ inhibition of the H₂-producing pathways |

### Key parameters

| Symbol | Meaning |
|---|---|
| $k_{m,\bullet}$, $K_{S,\bullet}$ | maximum Monod rate and half-saturation constant |
| $k_{\text{dec,}\bullet}$ | decay rate per biomass pool [d⁻¹] |
| $k_{\text{dis,PS/PF}}$, $k_{\text{hyd,ch/pr/li}}$ | disintegration / hydrolysis rates [d⁻¹] |
| $Y_\bullet$ | biomass yield per substrate COD [-] |
| $K_{a,\bullet}$ | acid dissociation constants |
| $K_{H,\text{ch4/co2/h2}}$ | Henry constants (gas solubility) |
| $K_w$ | ionic product of water |
| $k_{L}a$ | gas-liquid transfer coefficient [d⁻¹] |
| $k_p$ | gas-outlet flow coefficient (pipe flow) |
| $V_{\text{liq}}$, $V_{\text{gas}}$ | liquid / gas-phase volume [m³] |
| $T_{\text{ad}}$ | operating temperature (adiabatic) [K] |
| $R$, $RT$ | gas constant / product $R \cdot T_{\text{ad}}$ |
| $M_{\text{HAc}}$ | 60 g/mol (acetate molar mass) for VFA conversion |
| 64, 112, 160, 208 | COD equivalents for acetate / propionate / butyrate / valerate [g COD/mol] |
| $f_{\bullet}$ | stoichiometric product fractions (e.g. $f_{\text{ac,su}} = 0.41$) |
| $N_{\text{bac}}, N_{\text{aa}}, N_I$ | nitrogen contents per biomass / amino-acid / inert pool [kmol N/kg COD] |
| $D_{\text{in}}, D_{\text{out}}$ | dilution rates $Q/V_{\text{liq}}$ [d⁻¹] |
| $S_H$ | $H^+$ concentration from the Newton solution of the charge balance |

## Measurement-model equations (from the code)

### Gas volume flows

From [`ADM1.calc_gas`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py#L431-L467):

```python
# L. 451-454
q_gas = max(
    k_p * (p_total_wet - p_ext) / (RT/1000 * NQ) * V_liq,
    0.0,
)
# L. 456-460
p_gas    = pi_Sh2 + pi_Sch4 + pi_Sco2          # = p_gas_h2 + p_gas_ch4 + p_gas_co2
q_ch4    = q_gas * pi_Sch4 / (p_gas + p_gas_h2o)
q_co2    = q_gas * pi_Sco2 / (p_gas + p_gas_h2o)
```

| Observable | Directly depends on | Idx |
|---|---|---|
| $Q_{\text{gas}}$ | pTOTAL | 40 |
| $P_{\text{gas}}$ | $p_{\text{gas\_h2}}$, $p_{\text{gas\_ch4}}$, $p_{\text{gas\_co2}}$ | 37, 38, 39 |
| $Q_{\text{CH4}}$ | pTOTAL, $p_{\text{gas\_h2}}$, $p_{\text{gas\_ch4}}$, $p_{\text{gas\_co2}}$ | 37, 38, 39, 40 |
| $Q_{\text{CO2}}$ | pTOTAL, $p_{\text{gas\_h2}}$, $p_{\text{gas\_ch4}}$, $p_{\text{gas\_co2}}$ | 37, 38, 39, 40 |

### pH

From [`ADM1._calc_ph`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py#L1031-L1056):

```python
# L. 1046-1048
vfa_anions = S_ac_ion/64 + S_pro_ion/112 + S_bu_ion/160 + S_va_ion/208
fixed      = S_cation - S_anion + (S_nh4 - S_nh3) - S_hco3 - vfa_anions
# Newton: f(S_H) = fixed + S_H − K_w/S_H = 0
pH = -log10(S_H)
```

| Observable | Directly depends on | Idx |
|---|---|---|
| pH | $S_{nh4}, S_{nh3}, S_{hco3}, S_{ac\_ion}, S_{pro\_ion}, S_{bu\_ion}, S_{va\_ion}, S_{cation}, S_{anion}$ | 10, 36, 35, 34, 33, 32, 31, 29, 30 |

### VFA and TAC

From
[`Digester._compute_indicators`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/components/biological/digester.py#L338-L400):

```python
# L. 362-367  VFA in g HAc-eq / m³
vfa = M_HAc * (S_ac/64 + S_pro/112 + S_bu/160 + S_va/208)

# L. 389-399  TAC in g CaCO3 / m³ (endpoint pH 5)
# combines: free NH3/NH4, HCO3/CO2, all VFA ions, S_cation/S_anion.
```

| Observable | Directly depends on | Idx |
|---|---|---|
| VFA sum | $S_{va}, S_{bu}, S_{pro}, S_{ac}$ | 3, 4, 5, 6 |
| VFA individual (HPLC) | per acid individually | 3, 4, 5 or 6 |
| TAC | $S_{va}, S_{bu}, S_{pro}, S_{ac}, S_{co2}, S_{nh4}, S_{cation}, S_{anion}, S_{va\_ion}, S_{bu\_ion}, S_{pro\_ion}, S_{ac\_ion}, S_{hco3}, S_{nh3}$ | 3-6, 9, 10, 29-36 |
| FOS/TAC | Union of VFA and TAC | ditto |

## A — Online measurements

Continuously available, second-to-minute innovation. This list covers
the typical SCADA instrumentation of an agricultural biogas plant plus
optional upgrades.

### A.1 Standard SCADA (most plants)

| Sensor | What is measured | Direct ADM1da-state link | Class |
|---|---|---|---|
| Q_gas (thermal / turbine flowmeter) | biogas volume flow [m³/d] | pTOTAL (40) | D |
| Gas storage level (% filled) | storage Δ over time → contribution to Q_gas | as Q_gas | reconstructive |
| Flare on/off | status; boundary for Q_gas balance | — (boundary) | — |
| p_head | headspace pressure [mbar/bar] | pTOTAL (40) | D |
| T_reactor (PT100 in digester) | fermenter temperature | — (boundary, scales kinetics via Arrhenius) | — |
| T_ambient, T_inlet | heat-balance boundary | — | — |
| Substrate load cell / pump counter | $-\Delta W$ → Q_feed (solid or liquid) | $Q_{solid}$ / $Q_{liquid}$ (augmented) | D |
| CHP Q_ch4 consumption | methane consumption per CHP [m³/h] | pTOTAL + p_gas_ch4 (40, 38) via $\eta$ | I₁ |
| CHP P_el (electrical power) | electrical output [kW] | as Q_ch4, derived via $\eta_{el}$ | I₁ |
| Stirrer current / power | mixer — viscosity proxy | — (no ADM1 coupling) | — |
| Heating power | thermal output [kW] | — (boundary) | — |

### A.2 Online gas analytics (NDIR multi-gas sensor)

| Sensor | What is measured | Direct ADM1da-state link | Class |
|---|---|---|---|
| CH₄ fraction in biogas (NDIR) | $p_{gas,ch4}/p_{gas}$ [%] | $p_{gas,ch4}$ (38) | D |
| CO₂ fraction in biogas (NDIR) | $p_{gas,co2}/p_{gas}$ [%] | $p_{gas,co2}$ (39) | D |
| H₂ fraction in biogas (TCD/electrochem) | $p_{gas,h2}/p_{gas}$ [ppm-%] | $p_{gas,h2}$ (37) | D |
| O₂ fraction in biogas | quality check (air leakage) | — (no ADM1 state) | — |
| H₂S fraction in biogas | hydrogen sulfide [ppm] | — (no ADM1 state; desulfurization) | — |

### A.3 Online liquid-phase (premium instrumentation)

| Sensor | What is measured | Direct ADM1da-state link | Class |
|---|---|---|---|
| pH probe (glass electrode, ISFET) | liquid pH | 9 charge-balance states (10, 29-36) | D |
| ORP / redox potential | reduction potential [mV] | $S_{h2}$ (7), methanogenic activity | C |
| Conductivity (EC) | electrical conductivity [µS/cm] | $S_{cation} + S_{anion}$ + all ion species | C |
| Online TAC (FOS titrator) | alkalinity [g CaCO₃/L] | as lab TAC, see below | D |
| Online VFA (HPLC online) | individual acids [g/L] | $S_{va}, S_{bu}, S_{pro}, S_{ac}$ individually (3-6) | D |
| Online NH₄⁺ (ion-selective electrode) | ammonium [g/L] | $S_{nh4}$ (10) | D |
| Level / V_liq | fill level | — (constant in current model) | — |

## B — Lab analytics

Periodic (daily, weekly, monthly). Delivers sample-and-hold values; the
UKF uses them as **gated observations** at sporadic times.

### B.1 Standard AD routine (typ. weekly)

| Lab quantity | Method | Direct ADM1da-state link | Class |
|---|---|---|---|
| pH (lab) | glass electrode | as online pH | D |
| VFA (FOS titration Nordmann) | titrimetric | VFA sum: $S_{va}, S_{bu}, S_{pro}, S_{ac}$ as sum | D-sum |
| TAC (alkalinity titration) | titrimetric | as above (TAC entry) | D |
| FOS/TAC | derived | ratio | D |
| TS (total solids) | drying oven | $X_I$ directly, particulate pools as sum | I₁(Σ) |
| VS / oTS (loss on ignition) | ignition at 550 °C | analogous to TS, without mineral ash | I₁(Σ) |
| NH₄-N | indophenol or IC | $S_{nh4}$ (10) | D |
| Conductivity (lab) | EC meter | $S_{cation} + S_{anion}$ sum | C |

### B.2 Extended analytics (typ. weekly to monthly)

| Lab quantity | Method | Direct ADM1da-state link | Class |
|---|---|---|---|
| Individual VFA (acetate, propionate, n-/iso-butyrate, n-/iso-valerate) | HPLC or GC | $S_{ac}$ (6), $S_{pro}$ (5), $S_{bu}$ (4), $S_{va}$ (3) each | D per acid |
| TKN (Total Kjeldahl Nitrogen) | Kjeldahl digestion | $S_{nh4} + S_{nh3} + N$ content of all biomass and protein pools | D-sum over N |
| COD total | photometric | sum of all COD species | I₁(Σ) |
| COD dissolved (filtered 0.45 µm) | as above | only dissolved COD: $S_{su, aa, fa, va, bu, pro, ac, h2, ch4, I}$ sum | I₁(Σ-dissolved) |
| COD particulate | COD_total − COD_dissolved | particulate pools sum | I₁(Σ-part.) |
| Cation inventory (Na⁺, K⁺, Ca²⁺, Mg²⁺) | IC or ICP-OES | $S_{cation}$ (29) | D |
| Anion inventory (Cl⁻, SO₄²⁻, PO₄³⁻, NO₃⁻) | IC | $S_{anion}$ (30) | D |
| BOD5 | 5-day respiration test | biodegradable COD fraction | I₂(Σ) |

### B.3 Substrate analytics (for s_in of the influent balance)

| Lab quantity | What it delivers | Direct ADM1da-input link |
|---|---|---|
| Substrate TS, oTS, COD | per-substrate characterization | s_in[i] for particulate / inert pools per substrate slot |
| Substrate NH₄-N, TKN | nitrogen contribution | s_in[10], s_in[11] |
| Substrate disintegration fractions | from profiles or lab BMP test | f_ch_xc, f_pr_xc, f_li_xc, f_xi_xc, f_si_xc — feeds the calibration artifact, not state |

Substrate analytics does not affect the filter state directly, but the
**influent vector** $s_{in}$ — i.e. it shifts the initial condition of
the mass balance, not the state itself. A wrongly parametrized influent
leads to chronic filter biases.

## C — Research / specialized methods (for completeness)

| Method | What is measured | Direct ADM1da-state link | Note |
|---|---|---|---|
| qPCR / 16S rRNA sequencing | biomass populations quantitatively | $X_{su}, X_{aa}, X_{fa}, X_{c4}, X_{pro}, X_{ac}, X_{h2}$ (22-28) individually | expensive (~€100/sample), weeks lead time — rare in production |
| GC-MS for LCFA / trace VFA | individual LCFA species, trace acids | $S_{fa}$ (2) and finer subscale | research |
| In-situ H₂ probe (headspace) | online $p_{gas,h2}$ | $p_{gas,h2}$ (37) | see A.2 |
| Trace-element analytics (Fe, Co, Ni, Mo, Se) | important for kinetics | — (no ADM1 state, affects $k_{m,*}$) | as calibration hint |
| H₂S dissolved, sulfate, sulfide | sulfur inhibition | — (not in ADM1da) | for Hill-extended models |
| BMP test (batch methane potential) | specific methane yield per substrate | $f_{*}$ fractions, k_dis per substrate | input to calibration artifact |

## ODE coupling structure — who depends on whom

The right-hand side of every differential equation contains further
states. The dependencies below are extracted from
[`ADM_ODE`](https://github.com/dgaida/PyADM1ODE/blob/main/pyadm1/core/adm1.py#L540-L962):

### Gas-phase ODEs (L. 914-918)

```python
diff_p_h2  = Rho_T_h2  * RT/16 − p_gas_h2/pTOTAL  * Rho_T_11
diff_p_ch4 = Rho_T_ch4 * RT/64 − p_gas_ch4/pTOTAL * Rho_T_11
diff_p_co2 = Rho_T_co2 * RT    − p_gas_co2/pTOTAL * Rho_T_11
diff_pTOT  = RT/16 * Rho_T_h2 + RT/64 * Rho_T_ch4 + RT * Rho_T_co2 − Rho_T_11
```

with (L. 712-723):

```python
Rho_T_h2  = k_L_a * (S_h2  - 16*p_gas_h2 /(RT*K_H_h2 )) * V_liq/V_gas
Rho_T_ch4 = k_L_a * (S_ch4 - 64*p_gas_ch4/(RT*K_H_ch4)) * V_liq/V_gas
Rho_T_co2 = k_L_a * (S_co2_free - p_gas_co2/(RT*K_H_co2)) * V_liq/V_gas
S_co2_free = max(S_co2 - S_hco3, 0)
Rho_T_11  = k_p * (pTOTAL + p_gas_h2o − p_ext) * V_liq/V_gas
```

**→ A gas-flow measurement directly exposes within one time step:**
pTOTAL (D), $p_{gas,h2/ch4/co2}$ (1-step indirect via $\dot p_{TOT}$),
$S_{h2}, S_{ch4}, S_{co2}, S_{hco3}$ (all via $\rho_T$ terms).

### Methane chain (S_ch4 ODE, L. 828-834)

```python
diff_S_ch4 = D_in·s_in[8] - D_out·S_ch4 + (1-Y_ac)·Rho_ac + (1-Y_h2)·Rho_h2
             - V_gas/V_liq · Rho_T_ch4
```

with $\rho_{ac} = k_{m,ac}\cdot S_{ac}/(K_{S,ac}+S_{ac})\cdot X_{ac}\cdot I_{ac}$
and $\rho_{h2} = k_{m,h2}\cdot S_{h2}/(K_{S,h2}+S_{h2})\cdot X_{h2}\cdot I_{h2}$
(L. 686-687).

**→ Additionally reachable from Q_CH4 after 2 steps:** $X_{ac}, X_{h2},
S_{ac}$, and all inhibition factors $I_{ac}$ (= all ions + $S_{nh3}$).

### Acetate chain (S_ac ODE, L. 805-815)

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

**→ A VFA measurement (sees $S_{ac}$ directly) after one step exposes:**
$S_{su}, S_{aa}, S_{fa}, S_{va}, S_{bu}, S_{pro}$ and $X_{su}, X_{aa},
X_{fa}, X_{c4}, X_{pro}, X_{ac}$.

### Nitrogen chain (S_nh4 ODE, L. 840-853)

```python
diff_S_nh4 = D_in·s_in[10] - D_out·S_nh4
             - Y_su·N_bac·Rho_su + (N_aa - Y_aa·N_bac)·Rho_aa
             - Y_fa·N_bac·Rho_fa - ...
             + (N_bac - f_pr_bac·N_aa - f_p_bac·N_I)·sum_decay
             + Rho_A_IN
```

**→ NH4-N measurement additionally exposes:** all biomass decay pools,
$S_{nh3}$ via $\rho_{A,IN}$, and via $S_H$ the entire pH algebra.

### Particulate pools (L. 856-893)

Disintegration X_PS_* → X_S_* and hydrolysis X_S_* → S_su/aa/fa act on
**sums** — the ch/pr/li split appears identically everywhere and is
therefore **not separable** with aggregated measurements (TS, VS, COD).

## Master table: sensor → exposed states

Synthesis of direct + 1-step indirect dependencies, separated by
availability. Symbols: **D** direct · **I₁** 1-step · **I₂+** multi-step
· **C** correlative · **Σ** only separable as sum · *(blank)* no
realistic path.

### Online sensors

#### Dissolved components (0-11)

| Sensor | S_su | S_aa | S_fa | S_va | S_bu | S_pro | S_ac | S_h2 | S_ch4 | S_co2 | S_nh4 | S_I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q_gas | | | | | | | | I₁ | I₁ | I₁ | | |
| Q_ch4 (NDIR) | | | | | | | I₂ | I₁ | I₁ | I₂ | I₂ | |
| Q_co2 (NDIR) | | | | | | | I₂ | | | I₁ | I₂ | |
| H₂ gas (NDIR) | | | | | | | | I₁ | | I₂ | | |
| pH (online) | | | | | | | | | | | D | |
| Conductivity | | | | | | | | | | | C | |
| ORP | | | | | | | | C | | | | |
| Online VFA (HPLC) | | | | D | D | D | D | | | | | |
| Online NH₄⁺ (ISE) | | | | | | | | | | | D | |
| CHP Q_ch4 | | | | | | | I₂ | I₁ | I₁ | I₂ | | |

#### Biomass + augmented inputs

| Sensor | X_su | X_aa | X_fa | X_c4 | X_pro | X_ac | X_h2 | Q_solid | Q_liquid |
|---|---|---|---|---|---|---|---|---|---|
| Q_gas | I₂+ | I₂+ | I₂+ | I₂+ | I₂+ | I₂ | I₂ | | |
| Q_ch4 | I₂+ | I₂+ | I₂+ | I₂+ | I₂ | I₁ | I₁ | | |
| Online VFA | I₁ | I₁ | I₁ | I₁ | I₁ | I₁ | I₂+ | | |
| Hopper load cell | | | | | | | | D | |
| Pre-pit level / pump counter | | | | | | | | | D |

#### Charge balance + gas phase

| Sensor | S_cation | S_anion | S_va_ion | S_bu_ion | S_pro_ion | S_ac_ion | S_hco3 | S_nh3 | p_gas_h2 | p_gas_ch4 | p_gas_co2 | pTOTAL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q_gas | | | | | | | I₁ | | I₁ | I₁ | I₁ | D |
| Q_ch4 | | | | | | | I₂ | | D | D | I₁ | D |
| Q_co2 | | | | | | | I₁ | | D | I₁ | D | D |
| H₂ gas | | | | | | | | | D | | | I₁ |
| p_head | | | | | | | | | | | | D |
| pH (online) | D | D | D | D | D | D | D | D | | | | |
| Conductivity | C | C | | | | | | | | | | |
| Online VFA | | | | | | | | | | | | |

### Lab analytics

#### Dissolved components (0-11)

| Lab quantity | S_su | S_aa | S_fa | S_va | S_bu | S_pro | S_ac | S_h2 | S_ch4 | S_co2 | S_nh4 | S_I |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| VFA sum (FOS) | | | | D-Σ | D-Σ | D-Σ | D-Σ | | | | | |
| TAC | | | | D | D | D | D | | | D | D | |
| FOS/TAC | | | | D | D | D | D | | | D | D | |
| pH (lab) | | | | | | | | | | | D | |
| NH₄-N (lab) | | | | | | | | | | | D | |
| Individual VFA (HPLC) | | | | D | D | D | D | | | | | |
| TKN | | | | | | | | | | | D-Σ | |
| TS / VS (lab) | | | | | | | | | | | | I₂ |
| COD total | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | I₁(Σ) | | | I₁(Σ) |
| COD dissolved | D-Σ-diss | D-Σ-diss | D-Σ-diss | D-Σ-diss | D-Σ-diss | D-Σ-diss | D-Σ-diss | D-Σ-diss | D-Σ-diss | | | D-Σ-diss |
| Anion inventory | | | | | | | | | | | | |
| Cation inventory | | | | | | | | | | | | |
| Substrate TS/oTS/COD | (s_in[*] — influent, not state directly) | | | | | | | | | | | |

#### Biomass + charge balance (lab)

| Lab quantity | X_PS_* / X_PF_* / X_S_* | X_I | X_su-h2 | S_cation | S_anion | S_*_ion | S_hco3 | S_nh3 |
|---|---|---|---|---|---|---|---|---|
| TS / VS | I₁(Σ) | D | I₂+ | | | | | |
| COD particulate | D-Σ-part. | D-Σ-part. | D-Σ-part. | | | | | |
| TAC | | | | D | D | D | D | D |
| TKN | | | | | | | | I₁ |
| Anion inventory | | | | | D | D? | D | |
| Cation inventory | | | | D | | | | I₁ |

> Note on anion/cation inventory: $S_{cation}$ and $S_{anion}$ are
> **lumped** in the model as kmol/m³. A species-resolved inventory
> (Na⁺, K⁺, …) gives the pool sum directly; individual species do not
> exist in the state vector.

## Implications for the UKF

### Rules of thumb

1. **Only standard SCADA (Q_gas, CHP, T, hopper, pre-pit)**  
    * directly estimable: pTOTAL, $Q_{solid}, Q_{liquid}$ (augmented)  
    * 1-step: $S_{h2/ch4/co2}, S_{hco3}, p_{gas,*}$  
    * 2-step: $X_{ac}, X_{h2}, S_{ac}$  
    * Effectively 6-8 separable dimensions, $X_{ac}$/$X_{h2}$ partially  
      degenerate.

2. **+ NDIR gas analytics (CH₄/CO₂ fraction online)**  
    * $X_{ac}$ vs $X_{h2}$ separable (via CH₄/CO₂ ratio).  
    * $S_{co2}$ vs $S_{hco3}$ cleaner separation.  
    * Optional H₂ sensor: makes $p_{gas,h2}$ directly observable.  

3. **+ pH probe**  
    * 9 charge-balance states directly observable.  
    * Newton iteration in the update disappears — algebraically clean.  

4. **+ FOS/TAC routine lab (daily/weekly)**  
    * VFA sum + alkalinity sum directly.  
    * Acetate-methanogen chain robustly observable.  
    * Early-warning indicators quantitatively available.  

5. **+ Individual VFA (HPLC, monthly)**  
    * $S_{ac}, S_{pro}, S_{bu}, S_{va}$ individually rather than as sum.  
    * Separates acetoclast vs propionate-driven inhibitions.  

6. **+ NH4-N + TKN**  
    * $S_{nh4}$ directly, NH3 inhibition $I_{nh3}$ quantifiable.  
    * Decisive for nitrogen-rich substrates (poultry manure, abattoir  
      waste).

7. **+ Cation/anion inventory (monthly)**  
    * $S_{cation}, S_{anion}$ directly — otherwise very hard to observe.  

### What the UKF cannot do, regardless of instrumentation

* **Separate individual ch/pr/li fractions.** Disintegration and  
  hydrolysis rates act on sums; the split remains prior-determined —
  solution only via qPCR or substrate profiles.  
* **$X_I$/$S_I$ accumulation in hours.** Inert pools have time constants  
  of weeks to months — no innovation, no update.  
* **Substrate-specific disintegration online.** Even with lab VFA and  
  pH, the split between X_PS and X_PF (slow/fast) remains weakly
  observable. Solution: the calibration artifact sets these parameters,
  the filter does not co-estimate them.  
* **Individual biomass populations** without qPCR — only summary  
  effects through the gas-flow response are measured.

### Latency assessment

| Measurement set | Update rate | Reaction time to plant upset |
|---|---|---|
| Q_gas only | seconds | minutes — very fast |
| + NDIR + pH | seconds | immediate on pH shift |
| + FOS/TAC routine lab | daily | 1-2 day delay — UKF must bridge with prior |
| + Individual VFA / NH4-N / cations | weekly-monthly | very slow correction; more a bias anchor than reaction |
| + qPCR | annually or rare | validation, no reaction |

## STRIKE-GOLDD applicability for ADM1da

Short: **not directly for the full 41-state model.**

### Reference data from the literature

Hellmann et al. (2023) empirically show, using the STRIKE_GOLDD toolbox (Matlab):

| Model class | States | FISPO time | ORC-DF time |
|---|---|---|---|
| ADM1-R4 | 11 | 3 s | 7 s |
| BMR3+ABC | 13 | 12 s | 5 s |
| ADM1-R3 | 17 | 11,959 s (≈ 3.3 h) | 811 s |
| ADM1-R2 | >17 | **no result** (toolbox aborts) | **no result** |

### Our own Python/sympy benchmark

We rebuilt the STRIKE-GOLDD algorithm directly in Python/sympy
(`observability_experiment/subsystem_checker.py`): symbolic Lie
derivatives + numerical rank check at a random sample point
(Sedoglavic 2002). Exactly what ORC-DF does internally, but without
the Matlab↔Octave↔Python bridge.

A direct analysis of all 41 states is not practical: Lie-derivative
complexity grows exponentially (a polynomial fit from synthetic test
runs with n = 4, 6, 8 gives $t \propto 4.04^{n}$ seconds and
$m \propto 3.39^{n}$ MiB; already at n = 16 the RAM demand exceeds
256 GiB). Hellmann (2023) stops at n = 17 for the same reason.

**Solution: divide-and-conquer over five subsystems + one
super-subsystem.** The full model is decomposed along its topological
structure (gas / acidogenesis / hydrolysis / charge balance /
nitrogen) into blocks; each block is analysed in isolation, couplings
are resolved via *known inputs*.

### Results (Phase-1 sensor set: $Q_{gas}$ + CH4/CO2 NDIR + pH online + FOS/TAC)

| Subsystem | n | n_out | iters | rank | verdict | wall |
|---|---|---|---|---|---|---|
| A — Gas + methanogenesis      | 11 | 3 | 3 | 11 | **observable**            | 0.7 s |
| B — Acidogenesis              |  9 | 1 | 5 |  6 | partial (sympy toolchain limit; ceiling ≈ 9) | 51 min |
| C — Disintegration / hydrolysis | 10 | 3 | 12 |  7 | partial (PS/PF split truly non-separable) | 1.4 s |
| D — Charge balance / pH       |  8 | 2 | 3 |  8 | **observable**            | 5.6 min |
| E — Nitrogen + $S_I$          |  2 | 2 | 0 |  2 | **observable** (open-loop) | < 0.1 s |
| **A+D fused (Variante II)**   | **18** | **5** | **3** | **18** | **observable, no handshake assumptions** | **17.8 min** |

Total: ≈ 80 min compute, all runs inside the 1-hour-per-subsystem budget.

**Key findings:**

* **A+D fused** is the strongest composite proof: 18 of 41 states,  
  including the full pH algebra
  $S_H = (-fixed + \sqrt{fixed^2 + 4 K_w})/2$ and the inhibition
  factors $I_{ac}, I_{h2}, I_{HAc}, I_{nh3}$, are structurally
  observable without any "opaque" boundary inputs. So the gas phase,
  methanogenesis, and the complete acid-base chemistry are provably
  estimatable.

* **C exposes the real structural deficit**: TS and VS alone cannot  
  separate carbohydrates, proteins and lipids (rank 4/10). With
  COD-weighted measurement (1.03 / 1.5 / 2.9 gCOD/gVS for ch/pr/li)
  the rank rises to 7/10; the remaining three deficits are the PS/PF
  splits (slow vs. fast degradable), which are fundamentally not
  resolvable from process measurements and must be fixed by
  substrate characterization in the lab.

* **B is toolchain-limited, not physically limited**: $S_{fa}, X_{fa}$  
  were removed as analytically decoupled (they flow via $\rho_{fa}$
  into $S_{ac}, S_{h2}$ and thus into A, not into the VFA sum that
  FOS measures). The rank growth $1 \to 2 \to 3 \to 4 \to 5 \to 6$
  is steady at +1 per iteration; at iter 6 SymPy hits a CPython
  buffer limit (`bytesobject.c:3219`). The pattern strongly suggests
  the structural rank is 9, but only rank 6 is provable inside the
  current toolchain.

* **E** (S_nh4, S_I) is *open-loop observable*: every RHS element is  
  available via A+B+C+D, so the output Jacobian already reaches rank
  2 without Lie iteration. But without an NH4-N measurement there is
  no innovation channel; drift in the UKF must be caught by an
  OU prior.

**Composite balance:**

| Block | Provably observable | Plausibly observable |
|---|---|---|
| A+D fused | 18 / 18 | 18 / 18 |
| B (substrate side) | 5 / 9 | 9 / 9 (sympy limit) |
| C | 7 / 10 | 7 / 10 (PS/PF deficit is real) |
| E | 2 / 2 (open-loop) | 2 / 2 |
| **ADM1da total** | **32 / 41** | **36 / 41** |

The remaining 5–9 states are: three PS/PF splits in C (lab
characterization required), two nitrogen states in E (correction
channel only with NH4-N), and 0–4 biomass states in B (structurally
likely observable, unprovable inside the current toolchain).

Full details and reproduction instructions:
`observability_experiment/results.md`.

### Consequence for UKF design

* **Direct innovation channel** for 30 states: 18 A+D + 5  
  acidogenesis substrates + 7 hydrolysis modes.  
* **Open-loop propagation** for 11 states: 4 acidogenesis biomasses  
  (X_su, X_aa, X_c4, X_pro), 3 PS/PF splits in C, 2 nitrogen states
  in E, 2 FA states (S_fa, X_fa observed slowly via A). Model them
  as OU-drift channels in `StateVectorSpec`.

## Sources

* `pyadm1/core/adm1.py` (v0.3.4) — all line citations refer to this.  
* `pyadm1/components/biological/digester.py` — VFA/TAC computation.  
* Hellmann, S. et al. (2023). *Observability and Identifiability  
  Analyses of Process Models for Agricultural Anaerobic Digestion
  Plants.* arXiv:2301.05068v3.  
* Haugen, F. et al. (2014). *State Estimation and Model-Based Control of  
  a Pilot Anaerobic Digestion Reactor.* J. Control Sci. Eng., 572621.  
* Villaverde, A. F. (2022). *STRIKE_GOLDD 4.0.* arXiv:2207.07346.  
* Wolf, C., Gaida, D., Bongards, M. (2014). *Online-measurement systems  
  for agricultural and industrial AD plants — a review and practice
  test.* Compendium :metabolon.  
* [Step 1: Observability literature review](literature_review.md)  
