# ADM1da — Modell-Übersicht

Das mechanistische Prozessmodell hinter diesem Repo ist **ADM1da** (Schlattmann
2011), eine landwirtschaftliche Biogas-Erweiterung des klassischen
Anaerobic Digestion Model No. 1 (Batstone et al. 2002), implementiert in
[`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE).

## Zustandsvektor — 41 Komponenten

Indizes wie in `pyadm1/core/adm1.py`:

### Gelöst (0–11)

| Idx | Symbol | Bedeutung | Einheit |
|---:|---|---|---|
| 0 | `S_su` | Monosaccharide | kg COD/m³ |
| 1 | `S_aa` | Aminosäuren | kg COD/m³ |
| 2 | `S_fa` | Langkettige Fettsäuren | kg COD/m³ |
| 3 | `S_va` | Valerat (gesamt) | kg COD/m³ |
| 4 | `S_bu` | Butyrat (gesamt) | kg COD/m³ |
| 5 | `S_pro` | Propionat (gesamt) | kg COD/m³ |
| 6 | `S_ac` | Acetat (gesamt) | kg COD/m³ |
| 7 | `S_h2` | Gelöster Wasserstoff | kg COD/m³ |
| 8 | `S_ch4` | Gelöstes Methan | kg COD/m³ |
| 9 | `S_co2` | Anorganischer Kohlenstoff (S_IC) | kmol C/m³ |
| 10 | `S_nh4` | Anorganischer Stickstoff (S_IN) | kmol N/m³ |
| 11 | `S_I` | Löslicher Inert | kg COD/m³ |

### Partikuläre Sub-Fraktionen (12–21)

| Idx | Symbol | Bedeutung |
|---:|---|---|
| 12 | `X_PS_ch` | Langsam disintegrierende Kohlenhydrate |
| 13 | `X_PS_pr` | Langsam disintegrierende Proteine |
| 14 | `X_PS_li` | Langsam disintegrierende Lipide |
| 15 | `X_PF_ch` | Schnell disintegrierende Kohlenhydrate |
| 16 | `X_PF_pr` | Schnell disintegrierende Proteine |
| 17 | `X_PF_li` | Schnell disintegrierende Lipide |
| 18 | `X_S_ch` | Hydrolysierbare Kohlenhydrate |
| 19 | `X_S_pr` | Hydrolysierbare Proteine |
| 20 | `X_S_li` | Hydrolysierbare Lipide |
| 21 | `X_I` | Partikulärer Inert |

### Biomasse (22–28)

| Idx | Symbol | Bedeutung |
|---:|---|---|
| 22 | `X_su` | Zucker-Abbauer |
| 23 | `X_aa` | Aminosäure-Abbauer |
| 24 | `X_fa` | LCFA-Abbauer |
| 25 | `X_c4` | Valerat/Butyrat-Abbauer |
| 26 | `X_pro` | Propionat-Abbauer |
| 27 | `X_ac` | Acetoclasten (Methanogene) |
| 28 | `X_h2` | Hydrogenotrophe Methanogene |

### Ladungsbilanz (29–36)

| Idx | Symbol | Bedeutung |
|---:|---|---|
| 29 | `S_cation` | Kationen |
| 30 | `S_anion` | Anionen |
| 31 | `S_va_ion` | Valerat-Ion |
| 32 | `S_bu_ion` | Butyrat-Ion |
| 33 | `S_pro_ion` | Propionat-Ion |
| 34 | `S_ac_ion` | Acetat-Ion |
| 35 | `S_hco3_ion` | Bikarbonat |
| 36 | `S_nh3` | Freier Ammoniak |

### Gasphase (37–40)

| Idx | Symbol | Bedeutung |
|---:|---|---|
| 37 | `p_gas_h2` | H₂-Partialdruck (bar) |
| 38 | `p_gas_ch4` | CH₄-Partialdruck (bar) |
| 39 | `p_gas_co2` | CO₂-Partialdruck (bar) |
| 40 | `pTOTAL` | Gesamter Gasdruck (bar) |

## Wichtigste Erweiterungen gegenüber Standard-ADM1

1. **Zwei-Pool-Disintegration** — X_xc wird durch X_PS (slow, k=0.04 d⁻¹) und
   X_PF (fast, k=0.4 d⁻¹) ersetzt, jeweils in ch/pr/li gesplittet.
2. **Explizite hydrolysierbare Pools** X_S_ch/pr/li (Indices 18–20).
3. **Temperatur-korrigierte Kinetik** via Arrhenius-θ pro Organismengruppe.
4. **Modifizierte Inhibition** — quadratisch/kubisch pH-Inhibition,
   freie-Säure-Inhibition (KIHPRO, KIHAC), Acetat-Kompetition.
5. **Verdoppelte Decay-Rate** k_dec_ac = 0.04 d⁻¹ (klassisch 0.02).

## Eingebaute abgeleitete Größen

Das `ADM1`-Objekt in pyadm1 stellt direkt nutzbare Observables bereit, die als
Extraktoren im `ObservationModel` verwendet werden können:

| Methode | Zurückgegeben |
|---|---|
| `Q_GAS()`, `Q_CH4()`, `Q_CO2()` | Gas-Volumenströme |
| `P_GAS()` | Partialdrücke |
| `pH_l()` | Berechneter pH |
| `VFA()`, `TAC()` | Gewichtete VFA-Summe, Pufferkapazität |
| `FOSTAC()`, `AcvsPro()` | Frühwarn-Verhältnisse |

## Quellen

* Schlattmann, M. (2011). Agrar-erweitertes ADM1.
* Batstone, D. J. et al. (2002). *The IWA Anaerobic Digestion Model No 1.*
  Water Science and Technology 45(10):65–73.
* Weinrich, S. & Nelles, M. (2021). *Systematic simplification of the ADM1.*
  Bioresource Technology 333:125124.
