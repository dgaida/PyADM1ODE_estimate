# ADM1da — model overview

The mechanistic process model behind this repository is **ADM1da**
(Schlattmann 2011), an agricultural-biogas extension of the classical
Anaerobic Digestion Model No. 1 (Batstone et al. 2002), implemented in
[`PyADM1ODE`](https://github.com/dgaida/PyADM1ODE).

## State vector — 41 components

Indices as in `pyadm1/core/adm1.py`:

### Dissolved (0–11)

| Idx | Symbol | Meaning | Unit |
|---:|---|---|---|
| 0 | `S_su` | Monosaccharides | kg COD/m³ |
| 1 | `S_aa` | Amino acids | kg COD/m³ |
| 2 | `S_fa` | Long-chain fatty acids | kg COD/m³ |
| 3 | `S_va` | Valerate (total) | kg COD/m³ |
| 4 | `S_bu` | Butyrate (total) | kg COD/m³ |
| 5 | `S_pro` | Propionate (total) | kg COD/m³ |
| 6 | `S_ac` | Acetate (total) | kg COD/m³ |
| 7 | `S_h2` | Dissolved hydrogen | kg COD/m³ |
| 8 | `S_ch4` | Dissolved methane | kg COD/m³ |
| 9 | `S_co2` | Inorganic carbon (S_IC) | kmol C/m³ |
| 10 | `S_nh4` | Inorganic nitrogen (S_IN) | kmol N/m³ |
| 11 | `S_I` | Soluble inert | kg COD/m³ |

### Particulate sub-fractions (12–21)

| Idx | Symbol | Meaning |
|---:|---|---|
| 12 | `X_PS_ch` | Slowly disintegrating carbohydrates |
| 13 | `X_PS_pr` | Slowly disintegrating proteins |
| 14 | `X_PS_li` | Slowly disintegrating lipids |
| 15 | `X_PF_ch` | Fast disintegrating carbohydrates |
| 16 | `X_PF_pr` | Fast disintegrating proteins |
| 17 | `X_PF_li` | Fast disintegrating lipids |
| 18 | `X_S_ch` | Hydrolysable carbohydrates |
| 19 | `X_S_pr` | Hydrolysable proteins |
| 20 | `X_S_li` | Hydrolysable lipids |
| 21 | `X_I` | Particulate inert |

### Biomass (22–28)

| Idx | Symbol | Meaning |
|---:|---|---|
| 22 | `X_su` | Sugar degraders |
| 23 | `X_aa` | Amino-acid degraders |
| 24 | `X_fa` | LCFA degraders |
| 25 | `X_c4` | Valerate/butyrate degraders |
| 26 | `X_pro` | Propionate degraders |
| 27 | `X_ac` | Acetoclasts (methanogens) |
| 28 | `X_h2` | Hydrogenotrophic methanogens |

### Charge balance (29–36)

| Idx | Symbol | Meaning |
|---:|---|---|
| 29 | `S_cation` | Cations |
| 30 | `S_anion` | Anions |
| 31 | `S_va_ion` | Valerate ion |
| 32 | `S_bu_ion` | Butyrate ion |
| 33 | `S_pro_ion` | Propionate ion |
| 34 | `S_ac_ion` | Acetate ion |
| 35 | `S_hco3_ion` | Bicarbonate |
| 36 | `S_nh3` | Free ammonia |

### Gas phase (37–40)

| Idx | Symbol | Meaning |
|---:|---|---|
| 37 | `p_gas_h2` | H₂ partial pressure (bar) |
| 38 | `p_gas_ch4` | CH₄ partial pressure (bar) |
| 39 | `p_gas_co2` | CO₂ partial pressure (bar) |
| 40 | `pTOTAL` | Total gas pressure (bar) |

## Main extensions over standard ADM1

1. **Two-pool disintegration** — X_xc is replaced by X_PS (slow, k=0.04 d⁻¹)
   and X_PF (fast, k=0.4 d⁻¹), each split into ch/pr/li.
2. **Explicit hydrolysable pools** X_S_ch/pr/li (indices 18–20).
3. **Temperature-corrected kinetics** via Arrhenius-θ per organism group.
4. **Modified inhibition** — quadratic/cubic pH inhibition, free-acid
   inhibition (KIHPRO, KIHAC), acetate competition.
5. **Doubled decay rate** k_dec_ac = 0.04 d⁻¹ (classic: 0.02).

## Built-in derived observables

The `ADM1` object in pyadm1 exposes ready-to-use observables that can be
plugged in as extractors in the `ObservationModel`:

| Method | Returned |
|---|---|
| `Q_GAS()`, `Q_CH4()`, `Q_CO2()` | Gas volume flows |
| `P_GAS()` | Partial pressures |
| `pH_l()` | Computed pH |
| `VFA()`, `TAC()` | Weighted VFA sum, buffer capacity |
| `FOSTAC()`, `AcvsPro()` | Early-warning ratios |

## Sources

* Schlattmann, M. (2011). Agricultural extension of ADM1.
* Batstone, D. J. et al. (2002). *The IWA Anaerobic Digestion Model No 1.*
  Water Science and Technology 45(10):65–73.
* Weinrich, S. & Nelles, M. (2021). *Systematic simplification of the ADM1.*
  Bioresource Technology 333:125124.
