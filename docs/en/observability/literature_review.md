# Observability of ADM1 State Estimators — Literature Review

> **Step 1** of the observability work in this repo: what does the published
> literature say about which ADM1 states can be reconstructed from which
> sensor measurements?
>
> Step 2 (reconciliation with our ADM1da implementation) will follow separately.

## The central question

A UKF (or any other observer) can only reconstruct states that are made
**observable** by the available measurements. Concretely:

> Which ADM1 states are estimable from which subset of typical plant
> measurements? Which remain fundamentally undetermined, no matter how
> good the filter is?

Three works give structured answers:

| # | Source | Model class | Approach |
|---|---|---|---|
| 1 | Hellmann et al. 2023 ([arXiv:2301.05068](https://arxiv.org/abs/2301.05068)) | ADM1-R4, ADM1-R3, ADM1-R2 (simplified ADM1 variants) | Formal observability/identifiability analysis (algebraic + geometric) |
| 2 | Gaida et al. 2012 ([PMID:22797239](https://pubmed.ncbi.nlm.nih.gov/22797239/)) | Full ADM1 (37 states) | Pattern recognition / machine learning (instead of a classical observer) |
| 3 | Haugen et al. 2014 ([10.1155/2014/572621](https://doi.org/10.1155/2014/572621)) | Modified Hill (4 states + 1 augmented) | UKF with a single measurement |

These three works span the practical range: from the simplest model class with
minimal instrumentation (Haugen) through the analytically tractable middle
classes (Hellmann) to full complexity that can only be handled statistically
(Gaida).

---

## 1. Hellmann et al. 2023 — Formal observability for ADM1 variants

The cleanest analytical study. Systematically investigates which model
variants are *structurally* observable under which measurement sets.

### Setup

**Models (simplified ADM1 variants per Weinrich & Nelles 2021):**

| Model | States | Properties |
|---|---|---|
| ADM1-R4 | 11 | First-order hydrolysis + methanogenesis as a lumped reaction |
| ADM1-R3 | 17 | Explicit acetoclastic methanogenesis, pH inhibition, NH3 inhibition |
| ADM1-R2 | more | Explicit VFA spectrum (acetate, butyrate, propionate, valerate) |

Plus submodels in which individual model parts A–E (biomass decay, gas
solubility, N limitation, pH/NH3 inhibition, pH computation) are isolated and
omitted — to systematically assess which part breaks observability.

**Assumed measurements:**

| Model class | Online | Offline |
|---|---|---|
| ADM1-R4 | CH₄, CO₂ (partial pressures) | TS, VS, IN (Total Solids, Volatile Solids, Inorganic Nitrogen) |
| ADM1-R3 | CH₄, CO₂, pH | TS, VS, IN, (Sac only for algebraic check) |
| ADM1-R2 | CH₄, CO₂, pH, VFA (utopian) | TS, VS, IN |

Note: TS, VS, IN are **lab values** in real operation, modelled with
sample-and-hold.

### Methodology

Two independent approaches, both symbolic:

1. **Algebraic approach** (Mathematica): build Lie derivatives
   $y, \dot y, \ddot y, \dots$ of the outputs, assemble an equation system
   and solve symbolically for the states. Unique solution → globally
   observable; multiple solutions → locally observable.

2. **Geometric approach** (STRIKE_GOLDD toolbox in Matlab, algorithms
   FISPO and ORC-DF): check rank of an observability matrix $\mathcal{O}(x)$
   built from Lie derivatives. Rank $n$ → locally observable.

### Main results

#### ADM1-R4 (11 states): globally observable with CH₄, CO₂, TS, VS, IN

These five measurements suffice to reconstruct all 11 states. Crucially,
TS, VS and IN are **not replaceable**. Hellmann explains:

> *„This follows directly from the model equations: all three states only
> appear in their corresponding differential equation. Therefore, if they
> were not available as measurements, they could not be observable because
> they would not be introduced into the system of equations via other
> measurements, regardless of the degree of time derivatives."* (p. 10)

This is the **single-channel-state effect**: a state that appears only in
its own differential equation and nowhere else is only observable by direct
measurement. No clever time-derivative manipulation helps.

#### ADM1-R3 (17 states): locally observable with CH₄, CO₂, pH, TS, VS, IN (+ Sac)

* The algebraic approach fails for the full ADM1-R3 because the equation
  system becomes too complex (Mathematica kernel dies).
* The geometric approach (STRIKE_GOLDD) successfully shows all 17 states
  locally observable.
* For algebraic resolution of **submodels**, an online Sac (acetate)
  measurement is required — not realistic in the field. The geometric
  approach shows the full ADM1-R3 observable even without Sac.

#### ADM1-R2 (more states): **not observable** without online VFA

> *„In an agricultural setting, these acid measurements are not available
> online. Even when assuming them to be available online, both algorithms
> of the geometric approach failed to evaluate the respective observability
> rank condition, and thus did not allow to draw conclusive statements."* (p. 12)

This is a **hard practical bound**: models with an individual VFA spectrum
exceed what agricultural plant instrumentation can deliver.

#### Identifiability

Structurally identifiable (proven):

| Model | Identifiable parameters |
|---|---|
| ADM1-R4 | Hydrolysis and decay rate constants (time-variant) |
| ADM1-R3 | $\mu_{m,ac}$, $K_{S,ac}$, $K_{I,nh3}$ |

That is, the UKF can co-estimate not only states but also selected kinetic
parameters online — given the same measurement set.

### Structural takeaways for filter design

1. **Some measurements are mandatory, not "nice to have"**: every state
   that appears only in a single DE and nowhere else *must* be measured.
   In ADM1-R4 these are exactly IN, TS, VS.

2. **pH measurement is a structural lever**: pH measurement allows direct
   computation of $S_{H^+}$ (via $\mathrm{pH} = -\log_{10} S_{H^+}$). This
   makes the three ion states $S_{ion}$, $S_{ac^-}$, $S_{hco3^-}$
   **redundant** and removable from the state vector:

   > *„Measuring the pH allows to infer $S_{H^+}$ directly because these two
   > variables are linked via the negative common logarithm. […] However, as
   > $S_{H^+}$ can be directly determined from pH measurements, the states
   > $S_{ion}$, $S_{ac^-}$ and $S_{hco3^-}$ become redundant. Their respective
   > differential equations can be cut out of the system of equations."*
   > (p. 30, *„Neglecting model part E"*)

   A single pH probe effectively reduces the state dimension by 3 and
   eliminates an algebraically non-trivial subsystem (the charge balance $\Phi$).

3. **Complexity scales poorly**: geometric analysis time for ADM1-R3 (17
   states, FISPO): ~12,000 s; after removing individual model parts,
   BMR3+ABC (13 states): ~12 s. Factor 1000. For ADM1-R2 and higher,
   neither method works any more.

---

## 2. Gaida et al. 2012 — Pattern recognition instead of an observer

A different paradigm. Instead of constructing a mathematical observer, the
mapping *measurement → state* is treated as a statistical classification
problem.

### Setup

* **Model:** full ADM1 (Batstone et al. 2002), 37 states
* **Plant:** full-scale agricultural plant (simulation study)
* **Measurements:** biogas flow, CH₄ and CO₂ concentrations in the biogas,
  pH value, substrate quantity per substrate type (maize, grass, manure,
  manure solids)
* **Method:** discriminant analysis / machine learning — static mapping
  function measurement → operating state

### Main statement (from the abstract)

> *„The operating state vector of the modelled anaerobic digestion process can
> be predicted with an overall accuracy of about 90%."*

That is, **90 %** classification accuracy across *all* ADM1 states using only
the standard SCADA measurements.

### Interpretation and context

What Gaida shows is not classical observability in Hellmann's sense. It is
rather: *„under typical operating conditions, the ADM1 state lies on a
low-dimensional manifold that is statistically well predictable, even
though it is not formally reconstructible from the measurements."*

Three implications:

1. **Practical observability ≠ structural observability**: a system can be
   formally unobservable while still operating in a regime where statistical
   methods suffice.

2. **Learning methods need training data** with reliable "ground-truth"
   states. That is the scarcest commodity in the field.

3. **Template for hybrid approaches**: classical observer (UKF) for the
   structurally observable states + statistical model for the rest — exactly
   the idea of AP 4.4 (fusion) in this repository.

> Note: only the abstract of the Gaida paper is available to us. The
> statements above are aggregated from the abstract and from the reference
> in Haugen's paper (Sec. 2.1).

---

## 3. Haugen et al. 2014 — UKF with a single measurement

The other extreme position: no full ADM1 but a drastically simplified
mechanistic model, and only a single online measurement. It still works.

### Setup

* **Model:** Modified Hill Model — 4 states plus one augmented quantity:

  | State | Meaning |
  |---|---|
  | $S_{bvs}$ | Biodegradable Volatile Solids (substrate reservoir) |
  | $S_{vfa}$ | Volatile Fatty Acids (acetate surrogate) |
  | $X_{acid}$ | Acidogenic biomass |
  | $X_{meth}$ | Methanogenic biomass |
  | $S_{vs_{in}}$ | (augmented) Volatile Solids in feed — modelled as random walk |

* **Measurement:** exactly one online quantity — $F_{meth}$ (methane flow).

* **Reactor:** UASB pilot reactor (250 L), substrate cattle manure.

### Methodology

Standard UKF per Wan/van-der-Merwe; tuning of the diagonal elements of $Q$
proportional to the state magnitudes (with scaling factors $m_i$ for
fine tuning).

### Main result

Despite only one measurement, the UKF successfully estimates all 5 augmented
states. Notably, $S_{vs_{in}}$ converges from an intentional 20 % initial
error to the true value within ~15 days.

> *„The linearized reactor model, augmented with $S_{vs_{in}}$, is found
> observable at a number of typical operating points using the obsv function
> of the Matlab Control System Toolbox."* (p. 6f)

So structural observability is given in the linearized sense at typical
operating points.

### Lessons

* With drastic model reduction (5 vs 17 vs 37 states) few measurements
  suffice. The price: the model misses many biological mechanisms, and model
  errors show up as drift in the estimates (Haugen observes this in the plot
  for $S_{vfa}$, Fig. 4 — *„from $t=150\,d$, there is a noticeable difference
  between the estimate and the laboratory analysis of $S_{vfa}$"*, p. 7f).
* Augmented states like an unknown influent $S_{vs_{in}}$ can be
  co-estimated from a single indirect measurement, as long as the system is
  observable in the linearized sense.

---

## Synthesis — who says what about measurement-state dependencies?

### Channel-by-channel: which measurement unlocks which states?

Combined from the three sources. Column meanings:
**direct** = the measurement appears in the output vector and the state
appears only there.
**indirect** = the measurement couples nonlinearly with the state through
another measured signal (e.g. CH₄ flow depends on $X_{ac}$ via kinetics).
**constructed** = the measurement enables algebraic reduction (e.g. pH → $S_{H^+}$
→ ion states eliminated).

| Measurement | Directly unlocks | Indirectly unlocks | Constructed elimination |
|---|---|---|---|
| $p_{CH_4}$ (CH₄ partial pressure) | $p_{CH_4}$, $S_{ch4,gas}$ | $X_{ac}$, $S_{ac}$ (via methanogenesis kinetics) | — |
| $p_{CO_2}$ (CO₂ partial pressure) | $S_{co2,gas}$ | $S_{IC}$, $X_{ac}$ + $X_{h2}$ (via CH₄/CO₂ ratio) | — |
| $\mathrm{pH}$ | — | Inhibition $I_{ac}$ (via all biomass pools) | $S_{H^+}$ → $S_{ion}$, $S_{ac^-}$, $S_{hco3^-}$ eliminable |
| $S_{IN}$ (NH₄-N, lab) | $S_{IN}$ directly | NH₃ inhibition (via $S_{nh3}$) | — |
| TS (Total Solids, lab) | $X_{ash}$ directly (via $S_{h2o}$) | particulate pools $X_{ch/pr/li}$ summarily | — |
| VS (Volatile Solids, lab) | $X_I$ directly | summarily all particulate biological pools | — |
| $S_{ac}$ (acetate, lab or online) | $S_{ac}$ directly | $X_{ac}$ (via acetoclast kinetics) | — |
| $F_{meth}$ (methane flow) | weighted sum of methanogenic contributions | $X_{meth}$, $S_{bvs}$ (in Hill model coupled via kinetics) | — |
| $Q_{feed}$ (substrate flow) | augmented input channels directly | dilution rate $D$ as factor in *every* DE | — |

### Measurement sets vs. model size: rule of thumb

| Model | States | Minimal measurement set for full observability |
|---|---|---|
| Modified Hill (Haugen) | 5 | $F_{meth}$ alone (linear local sense) |
| ADM1-R4 (Hellmann) | 11 | CH₄, CO₂, TS, VS, IN (5 quantities) |
| ADM1-R3 (Hellmann) | 17 | CH₄, CO₂, pH, TS, VS, IN (6 quantities) — structurally proven via geometric |
| ADM1-R2 (Hellmann) | >17 | not observable with agriculture-typical sensors |
| Full ADM1 (Gaida) | 37 | not tractable classically — ML reaches ≈90 % classification accuracy |

Empirical observation: **per additional independent sensor channel, roughly
one state dimension becomes truly separable by the filter.** Crude but
practically tenable — matches the Hellmann table.

### Why not all states are estimable — the three reasons

The three sources together yield a consistent picture of why so many states
remain unobservable in practice:

1. **Single-channel states without a measurement are formally unobservable.**
   Hellmann proves this exactly for TS, VS, IN: no clever filter can
   estimate them if they are not measured directly.

2. **Many states with similar effect on few measurements — the filter
   distributes the innovation signal across the prior**, not across true
   information. In the Hill model the biological pools collapse to 2–3
   separable dimensions; in ADM1 the count is similar under standard
   instrumentation. This is the *practical* (not structural) bound that
   Gaida's empirics reflect: 90 % accuracy across *all* states implies the
   effective manifold is low-dimensional.

3. **Complexity bound of the analysis**: even where structural observability
   theoretically exists (e.g. ADM1-R2), the symbolic tools fail before
   filter implementation. Hellmann explicitly states that "advanced methods"
   not available in standard toolboxes would be required for full ADM1.
   Meaning: we **don't even know** what a UKF could in principle achieve.

### Structural levers — what the literature identifies as decisive

From the three works, three levers emerge that qualitatively (not gradually)
improve observability:

| Lever | Effect | Source |
|---|---|---|
| Adding a **pH probe** | Eliminates 3 ion states from the filter, makes $I_{ac}$ verifiable | Hellmann (Part E redundant), Gaida (pH is one of their 5 measurements) |
| **CH₄/CO₂ separation** (vs. total Q_gas only) | Separates acetoclastic ($X_{ac}$) from hydrogenotrophic ($X_{h2}$) methanogenesis | Hellmann (CH₄ and CO₂ as separate outputs), Gaida (both measurements) |
| **Lab values for TS/VS/IN** | Makes the respective states observable at all | Hellmann (single-channel argument) |
| **VFA lab values** (FOS/TAC) | Makes $S_{ac}$ directly observable, unlocks the VFA spectrum | Hellmann (ADM1-R3 with Sac), Haugen (implied via early-warning discussion) |

---

## Implications for our project

From the perspective of the UKF implemented here (ADM1da, 41 states):

* **ADM1da is closer to ADM1-R3 than to ADM1-R4** in complexity (sub-fraction
  disintegration, inhibitions, charge balance), but with substantially more
  states than ADM1-R3 (41 vs 17). Hellmann's ADM1-R2 result is the warning
  sign: beyond ~17 states without an online VFA sensor, the situation is
  precarious.
* **Real agricultural plants often have even fewer sensors than Hellmann's scenario**
  (no pH, no FOS/TAC, no NH₄-N). The literature rule of thumb is clear:
  from ADM1da we can realistically estimate only a **subset** of the states —
  probably a similar order of magnitude as Haugen's Hill model (4–6
  effectively separable dimensions plus augmented inputs).
* **Future levers**, in order of value:
  pH probe > CH₄/CO₂ split sensor > daily FOS/TAC lab values.
  Each one is marked in the literature as a structural jump, not a gradual
  improvement.

Step 2 (separate file) will concretize which `StateVectorSpec` channel
configuration is realistic for a given sensor suite and which remains
wishful thinking.

---

## References

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

Cited as background:

* Weinrich, S., Nelles, M. *Systematic simplification of the anaerobic
  digestion model no. 1 (ADM1) — model development and stoichiometric
  analysis.* Bioresource Technology, 333:125124, 2021.
* Villaverde, A. F., Barreiro, A., Papachristodoulou, A.
  *Structural identifiability of dynamic systems biology models.*
  PLoS Computational Biology, 12(10):e1005153, 2016.
  (Background on the STRIKE_GOLDD toolbox.)
