# Noise injection — the redesign

**Status: the dose-matching rule is approved; one item is open.** This is the specification for
what replaces the old six noise strategies. The rule that the noise level becomes the amount
actually delivered was approved by the author on 2026-08-21. **One question remains open — whether
Laplace is queued as a condition (§7).** Everything else in §7 is settled.

**It is no longer a paper design.** Chat A built the Rust half on 2026-08-26 (`d25bcb0`) and it
matches the reference implementation (§5.1c); the Python half is chat B's and §6.2 step 6 specifies
it. Section 6 marks what is built and what is not, row by row.

Last updated 2026-08-26.

**Rule for this document:** every number carries a peer-reviewed primary source and a
verbatim quote. Anything computed here is marked as computed. Anything unverified says so.
No preprints.

**Companion document**
- `RERUN_PLAN.md` — the single plan for the re-run: what is broken, what gets rebuilt, in what
  order, and which decisions are still open. It is the process document; this file is the
  evidence and the algebra.

**Which document owns what.** This file owns what the noise *is* — the strategies, the algebra,
the parameters, their sources, and the checks that are properties of the noise scheme.
`RERUN_PLAN.md` owns what gets *run* and in what order — the staged design, the replicate counts,
the representation set, the job scripts, the analysis. Neither restates the other; where a fact is
needed on both sides, it lives in its owner and the other points at it.

⚠️ An earlier version of this paragraph listed six state documents as having been deleted.
**They were restored intact on 2026-08-25 and all six are on disk** — see `RERUN_PLAN.md` §11 for
the audit of where each one's content went, and its opening note on `immediate_next_steps.md`,
which is explicitly *not* superseded. `REVISION_GUIDE.md` is the one that is genuinely gone; its
salvage is `RERUN_PLAN.md` §10b.

---

## 0. What this design can claim

Standalone claims, independent of any previous write-up. Status is honest: ✅ established
here, 🟡 shown on QM9 only and needs the experimental datasets, ⬜ not yet tested.

**About real measurement error** (from the literature, §3):
1. ✅ Bioactivity measurement error is **formally not Gaussian** — Anderson-Darling rejects
   normality at p < 2×10⁻¹⁶ on 41,733 comparisons, and a Laplace fits instead.
2. ✅ Measurement error **does not depend on the value being measured** — tested directly on
   16,844 repeat measurements and refuted, across potency and every ligand property examined.
3. ✅ **62% of measurement variance is between laboratories.** The dominant structure in real
   error is provenance, not label magnitude.
4. ✅ The far tail of disagreement is **database error, not imprecision** — essentially every
   disagreement beyond 2.5 log units is a wrong unit, wrong target or wrong assay.

**About how noise affects models** (from the experiments here, §5):
5. ✅ **Comparing noise types at a common nominal setting is confounded.** Six commonly-used
   types deliver between 0.49× and 2.00× the same amount of noise, and their apparent
   severity ordering is entirely explained by that.
6. 🟡 **The shape of random noise barely matters once the amount is matched.** Across
   heavy-tailed, grouped and sparse-contamination noise, differences stay under 0.02 in R²
   at realistic levels and never exceed 0.10. This extends a known result for evenly-spread
   noise to the concentrated case, which had not been tested.
7. 🟡 **Censoring is a different order of problem** — twelve times more damaging than any
   shape difference, because it biases labels in one direction rather than scattering them.
8. 🟡 **How much your model choice is worth depends on the kind of error.** For random error
   it is worth up to 0.27 in R², and the gap widens as error grows. For censored data it is
   worth nothing at any level.

**Not yet tested** (⬜): all of 6–8 on experimental data; whether any model can identify
*which* labels are unreliable; whether the grouped noise type behaves differently from the
others in ways accuracy alone does not reveal.

---

## 1. The control quantity — what replaces σ

There are **two separate jobs** here, and conflating them is what caused the original
confound. Keep them apart.

### Job 1 — make the strategies comparable to each other (within a dataset)

**Control the realised dose τ: the root-mean-square noise actually added, in the label's
own units.** Every strategy must deliver the same τ, or a comparison between strategies is
measuring amount rather than shape.

This is the fix. Today a single knob is set and each strategy delivers whatever it happens
to deliver — between 0.49× and 2.00× the Gaussian amount on QM9. **τ must be the result,
not the knob.**

### Job 2 — choose what τ to use, and report it comparably (across datasets)

Two different axes, and they answer different questions:

| Axis | Use it for | Why |
|---|---|---|
| **τ in the label's own units** (log units for the experimental sets) | **Choosing** the noise levels, and claiming realism | Assay error in log units is roughly constant across endpoints *regardless of label spread*, so a fixed value in log units is more defensible as "one unit of real error" than a spread-matched one |
| **k = τ / SD(y)** | **Reporting** and cross-dataset comparison | Unitless, so electronvolts and log units land on one axis, and it is what determines how hard the learning problem becomes |

**Report both.** They are not interchangeable: a dataset with a narrow label spread suffers
far more from the same absolute error, which is exactly why Caco-2 degrades first
(τ ≈ 0.35 log₁₀ but a narrow dynamic range).

**Practical consequence for the grid:** choose the levels in τ for the experimental datasets,
anchored to the assay error in §4, and in k for QM9, which has no assay error to anchor to. Report
the other quantity alongside in both cases.

**The grids themselves are in §6.4 and nowhere else.** An earlier draft of this section proposed
its own ladders; they were superseded by the range-finding run (§5.5) and have been removed rather
than left to disagree with §6.4.

**Precedent for the k form:** Zhao Y, Wang J, Sedykh A, Zhu H (2017). *ACS Omega*
2(6):2805–2812 — *"the standard deviation of each data set was multiplied by a parameter k
(k = 0.1, 0.2, 0.5, 1.0)."*

**Precedent for the absolute form:** Kolmar & Grulke put the delivered standard deviation on
the x-axis and then benchmark it against a published assay figure — *"1.1 log units of noise
were added, or 1.6 times the average standard deviation reported in ChEMBL."*

### The rule that makes every strategy honour τ

> For each strategy, compute its **unit dose** `G` — the root-mean-square of its
> per-molecule scale factors, times the shape's own unit standard deviation — from the
> **clean training labels only**. Then set the strategy's scale to `τ / G`.
>
> Write `G`, the realised `τ` and the resulting `k` into the results file so every figure
> traces back to them.

**Why root-mean-square and not something else:** RMSE and R² are second-moment quantities,
so matching the noise's second moment is what makes them comparable. The machine-learning
convention of fixing the *fraction* of corrupted labels does not transfer — Song et al.
(2023) *IEEE TNNLS* 34(11):8135–8153 define noise by a rate and state explicitly that they
*"focus on the classification setting"*. A flipped class label has no magnitude; a
perturbed continuous label is nothing but magnitude.

---

## 2. The strategies

| # | Strategy | What it is | Why it is in | Size | Shape |
|---|---|---|---|---|---|
| 1 | **Gaussian** | Every label nudged by a similar amount | The reference case, and what both direct predecessors used, so results stay comparable to theirs | `τ` | — |
| 2 | **Student-t** | Same, but badly-wrong labels far more common | Real error is formally non-normal (§3.1). Gaussian is this strategy's ν→∞ limit, so the two nest on one number | `τ` | **ν = 5**, settled 2026-08-27 — one setting, not three (§5.8) |
| 3 | **Laplace** *(stage 2 only, if at all)* | A specific heavy-tailed shape | The distribution actually **fitted** to real bioactivity differences (§3.1). Citational value — statistically it sits near ν = 6, and measured, it is indistinguishable from Gaussian (§5.8) | `τ` | fixed |
| 4a | **Grouped — wider** | Whole scaffold groups get wider errors, still centred on the true value | **Best-evidenced of the set.** Within-laboratory error must be multiplied by about three to reach between-laboratory error (§3.3). The only zero-mean condition where noise is predictable from structure, so the only one that tests whether a model can *spot* bad data | `τ` | λ = 3, affected **molecule** fraction ≈ 0.2 |
| 4b | **Grouped — shifted** ✅ | Whole scaffold groups have their labels pushed in one direction by a constant | 62% of real measurement variance sits **between** laboratories (§3.3) — and that describes laboratory *averages* differing, which is an offset, not a widening. Added 2026-08-26; see §2a | `τ` | ρ = 0.62, from the source |
| 5 | **Outlier** | A random few labels are simply wrong | Real contamination is transcription errors, wrong target, wrong assay (§3.4). Formally Huber's contamination model | `τ` | **p = 10%**, λ = 3, settled 2026-08-27 — one setting, not three (§5.8) |
| 6 | **Censoring** ✅ | Values past the assay limit recorded as the limit | The most *prevalent* real mechanism (§3.5), and the only one that is not zero-mean | **cannot be dose-matched** — separate axis | fraction censored: 10%, 25%, 40% |

Strategies 1–5 are dose-matched. 1, 2, 3, 4a and 5 are zero-mean; **4b is zero-mean in
expectation but not in any one run — that asymmetry is its mechanism, not a defect**. Strategy 6
is neither dose-matched nor zero-mean, which is why it needs its own axis and its own figure.

**✅ Which of these actually run, and at which stage, was settled on 2026-08-27 and lives in
`noise_conditions.json` at the repository root.** That file is read by `rust/tests/noise_gates.rs`
and by `scripts/test_noise_conditions.py`, so a grid that stops matching it fails a test rather than
quietly running. The evidence is §5.8. In short: Gaussian, both grouped conditions and censoring at
full grid; one Student-t setting and one Outlier setting at depth only; Laplace optional; the other
four settings dropped; the skewed draw never built.

### The exact algebra

| Strategy | Scale that delivers realised dose τ |
|---|---|
| Gaussian | `s = τ` |
| Student-t, ν d.f. | `s = τ·√((ν−2)/ν)` — **requires ν > 2**, below which the variance is undefined and "same amount of noise" stops meaning anything |
| Laplace | `s = τ/√2` |
| Grouped — wider | `s_low = τ / √(1 − f + f·λ²)`, `s_high = λ·s_low`, where `f` is the fraction of **molecules** in affected groups |
| Grouped — shifted | `ε_i = √ρ·τ·b_{g(i)} + √(1−ρ)·τ·e_i`, with `b_g` and `e_i` unit draws from the shape. No solver step — the two variances sum to `τ²` by construction. See §2a |
| Outlier | `s_base = τ / √(1 + p(λ² − 1))`, contaminated points get `λ·s_base` |

### Parameter values, each sourced

- **λ = 3** — four independent lines agree. Solubility within-laboratory error must be
  *"multiplied by a factor of 3"* to reach between-laboratory error (Avdeef 2019,
  *ADMET & DMPK* 7(3):210–219); solubility test sets 0.62/0.17 = 3.6 (Llinàs & Avdeef 2019,
  *JCIM* 59:3036–3040); potency 0.68/0.17–0.22 = 3.1–4.0 (Kalliokoski et al. 2013); and it
  is Tukey's conventional value for contaminated normals.
- **p = 1–10%** — Hampel (2001): *"for scientific routine data, not taken with utmost care,
  their fraction is typically between 1 percent and 10 percent."*
- **Affected group fraction ≈ 0.2** — **no published number exists.** Choose 0.2 and say so.

### Two implementation points settled by testing, not opinion

1. **The Outlier type must use a wider Gaussian, not label replacement.** The
   "replace the bad label with a random draw from the label range" variant **cannot be
   dose-matched** — realised dose came out +102% and +350% off target, because the dose is
   then fixed by the contaminated fraction and the label spread rather than by the knob.
2. **The Grouped type must compute the molecule fraction from the actual scaffold
   assignment**, not assume it equals the group fraction. With evenly-sized simulated
   clusters the two coincide; with real Murcko scaffolds, which are very unevenly sized,
   they will not.

### 2a. Grouped noise — the algebra, and two rules the real scaffolds force

Written 2026-08-26 by chat B, closing the 🔴 TODO left for chats A and B in `RERUN_PLAN.md` §13.3.
**Both implementations must follow this.**

#### The shifted condition

Every group *g* receives a constant offset; every molecule receives its own error on top.

> `ε_i = √ρ · τ · b_{g(i)}  +  √(1 − ρ) · τ · e_i`,   with `b_g`, `e_i` unit draws from the shape

The two variances sum to `τ²` by construction, so this condition is dose-matched to the same `τ`
as the other four **without a solver step**. `ρ` is the share of total variance carried by the
group-level term, and it comes from the source rather than from a judgement: **Bentz et al. (2013)
Table 7** gives laboratory 62%, laboratory × experiment 20%, residual error 10%, cell line 8%.

**ρ = 0.62.** (Recorded alternative, not taken: 62/92 = 0.674, excluding the cell-line term on the
grounds that it is not a provenance effect. The difference is well inside the run-to-run spread
below, so it is not worth an extra condition.)

**Do not centre the offsets.** This condition is not zero-mean in any one run, and that is exactly
the mechanism under test — the study's emerging result is that error pushing in one direction hurts
far more than error that scatters. Censoring shows that for a whole dataset; grouped-shifted shows
it for a chemical family, at matched amount.

It also closes a gap the wider condition has: the affected-group fraction has no published number
and has to be chosen. Under the shifted version there is nothing to choose.

#### Rule 1 — select groups by molecule fraction, never by group fraction

Real Murcko scaffolds are very unevenly sized, so a fraction of *groups* does not control who gets
hit. **Measured on the first 10,000 molecules of `data/QM9/raw/gdb9.sdf`** (chat B, 2026-08-26):
855 distinct scaffolds, **32.2% of molecules share one empty — acyclic — scaffold**, and 523
scaffolds are singletons. Over 200 draws at a nominal group fraction of 0.2:

| Selection rule | Realised affected molecule fraction | Realised dose (target 1.0) |
|---|---|---|
| A fraction of **groups** | 0.067 – 0.551 (SD 0.135) | 0.964 – 1.030 |
| Until a **molecule** fraction is reached | 0.200 – 0.515 (SD 0.108) | 0.975 – 1.035 |

Dose matching survives either way, because the solver uses the *realised* fraction — but who gets
hit swings eightfold under the group rule, and that is a property of the condition, not noise in it.

So: shuffle the groups, add them one at a time, skip any group that would take the cumulative
molecule fraction further from `f` than stopping would, and stop at the closest approach. **Write
the realised fraction to `affected_molecule_fraction` on every row** (`RERUN_PLAN.md` §5.2 already
requires the column; nothing populated it).

#### Rule 2 — the empty Murcko scaffold is not a group

Acyclic molecules become **singleton groups**. Otherwise a single offset draw moves a third of QM9
at once. Measured over 200 draws at ρ = 0.62, target dose 1.0:

| Grouping | Groups | Realised dose | Mean label shift |
|---|---|---|---|
| Raw Murcko, one empty group | 855 | 0.795 – 1.493 (**SD 11%**) | −0.63 to +0.75 |
| Empty scaffold split into singletons | 4,070 | 0.904 – 1.187 (SD 4.6%) | −0.25 to +0.25 |

Also record the group count and the largest group's share of molecules, per dataset and replicate.

#### Rule 3 — the flat-dose gate is about the solved scale, not one realisation

Grouped-shifted has few effective degrees of freedom, so its realised dose cannot meet a ±0.5%
per-run tolerance on QM9 by construction. **Fix the population dose and record what was
delivered** — the same ruling already made for Student-t ν = 3 in §5.1b, so the rule stays uniform
across conditions rather than special-casing one. Concretely, gate 1 of `RERUN_PLAN.md` §8 asserts:

1. `unit_dose × solved_scale == τ` **exactly**, for every dose-matched condition; and
2. the **mean** realised dose over at least 20 seeds is within tolerance of `τ`, and of every other
   condition's.

⚠️ **Twenty seeds is not enough, and the gate built on it failed at random.** Measured by chat G on
2026-08-26 against 3,200 real QM9 training labels with real scaffold groups: the per-run spread of
the delivered dose is 1.3% for Gaussian, **3.9% for grouped-shifted and 6.9% for Student-t ν = 3**.
Over twenty seeds those two conditions' means therefore wander by ±0.9% and ±1.5%, and the 3% flat-
dose criterion is breached by sampling noise alone — it reported a 3.39% spread and failed, on labels
where 400 seeds put grouped-shifted at **+0.03% ± 0.19%**, exactly on target. The gate now averages
**200 seeds** and the same labels give 1.29%. The companion gate that checks Student-t nests Gaussian
at ν → ∞ had the same defect in sharper form — it compared a *single* draw against a single draw, so
it failed about a quarter of the time — and now averages 50. Both are in `rust/src/main.rs`.

The per-run spread — roughly ±5% for grouped-shifted on QM9 — is then a number the Methods states,
not a check that fails.

### What is dropped, and why

| Current strategy | Verdict |
|---|---|
| `legacy` (Gaussian) | **Keep** — becomes strategy 1 |
| `outlier` | **Keep the name and the idea, change the selection rule** — victims chosen at random, not by whether the label is extreme |
| `heteroscedastic` | **Drop** — folded into Grouped |
| `value_proportional` | **Drop** — folded into Grouped |
| `threshold` | **Drop** — folded into Grouped |
| `quantile` | **Drop** — folded into Grouped |

Four of the six go for one reason: they all assume error depends on the measured value,
and that has been directly tested and disproved (§3.2).

`threshold` has a second, independent problem: its cut fires on `|y| ≥ 1.0` against raw
electronvolts, and QM9's smallest gap is 0.669 eV, so **99.99925% of the 133,885 molecules clear
it** — ten molecules in the whole dataset escape the cut. (Verified from
`data/QM9/raw/gdb9.sdf.csv`; earlier notes quoting 2.08 eV and 100% were computed from the first
10,000 molecules in file order, and the pipeline samples at random from the whole set.)
It is homoscedastic Gaussian noise at double dose, with no threshold behaviour at all. Had
the pipeline stayed in Hartree the same cut would have caught *zero* molecules. Its entire
character is an accident of a unit conversion.

---

## 3. The evidence

### 3.1 Real error is not Gaussian — and there is a formal test

**Krüger FA, Overington JP (2012).** Global analysis of small molecule binding to related
protein targets. *PLoS Comput Biol* 8(1):e1002333. Open access, read in full.
2,782 compound-ortholog pairs; 41,733 paralog combinations.

> "The density distribution of differences in bioactivity has a central peak at 0 and is
> **non-normal as established by Anderson-Darling test (p<2e-16)**."

They then fitted a **Laplace** distribution — *"Both distributions can be approximately
described by a Laplace distribution... the scale parameter b... is b = 0.7 for the paralogs
and b = 1.3 for human-to-rat orthologs."*

⚠️ **Cite this accurately.** The paper never uses the words "heavy-tailed", "leptokurtic"
or "Gaussian". Cite what they did — rejected normality, fitted Laplace. Also note their b
values describe differences between *related proteins* (biology); their separate
inter-assay control distribution (measurement error) is stated only to *"closely resemble"*
the ortholog one. **Do not inject b = 0.7 or 1.3 as measurement error.** Take the shape,
dose-match it like everything else.

**Kalliokoski et al. (2013)** could only fit a Gaussian after truncating at 1.5–2.5 log
units, and the fitted width grows with the cut (0.80 → 0.84 → 0.86). The largest single
disagreement is **7.7 log units** — a nine-sigma event that a Gaussian says should occur
6×10⁻¹⁵ times in 16,844 pairs. It occurred once.

How wrong a Gaussian is in the tail *(computed from their Figure 2)*:

| Disagreement | Observed | Gaussian predicts | Wrong by |
|---|---|---|---|
| > 2 log units | 9.0% | 4.7% | 1.9× |
| > 3 log units | 2.2% | 0.29% | 7.7× |
| > 4 log units | 0.40% | 0.007% | **58×** |

Fitting the full tail gives Student-t with **ν ≈ 4–6**; fitting only the truncated core
gives ν ≈ 8–16.

⚠️ **Honest limitation to state in the paper:** matching the *real* tail needs about
ν = 1.1, which has no finite variance and therefore cannot be dose-matched. Strategy 2 at
ν = 3 is a **conservative** heavy-tailed setting. Real data is heavier than anything that can
be fairly tested.

### 3.2 Error does not depend on the measured value

**Kalliokoski T, Kramer C, Vulpetti A, Gedeck P (2013).** Comparability of mixed IC50 data
— a statistical analysis. *PLoS ONE* 8(4):e61007. 16,844 repeat measurement pairs.

> "We checked whether the ΔpIC50 depends on the overall activity measured or on
> physicochemical ligand properties like logP, logD, molecular weight (MW), polar surface
> area (PSA), the number hydrogen bond acceptors (HBA), the number hydrogen bond donors
> (HBD) or the number of rotatable bonds... **The ΔpIC50's depend neither on the average
> measured pIC50 nor on any of the ligand properties examined.**"

> "One might assume that higher IC50 values show a larger variability than for example
> single digit µM IC50 values because of solubility limits. **However, our analysis shows
> that on the average this is clearly not the case.**"

Corroborated three ways:
- **Kramer C, Dahl G, Tyrchan C, Ulander J (2016).** *Drug Discov Today* 21(8):1213–1221.
  Every AstraZeneca assay 2005–2014: *"an average experimental uncertainty of less than a
  twofold difference and no technologies or assay types had higher variability than
  others."* A constant fold-difference is a constant standard deviation in log space.
- **Srinivasan B, Lloyd MD (2025).** *J Med Chem* 68(3):2052–2056. Potency data is
  lognormal; taking logs *"converts multiplicative variability... to arithmetic
  variability"* — constant error on the scale actually modelled.
- **Horwitz W, Albert R (2006).** *J AOAC Int* 89(4):1095. Error scales as concentration to
  the power 0.85 — near-constant *relative* error, i.e. near-constant error in logs.

Where value-dependence *does* exist, it takes the form of **censoring at the assay limits,
not smooth scaling.** Outside the working range an assay does not return a noisier number;
it returns no number, only a `>` or `<` qualifier. That is strategy 6, not a scale rule.

### 3.3 Most variance is between laboratories

**Bentz J, O'Connor MP, Bednarczyk D, et al. (2013).** *Drug Metab Dispos* 41(7):1347–1366.
23 participating laboratories. Variance decomposition of log efflux ratio (Table 7):

| Source | Share |
|---|---|
| **Laboratory** | **62%** |
| Laboratory × experiment | 20% |
| Residual error | 10% |
| Cell line | 8% |

Also, justifying the modelling scale: *"Log ER and log IC50 were used in variance component
analysis since data are log normally distributed."*

Corroborating: **Landrum GA, Riniker S (2024)**, *JCIM* 64(5):1560–1567 — curating IC50 data
down to a single assay cuts mean absolute disagreement from **0.50 to 0.27 log units**, so
roughly half the apparent noise in public potency data is provenance, not imprecision.

**This is the strongest single argument in the whole design.** The dominant structure in
real label error is *where the measurement came from*, not how big the value was.

### 3.4 The tail is database errors, not imprecision

Kalliokoski's Table 2 manually inspected 10 pairs per disagreement band. Number invalid:
Δ 4.7–7.8 → **9/10**; Δ 3.2 → **10/10**; Δ 2.5 → **8/10**; Δ 1.5 → 6/10; Δ 1.1 → 1/10;
Δ 0.02 → 0/10. Error types named: unit-transcription, receptor-subtype confusion,
cellular-versus-biochemical assay mix-up, undifferentiated stereochemistry, retracted papers.

> "The extreme disagreements are all due to clear errors."

This is why strategy 5 is called Outlier and why its victims are chosen at random: a
mistyped unit is a property of the *record*, not of the value.

### 3.5 Censoring is the most common mechanism

**Svensson et al. (2025)**, *Artificial Intelligence in the Life Sciences* 7:100128, Table 1.
Fifteen real industrial assays from a pharmaceutical company, with the censored fraction reported
for each. Counting left- and right-censoring together:

- **Thirteen of the fifteen carry censored labels.** Only two do not, and the paper says so
  outright: *"Finally, two of the target-based assays, Target 3 and Target 6, do not have any
  censored labels."*
- **Eight of the fifteen sit between 25% and 63%** — the three CYP assays at 61%, 63% and 58%,
  hERG at 42%, and four target assays at 43%, 35%, 33% and 25%.
- The lipophilicity assay is **88,114 compounds with 8% right-censored** and no left-censoring,
  which is the largest assay in the set.

⚠️ **Correction, 2026-08-26.** This section previously said "25–63% of labels censored in ten of
fifteen assays". The 25–63% range is right; **the count was not** — it is eight of fifteen in that
band, and thirteen of fifteen with any censoring at all. Checked against Table 1 of the paper.

Assay Guidance Manual (NCBI Bookshelf NBK91994): values outside the tested range *"should be
'<Xmin' or '>Xmax', as appropriate."*

### 3.6 What Heid et al. already published

**Heid E, McGill CJ, Vermeire FH, Green WH (2023).** *JCIM* 63(13):4012–4029.

> "We did not observe any difference in overall model performance between different error
> distributions, **as long as the mean and standard deviation of the noise was the same**,
> respectively. Though noise distributions found in real data may be non-Gaussian, **if
> homoscedastic**, they should still follow the same trends."

This confirms dose-matching is the right move. It also means "shape does not matter at equal
dose" is **already published for evenly-spread noise**. The open question — where novelty
remains — is whether it holds when noise is *concentrated*, either on a structural group
(strategy 4) or on a sparse random subset (strategy 5). Heid explicitly excludes that case.

⚠️ **Do not claim Heid et al. showed noise is structure-dependent.** A common misreading —
and one this project previously made — is that they *"used mean-variance estimation and
bias-variance decomposition to show that noise can be tied to specific modalities, for
example structure dependence."* They did not. They **imposed**
structure-dependent noise by hand — *"Gaussian noise of standard deviation 20 kcal/mol for
nitrogen-containing molecules and... 2 kcal/mol for non-nitrogen-containing molecules"* —
then showed a method could detect what they themselves injected. Rewrite the sentence.

---

## 4. Anchoring to reality — what a noise level means

| Endpoint | One unit of real error | Source |
|---|---|---|
| **pIC50**, mixed public data | **0.68 log units** | Kalliokoski et al. 2013, *PLoS ONE* 8(4):e61007 |
| **pKi**, mixed public data | **0.54 log units** | Kramer et al. 2012, *J Med Chem* 55(11):5165–5173 |
| **hERG** (Ki) | **0.5–0.7**; use 0.54 as the point estimate | Kramer 2012 as a labelled stand-in; hERG-specific bracketing from Alvarez Baron 2025 *Sci Rep* 15:29995 and Sato 2018 *PLoS ONE* 13:e0199348 |
| **Caco-2** (efflux ratio) | **≈0.35 log₁₀ units (≈2.2-fold)** | *Computed* from 11 laboratories' digoxin efflux ratios, Bentz et al. 2013, Table 6 |
| **logD** | **≈0.15** within a laboratory; **±0.5** between methods | Wenlock et al. 2011, *J Biomol Screen* 16(3):348–355; OECD Test Guideline 117 |

Notes:
- pIC50 and pKi are already single-measurement values — Kalliokoski divides the paired
  standard deviation by √2 explicitly. 0.68 means *68% of measurements agree within a
  factor of 4.8*.
- **hERG has no target-specific published standard deviation.** Kramer's 0.54 is
  ChEMBL-wide and must be labelled a stand-in. It is the right one: the endpoint matches
  (Ki), it is a standard deviation, and it is bracketed by a hERG cross-assay-format RMSD
  of 0.737 (Sato 2018, n=209) and a best-case standardised-protocol residual of 0.18
  (Alvarez Baron 2025).
- **Caco-2 is noisier per log unit than it looks**, because its dynamic range is narrower.
  Corroborated by Larregieu & Benet 2013 *AAPS J* 15(2):483–497 (median 0.33, computed) and
  O'Hagan & Kell 2015 *PeerJ* 3:e1405 (*"within a factor of 2–5"* = 0.30–0.70).
- **QM9 has no assay error to anchor to.** Kolmar & Grulke state it outright: *"quantum
  mechanical calculations do not have random experimental error, because the same
  calculation will give exactly the same number."* For QM9, `k` is the only honest axis and
  the noise is a controlled perturbation, not a simulated measurement.

**The ceiling this imposes**, verbatim from Kramer et al. 2012:

> "The maximum possible squared Pearson correlation coefficient (R²) on large data sets is
> estimated to be **0.81**."
> "Models that yield errors smaller than the experimental uncertainty are necessarily
> overtrained."

⚠️ Attribute as *estimated for heterogeneous public Ki data with SD 0.54 pKi units*, not as
a universal ceiling — the derivation is in the paywalled body and is unverified.

### 4c. The same anchors on the shared scale — what a QM9 level of 1.0 or 1.5 corresponds to

Added 2026-08-27, after the author asked what can be cited at a level between 1.0 and 1.5.

Since the one-shared-grid decision (`RERUN_PLAN.md` §2.12) every dataset sweeps the same ladder,
read as a fraction of that fold's clean training label spread. That makes the published assay
errors above convertible onto the QM9 axis: **divide the assay error in log units by that
dataset's clean label spread.**

The three spreads, and where each comes from:

| Dataset | Clean label spread | How it is known |
|---|---|---|
| hERG (pKi) | **0.896** | Measured directly on the real data after the shared-grid change — forest + ECFP4, level 1.00 delivered 0.896 log units against a spread of 0.896 (`RERUN_PLAN.md` §2.12) |
| Caco-2 | **≈0.44** | Derived: the top of the grid carries 1.9× the 0.35 assay error, so 1.5 × spread = 0.665 |
| logD | **≈1.19** | Derived: the top of the grid carries 11.9× the 0.15 within-lab error, so 1.5 × spread = 1.785 |

**Every anchor in §4 placed on the shared scale:**

| Source | What it measures | log units | On the shared scale |
|---|---|---|---|
| Wenlock et al. 2011 | logD, within one laboratory | 0.15 | **0.13** |
| OECD Test Guideline 117 | logD, between methods | 0.50 | **0.42** |
| Kramer et al. 2012 | pKi across public data (hERG stand-in) | 0.54 | **0.60** |
| Sato et al. 2018 | hERG, between assay formats (RMSD, n = 209) | 0.737 | **0.82** |
| Bentz et al. 2013 | Caco-2, median across 11 laboratories | 0.35 | **0.79** |
| O'Hagan & Kell 2015, reporting Hayeshi et al. 2008 | Caco-2, top of the published inter-laboratory range (*"within a factor of 2–5"* = 0.30–0.70) | 0.70 | **1.58** |

**The answer to "what can I cite between 1.0 and 1.5".**

1. **Caco-2 inter-laboratory variability at the top of its published range is 1.58 of the Caco-2
   label spread.** This is the only single published figure that lands there. Cite it to
   **O'Hagan & Kell 2015** as the secondary source for Hayeshi et al. 2008 — §4's blocklist forbids
   citing the fold figures to Hayeshi directly, and the *"within a factor of 2–5"* wording is the
   verified one (§4b item 5).
2. **Twice the published assay error**, which is already the rule the experimental grids were built
   on (§6.4: *"each grid brackets one unit of real error and runs to roughly twice it"*), puts
   **Caco-2 at 1.58 and hERG at 1.21**. This is the stronger argument of the two, because it is a
   rule the design already applies rather than a single number picked because it fits.
3. **Nothing anchors logD above 0.42.** Its label spread is wide and its assay is precise, so on
   this scale logD never leaves the bottom of the grid. Say so rather than letting a reader
   average across the three.

**What cannot be claimed, and it has to be stated in the Methods.** QM9 has no assay error at all —
Kolmar & Grulke, quoted in §4: *"quantum mechanical calculations do not have random experimental
error."* A QM9 level of 1.5 is **not** a realistic measurement error and must never be described as
one. What the conversion above buys is narrower and still worth having: a level of 1.5 is the amount
of error that the noisiest real assay in this study actually carries between laboratories. That is a
statement about the range being realistic for *some* endpoint, not about QM9.

⚠️ **One number needs settling and it is a one-line check.** §6.4 of this document says one unit of
real error is **0.76** of the Caco-2 label spread; `RERUN_PLAN.md` §2.12 says **0.79**. The two imply
Caco-2 label spreads of 0.461 and 0.443. Nothing in either document records the spread directly —
both figures are derived. Print the clean Caco-2 training label SD once on the cluster and fix both.
Until then, quote the anchor as **≈0.8 of the label spread** and the top-of-range figure as
**≈1.5–1.6**, not to three digits.

### 🚫 Numbers that must NOT enter the paper

Each is in circulation, each is wrong, each was traced to source during the August 2026
literature chase.

| Bad number | Why it is wrong | Use instead |
|---|---|---|
| "0.68 log units for **pKi**" | 0.68 is **pIC50**. The two get swapped constantly — this project's own earlier notes had them swapped | pKi = **0.54** (Kramer 2012); pIC50 = **0.68** (Kalliokoski 2013) |
| "**Matsson et al. 2019**, Caco-2 intra-lab CV ~14%" | **No such paper exists.** Matsson's only 2019 paper is about unbound intracellular drug fraction | **Prieto et al. 2010**, *ATLA* 38(5):367–386 — median CV 10.4% and 14.7% |
| "**Pham-The 2013**, Caco-2 inter-lab standard error 18.5%" | That paper contains **no reproducibility experiment**. 18.5% is a regression standard error | Bentz et al. 2013 |
| "Caco-2 SD reaches almost **0.6 log units**" | **Unit error.** Lanevskij & Didziapetris (2019) converted Lee et al. (2017)'s *range-normalised, dimensionless* RMSE of 0.581 into "log units" | **≈0.35 log₁₀**, computed from Bentz 2013 Table 6 |
| "Hayeshi 2008 found 4-fold / 44-fold / 16-fold variability" | Those are **Fagerholm's re-analysis in a rejected preprint**, not statements in Hayeshi | O'Hagan & Kell 2015 — *"within a factor of 2–5"*, cited as secondary |
| Any Caco-2 figure from **Fagerholm et al., bioRxiv 2022.09.27.509731** | Preprint, not peer reviewed | Bentz et al. 2013 |

**Rule going forward: no number is written down without a peer-reviewed primary source and a
verbatim quote.**

### 4a. Reconciling the two literature passes — read before citing anything here

There were **two** independent literature efforts, four days apart, and they do not agree. The
first (2026-08-20) produced the verified-quote layer now in §4b, with 35 quotes confirmed
character-for-character and 4 discarded. The second (2026-08-24) produced the anchor table above.
Where they differ, the difference is real and needs settling, not averaging.

**Settled here, from the primary source.** §4b attributes the *"within a factor of 2–5"*
characterisation of the Caco-2 inter-laboratory spread to "Kell DB, Oliver SG". **That
attribution is wrong.** Checked against the retrieved article itself
(`research_archive/28450b4e/kell.xml`, contributor block): PeerJ 2015;3:e1405 is
**O'Hagan S, Kell DB**, *The apparent permeabilities of Caco-2 cells to marketed drugs*. The
anchor table above has it right. The quote itself is confirmed verbatim in the retrieved text.

**Open, and yours to settle.**

| Endpoint | Anchor table above (2026-08-24) | Verified-quote layer, §4b (2026-08-20) |
|---|---|---|
| **Caco-2** | ≈0.35 log₁₀, *computed* from 11 laboratories' digoxin efflux ratios, Bentz 2013 Table 6 | No number from Hayeshi itself is quotable — the full text was never retrieved. Every figure is secondary: efflux ratios 0.18–3.76 (Chen et al. 2017) and a factor of 2–5 (O'Hagan & Kell 2015) |
| **logD** | ≈0.15 within a laboratory (Wenlock 2011); ±0.5 between methods (OECD 117) | Bruneau's 0.27 **failed verification and must not be used**. Repeat-test error is MAE 0.7 before curation and 0.48 after (Niu et al. 2024, PharmaBench Table 5, fully verified, open access) |

These are not small differences. On Caco-2 the two routes give 0.35 against a two-to-five-fold
spread, and one of them is a computation you would have to defend rather than a quote you can
cite. On logD they differ by a factor of three, and the anchor table's source is not in the
verified set while the verified set's source is not in the anchor table.

**My recommendation, and it is only that.** Take the verified layer as authoritative for what can
be *quoted*, and the anchor table as the working values where no quote exists — but say which is
which in the Methods. For logD specifically, use Niu 2024: it is open access, fully verified,
covers the endpoint directly, and removes the dependence on a number that failed checking.

**Two methodological points from §4b that the anchor table does not carry, and should.**

- A spread computed from *pairs* of measurements is √2 larger than the spread of a single
  measurement. Anything injected into a label must be the per-measurement figure. Cite
  Kalliokoski 2013 (Figure 5 caption) for this, not Kramer 2012.
- Kramer's 0.54 is sensitive to how outlying pairs are handled; a comparable re-analysis in
  Kalliokoski lands at 0.47. Quote ≈0.5 as an approximate anchor, not a constant.

---


## 4b. Verified verbatim quotes — the primary-source layer

Every string in quotation marks below was fetched from the source and then **independently re-fetched and re-checked character-for-character by a separate pass**. 35 quotes passed; 4 failed and were discarded. Anything not in quotation marks is summary, not quotable.

⚠ **Read the closing assessment at the end of this section before citing anything** — one number this guide previously asserted did NOT survive verification.

### Experimental-noise anchor quotes — verified verbatim material

Every string in quotation marks below was independently re-fetched and confirmed character-for-character. Anything not in quotation marks is my own summary and must not be re-quoted. Where a verification pass flagged a location error or an interpretive trap, that is recorded under the quote rather than silently corrected away.

Typographic note that applies throughout: subscripts (IC₅₀, pKᵢ, P_app) are flattened to plain text in these transcriptions, and a few publisher pages use thin or non-breaking spaces around `=` and `×`. Neither changes wording or numbers.

---

### 1. Kramer, Kalliokoski, Gedeck & Vulpetti (2012) — heterogeneous public Ki data

**Citation.** Kramer C, Kalliokoski T, Gedeck P, Vulpetti A. The experimental uncertainty of heterogeneous public Ki data. *J. Med. Chem.* 2012;55(11):5165–5173. DOI: 10.1021/jm300131x

**Access. ABSTRACT ONLY.** The ACS landing page (`https://pubs.acs.org/doi/10.1021/jm300131x`) returns HTTP 403. Abstract fetched from the Europe PMC REST API: `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1021/jm300131x%22&resultType=core&format=json`, and cross-checked against PubMed PMID 22643060. **No full text was retrieved for this paper.**

### Quotes from the abstract (primary)

> "The data deposited in ChEMBL was analyzed for reproducibility, i.e., the experimental uncertainty of independent measurements."

*Location: abstract, fifth sentence.*
Establishes that the quantity being measured is the spread between genuinely independent repeat measurements of the same quantity, which is exactly the quantity a σ-sized Gaussian perturbation of a log-unit label is meant to imitate.

> "The experimental uncertainty is estimated to yield a mean error of 0.44 pK(i) units, a standard deviation of 0.54 pK(i) units, and a median error of 0.34 pK(i) units."

*Location: abstract.*
Gives a directly citable magnitude — σ ≈ 0.54 in log units for heterogeneous public binding-affinity data — so a noise-injection sweep spanning roughly σ = 0.1 to 0.5 can be described as running from optimistic to realistic rather than as an arbitrary range.

*Note on rendering:* "pK(i)" is the MEDLINE ASCII de-subscripting convention. The ACS-published version prints pKᵢ with a subscript, so a transcription from the publisher PDF would read "0.54 pKi units".

> "Careful filtering of the data was required because ChEMBL contains unit-transcription errors, undifferentiated stereoisomers, and repeated citations of single measurements (90% of all pairs)."

*Location: abstract.*
Establishes that a headline σ is only meaningful after aggressive cleaning, so a noise-injection study should be explicit that its σ corresponds to curated data and that raw repository labels carry additional, non-Gaussian corruption.

*Caveat, important:* the abstract does **not** say 90% of pairs were removed. The parenthetical attaches to the third list item only — repeated citations of single measurements. Any gloss along the lines of "90% of pairs were discarded" goes beyond what the retrieved text supports, and the paywalled Methods would be needed to check it.

### Quotes *about* Kramer 2012, from an open-access source (secondary — cite to Kalliokoski)

Both of the following are sentences in Kalliokoski et al. 2013 (PLOS ONE 8(4):e61007, open access, `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061007`), which cites Kramer 2012 as reference [13]. They are **not** text from Kramer 2012 and must be attributed to Kalliokoski if quoted.

> "Note that since the σ here is calculated from pairs of measurements each containing experimental uncertainty and other sources of variability, it has to be divided by √2 in order to obtain the true σ of the individual measurements [13]."

*Location: Kalliokoski et al. 2013, caption of Figure 5.*
This is the single most important methodological point for a noise-injection paper: a σ computed from differences between duplicate measurements is √2 larger than the σ of one measurement, so the value injected into a label must be the per-measurement σ, not the pairwise one.

> "After dividing by √2, the σ for the Gaussian distribution fitted to all ΔpKi values <2.5 then becomes 0.47 (a bit lower than the σ value of 0.54 previously calculated for heterogeneous pKi data from ChEMBL version 12 data without upper threshold for ΔpKi data."

*Location: Kalliokoski et al. 2013, Results/Discussion, the paragraph following Table 3.*
Shows that the 0.54 figure is sensitive to how outlying pairs are handled — a comparable re-analysis lands at 0.47 — which supports quoting σ ≈ 0.5 as an approximate anchor rather than a precise constant.

*Note:* the unbalanced open parenthesis is genuinely in the published text and has been reproduced rather than tidied.

---

### 2. Kalliokoski, Kramer, Vulpetti & Gedeck (2013) — mixed IC50 data

**Citation.** Kalliokoski T, Kramer C, Vulpetti A, Gedeck P. Comparability of mixed IC50 data — a statistical analysis. *PLoS ONE* 2013;8(4):e61007. DOI: 10.1371/journal.pone.0061007

**Access. FULL TEXT (open access).** Fetched as JATS XML from `https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0061007&type=manuscript`, with the rendered article at `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0061007` and PMC4655101/PMC3628986 used for cross-checks.

> "The standard deviation of public ChEMBL IC50 data, as expected, resulted greater than the standard deviation of in-house intra-laboratory/inter-day IC50 data."

*Location: abstract.*
Establishes that label noise is not one number but a function of provenance — public aggregated data is noisier than single-laboratory data — which justifies sweeping σ rather than fixing a single value.

> "From the initially available 616.555 IC50 values with confidence score greater or equal to four 10.895 IC50 values for 3.480 Protein/Ligand systems remained, yielding 20.356 pairs of independent measurements."

*Location: Materials and Methods → "Dataset Preparation", the paragraph immediately preceding Table 1.* (A verification pass corrected this from "Results"; the paper uses "." as a thousands separator, so this is 616,555 → 10,895 values, 3,480 systems, 20,356 pairs.)
Establishes the sample size behind the σ estimate — roughly twenty thousand independent measurement pairs — so the quoted noise magnitude can be presented as well-powered rather than anecdotal.

> "For heterogeneous biochemical pIC50 data, we find a variability with σpIC50 = 0.68, MUEpIC50 = 0.55 and MedUEpIC50 = 0.43."

*Location: Summary and Conclusions; the same figures also appear in the Discussion.*
Gives the log-unit σ for mixed IC50 data, ≈ 0.68, which is the natural upper anchor for a σ sweep on activity labels drawn from heterogeneous public sources.

*Caveat:* the verification pass flagged that the 0.55 MUE in this sentence is an estimate obtained by scaling the pKi metrics upward by 25%; the paper's directly measured MUE from ΔpIC50 data with a 2.5 threshold is 0.54, not 0.55. That alternative sentence was not part of the independently confirmed set, so it is reported here as a caution only and must not be quoted from this document. The σ = 0.68 figure is unaffected.

> "A standard deviation of 0.68 corresponds to a factor of 4.8, meaning that 68.2% of all IC50 measurements agree within a factor of 4.8, even when measured in different laboratories under potentially different assay conditions."

*Location: Discussion, second paragraph.*
Translates a log-unit σ into a concentration-fold interpretation, which lets a noise-injection paper explain to a non-modelling reader what σ = 0.68 physically means without inventing its own conversion.

> "IC50 values measured in the same laboratory usually show a better reproducibility. From our in-house database, we extracted series of reference pIC50 values measured for assay standards. The plots in Figure 9 show the pIC50 values measured for rolipram on PDE4D and cilostamide on PDE3. The standard deviation of the pIC50 values are σ = 0.22 for rolipram/PDE4D and σ = 0.17 for cilostamide/PDE3."

*Location: Discussion, the paragraph introducing Figure 9.*
Provides the low end of the realistic range — σ ≈ 0.17–0.22 for repeat measurements within one laboratory — so a σ sweep can be anchored at both ends with measured values rather than round numbers.

---

### 3. Sato, Yuki, Ito, Tatsuzawa, Yoshida, Yamada, Kanamitsu, Uchida, Hisaka, Yamashita, Yoshimatsu, et al. (2018) — hERG database construction

**Citation.** Sato T, Yuki H, Takaya D, Sasaki S, Tanaka A, Honma T. Construction of an integrated database for hERG blocking small molecules. *PLoS ONE* 2018;13(7):e0199348. DOI: 10.1371/journal.pone.0199348

**Access. FULL TEXT (open access).** Fetched from `https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0199348&type=manuscript` and cross-checked against `https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0199348`.

> "The assay protocols of hERG blocking activities could roughly classified into electrostatic assays such as automated patch clamp assays that measure the change in the voltage between the cell-membrane by the presence of small molecules, and binding assays, such as radio-ligand replacement assays that measure the binding affinity of small molecules by the replacement ratio of radiolabeled inhibitors."

*Location: Materials and methods → "Formatting activity information".* (The missing "be" in "could roughly classified" is in the original.)
Establishes that the label noise in an aggregated hERG dataset is partly structural — two different assay technologies measuring different physical things — rather than purely random measurement scatter.

> "Since the integrated dataset contained heterogeneous data entries, the deviations of the hERG inhibitory activities due to the differences in the assay protocols were analyzed, to assess the influence of the deviations on the classification of the tested compounds into hERG inhibitors and non-inhibitors."

*Location: **Introduction** — a verification pass corrected this from "Materials and methods → Data set". In the Introduction it reads as a stated aim of the study, not as an executed procedure, and should be cited that way.*
Establishes that assay-protocol heterogeneity is treated in the literature as a first-class source of label deviation with downstream consequences for how compounds get classified.

> "To compare the two methods, 209 compounds for which the IC50 values were measured by both methods were investigated."

*Location: Results and discussion → "Comparison between binding assays and electro static assays".* (The two-word "electro static" is the published heading.)
Gives the sample size for the cross-assay agreement analysis, so the agreement statistics below can be cited with their basis stated.

> "The coefficient of determination and the root mean square deviation between the pIC50 values measured by binding assays and electrostatic assays were 0.517 and 0.737, respectively."

*Location: Results and discussion → "Comparison between binding assays and electro static assays".*
Gives a measured ceiling on agreement between two experimental methods — R² = 0.517, RMSD = 0.737 log units — which is the most direct available justification for the claim that a model cannot be expected to beat a certain error floor on aggregated hERG labels.

*Caveat:* R² = 0.517 here is agreement between two assay technologies, **not** a predictive model's R². If this is used as a performance ceiling it must be framed as inter-assay-method agreement.

> "Among the 263 compounds, 144 compounds showed consistent IC50 values with less than one order of magnitude differences between the maximum and minimum results. However, 47 compounds recorded more than 100-fold differences between the maximum and minimum IC50 values."

*Location: Results and discussion → "Deviation of IC50 values and classification of hERG inhibitors and inactive compounds" (the paragraph after Fig 4).*
Establishes that real label error is heavy-tailed — most compounds agree within one log unit but a substantial minority disagree by more than two — which is a concrete, citable limitation of a homoscedastic Gaussian noise model and a motivation for outlier-style or heteroscedastic noise strategies.

---

### 4. Bruneau & McElroy (2006) — logD7.4 modelling

**Citation.** Bruneau P, McElroy NR. logD7.4 modeling using Bayesian Regularized Neural Networks. Assessment and correction of the errors of prediction. *J. Chem. Inf. Model.* 2006;46(3):1379–1387. DOI: 10.1021/ci0504014

**Access. CONTESTED — see the gap note below.** The ACS page returns HTTP 403 and Unpaywall reports `oa_status: closed` with zero repository copies. One verification pass retrieved an author-hosted PDF via the Internet Archive (`https://web.archive.org/web/20240423025829if_/https://www.people.iup.edu/nate/docs/ci0504014.pdf`) and confirmed one string; other passes could not reach any full text at all.

**Only one quote from this paper survived verification:**

> "A measurement is more likely repeated if an apparently abnormal result is obtained. Thus, high ranges in duplicated measurements do not indicate a global variability of the experimental methodology."

*Location: reference/endnote list, note (32), final page — the note cited at the point in the text where the replicate-variability estimate is given.*
Establishes a selection-bias warning that directly constrains any noise-injection justification: replicate spreads in industrial datasets are not a random sample of measurement error, because repeats are preferentially triggered by suspicious results, so replicate-derived σ estimates skew high.

**The frequently cited "0.27 log units across 307 compounds" figure from this paper is not supported by a verified quote in this document.** See the closing section.

---

### 5. Hayeshi et al. (2008) — inter-laboratory Caco-2 comparison

**Citation.** Hayeshi R, Hilgendorf C, Artursson P, Augustijns P, Brodin B, Dehertogh P, Fisher K, Fossati L, Hovenkamp E, Korjamo T, Masungi C, Maubon N, Mols R, Müllertz A, Mönkkönen J, O'Driscoll C, Oppers-Tiemissen HM, Ragnarsson EG, Rooseboom M, Ungell AL. Comparison of drug transporter gene expression and functionality in Caco-2 cells from 10 different laboratories. *Eur. J. Pharm. Sci.* 2008;35(5):383–396. DOI: 10.1016/j.ejps.2008.08.004

**Access. ABSTRACT ONLY.** Full text is paywalled at Elsevier. Abstract fetched from `https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=DOI:%2210.1016/j.ejps.2008.08.004%22&resultType=core&format=json` (PMID 18782614), cross-checked via NCBI efetch.

> "In this study, Caco-2 cells from 10 different laboratories were compared in terms of mRNA expression levels of 72 drug and nutrient transporters, and 17 other target genes, including drug metabolising enzymes, using real-time PCR."

*Location: abstract, second sentence.*
Establishes the design of the canonical inter-laboratory permeability comparison — ten independent laboratories, one nominal cell system — which is the cleanest available demonstration that "the same assay" in different hands is not the same measurement.

> "Absolute expression of genes was variable indicating that small differences in culture conditions have a significant impact on gene expression, although the overall expression patterns were similar."

*Location: abstract, final sentence.*
Establishes a mechanism for between-laboratory label noise — small, unrecorded protocol differences propagate into the measured biology — supporting the framing of label noise as systematic-plus-random rather than purely random.

> "Atenolol permeability was more variable across laboratories than metoprolol permeability."

*Location: abstract.*
Establishes that noise magnitude is compound-dependent within a single assay, which is a citable argument for value-proportional or heteroscedastic noise strategies over a single global σ.

*Caveat:* the abstract attaches no number to this comparison, so it cannot support any quantitative claim about the size of that spread.

> "Talinolol efflux was observed by all the laboratories, whereas only five laboratories observed significant apical uptake of Gly-Sar."

*Location: abstract.*
Establishes that inter-laboratory disagreement can be qualitative — half the laboratories fail to detect an effect the others detect — which is a stronger form of label corruption than additive Gaussian scatter and worth acknowledging as a limitation.

### A quote *about* Hayeshi 2008, from an open-access source (secondary — cite to Kell & Oliver)

> "An interlaboratory comparison (Hayeshi et al., 2008) indicated that while on occasion measurements could vary by more than an order of magnitude, overall the groupings were normally reasonably tight (say within a factor of 2–5)."

*Location: Kell DB, Oliver SG, PeerJ 2015;3:e1405, Introduction. Retrieved from `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC4655101/fullTextXML`. The dash in "2–5" is an en dash in the original.*
Supplies a quantitative characterisation of the Hayeshi spread — typically two- to five-fold, occasionally more than ten-fold — which is the only citable magnitude available for that dataset without the paywalled full text.

*Caveat:* this is Kell & Oliver's own prose describing Hayeshi, not Hayeshi's text. The factor-of-2–5 figure is verified only as a secondary characterisation and remains unchecked against Hayeshi 2008 itself.

---

### 6. Chen, Slättengren, de Lange, Smith & Hammarlund-Udenaes (2017) — reporting Hayeshi data

**Citation.** Chen X, Slättengren T, de Lange ECM, Smith DE, Hammarlund-Udenaes M. Revisiting atenolol as a low passive permeability marker. *Fluids Barriers CNS* 2017;14:30. DOI: 10.1186/s12987-017-0078-x

**Access. FULL TEXT (open access).** The BMC and Springer URLs redirect to an authentication gate; text fetched from PMC/Europe PMC: `https://pmc.ncbi.nlm.nih.gov/articles/PMC5664587/` and `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5664587/fullTextXML` (PMID 29089037).

> "In a collaborative study comparing Caco-2 cells from 10 laboratories, atenolol showed highly variable permeability and its efflux ratios ranged from 0.18 to 3.76, indicating the possibility of an involvement of transporter-mediated transport [38]."

*Location: Discussion, the paragraph on ABC/Pgp transporter evidence, immediately before the sentence beginning "In summary, it is not clear which transporter(s)…". Reference 38 is Hayeshi et al. 2008.*
Puts a number on the Hayeshi inter-laboratory spread for one compound — efflux ratios from 0.18 to 3.76, a range spanning more than a factor of twenty — which is a directly quotable magnitude for permeability label noise.

*Caveat:* this range is Chen et al.'s characterisation of Hayeshi's data, so it should be cited as reported by Chen et al., or checked against Hayeshi directly.

> "The reported Papp values from other in vitro cell models bearing tight junctions for both A–B and B–A directions were in the range of 0.18 × 10−6 − 11 × 10−6 cm/s for Caco-2 cells and 0.13 × 10−6 − 0.8 × 10−6 cm/s for MDCKII (Madin-Darby canine kidney II cells) [37–40]."

*Location: Discussion, immediately preceding the sentence quoted next.*
Establishes that published permeability values for the same marker span roughly two orders of magnitude across cell models and laboratories, which is a citable justification for treating permeability endpoints as high-noise relative to a nominal σ.

*Caveat:* the range is attributed to references [37–40] collectively (Hakkarainen 2010, Hayeshi 2008, Wang 2005, Gartzke 2015). It must **not** be re-attributed to Hayeshi alone.

> "Although showing large inter-laboratory variation, these values and ranges are lower than the out-of-brain permeability estimated in the current study (70.8 × 10−6 cm/s), also suggesting the involvement of transporters in removing atenolol from the brain."

*Location: Discussion, the sentence following the one above.*
Establishes that the authors themselves describe the collected literature values as showing large inter-laboratory variation, which is the phrase available for citation when characterising permeability data quality.

> "38. Hayeshi R, Hilgendorf C, Artursson P, Augustijns P, Brodin B, Dehertogh P, Fisher K, Fossati L, Hovenkamp E, Korjamo T, Masungi C, Maubon N, Mols R, Mullertz A, Monkkonen J, O'Driscoll C, Oppers-Tiemissen HM, Ragnarsson EG, Rooseboom M, Ungell AL. Comparison of drug transporter gene expression and functionality in Caco-2 cells from 10 different laboratories. Eur J Pharm Sci. 2008;35:383–396."

*Location: reference list, entry 38.*
Confirms the exact bibliographic form of the Hayeshi citation as printed in a peer-reviewed source, useful because the Hayeshi full text itself could not be retrieved.

---

### 7. Niu et al. (2024) — PharmaBench

**Citation.** Niu Z, Xiao X, Wu W, Cai Q, Jiang Y, Jin W, Wang M, Yang G, Kong L, Jin X, Yang G, Chen H. PharmaBench: Enhancing ADMET benchmarks with large language models. *Scientific Data* 2024;11:985. DOI: 10.1038/s41597-024-03793-0

**Access. FULL TEXT (open access).** nature.com redirects through an identity provider; text fetched from `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11387650/fullTextXML` (PMC11387650, PMID 39256394) and, for Table 5, from `https://www.nature.com/articles/s41597-024-03793-0/tables/5` by direct HTML parse.

> "A raw dataset often contains multiple records for the same compound due to different sources and varying experimental conditions. Repeated testing compares the maximum and minimum values for the same compound under the same condition to validate the data quality."

*Location: Technical Validation → "Repeated test for data quality assessment".*
Establishes the standard modern procedure for estimating label noise from a public dataset — comparing repeat records of the same compound — which is the empirical basis a noise-injection study should point to when choosing σ.

> "The maximum and minimum experimental results for each group are selected as the worst-case scenario."

*Location: Technical Validation → "Repeated test for data quality assessment", immediately preceding Table 5.*
Establishes that the reported agreement statistics are deliberately worst-case, so any σ derived from them is an upper bound rather than a typical value.

> "If the experimental results are consistent for different data sources, the repeated test plot will exhibit higher correlation and a lower mean absolute error (MAE) for regression tests, and the confusion matrix will show higher accuracy (ACC), precision, and recall for classification tests."

*Location: Technical Validation → "Repeated test for data quality assessment".*
Establishes the interpretation rule linking curation quality to measurable agreement metrics, letting a noise-injection paper connect its σ to a documented data-quality diagnostic rather than an assumption.

> "LogD 0.774 1.196 0.7 0.881 0.881 0.48"

*Location: Table 5, first data row. Caption: "Comparison of Metrics Between the Regression Datasets Before and After the Data Processing Workflow." Column structure: ADMET Property Name | Before Data Processing Workflow (R, RMSE, MAE) | After Data Processing Workflow (R, RMSE, MAE). So LogD before: R = 0.774, RMSE = 1.196, MAE = 0.7; after: R = 0.881, RMSE = 0.881, MAE = 0.48. The repeated 0.881 is genuine, not a transcription error.*
Gives a contemporary, worst-case repeat-measurement error for LogD — MAE 0.7 log units before curation, 0.48 after — which brackets the σ range a LogD noise-injection experiment should cover.

> "The results of repeated tests for certain experiments, such as the LogD experiment, have significantly improved data quality after the data processing workflow, reaching a level comparable to that of traditional wet lab experiments."

*Location: Technical Validation → "Repeated test for data quality assessment", the paragraph following Tables 5 and 6.*
Establishes that even a well-curated modern LogD benchmark is characterised by its own authors as merely reaching wet-lab-comparable reproducibility, i.e. the label noise floor does not vanish with curation.

*Caveat:* this sentence cites no external wet-lab benchmark; the supporting comparison is the dataset against itself, before versus after processing. The next sentence in the paper explicitly excludes CYP and clearance endpoints from the claim.

---

### 8. Alvarez Baron et al. (2025) — multi-laboratory manual patch clamp hERG

**Citation.** Alvarez Baron C, Zhao J, Yu H, et al. Multi-laboratory comparisons of manual patch clamp hERG data generated using standardized protocols and following ICH S7B Q&A 2.1 best practices. *Sci. Rep.* 2025;15:29995. DOI: 10.1038/s41598-025-15761-8

**Access. FULL TEXT (open access).** nature.com redirects to an identity provider; text fetched from `https://www.ebi.ac.uk/europepmc/webservices/rest/PMC12357877/fullTextXML` (PMC12357877, PMID 40819150), with the nature.com HTML retrieved by direct curl for the Fig. 10 sentence.

> "Descriptive statistics and meta-analysis were applied to the dataset to estimate what the distribution in hERG block potencies would be if a laboratory were to test the same drug repeatedly. This measure, or hERG data variability, was ~ 5X."

*Location: abstract. (The space before "5X" is a thin space in the published version.)*
Provides a current, headline-level statement of hERG assay repeatability — about five-fold — that a noise-injection study can quote as the physical meaning of its chosen σ on hERG labels.

> "A goal of this study is to assess variability of hERG block potency, which is defined as the distribution of potencies for the same drug when measured repeatedly by the same laboratory."

*Location: Results, opening sentence of the section "Variability in hERG block potency".*
Supplies an explicit definition of the noise quantity — the distribution obtained on repeat measurement of the same drug — which is precisely what an injected Gaussian is standing in for.

> "These approaches used strategies to estimate and then remove drug- (i.e., potency) and laboratory-specific impacts (i.e., systematic differences to the group average) to reveal unexplained or residual variability in data not tied to drug or laboratory."

*Location: Results, section "Variability in hERG block potency", third sentence of the opening paragraph (a verification pass corrected this from "second sentence").*
Establishes that the reported variability is a residual after removing compound and laboratory effects, i.e. the irreducible component, which is the right quantity to match when calibrating injected noise.

> "The estimated overall variability (τ) not explained by drug and laboratory and expressed as SD was 0.18, and as 95% CI 0.69, corresponding to an IC50 ratio of 4.9X (95% CI: 4.0 to 6.2)."

*Location: Results, section "Variability in hERG block potency", the paragraph describing Fig. 9 (the mixed-effects meta-analysis).*
Gives the residual hERG label variability in both log-unit and fold terms, allowing a σ chosen for a noise sweep to be justified against a published mixed-effects estimate.

*Caveat, arithmetic:* 0.18 alone does not generate the 4.9× figure; 0.69 is the half-width of the 95% interval and 10^0.69 ≈ 4.9. If this sentence is used to justify a noise magnitude, cite the 0.69 / 4.9× pair, or the numbers will not reproduce. Also note the paper reports τ as an SD in pIC50 units — pIC50 is already a negative log10, so "log10 pIC50 units" is a unit-of-a-unit error.

> "The modelling results shown in Fig. 10 also suggest that variability in hERG block potency is laboratory-specific, ranging from 3.4X to 9.6X for different laboratories in this study."

*Location: Discussion, subsection "Limitations and lessons learned" — a verification pass corrected this from the subsection "Variability in hERG block potency". Cite the corrected subsection.*
Establishes that even under standardised protocols the noise magnitude differs nearly three-fold between laboratories, which supports reporting results across a σ range rather than at a single calibrated σ.

---

### Closing assessment: what can and cannot be quoted

### Endpoints with direct quotable support

**Binding affinity (Ki), public aggregated data.** Supported, abstract only. Kramer et al. 2012 can be quoted for σ = 0.54 log units, mean error 0.44, median error 0.34. The full text was never retrieved, so anything beyond those three numbers and the two other abstract sentences above is not quotable from this paper.

**IC50, mixed public data.** Fully supported from open-access full text. Kalliokoski et al. 2013 gives σ = 0.68, the 4.8-fold interpretation, the within-laboratory floor of σ = 0.17–0.22, and the sample size. This is the strongest single source in the set.

**The √2 correction for pairwise-derived σ.** Supported, but only via Kalliokoski's Figure 5 caption. If your text needs this methodological point, cite Kalliokoski 2013, not Kramer 2012.

**hERG IC50, database heterogeneity.** Fully supported from Sato et al. 2018: the two-assay-type structure, the 209-compound comparison, R² = 0.517 and RMSD = 0.737 between assay methods, and the heavy-tailed 263/144/47 breakdown.

**hERG IC50, standardised repeat measurement.** Fully supported from Alvarez Baron et al. 2025: ~5× overall, τ figures with confidence interval, and the 3.4×–9.6× per-laboratory range.

**LogD.** Supported for the *modern benchmark* case only, from PharmaBench (Niu et al. 2024): repeat-test MAE 0.7 before curation and 0.48 after, plus the worst-case-selection caveat that frames those as upper bounds.

**Caco-2 permeability, inter-laboratory.** Partially supported. The Hayeshi 2008 abstract yields four usable qualitative sentences about design, mechanism, compound-dependence, and qualitative disagreement. Every *number* characterising the size of the Hayeshi spread comes from a secondary source: the 0.18–3.76 efflux-ratio range from Chen et al. 2017, and the "factor of 2–5" characterisation from Kell & Oliver 2015. Both are verified verbatim in those papers, but neither is Hayeshi's own wording.

### Gaps — endpoints and numbers that cannot currently be quoted

**LogD from Bruneau & McElroy 2006: the "0.27 log units across 307 compounds" figure has no verified quote.** This is the significant gap. Three separate claimed sentences from this paper's Methodology and Results — the one containing 0.27 and 307, the preceding clause about the 2-log-unit exclusion threshold, and two sentences from the "Database" subsection including a 0.3-log-unit discard threshold — all failed verification with the verdict "could not access". The ACS full text returns HTTP 403, Unpaywall reports the article as closed with zero repository copies, and only the abstract could be read; the abstract contains none of those numbers. One verification pass did reach an author-hosted PDF through the Internet Archive and confirmed endnote 32 from it, so the article text is evidently reachable by that route, but the numeric sentences were never independently confirmed and therefore must not be presented as quotes.

Honest options for this number, in order of preference:

1. **Obtain the PDF manually** through Oxford institutional access at pubs.acs.org, or from the archived author copy, and check pages 1380 and 1382–1383 by eye. This is the only route that turns 0.27 into a quotable figure, and it is cheap.
2. **Cite without quoting.** Write "Bruneau and McElroy report a mean per-compound replicate standard deviation of approximately 0.27 log units for in-house logD7.4 data" with a plain citation and no quotation marks. This is defensible only if you have actually seen the number somewhere you trust; do not do this on the strength of this document alone, because this document has not confirmed it.
3. **Use a different source.** PharmaBench Table 5 (MAE 0.48–0.7 for LogD) is fully verified, open access, and covers the same endpoint. If the argument only needs "LogD labels carry roughly half a log unit of error", PharmaBench supports it outright and Bruneau is not required.

Also note two internal inconsistencies flagged during verification, which should be resolved before the guide is used: the repository currently cites Bruneau for 0.27 in `REVISION_GUIDE.md` line 357 and `DISCUSSION_TRACKER.md` line 116, while a separate claimed sentence from the same paper gives 0.3 as a *discard threshold* — different quantities that should not be conflated. Neither is verified.

**Hayeshi's own numbers.** No quantitative statement can currently be quoted from Hayeshi et al. 2008 itself, because the full text is paywalled at Elsevier and only the abstract was retrieved. If a number attributed directly to Hayeshi is load-bearing, either obtain the PDF or attribute the number explicitly to Chen et al. 2017 or Kell & Oliver 2015, marked as reported-by.

**Kramer's filtering statistics.** The "90% of all pairs" figure is quotable, but only as attached to the category "repeated citations of single measurements". Any claim about what fraction of the dataset was ultimately removed is not supported by the retrieved abstract and would need the paywalled Methods.

---

## 5. What has been tested

### 5.1 Dose matching delivers ✅

All six noise types implemented in Python and run against the real QM9 labels (133,885 molecules,
`data/QM9/raw/gdb9.sdf.csv`). Script: `scripts/test_noise_arms.py`.

Every noise type landed within **±1.5%** of the requested dose at k = 0.2 and k = 0.5.

### 5.1b ✅ A Rust reference implementation exists, builds, and agrees with Python

`rust/reference/noise_arms.rs` — self-contained (no RDKit, no memmap, no pipeline), builds
clean against `rand 0.9` / `rand_distr 0.5`. It exists to prove the design before
`rust/src/main.rs` is touched, and to be the fixed point the Python injector is checked against.

**Updated 2026-08-26 (chat B).** It had **no `Cargo.toml`**, so nobody but its author could build
it and §6.3 item 6 was unrunnable; there is one now, and the directory is tracked. It also
implemented only the five dose-matched types — **censoring and grouped-shifted have been added**,
so the gate covers every condition rather than most of them. `--json`, `--groups <file>` and
`--seeds N` were added so both implementations run on the same labels and the same group
assignment and are compared automatically:

```
cd rust/reference && cargo build --release
python scripts/crosscheck_injectors.py          # exits non-zero on any failure
```

Run against all 133,885 real QM9 labels, realised versus target dose:

| Noise type | Unit dose G | Error vs target |
|---|---|---|
| Gaussian | 1.0000 | −0.05% |
| Student-t ν=10 | 1.1180 | +0.14% |
| Student-t ν=5 | 1.2910 | −0.03% |
| Student-t ν=3 | 1.7321 | −2.58% (see below) |
| Laplace | 1.4142 | −0.38% |
| Grouped λ=3 f=0.2 | 1.6097 | +0.09% |
| Outlier p=0.01 | 1.0392 | −0.09% |
| Outlier p=0.05 | 1.1855 | +0.28% |
| Outlier p=0.10 | 1.3435 | +0.49% |

Identical at k = 0.25, 0.5 and 1.0, as the algebra requires (the scale map is linear in the
knob). The two independent implementations also agree on the *shape* diagnostics — fraction
of labels off by more than three times the dose: Gaussian 0.27% both; ν=10 0.73% both;
ν=5 1.18% vs 1.16%; ν=3 1.42% vs 1.37%; Grouped 2.19% vs 2.10%.

**The Student-t ν=3 deviation is sampling variability, not a bug.** Across 40 seeds the
error is unbiased (mean −0.30%) but its spread explodes as the tail heavies:

| ν | mean error | SD of error | range |
|---|---|---|---|
| 30 | −0.00% | 0.19% | −0.58% to +0.32% |
| 10 | −0.04% | 0.24% | −0.59% to +0.49% |
| 5 | −0.14% | 0.38% | −1.01% to +0.60% |
| 4 | −0.16% | 0.52% | −1.13% to +0.80% |
| **3** | **−0.30%** | **2.24%** | **−3.51% to +6.84%** |

**Consequence:** fix the *population* dose and report that. Never report a per-run empirical
dose or empirical kurtosis for ν ≤ 4 — the sample statistic is unstable by construction
because the fourth moment is infinite.

### 5.1c ✅ THE PIPELINE ITSELF NOW MATCHES THE REFERENCE — chat A, 2026-08-26

`rust/src/main.rs` is no longer a plan. The redesign is implemented there, the old noise
types are deleted, and the injector reproduces the reference table above **to the digit** on
the same 133,885 QM9 labels:

```
./rust/target/release/rust_processor --self-test <labels.csv> [--scaffold-file <groups.json>]
```

| Noise type | Unit dose G | Error vs target | Reference (§5.1b) |
|---|---|---|---|
| Gaussian | 1.0000 | −0.05% | −0.05% |
| Student-t ν=10 | 1.1180 | +0.14% | +0.14% |
| Student-t ν=5 | 1.2910 | −0.03% | −0.03% |
| Student-t ν=3 | 1.7321 | −2.58% | −2.58% |
| Laplace | 1.4142 | −0.38% | −0.38% |
| Outlier p=0.01 | 1.0392 | −0.09% | −0.09% |
| Outlier p=0.05 | 1.1855 | +0.28% | +0.28% |
| Outlier p=0.10 | 1.3435 | +0.49% | +0.49% |

The shape diagnostics reproduce too (fraction beyond three times the dose: 0.27%, 0.73%,
1.18%, 1.42%, 1.41%).

**On 4,000 real QM9 molecules with real Murcko scaffold groups**, acyclic molecules split into
singletons per §2a rule 2 — 1,703 groups, largest holding 7.0% of the molecules — the mean
delivered dose over 20 seeds sits within **0.74%** of target for every condition, and the
**spread between the conditions is 1.27%**. That is the whole point of the redesign, measured:

| Condition | mean over 20 seeds | per-run SD | affected molecules |
|---|---|---|---|
| Gaussian | +0.10% | 1.2% | 100% |
| Student-t ν=10 | +0.48% | 1.2% | 100% |
| Student-t ν=5 | +0.67% | 1.6% | 100% |
| Student-t ν=3 | +0.58% | 6.0% | 100% |
| Laplace | −0.52% | 1.6% | 100% |
| Outlier p=0.01 / 0.05 / 0.10 | +0.09 / +0.11 / +0.51% | 1.3–1.4% | 0.9 / 4.3 / 9.3% |
| Grouped — wider (f=0.2) | −0.01% | 1.2% | **20.9%** |
| Grouped — shifted (ρ=0.62) | +0.74% | **6.9%** | 100% |

Two things to read off it. The grouped-wider condition lands on 20.9% of molecules against a
request of 20% — under the old group-counting rule the same request gave 22.6%, and §2a's
measurements put that rule's range at 6.7–55.1%. And grouped-shifted's per-run spread is 6.9%,
matching §2a rule 3's "roughly ±5%" — which is why the gate is on the mean and on the
construction, not on one realisation.

#### What `affected_molecule_fraction` means → §5.1d finding 2

`rust/reference/noise_arms.rs` and `noiseInject` both carry a code comment pointing here for
this. It is one section further down: the convention is **1.0 wherever nothing selects**, and
the write-up is finding 2 of §5.1d.

#### The tolerance is derived, not chosen

The half a percent quoted above is not a universal constant — it is what 133,885 Gaussian draws
happen to give. For a second moment averaged over `n_eff` independent contributions with kurtosis
`k`, the relative standard error is `sqrt((k − 1) / (4·n_eff))`, and the square root halves it. At
n = 133,885 that is **0.19%**, which is exactly the standard deviation §5.1b measured across 40
seeds at ν = 30. The injector therefore computes its own band per run — kurtosis from the sample,
`n_eff` from the scale map (`(Σs²)² / Σs⁴`) and, for the shifted condition, from the group count
(`1 / (ρ²/n_groups + (1−ρ)²/n)`).

This matters because a flat band is wrong in both directions: it fails correct code on a small
dataset, and it would pass a broken solver on a large one.

---

### 5.1d ✅ THE THIRD LEG — the pipeline against the reference, chat A 2026-08-26

The scheme exists in three places and only two of them were tied together.
`scripts/crosscheck_injectors.py` (chat B) ties the **reference** to the **Python
injector**. Nothing tied either of them to the thing that actually noises QM9,
`rust/src/main.rs`. The reference is a clean-room prover with no memmap, no RDKit and no
pipeline around it; the pipeline is the code that touches the data. They can drift in
exactly the way the two injectors already drifted once, and the existing gate would pass
throughout.

`scripts/crosscheck_pipeline_reference.py` closes it. The chain is now
**Python injector ↔ reference ↔ pipeline.**

```
python scripts/crosscheck_pipeline_reference.py --labels <smiles,y> --groups <groups.json>
```

Result on 4,000 real QM9 molecules with real Murcko groups, 20 seeds, k = 0.5 — and at
k = 0.25 and k = 1.0:

| Compared | Outcome |
|---|---|
| Unit dose `G`, where it is fixed by algebra | **identical** — gaussian 1.0000, ν=10 1.1180, ν=5 1.2910, ν=3 1.7321, laplace 1.4142, grouped-shifted 1.0000, grouped-wider 1.6125 |
| Censoring limit and delivered dose — deterministic given the labels | **identical to six decimal places** at every level. This is the sharpest check in the file: it is what catches the two putting the assay limit in different places |
| Mean delivered dose over 20 seeds | within **0.82%** for every condition except ν=3, which is 2.53% and is discussed below |
| Shape diagnostics — median absolute error, worst-hit 5%'s share of the noise energy | within 10% relative |

Verified to fail: reverting the censoring limit to nearest-rank makes it report the limit
mismatch and the dose gap on three levels.

#### Three things the check found

**1. The censoring roster was missing its zero control.** §6.4's grid is 0, 10, 20, 25, 30,
40, 50%. The pipeline's roster started at 10%, so the negative control for the one condition
that is not zero-mean was silently absent. **Fixed.**

**2. ✅ SETTLED 2026-08-26 (chat B) — `affected_molecule_fraction` is 1.0 where nothing selects.**
Chat A raised this and left it for whoever merged the three implementations. For the conditions
with no selection rule (gaussian, Student-t, Laplace, grouped-shifted) the pipeline recorded
**1.0**, since every molecule is affected; the reference recorded **0.0**, meaning "no targeting
applies". Both read the name defensibly and they cannot share a column.

**1.0 everywhere.** It is the truthful reading — every molecule does receive noise — and the
alternative left the Python injector disagreeing with *itself*: grouped-shifted, which also
perturbs every molecule, already recorded 1.0 while the shape-only conditions recorded 0.0.
Changed in `rust/reference/noise_arms.rs` and `noiseInject`; the pipeline already agreed, so all
three now match and the cross-check compares the column on every condition rather than only where
something selects.

One consequence to carry into the figure script: **failure mode 6 in `RERUN_PLAN.md` §0.6 must be
scoped to conditions that have a cut-point.** Its guard asserts the affected fraction "is neither
near zero nor near one", which catches a degenerate *threshold* rule — but a uniform condition
legitimately reads 1.0 and would trip it.

**3. Without a group file the two behave differently on purpose.** The reference falls back
to 2,000 synthetic clusters; the pipeline refuses, because grouped noise over invented groups
is uniform noise wearing a grouped name. So the grouped conditions cannot be compared without
a real assignment — and the check now reports **PARTIAL and exits non-zero** rather than
saying 15 of 17 conditions agree and calling it a pass.

---

### 5.2 The noise types are genuinely distinguishable ✅

At k = 0.5, with **identical total noise** in every row:

| Strategy | Labels off by >3× the dose | Share of noise on worst-hit 5% | Median error |
|---|---|---|---|
| Gaussian | 0.27% | 28% | 0.434 |
| Student-t ν=10 | 0.71% | 33% | 0.405 |
| Student-t ν=5 | 1.16% | 41% | 0.364 |
| Student-t ν=3 | 1.37% | 57% | 0.286 |
| Grouped λ=3 f=0.2 | **2.10%** | 50% | 0.317 |
| Outlier p=0.05 | 1.25% | 42% | 0.382 |
| Outlier p=0.10 | 1.78% | 48% | 0.351 |

An **eight-fold spread** in how many labels end up badly wrong, at identical total noise.

Note the trade-off, which is the practical story: as noise concentrates, the *median* label
gets **more** accurate (0.434 → 0.286) while a minority gets far worse. Whether a dataset
where most labels are excellent and a few are junk is better or worse than one where
everything is slightly off has not been answered in the literature.

### 5.3 ⚠️ But the noise types barely differ in model accuracy

Pilot of the real experiment: 10,000 QM9 molecules, RDKit descriptors, real Murcko scaffold
split, noise on training labels only, scored on clean test labels. Seven noise types × three doses
× three models × three repeats. `scripts/pilot_noise_arms.py`, results
`results/pilot_noise_arms.csv`. Runtime 3.2 hours.

The experiment is mechanically sound: realised dose matched nominal in every cell, and
degradation magnitudes match the published picture (at k = 1.0 models retain 88–97% of clean
R², far above the 50% floor full overfitting would give — exactly Kolmar & Grulke's point
that unbiased label noise largely averages out).

Concentrated noise types paired against Gaussian at the same dose, model and repeat:

| Model | Mean advantage of concentrated noise | Paired t-test |
|---|---|---|
| **Random forest** | **−0.0006 R²** | p = 0.71 — **nothing** |
| LightGBM | +0.0033 R² | p = 0.067 |
| Ridge | +0.0073 R² | p < 0.001 |

At a realistic noise level all seven noise types sat within **0.006 R²** of each other, while the
dose itself cost 0.021.

**Honest reading:** noise shape, at matched dose, does not meaningfully change how much
accuracy a tree ensemble loses. That extends Heid's homoscedastic result to the concentrated
case — a real finding, but a negative one.

Limits: one dataset and an easy one (clean R² 0.93); three models, one representation, three
repeats; effect sizes near the replicate noise floor. And it says nothing about **uncertainty
estimates**, which is where strategy 4 was expected to earn its place — a model may lose the
same accuracy while being much better or worse at knowing which labels were corrupted.

### 5.3b 🔴 THE RESULT THAT CHANGES THE DESIGN — censoring is 12× bigger than everything else

Second pilot, N = 4,000 QM9 molecules, all six noise types including censoring.
`scripts/pilot_noise_arms.py` (N_MOL = 4000), 696 s.

**Validity check passed first.** Predicted before running: a smaller training set has less
data to average random noise away, so losses should be *larger* than the 10,000-molecule
run. Confirmed for all three models — retention at the heaviest noise fell from 0.881 to
0.771 (LightGBM), 0.922 to 0.897 (random forest), 0.974 to 0.933 (ridge). The pipeline
behaves as it should.

**The finding.** Censoring compared against random noise *interpolated to the identical
effective dose*, so this is a like-for-like comparison:

| Censored fraction | Effective dose | Extra damage beyond the same amount of random noise |
|---|---|---|
| 10% | 0.20 × label SD | **+0.002 R²** — nothing |
| 25% | 0.36 × label SD | **−0.054 R²** |
| 40% | 0.60 × label SD | **−0.266 R²** |

For scale: **the largest difference among all six zero-mean noise types was 0.022 R².** Censoring
40% of labels costs **0.266 R² beyond** what the same amount of random noise costs —
**twelve times larger than any noise-shape effect in the entire study.**

Two further things the numbers show:

1. **It is a threshold effect, not a gradient.** Nothing at 10%, mild at 25%, catastrophic
   at 40%. R² collapses from ~0.88 to ~0.59 between those last two points.
2. **It is the only noise type with a systematic direction.** Mean label shift −0.047, −0.162 and
   −0.397 eV at 10/25/40%. Every other noise type shifts labels by zero on average. That is
   precisely why models cope with it so badly — they can average away scatter, but not bias.

**Why this matters for the paper:** Svensson et al. report **25–63% of labels censored in eight
of fifteen real industrial assays** (§3.5). Real datasets therefore sit squarely in the range
where this goes from mild to catastrophic — while the differences between the zero-mean
noise shapes never exceed 0.02 R².

⚠️ **Caveats, stated plainly.** QM9 has no real assay limit — a density-functional
calculation always returns a number — so censoring here is an imposed mechanism, not a
simulated measurement. It is realistic for the three experimental endpoints, not for QM9.
And censoring 40% of labels removes the top of the label range outright, so part of the
damage is loss of range rather than corruption *per se*. That is, however, exactly what
censoring does to a real dataset.

**Recommendation for the design (yours to accept or reject):** censoring stops being an
optional sixth noise type and becomes the main event. The five zero-mean noise types are worth running as
the controlled comparison that establishes shape does not matter — a clean negative result
that is publishable and that nobody has established for concentrated noise — but the
positive finding is here.

### 5.5 ✅ RANGE-FINDING RUN — sets the grids, and produces the study's best practical result

QM9, N = 4,000, real scaffold split. Seven noise levels × four noise types × three repeats,
plus censoring swept at nine levels. `scratchpad/ranges.py`, 1,263 s.
Clean baselines: LightGBM 0.921, random forest 0.920, ridge 0.897.

#### Where the curve actually moves — this fixes the QM9 grid

| Noise level (fraction of label spread) | LightGBM | Random forest | Ridge | Spread across noise types |
|---|---|---|---|---|
| 0.1 | 0.919 | 0.920 | 0.898 | 0.002 |
| 0.2 | 0.913 | 0.917 | 0.897 | 0.004 |
| 0.3 | 0.901 | 0.911 | 0.894 | 0.008 |
| 0.5 | 0.869 | 0.891 | 0.885 | 0.016 |
| 0.75 | 0.811 | 0.863 | 0.865 | 0.028 |
| 1.0 | 0.727 | 0.825 | 0.836 | 0.042 |
| 1.5 | 0.482 | 0.731 | 0.752 | 0.101 |

**Below 0.2 nothing happens** — the models are indistinguishable from clean. **The noise
types only begin to separate above 1.0**, and even at 1.5 the spread (0.101) is smaller than
the gap between models.

**QM9 grid, set from this: 0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5.** Dropping 0.1 as
uninformative; keeping 1.5 because it is the only level where noise shape separates at all.

#### The censoring curve — smooth acceleration, no sharp knee

Extra damage beyond what the *same delivered dose* of random noise costs:

| Labels censored | Delivered dose | Extra damage |
|---|---|---|
| 5% | 0.17 | +0.003 — nothing |
| 10% | 0.20 | −0.001 — nothing |
| 15% | 0.24 | −0.009 |
| 20% | 0.29 | −0.027 |
| 25% | 0.36 | **−0.057** |
| 30% | 0.44 | **−0.117** |
| 35% | 0.53 | **−0.200** |
| 40% | 0.60 | **−0.273** |
| 50% | 0.74 | **−0.481** |

There is no knee — it is a smooth acceleration that becomes serious around 20% and severe
past 30%. **Svensson et al. report 25–63% of labels censored in eight of fifteen real industrial
assays** (§3.5), which is exactly the range running from −0.06 to −0.48.

**Censoring grid, set from this: 0, 10, 20, 25, 30, 40, 50%.** Keeping 10% as the null
anchor, concentrating resolution where the damage accelerates.

#### 🔴 The finding: model choice stops helping under censoring

Spread between the best and worst of the three models:

| Condition | Best-minus-worst R² |
|---|---|
| Random noise, level 0.1 | 0.022 |
| Random noise, level 0.5 | 0.022 |
| Random noise, level 0.75 | 0.054 |
| Random noise, level 1.0 | **0.109** |
| Random noise, level 1.5 | **0.270** |
| | |
| Censoring 5% | 0.022 |
| Censoring 25% | 0.021 |
| Censoring 35% | **0.0007** |
| Censoring 40% | 0.006 |
| Censoring 50% | 0.022 |

**Under random label noise, picking a robust model buys you steadily more as the noise
rises — up to 0.27 R². Under censoring it buys you essentially nothing at any level.** At
35% censored the three models land within 0.0007 R² of each other: 0.677, 0.678, 0.677.

The mechanism is straightforward. Random errors are unbiased, so a model that resists
overfitting can average them away — and models differ a lot in how well they do that.
Censoring is a systematic bias in the labels themselves. No amount of regularisation
recovers information that was never recorded.

**This is a standalone claim, and the most useful one the design produces:**

> How much your choice of model matters depends on what kind of error you have. For random
> measurement error, model choice is worth up to 0.27 in R² and the gap widens as the error
> grows. For censored data it is worth nothing at any level — every model fails equally, and
> the only remedy is to handle the censoring in the data or the loss function.

It is directly actionable, it is the opposite of what a practitioner would assume, and no
noise-robustness benchmark has established it.

⚠️ Same caveats as §5.3b: QM9 has no real assay limit, so censoring here is imposed rather
than observed, and clipping the top of the range removes label range as well as corrupting
values. Both need confirming on the experimental datasets, where censoring is real.

### 5.6 ✅ The two new conditions, verified on real labels — chat G, 2026-08-26

`scripts/setting_selection_test.py --self-check`, on the training half of a 4,000-molecule QM9
subsample: 3,200 labels, 1,416 scaffold groups after rule 2, label spread 1.299 eV.

**Every condition delivers what it was asked for.** Mean realised dose over 24 draws, as a fraction
of target, at each of the three levels tested (identical across levels, as the algebra requires —
the scale map is linear in the knob):

| Condition | Realised ÷ target − 1 | Per-run SD |
|---|---|---|
| Gaussian | −0.00% | 0.91% |
| Student-t ν = 10 | +0.36% | 1.20% |
| Student-t ν = 5 | −0.38% | 1.98% |
| Student-t ν = 3 | +1.33% | **16.18%** |
| Laplace | −0.82% | 2.00% |
| Outlier p = 1% | +0.48% | 1.57% |
| Outlier p = 5% | +0.04% | 2.03% |
| Outlier p = 10% | +0.21% | 1.86% |
| Grouped — wider | +0.36% | 1.74% |
| Grouped — shifted | +0.46% | 4.28% |
| Skewed draw *(proposed)* | −0.14% | 2.53% |

**Grouped — shifted does what it was added to do.** Its per-run mean label shift ranges from −0.117
to +0.085 eV against a target dose of 0.650, while Gaussian's ranges only −0.024 to +0.029 — a
one-directional push about four times larger, at matched amount. Over 24 seeds the shift averages
−0.003 eV, so the condition is zero-mean in the population and directional in every run, which is
exactly the specification. **Do not "fix" the per-run shift.**

**One number to expect and not be alarmed by.** The empirical between-group share of variance comes
out **0.766**, not the nominal ρ = 0.62. That is rule 2 working: once acyclic molecules are
singletons, a singleton group's "group mean" carries its own within-molecule term, which inflates the
measured between-group share. The parameter is still the sourced 0.62.

**Grouped — wider hits the fraction it is given.** 0.200 of *molecules* affected against a requested
0.2, using the molecule-fraction selection rule rather than counting groups.

### 5.7 ❌ The skewed draw — tested, and rejected 2026-08-27

§13.3 of `RERUN_PLAN.md` rejected a skewed *draw* for the three experimental datasets, on the sound
argument that log-scale potency error is symmetric. The condition below exists to measure what that
rejection costs on QM9, which is computed rather than measured and so has no assay to justify any
shape. **It did not separate, and it is not being built** — nothing at the reporting level, and
−0.045 R² for the random forest alone at the top of the grid. It stays in
`scripts/setting_selection_test.py` so the rejection remains reproducible, is recorded in
`noise_conditions.json` under `not_run` as never implemented, and a test on each side fails if it
appears in a grid.

> Centred Gamma: `ε = ((g − a)/√a)·τ` with `g ~ Gamma(a, 1)`. Mean zero, variance `τ²`, skewness
> `2/√a`. At `a = 1` the sample skewness on real labels came out **+2.26** against the target +2.00,
> with a mean shift of −0.017 eV on a dose of 0.650.

One parameter, exact dose match in the population, and it nests nothing — which is the point: it is
the only condition in the set that is asymmetric *by draw* rather than by mechanism.

### 5.8 ✅ WHICH SETTINGS EARN THEIR PLACE — chat G, 2026-08-26

`scripts/setting_selection_test.py`. Real QM9, 4,000 molecules per replicate drawn fresh, PDV
descriptors, real Murcko scaffold split, noise on training labels only, scored on clean test labels.
**Twelve replicates**, levels 0.5 and 1.5, three models on the pipeline's own defaults. Every
condition delivered the same amount of noise, within three standard errors of target.

**Shape does not earn separate settings. Direction does.**

At the reporting level, across every non-Gaussian condition except grouped-shifted, and all three
models: the largest mean difference against Gaussian is **0.0058 R²**, the largest ratio to the
replicate-to-replicate wobble is **0.29**, and the smallest paired *p* is 0.089 — against a
detectable floor of 0.0064–0.0208. The ν = 10 → 5 → 3 ladder and the p = 1% → 5% → 10% ladder are
both flat, every step under 0.006 R².

**This confirms §5.3 with a much better test.** That pilot found the same thing on three replicates
with the subsample, split and model seeds all held fixed — so its comparison ran against a wobble
that was far too small. Twelve replicates with everything redrawn per replicate give the same answer,
and now the answer is worth something.

**Grouped-shifted separates, and by a lot.** Against Gaussian at level 1.5: −0.127, −0.101 and
−0.330 R² for LightGBM, the random forest and ridge — 2.2×, 3.0× and 4.6× the wobble, every *p* ≤
0.002. Against **grouped-wider**, which differs only in whether the group's error is centred:
−0.142, −0.096 and −0.314, at 2.5× to 7.4× the wobble. Same amount of noise, same groups, same
targeting — the only difference is direction, and direction is worth fifty times what shape is.

**That is the censoring result, reproduced by a second mechanism.** §5.3b and §5.5 show one-directional
error doing twelve times more damage than any shape effect, at the level of the whole dataset.
Grouped-shifted shows it at the level of a chemical family. Two independent demonstrations of one
effect, which is what §13.3 of `RERUN_PLAN.md` argued the second grouped condition would buy.

**And the delivered-dose wobble was not smearing it.** Student-t ν = 3's per-run dose spread reaches
17% at level 1.5, so a second pass rescaled every draw to *exactly* the target amount. Nothing moved:
every condition except grouped-shifted stays within 0.048 R² of Gaussian at ratios to the wobble of
1.04 or less, while grouped-shifted holds at −0.111, −0.094 and −0.312 (2.1×, 2.6×, 4.4×, all
*p* ≤ 0.002).

**The skewed draw (§5.7) does not earn implementation.** Nothing at the reporting level; −0.045 R² for
the random forest alone at level 1.5 and nothing for the other two. §13.3's rejection of a skewed
draw stands, and this measures what it costs: one model out of three, at the top of the grid only.

⚠️ **One statistically significant nothing.** Outlier 5% versus 1% reaches *p* = 0.002 and *p* = 0.001
on differences of +0.005 and +0.003 R² — 0.36 and 0.14 of the wobble, and the *wrong sign* for a dose
response. Common random numbers make the paired difference precise, so a trivial difference can be
significant. Precision around zero is not an effect.

### 5.4 🔴 Two analysis errors of mine, recorded so they do not resurface

1. A pooled sign test across models gave *"96/135 favour concentrated noise, p = 1e-6"*.
   **Misleading** — driven almost entirely by Ridge (42/45); random forest was 25/45, a coin
   flip. Do not pool across models here.
2. A statistic reported as *"shape is 30–49% of the dose effect"* compared a max-minus-min
   *spread across seven noise types* against a *mean drop*. Different quantities; the ratio is
   meaningless. Deleted.

---

## 6. IMPLEMENTATION PLAN

**Principle: no legacy code.** Old noise types are deleted outright, not deprecated or
flag-guarded. Git history is the archive. Every deletion below is recoverable from
commit `6099659` or earlier.

### 6.0 The architecture already exists — use it

`rust/src/main.rs` already separates two concepts, and the redesign maps onto them cleanly:

| Existing concept | What it controls | Redesign uses it for |
|---|---|---|
| `NoiseDistribution` | the **shape** of each draw | Gaussian, Student-t, Laplace |
| `NoiseStrategy` | **who** gets hit and **how hard** (the per-molecule scale map) | Uniform, Grouped, Outlier, Censoring |

This is the right split and it is currently unused — `--distribution` is hard-wired to
`gaussian` at `scripts/process_and_train.py:241`, and six of the seven distribution variants
are unreachable. Building the redesign on this existing separation means the shape and the
targeting are independently selectable, which is exactly what dose-matching needs.

### 6.0a The specification has TWO implementations — this document covers both

Added 2026-08-26 by chat B. As written this section named `rust/src/main.rs` and nothing else,
which is the same omission that let the two injectors drift apart unnoticed for the life of the
project (`RERUN_PLAN.md` §2.3).

| Implementation | File | Produces |
|---|---|---|
| Rust | `rust/src/main.rs` | QM9 — every main-pipeline result |
| Python | `NoiseInject/noiseInject/core.py` | LogD, Caco-2, hERG — every experimental result **and every uncertainty number** |

The Python side additionally reaches the results through two callers that must move with it:
`KIRBy/tests/alternative_data_noise_robustness.py` (which imports the injector directly, not
through `noise_spec`) and `KIRBy/src/kirby/noise_spec.py`.

**Agreement between the two is an enforced gate, not an assumption** — §6.3 item 6, and gate 2 of
`RERUN_PLAN.md` §8. It cannot be an element-wise check: Rust's `StdRng` and numpy's generator
produce different streams, so identical draws are impossible and a check written that way would be
quietly disabled. It compares statistics on the same labels, target, groups and seed.

### 6.1 DELETE (phase 1, no replacement)

**`rust/src/main.rs` — ✅ DONE, chat A, 2026-08-26.** All six unreachable distribution variants
and all five superseded targeting rules are gone, along with `generate_value_based_noise_map`,
`generate_adaptive_noise`, `generate_noise_by_indices` and `sample_from_distribution`. No
deprecation, no flag guards; `git show 6099659:rust/src/main.rs` is the archive.
`scripts/noise_strategy_params.json` and the `--strategy-params` argument are deleted with them.
The two figure-script rows below belong to chat J and are **not** done.

**`rust/src/main.rs`:**

| Delete | Lines | Why |
|---|---|---|
| `NoiseDistribution::LeftTailed`, `RightTailed`, `UShaped`, `Uniform`, `DomainMpnn`, `DomainTanimoto` | enum ~87–92, plus branches at ~205–275 **and** ~461–505 | Unreachable — `--distribution` is always `gaussian`. Duplicated across two functions. `LeftTailed`/`RightTailed` also call `.powf(0.5)` on values that can be negative, so they would produce NaN if ever reached |
| `NoiseStrategy::ValueProportional` | enum ~103, branch ~309, CLI ~1167 | Premise disproved (§3.2) |
| `NoiseStrategy::Quantile` | enum ~109, branch ~320, CLI ~1181 | Premise disproved (§3.2) |
| `NoiseStrategy::Threshold` | enum ~118, branch ~356, CLI ~1192 | Premise disproved, and degenerate (§2) |
| `NoiseStrategy::Heteroscedastic` | enum ~133, branch ~405, CLI ~1211 | Premise disproved; also ranks molecules identically to value-proportional |
| `NoiseStrategy::ScaffoldBased` | enum ~141, branch ~420, CLI ~1222 | Never invoked from Python. Its intent is superseded by the new Grouped type, which is dose-matched |

**Elsewhere:**

| Delete | Why |
|---|---|
| **`NoiseInjectorRegression`'s six strategies** — `legacy`, `quantile`, `threshold`, `outlier` (z-score selection), `hetero`, `valprop`, `NoiseInject/noiseInject/core.py:87-211` | Four have a premise that was directly tested and disproved (§3.2); `outlier`'s selection rule is replaced by random selection; `legacy` is replaced by `uniform`/`gaussian`. Full clean break, author's decision 2026-08-26 |
| **`calibrate_sigma` and `calibrate_multiple_sigmas`**, `NoiseInject/noiseInject/calibration.py:16`, `:82` | A binary search on **mean \|Δy\| / SD** — the *first* moment. The design controls the second. At identical RMS dose, mean\|ε\|/RMS is 0.797 for Gaussian but 0.642 for Student-t ν=3, so calibrating this way hands the heavy-tailed conditions up to **24% more actual noise** at the same nominal level. It also re-uses one injector across its 20 iterations (`:49`, `:67`), so the objective it searches is stochastic. The closed-form solver replaces it: exact, deterministic, and identical to the Rust side. The classification calibrators stay |
| ✅ `scripts/noise_strategy_params.json` | Never passed to the binary (`process_and_train.py:1635` omitted `--strategy_params`). A latent trap: if anyone had ever wired it up, its `base_sigma: 0.1` would have silently flattened every value-proportional curve. **Deleted 2026-08-26** |
| ✅ `--strategy-params` argument, `process_and_train.py` | Dead argument for the dead file. **Deleted 2026-08-26**, along with `--sigma`, `--distribution` and `--noise-strategy` — see §6.2a |
**Three analysis-side deletions used to be listed here and have been moved out** — the retired v1
figure script, its stale output directory, and the synthetic methods-figure block. None of them is
a noise-scheme deletion, and duplicating them here meant two documents specifying the same change.
They are owned by `RERUN_PLAN.md` §5.4 and executed by chat J. The one point this document does
own about the methods figure: it must be redrawn from real labels through the real injector at
matched dose, because the block being deleted reimplements two noise types differently from the
pipeline.

### 6.2 BUILD (phase 2)

**Steps 1–5 are ✅ DONE in Rust, chat A, 2026-08-26.** Step 6 (Python) is chat B's. What was
built and how it is invoked is §6.2a; what it was measured to deliver is §5.1c.

**Step 1 — the dose solver.** One new function, and it is the change that fixes the confound.

```
unit_dose(strategy, distribution, clean_train_labels) -> G
    G = rms(per-molecule scale map) * shape_unit_sd
        where shape_unit_sd = 1                     for Gaussian
                            = sqrt(nu/(nu-2))       for Student-t
                            = sqrt(2)               for Laplace
scale = target_dose / G
```
Computed **once per (dataset, split, strategy, parameters)** from clean training labels only.
Working reference, already built and tested: `rust/reference/noise_arms.rs`.

**Step 2 — the three shapes** on `NoiseDistribution`: keep `Gaussian`, add `StudentT { nu }`
and `Laplace`. Student-t drawn as `z / sqrt(chi2(nu)/nu)`; Laplace by inverse transform.
**Reject `nu <= 2` at argument-parse time** — the variance is undefined there and the run
would be silently meaningless.

**Step 3 — the four targeting rules** on `NoiseStrategy`:
- `Uniform` — every molecule scale 1 (replaces `Legacy`)
- `Grouped { lambda, group_fraction }` — scaffold clusters, via the assignments already
  loadable at `rust/src/main.rs:280`. **Compute the affected *molecule* fraction from the
  actual assignment**, never assume it equals the group fraction
- `Outlier { p, lambda }` — **random** selection, not z-score on the label
- `Censoring { fraction, side }` — clip beyond a quantile of the training labels. Does not
  go through the dose solver; see §6.4

**Step 4 — fix the standardisation order.** Currently noise is added at `:751` and the
labels are standardised at `:759-760` **using the noisy standard deviation**, so the target
scale moves with the noise level. Standardise using the **clean training** mean and standard
deviation, computed before injection.

**Step 5 — emit the provenance.** Every results row carries `G`, realised dose in label
units, realised dose as a fraction of label spread, and the affected-molecule fraction. No
figure should ever again be un-traceable to the amount of noise actually delivered.

**Step 6 — the same five steps in Python**, in `NoiseInject/noiseInject/core.py`, mirroring the
Rust separation of shape from targeting so the two stay comparable line by line:

```
NoiseInjectorRegression(strategy=..., distribution=..., random_state=..., **params)

  distribution  gaussian | student_t (nu > 2, rejected at construction) | laplace
  strategy      uniform | grouped_wider | grouped_shifted | outlier | censoring
```

- `scale_map(y, groups, **p) -> (scales, affected_fraction)` — draws only for the two selection rules
- `unit_dose(scales, **p) -> G` — `√(mean(scale²)) × shape_unit_sd`; `solved = τ / G`
- `inject_verbose(y, dose, groups=, reference=) -> InjectionResult` — carries every provenance field
  in `RERUN_PLAN.md` §5.2, and unpacks as `(y_noisy, noise_scale, epsilon)` for existing callers
- `noise_scale(y, dose, reference=, groups=, reference_groups=)` — **keep this surface.** It scores
  held-out molecules against the pattern the *training* labels were exposed to, and the uncertainty
  confound control is built on it. Two references, and **both are needed**: `reference` fixes any
  label cut-point (censoring's limit), and `reference_groups` fixes *which groups were selected*.
  Without the second the selection is re-run over the held-out molecules' own groups and picks a
  different set — measured on a 40-group split, two of the eight groups corrupted in training went
  unmarked — so question B would be scored against an injection that never happened, for exactly the
  conditions §3.2 identifies as the only ones with a pattern to find.

  Under the new set the scale is constant for the three shape-only conditions **and for
  grouped-shifted**, whose group offsets are all drawn from one distribution, so every molecule is
  equally affected and what differs by group is the direction rather than a magnitude. For those
  four the "where is the noise" question is **undefined**, not zero, and the result says so
  (`scale_is_degenerate`) instead of returning a silent constant
- `CONDITIONS` — one registry name per run condition (`gaussian`, `student_t_nu5`, `grouped_shifted`,
  `outlier_p05`, `censoring_25`, …), so a job script, a results row and a figure label agree

### 6.2a ✅ What the Rust injector now takes, and what it writes — chat A, 2026-08-26

Shape and targeting are separately selectable, as §6.0 intended. `--sigma`,
`--noise_distribution`, `--noise_strategy` and `--strategy_params` are gone; `process_and_train.py`
**refuses** them by name rather than ignoring them, because a job script written against the old
scheme would otherwise run silently under the new one, where the level means something different.

| Argument | Values | Default |
|---|---|---|
| `--noise-level` | the dose to deliver; for censoring, the fraction of labels clipped | 0 |
| `--dose-units` | `spread` (a fraction of the clean training label SD) or `label` (the label's own units) | `spread` |
| `--noise-shape` | `gaussian`, `student_t`, `laplace` | `gaussian` |
| `--noise-targeting` | `uniform`, `grouped_wide`, `grouped_shift`, `outlier`, `censoring` | `uniform` |
| `--nu` | degrees of freedom; **refused at or below 2** | 5 |
| `--lambda` | how many times wider the affected molecules' error is | 3 (Avdeef 2019) |
| `--group-fraction` | affected **molecule** fraction for `grouped_wide` | 0.2 (a stated choice) |
| `--group-variance-share` | ρ, the group-level share of the variance for `grouped_shift` | 0.62 (Bentz 2013 Table 7) |
| `--outlier-p` | contaminated fraction | 0.05 (Hampel 2001) |
| `--censor-side` | `upper` or `lower` | `upper` |
| `--scaffold-file` | canonical SMILES → scaffold group id, written by `process_and_train.py` | `scaffold_groups_{file_no}.json` |
| `--noise-manifest` / `--noise-provenance` | where the provenance goes | `noise_{manifest,provenance}_{file_no}.{json,csv}` |
| `--self-test <labels>` | run the gates on a labels file and exit non-zero on failure. No pipeline needed | — |

**Two files come out of every run, and neither existed before.**

`noise_manifest_{file_no}.json` — the run-level record: `noise_type`, `noise_shape`,
`noise_targeting`, `noise_level`, `unit_dose`, `solved_scale`, `target_dose_in_label_units`,
`delivered_dose_in_label_units`, `delivered_dose_as_fraction_of_label_spread`, `mean_epsilon`,
`affected_molecule_fraction`, `effective_n`, `standardisation_mean`, `standardisation_sd`,
`clean_label_mean`, `clean_label_sd`, `seed`, `n_train`, and every condition parameter including
`n_scaffold_groups` and `largest_group_share_of_molecules`. `process_and_train.py` appends it to
`<results>_noise_manifest.csv`, so every results row can be joined back to the amount of noise
that produced it.

`noise_provenance_{file_no}.csv` — one row per molecule, **every split**:
`split, record_index, canonical_smiles, y_clean_raw, epsilon_raw, y_noisy_raw, y_written`. The
held-out rows carry `epsilon_raw = 0` exactly, so the file is itself the evidence that the
held-out labels were untouched.

**Scaffold groups are keyed by canonical SMILES, not by row position.** `process_and_train.py`
builds the map with `MurckoScaffoldSmiles` in `build_scaffold_groups`, splitting acyclic molecules
into singletons per §2a rule 2, and the injector **refuses to run** if more than 1% of the training
molecules are missing from it — otherwise a stale file would quietly turn grouped noise into
uniform noise while still calling itself grouped. The write path additionally asserts, molecule by
molecule, that the row it is about to apply noise to is the row the noise was drawn for. That is
the guard for the *class* the original held-out bug belonged to, not for that one instance.

### 6.3 VERIFY (phase 3, before any cluster time)

**Status, chat A 2026-08-26 — closed out.** Items 1, 2, 3, 5, 7 and 8 are implemented as checks
that fail rather than notes that reassure, and they pass. Item 6 is chat B's. Item 4 needs a
training run and is chat H's.

Four commands re-run the lot, none of them needing the cluster or the Python training stack:

```
cd rust && cargo test --release --test noise_gates          # 15 gates over real mmap files
./rust/target/release/rust_processor --self-test <labels.csv> --scaffold-file <groups.json>
python scripts/crosscheck_pipeline_reference.py --labels <labels.csv> --groups <groups.json>
python scripts/test_injector_wiring.py                      # the Python driver's helpers
```

The last of those exists because `process_and_train.py` cannot be imported without
`torch_geometric`, so the helpers that write the scaffold-group file and the manifest were only
ever read. It lifts them out with `ast` and runs them, including §2a rule 2 — falsified by
collapsing the acyclic molecules back into one group.

| Check | Where | Runs as |
|---|---|---|
| 1 — the pipeline against the reference implementation | `scripts/crosscheck_pipeline_reference.py` | exits non-zero on disagreement **and on incomplete coverage**; see §5.1d |
| 2, 5, 7 — flatness across conditions, the ν→∞ limit, exact zero | `rust/src/main.rs`, `self_test` | `rust_processor --self-test <labels> [--scaffold-file <groups>]`, exits non-zero on failure. **This is the preflight command** |
| The Python driver's helpers — the acyclic-singleton rule, the manifest columns, the retired-flag refusal | `scripts/test_injector_wiring.py` | runs without the training stack |
| 1, 3, 7, 8 plus the standardisation order, the molecule-identity guard, censoring's direction, the ν ≤ 2 refusal, a mismatched scaffold file, a short record stream, manifest completeness, and the effective group count | `rust/tests/noise_gates.rs` — 15 gates over real mmap files | `cargo test --release` |

Each was checked by removing the fix and confirming the check fails. Reverting the standardisation
order fails `standardisation_uses_the_clean_training_spread`. Re-applying noise to the held-out
splits fails `held_out_labels_are_bit_identical_across_levels`. Removing the dose solver fails
`gate_one_dose_is_flat_across_types`, with the conditions spread across 1.00–1.70× the Gaussian
dose — the original confound, reproduced on demand.

1. ✅ **Against the reference implementation.** `rust/reference/noise_arms.rs` reproduces every
   dose to within ±0.5% (±2.2% for Student-t ν=3, which is sampling variability — §5.1b).
   The pipeline must match it. **Executable: `scripts/crosscheck_pipeline_reference.py`**,
   which exits non-zero on any disagreement and on incomplete coverage. Results in §5.1d.
2. **Dose is flat across noise types.** At a fixed target, realised dose must be identical
   for every type. This is the single check that proves the confound is gone.
3. **Held-out labels are untouched.** Assert `y_true` is bit-identical across every noise
   level — the check that caught the original bug (§2.1 of `RERUN_PLAN.md`).
4. **Clean-label run reproduces the old clean-label run.** At zero noise nothing has changed,
   so R² must match the existing σ=0 numbers.
5. **Student-t reduces to Gaussian at large ν.** At ν = 200 the results must be
   indistinguishable from the Gaussian type.
6. ✅ **The two implementations agree.** Same labels, same target, same group assignment, same
   seeds — Rust and Python compared on realised dose, unit dose `G`, the fraction of labels off by
   more than 3τ, the median absolute error, the worst-hit 5%'s share of the total noise energy, and
   the realised affected-molecule fraction.

   **The dose tolerance is DERIVED per condition, not fixed.** `dose_tolerance` — the same function
   in `rust/src/main.rs`, `rust/reference/noise_arms.rs` and `noiseInject` — takes three standard
   errors of a root-mean-square estimate from the condition's own fourth moment and its effective
   number of independent contributions, floored at the half a percent §5.1b quotes for the full QM9
   column, with a flat 15% for Student-t at ν ≤ 4 where the fourth moment is infinite and the sample
   kurtosis is itself meaningless. The cross-check then divides by √seeds for a mean, and by
   √(2/seeds) for a difference between two independent means.

   An earlier draft of this item fixed the numbers at 0.5%, relaxed to 3% for Student-t ν = 3 and
   grouped-shifted. That is a hand-kept list of exceptions: it needs editing whenever a condition is
   added and silently stops covering the new one. Deriving it also *found* something — the effective
   count for grouped-shifted was being taken as the group count when the group term is averaged over
   molecules (§2a rule 3, `RERUN_PLAN.md` §2.3a).

   Executable: **`python scripts/crosscheck_injectors.py`**, which exits non-zero on any failure.
   Verified to do so: sabotaging the dose solver fails 40 of 154 checks with exit code 1, and a
   condition present in one implementation but not the other fails by name rather than being
   silently skipped.
7. **Zero dose records exactly zero** — not a small number. The negative control the old
   reconstruction never had.
8. **The recorded noise reconstructs the label exactly**: `y_clean + epsilon == y_noisy`, every
   condition, every level.

### 6.4 THE NOISE LEVELS — revised, and this supersedes the old σ ladder

The old approach set one knob and let each noise type deliver whatever it delivered. That is
gone. The revision has two parts, per §1:

**Experimental datasets — choose levels in log units, anchored to real assay error.**

| Dataset | One unit of real error | Proposed levels (log units) |
|---|---|---|
| logD | 0.15 within a lab; 0.50 between methods | 0, 0.15, 0.3, 0.5, 0.75, 1.0 |
| Caco-2 | 0.35 between 11 labs | 0, 0.1, 0.2, 0.35, 0.5, 0.7 |
| hERG | 0.54 (stand-in) | 0, 0.15, 0.3, 0.54, 0.8, 1.1 |

Each grid brackets one unit of real error and runs to roughly twice it. Report the resulting
fraction-of-spread alongside, because **one unit of real error is 0.13 of the label spread on
logD but 0.76 on Caco-2** — a factor of six. A single shared ladder would mean six different
experiments.

**QM9 — no assay error exists**, so fraction-of-label-spread is the only honest axis.
**Grid set by the range-finding run (§5.5): 0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5.** Below 0.2 the
models are indistinguishable from clean; 1.5 is retained because it is the only level where
the noise types separate at all.

**Censoring is swept separately** on its own axis — fraction of labels clipped — because it
has no variance parameter and is not zero-mean. **Grid set by the range-finding run (§5.5): 0, 10, 20, 25, 30, 40, 50%.** There is no sharp
knee — damage accelerates smoothly, becoming serious around 20% and severe past 30%.

### 6.5 ORDER OF WORK — the noise scheme only

**The order of the whole re-run is `RERUN_PLAN.md` §10 and is not repeated here.** This table
covers the build of the noise scheme itself, and doubles as its completion record.

| # | Step | Blocks |
|---|---|---|
| 1 | ✅ Range-finding run completes; fix the QM9 and censoring level grids | everything |
| 2 | ✅ Delete §6.1 — Rust done 2026-08-26. The two figure-script rows are chat J's | — |
| 3 | ✅ Build §6.2 — Rust steps 1–5 done 2026-08-26. Step 6 (Python) is chat B's | 2 |
| 4 | ✅ Verify §6.3 locally on QM9 at N=4,000 — done 2026-08-26, see §5.1c | 3 |
| 5 | Author the cluster scripts | 4 |
| 6 | QM9 re-run — also clears the held-out-label contamination | 5 |
| 7 | Experimental datasets re-run | 5 |
| 8 | Rebuild the figure script against the new columns | 6, 7 |

Steps 2–4 are local and cost nothing but time. **No cluster time is spent until the dose is
verified flat across noise types.**

## 7. Decisions

### Settled

| Decision | Outcome |
|---|---|
| **Censoring is in** | ✅ **Confirmed by the author, and the data agrees emphatically.** It causes 12× more damage than any difference between the zero-mean types (§5.3b). It is no longer an optional extra — it is the strongest effect in the study |
| **Keep the name "Outlier"** | ✅ It is the familiar term and matches Huber/Tukey usage. What changes is the selection rule (random, not extreme-label), not the name |
| **Drop the four value-dependent types** | ✅ Their premise was directly tested and disproved (§3.2). Independently corroborated: the uncertainty re-run's own review found threshold degenerate on hERG and found heteroscedastic and value-proportional ranking molecules identically |
| **Dose is the result, not the knob** | ✅ Every noise type solves for its scale so the delivered dose matches |
| **Anchor in log units for the experimental datasets, fraction-of-spread for QM9** | ✅ One unit of real error is 0.13 of the label spread on logD but 0.76 on Caco-2 — a shared ladder would mean six different experiments |
| **No separate artificial positive control is needed** | ✅ **Closed 2026-08-26.** This was open as "keep one label-keyed noise type as a deliberate positive control?" The question assumed the replacement set had no label-keyed condition. It has one: **censoring is keyed to the label by construction** — which molecules get clipped is a deterministic function of the value — so the condition the control was wanted for already exists, and the zero-noise subtraction does real work there. `RERUN_PLAN.md` §3.2 sets out the full mapping of which types have a learnable pattern and which are true nulls; §4 Decision 4 records the same conclusion. Reopen only if an *additional* deliberately unrealistic condition is wanted on top |
| **What determines the QM9 and censoring level grids** | ✅ **Closed 2026-08-26.** The range-finding run has completed (§5.5) and the grids are set from it, in §6.4 |

| **Which conditions run, and at which stage** | ✅ **Settled 2026-08-27**, on twelve replicates at matched delivered dose (§5.8). Four at full grid — Gaussian, both grouped conditions, censoring. One Student-t setting (ν = 5) and one Outlier setting (p = 10%) at depth. Four settings dropped, the skewed draw never built. It lives in `noise_conditions.json` at the repository root, which `rust/tests/noise_gates.rs` and `scripts/test_noise_conditions.py` both read — so a grid that stops matching it fails a test rather than quietly running |

### Still open

**Nothing.** The last item closed 2026-08-27.

**1. ✅ CLOSED 2026-08-27 — Laplace is kept, at depth.** *("Keep laplace".)* It runs in the depth
stage and not in the full grid. `noise_conditions.json` carries it without the `optional` marker, so
the file now says what was decided rather than leaving it to whoever reads it. The reasoning below
stands as the record of what the decision was made on.

**The question as it was:** *(Built and verified either way — chat A,
2026-08-26. Nothing is blocked on this: it is a question of whether the condition is queued, not
whether it exists.)*

**Narrowed 2026-08-27 (§5.8).** It is **out of the full grid on measurement** — indistinguishable
from Gaussian on every model at both levels tested, largest difference 0.0058 R² against a test that
could have seen 0.0086. So the question is no longer whether it joins the breadth stage at 4,680
training runs; it is whether it runs at depth for **720**. It adds nothing statistically — it sits
near Student-t at ν = 6 — but it is the only distribution family actually *fitted* to real
bioactivity data (Anderson-Darling rejects normality at p < 2×10⁻¹⁶; Laplace fitted with scale 0.7
and 1.3). **Buys a citation, not a result.** QM9 only if included. It is marked `optional` in
`noise_conditions.json`, so either answer passes the gates.

Items 2 and 3 were here and are now settled above — the positive-control question was answered by
censoring, and the grids were set by the range-finding run. **Laplace is the only item still open in
this document**, and it is now a 720-run yes/no. `RERUN_PLAN.md` §4 Decision 4 and §13.5 carry it on
the process side, with the context the author asked for.

## Sources

**Every source below now has an entry in `citations.bib`**, added 2026-08-26 with metadata taken
from Crossref DOI content negotiation (or the publisher record where no DOI exists). The BibTeX
key is given so the paper pass can cite directly. `RERUN_PLAN.md` §13.8 tracks the manuscript side.

| Source | BibTeX key |
|---|---|
| Alvarez Baron C, Zhao J, Yu H, Ren M, Thiebaud N, Guo D, et al. (2025). *Sci Rep* 15:29995. doi:10.1038/s41598-025-15761-8 | `AlvarezBaron2025` |
| Assay Guidance Manual — Iversen PW, Beck B, Chen Y-F, Dere W, Devanarayan V, Eastwood BJ, et al. (2017). *Assay Operations for SAR Support*. NCBI Bookshelf NBK91994, PMID 22553866 | `AssayGuidanceManual2017` |
| Avdeef A (2019). *ADMET & DMPK* 7(3):210–219. doi:10.5599/admet.698 | `Avdeef2019` |
| Bentz J, O'Connor MP, Bednarczyk D, Coleman J, Lee C, Palm J, et al. (2013). *Drug Metab Dispos* 41(7):1347–1366. doi:10.1124/dmd.112.050500 | `Bentz2013` |
| Bruneau P, McElroy NR (2006). *J Chem Inf Model* 46(3):1379–1387. doi:10.1021/ci0504014 | `Bruneau2006` |
| Chen X, Slättengren T, de Lange ECM, Smith DE, Hammarlund-Udenaes M (2017). *Fluids Barriers CNS* 14:30. doi:10.1186/s12987-017-0078-x | `Chen2017` |
| Hampel FR (2001). Robust statistics: a brief introduction and overview. Research Report 94, Seminar für Statistik, ETH Zürich. doi:10.3929/ethz-a-004158597 | `Hampel2001` |
| Hayeshi R, Hilgendorf C, Artursson P, Augustijns P, Brodin B, Dehertogh P, et al. (2008). *Eur J Pharm Sci* 35(5):383–396. doi:10.1016/j.ejps.2008.08.004 | `Hayeshi2008` |
| Heid E, McGill CJ, Vermeire FH, Green WH (2023). *J Chem Inf Model* 63(13):4012–4029. doi:10.1021/acs.jcim.3c00373 | `Heid2023` |
| Horwitz W, Albert R (2006). *J AOAC Int* 89(4):1095–1109. doi:10.1093/jaoac/89.4.1095 | `Horwitz2006` |
| Huber PJ (1964). *Ann Math Statist* 35(1):73–101. doi:10.1214/aoms/1177703732 | `huber1964robust` |
| Kalliokoski T, Kramer C, Vulpetti A, Gedeck P (2013). *PLoS ONE* 8(4):e61007. doi:10.1371/journal.pone.0061007 | `Kalliokoski2013` |
| Kolmar SS, Grulke CM (2021). *J Cheminform* 13:92. doi:10.1186/s13321-021-00571-7 | `Kolmar2021` |
| Kramer C, Kalliokoski T, Gedeck P, Vulpetti A (2012). *J Med Chem* 55(11):5165–5173. doi:10.1021/jm300131x | `Kramer2012` |
| Kramer C, Dahl G, Tyrchan C, Ulander J (2016). *Drug Discov Today* 21(8):1213–1221. doi:10.1016/j.drudis.2016.03.015 | `Kramer2016` |
| Krüger FA, Overington JP (2012). *PLoS Comput Biol* 8(1):e1002333. doi:10.1371/journal.pcbi.1002333 | `Kruger2012` |
| Landrum GA, Riniker S (2024). *J Chem Inf Model* 64(5):1560–1567. doi:10.1021/acs.jcim.4c00049 | `landrum2024` |
| Lange KL, Little RJA, Taylor JMG (1989). *JASA* 84(408):881–896. doi:10.1080/01621459.1989.10478852 | `Lange1989` |
| Larregieu CA, Benet LZ (2013). *AAPS J* 15(2):483–497. doi:10.1208/s12248-013-9456-8 | `Larregieu2013` |
| Llinàs A, Avdeef A (2019). *J Chem Inf Model* 59(6):3036–3040. doi:10.1021/acs.jcim.9b00345 | `Llinas2019` |
| Niu Z, Xiao X, Wu W, Cai Q, Jiang Y, Jin W, et al. (2024). *Sci Data* 11:985. doi:10.1038/s41597-024-03793-0 | `Niu2024` |
| O'Hagan S, Kell DB (2015). *PeerJ* 3:e1405. doi:10.7717/peerj.1405 | `OHagan2015` |
| OECD (2022). Test No. 117: Partition Coefficient (n-octanol/water), HPLC Method. doi:10.1787/9789264069824-en | `OECD2022` |
| Prieto P, Hoffmann S, Tirelli V, Tancredi F, González I, Bermejo M, De Angelis I (2010). *ATLA* 38(5):367–386. doi:10.1177/026119291003800510 | `Prieto2010` |
| Sato T, Yuki H, Ogura K, Honma T (2018). *PLoS ONE* 13(7):e0199348. doi:10.1371/journal.pone.0199348 | `Sato2018` |
| Song H, Kim M, Park D, Shin Y, Lee J-G (2023). *IEEE TNNLS* 34(11):8135–8153 | `Song2022` |
| Srinivasan B, Lloyd MD (2025). *J Med Chem* 68(3):2052–2056. doi:10.1021/acs.jmedchem.5c00131 | `Srinivasan2025` |
| Svensson E, Friesacher HR, Winiwarter S, Mervin L, Arany A, Engkvist O (2025). *Artif Intell Life Sci* 7:100128. doi:10.1016/j.ailsci.2025.100128 | `Svensson2025` |
| Tukey JW (1960). A survey of sampling from contaminated distributions. In: Olkin I, ed. *Contributions to Probability and Statistics: Essays in Honor of Harold Hotelling*. Stanford University Press, 448–485 | `Tukey1960` |
| Wenlock MC, Potter T, Barton P, Austin RP (2011). *J Biomol Screen* 16(3):348–355. doi:10.1177/1087057110396372 | `Wenlock2011` |
| Zhao L, Wang W, Sedykh A, Zhu H (2017). *ACS Omega* 2(6):2805–2812. doi:10.1021/acsomega.7b00274 | `Zhao2017` |

**Four corrections this list previously carried, all found 2026-08-26 when the entries were built
from Crossref:**

- **Zhao 2017's authors were wrong here.** This list said "Zhao Y, Wang J"; the paper is
  **Zhao Linlin, Wang Wenyi**, Sedykh A, Zhu H. It is the published precedent for the `k` axis in
  §1, so the attribution matters.
- **Sato 2018 has four authors, not six and not eleven.** Crossref: **Sato T, Yuki H, Ogura K,
  Honma T**. §4b's citation line named "Takaya D, Sasaki S, Tanaka A" and its section heading named
  eleven people; both were wrong.
- **Bruneau & McElroy is a 2006 journal issue** (JCIM 46(3):1379–1387) with an online-first date of
  December 2005. Cite 2006. Its widely quoted 0.27 log-unit figure still **fails verification** and
  must not be used — §4a.
- **Hampel 2001 is an institutional research report**, not a peer-reviewed article — Research Report
  94, Seminar für Statistik, ETH Zürich, with a stable DOI. It is the source of the p = 1–10%
  contamination fraction in §2, so its status is stated here rather than left to look like a journal
  paper. No peer-reviewed version was found; the canonical peer-reviewed alternative for the same
  material is Hampel, Ronchetti, Rousseeuw & Stahel, *Robust Statistics: The Approach Based on
  Influence Functions* (Wiley, 1986), which does not carry the 1–10% sentence verbatim.
