# Results and Conclusions — diagnosis and rework plan

Working document for rebuilding the Results and Conclusions of the NoiseInject paper.
Last updated 2026-08-24.

**Scope:** predictive performance, robustness, noise strategies, practical guidance.
Per-sample uncertainty is a **separate workstream** and is deliberately out of scope here.

**Companion documents**
- `NOISE_DESIGN.md` — the redesigned noise injection scheme, awaiting sign-off
- Superseded, no longer maintained: `REVISION_GUIDE.md`, `DISCUSSION_REWORK.md`,
  `DISCUSSION_TRACKER.md`, `immediate_next_steps.md`

**Status legend**
- ✅ Verified — computed this session from a named file, reproducible
- 🟡 Partly grounded — computed, but rests on one unchecked assumption
- ⬜ Needs a run — cannot be settled on this laptop
- ❓ Your call

---

## 1. 🔴 THE BLOCKER — every QM9 result must be regenerated

**Every QM9 number in the repository, and therefore every QM9 number in the paper, came
from runs in which the held-out labels were corrupted.**

1. `rust/src/main.rs:751` applies the noise map inside `write_data`, and `write_data`
   restarts its index counter at 0 for every split. The map is keyed by *training* index.
   Applied to validation and test, each held-out molecule received the noise drawn for the
   *training* molecule at the same position — corrupting held-out labels and attaching the
   corruption to the wrong molecules.
2. **Now fixed** — an `apply_noise` flag, true for training only
   (`rust/src/main.rs:1015-1075`), with a source comment describing the old behaviour.
3. **The fix landed 2026-08-24 (commit `9d7db67`).** The derived QM9 results in
   `results/paper_figures_v2/` are dated **8 July**. Nothing has been regenerated.
4. ✅ **The validation pipeline never had the bug.** Verified directly: in
   `results/validation_full/openadmet_caco2/QRF_ECFP4_uncertainty_values.csv` the held-out
   label vector is bit-identical across all 11 noise levels. **LogD, Caco-2 and hERG are
   sound.**

### Why this one is nastier than a normal bug

Corrupting held-out labels does not add scatter. It adds a smooth, predictable downward
bias that *mimics the thing being measured*. If held-out labels receive noise of variance
*v*, a model predicting perfectly still scores R² = 1/(1 + v/Var(y)) — which is exactly the
shape of a degradation curve, with no model dependence at all.

This is not hypothetical: fitting that curve to the contaminated QM9 results gave a
parameter of 1.288 against a label standard deviation of 1.293 — within 0.4% of the
pure-artefact prediction. Any "robustness" analysis built on the current QM9 files is
measuring the bug.

### Consequence for sequencing

Regenerating QM9 is the critical path. The experimental datasets are not blocked, and since
they are the only clean data available, **they should probably move from "validation" to
the front of the Results** — a structural change worth discussing (§6).

---

## 2. ✅ VERIFIED DEFECTS IN THE PAPER

These are document-consistency problems, independent of the re-run. All confirmed against
files this session.

### 2.1 The ANOVA table contradicts its own source data — and the prose beside it

`paper.tex:396-401` versus `results/paper_figures_v2/table1_anova_summary.csv`,
robustness columns:

| Strategy | Model η² paper → CSV | Residual η² paper → CSV |
|---|---|---|
| Gaussian | 48.7 → **43.8** | 31.2 → **34.2** |
| Quantile | 47.9 → **36.8** | 30.4 → **43.7** |
| Threshold | 48.0 → **54.7** | 13.9 → **14.8** |
| Heteroscedastic | **37.0 → 14.0** | **41.0 → 77.4** |
| Value-prop. | 52.8 → **52.5** | 18.2 → **21.6** |
| Outlier | 12.7 → **10.3** | 79.3 → **83.6** |

The heteroscedastic row is the serious one: the paper claims model architecture explains
37% of robustness variance; the data says **14%**, with 77% unexplained. Worse, the prose at
`paper.tex:380` already quotes the *correct* CSV values (83.6 and 77.4) while the table
beside it prints the old ones. **The paper contradicts itself within a single page.**

### 2.2 🔴 A non-significant result is reported as significant

`paper.tex:468-484` versus `results/paper_figures_v2/table3_wilcoxon_tests.csv`:

| Comparison | Paper Δ | CSV Δ | Paper p | **CSV p** | Marked significant? |
|---|---|---|---|---|---|
| NN-α → BNN-α | +0.056 | +0.031 | 2.9e-10 | 2.9e-11 | yes — correct |
| **NN-α → VBLL-α** | **+0.061** | **+0.011** | **1.2e-6** | **0.252** | **yes — WRONG** |
| NN-β → BNN-β | +0.096 | +0.053 | 2.9e-11 | 2.9e-11 | yes — correct |
| NN-β → VBLL-β | +0.124 | +0.062 | 1.2e-7 | 1.2e-7 | yes — correct |
| RF → QRF | −0.022 | −0.012 | 1.6e-10 | 2.9e-11 | yes — correct |

**The variational Bayesian last-layer transformation of NN-α does not significantly improve
robustness. The regenerated p-value is 0.252.** The paper prints 1.2×10⁻⁶ with a
significance asterisk.

This propagates: `paper.tex:466` states *"Both full BNN and VBLL transformations
significantly improved robustness for both NN-α and NN-β"* — false as written.

Every effect size in the table is also roughly double the regenerated value, the signature
of stale noise-degradation-slope numbers.

Additional wrinkle: `table3_probabilistic_comparison.csv` shows the VBLL-α variant has a
much lower clean baseline (R² = 0.763) than the NN-α it is compared against (0.867), so part
of any apparent gain is a baseline effect.

### 2.3 "Robustness is decoupled from baseline performance" is unsupported

`paper.tex:433` asserts it. `results/paper_figures_v2/deep_baseline_vs_robustness.csv`
records verbatim that the QM9 version is a **data gap**: *"GAP: QM9 baseline_r2 ... NEEDS
server dump"*. There is no QM9 evidence for the claim.

The validation evidence that exists contradicts it, inconsistently:

| Scope | Pearson r | Spearman ρ | p |
|---|---|---|---|
| Pooled, all three datasets | +0.525 | +0.587 | 1e-43 |
| OpenADMET LogD | **−0.306** | **−0.423** | 4e-08 |
| OpenADMET Caco-2 | +0.103 | +0.031 | 0.71 (n.s.) |
| ChEMBL hERG Ki | +0.304 | +0.330 | 4e-05 |

The sign flips between datasets, and the pooled positive is a pooling artefact. The honest
statement is **"the relationship is inconsistent and dataset-specific"** — weaker, but true,
and arguably more interesting since it means robustness cannot be inferred from a
clean-data leaderboard.

### 2.4 The paper contradicts itself about PDV

- `paper.tex:462` — *"PDV stood out as having particularly strong robustness to noise"*
- `paper.tex:493` — *"PDV configurations produced the shallowest mean NDSs"*
- `paper.tex:567` — *"PDVs ... are the **least** noise-resistant"*

✅ Settled from the data: physicochemical descriptors are the **most** robust representation
of those tested. **`paper.tex:567` is wrong and must go.** The same sentence also claims
embeddings and "particularly mol2vec" are most robust — mol2vec does come second, so that
half needs softening rather than deleting.

### 2.5 ✅ The exclusion threshold — settled, two statements are wrong

`results/paper_figures_v2/excluded_configs.csv` holds 48 excluded cells with baselines from
−0.190 to **0.072** — all far below 0.3, nothing between 0.3 and 0.6. **The code uses 0.3.**

- `paper.tex:247`, `:418`, `:422` — correct
- `paper.tex:464` — *"baseline R² < 0.6"* ❌ **change to 0.3**
- `paper.tex:663` (Additional file 5) — *"R² ≤ 0.6"* ❌ **change to 0.3**

### 2.6 ✅ A material omission about which models were unstable

The 48 exclusions are exactly four models × two representations × six strategies: BNN-α,
BNN-β, VBLL-α and VBLL-β, each on MHG-GNN and mol2vec.

`paper.tex:464` reports only half — *"all 24 VBLL × {MHG-GNN, mol2vec} configurations were
excluded"*. The count is right for VBLL, but **the two full Bayesian networks failed on
exactly the same representations, for another 24 exclusions, and the paper does not say so.**

That matters because the paper's story is that full Bayesian transformation is the robust
one and VBLL the representation-dependent one. On these two embeddings **both** collapsed
equally.

### 2.7 Smaller mismatches, checked individually

| Claim | Paper | Recomputed | Verdict |
|---|---|---|---|
| Kendall's W across strategies (`:433`) | 0.9121, p=3.55e-8 | **0.9374**, p=1.85e-8 | Mismatch — headline number |
| Spearman, ECFP4 vs PDV (`:413`) | 0.82, p=0.002 | 0.818, p=0.0021 | ✅ correct |
| Spread, outlier strategy (`:460`) | 0.02 | 0.0233 | ✅ correct |
| Spread, threshold strategy (`:460`) | 0.13 | 0.1288 | ✅ correct |

### 2.8 The retired metric is still everywhere

The robustness metric was replaced by the normalised retention area, but "NDS" survives in
the ANOVA table caption (`:387`), the Wilcoxon caption and column header (`:470`, `:475`),
body text (`:462`, `:464`, `:493`, `:495`), both validation figure captions (`:547`, `:556`),
the **entire Conclusion** (`:571-573`), the abbreviations list (`:598`), and six additional
file descriptions (`:660-669`).

Not cosmetic: the Conclusion currently defines the paper's headline metric as a slope that
is no longer computed. Note `:387` and `:409` caption the *same* numbers with two different
metrics.

### 2.9 The Methods figure does not show the experiment

`paper.tex:359` captions `fig_methods_noise_strategies` as showing *"the QM9 HOMO–LUMO gap
label distribution"*. The generating code
(`scripts/generate_paper_figures_v2.py:2541-2562`) uses a **synthetic three-component
Gaussian mixture** and reimplements two strategies differently from the Rust pipeline —
threshold as a **median split**, value-proportional as additive rather than multiplicative.
So "threshold" has three different definitions in three places, and the per-panel RMSE
annotations describe the synthetic data under the reimplemented rules.

### 2.10 The Methods misdescribe the noise injection

- `paper.tex:354` — *"samples with |y| > 1.0 (on normalized data)"*. **It is not normalised
  data.** Noise is added to raw labels (`:751`) and standardised afterwards (`:759-760`).
  On QM9 in electronvolts that cut catches 99.9992% of molecules.
- `paper.tex:313` — *"Validation and test data remain free of noise."* False for QM9 at the
  time the results were generated (§1).
- Standardisation uses the **noisy** standard deviation, so the target scale moves with the
  noise level — a second, separate confound.
- `paper.tex:186` — the Heid citation misrepresents its source. See `NOISE_DESIGN.md` §3.6.

---

## 3. ✅ WHAT SURVIVES ON CLEAN DATA

Computed on the uncontaminated validation pipeline.

### 3.1 The headline claim holds

Fitting a noise-tolerance parameter directly to the raw per-σ points on clean LogD data,
15 model × representation configurations, median fit R² = 0.906:

| | LogD (**clean**) | QM9 (contaminated) |
|---|---|---|
| Model architecture | **71.4%** | 74.9% |
| Molecular representation | **10.8%** | 8.6% |
| Interaction / residual | **17.8%** | 16.5% |

The two agree closely, and the clean one is the one that counts. **The paper's central
claim — that model architecture dominates representation in determining noise robustness —
is supported by data without the bug.** The paper's spine is intact even though its QM9
numbers are not.

⚠️ Caveats: only 4 models and 4 representations on LogD, n = 15 cells, against QM9's 11 × 9.
Caco-2 carries only ECFP4 so cannot test representation at all.

### 3.2 The σ=0.6 ANOVA you proposed — worth doing, shows a gradient

Run on clean LogD:

| Outcome | Model | Representation | Interaction | Model p |
|---|---|---|---|---|
| R² at σ = 0 | 47.6% | 11.4% | 41.1% | 0.11 |
| R² at σ = 0.3 | 46.9% | 12.6% | 40.5% | 0.11 |
| **R² at σ = 0.6** | **53.8%** | **9.7%** | 36.5% | 0.062 |
| R² at σ = 1.0 | 55.8% | 8.3% | 35.9% | 0.055 |

Three findings:
1. **It reveals a gradient, which beats the paper's dichotomy.** As noise rises, the model's
   share climbs and the representation's falls, monotonically. The paper currently tells this
   as a categorical inversion (`:380-381`). A smooth gradient is truer and easier to explain.
2. **It does not reproduce the "interaction dominates performance" claim.** On clean LogD,
   model architecture leads at *every* noise level including σ = 0. That claim is a QM9
   result, and QM9 is contaminated. ⬜ Only the re-run can settle it.
3. ⚠️ **Badly underpowered on validation data** — best p-value 0.062 at n = 17. The power is
   in the QM9 design (11 models, 9 representations, 10 replicates).

**Verdict: add it to the QM9 re-run, not the validation analysis, and report it as a series
across noise levels rather than at one value — the gradient is the finding.**

### 3.3 🔴 auc_norm is unsound on the experimental datasets

From `results/validation_full/openadmet_logd/all_results.csv` (1,020 points) and
`results/paper_figures_v2/table_validation_auc_full.csv` (465 rows):

- **16.1% of LogD retention ratios exceed 1** — the model scored *better* with noise added.
  Largest is **8.91**. All are measurement noise in the baseline, integrated straight in.
- 21 of 465 validation auc_norm values are **negative**; 5 exceed 1.
- Dispersion depends brutally on the baseline:

  | Baseline R² | n | auc_norm range | SD |
  |---|---|---|---|
  | 0.3 – 0.5 | 176 | −0.40 to +1.03 | **0.341** |
  | 0.5 – 0.7 | 151 | −0.25 to +1.00 | 0.269 |
  | 0.7 – 1.0 | 138 | +0.74 to +1.00 | **0.054** |

  **Six times noisier for weak configurations than strong ones** — least trustworthy exactly
  where robustness matters most.
- The guard does not guard: **zero** validation configurations fall below the 0.3 threshold,
  so the exclusion rule removes nothing there.

This is a dividing-by-a-noisy-small-number problem and cannot be patched by raising the
threshold. **auc_norm cannot carry the paper as the primary robustness metric.**

Replacement, in order of preference:
1. **Absolute R² at a fixed, stated noise-to-signal ratio** — your σ=0.6 sanity check,
   expressed as a fraction of the label standard deviation so it is comparable across
   datasets. No division by a noisy baseline at all.
2. A fitted noise-tolerance parameter, **fitted to raw R²-versus-noise points, never to
   retention areas** — fitted from auc_norm on validation it inherits the instability and
   even produces impossible negative values.

---

## 4. 📌 MATERIAL FOR THE PAPER

### 4.1 The assay-error anchor

The paper never states what any noise level means in real terms. This is the highest-value
writing fix available and needs no re-run. Full table, sources and the blocklist of bad
numbers: **`NOISE_DESIGN.md` §4.**

Summary: pIC50 **0.68** log units (Kalliokoski 2013); pKi **0.54** (Kramer 2012); hERG
**0.5–0.7**; Caco-2 **≈0.35 log₁₀**; logD **≈0.15** within a lab. QM9 has no assay error to
anchor to and must be worded separately.

Where it goes: Methods near `:240`; Results at any stated operating point (`:380`);
Validation near `:551`, where the hERG set's own retained noise floor (filtered at
inter-assay SD > 1.0, `:197`) approaches the largest noise injected — currently unacknowledged.

### 4.2 Background material — where real error comes from

For the Introduction. All peer-reviewed, all verified; full quotes in `NOISE_DESIGN.md` §3.

- **62% of real measurement variance is between laboratories** (Bentz et al. 2013,
  *Drug Metab Dispos* 41(7):1347, Table 7). Corroborated by Landrum & Riniker 2024: curating
  to a single assay cuts mean disagreement from 0.50 to 0.27 log units. **The dominant
  structure in real label error is provenance, not label magnitude.**
- **Bioactivity error is formally non-normal** — Anderson-Darling p < 2×10⁻¹⁶, with a Laplace
  fitted (Krüger & Overington 2012, *PLoS Comput Biol* 8(1):e1002333).
- **Error does not depend on the measured value** (Kalliokoski et al. 2013, 16,844 repeat
  measurements). This is the justification for dropping four of the six noise strategies and
  belongs in the Background, not a methods footnote.
- **Experimental uncertainty caps achievable accuracy** — *"the maximum possible squared
  Pearson correlation coefficient (R²) on large data sets is estimated to be 0.81"*
  (Kramer et al. 2012). Attribute as specific to heterogeneous public Ki data.

---

## 5. ⬜ PROVISIONAL figure and table list

**Provisional — depends on the QM9 re-run.** Current main text carries 8 figures and 6
tables, several supporting a single sentence.

### Proposed figures

1. **The dose collapse** — R² against nominal noise level (curves fanning apart) beside the
   same data against realised dose (curves collapsing). Replaces the strategy half of
   `fig1_global_overview` and most of the strategy prose. ✅ exists for LogD; ⬜ QM9 needs the re-run.
2. **Noise tolerance across models and representations** — heatmap, rows ordered by model
   median, with the label standard deviation marked as a reference. Replaces
   `fig2_anova_decomposition`, `fig_interaction`, and the heatmap panel of
   `fig1_global_overview` — three figures into one.
3. **Bayesian transformation raises the level and removes representation dependence** —
   plain versus full-Bayesian versus variational, showing both the shift and the collapse in
   spread. Replaces `fig_nn_family_comparison` with a stronger point.
4. **Practical guidance** — retained R² against noise as a fraction of label spread, one line
   per model, with a band for typical assay error. **New; the figure the paper lacks.**
5. **The experimental datasets** — does the QM9 ordering transfer. Replaces
   `fig_validation_overview` *and* `fig_validation_combined`.

Plus the **Methods figure**, which stays but must be redrawn from real labels through the
real definitions (§2.9), and should show strategies at **matched dose**.

**Net: 5 main figures plus 1 methods figure, down from 8.**

### Proposed tables

| # | Table | Status |
|---|---|---|
| T1 | Metrics summary | **Revise** — drop the retired slope metric and the calibration rows that moved out |
| T2 | Noise strategies | **Revise, critical** — add realised-dose and affected-fraction columns. Not having these is what hid the confound |
| T3 | Variance decomposition | **Replace** — one small table instead of the current 6×8 grid, whose numbers are stale (§2.1) |
| T4 | Model ranking | **Replace** `tab:auc_ranking` with retained R² at two or three stated noise levels |
| T5 | Bayesian transformation tests | **Keep but fix** the significance error (§2.2) |

**Net: 5 tables, two substantially rebuilt.**

Moving to additional files: the per-strategy robustness grid, the ECFP4 replication, the
exclusion list, redundancy and intraclass-correlation tables, per-dataset validation heatmaps.

---

## 6. ❓ OPEN DECISIONS

1. **Does "dose, not pattern" become the headline**, or stay a caveat inside the existing
   model-versus-representation story? Changes the Introduction, not just the Results.
2. **Do the experimental datasets move to the front?** They are the only clean data.
3. **Sign off the noise redesign** (`NOISE_DESIGN.md` §7) before the QM9 re-run, so the
   re-run emits what the new figures need in one pass.
4. **Given the pilot** (`NOISE_DESIGN.md` §5.3), is the accuracy experiment worth the compute,
   or does the design pivot to the uncertainty question?

---

## 7. What has deliberately NOT been done

- `paper.tex` untouched. `generate_paper_figures_v2.py` untouched.
- No conclusions drafted — they would rest on numbers that do not yet exist.
- Uncertainty analysis excluded throughout (separate workstream).
