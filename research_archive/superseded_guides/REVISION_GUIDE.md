# NoiseInject — Paper Revision Guide

> The author edits `paper.tex`; this guide tells them what to change and hands them replacement prose. Built from the submitted manuscript and the regenerated CSVs in `results/paper_figures_v2/` and `results/paper_figures/`.
>
> **Two changes, both large:** (1) robustness metric **NDS -> AUC_norm**; (2) uncertainty measured **within each sigma** (which reverses the per-sample finding). AUC_norm is simply the metric — there is no prior NDS publication to justify against, and there is one manuscript, not an "expanded" one. Weibull is not used.

---

# OPEN THREADS TRACKER (live — updated 2026-08-15)

Every live issue and its status. `SERVER` = waiting on an ARC dump; `DECIDE` = needs author call; `LOGGED` = fix written into this guide; `DONE` = resolved.

| # | Thread | Status | Detail / where |
|---|--------|--------|----------------|
| T1 | Per-sample uncertainty = NULL | DONE | max within-σ |ρ|=0.129; abstract→1-line null, SC→removed. |
| T2 | ANOVA residual-dominance (Outlier 83.6 / Hetero 77.4) REAL not artifact | DONE | roster=7 consistent; confirmed on ARC. |
| T3 | ANOVA **table in paper.tex is STALE** (L396–401 has 48.7/41.0…) | LOGGED | replace body with corrected table (guide "tab:anova_decomposition"). |
| T4 | "Robustness decoupled from baseline accuracy" (L433) | PROVISIONAL + nuance | QM9 decoupling from server `.out` (+0.046) — NOT locally reproducible (baseline CSV absent), mark provisional; KEEP L433/L460. ADME is dataset-specific & sign-flips (hERG +0.30, LogD −0.31, Caco-2 ~0) — NOT "coupled". *(I flip-flopped twice here: "remove it" then "+0.525 coupled" — both wrong. This is the corrected version.)* |
| T5 | PDV self-contradiction (L462/L493 "most" vs L567 "least" robust) | DECIDE | both NDS-stale; local auc_norm: continuous_pdv mid-cluster. See T7. |
| T6 | **Noise types differ — representation** | DECIDE (NEW, local) | rep spread: Threshold 0.060 / ValProp 0.059 (matters) vs Outlier 0.007 / Hetero 0.015 (irrelevant); best rep is noise-type-dependent (mol2vec graded, PDV hetero); corroborated on ADME. This should REPLACE the flat rep claims. |
| T7 | Rework rep claims L462/L493/L567 | DECIDE | reframe to T6 shape (no universal most/least robust). |
| T8 | Cross-strategy averaging audit (the 9-row table) | DECIDE | Model table+W = KEEP (all 6 shown); rep claims + validation table + ICC = collapse strategies w/ nothing shown → fix per T6. |
| T9 | Kendall W = 0.92 → **0.9121**; SVM not top-2 (5th) | LOGGED | L433, L573. |
| T10 | Stale "NDS" in 11 lines | PARTIAL | rename-only: L387,L464,L556. rename+recompute(SERVER): L470/L475 Wilcoxon Δ, L495 top-10. content-rework: L462/L493/L567/L573. |
| T11 | Use validation (KIRBy) data everywhere, not just QM9 | ONGOING | corroborates T6; validation has baseline+auc_norm locally. |
| T12 | Duplicated leftover sentence L381 | LOGGED | delete (restates L380, contradicts its nuance). |
| T13 | NGBoost/SVM "robust despite mediocre clean-data" (L460) | DONE | ✅ SUPPORTED: NGBoost lowest baseline (0.671) yet most robust (0.851, Gaussian); no model-level baseline→robustness trend. KEEP L460. |
| T14 | **valprop auc_norm corrupted in direct dump** | KNOWN | `calculate_robustness()` called directly does NOT apply the catastrophic-iteration filter → valprop shows garbage (−1.7e29). Use main-pipeline valprop (~0.66) or add the `filtered_catastrophic_iterations` filter. Any valprop number needs this guard. |
| T15 | PDV has best clean-data accuracy (L567 first half) | DONE | ✅ continuous_pdv baseline **0.857**, highest of all reps. Supports "PDVs produce strongest clean-data performance." |
| T25 | **$\sigma$ anchored to real experimental error** | LOGGED (needs D11) | New section "Noise magnitude — anchoring $\sigma$…". Published assay error is 0.27–0.74 log units across all three external endpoints (Kramer 2012 $pK_i$ 0.54; Bruneau 2006 logD 0.27; Hayeshi 2008 Caco-2 ~0.43) ⇒ **$\sigma \approx 0.6$ = one unit of real measurement error**. Units check out: injection is in raw label units and all three labels are logs. Needs 5 new `citations.bib` entries. **QM9 excluded** — computed data, no experimental error to match. |
| T24 | **ECE removed from the paper entirely** | LOGGED | Author decision 2026-08-19. Scripts already stripped (`generate_paper_figures_v2.py`, `generate_paper_figures.py`, `deep_analysis.py`). Paper edits listed in the "Metric removal — ECE" section below. Coverage at 1σ/2σ carries the calibration argument. |

**Server dumps pending (blocks T4, T10-partial, T13):** run the two blocks in the "Re-run on ARC" section → `qm9_auc_with_baseline.csv` (per model×rep×strategy auc_norm + baseline_r2).

---

# The big picture: how the two changes reshape the whole argument

This section is the intellectual spine of the revision. Read it before touching any individual section, because the two changes are not local edits — they propagate from the abstract to the conclusion and, in two places, *reverse* a headline claim. Everything below cites the verified numbers you should carry into the prose.

### The two changes, stated once

- **Change 1 (metric).** Every robustness number in the paper is currently a **Noise Degradation Slope (NDS)** — the straight-line slope $dR^2/d\sigma$, where "more negative = worse." Replace all of them with **AUC$_\text{norm}$**: the trapezoidal area under the *retention* curve $R^2(\sigma)/R^2(0)$ over $\sigma\in\{0,\dots,1.0\}$, roughly on $[0,1]$, where **higher = more robust**. AUC$_\text{norm}$ makes no linearity assumption and — critically — is **not coupled to baseline accuracy** (it divides by $R^2(0)$). There is no prior NDS publication and no "old paper": you are simply defining and using AUC$_\text{norm}$. Never write "we changed the metric," "unlike NDS," or anything justifying the swap. Weibull is gone; do not mention it.

- **Change 2 (uncertainty within-$\sigma$) — now a NULL result.** The paper currently computes the uncertainty–noise Spearman $\rho$ by **pooling all $\sigma$ levels together**. Pooling secretly measures the *population* trend (mean uncertainty rises as $\sigma$ rises) and mislabels it as *per-sample* detection. Worse, the pipeline's `fix_injected_noise` also **pooled all six noise strategies** into one regression (it omitted `strategy` from the group key), manufacturing a fake per-sample correlation. Recomputed correctly — **within each $\sigma$ level, one strategy at a time** — the finding **collapses to a null**: no model detects per-sample noise. Across all 143 model×representation×strategy combinations the largest within-$\sigma$ |ρ| is **0.129**, nothing above 0.15. The published ρ=0.485 "detector" was the pooling artifact. **The only surviving uncertainty result is the population-level trend** (mean uncertainty rises with σ — the Kolmar link). In the paper: abstract → one null sentence; Scientific Contribution → per-sample claim removed entirely.

### (a) The paper's spine — and how each change *sharpens* it

The paper has always argued a **two-track thesis**:

1. **Noise robustness is a MODEL property, not a representation property.** ANOVA attributes most robustness variance to model architecture; representation contributes <10% for most noise types; rankings are stable across noise strategies (Kendall's $W$).
2. **Uncertainty and noise are linked only at the population level.** As more label noise is injected, mean predicted uncertainty rises (the Kolmar link). This holds broadly across models. What does **not** hold — despite what the current paper claims — is *per-sample* detection: no model's uncertainty flags *which individual labels* were corrupted (null result, see Change 2).

Change 1 *reinforces* Track 1. Change 2 **overturns** the paper's old per-sample claim and reduces Track 2 to the population-level link plus an honest null — this is a structural change to the paper the author is working through with supervisors.

- **Change 1 sharpens Track 1's independence claim.** Under NDS, the paper had to keep *apologizing* for a confound: high-baseline representations "have more to lose," so PDV degraded fastest despite being the best representation, and low-baseline mol2vec looked "shallow" for the wrong reason (paper L471, L502). That whole caveat was a **pure artifact of an accuracy-coupled metric**. AUC$_\text{norm}$ normalizes by $R^2(0)$, so the artifact disappears and you can state cleanly that robustness is decoupled from baseline accuracy — **confirmed on QM9 2026-08-15** (within-Gaussian Pearson(baseline, auc_norm) = +0.046 ≈ 0; NGBoost lowest baseline yet most robust). The decoupling argument (L433, L460) is genuine and **stays**. *(Nuance for the validation section: the QM9 decoupling number (+0.046) is from the server run and NOT locally reproducible (baseline CSV absent) — call it provisional. On ADME the relation is dataset-specific and sign-flips — hERG +0.30, LogD −0.31, Caco-2 ~0 — so do NOT say "ADME re-couples"; say accuracy↔robustness coupling varies by dataset and can be positive or negative.)* The one cost: the ANOVA now shows **outlier and heteroscedastic noise are residual-dominated**, so "model always dominates" softens to "model dominates for four of six noise types" (see below). *(✅ Confirmed on ARC 2026-08-14: the residual-dominance of those two strategies is REAL and robust — not a roster/VBLL artifact. The roster is a consistent 7 models across all six strategies. Note the residual there reflects run-to-run variance under stochastic noise, not models being interchangeable — see the Variance-decomposition banner.)*

- **Change 2 overturns Track 2's per-sample claim.** The paper said uncertainty detection was representation-gated (embeddings fail, fingerprints work). The corrected within-$\sigma$, per-strategy recomputation shows the "fingerprints work" half was also an artifact: **no representation and no model produces per-sample detection** (max |ρ|=0.129). What remains true is only the **population-level** statement — mean uncertainty rises with σ. So Track 2 shrinks from "a conditional per-sample detector" to "a population-level trend, with per-sample detection a null." How much of the paper's uncertainty section survives, and in what form, is the author's supervisor conversation.

The old "deepest synthesis" (robustness and per-sample detection are orthogonal capabilities) **no longer applies** — there is no per-sample detection track for robustness to be orthogonal to. The honest synthesis is narrower: robustness is a model property; uncertainty rises with noise only in aggregate; and per-sample noise cannot be recovered from uncertainty by any model tested.

### (b) Every headline claim, triaged: BREAKS / STRENGTHENS / REFRAME

#### Abstract (L164–169)

- **"Noise robustness was measured as the slope of predictive performance degradation across increasing amounts of additional label noise."** → **REFRAME.** Replace with the AUC$_\text{norm}$ definition (normalized area under the retention curve; higher = more robust).
- **"NGBoost and SVMs showing the strongest robustness to noise."** → **REFRAME (half breaks).** NGBoost survives as **#1 by mean AUC$_\text{norm}$ (0.824)**. **SVM does not** — it is **5th (0.814)**, in a near-tie with XGBoost/LightGBM/RF, and *leads only under outlier noise* (AUC$_\text{norm}$ 0.956) and on the ADME data. New honest phrasing: "NGBoost was most robust overall, with the tree ensembles (RF, LightGBM, XGBoost) and SVM clustered just behind."
- **"applying Bayesian transformations to feed-forward neural networks improves their noise robustness."** → **REFRAME (mostly holds, one exception).** By mean AUC$_\text{norm}$: BNN improves *both* NN families (BNN-$\alpha$ 0.801 > NN-$\alpha$ 0.789; BNN-$\beta$ 0.802 > NN-$\beta$ 0.756). VBLL improves the $\beta$ family (VBLL-$\beta$ 0.792 > NN-$\beta$ 0.756) but **NOT the $\alpha$ family** (VBLL-$\alpha$ 0.781 < NN-$\alpha$ 0.789). So "both transformations improve both networks" is now FALSE. Say "Bayesian transformations generally improve NN robustness (full-BNN reliably; VBLL for the $\beta$ architecture)."
- **"NGBoost and Gaussian Processes displayed the strongest correlations between per-sample estimated uncertainty and injected noise."** → **BREAKS OUTRIGHT → NULL.** Per-sample tracking does not exist for any model (largest within-σ |ρ| across all 143 combinations = 0.129; the published ρ=0.485 was a strategy-pooling artifact — see the DECISION banner in the Abstract section). Abstract fix: replace with a **one-line null** — per-sample uncertainty did not reliably identify which labels were corrupted, in any model or representation.

#### Scientific Contribution (L167–169)

- **"per-sample uncertainty estimates track injected label noise … only for models with an explicit aleatoric noise term … whereas models whose uncertainty is purely epistemic … do not."** → **DELETE.** The whole per-sample sentence is removed from the Scientific Contribution (author's decision, 2026-08-14). No detector, no triple gate, no aleatoric/epistemic frame — the finding is null and the contribution reduces to the framework + the model-drives-robustness result.

#### Methods — Performance Metrics (L234–322)

- **NDS definition block (L254–260)** → **REFRAME.** Replace the $dR^2/d\sigma$ equation with the AUC$_\text{norm}$ definition. Note the gate change: baseline $R^2<0.3$ now excludes **48** configs (was $<0.6$ excluding 66). Drop "positive slopes … would indicate noise improves performance" (slope language is gone).
- **Metrics summary table, NDS row (L291–293)** → **REFRAME** to the AUC$_\text{norm}$ row (higher = more robust, ~[0,1]).
- **ANOVA outcome text (L262, "either $R^2$ … or NDS")** → **REFRAME** to "$R^2$ … or AUC$_\text{norm}$."
- **Uncertainty $\rho$ definition (L240)** → **REFRAME / EXPAND.** State explicitly that the uncertainty–noise $\rho$ is computed **within each $\sigma$ level** (and one noise strategy at a time) to isolate per-sample detection from the population trend. This is the correct method and what lets the paper report the per-sample result honestly — which is a **null** (no model resolves individual corrupted labels). Keep the sentence; the *result* it feeds is null, not a "new finding."

#### Results 4.1 — Variance decomposition (L389–439)

- **Performance ANOVA (interaction dominates, then model, then rep)** → **STRENGTHENS/UNCHANGED.** Performance roster is still 11 models; interaction is still the top term. Numbers barely move.
- **"for noise robustness … model architecture is instead the largest source of variance" (L391)** → **REFRAME with a real qualification.** True for Gaussian (Model 43.8%), Quantile (36.8%), Threshold (54.7%), Value-Prop (52.5%). **FALSE for Outlier (Model 10.3%, Residual 83.6%) and Heteroscedastic (Model 14.0%, Residual 77.4%)** — both are now **residual-dominated**. Rewrite as: "model architecture is the dominant *explained* factor for four of six strategies; under outlier and heteroscedastic noise, differences between models shrink into the residual." Also state the **robustness ANOVA roster is 7 models** (the four Bayesian/VBLL NN variants drop because they don't train on the mol2vec/MHG-GNN embeddings), distinct from the 11-model performance roster — the current text never says this and it matters for interpreting the $\eta^2$.
- **Table 2 / Fig 2 ($\eta^2$ values, L406–411, L419)** → **REPLACE all robustness columns** with the AUC$_\text{norm}$ values (Gaussian 43.8/5.2/16.9/34.2; Quantile 36.8/4.4/15.1/43.7; Threshold 54.7/7.9/22.6/14.8; Value-Prop 52.5/6.0/19.9/21.6; Hetero 14.0/0.7/8.0/77.4; Outlier 10.3/0.2/5.9/83.6). Caption must drop "noise degradation slope (NDS)" → AUC$_\text{norm}$.
- **"threshold and value-proportional … degraded the most … outlier noise only affects statistical outliers … model and interaction effects were both smaller" (L393)** → **STRENGTHENS.** This paragraph's *logic* now fits the numbers even better: threshold/value-prop are model-dominated (54.7/52.5%), and outlier collapses into residual (83.6%). Just re-anchor to AUC$_\text{norm}$ $\eta^2$ and add heteroscedastic to the "small model effect" group.
- **Interaction figure text — "SVM and full BNNs maintain consistent … NN-$\beta$ display greater variation" and "$\rho=0.73$ … ECFP4 vs PDV" (L423)** → **REFRAME.** Concept holds; recompute the cross-representation Spearman on AUC$_\text{norm}$ and update the figure/heatmap (it currently shows NDS with an accuracy-coupled scale). Rename "ECFP4" per the standing topological-fingerprint convention if you are also applying that rename.
- **"excluding 66 configurations where … baseline R$^2<0.6$" (L432)** → **REFRAME** to **48 configs at $R^2<0.3$** (matches the new gate).

#### Results 4.2 — Robustness across noise strategies (L441–504)

- **"NGBoost and SVM showed the smallest degradation … NN-$\beta$ the steepest" (L443)** → **REFRAME.** NGBoost stays #1 (0.824); **NN-$\beta$ stays worst** (mlp, 0.756) — that endpoint survives. But SVM drops to 5th; recast the top as "NGBoost, then a tight tree-ensemble cluster (RF 0.818, LightGBM 0.817, XGBoost 0.814) with SVM (0.814)."
- **Kendall's $W$ (L443, L577)** → **UPDATE** 0.92 → **0.9121** ($p=3.55\times10^{-8}$, 11 models, 6 strategies). Claim (rankings strategy-independent) **STRENGTHENS** — still well above 0.7.
- **"NDS clusters near $-0.38$ regardless of baseline R$^2$" (L443)** → **BREAKS as an NDS statement — delete the "$-0.38$ cluster" number.** But the follow-on decoupling point (paper L433) is **correct on QM9** (within-Gaussian corr +0.046 ≈ 0; NGBoost low-baseline/high-robust) — keep it, just without the "$-0.38$" figure. Add the ADME nuance in the validation section — coupling is dataset-specific and sign-flips (hERG +0.30, LogD −0.31, Caco-2 ~0), NOT a uniform "+0.525".
- **Table 3 (`tab:nds_ranking`, L445–467)** → **FULL REPLACE.** Rebuild from `table2_auc_by_strategy_pdv.csv`: higher = more robust; new row order NGBoost > RF > LightGBM > XGBoost > SVM > BNN-$\beta$ > BNN-$\alpha$ > VBLL-$\beta$ > NN-$\alpha$ > VBLL-$\alpha$ > NN-$\beta$. Per-strategy bold winners change (e.g., outlier winner is SVM at 0.956).
- **"VBLL-$\alpha$ … outperforming all tree-based models except NGBoost under threshold and value-prop" (L469)** → **BREAKS.** Under AUC$_\text{norm}$, VBLL-$\alpha$ is 10th overall and below all four tree ensembles on threshold and value-prop. Delete this claim.
- **"NGBoost and SVM … did not perform particularly well on clean data … decoupling" (L469)** → **STRENGTHENS** (see spine). Keep the message; you may keep SVM as an *example* of decoupling since it is genuinely robust-ish while not top on clean data, but do not call it a top-2 robustness model.
- **PDV "high baseline → more to lose → steepest slopes / mol2vec shallower" (L471 AND L502)** → **DELETE at both locations.** This is the central NDS artifact. Under AUC$_\text{norm}$ it is simply untrue and unnecessary. Replace with the clean statement: PDV gives the best clean-data accuracy *and* competitive normalized robustness under strongly-regularized models (SVM, full BNN); representation contributes little to robustness variance.
- **"representation explains <11% of NDS variance … VBLL representation-dependent … 24 VBLL×{MHG-GNN,mol2vec} excluded at R$^2<0.6$" (L473)** → **REFRAME.** Recompute the <11% on AUC$_\text{norm}$; exclusion threshold is now $R^2<0.3$ (re-verify the count of excluded VBLL×embedding configs against the new gate before quoting "24").
- **Bayesian-transformation improvement + Table 4 (`tab:wilcoxon_bnn`, L475–493)** → **REFRAME + RECOMPUTE.** The Wilcoxon $\Delta$ is currently "$\Delta$NDS." Recompute all five rows as $\Delta$AUC$_\text{norm}$ (sign flips to "positive = more robust" naturally). Headline nuance: full-BNN improves both families; **VBLL improves $\beta$ but not $\alpha$** (mean AUC$_\text{norm}$ VBLL-$\alpha$ 0.781 < NN-$\alpha$ 0.789) — reconcile the per-strategy Wilcoxon result with the mean ranking and state the exception explicitly. **QRF < RF stays** (mean AUC$_\text{norm}$ RF 0.818 > QRF; direction preserved) — that claim STRENGTHENS.
- **"Tree ensembles dominate the top 10 … NGBoost appears five times … no NN in top 10 … SVM ranks higher on ADME" (L504)** → **RE-DERIVE** the top-10 on AUC$_\text{norm}$; qualitatively likely to hold (tree-dominated), but the exact counts must be recomputed. The "SVM higher on ADME" tail is supported (validation table: svm/sns 0.888, svm/mhggnn 0.879 are the two best ADME configs).

#### Results 4.3 — Uncertainty estimation under label noise (L506–551)

**Change 2 lands here as a NULL result. The whole subsection collapses to: population-level link holds, per-sample detection fails.** The detailed rewrite of this subsection is in the placeholder in the "Results — Uncertainty estimation under label noise" section below, and is **pending the paper restructure**. The triage of the individual spans:

- **"GP and NGBoost … showed the strongest correlations … embeddings near-zero for all models" (L508)** → **BREAKS → NULL.** Both halves of the old claim were pooling artifacts. There is no leader; no model detects per-sample noise (max within-σ |ρ|=0.129).
- **Table 5 (`tab:top_unc_noise`, L510–538)** → **DELETE or replace with a null/population-level table.** The pooled table (GP/SNS 0.56) is an artifact; there is no per-sample within-σ table to replace it with (all cells ~0). Final form pending restructure. **Also: this table contains a stray `PDV (binary)` row (paper.tex L514) — remove it. Binary PDV was a mistake, was never a studied representation, and underperforms continuous PDV (local check: lower clean-data R² on hERG and LogD). Decision 2026-08-14: binary PDV dropped everywhere.**
- **Mechanistic paragraph (L547)** → **REWRITE to explain the null.** The GP's global $\sigma_n^2$ cannot be per-sample, so its within-σ ρ≈0; the same is now true of every model. Keep the mechanism sentences as an explanation of *why per-sample detection fails*, not why anything "wins."
- **Fig 6 (`fig:uncertainty_combined`, L540–545)** → **KEEP, RE-CAPTION as population-level.** This is the **surviving** result: mean uncertainty rises with σ (Kolmar link). Label it population-level so it is not read as per-sample detection. (The safe caption fix is in the placeholder section.) **Do NOT add** the old `within_sigma_uncertainty.png` per-sample figure — it showed the artifact.
- **Kolmar extension (L551)** → **REFRAME to population-only.** The population link holds broadly; the individual-sample link does **not** hold for any model. State the null plainly; do not claim a conditional per-sample link.

#### Results 4.4 — Validation on experimental datasets (L560–571)

- **"model architecture dominates robustness variance on all three datasets … trends generalize" (L562)** → **REFRAME to AUC$_\text{norm}$**, re-verify against Additional file 10; likely holds.
- **"NGBoost ranks first under Gaussian … SVM marginally more robust overall (pooled mean NDS −0.16 vs −0.19), leading on hERG/Caco-2, NGBoost on LogD" (L562)** → **RECOMPUTE on AUC$_\text{norm}$.** The direction is supported: `table_validation_auc.csv` shows SVM best overall on ADME (svm/sns mean 0.888; svm/mhggnn 0.879) with NGBoost close (ngboost/continuous_pdv 0.877; strongest on LogD ~0.985). Replace the "−0.16 vs −0.19" NDS numbers with the AUC$_\text{norm}$ means.
- **"XGBoost suffers the most" (L562) / Fig 8b "ensemble-dependent models (XGBoost) degrade" (L567)** → **STRENGTHENS.** XGBoost is unambiguously worst on ADME by AUC$_\text{norm}$ (0.563/0.537/0.484/0.477 means; collapses on Caco-2 to 0.05–0.21). Keep and quantify with AUC$_\text{norm}$.
- **"QRF consistently less robust than RF on every external dataset" (L571)** → **STRENGTHENS.** Validation AUC$_\text{norm}$ confirms RF > QRF on all three (e.g., continuous_pdv: RF 0.777 > QRF 0.683). Keep.
- **Figs 7 & 8 (`fig:validation_overview`, `fig:validation_combined`)** → **REGENERATE** on AUC$_\text{norm}$ (heatmaps currently NDS; note the black-cell threshold text says $R^2<0.3$ already — consistent with the new gate).

#### Conclusion (L573–581)

- **L575 "noise degradation slope (NDS), defined as the slope of $R^2$ … model architecture is the dominant factor, while representation explains less than 10%."** → **REFRAME.** AUC$_\text{norm}$ definition; qualify "dominant factor" to "for four of six noise types" (outlier/hetero residual-dominated). "Representation <10%" **STRENGTHENS** (AUC$_\text{norm}$ rep $\eta^2$ is 0.2–7.9%).
- **L577 "outlier noise barely separates them"** → **STRENGTHENS** (now literally residual-dominated, 83.6%). **"Kendall's $W=0.92$"** → 0.9121. **"NGBoost and SVM, the most noise-robust"** → REFRAME (NGBoost + tree cluster; SVM mid-pack). **"embeddings' more decisive weakness was uncertainty, not robustness slopes"** → KEEP, now precise: within-$\sigma$ they are ~0 even under outlier.
- **L579 "NGBoost and Gaussian Processes … strongest per-sample correlation."** → **BREAKS → NULL.** Delete the per-sample detection claim; state the null (no model tracks per-sample noise) and keep only the population-level uncertainty rise. Final wording pending restructure.
- **L581 (closing synthesis): "NGBoost and GPs … most robust … and often produce uncertainty estimates which track per-sample label noise."** → **BREAKS.** Delete the per-sample tracking half entirely. Robustness stands (NGBoost/tree ensembles); the uncertainty half reduces to the population-level trend plus a null on per-sample detection. Do not write an "orthogonality" thesis — there is no detection track. Final wording pending restructure.

#### Abbreviations / metric plumbing

- **Abbreviations list (L602)** — drop "NDS: Noise degradation slope," add the AUC$_\text{norm}$ definition if you abbreviate it.
- **Availability of data (L624) and NoiseInject framework text (L380)** — both list "noise-performance degradation slope / retention." Update to AUC$_\text{norm}$ (the retention curve it integrates is already the object described, so this is a light touch).
- **Additional-file captions (L664–673)** — files 2, 3, 4, 6, 7, 8, 11 all say "NDS" and must be re-labeled AUC$_\text{norm}$; several supplements must be regenerated.

### (c) The new narrative through-line (abstract → conclusion)

Carry this single arc, verbatim in spirit, from the abstract to the last sentence:

> **Noise robustness is primarily a property of the model, largely independent of the molecular representation and of clean-data accuracy** (rankings are strategy-stable at $W=0.9121$; NGBoost leads; decoupling from baseline confirmed on QM9, within-Gaussian corr +0.046). **Nuance:** QM9 decoupling is provisional (server-only number +0.046, not locally reproducible); on ADME the accuracy↔robustness relation is dataset-specific and sign-flips (hERG +0.30, LogD −0.31, Caco-2 ~0) — belongs in the validation discussion. **At the population level, mean predicted uncertainty rises with injected noise** (the Kolmar link — it holds broadly). **But per-sample uncertainty does not identify which individual labels were corrupted for any model or representation** (largest within-σ ρ = 0.129; a null result). So the population-level signal and per-sample detection must be kept strictly distinct, and the paper makes no per-sample detection claim.

The rhetorical shift from the current paper: drop "explicit-aleatoric models track noise, epistemic ones don't" (backwards), drop every "high baseline has more to lose" apology (metric artifact), and drop the per-sample "detector" story entirely (it was a pooling artifact). Replace with the cleaner **"model-driven robustness; population-level uncertainty rises with noise but per-sample detection fails"** frame.

> **NOTE:** The exact shape of the uncertainty narrative below (Results 4.3, the conclusion's detection sentences, main-points rows 10–17) is pending the paper restructure and is **not yet rewritten in this guide** — the through-line above is the settled direction; the section-level detail still shows the old (superseded) detector guidance.

### (d) Main points → where each lands

| # | Headline point | Fate | Key numbers to use | Where in paper.tex |
|---|----------------|------|--------------------|--------------------|
| 1 | Robustness is model-driven, not representation-driven | **STRENGTHENS** (qualify: 4 of 6 strategies) | Rep $\eta^2$ 0.2–7.9%; Model 43.8/36.8/54.7/52.5% (Gauss/Quant/Thresh/ValP); Outlier 10.3 & Hetero 14.0 residual-dominated (83.6/77.4) | L391–393, Table 2 (L406), Fig 2 (L419), L575 |
| 2 | Robustness vs clean-data accuracy (coupling) | **QM9 decoupled (server-only); ADME sign-flips** | QM9: server `.out` gave within-Gaussian +0.046 ≈ 0 (decoupled), but the baseline CSV isn't local so it's **not independently reproducible** — mark provisional. NGBoost lowest baseline yet most robust supports it. ADME is **NOT uniformly coupled** (my earlier "+0.525" was a pooled artifact): per dataset the sign flips — hERG **+0.30**, LogD **−0.31** (anti-coupled), Caco-2 **+0.10** (~none). Frame as: decoupled on clean QM9; on experimental data the accuracy↔robustness relation is dataset-specific and can go either way. | L433, L460, validation section |
| 3 | NGBoost most robust | **HOLDS** | NGBoost 0.824 (mean AUC$_\text{norm}$, #1 of 11) | Abstract L164, L443, L577 |
| 4 | SVM one of the two most robust | **BREAKS → REFRAME** | SVM 5th (0.814); leads only outlier (0.956) + ADME | Abstract L164, L443, L577, L581 |
| 5 | NN-$\beta$ (mlp) least robust | **HOLDS** | mlp 0.756 (last of 11) | L443, Table 3 |
| 6 | Bayesian transforms improve NN robustness | **REFRAME (1 exception)** | BNN both (0.801>0.789; 0.802>0.756); VBLL-$\beta$ yes (0.792>0.756), **VBLL-$\alpha$ no (0.781<0.789)** | Abstract L164, L475, Table 4 (L486) |
| 7 | QRF less robust than RF | **STRENGTHENS** | RF 0.818 > QRF; holds on all 3 ADME | L475, L490, L571 |
| 8 | Rankings stable across noise strategies | **STRENGTHENS** | Kendall's $W=0.9121$, $p=3.55\times10^{-8}$ | L443, L577 |
| 9 | Outlier noise barely separates models | **STRENGTHENS** | Outlier robustness residual $\eta^2=83.6\%$ | L393, L469, L577 |
| 10 | GP & NGBoost strongest per-sample uncertainty trackers | **BREAKS → NULL** | No per-sample detection anywhere; max within-σ |ρ|=0.129 across 143 combos | Abstract L164, SC L167, Table 5 (L519), L547, L579, L581 |
| 11 | Explicit-aleatoric term ⇒ tracks noise; epistemic ⇒ doesn't | **BREAKS → NULL** — delete frame | Neither aleatoric nor epistemic models detect per-sample noise | Scientific Contribution L167–169 |
| 12 | Embeddings (mol2vec/MHG-GNN) can't support uncertainty detection | **MOOT (subsumed by null)** | No representation supports per-sample detection, embeddings included | L508, L549, L577 |
| 13 | Population link: mean uncertainty rises with noise (Kolmar) | **HOLDS** — recaption as population-level | Fig 6 survives; this is the surviving uncertainty result | L540–545, L551 |
| 14 | Per-sample link is conditional on model×representation | **BREAKS → NULL** | No conditions produce per-sample detection | L549, L551, SC |
| 15 | XGBoost collapses on external data | **STRENGTHENS** | ADME AUC$_\text{norm}$ 0.48–0.56; Caco-2 0.05–0.21 | L562, Fig 8 (L567) |
| 16 | Trends generalize to ADME; model dominates there too | **HOLDS** (re-verify AF10 on AUC$_\text{norm}$) | SVM/NGBoost best ADME configs 0.86–0.99 | L562, L571 |
| 17 | Robustness & per-sample detection are orthogonal | **DROP** — there is no detection track to be orthogonal to | Only surviving split: robustness (per-model) vs population-level uncertainty rise | Conclusion L581, Discussion |
| 18 | Config exclusion gate | **REFRAME** | $R^2<0.3$ → 48 excluded (was <0.6 → 66) | L260, L432 |

**Per-sample uncertainty is a NULL result everywhere.** Rows 10–12, 14, 17 all collapse into one fact: no model's per-sample uncertainty tracks which individual labels were corrupted, under any representation or noise type (max within-σ |ρ| = 0.129). The only surviving uncertainty result is the **population-level** Kolmar link (row 13). The section-level rewrite of Results 4.3 / conclusion to reflect this is pending the paper restructure.

---

---

# How to use this guide

This walks the manuscript **section by section, in paper order**. For each section you get the exact lines to change — I only propose a change where your text is **wrong**: the robustness metric (AUC$_\text{norm}$, not a slope), the uncertainty finding (per-sample detection is a **null** — measured correctly within each σ and per strategy, no model tracks per-sample noise; only the population-level uncertainty rise survives), or a number the data contradict. Everything else is your prose, kept byte-for-byte. You make every edit; nothing here touches your files.

Every number below was read directly from the regenerated CSVs in `results/paper_figures_v2/` and `results/paper_figures/`; every `\cite` was checked against `citations.bib`. Each section closes with what was **kept / removed / replaced** and a **number + citation verification** line.

**Two things have to happen:**
1. **You re-run the figures on ARC** — one item still needs certifying from the server: the **validation ANOVA** (saturated one-obs-per-cell design; needs per-fold replicates). The robustness ANOVA rows were confirmed on ARC 2026-08-14 (residual-dominance is real, roster is a consistent 7 models); the per-sample "detector" is a code-fix null. Remaining open items and commands are in the "Re-run on ARC" section at the end.
2. **You edit the paper from this guide.**

**Fix first (citation audit — details at the end):** `\bibliography{sn-bibliography}` points at a file that doesn't exist → every citation renders `[?]`. And `Rogers2010` (ECFP) and `Islam2019` are cited but missing from `citations.bib`.

# Writing principles (applied per section, not just stated here)

Drawn from the nine Nature Machine Intelligence papers. Each section names the exemplar it follows and the concrete move it borrows — applied in the replacement text, never used to restyle a sentence that is already correct.

- **Numbers never appear in the Abstract or Scientific Contribution** — only in Results, tables, and captions.
- **A number rides behind a plain adjective, in parentheses, with full statistics** — "significantly more robust (ΔAUC$_\text{norm} = +0.053$, $p = 2.9\times10^{-11}$)", never a bare number, never "significant" without a test.
- **Results subheadings state the finding as a claim**, mechanism as grammatical subject.
- **Reversals are stated plainly and owned**; narrow results carry an explicit scope caveat.
- **The metric is simply defined and used** — no "unlike NDS", no "we changed the metric", no Weibull. NDS appears nowhere in the final manuscript.

Model-name map used in every table: NN-α=`dnn`, NN-β=`mlp`, BNN-α=`dnn_bnn_full`, BNN-β=`mlp_bnn_full`, VBLL-α=`dnn_vbll`, VBLL-β=`mlp_vbll`.

---

# The paper, section by section

---

## Abstract (the `\abstract{}` block, paper.tex L164–169 — abstract prose only; Scientific Contribution is handled in its own section below)

**Argument now:** The abstract must (i) describe robustness as retained performance under rising noise, not a slope (in plain prose — no symbol or acronym; the metric is named/defined in Methods); (ii) name the genuinely most-robust models (NGBoost + a tree ensemble, not SVM); and (iii) replace the per-sample uncertainty claim with a single honest null sentence — per-sample uncertainty did not reliably flag which individual labels were corrupted, in any model or representation. All with zero numbers, matching the current numberless abstract.

> **DECISION (2026-08-14): per-sample uncertainty tracking is a NULL result.** The published ρ=0.485 "detector" was a strategy-pooling artifact in `fix_injected_noise` (it omitted `strategy` from the group key, so all six noise types were regressed together and the recovered per-sample noise was contaminated). Re-run correctly — within each σ, one strategy at a time, across all 143 model×representation×strategy combinations — the largest |ρ| anywhere is **0.129**, with nothing above 0.15. There is no per-sample detector: not BNN, not GP, not NGBoost. In the abstract this becomes **one null sentence**; in the Scientific Contribution it is **removed entirely** (author's decision). The detailed uncertainty guidance further down this file — triage §4.3, the narrative through-line, main-points rows 10–17, Results 4.3, and the conclusion's detection sentences — is **SUPERSEDED and awaits the paper restructure; do not edit the paper from it yet.**

---

**Replace (L164, robustness-metric sentence):**
> Noise robustness was measured as the slope of predictive performance degradation across increasing amounts of additional label noise.

**With:**
> Noise robustness was measured as the proportion of predictive performance retained across increasing amounts of additional label noise.

*(Symbol-free by design. The original abstract sentence was already a plain-prose metric description; the only wrong span is "slope of…degradation," since the metric is no longer a slope. Do NOT inject `$R^2$` or the `AUC$_\text{norm}$` acronym here — an abstract for a framework paper doesn't formally define its yardstick metric, and the acronym is defined properly at first use in Methods. "retained" carries the direction, higher = more robust.)*

---

**Replace (L164, most-robust-models sentence):**
> We found that model architecture was the dominant factor in performance degradation, with NGBoost and SVMs showing the strongest robustness to noise.

**With:**
> We found that model architecture was the dominant factor in performance degradation, with NGBoost and random forests showing the strongest robustness to noise.

---

**Replace (L164, Bayesian + uncertainty sentence — first clause kept verbatim, only the post-semicolon uncertainty clause is category (b) and changes to a one-line null):**
> We also found that applying Bayesian transformations to feed-forward neural networks improves their noise robustness; and that NGBoost and Gaussian Processes displayed the strongest correlations between per-sample estimated uncertainty and injected noise.

**With:**
> We also found that applying Bayesian transformations to feed-forward neural networks improves their noise robustness, but that per-sample predictive uncertainty did not reliably identify which individual labels had been corrupted, in any model or representation tested.

*(One-line null, symbol-free. The first clause is kept verbatim. The post-semicolon claim of strong per-sample correlations is deleted, not reframed — it was the pooling artifact (see the DECISION banner above). "did not reliably identify which individual labels had been corrupted" is the honest population-vs-per-sample distinction stated in plain words. This is the ONLY place per-sample uncertainty appears in the abstract; the Scientific Contribution says nothing about it.)*

---

**Scientific Contribution** is inside the same `\abstract{}` block (L167–169), but its edit is written out in full in its own **`## Scientific Contribution`** section below — go there. Summary of the decision: **keep the first two sentences verbatim; delete the third (per-sample uncertainty) sentence entirely.** Nothing about uncertainty tracking survives in the Scientific Contribution.

---

**Decisions (folded from the old figures/lit/review material):**
- SYMBOL-FREE in the abstract: the only wrong span in the metric sentence is "slope" (the metric is no longer a slope), so the fix is "slope of…degradation" → "proportion of…retained" and nothing more. The `AUC$_\text{norm}$` acronym and its `$R^2$` definition are NOT introduced here — abstracts for framework papers don't formally define the yardstick metric, and the acronym is defined at first use in Methods (the abstract is self-contained; nothing downstream depends on it).
- REPLACED the full-draft's whole-sentence rewrite of the models claim ("tree ensembles… closely followed by SVM") with the minimal change-list swap "SVMs → random forests" — rule 1 forbids touching correct prose beyond the wrong span; RF is the verified #2.
- UNCERTAINTY → ONE-LINE NULL: the entire per-sample tracking claim is deleted, not reframed. Earlier versions of this guide reframed it (triple gate, BNN-on-PDV detector, ρ=0.485); that reframing was itself built on the pooling artifact and is now void. The abstract carries a single honest null sentence; nothing more.
- REMOVED all abstract numbers per rule 4 and ChatNT precedent — the abstract stays numberless.

**Verification — numbers** (none appear in the replacement prose; these justify the wrong-span swaps only):
- NGBoost mean AUC$_\text{norm}$ = 0.823966 (#1) | table2_auc_by_strategy_pdv.csv | OK
- RF mean = 0.817719 (#2) | table2_auc_by_strategy_pdv.csv | OK
- LGB 0.816752 (#3), XGBoost 0.814395 (#4), SVM 0.813671 (#5) | table2_auc_by_strategy_pdv.csv | OK — SVM is 5th, confirming the SVM→RF correction
- SVM Outlier = 0.955555 (leads outlier column: RF 0.954834, NGBoost 0.952861) | table2_auc_by_strategy_pdv.csv | OK — SVM's win is outlier-only, so it is not an abstract-level "strongest robustness" model
- Per-sample null: largest within-σ |ρ| across all 143 model×rep×strategy combinations = **0.129** (mlp_bnn_full, quantile, sns, σ=0.1; a single low-σ cell, not reproduced elsewhere); nothing > 0.15. Confirms the abstract's null sentence. The old ρ=0.485 was the strategy-pooling artifact — do not carry it into the abstract or anywhere else.

**Verification — citations:** The abstract and Scientific Contribution contain zero `\cite`/`\citep`/`\citet` keys (journal abstracts carry no citations); none to verify. No citation keys added.

---

## Scientific Contribution

**Argument now:** Keep the first two sentences verbatim (both correct and metric-agnostic). **Delete the third sentence entirely** — the per-sample uncertainty claim is a null result (see the DECISION banner in the Abstract section) and is removed from the contribution, not reframed. The corrected contribution is two sentences: the framework, and "model, not representation, drives robustness."

**Sentence 1 — keep verbatim:**
> This study introduces NoiseInject, an open-source benchmarking framework that performs controlled artificial noise injections and provides analysis tools to determine the impact of label noise.

**Sentence 2 — keep verbatim:**
> We demonstrate that the choice of model, rather than molecular representation, is the primary determinant of QSAR-model noise robustness, a conclusion we reached by comparing a model's relative ranking across different types of label noise.

**Sentence 3 — DELETE.** Remove the whole span (it crosses the L167→L169 paragraph break) and close the paragraph:
> ~~We further show that per-sample uncertainty estimates track injected label noise for some certain models, [¶] only for models with an explicit aleatoric noise term such as the observation-noise variance of a Gaussian Process or the predicted scale of NGBoost, whereas models whose uncertainty is purely epistemic, such as Bayesian Neural Networks, do not.~~

**Full corrected block, paste-ready:**

```latex
\textbf{Scientific contribution}. This study introduces NoiseInject, an open-source benchmarking framework that performs controlled artificial noise injections and provides analysis tools to determine the impact of label noise. We demonstrate that the choice of model, rather than molecular representation, is the primary determinant of QSAR-model noise robustness, a conclusion we reached by comparing a model's relative ranking across different types of label noise.
```

No numbers, no citations, no table/figure owned by this section.

---

## Introduction (research questions)

**Argument now:** The three-question closer must stay a clean, number-free enumeration, but Q2 can no longer pre-commit to the discarded pooled correlation. It must frame the uncertainty question as within-σ: does per-sample uncertainty track *which individual labels* were corrupted once the population-level rise in average uncertainty is controlled for. Everything else in the paragraph is correct and stays byte-for-byte.

Replace: `Second, we compare how noise-robust probabilistic models are versus their deterministic counterparts, and whether uncertainty estimates correlate with prediction error or label noise under noisy conditions.`
With:    `Second, we compare how noise-robust probabilistic models are versus their deterministic counterparts, and whether their per-sample uncertainty estimates track which individual labels have been corrupted, once the population-level rise in average uncertainty at each noise level is controlled for.`

All other sentences in the paragraph (Q1 "First, we investigate the contributions of molecular representation and model architecture…"; Q3 "Third, we assess the generalizability of noise-robustness patterns…"; the closer "Together, these questions aim to identify what, if anything, makes a QSAR model robust to noise."; and the "Finally, we introduce NoiseInject…" sentence) contain no NDS/slope language, no reversed finding, and no CSV-contradicted numbers — keep them EXACTLY as written. The `% TODO: edit this` comment on L201 may be deleted.

**Decisions (what I folded from the old figures/lit/review material):**
- REMOVED the old full-draft's inserted findings-preview paragraph (guide L181: "Our central finding is… Kendall's W = 0.91… NGBoost and tree ensembles… BNN on PDV… GP does not resolve…"). It injects results + numbers into the Introduction, which the author's structure keeps in Results; adding it violates "change only wrong spans" and the number-free-preview craft rule. The reversed uncertainty finding lands in Results/Methods, not here.
- KEPT the old change-list's core recommendation (guide L599, L1407) to tighten only Q2's pooled clause to within-σ; adopted the paper-voice replacement (guide L1408) but trimmed it so the author's first clause survives verbatim rather than being reworded to "compare the noise robustness of… against…".
- REMOVED the optional Q3 add-on "including experimentally measured endpoints" (guide L190) — it is foreshadowing, not a C1/C2 correction, so it falls under "do not touch correct claims."
- REMOVED the closer extension "…and whether robustness and noise-awareness are the same property or distinct ones" (guide L179) — an enhancement, not a correction.
- NOTED the W presentation conflict (guide L1351): the intro prints no W in this minimal version, so the "0.91 vs 0.9121" inconsistency does not arise here; the official 0.9121 lives in Results.

**Verification — numbers:** (owned paragraph prints NO numbers; these were checked to justify NOT importing them into the intro)
- Kendall W = 0.9121, p = 3.55e-8, 11 models, 6 strategies | results/paper_figures_v2/table6_kendalls_w.txt | OK (official value; kept out of the RQ prose)
- PDV MEAN AUC_norm ranking NGBoost .824 > RF .818 > LGB .817 … mlp .756 | results/paper_figures_v2/table2_auc_by_strategy_pdv.csv | OK (kept out of the RQ prose)

**Verification — citations:** The owned research-questions paragraph (L202) contains NO \cite/\citet/\citep keys — nothing to verify in-section. (Out-of-section courtesy check on the surrounding intro: all keys IN-BIB except `Rogers2010` MISSING at L193 — flag for whoever owns the background paragraphs; not in this section's span.)

---

I have everything needed. All numbers and citations verified. Writing the consolidated section.

## Metric removal — ECE (complete removal, author decision 2026-08-19)

**What and why.** Expected calibration error is removed from the paper entirely. It is not replaced: empirical coverage at $1\sigma$ and $2\sigma$ is already reported, is on a fixed $[0,1]$ scale with known targets (68\% and 95\%), and carries the same calibration argument in a form a reader can check.

Two grounds, both verified rather than asserted:
1. **The score has a floor that grows with the label scale.** The implementation compares binned mean *predicted uncertainty* (a standard deviation) against binned mean *absolute error*. For a perfectly calibrated Gaussian model those differ by a factor of $\sqrt{2/\pi} \approx 0.798$, so a flawless model does not score 0. Simulated (20{,}000 samples, uncertainties exactly correct by construction): ECE $= 0.198$ at label scale $\times 1$ and $2.031$ at scale $\times 10$. The score is therefore not comparable between models whose uncertainty magnitudes differ, nor across datasets.
2. **It is load-bearing nowhere.** ECE appears in paper.tex only as a definition, a metrics-table row, one column of `tab:top_unc_noise`, an abbreviation, and two Additional-file captions. **No prose sentence in the paper reports or interprets an ECE value.** Removing it costs one table column.

**Paper.tex locations to strike** (author edits; line numbers against the local 30 Jun snapshot — re-anchor to live Overleaf):

| Line(s) | What is there | Action |
|---|---|---|
| L234–238 | "The Expected Calibration Error (ECE) was computed by binning…" + the $\text{ECE} = \sum_b \ldots$ equation + "Lower ECE indicates better-calibrated…" | **Delete the whole block.** The coverage definition immediately around it stays. |
| L300–302 | `tab:metrics_summary` row: `ECE & $\sum_b \frac{|B_b|}{N}|\bar u_b - \bar e_b|$ & Calibration of predicted uncertainty…` | **Delete the row.** The Coverage row above it stays and now carries calibration alone. |
| L368 | Methods/NoiseInject paragraph: "uncertainty-calibration metrics: expected calibration error (ECE), empirical coverage at $1\sigma$ and $2\sigma$, mean prediction-interval width, and…" | **Drop "expected calibration error (ECE), "** — list starts at "empirical coverage". |
| L503–528 | `tab:top_unc_noise`: the **ECE** column header, its caption clause "\textbf{ECE} = expected calibration error (lower = better-calibrated);", and the ECE value in all 14 body rows | **Delete the column** (7 columns → 6; `\begin{tabular}{llrrrrr}` → `{llrrrr}`) and the caption clause. Regenerate the body from `table4c_top_unc_noise_correlations.csv` after the next run — the script no longer emits the column. |
| L587 | `\item[ECE:] Expected calibration error` in Abbreviations | **Delete the line.** |
| L620 | Availability paragraph: "uncertainty-calibration metrics (ECE, coverage at $1\sigma$/$2\sigma$, mean interval width, and…)" | **Drop "ECE, "**. |
| L667 | Additional file 9 caption: "Unc-Noise $\rho$, Unc-Error $\rho$, ECE, and coverage at $1\sigma$/$2\sigma$" | **Drop "ECE, "**. |

Nothing else in L234–322 changes; the coverage equation block and the $\sigma$/ANOVA/ICC material stay verbatim.

**Scripts — DONE 2026-08-19, full deletion (not commented out):**
- `scripts/generate_paper_figures_v2.py` — both ECE computation blocks removed (QM9 per-model and per-strategy×rep), `'ECE'` dropped from all three output row dicts, dropped from the `table4c_*` column list, and the validation-uncertainty ECE block removed. Zero occurrences remain.
- `scripts/generate_paper_figures.py` — same removals (this is the script `slurm_scripts_analysis/run_figures.sh` actually invokes, so leaving it would have re-emitted ECE on the next ARC run). Zero occurrences remain.
- `scripts/deep_analysis.py` — ECE dropped from the column rename map, the summary column list, and two print strings; verified it still runs end-to-end and its `deep_qm9_uncertainty_*.csv` outputs no longer carry the column.
- Consequence: `table4_uncertainty_metrics*.csv`, `table4_supp_uncertainty_by_strategy_rep.csv`, `table4c_top/bottom_*.csv` and `table_validation_uncertainty.csv` lose the ECE column at the next run. Existing local CSVs still have it until regenerated.
- **Not touched** (older exploratory scripts, no paper artefacts): `scripts/uncertainty_analysis.py` (emits `fig6c_ece.png`), `scripts/phase2_analysis.py`, `scripts/generate_figures.py`. Flag if these should be stripped too.

**Knock-on:** the VBLL "ECE ≈ 6–24" numbers disappear with the metric, but the underlying problem does not — coverage for VBLL on MHG-GNN and mol2vec is 0.27–0.45 at $1\sigma$ and 0.39–0.58 at $2\sigma$ against targets of 0.68/0.95. Low coverage despite very wide intervals means the errors exceed even the inflated uncertainty, so T20's "resolved — scale artifact" is incomplete. Tracked in DISCUSSION_TRACKER.md; needs an ARC check of the raw uncertainty CSVs.

---

## Noise magnitude — anchoring $\sigma$ to real experimental error (added 2026-08-20)

**The idea.** The paper sweeps $\sigma$ from 0 to 1 and never says what those numbers mean physically. A reviewer can reasonably ask "why 0.5? why 1.0? is any of this the amount of noise real data actually has?" There is a published answer, and it is a strong one: **for the endpoints in this paper, $\sigma \approx 0.6$ is approximately one unit of real experimental measurement error.** That converts an arbitrary knob into a claim worth making.

**Why the comparison is legitimate — the units line up.** Noise is injected as $y_\text{noisy} = y + \sigma \cdot \mathcal{N}(0,1)$ in the label's own **raw units** (`NoiseInject/noiseInject/core.py:68`; only $X$ is standardised, never $y$). And all three external labels are on a log scale: logD natively, hERG as $pK_i$, and Caco-2 efflux via `log_transform=True` at `KIRBy/tests/alternative_data_noise_robustness.py:1256`. So $\sigma$ is directly comparable to a published assay error quoted in log units — no conversion, no hand-waving.

**Published experimental error, in the same log units.** Verified against the primary sources (a fact-checking pass rejected several near-miss figures; only surviving numbers are listed):

| Endpoint | Error | Statistic / scope | Source |
|---|---|---|---|
| hERG $pK_i$ | **0.54** | SD of an individual measurement, inter-lab public data | Kramer et al. 2012, *J. Med. Chem.* **55**:5165, doi:10.1021/jm300131x |
| $pIC_{50}$ (general) | **0.68** | SD, inter-lab, 20{,}356 pairs, ChEMBL 14 | Kalliokoski et al. 2013, *PLoS ONE* **8**:e61007, doi:10.1371/journal.pone.0061007 |
| $pIC_{50}$, intra-lab floor | 0.17–0.22 | SD, one laboratory, inter-day | Kalliokoski et al. 2013 |
| hERG $pIC_{50}$ | 0.737 | RMSD, binding vs electrophysiology, 209 compounds | Sato et al. 2018, *PLoS ONE* **13**:e0199348 |
| logD$_{7.4}$ | ~~0.27~~ | ⚠ **UNVERIFIED — DO NOT USE.** Verification could not confirm this number exists in the paper (ACS returns 403, Unpaywall reports closed with no repository copy; the abstract contains neither 0.27 nor 307). See §11 closing assessment. | Bruneau \& McElroy 2006, *JCIM* **46**:1379, doi:10.1021/ci0504014 |
| logD, modern benchmark | **MAE 0.48** after curation (0.7 before) | repeat-test agreement, deliberately worst-case | Niu et al. 2024, *Sci. Data* **11**:985, Table 5 — **fully verified, open access; use this instead of Bruneau** |
| logD$_{7.4}$ | 0.43–0.62 | inter-source **upper bounds** (derived from MAE 0.48 / RMSE 0.881) | Niu et al. 2024, *Sci. Data* **11**:985, doi:10.1038/s41597-024-03793-0 |
| logP, best case | 0.07–0.09 | two laboratories, one prescribed protocol, $n=23$ | Takács-Novák \& Avdeef 1996, *JPBA* **14**:1405 |
| Caco-2 efflux (log$_{10}$) | ~0.43 | inter-lab; derived from the atenolol 20.9-fold spread across 10 labs | Hayeshi et al. 2008, *Eur. J. Pharm. Sci.* **35**:383, via Chen et al. 2017, *Fluids Barriers CNS* **14**:30 |

Every estimate falls roughly between 0.4 and 0.74, clustered near 0.5–0.6 — **regardless of endpoint**. That is the useful fact: assay error in log units is roughly scale-free across these assays. (The range was previously written as "0.27–0.74" on the strength of the Bruneau figure; with that number withdrawn, the verified logD support is Niu 2024's MAE 0.48, which sits comfortably inside the same band and does not change the conclusion.)

**Draft sentence for Methods (pending the D11 decision — do not insert yet):**
> Noise magnitudes are reported in the units of the label itself. Because all external endpoints are expressed on a logarithmic scale, the injected $\sigma$ is directly comparable to published estimates of experimental measurement error, which lie between roughly 0.4 and 0.7 log units for $pK_i$ [Kramer 2012], $pIC_{50}$ [Kalliokoski 2013], logD [Niu 2024] and hERG potency under standardised protocols [Alvarez Baron 2025]. An injected $\sigma$ of 0.6 therefore corresponds to approximately one additional unit of the measurement error already present in public bioactivity data.

*(Citation set updated after verification: Bruneau 2006 and Hayeshi 2008 are dropped from the draft sentence — neither number survived checking. Kalliokoski, Niu and Alvarez Baron are all open access and fully verified.)*

**Four caveats that must travel with the claim:**
1. **QM9 is computed data.** Its labels come from DFT, so there is no experimental error to match and this justification does not transfer. $\sigma$ on QM9 is a controlled perturbation, not a simulation of measurement noise, and the paper must say so rather than letting the reader carry the ADME framing across.
2. **The error is already in the data.** Public labels *already* carry this noise; injecting $\sigma = 0.6$ roughly doubles it rather than creating it. Errors add in quadrature: $\sqrt{0.54^2 + 0.6^2} \approx 0.81$. Phrase it as "an additional unit", never as "we introduce realistic noise into clean data".
3. **The Caco-2 evidence is the weakest of the three.** The most quotable figures (5.3-fold average range, 2.8-fold mean deviation) trace to Fagerholm 2022, bioRxiv — a single-author, non-peer-reviewed preprint from a commercial vendor. **Cite the underlying Hayeshi 10-lab study instead**, where atenolol alone spans 0.18 to 3.76 across laboratories.
4. **Intra- vs inter-laboratory matters and is easy to misquote.** The 0.17–0.22 $pIC_{50}$ figures are a single-lab repeatability floor; the 0.54–0.68 figures are inter-laboratory. Public databases aggregate many sources, so the inter-laboratory number is the relevant one. Alvarez Baron et al. 2025 (*Sci. Rep.* **15**:29995) report $\tau = 0.18$, which despite being a five-lab study is a *within*-lab residual — do not quote it as an inter-lab error.

**This also corrects an approach I proposed and then abandoned.** I had suggested matching $\sigma$ to each dataset's label spread, so that noise was a constant fraction of signal. That is the wrong normaliser: assay error in log units is roughly constant across endpoints *irrespective* of how spread out the labels are, so a fixed $\sigma$ in log units is more defensible than a per-dataset one. The label-spread ratios remain useful for one narrow purpose — explaining why Caco-2 degrades earliest, since its labels are the most tightly spread (SD 0.434 against 1.19 for logD, so a given $\sigma$ is ~2.7$\times$ harsher there).

**Citations to add to `citations.bib`** (none of these are among the 51 keys currently used): Kramer 2012, Kalliokoski 2013, Sato 2018, Bruneau 2006, Hayeshi 2008. Niu 2024 and Takács-Novák 1996 are optional supporting cites.

**Direct links.** Open access, no login:
- Kalliokoski et al. 2013, *PLoS ONE* — https://doi.org/10.1371/journal.pone.0061007
- Sato et al. 2018, *PLoS ONE* — https://doi.org/10.1371/journal.pone.0199348
- Chen et al. 2017, *Fluids Barriers CNS* (reports the Hayeshi 10-lab data) — https://doi.org/10.1186/s12987-017-0078-x
- Niu et al. 2024, *Scientific Data* — https://doi.org/10.1038/s41597-024-03793-0
- Alvarez Baron et al. 2025, *Sci. Rep.* — https://doi.org/10.1038/s41598-025-15761-8

Paywalled — institutional access needed:
- Kramer et al. 2012, *J. Med. Chem.* (the $pK_i$ SD 0.54) — https://doi.org/10.1021/jm300131x
- Bruneau \& McElroy 2006, *JCIM* (the logD 0.27) — https://doi.org/10.1021/ci0504014
- Hayeshi et al. 2008, *Eur. J. Pharm. Sci.* (the Caco-2 10-lab study) — https://doi.org/10.1016/j.ejps.2008.08.004
- Takács-Novák \& Avdeef 1996, *JPBA* — https://doi.org/10.1016/0731-7085(96)01773-6

⚠ **Verbatim quotes are being extracted and independently re-checked against each source** (workflow `wf_a08b90aa-22c`). Until that lands, treat every number above as correctly attributed but **not yet quotable** — cite, do not quote.

---

## Ranking at $\sigma = 0.6$ — what it shows, and what AUC$_\text{norm}$ misses (added 2026-08-20)

Once $\sigma = 0.6$ is fixed, three rankings can be compared directly: **clean R²** (σ=0), **R² at σ=0.6**, and **AUC$_\text{norm}$**. They do not agree, and the pattern of disagreement is the finding.

**⚠ Data provenance — read before quoting any number here.** These tables come from
`KIRBy/tests/results/validation_rerun/` (13 models × 4 reps × 3 datasets, 48{,}510 rows), which is the
**local per-σ source**. The paper pipeline is fed `alternative_full` on ARC, which carries only 7 models.
The two are different runs. Numbers below must be reproduced from whichever directory the final figures
use before they enter the paper. Representation held at PDV (`PRIMARY_REP`), baseline gate R² ≥ 0.3.
Backing CSVs: `scratchpad/rank_sigma06_pdv.csv`, `scratchpad/rank_sigma06_all_reps.csv`.

**Do the three measures rank models the same way? (Spearman, rep = PDV)**

| Dataset | Strategy | n | R²@0.6 vs clean | R²@0.6 vs AUC | clean vs AUC |
|---|---|---|---|---|---|
| LogD | hetero | 13 | 0.97 | −0.16 | −0.23 |
| LogD | legacy | 13 | 0.96 | 0.07 | −0.08 |
| LogD | outlier | 13 | 0.94 | 0.18 | −0.01 |
| LogD | quantile | 13 | 0.97 | −0.23 | −0.36 |
| LogD | **threshold** | 13 | **0.70** | 0.65 | 0.02 |
| LogD | valprop | 13 | 0.86 | 0.40 | −0.10 |
| hERG | hetero | 6 | 0.94 | 0.49 | 0.26 |
| hERG | legacy | 6 | 0.94 | 0.20 | 0.03 |
| hERG | outlier | 6 | 1.00 | 0.49 | 0.49 |
| hERG | quantile | 6 | 0.77 | 0.54 | 0.14 |
| hERG | **threshold** | 6 | **0.26** | 0.77 | −0.09 |
| hERG | **valprop** | 6 | **0.20** | 0.83 | −0.09 |
| Caco-2 | hetero | 13 | 0.91 | 0.14 | 0.04 |
| Caco-2 | legacy | 13 | 0.52 | 0.75 | 0.05 |
| Caco-2 | outlier | 13 | 0.59 | 0.82 | 0.29 |
| Caco-2 | quantile | 13 | 0.62 | 0.66 | 0.04 |
| Caco-2 | threshold | 13 | 0.67 | 0.77 | 0.25 |
| Caco-2 | valprop | 13 | 0.49 | 0.81 | 0.16 |

**Three things follow, all defensible:**

1. **Clean R² and AUC$_\text{norm}$ are essentially unrelated** — the last column runs from −0.36 to +0.49 and sits near zero in 14 of 18 cells. Whatever AUC$_\text{norm}$ ranks models by, it is not how accurate they are. This is the same near-independence seen at model-mean level (Spearman 0.179), now reproduced per dataset × strategy on 13 models.
2. **Under mild noise, ranking at σ=0.6 just reproduces the clean ranking** (0.91–1.00 for hetero/legacy/outlier/quantile on LogD and hERG). Noise of this magnitude does not change which model you would pick.
3. **Under the stress strategies it does change** — R²@0.6 vs clean falls to 0.26 (hERG threshold) and 0.20 (hERG valprop), and 0.70 on LogD threshold. **Threshold and value-proportional noise are where robustness actually decides the winner**; the other four largely do not.

**Top-3 by each measure, Gaussian noise, rep = PDV** — the disagreement made concrete:

| Dataset | by R² at σ=0.6 | by clean R² | by AUC$_\text{norm}$ |
|---|---|---|---|
| LogD | DNN, GP, MLP | DNN, GP, SVM | **NGBoost, MLP-VBLL-Full, BNN-Full** |
| hERG | RF, LightGBM, QRF | LightGBM, RF, QRF | **NGBoost, RF, SVM** |
| Caco-2 | MLP-VBLL-Full, DNN, MLP | DNN, SVM, GP | **MLP-BNN-Full, MLP-VBLL-Full, GP** |

On LogD the AUC$_\text{norm}$ podium and the accuracy podium share **no models at all**, and the AUC winners are the low-baseline ones — NGBoost's clean R² on PDV/LogD is 0.661 against DNN's 0.797. This is the clearest single demonstration that the retention metric crowns models for having less to lose.

**⚠ Also found, needs checking before it changes anything:** `validation_rerun` contains **13 models**, including MLP, BNN-Full, VBLL-Full, MLP-BNN-Full, MLP-VBLL-Full and GP. This contradicts the standing note that "KIRBy has no MLP/VBLL models" and the D2 finding that the BNN comparison pairs "never fire on validation". That note was true of `alternative_full`; it is **not** true of `validation_rerun`. Which directory the paper's validation section should use is now an open question — resolving it may unblock the BNN/VBLL validation comparisons that were written off as impossible. Do not act on this until the directory question is settled.

**Full evidence** (31-agent verified search, including the rejected claims and what was wrong with them): `tasks/wqrlyif68.output`; per-agent journal under `subagents/workflows/wf_b419480e-a74/`. Summary in memory as `experimental_noise_magnitudes.md`.

---

## Methods — Performance Metrics
**Argument now:** This section must (C1) define the robustness metric as AUC$_\text{norm}$ — the normalised area under the R² retention curve, higher = more robust, ~[0,1], baseline-decoupled — with no slope/NDS language and no "we changed it" framing; and (C2) state, as a plain methodological choice, that the uncertainty–noise correlation is computed *within* each fixed $\sigma$ (and, correctly, one noise strategy at a time — pooling strategies was the bug behind the discarded per-sample "detector"). This is the correct method and it is what lets the paper report the per-sample result honestly, which is a **null** — no model resolves which individual labels were corrupted. The methodological sentence stays regardless of the restructure; only the *result* it feeds changed. Everything else in the section is correct and stays byte-for-byte.

---

**Replace (L240):**
> Spearman's rank correlation coefficient ($\rho$) was used to quantify the relationship between predicted uncertainty ($u_i$) and absolute error ($|y_i - \hat{y}_i|$), as well as between predicted uncertainty and injected noise magnitude.

**With:**
> Spearman's rank correlation coefficient ($\rho$) was used to quantify the relationship between predicted uncertainty ($u_i$) and absolute error ($|y_i - \hat{y}_i|$), as well as between predicted uncertainty and injected noise magnitude. The uncertainty–noise correlation was computed within each fixed noise level $\sigma$ rather than by pooling across levels. Pooling conflates the population trend, in which mean uncertainty rises with $\sigma$, with genuine per-sample discrimination; only the within-$\sigma$ correlation isolates whether a model resolves which individual labels were perturbed.

*(The following sentence at L240, "Higher correlations indicate that the model's uncertainty estimates reliably identify predictions that are likely to be incorrect," is correct and stays verbatim.)*

---

**Replace (L255–258, the NDS definition block including the equation):**
> To evaluate the effect of label noise, we examined model performance degradation under increasing artificial noise with $\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$ using the Noise Degradation Slope (NDS), calculated as the slope of R$^2$ versus noise level:
> $$\text{NDS} = \frac{dR^2}{d\sigma}.$$

**With (paste-ready LaTeX):**
```latex
To evaluate the effect of label noise, we examined performance retention under increasing artificial noise with $\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$. For each configuration we recorded $R^2(\sigma)$ and normalised it by the clean-label value to obtain the retention ratio $R^2(\sigma)/R^2(0)$. Our robustness metric is the normalised area under this retention curve,
$$
\text{AUC}_{\text{norm}} = \frac{1}{\sigma_{\max}} \int_0^{\sigma_{\max}} \frac{R^2(\sigma)}{R^2(0)}\, d\sigma,
$$
evaluated by the trapezoidal rule over the eleven noise levels ($\sigma_{\max} = 1.0$).
```

---

**Replace (L260):**
> Values closer to zero indicate that noise has less of an effect on performance, negative values indicate a higher sensitivity to noise. Positive slopes, which were not observed in this study, would indicate that the noise \textit{improves} model performance. Configurations with baseline R$^2 < 0.6$ were excluded from robustness analysis, as poor performance on clean labels tends to remain poor for noisy labels as well, producing misleadingly shallow or even positive slopes that represent poor performance across the board.

**With:**
> $\text{AUC}_{\text{norm}}$ lies on approximately $[0, 1]$: a value near 1 means performance is retained at every noise level, and lower values indicate faster degradation. Because each curve is normalised by its own clean-label performance, $\text{AUC}_{\text{norm}}$ is not confounded by baseline accuracy, and because it integrates the whole curve it makes no assumption that degradation is linear. We filtered out model--representation configurations with baseline R$^2 < 0.3$, excluding 48 configurations where an unstable clean-label denominator would distort the retention ratio (Additional file~5).

*(Matches your paper's convention at L432: count in the prose, file as a bare trailing `(Additional file~5)` — not `(48 configurations; Additional file~5)`. "Additional file~5" is correct: L667 lists it as the "Configuration[s]" supplement. Count "48" verified below.)*

---

**Replace (L262):**
> A two-way analysis of variance (ANOVA) decomposition was conducted separately for each noise strategy to identify the relative contributions of molecular representation and model architecture choice on both predictive performance and noise robustness. We chose ANOVA with $\eta^2$ effect sizes rather than pairwise significance tests because our primary question is variance attribution: how much of the variation in robustness is explained by model architecture versus molecular representation. Pairwise tests can establish whether two models differ significantly but cannot partition the total variance among factors---the key distinction our analysis requires. This per-strategy approach avoids inappropriate aggregation across fundamentally different noise types. For a given metric $y$ (either $R^2$ at fixed noise or NDS):

**With:** *(identical except the final clause)*
> …This per-strategy approach avoids inappropriate aggregation across fundamentally different noise types. For a given metric $y$ (either $R^2$ at fixed noise or AUC$_\text{norm}$):

*(Only "NDS" → "AUC$_\text{norm}$" changes; the entire preceding justification is correct and stays verbatim.)*

---

**Table `tab:metrics_summary` — NDS row (L291–293). Replace:**
```latex
NDS
  & $dR^2 / d\sigma$ (slope of $R^2$ vs $\sigma$)
  & Robustness to label noise; values near zero indicate stability, more negative values indicate sensitivity \\
```
**With (paste-ready):**
```latex
AUC$_\text{norm}$
  & $\frac{1}{\sigma_{\max}}\int_0^{\sigma_{\max}} R^2(\sigma)/R^2(0)\, d\sigma$ (trapezoidal)
  & Robustness to label noise; approximately $[0, 1]$, higher indicates more performance retained across noise levels \\
```

**Table `tab:metrics_summary` — Spearman's $\rho$ row (L306–308). Replace:**
```latex
Spearman's $\rho$
  & Rank correlation, $[-1, 1]$
  & Calibration of predicted uncertainty against (i) absolute error and (ii) injected noise magnitude \\
```
**With (paste-ready):**
```latex
Spearman's $\rho$
  & Rank correlation, $[-1, 1]$
  & Calibration of predicted uncertainty against (i) absolute error and (ii) injected noise magnitude (computed within each $\sigma$) \\
```

*(All other rows of `tab:metrics_summary` — RMSE, $R^2$, Wilcoxon, Kendall's $W$, ANOVA $\eta^2$, ICC(1,1), Coverage — and the table caption are correct and stay verbatim. The ECE row is **deleted**; see "Metric removal — ECE".)*

---

**Everything else in L234–322 stays verbatim** *(except the ECE deletions listed in "Metric removal — ECE")*, including: the $\sigma$-definition paragraph (L236), the Wilcoxon/Kendall paragraph (L238, correct — no numbers), the coverage equation block (L242–252), the full ANOVA-model equation and $\eta^2$ derivation (L263–272), and the independence/ICC/redundancy paragraphs (L274–276). None of these contains an NDS reference, a pooled-uncertainty claim, or a CSV-contradicted number.

---

**Decisions (folded from the old figures / change-list / paper-voice material):**
- **Kept** the guide's five paper-voice Replace/With blocks (REVISION_GUIDE.md L1413–1449) verbatim as the section's spine — they already satisfy every hard rule (no "NDS"/"slope"/"unlike"/"previously", no number in metric-defining sentence).
- **Kept** the change-list gate note (L51): R²<0.3 excludes **48** configs (was <0.6 excluding 66); folded the "48" directly into the L260 replacement.
- **Removed** the change-list's parenthetical "(was $<0.6$ excluding 66)" from the paper prose — that is guide-internal bookkeeping and would read as a "we changed it" contrast, which is forbidden.
- **Removed** the whole "positive slopes… would indicate noise improves performance" sentence (slope language is gone; no NDS-era stability/sensitivity framing survives).
- **Deferred** the cross-rep Spearman $\rho=0.82$ (L1459 of guide) — it belongs to Results §4.1, not this Methods section, so it is not printed here.
- **Flagged** (not silently kept) the "Additional file~5" pointer for the author to reconcile with the paper's existing "Additional files~2--4" numbering.

**Verification — numbers:**
- 48 excluded configurations (baseline R²<0.3) | `results/paper_figures_v2/excluded_configs.csv` | **OK** — file has exactly 48 rows; the `baseline` column is <0.3 (all near-zero/negative) for every row; rows are precisely {dnn_bnn_full, dnn_vbll, mlp_bnn_full, mlp_vbll} × {mhggnn, mol2vec} × 6 strategies (4×2×6=48). The old `results/paper_figures/excluded_configs.csv` has 66 rows (the R²<0.6-era gate), consistent with the "was 66" bookkeeping.
- $\sigma_{\max}=1.0$, 11 noise levels {0, 0.1, …, 1.0} | matches author's L236/L255 verbatim | **OK** (unchanged author text).
- No other numeric value is printed in this section (the $\eta^2$, AUC ranking, Kendall $W$, Wilcoxon, and within-$\sigma$ $\rho$ values from the ledger all belong to Results sections, not Methods–Metrics).

**Verification — citations:** The Performance Metrics subsection body (L234–322) contains **no `\cite`/`\citep`/`\citet` keys** — all citations sit in the preceding Models subsection (L228–232). For completeness, every key in that adjacent block was checked against `citations.bib`: Obrezanova2007, gauche, Rasmussen2005, quinonero2005, lakshminarayanan2017, Ralaivola2005, moss2020, pytorchGeometric, gal2016, Harrison2024, kendall2017, Duan2020, Meinshausen2006 — all **IN-BIB**. No new citation is introduced by any replacement above.

---

I have everything I need. My section is number-free with a single substantive edit at L380. Here is the consolidated guide block.

## Methods — Dataset / Representations / Models, Noise Strategies, NoiseInject Framework

**Argument now:** These subsections are the paper's methodological scaffolding and are almost entirely untouched by both corrections — the dataset, representation, and model descriptions are C1/C2-neutral, and the noise-strategy definitions are the *inputs* to robustness, not the metric. The only load-bearing edit is the NoiseInject software's self-description (L380), where the framework must be said to report AUC$_{\text{norm}}$ (normalised area under the $R^2$ retention curve, higher = more robust), not a degradation slope. The L232 uncertainty-decomposition passage stays verbatim: it is decomposition *mechanics*, not the reversed uncertainty *finding*, so C2 does not reach it.

---

**L380 — the one substantive edit.**

Replace (verbatim author line):
> The framework computes standard regression metrics (RMSE, $R^2$, mean absolute error [MAE]) and classification metrics (accuracy, precision, recall, and F1 score, both macro- and weighted-averaged, with per-class breakdowns). For probabilistic models, it additionally computes uncertainty-calibration metrics: expected calibration error (ECE), empirical coverage at $1\sigma$ and $2\sigma$, mean prediction-interval width, and the Spearman correlations between predicted uncertainty and both absolute error and injected noise. It reports noise robustness metrics, including the noise-performance degradation slope and retention percentage, to quantify performance degradation across noise levels. Results are returned as structured \texttt{pandas} DataFrames (per-noise-level values and an aggregate summary) for downstream analysis.

With (only the wrong span changed; everything else byte-for-byte identical):
> The framework computes standard regression metrics (RMSE, $R^2$, mean absolute error [MAE]) and classification metrics (accuracy, precision, recall, and F1 score, both macro- and weighted-averaged, with per-class breakdowns). For probabilistic models, it additionally computes uncertainty-calibration metrics: empirical coverage at $1\sigma$ and $2\sigma$, mean prediction-interval width, and the Spearman correlations between predicted uncertainty and both absolute error and injected noise. It reports noise robustness metrics, including the normalised area under the $R^2$ retention curve (AUC$_{\text{norm}}$; higher values indicate greater robustness) and retention percentage, to quantify performance degradation across noise levels. Results are returned as structured \texttt{pandas} DataFrames (per-noise-level values and an aggregate summary) for downstream analysis.

**Everything else in the section stays EXACTLY as written — explicit no-edit ledger:**
- **L206–213 (Dataset):** verbatim. QM9, scaffold-split, HOMO–LUMO, ADME datasets — all C1/C2-neutral. No change.
- **L215–219 (Representations):** verbatim. No change. (The standing "ECFP4 → topological fingerprint" rename is a *separate* correction not in this task's C1/C2 scope; do NOT apply it here.)
- **L221–230 (Models):** verbatim. Architecture/BNN/VBLL descriptions are neutral.
- **L232 (uncertainty decomposition):** **KEEP VERBATIM.** This is decomposition mechanics ("we derive epistemic uncertainty from the posterior variance… aleatoric from a learned observation noise term"), not the reversed pooled uncertainty *finding*. C2 rule 1(b) targets the finding statement and the aleatoric-vs-epistemic *organizing frame for the result* (handled in Results/Discussion by another owner), not the methods description of how components are computed. Do not lean on this paragraph elsewhere; do not delete it.
- **L324–366 (Noise Strategies + Table `tab:regression_noise`):** verbatim. These define the noise *inputs* ($s_i$ per strategy), independent of the robustness metric. No change.

---

**Decisions (folded from the old REVISION_GUIDE material):**
- **Kept** the guide's L380 instruction ("both list 'noise-performance degradation slope / retention' → update to AUC$_{\text{norm}}$; the retention curve it integrates is already the object described, so this is a light touch") — implemented as the single Replace/With above.
- **Removed** the old guide's framing that lumped L380 with "Availability of data (L624)" — L624 is outside this section (Back-matter owner); noted here only so it is not double-edited.
- **Removed/declined** the topological-fingerprint rename in L217: it is a standing convention, not a C1/C2 correction, and Hard Rule 1 forbids touching correct prose outside the two changes.
- **Declined** any edit to L232 despite the old guide's aggressive "delete the aleatoric-vs-epistemic frame entirely" note — that note targets the *Results/Discussion finding*, not this methods mechanic; the task brief explicitly says "leave mechanics, do not lean on it."
- **Confirmed** the section owns no table/figure that carries a robustness number: `tab:regression_noise` (L330) and `fig:noise_strategies` (L368) describe noise injection only, so no verified-value paste-ins are needed here. (The NDS-bearing `tab:metrics_summary` at L278–322 and the NDS definition at L255–260 are in the *Performance Metrics* subsection, a different section owner — not edited here.)

**Verification — numbers:** None. This section (Dataset, Representations, Models, Noise Strategies, NoiseInject Framework, incl. the L380 edit) contains no data-derived numbers; the replacement prose introduces none. Confirmed number-free per Hard Rule 4 (no numbers required, none printed). All quantitative content (N sizes, dimensions, kernel params, noise multipliers) is design/config, not results, and is left verbatim.

**Verification — citations** (all keys appearing in L206–233, L324–366, L380; grepped against `citations.bib`):
Ramakrishnan2014 IN-BIB · deepchem IN-BIB · Heid2023 IN-BIB · Islam2019 IN-BIB · Hllermeier2021 IN-BIB · Fediai2023 IN-BIB · openadmet IN-BIB · Zdrazil2023 IN-BIB · landrum2024 IN-BIB · rdkit IN-BIB · sns IN-BIB · Cherkasov2014 IN-BIB · Bertoni2020 IN-BIB · jaeger2018mol2vec IN-BIB · kishimoto2023 IN-BIB · Kolmar2021 IN-BIB · Svetnik2003 IN-BIB · Breiman2001 IN-BIB · Meinshausen2006 IN-BIB · Mustapha2016 IN-BIB · Tian2022 IN-BIB · ke2017lightgbm IN-BIB · Duan2020 IN-BIB · Vapnik1995 IN-BIB · Obrezanova2007 IN-BIB · gauche IN-BIB · Rasmussen2005 IN-BIB · quinonero2005 IN-BIB · lakshminarayanan2017 IN-BIB · Ralaivola2005 IN-BIB · moss2020 IN-BIB · pytorchGeometric IN-BIB · gal2016 IN-BIB · Harrison2024 IN-BIB · kendall2017 IN-BIB · noiseinject IN-BIB. (37/37 present; the L380 edit adds no new citation.)

---

I have everything I need. No `\cite` keys exist in this section. Here is the consolidated guide.

## Results — Variance decomposition

**Argument now:** The interaction term still dominates *performance*; but *robustness* (now AUC$_\text{norm}$, higher = more robust) inverts to model-dominance for only **four of six** strategies — outlier and heteroscedastic noise are residual-dominated. The section must own that partial reversal plainly and drop every slope/NDS reference, keeping the author's mechanistic reasoning intact.

---

**(1) Paragraph L391 — the inversion paragraph.** Change only the two wrong spans; the rest stays verbatim.

Replace: `However, for noise robustness as measured by the noise degradation slope (NDS), this ordering shifts. The choice of model architecture is instead the largest source of variance, while the interaction term drops to second and representation third.`
With:    `However, for noise robustness, quantified by the normalised area under the R$^2$ retention curve (AUC$_\text{norm}$; higher values indicate greater robustness), this ordering shifts, though not uniformly across strategies. For four of the six strategies (Gaussian, quantile, threshold and value-proportional) the choice of model architecture becomes the largest structured source of variance (36.8--54.7\%), while the interaction term drops to second and representation to a small residual role; under outlier and heteroscedastic noise, however, the differences between models shrink relative to the run-to-run variance, and the residual term dominates (83.6\% and 77.4\%).`

> **✅ CONFIRMED (2026-08-14).** The final clause — "under outlier and heteroscedastic noise… the residual term dominates (83.6% and 77.4%)" — is verified on ARC and safe to publish (robust to roster and SS-type; see the table banner below). One phrasing caution: this residual reflects **run-to-run variance under stochastic noise**, not models becoming interchangeable (at cell-mean granularity Model rises to 55%/71%). So keep the clause but frame it as "differences *between models* shrink relative to run-to-run variance," rather than implying model architecture stops mattering.

Everything else in L391 ("This observed inversion makes sense because…" through "…stronger impact on how it handles noise.") is correct and stays **byte-for-byte**.

**(2) Paragraph L393 — per-strategy interpretation.** No wrong span (no NDS, no CSV-contradicted number; threshold/value-prop are genuinely the most-degraded and most model-dominated, and "for outlier noise, model and interaction effects were both smaller" is true). **KEEP VERBATIM.** The heteroscedastic-residual fact is now carried by paragraph (1) and the table, so no edit is forced here.

**(3) Table `tab:anova_decomposition` — caption L397 + body L406–411.**

> **✅ CONFIRMED on ARC (2026-08-14) — all six rows are final; the earlier "roster artifact" worry was wrong.** `reproduce_robustness_anova.py` + `robustness_anova_sensitivity.py` settle it: (i) the robustness ANOVA roster is a **clean, consistent 7 models across all six strategies** — `dnn, lgb, mlp, ngboost, rf, svm, xgboost` (VBLL/BNN never enter; they're gated out on the embeddings). It does **not** grow to 9 or 10. Every η² reproduces the CSV exactly (`|code−CSV| = 0`). (ii) The **Heteroscedastic (77.4) and Outlier (83.6) residual-dominance is REAL, not a VBLL artifact:** adding the NN/Bayesian models back (11-model roster) barely moves it (Outlier 85.5→85.9, Hetero 82.8→83.1), and it is stable across SS-type (Type I = Type II). (iii) So all six robustness rows are publishable as-is; no re-run needed. **Model count for the caption = 7 for robustness** (eleven for performance).
>
> **Nuance to phrase correctly (from the granularity check):** the high Outlier/Hetero residual reflects **run-to-run (replicate) variance**, not models being interchangeable. At cell-mean granularity the residual collapses to ~0 and Model jumps to **55% (Outlier) / 71% (Heteroscedastic)**. So write the finding as "under outlier and heteroscedastic noise, differences *between models* shrink relative to run-to-run variance, and the residual term dominates (83.6% and 77.4%)" — not "model architecture stops mattering." The stochastic subset-selection in these two noise types is what inflates within-cell variance.

> **⚠ YOUR PAPER'S TABLE IS CURRENTLY STALE — this is a live text/table mismatch (found 2026-08-15).** paper.tex table rows (L396–401) still hold OLD robustness numbers (e.g. Gaussian Model **48.7**, Heteroscedastic Resid **41.0**) that match neither the current code nor your own prose. Your text at L380 already cites the correct **83.6 / 77.4 / 43.8**. So replace the whole table body with the block below and text+table will finally agree.

**Complete paste-ready table** (`tab:anova_decomposition`). Every value verified against `results/paper_figures_v2/table1_anova_summary.csv`; robustness columns independently confirmed by `reproduce_robustness_anova.py`. All six rows final. Needs `\usepackage{booktabs}`.
```latex
\begin{table}[htbp]
\centering
\caption{ANOVA variance decomposition by noise strategy on the QM9 HOMO--LUMO gap. Each cell reports $\eta^2$ (\%), the share of variance explained by that factor. \textbf{Model} = model architecture effect; \textbf{Rep} = molecular representation effect; \textbf{Inter.}\ = model$\times$representation interaction; \textbf{Resid.}\ = residual. Performance uses R$^2$ at $\sigma = 0.3$ (eleven models); robustness uses AUC$_\text{norm}$, the normalised area under the R$^2$ retention curve, higher values indicating greater robustness (seven models: the tree ensembles, SVM and both plain neural networks, i.e.\ those with baseline R$^2 \ge 0.3$ on all five representations). $\eta^2 = SS_\text{factor}/SS_\text{total}$ from weighted marginal sums of squares (equivalent to Type~I/II for balanced designs). Bold marks the dominant factor in each half-row.}
\label{tab:anova_decomposition}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
 & \multicolumn{4}{c}{\textbf{Performance} (R$^2$ at $\sigma=0.3$)} & \multicolumn{4}{c}{\textbf{Robustness} (AUC$_\text{norm}$)} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9}
\textbf{Strategy} & Model & Rep & Inter. & Resid. & Model & Rep & Inter. & Resid. \\
\midrule
Gaussian        & 24.6 & 22.8 & \textbf{48.1} & 4.5 & \textbf{43.8} & 5.2 & 16.9 & 34.2 \\
Quantile        & 25.1 & 22.4 & \textbf{46.9} & 5.6 & 36.8 & 4.4 & 15.1 & \textbf{43.7} \\
Threshold       & 25.0 & 21.2 & \textbf{48.0} & 5.8 & \textbf{54.7} & 7.9 & 22.6 & 14.8 \\
Value-prop.     & 27.8 & 22.0 & \textbf{46.5} & 3.8 & \textbf{52.5} & 6.0 & 19.9 & 21.6 \\
Heteroscedastic & 27.9 & 23.5 & \textbf{46.2} & 2.5 & 14.0 & 0.7 & 8.0 & \textbf{77.4} \\
Outlier         & 25.3 & 22.0 & \textbf{49.2} & 3.5 & 10.3 & 0.2 & 5.9 & \textbf{83.6} \\
\bottomrule
\end{tabular}
\end{table}
```
(Bolding rule now "largest cell in row": for Quantile/Hetero/Outlier the residual is largest, so bold moves off Model onto Resid — matches the numbers. Old table wrongly bolded Model for every robustness row. All six rows are confirmed final — no re-run pending.)

**(4) Figure caption L419 — `fig:anova_decomposition`.**

Replace: `\caption{ANOVA variance decomposition ($\eta^2$, \%) for a) predictive performance (R$^2$ at $\sigma=0.3$) and b) noise robustness, quantified by the noise degradation slope (NDS), by noise strategy. Lower (more negative) NDS indicates faster performance loss under noise.}`
With:    `\caption{ANOVA variance decomposition ($\eta^2$, \%) for a) predictive performance (R$^2$ at $\sigma=0.3$) and b) noise robustness, quantified by the normalised area under the R$^2$ retention curve (AUC$_\text{norm}$; higher values indicate greater robustness), by noise strategy.}`
(Image `fig2_anova_decomposition.png` must be regenerated from `paper_figures_v2/` on the AUC$_\text{norm}$ scale — author action.)

**(5) Paragraph L423 — interaction / Spearman.** Change only the wrong statistic; "12 models"→"11 models" (verified 11 common models).

Replace: `Model robustness on one representation strongly predicts robustness on another (Spearman $\rho = 0.73$, $p = 0.007$ for ECFP4 vs PDV; 12 models).`
With:    `Model robustness on one representation strongly predicts robustness on another (Spearman $\rho = 0.82$, $p = 0.002$ for ECFP4 vs PDV; 11 models).`
The first sentence ("While some model architectures, like SVM and full BNNs maintain consistent noise robustness scores…") and the trailing "However, the scatter around the diagonal reflects the interaction term identified by the ANOVA." stay **verbatim**.

**(6) Figure caption L428 — `fig:interaction`.**

Replace: `\caption{Representation--model interaction effects on the QM9 HOMO--LUMO gap. a) Noise degradation slope (NDS) heatmap showing model robustness across representations under Gaussian noise; lower (more negative) NDS indicates faster degradation. ``N/A'' indicates configurations excluded due to baseline $R^2 < 0.6$ or kernel incompatibility (GP uses a Tanimoto kernel defined on binary vectors, which is incompatible with PDV). b) NDS on PDV versus ECFP4 fingerprint for each model under Gaussian noise, with Spearman correlation.}`
With:    `\caption{Representation--model interaction effects on the QM9 HOMO--LUMO gap. a) AUC$_\text{norm}$ heatmap showing model robustness across representations under Gaussian noise; higher values indicate more performance retained under noise. ``N/A'' indicates configurations excluded due to baseline $R^2 < 0.3$ or kernel incompatibility (GP uses a Tanimoto kernel defined on binary vectors, which is incompatible with PDV). b) AUC$_\text{norm}$ on PDV versus ECFP4 fingerprint for each model under Gaussian noise, with Spearman correlation.}`
(`fig_interaction.png` must be regenerated on the AUC$_\text{norm}$ scale — author action.)

**(7) Paragraph L432 — filtering / gate.** Three wrong spans.

Replace: `with baseline predictive performance R$^2 < 0.6$, excluding 66 configurations where poor clean-data performance would produce misleadingly shallow degradation slopes (Additional file~5).`
With:    `with baseline predictive performance R$^2 < 0.3$, excluding 48 configurations for which a near-zero clean-data denominator would make the retention ratio unstable (Additional file~5).`

Replace: `considering both predictive performance as measured by $R^2$ as well as the noise degradation slope (NDS).`
With:    `considering both predictive performance as measured by $R^2$ as well as robustness as measured by AUC$_\text{norm}$.`

Replace: `We observe that NDS varies substantially across architectures and noise strategies, as seen in Figure~\ref{fig:global_overview}.`
With:    `We observe that AUC$_\text{norm}$ varies substantially across architectures and noise strategies, as seen in Figure~\ref{fig:global_overview}.`
(The sentence "We evaluated all model architectures under increasing label noise on the PDV representation across six strategies…" up to the metric clause stays verbatim.)

**(8) Figure caption L437 — `fig:global_overview`.**

Replace: `b) Noise degradation slope (NDS) heatmap across all model architectures and noise strategies on PDV; lower (more negative) NDS indicates faster degradation under noise.`
With:    `b) AUC$_\text{norm}$ heatmap across all model architectures and noise strategies on PDV; higher values indicate more performance retained under noise.`
Panel a) sentence and the "Tanimoto GP is absent…RBF kernel is included for PDV-specific analyses." sentence stay **verbatim**. (`fig1_global_overview.png` regenerated on AUC$_\text{norm}$ scale — author action.)

---

**Decisions (folded from old figures/lit/review material):**
- KEPT the revision guide's robustness-η² table body and both per-strategy paragraphs' logic (they match `table1_anova_summary.csv` exactly) and the residual-dominance reframe for outlier + heteroscedastic.
- KEPT the corrected filtering gate (R²<0.3, 48 configs) and the retention-ratio-instability rationale.
- REPLACED the revision-guide draft's Spearman `ρ=0.86, p<0.001, 12 models` with the **verified** `ρ=0.82, p=0.002, 11 models` (both the paper's 0.73/12 and the old draft's 0.86/12 are wrong for the 11-model AUC$_\text{norm}$ set).
- REMOVED the revision-guide draft's heavier full-paragraph rewrite of L391/L393 (added ranges, extra sentences): applied minimal-span edits instead, per the hard rule to keep correct prose verbatim.
- CORRECTED the roster claim: an earlier draft called the robustness ANOVA a "balanced seven-model design." The forensic pass showed that is wrong (the code uses 9 models, growing to 10 for Hetero/Outlier). The caption no longer states a count; the true count is confirmed by the ARC re-run (see the provisional banner and "Re-run on ARC").
- LEFT "ECFP4" as the author wrote it (the topological-fingerprint rename is a separate optional convention, not a factual error here).
- L393 receives **no edit** (contains no NDS/slope and no CSV-contradicted number) — the old change-list's "add heteroscedastic" is satisfied in paragraph (1) instead.

**Verification — numbers:**
- Robustness η² all six rows (Model/Rep/Inter/Resid) | table1_anova_summary.csv | OK (Gaussian 43.8/5.2/16.9/34.2; Quantile 36.8/4.4/15.1/43.7; Threshold 54.7/7.9/22.6/14.8; Hetero 14.0/0.7/8.0/77.4; Value-Prop 52.5/6.0/19.9/21.6; Outlier 10.3/0.2/5.9/83.6)
- Model-dominant range 36.8–54.7% (4 strategies) | table1_anova_summary.csv | OK
- Residual 83.6% (outlier) / 77.4% (hetero) | table1_anova_summary.csv | OK
- Performance η² (Model 24.6–27.9, Rep 21.2–23.5, Inter 46.2–49.2) | table1_anova_summary.csv | OK (unchanged from paper; "largest = interaction" and "model>rep" both hold)
- Spearman ρ=0.82, p=0.002, n=11 (continuous_pdv vs ecfp4, Gaussian) | table2_supp_auc_all_reps.csv (computed) | OK — MISMATCH vs paper's 0.73/0.007/12 and vs old draft's 0.86/12
- Excluded configs = 48 (4 NN-Bayesian variants × {mhggnn,mol2vec} × 6; VBLL×embedding = 24) | excluded_configs.csv | OK — MISMATCH vs paper's "66"
- All excluded baselines negative/near-zero (−0.02 to −0.18) → supports retention-ratio-instability rationale | excluded_configs.csv | OK

**Verification — citations:** No `\cite`/`\citep`/`\citet` keys appear anywhere in L389–439. None to check.

---

I have everything verified. Producing the consolidated section guide.

## Results — Robustness across noise strategies (paper.tex L441–504)

**Argument now:** Robustness is AUC_norm (fraction of baseline R² retained, higher = more robust, baseline-decoupled by construction). The reversal to own: because AUC_norm normalises out the clean-data level, PDV's best-in-class accuracy is *not* penalised — it is among the most robust representations, killing the "high baseline → most to lose → steepest slope" story that ran through this whole section. Secondary reversal: VBLL helps NN-β but not NN-α; the "both transformations help both networks" claim is false.

---

### Heading (L441) — unchanged (not factually wrong).

### Opening paragraph (L443)

**Replace:**
> NGBoost and SVM showed the smallest degradation in predictive performance across all strategies, while NN-$\beta$ showed the steepest (Table~\ref{tab:nds_ranking}). When sorting models by NDS, their ranks across noise strategies were consistent: Kendall's $W = 0.92$ ($p = 2.7 \times 10^{-8}$; 11 models, 6 strategies), indicating that a model's relative robustness is largely strategy-independent. The same ranking pattern holds on the ECFP4 fingerprint (Additional file~6). Robustness is also largely decoupled from baseline predictive performance: across models on PDV under Gaussian noise, NDS clusters near $-0.38$ regardless of baseline R$^2$ (Additional file~7), so a high-performing model is not automatically a more robust one.

**With:**
> NGBoost and random forests retained the most predictive performance across all strategies, while NN-$\beta$ retained the least (Table~\ref{tab:auc_ranking}). When ranking models by AUC$_{norm}$, their ranks across noise strategies were consistent: Kendall's $W = 0.9121$ ($p = 3.55 \times 10^{-8}$; 11 models, 6 strategies), indicating that a model's relative robustness is largely strategy-independent. The same ranking pattern holds on the ECFP4 fingerprint (Additional file~6). Robustness is also decoupled from baseline predictive performance: because AUC$_{norm}$ measures the fraction of baseline R$^2$ retained rather than its absolute level, a high clean-data R$^2$ does not inflate it (Additional file~7), so a high-performing model is not automatically a more robust one.

> **Note on the Kendall W basis (verified):** W = 0.9121 is computed on **AUC$_\text{norm}$ averaged across representations** (11 models), which is what `table6_kendalls_w.txt` outputs. The ranking table beside it (`tab:auc_ranking`) is **PDV-only** — if you instead computed W on the PDV columns alone it would be 0.9374. Both are internally correct; just make sure the text doesn't imply the 0.9121 is the W *of the PDV table*. Rounds to 0.91 (the old "0.92" in L443/L577 was the slope-era value).

### Table `tab:nds_ranking` → `tab:auc_ranking` (L445–467) — paste-ready

```latex
\begin{table}[htbp]
\centering
\caption{Normalised retention area (AUC$_{norm}$) by model on PDV (QM9 HOMO--LUMO gap), ranked by mean across six strategies. Higher AUC$_{norm}$ (closer to 1) indicates greater robustness. Strategy abbreviations: \textbf{Gauss.}\ = Gaussian, \textbf{Quant.}\ = Quantile, \textbf{Thresh.}\ = Threshold, \textbf{Hetero.}\ = Heteroscedastic, \textbf{Val.-P.}\ = Value-Proportional. Bold rows mark the two most robust models overall; bold cell entries mark each strategy's most robust configuration.}
\label{tab:auc_ranking}
\small
\begin{tabular}{lccccccc}
\toprule
\textbf{Model} & \textbf{Gauss.} & \textbf{Outlier} & \textbf{Quant.} & \textbf{Thresh.} & \textbf{Hetero.} & \textbf{Val.-P.} & \textbf{Mean} \\
\midrule
\textbf{NGBoost}    & $\mathbf{0.851}$ & $0.953$ & $\mathbf{0.873}$ & $\mathbf{0.645}$ & $\mathbf{0.922}$ & $\mathbf{0.701}$ & $\mathbf{0.824}$ \\
\textbf{RF}         & $0.846$ & $0.955$ & $0.872$ & $0.626$ & $0.921$ & $0.687$ & $\mathbf{0.818}$ \\
LightGBM            & $0.845$ & $0.951$ & $0.865$ & $0.630$ & $0.919$ & $0.690$ & $0.817$ \\
XGBoost             & $0.844$ & $0.946$ & $0.864$ & $0.628$ & $0.918$ & $0.686$ & $0.814$ \\
SVM                 & $0.839$ & $\mathbf{0.956}$ & $0.870$ & $0.620$ & $0.916$ & $0.682$ & $0.814$ \\
BNN-$\beta$         & $0.834$ & $0.941$ & $0.846$ & $0.609$ & $0.913$ & $0.668$ & $0.802$ \\
BNN-$\alpha$        & $0.829$ & $0.942$ & $0.846$ & $0.606$ & $0.911$ & $0.669$ & $0.801$ \\
VBLL-$\beta$        & $0.820$ & $0.943$ & $0.840$ & $0.595$ & $0.904$ & $0.654$ & $0.792$ \\
NN-$\alpha$         & $0.820$ & $0.938$ & $0.839$ & $0.583$ & $0.904$ & $0.650$ & $0.789$ \\
VBLL-$\alpha$       & $0.811$ & $0.932$ & $0.823$ & $0.577$ & $0.902$ & $0.641$ & $0.781$ \\
NN-$\beta$          & $0.789$ & $0.934$ & $0.816$ & $0.516$ & $0.892$ & $0.590$ & $0.756$ \\
\bottomrule
\end{tabular}
\end{table}
```
*(Bold rows are now NGBoost + RF; SVM keeps its Outlier cell (0.956). Update the two `\ref{tab:nds_ranking}` at L443 and L469 to `\ref{tab:auc_ranking}` — grep confirms these are the only two references.)*

### Spread + VBLL-α paragraph (L469)

**Replace:**
> The strategies differed not only in predictive performance severity, but also in their ability to discriminate between architectures. The spread of NDS across models differed by noise strategy, with outlier noise having a spread of 0.06 and threshold noise having 0.23, presumably exposing architectural differences that are masked by other types of noise. As seen in Table~\ref{tab:nds_ranking}, NGBoost was the most resistant to noise across strategies. With respect to the Variational Bayesian Last Layer (VBLL) models, VBLL-$\alpha$ showed a similar pattern, outperforming all tree-based models except NGBoost under threshold and value-proportional noise despite ranking below them under Gaussian noise. This suggests that its learned noise variance is most beneficial when label corruption is systematic rather than uniform. This observation with VBLL is not alone; predictive performance on clean data is not well-correlated with noise robustness. In fact, some of the most noise-robust models across strategies, such as NGBoost and SVM, did not perform particularly well on clean data. This decoupling indicates that noise robustness is not a by-product of clean-data accuracy: the inductive biases, regularization, and ensembling that limit how much a model fits corrupted labels are distinct from the mechanisms that maximize fit on clean data.

**With:**
> The strategies differed not only in predictive performance severity, but also in their ability to discriminate between architectures. The spread of AUC$_{norm}$ across models differed by noise strategy, with outlier noise having a spread of 0.02 and threshold noise having 0.13, presumably exposing architectural differences that are masked by other types of noise. This is consistent with the variance decomposition: outlier and heteroscedastic noise leave most of the AUC$_{norm}$ variance in the residual term, precisely where the spread is narrowest. As seen in Table~\ref{tab:auc_ranking}, NGBoost was the most resistant to noise across strategies, leading five of the six strategies. The Variational Bayesian Last Layer (VBLL) models did not recover this level of robustness: VBLL-$\alpha$ ranked below every tree-based ensemble on all six strategies (mean AUC$_{norm}$ = 0.781, versus 0.814--0.824 for the trees), so its learned noise variance did not, on its own, confer tree-level robustness. Predictive performance on clean data is not well-correlated with noise robustness. In fact, some of the most noise-robust models across strategies, such as NGBoost and SVM, did not perform particularly well on clean data. This decoupling indicates that noise robustness is not a by-product of clean-data accuracy: the inductive biases, regularization, and ensembling that limit how much a model fits corrupted labels are distinct from the mechanisms that maximize fit on clean data.

### ANOVA restatement + PDV paragraph (RE-ANCHORED 2026-08-14 to live paper.tex ~L462)

> **⚠ Anchor corrected + robustness claim tempered.** The June-30 local snapshot had a longer "PDV degrades fastest / most-to-lose" version; your live text is the shorter version below (the old guide quoted the snapshot — wrong). Now anchored to your real text.
>
> **Do NOT rank representations by mean AUC$_\text{norm}$.** That is *retention* (baseline-normalized): a low-baseline representation retains a higher fraction of less, so the AUC$_\text{norm}$ per-rep table (`table2_supp_auc_all_reps.csv`) is confounded by baseline and cannot support a "most robust representation" claim. (This is exactly what made the mistaken **binary PDV** look #1 — now dropped.) What the data *do* support: (i) representation differences in robustness are **small** (rep $\eta^2 \le 8\%$; whole AUC$_\text{norm}$ range ≈0.78–0.81), and (ii) **PDV (= continuous_pdv) has the strongest clean-data performance**. So the honest claim is "representation matters little for robustness; PDV's edge is clean-data accuracy," NOT "PDV is the most/least robust representation." Your original "PDV stood out / preferable for noisy data" overstates a robustness ranking the data can't cleanly support.

**Replace:**
> As established by the ANOVA, model architecture and the model--representation interaction term dominate NDS variance. A handful of models stood out as being noise-robust regardless of representation, namely SVM and full BNNs. Both methods rely on inductive biases, SVM on margin maximization and BNN on weight priors. However, other models like RF and NN-$\beta$ show the opposite pattern. These models can be robust to noise, but only when paired with particular representations. The pairings differ by model; however, PDV stood out as having particularly strong robustness to noise. This suggests that PDVs are preferable when dealing with noisy data.

**With:**
> As established by the ANOVA, model architecture and the model--representation interaction term dominate AUC$_\text{norm}$ variance. A handful of models stood out as being noise-robust regardless of representation, namely SVM and full BNNs. Both methods rely on inductive biases, SVM on margin maximization and BNN on weight priors. However, other models like RF and NN-$\beta$ show the opposite pattern. These models can be robust to noise, but only when paired with particular representations. The pairings differ by model. Representation itself explains only a small share of robustness variance ($\eta^2 \le 8\%$), so no single representation is decisively the most robust. PDV's distinguishing strength is instead its clean-data performance, the highest of any representation; combined with robustness that is competitive rather than exceptional, this makes it a reasonable default when working with noisy data.

*(Claim deliberately avoids a "most robust representation" ranking — that would rest on mean AUC$_\text{norm}$, which is baseline-confounded. See the note above. If you want a number, cite the small rep $\eta^2$, not a per-rep AUC$_\text{norm}$ ordering.)*

### Bayesian-transformation paragraph (L473)

**Replace:**
> By comparing models with the same core architecture, we observe that with full Bayesian transformations, representation explains less than $11\%$ of NDS variance ($p > 0.25$) while variational Bayesian inference on the same NNs is representation-dependent.

**With:**
> By comparing models with the same core architecture, we observe that with full Bayesian transformations, representation is a minor, mostly non-significant contributor to AUC$_{norm}$ variance (representation $\eta^2 \approx 11\%$ for BNN-$\alpha$), while variational Bayesian inference on the same NNs retains residual representation-dependence under some strategies (for VBLL-$\alpha$ under threshold noise, $\eta^2 = 27\%$, $p = 0.02$).

*(Rest of L473 kept verbatim: the "all 24 VBLL × {MHG-GNN, mol2vec} configurations were excluded due to baseline R$^2 < 0.6$" sentence — the count 24 is CSV-verified; see flag on the 0.6 gate below.)*

### Transformation-improvement paragraph (L475) — one sentence

**Replace:**
> Both full BNN and VBLL transformations significantly improved robustness for both NN-$\alpha$ and NN-$\beta$ (Table~\ref{tab:wilcoxon_bnn}).

**With:**
> The full BNN transformation significantly improved robustness for both NN-$\alpha$ and NN-$\beta$, while the VBLL transformation significantly improved NN-$\beta$ but not NN-$\alpha$ (Table~\ref{tab:wilcoxon_bnn}).

*(All other sentences in L475 kept verbatim — "reduced the degradation… the most", the VBLL-cost sentence, the SMILES/Additional file 8 sentence, and "QRF was significantly less robust than RF" (CSV-confirmed).)*

### Table `tab:wilcoxon_bnn` (L477–493) — paste-ready

```latex
\begin{table}[htbp]
\centering
\caption{Wilcoxon signed-rank tests for Bayesian and probabilistic transformations on the QM9 HOMO--LUMO gap. \textbf{Base} is the deterministic reference model; \textbf{Variant} is the probabilistic/Bayesian transformation tested against it. \textbf{$\Delta$ AUC$_{norm}$} = mean change in the normalised retention area from base to variant; positive values mean the variant is more robust (retains more of its baseline R$^2$ under noise). \textbf{Sig.}\ marks pairs that are statistically significant at $p < 0.05$ (asterisk).}
\label{tab:wilcoxon_bnn}
\small
\begin{tabular}{llrrl}
\toprule
\textbf{Base} & \textbf{Variant} & \textbf{$\Delta$ AUC$_{norm}$} & \textbf{$p$-value} & \textbf{Sig.} \\
\midrule
NN-$\alpha$ & BNN-$\alpha$  & $+0.031$ & $2.9 \times 10^{-11}$ & \textbf{*} \\
NN-$\alpha$ & VBLL-$\alpha$ & $+0.011$ & $0.25$ & \\
NN-$\beta$  & BNN-$\beta$   & $+0.053$ & $2.9 \times 10^{-11}$ & \textbf{*} \\
NN-$\beta$  & VBLL-$\beta$  & $\mathbf{+0.062}$ & $1.2 \times 10^{-7}$ & \textbf{*} \\
RF          & QRF           & $\mathbf{-0.012}$ & $2.9 \times 10^{-11}$ & \textbf{*} \\
\bottomrule
\end{tabular}
\end{table}
```
*(VBLL-α asterisk removed — p = 0.25, non-significant. Largest significant gain is now NN-β→VBLL-β +0.062.)*

### Figure `fig:nn_family_comparison` (L495–500) — caption unchanged

Caption plots "R$^2$ versus $\sigma$", metric-agnostic; no NDS token. **Keep verbatim.** (Flag: confirm the PNG was regenerated in `paper_figures_v2/`.)

### Interaction + second most-to-lose paragraph (L502)

**Replace:**
> While representation may not be a dominant determinant in NDS variance, the model--representation interaction term does have an impact. For SMILES, model choice explains over 91\% of robustness variance, while on PDV, model choice explains 72\%. Across all six noise strategies, PDV's strongest configurations came from models with strong inductive biases like SVM and full BNN, suggesting that PDV's compact feature space complements margin-based and Bayesian regularization. As noted above, PDV's high baseline accuracy means it degrades among the fastest of all representations by raw NDS; embeddings such as mol2vec show shallower slopes but struggle most when paired with NNs. These patterns are aligned with the importance of the model--representation interaction term, indicating that representation choice modulates noise robustness within a given architecture.

**With:**
> While representation may not be a dominant determinant in AUC$_{norm}$ variance, the model--representation interaction term does have an impact. For SMILES, model choice explains about 75\% of robustness variance on average (rising to 94\% under threshold noise), while on PDV it explains about 50\%. Across all six noise strategies, PDV's strongest configurations came from models with strong inductive biases like SVM and full BNN, suggesting that PDV's compact feature space complements margin-based and Bayesian regularization. As noted above, PDV's high baseline accuracy is retained under noise rather than penalised; embeddings such as mol2vec retain comparably but struggle most when paired with NNs. These patterns are aligned with the importance of the model--representation interaction term, indicating that representation choice modulates noise robustness within a given architecture.

### Top-10 ranking paragraph (L504)

**Replace:**
> Taking a closer look at model choice for noise robustness, we ranked all model--representation configurations on QM9 by their Gaussian NDS. Tree-based ensembles dominate the top 10: NGBoost appears five times, XGBoost and RF twice each, and LightGBM once. No neural-network configuration appears the top 10. Rankings are consistent across strategies and the top QM9 configurations maintain their advantage on ADME datasets, though SVM configurations rank higher on these ADME datasets than NGBoost.

**With:**
> Taking a closer look at model choice for noise robustness, we ranked all model--representation configurations on QM9 by their mean AUC$_{norm}$. Tree-based ensembles dominate the top 10: NGBoost appears seven times, RF twice, and XGBoost once. No neural-network configuration appears in the top 10. Rankings are consistent across strategies and the top QM9 configurations maintain their advantage on ADME datasets, though SVM configurations rank higher on these ADME datasets than NGBoost.

---

**Decisions (folded from the old figures/lit/review material):**
- **Kept** the old full-draft's table bodies (`tab:auc_ranking`, `tab:wilcoxon_bnn`) — re-verified cell-by-cell against the CSVs; they match.
- **Kept** the old draft's opener/decoupling reframe and the VBLL-α-is-10th correction — both survive verification.
- **Replaced** the old change-list's L473 "less than 8% of AUC_norm variance (p>0.25)": that used the *overall* ANOVA rep term, but the sentence is a *within-architecture* simple-effects claim — I used the verified simple-effects value (BNN-α η²≈11%) plus a verified VBLL-α threshold counter-example, which is truer to the sentence's meaning.
- **Replaced** the old change-list's "drop the 24 count / change gate to R²<0.3": the ledger + `excluded_configs.csv` confirm 24 VBLL×embedding is CORRECT, so I kept "24"; I left "0.6" unchanged (not CSV-contradicted) and flag it below.
- **Replaced** the old change-list's "drop the top-10 counts / drop SMILES 91%/PDV 72%": the ledger supplies verified replacements (NGBoost×7/RF×2/XGB×1; SMILES ~75%, PDV ~50%), so I restored concrete counts rather than vaguen them.
- **Removed** the old draft's per-strategy spread parenthetical "(from SVM at 0.956 down to VBLL-α at 0.932)" for economy — the two spread numbers plus the residual-dominance tie-in carry the point.
- **Folded, not printed:** the Additional-file-8 SMILES-benefit sentence kept verbatim (no metric token, not contradicted).

**Verification — numbers:**
- NGBoost mean AUC_norm 0.824, RF 0.818, LGB 0.817, XGB 0.814, SVM 0.814, BNN-β 0.802, BNN-α 0.801, VBLL-β 0.792, NN-α 0.789, VBLL-α 0.781, NN-β 0.756 | table2_auc_by_strategy_pdv.csv | **OK** (all per-strategy cells in table also match)
- SVM Outlier best 0.956 (0.95556) | table2_auc_by_strategy_pdv.csv | **OK**
- NGBoost leads 5 of 6 strategies (Gauss/Quant/Thresh/Hetero/Val-P; SVM wins Outlier) | table2_auc_ranks_pdv.csv | **OK**
- Kendall W = 0.9121, p = 3.55×10⁻⁸, 11 models, 6 strategies | table6_kendalls_w.txt | **OK** (paper's "0.92 / 2.7×10⁻⁸" → corrected)
- Spread outlier 0.023, threshold 0.129, value-prop 0.111 (2nd) | computed from table2_auc_by_strategy_pdv.csv | **OK** (rounded to 0.02 / 0.13 in prose)
- VBLL-α mean 0.781 below trees 0.814–0.824 | table2_auc_by_strategy_pdv.csv | **OK**
- PDV (continuous_pdv) rep-mean AUC_norm across models = 0.801 | table2_supp_auc_all_reps.csv | ⚠ **do NOT use this to rank reps.** AUC_norm is retention (baseline-normalized), so a weak low-baseline rep scores high. Local check (KIRBy hERG/LogD): binary PDV has *lower* clean-data R² than continuous PDV (0.574 vs 0.605; 0.703 vs 0.789), confirming its higher AUC_norm was a low-baseline mirage. **Binary PDV dropped.** Rep robustness claims must cite the small rep $\eta^2$, not a per-rep AUC_norm ordering.
- Wilcoxon: dnn→bnn_full +0.031 p2.9×10⁻¹¹ sig; dnn→vbll +0.011 p0.25 n.s.; mlp→bnn_full +0.053 p2.9×10⁻¹¹ sig; mlp→vbll +0.062 p1.2×10⁻⁷ sig; rf→qrf −0.012 p2.9×10⁻¹¹ sig | table3_wilcoxon_tests.csv | **OK** (paper's ∆NDS values +0.056/+0.061/+0.096/+0.124/−0.022 and VBLL-α asterisk were **MISMATCH** → corrected)
- BNN-α rep-effect η² ≈ 11% (mean 11.4%) | table1_supp_simple_effects.csv (Robustness, Rep effect, dnn_bnn_full) | **OK** *(caveat: BNN-β is 21.6% — flagged; I anchored on BNN-α)*
- VBLL-α threshold rep η² = 27.4% (→27%), p = 0.016 (→0.02) | table1_supp_simple_effects.csv | **OK**
- SMILES model-effect η² mean 75.4% (→~75%), max 93.9% under threshold (→94%); continuous_pdv 50.1% (→~50%) | table1_supp_simple_effects.csv (Robustness, Model effect) | **OK** (paper's 91%/72% were NDS-era → corrected)
- Top-10 by mean AUC_norm: NGBoost ×7, RF ×2, XGBoost ×1, no plain NN | table2_supp_auc_all_reps.csv | **OK** (paper's "NGBoost 5 / XGB+RF 2 / LGB 1" was Gaussian-NDS → corrected)
- 24 VBLL × {MHG-GNN, mol2vec} excluded (of 48 total; the other 24 are the two full-BNN variants) | excluded_configs.csv | **OK** — "24" kept
- **FLAG (unverifiable from listed CSVs):** the baseline gate "R$^2 < 0.6" — every excluded baseline is ≤0.07, so 0.6 is not *contradicted*; MEMORY notes a commit "relax robustness baseline gate to 0.3." Left "0.6" unchanged per the change-only-wrong-spans rule; author should reconcile against the pipeline's current gate constant.

**Verification — citations:** No `\cite`/`\citep`/`\citet` keys appear anywhere in L441–504 (grep returned NONE) — nothing to check. The only cross-references are internal `\ref`/`\label` (tab:auc_ranking, tab:wilcoxon_bnn, fig:nn_family_comparison) and "Additional file 5/6/7/8", all resolved within the paper.

**Out-of-section flag:** `fig:global_overview` (L437, previous subsection) still says "Noise degradation slope (NDS) heatmap … lower (more negative) NDS indicates faster degradation" — belongs to the variance-decomposition subsection's edit, but must be converted to AUC_norm for consistency.

---

All verified. Here is the consolidated section guide.

## Results — Uncertainty estimation under label noise

> **SUPERSEDED — awaits the paper restructure. Do not edit the paper from an old copy of this section.**
>
> The entire detailed rewrite that used to live here (a within-$\sigma$ detector table topped by BNN-$\alpha$ at $\rho=0.485$, a representation-gate table, a `within_sigma_uncertainty.png` figure, the "triple gate" mechanism paragraph, and the Kolmar per-sample extension) has been **deleted**. It was all built on the strategy-pooling artifact and is false. There is **no per-sample detector**: across all 143 model×representation×strategy combinations the largest within-$\sigma$ |ρ| is **0.129**, with nothing above 0.15.

**What is settled about this subsection:**
- **The per-sample tracking claim is a NULL result** and must be reported as such. No model — BNN, GP, NGBoost, QRF, VBLL — resolves which individual labels were corrupted, on any representation or noise type.
- **The one surviving uncertainty result is the population-level Kolmar link:** mean predicted uncertainty rises with $\sigma$ (Figure `fig:uncertainty_combined`). Keep this figure; re-caption it explicitly as *population-level*, not per-sample detection.
- **The GP mechanism story still explains the null:** a single global observation-noise term $\sigma^2_n$ cannot vary per sample, so the GP's high *pooled* correlation was the population trend leaking in — exactly consistent with its within-$\sigma$ $\rho\approx0$. This is now an explanation of *why detection fails*, not of why the GP "wins."

**What is NOT settled (your supervisor conversation):** how prominently the null appears, whether the subsection shrinks to a paragraph or stays a section, and how the work is distinguished from Kolmar. Until that is decided, **this section is intentionally left as a placeholder** — the surgical Replace/With blocks for the opening paragraph, tables, figure, mechanism, and Kolmar text will be rewritten to the null only after the structure is fixed.

**Minimal caption fix that is safe to apply now** (independent of the restructure — just stops Fig 6 being read as per-sample detection):
```
Replace: \caption{Uncertainty estimates under label noise on the QM9 HOMO--LUMO gap. a) Mean predicted uncertainty versus $\sigma$ for probabilistic models (PDV, Gaussian strategy). b) Aleatoric and epistemic uncertainty components versus $\sigma$ for VBLL models.}
With:    \caption{Population-level uncertainty response to label noise on the QM9 HOMO--LUMO gap (a population trend, not per-sample noise detection). a) Mean predicted uncertainty versus $\sigma$ for probabilistic models (PDV, Gaussian strategy). b) Aleatoric and epistemic uncertainty components versus $\sigma$ for VBLL models.}
```

---

## Results — Validation on experimental datasets
**Argument now:** On real ADME endpoints, noise robustness is almost entirely a model property (model η² 91.8–95.2%), and the QM9 robustness ordering transfers — SVM and NGBoost lead, XGBoost is the owned exception that collapses. Everything is restated on AUC_norm (higher = more retained); no per-sample-uncertainty claim lives in this section, so C2 does not touch it.
---

**Figure `fig:validation_overview` caption (paper.tex L556).**

Replace: `Noise degradation slope (NDS) heatmaps for three external validation datasets: a) LogD, b) Caco-2 Efflux, and c) hERG K$_i$. Lower (more negative) NDS indicates faster degradation under noise. Black cells indicate baseline R$^2 < 0.3$ for that (model, strategy) pair; ``N/A'' indicates filtered extreme values ($|$NDS$| > 2$). Models whose baseline R$^2$ fell below 0.3 across all strategies are omitted entirely, accounting for the smaller set of rows on hERG K$_i$.`

With:    `AUC$_\text{norm}$ heatmaps for three external validation datasets: a) LogD, b) Caco-2 Efflux, and c) hERG K$_i$. Lower AUC$_\text{norm}$ indicates faster degradation under noise. Black cells indicate baseline R$^2 < 0.3$ for that (model, strategy) pair. Models whose baseline R$^2$ fell below 0.3 across all strategies are omitted entirely, accounting for the smaller set of rows on hERG K$_i$.`

---

**Results paragraph (paper.tex L562).** Only the NDS/endpoint sentence is a wrong span; every other sentence in this paragraph is correct and kept byte-for-byte.

Keep verbatim: `In addition to working with ``clean'' quantum molecular properties, we tested our findings on experimentally-obtained data, replicating our process of adding artificial noise. Unlike QM9, where $\sigma = 0$ corresponds to a noise-free baseline, the experimental endpoints carry their own measurement noise in both training and test labels; the injected $\sigma$ therefore stacks on top of an unknown noise floor.`

Replace: `Per-dataset NDS heatmaps across all model architectures and noise strategies are shown in Figure~\ref{fig:validation_overview}.`

With:    `Per-dataset AUC$_\text{norm}$ heatmaps across all model architectures and noise strategies are shown in Figure~\ref{fig:validation_overview}.`

Keep verbatim (the qualitative claim): `The ANOVA on these additional datasets confirms that model architecture dominates robustness variance on all three datasets, as seen in Additional file~10, and that the trends with respect to noise robustness generalize.` — the direction (model dominates) is right, so the sentence stays. **But do NOT print the η² values (91.8/92.4/95.2) yet:**

> **⚠ PROVISIONAL — Additional file 10 (validation ANOVA) must be refit before publishing its numbers.** The forensic pass found this ANOVA is a **saturated, one-observation-per-cell design**: the residual is **0.0 by construction**, there are no F-statistics or p-values, and the roster wrongly includes QRF (QM9 excludes it). So "model η² = 91.8/92.4/95.2 with residual 0" is not a credible decomposition. Fix: refit keeping **per-CV-fold AUC$_\text{norm}$ as replicate rows** within each (model, rep) cell so a real residual and F/p exist (see "Re-run on ARC"). The qualitative "model dominates on ADME" claim can stay; the specific η² numbers wait for the refit.

Replace: `NGBoost ranks first under Gaussian noise on both QM9 and the external datasets, and remains among the most robust when results are pooled across all six strategies; on the external data SVM is marginally more robust overall (pooled mean NDS $-0.16$ versus $-0.19$), leading on hERG~K$_i$ and Caco-2 while NGBoost leads on LogD.`

With:    `NGBoost ranks first under Gaussian noise on QM9 and remains among the most robust when results are pooled across all six strategies; on the external data SVM is marginally more robust overall (best-config mean AUC$_\text{norm}$ $0.888$ for SVM/SNS versus $0.877$ for NGBoost/PDV), leading on Caco-2 Efflux while NGBoost leads on hERG~K$_i$ ($0.914$) and LogD ($0.985$).`

Keep verbatim: `Some models' predictive capabilities are limited on these external datasets, likely due to a combination of smaller training sets, narrower chemical coverage, and the unknown experimental noise in the labels. XGBoost suffers the most here, as seen in Figure~\ref{fig:validation_combined}a.` — XGBoost-worst is confirmed (mean AUC_norm 0.477–0.563; Caco-2 0.055–0.211); the sentence names no metric, so it is correct and untouched. (The old guide deleted this sentence to fold it into an expanded recut; I keep it — deleting a correct, figure-anchored sentence violates minimal-change.)

---

**Figure `fig:validation_combined` caption (paper.tex L567).**

Replace: `a) Model robustness measured by the noise degradation slope (NDS) across all four datasets; lower (more negative) NDS indicates faster degradation under noise.`

With:    `a) Model robustness measured by AUC$_\text{norm}$ across all four datasets; lower AUC$_\text{norm}$ indicates faster degradation under noise.`

---

**QRF/representation paragraph (paper.tex L571).**

Keep verbatim: `QRF was consistently less robust than RF on every external data set (Additional file~11). As seen with QM9, choice of molecular representation has a minimal effect on noise robustness across all three external datasets, reinforcing the finding that model architecture, not representation, is the main driver of noise robustness.` — direction verified (RF/PDV 0.777 > QRF/PDV 0.683) and strengthened; rep η² ≤ 4.2 confirms the minimal-effect claim. Neither sentence names NDS, so both are correct spans and kept exactly. (The old guide appended "by AUC$_\text{norm}$" here; I drop that edit — the sentence is metric-agnostic and correct, so adding words breaks HARD RULE 1.)

---

**Decisions (folded from old figures/lit/review material):**
- Kept: the AUC_norm recut of the NGBoost/SVM sentence and both figure-caption metric/direction swaps (guide L1662–1690, L775–777) — verified and reconciled into single Replace/With blocks.
- Kept: the endpoint flip (SVM→Caco-2 only; NGBoost→hERG+LogD), which the old change-list (L522) flagged; folded into the one sentence rather than a separate note.
- Removed: the guide's expanded full-draft version (L503–507) that grafted η² numbers into the correct ANOVA sentence and rebuilt the XGBoost sentence — over-edits correct prose; minimal-change keeps those sentences verbatim.
- Removed: the guide's deletion of the standalone "XGBoost suffers the most…Figure~b" sentence (L1680) — it is correct and figure-anchored, so retained.
- Removed: the guide's "by AUC$_\text{norm}$" addition to the QRF sentence (L1696) — sentence is metric-agnostic and correct.
- Removed/flagged: the guide's alternate reading "SVM leads external / NGBoost leads QM9 and LogD" (L1191) — superseded; the per-endpoint CSV gives the sharper, verified split used above.

**Verification — numbers:**
- SVM/SNS mean AUC_norm 0.888 | results/paper_figures_v2/table_validation_auc.csv (0.887920) | OK
- NGBoost/continuous-PDV mean AUC_norm 0.877 | table_validation_auc.csv (0.877103) | OK
- NGBoost/cpdv hERG-Ki 0.914 | table_validation_auc.csv (0.914059) | OK
- NGBoost/mhggnn LogD 0.985 (NGBoost LogD leader) | table_validation_auc.csv (0.984973) | OK
- SVM/SNS Caco-2 Efflux 0.810 (Caco-2 leader) | table_validation_auc.csv (0.810480) | OK
- SVM pooled mean 0.877 > NGBoost 0.857 (basis for "marginally more robust overall") | computed from table_validation_auc.csv groupby model | OK
- XGBoost mean AUC_norm range 0.477–0.563 (worst) | table_validation_auc.csv | OK
- XGBoost Caco-2 collapse 0.055–0.211 | table_validation_auc.csv (0.054767–0.210659) | OK
- RF/cpdv 0.777 > QRF/cpdv 0.683 | table_validation_auc.csv (0.776606, 0.683418) | OK
- Validation ANOVA model η² LogD 91.8 / Caco-2 92.4 / hERG 95.2; Rep ≤4.2; Residual ~0 | results/paper_figures_v2/table_validation_anova.csv | OK
- Old "pooled mean NDS −0.16 versus −0.19" | not in any CSV; NDS metric retired | MISMATCH (deleted)
- Old endpoint attribution "SVM leads hERG & Caco-2, NGBoost LogD" | table_validation_auc.csv gives the reverse on hERG | MISMATCH (corrected)

**Verification — citations:** Section L560–571 contains **no** inline `\cite/\citep/\citet` (only "Additional file" cross-refs). Adjacent dataset keys checked against citations.bib: openadmet IN-BIB; Zdrazil2023 IN-BIB; Kolmar2021 IN-BIB; wu2018 IN-BIB.

---

I have everything verified. Producing the consolidated section guide.

## Conclusion (paper.tex L573–581)

**Argument now:** After the two corrections the Conclusion must (i) name AUC$_\text{norm}$ (normalised area under the $R^2$ retention curve, higher = more robust) wherever the author wrote NDS/slope — these C1 metric fixes are settled and given below — and (ii) **delete the per-sample uncertainty-tracking claim**, which is a null result (no model tracks per-sample noise; max within-σ |ρ|=0.129). The robustness half of the conclusion stands (NGBoost/tree ensembles most robust); the uncertainty half reduces to the population-level Kolmar link plus the null. **The exact C2 wording of L579 and L581 is pending the paper restructure** — the blocks below give a safe null-based placeholder, not final prose. The C1 (metric) spans are correct and can be applied now; everything else in the author's four paragraphs stays byte-for-byte.

---

### The consolidated change guide (minimal Replace/With, author prose kept verbatim except wrong spans)

**L575 [C1 — metric only].** Only the NDS-definition span is wrong; "model architecture is the dominant factor, while representation explains less than 10\% … for most types" is CSV-consistent (all rep $\eta^2 \le 7.9\%$) and stays verbatim.

Replace: `For predictive performance degradation under label noise, we look at the noise degradation slope (NDS), defined as the slope of $R^2$ with respect to the noise scaling factor $\sigma$, and see that model architecture is the dominant factor, while representation explains less than 10\% of variance for most types of label noise.`

With:    `For predictive performance degradation under label noise, we use AUC$_\text{norm}$, the normalised area under the $R^2$ retention curve, where higher values indicate more performance retained across noise levels, and see that model architecture is the dominant factor, while representation explains less than 10\% of variance for most types of label noise.`

---

**L577a [C1 — metric name only].** Spread claim is CSV-verified (Threshold 0.129, Value-Prop 0.111 widest; Outlier 0.023 smallest); keep the sentence and its semicolon.

Replace: `Threshold and value-proportional noise produce the widest spread of NDS across models; outlier noise barely separates them.`

With:    `Threshold and value-proportional noise produce the widest spread of AUC$_\text{norm}$ across models; outlier noise barely separates them.`

---

**L577b [C1 — number only].** $W=0.92$ mismatches the pipeline's official $W=0.9121$; fix the number only. "independent" is a correct claim, kept verbatim.

Replace: `Model rankings are highly concordant across strategies (Kendall's $W = 0.92$), suggesting that a model's robustness to noise is independent of type of noise.`

With:    `Model rankings are highly concordant across strategies (Kendall's $W = 0.9121$), suggesting that a model's robustness to noise is independent of type of noise.`
(Optional, if the author wants the stat behind the adjective per MSR/paper-craft: `(Kendall's $W = 0.9121$, $p = 3.55 \times 10^{-8}$)` — both values verified from `table6_kendalls_w.txt`.)

---

**L577c [no change required].** "NGBoost and SVM, the most noise-robust models" is a qualitative ranking claim, not the metric name and not a number the CSV directly contradicts (SVM mean AUC$_\text{norm}$ .814 ties XGBoost and wins Outlier .956). Under HARD RULE 1 (do not touch correct claims) this sentence is **kept verbatim**. *Optional sharpening the author may accept for cross-section consistency with the revised Results ranking (NGBoost .824 > RF .818 > LGB .817 > XGB .814 = SVM .814): "…NGBoost and random forests, the most noise-robust models, did not perform particularly well on clean data, with SVM strongest under outlier noise and on the ADME datasets." Flagged, not forced.*

---

**L577d [C1 metric fix now; C2 uncertainty clause PENDING].** "slopes" is a forbidden metric term (RULE 4) — apply the AUC$_\text{norm}$ swap now. **But the uncertainty clause is now misleading:** it contrasts embeddings' "failure to track" against an implied success elsewhere, and under the null *no* representation tracks per-sample noise. Once the uncertainty section is restructured, this clause should drop the per-sample-tracking contrast (embeddings' weakness is simply robustness, or the clause is cut). Flagged, not yet finalised.

Replace: `SVM and full BNNs maintained consistent NDS across all representations. Embeddings (MHG-GNN, mol2vec) degraded most when paired with neural network architectures; their more decisive weakness, however, was that their per-sample uncertainty failed to track injected noise (discussed below), not their robustness slopes.`

With (C1 metric only — C2 clause pending restructure):    `SVM and full BNNs maintained consistent AUC$_\text{norm}$ across all representations. Embeddings (MHG-GNN, mol2vec) degraded most when paired with neural network architectures; their more decisive weakness, however, was that their per-sample uncertainty failed to track injected noise (discussed below), not their robustness.`

---

**L577e [no change].** `On the flip side, QRF was significantly less noise-robust than RF across datasets, suggesting that quantile regression may be overfitting to noisy labels rather than absorbing them.` — CSV-consistent (Wilcoxon rf→qrf $\Delta$AUC$_\text{norm}=-0.012$, $p=2.9\times10^{-11}$ SIG). Kept verbatim.

---

**L579 [C2 CRITICAL — the per-sample tracking claim is a NULL; the first two sentences must be deleted/rewritten. Exact wording PENDING RESTRUCTURE].** The pooled "NGBoost and Gaussian Processes produced the strongest per-sample correlation" finding is a **pooling artifact** — there is no per-sample detector (max within-σ |ρ|=0.129). Sentences 3→end (QM9 clean baseline, validation-dataset description with the two \citep, XGBoost exception, future work) are correct and kept **verbatim**.

Delete (sentences 1–2, the per-sample tracking claim): `With regard to uncertainty estimation, models that learn a separate scale or noise parameter during training, particularly NGBoost and Gaussian Processes, produced the strongest per-sample correlation between estimated uncertainty and injected noise magnitude. Non-embedding representations (fingerprints and the PDV descriptor) produced moderate-to-strong uncertainty-noise correlations, while the learned embeddings (MHG-GNN, mol2vec) produced near-zero or negative correlations for all models.`

> **PLACEHOLDER (null-based, NOT final — awaits restructure).** The safe replacement states the population link and the null, e.g.: "With regard to uncertainty estimation, mean predicted uncertainty rose with injected noise at the population level for most probabilistic models, but this aggregate trend did not translate to per-sample detection: computed within each noise level, no model's uncertainty reliably identified which individual labels had been corrupted, on any representation or noise type." Do not paste this yet — the final wording depends on how prominently the null appears after the restructure.

Unchanged remainder of L579 (keep exactly): `QM9 served as a relatively clean baseline; it is computationally-derived with negligible measurement noise. The validation datasets, LogD, Caco-2 \citep{openadmet}, and hERG Ki \citep{Zdrazil2023}, tested are smaller and cover a narrow range of the chemical and biological space, and have an unknown inherent amount of noise from experimentally-determined values, on top of the added artificial noise. These validation datasets supported the QM9 findings, with XGBoost the notable exception, degrading on the external datasets. Future work should extend this type of noise robustness benchmarking to classification tasks and larger experimental datasets with better-known uncertainty estimates.`

---

**L581 [C1 + C2 — closing paragraph. Keep the opening sentence; the rest asserts per-sample tracking (NULL) and must be rewritten. Exact wording PENDING RESTRUCTURE].** Sentence 1 is a still-true model-driven framing and is **kept verbatim** (though "noise-aware uncertainty estimation" may want softening once the null is settled). Sentences 2–4 fuse robustness with a per-sample tracking claim that is a null; the "orthogonal capabilities" framing no longer applies (there is no detection track).

Keep verbatim (sentence 1): `When working with noisy experimental data in QSAR settings, it is important to keep in mind that noise robustness and noise-aware uncertainty estimation are primarily driven by the model's mechanism of training, and to a certain extent the choice of molecular representation, usually with respect to the model.`

Delete (sentences 2–4, per-sample tracking claim): `While Bayesian transformations on neural networks improve noise robustness, they do not produce uncertainty estimates that correlate particularly strongly with noise. Models like NGBoost and GPs, which include a learned noise or scale parameter in the training objective, are the most robust to noise across both various types of noise and datasets and often produce uncertainty estimates which track per-sample label noise. Pairing these models with fingerprint-based representations provides additional benefits in restricting, and potentially detecting and mitigating label noise.`

> **PLACEHOLDER (null-based, NOT final — awaits restructure).** A safe replacement keeps only what holds: "Bayesian transformations on neural networks improve their noise robustness. The most robust models overall are NGBoost and the tree ensembles. Predicted uncertainty rises with injected noise at the population level, but it does not identify which individual labels are corrupted for any model or representation we tested, so uncertainty-based detection or mitigation of label noise at the per-sample level did not prove viable here." Do not paste yet — final wording depends on the restructure.

---

**Optional insert — Limitations paragraph (NEW).** Not a Replace/With; the guide proposes it as an insertion immediately before the closing L581 paragraph. Include only if the author wants it. **The per-sample-detection sentences (opening and closing) have been DELETED** — they cited the ρ=0.485 / VBLL-α 0.342 detector, which is the pooling artifact. What remains is verified and safe. Paste-ready:

```latex
\paragraph{Limitations.} Several constraints bound the strength of these conclusions. First, the robustness ranking is built on a tight top cluster: NGBoost (0.824), the tree ensembles (RF 0.818, LightGBM 0.817, XGBoost 0.814) and SVM (0.814) are separated by hundredths of an AUC$_\text{norm}$ unit, so the claim that these architectures are the most robust rests on the concordance of their ranks across noise strategies (Kendall's $W = 0.9121$) rather than on large pairwise gaps. Second, the external ADME datasets (LogD, Caco-2, hERG Ki) are small, cover a narrow region of chemical and biological space, and carry an unknown quantity of intrinsic experimental noise on top of the injected noise; on the noisier endpoints---most sharply Caco-2 Efflux, where XGBoost falls to a mean AUC$_\text{norm}$ near 0.05--0.21---some models degrade below usable predictive performance, so the validation results should be taken as corroboration of the QM9 trends rather than as independent benchmarks. Third, our evidence is confined to regression, on a single computational QM9 target and three ADME endpoints. Finally, the uncertainty analysis rests on single-seed runs; while this does not affect the population-level trend, a per-sample analysis with multiple seeds would be needed to place confidence intervals on any future per-sample claim.
```

> **NOTE:** the deleted first sentence ("the per-sample uncertainty analysis rests on single-seed runs … leading BNN-α/PDV/outlier value of ρ = 0.485") and the deleted last sentence (VBLL-α ρ = 0.342 dirty σ=0 control) both presupposed a real detector. With the null result they are void. If the restructure keeps a per-sample null as a stated result, add one clean sentence saying the null was checked exhaustively (143 model×rep×strategy combinations, max within-σ |ρ| = 0.129) rather than these two.

---

**Decisions (folded from the old figures/lit/review material):**
- **Kept** the guide's Replace/With list (REVISION_GUIDE.md L1706–1772) as the spine — it is minimal-change and rule-1-compatible.
- **Removed** the standalone "two-mechanism synthesis passage" (L246–270 / full-draft L539–549) as a separate insert — it over-rewrites correct author prose (L575 ANOVA sentence, L577 opening) beyond the "change only wrong spans" mandate; its orthogonality thesis is folded into the L581 rewrite instead.
- **Reverted** two guide over-edits that touch *correct* author claims: "less than 10% for most types" → the guide's "under 8% throughout" (reverted; CSV supports author's wording), and "independent" → "largely independent" (reverted, word-choice only).
- **Kept as optional** the L577c sharpening (SVM demotion) and the Kendall p-value add — both defensible but exceed strict rule 1, so flagged not forced.
- **Kept** the Limitations paragraph as an optional NEW insert (numbers verified), trimmed to remove speculative claims; author's call whether to add.
- **Not in this section:** the `fig:validation_combined` caption "noise degradation slope (NDS)" (L567) belongs to the Validation subsection, not the Conclusion — flag for that section's owner.

**Verification — numbers:**
- NGBoost mean AUC$_\text{norm}$ 0.824 | table2_auc_by_strategy_pdv.csv (0.823966) | OK
- RF 0.818 / LGB 0.817 / XGBoost 0.814 / SVM 0.814 | table2_auc_by_strategy_pdv.csv (.817719/.816752/.814395/.813671) | OK
- SVM Outlier 0.956 | table2_auc_by_strategy_pdv.csv (0.955555) | OK
- Kendall $W=0.9121$, $p=3.55\times10^{-8}$, 11 models, 6 strategies | table6_kendalls_w.txt | OK
- ~~BNN-$\alpha$ within-$\sigma$ $\rho=0.485$ outlier, cont-PDV, $\sigma=0.6$~~ | ❌ **VOID — this was the strategy-pooling artifact.** The 0.485 came from `fix_injected_noise` regressing all six noise strategies together (it omitted `strategy` from the group key). Recomputed correctly (within σ, one strategy at a time), BNN-α on cont-PDV under outlier is **not** a detector; the largest within-σ |ρ| *anywhere across all 143 model×rep×strategy combinations* is 0.129. Do not publish 0.485 or any BNN "leads" direction — the finding is a null.
- GP call-out is grounded LOCALLY: across the reps GP appears in the local aggregate, pooled $\rho$ is high but within-$\sigma$ collapses to $\approx 0$ | table4_uncertainty_metrics.csv: GP pooled up to 0.535 (sns) / 0.527 (morgan); within-$\sigma$ at $\sigma=0.6$ is −0.056 (ecfp4), −0.015 (morgan), −0.041 (pdv, **binary**), −0.043 (sns) | OK — reproducible proof that GP's pooled signal is a population artifact. The mechanism (input-geometry/global-noise variance, label-blind) predicts within-$\sigma$ $\approx 0$ on **every** representation, so the paper-text GP claims are stated representation-agnostically and hold regardless.
- ⚠ **GP on continuous_pdv — do NOT claim it "was never run"; VERIFY on the server.** GP/gauche is ABSENT from the *local* `table4_uncertainty_metrics_continuous_pdv.csv`, but the other models from the same 2026-03-03 uncertainty batch (BNN-α/β, NGBoost, QRF, VBLL-α/β) ARE present — so this is not wholesale staleness. A gauche × continuous_pdv uncertainty job was submitted (**`unc_cpdv_gauche.sh`, job 11442993**), and on continuous features gauche uses an RBF kernel (ANOVA-invalid, uncertainty-valid). Two live possibilities: the gauche/cpdv run **failed or produced no usable output** (memory's `gp_gauche_data_gap.md` documents exactly a gauche-RBF-on-continuous_pdv gap), OR it **landed after this aggregate was built**. Resolve by checking the server output for 11442993 and the re-run: **if a GP row appears on continuous_pdv**, its within-$\sigma$ $\rho$ is expected $\approx 0$ (mechanism is representation-agnostic) — report it as the direct same-representation GP-vs-BNN comparison, which strengthens rather than changes the conclusion.
- Per-strategy AUC$_\text{norm}$ spread widest Threshold/Value-Prop, narrowest Outlier | table2_auc_by_strategy_pdv.csv (max−min: Thresh 0.129, ValProp 0.111, Outlier 0.023) | OK
- rf→qrf Wilcoxon $\Delta=-0.012$, $p=2.9\times10^{-11}$ (L577e support) | ledger table3_wilcoxon_tests.csv | OK (not re-opened; ledger-cited, no CSV cell printed in prose)
- Limitations: RF 0.818/LGB 0.817/XGB 0.814/SVM 0.814 | table2_auc_by_strategy_pdv.csv | OK
- Limitations: XGBoost Caco-2 collapse 0.05–0.21 | table_validation_auc.csv (xgb Caco2: cpdv 0.0548, sns 0.1075, mhggnn 0.2107, ecfp4 0.2076) | OK
- ~~Limitations: VBLL-$\alpha$ outlier $\rho=0.342$~~ | ❌ **VOID — same pooling artifact** (the `within_sigma_panelA_*` CSV was built on the strategy-pooled `fix_injected_noise` output). No VBLL detector; deleted from the Limitations paragraph.
- (context, not printed) Validation best SVM/sns .888, SVM/mhggnn .879, NGBoost/cpdv .877; RF .777 > QRF .683 on cpdv | table_validation_auc.csv | OK

**Verification — citations:**
- openadmet | IN-BIB
- Zdrazil2023 | IN-BIB

---

## Abbreviations; Availability of data and materials; Description of additional data files

**Argument now:** These are reference/plumbing sections — they must silently carry the AUC$_\text{norm}$ metric name (never NDS) and the C2 within-$\sigma$ framing in the one uncertainty-supplement caption, while every factual pointer (repo, datasets, file contents) stays a plain, number-free attribution. Nothing here argues a finding; it must not leak "we changed the metric."

---

### Abbreviations (L602)

Delete the NDS entry (the metric no longer exists in the paper) and add an AUC$_\text{norm}$ entry in alphabetical position (after ANOVA, L587, before BNN, L588).

Replace: `    \item[NDS:] Noise degradation slope`
With:    *(delete this line entirely)*

Insert after L587 (`\item[ANOVA:] Analysis of variance`):
```latex
    \item[AUC$_\text{norm}$:] Normalised area under the $R^2$ retention curve
```

---

### Availability of data and materials (L624)

Only the metric parenthetical changes; every other clause is kept byte-for-byte.

Replace: `The NoiseInject benchmarking framework, implementing all six regression and six classification noise injection strategies together with the robustness metrics (noise degradation slope and retention) and uncertainty-calibration metrics (ECE, coverage at $1\sigma$/$2\sigma$, mean interval width, and uncertainty--error / uncertainty--noise correlation) described in this work, is available as an open-source Python package under an MIT license \citep{noiseinject}.`
With:    `The NoiseInject benchmarking framework, implementing all six regression and six classification noise injection strategies together with the robustness metric (AUC$_\text{norm}$, the normalised area under the $R^2$ retention curve) and uncertainty-calibration metrics (coverage at $1\sigma$/$2\sigma$, mean interval width, and uncertainty--error / uncertainty--noise correlation) described in this work, is available as an open-source Python package under an MIT license \citep{noiseinject}.`

The rest of the subsection (project-metadata itemize L626–635, dataset-provenance paragraph L637) is correct and untouched — verbatim.

---

### Description of additional data files (L662–675)

Files 1, 10, 12 name no metric symbol and contain no NDS/C2 span → **kept verbatim.** (File 10's "$\eta^2$ contributions from model, representation, and interaction" is metric-neutral; its AUC$_\text{norm}$ regeneration is the validation owner's job, no caption edit here.)

**File 2 (L664):**
Replace: `    \item[Additional file 2 (PDF):] \textit{Pairwise Spearman rank correlations between model NDS profiles.} Model--model NDS correlations ($\rho \geq 0.95$) on QM9; flags pairs excluded from the ANOVA for near-redundancy.`
With:    `    \item[Additional file 2 (PDF):] \textit{Pairwise Spearman rank correlations between model AUC$_\text{norm}$ profiles.} Model--model AUC$_\text{norm}$ correlations ($\rho \geq 0.95$) on QM9; flags pairs excluded from the ANOVA for near-redundancy.`

**File 3 (L665):**
Replace: `    \item[Additional file 3 (PDF):] \textit{Pairwise Spearman rank correlations between representation NDS profiles.} Representation--representation NDS correlations on QM9; flags pairs excluded from the ANOVA.`
With:    `    \item[Additional file 3 (PDF):] \textit{Pairwise Spearman rank correlations between representation AUC$_\text{norm}$ profiles.} Representation--representation AUC$_\text{norm}$ correlations on QM9; flags pairs excluded from the ANOVA.`

**File 4 (L666):**
Replace: `    \item[Additional file 4 (PDF):] \textit{Intraclass correlation coefficients (ICC(1,1)) for model pairs.} ICC(1,1) on QM9 NDS profiles for pairs with ICC $\geq 0.5$, with Family-grouping (Tree, $\alpha$/$\beta$, $\alpha$, Cross).`
With:    `    \item[Additional file 4 (PDF):] \textit{Intraclass correlation coefficients (ICC(1,1)) for model pairs.} ICC(1,1) on QM9 AUC$_\text{norm}$ profiles for pairs with ICC $\geq 0.5$, with Family-grouping (Tree, $\alpha$/$\beta$, $\alpha$, Cross).`

**File 5 (L667):** gate threshold moves with the metric (retention ratios go unstable near a zero clean-label denominator, so the gate is baseline $R^2<0.3$, not $<0.6$; the v2 supplement is the 0.3-gate artifact = 48 configs).
Replace: `    \item[Additional file 5 (PDF):] \textit{Configurations excluded from robustness analysis.} Model$\times$representation cells removed because baseline R$^2 < 0.6$ on QM9, summarized by number of strategies excluded.`
With:    `    \item[Additional file 5 (PDF):] \textit{Configurations excluded from robustness analysis.} Model$\times$representation cells removed because baseline R$^2 < 0.3$ on QM9, summarized by number of strategies excluded.`

**File 6 (L668):** metric only — **ECFP4 kept verbatim** (the topological rename is outside the C1/C2/number mandate; see decisions).
Replace: `    \item[Additional file 6 (PDF):] \textit{Global overview of noise robustness on the ECFP4 representation.} Performance degradation curves and the NDS heatmap across all model architectures and noise strategies, for ECFP4 on QM9.`
With:    `    \item[Additional file 6 (PDF):] \textit{Global overview of noise robustness on the ECFP4 representation.} Performance degradation curves and the AUC$_\text{norm}$ heatmap across all model architectures and noise strategies, for ECFP4 on QM9.`

**File 7 (L669):** relabel metric; the L443 sentence that cites this file survives (rewritten to keep the "Additional file~7" reference), so the file stays — see flag in decisions.
Replace: `    \item[Additional file 7 (PDF):] \textit{Baseline R$^2$ versus NDS scatter (PDV, Gaussian).} Per-model scatter of baseline predictive performance against NDS on QM9 (PDV, Gaussian noise).`
With:    `    \item[Additional file 7 (PDF):] \textit{Baseline R$^2$ versus AUC$_\text{norm}$ scatter (PDV, Gaussian).} Per-model scatter of baseline predictive performance against AUC$_\text{norm}$ on QM9 (PDV, Gaussian noise).`

**File 8 (L670):**
Replace: `    \item[Additional file 8 (PDF):] \textit{Neural-network Bayesian transformation effects by representation.} Wilcoxon signed-rank tests on QM9 showing how BNN and VBLL transformations change NDS for each representation.`
With:    `    \item[Additional file 8 (PDF):] \textit{Neural-network Bayesian transformation effects by representation.} Wilcoxon signed-rank tests on QM9 showing how BNN and VBLL transformations change AUC$_\text{norm}$ for each representation.`

**File 9 (L671):** C2 touch — the uncertainty–noise correlation is now measured within each fixed $\sigma$; qualify that one metric accordingly. **ECFP4 kept verbatim** (topological rename out of scope).
Replace: `    \item[Additional file 9 (PDF):] \textit{Uncertainty quantification metrics for probabilistic models on the ECFP4 representation.} Unc-Noise $\rho$, Unc-Error $\rho$, ECE, and coverage at $1\sigma$/$2\sigma$ for probabilistic models under Gaussian noise on QM9.`
With:    `    \item[Additional file 9 (PDF):] \textit{Uncertainty quantification metrics for probabilistic models on the ECFP4 representation.} Within-$\sigma$ Unc-Noise $\rho$, Unc-Error $\rho$, and coverage at $1\sigma$/$2\sigma$ for probabilistic models under Gaussian noise on QM9.`

**File 11 (L673):**
Replace: `    \item[Additional file 11 (PDF):] \textit{RF vs QRF on validation datasets.} NDS comparison of random forest and quantile regression forest on the three external validation datasets.`
With:    `    \item[Additional file 11 (PDF):] \textit{RF vs QRF on validation datasets.} AUC$_\text{norm}$ comparison of random forest and quantile regression forest on the three external validation datasets.`

---

**Decisions (folded from the old figures / change-list / paper-voice material):**
- **KEPT** the REVISION_GUIDE's paste-ready per-line Replace/With for additional files 2,3,4,7,8,11 (L1808–1841) verbatim — pure NDS→AUC$_\text{norm}$ swaps, no scope conflict.
- **KEPT** the guide's file-5 gate change 0.6→0.3 (L1820) and the availability metric-parenthetical touch (L106); both are C1 metric-plumbing tied to the v2 supplement.
- **REPLACED** the guide's file-6 and file-9 "With" text: I **reverted the ECFP4→topological rename** the guide applied there, because that rename is not a C1/C2/number correction and my mandate forbids touching non-wrong spans (the guide's own audit at L1349 flags this rename as inconsistently applied). Changed only the metric (file 6) and added the within-$\sigma$ qualifier (file 9, C2).
- **REMOVED** nothing from the abbreviation list beyond the now-dead NDS entry; **ADDED** an AUC$_\text{norm}$ entry the guide (L105) left optional — added it because the metric symbol now appears in captions and the availability text.
- **FLAG (folded from guide L1338/L1363):** Additional file 7 is retained (the L443 sentence citing it survives in rewritten form), so no orphan — but the underlying supplement PDF must be regenerated on the AUC$_\text{norm}$ scale, as must files 2,3,4,6,8,11. Caption text is fixed here; PDF regeneration is a separate author action.

**Verification — numbers:**
- Excluded-configs count 48 (v2 gate R²<0.3) | results/paper_figures_v2/excluded_configs.csv (48 rows; all four NN-Bayesian variants × {mhggnn, mol2vec} × 6 strategies, baselines −0.01 to −0.09) | **OK**
- Old count 66 (superseded R²<0.6 gate) | results/paper_figures/excluded_configs.csv (66 rows) | **OK** (context only; file 5 caption prints no count, only the 0.3 threshold)
- File 5 gate threshold R²<0.3 | consistent with v2 excluded_configs.csv (all excluded baselines ≪ 0.3) | **OK**

**Verification — citations** (all keys appearing in this section's Availability paragraph):
- noiseinject | **IN-BIB**
- qm9_dataset | **IN-BIB**
- Ramakrishnan2014 | **IN-BIB**
- openadmet | **IN-BIB**
- Zdrazil2023 | **IN-BIB**
- lhasa | **IN-BIB**
- landrum2024 | **IN-BIB** (cited in Methods L211, not this section; verified for completeness)

---

# Reference — status of every number, citations, figures, and the ARC re-run

## Status of every result (from the forensic pass)

Every value below was traced to the code and reproduced from the committed v2 CSVs unless marked ⚠. Verdicts: **OK** = reproduced exactly, safe to write in; **RELABEL** = value fine, description/label needs fixing; **⚠ RE-RUN** = not certifiable from local data, needs the ARC re-run before publishing.

| Result | Verdict | Note |
|---|---|---|
| Performance ANOVA (24 cells) | **OK** | Byte-stable; interaction dominates all 6; only relabel co-located "NDS" name |
| Robustness ANOVA — Gaussian, Quantile | **OK** | Balanced design; reproduced exactly |
| Robustness ANOVA — Threshold, Value-Prop | **RELABEL** | Values fine; method is weighted-marginal SS (= Type I only under balance), not "Type I sequential" |
| Robustness ANOVA — **Heteroscedastic, Outlier** | **OK (confirmed 2026-08-14)** | Residual dominance (77.4 / 83.6) is REAL, not a roster artifact — roster is a consistent 7 models; 11-model roster barely changes it (82.8→83.1, 85.5→85.9); stable across SS-type. Residual = run-to-run variance (cell-means: Model 55%/71%). |
| AUC ranking (0.824…0.756), ranks, spreads, SVM outlier 0.956 | **OK** | Reproduced exactly |
| Kendall W = 0.9121, p = 3.55×10⁻⁸ | **OK (label)** | Rounds to **0.91**; basis is **across-representation mean, n=11** (state it — the PDV-only table it sits beside would give 0.9374) |
| Wilcoxon table (5 rows) | **OK** | ΔAUC$_\text{norm}$; **VBLL-α is non-significant (p=0.252)** → "both transformations improve both networks" is false |
| Interaction Spearman ρ=0.82, p=0.002, **n=11** | **OK** | Paper's 0.73/12 and the draft's 0.86 are both wrong |
| ~~Within-σ tables + BNN-α 0.485 lead~~ | ❌ **VOID (artifact)** | Was strategy-pooled in `fix_injected_noise`. Per-strategy, within-σ: **no detector** — max |ρ| across all 143 combos = 0.129. Per-sample tracking is a **null result**. |
| ~~VBLL-α within-σ 0.342~~ | ❌ **VOID (artifact)** | Same pooling artifact; deleted. |
| Simple-effects SMILES ~75% / PDV ~50% | **OK** | Both vary by strategy (SMILES 23–94%, PDV 19–69%); the paper's 91%/72% were slope-era |
| Top-10 (NGBoost 5 / RF 2 / SVM 1 / LGB 1 / XGB 1, no NN) | **OK** | By mean AUC$_\text{norm}$; paper's "by Gaussian NDS" counts were different |
| Excluded configs = 48 (24 VBLL + 24 BNN, gate 0.3) | **OK** | Paper's "66 at R²<0.6" is the old gate |
| Validation AUC leaderboard, XGBoost collapse, RF>QRF | **OK** | RF/PDV 0.777 > QRF/PDV **0.683** (verified from CSV); endpoint leaders were backwards in the paper (NGBoost leads hERG Kᵢ + LogD; SVM leads only Caco-2) |
| **Validation ANOVA (91.8 / 92.4 / 95.2, residual 0.0)** | **⚠ RE-RUN** | Saturated one-obs-per-cell design → residual is 0 by construction, no F/p; not credible as printed. Refit with per-fold replicates. |
| ICC / redundancy supplements | **OK** | Reproduced; captions need metric relabel |

## Figures to regenerate on the AUC$_\text{norm}$ scale

`fig1_global_overview.png`, `fig2_anova_decomposition.png`, `fig_interaction.png` (annotate **ρ = 0.82**), `fig_validation_overview.png`, `fig_validation_combined.png`, the ECFP4 supplement, and the validation-ANOVA supplement. **Re-caption only** (no regen): `fig_uncertainty_combined.png` → population-level. **Do NOT add** `within_sigma_uncertainty.png` — it displayed the pooling artifact (per-sample detection is a null; there is no per-sample figure to add). Metric-agnostic (no change): `fig_nn_family_comparison.png`, `fig_methods_noise_strategies.png`.

## Citation audit (`citations.bib`, 193 entries, 51 keys used)

- **Build-breaking:** `\bibliography{sn-bibliography}` → the file doesn't exist; change to `\bibliography{citations}` or every `\cite` renders `[?]`.
- **Cited but missing (add):** `Rogers2010` (ECFP/Morgan), `Islam2019`.
- **Duplicate key:** `Xu2019` defined twice (uncited; rename if used).
- **New methods citations:** already present — reuse `Kolmar2021`, `kendall2017` (lowercase), `Northcutt2021`. Need adding — `Hendrycks2019`, `Depeweg2018`, `Kohler2019`, `Kievit2013`, `Simpson1951`, `Robinson1950`.

## Re-run on ARC (your step — I have no server access)

The contaminated/uncertain numbers need `generate_paper_figures_v2.py` re-run on `gateway.arc.ox.ac.uk`. Base: `/data/stat-cadd/scat9264/qsar_qm_models`, branch `additional_reps`, `--account=stat-cadd`.

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models && git branch      # confirm additional_reps
# full regen — certifies most CSVs match what this guide used
python scripts/generate_paper_figures_v2.py --qm9-dir results --output-dir results/paper_figures_v2
# the two reproducers (already pushed to scripts/) settle the ANOVA questions:
python scripts/reproduce_robustness_anova.py --results-dir results \
    --csv results/paper_figures_v2/table1_anova_summary.csv     # prints true post-gate roster + OK/MISMATCH per cell
python scripts/robustness_anova_sensitivity.py --qm9-dir results  # SS-type / roster / gate sweep
```

Outstanding items:
1. ~~**Heteroscedastic + Outlier robustness ANOVA — roster**~~ ✅ **RESOLVED 2026-08-14.** The reproducers confirm the roster is a consistent 7 models across all six strategies (not 9/10); the residual-dominance is real and robust to roster and SS-type. No re-run needed — all six ANOVA rows are final. (Phrase the residual as run-to-run variance, not "models don't matter" — cell-means gives Model 55%/71%.)
2. **Validation ANOVA** — still open: refit keeping per-CV-fold AUC$_\text{norm}$ as replicate rows per (model, rep) cell, so a real residual and F/p exist (current 91.8/92.4/95.2 with residual 0.0 is structurally invalid).
3. **Per-sample uncertainty — CODE FIX, not a re-run.** The 0.485 "detector" was an artifact of `fix_injected_noise` omitting `strategy` from its group key (it pooled all six noise strategies into one regression). Fix the group key to include `strategy`, then the within-σ per-strategy correlations are ~0 for every model (max |ρ|=0.129 across all 143 combos) — a null result. No extra seeds are needed to establish the null; seeds would only matter if a future per-sample analysis found real signal (none exists here).

---

# §9 — PATTERN FINDINGS (added 2026-08-19, grounded via workflows wz5d8h5jy + wb1ogemra)

> These are the deeper robustness patterns. Prose below is paste-ready in spirit; every **[REP: TBD]** marks a representation the AUTHOR must pick (see DISCUSSION_TRACKER D-rep). Items marked ⚠PENDING did NOT reconcile with the aggregate CSVs — do NOT write them until resolved.

## 9.1 The metric reframe — "decoupling" is a mirage, not a result (F1/F2/F5) — SUPERSEDES T4/T13
AUC_norm = (1/σ_max)∫ R²(σ)/R²(0) dσ divides out R²(0), so **retention is baseline-free BY CONSTRUCTION** — that is arithmetic, not a finding. The QM9 within-Gaussian Pearson(baseline, auc_norm)=+0.046 only confirms the identity; it does NOT show delivered performance is baseline-free.
- **Do NOT** tell the author to "strengthen the decoupling" (reverses old T4/T13). **Do NOT** cite "NGBoost lowest baseline yet most robust" as *evidence for* decoupling — that pattern IS the artifact.
- **Replacement for L433/L460:** "Because AUC_norm reports the *fraction* of clean-label R² retained rather than its absolute level, it is by construction independent of baseline accuracy. Retention and delivered accuracy are therefore distinct axes: a high AUC_norm certifies graceful degradation, not strong absolute performance under noise, which still scales with the clean-data baseline (delivered R² ≈ baseline × retention)."
- Grounding: AUC_norm definition (L241-247); NGBoost mean auc 0.824 #1 with LOWEST baseline 0.671; delivered-tracks-baseline is INFERRED on QM9 (abs-R² server-blocked), GROUNDED on validation (baseline↔abs-R² Pearson +0.75).

## 9.2 State what AUC_norm captures (F2)
Add one Methods sentence after the AUC_norm definition: "AUC_norm quantifies how gracefully a model degrades, independent of how well it performs; interpret it alongside baseline R², since absolute accuracy under noise is approximately their product." Consequence to carry through Results: **tab:auc_ranking, Kendall's W (0.9121), and the robustness ANOVA all decompose *fragility*, not delivered accuracy.** Whenever a model is called "most robust," make clear it is a retention rank.

## 9.3 ANOVA residual-dominance = ceiling-strategy mechanism (F3) — augments L380
L380 reports Outlier 83.6 / Hetero 77.4 residual and calls it "run-to-run variance" but omits WHY. Add: "— because these two strategies barely degrade any model (mean retention >0.9), there is little between-model variance to attribute, so stochastic replicate variance dominates the decomposition rather than the models being genuinely indistinguishable." Ties the residual directly to the ceiling tier (§9.6). (Also fix caption L387 "NDS"→AUC_norm.)

## 9.4 NGBoost is retention-flattered (F5) — qualify the crown at L164/L443/L577
Keep NGBoost as most robust (it genuinely degrades most gracefully) but qualify everywhere: "NGBoost retains the highest fraction of its clean-data performance, though its low baseline R² (0.671) means high retention does not make it the top performer in absolute terms under noise" (validation: LogD delivered #7/8, hERG #5/6; QM9 delivered server-blocked → INFERRED).

## 9.5 NEW results for §4.4 Validation (F6/F7/F8) — need rep + data decisions
- **F6 model-family × noise-type selectivity:** threshold/value-prop preferentially kill boosting (XGBoost worst, goes negative) and QRF; SVM resists (greedy residual-fitting chases structured noise; margins/smoothing average it out). Gaussian hits DNN+SVM hardest. Outlier only bites low-baseline Caco-2. **DATA FACTS (verified 2026-08-19, corrects an agent error):** ALL 8 models × 4 reps WERE run on validation (990 rows each) — NO data gap. SVM/DNN are on PDV too, but DIVERGE on specific hard cells (SVM PDV-hERG baseline −1.9e7; DNN diverges on MHG-GNN broadly + PDV-hERG) and get baseline-gated → that is a model-STABILITY issue, not a missing run. **GP is PDV-ONLY by design** (RBF kernel on continuous features; Tanimoto would be needed for fingerprints — never run there), and weak on hERG (baseline 0.04). **Rep choice for F6:** on a fingerprint rep (ECFP4/SNS/MHG-GNN) the kernel family = **SVM only** (no GP); on PDV you get GP+SVM but only on Caco-2+LogD (both kernels unusable on hERG). **[REP: ECFP4] recommended for STABILITY** (fewest divergent cells), not availability.
- **F7 dataset-size-dependent fragility:** boosting fragility is validation-only, vanishes on large clean QM9 (there NNs lag) → "model choice matters far more on small real-world data." Frame widening spread as size AND intrinsic noise (Caco-2 noisiest), not size alone.
- **F8 floor/ceiling profiles:** RF/LightGBM = high floor / safe default (graceful under every strategy); SVM = high ceiling / low floor (top only with the right rep). Back with a validation-[REP: ECFP4] table: MEAN, STD, FLOOR(worst strategy), CEILING(best), BASELINE R² per model.

## 9.6 Strategy three-tier hierarchy (F11) — extends L383
Universal on QM9 (validation STRESS tier partial): **CEILING** = Heteroscedastic + Outlier (retention >0.9, no discrimination); **MILD** = Gaussian + Quantile; **STRESS** = Threshold + Value-Prop, which gut performance by destroying the **high-|y| tail** that carries most of the R² signal. Add hetero-as-ceiling explicitly (currently the paper only notes threshold/valprop as harshest).

## ⚠ PENDING RECONCILIATION — do NOT write these until resolved
- **F4 (SVM rep-dependence):** the pattern-hunt said "SVM most rep-dependent, sign-flips negative on PDV" and I flagged L573 as WRONG. **This does NOT reproduce in the aggregate CSVs** — those show SVM has the SMALLEST rep-spread (0.016 QM9 / 0.024 validation) and is POSITIVE on PDV (threshold 0.62, valprop 0.68). The "collapse" came from raw divergent cells. **⇒ do NOT reverse L573; likely only NDS→AUC_norm.** Needs per-config source check.
- **F9 (cross-dataset rank agreement):** the "hERG↔LogD Spearman +0.09" does NOT reproduce — aggregate is +0.79 to +0.93; the near-zero/negative agreements appear ONLY at per-strategy×per-rep granularity under ceiling strategies. Report the grounded pair (aggregate +0.79–0.93; ceiling-tier collapse), NOT +0.09.
- Note: one draft agent claimed "value-proportional never run on validation" — that is FALSE (valprop IS in the validation data); ignore that claim.

## 9.7 GP / SVM kernels are representation-tied — comparisons must be kernel-aware (added 2026-08-19, verified from code)
- **There is ONE GP model (`Gauche`, models.py L1690-1789), not two.** The kernel is a parameter (default `tanimoto`); the results label is set by the kernel alone (`model_name = 'gauche_rbf' if kernel=='RBF' else 'gauche'`, L1787). So `gauche` (fingerprint/Tanimoto kernel) and `gauche_rbf` (RBF kernel) are the SAME model with different kernels. RBF is used on continuous PDV only because **Tanimoto is mathematically defined only on fingerprint vectors** — it cannot run on PDV. Consequence for the paper: GP/PDV's strength is partly the **RBF kernel**, not "representation luck," so any GP comparison across reps is also a comparison across kernels and must say so.
- **Validation GP = RBF on PDV ONLY** (KIRBy L922, `GaussianProcessGauche(kernel='rbf')`, GP added only for PDV). **Tanimoto-GP (`gauche`) was NEVER run on validation fingerprints** — that IS a genuine validation gap (the QM9 study has it; validation doesn't). Fillable by a re-run.
- **SVM uses FIXED RBF on every rep, both studies** (KIRBy L913-914 `SVR(kernel='rbf')`; qsar models.py L1454 default 'rbf', tuned only among rbf/poly/sigmoid). ⇒ SVM cross-rep comparisons are FAIR (same kernel) — this is why "SVM rep-consistent" holds and L573 stands. **⚠ DISCREPANCY:** paper L197 / Additional file 12 says SVM used a **Tanimoto kernel for binary representations, RBF for continuous** — the code does NOT do this (SVM is RBF throughout). Reconcile: either the paper's SVM-kernel sentence is wrong (it may have conflated GP's kernel scheme), or a Tanimoto-SVM path must be located. Do NOT leave the paper claiming rep-specific SVM kernels if the code used fixed RBF.
- **Author's open question (logged):** GP defaulted to Tanimoto on fingerprints; whether RBF-GP should also be run on fingerprints (to separate kernel from representation) is a modeling decision → part of the clean-kernel re-run.

## 9.8 SVM kernel fix + GP-into-ANOVA re-run plan (2026-08-19, code-grounded)
- **PAPER FIX (no re-run):** L197/Add. file 12 "Tanimoto kernel for SVM on binary reps" is WRONG and **contradicts your own L262** ("RBF was used across all representations" for SVM). Code confirms L262 (models.py L1454 default rbf; KIRBy L914 fixed rbf; KIRBy comment: SVM fixed-RBF "free of the kernel–representation confound"). ⇒ change L197 to state SVM used **RBF across all representations**. **SVM's ANOVA inclusion is UNCHANGED** — it is included *because* it is RBF-everywhere; the fix removes an internal contradiction and strengthens the ANOVA justification.
- **GP is ANOVA-excluded ONLY because of the kernel/rep split** (gauche=Tanimoto on fingerprints, gauche_rbf=RBF on PDV; no single kernel spans all reps). QM9 ran `-m gauche -r ecfp4 pdv sns smiles randomized_smiles` (Tanimoto on fingerprints; RBF on PDV). Validation gates GP to PDV+RBF only (KIRBy L920).
- **RE-RUN DESIGN (author-approved direction):** run **RBF-GP on ALL reps, QM9 + validation** → GP becomes one consistent cross-rep model → **includable in the ANOVA like SVM**. QM9 already has gauche_rbf on PDV, so ADD RBF-GP on the fingerprint reps; validation: run GP on ECFP4/SNS/MHG-GNN (lift PDV gate), RBF. PLUS keep **Tanimoto-gauche on fingerprints** (QM9 has it; add on validation) as the RBF-vs-Tanimoto head-to-head. If RBF≈Tanimoto on fingerprints → commit to RBF everywhere, fold GP into ANOVA. Commands: TBD (locate --kernel flag in training entry point first).

---

# §10 — SIGMA SELECTION + METRIC EVIDENCE DUMP (2026-08-20)

Raw material, added without restructuring. Everything here is locally re-derived; provenance stated per block. Backing CSVs in `scratchpad/`.

## 10.1 What σ physically means — the units problem

Noise is injected as `y_noisy = y + σ·N(0,1)` in the label's **raw units** (`NoiseInject/noiseInject/core.py:68`, `_legacy`). Only `X` is standardised; `y` is passed raw (`KIRBy/tests/alternative_data_noise_robustness.py:750-760`). QM9 uses the identical grid (`--sigma 0.0 0.1 … 1.0` in `slurm_scripts_*`).

Label SD back-computed as `RMSE²/(1−R²)` from the σ=0 rows (n = 1219–1362 per validation dataset; QM9 from 292 local `mol2vec_investigate_*.csv` rows):

| dataset | label SD | σ=0.5 | σ=0.6 | σ=0.8 | σ=1.0 |
|---|---|---|---|---|---|
| OpenADMET-LogD | 1.191 | 0.42× | 0.50× | 0.67× | 0.84× |
| ChEMBL-hERG-Ki | 0.905 | 0.55× | 0.66× | 0.88× | 1.10× |
| OpenADMET-Caco2_Efflux | 0.434 | 1.15× | 1.38× | **1.84×** | 2.30× |
| QM9 | 1.051 (IQR 1.041–1.065 ⇒ effectively standardised) | 0.48× | 0.57× | 0.76× | 0.95× |

**Consequence for the paper as it stands:** a fixed σ is *not* a fixed noise condition. Caco-2 receives ~2.7× the noise-to-signal of LogD at any given σ. Cross-dataset comparisons at one σ are not like-for-like, and this is the mechanism behind Caco-2 supplying most of the negative-R² cells (it is not purely model fragility). One honest sentence is required somewhere in Methods or the validation section.

⚠ In-house fix exists but was **not used**: `KIRBy/src/kirby/noise_spec.py` expresses level as a fraction of label SD and binary-searches the raw σ. Its own docstring: *"This is what makes a level comparable across datasets; the four ad-hoc definitions never were."* The paper run hardcodes raw `SIGMA_LEVELS` (L79).

## 10.2 σ selection — the discrimination evidence (superseded as the *method*, kept as evidence)

Validation, fold-averaged, baseline-gated ≥0.3 (709 configs). "Separation" = IQR of R² across models within a dataset×rep×strategy cell. "Baseline echo" = Spearman of R²@σ against R²@0.

| σ | separation | echo of baseline ranking | %R²>0 | %diverged (R²<−1) |
|---|---|---|---|---|
| 0.0 | 0.068 | — | 100.0 | 0.0 |
| 0.2 | 0.069 | 0.921 | 99.4 | 0.3 |
| 0.4 | 0.075 | 0.860 | 99.2 | 0.3 |
| 0.5 | 0.086 | 0.833 | 98.6 | 0.3 |
| 0.6 | 0.124 | 0.810 | 94.2 | 0.7 |
| 0.7 | 0.142 | 0.791 | 92.1 | 0.4 |
| 0.8 | 0.177 | 0.760 | 87.7 | 0.7 |
| 0.9 | 0.221 | 0.737 | 82.9 | 1.0 |
| 1.0 | 0.257 | 0.708 | 79.1 | 3.0 |

- **Below σ≈0.5 the experiment is uninformative**: separation (0.066–0.086) is no larger than at σ=0 (0.068), and the ranking echoes the clean ranking at ρ ≥ 0.83.
- **Ranking is locally stable**: adjacent-σ Spearman ≥ 0.947 throughout (0.7↔0.8 = 0.982). So no σ choice is knife-edge — but the choice matters across the range (σ=0.2 vs 1.0 = 0.801).

**⚠ Heteroscedastic noise never separates models at any σ** — separation flat at 0.061–0.071 from σ=0 to σ=1.0, median R² only 0.485 → 0.400. Per-strategy separation at σ=0.8: threshold 0.390, valprop 0.219, legacy 0.139, quantile 0.128, outlier 0.122, **hetero 0.063**. This is a property of the strategy, not of the cutoff, and any hetero column in a robustness ranking essentially reproduces the baseline ranking. Needs an explicit caveat sentence wherever hetero is reported.

## 10.3 Where each dataset actually breaks (the threshold rule)

% of model configs still clearing an R² threshold, per dataset per σ. R²=0.3 is the only workable threshold: R²=0 is never crossed (except Caco-2 at σ=0.9), and R²=0.5 is already failed at σ=0 by hERG (29.6% above) and Caco-2 (6.6%).

| rule (R²>0.3) | Caco-2 | hERG | LogD |
|---|---|---|---|
| fewer than 75% remain | σ=0.5 (55%) | σ=0.6 (69%) | **never** (88% at σ=1.0) |
| fewer than 50% remain | σ=0.6 (38%) | σ=0.9 (42%) | **never** (88%) |
| fewer than 25% remain | σ=1.0 (23%) | never (41%) | **never** (88%) |

**⚠ LogD reaches no threshold at all** — 88.1% of configs still clear R²=0.3 and 97.7% still clear R²=0 at maximum noise. LogD is insensitive to the entire noise range tested. That is a reportable finding in its own right (and independently matches the earlier "LogD barely discriminates" note), not a defect to hide.

## 10.4 Full R² at σ=0.6 tables, rep = PDV

Source: `validation_rerun` (see the provenance warning in §"Ranking at σ=0.6"). CLEAN = R² at σ=0. Sorted by Gaussian (`legacy`) column.

**OpenADMET-LogD**

| model | CLEAN | legacy | outlier | quantile | threshold | valprop | hetero |
|---|---|---|---|---|---|---|---|
| DNN | 0.797 | 0.751 | 0.769 | 0.761 | 0.699 | 0.723 | 0.788 |
| GP | 0.784 | 0.746 | 0.759 | 0.749 | 0.705 | 0.728 | 0.772 |
| MLP | 0.770 | 0.744 | 0.745 | 0.730 | 0.691 | 0.720 | 0.777 |
| SVM | 0.779 | 0.744 | 0.771 | 0.757 | 0.700 | 0.723 | 0.769 |
| LightGBM | 0.758 | 0.729 | 0.731 | 0.717 | 0.644 | 0.698 | 0.753 |
| MLP-VBLL-Full | 0.751 | 0.719 | 0.736 | 0.716 | 0.682 | 0.703 | 0.749 |
| RF | 0.709 | 0.680 | 0.695 | 0.675 | 0.621 | 0.660 | 0.702 |
| MLP-BNN-Full | 0.682 | 0.664 | 0.675 | 0.650 | 0.623 | 0.648 | 0.666 |
| XGBoost | 0.740 | 0.663 | 0.723 | 0.688 | **0.492** | 0.598 | 0.737 |
| QRF | 0.703 | 0.658 | 0.677 | 0.661 | **0.535** | 0.615 | 0.697 |
| NGBoost | 0.661 | 0.657 | 0.656 | 0.657 | 0.653 | 0.652 | 0.661 |
| VBLL-Full | 0.686 | 0.634 | 0.657 | 0.629 | 0.530 | 0.597 | 0.664 |
| BNN-Full | 0.608 | 0.595 | 0.574 | 0.591 | 0.539 | 0.517 | 0.610 |

Note NGBoost: lowest spread across strategies of any model (0.652–0.661) but also near the bottom on absolute R². This is the retention-flattery case made visible in one row — it barely moves because it barely had anything to lose.

**ChEMBL-hERG-Ki**

| model | CLEAN | legacy | outlier | quantile | threshold | valprop | hetero |
|---|---|---|---|---|---|---|---|
| RF | 0.497 | 0.457 | 0.473 | 0.449 | 0.327 | 0.319 | 0.486 |
| LightGBM | 0.509 | 0.424 | 0.483 | 0.433 | 0.182 | 0.180 | 0.480 |
| QRF | 0.485 | 0.417 | 0.439 | 0.413 | 0.207 | 0.213 | 0.445 |
| SVM | 0.463 | 0.412 | 0.429 | 0.427 | 0.307 | 0.303 | 0.437 |
| NGBoost | 0.446 | 0.400 | 0.422 | 0.420 | 0.301 | 0.315 | 0.429 |
| XGBoost | 0.440 | 0.289 | 0.370 | 0.282 | **−0.155** | **−0.039** | 0.399 |
| MLP | 0.323 | NaN | NaN | NaN | NaN | NaN | **−41.857** |

MLP diverges on hERG/PDV — same family of numerical instability as SVM/PDV/hERG (baseline −1.9e7) and DNN/MHG-GNN. Divergence, not noise sensitivity; must be excluded or flagged, never averaged in.

**OpenADMET-Caco2_Efflux**

| model | CLEAN | legacy | outlier | quantile | threshold | valprop | hetero |
|---|---|---|---|---|---|---|---|
| MLP-VBLL-Full | 0.479 | 0.371 | 0.253 | 0.336 | 0.330 | 0.362 | 0.454 |
| DNN | 0.506 | 0.365 | 0.392 | 0.375 | 0.395 | 0.337 | 0.437 |
| MLP | 0.470 | 0.361 | 0.379 | 0.382 | 0.421 | 0.338 | 0.427 |
| GP | 0.500 | 0.357 | 0.344 | 0.375 | 0.383 | 0.348 | 0.466 |
| NGBoost | 0.444 | 0.315 | 0.246 | 0.225 | 0.320 | 0.299 | 0.419 |
| VBLL-Full | 0.446 | 0.315 | 0.231 | 0.305 | 0.299 | 0.344 | 0.422 |
| SVM | 0.500 | 0.276 | 0.436 | 0.397 | 0.425 | 0.255 | 0.455 |
| MLP-BNN-Full | 0.398 | 0.269 | 0.313 | 0.291 | 0.332 | 0.255 | 0.399 |
| RF | 0.460 | 0.219 | 0.192 | 0.229 | 0.324 | 0.174 | 0.410 |
| BNN-Full | 0.320 | 0.175 | 0.230 | 0.169 | 0.225 | 0.168 | 0.302 |
| QRF | 0.436 | 0.089 | 0.319 | 0.279 | 0.312 | 0.050 | 0.394 |
| LightGBM | 0.486 | **0.008** | 0.041 | 0.043 | 0.247 | **−0.037** | 0.433 |
| XGBoost | 0.401 | **−0.233** | −0.033 | −0.213 | 0.117 | **−0.269** | 0.348 |

**This table alone kills several claims.** LightGBM has the second-best clean R² on Caco-2 (0.486) and collapses to 0.008 under Gaussian noise — a model that looks excellent clean and is useless at σ=0.6. XGBoost goes negative under three strategies. Meanwhile the neural models (MLP-VBLL, DNN, MLP) hold up best. Note this is the **opposite** of the QM9 picture, where neural models are the fragile ones — direct support for the dataset-size argument already in §9 (F6b).

## 10.5 Corrections to earlier claims in this guide

- **"SVM is a retention mirage" — WITHDRAWN.** Like-for-like on the 45 cells where all 7 pipeline models are present, SVM ranks **#1 on both** retention (0.900) and absolute R² under noise (0.574). The earlier claim came from an ungated raw computation where SVM's single divergent PDV/hERG cell dragged its mean down. **The genuine retention-flattery case is NGBoost** (retention rank 2, absolute rank 5, lowest baseline 0.536); **LightGBM is the reverse** (retention rank 4, absolute rank 2 — the metric under-credits it). Update §9.4 accordingly; §9.4's NGBoost point stands and is now better evidenced.
- **`results/paper_figures/` is stale NDS output.** `scripts/generate_paper_figures.py` (v1) has 216 `nds` mentions and **zero** `auc_norm`; v2 has 122. `run_figures.sh` invokes v1; **`run_figures_v2.sh` is the correct regen script** and writes to `results/paper_figures_v2/`. Do not read numbers out of `paper_figures/`.
- **Validation directory ambiguity is unresolved** and affects everything in §10.4 — `alternative_full` (7 models, what the pipeline is fed on ARC) vs `validation_rerun` (13 models, the local per-σ source). Settle before any of these tables enters the paper.

---

# §11 — VERIFIED VERBATIM QUOTES (2026-08-20)

Every string in quotation marks below was fetched from the source and then **independently re-fetched and re-checked character-for-character by a separate pass**. 35 quotes passed; 4 failed and were discarded. Anything not in quotation marks is summary, not quotable.

⚠ **Read the closing assessment at the end of this section before citing anything** — one number this guide previously asserted did NOT survive verification.

# Experimental-noise anchor quotes — verified verbatim material

Every string in quotation marks below was independently re-fetched and confirmed character-for-character. Anything not in quotation marks is my own summary and must not be re-quoted. Where a verification pass flagged a location error or an interpretive trap, that is recorded under the quote rather than silently corrected away.

Typographic note that applies throughout: subscripts (IC₅₀, pKᵢ, P_app) are flattened to plain text in these transcriptions, and a few publisher pages use thin or non-breaking spaces around `=` and `×`. Neither changes wording or numbers.

---

## 1. Kramer, Kalliokoski, Gedeck & Vulpetti (2012) — heterogeneous public Ki data

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

## 2. Kalliokoski, Kramer, Vulpetti & Gedeck (2013) — mixed IC50 data

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

## 3. Sato, Yuki, Ito, Tatsuzawa, Yoshida, Yamada, Kanamitsu, Uchida, Hisaka, Yamashita, Yoshimatsu, et al. (2018) — hERG database construction

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

## 4. Bruneau & McElroy (2006) — logD7.4 modelling

**Citation.** Bruneau P, McElroy NR. logD7.4 modeling using Bayesian Regularized Neural Networks. Assessment and correction of the errors of prediction. *J. Chem. Inf. Model.* 2006;46(3):1379–1387. DOI: 10.1021/ci0504014

**Access. CONTESTED — see the gap note below.** The ACS page returns HTTP 403 and Unpaywall reports `oa_status: closed` with zero repository copies. One verification pass retrieved an author-hosted PDF via the Internet Archive (`https://web.archive.org/web/20240423025829if_/https://www.people.iup.edu/nate/docs/ci0504014.pdf`) and confirmed one string; other passes could not reach any full text at all.

**Only one quote from this paper survived verification:**

> "A measurement is more likely repeated if an apparently abnormal result is obtained. Thus, high ranges in duplicated measurements do not indicate a global variability of the experimental methodology."

*Location: reference/endnote list, note (32), final page — the note cited at the point in the text where the replicate-variability estimate is given.*
Establishes a selection-bias warning that directly constrains any noise-injection justification: replicate spreads in industrial datasets are not a random sample of measurement error, because repeats are preferentially triggered by suspicious results, so replicate-derived σ estimates skew high.

**The frequently cited "0.27 log units across 307 compounds" figure from this paper is not supported by a verified quote in this document.** See the closing section.

---

## 5. Hayeshi et al. (2008) — inter-laboratory Caco-2 comparison

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

## 6. Chen, Slättengren, de Lange, Smith & Hammarlund-Udenaes (2017) — reporting Hayeshi data

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

## 7. Niu et al. (2024) — PharmaBench

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

## 8. Alvarez Baron et al. (2025) — multi-laboratory manual patch clamp hERG

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

# Closing assessment: what can and cannot be quoted

## Endpoints with direct quotable support

**Binding affinity (Ki), public aggregated data.** Supported, abstract only. Kramer et al. 2012 can be quoted for σ = 0.54 log units, mean error 0.44, median error 0.34. The full text was never retrieved, so anything beyond those three numbers and the two other abstract sentences above is not quotable from this paper.

**IC50, mixed public data.** Fully supported from open-access full text. Kalliokoski et al. 2013 gives σ = 0.68, the 4.8-fold interpretation, the within-laboratory floor of σ = 0.17–0.22, and the sample size. This is the strongest single source in the set.

**The √2 correction for pairwise-derived σ.** Supported, but only via Kalliokoski's Figure 5 caption. If your text needs this methodological point, cite Kalliokoski 2013, not Kramer 2012.

**hERG IC50, database heterogeneity.** Fully supported from Sato et al. 2018: the two-assay-type structure, the 209-compound comparison, R² = 0.517 and RMSD = 0.737 between assay methods, and the heavy-tailed 263/144/47 breakdown.

**hERG IC50, standardised repeat measurement.** Fully supported from Alvarez Baron et al. 2025: ~5× overall, τ figures with confidence interval, and the 3.4×–9.6× per-laboratory range.

**LogD.** Supported for the *modern benchmark* case only, from PharmaBench (Niu et al. 2024): repeat-test MAE 0.7 before curation and 0.48 after, plus the worst-case-selection caveat that frames those as upper bounds.

**Caco-2 permeability, inter-laboratory.** Partially supported. The Hayeshi 2008 abstract yields four usable qualitative sentences about design, mechanism, compound-dependence, and qualitative disagreement. Every *number* characterising the size of the Hayeshi spread comes from a secondary source: the 0.18–3.76 efflux-ratio range from Chen et al. 2017, and the "factor of 2–5" characterisation from Kell & Oliver 2015. Both are verified verbatim in those papers, but neither is Hayeshi's own wording.

## Gaps — endpoints and numbers that cannot currently be quoted

**LogD from Bruneau & McElroy 2006: the "0.27 log units across 307 compounds" figure has no verified quote.** This is the significant gap. Three separate claimed sentences from this paper's Methodology and Results — the one containing 0.27 and 307, the preceding clause about the 2-log-unit exclusion threshold, and two sentences from the "Database" subsection including a 0.3-log-unit discard threshold — all failed verification with the verdict "could not access". The ACS full text returns HTTP 403, Unpaywall reports the article as closed with zero repository copies, and only the abstract could be read; the abstract contains none of those numbers. One verification pass did reach an author-hosted PDF through the Internet Archive and confirmed endnote 32 from it, so the article text is evidently reachable by that route, but the numeric sentences were never independently confirmed and therefore must not be presented as quotes.

Honest options for this number, in order of preference:

1. **Obtain the PDF manually** through Oxford institutional access at pubs.acs.org, or from the archived author copy, and check pages 1380 and 1382–1383 by eye. This is the only route that turns 0.27 into a quotable figure, and it is cheap.
2. **Cite without quoting.** Write "Bruneau and McElroy report a mean per-compound replicate standard deviation of approximately 0.27 log units for in-house logD7.4 data" with a plain citation and no quotation marks. This is defensible only if you have actually seen the number somewhere you trust; do not do this on the strength of this document alone, because this document has not confirmed it.
3. **Use a different source.** PharmaBench Table 5 (MAE 0.48–0.7 for LogD) is fully verified, open access, and covers the same endpoint. If the argument only needs "LogD labels carry roughly half a log unit of error", PharmaBench supports it outright and Bruneau is not required.

Also note two internal inconsistencies flagged during verification, which should be resolved before the guide is used: the repository currently cites Bruneau for 0.27 in `REVISION_GUIDE.md` line 357 and `DISCUSSION_TRACKER.md` line 116, while a separate claimed sentence from the same paper gives 0.3 as a *discard threshold* — different quantities that should not be conflated. Neither is verified.

**Hayeshi's own numbers.** No quantitative statement can currently be quoted from Hayeshi et al. 2008 itself, because the full text is paywalled at Elsevier and only the abstract was retrieved. If a number attributed directly to Hayeshi is load-bearing, either obtain the PDF or attribute the number explicitly to Chen et al. 2017 or Kell & Oliver 2015, marked as reported-by.

**Kramer's filtering statistics.** The "90% of all pairs" figure is quotable, but only as attached to the category "repeated citations of single measurements". Any claim about what fraction of the dataset was ultimately removed is not supported by the retrieved abstract and would need the paywalled Methods.
---

# §12 — FIGURE CHANGES FOLLOWING THE D11 DECISION (2026-08-20)

**The decision, in the author's framing:** AUC$_\text{norm}$ **stays** as the robustness metric. It is not replaced. The problem is narrower than "wrong metric" — AUC$_\text{norm}$ needs to carry baseline performance alongside it, because on its own it rewards a model for having had little to lose. **R² at $\sigma = 0.6$ is a sanity check on AUC$_\text{norm}$, not a competing metric.** Other noise levels exist; this one catches a useful set of cases.

That framing is what these changes implement. Nothing here proposes a new metric or a new figure.

## 12.1 fig1 Panel B — add a baseline column (highest payoff, smallest change)

`create_figure1`, `generate_paper_figures_v2.py` L2647; Panel B is the model × strategy AUC$_\text{norm}$ heatmap built from L2680 onward, PDV-only.

**Problem.** Panel B is the figure a reader ranks models from, and it contains no baseline information whatsoever. A row reading 0.989 looks like the best model on the page. NGBoost's clean R² on PDV/LogD is 0.661 against DNN's 0.797 — the reader cannot see this.

**Change.** Prepend one column to the heatmap: **R² at σ=0**, visually separated from the strategy columns (its own colour scale or a gap, since it is a different quantity and must not be read as a seventh strategy). Plumbing exists — `auc_df` already carries `baseline_r2`, and `calculate_robustness` returns it.

**Free companion change.** Panel A already plots absolute R² against σ, so the sanity check is *already in the figure* — just visually disconnected from Panel B. Add a vertical reference line at σ = 0.6 in Panel A. This ties the two panels together at no cost and makes the chosen level explicit on the page.

## 12.2 fig3 — colour by delivered accuracy, and fix the axis zoom

`create_figure3`, L2910. Already plots `baseline_r2` (x) against `auc_norm` (y) for PDV/Gaussian — this figure is closest to right already.

**Change 1.** Colour (or size) each point by **R² at σ=0.6**. A point sitting high on the AUC axis but pale is a model that retains well and delivers little. This turns the scatter into the sanity check without adding a figure.

**Change 2 — the y-axis zoom is now arguing for a retired claim.** L2928–2934 deliberately tightens the y-range to the data (`auc_norm` ≈ 0.78–0.86, padded by 25%) with the stated rationale *"so the flatness (the whole point) fills the panel"*. That flatness **was** the point when the paper claimed robustness is decoupled from accuracy. §9.1 retires that claim. A zoomed axis that magnifies a 0.08 spread into a full panel visually manufactures the very effect being withdrawn. On a [0,1] axis the same data reads as a tight cluster, which is the honest picture. **Change the axis or change the caption — do not leave the zoom paired with the new text.**

## 12.3 Validation figures — same baseline treatment

`create_validation_figures`, L1441. Both `fig_validation_overview` and `fig_validation_combined` present validation robustness without baseline context. Apply the same fix as 12.1: a baseline R² strip or column adjacent to the AUC$_\text{norm}$ display. `val_auc_df` already carries `baseline_r2` (it is one of the six columns in `table_validation_auc_full.csv`).

This also interacts with the still-open D1 question about how `fig_validation_combined` handles the strategy axis — settle D1 first, then apply the baseline strip to whatever layout wins.

## 12.4 One new PANEL (not a new figure)

R² at σ=0.6 on one axis, AUC$_\text{norm}$ on the other, one point per model. Bottom-right = high retention, low delivered accuracy — the flattered quadrant. This is the sanity check as a single image.

**Add it as a third panel on fig3, not as a standalone figure** (the standing instruction is no new figures). The worked cases that land in that quadrant, all from §10.4:
- **BNN-Full, LogD, quantile** — AUC$_\text{norm}$ rank **1** (0.999), R²@0.6 rank **13 of 13**. Clean R² 0.585 against DNN's 0.801. The clearest single example in the dataset.
- **NGBoost, LogD, Gaussian** — AUC rank 1 (0.989), R²@0.6 rank 11.
- **LightGBM, Caco-2** — the reverse case the metric under-credits: clean rank 4, R²@0.6 rank 12, and −8 rank places under Gaussian, outlier, quantile *and* valprop. R² falls from a clean 0.486 to 0.008 at σ=0.6 under Gaussian. Consistent across strategies, not a fluke.

## 12.5 What does NOT change

- **fig2 (ANOVA) stays on AUC$_\text{norm}$.** It decomposes variance in robustness, which remains the right question; adding baseline would change what is being asked, not improve it.
- **Kendall's W and the ranking-stability work stay as they are** — they measure whether robustness rankings are consistent, which is unaffected.
- **No metric is removed and no figure is added.** Total: two panel edits, one axis decision, one baseline strip on the validation figures, one added panel.

## 12.6 Caveat that travels with every AUC$_\text{norm}$ heatmap

**Heteroscedastic noise does not separate models at any σ** — separation flat at 0.061–0.071 from σ=0 to σ=1.0, median R² only 0.485 → 0.400 (§10.2). Its column in any AUC$_\text{norm}$ heatmap is therefore approximately the baseline ranking in disguise. This is a property of the strategy, not of the figure, and cannot be fixed by design — it needs one footnote wherever the heatmap appears. The same applies to the hetero column in fig1 Panel B once 12.1 lands, where it will sit directly beside the actual baseline column and the similarity will be visible.

## 12.7 Order of work

1. Settle the validation directory question (`alternative_full` 7 models vs `validation_rerun` 13) — 12.3 depends on it.
2. Do 12.1 (fig1 Panel B baseline column + σ=0.6 line in Panel A). Self-contained, highest payoff.
3. Do 12.2 (fig3 colouring + axis decision). The axis decision is an author call, not a code question.
4. Do 12.4 (third panel on fig3).
5. Do 12.3 once D1 is settled.
