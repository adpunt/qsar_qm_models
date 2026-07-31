# NoiseInject — Paper Revision Guide

> The author edits `paper.tex`; this guide tells them what to change and hands them replacement prose. Built from the submitted manuscript and the regenerated CSVs in `results/paper_figures_v2/` and `results/paper_figures/`.
>
> **Two changes, both large:** (1) robustness metric **NDS -> AUC_norm**; (2) uncertainty measured **within each sigma** (which reverses the per-sample finding). AUC_norm is simply the metric — there is no prior NDS publication to justify against, and there is one manuscript, not an "expanded" one. Weibull is not used.

---

# The big picture: how the two changes reshape the whole argument

This section is the intellectual spine of the revision. Read it before touching any individual section, because the two changes are not local edits — they propagate from the abstract to the conclusion and, in two places, *reverse* a headline claim. Everything below cites the verified numbers you should carry into the prose.

### The two changes, stated once

- **Change 1 (metric).** Every robustness number in the paper is currently a **Noise Degradation Slope (NDS)** — the straight-line slope $dR^2/d\sigma$, where "more negative = worse." Replace all of them with **AUC$_\text{norm}$**: the trapezoidal area under the *retention* curve $R^2(\sigma)/R^2(0)$ over $\sigma\in\{0,\dots,1.0\}$, roughly on $[0,1]$, where **higher = more robust**. AUC$_\text{norm}$ makes no linearity assumption and — critically — is **not coupled to baseline accuracy** (it divides by $R^2(0)$). There is no prior NDS publication and no "old paper": you are simply defining and using AUC$_\text{norm}$. Never write "we changed the metric," "unlike NDS," or anything justifying the swap. Weibull is gone; do not mention it.

- **Change 2 (uncertainty within-$\sigma$).** The paper currently computes the uncertainty–noise Spearman $\rho$ by **pooling all $\sigma$ levels together**. Pooling secretly measures the *population* trend (mean uncertainty rises as $\sigma$ rises) and mislabels it as *per-sample* detection. Recomputed **within each $\sigma$ level**, the finding flips: **GP does not detect per-sample noise at all** ($\rho\approx0$ at every $\sigma$, on every representation), and the genuine per-sample detector is an **epistemic BNN** — the opposite of what the current Scientific Contribution asserts.

### (a) The paper's spine — and how each change *sharpens* it

The paper has always argued a **two-track thesis**:

1. **Noise robustness is a MODEL property, not a representation property.** ANOVA attributes most robustness variance to model architecture; representation contributes <10% for most noise types; rankings are stable across noise strategies (Kendall's $W$).
2. **Per-sample uncertainty detection is a SEPARATE, conditional capability.** Not every robust model can flag which individual labels are noisy; it depends on how the model represents uncertainty and on the representation feeding it.

Both changes *reinforce* this spine rather than threaten it — that is the reassuring headline for the author.

- **Change 1 sharpens Track 1's independence claim.** Under NDS, the paper had to keep *apologizing* for a confound: high-baseline representations "have more to lose," so PDV degraded fastest despite being the best representation, and low-baseline mol2vec looked "shallow" for the wrong reason (paper L471, L502). That whole caveat was a **pure artifact of an accuracy-coupled metric**. AUC$_\text{norm}$ normalizes by $R^2(0)$, so the artifact disappears: you can now state cleanly that robustness is genuinely decoupled from baseline accuracy — the decoupling argument (L443, L469) becomes *stronger and simpler*, not weaker. The one cost: the ANOVA now shows **outlier and heteroscedastic noise are residual-dominated**, so "model always dominates" softens to "model dominates for four of six noise types" (see below). *(⚠ The residual-dominance of those two strategies specifically is **provisional** — the forensic pass traced it to a roster artifact that needs a re-run; the four model-dominated strategies are solid. See the Variance-decomposition provisional banner. The spine below does not depend on the two provisional rows.)*

- **Change 2 sharpens Track 2's conditionality claim.** The paper already said uncertainty detection is representation-gated (embeddings fail). The within-$\sigma$ recomputation makes the gating **far more specific and more interesting**: per-sample detection requires *three things at once* — a subset-targeting **noise type** (only outlier/quantile), a **non-embedding representation** (PDV/fingerprints), and a **distributional model**. This is a genuinely new, publishable mechanism, and it turns a vague "some pairings work" into a crisp, three-way conjunction with a named leader (BNN-$\alpha$ on PDV under outlier, within-$\sigma$ $\rho=0.485$).

The deepest synthesis — which should appear once, near the end — is that **the two tracks do not travel together**: SVM is robust but has no uncertainty channel at all; GP tracks the *population* noise level but cannot finger *individual* noisy samples; BNN-$\alpha$ detects individual noisy samples but is only mid-pack for robustness (7th of 11 by mean AUC$_\text{norm}$). Robustness and per-sample noise-awareness are **orthogonal capabilities**. Both metric changes are what let you say that without hand-waving.

### (b) Every headline claim, triaged: BREAKS / STRENGTHENS / REFRAME

#### Abstract (L164–169)

- **"Noise robustness was measured as the slope of predictive performance degradation across increasing amounts of additional label noise."** → **REFRAME.** Replace with the AUC$_\text{norm}$ definition (normalized area under the retention curve; higher = more robust).
- **"NGBoost and SVMs showing the strongest robustness to noise."** → **REFRAME (half breaks).** NGBoost survives as **#1 by mean AUC$_\text{norm}$ (0.824)**. **SVM does not** — it is **5th (0.814)**, in a near-tie with XGBoost/LightGBM/RF, and *leads only under outlier noise* (AUC$_\text{norm}$ 0.956) and on the ADME data. New honest phrasing: "NGBoost was most robust overall, with the tree ensembles (RF, LightGBM, XGBoost) and SVM clustered just behind."
- **"applying Bayesian transformations to feed-forward neural networks improves their noise robustness."** → **REFRAME (mostly holds, one exception).** By mean AUC$_\text{norm}$: BNN improves *both* NN families (BNN-$\alpha$ 0.801 > NN-$\alpha$ 0.789; BNN-$\beta$ 0.802 > NN-$\beta$ 0.756). VBLL improves the $\beta$ family (VBLL-$\beta$ 0.792 > NN-$\beta$ 0.756) but **NOT the $\alpha$ family** (VBLL-$\alpha$ 0.781 < NN-$\alpha$ 0.789). So "both transformations improve both networks" is now FALSE. Say "Bayesian transformations generally improve NN robustness (full-BNN reliably; VBLL for the $\beta$ architecture)."
- **"NGBoost and Gaussian Processes displayed the strongest correlations between per-sample estimated uncertainty and injected noise."** → **BREAKS OUTRIGHT.** This is the sentence Change 2 destroys. Within-$\sigma$, **GP $\rho\approx0$ everywhere** — its pooled value was a population artifact. Replace with: the strongest genuine per-sample detector is **BNN-$\alpha$ on the PDV descriptor under outlier noise ($\rho=0.485$ at $\sigma=0.6$)**, and per-sample tracking appears only under subset-targeting noise (outlier/quantile) on non-embedding representations.

#### Scientific Contribution (L167–169)

- **"per-sample uncertainty estimates track injected label noise … only for models with an explicit aleatoric noise term such as the observation-noise variance of a Gaussian Process or the predicted scale of NGBoost, whereas models whose uncertainty is purely epistemic, such as Bayesian Neural Networks, do not."** → **BREAKS — and is exactly backwards.** The lead per-sample detector *is* the epistemic BNN; the GP (the paper's poster child for an explicit aleatoric term) does **not** detect per-sample noise at all. **Delete the aleatoric-vs-epistemic organizing frame entirely.** Replace with the conjunctive gate: per-sample detection requires (i) subset-targeting noise, (ii) a non-embedding representation, and (iii) a distributional model — and even then it is narrow. Anchor the strong claim on **BNN-$\alpha$**, not VBLL (VBLL-$\alpha$'s apparent 0.342 outlier signal rides on a *dirty* $\sigma=0$ control of ~0.22, an $|y|$-magnitude confound; BNN-$\alpha$ has a clean $\sigma=0$).

#### Methods — Performance Metrics (L234–322)

- **NDS definition block (L254–260)** → **REFRAME.** Replace the $dR^2/d\sigma$ equation with the AUC$_\text{norm}$ definition. Note the gate change: baseline $R^2<0.3$ now excludes **48** configs (was $<0.6$ excluding 66). Drop "positive slopes … would indicate noise improves performance" (slope language is gone).
- **Metrics summary table, NDS row (L291–293)** → **REFRAME** to the AUC$_\text{norm}$ row (higher = more robust, ~[0,1]).
- **ANOVA outcome text (L262, "either $R^2$ … or NDS")** → **REFRAME** to "$R^2$ … or AUC$_\text{norm}$."
- **Uncertainty $\rho$ definition (L240)** → **REFRAME / EXPAND.** State explicitly that the uncertainty–noise $\rho$ is computed **within each $\sigma$ level** to isolate per-sample detection from the population trend. This methodological sentence is what licenses the entire new finding — do not omit it.

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
- **"NDS clusters near $-0.38$ regardless of baseline R$^2$" (L443)** → **BREAKS as written** (it's an NDS statement). The *point* (robustness decoupled from baseline accuracy) is now **cleaner** because AUC$_\text{norm}$ is baseline-normalized by construction; restate as decoupling without the "$-0.38$ cluster" number and update Additional file 7.
- **Table 3 (`tab:nds_ranking`, L445–467)** → **FULL REPLACE.** Rebuild from `table2_auc_by_strategy_pdv.csv`: higher = more robust; new row order NGBoost > RF > LightGBM > XGBoost > SVM > BNN-$\beta$ > BNN-$\alpha$ > VBLL-$\beta$ > NN-$\alpha$ > VBLL-$\alpha$ > NN-$\beta$. Per-strategy bold winners change (e.g., outlier winner is SVM at 0.956).
- **"VBLL-$\alpha$ … outperforming all tree-based models except NGBoost under threshold and value-prop" (L469)** → **BREAKS.** Under AUC$_\text{norm}$, VBLL-$\alpha$ is 10th overall and below all four tree ensembles on threshold and value-prop. Delete this claim.
- **"NGBoost and SVM … did not perform particularly well on clean data … decoupling" (L469)** → **STRENGTHENS** (see spine). Keep the message; you may keep SVM as an *example* of decoupling since it is genuinely robust-ish while not top on clean data, but do not call it a top-2 robustness model.
- **PDV "high baseline → more to lose → steepest slopes / mol2vec shallower" (L471 AND L502)** → **DELETE at both locations.** This is the central NDS artifact. Under AUC$_\text{norm}$ it is simply untrue and unnecessary. Replace with the clean statement: PDV gives the best clean-data accuracy *and* competitive normalized robustness under strongly-regularized models (SVM, full BNN); representation contributes little to robustness variance.
- **"representation explains <11% of NDS variance … VBLL representation-dependent … 24 VBLL×{MHG-GNN,mol2vec} excluded at R$^2<0.6$" (L473)** → **REFRAME.** Recompute the <11% on AUC$_\text{norm}$; exclusion threshold is now $R^2<0.3$ (re-verify the count of excluded VBLL×embedding configs against the new gate before quoting "24").
- **Bayesian-transformation improvement + Table 4 (`tab:wilcoxon_bnn`, L475–493)** → **REFRAME + RECOMPUTE.** The Wilcoxon $\Delta$ is currently "$\Delta$NDS." Recompute all five rows as $\Delta$AUC$_\text{norm}$ (sign flips to "positive = more robust" naturally). Headline nuance: full-BNN improves both families; **VBLL improves $\beta$ but not $\alpha$** (mean AUC$_\text{norm}$ VBLL-$\alpha$ 0.781 < NN-$\alpha$ 0.789) — reconcile the per-strategy Wilcoxon result with the mean ranking and state the exception explicitly. **QRF < RF stays** (mean AUC$_\text{norm}$ RF 0.818 > QRF; direction preserved) — that claim STRENGTHENS.
- **"Tree ensembles dominate the top 10 … NGBoost appears five times … no NN in top 10 … SVM ranks higher on ADME" (L504)** → **RE-DERIVE** the top-10 on AUC$_\text{norm}$; qualitatively likely to hold (tree-dominated), but the exact counts must be recomputed. The "SVM higher on ADME" tail is supported (validation table: svm/sns 0.888, svm/mhggnn 0.879 are the two best ADME configs).

#### Results 4.3 — Uncertainty estimation under label noise (L506–551)

This subsection needs the **heaviest rewrite** — Change 2 lands here.

- **"GP and NGBoost … showed the strongest correlations … embeddings near-zero for all models" (L508)** → **HALF BREAKS.** The embedding-failure half **STRENGTHENS** (within-$\sigma$, mol2vec/MHG-GNN are ~0 even under outlier: BNN-$\alpha$ mol2vec −0.10, MHG-GNN 0.05). The "GP/NGBoost strongest" half **BREAKS** — replace the leaders with BNN-$\alpha$/BNN-$\beta$ under outlier, on PDV.
- **Table 5 (`tab:top_unc_noise`, L510–538)** → **FULL REPLACE.** It is a *pooled, Gaussian-only* table topped by GP (SNS 0.56, Morgan 0.53). Rebuild from the within-$\sigma$ panels: at $\sigma=0.6$ on PDV, **outlier** — BNN-$\alpha$ 0.485, BNN-$\beta$ 0.363, VBLL-$\alpha$ 0.342, QRF 0.289, VBLL-$\beta$ 0.217, NGBoost 0.210, **GP ~0**; **quantile** — NGBoost 0.241, BNN-$\beta$ 0.207, VBLL-$\alpha$ 0.191, QRF 0.181, VBLL-$\beta$ 0.158, BNN-$\alpha$ 0.130; **Gaussian/threshold/hetero/value-prop all ~0 for everyone**. Add the representation gate under outlier (BNN-$\alpha$: cont-PDV 0.485 vs fingerprints 0.10–0.23 vs mol2vec −0.10 / MHG-GNN 0.05).
- **Mechanistic paragraph (L547): "GP … strong empirical correlation … NGBoost … per-sample uncertainty rises with per-sample noise … GP/NGBoost produced the strongest correlations."** → **REWRITE.** The GP mechanism story now *predicts the correct answer*: a **global** $\sigma_n^2$ cannot be per-sample, so GP's within-$\sigma$ $\rho\approx0$ is exactly what its global-noise term implies — the pooled "strong" GP number was the population trend leaking in. Use this to explain *why* the epistemic BNN (input-dependent posterior broadening) can, under subset-targeting noise, out-detect the "explicit-aleatoric" GP. This inverts the paper's current mechanistic moral.
- **Fig 6 (`fig:uncertainty_combined`, mean uncertainty vs $\sigma$; VBLL aleatoric/epistemic, L540–545)** → **KEEP, RE-CAPTION as population-level.** The Kolmar link (mean uncertainty rises with $\sigma$) **survives and STRENGTHENS** as a *population* statement. Explicitly label it population-level so it is not confused with per-sample detection. Add a new figure/panel for the within-$\sigma$ per-sample result (`within_sigma_uncertainty.png`).
- **Kolmar extension (L551): "we extend … the link … holds at the individual sample level, contingent on a compatible model and representation pairing."** → **REFRAME, don't overclaim.** The population link holds broadly; the *individual-sample* link holds only under the narrow triple gate (subset-targeting noise + non-embedding rep + distributional model). Say both, distinctly.

#### Results 4.4 — Validation on experimental datasets (L560–571)

- **"model architecture dominates robustness variance on all three datasets … trends generalize" (L562)** → **REFRAME to AUC$_\text{norm}$**, re-verify against Additional file 10; likely holds.
- **"NGBoost ranks first under Gaussian … SVM marginally more robust overall (pooled mean NDS −0.16 vs −0.19), leading on hERG/Caco-2, NGBoost on LogD" (L562)** → **RECOMPUTE on AUC$_\text{norm}$.** The direction is supported: `table_validation_auc.csv` shows SVM best overall on ADME (svm/sns mean 0.888; svm/mhggnn 0.879) with NGBoost close (ngboost/continuous_pdv 0.877; strongest on LogD ~0.985). Replace the "−0.16 vs −0.19" NDS numbers with the AUC$_\text{norm}$ means.
- **"XGBoost suffers the most" (L562) / Fig 8b "ensemble-dependent models (XGBoost) degrade" (L567)** → **STRENGTHENS.** XGBoost is unambiguously worst on ADME by AUC$_\text{norm}$ (0.563/0.537/0.484/0.477 means; collapses on Caco-2 to 0.05–0.21). Keep and quantify with AUC$_\text{norm}$.
- **"QRF consistently less robust than RF on every external dataset" (L571)** → **STRENGTHENS.** Validation AUC$_\text{norm}$ confirms RF > QRF on all three (e.g., continuous_pdv: RF 0.777 > QRF 0.683). Keep.
- **Figs 7 & 8 (`fig:validation_overview`, `fig:validation_combined`)** → **REGENERATE** on AUC$_\text{norm}$ (heatmaps currently NDS; note the black-cell threshold text says $R^2<0.3$ already — consistent with the new gate).

#### Conclusion (L573–581)

- **L575 "noise degradation slope (NDS), defined as the slope of $R^2$ … model architecture is the dominant factor, while representation explains less than 10%."** → **REFRAME.** AUC$_\text{norm}$ definition; qualify "dominant factor" to "for four of six noise types" (outlier/hetero residual-dominated). "Representation <10%" **STRENGTHENS** (AUC$_\text{norm}$ rep $\eta^2$ is 0.2–7.9%).
- **L577 "outlier noise barely separates them"** → **STRENGTHENS** (now literally residual-dominated, 83.6%). **"Kendall's $W=0.92$"** → 0.9121. **"NGBoost and SVM, the most noise-robust"** → REFRAME (NGBoost + tree cluster; SVM mid-pack). **"embeddings' more decisive weakness was uncertainty, not robustness slopes"** → KEEP, now precise: within-$\sigma$ they are ~0 even under outlier.
- **L579 "NGBoost and Gaussian Processes … strongest per-sample correlation."** → **BREAKS.** Rewrite around BNN-$\alpha$/PDV/outlier as the per-sample detector; GP demoted to population-level tracking only.
- **L581 (closing synthesis): "NGBoost and GPs … most robust … and often produce uncertainty estimates which track per-sample label noise."** → **BREAKS the coupling.** This sentence fuses the two tracks that the corrected data pulls apart. Rewrite as the orthogonality thesis: the most-robust model (NGBoost) is a *decent* but not the *best* per-sample detector; the best per-sample detector (BNN-$\alpha$) is only mid-pack for robustness; the GP tracks the population but not individuals; SVM is robust with no per-sample channel. Pair the "fingerprints help detection" advice with the tighter caveat (only under outlier/quantile noise).

#### Abbreviations / metric plumbing

- **Abbreviations list (L602)** — drop "NDS: Noise degradation slope," add the AUC$_\text{norm}$ definition if you abbreviate it.
- **Availability of data (L624) and NoiseInject framework text (L380)** — both list "noise-performance degradation slope / retention." Update to AUC$_\text{norm}$ (the retention curve it integrates is already the object described, so this is a light touch).
- **Additional-file captions (L664–673)** — files 2, 3, 4, 6, 7, 8, 11 all say "NDS" and must be re-labeled AUC$_\text{norm}$; several supplements must be regenerated.

### (c) The new narrative through-line (abstract → conclusion)

Carry this single arc, verbatim in spirit, from the abstract to the last sentence:

> **Noise robustness is primarily a property of the model, largely independent of the molecular representation and of the model's clean-data accuracy** (AUC$_\text{norm}$ normalizes accuracy out; rankings are strategy-stable at $W=0.9121$; NGBoost leads, the tree ensembles and SVM follow closely). **Per-sample noise *detection* is a separate, narrow capability** that switches on only when three conditions coincide — subset-targeting noise (outlier/quantile), a non-embedding representation (PDV/fingerprints), and a distributional model — with an **epistemic BNN on PDV under outlier noise ($\rho=0.485$) as the strongest genuine detector**, while a Gaussian Process tracks only the *population* noise level, not individual noisy samples. **These two capabilities are orthogonal**: robustness and per-sample noise-awareness do not come from the same models, so practitioners must choose for whichever they need — and, when they need detection, pair a distributional model with a fingerprint/descriptor representation and hope the noise is of the subset-targeting kind.

The rhetorical shift from the current paper: drop "explicit-aleatoric models track noise, epistemic ones don't" (backwards) and drop every "high baseline has more to lose" apology (metric artifact). Replace both with the cleaner **"model-driven robustness + narrowly-gated, orthogonal detection"** frame.

### (d) Main points → where each lands

| # | Headline point | Fate | Key numbers to use | Where in paper.tex |
|---|----------------|------|--------------------|--------------------|
| 1 | Robustness is model-driven, not representation-driven | **STRENGTHENS** (qualify: 4 of 6 strategies) | Rep $\eta^2$ 0.2–7.9%; Model 43.8/36.8/54.7/52.5% (Gauss/Quant/Thresh/ValP); Outlier 10.3 & Hetero 14.0 residual-dominated (83.6/77.4) | L391–393, Table 2 (L406), Fig 2 (L419), L575 |
| 2 | Robustness decoupled from clean-data accuracy | **STRENGTHENS** (artifact removed) | AUC$_\text{norm}$ baseline-normalized; delete "$-0.38$ cluster" & "more to lose" | L443, L469, L471, L502 |
| 3 | NGBoost most robust | **HOLDS** | NGBoost 0.824 (mean AUC$_\text{norm}$, #1 of 11) | Abstract L164, L443, L577 |
| 4 | SVM one of the two most robust | **BREAKS → REFRAME** | SVM 5th (0.814); leads only outlier (0.956) + ADME | Abstract L164, L443, L577, L581 |
| 5 | NN-$\beta$ (mlp) least robust | **HOLDS** | mlp 0.756 (last of 11) | L443, Table 3 |
| 6 | Bayesian transforms improve NN robustness | **REFRAME (1 exception)** | BNN both (0.801>0.789; 0.802>0.756); VBLL-$\beta$ yes (0.792>0.756), **VBLL-$\alpha$ no (0.781<0.789)** | Abstract L164, L475, Table 4 (L486) |
| 7 | QRF less robust than RF | **STRENGTHENS** | RF 0.818 > QRF; holds on all 3 ADME | L475, L490, L571 |
| 8 | Rankings stable across noise strategies | **STRENGTHENS** | Kendall's $W=0.9121$, $p=3.55\times10^{-8}$ | L443, L577 |
| 9 | Outlier noise barely separates models | **STRENGTHENS** | Outlier robustness residual $\eta^2=83.6\%$ | L393, L469, L577 |
| 10 | GP & NGBoost strongest per-sample uncertainty trackers | **BREAKS OUTRIGHT** | Within-$\sigma$ GP $\rho\approx0$; BNN-$\alpha$ 0.485 leads | Abstract L164, SC L167, Table 5 (L519), L547, L579, L581 |
| 11 | Explicit-aleatoric term ⇒ tracks noise; epistemic ⇒ doesn't | **BREAKS (backwards)** — delete frame | Epistemic BNN-$\alpha$ is the lead detector; GP (aleatoric) fails | Scientific Contribution L167–169 |
| 12 | Embeddings (mol2vec/MHG-GNN) can't support uncertainty detection | **STRENGTHENS** | Within-$\sigma$ outlier: BNN-$\alpha$ mol2vec −0.10, MHG-GNN 0.05 | L508, L549, L577 |
| 13 | Population link: mean uncertainty rises with noise (Kolmar) | **HOLDS** — recaption as population-level | Fig 6 survives | L540–545, L551 |
| 14 | Per-sample link is conditional on model×representation | **STRENGTHENS + narrows** to triple gate | noise type + non-embedding rep + distributional model | L549, L551, SC |
| 15 | XGBoost collapses on external data | **STRENGTHENS** | ADME AUC$_\text{norm}$ 0.48–0.56; Caco-2 0.05–0.21 | L562, Fig 8 (L567) |
| 16 | Trends generalize to ADME; model dominates there too | **HOLDS** (re-verify AF10 on AUC$_\text{norm}$) | SVM/NGBoost best ADME configs 0.86–0.99 | L562, L571 |
| 17 | Robustness & per-sample detection are orthogonal | **NEW synthesis to add** | SVM robust/no detection; GP population-only; BNN-$\alpha$ detects but 7th robustness | Conclusion L581, Discussion |
| 18 | Config exclusion gate | **REFRAME** | $R^2<0.3$ → 48 excluded (was <0.6 → 66) | L260, L432 |

**One caution to bake in everywhere per-sample uncertainty is claimed:** anchor the strong per-sample result on **BNN-$\alpha$** (clean $\sigma=0$ control), and treat **VBLL-$\alpha$'s** apparent outlier signal (0.342) as suspect — its $\sigma=0$ control is ~0.22, an $|y|$-magnitude confound that inflates the within-$\sigma$ $\rho$. Mention VBLL only as corroborating, never as the headline.

---

---

# How to use this guide

This walks the manuscript **section by section, in paper order**. For each section you get the exact lines to change — I only propose a change where your text is **wrong**: the robustness metric (AUC$_\text{norm}$, not a slope), the reversed uncertainty finding (measured within each σ; the epistemic BNN-α on the PDV descriptor under outlier noise is the genuine per-sample detector, not the GP), or a number the data contradict. Everything else is your prose, kept byte-for-byte. You make every edit; nothing here touches your files.

Every number below was read directly from the regenerated CSVs in `results/paper_figures_v2/` and `results/paper_figures/`; every `\cite` was checked against `citations.bib`. Each section closes with what was **kept / removed / replaced** and a **number + citation verification** line.

**Two things have to happen:**
1. **You re-run the figures on ARC** — a handful of numbers can't be certified from the committed CSVs (the two contaminated ANOVA rows, the validation ANOVA, the within-σ confidence intervals). Those are marked **⚠ PROVISIONAL — RE-RUN** inline, and the exact commands are in the "Re-run on ARC" section at the end.
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

## Abstract (the `\abstract{}` block, paper.tex L164–169 — abstract prose + Scientific Contribution)

**Argument now:** The abstract must (i) define robustness as retained performance under rising noise (AUC$_\text{norm}$), not a slope; (ii) name the genuinely most-robust models (NGBoost + a tree ensemble, not SVM); and (iii) restate per-sample uncertainty as a separate, narrow capability owned by a Bayesian NN on PDV — the exact reversal of the GP/NGBoost claim — all with zero numbers, matching the current numberless abstract.

---

**Replace (L164, robustness-metric sentence):**
> Noise robustness was measured as the slope of predictive performance degradation across increasing amounts of additional label noise.

**With:**
> Noise robustness was measured as the normalised area under the $R^2$ retention curve across increasing amounts of additional label noise (AUC$_\text{norm}$; higher values indicate greater robustness).

---

**Replace (L164, most-robust-models sentence):**
> We found that model architecture was the dominant factor in performance degradation, with NGBoost and SVMs showing the strongest robustness to noise.

**With:**
> We found that model architecture was the dominant factor in performance degradation, with NGBoost and random forests showing the strongest robustness to noise.

---

**Replace (L164, Bayesian + uncertainty sentence — first clause kept verbatim, only the post-semicolon uncertainty clause is category (b) and changes):**
> We also found that applying Bayesian transformations to feed-forward neural networks improves their noise robustness; and that NGBoost and Gaussian Processes displayed the strongest correlations between per-sample estimated uncertainty and injected noise.

**With:**
> We also found that applying Bayesian transformations to feed-forward neural networks improves their noise robustness, and that per-sample uncertainty tracked injected noise only under a narrow set of conditions, namely subset-targeting noise, a non-embedding representation, and a distributional model, with a Bayesian neural network on the physicochemical descriptor vector the strongest detector.

---

**Scientific Contribution (inside the same `\abstract{}` block, L167–169). Keep the first two sentences verbatim; the per-sample half is category (b) and is exactly backwards.**

**Replace (L167 + L169, the broken-across-a-paragraph-break sentence):**
> We further show that per-sample uncertainty estimates track injected label noise for some certain models, [¶] only for models with an explicit aleatoric noise term such as the observation-noise variance of a Gaussian Process or the predicted scale of NGBoost, whereas models whose uncertainty is purely epistemic, such as Bayesian Neural Networks, do not.

**With:**
> We further show that per-sample uncertainty tracking is a separate and narrow capability. It emerges only when three conditions coincide: the noise targets an identifiable subset of samples, the representation is a non-embedding descriptor or fingerprint, and the model is distributional. Under these conditions a Bayesian neural network on the physicochemical descriptor vector is the strongest detector, whereas a Gaussian Process tracks only the population-level trend and does not resolve which individual labels were perturbed.

**Keep verbatim** (Scientific Contribution, first two sentences, unchanged): "This study introduces NoiseInject… impact of label noise." and "We demonstrate that the choice of model, rather than molecular representation, is the primary determinant of QSAR-model noise robustness, a conclusion we reached by comparing a model's relative ranking across different types of label noise." *(The old guide's "a model's"→"each model's" grammar tweak is NOT applied — not an (a)/(b)/(c) span; rule 1 forbids it.)*

---

**Decisions (folded from the old figures/lit/review material):**
- KEPT the full-draft/change-list consensus that names AUC$_\text{norm}$ ("normalised area under the $R^2$ retention curve") — required by rule 1(a); it carries no numbers, so it is abstract-safe.
- REMOVED the paper-voice "SPECTRA" symbol-free variant ("how much of its predictive performance a model retains…"): rule 1(a) mandates the AUC$_\text{norm}$ definition, so the metric is named, not paraphrased away. (Author's call if they prefer symbol-free — the science is identical.)
- REPLACED the full-draft's whole-sentence rewrite of the models claim ("tree ensembles… closely followed by SVM") with the minimal change-list swap "SVMs → random forests" — rule 1 forbids touching correct prose beyond the wrong span; RF is the verified #2.
- KEPT the change-list/paper-voice uncertainty clause (triple gate, BNN on PDV) over the full-draft's longer version; DROPPED the full-draft's added "at the population level, mean predicted uncertainty rose…" tail from the abstract (extra material, not a wrong-span fix) — it survives in the SC where the GP contrast lives.
- REMOVED all abstract numbers per rule 4 and ChatNT precedent; the figures-section material (ρ=0.485, W=0.9121, 0.824/0.818) stays out of the abstract and is used only for justification below.
- Scientific Contribution: KEPT the paper-voice three-sentence split (fixes the L167→L169 broken sentence); DROPPED the "aleatoric" organizing frame entirely per the change-list.

**Verification — numbers** (none appear in the replacement prose; these justify the wrong-span swaps only):
- NGBoost mean AUC$_\text{norm}$ = 0.823966 (#1) | table2_auc_by_strategy_pdv.csv | OK
- RF mean = 0.817719 (#2) | table2_auc_by_strategy_pdv.csv | OK
- LGB 0.816752 (#3), XGBoost 0.814395 (#4), SVM 0.813671 (#5) | table2_auc_by_strategy_pdv.csv | OK — SVM is 5th, confirming the SVM→RF correction
- SVM Outlier = 0.955555 (leads outlier column: RF 0.954834, NGBoost 0.952861) | table2_auc_by_strategy_pdv.csv | OK — SVM's win is outlier-only, so it is not an abstract-level "strongest robustness" model
- Kendall's W = 0.9121, p = 3.55e-8, 11 models, 6 strategies | table6_kendalls_w.txt | OK (not printed in abstract; confirms ledger, no stale value carried in)

**Verification — citations:** The abstract and Scientific Contribution contain zero `\cite`/`\citep`/`\citet` keys (journal abstracts carry no citations); none to verify. No citation keys added.

---

I have everything needed. The COMET abstract carries zero numbers, introduces its acronym inline, and states its advance as a plain claim with a plainly-owned scope limit ("beyond the simplistic binary reduction to cases and controls"). Here is the consolidated section.

## Scientific Contribution

**Argument now:** Keep the first two sentences verbatim (NoiseInject framing + "model, not representation, drives robustness" — both correct and metric-agnostic). Rewrite only the third sentence: the aleatoric-vs-epistemic frame is backwards and must be replaced by the triple-gated, epistemic-BNN-leads finding, stated plainly and number-free.

---

**Sentence 1 — keep verbatim (no change):**
> This study introduces NoiseInject, an open-source benchmarking framework that performs controlled artificial noise injections and provides analysis tools to determine the impact of label noise.

**Sentence 2 — keep verbatim (no change; correct, and names no metric):**
> We demonstrate that the choice of model, rather than molecular representation, is the primary determinant of QSAR-model noise robustness, a conclusion we reached by comparing a model's relative ranking across different types of label noise.

**Sentence 3 — the only edit. This one span crosses the L167→L169 paragraph break; replace the whole thing:**

Replace: `We further show that per-sample uncertainty estimates track injected label noise for some certain models,` `only for models with an explicit aleatoric noise term such as the observation-noise variance of a Gaussian Process or the predicted scale of NGBoost, whereas models whose uncertainty is purely epistemic, such as Bayesian Neural Networks, do not.`

With: `We further show that per-sample uncertainty tracking is a separate and narrow capability. It emerges only when three conditions coincide: the noise targets an identifiable subset of samples, the representation is a non-embedding descriptor or fingerprint, and the model is distributional. Under these conditions a Bayesian neural network on the physicochemical descriptor vector is the strongest detector, whereas a Gaussian Process tracks only the population-level trend and does not resolve which individual labels were perturbed.`

**Full corrected block, paste-ready (unchanged spans verbatim; the only difference is sentence 3, and the stray L167/169 paragraph break is closed):**

```latex
\textbf{Scientific contribution}. This study introduces NoiseInject, an open-source benchmarking framework that performs controlled artificial noise injections and provides analysis tools to determine the impact of label noise. We demonstrate that the choice of model, rather than molecular representation, is the primary determinant of QSAR-model noise robustness, a conclusion we reached by comparing a model's relative ranking across different types of label noise. We further show that per-sample uncertainty tracking is a separate and narrow capability. It emerges only when three conditions coincide: the noise targets an identifiable subset of samples, the representation is a non-embedding descriptor or fingerprint, and the model is distributional. Under these conditions a Bayesian neural network on the physicochemical descriptor vector is the strongest detector, whereas a Gaussian Process tracks only the population-level trend and does not resolve which individual labels were perturbed.}
```

No table or figure is owned by this section.

**Decisions (what I folded from the old figures/lit/review material):**
- KEPT (as primary replacement) the "paper-voice line replacements" version at REVISION_GUIDE L1402 (COMET-voiced, concise, number-free) — best match to this contribution's numberless house style.
- REPLACED the longer "full draft" version at REVISION_GUIDE L165 with the L1402 version — the full-draft one restates the same triple gate at greater length and adds the "orthogonal properties" coda, which belongs in the Discussion synthesis (L581), not the contribution block.
- FOLDED the change-list rows (L47, L131, L593) into a single edit — they all say the same thing (delete the aleatoric/epistemic frame; anchor on epistemic BNN-α; GP is a pooled population artifact). Their numeric anchors (ρ=0.485 at σ=0.6) stay OUT of this block per the numberless rule; they live in the Abstract/Results.
- REMOVED any mention of VBLL-α as a detector: guide L47 flags its σ=0 control is dirty (|y|-confound), so the strong claim is anchored on BNN-α only — consistent with keeping the contribution to the single cleanest exemplar.

**Verification — numbers:** None. This section carries no numbers, and the replacement introduces none (confirmed: no digit or stat in the corrected block; all supporting figures — ρ=0.485 etc. — are quoted elsewhere, not here). Rule 4 (no numbers in Scientific Contribution) satisfied. Forbidden-word scan of the replacement ("NDS", "slope", "Weibull", "previously", "unlike", "we changed"): none present — OK.

**Verification — citations:** None. `grep` for `\cite`/`\citep`/`\citet` in paper.tex L167–169 returns empty; the replacement adds no citations. Nothing to check against citations.bib.

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

## Methods — Performance Metrics
**Argument now:** This section must (C1) define the robustness metric as AUC$_\text{norm}$ — the normalised area under the R² retention curve, higher = more robust, ~[0,1], baseline-decoupled — with no slope/NDS language and no "we changed it" framing; and (C2) state, as a plain methodological choice, that the uncertainty–noise correlation is computed *within* each fixed $\sigma$, since that one sentence is what licenses the reversed uncertainty finding downstream. Everything else in the section is correct and stays byte-for-byte.

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

**With:**
> To evaluate the effect of label noise, we examined performance retention under increasing artificial noise with $\sigma \in \{0, 0.1, 0.2, \ldots, 1.0\}$. For each configuration we recorded $R^2(\sigma)$ and normalised it by the clean-label value to obtain the retention ratio $R^2(\sigma)/R^2(0)$. Our robustness metric is the normalised area under this retention curve,
> $$\text{AUC}_{\text{norm}} = \frac{1}{\sigma_{\max}} \int_0^{\sigma_{\max}} \frac{R^2(\sigma)}{R^2(0)}\, d\sigma,$$
> evaluated by the trapezoidal rule over the eleven noise levels ($\sigma_{\max} = 1.0$).

*(Note: keep the `% Edit mark` comment on L254 as-is.)*

---

**Replace (L260):**
> Values closer to zero indicate that noise has less of an effect on performance, negative values indicate a higher sensitivity to noise. Positive slopes, which were not observed in this study, would indicate that the noise \textit{improves} model performance. Configurations with baseline R$^2 < 0.6$ were excluded from robustness analysis, as poor performance on clean labels tends to remain poor for noisy labels as well, producing misleadingly shallow or even positive slopes that represent poor performance across the board.

**With:**
> $\text{AUC}_{\text{norm}}$ lies on approximately $[0, 1]$: a value near 1 means performance is retained at every noise level, and lower values indicate faster degradation. Because each curve is normalised by its own clean-label performance, $\text{AUC}_{\text{norm}}$ is not confounded by baseline accuracy, and because it integrates the whole curve it makes no assumption that degradation is linear. Configurations with baseline R$^2 < 0.3$ were excluded from the robustness analysis (48 configurations; Additional file~5), since retention ratios become unstable when the clean-label denominator is near zero.

*(⚠ author check: confirm "Additional file~5" is the correct file index for the excluded-configs table — the paper elsewhere cites "Additional files~2--4" at L276; renumber if needed. The count "48" is verified below.)*

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

*(All other rows of `tab:metrics_summary` — RMSE, $R^2$, Wilcoxon, Kendall's $W$, ANOVA $\eta^2$, ICC(1,1), Coverage, ECE — and the table caption are correct and stay verbatim.)*

---

**Everything else in L234–322 stays verbatim**, including: the $\sigma$-definition paragraph (L236), the Wilcoxon/Kendall paragraph (L238, correct — no numbers), the coverage and ECE equation blocks (L242–252), the full ANOVA-model equation and $\eta^2$ derivation (L263–272), and the independence/ICC/redundancy paragraphs (L274–276). None of these contains an NDS reference, a pooled-uncertainty claim, or a CSV-contradicted number.

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
> The framework computes standard regression metrics (RMSE, $R^2$, mean absolute error [MAE]) and classification metrics (accuracy, precision, recall, and F1 score, both macro- and weighted-averaged, with per-class breakdowns). For probabilistic models, it additionally computes uncertainty-calibration metrics: expected calibration error (ECE), empirical coverage at $1\sigma$ and $2\sigma$, mean prediction-interval width, and the Spearman correlations between predicted uncertainty and both absolute error and injected noise. It reports noise robustness metrics, including the normalised area under the $R^2$ retention curve (AUC$_{\text{norm}}$; higher values indicate greater robustness) and retention percentage, to quantify performance degradation across noise levels. Results are returned as structured \texttt{pandas} DataFrames (per-noise-level values and an aggregate summary) for downstream analysis.

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
With:    `However, for noise robustness, quantified by the normalised area under the R$^2$ retention curve (AUC$_\text{norm}$; higher values indicate greater robustness), this ordering shifts, though not uniformly across strategies. For four of the six strategies (Gaussian, quantile, threshold and value-proportional) the choice of model architecture becomes the largest structured source of variance (36.8--54.7\%), while the interaction term drops to second and representation to a small residual role; under outlier and heteroscedastic noise, however, model, representation and interaction effects are all small and the residual term dominates (83.6\% and 77.4\%).`

> **⚠ PROVISIONAL clause.** The final clause — "under outlier and heteroscedastic noise… the residual term dominates (83.6% and 77.4%)" — rests on the two contaminated ANOVA rows (see the table banner below). The 36.8–54.7% model-dominance for the other four strategies is solid. If you'd rather not state the residual-dominance until the roster-fixed re-run, end the sentence at "…representation to a small residual role" and add the outlier/heteroscedastic clause back once re-run. Otherwise keep it and update the two numbers after re-running.

Everything else in L391 ("This observed inversion makes sense because…" through "…stronger impact on how it handles noise.") is correct and stays **byte-for-byte**.

**(2) Paragraph L393 — per-strategy interpretation.** No wrong span (no NDS, no CSV-contradicted number; threshold/value-prop are genuinely the most-degraded and most model-dominated, and "for outlier noise, model and interaction effects were both smaller" is true). **KEEP VERBATIM.** The heteroscedastic-residual fact is now carried by paragraph (1) and the table, so no edit is forced here.

**(3) Table `tab:anova_decomposition` — caption L397 + body L406–411.**

> **⚠ PROVISIONAL — RE-RUN NEEDED for two rows.** The forensic pass found: (i) the robustness ANOVA roster is **not** "7 models" — the code actually uses **9** models for Gaussian/Quantile/Threshold/Value-Prop and silently grows to **10** for Heteroscedastic/Outlier by re-admitting `dnn_vbll`; (ii) that VBLL re-admission, whose runs diverge on the mol2vec/MHG-GNN embeddings, **inflates the residual** for exactly those two rows — so the **Heteroscedastic (77.4) and Outlier (83.6) residual-dominance is a roster artifact, not a finding**; (iii) the method is weighted-marginal SS (= Type I only under balance), not literal "Type I sequential." **Gaussian, Quantile, Threshold, Value-Prop robustness values are solid.** Before publishing, re-run with one consistent roster across all six strategies (drop `dnn_vbll`/`mlp_vbll` everywhere) per the "Re-run on ARC" section; Hetero/Outlier will change. Also re-run to confirm the true model count for the caption.

Paste-ready (performance columns unchanged/correct; robustness columns replaced; bold moved to the largest cell per row). Caption avoids committing to a roster count until the re-run confirms it:

Replace caption L397 with:
```latex
\caption{ANOVA variance decomposition by noise strategy on the QM9 HOMO--LUMO gap. Each cell reports $\eta^2$ (\%), the share of variance explained by that factor. Columns: \textbf{Model} = model architecture effect; \textbf{Rep} = molecular representation effect; \textbf{Inter.}\ = model$\times$representation interaction; \textbf{Resid.}\ = residual unexplained variance. ``Performance'' uses R$^2$ at $\sigma = 0.3$ as the outcome (eleven models); ``Robustness'' uses AUC$_\text{norm}$, the normalised area under the R$^2$ retention curve, higher values indicating greater robustness. $\eta^2 = SS_\text{factor}/SS_\text{total}$ from weighted marginal sums of squares (equivalent to Type I/II for balanced designs). Bold entries mark the dominant factor in each row.}
```
Replace body rows L406–411 with (Gaussian/Quantile/Threshold/Value-prop final; **Heteroscedastic + Outlier ⚠ provisional — see banner**):
```latex
Gaussian        & 24.6 & 22.8 & \textbf{48.1} &  4.5 & \textbf{43.8} &  5.2 & 16.9 & 34.2 \\
Quantile        & 25.1 & 22.4 & \textbf{46.9} &  5.6 & 36.8 &  4.4 & 15.1 & \textbf{43.7} \\
Threshold       & 25.0 & 21.2 & \textbf{48.0} &  5.8 & \textbf{54.7} &  7.9 & 22.6 & 14.8 \\
Heteroscedastic & 27.9 & 23.5 & \textbf{46.2} &  2.5 & 14.0 &  0.7 &  8.0 & \textbf{77.4} \\
Value-prop.     & 27.8 & 22.0 & \textbf{46.5} &  3.8 & \textbf{52.5} &  6.0 & 19.9 & 21.6 \\
Outlier         & 25.3 & 22.0 & \textbf{49.2} &  3.5 & 10.3 &  0.2 &  5.9 & \textbf{83.6} \\
```
(Bolding rule now "largest cell in row": for Quantile/Hetero/Outlier the residual is largest, so bold moves off Model onto Resid — matches the numbers. Old table wrongly bolded Model for every robustness row. The Hetero/Outlier residual values will move on the roster-fixed re-run.)

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

### ANOVA restatement + PDV "most-to-lose" paragraph (L471)

**Replace:**
> As established by the ANOVA, model architecture and the model--representation interaction term dominate NDS variance. A handful of models stood out as being noise-robust regardless of representation, namely SVM and full BNNs. Both methods rely on inductive biases, SVM on margin maximization and BNN on weight priors. However, other models like RF and NN-$\beta$ show the opposite pattern. These models can be robust to noise, but only when paired with particular representations. The pairings differ by model. Notably, although PDV gave the strongest clean-data performance of any representation, this did not translate into the slowest degradation under noise: because NDS is the slope of $R^2$ against $\sigma$, PDV's high baseline $R^2$ leaves more performance to lose, so its degradation slopes are among the steepest of all representations. Lower-baseline representations such as the mol2vec embedding show shallower slopes, part of which reflects this reduced headroom rather than intrinsic noise resistance. PDV's value therefore lies in pairing the strongest predictive performance with robust behaviour under strongly-regularized models such as SVM and full BNNs, rather than in being the most noise-robust representation overall.

**With:**
> As established by the ANOVA, model architecture and the model--representation interaction term dominate AUC$_{norm}$ variance. A handful of models stood out as being noise-robust regardless of representation, namely SVM and full BNNs. Both methods rely on inductive biases, SVM on margin maximization and BNN on weight priors. However, other models like RF and NN-$\beta$ show the opposite pattern. These models can be robust to noise, but only when paired with particular representations. The pairings differ by model. Notably, PDV gave the strongest clean-data performance of any representation, and because AUC$_{norm}$ measures the fraction of baseline R$^2$ retained rather than an absolute drop, this strength is not penalised: PDV retained its performance competitively under noise, with a mean AUC$_{norm}$ across models of 0.801, among the highest of any representation. Embeddings such as mol2vec retained comparably on the models they supported but collapsed when paired with Bayesian neural networks. PDV's value therefore lies in pairing the strongest predictive performance with among the most robust behaviour under noise, particularly under strongly-regularized models such as SVM and full BNNs.

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
- PDV (continuous_pdv) rep-mean AUC_norm across models = 0.801 | table2_supp_auc_all_reps.csv (per-rep mean of MEAN) | **OK** (binary pdv 0.812, mol2vec 0.806 higher → "among the highest" is fair)
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

**Argument now:** This subsection must land the reversal cleanly: pooling all σ together secretly measured a *population* trend (mean uncertainty rises with σ), and once the uncertainty–noise correlation is recomputed *within each σ level*, genuine per-sample detection is narrow and triple-gated — a subset-targeting noise type (outlier/quantile) × a non-embedding representation (PDV/fingerprints) × a distributional model — with the epistemic BNN-α on the PDV descriptor under outlier noise the strongest genuine detector (ρ = 0.485). The GP, top of the old pooled table, tracks the population but not individuals. Robustness and per-sample noise-awareness are separate capabilities.

---

**Subsection heading** (paper-craft: encode the corrected C2 finding as the claim-heading; folded from old guide L1786):
```
Replace: \subsection{Uncertainty estimation under label noise}
With:    \subsection{Uncertainty tracks label noise only for distributional models on non-embedding representations}
```

---

**Opening paragraph (L508)** — the GP/NGBoost-strongest clause is the pooled artifact (C2b); the embedding-failure clause is correct and kept verbatim:
```
Replace: To paint the complete picture of models and noise robustness, we examine uncertainty estimates. One key sign that a model is able to handle noise is its ability to track it, represented by the correlation between per-sample artificial noise and uncertainty. These correlations vary drastically across models and representations, as seen in Table~\ref{tab:top_unc_noise}. While GP and NGBoost on non-embedding representations (fingerprints and the PDV descriptor) showed the strongest correlations, uncertainty estimates on graph-based (MHG-GNN) and substructure-embedding (mol2vec) representations show near-zero or negative noise correlations for all models tested.

With:    To paint the complete picture of models and noise robustness, we examine uncertainty estimates. One key sign that a model is able to handle noise is its ability to track it, represented by the correlation between per-sample artificial noise and uncertainty. Pooling this correlation across all noise levels, however, conflates a population trend---mean predicted uncertainty rises with $\sigma$ for most probabilistic models---with genuine per-sample discrimination at a fixed noise level. Computed \emph{within} each $\sigma$ level, the correlations vary drastically across models, representations, and noise strategies (Table~\ref{tab:top_unc_noise}): most combinations sit at $\rho \approx 0$, and per-sample detection emerges only under a narrow combination of conditions. The strongest genuine detector is the epistemic BNN-$\alpha$ on the PDV descriptor under outlier noise ($\rho = 0.485$ at $\sigma = 0.6$), while uncertainty estimates on graph-based (MHG-GNN) and substructure-embedding (mol2vec) representations show near-zero or negative within-$\sigma$ correlations for all models tested.
```

---

**Table `tab:top_unc_noise` (L510–538)** — FULL REPLACE. The current table is pooled/Gaussian-only, topped by GP/SNS 0.56 (a population artifact that vanishes within σ). Replace with the within-σ noise-type table (values verified against `within_sigma_panelA_continuous_pdv_by_strategy.csv`):
```latex
\begin{table}[htbp]
\centering
\caption{Within-$\sigma$ per-sample uncertainty--noise correlation on the QM9 HOMO--LUMO gap, PDV descriptor, at $\sigma = 0.6$. Each cell is the Spearman correlation between predicted per-sample uncertainty and injected per-sample noise magnitude $|\epsilon|$, computed \emph{within} the fixed noise level (higher = better per-sample noise detection). Only the two subset-targeting strategies (outlier, quantile) yield a nonzero signal; Gaussian, threshold, heteroscedastic, and value-proportional noise leave every model at $\rho \approx 0$. GP produces no per-sample signal (see Table~\ref{tab:unc_repgate}). Dashes indicate configurations not run.}
\label{tab:top_unc_noise}
\small
\begin{tabular}{lrrrrrr}
\toprule
\textbf{Model} & \textbf{Gauss.} & \textbf{Thresh.} & \textbf{Hetero.} & \textbf{Val-Prop} & \textbf{Quant.} & \textbf{Outlier} \\
\midrule
\textbf{BNN-$\alpha$} & $-0.07$ & $0.03$ & $-0.01$ & --- & $0.13$ & $\mathbf{0.485}$ \\
BNN-$\beta$   & $-0.04$ & $0.01$ & $-0.02$ & $-0.01$ & $0.21$ & $0.363$ \\
VBLL-$\alpha$ & $0.02$  & $0.02$ & $0.06$  & $0.01$  & $0.19$ & $0.342$ \\
QRF          & $0.02$  & $-0.01$ & $0.02$ & $0.03$  & $0.18$ & $0.289$ \\
VBLL-$\beta$  & $0.03$  & $-0.01$ & $0.06$ & $0.01$  & $0.16$ & $0.217$ \\
NGBoost      & $0.02$  & $0.02$ & $0.05$  & $0.03$  & $0.24$ & $0.210$ \\
GP           & ---     & ---    & ---     & ---     & ---    & ---    \\
\bottomrule
\end{tabular}
\end{table}
```
*(ECE / coverage columns from the old pooled table are dropped — they were Gaussian-pooled per-cell metrics that no longer fit the within-σ, per-strategy layout; retain them, if wanted, only in Additional file 9.)*

---

**New table `tab:unc_repgate`** — ADD immediately after `tab:top_unc_noise` (the representation gate; values verified against `within_sigma_panelB_outlier_by_rep.csv`):
```latex
\begin{table}[htbp]
\centering
\caption{Representation gate for per-sample uncertainty detection: within-$\sigma$ uncertainty--noise correlation under outlier noise at $\sigma = 0.6$, across representations. Detection is strongest on the PDV descriptor and fingerprints and collapses to $\approx 0$ on the learned embeddings (mol2vec, MHG-GNN) for every model. Dashes indicate configurations not run.}
\label{tab:unc_repgate}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrr}
\toprule
\textbf{Model} & \textbf{c-PDV} & \textbf{PDV} & \textbf{Topo.} & \textbf{SNS} & \textbf{Morgan} & \textbf{SMILES} & \textbf{Mol2vec} & \textbf{MHG-GNN} \\
\midrule
\textbf{BNN-$\alpha$} & $\mathbf{0.485}$ & $0.14$ & $0.18$ & $0.23$ & $0.12$ & $0.10$ & $-0.10$ & $0.05$ \\
BNN-$\beta$   & $0.363$ & $0.25$ & $0.20$ & $0.28$ & $0.16$ & $0.17$ & $0.08$  & $0.01$ \\
VBLL-$\alpha$ & $0.342$ & $0.16$ & $0.32$ & ---    & ---    & $0.17$ & $0.05$  & $0.01$ \\
QRF          & $0.289$ & $0.23$ & $0.10$ & $0.11$ & $0.08$ & ---    & ---     & ---    \\
VBLL-$\beta$  & $0.217$ & $0.17$ & $0.27$ & ---    & ---    & $0.15$ & $0.02$  & $0.02$ \\
NGBoost      & $0.210$ & $0.21$ & $0.25$ & $0.20$ & $0.23$ & ---    & ---     & ---    \\
GP           & ---     & $0.06$ & $-0.02$ & $-0.04$ & $0.04$ & ---   & ---     & ---    \\
\bottomrule
\end{tabular}
\end{table}
```

---

**New figure `fig:within_sigma`** — ADD after the two tables, before `fig:uncertainty_combined` (this is the figure that carries the corrected finding; PNG `within_sigma_uncertainty.png` in `results/paper_figures/`):
```latex
\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{within_sigma_uncertainty.png}
\caption{Per-sample noise detection, measured \emph{within} each noise level to remove the population trend that mean uncertainty rises with $\sigma$. Cells report the within-$\sigma$ Spearman correlation between per-sample predicted uncertainty and injected noise magnitude $|\epsilon|$ at $\sigma = 0.6$ (higher = better per-sample detection). a) Noise-type dependence on the continuous PDV descriptor: only outlier and quantile noise---which concentrate corruption on an identifiable subset of samples---yield any per-sample signal, while Gaussian, threshold, heteroscedastic and value-proportional noise give $\rho \approx 0$ for every model. b) Representation gate under outlier noise: the signal is confined to the non-embedding descriptors, peaking for BNN-$\alpha$ on continuous PDV ($\rho = 0.485$) and collapsing to $\approx 0$ on the mol2vec and MHG-GNN embeddings. The GP shows no per-sample detection on any representation, confirming that its high pooled correlation was a population-level artifact.}
\label{fig:within_sigma}
\end{figure}
```

---

**Figure `fig:uncertainty_combined` caption (L543)** — factually correct (it *is* mean-uncertainty-vs-σ = population). Minimal clarifier so it is not read as per-sample detection:
```
Replace: \caption{Uncertainty estimates under label noise on the QM9 HOMO--LUMO gap. a) Mean predicted uncertainty versus $\sigma$ for probabilistic models (PDV, Gaussian strategy). b) Aleatoric and epistemic uncertainty components versus $\sigma$ for VBLL models.}
With:    \caption{Population-level uncertainty response to label noise on the QM9 HOMO--LUMO gap (not per-sample detection; see Figure~\ref{fig:within_sigma}). a) Mean predicted uncertainty versus $\sigma$ for probabilistic models (PDV, Gaussian strategy). b) Aleatoric and epistemic uncertainty components versus $\sigma$ for VBLL models.}
```

---

**Mechanism paragraph (L547)** — shown as a full corrected paragraph. The author's mechanism sentences (NGBoost scale, GP global σ²ₙ, BNN posterior broadening, VBLL global scalar) are kept **byte-for-byte**; the only changed spans are the pooled/aleatoric-frame conclusions (bolded here for review — remove bold in paste). The population-level closing sentence is kept verbatim:

> The different methods by which uncertainty is estimated all respond differently to label noise. NGBoost learns a per-sample mean $\mu$ and scale $\sigma$ as direct targets of its training objective \citep{Duan2020}: when a sample has a large residual, a typical indication of label noise, its predicted scale grows to absorb that residual \citep{kendall2017}, so per-sample uncertainty rises with per-sample noise. GPs instead learn a posterior mean and variance together with a single \emph{global} observation noise term $\sigma^2_n$ \citep{Rasmussen2005, Obrezanova2007}; because $\sigma^2_n$ is shared across all samples, the GP's per-sample uncertainty derives from its posterior variance, governed by kernel distance, rather than from an input-dependent noise term, **so within a fixed noise level it cannot, by construction, tell one sample apart from another: the GP's within-$\sigma$ correlation is $\approx 0$ on every representation it was run on (Table~\ref{tab:unc_repgate}: PDV $0.06$, topological $-0.02$, SNS $-0.04$, Morgan $0.04$), and its high pooled correlation was the population trend leaking in rather than per-sample detection.** BNNs treat large residuals differently, they broaden their weight posteriors, increasing predicted uncertainty without explicitly modeling observation noise \citep{gal2016, kendall2017}. VBLL adds a learned noise variance to the BNN loss \citep{Harrison2024}, but this value is a global scalar, independent of the input. **This input-dependent posterior broadening is precisely what an outlier label triggers, so under subset-targeting noise the purely epistemic BNN-$\alpha$ becomes the strongest genuine per-sample detector (within-$\sigma$ $\rho = 0.485$ on PDV under outlier noise), out-resolving the GP whose explicit noise term is global (Table~\ref{tab:top_unc_noise}, Table~\ref{tab:unc_repgate}, and Additional file~9); NGBoost and QRF, which parameterise per-sample distributions directly, show modest but real within-$\sigma$ signal under the subset-targeting strategies (outlier and quantile), while every model falls to $\rho \approx 0$ under the uniform or smoothly-graded strategies. Per-sample noise detection is therefore not a property of carrying an explicit aleatoric term, but of representing uncertainty in an input-dependent way and feeding it a representation in which corrupted labels remain separable.** At the population level, mean predicted uncertainty increases with $\sigma$ for most probabilistic models, and for VBLL both the aleatoric and epistemic components rise with the injected noise (Figure~\ref{fig:uncertainty_combined}).

**Add one scope-caveat sentence** after this paragraph (folded from old guide L486; the σ=0-control detail — paper-craft: narrow result gets its explicit caveat):
> VBLL-$\alpha$'s apparent outlier signal ($0.342$) is close to BNN-$\beta$'s, but its $\sigma = 0$ control is not clean --- at zero injected noise its within-$\sigma$ correlation is already $\approx 0.22$, indicating a label-magnitude confound rather than noise detection --- so the strong claim rests on BNN-$\alpha$, whose $\sigma = 0$ control is clean, with VBLL-$\alpha$ reported only as corroboration.

---

**Fingerprint-representation claim (L549)** — reframe to the triple gate (C2b); the embedding-failure observation is kept verbatim:
```
Replace: Graph-based and learned embeddings, which encode molecules in continuous, high-dimensional spaces, specifically MHG-GNN and mol2vec, did not provide useful signals for uncertainty estimation. This is in contrast to fingerprints which appear to provide a clear signal that better distinguishes noise-induced label perturbations from structural variation. Although further research could be done in this area, it appears as if uncertainty-based noise detection and mitigation methods may only be viable for certain model-representation combinations, particularly fingerprint representations.

With:    Graph-based and learned embeddings, which encode molecules in continuous, high-dimensional spaces, specifically MHG-GNN and mol2vec, did not provide useful signals for uncertainty estimation. This is in contrast to the PDV descriptor and fingerprints, which appear to provide a clear signal that better distinguishes noise-induced label perturbations from structural variation. Uncertainty-based noise detection therefore appears viable only when three conditions coincide: a subset-targeting noise type (outlier or quantile), a non-embedding representation (the PDV descriptor or a fingerprint; the learned embeddings collapse to zero), and a distributional model. Removing any one condition collapses the signal, so uncertainty-based noise detection and mitigation methods are restricted to this specific combination of model, representation, and noise type.
```

---

**Kolmar extension (L551)** — split the surviving population link from the narrow per-sample link (C2b); the robustness-transfer sentence at the end is kept verbatim:
```
Replace: \citet{Kolmar2021} found that models trained on increasingly noisy data maintain stable predictions on clean test sets, even as their apparent error on noisy test sets rises with the noise level. Their data was sourced from various targets across MoleculeNet \citep{wu2018}. While this holds across models within a given dataset, it becomes highly variable across datasets. They also established a population-level link between GP prediction uncertainty and noise magnitude when using PaDEL descriptors \citep{Kolmar2021}. We extend these insights by showing that the link between uncertainty and noise holds at the individual sample level, contingent on a compatible model and representation pairing. We also observe that while the predictive performance of a given model or representation may not be consistent across noise strategies, a model's relative noise robustness on one strategy does predict its robustness on others.

With:    \citet{Kolmar2021} found that models trained on increasingly noisy data maintain stable predictions on clean test sets, even as their apparent error on noisy test sets rises with the noise level. Their data was sourced from various targets across MoleculeNet \citep{wu2018}. While this holds across models within a given dataset, it becomes highly variable across datasets. They also established a population-level link between GP prediction uncertainty and noise magnitude when using PaDEL descriptors \citep{Kolmar2021}. This population-level link holds broadly in our data and across many more models (Figure~\ref{fig:uncertainty_combined}), but it does not by itself imply per-sample detection: within a fixed noise level the individual-sample link appears only under the combination of a subset-targeting noise type, a non-embedding representation, and a distributional model, and---notably---a Gaussian Process, the very model whose population link Kolmar reported, does not resolve noise at the individual-sample level. We also observe that while the predictive performance of a given model or representation may not be consistent across noise strategies, a model's relative noise robustness on one strategy does predict its robustness on others.
```

---

**Decisions (folded from old figures/change-list/line-replacement material):**
- **Kept:** the old guide's full-rebuild within-σ table (`tab:top_unc_noise`) and rep-gate table (`tab:unc_repgate`) and the `within_sigma_uncertainty.png` figure block — all match my CSV re-verification exactly; kept the claim-heading line-replacement (L1786).
- **Removed:** the old guide's *wholesale paragraph rewrites* (guide L435–488) — Hard Rule 1 requires starting from the author's verbatim prose, so I converted them into surgical Replace/With and one full-paragraph-corrected block that preserves the author's correct mechanism sentences byte-for-byte; also removed the old pooled ECE/coverage columns from the main table.
- **Replaced:** the pooled/Gaussian `tab:top_unc_noise` (GP/SNS 0.56 leader) with the within-σ table; the L547 "GP and NGBoost strongest" + aleatoric-frame conclusions with the epistemic-BNN-α/GP-demotion mechanism; the L549 "fingerprint" viability claim with the triple gate; the L551 Kolmar "individual-sample" extension with the population-vs-per-sample split.
- **Added (new, folded from guide L486/L759):** the VBLL-α dirty-σ=0-control caveat sentence, per the ledger and paper-craft scope-caveat principle.
- **Not in this section (flag for adjacent owners):** `fig:validation_overview` caption at L556 still says "Noise degradation slope (NDS)" — belongs to the Validation subsection; the abbreviations/additional-file NDS→AUC_norm relabels (guide L1794–1838) belong to the back-matter owner.

**Verification — numbers** (all from `results/paper_figures/`):
- BNN-α outlier ρ = 0.485 (0.484779) | within_sigma_panelA_continuous_pdv_by_strategy.csv & panelB | OK
- BNN-β outlier 0.363, VBLL-α 0.342, QRF 0.289, VBLL-β 0.217, NGBoost 0.210 | panelA Outlier col | OK
- Quantile: NGBoost 0.241, BNN-β 0.207, VBLL-α 0.191, QRF 0.181, VBLL-β 0.158, BNN-α 0.130 | panelA Quantile col | OK
- BNN-α Gaussian −0.07(−0.067), Thresh 0.03(0.032), Hetero −0.01(−0.009), Val-Prop NaN(dash), | panelA | OK
- Non-signal strategies (Gauss/Thresh/Hetero/Val-Prop) all within −0.07..+0.06 for every model | panelA | OK
- Rep gate BNN-α: c-PDV 0.485, PDV 0.14(0.137), Topo 0.18(0.176), SNS 0.23(0.233), Morgan 0.12(0.123), SMILES 0.10(0.104), Mol2vec −0.10(−0.102), MHG-GNN 0.05(0.048) | panelB | OK
- GP rep gate: PDV 0.06(0.065), Topo −0.02(−0.015), SNS −0.04(−0.036), Morgan 0.04(0.043); c-PDV NaN(dash) | panelB | OK
- All other cells in both tables (BNN-β/VBLL-α/VBLL-β/QRF/NGBoost rows) | panelA/panelB | OK
- VBLL-α σ=0 control ≈0.22 (dirty) vs BNN-α clean | ledger only — no per-σ=0 CSV provided; carried from ledger, flagged as not independently re-verified

**Verification — citations** (grepped `citations.bib`):
- Duan2020 IN-BIB · kendall2017 IN-BIB · Rasmussen2005 IN-BIB · Obrezanova2007 IN-BIB · gal2016 IN-BIB · Harrison2024 IN-BIB · Kolmar2021 IN-BIB · wu2018 IN-BIB (all 8 keys in the section confirmed present)

---

I have everything verified. Producing the consolidated section guide.

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

**Argument now:** After the two corrections the Conclusion must (i) name AUC$_\text{norm}$ (normalised area under the $R^2$ retention curve, higher = more robust) wherever the author wrote NDS/slope, and (ii) break the coupling the closing paragraph asserts — robustness and per-sample noise detection are orthogonal: NGBoost/tree-ensembles are most robust, GP tracks the population not individuals, and the epistemic BNN-$\alpha$ on the PDV descriptor under outlier noise is the one genuine per-sample detector, gated by subset-targeting noise × non-embedding representation × distributional model. Everything else in the author's four paragraphs is correct and stays byte-for-byte.

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

**L577d [C1 — metric name + drop "slopes"].** "slopes" is a forbidden metric term (RULE 4); the embeddings/uncertainty clause already matches the within-$\sigma$ finding (embeddings give no per-sample signal) and stays.

Replace: `SVM and full BNNs maintained consistent NDS across all representations. Embeddings (MHG-GNN, mol2vec) degraded most when paired with neural network architectures; their more decisive weakness, however, was that their per-sample uncertainty failed to track injected noise (discussed below), not their robustness slopes.`

With:    `SVM and full BNNs maintained consistent AUC$_\text{norm}$ across all representations. Embeddings (MHG-GNN, mol2vec) degraded most when paired with neural network architectures; their more decisive weakness, however, was that their per-sample uncertainty failed to track injected noise (discussed below), not their robustness.`

---

**L577e [no change].** `On the flip side, QRF was significantly less noise-robust than RF across datasets, suggesting that quantile regression may be overfitting to noisy labels rather than absorbing them.` — CSV-consistent (Wilcoxon rf→qrf $\Delta$AUC$_\text{norm}=-0.012$, $p=2.9\times10^{-11}$ SIG). Kept verbatim.

---

**L579 [C2 CRITICAL — reverse the first two sentences only].** The pooled "NGBoost and Gaussian Processes produced the strongest per-sample correlation" finding is the reversed claim. Sentences 3→end (QM9 clean baseline, validation-dataset description with the two \citep, XGBoost exception, future work) are correct and kept **verbatim**.

Replace (sentences 1–2 only): `With regard to uncertainty estimation, models that learn a separate scale or noise parameter during training, particularly NGBoost and Gaussian Processes, produced the strongest per-sample correlation between estimated uncertainty and injected noise magnitude. Non-embedding representations (fingerprints and the PDV descriptor) produced moderate-to-strong uncertainty-noise correlations, while the learned embeddings (MHG-GNN, mol2vec) produced near-zero or negative correlations for all models.`

With:    `Per-sample uncertainty tracking is a separate, narrow capability. Computed within each noise level, it emerges only under subset-targeting noise (outlier or quantile), only with a non-embedding representation, and only for a distributional model, with a Bayesian neural network on the physicochemical descriptor vector the strongest detector (within-$\sigma$ $\rho = 0.485$ under outlier noise). A Gaussian Process, whose pooled correlation is high, does not resolve noise at the individual-sample level; its signal is a population-level effect. Learned embeddings (MHG-GNN, mol2vec) produced no per-sample signal for any model.`

Unchanged remainder of L579 (keep exactly): `QM9 served as a relatively clean baseline; it is computationally-derived with negligible measurement noise. The validation datasets, LogD, Caco-2 \citep{openadmet}, and hERG Ki \citep{Zdrazil2023}, tested are smaller and cover a narrow range of the chemical and biological space, and have an unknown inherent amount of noise from experimentally-determined values, on top of the added artificial noise. These validation datasets supported the QM9 findings, with XGBoost the notable exception, degrading on the external datasets. Future work should extend this type of noise robustness benchmarking to classification tasks and larger experimental datasets with better-known uncertainty estimates.`

---

**L581 [C1 + C2 — closing paragraph, keep the opening sentence, break the coupling in the rest].** Sentence 1 is a still-true model-driven framing and is **kept verbatim**. Sentences 2–4 fuse the two tracks and mis-state the BNN uncertainty result; replace them.

Keep verbatim (sentence 1): `When working with noisy experimental data in QSAR settings, it is important to keep in mind that noise robustness and noise-aware uncertainty estimation are primarily driven by the model's mechanism of training, and to a certain extent the choice of molecular representation, usually with respect to the model.`

Replace (sentences 2–4): `While Bayesian transformations on neural networks improve noise robustness, they do not produce uncertainty estimates that correlate particularly strongly with noise. Models like NGBoost and GPs, which include a learned noise or scale parameter in the training objective, are the most robust to noise across both various types of noise and datasets and often produce uncertainty estimates which track per-sample label noise. Pairing these models with fingerprint-based representations provides additional benefits in restricting, and potentially detecting and mitigating label noise.`

With:    `Bayesian transformations on neural networks improve noise robustness, and their uncertainty does not track noise except for a full BNN under subset-targeting noise on descriptor representations. Noise robustness and per-sample uncertainty tracking are distinct capabilities and are not maximised by the same models: NGBoost and random forests are the most robust, SVM is robust but yields no per-sample uncertainty, a Gaussian Process tracks the population trend but not individual samples, and a Bayesian neural network detects per-sample noise, under outlier noise on descriptor representations, without being the most robust. Detecting or mitigating label noise from uncertainty is therefore possible only under a narrow, jointly-gated combination of noise type, representation and model.`

---

**Optional insert — Limitations paragraph (NEW).** Not a Replace/With; the guide proposes it as an insertion immediately before the closing L581 paragraph. Include only if the author wants it (it is genuinely new content, outside the "change wrong spans" mandate). Every number in it is verified below. Paste-ready:

```latex
\paragraph{Limitations.} Several constraints bound the strength of these conclusions. First, the per-sample uncertainty analysis rests on single-seed runs, so the within-$\sigma$ correlations we report carry no confidence intervals; individual cell rankings---including the leading BNN-$\alpha$/PDV/outlier value of $\rho = 0.485$---should be read as point estimates whose ordering, rather than whose exact magnitude, is the reliable signal. Second, the robustness ranking is built on a tight top cluster: NGBoost (0.824), the tree ensembles (RF 0.818, LightGBM 0.817, XGBoost 0.814) and SVM (0.814) are separated by hundredths of an AUC$_\text{norm}$ unit, so the claim that these architectures are the most robust rests on the concordance of their ranks across noise strategies (Kendall's $W = 0.9121$) rather than on large pairwise gaps. Third, the external ADME datasets (LogD, Caco-2, hERG Ki) are small, cover a narrow region of chemical and biological space, and carry an unknown quantity of intrinsic experimental noise on top of the injected noise; on the noisier endpoints---most sharply Caco-2 Efflux, where XGBoost falls to a mean AUC$_\text{norm}$ near 0.05--0.21---some models degrade below usable predictive performance, so the validation results should be taken as corroboration of the QM9 trends rather than as independent benchmarks. Fourth, our evidence is confined to regression, on a single computational QM9 target and three ADME endpoints. Finally, the apparent outlier-noise signal for VBLL-$\alpha$ ($\rho = 0.342$) sits on a contaminated $\sigma = 0$ control (baseline $\rho \approx 0.22$, an $|y|$-magnitude confound), which is why we anchor the per-sample detection claim on BNN-$\alpha$, whose clean-label control is at chance.
```

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
- BNN-$\alpha$ within-$\sigma$ $\rho=0.485$ outlier, cont-PDV, $\sigma=0.6$ | within_sigma_panelA_continuous_pdv_by_strategy.csv (0.484779) | OK
- GP within-$\sigma$ = NaN (not run on cont-PDV) | within_sigma_panelA_continuous_pdv_by_strategy.csv (GP row all NaN) | OK
- Per-strategy AUC$_\text{norm}$ spread widest Threshold/Value-Prop, narrowest Outlier | table2_auc_by_strategy_pdv.csv (max−min: Thresh 0.129, ValProp 0.111, Outlier 0.023) | OK
- rf→qrf Wilcoxon $\Delta=-0.012$, $p=2.9\times10^{-11}$ (L577e support) | ledger table3_wilcoxon_tests.csv | OK (not re-opened; ledger-cited, no CSV cell printed in prose)
- Limitations: RF 0.818/LGB 0.817/XGB 0.814/SVM 0.814 | table2_auc_by_strategy_pdv.csv | OK
- Limitations: XGBoost Caco-2 collapse 0.05–0.21 | table_validation_auc.csv (xgb Caco2: cpdv 0.0548, sns 0.1075, mhggnn 0.2107, ecfp4 0.2076) | OK
- Limitations: VBLL-$\alpha$ outlier $\rho=0.342$ | within_sigma_panelA_continuous_pdv_by_strategy.csv (0.341505) | OK
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
With:    `The NoiseInject benchmarking framework, implementing all six regression and six classification noise injection strategies together with the robustness metric (AUC$_\text{norm}$, the normalised area under the $R^2$ retention curve) and uncertainty-calibration metrics (ECE, coverage at $1\sigma$/$2\sigma$, mean interval width, and uncertainty--error / uncertainty--noise correlation) described in this work, is available as an open-source Python package under an MIT license \citep{noiseinject}.`

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
With:    `    \item[Additional file 9 (PDF):] \textit{Uncertainty quantification metrics for probabilistic models on the ECFP4 representation.} Within-$\sigma$ Unc-Noise $\rho$, Unc-Error $\rho$, ECE, and coverage at $1\sigma$/$2\sigma$ for probabilistic models under Gaussian noise on QM9.`

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
| Robustness ANOVA — **Heteroscedastic, Outlier** | **⚠ RE-RUN** | Residual dominance (77.4 / 83.6) is a VBLL-divergence artifact — roster silently grows to 10. See re-run. |
| AUC ranking (0.824…0.756), ranks, spreads, SVM outlier 0.956 | **OK** | Reproduced exactly |
| Kendall W = 0.9121, p = 3.55×10⁻⁸ | **OK (label)** | Rounds to **0.91**; basis is **across-representation mean, n=11** (state it — the PDV-only table it sits beside would give 0.9374) |
| Wilcoxon table (5 rows) | **OK** | ΔAUC$_\text{norm}$; **VBLL-α is non-significant (p=0.252)** → "both transformations improve both networks" is false |
| Interaction Spearman ρ=0.82, p=0.002, **n=11** | **OK** | Paper's 0.73/12 and the draft's 0.86 are both wrong |
| Within-σ tables + BNN-α 0.485 lead | **OK (single-seed)** | Clean σ=0 control; single run (you already state uncertainty was run once) — no CI |
| VBLL-α within-σ 0.342 | **RELABEL** | Dirty σ=0 control (0.221, \|y\|-confound); true increment ≈0.12 — corroboration only, never the headline |
| Simple-effects SMILES ~75% / PDV ~50% | **OK** | Both vary by strategy (SMILES 23–94%, PDV 19–69%); the paper's 91%/72% were slope-era |
| Top-10 (NGBoost 5 / RF 2 / SVM 1 / LGB 1 / XGB 1, no NN) | **OK** | By mean AUC$_\text{norm}$; paper's "by Gaussian NDS" counts were different |
| Excluded configs = 48 (24 VBLL + 24 BNN, gate 0.3) | **OK** | Paper's "66 at R²<0.6" is the old gate |
| Validation AUC leaderboard, XGBoost collapse, RF>QRF | **OK** | RF/PDV 0.777 > QRF/PDV **0.683** (verified from CSV); endpoint leaders were backwards in the paper (NGBoost leads hERG Kᵢ + LogD; SVM leads only Caco-2) |
| **Validation ANOVA (91.8 / 92.4 / 95.2, residual 0.0)** | **⚠ RE-RUN** | Saturated one-obs-per-cell design → residual is 0 by construction, no F/p; not credible as printed. Refit with per-fold replicates. |
| ICC / redundancy supplements | **OK** | Reproduced; captions need metric relabel |

## Figures to regenerate on the AUC$_\text{norm}$ scale

`fig1_global_overview.png`, `fig2_anova_decomposition.png`, `fig_interaction.png` (annotate **ρ = 0.82**), `fig_validation_overview.png`, `fig_validation_combined.png`, the ECFP4 supplement, and the validation-ANOVA supplement. **Re-caption only** (no regen): `fig_uncertainty_combined.png` → population-level. **Add:** `within_sigma_uncertainty.png` (fig:within_sigma). Metric-agnostic (no change): `fig_nn_family_comparison.png`, `fig_methods_noise_strategies.png`.

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

The three things the re-run must fix before those rows are publishable:
1. **Heteroscedastic + Outlier robustness ANOVA** — force one consistent roster across all six strategies (drop `dnn_vbll`/`mlp_vbll` everywhere) so the residual isn't inflated by VBLL divergence; the residual-dominance claim depends on this.
2. **Validation ANOVA** — refit keeping per-CV-fold AUC$_\text{norm}$ as replicate rows per (model, rep) cell, so a real residual and F/p exist (current 91.8/92.4/95.2 with residual 0.0 is structurally invalid).
3. **Within-σ CIs** — re-run the uncertainty experiment with ≥5 seeds if you want a confidence interval on the BNN-α 0.485 lead (optional; the single-run limitation is already stated).

Then drop the certified numbers into the ⚠ PROVISIONAL spots in the guide.
