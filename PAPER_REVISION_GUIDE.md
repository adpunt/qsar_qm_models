# NoiseInject Paper: Revision Guide

> Big-picture rebuild plan plus full draft replacements. The drafts are written plainly so you can
> re-voice them in your own words. Every number is from your own tables or confirmed against your
> results CSVs. Line numbers drift as you edit, so anchor on the quoted text.
>
> **MAJOR UPDATE (uncertainty section fully overhauled).** The uncertainty–noise metric in the
> submitted paper (Table 7, the abstract sentence, the "individual sample level" Kolmar extension)
> was computed by **pooling all σ levels together**, which measures the population-level (Kolmar)
> trend, NOT per-sample detection. A corrected **within-σ** re-analysis (verified against the
> regenerated CSVs, run 2026-07) changes the uncertainty story substantially. Everything in this
> guide touching uncertainty has been rewritten around the corrected finding; the robustness content
> (ANOVA inversion, NDS, Kendall's W, PDV tradeoff, validation) is unchanged and still verified.
> The corrected numbers live in `table4_supp_uncertainty_by_strategy_rep.csv` (per-σ columns
> `Unc-Noise ρ σ=0.0/0.3/0.6`); the new figure is `results/paper_figures/within_sigma_uncertainty.png`.
> One item is still pending a server check: the BNN-vs-VBLL "genuine detection vs |y|-magnitude
> confound" subset test (flagged inline).
>
> **The corrected uncertainty finding in one line:** per-sample uncertainty tracks injected noise
> only under noise types that corrupt an *identifiable subset* (outlier strongly, quantile
> moderately; NOT Gaussian), only on *non-embedding* representations, and best with a *distributional
> BNN* — the strongest genuine detector is **BNN-α on the PDV descriptor under outlier noise
> (within-σ ρ = 0.49)**, not GP/NGBoost. GP does **not** detect per-sample at all (its high pooled
> rank — 0.56 in the submitted table, 0.54 on the regenerated data — was pure population trend).

> **MAJOR UPDATE 2 (robustness metric replaced — 2026-07).** The robustness scalar is being
> changed from **NDS** (the OLS slope of R²(σ), a straight-line fit) to **AUC-norm** as the sole
> robustness metric. This SUPERSEDES the note above
> that "the robustness content (…NDS…) is unchanged" — all robustness numbers (ANOVA η², Kendall's
> W, the NDS-ranking table, Wilcoxon ΔNDS, the baseline-vs-robustness scatter) must be re-derived
> from the new v2 CSVs (`results/paper_figures_v2/`, produced by `scripts/generate_paper_figures_v2.py`).
> (Weibull β was trialled as a supplementary shape descriptor and DROPPED 2026-07 — weak
> discriminator, 63–92% ANOVA residual; removed from the figure script entirely.)
>
> **>>> WRITE-UP TODO (motivation for the metric change — must be explained in the expanded paper):**
> Explain *why* AUC-norm is used instead of a slope. The point: we needed a metric
> **separate from raw predictive performance** that describes the **nature of the degradation curve
> itself**. A best-fit line (NDS) does not cut it:
>   - The R²(σ) degradation curves are **nonlinear** — they plateau, then fall off a cliff, or decay
>     early then flatten. A single linear slope mischaracterizes that shape and its linearity
>     diagnostic (old `r2_fit`) confirmed the straight-line assumption was often poor.
>   - NDS was also **coupled to baseline performance** (a high-baseline model has "more to lose," so
>     a steeper slope) — so it partly re-measured performance rather than isolating robustness.
>   - **AUC-norm** = normalised area under the *retention* curve R²(σ)/R²(0): baseline-decoupled
>     (baseline coupling weak/n.s.: config-level ρ≈+0.06, Gaussian +0.13 n.s., vs NDS coupled −0.36 p=5e-4), makes no shape assumption, higher = more robust. It answers
>     "what fraction of skill is retained across the noise range."
>   - Frame it as a **curve-descriptive** metric decoupled from the performance axis — the whole
>     reason for introducing it is that robustness is a property of the *curve*, not of any one
>     point on it, and a line through the points throws that away.
>
> **Related gap for the expansion:** actual R²-vs-σ degradation curves currently appear only for
> **key models, on PDV/ECFP4, under Gaussian noise** (fig1 Panel A, fig1_supp Panel A,
> fig_nn_family). Per-dataset validation degradation curves were cut from v2. If the expansion wants
> to *show* the nonlinear/cliff behaviour that motivates AUC-norm+β, we likely need broader curve
> panels (more reps, more strategies, validation datasets) — easy to add back.

---

## The current main picture — read this first

**Two properties, two different drivers. That is the whole paper.**

1. **Noise robustness** — how much accuracy a model keeps as its labels get corrupted — is set mainly by the
   **model**, barely by the representation. Under auc_norm the model explains **10–55%** of the variance
   (depending on noise type) and the representation **under ~8%**. Rankings are stable across all six noise types
   (**Kendall's W = 0.9121**). NGBoost and the tree ensembles (RF, LightGBM, XGBoost), with SVM alongside, form
   the robust cluster; the plain neural networks are worst.
2. **Per-sample uncertainty detection** — whether a model's uncertainty can flag *which individual labels* were
   corrupted — is a **separate, much narrower** capability, driven first by the **noise type** and the
   **representation**, not the model. It appears only under subset-targeting noise (outlier/quantile, **not**
   Gaussian), only on non-embedding representations, and best from a Bayesian neural network on the PDV
   descriptor (within-σ ρ up to **0.49**). The GP does not do it at all.

The two do **not** travel together — that is the sharpest version of the argument. This spine belongs in the
abstract, the intro close, the head of each results subsection, and the conclusion.

> **How to read the rest of this guide.** The full-draft rewrites below (§1 abstract, §5 two-mechanism, §6
> conclusion) were written in the **NDS era**. Use them for *structure and voice*, but take every robustness
> number and the specifics below from the **AUC_norm cross-examination section** further down, which supersedes
> them:
> - Robustness metric is **auc_norm** (retention area, higher = more robust), not the NDS slope. All robustness
>   η², rankings, and Wilcoxon numbers come from `results/paper_figures_v2/`.
> - **"NGBoost and SVM are the most robust" is a false pairing** — SVM is 5th by mean auc_norm; it leads only
>   under outlier noise and on the external ADME sets.
> - **The PDV "highest baseline → most to lose → degrades fastest" argument is DELETED** — an NDS artifact
>   auc_norm removes. Do not re-fit it.
> - Kendall's **W = 0.9121** (not 0.92); robustness ANOVA is on **7 models**, the ranking/Kendall set is **11**.
> - **Weibull is dropped**; auc_norm is the sole robustness metric.

---

## The spine

A QSAR model's behavior under label noise has two faces, and the corrected data show they are driven by *different* things — which is itself the sharpest version of the spine. **Noise robustness** — how much predictive accuracy a model loses as labels are corrupted — is set mainly by the model's training mechanism, not the representation (the representation only modulates). **Per-sample uncertainty detection** — whether a model's uncertainty can flag which individual labels were corrupted — is a separate, much narrower capability that is *not* primarily model-driven: it is gated first by the type of noise and the representation, and only then requires a suitable (distributional) model. The unifying message is not "the model decides everything," but "robustness is the model's job, per-sample detection is a conditional property that depends on the noise and the representation" — and the two do not travel together. This should be visible in the abstract, at the end of the intro, at the head of each results subsection, and in the conclusion. Right now the spine appears in only one place, the final paragraph.

The deepest version of the spine separates two properties that are easy to confuse.

- **Robustness**: resistance to performance loss as labels are corrupted. It comes from the model's inductive bias, regularization, and ensembling, the mechanisms that limit how much it fits corrupted labels. Across the model set it depends little on the representation, though RF and NN-beta are representation-dependent while SVM, BNN-alpha, and BNN-beta are not.
- **Uncertainty tracking (per-sample)**: whether a model's uncertainty rises on the *individual* labels that were actually corrupted. This is narrow and *conditional*: it needs (i) a noise type that corrupts an identifiable subset of samples (outlier/quantile — NOT Gaussian), (ii) a non-embedding representation, and (iii) a model that learns a distribution over outputs (a distributional BNN, NGBoost, QRF). It is a distinct capability from robustness and is not driven primarily by the model.

These two properties do not always go together, and the corrected within-σ data show it (per-sample column = within-σ ρ at σ=0.6; "best case" = its strongest noise-type × representation cell):

| Model | Robust? | Per-sample uncertainty tracks noise? |
|---|---|---|
| SVM | Yes (margin, works across reps) | No uncertainty output at all |
| BNN-α (dnn_bnn_full) | Yes (weight priors, works across reps) | **Strongest genuine detector** — PDV/outlier ρ=0.49 (clean σ=0 control); ≈0 on Gaussian and on embeddings |
| BNN-β (mlp_bnn_full) | Yes | Strong — PDV/outlier ρ=0.36; ≈0 on Gaussian |
| NGBoost | Yes | Moderate — only under outlier/quantile (ρ≈0.21–0.25); ≈0 on Gaussian |
| QRF | Less than RF | Moderate under outlier/quantile (PDV ρ≈0.29); ≈0 on Gaussian |
| GP | Yes | **No** — ≈0 within-σ everywhere (−0.06 to +0.07), including under outlier/quantile; its pooled rank was a population artifact |
| VBLL-α/β | Yes (improve NN robustness) | Apparent signal under outlier, but VBLL-α has a dirty σ=0 control (≈0.22) → partly a \|y\|-magnitude confound (pending subset test) |

Being robust does not guarantee per-sample uncertainty tracking, and vice versa. SVM is very robust but gives no uncertainty at all; GP is robust and topped the (flawed) pooled ranking but does not localize noise per-sample; the strongest genuine detector, a Bayesian NN, is not among the most robust models. Per-sample detection is gated first by the **noise type** and the **representation**, not by which model — the opposite of the robustness result, where model architecture dominates.

---

## Your main points: where each is reflected

| Main point | Where it lands in the drafts |
|---|---|
| Clean performance driven by model x representation; robustness dominated by model architecture | Abstract, Intro preview, Conclusion (the inversion) |
| Most robust: NGBoost and SVM; rankings independent of noise type | Abstract, Conclusion, robustness subsection |
| Bayesian transformations improve robustness | Abstract, Conclusion |
| QRF less robust, added flexibility overfits noise | Conclusion, robustness subsection |
| Robustness not correlated with predictive performance | Intro preview, Conclusion; fixes the line 469 self-contradiction |
| RF and NN-beta are representation-dependent; SVM and full BNN are not | Two-mechanism passage, Conclusion |
| PDV: best clean-data performance, but NOT the most robust by NDS (high baseline → steepest slope); representations trade off across the three properties | Two-mechanism passage, Conclusion |
| Uncertainty (CORRECTED): per-sample noise detection is real but *narrow* — needs subset-targeting noise (outlier/quantile) + non-embedding rep + a distributional model | Abstract, Scientific Contribution, Uncertainty section, Conclusion |
| Strongest genuine per-sample detector is a Bayesian NN (BNN-α on PDV under outlier, ρ=0.49) — NOT GP/NGBoost | Uncertainty section, Conclusion |
| GP does NOT track per-sample (pooled rank was a population artifact) — population ≠ per-sample | Uncertainty section |
| Per-sample tracking collapses on Gaussian noise for every model | Uncertainty section, Abstract |
| Uncertainty tracking collapses for embeddings (mol2vec, MHG-GNN) — even under outlier, the best case | Uncertainty section, Conclusion |
| Noise *type* changes the answer qualitatively — the only place in the paper where it does (robustness rankings are stable across types, W=0.9121) | Uncertainty section, two-mechanism passage |
| Population-level: mean uncertainty rises with σ (the valid Kolmar extension) | Uncertainty section |
| Conclusions confirmed on multiple datasets | Abstract, Conclusion, validation subsection |

**One note on the BNN point (now reversed and strengthened).** The submitted paper *demoted* BNNs ("track noise weakly / do not"). The corrected within-σ data reverse this: the single strongest genuine per-sample detector is **BNN-α on continuous_pdv under outlier (ρ=0.49, clean σ=0 control −0.035, rising 0.35→0.49 with σ)**, with BNN-β close behind (0.36). BNNs are the *lead* detectors, not the laggards — but only under subset-targeting noise (outlier/quantile) and only on non-embedding reps; on Gaussian they, like everyone, sit at ≈0. Use "non-embedding" (fingerprints *and* the PDV descriptor) throughout — PDV is the single best detector representation.

---

## The big picture: seven whole units to rebuild, not patch

1. **Abstract**: a flat list of findings. It never states the spine or the inversion.
2. **Scientific Contribution**: broken mid-edit, and built on the wrong organizing idea ("aleatoric").
3. **Introduction close**: the research questions do not match the findings, and there is no "what we find" preview.
4. **Uncertainty Results section**: it meanders, states the mechanism loosely, and buries its own novelty.
5. **Two-mechanism synthesis passage**: does not exist. It is the piece that turns the paper into one argument.
6. **Conclusion**: a list, not a spine.
7. **Limitations paragraph**: does not exist. Reviewers will expect one.

---

# FULL REBUILDS

## 1. Abstract: full rewrite

**Why wholesale:** the current abstract is ordered by the order you ran experiments. This version is ordered by the argument: problem, then tool, then the inversion, then robustness, then uncertainty. It states the spine, removes SVM from the uncertainty claim (SVM produces no per-sample uncertainty), and surfaces the per-sample novelty.

> Quantitative structure-activity relationship (QSAR) models are widely used to screen compounds in drug discovery by predicting their molecular properties. The experimental labels used to train them carry noise from assay variability, measurement error, and mixed data sources. It is not yet clear how much of a model's resistance to that noise comes from the molecular representation and how much comes from the model architecture. We present NoiseInject, an open-source framework that adds controlled label noise under six different noise patterns and measures both predictive robustness and uncertainty calibration. We apply it to a wide range of representations and architectures on the QM9 dataset, using the HOMO-LUMO gap as the main task, and on three experimental ADME datasets: LogD, Caco-2 efflux, and hERG Ki. We find that the main driver depends on the question being asked. On clean labels, the interaction between model and representation explains the most variance in predictive performance. Under noise, model architecture becomes the dominant factor in the rate of performance loss, and representation explains under 10% of the variance. Robustness rankings are stable across noise types (Kendall's W = 0.92) and are not closely tied to clean-data accuracy. NGBoost and SVM are the most robust, and converting the feed-forward networks NN-alpha and NN-beta into their BNN and VBLL variants improves their robustness. Finally, we find that whether a model's uncertainty can flag which individual labels are noisy is a narrow, conditional capability: it holds only under noise that corrupts an identifiable subset of samples (not Gaussian), only on non-embedding representations, and best for Bayesian neural networks on the PDV descriptor, while the previously reported link between average uncertainty and average noise holds broadly at the population level. Noise type, which barely affects robustness rankings, is decisive here.

---

## 2. Scientific Contribution: full rewrite

**Why wholesale:** the current text is broken mid-edit and uses "aleatoric" as its organizing idea, which contradicts your own Methods statement that NGBoost cannot be decomposed into aleatoric and epistemic parts. This version drops "aleatoric", drops SVM from the uncertainty claim, removes the absolute "do not", and ends on the two-properties framing the rest of the paper delivers.

> This study introduces NoiseInject, an open-source benchmarking framework that adds controlled artificial label noise under several noise patterns and measures its effect on both predictive performance and uncertainty calibration. Using it, we show that the choice of model, not the molecular representation, is the main determinant of QSAR noise robustness. We reached this by comparing each model's relative ranking across different noise types, representations, and datasets. This is not true of clean-data accuracy, which is instead governed by the interaction between model and representation. We further show that a model's uncertainty can flag which individual labels are corrupted only under narrow conditions — a noise process that targets an identifiable subset of samples, a non-embedding representation, and a model that learns a distribution over outputs, most cleanly a Bayesian neural network on a physicochemical descriptor — whereas the link between average uncertainty and average noise holds broadly. Together these results separate two model properties, resistance to noise and the per-sample detection of it, and show that resistance is driven mainly by the model's training mechanism while per-sample detection is gated first by the noise type and the representation.

*(Do NOT organize this around "aleatoric": the corrected data show the GP's global aleatoric term does not produce per-sample detection, so the aleatoric-vs-epistemic framing that the submitted version used is backwards — the strongest per-sample detector is an epistemic Bayesian NN. Drop "aleatoric" as an organizing idea entirely.)*

---

## 3. Introduction close: recut research questions plus a new preview (replaces the `% TODO` paragraph, around line 202)

**Why wholesale:** the current RQ2 ties a robustness sub-finding (probabilistic versus deterministic) to the uncertainty result. RQ3 dissolves into two robustness sub-claims. The decoupling of robustness from clean accuracy is never asked. And there is no preview of the spine.

**Research-question paragraph:**
> This research addresses three questions about how QSAR models behave under label noise. First, how do molecular representation and model architecture split the variance in predictive performance on clean data, compared with noise robustness? Second, which architectures are most robust to noise, is robustness tied to clean-data accuracy, and is it stable across noise types and molecular properties? Third, when and why do per-sample uncertainty estimates track injected label noise? Finally, we introduce NoiseInject, an open-source Python package for benchmarking noise robustness and uncertainty quantification in machine learning models.

**New preview paragraph (this is the missing piece, and it states the spine up front):**
> We find that the roles of model and representation come apart. On clean labels, the interaction between model and representation explains the most variance in performance. Under noise, model architecture takes over and representation explains under 10% of the variance. NGBoost and SVM, the most robust models, are not the best clean-data predictors, and their advantage is stable across all six noise types and carries over to the experimental datasets. Uncertainty behaves differently again: whether a model's uncertainty can flag which individual labels are noisy is a narrow capability, present only under noise that hits an identifiable subset of samples, only on non-embedding representations, and best for Bayesian neural networks — and here, unusually, the type of noise is decisive rather than the model. Together these show that noise robustness is set mainly by the model's training mechanism, while per-sample noise detection is a separate, conditional property gated by the noise type and the representation.

**RQ to results mapping (for your own bookkeeping):**

| RQ | Delivers | Results section |
|---|---|---|
| RQ1, variance attribution | the inversion, the interaction, representation dependence, PDV | 4.1 |
| RQ2, which models and is robustness intrinsic | NGBoost and SVM, Kendall W, Bayesian transforms, QRF, the decoupling, external datasets | 4.2, 4.4 |
| RQ3, uncertainty | the scale-parameter mechanism, BNN moderate, QRF poor, embedding collapse | 4.3 |

---

## 4. Uncertainty Results section: full rebuild (currently around lines 506 to 551)

**Why wholesale:** the submitted section rests on a metric that pooled all σ levels together, which measures the population-level (Kolmar) trend, not per-sample detection. Corrected within-σ, the "GP and NGBoost strongest" ranking evaporates, GP turns out not to detect per-sample at all, and the real per-sample signal appears somewhere the submitted paper never looked: under specific noise *types*. Rebuilt as: population result → the sharper per-sample question → the metric correction → the noise-type dependence → the representation gate → the model story (with the GP reversal) → the scoped novelty.

**The correction, stated plainly (put a version of this in the text, and in Methods at the metric definition):** the per-sample uncertainty–noise correlation must be computed *within each noise level σ*. Pooling σ levels makes both predicted uncertainty and injected-noise magnitude rise together with σ, so the pooled Spearman reports that shared σ ramp — the population trend — as a strong correlation even when there is no within-σ signal. The submitted Table 7 reported these as strong (GP/SNS 0.56, NGBoost/PDV 0.47, BNN-α/SNS 0.42; "PDV" = the continuous descriptor). Re-computing the pooled correlation on the regenerated data reproduces that ranking almost exactly (GP/SNS 0.54, NGBoost/PDV 0.47, BNN-α/SNS 0.42), but within each σ level it collapses to ≈0 (GP/SNS −0.04, NGBoost/PDV 0.02, BNN-α/SNS 0.00). This is exactly the population-vs-per-sample distinction the section is supposed to be about. (Use the regenerated pooled values, from the same run as the within-σ numbers, when showing the collapse, so "before" and "after" are apples-to-apples.)

> Kolmar et al. showed that, at the population level, a Gaussian Process's mean predicted uncertainty rises with the amount of label noise in the training data. Most of the probabilistic models we benchmark behave the same way: mean predicted uncertainty increases as more noise is added (Figure~\ref{fig:uncertainty_combined}a). We ask a sharper question: within a fixed amount of noise, can a model's uncertainty point to *which individual labels* were the ones corrupted? We measured this as the Spearman correlation between predicted per-sample uncertainty and injected per-sample noise magnitude, computed *within each noise level* (pooling levels conflates the population trend with per-sample detection). The answer depends on three things at once — the noise type, the representation, and the model — and it is narrow.
>
> The dominant factor is the **noise type**. Per-sample tracking appears only when the noise corrupts an identifiable subset of samples: under outlier noise (which amplifies the noise on statistical outliers) the correlation is strong (within-σ ρ up to 0.49), and under quantile noise (which hits the label tails) it is moderate (up to 0.24). Under Gaussian noise, and under the strategies that spread noise smoothly across all samples (heteroscedastic and value-proportional), it is indistinguishable from zero for every model and representation (all ≤ 0.10). Threshold noise also collapses to zero, because on the eV-scale HOMO–LUMO gap its fixed ±1.0 cutoffs place essentially every sample in one bin, degenerating it to uniform noise. Mechanistically this is expected: a model can only flag which labels are noisy when "noisy" corresponds to a coherent, structure-linked region of samples — as it does for outliers and label tails, but not for noise sprinkled independently on every label.
>
> [[ TABLE tab:top_unc_noise GOES HERE — rebuilt: within-σ ρ, models × the six noise strategies, on PDV; see Table spec below ]]
>
> The **representation** gates whether even that is possible. Under outlier noise, the physicochemical descriptor PDV and the fingerprints carry the signal (BNN-α on PDV reaches ρ=0.49), while the learned embeddings mol2vec and MHG-GNN stay at ≈0 for every model — they fail even under the one noise type where everything else lights up (Figure~\ref{fig:within_sigma}b). So the divide is between learned embeddings and everything else, and PDV is the single best detector representation, not a fingerprint at all.
>
> [[ FIGURE fig:within_sigma GOES HERE — within_sigma_uncertainty.png: Panel a noise-type dependence on PDV, Panel b representation gate under outlier ]]
>
> The **model** matters least of the three, and here the picture inverts relative to the population trend. The strongest genuine per-sample detectors are the Bayesian neural networks: BNN-α on PDV under outlier reaches ρ=0.49, rising monotonically with σ (0.35 at σ=0.3, 0.49 at σ=0.6) from a clean zero baseline, with BNN-β close behind (0.36). NGBoost and QRF detect moderately (ρ≈0.21–0.29) under the same conditions. The Gaussian Process, which topped the pooled ranking, does *not* detect per-sample at all: its within-σ correlation is between −0.06 and +0.08 everywhere, including under outlier and quantile noise. Its high pooled value came entirely from its mean uncertainty rising with σ — a population effect — not from resolving individual noisy labels. This is the clearest illustration of why the pooled metric was misleading. (Two caveats on specific cells, see below: VBLL-α's apparent outlier signal is partly a magnitude confound, and NGBoost's quantile/PDV cell rests on a single σ level.)
>
> Where Kolmar et al. established that *average* uncertainty rises with *average* noise, we find that this population-level link holds broadly across our models and representations, but that it extends to the *individual-sample* level only under narrow conditions: a noise process that targets an identifiable subset of samples, a non-embedding representation, and a model that learns a distribution over outputs. Under those conditions — most cleanly a Bayesian neural network on the PDV descriptor under outlier noise — a model's uncertainty does point to which specific labels are corrupted. This is also the only place in our study where the *type* of label noise changes the qualitative result rather than merely its magnitude: robustness rankings are near-identical across all six noise types (Kendall's W = 0.92), but per-sample uncertainty detection exists under two of them and vanishes under the other four.

**Two caveats to state honestly in the text (both from the data):**
1. **VBLL-α magnitude confound (pending a subset test).** VBLL-α shows apparent outlier detection (ρ=0.34 on PDV) but has a *dirty σ=0 control* (ρ≈0.22 where it should be ~0): its uncertainty tracks label magnitude |y|, and outlier noise targets |y|-extreme samples, so part of its "detection" is flagging extreme molecules rather than responding to noise. BNN-α is clean here (σ=0 control −0.035), so anchor the strong claim on BNN-α. **[PENDING] A direct subset test** (does uncertainty on the corrupted subset *rise* from σ=0 to σ=0.6, or was it already high at σ=0?) will confirm BNN-α genuine vs VBLL-α confounded — awaiting the raw per-sample CSVs from the server.
2. **NGBoost/quantile/PDV thin slice.** That cell's σ=0.3 point is missing (n=0), so its 0.24 rests on the σ=0.6 slice alone. Report it with that caveat or drop it; it is not load-bearing (the headline is BNN-α/outlier).

**Verify before finalizing:** the GP mechanism question ("is the Gauche GP likelihood homoscedastic?") is now *moot for the per-sample claim* — GP does not detect per-sample regardless, so no strong GP claim needs defending. The old worry about the GP correlation being a data-density artifact is resolved: it was a σ-pooling artifact. Anchor the strong per-sample claim on **BNN-α under outlier**, whose within-σ signal is clean and monotone in σ.

**New Table 7 content (within-σ ρ at σ=0.6, representation = continuous_pdv; models × noise strategies).** Source: `table4_supp_uncertainty_by_strategy_rep.csv`, columns `Unc-Noise ρ σ=0.6`. Keep the Unc-Error ρ / ECE / coverage columns from the old table (unaffected by the σ-pooling bug) as secondary columns if you like, but the headline column is the within-σ noise ρ. State in the caption that ρ is computed within a fixed σ and that σ=0.6 is shown (σ=0.3 mirrors it, lower).

| Model | Gaussian | Threshold | Hetero. | Value-Prop. | Quantile | Outlier |
|---|---|---|---|---|---|---|
| BNN-α | −0.07 | 0.03 | −0.01 | n/a | 0.13 | **0.48** |
| BNN-β | −0.04 | 0.01 | −0.02 | −0.01 | 0.21 | **0.36** |
| VBLL-α† | 0.02 | 0.02 | 0.06 | 0.01 | 0.19 | 0.34 |
| VBLL-β | 0.03 | −0.01 | 0.06 | 0.01 | 0.16 | 0.22 |
| QRF | 0.02 | −0.01 | 0.02 | 0.03 | 0.18 | 0.29 |
| NGBoost | 0.02 | 0.02 | 0.05 | 0.03 | 0.24‡ | 0.21 |
| GP | — | — | — | — | — | — |

- Only the **Outlier** and **Quantile** columns are non-zero; the other four are ≈0 (the noise-type dependence — this is the table's point).
- **† VBLL-α** has a dirty σ=0 control (≈0.22) — its outlier/quantile values are partly a \|y\|-magnitude confound (pending the subset test); footnote it.
- **‡ NGBoost/Quantile** rests on σ=0.6 only (σ=0.3 slice missing, n=0); footnote or drop.
- **GP** has no continuous_pdv uncertainty run (a data gap, `gauche_rbf` was never run in the main pipeline). On the reps where GP *does* run (pdv, ecfp4, sns, morgan), its within-σ ρ is between −0.06 and +0.08 under *every* strategy including outlier/quantile — so add a GP row from `pdv` (Outlier 0.07, Quantile 0.07, Gaussian −0.04) or state the GP-does-not-detect result in the caption/text. Do not leave GP silently absent, since "GP does not detect per-sample" is a key point.
- **Companion (representation gate):** a second small table or the Panel b heatmap — within-σ ρ under Outlier, models × representations — carries "embeddings collapse even under outlier" (mol2vec/MHG-GNN ≤ 0.08 for every model vs PDV 0.49).

---

## 5. New: two-mechanism synthesis passage (the keystone, end of Results or top of Conclusion)

**Why new:** this passage does not exist anywhere, and it is the single thing that turns the paper from a list of findings into one argument.

> Two questions run through these results, and each is controlled by a different mechanism. The first is noise robustness: how well a model keeps its predictive accuracy when its training labels are corrupted. This is set mainly by the model's training mechanism, the inductive bias, regularization, and ensembling that limit how much it fits corrupted labels. In absolute terms NGBoost and SVM were the most robust, followed closely by RF, LightGBM, and XGBoost, while NN-alpha and NN-beta were the least robust. A separate axis is consistency across representations: SVM, BNN-alpha, and BNN-beta stayed robust on every representation, SVM through margin maximization and the full BNNs through weight priors, whereas RF and NN-beta were robust only with certain representations. The representation itself plays a secondary role, accounting for under 10% of NDS variance for most noise types, though the interaction between model and representation is real. The representations trade off across the three properties rather than one dominating: PDV gives the strongest clean-data performance and tracks per-sample noise as well as the best fingerprint, but by raw NDS it degrades among the *fastest* of all representations, because its high baseline R² leaves the most performance to lose; the embeddings that degrade *slowest* by NDS (mol2vec, MHG-GNN) — partly a low-baseline headroom effect — collapse completely on uncertainty tracking. No single representation wins on all three. (Data: `table2_supp_auc_all_reps.csv`, `table_all_configurations.csv` baseline R², `table4_uncertainty_metrics_{mol2vec,mhggnn}.csv`.)
>
> The second question is whether a model's uncertainty can flag *which individual labels* were corrupted, within a fixed amount of noise. This is a much narrower capability than robustness, and it is gated by three things at once. It requires, first, a noise process that corrupts an identifiable subset of samples: it appears under outlier and quantile noise, which amplify the corruption on statistical outliers and label tails, but vanishes under Gaussian noise and the strategies that spread noise smoothly across all samples. Second, it requires a non-embedding representation — fingerprints and the PDV descriptor expose the signal, while the learned embeddings MHG-GNN and mol2vec hide it even under the noise types where everything else works. Third, it requires a model that learns a distribution over outputs; the strongest genuine detector is a Bayesian neural network on PDV under outlier noise (within-σ ρ up to 0.49), with NGBoost and quantile random forests moderate. Notably, the Gaussian Process — which appears strongest if the correlation is (incorrectly) pooled across noise levels — does not resolve individual noisy labels at all; its apparent signal is the population-level rise of mean uncertainty with noise, not per-sample detection. Unlike robustness, this property is set primarily by the noise type and the representation, not by the model.
>
> The two properties do not always go together, and they are driven by different factors. SVM is among the most robust models but gives no uncertainty at all. The Gaussian Process is robust and its mean uncertainty tracks the overall noise level, but it cannot point to individual noisy labels. The Bayesian neural networks are robust *and* the best per-sample detectors, but only under the right noise type and representation. A practitioner choosing a model that survives noisy labels, and one that can also flag which labels are noisy, is choosing against two different criteria — and where robustness is set by the model's training mechanism almost regardless of representation or noise type, per-sample detection is the opposite: it is a narrow, conditional capability that depends first on the kind of noise and the representation.

---

## 6. Conclusion: full restructure

**Why wholesale:** the current conclusion is a list. This version follows the spine: the inversion, then robustness (with the decoupling and the stability), then the two-properties uncertainty result, then generalization with caveats, then the practical takeaway.

> We investigated how molecular representation and model architecture shape the behavior of QSAR models under increasing label noise. We used six noise-injection strategies on the QM9 HOMO-LUMO gap task and three experimental ADME datasets. The model and the representation play different roles depending on the question. For predictive performance on clean labels, the interaction between model and representation explained the most variance. For robustness, measured by the noise degradation slope, model architecture was the dominant factor, and representation explained under 10% of the variance for most noise types. Label noise corrupts the targets, not the features, so the contribution of the representation shrinks, while the model's regularization, ensembling, and priors decide how corrupted labels are absorbed.
>
> Within robustness, model rankings were highly concordant across noise types (Kendall's W = 0.92) and were largely decoupled from clean-data accuracy. NGBoost and SVM were among the most robust without being the best clean-data predictors. SVM, BNN-alpha, and BNN-beta held their robustness across all representations, while RF and NN-beta were robust only with certain representations. No single representation dominated. PDV gave the best clean-data performance and tracked per-sample noise as well as the best fingerprints, but by raw NDS it degraded fastest — its high baseline R² leaves the most to lose — while the embeddings that degraded slowest by NDS (mol2vec, MHG-GNN) failed catastrophically at per-sample uncertainty tracking. With neural networks specifically, the embeddings were the weakest. Converting NN-alpha and NN-beta into their BNN and VBLL variants reliably improved robustness. QRF was consistently less robust than RF, which suggests that its added flexibility overfits noisy labels rather than absorbing them.
>
> Robustness and per-sample uncertainty detection are different properties with different drivers. The population-level link between uncertainty and noise reported by Kolmar et al. — mean uncertainty rising with the overall noise level — held broadly across our models. But whether a model's uncertainty can point to *which individual labels* were corrupted proved narrow and conditional: it required a noise process that targets an identifiable subset of samples (outlier or quantile noise, not Gaussian), a non-embedding representation, and a model that learns a distribution over outputs. Under those conditions the Bayesian neural networks were the strongest detectors, most cleanly on the PDV descriptor under outlier noise; the Gaussian Process, whose mean uncertainty tracks the overall noise level, did not resolve individual noisy labels. Uniquely among our results, the *type* of label noise was decisive here, whereas robustness rankings were near-identical across all six types.
>
> These conclusions held on experimental data. Model architecture again dominated robustness on all three ADME datasets, though the smaller and noisier external sets reordered some rankings, and XGBoost in particular degraded. In practice, resistance to noise and the ability to detect which labels are noisy are governed differently: robustness is set mainly by the model's training mechanism, and the most robust choices (NGBoost, SVM) hold across representations, whereas per-sample noise detection is a narrow, conditional property — available only under noise that targets an identifiable subset of samples, only on non-embedding representations, and best from a Bayesian neural network on a physicochemical descriptor. A practitioner who needs both a model that survives noisy labels and one that can flag which labels are noisy is optimizing two different, only partly overlapping criteria.

---

## 7. New: Limitations paragraph (before the closing paragraph of the Conclusion)

**Why new:** your caveats are currently scattered. JCI reviewers expect one consolidated paragraph, and it pre-empts the three main objections (single-seed uncertainty runs, the tight top ranking, the small external sets).

> This study has several limitations. The uncertainty experiments were run once per configuration, so the uncertainty-noise correlations are indicative rather than precisely estimated. Replication would allow confidence intervals on the per-sample rankings. The robustness differences among the top models (NGBoost, SVM, RF, LightGBM, XGBoost) are small compared with the gap to NN-alpha and NN-beta, and our claims rest on the stability of their relative ranking rather than on large pairwise gaps. QM9 served as a clean baseline with negligible measurement noise, while the experimental datasets carry an unknown noise floor on top of the injected noise and are small enough that several models, including XGBoost and the BNN variants, fell below usable performance. Finally, we limited the analysis to regression on a single quantum-mechanical target and three ADME endpoints. Classification, which NoiseInject supports, was not tested.

---

# Two structural moves (reorganization, not prose)

- **Split the overloaded robustness subsection (around lines 443 to 504).** It crams five findings into one block. Break it into three: (i) which models are robust, and the decoupling from clean performance; (ii) representation modulates robustness (the interaction, SVM and full BNN being representation-agnostic versus RF and NN-beta being representation-dependent, and PDV); (iii) the within-family probes (Bayesian transforms, QRF), which are your cleanest mechanism-isolating evidence.
- ~~**Fix the figure ordering.**~~ **OBSOLETE — no longer applies.** Verified against the current `paper.tex`: figures are referenced in strict numerical order. The "global overview" is **Figure 4** (`fig1_global_overview.png` despite the filename), correctly referenced *after* the ANOVA (Fig 2) and interaction (Fig 3) figures. This item described an older draft; no change needed.

---

# Beyond wording: credibility issues to decide on (likely with your supervisor)

> **Note (auc_norm era):** items **#1** and **#3** were written under NDS and are partly overtaken by the metric
> switch. **#1** — the tight-cluster point still holds under auc_norm (NGBoost 0.824 / RF 0.818 / LGB 0.817 /
> XGB 0.814 / SVM 0.814 within ~0.01), but note SVM is now **5th**, not co-leader; the "can't single out SVM
> without per-seed spread" caution is *stronger*, not weaker (see the cross-examination section). **#3** — the
> range-restriction worry is largely **resolved**: auc_norm is baseline-decoupled, and the gate is now R²<0.3
> (48 configs excluded), not R²<0.6 (66). Items #2, #4, #5, #6 are unaffected by the metric change.

1. **The top NDS ranking is a tight cluster.** Confirmed in the data: on PDV under Gaussian noise, RF, SVM, LightGBM, XGBoost, and QRF all fall within 0.013 of each other (-0.359 to -0.372), with NGBoost marginally ahead (-0.331). The plain NNs are clearly worse (DNN -0.407, MLP -0.488). So the defensible claim is a tight top cluster of tree and SVM models, with the NNs separated below. Singling out SVM over RF, LightGBM, and XGBoost needs a per-seed spread, which is not in the local data and would have to come from the server. Kendall W = 0.92 is about rank agreement across strategies, not a test that models differ from each other.
2. **The uncertainty results are single-seed.** Confirmed: each uncertainty configuration was run once. The corrected within-σ correlations are each computed over many samples per σ slice (n = 1,000–10,000), so within a run they are well-powered, but there is still only **one training seed per cell** — so the *ranking* among detectors (e.g. BNN-α 0.49 vs BNN-β 0.36 vs QRF 0.29 under outlier) has no error bars across seeds. Present the finding as a gradient (strong / moderate / absent) and as a qualitative noise-type × representation pattern rather than a precise per-cell ranking; a few extra seeds on the headline cells (BNN-α/PDV/outlier especially) would let you put a CI on the lead. Two cells are additionally thin and must be flagged or dropped: NGBoost/quantile/PDV (σ=0.3 slice missing, n=0) and any QRF cell with an n=1 σ-slice. The Limitations paragraph covers the single-seed point.
3. **The inversion may be partly range restriction.** "NDS clusters near -0.38 regardless of baseline R2" is computed after dropping 66 configurations with R2 below 0.6. Restricting the R2 range can mechanically lower the correlation with NDS. Defend it (show the relationship with the filter relaxed) or argue that the filtered range is the relevant one.
4. **The word "comprehensive" over-promises.** This is a controlled benchmark, not a survey. "Comprehensive" invites "you did not test X". Consider softening. Minor.
5. **External "confirmation" is overstated.** The external sets are small, XGBoost and the BNN variants drop below R2 of 0.3 on the hardest set (hERG), and rankings reorder (SVM overtakes NGBoost when all six strategies are pooled). Soften to "broadly consistent, with notable exceptions", which is close to what you already say.
6. **The framework advertises classification and conformal prediction, which never appear in the results.** Either show a small example or state clearly that they are framework features, not benchmarked here.

---

# Mechanical fixes

| Location | Issue | Fix |
|---|---|---|
| Line 164 | SVM listed in the uncertainty sentence | Remove SVM (it has no per-sample uncertainty) |
| Lines 167 to 169 | Scientific Contribution broken mid-edit AND built on the reversed aleatoric framing | Replace per §2 (drop "aleatoric" as the organizing idea) |
| **Line 240** | **Metric definition** — "Spearman between uncertainty and injected noise magnitude" is computed POOLED over σ (the root of the error) | **State it is computed *within each σ level*; the pooled version measures the population trend, not per-sample detection.** Fix here or every downstream number stays self-contradicting |
| **Lines 306–308, 380, 624** | Restatements of the same metric (Table 1 metrics box; NoiseInject framework description) | Same within-σ clarification, or label pooled versions "population-level" |
| Line 169 / 547 / 581 | "Bayesian Neural Networks... do not [track noise]" / "track noise weakly" | **REVERSED by the data — do NOT soften to "weakly"; BNNs are the strongest genuine per-sample detectors (BNN-α/PDV/outlier ρ=0.49). Rewrite per §4.** |
| Line 469 | "more prone to picking up on noise" contradicts robustness | State the decoupling instead |
| Lines 508, 549, 579 | "fingerprints" undersells PDV (a descriptor) | "non-embedding representations" |
| Lines 504 vs 562 | SVM vs NGBoost on ADME: the two sentences use different noise slices | State one lens, or report both (NGBoost under Gaussian, SVM across all six strategies) |
| **Table 7 (lines 517–537) whole table** | Every Unc-Noise ρ value is a pooled-σ (population) artifact; the table is Gaussian-only, where per-sample ρ≈0 | **Regenerate as within-σ, models × noise strategies (see §4 Table spec). Unc-Error ρ / ECE / coverage columns are unaffected and can stay.** The Morgan/ECFP4 naming and "PDV (binary)" issues are subsumed by the regeneration |
| **Additional file 9 (line 671)** | Gaussian, pooled Unc-Noise ρ — same artifact | Regenerate within-σ, or relabel its Unc-Noise ρ as population-level |
| Lines 260 vs 432 | "R2 <= 0.6" vs "R2 < 0.6" | Pick one |
| Line 274 | "observations... are independence" | "are independent" |
| ~~Figures~~ | ~~ANOVA/interaction figs before global-overview~~ | **OBSOLETE** — current `paper.tex` references figures in numerical order; global overview is Fig 4, correctly after ANOVA/interaction |

---

# Figure accuracy audit (all 8 figures viewed against the rendered PNGs)

Each figure was opened and checked against its caption and the body sentences that cite it. **Re-checked against
the regenerated `results/paper_figures_v2/` (auc_norm) figures — 2026-07.** Several items the original NDS-era
audit flagged are now **resolved by the v2 regeneration** (marked ✅ RESOLVED below); the remaining items are
paper.tex wording/caption fixes. Where a number changed under auc_norm, the new value is given.

- **Fig 1 — noise strategies** (`fig_methods_noise_strategies.png`): accurate. No change.
- **Fig 2 — ANOVA decomposition** (`fig2_anova_decomposition.png`): ✅ **RESOLVED by v2.** The v2 figure now
  plots a fourth grey **Residual** bar, so the old "Residual is dropped / Outlier bars misread" problem is gone —
  the residual-dominated panels are now visible directly (Outlier Residual ≈84%, Heteroscedastic ≈77%). Panel
  titles now state the roster: **panel a "11 models" (performance), panel b "7 models" (robustness)** — use these,
  and note in the caption that the robustness panel is the 7-model balanced set.
- **Fig 3 — interaction** (`fig_interaction.png`): accurate (heatmap + scatter, N/A cells present). Two updates
  under auc_norm: (i) the scatter Spearman is now **ρ = 0.86** (was 0.73 under NDS) — update the body number;
  (ii) the "11 models" body text is still off — the scatter plots **12** points (all models except GP, which is
  N/A on PDV). Change "11 models" → "12 models" and ρ → 0.86 at the ECFP4-vs-PDV sentence (paper.tex:~423).
  Heatmap is now Gaussian AUC$_{norm}$ (13 model rows × 5 reps), higher = more robust.
- **Fig 4 — global overview** (`fig1_global_overview.png`): now **two panels** in v2. Panel a = R² degradation
  curves (PDV, Gaussian) for RF, QRF, NN-α, NN-β, NGBoost, GP (RBF), XGBoost — GP (RBF) top, **NN-β lowest/steepest**
  (note: SVM is **not** in panel a, so don't cite "NGBoost/SVM shallowest" against it). Panel b = a new
  **AUC$_{norm}$ heatmap** (13 models × 6 strategies, PDV), higher = more robust — NN-β lowest (Gaussian 0.79),
  the tree ensembles + NGBoost highest. Re-word any "shallowest/steepest slope" caption phrasing to auc_norm terms.
- **Fig 5 — NN family comparison** (`fig_nn_family_comparison.png`): **still valid under v2.** Panel b (NN-β,
  BNN-β on top) and panel c (RF vs QRF, near-overlapping with QRF marginally below) support the claims. **Panel a
  (NN-α) still does not**: in absolute R², BNN-α sits on top of plain NN-α and **VBLL-α is the lowest curve at
  every σ**. This now aligns with the cross-examination finding that **VBLL-α does not significantly improve NN-α
  (Δ+0.011, p=0.25, NS)** — only the full BNNs (both families) and VBLL-β do. So temper "both NN-α and NN-β
  improve" to "the full BNNs improve both families; VBLL improves NN-β but not NN-α," and state that robustness
  here means **auc_norm / the Wilcoxon test**, not absolute R² (paper.tex:~475).
- **Fig 6 — uncertainty combined** (`fig_uncertainty_combined.png`): **still valid under v2** (uncertainty
  figure, unaffected by the robustness metric swap). Shows only the **population-level** story (mean uncertainty
  vs σ, panels a/b) — which SURVIVES the correction and is fine, but it must be explicitly captioned as
  population-level and NOT read as per-sample detection. Two panel-caption claims still need the earlier wording
  fixes (panel a: not *every* model rises steadily — QRF spikes to ≈0 at σ=0.3, VBLL-α erratic, GP absent; panel
  b: epistemic does **not** stay flat — both aleatoric and epistemic rise; paper.tex:547 still says "epistemic
  remains nearly flat", fix there too).
  **NEW — add the per-sample figure.** The corrected per-sample result needs its own figure: `results/paper_figures/within_sigma_uncertainty.png` (generated from the data, script `build_within_sigma.py`, panel data in `within_sigma_panel{A,B}_*.csv`). Panel a = within-σ ρ at σ=0.6, models × the six noise strategies on PDV (outlier/quantile hot, the other four ≈0 — the noise-type dependence); panel b = within-σ ρ under outlier, models × representations (PDV/fingerprints carry it, mol2vec/MHG-GNN ≈0 — the representation gate). This is the figure that carries the corrected finding; pair it with the rebuilt Table 7. Together they replace the reliance on the old Gaussian pooled Table 7 as evidence of "detection."
- **Fig 7 — validation overview** (`fig_validation_overview.png`): ✅ **RESOLVED/CHANGED by v2** — but the
  paper.tex **caption must be rewritten to match**. The v2 figure is now titled *"Robustness (AUC$_{norm}$) on PDV
  representation"*, three panels (LogD / Caco-2 Efflux / hERG K$_i$), the **same 5 models** (RF, QRF, XGBoost,
  LightGBM, NGBoost) in each. The old NDS caption language (black cells for R²<0.3, "N/A" for |NDS|>2, fewer hERG
  rows) is now **moot**: the figure uses the auc_norm colour scale (−0.5→1.1), shows **negative auc_norm directly
  in purple** (e.g. XGBoost/Caco-2/Gaussian −0.40, XGBoost/hERG/Threshold −0.16), and has exactly **one grey N/A
  cell** (XGBoost × Value-Prop on Caco-2, a missing config). Rewrite the caption to: 5 models × 6 strategies ×
  3 datasets, auc_norm (higher = more robust), negative = worse-than-baseline, one N/A cell. Drop the |NDS|/black-cell wording.
- **Fig 8 — validation combined** (`fig_validation_combined.png`): **still valid under v2.** Panel a = 4-dataset
  auc_norm bars for 7 models (RF, QRF, XGBoost, LightGBM, NGBoost, SVM, NN-α); panel b = QM9-vs-external scatter
  with SVM/NGBoost stable and **XGBoost clearly lowest external** (Caco-2 ≈0.14, external mean ≈0.53). But body
  line 562 cites panel a for "XGBoost **and BNN variants** suffer the most" — **no BNN models appear in this
  figure** (none exist in the external data). Drop "and BNN variants" from that sentence.

---

# Verify in data or code (not wording)

- SVM uncertainty output: **resolved**. SVM appears in zero uncertainty artifacts. The abstract claim is unsupportable.
- ADME ranking: **resolved**. NGBoost leads under Gaussian noise only. SVM leads across all six strategies (more robust on hERG and Caco-2, mean -0.158 vs -0.186; NGBoost wins LogD). Reconcile the two sentences.
- Single-seed uncertainty: **resolved**. Confirmed n = 1, no error bars.
- PDV versus fingerprints: **resolved (within-σ).** The old "PDV 0.358 ≈ SNS 0.360" comparison was pooled-Gaussian (population artifact). Within-σ under outlier noise, **PDV is the single best detector representation** — BNN-α/PDV/outlier ρ=0.49 vs ~0.12–0.28 for the same models on fingerprints (e.g. BNN-α/SNS 0.23, BNN-α/topological 0.18, BNN-α/Morgan 0.12) — and the learned embeddings (mol2vec, MHG-GNN) collapse to ≈0 even under outlier. Use "non-embedding", and note PDV leads.
- PDV as "most robust representation": **resolved — CONTRADICTED (QM9).** Holding the model fixed, continuous_pdv ("PDV") degrades *faster* (steeper NDS) than mol2vec/morgan for every model — e.g. NGBoost mol2vec −0.292 / morgan −0.288 / PDV −0.357; SVM −0.355 / −0.342 / −0.385; RF −0.306 / −0.330 / −0.386 (`table2_supp_auc_all_reps.csv`). Cause: PDV has the highest baseline R² (~0.84–0.89 vs mol2vec ~0.69–0.82, `table_all_configurations.csv`), so it has the most performance to lose → steepest slope. mol2vec has the shallowest mean NDS but collapses on uncertainty tracking (ρ≈0, ECE up to 24). The "PDV most robust" claim was **removed** from `paper.tex` (lines 471, 502) and this guide, replaced with the three-way tradeoff framing. Note the partial range-restriction artifact (issue #3): mol2vec's apparent robustness is partly a low-baseline headroom effect, not intrinsic noise resistance.
- Per-seed NDS spread of the top models: **still open**. Not in the local repo. Pull from the server if you want a pairwise significance claim.
- Gauche GP likelihood (homoscedastic or not): **now MOOT.** The corrected within-σ analysis shows GP does not detect per-sample at all (ρ between −0.06 and +0.08 everywhere, including under outlier/quantile), so there is no strong GP per-sample mechanism to defend. Whether its likelihood is homo- or heteroscedastic no longer bears on any claim.
- **[PENDING] BNN-vs-VBLL subset test (the one open data task).** Confirm that BNN-α's outlier signal is genuine noise detection (uncertainty on the corrupted \|z\|>2 subset *rises* from σ=0 to σ=0.6) rather than a \|y\|-magnitude confound, and quantify how much of VBLL-α's apparent signal is confound (its σ=0 control is a dirty ≈0.22). Needs four raw per-sample CSVs from the server (`uncertainty_{outlier,legacy}_continuous_pdv_{dnn_bnn_full,mlp_bnn_full,dnn_bnn_full_variational}_uncertainty_values.csv`). Until then, the guide credits **BNN-α** (clean σ=0 control) as the genuine detector and flags VBLL-α as "possibly magnitude-linked."
- **Cross-section link — leverage it (paper line 393).** The Variance-decomposition section *already* characterizes the noise strategies mechanistically: "outlier noise only affects statistical outliers," while threshold/value-proportional "corrupted label regions." The first half is a gift — the paper already establishes that outlier noise hits an identifiable subset, which is *exactly* the property that makes it per-sample detectable. Connect the uncertainty result back to this sentence so the two sections reinforce each other (subset-targeting noise → both a distinct ANOVA signature *and* the only regime where uncertainty localizes noise).
- **Threshold degeneracy — reconcile, and verify the raw target range.** The code analysis found `threshold`'s fixed ±1.0 cutoffs sit below the eV-scale HOMO–LUMO gap, so ~100% of samples land in the single `2σ` bin → it degenerates to *uniform 2σ noise*, not region-corrupting. This is consistent with the data on both sides (threshold's within-σ uncertainty ρ ≈ 0 like Gaussian, AND threshold degrading performance the most — 2σ everywhere). But it means line 393's "threshold... corrupted label regions" is imprecise, and that `threshold` is not really testing a distinct noise *pattern* on this target. **Verify:** confirm the raw (pre-normalization) HOMO–LUMO gap range is > 1.0 eV (agent reported ≈0.67–16.9 eV, only 1/133,885 below 1.0) before leaning on this in text — then either footnote the degeneracy or drop threshold from the "region-corrupting" grouping.
- **[SEPARATE, robustness-side] valprop noise may not scale with the σ-sweep.** The code sets `value_proportional`'s `base_sigma` to a fixed 0.1 (read from `noise_strategy_params.json`), independent of the σ level being swept. If confirmed, the value-proportional NDS/degradation results are computed at an effectively constant noise level rather than increasing σ — which would affect the robustness (not uncertainty) analysis. Worth checking against the raw valprop injected-noise magnitudes per σ. Not addressed in this guide's uncertainty rewrite; flag for the robustness pass.
- **[SEPARATE, methods] NN-β architecture mismatch.** Methods states NN-β is [128,128], but the code builds a uniform-width Optuna-tuned MLP (default 32×2), not a hard-coded [128,128]. Reconcile the Methods description with what the code builds (the α↔dnn / β↔mlp identity is correct; only the layer sizes are off).

---

# Suggested priority order

0. **Load-bearing data fix first: regenerate Table 7 and Additional file 9 within-σ, add the within-σ figure, and correct the metric definition (line 240).** Everything else in the uncertainty story depends on this — if the prose is fixed but the pooled numbers stay, the section contradicts itself. The regenerated numbers and figure already exist (`table4_supp_uncertainty_by_strategy_rep.csv`, `within_sigma_uncertainty.png`); the one open task is the BNN-vs-VBLL subset test.
1. Rebuild the uncertainty section (#4) around the noise-type-dependent finding, and align the abstract and Scientific Contribution (#1, #2) — this is now the largest change and the one that most affects the paper's claims. Drop the reversed "GP/NGBoost strongest per-sample / BNNs weak" framing everywhere.
2. Drop in the two-mechanism synthesis (#5) and lead with the inversion. Biggest narrative payoff on the robustness side.
3. Recut the research questions and add the preview (#3), restructure the conclusion (#6), and add the Limitations paragraph (#7).
4. Do the structural splits, the mechanical fixes, and the credibility decisions last.

---

# AUC_norm CROSS-EXAMINATION (2026-07 — supersedes NDS content above)

> Produced by a multi-agent cross-examination of the submitted `paper.tex` against the regenerated
> `results/paper_figures_v2/` CSVs after the NDS → **auc_norm** (sole robustness metric)
> change. Verified against the CSVs, not memory. (Weibull β, mentioned as "supplementary" throughout
> this section, was later DROPPED — 2026-07; ignore every weibull_beta note below.) **This section supersedes every NDS-based number and
> claim in the drafts above** (including the §1/§5/§6 full-draft rewrites, which were written in the NDS era —
> see the "do not paste verbatim" warning at the end). Figures were re-audited and the code fixes are already
> applied to `generate_paper_figures_v2.py`.

## Verdict: the spine survives and is *strengthened* — but three claims break

The two-mechanism spine (**robustness is the model's job; per-sample uncertainty detection is a separate
conditional capability**) holds under auc_norm and is cleaner than before:
- **Model dominates robustness, representation is negligible** — auc_norm ANOVA Model η² = 10–55%, Rep η² =
  **0.2–7.9%** (even smaller than under NDS). Rankings stay concordant across the 6 strategies: **Kendall's W =
  0.9121** (n=11, χ²=54.73, p=3.55×10⁻⁸).
- **auc_norm removes NDS's baseline coupling (metric/config level)** — ⚠ CORRECTED after the real-data figure
  audit. The honest, verified claim is a *relative, config-level* one: across all model×rep configs under Gaussian
  noise, **NDS was significantly coupled to baseline (ρ = −0.36, p = 5×10⁻⁴)** — the "more to lose" artifact —
  whereas **auc_norm is not (ρ = +0.13, p = 0.26, n.s.;** all-config ρ ≈ +0.06). So the metric no longer
  mechanically re-measures clean performance. **This is NOT the same as "robustness and accuracy are unrelated."**
  See the fig3 decision below — my earlier "|ρ|≈0.006, fig3 flat" was wrong on two counts (0.006 came from old
  screening data; and fig3 plots n=11 model *means*, where ρ = +0.43, i.e. better models are both more accurate
  and more robust — a real empirical pattern, not decoupling).

**Three claims break and must be fixed (not patched):**
1. **"NGBoost and SVM are the most robust" → false pairing.** By mean auc_norm on PDV, SVM is **5th**
   (0.814), behind RF (0.818), LightGBM (0.817) and level with XGBoost (0.814). The robust cluster is NGBoost +
   the three tree ensembles; SVM is a member, and leads *only* under outlier noise and on the external ADME sets.
2. **The PDV "headroom / most-to-lose" argument → DELETE, do not re-fit.** "PDV has the highest baseline so it
   degrades fastest / mol2vec is shallowest" is a pure NDS baseline-coupling artifact — exactly what auc_norm
   removes. Under auc_norm PDV is mid-pack-to-strong and mol2vec is no longer "shallowest." Leaving this in
   *contradicts the paper's own new flat-decoupling result* (lines 469, 471, 502; the §5 keystone passage).
3. **"Model dominant" and "Bayesian transforms help" are over-general.** Model η² dominates only 4/6
   strategies; **Heteroscedastic joins Outlier as residual-dominated** (η² Model 14% / 10%, Residual 77% / 84%).
   And **VBLL-α no longer significantly improves NN-α** (Δ+0.011, p=0.25, NS) — only full-BNN (both families)
   and VBLL-β do.

**Weibull β was DROPPED (2026-07).** It was a weak discriminator (ANOVA 63–92% residual; model signal only
on threshold 18.6% / valprop 12.8%), so it has been removed from the figure script and the paper. auc_norm is
the sole robustness metric. Keep the two-mechanism spine; do NOT introduce a shape axis.

## The metric definition rewrite — Methods §Performance Metrics (lines 234–323) [CRITICAL]

This is the load-bearing edit. Replace the NDS definition with:
- **AUC_norm** (sole robustness metric) = normalised area under the retention curve R²(σ)/R²(0), trapezoidally integrated over
  σ∈{0…1.0}; range ≈[0,1], **higher = more robust**, no shape assumption, weakly/non-significantly coupled to baseline (config-level ρ≈+0.06; NDS was −0.36 under Gaussian).
- **Motivation to state explicitly** (this is what makes the decoupling headline coherent, and is a genuine
  *third contribution*): (i) R²(σ) curves are nonlinear (plateau-then-cliff / early-collapse), so a single OLS
  slope mischaracterises shape — cite the old `r2_fit` linearity diagnostic showing the straight line was often
  poor; (ii) a slope is baseline-coupled, so NDS partly re-measured clean performance; (iii) auc_norm fixes both.
- Rewrite line 260's "values near zero / negative = sensitive / positive slopes" language entirely (auc_norm
  near 1 = robust; well below 1 = degraded).
- Update the **exclusion rationale**: gate relaxed to **baseline R² < 0.3** (`ROBUSTNESS_BASELINE_THRESHOLD`,
  justified because auc_norm is baseline-decoupled) **+ ≥5 valid iterations/cell**; the old "misleadingly shallow
  slopes" wording is slope-specific and now wrong.
- **Metrics-summary table (tab:metrics_summary, 291–293):** delete the NDS row; add an AUC_norm row.
- **ANOVA metric line 262:** "either R² at fixed noise or NDS" → "…or AUC_norm".
- **7-vs-11 model caveat** (state wherever the metric/roster is introduced): the robustness ANOVA runs on **7
  models** (cells need ≥5 valid iters on every rep for balance; the four Bayesian/VBLL NN variants — dnn_bnn_full,
  mlp_bnn_full, dnn_vbll, mlp_vbll — drop out on the embedding reps mhggnn/mol2vec) while the ranking table and
  **Kendall's W use 11 models**. Excluded configs **66 → 48**. (Confirmed against fig2: panel a "11 models",
  panel b "7 models".)

## Section-by-section change list (verified numbers)

### Results: Robustness across strategies (441–505) [CRITICAL]
The `tab:nds_ranking` table must be **rebuilt entirely**. Model order flips from NDS to:
`ngboost 0.824 > rf 0.818 > lgb 0.817 > xgboost 0.814 > svm 0.814 > BNN-β 0.802 > BNN-α 0.801 > VBLL-β 0.792 >
NN-α(dnn) 0.789 > VBLL-α 0.781 > NN-β(mlp) 0.756`. Bold NGBoost **and RF** (not SVM). NGBoost mean auc_norm 0.824.

**Wilcoxon table (tab:wilcoxon_bnn) — exact ΔNDS → Δauc_norm:**

| Comparison | paper ΔNDS (p) | v2 Δauc_norm (p) | note |
|---|---|---|---|
| NN-α → BNN-α | +0.056 (2.9e-10) * | **+0.031 (2.9e-11) *** | holds |
| NN-α → VBLL-α | +0.061 (1.2e-6) * | **+0.011 (0.252) NS** | **significance FLIPS** |
| NN-β → BNN-β | +0.096 (2.9e-11) * | **+0.053 (2.9e-11) *** | holds |
| NN-β → VBLL-β | +0.124 (1.2e-7) * | **+0.062 (1.2e-7) *** | still largest |
| RF → QRF | −0.022 (1.6e-10) * | **−0.012 (2.9e-11) *** | QRF still worse |

Other fixes: line 443 "NDS clusters near −0.38 regardless of baseline" → "auc_norm clusters near ~0.83; baseline
auc_norm baseline coupling weak/n.s. (config-level ρ≈+0.06; NDS was −0.36) — but see the fig3 caveat: at model-mean level ρ≈+0.43". Per-strategy spread (469): outlier ~0.023, threshold ~0.129 (threshold
widest — direction preserved). **Delete the VBLL-α "systematic-noise benefit" narrative** (line ~471). Line 502
"PDV model-choice 72%" is badly stale (28.1% under Gaussian; range 13–55%).

### Results: Variance decomposition / ANOVA (389–439) [MAJOR]
Rebuild the η² table (`table1_anova_summary.csv`). Robustness rows paper → v2 (Model/Rep/Inter/Resid):

| Strategy | paper (NDS) | v2 (auc_norm) | shift |
|---|---|---|---|
| Gaussian | 48.7/8.2/11.9/31.2 | 43.8/5.2/16.9/34.2 | model still dominant |
| Quantile | 47.9/6.8/14.9/30.4 | 36.8/4.4/15.1/**43.7** | residual now leads |
| Threshold | 48.0/10.1/28.0/13.9 | **54.7**/7.9/22.6/14.8 | model dominant |
| Heteroscedastic | 37.0/4.5/17.5/41.0 | 14.0/0.7/8.0/**77.4** | **bold moves Model→Resid** |
| Value-Prop. | 52.8/6.9/22.2/18.2 | 52.5/6.0/19.9/21.6 | model dominant |
| Outlier | 12.7/3.2/4.7/79.3 | 10.3/0.2/5.9/**83.6** | residual dominant |

Line 391 "interaction dominates NDS variance" → model is the largest *explained* factor (Model>Interaction>Rep),
but for diffuse noise (outlier/hetero, +now quantile-ish) most variance is residual. Cross-rep Spearman (423,
"ρ=0.73, 12 models") must be recomputed on auc_norm with the corrected model count (9 ANOVA / 11 ranking, not 12).

### Abstract + Scientific Contribution (164–169) [MAJOR]
- 164 metric sentence → retention-AUC definition with the nonlinear+decoupled motivation (see big-picture edits).
- "NGBoost and SVMs strongest" → "NGBoost and the tree ensembles (RF/LightGBM/XGBoost)".
- "Bayesian transformations improve robustness" → "full-BNN improves both families; VBLL improves one (NN-β)".
- Add a **third contribution**: the pair of curve-descriptive, baseline-decoupled robustness metrics.
- The "NGBoost & GP strongest per-sample" clause is already dead from the uncertainty correction — remove.
- Fix the mid-sentence break at 167–169; drop the "aleatoric" organizing frame.
- Numbers: W=0.9121; interaction explains most on clean data (46–49%); rep <8% under noise.

### Introduction (191–203) [MAJOR]
No literal "NDS" in the body, but the `% TODO` preview (202) must inherit the new metric. Add a one-line
motivation (robustness is a property of the whole *curve*, not a slope) and a "what we find" preview citing the
v2 splits + the decoupling. Do **not** re-import "SVM among most robust" or the headroom logic. If a concordance
number is quoted, use W=0.9121 (not 0.887/13 or 0.92).

### Methods: Noise Strategies + NoiseInject (324–386) [MAJOR]
Line 380 "reports … degradation slope and retention percentage" → auc_norm + per-level retention. Introduce the
gating methodology (0.3 gate + ≥5-iters + 7-vs-11 caveat) where robustness metrics are described.

### Results: Validation on experimental datasets (560–572) [MAJOR]
Direction HOLDS everywhere; only the framing inverts. "pooled mean NDS −0.16 (SVM) vs −0.19 (NGBoost)" →
"pooled mean auc_norm **0.877 (SVM) vs 0.857 (NGBoost)**"; SVM leads on hERG/Caco-2, NGBoost on LogD (marginal).
External ANOVA Model η² = **91.8/92.4/95.2%** (LogD/Caco-2/hERG), Rep 4.2/0.9/0.2%, Residual 0.0 (Type-I sequential SS), n_models=7, n_reps=3.
RF→QRF Δauc_norm: hERG −0.132, Caco-2 −0.124, LogD −0.042 (QRF worse, holds). XGBoost worst (holds). Invert both
figure captions (556, 567) to "higher auc_norm = more robust"; drop the "|NDS|>2 filter" language.

### Results: Uncertainty (506–559) [MAJOR — but NOT a metric-swap edit]
**Untouched by the robustness metric.** Table 4 values match v2 to displayed precision (tiny rounding only: GP/SNS
ρ 0.56→0.54, ECE 0.14→0.13, coverage 71.2→71.4%; BNN-α/SNS 0.21→0.20). The only real work here is the *separate*
within-σ pooling correction already covered in MAJOR UPDATE 1. The `fig:validation_overview` caption that currently
floats here (556) is NDS-worded — fix it in the Validation-section pass, not here. Flag so this section is NOT
accidentally rewritten during the metric migration.

### Conclusion (573–685) [MAJOR]
Same set of fixes as the abstract, in prose: W 0.92→0.9121; qualify "model dominant" (4/6 strategies); NGBoost-#1
with tree cluster behind (not NGBoost+SVM); strengthen the decoupling to a design property; qualify the Bayesian
claim (VBLL-α NS); "threshold/valprop widest, outlier barely separates" → phrase in auc_norm η² terms. Update the
**Abbreviations** list (drop NDS, add auc_norm) and **Additional file 5** caption
(baseline R² < 0.3, 48 configs, +≥5-iters rule).

### Methods: Dataset/Reps/Models (206–233) [MINOR]
Dataset sizes unchanged. Add the 7-vs-11 roster caveat and the balance-gate inclusion criterion. Clarify GP/QRF
are analysed pairwise/rep-specifically, not in the primary cross-rep ANOVA. Note VBLL×embedding instability is
now expressed via the iteration-count gate, not the baseline filter.

## Figure audit — fixes applied, decisions remaining

All 11 figures render with correct direction and labels. **Applied to `generate_paper_figures_v2.py`** (commit
373ca34): fig3 now titled for the decoupling + ρ annotation + un-squeezed y-axis; fig2 shows a Residual bar and
discloses "11 models" / "7 models" per panel (real data drops to 7 — see model-count correction); fig_validation_overview discloses the PDV representation, widens the
colour scale to [−0.5, 1.1] so negatives aren't collapsed, and greys N/A; fig1/fig1_supp share a [0.4, 1.0] anchor.

**FIXED (applied):**
- **fig_validation_anova negative residuals** — the unbalanced design (7 models × 3 reps, missing cells) previously
  made Model+Rep+Interaction sum >100%, giving a **negative Residual η²** (−1.9/−11.7/−9.3). Now computed via a
  dependency-free **Type-I sequential SS** helper (`_two_way_eta2_unbalanced`, nested OLS via numpy) that guarantees
  residual ≥ 0 and shares summing to 100%. Method-consistent with the paper's stated Type-I SS. **NB:** the validation
  ANOVA uses one auc_norm value per model×rep cell (no replication), so residual is ~0 and the three effects partition
  100% — legitimate for a saturated design, but state it as such (it is not the same 10-replicate error term as QM9).
- **`nds` → `auc_norm` cleanup** — all metric-carrying variables, stale comments, and the mislabeled output CSVs were
  renamed. Output filenames changed: `table2_nds_*` → `table2_auc_*`, `table_validation_nds*` → `table_validation_auc*`,
  `table_nn_family_nds` → `table_nn_family_auc`. **Paper Additional-file references and captions must use the new names.**
  (Legacy input-column handling `NDS_r2`/`NSI` and the intentional "old NDS slope" explanations were deliberately kept.)

## ⚠ REAL-DATA FIGURE AUDIT — three corrections (6-agent pixel-level check of the regenerated figures)

Every figure was verified against its regenerated CSV. Heatmap cells, the Wilcoxon table, and the validation
figures all match the data **exactly**. The figures are faithful; three claims in *this guide* were wrong and are
now corrected. **In every case the figure is right and the guide's number was the error.**

1. **🔴 fig3 / the decoupling claim — FRAMING DECISION NEEDED (do not paste the old claim).** fig3 plots **11
   PDV/Gaussian model *means***, and prints **ρ = +0.49** — a *visible positive tilt*, NOT a flat cloud. So the
   figure's own "decoupled" title contradicts its data. The real, defensible decoupling is a **metric/config-level**
   statement (NDS vs auc_norm coupling to baseline):
   - Within Gaussian, all configs: **NDS ρ = −0.36 (p = 5×10⁻⁴, coupled)** vs **auc_norm ρ = +0.13 (n.s.)**.
   - All configs: NDS −0.08 vs auc_norm +0.06.
   - fig3's n=11 model means: **+0.43** (better models are both more accurate *and* more robust — a genuine
     empirical pattern, not a metric artifact).
   **RESOLVED (no new figures):** fig3's title was reverted from the (wrong) "Robustness is decoupled from baseline
   accuracy" back to the neutral **"Baseline accuracy vs. robustness (PDV, Gaussian)"**; the ρ annotation stays and
   honestly shows the model-level tilt. The figure now simply shows what it shows. **The metric-decoupling argument
   moves to TEXT** (Methods, where auc_norm is defined): state in one sentence that, unlike NDS, auc_norm does not
   mechanically track baseline accuracy — NDS ρ(baseline) = −0.36 (p=5×10⁻⁴) under Gaussian vs auc_norm ρ = +0.13
   (n.s.); no figure needed. **Abstract/intro/conclusion:** do NOT claim "robustness is decoupled from clean accuracy"
   as an empirical finding — at the model level they are moderately *related* (more robust models tend to be more
   accurate). Keep only the narrow, correct statement that the *metric* isn't a proxy for baseline the way NDS was.

2. **fig2 robustness ANOVA = 7 models, not 9.** Correct number: **7**. The 4 NN Bayesian/VBLL variants
   (BNN-α, BNN-β, VBLL-α, VBLL-β) are **intentionally excluded from the cross-rep ANOVA** — they don't produce
   usable results on the embedding fingerprints (mol2vec, MHG-GNN), so they can't enter a design that requires all
   five reps. This is expected and correct; no action. (My guide's earlier "9" was a wrong synthetic estimate — the
   only fix was updating the number to 7.) Their robustness is reported, as intended, in the ranking + Wilcoxon +
   nn-family analyses. Just make sure the paper text/caption for fig2 states the 7-model roster.

3. **Validation ANOVA Rep η² = 4.2/0.9/0.2, not 4.8/7.9/5.5** (Type-I sequential SS changed the split; Residual 0,
   Model η² 91.8/92.4/95.2 unchanged). Fixed inline above. Representation is now *even more* negligible on the
   external sets — the "model dominates" claim is strengthened.

**Still open (your call):**
- **fig_interaction NN-β×mol2vec = 0.846** — CONFIRMED real data (not a thin-cell artifact); mlp is genuinely ~+0.06
  more robust on mol2vec than its other reps. Faithful render; keep, but worth a sentence if it draws a reviewer's eye.
- **fig_interaction Panel B** now prints Spearman ρ = 0.86 (was NDS ρ=0.73) — update any text citing 0.73.
- **fig_uncertainty_combined QRF σ=0.3 spike** — near-zero-mean artifact, unrelated to the metric.
- **QRF/GP rows** appear in fig1/fig_interaction heatmaps but not in the `table2_auc*` ranking CSVs (they're computed
  but excluded from the 11-model ranking set) — confirm their heatmap source is intended.
- **Broader degradation curves** — auc_norm's motivation is "curves are nonlinear," but curves are only shown for
  key models, PDV/ECFP4, Gaussian. A reviewer can call the motivation unsubstantiated. Either cite the `r2_fit`
  linearity diagnostic or add broader curve panels (more reps/strategies + validation).

## Literature backing (verified — 18/28 candidates survived adversarial checking)

The metrics are established practice. **Cite:**

**AUC_norm (normalised area / aggregate over a degradation-severity curve, decoupled from clean accuracy):**
- **Hendrycks & Dietterich, ICLR 2019** (arXiv:1903.12261) — mCE / Relative mCE: error aggregated over severity
  levels, normalised to a reference; Relative mCE explicitly removes the clean-accuracy confound. *The canonical
  citation.* (Cite 1903.12261, not the earlier 1807.01697.)
- **Wang et al., 2025** (arXiv:2503.16183) — defines **relative AUC (rAUC) = AUC(method)/AUC(upper-bound)** over
  the noise range. Closest exact match to auc_norm.
- **Rajhans & Khawarey, ICAART 2026** (arXiv:2602.06395) — "Robustness Index = area under the accuracy-perturbation curve."
- **Kolmar & Grulke, *J. Cheminform.* 13:92, 2021** — **same domain, same journal**: normalised RMSE/RMSE₀ vs
  σ/RMSE₀, ratio m_noise/m_true as a single scalar decoupled from absolute error. *Frame auc_norm as the natural,
  shape-agnostic evolution of Kolmar's normalised-degradation approach* (this is also the Kolmar work the uncertainty
  section already builds on). NB: it uses linear slopes, not an AUC — cite as precedent for *normalised, decoupled*
  robustness scalars in QSAR, not for the area itself.
- **Ben Braiek & Khomh, 2024** (arXiv:2404.00897, ML Robustness primer) — surveys mCE/rCE as standard practice.
- Supporting the *decoupling principle*: **Taori et al., NeurIPS 2020** ("effective/relative robustness");
  **Göpfert et al., ECML-PKDD 2019** ("adversarial robustness curves"); **Wang et al., ACL 2023** ReCode ("Robust Drop").

**weibull_beta references — NO LONGER NEEDED (metric dropped 2026-07).** Kept for reference only in case the
shape descriptor is ever revisited: Klakattawi *PLoS ONE* 17(2):e0264229 2022; Safari/Masseran/Majid et al.
*Sci. Rep.* 15:11516 2025; Huang & Ferrell *PNAS* 93:10078 1996 (Hill-coefficient steepness analogy).

**Do NOT cite as metric support** (real papers, but the match was overstated): Rolnick 2017, Jiang ICML 2020,
Song survey 2022, Northcutt 2021, NoiseMol 2023, Fooladi 2025 — these only sweep a noise axis / report point-wise
metrics. And **ContextShift (arXiv:2606.09495) fabricated its rAUC claim** — exclude entirely.

## ⚠ Warning: the guide's own NDS-era drafts (§1, §5, §6) are now stale

The full-draft rewrites above (abstract §1, two-mechanism keystone §5, conclusion §6, preview line 149) predate
this metric change. They still say "rate of performance loss / slope," cite "Kendall's W = 0.92," lean on the PDV
**headroom argument**, and assert blanket "BNN+VBLL improve robustness." **Pasting them verbatim re-imports exactly
the NDS framing and the self-contradiction this change removes.** Re-derive them against auc_norm before use — the
per-section fixes above are the authority where they conflict.
