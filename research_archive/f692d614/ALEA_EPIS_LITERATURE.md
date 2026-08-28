# Aleatoric vs epistemic uncertainty decomposition in molecular ML — evidence review

Compiled 2026-08-21. Every claim below is tied to a document I actually retrieved and read.
Anything I could not retrieve is listed at the bottom and is **not** described.

---

## A. The formal recipe, and what a model must output

### A.1 The variance decomposition (law of total variance)

The canonical statement is Depeweg, Hernández-Lobato, Doshi-Velez & Udluft, *Decomposition of
Uncertainty in Bayesian Deep Learning for Efficient and Risk-sensitive Learning*, ICML 2018
(arXiv:1710.07283). Read at https://arxiv.org/html/1710.07283.

Total predictive standard deviation:

```
σ(y*|x*) = { σ²_q(W)( E[y*|W,x*] )  +  E_q(W)[ σ²(y*|W,x*) ] }^(1/2)      (their Eq. 4)
```

with their own labelling: "σ²_q(W)(E[y*|W,x*]) corresponds to the epistemic uncertainty …
By contrast, the term E_q(W)[σ²(y*|W,x*)] represents the aleatoric uncertainty."

They also give the information-theoretic version, Eq. (3):
`H[y*|x*] − E_q(W)[H(y*|W,x*)] = I(y*,W)` — total entropy minus expected conditional entropy
equals the mutual information between prediction and weights (the epistemic part).

**What the model must produce for this to be computable (their own list):**
1. samples from a posterior over parameters, `W ~ q(W)`;
2. a *per-sample predictive distribution* `p(y*|W,x*)` whose mean **and variance** are computable.

Both are required. One of them alone gives you one term, never two.

### A.2 The deep-learning instantiation

Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*,
NeurIPS 2017 (arXiv:1703.04977). Read the full PDF.

Their Eq. (9), the MC-sampling estimator:

```
Var(y) ≈ (1/T) Σ_t ŷ_t²  −  ( (1/T) Σ_t ŷ_t )²   +   (1/T) Σ_t σ̂_t²
         └──────────── epistemic ─────────────┘       └── aleatoric ──┘
```
"with {ŷ_t, σ̂_t²}_{t=1..T} a set of T sampled outputs: ŷ_t, σ̂_t² = f^Ŵt(x) for randomly masked
weights Ŵ_t ~ q(W)."

Crucially, §2.2 explains what happens without a variance head: "When not explicitly modeled …
this observation noise parameter is often fixed as part of the model's weight decay, and ignored."
And §2.2 closes: a MAP-trained heteroscedastic net "does not capture epistemic model uncertainty,
as epistemic uncertainty is a property of the model and not of the data."

Their §5.2 results are the reference behaviour:
- "Aleatoric uncertainty cannot be explained away with more data"
- "Aleatoric uncertainty does not increase for out-of-data examples … whereas epistemic uncertainty does"
- Table 3 layout: rows = training-set fraction / test set, columns = aleatoric and epistemic. This is
  exactly the table shape the user should copy (rows = σ level, columns = the two components).

### A.3 The conceptual caveats

Hüllermeier & Waegeman, *Aleatoric and Epistemic Uncertainty in Machine Learning: An Introduction to
Concepts and Methods*, Machine Learning 110:457–506 (2021), arXiv:1910.09457. Read via ar5iv.

- "aleatoric (aka statistical) uncertainty refers to the notion of randomness, that is, the variability
  in the outcome of an experiment which is due to inherently random effects."
- "epistemic (aka systematic) uncertainty refers to uncertainty caused by a lack of knowledge (about the
  best model)."
- "epistemic uncertainty refers to the reducible part of the (total) uncertainty, whereas aleatoric
  uncertainty refers to the irreducible part."
- **Important hedge**: "aleatoric and epistemic uncertainty should not be seen as absolute notions.
  Instead, they are context-dependent in the sense of depending on the setting (X,Y,H,P)."

### A.4 Answers to the two pointed sub-questions

**"Can you get aleatoric from a point prediction plus ensemble spread?"** No. Ensemble spread is the
first term only. Heid et al. (JCIM 2023, below) put it flatly: "ensembling is a measurement of variance
error and does not directly incorporate noise error." Scalia et al. (JCIM 2020) say the same of
MC-dropout/ensembles/bootstrap: "these approaches have been designed to model NN-weight uncertainties,
therefore they are directly related to epistemic uncertainty estimation."

**"Can you get epistemic from NGBoost or a QRF?"** Not from a single fitted model. NGBoost outputs one
conditional distribution per input (a fitted mean and variance), i.e. the aleatoric term only; there is
no posterior over the boosted model, so the first term of the decomposition is identically zero. I did
not find a molecular-ML paper that extracts epistemic uncertainty from NGBoost. To get it you must
create model-to-model variation yourself (bagging / bootstrap / multi-seed NGBoost) and take the
variance of the *means* across those fits — the outer term of the same equation.

A QRF is different: it is *already* an ensemble, so both terms exist. A published recipe:
Al-Aghbary, Sobh, Stål, Awaleh, Jalludin & Gerhards, *Uncertainty quantification using clustering-based
quantile regression forests*, Geophysical Journal International 245(2), 2026,
doi:10.1093/gji/ggag113. They write `σ²_t(x) = σ²_a(x) + σ²_e(x)` with
`σ²_a(x) = (1/M) Σ σ²_i(x)` (mean of per-tree variances) and
`σ²_e(x) = (1/M) Σ μ²_i(x) − μ̄²(x)` (variance of per-tree means), stating that for tree models
epistemic uncertainty "stems from variability in model structure introduced by bootstrap resampling,"
while aleatoric "captures the irreducible noise in the response, as reflected by variability of
observed outcomes within each terminal node." This is directly transplantable to a QSAR QRF.

For random forests in *classification*, the equivalent construction is Shaker & Hüllermeier,
*Aleatoric and Epistemic Uncertainty with Random Forests*, IDA 2020 / Springer LNCS
(arXiv:2001.00893) — found by search, abstract read only, and it is classification-only, so it is
supporting rather than load-bearing here.

---

## B. Model-by-model verdict for the user's roster

| Model | Aleatoric available? | Epistemic available? | How the literature does it |
|---|---|---|---|
| **GP** | Yes — the learned noise variance σ_n² | Yes — the posterior variance of the latent function | Rasmussen & Williams, *GPML* ch. 2 (read the PDF): the GP algorithm "returns the predictive mean and variance for noise free test data—to compute the predictive distribution for noisy test data y*, simply add the noise variance σ_n² to the predictive variance of f*." So epistemic = V(f*) (their Eq. 2.26), aleatoric = σ_n². **Caveat: a standard GP has one global σ_n², i.e. homoscedastic aleatoric — it is a single number per model, not a per-molecule value.** |
| **NGBoost** | Yes — the fitted per-input variance | **No, not as run.** A single NGBoost fit has no parameter posterior | NGBoost (Duan et al., ICML 2020, arXiv:1910.03225 — abstract/search only) outputs a full conditional distribution per input; that is the aleatoric term. Epistemic requires an outer ensemble of NGBoost fits, computed as the variance of their means. |
| **QRF** | Yes — spread of the conditional distribution at a leaf | **Yes, if you compute it per tree** | Al-Aghbary et al. (GJI 2026) recipe above. Requires access to per-tree leaf values, which `quantile-forest`/sklearn expose. |
| **MC-sampled BNN** | **Only if the network has a variance output head.** With a fixed homoscedastic noise term it reports a constant, not a usable aleatoric column | Yes — variance of the sampled means | Kendall & Gal Eq. (6): "[ŷ, σ̂²] = f^Ŵ(x) … a single network to transform the input x, with its head split to predict both ŷ as well as σ̂²", trained with the heteroscedastic NLL, Eq. (8): `L = (1/D) Σ_i ½ exp(−s_i)‖y_i−ŷ_i‖² + ½ s_i` where `s_i := log σ̂_i²`. Scalia et al. do exactly this for molecules: same MVE head bolted onto MC-dropout, ensembles and bootstrap alike. **This is the fix for the user's BNN row.** |
| **VBLL** | Yes — an explicit noise covariance Σ | Yes — the last-layer weight posterior term | Harrison, Willes & Snoek, *Variational Bayesian Last Layers*, ICLR 2024 (arXiv:2404.11599), read the PDF. Eq. (2): `p(y|x,η,θ) = N(w̄ᵀφ, φᵀSφ + Σ)`. The `φᵀSφ` term is epistemic (last-layer weight covariance S); `Σ` is the aleatoric noise covariance, which they estimate by MAP: "While variational inference for the noise covariance is possible, we choose (MAP) point estimation." **Note Σ is input-independent unless you make it so** — same homoscedastic caveat as the GP. |

Chemprop, the most-used package in this space, encodes the same rules. Its docs
(https://chemprop.readthedocs.io/en/latest/uncertainty.html, and the DeepWiki mirror) list the
regression estimators as `dropout`, `ensemble`, `quantile-regression`, `mve`, and
`evidential-total` / `evidential-epistemic` / `evidential-aleatoric`. Ensemble and dropout are
described as estimating **epistemic** uncertainty; MVE "predicts both the mean and variance for each
target"; evidential regression "provides a principled way to capture both aleatoric and epistemic
uncertainty." The naming convention (`-total`, `-epistemic`, `-aleatoric` as separate selectable
outputs) is itself a useful precedent for how to label table columns.

---

## C. What real papers do when only some models support the split

Four distinct, all legitimate, strategies observed:

**C.1 Standardise the aleatoric estimator across every method, so the table is full.**
Scalia, Grambow, Pernici, Li & Green, *Evaluating Scalable Uncertainty Estimation Methods for
Deep Learning-Based Molecular Property Prediction*, JCIM 60:2697–2717 (2020), arXiv:1910.03127
(full PDF read). They compare MC-dropout, ensembling and bootstrapping — all three are epistemic-only
mechanisms — and bolt the **same** heteroscedastic variance head onto all of them:

> "All the different methods use the same aleatoric approximation scheme but the way epistemic
> uncertainty is modeled affects also aleatoric uncertainty results, thus resulting in different
> outputs (ref. Eq. (4)). This also allows drawing conclusions about aleatoric uncertainty which do
> not depend on the uncertainty model used for the NN-weights."

They then report **three parallel evaluations — aleatoric alone, epistemic alone, and total** — under
the same metric set. This is the closest published analogue to what the user wants and is the model
to copy.

**C.2 Report a single "total uncertainty" column for everything, and mark the cells where a metric
is undefined as excluded.** Hirschfeld, Swanson, Yang, Barzilay & Coley, *Uncertainty Quantification
Using Neural Networks for Molecular Property Prediction*, JCIM 60:3770–3780 (2020),
arXiv:2005.10036 (full PDF read). They evaluate ~a dozen heterogeneous UQ methods (ensembles,
bootstrap, snapshot, MC-dropout, MVE, Tanimoto/latent distances, DNN-GP, DNN-RF, FP-GP, FP-RF) as one
scalar `U(x)` each, and explicitly refuse to compute variance-based metrics for methods that do not
produce variances: "it is inappropriate to calculate NLL for relative UQ metrics, so values for
Tanimoto and latent space distances are excluded." Excluding a cell with a stated reason is the
accepted move; inventing a surrogate is not.

**C.3 Force every backbone to emit a variance so the grid is complete.** MUBen
(arXiv:2306.10060, HTML read) applies each UQ method to each backbone and reports Gaussian NLL and
calibration error for regression across the full grid. The tables are complete, not ragged.

**C.4 Name the component in the method label.** Chemprop's `evidential-total` /
`evidential-epistemic` / `evidential-aleatoric` naming.

---

## D. Is "aleatoric rises with injected label noise" an anomaly? **No. It is the confirmatory result.**

This is the single most important finding for the user's narrative, and it is directly on point.

### D.1 The exact experiment already exists, in a target-adjacent journal

Ryu, Kwon & Kim, *Uncertainty quantification of molecular property prediction with Bayesian neural
networks*, arXiv:1903.08375 (full PDF read; published in revised form as *A Bayesian graph
convolutional network for reliable prediction of molecular properties with uncertainty
quantification*, Chemical Science 10:8438–8446, 2019 — I read the preprint, not the journal version).

They use their Eq. (9), which is Kendall & Gal's decomposition:

```
V̂ar[y*|x*] = (1/T)Σ(ŷ*_t)² − ((1/T)Σŷ*_t)²  +  (1/T)Σσ̂²_t
             └──────── epistemic ─────────┘     └ aleatoric ┘
```

They take a **noise-free synthetic label** (RDKit logP on ZINC) precisely so there is no baseline
aleatoric term, then inject Gaussian noise `ε ~ N(0, σ²)` at increasing σ² and plot all three
uncertainties:

> "As the noise level increases, the aleatoric and total uncertainties increase, but the epistemic
> uncertainty is slightly changed. This result verifies that the aleatoric uncertainty arises from
> data inherent noises, while the epistemic uncertainty does not depend on data quality.
> **Theoretically, the epistemic uncertainty should not increase by the changes in the amount of data
> noise.** We guess that the slight change of the epistemic uncertainty arises from the stochastic
> numerical optimization of model parameters."

So: aleatoric ↑ with σ is the **expected, confirmatory** result. Epistemic staying roughly flat is
also expected. A small epistemic drift is explicitly attributed by the authors to optimisation
stochasticity, not treated as a finding.

### D.2 The same result in J Cheminform, on QM9

Yang & Li, *Explainable uncertainty quantifications for deep learning-based molecular property
prediction*, J Cheminform 15 (2023), doi:10.1186/s13321-023-00682-3 (PMC full text read). Deep
ensemble of D-MPNNs with heteroscedastic loss; decomposition:
`σ²_ale = (1/M)Σσ²_m`, `σ²_epi = (1/M)Σ(μ_m − μ_ens)²`.

They add artificial Gaussian noise scaled to the number of nitrogen atoms:
> "The distribution of aleatoric uncertainty shifts right (increases) as the number of nitrogen atoms
> in the molecules increases in the nitrogen-noisy model, which suggests that the model can
> successfully learn the artificial noise introduced."

And separately, training with nitrogen compounds held out:
> "the epistemic uncertainties greatly increase for the nitrogen-containing molecules, which indicates
> the self-awareness of ignorance."

Their own caution, worth quoting in the user's paper: "a high estimate of aleatoric uncertainty is not
always caused by data noise."

### D.3 The aleatoric limit — the deeper JCIM reference for a label-noise study

Heid, McGill, Vermeire & Green, *Characterizing Uncertainty in Machine Learning for Chemistry*,
JCIM 63:4012–4029 (2023), doi:10.1021/acs.jcim.3c00373 (full PDF read via MIT DSpace).

They construct a synthetic **noise-free** group-additivity dataset (from QM9 groups applied to GDB11,
7.9M molecules) specifically because no noise-free chemical dataset exists, then inject controlled
noise. Findings the user should cite:

- Test-set noise creates an **aleatoric limit**: "The learning curve of the noisy test and training
  sets approaches an asymptote at 1 kcal/mol, which is the standard deviation of the employed noise
  distribution … The effect of noise in the test set on the perceived model error is irreducible."
- Noise in the *training* set is **not** purely irreducible: "Though noise-based, the error from noise
  introduced while training is not irreducible."
- Ensembles cannot see noise: "ensembling is a measurement of variance error and does not directly
  incorporate noise error." In their systematic-noise experiment, MVE recovered the two noise regimes
  (mean predicted SD 2.35 vs 20.0 kcal/mol against true 2 vs 20), while a scaled ensemble did not
  (10.6 vs 11.5 — essentially no discrimination). They call this "an example of poor dispersion, even
  when scaled to a calibrated level."
- Noise distribution shape barely matters: Gaussian, uniform, hyperbolic and bimodal noise at the same
  SD gave "very similar performance (points overlap in the figure)". **This is a direct challenge to
  the user's six-strategy design and should be engaged with, not ignored** — it is also an opportunity,
  because their strategies include heteroscedastic/outlier variants that Heid et al.'s four do not.
- Their guidance: "When there is reason to believe that a data set is affected by systematic noise, we
  recommend testing a model trained using mean-variance estimation or similar and comparing it against
  a simple model architecture."

### D.4 The counterweight: the two components are not cleanly separable in practice

Mucsányi, Kirchhof & Oh, *Benchmarking Uncertainty Disentanglement: Specialized Uncertainties for
Specialized Tasks*, NeurIPS 2024 D&B (arXiv:2402.19460, abstract page read). Headline: "No existing
approach provides pairs of disentangled uncertainty estimators in practice."

The ICLR 2025 Blogposts entry *Reexamining the Aleatoric and Epistemic Uncertainty Dichotomy*
(https://iclr-blogposts.github.io/2025/blog/reexamining-the-aleatoric-and-epistemic-uncertainty-dichotomy/)
— a peer-reviewed blog track entry, not a journal article — summarises the evidence I could verify by
reading it: Mucsányi et al. report aleatoric/epistemic rank correlations "between 0.8 and 0.999 on all
twelve methods they test, from deep ensembles over Gaussian processes to evidential deep learning",
and find "the epistemic uncertainty estimators are as predictive of human annotator noise (an aleatoric
task) as aleatoric estimators." It further cites Gruber et al. that "even in this very simple [linear]
model one cannot additively decompose the total predictive uncertainty into aleatoric and estimation
uncertainty" because the terms "interact non-linearly."

Valdenegro-Toro & Saromo, *A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement*
(arXiv:2204.09308, abstract read): "there is an interaction between learning aleatoric and epistemic
uncertainty, which is unexpected and violates assumptions on aleatoric uncertainty"; "aleatoric
uncertainty is unreliable in the out-of-distribution setting"; "Ensembles provide overall the best
disentangling quality."

**Net:** if the user observes epistemic rising with σ, that is a real, publishable, *literature-anticipated*
observation of entanglement — not an anomaly and not a failure. It has two candidate mechanisms, both
citable: (i) noisier labels genuinely make the fitted model less determined, so ensemble/posterior
spread grows (a real increase in epistemic uncertainty in the Hüllermeier sense — the "best model"
becomes less identifiable); and (ii) estimator entanglement (Mucsányi et al.). The user should say
which they can distinguish and which they cannot.

---

## E. What is reported alongside the split — the community-standard metric set

Assembled from the four benchmark papers I read in full:

| Metric | Scalia 2020 | Hirschfeld 2020 | Busk 2022 | UNIQUE 2024 |
|---|---|---|---|---|
| Spearman rank corr. (uncertainty vs \|error\|) | via confidence curves | **yes, primary** | ranking plot | **yes** |
| Confidence / sparsification curve + AUCO | **yes** | — | yes | — |
| Error drop (ratio first vs last quantile) | **yes** | — | — | — |
| Calibration curve (observed vs expected coverage) | yes | yes | yes | yes |
| Miscalibration area | — | **yes** | SSE of quantile calibration | MACE |
| ENCE (RMV vs RMSE in bins) | **yes** | — | **yes** | — |
| NLL (and calibrated NLL) | — | **yes, both** | **yes** | **yes** |
| Sharpness (RMV) + dispersion (coefficient of variation) | **yes (c_v)** | — | **yes (CV)** | — |

Sources:
- Scalia et al.: confidence curve, AUCO (area under confidence–oracle error), error drop, decrease
  ratio, confidence-based and error-based (ENCE) calibration, AUCE, MCE, coefficient of variation, all
  re-evaluated in-domain and out-of-domain by scaffold split.
- Hirschfeld et al.: Spearman ρ (their Eq. 8), miscalibration area, NLL (their Eq. 9:
  `½|D|⁻¹ Σ ln(2π) + ln(U(x)) + (M(x)−y)²/U(x)`), calibrated NLL with `σ̂²(x) := aU(x)+b`.
- Busk, Jørgensen, Bhowmik, Schmidt, Winther & Vegge, *Calibrated uncertainty for molecular property
  prediction using ensembles of message passing neural networks*, Mach. Learn.: Sci. Technol. 3:015012
  (2022), doi:10.1088/2632-2153/ac3eb3 (full PDF read). Their Eq. (8) is the decomposition:
  `σ²_*(g) = (1/M)Σσ²_θm(g) [aleatoric] + (1/M)Σμ²_θm(g) − μ²_*(g) [epistemic]`.
  Their Eq. (9) is ENCE = `(1/K) Σ_k |RMV_k − RMSE_k| / RMV_k`. Their Eq. (10) is the coefficient of
  variation. They also do post-hoc recalibration on a held-out set.
  Their key warning: "Calibration alone is not sufficient to ensure that individual uncertainty
  estimates are informative … a regression model that predicts constant uncertainty corresponding to
  its average empirical error is well calibrated in terms of ENCE and SSE but the uncertainty estimates
  are clearly not very useful."
- UNIQUE (Novartis), *UNIQUE: A Framework for Uncertainty Quantification Benchmarking*, JCIM
  64:8379–8386 (2024), doi:10.1021/acs.jcim.4c01578 (PMC full text read). Its three evaluation types
  are ranking-based (Spearman), calibration-based (MACE), and proper scoring rules (NLL); it selects a
  recommended metric by counting Wilcoxon rank-sum wins with Bonferroni correction. Note it does **not**
  use aleatoric/epistemic language — it splits UQ metrics into "data-based" and "model-based" instead.

**Minimal defensible set for a J Cheminform submission:** (1) Spearman ρ of uncertainty vs absolute
error, (2) a calibration curve summarised by miscalibration area *or* ENCE, (3) NLL, and (4) a sharpness
/ dispersion number (RMV and coefficient of variation) so a flat-uncertainty model cannot look good.
Items (1)+(2)+(3) are shared by three of the four benchmarks; (4) is Busk's explicit warning and Heid
et al.'s "sharpness and dispersion" recommendation.

Heid et al. add: "Application of calibration methods may serve to improve some uncertainty evaluation
metrics, such as miscalibration area, while still providing uncertainty quantifications with functional
shortcomings … We caution the reader to apply calibration methods carefully and check their validity
using multiple evaluation metrics."

---

## F. Concrete, prioritised recommendations for this benchmark

### Priority 1 — make the table square by fixing the estimator, not the table

The ragged table is a *modelling* gap, not a reporting gap, and every fix is small.

1. **BNN (both backbones): add a two-output head (mean, log-variance) and train with the
   heteroscedastic NLL.** This is Kendall & Gal Eq. (6)+(8), and it is exactly what Scalia et al. did
   to make MC-dropout/ensembles/bootstrap comparable on aleatoric. Then aleatoric = mean of sampled
   σ̂_t², epistemic = variance of sampled means. Cite Kendall & Gal 2017 + Scalia et al. 2020.
   Without this the BNN aleatoric column *cannot* exist and saying so is the honest alternative.
2. **QRF: compute per-tree leaf mean and leaf variance.** epistemic = variance across trees of the
   leaf means; aleatoric = mean across trees of the within-leaf variances. Cite Al-Aghbary et al.
   (GJI 2026) for the recipe and Busk et al. Eq. (8) for the general form. This is a post-hoc
   computation on an already-trained forest — no retraining.
3. **NGBoost: either (a) bag it** — fit K NGBoost models on bootstrap resamples or different seeds and
   take variance-of-means as epistemic — **or (b) declare epistemic not available and leave the cell
   blank with a footnote.** Option (a) costs K× training; option (b) costs nothing and is defensible
   (Hirschfeld et al.'s precedent of excluding undefined cells with a stated reason).
4. **GP and VBLL: keep both columns but flag the homoscedastic caveat.** Both have a *single global*
   noise term (GP σ_n², VBLL Σ), so their aleatoric column is one number per model per σ, not a
   per-molecule quantity — it cannot be rank-correlated against per-molecule error the way an MVE
   aleatoric can. State this explicitly; it will otherwise be a reviewer's first question.

### Priority 2 — flip the narrative from "anomaly" to "confirmation plus one genuine finding"

- **Aleatoric rising with σ is the sanity check that the analysis works.** Ryu, Kwon & Kim (2019) and
  Yang & Li (2023) both report exactly this. Frame it as: "our injected-noise design reproduces the
  expected aleatoric response reported by [Ryu 2019] and [Yang & Li 2023], which validates the
  decomposition before we use it." Then quantify it — does aleatoric track σ **one-for-one**? A slope
  of ~1 on a plot of estimated aleatoric SD vs injected σ is a much stronger claim than "it goes up",
  and Heid et al.'s aleatoric-limit analysis gives you the reference line.
- **Epistemic rising too is the actual result, and it is publishable.** Ryu et al. state the
  theoretical expectation ("Theoretically, the epistemic uncertainty should not increase by the changes
  in the amount of data noise") and attribute their own small drift to optimiser stochasticity. If the
  user sees a *large* rise, that is a genuine, quantifiable departure, and the disentanglement
  literature (Mucsányi et al. NeurIPS 2024; Valdenegro-Toro & Saromo 2022) supplies the frame:
  estimated components are entangled in practice. Report the rank correlation between the user's own
  aleatoric and epistemic estimates per model — Mucsányi et al. found 0.8–0.999 across twelve methods,
  so a direct comparison is available and would be a novel chemistry-domain data point.
- **A clean falsifiable claim the user can make:** "Under injected label noise, aleatoric uncertainty
  tracks the injected σ for models with an explicit noise model, while epistemic uncertainty rises by
  X% — i.e. predictive uncertainty signals that labels are bad, but *which component* signals it is
  not reliably the one theory assigns." That is defensible, novel in the QSAR setting, and consistent
  with everything above.

### Priority 3 — table layout

Copy Kendall & Gal's Table 3 shape and Scalia's three-panel evaluation:

- Rows: σ level (all 11), grouped by noise strategy (all 6 — do not average, per the user's own rule).
- Columns, per model: `aleatoric` | `epistemic` | `total`, with `total` reported for every model
  because every model has it. Models missing a component get an em-dash and a footnote naming the
  reason ("no parameter posterior; epistemic undefined for a single NGBoost fit").
- A separate table of evaluation metrics (Spearman ρ, miscalibration area or ENCE, NLL, RMV + CV)
  computed **three times**: on aleatoric alone, epistemic alone, and total — Scalia's structure. This
  is what makes the split earn its place rather than just being reported.

### Priority 4 — things to consider dropping or de-emphasising

- Any claim that the six noise strategies produce distinguishable model behaviour needs checking
  against Heid et al.'s result that Gaussian/uniform/hyperbolic/bimodal noise at matched SD gave
  overlapping learning curves. If the user's strategies differ only in shape at matched SD, expect a
  reviewer to ask. If some strategies are heteroscedastic or structure-dependent (e.g. outlier
  injection concentrated on a chemotype), *that* is the distinguishing axis worth foregrounding — and
  Heid et al.'s systematic-noise experiment plus Yang & Li's nitrogen experiment are the precedents.
- Avoid claiming the decomposition is "the" true split. Hüllermeier & Waegeman: the notions "are
  context-dependent in the sense of depending on the setting."

### Priority 5 — citations to add

Methods/decomposition: Kendall & Gal 2017; Depeweg et al. 2018; Hüllermeier & Waegeman 2021.
Chemistry benchmarks: Scalia et al. JCIM 2020; Hirschfeld et al. JCIM 2020; Busk et al. MLST 2022;
Heid et al. JCIM 2023; Yang & Li J Cheminform 2023; Ryu, Kwon & Kim (Chem Sci 2019 / arXiv 2019);
Soleimany et al. ACS Cent Sci 2021; UNIQUE JCIM 2024.
Entanglement caveat: Mucsányi, Kirchhof & Oh NeurIPS 2024; Valdenegro-Toro & Saromo 2022.
Per-model recipes: Rasmussen & Williams 2006 ch. 2 (GP); Al-Aghbary et al. GJI 2026 (QRF);
Harrison, Willes & Snoek ICLR 2024 (VBLL); Duan et al. ICML 2020 (NGBoost).

---

## Sources actually read (URLs successfully fetched)

1. https://arxiv.org/pdf/1910.03127 — Scalia et al., JCIM 60:2697 (2020). Full PDF. Same MVE aleatoric head across all epistemic methods; reports aleatoric/epistemic/total separately; AUCO, error drop, ENCE, dispersion metrics.
2. https://ar5iv.labs.arxiv.org/html/1910.03127 — HTML of the same, used for the metric list.
3. https://arxiv.org/pdf/2005.10036 — Hirschfeld et al., JCIM 60:3770 (2020). Full PDF. Single scalar U(x) per method; Spearman/miscalibration area/NLL/cNLL; excludes NLL for non-variance metrics; synthetic CLogP dataset with "no aleatoric uncertainty".
4. https://arxiv.org/pdf/1703.04977 — Kendall & Gal, NeurIPS 2017. Full PDF. Eq. (6)(8)(9); Table 3 layout; §5.2 behaviour of each component.
5. https://arxiv.org/html/1710.07283 — Depeweg et al., ICML 2018. Entropy Eq. (3) and law-of-total-variance Eq. (4); requirements for computability.
6. https://ar5iv.labs.arxiv.org/html/1910.09457 — Hüllermeier & Waegeman, Machine Learning 110:457 (2021). Definitions; context-dependence caveat.
7. https://arxiv.org/abs/1910.09457 — abstract page (full text came from ar5iv).
8. https://backend.orbit.dtu.dk/ws/files/267165667/Busk_2022_Mach._Learn._Sci._Technol._3_015012.pdf — Busk et al., MLST 3:015012 (2022). Full PDF. Eqs. (5)–(10): mixture decomposition, ENCE, CV; recalibration; the "constant uncertainty is well calibrated but useless" warning.
9. https://arxiv.org/abs/2107.06068 — Busk et al. arXiv abstract page (confirms venue/authors).
10. https://dspace.mit.edu/bitstream/handle/1721.1/159977/heid-et-al-2023-characterizing-uncertainty-in-machine-learning-for-chemistry.pdf — Heid, McGill, Vermeire & Green, JCIM 63:4012 (2023). Full PDF. Aleatoric limit; noise-distribution-shape null result; MVE vs ensemble under systematic noise; sharpness/dispersion advice.
11. https://pmc.ncbi.nlm.nih.gov/articles/PMC9898940/ — Yang & Li, J Cheminform 15 (2023), doi:10.1186/s13321-023-00682-3. Deep-ensemble decomposition equations; nitrogen-noise experiment showing aleatoric rises; held-out-nitrogen experiment showing epistemic rises.
12. https://arxiv.org/pdf/1903.08375 — Ryu, Kwon & Kim (2019 preprint of the Chem Sci paper). Full PDF. Eq. (9) decomposition; logP + Gaussian noise sweep; aleatoric ↑, epistemic ~flat; the "theoretically epistemic should not increase" statement.
13. https://pmc.ncbi.nlm.nih.gov/articles/PMC11600502/ — UNIQUE, JCIM 64:8379 (2024). Data-based vs model-based UQ metrics; Spearman/MACE/NLL; Wilcoxon-based method selection; does not use aleatoric/epistemic language.
14. https://pmc.ncbi.nlm.nih.gov/articles/PMC12848971/ — JCIM (2026), "Uncertainty Quantification in Molecular Machine Learning for Property Predictions under Data Shifts". Chemprop GNN ensembles + distance/error-model UQ; Spearman as primary metric; does not separate aleatoric/epistemic.
15. https://pmc.ncbi.nlm.nih.gov/articles/PMC8393200/ — Soleimany et al., ACS Cent Sci 7:1356 (2021), evidential deep learning. NIG parameterisation; baselines = ensembles, dropout, MVE; metrics = confidence-based ranking, Spearman, calibration. (I could not extract explicit aleatoric/epistemic formulas from the fetched text.)
16. https://arxiv.org/pdf/2404.11599 — Harrison, Willes & Snoek, VBLL, ICLR 2024. Full PDF. Eq. (2) `N(w̄ᵀφ, φᵀSφ + Σ)`; MAP point estimate of the noise covariance.
17. https://gaussianprocess.org/gpml/chapters/RW2.pdf — Rasmussen & Williams, GPML ch. 2. Eq. (2.26) and the statement that noisy-target predictive variance = latent variance + σ_n².
18. https://academic.oup.com/gji/article/245/2/ggag113/8527733 — Al-Aghbary et al., Geophys. J. Int. 245(2) 2026, doi:10.1093/gji/ggag113. Explicit QRF aleatoric/epistemic decomposition equations.
19. https://arxiv.org/abs/2402.19460 — Mucsányi, Kirchhof & Oh, NeurIPS 2024 D&B. Abstract page. "No existing approach provides pairs of disentangled uncertainty estimators in practice."
20. https://iclr-blogposts.github.io/2025/blog/reexamining-the-aleatoric-and-epistemic-uncertainty-dichotomy/ — ICLR 2025 Blogposts track. Sourced summary of the 0.8–0.999 rank correlations and the non-additivity argument. (Blog track, not a journal paper — cite the primaries it points to.)
21. https://arxiv.org/abs/2204.09308 — Valdenegro-Toro & Saromo, "A Deeper Look into Aleatoric and Epistemic Uncertainty Disentanglement". Abstract. Interaction between the two; ensembles disentangle best.
22. https://arxiv.org/html/2306.10060v3 — MUBen benchmark. UQ methods × backbones grid is complete; regression metrics RMSE/MAE/Gaussian NLL/calibration error.
23. https://arxiv.org/abs/2005.10036 and https://arxiv.org/abs/1910.03127 — arXiv abstract pages (metadata confirmation).
24. https://chemprop.readthedocs.io/en/latest/uncertainty.html and https://deepwiki.com/chemprop/chemprop/3.4-uncertainty-estimation — Chemprop uncertainty estimator list; ensemble/dropout labelled epistemic; MVE predicts mean and variance; evidential split into total/epistemic/aleatoric.
25. https://arxiv.org/pdf/2502.03982 — Friesacher, Svensson, Winiwarter, Mervin, Arany & Engkvist, "Temporal Distribution Shift in Real-World Pharmaceutical Data: Implications for Uncertainty Quantification in QSAR Models" (2025). Read partially. Classification-only, AstraZeneca assays; states the aleatoric/epistemic distinction but does not perform a regression decomposition — background only.
26. https://arxiv.org/abs/1903.08375 — abstract page for #12.

## Could not retrieve

- **Heid, Greenman, McGill et al., "Chemprop: A Machine Learning Package for Chemical Property
  Prediction", JCIM 64 (2024), doi:10.1021/acs.jcim.3c01250.** ACS returned 403 and the PDF endpoint
  served an HTML challenge page. I did **not** read it; I only read the Chemprop *documentation*. Do
  not attribute anything to this paper on my account.
- **Heid et al. 2023 via ACS (doi:10.1021/acs.jcim.3c00373)** returned 403 — but I read the identical
  full text from the MIT DSpace author copy, so its content above is verified.
- **Ryu, Kwon & Kim, Chemical Science 10:8438 (2019)** — the RSC HTML returned 403 and the KAIST
  repository link served HTML, not the PDF. I read the **arXiv preprint** (1903.08375) only. The
  quotes above are from the preprint; the journal version has a different title and may differ.
- **"Are we fitting data or noise?", Faraday Discussions, doi:10.1039/d4fd00091a** — RSC returned 403.
  Not read, not described.
- **Uncertainty quantification for molecular property predictions with graph neural architecture
  search, Digital Discovery 3:1534 (2024)** — RSC returned 403. Not read.
- **Uncertainty quantification with graph neural networks for efficient molecular design, Nature
  Communications (2025), s41467-025-58503-0** — redirected to an authentication endpoint. Not read.
- **Greenman / Gómez-Bombarelli multi-fidelity calibrated uncertainty work** — I did not locate and
  retrieve a specific paper; nothing is claimed about it.
- **Duan et al., NGBoost, ICML 2020 (arXiv:1910.03225)** — search results only, full text not fetched.
  The statement that NGBoost outputs a full conditional distribution comes from the search-result
  description and from its documented API, not from a passage I read in the paper.
- **Shaker & Hüllermeier, "Aleatoric and Epistemic Uncertainty with Random Forests" (arXiv:2001.00893)**
  — abstract/search description only, full text not fetched. It is classification-only.
- **Deshwal / Doppa** — not searched in depth; nothing claimed.
