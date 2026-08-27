## A) EXACTLY WHAT KOLMAR & GRULKE (2021) DID

Source file: `/Users/apunt/Downloads/13321_2021_Article_571.pdf` (19 pp, J. Cheminform. 13:92). Text extract used for line cites: `/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/28450b4e-4a9e-4197-993c-548e1dd48b09/scratchpad/kolmar.txt`

**Datasets — 8, spanning 6 endpoint types** (PDF p.6, Table 1; kolmar.txt:288–303):

| Dataset | Category | Entries | Endpoint | Range |
|---|---|---|---|---|
| G298_atom | Quantum mechanical | 131,082 | ΔG°at (kcal/mol) | −2417 to −288 |
| Alpha | Quantum mechanical | 131,082 | α (Bohr³) | 9.0 to 27.8 |
| Lip | Physicochemical | 4,200 | logD | −1.5 to 4.5 |
| Solv | Physicochemical | 642 | ΔG°hyd (kcal/mol) | −25.5 to 3.4 |
| BACE | Biochemical | 1,513 | pIC50 | 2.5 to 10.5 |
| Tox_102 | Tox in vitro | 971 | logAC50 | −2.1 to 4.7 |
| Tox_134 | Tox in vitro | 1,347 | logAC50 | −4.0 to 2.8 |
| LD50 | Tox in vivo | 5,003 | logLD50 (mg/kg) | −1.9 to 4.8 |

Any dataset >1000 molecules was **randomly subsampled to 1,000 before modelling** (kolmar.txt:299, 317–319). So the two 131k QM9-derived sets were modelled at n=1000. Sources: MoleculeNet, EPA ToxCast, DSSTox/Gadaleta LD50 (kolmar.txt:251–260).

**Noise model — ONE: additive zero-mean Gaussian on the target only** (kolmar.txt:325–345):
- `Y_noise_n,i = Y + N(0, σ_noise_n)` (Eq. 7)
- `σ_noise_n = (Y_max − Y_min) * multiplier * n` (Eq. 8), **multiplier = 0.01**, `n ∈ (0,…,14)`, `i ∈ (1,…,5)`
- "Noise was added only to the target variables and not to the descriptors" (kolmar.txt:326–327)
- **Noise magnitude is parameterised as a fraction of the endpoint's observed range** — i.e. 0% to 14% of range in 1% steps. This is dataset-relative, not absolute.

**The crucial design element — two test sets.** Noise is added to a full copy of the dataset, which is then split 75/25. The model trains on `Train_noise_n` and predicts on BOTH `Test_noise_n` (error-laden) and `Test_true` (the original, un-noised labels for those same molecules) (kolmar.txt:354–358, Fig. 2). This yields RMSE_noise/R²_noise and RMSE_true/R²_true for every cell.

**Representations — exactly ONE.** PaDEL descriptors via PadelPy or OPERA; "only the 1,444 1D and 2D Padel descriptors were used" (kolmar.txt:308–309). Representation is never varied. The only descriptor-space experiment is PCA on vs. off (Table 5, kolmar.txt:614–625).

**Models — 5** (Table 2, kolmar.txt:362–386): Ridge, kNN, SVR (RBF kernel, C ∈ {0.01,0.1,1,10}), Random Forest, Gaussian Process (kernel chosen per-dataset, not tuned, because "for most datasets, only a single kernel converged", kolmar.txt:386). Pipeline = StandardScaler → PCA → estimator, with PCA n_components ∈ (1,3,…,59) tuned jointly; 5-fold GridSearchCV (RandomSearchCV, 500 iters, for RF).

**Replicates — 5 per noise level.** 15 levels × 5 reps = 75 datasets → 75 models → 75 RMSE, 75 R², 75 RMSE_true, 75 R²_true per dataset-algorithm cell (kolmar.txt:327–335).

**Metrics:**
- RMSE_observed and RMSE_true (Eqs. 5, 6; kolmar.txt:195–204), R²_noise, R²_true
- Normalised axes: y = RMSE/RMSE₀, x = σ/RMSE₀, where RMSE₀ is the noiseless-baseline RMSE (kolmar.txt:477–487)
- **m_noise** = slope of RMSE_noise/RMSE₀ vs σ/RMSE₀; **m_true** = slope of RMSE_true/RMSE₀ vs σ/RMSE₀; **ratio m_noise/m_true** as the headline single-number statistic (kolmar.txt:539–584). Every slope is reported with ± SE and a p-value.
- GP only: mean prediction uncertainty `mean σŷ` (Eq. 21) and the **95% CI of σŷ** (Eq. 22) — i.e. the *spread* of per-molecule uncertainties, aggregated to a population number (kolmar.txt:745–748).

---

## B) EVERY CONCLUSION THEY DRAW — QUOTED

Abstract (kolmar.txt:27–32):
> "The results show that for each level of added error, the RMSE for evaluation on the error free test sets was always better. The results support the hypothesis that, at least under the conditions of Gaussian distributed random error, QSAR models can make predictions which are more accurate than their training data, and that the evaluation of models on error laden test and validation sets may give a flawed measure of model performance."

Results (kolmar.txt:515–528):
> "For each dataset and algorithm, the RMSE_noise/RMSE₀ clearly increases as σ/RMSE₀ increases. The RMSE_true/RMSE₀ values increase slightly or stay essentially constant, depending on the dataset… The RMSE_noise being consistently higher than RMSE_true for each algorithm and dataset indicates that while the models are retaining their accuracy, our ability to validate the models as being accurate using Test_noise is significantly degraded."

**The consistency-across-algorithms conclusion — directly relevant to us** (kolmar.txt:553–573):
> "For a given dataset, m_noise and m_true are reasonably constant across algorithms (across rows). This observation is consistent with the consistent behavior across algorithms that Cortés-Ciriano observed. For a given algorithm, m (and to a lesser extent m_true) vary more significantly over datasets (down columns). This indicates that the RMSE response to added error is consistent for a given dataset with different algorithms, and that the RMSE response is highly variable for a given algorithm across different datasets."

(kolmar.txt:617–632):
> "For example, the quantum mechanical datasets have high m_noise values (approaching 1) while toxicity datasets have more moderate slopes (near 0.5)… This suggests that the RMSE response to additional noise likely decreases as the amount of native error in a dataset increases. In contrast, m_true varies little and does not follow a decreasing trend over datasets. This observation indicates that these algorithms are capable of finding the 'true' values as simulated error was added, regardless of the amount of native error in the original dataset."

On the metric's instability (kolmar.txt:644–651):
> "This variability comes from the fact that m_true is generally very small, so small changes in this small number lead to large fluctuations in the m_noise/m_true ratios. This instability could be viewed as one detriment of this metric."

On PCA (kolmar.txt:634–653, Table 5):
> "The most dramatic effect is seen across each dataset using the Ridge algorithm, for which the ratios all drop significantly… For kNN and SVR however, the ratios are not sensitive to the use of PCA."

GP uncertainty (kolmar.txt:806–812):
> "when the measurement uncertainty σy is withheld from the algorithm, the slopes of mean σŷ versus σ are all positive. This indicates that prediction precision gets worse as noise is added into the data. These slopes also generally become smaller as the qualitative complexity of the datasets increase."

(kolmar.txt:829–843) — **the strongest actionable claim in the paper**:
> "This result shows that even when datasets have large uncertainty in the measurements, the predictions from GP can apparently become more precise as more error is introduced as long as the magnitude of that error is known, the error is normally distributed, and the error is provided as an input. Error in datasets is not always known, nor is it always normally distributed."

(kolmar.txt:825–828):
> "Including measurement uncertainty in the calculation decreases the slope for each of the datasets, even causing some of the slopes to become negative… This reductive effect is mild for the quantum mechanical and physiochemical datasets but becomes more pronounced for the in vitro and in vivo datasets."

Discussion/conclusions (kolmar.txt:900–917):
> "Because QSAR models are evaluated on these test and validation sets, this means that QSAR models are being judged by their ability to predict error laden values, when they should be judged by their ability to predict the population means of measurements… A more exact statement would be that cross/external validation statistics (our standard metrics of predictivity) for QSAR models are limited based on the accuracy of the dataset."

(kolmar.txt:918–937):
> "The results also show that the difference between the observed RMSE and the unknown RMSE_true depends on algorithm and dataset complexity. This is an important observation, because it suggests that when models using different algorithms are compared, they may have significantly different accuracies, even if the observed RMSEs are very close… Because real world datasets are undeniably rife with unknown amounts of error, this example demonstrates that comparing QSAR models through error laden test sets may be producing misleading conclusions in terms of model performance."

(kolmar.txt:938–953):
> "It is important to recognize that error in training sets appears to result in only a minor increase in 'true' predictive error as assumed in this work (at least when work with datasets containing 1000 datapoints). In general, QSAR evaluation techniques cause us to perceive large amounts of predictive error when our training sets have error… These observations were made by Cortés-Ciriano and coworkers on pIC50 datasets, and the current work complements and extends those initial studies."

(kolmar.txt:953–972):
> "Therefore, new learning methods will not resolve the issue. While some methods like Gaussian Processes and Conformal Prediction take error into account as part of training and allow modelers to estimate prediction precision, there are associated limitations. Conformal Prediction requires that a segment of the training set be put aside for calibration, while Gaussian Process requires a reasonable prior distribution and some knowledge of the experimental uncertainty to be effective… Efforts towards estimating uncertainties of other common QSAR endpoints would be welcome."

Self-declared limitations (kolmar.txt:207–209, 888–899): the assumption that original dataset values are true "directly contradicts our premise that doing so is dangerous"; only single experimental values exist; test/validation sets are assumed error-free.

---

## C) FIGURES AND TABLES

**6 numbered figures + 1 graphical abstract; 7 tables.**

| Item | Content | Source |
|---|---|---|
| Graphical abstract | PDF p.2 | kolmar.txt:57 |
| Fig. 1 | Concept schematic: rows of horizontal bar triplets — **red = population mean ("true"), grey = experimental value, blue = prediction**. Makes the ε_true vs ε_observed argument visually. | kolmar.txt:122–127 |
| Fig. 2 | Modelling workflow / ML pipeline schematic: noisy-dataset generation → split into Test_noise_n/Train_noise_n and Test_true/Train_true → GridSearchCV → Best estimator → fit on Train_noise → predict both test sets. | kolmar.txt:354–358 |
| **Fig. 3** | **The key figure.** RMSE vs added error, 8 panels. | kolmar.txt:498–501; PDF p.9 |
| Fig. 4 | R² vs added error, same 8-panel layout, same two-colour scheme. | kolmar.txt:507–510; PDF p.10 |
| Fig. 5 | GP only: RMSE vs added error, 4 panels (Solv, Tox134) × (no σy fed / σy fed). | kolmar.txt:766–768; PDF p.14 |
| Fig. 6 | GP only: top row = mean 95% CI of prediction uncertainty vs added error; bottom row = mean prediction uncertainty vs added error; G298_atom and Tox134. | kolmar.txt:818–820; PDF p.15 |
| Table 1 | 8 datasets: category, entries, endpoint, range, refs | kolmar.txt:287–303 |
| Table 2 | 5 algorithms + hyperparameter search spaces | kolmar.txt:362–386 |
| Table 3 | m_noise and m_true, 8 datasets × 4 algorithms, ± SE | kolmar.txt:588–608 |
| Table 4 | m_noise/m_true ratios, 8 × 4, ± SE | kolmar.txt:521–546 |
| Table 5 | m_noise/m_true **without PCA**, 8 datasets × 3 algorithms (RF omitted — compute cost) | kolmar.txt:614–625 |
| Table 6 | GP m/m_true, 7 datasets, No σy vs With σy | kolmar.txt:773–785 |
| Table 7 | GP slopes of mean σŷ and σŷ 95% CI vs σ, No σy vs With σy | kolmar.txt:773–793 |

**Visual design of Fig. 3 (verified by rendering PDF p.9 to image):** a boxed 4-row × 2-column grid of eight small line plots — left column G298_atom, right column Tox134; rows Ridge / KNN / SVR / RF. Each panel title is `dataset, algorithm`. Two series per panel: a **blue** line (evaluated on the noisy test set) that climbs steeply and near-linearly, and an **orange** line (evaluated on the noise-free test set) that stays essentially flat just above 1.0. Both carry vertical error bars (spread over the 5 replicates at each level). Each panel has a small **inline text box in the upper-left containing the fitted slope ± SE and the p-value for both series** — e.g. G298_atom/Ridge: RMSE_noise slope 0.93 ± 0.02, p = 7.05e-17; RMSE_true slope 0.16 ± 0.01, p = 5.82e-11. Axes are `RMSE / RMSE₀` (y) vs `σ / RMSE₀` (x), so both are dimensionless and every panel is directly comparable. The x-range differs per panel because σ is range-relative and RMSE₀ differs.

**How they present the noise-vs-performance relationship:** as a *pair of divergent lines on doubly-normalised axes*, with the entire argument carried by the **gap between them**. The reader's eye is drawn to a widening wedge. They then compress each panel to two slopes and report the ratio. There is no heatmap anywhere in the paper. There is no area-under-curve metric. There is no ranking of models by robustness.

---

## D) PRACTICAL GUIDANCE FOR A READER WHO SUSPECTS NOISY DATA

Honestly: **they give very little, and they say so.** The paper is a diagnosis, not a prescription. What exists, quoted verbatim:

1. **Reinterpret your validation statistics; don't conclude your model is bad** (kolmar.txt:894–917):
> "The RMSE, when calculated for these test sets, may be quite high, and thus the model is judged to be flawed… These results show that those models may very well be predicting the population means of those measurements, but this fact is obscured by the error in the test sets. Even from a very conservative interpretation of the results shown here, this study indicates that this situation is plausible."

2. **Do not rank models on noisy test statistics** (kolmar.txt:921–937):
> "it suggests that when models using different algorithms are compared, they may have significantly different accuracies, even if the observed RMSEs are very close. For example, examining the Solv row in Table 3, the m_noise/m_true ratio is 3.3 for SVR and 6.1 for RF. This means that in a real modeling situation, if these SVR and RF algorithms produced the same RMSE for the Solv dataset, the RMSE_true's (and the relevant comparison) would be different by a factor of 1.8."
   *(Note their own internal inconsistency: text says "Table 3… 6.1 for RF"; the number 6.0 for Solv/RF is in Table 4, kolmar.txt:543. Cite carefully if quoting.)*

3. **Feed the known measurement uncertainty to a GP if you have it** (kolmar.txt:829–843, quoted in full in §B). With the three explicit preconditions: magnitude known, error normally distributed, error provided as input. Immediately hedged: "which, admittedly, is not a situation that is common in QSAR modeling."

4. **Don't expect a new algorithm to save you** (kolmar.txt:953–954):
> "Therefore, new learning methods will not resolve the issue."

5. **Benchmark your σ against published assay-error estimates** (kolmar.txt:676–681):
> "Kramer, Kalliokoski and colleagues found from an examination of the ChemBL database that heterogeneous pIC50 data has an average standard deviation of 0.68 log units. For the BACE dataset, which uses a pIC50 endpoint, 1.1 log units of noise were added, or 1.6 times the average standard deviation reported in ChemBL."

6. **Sanity-check a probabilistic model's predicted uncertainty against the literature assay SD** (kolmar.txt:861–868):
> "Using the RMSE₀ value of 0.98 for the GP calculations on the BACE dataset, the mean σŷ is 0.79 log units. The estimated experimental uncertainty for pIC50 is 0.68 log units, so GP's prediction uncertainty is 1.2 times the experimental estimate, when no simulated noise has been added to the dataset."

7. **Call to the field** (kolmar.txt:971–972): "Efforts towards estimating uncertainties of other common QSAR endpoints would be welcome."

---

## E) THE GAP — WHAT WE DO THAT THEY DO NOT

Read from `/Users/apunt/repos/qsar_qm_models/paper.tex` lines 375–577.

### Genuinely novel in ours

**E1. Noise *structure*, not just noise *magnitude*. This is our strongest and cleanest novelty.** Kolmar has exactly one noise model: additive homoscedastic Gaussian on the target (kolmar.txt:325–327). We define six — Gaussian, outlier (z>2 → 3σ, else 0.1σ), quantile (outside Q10/Q90 → 2σ), threshold (|y|>1 → 2σ), value-proportional (σ(1+0.1|y|)), heteroscedastic (σ√(0.1+0.05|y|)) — at `paper.tex:328–341` and `paper.tex:354`. Kolmar explicitly flags this as their own limitation twice ("this condition is certainly not representative of every real-world data situation", kolmar.txt:173–175; "Error in datasets is not always known, nor is it always normally distributed", kolmar.txt:834–836). We fill exactly the hole they name. Our finding that structure changes the *severity* (AUC_norm spread 0.02 under outlier vs 0.13 under threshold, `paper.tex:460`) while barely changing the *ranking* (Kendall's W = 0.9121, p = 3.55e-8, 11 models × 6 strategies, `paper.tex:433`) is new — Kolmar could not have produced it.

**E2. Representation as an experimental factor, with a variance decomposition.** Kolmar has one descriptor set (1,444 PaDEL 1D/2D, kolmar.txt:308–309) and never varies it; their only descriptor-space manipulation is PCA on/off (Table 5). We cross model architecture × representation (ECFP4/Morgan, PDV, SMILES/SNS, MHG-GNN, mol2vec, PDV-binary) and run an ANOVA η² decomposition per noise strategy (`paper.tex:396–401`, Table `tab:anova_decomposition`). Nobody in Kolmar's design can ask "is it the model or the features?" This is a real gap we fill.

**E3. Per-sample uncertainty–noise tracking.** Our paper claims this explicitly and correctly at `paper.tex:499`: *"\citet{Kolmar2021} found that the mean predicted uncertainty derived from the GPs increases with the amount of label noise in the training data. We go beyond the population level and instead ask if a model's per-sample uncertainty tracks label noise."* **Verified: this is accurate.** Kolmar's Eq. 21 is the *mean* of σŷ and Eq. 22 is the population 95% CI of σŷ (kolmar.txt:745–748); Table 7 reports only the slopes of those two aggregates. They never correlate a molecule's predicted uncertainty with that molecule's injected error. Our Spearman ρ(uncertainty, |ε|) per molecule (`paper.tex:503`, `paper.tex:510–526`) is a strictly finer measurement. Note also that Kolmar states the per-sample uncertainty *cannot* respond to label noise for their GP — "the prediction uncertainty σŷ is completely dependent on the descriptor values and is independent of whether the prediction is evaluated using the true test set or the noisy test set" (kolmar.txt:748–754) — which is a hypothesis our GP ρ = 0.56 on SNS (`paper.tex:510`) directly tests.

**E4. Model roster.** They have Ridge, kNN, SVR, RF, GP (kolmar.txt:362–386). We have 11 in the ranking table (`paper.tex:444–454`): NGBoost, RF, LightGBM, XGBoost, SVM, BNN-α/β, VBLL-α/β, NN-α/β, plus QRF and GP elsewhere. Modern gradient boosting and modern variational Bayesian layers (VBLL, `paper.tex:216`) simply did not exist in their comparison. The controlled base→variant Wilcoxon design (NN→BNN, NN→VBLL, RF→QRF; `paper.tex:477–481`) is a cleaner causal contrast than anything in Kolmar.

**E5. Replicates.** 10 vs their 5 (`paper.tex:380` context / `paper.tex:460`; kolmar.txt:328).

### Where we are NOT novel — say this plainly

**E6. "Models are robust to label noise; performance on clean labels degrades slowly." This is Kolmar's result, and Cortés-Ciriano's before that.** Our AUC_norm curves measure the retention of R² on a **clean test set** — the framework "applies artificial noise to the training labels while preserving the integrity of the test set" (`paper.tex:366`), and this is confirmed in the code: `rust/src/main.rs:750` gates noise on `config.noise && apply_noise`, and the val and test `write_data` calls pass `false` (`rust/src/main.rs:1054`, `rust/src/main.rs:1078`) while train passes `true` (`rust/src/main.rs:1028`). **That means every R²(σ) curve we plot is Kolmar's orange line and nothing else.** Their m_true values across 8 datasets × 4 algorithms span 0.00 to 0.27 against m_noise 0.36 to 0.98 (kolmar.txt:591–606) — they already showed, quantitatively and with p-values, that clean-test degradation is small. If our headline is "models are more robust than you'd think", that is theirs. Ours must be the *ordering* and the *decomposition*, not the phenomenon.

**E7. "Algorithms differ in their noise response."** Attributed by Kolmar to Cortés-Ciriano 2015 (kolmar.txt:146–155): *"algorithms have different levels of sensitivity to added random experimental error, such that while algorithm A might have a lower RMSE than algorithm B at low noise levels, algorithm A can have a higher RMSE than algorithm B at high noise levels."* Twelve algorithms, twelve datasets, ten noise levels. Our 11-model ranking is a refresh of that with a modern roster, not a new question.

**E8. 🔴 DIRECT CONFLICT — our headline claim is the opposite of theirs, and we do not acknowledge it anywhere in 375–577.** Our conclusion (`paper.tex:571`): *"model architecture is the dominant factor, while representation explains less than 10% of variance"*, and `paper.tex:562`: *"noise robustness… is primarily determined by the model's training mechanism."* Kolmar found (kolmar.txt:553–563): *"For a given dataset, m_noise and m_true are reasonably constant across algorithms… For a given algorithm, m… vary more significantly over datasets… the RMSE response to added error is consistent for a given dataset with different algorithms, and… highly variable for a given algorithm across different datasets."* They put the variance in the **dataset**; we put it in the **model**. Both can be true (they varied dataset and held representation fixed; we varied representation and mostly held dataset fixed at QM9 — 1 QM9 + 3 ADME, `paper.tex:556`), but a JCheminf reviewer who knows this paper will ask, and right now `paper.tex` line 499 is the *only* place Kolmar is engaged with at all. **This needs a paragraph.**

**E9. Not novel: "PDV is preferable when dealing with noisy data."** `paper.tex:462` says PDV "stood out as having particularly strong robustness to noise" and "PDVs are preferable when dealing with noisy data" — but `paper.tex:567` says the opposite: *"PDVs produce the strongest predictive performance on clean data, but are the least noise-resistant."* These two sentences are in the same Results section and contradict each other. Independent of Kolmar, this must be fixed.

---

## F) WHAT KOLMAR & GRULKE WOULD SAY IS MISSING FROM OURS

**F1. We never compute their central quantity. We only have one of the two test sets.** Their entire contribution is `RMSE_noise` vs `RMSE_true` — the wedge in Fig. 3. Our pipeline holds the test set clean (`paper.tex:366`; `rust/src/main.rs:1054`, `:1078`), so we report only RMSE_true / R²_true. We therefore cannot say anything about what a practitioner *observes* when their own held-out labels are noisy — which is the situation every real modeller is in. Kolmar would say: you have measured the quantity nobody can measure in practice, and omitted the quantity everybody actually sees. Adding a noisy-test-set arm would be cheap (the noise map already exists) and would let us do the model×representation×strategy decomposition on the **gap**, which nobody has done. That is arguably a bigger paper than the one we have.

**F2. Our σ has no physical anchor in the Results.** Kolmar ties σ to the endpoint range (Eq. 8, kolmar.txt:341), then converts to log units and benchmarks against Kalliokoski's ChEMBL pIC50 SD of 0.68 log units, stating BACE received 1.1 log units = 1.6× the published assay SD (kolmar.txt:676–681). Our σ ∈ {0, 0.1, …, 1.0} (`paper.tex:240`) is on normalised labels and nowhere in `paper.tex:375–577` is a single σ value translated into log units or compared to a published assay SD. A reader cannot tell whether σ = 0.3 (the ANOVA operating point, `paper.tex:380`) is negligible or catastrophic in assay terms. **This is the most fixable and highest-value criticism.** They would also note their per-strategy doses are not equal — a σ label means a different physical perturbation under Gaussian vs threshold vs outlier — and our claim at `paper.tex:354` that "the difficulty scaling controlled by σ is consistent" across strategies is asserted, not demonstrated.

**F3. No native-error accounting on the validation sets.** Kolmar's own stated limitation is assuming original values are true (kolmar.txt:207–209, 888–899). For QM9 we are genuinely stronger (computed labels, `paper.tex:575`). But for LogD, Caco-2 and hERG Ki, we acknowledge the noise floor in exactly one sentence (`paper.tex:551`: "the injected σ therefore stacks on top of an unknown noise floor") and then draw conclusions from those datasets anyway (`paper.tex:551`, `:560`). Kolmar's finding that "the RMSE response to additional noise likely decreases as the amount of native error in a dataset increases" (kolmar.txt:623–625) predicts exactly this confound and we do not test for it.

**F4. Never feeding known noise magnitude to the model.** Their single most positive result is that giving σy to the GP changes the picture qualitatively (Table 6: G298_atom ratio 6.9±1.5 → 1.7±0.26; Alpha 1.8±0.11 → 9.3±0.37; BACE 3.7±2.0 → 8.6±1.6; kolmar.txt:777–785). We inject noise we know the magnitude of and never tell any model. Our VBLL "learns a scalar observation noise variance" (`paper.tex:216`) — an obvious oracle-vs-learned comparison sitting unused.

**F5. Dataset and endpoint diversity.** 8 datasets across 6 endpoint categories, deliberately chosen to span a native-error gradient (QM → physchem → biochem → in vitro → in vivo, kolmar.txt:262–281) vs. our 1 QM9 task + 3 ADME sets. Their design is what licenses their dataset-dominance claim; ours cannot rebut it on 4 datasets.

**F6. No uncertainty on our robustness numbers.** Every Kolmar slope carries ± SE and a p-value, in the tables and printed inside every Fig. 3 panel. Our `tab:auc_ranking` (`paper.tex:444–454`) gives bare point estimates to 3 decimals, and the gaps we treat as meaningful (NGBoost 0.851 vs RF 0.846 vs LightGBM 0.845 under Gaussian) are smaller than any spread we report. Kolmar would ask whether NGBoost's rank-1 finish survives the replicate spread — and they'd point at their own warning about ratio metrics built on small denominators being unstable (kolmar.txt:644–651), which applies directly to AUC_norm normalising by clean-data R².

**F7. Their PCA experiment is our missing control.** Table 5 (kolmar.txt:614–625) shows the robustness metric is *algorithm-dependently* sensitive to dimensionality reduction: Ridge ratios collapse without PCA, kNN and SVR are unaffected. Our representations differ enormously in dimensionality (ECFP4 vs PDV vs 300-d embeddings) and we attribute the differences to representation *content*. Kolmar would say some of our model×representation interaction may just be dimensionality.

---

## G) FLAGS FOUND WHILE READING (must fix regardless of comparator)

- **NDS is still in the paper in 11 places inside 375–577**, after the metric was replaced by AUC_norm: `paper.tex:387` (Table `tab:anova_decomposition` caption says "Robustness uses the noise degradation slope (NDS)" while `paper.tex:409`, the figure caption for the *same numbers*, says AUC_norm), `:462`, `:464`, `:470`, `:471`, `:475`, `:493`, `:495`, `:551`, `:556`, `:571`, `:573`.
- **Table `tab:anova_decomposition` (`paper.tex:396–401`) does not match the prose above it.** Prose at `paper.tex:380` says model architecture is 36.8–54.7% and residual dominates at 83.6% / 77.4% for outlier and heteroscedastic. The table gives Heteroscedastic model 37.0 / residual 41.0 and Outlier model 12.7 / residual 79.3. 83.6 and 77.4 appear nowhere in the table. Trace both to `results/paper_figures_v2/table1_anova_summary.csv` before either survives.
- **Averaging flags.** (a) `tab:auc_ranking` "Mean" column (`paper.tex:442–457`) averages AUC_norm across all six noise strategies — this hides that threshold spread is 0.13 and outlier spread is 0.02 (`paper.tex:460`), i.e. the mean is dominated by strategies where models barely differ. (b) `paper.tex:495` ranks configurations by "Gaussian NDS" and reports a top-10 census — a ranking over an average across 10 replicates with no spread. (c) `paper.tex:560` "QRF was consistently less robust than RF on every external data set" — check whether "consistently" is per-replicate or on replicate means. (d) Every AUC_norm cell is a mean over 10 replicates with no dispersion reported anywhere in 375–577.
- **Self-contradiction on PDV**: `paper.tex:462` ("PDV stood out as having particularly strong robustness to noise… PDVs are preferable when dealing with noisy data") vs `paper.tex:567` ("PDVs… are the least noise-resistant").
- **Uncertainty workstream (one line, per instruction):** ECE columns are still present in `tab:top_unc_noise` (`paper.tex:503`, `:508`, `:510–526`) despite the metric having been removed elsewhere — collides with the NDS/AUC_norm cleanup pass on the same table region.
- `paper.tex:381` is a duplicated fragment of `paper.tex:380` ("The choice of model architecture is instead the largest source of variance…") — leftover text.

---

## H) OVERINGTON 2017 — WHAT IT GIVES US ON REAL ASSAY-NOISE STRUCTURE

Source: `/Users/apunt/Downloads/2017.Overington.Big Data, Noisy Data, and Hard to Find Data in Drug Discovery.pdf`. **This is a PowerPoint slide deck ("Oxford DTC - February 2017", 30+ slides), not a paper.** Numeric content lives in figure images, not text. Slides rendered and read directly.

- **Slide 20, "Inter-lab Variability" — same compound, same species, different publication, n = 3,000** (Krüger & Overington 2012, PLoS Comp. Biol. DOI:10.1371/journal.pcbi.1002333). Left: pKi Assay1 vs pKi Assay2 scatter, visibly wide. Right: density of `diff(assay1, assay2)`, centred on 0, peak density ≈ 0.58, **x-axis spanning −4 to +4 log units with clearly non-zero density out to ±3**.
- **Slide 19, "Inter-species Assay Variability" — same compound, same endpoint, rat vs human orthologs, n = 2,781**, same plot pair, peak density ≈ 0.6.
- **Slide 21, "Inter-species vs Inter-lab Variability"** overlays both densities on one axis (x = pKi_ii − pKi_ij, −6 to +6). The two curves are nearly superimposed; **inter-laboratory (black) is slightly wider in the shoulders than inter-species (red)**, both peaking at ~0.56–0.57.
- **No standard deviation is stated numerically anywhere in the deck.** NO SOURCE FOUND for an SD value in this file — the underlying Krüger & Overington 2012 PLoS Comp. Biol. paper would need to be read for one.
- Slide 17, "Errors in ChEMBL" (Tiikkainen et al. 2013, JCIM DOI:10.1021/ci400099q): bar chart "Activity parameter error rates", y-axis %, four parameter classes — **Ligand highest at roughly 5–7% error rate across WOMBAT/ChEMBL/Evolvus/All-discrepant; Target ~2.5–3.2%; Activity value ~0.2–2.7%; Activity type <0.3%.** Quoted caption: *"The more complex the parameter, the more frequent the errors."* (over.txt:168–172). These are read off a chart image, not from text — treat as approximate and cite Tiikkainen 2013 directly if used.

**What this means for our σ anchoring — three points, and only the third is a number:**

1. **The shape argument is the useful one.** The inter-lab difference density on slide 20/21 is sharply peaked with long tails out to ±3–4 log units. That is **not** Gaussian; it is leptokurtic. A Gaussian fitted to that distribution over-weights the shoulders and under-weights both the spike at zero and the far tail. This is direct, citable support for our six-strategy design — specifically for the outlier strategy (most molecules barely perturbed, a minority moved hard) — and it is exactly the objection Kolmar raises against their own Gaussian-only design (kolmar.txt:834–836). **This is the single best use of the Overington material in our paper.**

2. **A difference between two measurements has √2 times the SD of a single measurement.** The plotted quantity is `pKi_assay1 − pKi_assay2`, not a single-measurement error. Any SD read off these curves must be divided by √2 before it is comparable to our injected σ, which is a single-label perturbation. If we cite width from these plots we must say which we mean.

3. **The only hard number available to anchor σ remains the one Kolmar already used**: heterogeneous ChEMBL pIC50 SD = **0.68 log units** (Kalliokoski et al. 2013, PLoS ONE 8(4):e61007, cited at kolmar.txt:676–681). Kolmar's own normalisation — "1.1 log units of noise were added, or 1.6 times the average standard deviation reported in ChemBL" (kolmar.txt:678–681) — is the exact sentence pattern our paper is missing. **Recommendation: state, for at least the three ADME validation sets, what σ = 0.3 and σ = 1.0 correspond to in log units and as a multiple of that endpoint's published assay SD.** Do not read that multiplier off the Overington slides; take it from the primary papers.