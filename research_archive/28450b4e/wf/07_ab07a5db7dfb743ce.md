# PAPER 1 — Dablander et al. 2023, J Cheminform 15:47

## A) Results subsection headings, in order

Section is titled "Results and discussion" (single combined section, per JCheminf convention), followed by a separate "Conclusions".

1. **"QSAR-prediction performance"** — *Which representation/regressor gives the lowest error on the ordinary single-molecule task?* Answer: ECFP > GIN > PDV; MLP-ECFP lowest MAE on all three targets.
2. **"AC-classification performance"** — *Can these same models detect activity cliffs, and does that depend on how much is known at test time?* Answer: strong on M_inter, collapses on M_test/M_cores; GIN wins here.
3. **"PD-classification performance"** — *Can they at least say which of the pair is more potent?* Answer: weak overall (~0.7 / ~0.6), but >0.9 / >0.8 once restricted to pairs the model itself predicted to be cliffs.
4. **"Linear relationship between QSAR-MAE and AC-MCC"** — *Are the two abilities the same ability?* Answer: yes, tightly linear — the paper's synthesis subsection, which exists purely to connect subsections 1 and 2.

The four headings track: task A → task B → task C → relationship between A and B.

## B) Every figure and table

| Item | Type | What it shows | Variables resolved at once | The one sentence it exists to support |
|---|---|---|---|---|
| **Table 1** | Counts table | Compounds / MMPs / ACs / half-ACs / non-ACs / AC:non-AC ratio for 3 datasets | dataset × 6 counts | "AC prediction is a severely imbalanced task, and the imbalance differs per target (≈1:68, ≈1:8, ≈1:20)." |
| **Fig. 1** | Chemical structure pair | One factor Xa activity cliff, ChEMBL assay 658338 | — | "A tiny structural change can move activity by ~3 orders of magnitude." |
| **Fig. 2** | Schematic (set diagram) | The M_train / M_inter / M_test / M_cores split | 4 evaluation regimes | "Our split defines four distinct AC-prediction scenarios with different amounts of leakage." |
| **Fig. 3** | Schematic (factorial grid) | 3 representations × 3 regressors → 9 models, each run through 2-fold CV × 3 seeds with an inner hyperparameter loop | 3 × 3 × 3 tasks × 6 trials | "Every representation is combined with every regressor, so representation and model effects can be read off separately." |
| **Figs. 4, 5, 6** | 3×3 grid of **scatter plots with error bars** — one figure per target (D2, factor Xa, SARS-CoV-2 Mpro) | y = QSAR-MAE on D_test; x = AC metric. Columns = MMP-set (M_inter, M_test, M_cores); rows = metric (MCC, sensitivity, precision). 9 labelled points per panel = the 9 models. Error bar = 2 SD over the 6 hyperparameter-optimised runs | model (9) × MMP-set (3) × AC-metric (3) × 2 prediction tasks simultaneously, ×3 targets across the figure set | "Models that predict activities well also classify cliffs well, but only when one activity is already known." |
| **Figs. 7, 8, 9** | Same 3-column layout, 2 rows — one figure per target | y = QSAR-MAE; x = PD accuracy. Upper row = all MMPs, lower row = only MMPs the model predicted to be ACs | model (9) × MMP-set (3) × restriction (2) × 2 tasks, ×3 targets | "Potency direction is near-random in general but reliable on the pairs the model flags as cliffs." |

Note the discipline: **no heatmaps, no bar charts, no ranking table of the sweep.** The entire factorial result is delivered as labelled scatter points in a small-multiple grid.

## C) How the model×representation grid is presented without averaging

- Every one of the 9 models is a **separately labelled point** in every panel. Nothing is collapsed to a model marginal or a representation marginal; the reader does the marginalising by eye.
- **Both axes are metrics**, so a point's position states two results at once. The lower-right corner is defined as good on both — stated in every caption: *"For each plot, the lower right corner corresponds to strong performance at both prediction tasks."*
- The **three datasets are three separate figures with identical layout** (Figs 4/5/6, then 7/8/9). Generalisation is demonstrated by visual repetition, never by averaging across targets.
- Replicate spread is shown, not hidden: *"The total length of each error bar equals twice the standard deviation of the performance metric measured over all mk = 3 * 2 = 6 hyperparameter-optimised models."*
- Comparisons in the text are always stated as **held-one-factor-fixed contrasts**, e.g. *"the combination GIN-kNN consistently performs considerably better for AC-classification than the combinations ECFP-kNN and PDV-kNN"* — regressor fixed, representation varied.
- Missing cells are declared rather than imputed: *"The precision of the AC-classification task is lacking for the ECFP + kNN technique on M_test and M_cores since this method produced only negative AC-predictions for all trials on this data set"* (Fig. 6 caption).

## D) Practical guidance — actual sentences

It is a **named-baseline recommendation**, delivered as prose sentences inside the Results, then repeated in the Conclusions. Not a flowchart, not a decision tree.

- *"The combinations GIN-MLP, GIN-RF and ECFP-MLP exhibit particularly high AC-MCC values relative to the other methods. We recommend using at least one of these three models as a baseline against which to compare tailored AC-prediction models; the practical utility of any AC-prediction technique that cannot outperform these three common QSAR methods is questionable."*
- *"The combination ECFP-MLP reaches the strongest PD-accuracy in the majority of cases and we recommend starting with this model as a baseline for more advanced PD-prediction methods."*
- *"We thus recommend using GINs as an AC-classification baseline since such an agreed-upon baseline is currently lacking."*
- Scenario-conditional guidance: *"if the activity of one MMP-compound is known (i.e., present in the training set) then AC-sensitivity increases substantially; for query compounds with known activities, QSAR methods can therefore be used as simple AC-prediction-, compound-optimisation- and SAR-knowledge-acquisition tools."*

The rhetorical move: guidance is framed as *what other researchers must now beat*, which converts a benchmark into an obligation on the field.

## E) Handling "no single method wins"

They never claim one winner. Instead they **split the win by task and say so explicitly**, and give a mechanism for each half:

- *"while GINs appear to be inferior to ECFPs for QSAR-prediction, they tend to be advantageous for AC-classification; their highly parametric nature might simultaneously lead to increased overfitting but to a better modelling of the more jagged regions of the SAR-landscape."*
- *"RFs tend to exhibit the strongest AC-precision and the weakest AC-sensitivity. This might be as a result of their ensemble nature which should intuitively lead to conservative but trustworthy predictions of extreme effects such as ACs."*
- Then a **cross-task law** rescues it from being a list of exceptions — the fourth subsection: *"The results suggest that for real-world QSAR models the AC-MCC and the QSAR-MAE are strongly predictive of each other; while this observation only rests on nine models, it is highly consistent across MMP-sets and pharmacological targets."*
- And a **forward-looking consequence**: *"it might be possible to considerably boost the performance of common QSAR models by developing techniques to increase their AC-sensitivity which could potentially provide a fruitful direction for future research."*

Split the win by task + explain each split mechanistically + supply one law that holds across the whole grid + state what the field should build next. No shrug.

## F) Main-text : additional-file figure ratio

**9 main-text figures + 1 main-text table : 0 additional files.** A full-text scan for "additional file", "supplementary", "Figure S", "Table S" returns nothing. Everything overflow goes to the public GitHub repo: *"All used data sets, the code to reproduce and visualise the experimental results, and the exact numerical results generated by the original experiments are available in our public code repository."*

---

# PAPER 2 — Venkatraman 2021, FP-ADMET, J Cheminform 13:75

## A) Results subsection headings, in order

Section "Results and discussion" is very short and has **only one internal heading**, followed by a separate "Conclusion":

1. *(unheaded opening block)* — *How good are the best fingerprint models per endpoint, and which endpoints are actually modellable?* Answer: BACC > 0.80 for a named list; moderate 0.71–0.78 for a second named list; "somewhat average" for a third; regression mostly poor except pKa, logS, logD, HSA, skin penetration.
2. *(unheaded block, keyed to Figs 1 and 2)* — *Which fingerprints work across endpoints?* Answer: pharmacophore (2PPHAR/3PPHAR) poor everywhere; PUBCHEM/MACCS/KR/ECFP/FCFP good; PUBCHEM, ECFP4, ASP best for regression.
3. *(unheaded block, keyed to Fig 3)* — *Do fingerprints match published 2D/3D-descriptor models?* Answer: comparable, with named exceptions in both directions.
4. *(unheaded block)* — *Can you tell when to trust a prediction?* Prediction intervals + conformal confidence/credibility, both pushed to Additional file 1.
5. **"Software usage"** — *How do I actually run this on my molecules?* A literal command line.

## B) Every figure and table

| Item | Type | What it shows | Variables resolved | The one sentence it exists to support |
|---|---|---|---|---|
| **Table 1** | Spec table | 20 fingerprints and their bit lengths (MACCS 166 … KR 4860) | 1 | "The sweep covers substructure, circular, path-based and pharmacophore families." |
| **Table 2** | Inventory table | ~56 classification endpoints: model type (BC/MC), #compounds, ADMET group, data source | endpoint × 4 attributes | "This is a comprehensive, fully sourced endpoint collection, not a cherry-picked subset." |
| **Table 3** | Inventory table | 19 regression endpoints, same columns | endpoint × 4 | same, for regression |
| **Table 4** | **Ranking / lookup table** | Per endpoint: the *winning* fingerprint name + BACC and AUC on calibration and validation | endpoint (~56) × best-FP × 2 metrics × 2 splits | "For every endpoint we name the single fingerprint you should use and what accuracy to expect." |
| **Table 5** | **Ranking / lookup table** | Same for the 19 regression endpoints: best FP + R², RMSE, MAE, calibration and validation | endpoint (19) × best-FP × 3 metrics × 2 splits | same, regression |
| **Table 6** | Worked example | pKa and anticommensal-effect predictions for 3 molecules with confidence, credibility, Q=0.025, Q=0.975 | 3 molecules × 6 outputs | "Here is what the tool literally hands you, uncertainty included." |
| **Fig. 1** | **Heatmap** | Cross-validated balanced accuracy, endpoint (rows) × fingerprint (20 columns), classification | endpoint × fingerprint × BACC — the *entire* sweep in one image | "Pharmacophore fingerprints fail everywhere; substructure-key fingerprints are broadly good." |
| **Fig. 2** | **Heatmap** | Same layout, cross-validated R², regression | endpoint × fingerprint × R² | same claim, regression side |
| **Fig. 3** | **Grouped bar chart, small multiples** (one panel per endpoint, ~20 panels; bars labelled with the winning method name, e.g. "PUBCHEM" vs "1D/2D") | This study's accuracy ("Current") vs the previously published descriptor model ("Original") | endpoint × 2 methods × accuracy, with method names printed on the bars | "Fingerprint models are comparable with published 2D/3D descriptor models on most endpoints." |

## C) How the model×representation grid is presented without averaging

Note the difference from Dablander: Venkatraman fixes the model (**random forest only**, with a brief SVM aside) and sweeps representation × endpoint. So the grid is 20 fingerprints × 75 responses = 1500 models.

- The **whole grid goes in as two heatmaps** (Figs 1 and 2), unaveraged: endpoint on one axis, fingerprint on the other, colour = the metric. Nothing is marginalised into a "mean over endpoints" bar.
- The heatmaps are read for **patterns, not point values**: *"While the pharmacaphore fingerprints (2PPHAR/3PPHAR) perform poorly on all datasets, others such as the PUBCHEM, MACCS and KR encodings show moderate to high accuracies for a majority of the modelled endpoints."*
- **Tables 4 and 5 are the argmax slice of the heatmap** — for each endpoint row, the winning fingerprint and its numbers. Argmax, not mean; the identity of the winner is preserved (PUBCHEM for BBB, FCFP4 for BCRP, AT2D for HLM stability, ASP for PGP substrate…).
- Calibration and validation columns sit side by side on every row, so overfitting is visible per endpoint rather than summarised.
- Small print does declare one average: *"average of 3 independent runs"* (Table 4 and 5 footnotes, Fig. 1 and Fig. 2 captions) — the replicate average only, and it is disclosed in every caption.
- The complete unreduced numbers go to Additional file 1: *"The complete performance summary for the training and validations sets is listed in Additional file 1: Tables S1 and S2."*

## D) Practical guidance — actual sentences

Three delivery mechanisms, stacked from most to least abstract:

1. **Lookup table** — Tables 4/5. A practitioner finds their endpoint row, reads the fingerprint name. That *is* the guidance; no prose needed.
2. **Prose rules over the heatmaps**: *"While the pharmacaphore fingerprints (2PPHAR/3PPHAR) perform poorly on all datasets, others such as the PUBCHEM, MACCS and KR encodings show moderate to high accuracies for a majority of the modelled endpoints."* and *"here too the R²cv for PUBCHEM, ECFP4, and ASP fingerprints yield better models than the other fingerprints tested."*
3. **A runnable command plus a worked interpretation** — the "Software usage" subsection: *"bash runadmet.sh -f molecule.smi -p ## -a"*, then Table 6 read aloud: *"A confidence value of 0.95 suggests that the classifier is quite certain that the prediction is likely to be a single label. A relatively low value of credibility (0.57) suggests that the compounds like G00001 are not sufficiently represented in the training set and that the user needs to treat the prediction with caution."*
4. **An operational filtering rule** — *"Such a strategy that allows for compound selection based on static thresholds for the confidence/credibility offer a way to reduce the number of compounds that typically undergo experimental testing."*

## E) Handling "no single method wins"

- It **converts the absence of a global winner into a per-endpoint answer** and ships that as the deliverable. Tables 4 and 5 have a different fingerprint in the FP column on nearly every row; that heterogeneity is the product, not an embarrassment.
- It **states a floor claim that does hold globally**: *"We find that for a majority of the properties, fingerprint-based random forest models yield comparable or better performance compared with traditional 2D/3D molecular descriptors."*
- It **names the losers unambiguously**, which is a real result: pharmacophore fingerprints poor on all datasets.
- It is **specific about where the competition wins and why**: *"for some endpoints such as myelotoxicity, ototoxicity, myopathy accuracies obtained using 2D/3D descriptors are only marginally better… For phototoxicity in particular, quantum chemistry-based 3D descriptors are used which can add to the time taken. It must however be pointed out that some of the better performing models take advantage of deep learning."*
- It **ends on an artefact, not an opinion**: *"A total of 1500 models were analysed spanning 75 responses and 20 fingerprints… the best performing models have been compiled into an open access software package called FPADMET."*

## F) Main-text : additional-file figure ratio

**3 main-text figures + 6 main-text tables : 1 Additional file containing Tables S1–S2 and Figures S1–S2** (plus endpoint descriptions). So 3 main figures : 2 supplementary figures; 6 main tables : 2 supplementary tables. Every main-text item is load-bearing; the supplement holds the full unreduced numeric dump and two diagnostic plots.

---

# COMPARISON — three structural things our paper does worse

Sources: `/Users/apunt/repos/qsar_qm_models/paper.tex`.

### 1. We collapse the factorial sweep into a mean-ranked table and one representation; neither comparator ever does that

`paper.tex:437` — the caption of `tab:auc_ranking` reads *"Normalised retention area (AUC_norm) by model on PDV (QM9 HOMO–LUMO gap), **ranked by mean across six strategies**"*, and `paper.tex:442–455` carries a `\textbf{Mean}` column that sets the row order. **This is an average over the six noise strategies.** The six columns are shown, which is the minimum, but ordering by the mean is what the reader takes away, and it hides the thing the paper itself says matters: threshold spread is 0.13 while outlier spread is 0.02 (`paper.tex:460`), so the mean is dominated by strategies that do not discriminate, and "NGBoost is most robust" (`paper.tex:433`) is a statement about a number no experiment produced.

Compounding it: that table is **PDV only**, with ECFP4 exiled to Additional file 6 (`paper.tex:433`, `paper.tex:664`), so the representation axis — half the paper's stated subject — is absent from the main comparison. Dablander shows all 9 model×representation combinations as individually labelled points in every one of ~9 panels per figure, and repeats the whole figure per target rather than pooling targets. Venkatraman puts the entire 20-fingerprint × ~56-endpoint grid on screen as Figs 1 and 2 and then takes the **argmax per row** into Tables 4/5 — never the mean. Two other averages compound this: `\Delta NDS` in `tab:wilcoxon_bnn` is described as *"mean change in noise degradation slope"* (`paper.tex:470`) pooling across representations and strategies, and `paper.tex:495` reduces the sweep to a top-10 count of a **single** strategy (Gaussian).

**Fix shape:** replace the mean-ranked table with either (a) six small-multiple panels, one per strategy, models as labelled points — Dablander's device; or (b) a full model × strategy × representation heatmap with a per-strategy winner column — Venkatraman's device. Delete the Mean column and rank by nothing, or rank per strategy.

### 2. Our headline claim has no main-text figure and no subsection; Dablander gives exactly this claim its own heading

"Robustness is decoupled from clean-data performance" is asserted four times — `paper.tex:433`, `paper.tex:460`, `paper.tex:567`, `paper.tex:573` — and its only evidence is **Additional file 7**, *"Baseline R² versus NDS scatter (PDV, Gaussian)"* (`paper.tex:665`). The one plot that would prove the paper's most quotable claim is in the supplement, for one representation and one strategy.

Dablander's structurally equivalent claim gets a **dedicated Results subsection with its own heading** ("Linear relationship between QSAR-MAE and AC-MCC") and is visible in *every* main-text figure, because their whole plotting design puts the general metric on y and the specialist metric on x. Our figures never plot performance against robustness together at all: `fig:global_overview` (`paper.tex:427`) has degradation curves and a heatmap; `fig:interaction` (`paper.tex:418`) plots AUC_norm against AUC_norm.

**Fix shape:** promote a baseline-R² vs AUC_norm scatter, all models labelled, faceted by strategy (six panels, no averaging), into the main text under its own subsection heading; make the two-axis "good in both corners" convention explicit in the caption the way Dablander does in all six of Figs 4–9.

### 3. We produce no practitioner deliverable, and we end on the shrug both comparators avoid

`paper.tex:567`: *"At the end of the day, the choice of both model and representation comes down to your problem, including the amount of noise in your data, your objectives, and your compute limits."* That sentence is the shrug. Against it:

- Dablander: *"We recommend using at least one of these three models as a baseline against which to compare tailored AC-prediction models; the practical utility of any AC-prediction technique that cannot outperform these three common QSAR methods is questionable."*
- Venkatraman: Tables 4/5, where a reader looks up their endpoint and is handed a fingerprint name, plus `bash runadmet.sh -f molecule.smi -p ## -a`.

Ours has no named baseline to beat, no lookup keyed on anything the reader possesses (their noise regime, their dataset size, their compute budget), and no artefact. The material exists — `paper.tex:462` (SVM and full BNNs robust across all representations; RF and NN-β only with particular ones), `paper.tex:531`/`565` (models with a learned per-sample scale track noise; posterior-over-weights models do not) — but it is left as observations rather than turned into a rule of the form "if your labels carry roughly X, start with Y and report against it."

**Fix shape:** a guidance table keyed on the practitioner's situation (noise regime × objective: best retention / best per-sample noise tracking / cheapest), with a named default configuration per cell and the number it must beat — Venkatraman's argmax-lookup pattern — plus one Dablander-style sentence naming the baseline the field should now compare against. Then cut `paper.tex:567`.

---

# Flags required by the brief

**NDS still present throughout paper.tex, though the metric was replaced by AUC_norm.** Occurrences: `paper.tex:387` (the `tab:anova_decomposition` caption says the Robustness columns use *"the noise degradation slope (NDS)"*, while `fig:anova_decomposition`'s caption at `paper.tex:409` says the same quantity is AUC_norm — **the table and figure captions contradict each other about which metric the numbers are**), and `paper.tex:462, 464, 470, 475, 493, 495, 551, 556, 560(implicit), 571, 573, 598, 660, 661, 662, 664, 665, 666, 669`. Note `tab:wilcoxon_bnn` (`paper.tex:470–481`) reports Δ NDS values with no AUC_norm equivalent, and `fig:validation_overview` at `paper.tex:547` is captioned AUC_norm while `fig:validation_combined` at `paper.tex:556` is captioned NDS — the two validation figures use different metrics.

**Averages flagged:** `paper.tex:437`+`442–455` (Mean over 6 strategies, and the table is ordered by it — hides that outlier spread is 0.02 vs threshold 0.13, per `paper.tex:460`); `paper.tex:470` (Δ NDS = mean change, pools representations and strategies); `paper.tex:495` (top-10 count from Gaussian alone, presented as a general conclusion); `paper.tex:562` (*"accounting for less than ~10% of the variance in robustness"* — a claim across strategies that `paper.tex:380` itself contradicts for outlier and heteroscedastic, where residual dominates at 83.6% and 77.4%).

**Additional-file ratio:** ours is **8 main-text figures + 6 main-text tables : 12 additional files** (`paper.tex:658–671`) — inverted versus Dablander (9 figures : 0 additional files) and Venkatraman (3 figures + 6 tables : 1 additional file). More of our evidence sits outside the paper than inside it.

**One-line uncertainty note (separate workstream, collides here only structurally):** `tab:top_unc_noise` at `paper.tex:503–528` still has an ECE column, and the ANOVA-exclusion threshold is stated as baseline R² < 0.3 at `paper.tex:422` but R² ≤ 0.6 in Additional file 5's description at `paper.tex:663`, and R² < 0.6 at `paper.tex:464` — three different thresholds for the same exclusion rule.