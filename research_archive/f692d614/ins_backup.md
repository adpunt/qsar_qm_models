# Immediate next steps

Step 1 of 5: triage what exists, decide what the results actually say, re-plan the figures.
Nothing here has been applied to `paper.tex` or to any script.

Every number below was read from a file in this session. The source file is named next to it.
Where a number does not exist anywhere, it says so instead of guessing.

Last updated 2026-08-21.

---

## The one-line argument, kept in view

> What, if anything, makes a QSAR model robust to noise — and can a model's uncertainty tell you
> when your data is bad?

Two halves. The first half is answerable and the answer is clearer than the paper currently says.
The second half is in trouble and item 4 explains why.

---

## 0. Two things in the noise-injection code that have to be settled first

Both found by reading `rust/src/main.rs` directly. Neither is an analysis problem — they are upstream of
every QM9 number in the paper. Nothing else in this document should be acted on before these are resolved.

### 0a. The QM9 test and validation sets are being given the training set's noise

The noise map is built over training indices only — `(0..train_count)`. Then `preprocess_data` calls
`write_data` three times, once for train, once for validation, once for test, **passing the same noise
map every time**. And `write_data` restarts its counter at zero on each call: `for index in 0..data_count`.

Because the test and validation splits are smaller than the training split, every one of their indices
finds an entry in the map. So test molecule number 7 gets the noise that was drawn for **training**
molecule number 7.

Two consequences:

1. **The test labels are corrupted.** The Methods say validation and test data remain noise-free. As
   the code stands, they do not. Every reported R² above σ = 0 on QM9 is measured against a corrupted
   reference, which mixes "the model got worse" together with "the target moved".
2. **The corruption is attached to the wrong molecules.** So any correlation between a molecule's
   predicted uncertainty and "its" injected noise is zero by construction. This is the single reason
   the per-sample uncertainty result in item 4 cannot be believed.

**The experimental datasets do not have this problem.** They go through the KIRBy pipeline, not the
Rust one. I checked directly: in `results/validation_full/openadmet_logd/QRF_PDV_uncertainty_values.csv`
the true labels are **bit-identical at all eleven noise levels** — the test set is perfectly clean.

So QM9 and the experimental datasets are currently not running the same experiment, and any comparison
between them is comparing noisy-test results against clean-test results.

**What to do.** Gate the noise lookup on the training split in `write_data`, rebuild, re-run QM9.
Before committing to that, confirm the deployed binary matches this source with one command on ARC —
open any `*_uncertainty_values.csv` from a QM9 run and check whether the true-label column changes
between σ = 0 and σ = 1. If it does not change, only the analysis is affected. If it does, everything
QM9 needs re-running.

### 0b. The noise-strategy parameter file is never used

`scripts/process_and_train.py` defines a `--strategy-params` option pointing at
`scripts/noise_strategy_params.json`, but **it does not pass that flag when it launches the Rust
binary**. Rust therefore sees an empty parameter object and falls back to its own built-in defaults
on every run.

This is mostly good news, and it corrects something in the current draft:

- The paper says value-proportional uses a factor of 0.1. **The paper is right** — 0.1 is the Rust
  default. The JSON's 0.05 was never read. (An earlier note in the revision guide saying the code
  uses 0.05 is wrong; ignore it.)
- The JSON file is dead code that contradicts what actually ran. Either delete it or make the flag
  get passed — but if you make it get passed, every number changes.

What the strategies actually do, at noise level σ, computed from the real QM9 gap values
(first 10,000 molecules, `data/QM9/raw/gdb9.sdf.csv`, converted to eV; mean 7.00, SD 1.35, range
2.08–16.93):

| Strategy | Noise SD applied | RMS dose | Per-molecule spread |
|---|---|---|---|
| Gaussian | σ on every molecule | 1.000 | none |
| Threshold | 2σ on every molecule | 2.000 | none |
| Value-prop. | σ·(1 + 0.1·y) | 1.701 | 2.2× |
| Heteroscedastic | σ·√(0.1 + 0.05·y) | 0.669 | 2.2× |
| Quantile | 2σ on the top and bottom 10 %, 0.1σ on the rest | 0.899 | 20× |
| Outlier | 3σ where \|z\| > 2, 0.1σ elsewhere | 0.502 | 30× |

**Threshold noise is inert on QM9.** Its rule is "add 2σ if the label is above 1.0". The smallest
HOMO–LUMO gap in the data is 2.08 eV, so **100.00 % of molecules** clear the cut and every one gets
the same multiplier. On QM9, threshold noise *is* Gaussian noise at double strength — nothing more.
Either set the cut-points from the target's own quantiles or say plainly in the Methods that this
strategy reduces to a dose change on this dataset.

This table is also the mechanistic explanation for the strategy clustering in item 8, and the
explanation for the residual in item 5.

---

## 1. Representation, model, interaction — what the ANOVA actually says

Source: `results/paper_figures_v2/table1_anova_summary.csv` and
`table1_supp_simple_effects.csv`. Roster is 11 models × 5 representations
(continuous PDV, ECFP4, MHG-GNN, Mol2vec, SMILES). Performance is R² at σ = 0.3;
robustness is `auc_norm`.

### The table

| Strategy | Perf: Model | Rep | Interaction | Resid | Robust: Model | Rep | Interaction | Resid |
|---|---|---|---|---|---|---|---|---|
| Gaussian | 24.6 | 22.8 | **48.1** | 4.5 | 43.8 | 5.2 | 16.9 | 34.2 |
| Quantile | 25.1 | 22.4 | **46.9** | 5.6 | 36.8 | 4.4 | 15.1 | 43.7 |
| Threshold | 25.0 | 21.2 | **48.0** | 5.8 | 54.7 | 7.9 | 22.6 | 14.8 |
| Heteroscedastic | 27.9 | 23.5 | **46.2** | 2.5 | 14.0 | 0.7 | 8.0 | **77.4** |
| Value-prop. | 27.8 | 22.0 | **46.5** | 3.8 | 52.5 | 6.0 | 19.9 | 21.6 |
| Outlier | 25.3 | 22.0 | **49.2** | 3.5 | 10.3 | 0.2 | 5.9 | **83.6** |

### Reading it, for performance

**Neither representation nor model drives accuracy. The pairing does.**
Interaction is the biggest single term on every one of the six strategies — 46 to 49 percent —
and it is more stable across strategies than anything else in the table. Model and representation
each take about a quarter and are within a few points of each other (model's share of the two main
effects is 52–56 % on every strategy).

This is a different claim from the one in the paper, and it is a better one. It says the question
"which representation is best?" has no answer on its own, and neither does "which model is best?".

The simple-effects table shows the shape of that interaction, and the shape is usable:

**How much your model choice matters depends on your representation.** Model effect within each
representation, averaged over the six strategies:

| Representation | Model effect on accuracy |
|---|---|
| Mol2vec | 98.9 % |
| MHG-GNN | 97.4 % |
| SMILES | 93.5 % |
| Continuous PDV | 78.3 % |
| ECFP4 | **61.3 %** |

On ECFP4 it barely matters which model you use. On Mol2vec it is almost the only thing that matters —
because on Mol2vec some models work and some do not work at all.

**And how much your representation choice matters depends on your model.** Representation effect
within each model, same averaging:

| Model | Rep effect on accuracy |
|---|---|
| DNN-BNN, MLP-BNN | 99.8, 99.7 % |
| MLP-VBLL, DNN-VBLL | 98.6, 98.2 % |
| NGBoost | 97.9 % |
| RF | 95.2 % |
| XGBoost, LightGBM | 91.2, 90.1 % |
| SVM | 74.8 % |
| MLP, DNN | **61.7, 59.4 %** |

Bayesian neural networks live or die by the representation. Plain networks and SVM are the least
sensitive to it.

Plain-English version of the whole performance half: **the more you constrain a model — Bayesian
priors, boosting, a fixed kernel — the more it depends on getting the representation right. Plain
flexible models care less. And the better the representation, the less your model choice matters.**

### Reading it, for robustness

Here the picture is simple and consistent, and it is the paper's strongest surviving result.

**Representation is essentially irrelevant to robustness.** It never exceeds 7.9 %, and on two
strategies it is under 1 %. Of the variance that model and representation jointly explain, model
takes 87–98 % on every single strategy. That ratio is the thing that generalises across noise
mechanisms — it is far more stable than the raw numbers make it look.

So: **how well you predict is a joint property of model and representation; how fast you lose that
prediction under noise is a property of the model alone.** That is a clean, quotable finding, it
answers research question 1 properly, and it is true on all six strategies.

### The residual, and why the two halves of the table look so different

See item 5 for this in full. Short version: the residual is not a finding about heteroscedastic and
outlier noise. It is a measurement-floor effect — those two strategies barely damage anything, so
there is barely anything for the ANOVA to explain.

### What is missing, and it matters

`calculate_robustness` in the figure script emits only `auc_norm`, the clean-data R², and the number
of σ levels. **R² at σ = 0.6 does not exist for QM9 in any local file.** Neither does R² per model
per representation per σ. Confirmed by reading every CSV in `results/paper_figures_v2/`: the only
QM9 clean-data R² anywhere is `table3_probabilistic_comparison.csv`, which covers 8 models on PDV
under Gaussian noise only.

That is why the accuracy-under-noise comparison on QM9 cannot be written yet. It is a step-2 job and
it is item 8 below.

---

## 2. The validation ANOVA — why it is broken, and the fix (done locally)

### Why it is broken

`table_validation_anova.csv` shows residual η² = 0.0 on all three datasets. That is not missing data.
It is a design collapse, and here is the mechanism:

1. The validation loader averages the 5 cross-validation folds together at each noise level **before**
   integrating the retention curve. One `auc_norm` per model × representation × strategy.
2. The validation ANOVA then filters to Gaussian noise only.
3. That leaves **exactly one observation per model × representation cell** — 18 or 19 observations
   for a 7 × 3 grid.

With one observation per cell there is no within-cell variation left, so the residual is forced to
zero by arithmetic and the interaction term absorbs everything unexplained. The reported
Model η² of 91.8 / 92.4 / 95.2 % is an artefact of that, not a measurement.

The roster is also thin: 7 models × 3 representations, against 11 × 5 on QM9. Not comparable.

### The fix — and you already have the data locally

`/Users/apunt/repos/KIRBy/tests/results/validation_rerun/` holds 48,510 rows: **13 models × 4
representations × 3 datasets × 6 strategies × 11 σ × 5 folds, with the folds kept separate.**
Three cells are missing (the Gaussian process was only ever run on PDV). Nothing else is missing.

I recomputed the ANOVA from it: `auc_norm` per fold, folds as replicates, Gaussian process excluded,
SNS dropped to match the QM9 convention, same integration formula as the figure script. Results
(robustness, `auc_norm`):

| Dataset | Strategy | Model | Rep | Interaction | Residual | n | models |
|---|---|---|---|---|---|---|---|
| LogD | Gaussian | 13.5 | 5.1 | 44.9 | 36.5 | 103 | 7 |
| LogD | Quantile | 11.1 | 7.6 | 30.8 | 50.5 | 117 | 8 |
| LogD | Threshold | 11.0 | 6.8 | 28.3 | 54.0 | 118 | 8 |
| LogD | Heteroscedastic | 16.6 | 11.5 | 36.1 | 35.7 | 129 | 9 |
| LogD | Value-prop. | 14.7 | 5.0 | 45.3 | 35.1 | 103 | 7 |
| LogD | Outlier | 16.9 | 9.6 | 55.9 | 17.5 | 118 | 8 |
| Caco-2 | Gaussian | 60.2 | 0.7 | 6.1 | 33.0 | 86 | 6 |
| Caco-2 | Quantile | 61.3 | 5.2 | 8.4 | 25.1 | 86 | 6 |
| Caco-2 | Threshold | 17.9 | 2.4 | 43.1 | 36.6 | 99 | 7 |
| Caco-2 | Heteroscedastic | 9.5 | 3.6 | 29.0 | 58.0 | 98 | 7 |
| Caco-2 | Value-prop. | 59.8 | 0.9 | 5.5 | 33.9 | 86 | 6 |
| Caco-2 | Outlier | 53.0 | 4.9 | 14.0 | 28.1 | 86 | 6 |
| hERG | Gaussian | 62.3 | 3.7 | 2.2 | 31.9 | 88 | 6 |
| hERG | Quantile | 66.8 | 2.2 | 2.8 | 28.2 | 88 | 6 |
| hERG | Threshold | 73.5 | 3.2 | 3.1 | 20.3 | 88 | 6 |
| hERG | Heteroscedastic | 26.8 | 2.0 | 27.5 | 43.6 | 99 | 7 |
| hERG | Value-prop. | 28.8 | 0.4 | 17.9 | 52.9 | 100 | 7 |
| hERG | Outlier | 46.9 | 1.8 | 5.3 | 46.0 | 88 | 6 |

Working script: `scratchpad/val_anova2.py`; per-fold metrics in `scratchpad/val_fold_metrics.csv`.

**Representation η² is 0.4–11.5 % on every dataset and every strategy — the same near-zero
representation effect as QM9.** That is research question 3 answered honestly: the pattern does
generalise, and it generalises on the part of the result that was always the strongest.

The Model η² varies a lot (9.5 % to 73.5 %) and that variation tracks dataset, not strategy. On
Caco-2 and hERG the model dominates; on LogD, which is the highest-signal dataset, the interaction
term takes over instead.

### The thing this fix uncovered, which is bigger than the fix

Clean-data R² by model and representation on hERG, from the same per-fold data:

| Model | ECFP4 | MHG-GNN | PDV | SNS |
|---|---|---|---|---|
| LightGBM | 0.507 | 0.496 | 0.509 | 0.533 |
| RF | 0.496 | 0.514 | 0.497 | 0.536 |
| DNN | 0.484 | 0.362 | **−389** | 0.416 |
| MLP | 0.461 | 0.153 | **−32.7** | 0.452 |
| VBLL-Full | 0.400 | −0.618 | **−204** | 0.431 |
| BNN-Full | 0.029 | 0.030 | **−544** | 0.103 |

Every neural model works fine on fingerprints and fails catastrophically on the continuous
descriptor vector — on the smallest dataset only. On LogD and Caco-2 the same networks are fine on
PDV, and it is MHG-GNN that breaks them instead (−0.08 to −16.2).

**This is what the 46–49 % QM9 interaction term is made of.** It is not a gentle "some pairings work
better". It is specific model × representation pairs collapsing entirely, and the pattern is
consistent: dense continuous features break neural networks when the dataset is small. On QM9
(10,000 molecules) PDV is fine; on hERG (1,482) it is not. And PDV is the paper's primary
representation.

### Decisions needed

- **Switch the validation source from `alternative_full` to `validation_rerun`.** 13 models instead
  of 7, four representations, folds preserved. The figure script needs one flag changed.
- **Keep folds as replicates.** Do not average before integrating. Same rule as QM9's iterations.
- **Run the validation ANOVA per strategy**, matching QM9, rather than Gaussian only.
- **Decide what to do about the Gaussian process.** It only ever ran on PDV, so it cannot enter a
  cross-representation ANOVA. Either run it on the other three or drop it from the ANOVA and say so.
- **Report the neural-network failures explicitly** rather than gating them away silently. See item 6.

---

## 3. Wilcoxon tables — to fix

Current state: `table3_wilcoxon_tests.csv` reports one comparison per model pair, on `auc_norm` only,
pooled across four representations and six strategies at once — while sitting in a section that is
about one representation. The function accepts a representation argument and a strategy argument and
then ignores both.

The paper's printed values do not match this file, and one significance verdict flips: DNN → DNN-VBLL
is p = 0.25 in the current output, i.e. **not significant**, so "both transformations improved both
networks" is three of four, not four of four.

**What it should become:** for each model pair, the change in all three headline quantities —
clean R², R² at σ = 0.6, and `auc_norm` — reported **per strategy**, not pooled. Six rows per pair
instead of one. Paired across replicates within a strategy, at a single stated representation.

The current test is answering "does this transformation help on average over everything", which is
the averaging you have said repeatedly you do not want.

---

## 4. Uncertainty — are these real results?

**No.** Not the per-sample half. Here is the honest answer, in order of how much each point matters.

### The rows are test molecules, and test molecules were not supposed to have noise

`save_uncertainty_values` is called with the test-set predictions and the test-set labels. So the
question the analysis is asking — "does uncertainty flag the molecules whose labels I corrupted?" —
is being asked about molecules that were never meant to be corrupted. Those are also the molecules
that, because of item 0a, got *someone else's* noise. The correlation is zero by construction.

**To ask the question at all you need per-sample uncertainty on the training molecules.** Those rows
are never written. This is not a bug in the analysis; the data does not exist.

### The injected noise is reconstructed by regression, and it does not survive the reconstruction

The uncertainty files never recorded how much noise each molecule got — the normalisation mean and
standard deviation were not saved either, so the labels cannot even be put back on their original
scale. `fix_injected_noise` therefore fits a straight line from the clean label to the noisy label
and calls the leftovers "the noise".

That fails in two specific ways:

- **At σ = 0 no noise was injected at all**, so the leftovers are floating-point rounding — whose size
  grows with the size of the label, which is also where uncertainty is largest. That is exactly why
  the zero-noise control has a *higher* average correlation (0.0799) than σ = 0.6 does (0.0568). The
  control fails, and it fails in the direction that manufactures a positive result.
- **For outlier and quantile noise**, where a few molecules get very large errors, those few points
  drag the fitted line. What is left over for the other 80–95 % of molecules is roughly "how extreme
  is this label", which correlates with uncertainty for reasons that have nothing to do with noise.
  Outlier and quantile are precisely the two strategies that showed any signal.

The reconstruction also groups by model, representation, noise level and replicate — **but not by
strategy** — so runs normalised on different scales are pooled into one fit.

### The effect sizes would not be useful even if they were real

Verified from `table4_supp_uncertainty_by_strategy_rep.csv`: pooled correlation averages 0.259, but
once you condition on noise level it drops to 0.033 at σ = 0.3 and 0.047 at σ = 0.6. Outside outlier
and quantile the surviving correlations run 0.026 to 0.095 — the same size as the σ = 0 rounding
artefact.

In practical terms: at a correlation of 0.05, the most-uncertain tenth of your molecules is 11.7 %
bad labels instead of 10 % — a 1.17× enrichment. At 0.26 it is 20.3 %, a 2.03× enrichment. The first
is worthless. The second would be interesting if it were real, and it is not, because it comes
entirely from the two strategies where the reconstruction is known to break.

### Noise levels

The analysis conditions at three levels (0, 0.3, 0.6) out of eleven. Extending it to 0.8 and 1.0 is
worth doing **as a diagnostic, not as a stronger test** — the reconstruction artefact grows with the
amount of injected noise, so a stronger correlation at higher σ would be evidence of the artefact,
not of detection. σ = 0 must stay in permanently as the control.

### Experimental datasets

Per-sample uncertainty exists for six files only: LogD (QRF on all four representations, GP on PDV)
and Caco-2 (QRF on ECFP4). **Nothing at all for hERG.** The columns are noise level, sample index,
true label, prediction and uncertainty — no injected noise, no clean-versus-noisy pair, and values
already averaged over the CV folds. **The uncertainty-versus-noise claim is QM9-only** and cannot
currently be checked anywhere else.

### What survives

One thing, and it is worth keeping: **uncertainty ranks a model's own errors**, weakly but almost
everywhere. From `deep_qm9_uncertainty_by_model.csv` — QRF 0.263, NGBoost 0.220, GP 0.176, the
Bayesian nets 0.111–0.147. That is a real, defensible, modest result, and QRF being best is
interesting because the paper currently dismisses it.

But note this is measured against the noisy label, which means the "error" being ranked contains the
noise itself. It should be recomputed against the clean label before it is quoted.

### What has to change on the server

1. **Fix item 0a first.** Nothing here means anything until it is done.
2. **Write out the true injected noise from the Rust side** — split, index, epsilon — and join it on.
   Stop reconstructing.
3. **Save uncertainty for training molecules.** Without them the headline question cannot be asked.
4. **Save the normalisation mean and standard deviation, the strategy, the split, and a real molecule
   identifier.** The current `sample_idx` is a row position, so rows cannot be linked to molecules or
   matched across replicates.
5. **Score errors and coverage against the clean label**, not the noisy one.
6. **Compute the conditioned correlation inside a single replicate**, not pooled across replicates.

### What this means for the paper's argument

The second half of your one-line argument — *can a model's uncertainty tell you when your data is
bad?* — currently has no supporting evidence, and the honest answer on present data is "we cannot
tell". That is worth stating plainly rather than reporting a correlation of 0.26 that is an artefact.

The population-level version of the question is still answerable and still interesting: **average
uncertainty does rise as you add noise.** That is a real signal, it is what the pooled 0.259 is
actually measuring, and it should be presented as what it is — a population-level response, not
per-sample detection.

---

## 5. The residual, in plain terms

You asked why heteroscedastic and outlier noise have a residual of 77 % and 84 % while threshold has
15 %. Here it is without the jargon.

**Step one: the six strategies do not do the same amount of damage.**
Average retention across all 11 models (`deep_qm9_model_robustness_by_strategy.csv`), where 1.0 means
no damage at all:

| Strategy | Retention | |
|---|---|---|
| Outlier | 0.944 | barely scratches the model |
| Heteroscedastic | 0.908 | |
| Quantile | 0.846 | |
| Gaussian | 0.825 | |
| Value-prop. | 0.654 | |
| Threshold | 0.587 | wrecks the model |

**Step two: the amount of damage decides how far apart the models end up.**
Gap between the best and worst model's retention, same file:

| Strategy | Best−worst gap |
|---|---|
| Outlier | 0.034 |
| Heteroscedastic | 0.042 |
| Quantile | 0.069 |
| Gaussian | 0.083 |
| Value-prop. | 0.140 |
| Threshold | 0.180 |

Threshold noise spreads the models out **five times further** than outlier noise does.

**Step three: that is the whole story.** Line the two columns up against how much variance the ANOVA
explains and they are in exactly the same order — perfect rank agreement, all six strategies, no
exceptions (Spearman = −1.00 against total explained variance, and −1.00 against the model term
alone):

| Strategy | Retention | Model η² | Residual η² |
|---|---|---|---|
| Threshold | 0.587 | 54.7 | 14.8 |
| Value-prop. | 0.654 | 52.5 | 21.6 |
| Gaussian | 0.825 | 43.8 | 34.2 |
| Quantile | 0.846 | 36.8 | 43.7 |
| Heteroscedastic | 0.908 | 14.0 | 77.4 |
| Outlier | 0.944 | 10.3 | 83.6 |

**So the plain statement is:** when noise barely hurts, every model looks the same, and the leftover
run-to-run wobble is most of what is left to explain. The 83.6 % residual under outlier noise does
not mean "model architecture stops mattering when labels have outliers". It means outlier noise at
these σ values is too gentle to tell the models apart. It is a sensitivity floor, not a result.

**Does anything need to change?** Yes, two things, both small:

1. **Stop presenting the residual as a finding.** The current text reads the large residual as
   evidence that architecture stops mattering. Replace with the measurement statement above.
2. **Print how much damage each strategy does, next to the ANOVA.** One extra column of retention,
   or a σ-rescaling so the strategies are compared at equal damage rather than equal σ. Without it
   the reader has no way to know the six columns are not on a common scale, and the Methods currently
   claim they are.

---

## 6. Neural-network failures — you were right

You are already filtering these, twice. There is a per-run gate that deletes any training run whose
R² falls below −0.5, and a separate gate that drops any model × representation × strategy whose
clean-data R² is below 0.3.

Looking at what those gates actually caught (`filtered_catastrophic_iterations.csv`, 245 rows, and
`excluded_configs.csv`, 48 rows), the earlier framing was wrong. It is not "neural networks are
unstable". It is much narrower:

- The 245 deleted runs are almost entirely **MLP-VBLL (137) and DNN-VBLL (95)**, on **Mol2vec (146)
  and MHG-GNN (97)**. Everything else combined: 13 runs.
- All 48 dropped configurations are the four Bayesian neural variants on Mol2vec and MHG-GNN, and
  their clean-data R² is between −0.02 and −0.19 — i.e. **at zero noise they are no better than
  predicting the mean.** They never trained. That is not a robustness result at all.

So there is no separate "neural networks fail" takeaway. There is one sentence, and it belongs with
the interaction result in item 1: **four Bayesian neural variants do not train on Mol2vec or MHG-GNN
at all, so they carry no robustness number on those inputs** — and the same thing happens on the
experimental data, where every neural model collapses on PDV/hERG and on MHG-GNN/LogD.

The only thing worth adding is one limitation sentence: the filtering is not random with respect to
the question. It removes the worst runs of the least stable configurations, which nudges their
retention scores upward. Worth stating; not worth a paragraph.

### ➜ CARRY TO REVISION GUIDE

Two short additions, to be written into `REVISION_GUIDE.md` when step 3 happens:

- **Methods / exclusions.** One sentence stating that four Bayesian neural variants (DNN-BNN,
  DNN-VBLL, MLP-BNN, MLP-VBLL) do not train on Mol2vec or MHG-GNN — clean-data R² between −0.02 and
  −0.19 at zero noise — and therefore carry no robustness number on those inputs. This is a
  representation-compatibility fact, not a noise result, and it belongs next to the exclusion table.
- **Limitations.** One sentence stating that the catastrophic-run filter is not random with respect
  to the question: it removes the worst runs of the least stable configurations, which biases their
  retention scores upward.

---

## 7. Aleatoric / epistemic — keeping it, and how

You are not dropping it, so here is how real papers do it, what you are doing differently, and what
to change. Sources are in `scratchpad/ALEA_EPIS_LITERATURE.md` with the URLs that were actually read.

### The finding that changes the framing

**"Aleatoric uncertainty rises when I inject label noise" is the expected result, not an anomaly.**

Ryu, Kwon and Kim (Chemical Science, 2019) ran essentially your experiment — noise-free synthetic
data, Gaussian noise injected at increasing levels — and got aleatoric and total uncertainty rising
while epistemic stayed roughly flat. Their words: *theoretically, the epistemic uncertainty should
not increase by the changes in the amount of data noise.* Yang and Li (J Cheminform, 2023) report the
same on QM9.

So your aleatoric result is the **sanity check that the decomposition is working**, and it is
stronger if you show the slope is roughly one-for-one against injected σ.

Your epistemic component rising *is* the interesting part, and there is a place to put it:
Mucsányi et al. (NeurIPS 2024) found that across twelve different methods, aleatoric and epistemic
estimates rank-correlate between 0.8 and 0.999 — the two components are entangled in practice
everywhere, not just in your hands. Reporting the correlation between your own two components would
be a novel data point in a chemistry setting.

That is a much better paragraph than the one currently in the draft.

### The formal requirement, and why your table is ragged

The split is: total predictive variance = spread of the *mean* predictions across plausible versions
of the model (epistemic) + average of the *variances* those versions predict (aleatoric). Depeweg et
al. (ICML 2018) and Kendall & Gal (NeurIPS 2017).

You need both ingredients at once. Several draws of the model, and each draw emitting a variance as
well as a mean. Ensemble spread alone does not give you aleatoric — Heid et al. (JCIM 2023) state
that ensembling measures variance error and does not incorporate noise error.

Against your roster:

| Model | Can it split? | What is missing |
|---|---|---|
| GP | Both | Its aleatoric is one global number, not per-molecule |
| VBLL | Both, cleanly | Same — the noise term is global |
| QRF | **Both, and you already have it** | See below |
| NGBoost | Aleatoric only | One distribution per molecule, no model-draw axis |
| BNN | Epistemic only | A fixed noise term is not an aleatoric estimate |

So the ragged table is not a bug in your analysis. It is the honest consequence of the models you
chose. But two of the holes can be filled.

### What to change, in priority order

1. **Give the Bayesian networks a two-output head** — mean and log-variance, trained with the
   heteroscedastic loss. This is exactly what Scalia et al. (JCIM 2020) did: they bolted the same
   variance head onto MC-dropout, ensembles and bootstrap specifically so the aleatoric column would
   be comparable across methods. This one change fills the biggest hole in your table.
2. **Compute QRF's epistemic component from the trees you already have** — variance *across* trees of
   the per-tree leaf means is epistemic; average within-leaf variance is aleatoric. No retraining
   required. Published recipe, cited in the scratchpad file.
3. **NGBoost**: either bag it over seeds and take the variance of the means, or leave the cell blank
   with a footnote. Do not invent a surrogate — no paper in the search did.
4. **Flag that GP and VBLL aleatoric is a single global number**, so it cannot be rank-correlated
   per molecule. A reviewer will ask.
5. **Lay the table out the way Kendall & Gal do**: rows are noise level within strategy; columns are
   aleatoric / epistemic / total per model; em-dash plus footnote where a component is undefined.
   Then repeat the metric table once per component.

### Metrics to report alongside

The shared minimum across the four benchmark papers read in full: Spearman correlation of predicted
uncertainty against absolute error; a calibration curve summarised as **miscalibration area**; NLL;
and a sharpness or dispersion number.

That last one matters and you are currently missing it. Busk et al. (MLST 2022) point out that a model
predicting one constant uncertainty for everything scores perfectly on calibration while being
useless — you need the coefficient of variation of the predicted uncertainties to rule that out.

Miscalibration area is the direct regression replacement for the ECE that was removed. Your existing
coverage at 1σ and 2σ are literally two points on the curve it integrates, so it is a small addition,
not a new direction.

### One objection to prepare for

Heid et al. found that Gaussian, uniform, hyperbolic and bimodal noise at **matched standard
deviation** produced overlapping learning curves. Given item 0b, four of your six strategies differ
mainly in dose rather than shape, so expect this question directly. The defensible answer is the one
in item 8: foreground the strategies that are structurally different (outlier, quantile), and present
the rest as a dose axis.

---

## 8. Noise strategies: do you need all six?

You said the goal is not to prove the six strategies are distinct. Good, because on QM9 they mostly
are not, and the evidence for that is already in your own outputs.

**How similarly do the six strategies rank the 11 models?** Pairwise Spearman correlation of the
model rankings, from `deep_qm9_model_robustness_by_strategy.csv`:

|  | Gauss | Quant | Thresh | Hetero | ValProp | Outlier |
|---|---|---|---|---|---|---|
| **Gaussian** | 1.00 | 0.81 | 0.96 | 0.96 | 0.96 | 0.73 |
| **Quantile** | | 1.00 | 0.86 | 0.88 | 0.88 | 0.95 |
| **Threshold** | | | 1.00 | 0.98 | 0.98 | 0.79 |
| **Hetero** | | | | 1.00 | **1.00** | 0.83 |
| **Value-prop.** | | | | | 1.00 | 0.83 |
| **Outlier** | | | | | | 1.00 |

Mean off-diagonal 0.895; Kendall's W across all six is 0.912 (`table6_kendalls_w.txt`).

Two clusters, cleanly:

- **Spread-the-noise-everywhere**: Gaussian, threshold, heteroscedastic, value-proportional.
  Mutually 0.96–1.00. Heteroscedastic and value-proportional correlate at **exactly 1.00** — on QM9
  they are interchangeable.
- **Corrupt-a-subset**: quantile and outlier, correlating at 0.95 with each other and 0.73–0.88 with
  the first group.

**And that clustering matches the mechanics exactly.** Going back to the dose table in item 0b: the
first four strategies apply noise to every molecule, with a per-molecule spread of at most 2.2×
(threshold's is literally 1.0×, i.e. none at all). The last two apply large noise to a small subset
and almost nothing to the rest — spreads of 20× and 30×. The empirical clustering and the mechanistic
taxonomy are the same partition, which is a good sign that neither is an accident.

So the honest description is **six strategies, two mechanisms, differing mostly in how hard they
hit** (which is item 5). Heteroscedastic and value-proportional are smoothly label-dependent rather
than truly heteroscedastic in effect — the scaling factor varies about two-fold, which is not enough
to make them behave differently from plain Gaussian noise at the same dose.

**What I would cut, and why.** Move to **three in the main text**:

- **Gaussian** — the reference every prior paper uses, and the anchor for the comparison to
  Kolmar & Grulke.
- **Outlier** — the only genuinely different mechanism, and the only one where per-sample
  uncertainty shows any signal at all.
- **Threshold** — the harsh end of the dose range, which is what makes the model differences
  measurable in the first place.

Move heteroscedastic, value-proportional and quantile to the additional files, with the correlation
table above as the justification. That is a defensible cut, not a hidden one: you are showing all
six and arguing from the numbers that three carry the information.

**One caution before committing.** The strategies agree far less on the experimental datasets than
on QM9. Same calculation on the per-fold validation data, PDV representation:

| Dataset | Agreement on `auc_norm` | Agreement on R² at σ = 0.6 |
|---|---|---|
| LogD | 0.44 (min −0.07) | 0.88 (min 0.69) |
| Caco-2 | 0.56 (min 0.06) | 0.72 (min 0.53) |
| hERG | 0.80 (min 0.46) | 0.88 (min 0.82) |

Which brings up something more important than the cut itself, in item 9.

---

## 9. What each of the three headline numbers actually tells you

These are three different views of the same curve, not competitors. Nothing below is about picking
a winner — it is about what each one is measuring and where each one lies to you.

- **Clean R² (σ = 0)** — can this model and this representation learn the property at all. It is the
  precondition for the other two meaning anything.
- **R² at σ = 0.6** — what you actually get out of the model when the labels carry roughly one unit
  of real assay error. It is the number a chemist would care about.
- **`auc_norm`** — the shape of the degradation curve: what fraction of its own clean performance the
  model holds on to, averaged over the whole noise range. It deliberately removes the baseline so
  that models with different starting accuracy can be compared on decline alone.

That last property is the point of the metric and also its trap. Because each model is divided by its
own clean score, a model that starts weak has less to lose and scores well on decline.

**Evidence, from the per-fold validation data across all 72 dataset × representation × strategy cells:**

- The two metrics name the same best model in **20 of 72 cells (28 %)**.
- On LogD, LightGBM has the highest R² at σ = 0.6 in **17 of 24** cells and the highest retention in
  **none**. NGBoost is the exact mirror — 13 of 24 on retention, none on delivered accuracy — because
  its clean R² is 0.661 against LightGBM's 0.758.
- The clearest single case is the Gaussian process on hERG: clean R² of −0.156, worse than predicting
  the mean, and it still produces a retention number.

**And the two behave differently under a change of noise strategy.** Retention's model ranking moves
a lot between strategies; R² at σ = 0.6 barely moves:

| Dataset | Strategies agree on retention | on R² at σ = 0.6 |
|---|---|---|
| LogD | 0.44 (min −0.07) | 0.88 (min 0.69) |
| Caco-2 | 0.56 (min 0.06) | 0.72 (min 0.53) |
| hERG | 0.80 (min 0.46) | 0.88 (min 0.82) |

That is not "one metric is better". It is telling you something real: **delivered accuracy under
noise is largely a property of the model, while the shape of its decline is sensitive to which
mechanism the noise came from.** Retention is the more strategy-specific quantity — which makes it
the right instrument for research question 3 and the wrong instrument for a single ranking.

### The reporting rules that follow

- **Always print all three together in the same cell.** Kolmar & Grulke do exactly this in the same
  journal for the same kind of ratio, and it is why their metric survives scrutiny.
- **Refuse to compute retention where clean R² is degenerate.** A 0.3 gate already exists in the code;
  it is simply not visible in the printed tables, so a reader cannot tell a real 0.97 from a
  meaningless one.
- **Use retention where the question is about the noise mechanism**, and R² at σ = 0.6 where the
  question is about what a model delivers.

### Does the assessment σ need to change?

Two separate questions, and they have different answers.

**Is 0.6 the right point on the axis?** It has an independent justification that has nothing to do
with the data: published assay error is roughly 0.54 log units for hERG pKi, 0.27–0.62 for logD and
about 0.43 for Caco-2, so σ = 0.6 is approximately one unit of real experimental error on the
log-scale datasets. That is a defensible anchor and it should be stated as the reason.

**Should it be the same σ for every strategy?** This is the harder one, and the answer is probably
no — see item 0b. At σ = 0.6 the six strategies deliver RMS doses ranging from 0.30 (outlier) to 1.20
(threshold), a four-fold spread. Comparing them at equal σ compares different amounts of damage,
which is exactly what produces the residual pattern in item 5. The alternative is to report each
strategy at its own **equal-damage** σ. This is being analysed and the recommendation will land here.

*(Under analysis — the equal-damage σ values and whether the σ grid itself should change are being
computed now.)*

## 10. What has to be regenerated (step 2)

Grouped by what unblocks what.

### Before anything else — the noise-injection code

0. **Confirm and then fix the test/validation noise leak** (item 0a). Rebuild the Rust binary, re-run
   QM9. Everything below that touches QM9 numbers depends on this.
0b. **Decide what to do about the dead parameter file and the inert threshold cut-points** (item 0b).
   If the cut-points change, the threshold strategy has to be re-run.
0c. **Write out the true injected noise, the normalisation constants, the split, and a molecule
   identifier** alongside the uncertainty values, and **save uncertainty for training molecules**
   (item 4). This is the only way the paper's second research question becomes answerable.

### QM9 — figure script changes, then re-run

1. **`calculate_robustness` must also emit R² at σ = 0.6 and the drop from clean.** It currently
   emits only retention, clean R², and the σ count. This single change unblocks items 1, 3 and 9 on
   QM9, and it is the answer to "how do you not have this information".
2. **Emit the full R²-by-σ curve per model × representation × strategy**, not just the integral.
3. **Run the performance ANOVA at σ = 0.6 as well as σ = 0.3.** The σ value is currently hardcoded.
4. **Add a strategy-dose column** — how much each strategy actually perturbs a label per unit σ.
   No training needed; it is a property of the injection code.
5. **Rewrite the Wilcoxon table per item 3** — three quantities, per strategy, one representation.

### Validation — no re-run needed, just repoint

6. **Point the figure script at `validation_rerun` instead of `alternative_full`** and keep folds
   separate through the integration. Everything in item 2 then falls out locally.
7. **Decide the Gaussian process question** (item 2).

### Still genuinely blocked on the server

8. Per-sample uncertainty for the experimental datasets — none exists. See item 4.
9. Repeat training seeds for the neural models, if the Bayesian-vs-deterministic comparison is to
   survive. See item 3.

---

## 11. Where the figures land

Not decided yet — this is the next conversation, once items 4 and 7 are back. Current holdings are
8 figures and 6 tables; the comparable J Cheminform paper on the same topic carries 6 figures of
which 4 carry results.

The one new float worth arguing for: **six panels, one per noise strategy, clean R² on the horizontal
axis against R² at σ = 0.6 on the vertical, diagonal drawn.** Distance below the diagonal is the
damage. It makes item 9 self-evident without any argument, and it puts all three headline numbers on
one page.

---

## 12. Running list — threads that must survive into the revision guide

Anything logged here gets written into `REVISION_GUIDE.md` at step 3. Add to it; do not delete from it.

| # | Thread | Where it belongs in the paper | Status |
|---|---|---|---|
| T1 | Four Bayesian NN variants never train on Mol2vec/MHG-GNN (clean R² −0.02 to −0.19) — representation compatibility, not noise robustness | Methods, next to the exclusion table | ready to write |
| T2 | Catastrophic-run filter is not random w.r.t. the question; biases unstable configs' retention upward | Limitations | ready to write |
| T3 | Threshold noise is inert on QM9 — every label clears the ±1.0 cut, so it is Gaussian at double dose | Methods, noise strategies | ready to write |
| T4 | `noise_strategy_params.json` was never passed; paper's value-proportional factor of 0.1 is correct | Methods — no text change needed, but the earlier "code uses 0.05" note in REVISION_GUIDE.md must be deleted | ready to write |
| T5 | σ = 0.6 anchored to published assay error (hERG pKi 0.54, logD 0.27–0.62, Caco-2 0.43), not chosen post hoc | Methods, performance metrics | ready to write |
| T6 | The six strategies are two mechanisms at a range of doses; dose table + cluster correlations as evidence | Methods + Results | pending item 8 decision |
| T7 | Test/validation noise leak in the Rust pipeline — whether it needs a Methods correction depends on whether the deployed binary had it | Methods + Limitations | blocked on ARC check |
| T8 | Per-sample uncertainty detection cannot currently be claimed; population-level rise in uncertainty can | Results and discussion, uncertainty section | ready to write |
| T9 | SVM uses an RBF kernel throughout — paper's representation-specific-kernel claim is wrong in two places | Methods + Additional file 12 | ready to write |
| T10 | Retired metric NDS still appears throughout, including the Conclusion | Everywhere | ready to write |
