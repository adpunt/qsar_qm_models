# Immediate next steps

Standalone working document for the NoiseInject paper. Nothing here has been applied to `paper.tex`.

Every number is traced to the file it came from. Where a number does not exist anywhere, this says so
rather than guessing.

Last updated 2026-08-21.

**The argument the paper is making:** what, if anything, makes a QSAR model robust to noise — and can
a model's uncertainty tell you when your data is bad?

---

# A. Blockers in the noise-injection code

Both found by reading `rust/src/main.rs`. These are upstream of every QM9 number, so nothing else here
can be finalised until they are resolved.

## A1. QM9 held-out sets were given the training set's noise — fixed in code, not yet re-run

The noise map is keyed by training index, `0..train_count`. `preprocess_data` then called `write_data`
three times — train, validation, test — passing that same map each time, and `write_data` restarted
its counter at zero on every call. Because validation and test are smaller than train, every held-out
index found an entry. Test molecule 7 received the noise drawn for **training** molecule 7.

Two consequences:

1. **Held-out labels were corrupted**, contrary to the Methods. Every QM9 R² above σ = 0 was scored
   against a moving reference, mixing "the model got worse" with "the target moved".
2. **The corruption was attached to the wrong molecules**, so any correlation between a molecule's
   predicted uncertainty and "its" injected noise is zero by construction.

**Status: fixed.** `write_data` now takes an `apply_noise` flag, true only for the training call.
`cargo check` passes (exit 0; warnings are all pre-existing dead code).

**Two things this does not affect:**

- **Clean-data R² at σ = 0.** `process_and_train.py` sets `'noise': s > 0`, so at σ = 0 the Rust noise
  path is switched off entirely for all three splits.
- **The experimental datasets.** They run through the KIRBy pipeline, not this one. Verified directly:
  in `results/validation_full/openadmet_logd/QRF_PDV_uncertainty_values.csv` the true labels are
  bit-identical across all eleven noise levels.

**Confirm before spending compute.** The source had the bug; whether the *deployed* binary did is a
separate question. On ARC, open any QM9 `*_uncertainty_values.csv` and compare the clean-label column
between σ = 0 and σ = 1. Identical means only the analysis was affected. Different confirms the leak
and section B applies in full.

### A1b. The fix exposed a bigger problem — DECISION NEEDED

The fix itself was independently verified: the flag is in the right argument position at all three
call sites, validation and test are both excluded, and `cargo check` is clean.

But **four model families merge validation into training** — `y_train = np.hstack((y_train, y_val))`
in `models/models.py` (lines 1382, 1457, 1622, 1672, plus the `x_val_train` and `x_full` variants at
1501, 1506, 1732, 1737 and 3576). The split is 80/10/10, so those models train on 9,000 molecules.

- **Before the fix:** validation carried noise (mis-indexed, but noise), so all 9,000 training labels
  were noisy.
- **After the fix:** validation is clean, so **11 % of the training labels are now clean.**

Both states are wrong. **The correct design is train noisy, validation noisy with its own correctly
keyed noise map, test clean.** That also fixes early stopping, which currently selects on clean
labels — an oracle you would never have in reality.

This is a design decision about the experiment, so nothing has been changed. It has to be settled
before the re-run in section B.

### A1c. Two further consequences of the fix

- **RMSE and MAE are not comparable across noise levels.** They are reported in standardised units,
  and the normalisation standard deviation grows with injected noise, so the unit itself shrinks as σ
  rises. R² and Pearson r are unaffected. Either emit the raw mean and standard deviation from Rust so
  errors can be de-standardised, or report only R² and Pearson across σ.
- **The `injected_noise` column is now identically zero** for test rows, because the test labels are
  clean. Anything that reads it is dead — see D3.

## A2. The noise-strategy parameter file is never passed to the binary

`scripts/process_and_train.py` defines a `--strategy-params` option pointing at
`scripts/noise_strategy_params.json`, but does not include that flag in the subprocess call to the
Rust binary. Rust receives an empty parameter object and falls back to its built-in defaults on every
run.

**It was never passed on this branch.** Git history settles it: the flag appears in exactly one commit,
`f4b749d` on the abandoned `noise_strategy` branch, which is not an ancestor of HEAD. Its mainline twin
`fc06a18` (same parent, an hour later) added the argparse option but not the subprocess line. So the
JSON has been dead since the day it was added, and a single fixed parameter set was in force for every
run in `results/`. No SLURM script in any commit has ever passed it either.

Consequences:

- **The paper's value-proportional factor of 0.1 is correct** — that is the Rust default. The JSON's
  0.05 was never read. Commit `262423b` changed the *paper* from 0.05 to 0.1 to match the Rust default,
  which was the right direction.
- **Do not "fix" this by starting to pass the flag.** It would silently change value-proportional noise
  on every future run — factor 0.1 → 0.05, and base_sigma from σ to a pinned 0.1, which really would
  break the σ sweep — making new numbers incomparable with everything already computed. Delete the JSON
  and the dead argparse option, or rewrite the JSON so its values equal the Rust defaults and mark it
  documentation-only. Either way the Methods should quote the Rust defaults.

**The non-Gaussian noise distributions have bugs, but none of them ever ran.** The Rust code's U-shaped
sampler uses the wrong constant (noise comes out about 2.4× the nominal σ) and its left-tailed sampler
is a one-sided negative shift rather than a zero-mean skewed distribution. Neither matters: the default
is `gaussian`, and **no SLURM script in the repository passes `--distribution` at all**. Fix them if the
option is ever used; ignore them otherwise.

What the strategies actually do at noise level σ, computed from the real QM9 gap values (first 10,000
molecules of `data/QM9/raw/gdb9.sdf.csv` converted to eV; mean 7.00, SD 1.35, range 2.08–16.93):

| Strategy | Noise SD applied | RMS dose at σ=1 | Per-molecule spread |
|---|---|---|---|
| Gaussian | σ on every molecule | 1.000 | none |
| Threshold | 2σ on every molecule | 2.000 | none |
| Value-prop. | σ·(1 + 0.1·y) | 1.701 | 2.2× |
| Heteroscedastic | σ·√(0.1 + 0.05·y) | 0.669 | 2.2× |
| Quantile | 2σ on top and bottom deciles, 0.1σ on the rest | 0.899 | 20× |
| Outlier | 3σ where \|z\| > 2, 0.1σ elsewhere | 0.502 | 30× |

**Threshold noise is inert on QM9.** Its rule adds 2σ when the label exceeds 1.0, and the smallest
HOMO–LUMO gap in the data is 2.08 eV — so 100.00 % of molecules clear the cut and all receive the same
multiplier. On QM9, threshold noise is Gaussian noise at double strength.

This table underpins C3 and C5, and it is the reason D1 exists.

---

# B. What has to be re-run

Applies in full only if the ARC check in A1 confirms the leak reached the deployed binary.

## B1. QM9 — the main grid

Everything at σ > 0. Clean-data results at σ = 0 are valid and could be kept, but re-running all
eleven levels together is safer than splicing two runs and costs about 9 % more.

The job unit is one SLURM script per (model, strategy), each sweeping all representations, all eleven
σ levels and ten replicates. The ANOVA roster is 11 models × 5 representations × 6 strategies.

| Group | Scripts | Where |
|---|---|---|
| Core models × 6 strategies | ~66 | `slurm_scripts_anova/`, `slurm_scripts_missing/`, `slurm_scripts_mol2vec/` |
| Bayesian NN variants (BNN, VBLL, both backbones) | ~24 | `slurm_scripts_vbll/`, `slurm_scripts_full_vbll/`, `slurm_scripts_missing/` |
| Continuous-PDV grid | ~15 | `slurm_scripts_continuous_pdv/`, `slurm_scripts_cpdv_missing/` |
| GP (`gauche_rbf`) across representations | ~24 | `slurm_scripts_gauche_rbf/` |
| Uncertainty runs (the 7 probabilistic models) | ~14 | `slurm_scripts_uncertainty/`, `slurm_scripts_continuous_pdv/unc_*.sh` |

Everything writes to `../results/anova_{strategy}_{rep}_{model}.csv`, which the figure script globs.

**Archive the current results before overwriting.** They are the only record of what the paper claims
today, and the revision guide will need to reference them.

## B2. Things to add to that same re-run

Adding these now avoids a second full re-run later.

- **Per-sample uncertainty on TRAINING molecules.** Only test molecules are saved today. The paper's
  second research question is about detecting corrupted *training* labels, so without these rows the
  question cannot be asked at all. Single most valuable addition.
- **The true injected noise written out from Rust** — split, index, epsilon — instead of being
  reconstructed downstream by regression. See D3.
- **The normalisation mean and standard deviation**, the strategy name, the split name, and a real
  molecule identifier on every uncertainty row. The current `sample_idx` is a row position, so rows
  cannot be linked to molecules or matched across replicates.
- **A variance output head on the Bayesian networks**, if the aleatoric/epistemic table is to be
  filled. See D4. This changes the model, so it must be decided before the re-run starts.
- **Whatever σ grid and cut-points come out of D1.** Deciding those after the re-run means running
  twice.

## B2b. Two latent corruption risks to close before starting a long run

Both verified in `rust/src/main.rs`. Neither is known to have fired, but both are silent if they do,
and both are cheap to guard.

- **Truncated records in `write_data`.** Inside the ECFP4 block there are two `continue` statements —
  one when the fingerprint is not 2048 bits, one when the molecule fails to parse. Both sit *after* the
  earlier fields of that record have already been written and *before* the label is written. If either
  fires, the output file contains a short record, and **every molecule after it is read at the wrong
  offset.** Move the checks before the first write, or emit a zero-filled 256-byte block instead of
  skipping.
- **Index drift in `read_all_target_values`.** It pushes only successfully parsed records, but the
  noise map is keyed by loop position. A SMILES shorter than 5 characters or longer than 300 is
  rejected, so one rejection shifts every subsequent target value against its index. Push a placeholder
  so positions stay aligned.

Worth noting these could already explain some of the catastrophic runs being filtered out — a
misaligned record stream would produce exactly the wildly negative R² values seen in
`filtered_catastrophic_iterations.csv`. Not established, but cheap to rule out.

## B2c. NEW EXPERIMENT — skewed noise on a reduced grid

**Purpose.** All six current strategies draw from a symmetric Gaussian and differ only in *where* and
*how much* noise goes. Nothing in the study varies the *shape* of the error distribution. Heid et al.
(*JCIM* 2023) found Gaussian, uniform, hyperbolic and bimodal noise at matched standard deviation gave
overlapping learning curves, so the expected result is "shape does not matter" — which is a perfectly
good result to be able to state on QSAR data rather than borrow. It also answers the most likely
referee question about the Gaussian-only design, and it partly addresses the censoring gap in T20,
since right-censoring at an assay limit produces skewed error.

### Prerequisite — the existing skew samplers are broken and must be rewritten

`rust/src/main.rs` already has `LeftTailed`, `RightTailed` and `UShaped` distribution options, but they
cannot be used as they stand:

- **`LeftTailed` maps every draw to a non-positive value.** It is a one-sided downward shift, not
  zero-mean noise. Used as-is it would move the labels' mean, confounding shape with bias.
- **`UShaped` uses the wrong constant**, injecting about 2.45x the nominal amount.
- Both apply fractional powers to a signed sample, which is not a defined skewing transform.

**Replace them with a skew-normal draw, centred and scaled so the injected noise has zero mean and a
standard deviation of exactly sigma** — matching the D1 decision that the parameter is the achieved
noise SD. For shape parameter a, with d = a / sqrt(1 + a^2):

```
scale  w = sigma / sqrt(1 - 2*d^2/pi)
offset x = -w * d * sqrt(2/pi)
noise    = x + w * Z,   Z ~ standard skew-normal(a)
```

Suggested a = +4 and a = -4, giving a skewness of +/-0.78 while keeping mean 0 and SD sigma. Cite
Azzalini for the distribution. Note the skew-normal caps at a skewness of about +/-1; if a heavier tail
is wanted later, a centred lognormal is the next step, but that is a separate decision.

**Verify before launching:** draw 10^6 samples at sigma = 0.6 and confirm the sample mean is about 0
and the sample SD is about 0.6 for both shape values. If either fails, the experiment is confounded
and worthless.

### The reduced grid

Deliberately small. The question is narrow — does error shape change the conclusions — so it is asked
of the models the paper actually discusses, not the whole roster.

**Models (6).** Chosen to span the full robustness range and to cover each family the Results discuss:

| Model | Why it is in | QM9 retention |
|---|---|---|
| NGBoost | Highest retention, and a probabilistic model the paper leans on | 0.824 |
| LightGBM | Most accurate model; the retention-vs-accuracy contrast in C4 rests on it | 0.817 |
| SVM | Kernel family, mid-range | 0.814 |
| MLP-BNN-Full | Best of the Bayesian networks; carries the uncertainty story | 0.802 |
| DNN | Deterministic network, low robustness | 0.789 |
| MLP | Worst retention on QM9 — the bottom of the range | 0.756 |

**Representations (2).** `continuous_pdv`, the primary representation, plus `ecfp4`. Those two are the
extremes of the interaction result in C1 — model choice explains 78 % of accuracy variance within PDV
against 61 % within ECFP4 — so if shape interacts with representation at all, it shows up here.

**Placement strategies (2).** Gaussian and outlier. Gaussian is the reference; outlier is the only
strategy that behaves differently at matched dose (C5). Running both tests whether shape interacts
with placement, which one strategy alone cannot.

**Noise levels (4).** sigma = 0, 0.3, 0.6, 0.9. sigma = 0 is the shared control and is identical across
distributions, so it only needs running once per model x representation.

**Distributions (3).** Gaussian (control), skew-normal a = +4, skew-normal a = -4.

**Replicates: 10**, as elsewhere. Do not cut this — C6 shows run-to-run variation on the neural models
is large enough to swamp small effects, and a null result from an underpowered design is worthless.

### Cost

6 models x 2 new distributions (the Gaussian arm comes free from the main re-run) = **12 new SLURM
scripts**, each covering 2 representations x 3 noise levels x 2 placement strategies x 10 replicates.
Against roughly 145 scripts for the main QM9 re-run, this is under a tenth of the cost.

### What the result licenses

- **If shape does not matter** (the expected outcome): one sentence in Methods justifying the
  Gaussian-only design on this study's own data rather than on Heid et al.'s, which is strictly
  stronger. Report it in an additional file.
- **If shape does matter**: that is a genuinely novel finding — it would contradict Heid et al. on
  QSAR data — and it earns a main-text figure.

Either way, run it before the paper claims anything about error shape, and report the negative result
if that is what comes out.

## B3. Validation — no re-run needed, with two exceptions

The experimental datasets are clean and complete. `KIRBy/tests/results/validation_rerun/` already holds
48,510 rows: 13 models × 4 representations × 3 datasets × 6 strategies × 11 σ × 5 folds, folds
preserved. All the analysis in C can be done against it locally.

Two gaps that need server time only if you want them closed:

- **The Gaussian process was only ever run on PDV**, so it cannot enter a cross-representation ANOVA.
  Either run `val_gp_*` on ECFP4, MHG-GNN and SNS (9 scripts), or exclude it and say so.
- **Per-sample uncertainty exists for six files only** — LogD (QRF on all four representations, GP on
  PDV) and Caco-2 (QRF on ECFP4). Nothing for hERG. If the uncertainty claim is to extend beyond QM9,
  the probabilistic models need re-running on the experimental sets with uncertainty saving on.

## B4. What does not need re-running

- Clean-data (σ = 0) QM9 results.
- Anything on the experimental datasets already in `validation_rerun`.
- The conformal arm. `results/calibration_grid/` contains no intervals, coverage or widths, and
  nothing in the codebase reads that directory — there is nothing in it to analyse either way.

---

# C. What the results say

## C1. Accuracy is a property of the pairing, not of either factor

Source: `results/paper_figures_v2/table1_anova_summary.csv`. Roster 11 models × 5 representations,
performance at σ = 0.3.

| Strategy | Model | Rep | Interaction | Residual |
|---|---|---|---|---|
| Gaussian | 24.6 | 22.8 | **48.1** | 4.5 |
| Quantile | 25.1 | 22.4 | **46.9** | 5.6 |
| Threshold | 25.0 | 21.2 | **48.0** | 5.8 |
| Heteroscedastic | 27.9 | 23.5 | **46.2** | 2.5 |
| Value-prop. | 27.8 | 22.0 | **46.5** | 3.8 |
| Outlier | 25.3 | 22.0 | **49.2** | 3.5 |

Interaction is the largest term on all six strategies and the most stable number in the table. Model
and representation each take about a quarter and sit within a few points of each other. So neither
"which representation is best" nor "which model is best" has an answer on its own.

The simple-effects table (`table1_supp_simple_effects.csv`) gives the shape of that interaction, and
it runs both ways.

**How much your model choice matters depends on your representation** (model effect within each
representation, averaged over the six strategies):

| Representation | Model effect |
|---|---|
| Mol2vec | 98.9 % |
| MHG-GNN | 97.4 % |
| SMILES | 93.5 % |
| Continuous PDV | 78.3 % |
| ECFP4 | **61.3 %** |

**How much your representation choice matters depends on your model:**

| Model | Rep effect |
|---|---|
| DNN-BNN, MLP-BNN | 99.8, 99.7 % |
| MLP-VBLL, DNN-VBLL | 98.6, 98.2 % |
| NGBoost | 97.9 % |
| RF | 95.2 % |
| XGBoost, LightGBM | 91.2, 90.1 % |
| SVM | 74.8 % |
| MLP, DNN | **61.7, 59.4 %** |

In plain terms: the more constrained the model — Bayesian priors, boosting, a fixed kernel — the more
it depends on getting the representation right, while plain flexible models care less. And the better
the representation, the less the model choice matters.

## C2. Robustness is a property of the model alone

Same source, robustness measured on `auc_norm`.

| Strategy | Model | Rep | Interaction | Residual |
|---|---|---|---|---|
| Gaussian | 43.8 | 5.2 | 16.9 | 34.2 |
| Quantile | 36.8 | 4.4 | 15.1 | 43.7 |
| Threshold | 54.7 | 7.9 | 22.6 | 14.8 |
| Heteroscedastic | 14.0 | 0.7 | 8.0 | 77.4 |
| Value-prop. | 52.5 | 6.0 | 19.9 | 21.6 |
| Outlier | 10.3 | 0.2 | 5.9 | 83.6 |

Representation never exceeds 7.9 % and is under 1 % on two strategies. Of the variance model and
representation jointly explain, model takes 87–98 % on every single strategy — that ratio is the thing
that generalises across noise mechanisms.

**It holds on the experimental datasets too.** Recomputed from `validation_rerun` with folds as
replicates, GP excluded, SNS dropped to match the QM9 convention (`scratchpad/val_anova2.py`; per-fold
metrics in `scratchpad/val_fold_metrics.csv`):

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

Representation is 0.4–11.5 % everywhere — the same near-zero effect as QM9.

**Why the old validation ANOVA was degenerate.** `table_validation_anova.csv` shows residual η² = 0.0
on all three datasets. The loader averaged the five folds together at each noise level *before*
integrating the retention curve, and the ANOVA then filtered to Gaussian only — leaving exactly one
observation per cell, which forces the residual to zero arithmetically. Fix in E2.

## C3. The big ANOVA residuals are a dose effect, not a finding

Average retention across all 11 models (`deep_qm9_model_robustness_by_strategy.csv`), the best-minus-
worst model gap, and the ANOVA terms, lined up:

| Strategy | Retention (1.0 = no damage) | Best−worst gap | Model η² | Residual η² |
|---|---|---|---|---|
| Threshold | 0.587 | 0.180 | 54.7 | 14.8 |
| Value-prop. | 0.654 | 0.140 | 52.5 | 21.6 |
| Gaussian | 0.825 | 0.083 | 43.8 | 34.2 |
| Quantile | 0.846 | 0.069 | 36.8 | 43.7 |
| Heteroscedastic | 0.908 | 0.042 | 14.0 | 77.4 |
| Outlier | 0.944 | 0.034 | 10.3 | 83.6 |

Perfect rank agreement across all six strategies (Spearman −1.00 against both total explained variance
and the model term alone). Threshold noise spreads the models five times further apart than outlier
noise does.

**When noise barely hurts, every model looks the same, and run-to-run wobble is most of what is left
to explain.** The 83.6 % residual under outlier noise does not mean architecture stops mattering — it
means outlier noise at these σ values is too gentle to tell models apart. That is a sensitivity floor,
not a result. The ANOVA table should be printed with the retention column beside it so a reader can
see the six columns are not on a common scale.

## C4. What each of the three headline numbers measures

Three views of the same curve, not competitors.

- **Clean R² (σ = 0)** — can this model and representation learn the property at all. Precondition for
  the other two meaning anything.
- **R² at σ = 0.6** — what you actually get when labels carry roughly one unit of real assay error.
- **`auc_norm`** — the shape of the decline, each model divided by its own clean score so models with
  different starting accuracy can be compared on decline alone.

That last property is the point of the metric and also its trap: a model that starts weak has less to
lose. Across all 72 validation dataset × representation × strategy cells:

- The two metrics name the same best model in **20 of 72 cells (28 %)**.
- On LogD, LightGBM has the highest R² at σ = 0.6 in **17 of 24** cells and the highest retention in
  **none**. NGBoost is the mirror image — 13 of 24 on retention, none on accuracy — because its clean
  R² is 0.661 against LightGBM's 0.758.
- The Gaussian process on hERG has a clean R² of −0.156, worse than predicting the mean, and still
  produces a retention number.

**They also respond differently to a change of noise strategy:**

| Dataset | Strategies agree on retention | on R² at σ = 0.6 |
|---|---|---|
| LogD | 0.44 (min −0.07) | 0.88 (min 0.69) |
| Caco-2 | 0.56 (min 0.06) | 0.72 (min 0.53) |
| hERG | 0.80 (min 0.46) | 0.88 (min 0.82) |

Informative rather than a defect: delivered accuracy under noise is largely a property of the model,
while the shape of the decline is sensitive to which mechanism the noise came from. **Retention is the
strategy-sensitive instrument, which makes it the right tool for the generalisation question and the
wrong tool for a single ranking.**

Reporting rules that follow: print all three together in the same cell (Kolmar & Grulke do exactly this
in the same journal for the same kind of ratio); and make the existing 0.3 clean-R² gate visible in the
printed tables, so a reader can tell a real 0.97 from a meaningless one.

## C5. Six strategies, two mechanisms

Pairwise Spearman of the model rankings on QM9 (`deep_qm9_model_robustness_by_strategy.csv`):

|  | Gauss | Quant | Thresh | Hetero | ValProp | Outlier |
|---|---|---|---|---|---|---|
| **Gaussian** | 1.00 | 0.81 | 0.96 | 0.96 | 0.96 | 0.73 |
| **Quantile** | | 1.00 | 0.86 | 0.88 | 0.88 | 0.95 |
| **Threshold** | | | 1.00 | 0.98 | 0.98 | 0.79 |
| **Hetero** | | | | 1.00 | **1.00** | 0.83 |
| **Value-prop.** | | | | | 1.00 | 0.83 |
| **Outlier** | | | | | | 1.00 |

Mean off-diagonal 0.895; Kendall's W = 0.912 (`table6_kendalls_w.txt`). Two clusters: Gaussian,
threshold, heteroscedastic and value-proportional mutually at 0.96–1.00 (the last two at exactly
1.00), and quantile with outlier at 0.95.

**The clustering matches the mechanics** — from the dose table in A2, the first four apply noise to
every molecule with at most 2.2× per-molecule spread, the last two apply large noise to a small subset,
with 20× and 30× spread.

**But the clustering is a QM9 artefact and does not survive on the experimental data.** Recomputing the
same rank correlations from `validation_rerun` on mean R² across σ:

| Pair | QM9 | hERG | Caco-2 | LogD |
|---|---|---|---|---|
| Heteroscedastic – value-prop. | **1.000** | 0.929 | 0.918 | 0.951 |

On Caco-2 that pair is the **3rd lowest of all 15 pairs**, not the highest. And there is no cluster
structure at all on the experimental data — every one of the 15 pairs sits between 0.868 and 0.99 on
all three datasets. On delivered accuracy the six strategies agree with each other essentially
everywhere; it is only the *retention shape* that diverges (see C4).

**And at matched dose the two-mechanism story only half holds.** Comparing each strategy against
Gaussian noise at the same RMS dose rather than the same σ: outlier does 2.1–3.0× more damage than
dose-matched Gaussian, so concentration is real for outlier. Quantile does only 1.1–1.2× more — at
matched dose **quantile behaves like Gaussian, not like outlier.** So the "concentrated" mechanism has
one member, not two.

This is why the strategy cut in D2 does not go ahead as originally drafted.

## C6. Neural-network failures are a representation-compatibility fact

Two filters already run: any training run with R² below −0.5 is deleted, and any configuration with
clean R² below 0.3 is dropped from the robustness analysis. What they caught
(`filtered_catastrophic_iterations.csv`, 245 rows; `excluded_configs.csv`, 48 rows):

- 232 of the 245 deleted runs are MLP-VBLL (137) and DNN-VBLL (95), on Mol2vec (146) and MHG-GNN (97).
  Everything else combined is 13 runs.
- All 48 dropped configurations are the four Bayesian neural variants on Mol2vec and MHG-GNN, with
  clean R² between −0.02 and −0.19. **At zero noise they are no better than predicting the mean** —
  they never trained. That is not a robustness result.

The same pattern appears on the experimental data. Clean R² by model and representation on hERG:

| Model | ECFP4 | MHG-GNN | PDV | SNS |
|---|---|---|---|---|
| LightGBM | 0.507 | 0.496 | 0.509 | 0.533 |
| RF | 0.496 | 0.514 | 0.497 | 0.536 |
| DNN | 0.484 | 0.362 | **−389** | 0.416 |
| MLP | 0.461 | 0.153 | **−32.7** | 0.452 |
| VBLL-Full | 0.400 | −0.618 | **−204** | 0.431 |
| BNN-Full | 0.029 | 0.030 | **−544** | 0.103 |

Every neural model is fine on fingerprints and collapses on the continuous descriptor vector — on the
smallest dataset only. On LogD and Caco-2 the same networks are fine on PDV and it is MHG-GNN that
breaks them instead (−0.08 to −16.2).

**This is what the 46–49 % interaction term in C1 is made of** — not gentle pairing preferences, but
specific pairs collapsing entirely, with dense continuous features breaking neural networks when the
dataset is small. Note that PDV is the paper's primary representation.

---

# D. Open questions to settle

## D1. The noise-level parameter — DECIDED: report noise as an achieved standard deviation

**Decision: the noise level is defined as the realised standard deviation of the injected noise, in
label units, and each strategy's internal multipliers are normalised so that the parameter means the
same amount of corruption in all six.**

### Why, and what to cite

Kolmar & Grulke (*J Cheminform* 2021, **13**:92 — the direct predecessor, same journal, same
experiment) already parameterise noise by its standard deviation: their Eq. 7 is
Y_noise = Y + N(0, σ_noise), with σ_noise set per noise level. They also convert back to label units
when interpreting, writing that for BACE *"1.1 log units of noise were added, or 1.6 times the average
standard deviation reported in ChemBL"*. That is the precedent for both halves of this decision —
parameterising by SD, and reporting the achieved magnitude in units a chemist can read.

For matching different noise *mechanisms* at a common magnitude, the citable precedent is Heid et al.
(*JCIM* 2023, **63**:4012), who compared Gaussian, uniform, hyperbolic and bimodal noise where
*"each distribution had a standard deviation of 1 kcal/mol and was centered around 0 kcal/mol"*.

**State the extension honestly.** Heid matched four *homoscedastic distributions*. Threshold, quantile,
outlier, value-proportional and heteroscedastic are not homoscedastic, so matching their marginal
standard deviation is an extension of that practice, not a direct application of it. One sentence in
Methods saying so pre-empts the obvious referee question.

### What changes in practice

Currently one unit of the parameter delivers very different amounts of noise per strategy. Measured on
QM9 (first 10,000 molecules, gap in eV):

| Strategy | RMS noise per unit σ, as implemented |
|---|---|
| Threshold | 2.000 |
| Value-proportional | 1.701 |
| Gaussian | 1.000 |
| Quantile | 0.899 |
| Heteroscedastic | 0.669 |
| Outlier | 0.502 |

Normalising each strategy by its own factor makes the parameter mean one thing. Three consequences to
handle:

1. **The parameter is no longer called σ.** Across every paper checked, σ denotes a quantity in label
   units; when the tunable parameter is dimensionless it gets a different name — Kolmar & Grulke use
   an integer index `n`. Under this decision the parameter *is* in label units, so σ remains
   appropriate, but the Methods must define it as the realised noise SD rather than as a multiplier.
2. **Four of the six factors are dataset-dependent, not constants.** Threshold's factor is 2.000 only
   where every label clears its cut-point (true on QM9, false elsewhere — see D1b); heteroscedastic's
   depends on the mean absolute label; value-proportional's is affine in the label so it varies with σ
   itself. Only quantile and, approximately, outlier are structural. **The factors must be recomputed
   per dataset from the injection code, not carried over from the QM9 table above.**
3. **The σ = 0.6 anchor survives and gets stronger.** Published assay error is 0.54 log units for hERG
   pKi, 0.27–0.62 for logD and about 0.43 for Caco-2 (Kalliokoski et al., *PLoS ONE* 2013, **8**:e61007
   reports σ_pIC50 = 0.68). Once σ *is* the achieved noise SD, "σ = 0.6 is one unit of assay error"
   becomes literally true for every strategy rather than true only for Gaussian.

### What must not be claimed

σ = 0.6 is **not** where models separate best. Model discrimination — between-model spread over
between-fold spread — is flat across the whole grid (1.88 to 2.34, no peak). That is convenient rather
than a problem: 0.6 costs nothing in discriminating power, so the assay-error argument carries it
alone. Do not manufacture a data-driven justification the data does not support.

Also delete the claim at paper.tex L354 that *"the difficulty scaling controlled by σ is consistent"*
across strategies. Under the current code it is false; under this decision it becomes true, and the
sentence should be rewritten to say how it was made true.

### Other σ questions, settled

- **Do not truncate the grid.** Only about half the total damage has happened by σ = 0.6 and the
  curves are still steep above it. Report a second point at σ = 0.9 where robustness rather than
  baseline skill dominates the ordering.
- **Report damage per dataset, not pooled.** Mean relative drop at σ = 0.6 differs 3.4-fold: LogD
  0.155, hERG 0.356, Caco-2 0.529.
- **σ = 0 stays permanently** as a negative control.
- **Do not rescale the grid by each dataset's label standard deviation.** Assay error is absolute in
  log units and roughly constant across these three assays, so an absolute scale is the physically
  correct instrument. A label-SD rescaling is worth at most one diagnostic figure showing that the
  datasets differ in dose received rather than in kind.

## D1b. Threshold noise — CODE FIX REQUIRED

The paper and the code disagree, and the paper is right.

paper.tex L354 states that threshold noise applies the higher multiplier to samples with |y| > 1.0
**"(on normalized data)"**. The code applies the cut to the raw label, before standardisation.

| | Cut applied to | QM9 affected | LogD affected |
|---|---|---|---|
| What the paper says | standardised label, \|z\| > 1.0 | **33.8 %** | **33.2 %** |
| What the code does | raw label, \|y\| > 1.0 | **100.00 %** | **83.8 %** |

The paper's version is scale-free and selects a consistent third of molecules on both datasets, which
is what a well-specified strategy should do. The code's version depends entirely on where the label
happens to sit: 100 % on QM9 gaps in eV (minimum 2.08, so every molecule clears the cut), 83.8 % on
logD, and it would be far lower on Caco-2 efflux, whose labels have a standard deviation of 0.44.

**Fix: apply the threshold to the standardised label.** In `rust/src/main.rs`, the Threshold branch of
`generate_adaptive_noise` compares raw target values against ±1.0; it must compare
(y − mean) / sd instead, using the training mean and standard deviation. Do not re-specify the paper.

At the corrected specification, threshold corrupts 33.8 % of molecules against quantile's 19.2 % and
outlier's 2.7 % — three band widths on one axis, which is a coherent design and makes the strategy's
name meaningful.

**This also revises a claim in D2.** Threshold's reversal of ordering between datasets — least robust
on hERG and LogD, second mildest on Caco-2 — was previously read as evidence for the generalisation
question. It is not; it is this bug. One of the four arguments for keeping all six strategies is
therefore weaker than stated, though the other three stand.

**Check the same class of bug elsewhere.** Outlier uses a z-score and is scale-free. Quantile uses
deciles of the training targets and is scale-free. But value-proportional and heteroscedastic both
scale with the *raw* absolute label, so on QM9 — where every gap is around 7 eV — they carry a large
floor that would vanish on a centred variable. Whether that is intended needs deciding.

## D2. Whether to cut noise strategies — recommendation: keep all six

A cut was drafted and then withdrawn under adversarial checking. Four reasons it does not survive, all
verified against the data:

1. **The redundancy evidence is QM9-only.** Heteroscedastic and value-proportional correlate at exactly
   1.00 on QM9, which was the flagship argument — but on the experimental data they sit at 0.918–0.951,
   and on Caco-2 that is the 3rd *lowest* of all 15 pairs (C5). The pair is not interchangeable in
   general, and their magnitudes differ enormously anyway: QM9 retention 0.908 against 0.654.
2. **At matched dose the two clusters do not hold.** Quantile behaves like Gaussian (1.1–1.2× the
   damage of dose-matched Gaussian), not like outlier (2.1–3.0×). So the "concentrated" mechanism has
   one member, and no cut can give two implementations per mechanism.
3. **The redundancy criterion cannot be applied consistently.** Under "high rank correlation means
   redundant", the strategies that would be kept still include pairs correlating at 0.955–0.964 — above
   several pairs that would be cut. Two different criteria reaching a predetermined set of three or
   four is exactly what a reviewer reads as post hoc.
4. **The most discriminating strategy is the one that would be cut.** Threshold has the largest
   between-model spread of any strategy on QM9 (SD 0.0547 against outlier's 0.0100) and is the only one
   that reverses ordering between datasets — least robust on hERG and LogD, second *mildest* on Caco-2.
   That instability is not a disqualification; it is the answer to the generalisation question.

There is also a direct conflict with a result already in the paper: the ANOVA reports residual
dominance specifically under **outlier and heteroscedastic** (83.6 % / 77.4 %) against model dominance
under the other four. That grouping pairs outlier with heteroscedastic, contradicting any
outlier-plus-quantile cluster — and it only exists because all six were run.

**What to do instead.** The problem the cut was trying to solve is real — six near-identical panels
carry little extra information — but the fix is presentational, not a deletion:

- **State the redundancy as a finding rather than a reason to cut**: four of the six behave like
  Gaussian noise at a different dose, and only outlier reliably behaves differently at matched dose.
  That *is* the answer to research question 3.
- How to present six strategies without six near-identical panels is OPEN — not decided.

## D3. Whether the uncertainty claim can be rescued

As things stand the per-sample half of the paper's argument has no support.

- **The rows are test molecules.** Uncertainty is never saved for training molecules, so the question
  "does uncertainty flag the labels I corrupted" is being asked about molecules that were never meant
  to be corrupted — and which, because of A1, received someone else's noise.
- **The injected noise is reconstructed, not recorded.** The normalisation constants were never saved,
  so `fix_injected_noise` fits a line from clean label to noisy label and calls the leftovers noise. At
  σ = 0 no noise was injected, so the leftovers are floating-point rounding — whose size grows with the
  label, which is also where uncertainty is largest. That is why the zero-noise control shows a
  *higher* mean correlation (0.080) than σ = 0.6 does (0.057). The control fails, in the direction that
  manufactures a positive result.
- **For outlier and quantile noise**, a few very large errors drag the fitted line, leaving something
  close to "how extreme is this label" for the other 80–95 % of molecules. Those are precisely the two
  strategies that showed any signal.
- The reconstruction groups by model, representation, σ and replicate but **not by strategy**, pooling
  runs normalised on different scales.

Effect sizes, from `table4_supp_uncertainty_by_strategy_rep.csv`: pooled correlation averages 0.259,
dropping to 0.033 at σ = 0.3 and 0.047 at σ = 0.6 once conditioned on noise level. Outside outlier and
quantile the surviving values run 0.026–0.095, the same size as the σ = 0 artefact. In practical terms
a correlation of 0.05 means the most-uncertain tenth of molecules is 11.7 % bad labels instead of 10 %;
at 0.26 it is 20.3 %.

**What survives and is worth keeping:** uncertainty ranks a model's own errors, weakly but almost
everywhere — QRF 0.263, NGBoost 0.220, GP 0.176, the Bayesian nets 0.111–0.147
(`deep_qm9_uncertainty_by_model.csv`). Real and modest, and QRF being best is interesting because the
paper currently dismisses it. It is measured against the noisy label, so the "error" being ranked
contains the noise itself; recompute against the clean label.

**The population-level version of the question is answerable** and should be presented as what it is:
average uncertainty rises as noise is added. That is what the pooled 0.259 actually measures.

Conditioning at higher σ (0.8, 1.0) is worth adding **as a diagnostic, not a stronger test** — the
reconstruction artefact grows with injected noise, so a stronger correlation up there would be evidence
of the artefact rather than of detection.

## D4. How to do the aleatoric/epistemic split properly

**The framing in the current draft is backwards.** Ryu, Kwon & Kim (Chemical Science, 2019) ran
essentially this experiment — noise-free synthetic data, Gaussian noise at increasing levels — and got
aleatoric and total uncertainty rising while epistemic stayed flat, writing that *theoretically, the
epistemic uncertainty should not increase by the changes in the amount of data noise*. Yang & Li
(J Cheminform, 2023) report the same on QM9. So aleatoric rising with σ is the sanity check that the
decomposition works, not an anomaly — and it is stronger if the slope is shown to be roughly
one-for-one against injected σ.

**The epistemic component rising is the interesting result**, and there is a place to put it: Mucsányi
et al. (NeurIPS 2024) found aleatoric and epistemic estimates rank-correlate 0.8–0.999 across twelve
methods, so the entanglement is universal. Reporting the correlation between the two components in a
chemistry setting would be a genuine contribution.

**Why the table is ragged.** The split requires several draws of the model, each emitting a variance as
well as a mean (Depeweg et al., ICML 2018; Kendall & Gal, NeurIPS 2017). Ensemble spread alone does not
give aleatoric — Heid et al. (JCIM 2023) state that ensembling measures variance error and does not
incorporate noise error.

| Model | Can it split? | What is missing |
|---|---|---|
| GP | Both | Aleatoric is one global number, not per-molecule |
| VBLL | Both, cleanly | Same — the noise term is global |
| QRF | Both, from the fitted forest | Not currently computed |
| NGBoost | Aleatoric only | One distribution per molecule, no model-draw axis |
| BNN | Epistemic only | A fixed noise term is not an aleatoric estimate |

**The decomposition itself is two lines**, and it is identical in every implementation located and read
as source — Chemprop, Ryu et al., Scalia et al. and VBLL. Given `M` forward passes each returning a
mean and a variance:

```python
aleatoric = mean(predicted_variances, axis=0)   # average of the predicted variances
epistemic = var(predicted_means, axis=0)        # spread of the predicted means
total     = aleatoric + epistemic
```

Working code for all of it — the two-output layer, the heteroscedastic loss, the QRF split, and the
calibration metrics — is in `scratchpad/ALEA_EPIS_CODE.md`, with source URLs and line numbers. The
runnable QRF function is also saved standalone at `scratchpad/forest_ae.py`.

### The code to delete and replace

There are two decomposition functions in `scripts/utils.py`. One is correct, one is a stub.

- **`decompose_uncertainty_sampling_heteroscedastic` (line 88)** — correct. Epistemic is the spread of
  the predicted means, aleatoric is the average of the predicted variances. **Nothing calls it.**
- **`decompose_uncertainty_sampling` (line 62)** — a stub. It hardcodes `aleatoric = None` and sets
  `total = epistemic`. **This is the one the BNNs call.**

Swapping the call is not enough on its own: the stub exists because the network outputs one number per
molecule, so there is no predicted variance for the correct function to average. The network has to
change too.

**DELETE — but only together with the replacement below, never on its own:**

1. `decompose_uncertainty_sampling` in `scripts/utils.py`, lines 62–85. It cannot ever return an
   aleatoric component.
2. **All four of its call sites**, not just the broken one: `models/models.py` line 2121
   (`train_dnn_model`), line 2724 (`train_mlp_variant_model`), line 3133 (`train_gnn`), and the import
   at line 3077. Lines 2121 and 2724 are **live** — they are the non-VBLL Bayesian-network uncertainty
   path. Deleting the function without replacing those two calls breaks every BNN uncertainty run.
   Line 3133 passes one argument to a function that takes two and unpacks two values from a function
   that returns three, so that path raises on contact and has never run.

**REPLACE WITH:**

1. **A two-output network.** `DNNRegressionModel` (`models/models.py` ~line 1015) ends in
   `nn.Linear(hidden_size2, 1)`. Make it output 2 — a mean and a log-variance — and return both.
   `apply_bayesian_transformation` converts Linear layers to Bayesian ones and carries the wider
   output layer through unchanged.
2. **A Gaussian negative log-likelihood loss** in place of `nn.MSELoss()`. Use `nn.GaussianNLLLoss`,
   or lift the expression already sitting inside `VBLLLoss` (~line 1226), which computes exactly
   `0.5*log(var) + 0.5*(pred-target)^2/var`.
3. **Calls to `decompose_uncertainty_sampling_heteroscedastic`**, collecting both the means and the
   variances across the 100 Monte Carlo passes instead of the means alone.

**One trap:** when undoing target standardisation, scale the **variances**, not the standard
deviations. The reference implementations call this out explicitly because it is easy to get wrong.

**Actions, in priority order:**

1. Give the Bayesian networks a two-output head — mean and log-variance, trained with the
   heteroscedastic loss. Scalia et al. (JCIM 2020) bolted exactly this onto MC-dropout, ensembles and
   bootstrap specifically so the aleatoric column would be comparable across methods; their code is at
   `github.com/gscalia/chemprop`, branch `uncertainty`, and it is the closest match to this experiment
   because it already handles the target scaler. **One detail that is easy to get wrong: when undoing
   target standardisation, scale the variances, not the standard deviations.** Ryu et al.'s repo is
   `github.com/SeongokRyu/uq_molecule`. Fills the biggest hole, and **must be decided before the B1
   re-run starts.**
2. Compute QRF's epistemic component post-hoc from the trees already fitted — variance across trees of
   the per-tree leaf means is epistemic, average within-leaf variance is aleatoric. No retraining.
   **Two caveats from testing it:** the epistemic term barely moved between 200 and 2,000 training
   points, so treat it as an ordinal disagreement signal rather than a calibrated variance and do not
   quote coverage from it uncalibrated; and the aleatoric term is within-leaf variance, which is biased
   upward by whatever real signal remains inside the leaf. Say both in the paper.
3. For NGBoost, either bag over seeds and take the variance of the means, or leave the cell blank with
   a footnote. No paper found in the search invents a surrogate.
4. Flag that GP and VBLL aleatoric is a single global number and so cannot be rank-correlated per
   molecule. A reviewer will ask. **This may also explain an anomaly already on the books:** VBLL's
   noise parameter appears to be one scalar per output in the upstream library, meaning VBLL cannot
   represent per-molecule noise at all — which would account for its coverage at 1σ sitting at 0.27–0.45
   against a target of 0.68. Verify against the actual instantiation in this repo before writing it
   down, then it becomes an explanation rather than an unexplained oddity.
5. Lay the table out the way Kendall & Gal do: rows are σ within strategy, columns are aleatoric /
   epistemic / total per model, em-dash plus footnote where undefined; then repeat the metric table once
   per component.

Prefer installing `uncertainty-toolbox` to reimplementing the calibration metrics. If you do
reimplement, note that sharpness is `sqrt(mean(sigma**2))`, not `mean(sigma)`.

**Metrics to report alongside** — the shared minimum across the benchmark papers read: Spearman of
predicted uncertainty against absolute error; a calibration curve summarised as **miscalibration area**;
NLL; and a **sharpness or dispersion** number. That last one is currently missing and matters — Busk et
al. (MLST 2022) point out that a model predicting one constant uncertainty scores perfectly on
calibration while being useless. Miscalibration area is the direct regression replacement for the
removed ECE, and the existing coverage at 1σ and 2σ are two points on the curve it integrates.

**One objection to prepare for:** Heid et al. found Gaussian, uniform, hyperbolic and bimodal noise at
matched standard deviation gave overlapping learning curves. Given A2, four of the six strategies differ
mainly in dose rather than shape, so expect this question directly.

---

## D5. Dead code — what is verified safe to delete, and what is not

A delete list was drafted and then checked symbol by symbol against the whole repository. **Most of it
was wrong and would have broken the pipeline.** Recorded here so the same mistakes are not repeated.

**Verified dead — safe to remove.** All in `rust/src/main.rs`, confirmed by repo-wide grep plus
compiler dead-code warnings; the crate has a single source file and no library target, so there are no
external consumers:

`DELIMITER` (line 27), `PlotPoint` (line 79), `tanimoto_distance` (line 866), `mean_absolute_error`
(885), `mean_squared_error` (891), `root_mean_squared_error` (897), `r2_score` (901). The last two go
together — `root_mean_squared_error` is the only caller of `mean_squared_error`.

**NOT dead — do not delete:**

| Item | Why it must stay |
|---|---|
| `generate_noise_by_indices` (rust, line 193) | The **only** noise generator for the Gaussian strategy, which is the primary strategy and appears in 194 SLURM invocations |
| `sample_from_distribution` (rust, line 445) | Used by every non-Gaussian strategy. Deleting it removes all value-proportional, quantile, threshold, outlier and heteroscedastic noise |
| The U-shaped / left-tailed / right-tailed samplers | Genuinely buggy (U-shaped injects √6 ≈ 2.45× the nominal amount; left-tailed maps every draw to a negative value, so it is a downward shift rather than zero-mean noise) but reachable from the command line and referenced by older experiment scripts. **Fix, do not delete** |
| `ScaffoldBased` and `load_scaffold_assignments` | Reachable via `--noise_strategy scaffold`. It is semantically broken — it loads the scaffold map, discards it, and applies the training σ to everything — so this is a fix-or-explicitly-retire decision, not a dead-code deletion. No current SLURM script uses it |
| The `morgan` buffer in the Rust record | Part of the memory-map record layout and reachable via `-r morgan`. The real defect here is different: the Python writer never emits a Morgan block and the Python reader never expects one, so `-r morgan` would desync the Rust reader by 256 bytes per record. Diagnose that before touching anything |
| `scripts/noise_strategy_params.json` | Never read (A2), but it is the only surviving record of the intended parameters and the Methods need them quoted. Keep it, or rewrite it to match the Rust defaults and mark it documentation-only. The dead `--strategy-params` argparse option at `process_and_train.py` line 260 **is** safe to delete |

**Incomplete elsewhere:** removing the `injected_noise` column means also removing `fix_injected_noise`
in `generate_paper_figures_v2.py` (lines 980–1020) and its call at line 3997, not just the three read
sites — otherwise a function is left regressing on a column that no longer exists.

# E. Changes to the analysis code

`scripts/generate_paper_figures_v2.py` unless stated otherwise.

## E1. QM9

1. **`calculate_robustness` must also emit R² at σ = 0.6 and the drop from clean.** It currently emits
   only retention, clean R² and the σ count, which is why **R² at σ = 0.6 does not exist for QM9 in any
   file** — confirmed by reading every CSV in `results/paper_figures_v2/`. The only QM9 clean-data R²
   anywhere is `table3_probabilistic_comparison.csv`, covering 8 models on PDV under Gaussian noise.
2. **Emit the full R²-by-σ curve** per model × representation × strategy, not just the integral.
3. **Run the performance ANOVA at σ = 0.6 as well as σ = 0.3.** The σ value is currently hardcoded.
4. **Add a strategy-dose column** — how much each strategy perturbs a label per unit σ. No training
   needed; it is a property of the injection code and the values are in A2.

## E2. Validation

5. **Point the script at `validation_rerun` instead of `alternative_full`** — 13 models rather than 7,
   four representations, folds preserved.
6. **Keep folds separate through the integration.** Do not average before integrating; treat folds as
   replicates the way QM9 treats iterations.
7. **Run the validation ANOVA per strategy**, matching QM9, rather than Gaussian only.
8. **Decide the Gaussian process question** (B3) and make the decision visible in the output.

## E3. Statistical tests

9. **Rebuild the Wilcoxon tables.** `table3_wilcoxon_tests.csv` currently reports one comparison per
   model pair, on retention only, pooled across four representations and six strategies at once — while
   sitting in a section about a single representation. The function accepts a representation argument
   and a strategy argument and ignores both. It should instead report **the change in all three headline
   quantities — clean R², R² at σ = 0.6, retention — per strategy**, paired across replicates within a
   strategy, at one stated representation. Six rows per pair instead of one.

   The printed values in the paper do not match this file, and one verdict flips: DNN → DNN-VBLL is
   p = 0.25, i.e. not significant, so "both transformations improved both networks" is three of four.

10. **Recompute uncertainty–error correlations against the clean label**, not the noisy one (D3).
11. **Compute the conditioned uncertainty correlation inside a single replicate**, not pooled across
    replicates.

---

# F. Figures and tables

Current holdings are 8 figures and 6 tables. The comparable J Cheminform paper on the same topic
(Kolmar & Grulke) carries 6 figures of which 4 carry results.

Known redundancies: the ANOVA figure and table show identical numbers; the overview figure's second
panel duplicates the ranking table; three heatmaps carry the same message; and the combined validation
figure duplicates the overview figure with strategies averaged away — it is the only float that averages
across strategies, and it shows an incomplete model roster.

The one new float worth arguing for: **six panels, one per strategy, clean R² on the horizontal axis
against R² at σ = 0.6 on the vertical, with the diagonal drawn.** Distance below the diagonal is the
damage. It makes C4 self-evident without argument and puts all three headline numbers on one page.

Final selection is deferred until D1 and D2 are settled.

---

# G. Threads for the revision guide

Written into `REVISION_GUIDE.md` at step 3. Add to this list; do not delete from it.

| # | Thread | Where it belongs | Status |
|---|---|---|---|
| T1 | Four Bayesian NN variants never train on Mol2vec/MHG-GNN (clean R² −0.02 to −0.19) — representation compatibility, not noise robustness | Methods, next to the exclusion table | ready |
| T2 | The catastrophic-run filter is not random with respect to the question; it biases unstable configurations' retention upward | Limitations | ready |
| T3 | Threshold noise is inert on QM9 — every label clears the ±1.0 cut, so it is Gaussian at double dose | Methods, noise strategies | ready |
| T4 | `noise_strategy_params.json` was never passed; the paper's value-proportional factor of 0.1 is correct. The "code uses 0.05" note in `REVISION_GUIDE.md` must be deleted | Methods | ready |
| T5 | σ = 0.6 is anchored to published assay error (hERG pKi 0.54, logD 0.27–0.62, Caco-2 0.43), not chosen post hoc | Methods, performance metrics | ready |
| T6 | Four of the six strategies behave like Gaussian noise at a different dose; only outlier differs at matched dose. Report as a finding, keep all six | Methods and Results | ready |
| T14 | σ = 0.6 is the halfway point of the damage range (51 % of the σ = 1.0 drop) — but is NOT a discrimination optimum; do not claim it is | Methods, performance metrics | ready |
| **T17** | **σ is redefined as the realised standard deviation of the injected noise, with each strategy normalised so the parameter means the same corruption in all six. Cite Kolmar & Grulke 2021 (*J Cheminform* 13:92, Eq. 7) for SD-based parameterisation and for reporting achieved noise in label units; cite Heid et al. 2023 (*JCIM* 63:4012) for matching mechanisms at equal SD. State explicitly that extending equal-SD matching to non-homoscedastic mechanisms is our extension.** | **Methods, noise strategies — MUST appear in the paper** | **ready** |
| T18 | Threshold noise applies its cut to the raw label; the paper says normalised. Code fix, not a paper fix. Once fixed, threshold corrupts 33.8 % of molecules vs quantile 19.2 % and outlier 2.7 % | Methods — no text change if the code is fixed to match | ready |
| T19 | Delete the paper.tex L354 claim that σ scaling is consistent across strategies; replace with how it was made consistent (T17) | Methods, noise strategies | ready |
| T20 | All six strategies are zero-mean random perturbations, so censoring at assay limits and constant inter-laboratory offsets are not represented | Limitations | ready |
| T21 | Do not present value-proportional and heteroscedastic as models of real assay error — Kalliokoski et al. found error does not depend on the measured value. They are dynamic-range proxies | Methods, noise strategies | ready |
| T22 | Justify the Gaussian-only design: Heid et al. found Gaussian, uniform, hyperbolic and bimodal noise at matched SD gave overlapping learning curves — and, once B2c has run, on this study's own data too | Methods, noise strategies | pending B2c |
| T24 | Skewed-noise experiment (B2c): report the result whichever way it comes out, and cite Azzalini for the skew-normal | Methods + additional file, or main text if shape matters | pending B2c |
| T23 | Validation labels are merged into training for four model families, so the design decision in A1b changes what the Methods must say about which splits carry noise | Methods, NoiseInject framework | blocked on A1b decision |
| T15 | Report damage per dataset, not pooled — relative drop at σ = 0.6 is 0.155 (LogD), 0.356 (hERG), 0.529 (Caco-2) | Results | ready |
| T16 | Value-proportional noise varies per molecule by 2.2×, not the "under 10 %" figure in earlier drafts | Methods | ready |
| T7 | The held-out noise leak — whether a Methods correction is needed depends on whether the deployed binary had it | Methods and Limitations | blocked on ARC check |
| T8 | Per-sample uncertainty detection cannot currently be claimed; the population-level rise in uncertainty can | Results and discussion | ready |
| T9 | SVM uses an RBF kernel throughout — the representation-specific-kernel claim is wrong in two places | Methods and Additional file 12 | ready |
| T10 | The retired metric NDS still appears throughout, including in the Conclusion, where it is defined as *the* robustness metric | Everywhere | ready |
| T11 | ECE is defined and tabulated in the paper but has been deleted from the code, so those cells are unfillable | Metrics section and Table 7 | ready |
| T12 | PDV is described as both the most and the least noise-robust representation in the same section | Results | ready |
| T13 | The exclusion threshold is stated as R² < 0.3 in four places and 0.6 in two; the live gate is 0.3 | Methods and Additional files | ready |

---

# H. Master thread register

Added 2026-08-21. **Purpose: one place where every open thread lives, so nothing is dropped.**

## H0. Read this first — the numbering collision

There are **three independent `T`-registers** in this repo and their numbers mean different things.
Quoting a bare "T17" is ambiguous and has already caused confusion.

| Register | Lives in | Range | What its threads are about |
|---|---|---|---|
| `INS-T*` | `immediate_next_steps.md` §G | T1–T24 | threads to write into the revision guide at step 3 |
| `RG-T*` | `REVISION_GUIDE.md` "OPEN THREADS TRACKER" | T1–T25 | paper-text find/replace threads |
| `RS-T*` | `REVISION_STATUS.md` §4 | T3–T23 | admin/status threads (**T20 appears twice in that table**) |

Collisions, for the record:

- **T17** — `INS-T17` = redefine σ as the achieved noise SD. `RS-T17` = refit the saturated validation ANOVA. Unrelated.
- **T14** — `INS-T14` = σ = 0.6 is the halfway damage point. `RG-T14`/`RS-T14` = value-proportional corrupted in the direct dump. Unrelated.
- **T20** — `INS-T20` = censoring/offset noise not represented. `RG-T20`/`RS-T20` = VBLL "broken". `RS-T20` also appears a *second* time in the same table for the VBLL root cause. Unrelated.
- **T24** — `INS-T24` = skewed-noise experiment. `RG-T24` = ECE removal. Unrelated.

**Convention from now on: always write the prefix.** `INS-T17`, not `T17`.

## H1. The register

Status values: ⬜ not started · 🔶 in progress · ⛔ blocked (says on what) · ✅ done · ⏸ parked (says why).
"Gate" = the earliest step at which it must be resolved: **CODE** (before compute), **RUN** (during the re-run),
**ANALYSIS** (figure-script work on existing data), **PAPER** (author text edit, no compute).

_(Rows are filled in as each thread is verified against code. A row with no verification note has NOT been
re-checked in this pass and its claim is inherited from an earlier session.)_

### H1a. Code — must be settled before compute is spent

| ID | Thread | Gate | Status |
|---|---|---|---|
| A1 | Held-out splits received the training noise map | CODE | ✅ fixed in working tree (`rust/src/main.rs`, `apply_noise` flag, 3 call sites) — **uncommitted** |
| A1-verify | Confirm the *deployed* ARC binary had the bug before re-running everything | CODE | ⛔ needs ARC check |
| A1b | Four model families merge validation into training ⇒ after the A1 fix, 11 % of training labels are clean. Decide the intended design | CODE | ⬜ **DECISION NEEDED** |
| A1c-1 | RMSE/MAE are in standardised units whose scale moves with σ ⇒ not comparable across σ | CODE or ANALYSIS | ⬜ |
| A1c-2 | `injected_noise` column is now identically zero on test rows | ANALYSIS | ⬜ |
| A2 | `--strategy-params` never passed; Rust defaults were always in force | CODE | ⬜ decide delete-vs-document |
| B2b-1 | `write_data` ECFP4 `continue` statements can emit a short record and desync every later molecule | CODE | ⬜ |
| B2b-2 | `read_all_target_values` index drift vs the noise map key | CODE | ⬜ |
| D1 | σ redefined as the achieved noise SD, each strategy renormalised (= `INS-T17`) | CODE | ✅ decided, **not implemented** |
| D1b | Threshold cut applies to the raw label; paper says standardised | CODE | ✅ decided (fix code), **not implemented** |
| D1b-2 | Value-proportional and heteroscedastic also scale with the raw label — intended? | CODE | ⬜ |
| D4-head | Two-output head + Gaussian NLL on the Bayesian nets, needed for any aleatoric/epistemic split | CODE | ⬜ **DECISION NEEDED** |
| B2-unc | Save per-sample uncertainty for TRAINING molecules | CODE | ⬜ **DECISION NEEDED** |
| B2-prov | Emit true injected epsilon, normalisation mean/sd, molecule id, split name, strategy name | CODE | ⬜ |
| B2c | Skewed-noise experiment; requires rewriting the skew samplers as centred skew-normal first | CODE | ⬜ **DECISION NEEDED** |
| D5-dead | Verified-dead Rust symbols safe to delete; the rest are NOT dead | CODE | ⬜ low priority |
| D5-morgan | `-r morgan` desyncs the Rust reader by 256 bytes/record | CODE | ⬜ |
| D5-scaffold | `ScaffoldBased` strategy loads the scaffold map then discards it | CODE | ⬜ fix-or-retire |
| GP-rerun | RBF-GP across all representations so GP can enter the ANOVA | CODE/RUN | 🔶 26 jobs submitted 2026-08-19 (12822669–12822694), outcome unverified |

### H1b. Run — scope of the re-run itself

| ID | Thread | Gate | Status |
|---|---|---|---|
| B1 | QM9 main grid, all σ > 0 | RUN | ⛔ on the H1a decisions |
| B1-archive | Archive current `results/` before overwriting | RUN | ⬜ |
| B3-gp | GP only ever run on PDV for validation | RUN | ⬜ include-or-exclude |
| B3-unc | Per-sample uncertainty exists for 6 validation files only; nothing for hERG | RUN | ⬜ |
| B4 | What does NOT need re-running (σ = 0 QM9, validation, the empty conformal arm) | RUN | ✅ recorded |
| R2 | One ARC figure regen after the analysis edits land | RUN | ⬜ |

### H1c. Analysis — figure-script work on data that already exists

| ID | Thread | Gate | Status |
|---|---|---|---|
| E1-1 | `calculate_robustness` must emit R² at the reporting σ and the drop from clean | ANALYSIS | ⬜ |
| E1-2 | Emit the full R²-vs-σ curve per model × rep × strategy | ANALYSIS | ⬜ |
| E1-3 | Run the performance ANOVA at more than the hardcoded σ | ANALYSIS | ⬜ |
| E1-4 | Add a per-strategy dose column | ANALYSIS | ⬜ |
| E2-5 | Point the script at the correct validation directory | ANALYSIS | ⬜ **DECISION NEEDED** (`alternative_full` 7 models vs `validation_rerun` 13) |
| E2-6 | Keep folds separate through the integration | ANALYSIS | ⬜ |
| E2-7 | Validation ANOVA per strategy, not Gaussian only (= `RS-T17`) | ANALYSIS | ⬜ |
| E3-9 | Rebuild the Wilcoxon tables per strategy at one stated representation | ANALYSIS | ⬜ |
| E3-10 | Uncertainty–error correlation against the clean label | ANALYSIS | ⬜ |
| E3-11 | Conditioned uncertainty correlation inside a single replicate | ANALYSIS | ⬜ |
| D2-present | Six strategies without six near-identical panels | ANALYSIS | ⬜ **OPEN** |
| D3-unc | Whether the per-sample uncertainty claim can be rescued | ANALYSIS/CODE | 🔶 |
| DT-D1 | `fig_validation_combined` averages away strategies and datasets | ANALYSIS | 🔶 |
| DT-D2 | Latent averaged tables | ANALYSIS | ✅ done + implemented |
| DT-D3 | `table_supp_icc` averages across strategies before the ICC | ANALYSIS | ⬜ |
| DT-D4 | `table2_*_pdv` MEAN/STD/Mean_Rank across strategies — confirm keep | ANALYSIS | ⬜ |
| DT-D5 | Add a QM9 baseline-R² dump | ANALYSIS | ⬜ |
| DT-D6 | Add per-strategy baseline↔robustness tables | ANALYSIS | ⬜ |
| DT-D7 | Add a representation × strategy table | ANALYSIS | ⬜ |
| DT-D8 | Fold `deep_analysis.py` into the main script | ANALYSIS | ⬜ |
| DT-D10 | Self-guard `calculate_robustness` against catastrophic iterations | ANALYSIS | ⬜ |
| DT-D11 | Retention vs delivered accuracy | ANALYSIS | ✅ metric decided (σ = 0.6, `auc_norm` stays); figure work specced in guide §12, **not implemented** |
| DT-D11b | fig3 y-axis: keep the zoom or go to [0, 1] | ANALYSIS | ⬜ **DECISION NEEDED** |
| R1 | Run the script so `table_validation_uncertainty.csv` is actually emitted | ANALYSIS | ⬜ |
| v1-retire | `generate_paper_figures.py` + `run_figures.sh` are the dead NDS pair and still tracked | ANALYSIS | ⬜ **DECISION NEEDED** |
| F-floats | Final figure/table selection | ANALYSIS | ⛔ on the takeaway plan |

### H1d. Paper — author text edits, no compute

Carried verbatim from `immediate_next_steps.md` §G (`INS-T*`), `REVISION_GUIDE.md` (`RG-T*`) and
`REVISION_STATUS.md` §4 (`RS-T*`). These are step-3/step-4 work and are listed so they are not lost,
not because they are actionable now.

| ID | Thread | Status |
|---|---|---|
| INS-T1 | Four Bayesian NN variants never train on Mol2vec/MHG-GNN | ready |
| INS-T2 | The catastrophic-run filter biases unstable configurations upward | ready |
| INS-T3 | Threshold noise is inert on QM9 | ready |
| INS-T4 | `noise_strategy_params.json` was never passed; the 0.1 factor is correct | ready |
| INS-T5 | σ = 0.6 anchored to published assay error | ready |
| INS-T6 | Four of six strategies behave like Gaussian at a different dose | ready |
| INS-T7 | The held-out noise leak — Methods/Limitations wording | ⛔ ARC check |
| INS-T8 | Per-sample detection cannot be claimed; the population-level rise can | ready |
| INS-T9 | SVM is RBF throughout — the kernel claim is wrong in two places | ready |
| INS-T10 | NDS still appears throughout, including in the Conclusion | ready |
| INS-T11 | ECE is defined and tabulated in the paper but deleted from the code | ready |
| INS-T12 | PDV described as both most and least noise-robust | ready |
| INS-T13 | Exclusion threshold stated as 0.3 in four places and 0.6 in two | ready |
| INS-T14 | σ = 0.6 is the halfway damage point but NOT a discrimination optimum | ready |
| INS-T15 | Report damage per dataset, not pooled | ready |
| INS-T16 | Value-proportional varies per molecule by 2.2× | ready |
| INS-T17 | σ redefined as the achieved noise SD (must appear in the paper) | ready |
| INS-T18 | Threshold cut is raw-label in code, standardised in the paper | ready |
| INS-T19 | Delete the "σ scaling is consistent" claim | ready |
| INS-T20 | Censoring and inter-laboratory offsets are not represented | ready |
| INS-T21 | Do not present valprop/hetero as models of real assay error | ready |
| INS-T22 | Justify the Gaussian-only design | ⛔ B2c |
| INS-T23 | Which splits carry noise — Methods wording | ⛔ A1b |
| INS-T24 | Skewed-noise experiment, report whichever way it comes out | ⛔ B2c |
| RG-T3 / RS-T3 | ANOVA table body in paper.tex is stale | logged |
| RG-T5,T7 / RS-T5,T7 | PDV self-contradiction and the representation claims | decide |
| RG-T6 / RS-T6 | "Noise types differ" as an organising theme | decide |
| RG-T8 / RS-T8 | Cross-strategy averaging audit | decide |
| RG-T9 / RS-T9 | Kendall W 0.92 → 0.9121; SVM is 5th not top-2 | logged |
| RG-T10 / RS-T10 | 11 stale NDS lines | partial |
| RG-T12 / RS-T12 | Delete the duplicated sentence at L381 | logged |
| RG-T13 | "Robust despite mediocre clean-data accuracy" | ⚠ conflicts with §9.1 — see H2 |
| RG-T4 / RS-T4 | The "decoupling" claim | ⚠ conflicts with §9.1 — see H2 |
| RG-T25 | σ anchored to real experimental error | logged |
| RS-T16 | Uncertainty section reframe | decide |
| RS-T19 | Which figures to regenerate | open |
| RS-T22 | Validation coverage gaps | decide |
| RS-T23 | QRF robustness asymmetry on QM9 | ⛔ server |
| RS-8F1..10 | The ten author decisions in `REVISION_STATUS.md` §8F | see that section |
| RG-9.x | Guide §9 pending-reconciliation items (F4 SVM, F9 cross-dataset agreement) | ⚠ do not write |

## H2. Known internal contradictions — must be resolved, not averaged over

| # | The contradiction | Where |
|---|---|---|
| X1 | `RG-T4`/`RG-T13` say KEEP the "robustness is decoupled from accuracy" claim and cite NGBoost as evidence. Guide §9.1 says that claim is an artefact of the metric dividing out the baseline, and that citing NGBoost for it is citing the artefact. **Both are written into the guide.** | `REVISION_GUIDE.md` L62–134 vs §9.1 |
| X2 | `INS-T3` says threshold noise is inert on QM9 (a finding). `D1b` says that is a code bug to be fixed. If the code is fixed, `INS-T3` describes the *old* runs only. | `immediate_next_steps.md` §A2 vs §D1b |
| X3 | Which validation directory is canonical. `DISCUSSION_TRACKER` asserts `validation_rerun` in one place and `alternative_full` in another, each with evidence. | `DISCUSSION_TRACKER.md` |
| X4 | `RS-T20` says VBLL is fine (a scale artefact). The ECE block says low coverage despite wide intervals is NOT only a units artefact. | `REVISION_STATUS.md` §4 vs `DISCUSSION_TRACKER.md` |


## H3. Verification log — this session (2026-08-21)

Every entry below was read from code in this session. Nothing inherited.

| # | Checked | Finding |
|---|---|---|
| V1 | `git diff rust/src/main.rs` | The A1 fix is real and **uncommitted**: `apply_noise: bool` added to `write_data`, passed `true` / `false` / `false` at the train / validation / test call sites in `preprocess_data`. 12 insertions, 2 deletions. |
| V2 | `paper.tex` L188 | The three research questions are stated in a single paragraph, not as a numbered list. **RQ1** representation vs architecture, on both performance and robustness. **RQ2** probabilistic vs deterministic robustness, *and* whether per-sample uncertainty tracks which labels were corrupted once the population-level rise is controlled for. **RQ3** whether robustness patterns generalise across noise mechanisms and across properties. Plus the NoiseInject package. |
| V3 | `paper.tex` structure | Sections are `Results` then `Conclusion`. J Cheminform expects a combined **Results and discussion** plus **Conclusions**. Structural edit, author-side. |
| V4 | `_retention_auc_norm`, L1791 | `auc_norm = trapezoid(R²(σ)/R²(0), σ) / (σmax − σmin)`. It divides by the baseline, so it is a **fraction retained**. |
| V5 | `calculate_robustness`, L1803 | **Averages across iterations before computing `auc_norm`** (`group.groupby('sigma')['r2'].mean()`, L1830). One value per model × rep × strategy, no replicate spread. This is the same defect as the validation fold-averaging (thread `E2-6`) and it affects every ranking table and heatmap fed from `auc_df`. |
| V6 | `run_robustness_anova`, L2068 | **Correct** — groups by `model, rep, iteration`, so the QM9 robustness ANOVA has genuine within-cell replicates and its residual term is meaningful. The saturation problem is validation-only. Baseline is taken per iteration. |
| V7 | `run_simple_effects` call sites, L2181 and L2277 | Also per-iteration. The `table1_supp_simple_effects` numbers are therefore replicate-backed. |
| V8 | `REVISION_GUIDE.md` §9.1 vs L62–134 | Contradiction `X1` confirmed present in one file: §9.1 says the decoupling claim is an arithmetic artefact and must not be evidenced by NGBoost; `RG-T4`/`RG-T13` say keep the claim and cite NGBoost. **Note a subtlety for whichever way this is settled:** dividing by the baseline removes it *mechanically*, but does not force the correlation between baseline and retention to be zero — that correlation is still an empirical quantity. "Decoupled by construction" is too strong as written. |


---

# I. Step-1 close-out — findings from the 2026-08-21 grounded pass

17 agents read the figure script, `paper.tex`, `rust/src/main.rs`, `models/models.py`, every local
results file and all 30 SLURM directories. Everything below is code- or data-grounded. Corrections to
the agents' own claims are marked ⚠.

## I1. New facts that change the plan

| # | Fact | Evidence |
|---|---|---|
| N1 | **No release binary exists in this tree.** `rust/target/release/` is absent; `process_and_train.py:1849` invokes `../rust/target/release/rust_processor`. The ARC binary predates the held-out-noise fix. Nothing can produce corrected runs until the fix is committed and `cargo build --release` is run on ARC. | dir listing; `cargo check` passes on debug, exit 0, 19 dead-code warnings |
| N2 | **The QM9 re-run costs 11,132 requested wall-hours (44,528 core-hours)** across 216 job scripts and 906 distinct output CSVs. | SLURM `--time` sweep across all 30 dirs |
| N3 | **The external pipeline is clean and complete**: 16,170 rows per dataset, model × representation × strategy × σ × fold all present, noise into `y_train` only. Test labels bit-identical across all 11 σ in 27 of 27 uncertainty files. | `KIRBy/tests/results/validation/{logd,caco2,herg}/all_results.csv`; `alternative_data_noise_robustness.py:760` |
| N4 | **`injected_noise` is now a zero column.** It was never recorded — it is reconstructed by regressing the noisy label on the clean one (`scripts/utils.py:216-224`). Before the fix it held the *wrong molecule's* noise; after the fix test labels are clean, so it is identically zero. | code read |
| N5 | **The question "does uncertainty flag corrupted labels" is undefined on held-out data**, not merely unmeasured. Corrupted labels only ever enter training; uncertainty is only ever saved for test molecules. All 13 `save_uncertainty_values` call sites pass test arrays; `predict(x_train` appears nowhere in `models/models.py`. | code read |
| N6 | **A two-output (mean + variance) head already exists and is disabled.** `models/models.py:2031` builds it when `loss_name == 'heteroscedastic'`, but the variance column is discarded at `:2088` and `:2103`, `decompose_uncertainty_sampling_heteroscedastic` has zero callers, and `--loss` is never passed by any SLURM script. | code read |
| N7 | **Two models already compute a real per-molecule aleatoric term and throw it away**: `train_heteroscedastic_gp` (`models/models.py:6835-6837`, only `total_std` saved at `:6851`) and `train_evidential_kernel` (`:6963-6964`, `:6977`). Wiring them is a two-line change. | code read |
| N8 | **The Gaussian process was never run on the paper's primary representation.** `slurm_scripts_gauche_rbf/` covers ecfp4/mhggnn/mol2vec/smiles — no `continuous_pdv` target. `slurm_scripts_continuous_pdv/gauche.sh` runs the Tanimoto default on a real-valued descriptor. The figure script asks for `gauche_rbf` at `continuous_pdv` (`:2657-2658`) and gets nothing. | dir + code read |
| N9 | **σ = 0 is duplicated six times for the deterministic models.** `process_and_train.py:1609` sets `'noise': s > 0`, so the six strategy runs at σ = 0 are byte-identical. Confirmed empirically: LightGBM, QRF, SVM, XGBoost have exactly zero across-strategy SD at σ = 0 in 100 % of cells on all three external datasets. Every σ = 0 standard error and p-value is inflated sixfold for those models. | code + data |
| N10 | **Two different baseline gates are in force under one headline.** `ROBUSTNESS_BASELINE_THRESHOLD = 0.3` (`:429`) governs `table1_anova_summary.csv`; `BASELINE_THRESHOLD = 0.6` (`:423`) governs `table1_supp_simple_effects.csv`. Different configuration sets. | code read |
| N11 | **The methods figure plots synthetic data with two formulas that do not match the injector.** `create_methods_figure` generates a 3-component Gaussian mixture (`:2538-2546`); its value-proportional noise is additive where Rust's is multiplicative (`rust/src/main.rs:313`), and its threshold is a median split where Rust uses an absolute ±1.0 cut (`:364-373`). `paper.tex:359` implies it shows QM9. | code read |
| N12 | **The two injectors disagree.** Rust's value-proportional factor is 0.1 (`main.rs:1177`); NoiseInject's is 0.05 (`NoiseInject/noiseInject/core.py:157`). QM9 and the external datasets were noised by different code with different constants. Quantile also differs — Rust indexes a sorted array, NumPy interpolates. | code read |
| N13 | **The record-truncation risk is real and silent.** Two `continue` statements in the ECFP4 block (`main.rs:831`, `:848`) fire *after* every earlier field of the record is written, leaving it 256 bytes short; `parse_mmap` reads a fixed offset and its bare `except: continue` (`process_and_train.py:1224-1227`) swallows the misparse, so every later molecule in the file is read at the wrong offset. | code read |
| N14 | **The index-drift risk does NOT fire on QM9** but is live elsewhere. The reject condition is SMILES length < 5; QM9 SMILES retain explicit hydrogens, minimum length 6 over the first 3,000 molecules. It is live for the ADME sets, where `CCO` and `CC` are plausible. | code + data |
| N15 | **Threshold's raw ±1.0 cut makes it a different strategy per endpoint.** Fraction of molecules receiving the doubled dose: QM9 100 %, hERG 100 %, LogD 85.4 %, log10 Caco-2 efflux 9.2 %. | computed from label files |
| N16 | **σ does not control dose.** Realised RMS perturbation at σ = 1 spans 4×: outlier 0.502, heteroscedastic 0.669, quantile 0.899, Gaussian 1.000, value-proportional 1.701, threshold 2.000 eV. Mean retention per strategy is a perfect rank-inversion of that ordering (Spearman −1.000 on QM9; −0.943 / −1.000 / −0.943 on LogD / Caco-2 / hERG). `paper.tex:222` and `:354` promise the opposite. | computed |
| N17 | **Concentration differs where dose does not.** Share of total injected variance landing on the top 10 % of molecules: Gaussian 10.0, threshold 10.0, value-proportional 12.4, heteroscedastic 12.6, quantile 49.4, outlier 96.3. | computed |
| N18 | **Some QM9 uncertainty files do not exist anywhere.** No `*_uncertainty_values.csv` under `results/`; no VBLL uncertainty CSVs in either repo. The uncertainty figure's source data is external-only. | dir listing |

## I2. ⚠ Corrections to the agents' own claims

| # | Claim made | Correction |
|---|---|---|
| ⚠1 | "The variance decomposition rests on an unstated `r2.clip(lower=0)`, and unflooring it flips every η² (neural representation 53.5 → 8.1 on LogD)." | **No such clipping exists in `generate_paper_figures_v2.py`** — the only `clip` is a colour-scale clamp at `:1549`. Verified by grep. The flooring was the agent's own recomputation choice, not the pipeline's. **What IS real and unstated in the paper:** whole iterations with any R² below −0.5 are deleted (`filter_catastrophic_iterations`, threshold at `:437`), and the performance ANOVA then additionally drops rows with R² below −10 (`:1946`, `:2147`, `:2260`). Those two choices are undeclared and the decomposition is sensitive to them. The sensitivity is worth reporting; the specific 53.5 → 8.1 numbers are not the pipeline's. |
| ⚠2 | "Model is not the largest ANOVA term in 16 of 72 external cells." | Reproducible in principle but computed under the same non-pipeline flooring. Recompute through the shipped code path before quoting. |

## I3. Candidate key takeaways, after adversarial audit

Four drafting agents proposed takeaways in four areas; four auditing agents tried to refute each one.

**Survived intact**
- Six noise strategies differ in *how much* they corrupt far more than in *how* — realised dose spans 4× at the same σ, and retention ordering follows dose almost perfectly.
- Threshold noise is a different strategy on each dataset because its cut is applied to the raw label.
- Concentration is real for exactly two strategies: quantile puts 49 % of the injected variance on a tenth of molecules, outlier 96 %; the other four sit at 10–13 %.
- Baseline failure is a property of the model-and-representation pair, not of either alone — every neural model fails on hERG with the descriptor vector while the same pairs are fine on the other two datasets.
- Predicted uncertainty stops distinguishing between molecules as noise rises: its spread across molecules falls from 0.398 to 0.156, in 27 of 27 configurations.

**Survived with a caveat that must be stated**
- Best-by-retention and best-by-accuracy-under-noise are different rankings — they agree in 18 of 72 cells, and in 0 of 24 on LogD.
- Uncertainty inflates faster than error does (ratio 1.12 → 2.01) and coverage drifts from 0.635 to 0.884 against a 0.68 target.
- Within a single noise level, the correlation between predicted uncertainty and actual error collapses from 0.289 to 0.069, in 27 of 27 configurations.
- "No kernel model ever fails" is **false as written** — the Gaussian process fails on hERG with the descriptor vector (18 of 30 clean runs below zero, median −0.288). The true statement is "no tree ensemble or SVM fails".

**Died**
- "Model architecture is the largest source of variance at every noise level under every strategy" — model is not the largest term in 16 of 72 external cells (see ⚠2 for the caveat on how this was computed).
- "Representation is the dominant term among neural models" — sensitive to how catastrophic runs are handled.
- Any float built on per-sample noise detection.


# J. Proposed figure and table plan

Five main figures, four main tables (currently 8 and 6). Three of the five figures and two of the four
tables are buildable **today**, with no server access and no re-run.

| Float | What it shows | Buildable |
|---|---|---|
| **Fig 1** — what σ actually injects | (a) spread of per-molecule noise scale at σ=1, one strip per strategy, with realised dose and its variability; (b) 4 datasets × 6 strategies heat map of dose in units of each dataset's label spread, with the % of molecules hit by the high multiplier; (c) share of injected variance landing on the worst-hit molecules | today — analytic, no training runs. Blocked only on deciding which value-proportional constant is the real one |
| **Fig 2** — clean-data failure is a property of the pair | 4 facets (QM9 + 3 external), model × representation, median clean R², cells below zero hatched, plus a count of negative runs per cell | 3 external facets today; QM9 facet needs the σ=0 slice copied off ARC — **no re-run** |
| **Fig 3** — retention and accuracy are different rankings | 3 facets, accuracy at σ=0.6 against retention, one point per model × representation, fold error bars both axes | needs a ~20-line change so folds are kept before the retention curve is integrated. No compute |
| **Fig 4** — strategy severity is dose | (a) 4 datasets × 6 strategies retention with spread bars; (b) retention against realised dose, 24 points, per-dataset correlation | 3 external rows today; the QM9 row needs the re-run |
| **Fig 5** — what noise does to predicted uncertainty | 4 rows × 3 model families, every panel against σ: uncertainty-to-error ratio; spread of uncertainty across molecules; correlation of uncertainty with error; coverage against its 0.68 target | today — 27 verified files |
| **Table 1** — injector specification and realised dose | what each strategy does in each of the two injectors, whether they agree, realised dose and its variability per dataset | today |
| **Table 2** — retention and clean accuracy per model | 13 models × 3 datasets at one representation; clean R², all six retention columns, accuracy at σ=0.6. The cross-strategy MEAN and STD columns are deleted | same change as Fig 3 |
| **Table 3** — uncertainty metrics by σ | 3 models × 3 datasets at σ ∈ {0, 0.5, 1.0}: mean uncertainty, its spread, correlation with error, coverage, R². The noise-detection column is deleted and retracted | today |
| **Table 4** — metric definitions | one definition of the retention metric; NDS and ECE purged | editorial |

**Cut:** the ANOVA table and figure; the Wilcoxon table; the interaction figure; the global overview
figure; the neural-family figure; the combined uncertainty figure; both validation figures; the
supplementary validation ANOVA.

**Consequence for the paper's second research question.** Its second half — whether per-sample
uncertainty tracks which individual labels were corrupted — cannot be supported by any float, on any
dataset, with or without a re-run. It has to be rewritten as what Figure 5 does answer: how predicted
uncertainty responds to label noise it cannot see.

**Two code changes gate most of the rest**, both in `scripts/generate_paper_figures_v2.py`:
keep the fold axis before integrating the retention curve (`calculate_validation_auc`, `:1318-1393`),
and delete `fix_injected_noise` (`:980-1021`), which now backs no surviving float.

# K. Decisions needed before compute is spent

Ordering constraint: the dose decision redefines what a σ unit is, so it comes before the threshold
and grid decisions. The validation-noise decision changes every fitted model, so it comes before
anything is submitted. The scope decision sets the budget everything else draws against.

| # | Decision | Cost if wrong |
|---|---|---|
| K1 | Scope: full QM9 re-run, reduced replicates, fewer representations, or lead with the external data | the entire budget |
| K2 | Do validation labels carry their own noise? Six model families paste validation into training | every fitted model, both pipelines |
| K3 | Renormalise so one unit of σ means the same corruption in all six strategies? | every noise level above zero for five of six strategies |
| K4 | Move the threshold cut onto the standardised label? | every threshold run |
| K5 | Is the per-sample uncertainty question retired, or answered with cross-fitted training-set uncertainty? | a whole extra arm if answered |
| K6 | Two-output head on the neural models, or state plainly that no model predicts a per-molecule noise level? | every neural run |
| K7 | Keep the 11-point σ grid or coarsen it? | all results must move together |
| K8 | Run the skewed-noise experiment, or delete the three broken samplers? | an extra arm |
| K9 | Run the Gaussian process on the primary representation? (6 jobs) | a visible hole in two figures |
| K10 | Stop paying for six identical σ=0 runs per model? | ~9 % of the bill, and every σ=0 p-value |


---

# L. Compute cost — what the numbers actually are (measured 2026-08-21)

## L1. The "11,000 hours" figure is a request, not a cost

It is the sum of `#SBATCH --time` over the job scripts. My own sweep over all 530 scripts that declare
a time gives **19,615 requested wall-hours**; restricting to the QM9 scripts gives roughly the 11,132
the agent reported. Neither is a measurement of anything.

Nearly every script asks for `47:59:00` or `71:59:00` — the partition ceiling — and each of those
requests covers **ten sequential training campaigns** inside one script. There are **no job output
files anywhere in the repository**, so actual runtimes have never been recorded locally.

**To get the real number, one command on ARC:**

```bash
sacct -S 2026-01-01 -u $USER --account=stat-cadd \
      --format=JobID%16,JobName%36,State,Elapsed,Timelimit,AllocCPUS,MaxRSS \
      | grep -v '\.batch\|\.extern' > ~/runtimes.txt
```

Then `scp` it down. Until that exists, any scope decision is being made on a padded ceiling.

## L2. The honest unit of work

A clean re-run of the main design is:

> 11 models × 5 representations × 6 strategies × 11 noise levels × 10 replicates = **36,300 model fits**

The SLURM directories currently specify **117,275 fits across 964 distinct output files** — roughly
three times the clean design — because the `fixup`, `rerun`, `missing`, `remaining` and investigation
directories overlap heavily. Regenerating one deduplicated script set is a saving on its own, before
any statistical compromise.

Neural models dominate the specified work: `mlp` 22,506 + `dnn` 22,506 + `flexible_dnn` 16,500 =
**52 % of all specified fits**.

## L3. A measured inefficiency: data preparation is redone once per noise level

`process_and_train.py:1866-1897` loops `for s in sigma: for iteration in range(b):` and calls
`split_qm9` **inside both loops**. Every one of those calls re-shuffles QM9, recomputes the DeepChem
scaffold split, and recomputes every molecular representation from scratch (`process_and_train.py:
560-592`), then the Rust binary re-reads and rewrites the whole file (`:1628-1641`).

**None of that depends on the noise level.** The shuffle and the split are seeded by `iteration_seed`,
which is a function of `iteration` alone (`:1871`). Only the training labels change with σ.

Measured on this machine, 10,000 QM9 molecules:

| Stage | Time | Recomputed |
|---|---|---|
| Read SDF + canonical SMILES | 5.0 s | 110× per output file |
| **RDKit descriptor set (208 descriptors — this is PDV)** | **220 s** | **110× per output file** |
| ECFP4 fingerprints | 0.8 s | 110× per output file |
| DeepChem scaffold split | not measurable locally (broken PyTorch extension); typically 10–60 s | 110× per output file |

For a descriptor-representation output file that is about **6.7 hours of descriptor computation, of
which 6.1 hours is redundant**. Caching the prepared split per replicate and re-injecting only the
labels cuts the preparation stage by ~91 % and costs nothing statistically.

`mol2vec` and `mhggnn` are neural fingerprints over 10,000 molecules and are almost certainly slower
than the RDKit descriptors; they are recomputed on the same schedule. Not measurable here.

## L4. Savings, ranked, with what each costs

| Saving | Size | Statistical cost |
|---|---|---|
| Cache the prepared split + features per replicate; re-inject labels only | ~91 % of the preparation stage | **none** |
| Rebuild one deduplicated script set from the clean design | up to ~3× on specified work | **none** |
| Reuse σ = 0 across the six strategies for deterministic models | ~9 % | **none** — the six runs are byte-identical (`process_and_train.py:1609`; verified: LightGBM, QRF, SVM, XGBoost have exactly zero across-strategy spread at σ=0 in 100 % of cells) |
| Keep σ = 0 results from the existing runs | 1 of 11 levels | **none** — no noise is injected at σ = 0, so those results are unaffected by every bug and by both decisions taken today |
| 10 replicates → 5 | ~50 % | real but bounded: every gate in the analysis needs ≥ 5 (`MIN_CELL_ITERS = 5`, `generate_paper_figures_v2.py:434`). Costs precision on the residual term, which is itself a reported result |
| 11 noise levels → 6 | ~45 % | all results must move together; retention values shift slightly |
| Two representations instead of five | ~60 % | **guts research question 1** — do not do this |

## L5. Local environment note

The project's Python environment on this machine is broken: `torch_cluster` and `torch_sparse` are
compiled against a different PyTorch than the installed 2.2.2, so `import torch_geometric` fails and
`process_and_train.py` cannot run locally at all. **Not touched** — fixing it means reinstalling into
the author's conda environment. It is why the timings above are stage-by-stage rather than end-to-end.

The release binary **has now been built locally**: `cargo build --release`, 4 m 55 s, exit 0.
It still needs building on ARC after the fix is committed.

---

# M. Proposals on the table — NOTHING DECIDED

**Status 2026-08-23: no decision has been made on any item in this section.** An earlier version of
this file recorded M1 and M2 as author decisions. That was wrong and has been retracted. They are
proposals, written up so the implementation cost is visible, not choices that have been made.

**Standing instruction from the author (2026-08-23): the paper is fixed by RE-RUNNING, not by
rewording. Do not propose retiring, narrowing or rephrasing a research question as a way out of a
data problem. If the current data cannot answer the question, the answer is to design the run that
can.**

## ⬜ M1. PROPOSAL — renormalise the noise dose per strategy

One unit of the noise dial currently injects four times more corruption under one strategy than
another, and damage tracks dose almost perfectly (§I1 N16). The proposal is to rescale each strategy
so one unit means the same amount of corruption everywhere, leaving only *where* the damage lands as
the difference between strategies.

Implementation, both injectors, same commit:
- `rust/src/main.rs:302-430` — after computing each molecule's noise scale, divide by the realised RMS
  over the training labels and multiply by the target dose. Computed per dataset, from training labels
  only, before any noise is drawn.
- `NoiseInject/noiseInject/core.py:71-170` — same.
- Verification before launch: draw all six strategies at one dose on the real training labels and
  confirm the realised standard deviation matches in all six. If any misses, the run is confounded.
- Touches the σ-definition half of `D1`; would make `INS-T17` and `INS-T19` live paper edits.
- Cost: every noise level above zero, five of six strategies (Gaussian unchanged by construction).

## ⬜ M2. PROPOSAL — validation labels carry their own noise

After the held-out fix, six model families that merge validation into training end up with 11 % clean
training labels, while the neural networks stop training against a clean validation score they would
never have in reality (§I1, `A1b`).

Implementation:
- `rust/src/main.rs:1142-1147` — draw a second noise map keyed over validation indices, independent of
  the training map.
- `rust/src/main.rs:1055` — `apply_noise = true` for the validation call, with that map.
- `rust/src/main.rs:1082` — test stays clean.
- `KIRBy/tests/alternative_data_noise_robustness.py:978-981` — inject before validation is carved out
  of `y_train_full`.
- Cost: every noise level above zero, all six strategies, both pipelines.

## ⬜ M3. Scope of the re-run — cannot be costed yet

The 11,000-hour figure was a sum of requested time limits, not measured cost (§L1). The `sacct`
command in §L1 gives the real number. The three savings in §L4 that carry no statistical penalty apply
whatever the scope turns out to be.

## ⬜ M4. PROPOSAL — make the per-sample uncertainty question answerable

**Not a rewording. A protocol change plus a re-run.** See §N.

---

# N. Making the per-sample uncertainty question answerable — by re-running it

The question, unchanged from `paper.tex:188`: **do a model's per-molecule uncertainty estimates track
which individual labels have been corrupted?**

It is currently unanswerable because of three things the pipeline does not do. All three are fixable
in code, and then the question is measured, not reworded.

## N1. The three things that have to change

**(1) Corrupted molecules must be scored.**
Corruption only ever enters the training split; uncertainty is only ever saved for test molecules
(all 13 `save_uncertainty_values` call sites pass test arrays; `predict(x_train` appears nowhere in
`models/models.py`). So today the question is being asked about molecules that were never corrupted.
Training molecules must be scored.

**(2) Those scores must come from a model that never saw the molecule's corrupted label.**
Predicting on a molecule the model trained on measures memorisation, not uncertainty: the Gaussian
process has zero posterior variance at its own training inputs by construction, and the forests have
fitted those exact rows. The fix is **cross-fitting inside the training set** — split training into K
parts, train on K−1, score the held-out part, rotate. Every training molecule then carries an
uncertainty from a model that never saw its corrupted label. This is the standard out-of-fold
construction; it is not an approximation.

**(3) The injected noise must be recorded, not reconstructed.**
Rust writes no record of the noise it draws — the map is built at `rust/src/main.rs:1242-1247`, added
to the label at `:751`, and discarded. The `injected_noise` column in the uncertainty CSVs is a linear
regression of the noisy label on the clean one (`scripts/utils.py:216-224`), whose residual at σ = 0
is floating-point rounding that grows with the label — which is exactly where uncertainty is largest.
That is why the zero-noise control currently shows a *higher* correlation than the real noise levels
do. The control fails in the direction that manufactures a positive result.

## N2. What Rust must emit

A companion file per run, written between `rust/src/main.rs:1247` and `:1252`, where the noise map,
`mean` and `std_dev` are all still in scope:

| Column | Why |
|---|---|
| `split` | train / validation / test |
| `record_index` | the position the noise map is keyed on |
| `canonical_smiles` | the only real molecule identifier; already decoded at `process_and_train.py:1080` and then discarded |
| `y_clean_raw` | the uncorrupted label, raw units |
| `epsilon_raw` | the noise actually drawn for this molecule, raw units |
| `norm_mean`, `norm_sd` | so anything can be converted back to label units |
| `strategy`, `sigma`, `seed` | provenance — none of these appear in any output today |

This also fixes three other things at once: it makes the error metrics de-standardisable across noise
levels (`A1c-1`), it gives every uncertainty row a molecule identifier instead of a row counter, and it
lets rows be matched across replicates.

## N3. What the analysis then is

Computed **within a single noise level and within a single strategy and within a single replicate** —
pooling across any of those is what produced the discredited result.

- **Continuous strategies** (Gaussian, value-proportional, heteroscedastic, threshold): Spearman
  correlation between out-of-fold predicted uncertainty and the size of the injected noise, across
  training molecules.
- **Concentrated strategies** (quantile, outlier): the injected noise is bimodal — a small subset gets
  a large dose and everything else gets almost none. The right measure is how well uncertainty ranks
  the corrupted subset above the rest, reported as a ranking score, not a correlation.
- **A chemist-readable number alongside both:** of the tenth of molecules the model is most uncertain
  about, what fraction are in the tenth that were most corrupted.
- **Negative control:** at σ = 0 no noise is drawn, so the recorded epsilon is exactly zero for every
  molecule and the measure must be undefined. If it is not, the pipeline is still leaking.

## N4. What this costs

It is an **added arm on the probabilistic models only** — the main accuracy and robustness numbers
come from the normal train/test protocol and are unaffected.

- Roster: the 7 models that produce a per-molecule uncertainty.
- Cost multiplier: K, the number of cross-fitting folds. K = 5 is the usual choice.
- **The multiplier lands on training only, and only for those models.** With the data-preparation
  caching in §L3 in place, the features and the split are computed once per replicate and reused
  across all K folds and all 11 noise levels, so the K-fold cost is K model fits, not K full pipeline
  passes.
- Honest caveat to state in Methods: cross-fitted models see 4/5 of the training data, so they are
  marginally weaker than the headline models. That is inherent to the construction and is why the
  headline numbers stay on the normal protocol.

## N5. What has to be decided before this can be scheduled

- How many cross-fitting folds.
- Whether validation molecules are scored too (they carry noise under the §M2 proposal).
- Whether every probabilistic model is cross-fitted or only a stated subset.

---

# O. What happens next — the ordered work list

Step 1 (triage and replan) is essentially finished; §I–§N are its output. Step 2 is code, then compute.
**Nothing here is decided.** It is the list as it stands after reading the code.

## O0. Where things actually are, right now

- One fix exists, uncommitted: the held-out splits no longer receive the training noise
  (`rust/src/main.rs`, 12 lines).
- The release binary now builds cleanly locally (4 m 55 s, exit 0). It does not exist on ARC.
- Nothing has been re-run. Every QM9 number above zero noise is still invalid.
- The local Python environment cannot run the pipeline at all (§L5), so nothing can be smoke-tested
  end to end on this machine until that is fixed.

## O1. Code, before any compute

Ordered by dependency. Items 1–4 change what the noise *is*, so they must all land before anything is
submitted; getting one of them wrong means running twice.

| # | Change | Files | Blocked on |
|---|---|---|---|
| 1 | Commit the held-out-noise fix | `rust/src/main.rs` | — |
| 2 | Validation split gets its own noise map | `rust/src/main.rs:1055`, `:1142-1147`; `alternative_data_noise_robustness.py:978-981` | §M2 |
| 3 | Renormalise dose per strategy, both injectors | `rust/src/main.rs:302-430`; `NoiseInject/noiseInject/core.py:71-170` | §M1 |
| 4 | Threshold cut moves onto the standardised label | `rust/src/main.rs:364-373`; `core.py:97-115` | `D1b` |
| 5 | Reconcile the two injectors — value-proportional is 0.1 in Rust and 0.05 in NoiseInject; quantile indexes a sorted array in one and interpolates in the other | both | — |
| 6 | Record the injected noise and provenance (§N2) | `rust/src/main.rs:1247-1252` | — |
| 7 | Cross-fitted uncertainty on training molecules (§N) | `process_and_train.py`, `models/models.py`, `scripts/utils.py` | §N5 |
| 8 | Fix the record-truncation bug — the two `continue` statements leave a 256-byte-short record and desync every later molecule | `rust/src/main.rs:831`, `:848` | — |
| 9 | Fix index drift — the noise map is keyed on loop position, the target array on push count | `rust/src/main.rs:180-188`, `:951-967` | — |
| 10 | Fix or delete the three broken noise-shape samplers (one has support entirely below zero; one injects 2.45× what it claims) | `rust/src/main.rs:211-264`, `:461-496` | `K8` |
| 11 | Decide `--strategy-params`: pass it or delete it. It has never been passed, so the JSON has never been read | `process_and_train.py:260`, `:1628-1641` | — |
| 12 | Cache the prepared split and features per replicate instead of per noise level — the free 91 % saving (§L3) | `process_and_train.py:1866-1897` | — |
| 13 | Run σ = 0 once per model and representation instead of six times | script generator | — |
| 14 | Two-output head, or wire the two models that already compute a per-molecule aleatoric term and throw it away | `models/models.py:2031`, `:2088`, `:6851`, `:6977` | `K6` |
| 15 | Fix the local Python environment so any of this can be tested before it is submitted | conda env | — |

## O2. Verify, before any compute

Every one of these is a small local test, not a cluster job.

1. Draw all six strategies at one dose on the real QM9 training labels; confirm the realised standard
   deviation matches across all six. **If this fails the whole re-run is confounded.**
2. Confirm the threshold cut now selects a similar fraction of molecules on QM9, LogD, Caco-2 and hERG
   instead of 100 / 85 / 9 / 100 %.
3. Confirm the recorded epsilon reproduces the noisy label exactly: `y_clean + epsilon == y_noisy`.
4. Confirm validation labels are noisy and test labels are bit-identical across all noise levels.
5. Feed a record that fails the fingerprint check and confirm the file no longer desyncs.
6. Run the smallest possible end-to-end job and confirm every new column is populated.

## O3. Regenerate the job scripts

From the clean design, not by editing 559 existing files. The current directories specify roughly three
times the work the design actually needs (§L2).

## O4. Then run. Then figures.

The figure plan is §J. Three of five figures and two of four tables do not need the re-run at all and
can be built while it is queued.

---

# P. Uncertainty re-run — BUILT AND TESTED (2026-08-23)

Author instruction: *"There is no rewording, there is only re-running."* This section
records the code that makes both uncertainty questions measurable, and the jobs that produce
the data. **Nothing has been submitted.**

## P1. What was changed

| Repo | File | Change |
|---|---|---|
| `NoiseInject` | `noiseInject/core.py` | `noise_scale()` — per-molecule noise scale, consumes no randomness, so it can be computed for molecules that are never corrupted. `inject_verbose()` — returns the epsilon actually drawn. All six strategies refactored onto one shared scale function. |
| `KIRBy` | `tests/alternative_data_noise_robustness.py` | Runners return an `extras` dict with the true injected noise, the per-molecule scale for train *and* test, and out-of-fold training predictions. Uncertainty written for all six strategies with `split` / `strategy` / `fold` / `noise_scale` / `injected_noise` columns. New flags `--strategies`, `--unc-strategies`, `--oof-folds`. |
| `qsar_qm_models` | `slurm_scripts_uncertainty_rerun/` | 7 job-array scripts (72 tasks each, 504 total), `preflight.sh`, `merge_results.py`, `generate_scripts.py`, `RUNBOOK.md`. |

## P2. Three bugs found while patching — all pre-existing

| # | Bug | Consequence |
|---|---|---|
| P-B1 | The uncertainty writer appended one frame per fold but wrote to a filename with no fold in it, so the file was rewritten five times and **only the last fold survived**. It also averaged over `sample_idx`, a *within-fold* position — so it was averaging different molecules together. | **Every existing `*_uncertainty_values.csv` is one fold of five.** Any statement resting on those files rests on 20 % of the data. Verified on `validation_rerun/ecfp4_caco2/caco2/QRF_ECFP4_uncertainty_values.csv`: 4,752 rows = 432 molecules × 11 noise levels, i.e. a single fold, and the file carries no `fold` or `strategy` column. |
| P-B2 | `--sigmas` was never passed through for hERG, so hERG always ran all 11 levels regardless of the flag. | Silent; wasted compute on any partial run. |
| P-B3 | The hERG loader hard-coded a `pKi` column; the cached file has `pChEMBL`. | `KeyError` kills the whole run on contact. Would have failed on the cluster. |

## P3. Verification performed

| Check | Result |
|---|---|
| Patched injector reproduces the old one | **336 / 336 exact**, across 6 strategies × 7 noise levels × 4 label distributions |
| Patched pipeline reproduces pre-patch predictions | **exact (0.000e+00)** on all six strategies, single-threaded |
| Control for the above | pre-patch code compared against **itself** gives the same ~1.8e-15 wobble with `n_jobs=2` and 0 with `n_jobs=1` ⇒ the wobble is the forest library's parallel summation order, **not** this change. Worth knowing: tree results are not bit-reproducible at `n_jobs=-1`. |
| Recorded epsilon reconstructs the noisy label | exact, all six strategies |
| σ = 0 records exactly zero noise | **passes** — a true negative control. The old `injected_noise` was a regression residual and was non-zero at σ = 0, which is why the zero-noise control showed a *stronger* correlation than the real noise levels |
| Test-side scale uses the training cut-points | passes — 84/150 and 48/150 molecules classified differently than if the test set defined its own |
| Out-of-fold error exceeds in-sample error | passes (+66 %) ⇒ not measuring memorisation |
| Package test suite | 19 passed |
| Signal recovered on synthetic data | (A) training: Gaussian −0.08, outlier +0.12, quantile +0.13; σ=0 undefined as required. (B) test: Gaussian **flat by construction (the control)**, outlier +0.22, quantile +0.36 |

## P4. Environment risks found

- **QRF cannot be fitted in the local environment** — `quantile_forest` and
  `scikit-learn` disagree (`Invalid parameter 'monotonic_cst'`). If this reproduces on the
  cluster, every QRF job fails on contact. `preflight.sh` checks it explicitly.
- **Two KIRBy checkouts exist on the cluster** — `/data/stat-cadd/…` (used by the qsar job
  scripts) and `/data/stat-ecr/…` (used by KIRBy's own scripts). The new scripts use
  stat-cadd, matching the working `slurm_scripts_validation_rerun`. Confirm before submitting.
- **`NoiseInject` must be installed editable** from the checkout that gets pulled, or the
  patch has no effect. Preflight asserts `inject_verbose` exists.

## P5. Design of the run

- One array task per (dataset, representation, strategy); one script per model.
- **Every task writes its own results directory.** The pipeline merges results by
  read-modify-write, so tasks sharing a directory would race and silently lose rows.
  `merge_results.py` stitches them back and emits `coverage.csv`.
- Cross-fitting applies only to the seven models that emit a per-molecule uncertainty;
  the pipeline enforces this so the roster cannot drift from the SLURM generator.
- Cost per task: 11 noise levels × 5 folds × (1 + 5) fits = 330, against 55 without
  cross-fitting.

## P5b. The confound in question B, and how it is controlled

The noise scale a molecule receives is a deterministic function of its label. A model's
uncertainty may already track the label because extreme molecules are harder to predict, so a
raw correlation between uncertainty and noise scale would be partly manufactured.

Every output row therefore carries two columns: `noise_scale` (what was actually applied,
exactly zero at σ = 0) and `noise_pattern` (the *shape* — which molecules the strategy hits
hardest — taken at a fixed reference level and therefore **identical at every σ, including
zero**).

The defensible effect is **ρ(uncertainty at σ, pattern) − ρ(uncertainty at σ = 0, pattern)**.
The σ = 0 model trained on entirely clean labels but saw the same label distribution, so its
correlation *is* the confound. Gaussian noise is the second control: it gives every molecule the
same scale, so the pattern is flat and the correlation is undefined by construction. Verified:
pattern constant across σ for all six strategies, non-degenerate for the five uneven ones, flat
for Gaussian.

## P7. Nine defects found in adversarial review — all fixed (2026-08-24)

A ten-agent adversarial review of the patch found nine defects plus two of my own making. All fixed
and regression-tested (`KIRBy/tests/smoke/smoke_nine_fixes.py`).

| # | Defect | Consequence if shipped | Fix |
|---|---|---|---|
| 1 | The neural cross-fitting consumed the global torch generator | The **main** neural results at every noise level would silently differ from a run made without cross-fitting, so this run's R² could not be compared with any other | Snapshot and restore the generator around the out-of-fold block |
| 2 | Uncertainty held in memory, written only after all five folds | A wall-clock timeout destroyed everything the run exists to produce, while leaving a partial-results file that made the job look productive | Flushed after every fold, atomically, keyed so a re-flush replaces rather than duplicates |
| 3 | Cross-fitting used a **random** split | Breaks the standing scaffold-split rule, and puts out-of-fold uncertainty in an interpolation regime while the test set is extrapolation — the two are then not on one scale | Scaffold groups threaded through; inner split is `GroupKFold`, with a logged fallback when a fold has too few scaffolds |
| 4 | The Gaussian process is capped at 2,000 training molecules and its out-of-fold rows were numbered against that subsample | "Do QRF and GP flag the same molecules?" — the obvious analysis — would have paired **different molecules** and returned a spurious near-zero agreement | The subsample's real row indices are carried through and written as `sample_idx` |
| 5 | A failed inner fold was caught, blanked and written | Coverage marked all-blank cells OK; per-cell correlations would be computed on undocumented partial subsets | Returns how many folds succeeded; all-blank blocks skipped; `oof_folds_ok` on every row; coverage reports `OOF_ALL_NAN` |
| 6 | The merge built one frame of ~100 million rows | Out of memory **after** the multi-day run finished, taking the coverage report with it | Streams in chunks; coverage accumulated per file; `--parquet` option |
| 7 | Threshold noise cuts the raw label at ±1.0 and every hERG value clears it | 28 tasks producing a **constant** column; question B undefined for that arm | Run-time warning naming every constant arm; preflight prints the exact array indices to skip; `--threshold-quantile` derives cut-points from the labels |
| 8 | Heteroscedastic and value-proportional rank molecules identically (Spearman 1.000 on all three datasets) | A third of the run duplicate for these questions; "both show it, so it replicates" would be one observation reported twice | Documented as one arm; `--drop-strategies hetero` reclaims 84 tasks with no loss of rank information |
| 9 | The neural models early-stopped against **clean** validation labels | The four Bayesian nets were explicitly selected *not* to fit the injected noise — suppressing the measured quantity, for the neural half of the roster only, making any tree-versus-neural comparison meaningless | Validation carries the same strategy at the same level, from an independent generator so the training draw is untouched; `--no-noise-validation` restores the old behaviour |

**Two more, of my own making, found by executing the scripts rather than reading them:**

- Adding `set -u` made the unguarded `$CONDA_PREFIX` reference fatal — **all 504 tasks would have died
  in under a second**, before python was reached. The reference scripts survive only because they do
  not set `-u`.
- A dangling line continuation that `bash -n` accepts but which splits the command in two, so
  `--results-root` would have been run as a program.

Both are why the job scripts are now *executed* against a stubbed interpreter in testing, not just
parsed.

## P8. The analysis plan, written down so nobody manufactures a result

Recorded in full in `slurm_scripts_uncertainty_rerun/RUNBOOK.md`. The three points that matter:

- **Question A's statistic is not the raw correlation between uncertainty and injected noise.** Under
  cross-fitting the scoring model never saw that molecule's noise, so it cannot know the individual
  draw — under Gaussian noise that correlation is zero *by construction* and the design forbids any
  other answer. The informative quantities are the cross-fitted residual, whether uncertainty adds
  anything on top of it, and uncertainty against the noise *scale* (the region), which the model can
  learn from other molecules. Ship a permutation null with every number.
- **Question B must subtract the σ = 0 baseline**, and should carry a sham-ceiling check: recompute the
  pattern from the model's *predicted* label; if the correlation is just as strong, the model is
  tracking its own prediction rather than the noise. Computable from the saved predictions.
- **Gaussian is not the control for question B.** Its noise scale is constant, so the correlation is
  undefined rather than zero, and a control has to produce a number. The control is σ = 0 within the
  same strategy. Gaussian stays as question A's leakage check.

## P6. Still open

- **Scope of the QM9 side.** These jobs cover the three experimental datasets only. QM9
  needs a full re-run for unrelated reasons (§K), and its uncertainty arm should ride along
  with that rather than being run twice. Not started.
- **The dose question (§M1) interacts with this.** If σ is later renormalised, these runs
  sit at different absolute noise levels. Both analyses are rank-based so the conclusions
  transfer, but the axis values move.
