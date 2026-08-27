# Audit of the per-sample uncertainty analysis (QM9 noise-robustness paper)

Date: 2026-08-21. Every number below was read from a file in this session; the file and the
code that produces it are named in each case. Nothing is quoted from memory.

---

## 0. Bottom line

The claim "a model's per-sample predictive uncertainty tells you which individual labels were
corrupted" **cannot be supported by this data, and is not what the reported numbers measure.**

Three independent reasons, any one of which is fatal:

1. **The rows are TEST-set molecules.** Noise was intended for training labels only. Because of an
   indexing bug in the Rust preprocessor, test and validation labels also get noise added — but the
   noise value applied to test row *i* is the noise that was drawn for **training** molecule *i*, a
   different molecule. So the "injected noise" attached to each uncertainty row is a number that has
   no causal relation to that molecule at all. Detecting it is impossible in principle.
2. **The "injected noise" column is reconstructed, not recorded**, by regressing the noisy label on
   the clean label and taking residuals. At σ=0 (no noise injected anywhere) this reconstruction
   produces float32 rounding error, and the reported correlations at σ=0 are **larger in magnitude**
   than at σ=0.6. The control fails.
3. **Two of the six strategies are degenerate and one ignores σ entirely** (see §7), so the
   strategy-level pattern in the results is an artefact of the noise generator's parameter file.

The only within-σ correlations that exceed 0.1 come from the two strategies that produce *bimodal*
noise (Outlier, Quantile) — and that is exactly the configuration in which the regression-residual
reconstruction is known to leak label extremity into the "noise" column.

---

## 1. What the code actually computes

### 1.1 `load_uncertainty_data` (`scripts/generate_paper_figures_v2.py`, ~L876)

Globs `results/uncertainty_*_uncertainty_values.csv` **and** `results/*_uncertainty_values.csv`
(so `anova_*` files are swept in too), renames `representation`→`rep`, parses the strategy out of the
filename when the CSV lacks a strategy column, patches BNN model names that were saved as plain
`dnn`/`mlp`, deduplicates on (model, rep, strategy, sigma, iteration, sample_idx) keeping the last
occurrence, applies the BNN name map and the global model exclusion list.

Note: **`sample_idx` is a row position, not a molecule identifier.** `save_uncertainty_values` writes
`'sample_idx': i` for `i in range(len(y_pred_mean))`. There is no SMILES, no dataset index. Nothing in
these files can be traced back to a molecule.

### 1.2 `fix_injected_noise` (~L980–1020)

```
slope, intercept = linregress(y_true_original, y_true_noisy)   # per group
injected_noise   = y_true_noisy - (slope*y_true_original + intercept)
```
Groups are `[c for c in {'model','rep','sigma','iteration'} if c in df.columns]`.
It is called once, at L3997, right after loading.

### 1.3 `WITHIN_SIGMA_LEVELS` (~L1028) and `within_sigma_unc_noise_rho` (~L1031–1090)

`WITHIN_SIGMA_LEVELS = [0.0, 0.3, 0.6]`. For each of those three σ, the function masks rows to that σ
(`np.isclose`, atol 1e-6) intersected with the caller's validity mask, requires `n > 100`, and returns
`spearmanr(uncertainty, |injected_noise|)`. The docstring is honest about why the pooled version is
wrong ("the Kolmar effect… the shared σ ramp… NOT per-sample detection").

### 1.4 What writes the table4 files

- **`table4_uncertainty_metrics{_rep}.csv`** (~L3360–3506): Gaussian ("legacy") strategy only, one
  file per representation. Per model it computes Unc-Error ρ, pooled Unc-Noise ρ, coverage at 1σ/2σ,
  mean uncertainty, mean aleatoric, mean epistemic, plus `Unc-Noise ρ σ={0.0,0.3,0.6}` and the slice
  sizes.
- **`table4_supp_uncertainty_by_strategy_rep.csv`** (~L3509–3616): the same metrics for every
  (strategy × rep × model) cell. 234 rows locally.
- **`table4c_top/bottom_unc_noise_correlations.csv`** (~L3630–3660): reads the supp file back,
  filters to `Strategy == 'Gaussian'`, and takes `nlargest(15)` / `nsmallest(10)` **ranked by
  `Unc-Noise ρ σ=0.3`**, carrying the pooled and all per-σ columns.

---

## 2. The reconstruction problem

**(a) Yes, that is what it does** — see §1.2. And it does it *twice*: the identical regression is
already performed at write time in `scripts/utils.py` `save_uncertainty_values` (L200–260), which
computes `injected_noise` the same way before writing the CSV. `fix_injected_noise` re-derives it from
the same two columns.

**(b) Columns actually in the QM9 uncertainty CSVs** (from `save_uncertainty_values`):
`model, representation, sigma, iteration, file_no, sample_idx, y_pred_mean, y_pred_std_uncalibrated,
y_true_original, y_true_noisy, injected_noise, y_pred_std_calibrated, temperature,
epistemic_uncertainty, aleatoric_uncertainty`.

The two label columns are in **different units**:
- `y_true_original` = the raw QM9 target as read from the mmap (`target_value`, PyG QM9 index 4 =
  HOMO–LUMO gap in eV — the raw Hartree column in `data/QM9/raw/gdb9.sdf.csv` has mean 0.2511,
  min 0.0246, max 0.6221, i.e. 6.83 / 0.67 / 16.93 eV).
- `y_true_noisy` = `processed_target` = `(raw + noise − mean)/std`, where mean and std come from the
  **noisy training labels** (`generate_aggregate_stats`, `rust/src/main.rs` L931–974).

So the exact relation is `y_noisy = (1/std)·y_orig + (noise − mean)/std`. The regression is a way of
recovering `1/std` and `−mean/std` without ever having saved them. It would be unnecessary if the
normalisation constants were written out.

**(c) Yes — σ=0 is contaminated by construction.** `process_and_train.py` L1609 sets
`'noise': s > 0`, so at σ=0 the Rust side adds nothing and `y_noisy = (y_orig − mean)/std` exactly in
float32. The regression therefore fits perfectly and the residual is **pure float32 rounding error**.
Because relative float error scales with magnitude, |residual| is systematically larger for molecules
with extreme gaps — and predictive uncertainty is also larger for extreme molecules. The result, from
`table4_supp_uncertainty_by_strategy_rep.csv`:

| σ | cells with ρ | mean ρ | mean \|ρ\| | median | min | max | frac ρ>0.1 |
|---|---|---|---|---|---|---|---|
| 0.0 | 232 | +0.0326 | **0.0799** | +0.0416 | −0.2438 | +0.3637 | 0.198 |
| 0.3 | 229 | +0.0325 | 0.0455 | +0.0069 | −0.0685 | +0.3884 | 0.127 |
| 0.6 | 230 | +0.0465 | 0.0568 | +0.0135 | −0.1018 | +0.4848 | 0.204 |

**The no-noise control has a larger mean |ρ| (0.0799) than the high-noise condition (0.0568), and a
larger fraction of cells above 0.1 (19.8% vs 20.4% — effectively identical).** Verified. (The prior
figure of 0.0796 is the same quantity; I get 0.0799 from the file as it stands.)

**(d) Where the reconstruction systematically fails.**

- *Value-proportional and heteroscedastic*: the per-molecule σ depends on |y|. A single OLS line
  cannot represent that; part of the noise is absorbed into the slope, and the leftover residual is a
  mixture of noise and a term linear in y. In this dataset the point is largely moot because both
  strategies turn out to be near-homogeneous anyway (§7) — but the method is wrong in general and
  would fail badly on a dataset where they are not.
- *Outlier and Quantile*: these produce **bimodal** noise (a 5%/20% minority gets 30×/20× the σ of the
  rest). A handful of extreme residuals drags the fitted slope away from the true 1/std by
  δ = cov(ε, y)/var(y). For the 80–95% of rows with small ε the residual then becomes
  `≈ ε/std − δ·(y − ȳ)`, so **|reconstructed noise| ≈ |δ|·|y − ȳ|** — a proxy for label extremity.
  Since |·| is taken, the induced correlation with uncertainty is positive **whatever the sign of δ**.
  That is a positively-biased artefact, not a detection signal.
- *σ=0, all strategies*: as in (c), rounding error, magnitude ∝ |y|.

This mechanism predicts exactly the pattern in the data. From the same file, ρ(σ=0.6) > 0.1:

| Strategy | cells ρ>0.1 / cells | mean ρ | mean \|ρ\| | max ρ |
|---|---|---|---|---|
| Outlier | **28 / 42** | +0.1566 | 0.1639 | +0.4848 |
| Quantile | **19 / 40** | +0.0842 | 0.0898 | +0.2413 |
| Gaussian | 0 / 42 | −0.0025 | 0.0178 | +0.0868 |
| Threshold | 0 / 42 | +0.0054 | 0.0188 | +0.0774 |
| Value-Prop. | 0 / 31 | +0.0060 | 0.0135 | +0.0950 |
| Heteroscedastic | 0 / 33 | +0.0137 | 0.0191 | +0.0759 |

0 of 148 across Gaussian/Threshold/Value-Prop./Heteroscedastic. Verified exactly as reported.
At σ=0.3 the same split holds: Outlier 21/41, Quantile 8/39, 0 for the other four (0/149).

Additional defect: `fix_injected_noise` groups by model/rep/sigma/iteration but **not by strategy and
not by file_no**. Different strategies were normalised with *different* mean/std (the normaliser is
computed from that run's noisy training labels), so pooling them into one regression fits a compromise
slope and injects a further linear-in-y term into every strategy's residuals. Also, the group-column
list is built by iterating a Python `set`, so the group key ordering is not deterministic between runs
(harmless for the result, but it means the code's behaviour is not reproducible on its face).

---

## 3. Train or test? — **TEST molecules, with mis-assigned noise**

`scripts/utils.py::save_uncertainty_values` is called from ~20 sites in `models/models.py`
(L1403, 1555, 1793, 2155, 2272, 2494, 2761, 3196, 3397, 4000, 6851, 6977, 7183, …). Every call passes:

```
y_true_original = y_test_original
y_true_noisy    = y_test
y_pred_mean     = y_pred        # predictions on x_test
```

So one row = one **test-set** molecule. Test size is `int(sample_size*0.1)`
(`process_and_train.py` L1884), i.e. 1000 molecules for the `-n 10000` runs in
`slurm_scripts_uncertainty/*.sh` (80/10/10 scaffold split, L630).

### The indexing bug

`rust/src/main.rs` L1134:
```rust
let noise_indices: Vec<usize> = if config.noise {
    (0..config.train_count).collect()      // 0 .. 7999
} else { Vec::new() };
```
The noise map is keyed by **train** index. For the value-dependent strategies the per-index σ is
computed from `read_all_target_values` (L173–191), which reads **only the train mmap**.

`preprocess_data` (L975–1075) then calls `write_data` three times — train, val, test — passing the
**same** `noise_map` each time. Inside `write_data` (L610 onwards):
```rust
for index in 0..data_count {          // restarts at 0 for each file
    ...
    if config.noise {
        if let Some(&artificial_noise) = noise_map.get(&index) {
            property_value += artificial_noise;
```
`data_count` is `train_count` (8000), then `val_count` (1000), then `test_count` (1000). Since
1000 < 8000, **every** test index 0–999 finds an entry and gets noise added — the noise that was drawn
for train molecules 0–999. Same for validation.

Consequences:
- Test labels are corrupted, contradicting the paper. `noise_inject.tex` L325 states "Validation and
  test data remain free of noise" and L378 "applies artificial noise to the training labels while
  preserving the integrity of the test set." That is true of the KIRBy/NoiseInject path (§6) and
  **false of the QM9 Rust path**.
- For the value-dependent strategies, the noise magnitude attached to a test molecule was computed
  from a *different* molecule's label, so it is not even a valid draw from the stated strategy.
- Every QM9 R² at σ>0 is depressed partly by test-label corruption, not only by training corruption.
  This is not confined to the uncertainty analysis — it affects auc_norm, the ANOVA, and every QM9
  figure.
- For the uncertainty claim specifically: the model never sees the test label, and the noise attached
  to it was generated from an unrelated molecule. The true correlation between predictive uncertainty
  and that number is **exactly zero by construction**. Any non-zero ρ is an artefact.

---

## 4. Effect size reality check

Source: `results/paper_figures_v2/table4_supp_uncertainty_by_strategy_rep.csv` (234 rows;
7 models × 8 reps × 6 strategies with gaps).

- **n per correlation**: the slice sizes are 1000 or 10000 (a few 8000/9000, some 0 and 1).
  1000 = one iteration × 1000 test molecules. 10000 = **ten iterations pooled**. See §5.
- **SE of a Spearman ρ**: 1/√(n−3) = **0.0317** at n=1000, **0.0100** at n=10000.
- Uncorrected |ρ| needed for p<0.05: 0.062 (n=1000), 0.020 (n=10000).
- Bonferroni over the ~230 cells: 0.117 (n=1000), 0.037 (n=10000).

Fisher-z tests over the 229/230 cells with n>100:

| σ | raw p<0.05 | BH q<0.05 | Bonferroni p<0.05 |
|---|---|---|---|
| 0.3 | 76/229 | 69 | 52 |
| 0.6 | 88/230 | 78 | 62 |

BH-significant cells at σ=0.6 by strategy: Outlier 35, Quantile 32, Threshold 4, Gaussian 3,
Value-Prop. 2, Heteroscedastic 2. Outside Outlier/Quantile the BH-significant |ρ| values span
**0.026 to 0.095** — i.e. they are "significant" only because n is 10 000, and they are the same size
as the σ=0 rounding-error correlations (mean |ρ| 0.0799). Multiple-comparison correction does not
rescue them; passing a significance test at n=10 000 is not evidence of a usable effect when the
negative control passes the same test.

**Practical utility.** Model a chemist triaging the top decile of predicted uncertainty and asking how
often those molecules are in the top decile of |injected noise| (bivariate-normal approximation,
Pearson r = 2·sin(πρ_s/6)):

| Spearman ρ | precision at top-10% uncertainty | enrichment over random |
|---|---|---|
| 0.05 | 11.7% | 1.17× |
| 0.16 | 15.9% | 1.59× |
| 0.26 (the pooled mean) | 20.3% | 2.03× |
| 0.39 (GP pooled) | 27.0% | 2.70× |
| 0.56 (best pooled in the paper) | 37.5% | 3.75× |

Blunt answer: **ρ = 0.05 is useless** — you would re-measure 100 compounds to find 12 bad ones instead
of 10. ρ = 0.26 is a 2× enrichment, which is marginal even if it were real; and it is not real, because
the pooled value is dominated by the σ ramp (the script's own docstring says so) and by test-label
corruption. Nothing here supports telling a chemist which label is bad.

---

## 5. σ range, and pooling across iterations

**σ levels actually run**: `0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0` — 11 levels
(`slurm_scripts_uncertainty/unc_qrf_legacy.sh` L24 and every sibling script). `WITHIN_SIGMA_LEVELS`
uses **3 of 11**. The paper's own robustness work already prefers σ=0.8 as the headline stress level
(`DISCUSSION_TRACKER.md` L67), so the uncertainty analysis and the robustness analysis condition at
different noise levels.

**Would higher σ help?** No local file carries ρ at σ=0.8 or 1.0 — those columns are simply not
computed, so I cannot quote a number. What the local files *do* show is the direction: Outlier goes
+0.124 (σ=0.3) → +0.157 (σ=0.6); Quantile +0.058 → +0.084; the other four stay flat within ±0.02.
That is precisely the signature of the slope-bias artefact, whose size scales with sd(ε) and therefore
with σ. **Conditioning higher will make the artefact bigger and will not create a signal in the four
homogeneous strategies.** Adding σ=0.8 and 1.0 is worth doing, but as a diagnostic of the artefact,
not as a stronger test of the claim.

**Pooling across iterations — confirmed.** `within_sigma_unc_noise_rho` masks on σ only; there is no
iteration term in the mask, and the caller (`model_data` = all rows for one model within one
strategy×rep subset) does not split by iteration either. The n=10000 slices are 10 iterations × 1000
test molecules stacked. Each iteration is a different scaffold split, a different model fit, and a
different noise realisation, so between-run differences in mean uncertainty are folded into the
correlation — the same defect as pooling across σ, one level down. The n=1000 cells (single iteration)
are the only clean ones, and they are also the ones with the weakest power (SE 0.032).

Note also that the same molecule can appear in several iterations, so the 10 000 points are not
independent and the nominal p-values are anticonservative on top of everything else.

---

## 6. Validation datasets — the claim is QM9-only

Per-sample uncertainty files that exist locally under `results/validation_full/`:

| File | rows | σ levels | columns |
|---|---|---|---|
| `openadmet_logd/GP_PDV_uncertainty_values.csv` | 11088 | 0.0–1.0, 11 levels | sigma, sample_idx, y_true, y_pred, uncertainty |
| `openadmet_logd/QRF_ECFP4_…`, `QRF_MHGGNNpretrained_…`, `QRF_PDV_…`, `QRF_SNS_…` | — | — | same |
| `openadmet_caco2/QRF_ECFP4_uncertainty_values.csv` | 4752 | 0.0–1.0 | same |

Six files, two datasets, two models (GP on PDV; QRF on four reps). **No hERG per-sample uncertainty
data at all.** No `injected_noise`, no `y_true_original`, no molecule id.

`results/paper_figures_v2/deep_validation_uncertainty_rfqrf.csv` is not per-sample data — it is three
rows of RF-vs-QRF auc_norm comparisons.

`results/paper_figures_v2/table_validation_uncertainty.csv` — the file
`create_validation_uncertainty_table` (L1244–1314) is supposed to write — **does not exist** in the
output directory. Either the last regeneration ran without `--validation-dir`, or the loader found
nothing.

The script's own docstring for `load_validation_uncertainty` (L1173–1190) already says it: "there is no
per-sample `injected_noise`, so the within-σ noise-tracking analysis cannot be run on validation data."

Importantly, the validation pipeline is the **correct** one:
`KIRBy/tests/alternative_data_noise_robustness.py` L756–760 and L791–795 do
`y_noisy = y_train if sigma == 0.0 else injector.inject(y_train, sigma)` — **training labels only**;
the saved `y_true` is the clean test label. So the two pipelines are not doing the same experiment,
and the QM9 vs validation comparisons in the paper compare noisy-test against clean-test results.

Also worth noting: the validation uncertainty files are averaged across CV folds
(`groupby(['sigma','sample_idx']).agg({'uncertainty':'mean'})`, L1140–1145), so even the surviving
per-sample values are fold-averages.

**Conclusion: every per-sample uncertainty-vs-noise number in the paper is QM9 HOMO–LUMO gap only,
and rests entirely on the buggy pipeline.**

---

## 7. Other implementation defects found

### 7.1 Errors and coverage are computed against the NOISY label

`generate_paper_figures_v2.py` ~L3400 (Table 4) and ~L3436 (coverage), and again at ~L3546/L3568 for
Table 4b, prefer `y_true_noisy` over `y_true_original`, with the comment "Use y_true_noisy (normalized
space) to match y_pred_mean (normalized space)". The fallback to `y_true_original` is explicitly
flagged as scale-mismatched.

This is a workaround for the missing normalisation constants, and it is the **wrong target** for both
metrics being claimed:
- *Unc-Error ρ* ("does the model's uncertainty track its own error") should use the **clean** label.
  Measured against the noisy label, the error contains the injected noise itself, so any model whose
  uncertainty grows with σ scores well for a reason unrelated to error tracking.
- *Coverage at 1σ/2σ* is a calibration statistic. Against corrupted test labels it degrades with σ for
  reasons that have nothing to do with calibration. Since ECE was removed (2026-08-19), coverage now
  carries calibration alone in the paper — on a corrupted reference.

The correct fix is to save the normalisation `mean`/`std` (or to write `y_true_original` already
normalised), then compute errors and coverage against the clean label in normalised space.

### 7.2 `value_proportional` ignores the σ sweep entirely

`scripts/noise_strategy_params.json` sets `"base_sigma": 0.1`. `rust/src/main.rs` L1167 reads
`base_sigma: params.get("base_sigma")...unwrap_or(sigma)` — because the key **is present**, the
`--sigma` argument is never used. The per-molecule σ is
`0.1·(1 + 0.05·|y|) ≈ 0.134 eV` at every requested σ from 0.1 to 1.0.

So the value-proportional noise axis is a step function: 0 at σ=0 (noise is disabled entirely when
`s == 0`), then a constant ≈0.134 eV for all σ>0. **Every valprop robustness curve, auc_norm and ANOVA
cell is measuring a flat line.** This is not confined to the uncertainty section.

### 7.3 `threshold` is degenerate on QM9

`high_threshold: 1.0`, `low_threshold: -1.0`, applied to the **raw** target. PyG QM9 target index 4 is
the gap in eV: from `data/QM9/raw/gdb9.sdf.csv` the Hartree gap has min 0.0246 → 0.669 eV, mean
0.2511 → 6.83 eV. Essentially every molecule clears the 1.0 threshold, so all of them receive
`high_sigma = 2·σ`. "Threshold" is homogeneous Gaussian noise at 2σ.

### 7.4 `heteroscedastic` is near-degenerate on QM9

`σ_i = √(σ²(0.1 + 0.05·|y|))`. With mean |y| = 6.83 eV and sd 1.29 eV, σ_i ≈ 0.664·σ with roughly ±7%
variation across molecules. Effectively homogeneous Gaussian at 0.66σ.

Taken together, 7.2–7.4 mean that **four of the six "strategies" produce homogeneous or
near-homogeneous noise** (Gaussian, threshold, valprop, hetero) and only two (Outlier: 5% at 3σ vs 95%
at 0.1σ; Quantile: 20% at 2σ vs 80% at 0.1σ) produce real per-molecule heterogeneity. That is the
whole of the "strategy effect" in the uncertainty results, and it is a property of the parameter file
rather than of the models.

### 7.5 The local table4 outputs are stale relative to the script

`table4_supp_uncertainty_by_strategy_rep.csv` and both `table4c_*.csv` still carry an **ECE** column.
`grep -c ECE scripts/generate_paper_figures_v2.py` returns **0** — ECE was removed. So the CSVs in
`results/paper_figures_v2/` were produced by an older version of the script and do not correspond to
what a re-run would produce.

### 7.6 The paper's Table 7 and the current `table4c` no longer agree

`table4c_top_unc_noise_correlations.csv` is now sorted by `Unc-Noise ρ σ=0.3`. Reproducing the
paper's ordering requires sorting the Gaussian rows of the supp table by **pooled** ρ, which gives:

| Model | Rep | pooled ρ | ρ σ=0.3 | ρ σ=0.6 |
|---|---|---|---|---|
| GP | SNS | 0.5354 | **−0.0430** | **−0.0427** |
| GP | Morgan | 0.5269 | −0.0128 | −0.0153 |
| NGBoost | SNS | 0.4722 | +0.0761 | +0.0868 |
| NGBoost | continuous PDV | 0.4720 | +0.0184 | +0.0230 |
| NGBoost | PDV (binary) | 0.4434 | +0.0001 | +0.0064 |
| NGBoost | Morgan | 0.4386 | +0.0162 | +0.0274 |
| BNN-α | SNS | 0.4173 | −0.0042 | −0.0016 |
| BNN-β | SNS | 0.4001 | +0.0033 | −0.0084 |
| BNN-β | Morgan | 0.3802 | +0.0084 | −0.0033 |
| BNN-β | continuous PDV | 0.3759 | −0.0504 | −0.0438 |

That is the paper's Table 7 row-for-row. **Every headline number in it collapses to within ±0.09 of
zero once σ is conditioned on, and the top row goes negative.**

Two discrepancies against the printed table:
- Paper prints **0.56** for GP/SNS; the Gaussian row in the supp file is **0.5354**. 0.560017 is the
  **Threshold**-strategy GP/SNS value. Either the paper row was taken from the wrong strategy or from
  an older run. It must be re-checked before submission.
- The paper's captions describe Unc-Noise ρ as "the per-sample Spearman correlation between predicted
  uncertainty and injected noise magnitude" (L508, L512). It is the σ-pooled correlation, which the
  script itself annotates as "population-level / Kolmar trend — comparison only… do NOT report as
  per-sample".

### 7.7 GP is the sharpest illustration

All 18 GP rows in the supp table: mean pooled ρ **+0.3937**, mean ρ at σ=0.3 **−0.0085**, mean ρ at
σ=0.6 **−0.0117**. Verified. GP has a single global observation-noise term — which the paper's own
mechanistic paragraph (L540) says should *not* absorb individual residuals. The within-σ result agrees
with the mechanism; the pooled result does not. The pooled number is the σ ramp.

Excluding Outlier/Quantile, within-σ ρ at σ=0.6 by model:
BNN-α +0.0003, BNN-β +0.0035, GP −0.0327, NGBoost +0.0408, QRF +0.0079, VBLL-α +0.0059, VBLL-β +0.0046.
There is no model ranking to report here.

### 7.8 Data gaps

`results/paper_figures_v2/uncertainty_gaps.csv` (21 data rows) lists missing σ levels, e.g.
`dnn_bnn_full/continuous_pdv/legacy` missing σ=0.4, `dnn_vbll/continuous_pdv/valprop` missing σ=0.7,
`gauche/ecfp4/legacy` missing σ=0.7. Several within-σ cells have n = 0 or 1 (4 cells at σ=0.6 with
n=0), so the "234 cells" are not a complete design.

---

## 8. What has to be re-run or re-saved on the server to settle this

### 8.1 Fix the Rust preprocessor first (this is not optional)

`rust/src/main.rs`, `write_data`: noise must be applied to the training file only. Concretely, add a
flag or a split-name argument to `write_data` and gate the `noise_map.get(&index)` lookup on it, or
key the noise map by a global dataset index rather than a per-file position. Then rebuild and re-run.

Until this is fixed, **every QM9 number in the paper at σ>0 is measured against corrupted test
labels** — not just the uncertainty section.

### 8.2 Columns that must be added to `save_uncertainty_values` (`scripts/utils.py`)

| Column | Why |
|---|---|
| `norm_mean`, `norm_std` | Kills the regression reconstruction outright. Everything becomes exact. |
| `y_true_original_normalized` = `(y_orig − norm_mean)/norm_std` | Lets error and coverage be computed against the **clean** label in the same space as `y_pred_mean`. |
| `strategy` | Currently inferred from the filename; makes grouping unambiguous. |
| `split` (`train`/`val`/`test`) | Makes the train/test question answerable from the file. |
| `mol_index` or `smiles` | `sample_idx` is a row position; there is currently no way to link a row to a molecule or across iterations. |
| `injected_noise_true` | The **actual** ε the generator added, written by the Rust side, not reconstructed. This is the single most important addition. |

Getting `injected_noise_true` out of Rust means dumping the noise map: write
`noise_map_{file_no}.csv` with `split, index, epsilon` next to the mmaps, and have
`process_and_train.py` join it onto the uncertainty rows.

### 8.3 Which experiment actually tests the claim

The per-sample "which label is corrupted" question is a **training-set** question. It requires
per-sample uncertainty on **training** molecules whose labels were corrupted — currently never saved.
Add a `save_uncertainty_values(...)` call on `x_train` alongside the existing test-set call, with the
true ε attached. Without that, the claim cannot be tested at all, no matter how the correlation is
computed.

### 8.4 Fix the strategy parameters

- `scripts/noise_strategy_params.json`: **remove `"base_sigma": 0.1`** from `value_proportional` so
  the `--sigma` sweep is honoured, or state clearly in the paper that valprop is a fixed-magnitude
  strategy. Then re-run every valprop job.
- `threshold`: `high_threshold: 1.0` / `low_threshold: -1.0` are meaningless for a target with a
  6.8 eV mean. Set them from quantiles of the actual target, or drop the strategy.
- `heteroscedastic`: `alpha_multiplier`/`beta_multiplier` give ±7% variation. Either widen them so the
  strategy is genuinely heteroscedastic, or report it as near-Gaussian.

### 8.5 Analysis-side changes

- Compute the within-σ ρ **inside a single (model, rep, strategy, σ, iteration) cell**, then report the
  per-iteration values (or their spread), never a pooled-across-iterations correlation.
- Extend `WITHIN_SIGMA_LEVELS` to all 11 levels, or at minimum `[0.0, 0.3, 0.6, 0.8, 1.0]` to align
  with the σ=0.8 choice in `DISCUSSION_TRACKER.md`.
- Keep σ=0 as a reported negative control in every table. It currently fails, and that failure is the
  most informative single number in the analysis.
- Group `fix_injected_noise` by strategy and file_no as well (or delete the function once the
  normalisation constants are saved).
- Compute Unc-Error ρ and coverage against the clean normalised label.
- Report the pooled ρ only if it is explicitly labelled as the population-level σ trend
  (a Kolmar-style replication), never as per-sample detection.

### 8.6 Server files that would settle the remaining open questions

Only two things cannot be determined from local files:

1. **Do the QM9 test labels really differ from the clean labels at σ>0?**
   Settled by one file: any
   `/data/stat-cadd/scat9264/qsar_qm_models/results/uncertainty_legacy_pdv_qrf_uncertainty_values.csv`.
   Check whether, within one (sigma, iteration) group, `y_true_noisy` is a *perfect* linear function of
   `y_true_original` (residual ~1e-7 ⇒ test is clean) or has residuals of order σ/std (⇒ test is
   corrupted). The code says corrupted; this confirms it empirically in one command.
2. **ρ at σ=0.8 and 1.0.** Not computed anywhere. Needs the analysis change in 8.5 and a re-run of
   `sbatch run_figures_v2.sh`.

A copy-paste check for (1):

```bash
cd /data/stat-cadd/scat9264/qsar_qm_models/results
python - <<'PY'
import pandas as pd, numpy as np
from scipy.stats import linregress
d = pd.read_csv('uncertainty_legacy_pdv_qrf_uncertainty_values.csv')
for (s, it), g in d.groupby(['sigma','iteration']):
    m = np.isfinite(g.y_true_original) & np.isfinite(g.y_true_noisy)
    sl, ic, r, _, _ = linregress(g.y_true_original[m], g.y_true_noisy[m])
    res = g.y_true_noisy - (sl*g.y_true_original + ic)
    print(f"sigma={s} iter={it} n={m.sum()} R2={r**2:.8f} resid_sd={res.std():.6g}")
PY
```
If `resid_sd` is ~1e-7 at every σ, the test labels are clean and only the reconstruction is broken.
If it grows with σ, the test labels are corrupted and §3 applies in full.

---

## 9. Summary of verification against the prior findings

| Prior finding | Verdict | Value from file |
|---|---|---|
| pooled ρ mean 0.2591; within-σ 0.0325 (σ=0.3), 0.0465 (σ=0.6) | **Confirmed** | pooled mean 0.2591 (n=234); within-σ means 0.0325 and 0.0465 |
| σ=0 mean \|ρ\| 0.0796 > 0.0568 at σ=0.6 | **Confirmed** (I read 0.0799 vs 0.0568) | mean \|ρ\|: 0.0799 / 0.0455 / 0.0568 at σ = 0 / 0.3 / 0.6 |
| ρ(σ=0.6)>0.1: Outlier 28/42, Quantile 19/40, 0/148 elsewhere | **Confirmed exactly** | 28/42, 19/40; 0 across 42+42+31+33 = 148 |
| GP within-σ slightly negative despite pooled ~0.39 | **Confirmed** | GP (18 rows): pooled +0.3937, σ=0.3 −0.0085, σ=0.6 −0.0117 |

Files used: `results/paper_figures_v2/table4_supp_uncertainty_by_strategy_rep.csv`,
`table4c_top_unc_noise_correlations.csv`, `table4c_bottom_unc_noise_correlations.csv`,
`uncertainty_gaps.csv`, `deep_validation_uncertainty_rfqrf.csv`, `deep_qm9_uncertainty_by_*.csv`;
`scripts/generate_paper_figures_v2.py`, `scripts/utils.py`, `scripts/process_and_train.py`,
`scripts/noise_strategy_params.json`, `models/models.py`, `rust/src/main.rs`;
`slurm_scripts_uncertainty/unc_qrf_legacy.sh`; `data/QM9/raw/gdb9.sdf.csv`;
`results/validation_full/**/[A-Z]*_uncertainty_values.csv`;
`/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py`; `noise_inject.tex`;
`DISCUSSION_TRACKER.md`.
