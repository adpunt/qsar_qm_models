# PDV Binarization Investigation Report

## Background

For over a year, the QSAR/QM noise robustness study used a Physiochemical Descriptor Vector (PDV) representation that was inadvertently binarized. The PDV consists of 200 RDKit molecular descriptors computed via `MolecularDescriptorCalculator` — these are continuous-valued properties like molecular weight, LogP, partial charges, surface area bins, and functional group counts. During the mmap serialization step in `process_and_train.py`, the following transformation was applied:

```python
pdv_binary = (pdv > 0).astype(np.uint8)
pdv_packed = np.packbits(pdv_binary, bitorder='little')
```

This converts every descriptor value into a single bit: 1 if the value is positive, 0 otherwise. A `continuous_pdv` code path exists but was affected by a separate bug (`astype(np.uint8)` cast at parse time, now fixed).

The question: why did this "broken" binary representation work well enough to go unnoticed for a year?

## What Binarization Does to the 200 Descriptors

Analysis on 10,000 QM9 molecules reveals that the 200 descriptors fall into five categories under `(value > 0)` binarization:

| Category | Count | What happens | Examples |
|---|---|---|---|
| **ALWAYS_1** | 13 | Always positive for any molecule. Becomes constant=1 after binarization. Dead feature. | ExactMolWt, MolWt, HeavyAtomCount, MolMR, LabuteASA, qed |
| **ALWAYS_0** | 30 | Always zero or negative on QM9. Becomes constant=0. Dead feature. | HallKierAlpha, MinPartialCharge, fr_C_S, fr_SH, fr_azide, SlogP_VSA7 |
| **MOSTLY_1** | 29 | >90% positive. Low discriminating power but not dead. | Chi0-Chi4, BertzCT, TPSA, Kappa1-3, FractionCSP3, SlogP_VSA2 |
| **MOSTLY_0** | 52 | <10% positive. Sparse binary features. | Most fr_* counts (fr_ester, fr_halogen, fr_epoxide, fr_morpholine, ...) |
| **INFORMATIVE** | 76 | 10-90% positive. Highest discriminating power. | EState_VSA bins, PEOE_VSA bins, SMR_VSA bins, fr_ether, fr_C_O, fr_NH0/1/2, NumAromaticRings |

**After binarization, 43 features are constant (wasted), and ~157 carry some discriminating information.**

The binary PDV is effectively a **functional group fingerprint + coarse surface area fingerprint**:

1. The ~52 sparse `fr_*` fragment counts become presence/absence indicators for functional groups — conceptually identical to MACCS keys or pharmacophore fingerprints, which are well-established representations.

2. The ~50 VSA/EState/PEOE binned descriptors (which partition molecular surface area by polarity, refractivity, or electrotopological state) become "does this molecule have *any* surface area in this bin?" — a coarse but meaningful chemical signal.

3. The ~29 mostly-positive features (FractionCSP3, topological indices, TPSA) become near-constant, contributing little but not nothing.

## Experiment Design

**Dataset**: QM9 homo-lumo gap, n=10,000, scaffold split (80/10/10)
**Model**: Random Forest (500 trees, sqrt features, min_samples_leaf=2)
**Noise**: Legacy Gaussian injection via NoiseInject at sigma = [0.0, 0.2, 0.25, 0.3, 0.5]
**Metric**: Baseline R² (sigma=0) and NDS (slope of R² vs sigma; more negative = less robust)
**Repetitions**: 5 iterations with seeds 42-46

### PDV Variants Tested

| Variant | Description |
|---|---|
| `binary_gt0` | `(pdv > 0)` — the version used in all main experiments |
| `continuous_raw` | Raw float descriptors, no normalization |
| `continuous_zscore` | Z-score normalized per feature (mean=0, std=1, fit on train) |
| `continuous_minmax` | Min-max scaled to [-1, 1] (fit on train) |
| `binary_mean` | `(pdv > per-feature mean)` — threshold at training set mean |
| `binary_median` | `(pdv > per-feature median)` — threshold at training set median |
| `binary_nonzero` | `(pdv != 0)` — captures negative values as 1 (differs from >0 for LogP, charges, etc.) |
| `binary_gt0_no_dead` | Same as binary_gt0 but drops 46 constant columns (154 features remain) |

## Results

| Variant | # Features | Baseline R² | NDS (slope) |
|---|---|---|---|
| continuous_raw | 200 | 0.7987 +/- 0.0024 | -5.79 +/- 0.83 |
| continuous_zscore | 200 | 0.7987 +/- 0.0025 | -5.76 +/- 0.82 |
| continuous_minmax | 200 | 0.7987 +/- 0.0022 | -5.77 +/- 0.83 |
| **binary_mean** | **200** | **0.7771 +/- 0.0011** | **-4.95 +/- 0.68** |
| binary_median | 200 | 0.7685 +/- 0.0015 | -5.13 +/- 0.65 |
| binary_gt0_no_dead | 154 | 0.7671 +/- 0.0014 | -5.45 +/- 0.72 |
| binary_gt0 | 200 | 0.7601 +/- 0.0011 | -5.13 +/- 0.70 |
| binary_nonzero | 200 | 0.7590 +/- 0.0021 | -4.92 +/- 0.60 |

## Key Findings

### 1. Continuous PDV is better at baseline, but binary is more noise-robust

Continuous variants achieve R² = 0.799 vs binary_gt0 at R² = 0.760 — a 0.039 gap. However, binary_gt0 has better noise robustness: NDS = -5.13 vs -5.79 for continuous. The information loss from binarization acts as implicit regularization that helps under noisy training conditions.

### 2. Normalization makes no difference for RF on continuous PDV

All three continuous variants (raw, z-score, min-max [-1,1]) produce identical baseline R² (0.799) and nearly identical NDS. This is expected: Random Forests are scale-invariant — they use threshold-based splits, so linear rescaling of features cannot change the model. Normalization would only matter for distance-based or gradient-based models (SVM, DNN, etc.).

### 3. Mean-threshold binarization is the best binary variant

`binary_mean` (threshold at per-feature training mean) achieves R² = 0.777, splitting the difference between continuous (0.799) and binary_gt0 (0.760), while retaining the best NDS of any variant (-4.95). This makes sense: thresholding at the mean rather than zero captures more nuanced splits for the always-positive descriptors (MolWt, TPSA, etc.) that are dead under >0 binarization.

### 4. The 46 dead features don't hurt (much)

`binary_gt0_no_dead` (154 features) performs comparably to `binary_gt0` (200 features): R² 0.767 vs 0.760. The 46 constant features are simply ignored by RF. They add noise to feature sampling (sqrt(200) vs sqrt(154) candidates per split) but the effect is minor.

### 5. Feature importance tells the story

For **binary_gt0**, the top features are all INFORMATIVE category: SMR_VSA10, SMR_VSA7, SlogP_VSA6, VSA_EState2, fr_C_O_noCOO. These are the surface area bins and functional group indicators — the features that actually carry discriminating information after binarization.

For **continuous_raw**, the top features shift: FractionCSP3 (0.086) and HallKierAlpha (0.078) dominate — both are ALWAYS_0 or MOSTLY_1 under binarization (i.e., dead or low-info). The continuous representation unlocks these features that binarization kills.

## Why Binary PDV "Worked"

1. **The >0 threshold is semantically meaningful for most descriptors.** For functional group counts (fr_*), >0 means "present" — a legitimate molecular fingerprint. For VSA bins, >0 means "has surface area in this property range." These are reasonable chemical questions.

2. **RF is robust to dead features.** The 43 constant features are harmlessly ignored. RF's feature subsampling means they rarely interfere with useful splits.

3. **Information loss ≈ regularization.** The ~4% R² gap (0.76 vs 0.80) was small enough to go unnoticed, and the binary representation is genuinely more stable under noise injection.

4. **QM9 is "easy."** Small organic molecules with limited chemical diversity. A coarse fingerprint captures most of the relevant structure-property relationships.

## Deeper Dive: Model Comparison

To test whether the RF findings generalize across model families, we ran RF, XGBoost, SVM, BNN Full, and VBLL on three representative PDV variants: `continuous_raw`, `binary_mean`, and `binary_gt0_no_dead`. All models used Optuna-tuned hyperparameters from the main pipeline (`scripts/results/fig6*.json`), and Y-targets were z-score normalized (matching the Rust pipeline). Single iteration (seed=42).

**Pipeline-matching setup:**
- **Y normalization**: z-score `(y - mean) / std`, predictions denormalized before scoring
- **X normalization**: z-score for continuous PDV (SVM/NN need it); no normalization for binary PDV (already {0,1})
  - **Note**: This investigation applied z-score X normalization for `continuous_raw` (fit on train, applied to train/test/val), which the original pipeline did NOT do for `continuous_pdv`. The pipeline has since been fixed to match this investigation's preprocessing (float32 precision + z-score X normalization). All v2 results below reflect z-score-normalized continuous features.
- **RF**: 1552 trees, max_features=1.0, min_samples_split=12, bootstrap=True
- **XGBoost**: 1808 trees, subsample=0.67, max_depth=10, learning_rate=0.037
- **SVM**: poly kernel degree 2, C=2.04, coef0=1.91, gamma='scale'
- **BNN Full**: DNN [64, 32] tanh, all layers BayesLinear (torchbnn), prior N(0, 0.1), MSE loss, 100 MC samples
- **VBLL**: DNN [64, 32] tanh, all layers VBLLLayer, ELBO loss (NLL + KL/n), learned noise, 100 MC samples
- **Training**: Adam lr=0.001, early stopping (patience=20, tol=0.0001), max 500 epochs, batch size 256

### Results

| Model | PDV Variant | # Features | Baseline R² | NDS |
|---|---|---|---|---|
| **XGBoost** | continuous_raw | 200 | **0.8949** | -27.19 |
| RF | continuous_raw | 200 | 0.8885 | -2.88 |
| SVM | continuous_raw | 200 | 0.8772 | -4.11 |
| BNN Full | continuous_raw | 200 | 0.8303 | -0.53 |
| VBLL | continuous_raw | 200 | 0.7744 | -0.47 |
| RF | binary_mean | 200 | 0.8644 | -2.95 |
| XGBoost | binary_mean | 200 | 0.8550 | -35.04 |
| BNN Full | binary_mean | 200 | 0.8434 | -0.60 |
| SVM | binary_mean | 200 | 0.8355 | -1.58 |
| VBLL | binary_mean | 200 | 0.7889 | -0.52 |
| RF | binary_gt0_no_dead | 154 | 0.8576 | -4.65 |
| XGBoost | binary_gt0_no_dead | 154 | 0.8456 | -47.45 |
| BNN Full | binary_gt0_no_dead | 154 | 0.8286 | -0.77 |
| SVM | binary_gt0_no_dead | 154 | 0.8018 | -0.87 |
| VBLL | binary_gt0_no_dead | 154 | 0.7613 | -0.66 |

### Per-sigma detail (R² at each noise level)

| Model | PDV Variant | σ=0.0 | σ=0.2 | σ=0.25 | σ=0.3 | σ=0.5 |
|---|---|---|---|---|---|---|
| XGBoost | continuous_raw | 0.895 | -1.171 | -2.158 | -3.568 | -12.764 |
| RF | continuous_raw | 0.889 | 0.549 | 0.413 | 0.289 | -0.558 |
| SVM | continuous_raw | 0.877 | -0.042 | -0.273 | -0.508 | -1.166 |
| BNN Full | continuous_raw | 0.830 | 0.603 | 0.522 | 0.631 | 0.551 |
| VBLL | continuous_raw | 0.774 | 0.675 | 0.651 | 0.624 | 0.538 |
| XGBoost | binary_mean | 0.855 | -1.922 | -3.852 | -5.362 | -16.676 |
| RF | binary_mean | 0.864 | 0.537 | 0.390 | 0.236 | -0.609 |
| SVM | binary_mean | 0.836 | 0.543 | 0.455 | 0.379 | 0.045 |
| BNN Full | binary_mean | 0.843 | 0.619 | 0.608 | 0.568 | 0.544 |
| VBLL | binary_mean | 0.789 | 0.651 | 0.633 | 0.613 | 0.525 |
| XGBoost | binary_gt0_no_dead | 0.846 | -2.875 | -5.093 | -7.354 | -22.932 |
| RF | binary_gt0_no_dead | 0.858 | 0.384 | 0.166 | -0.081 | -1.468 |
| SVM | binary_gt0_no_dead | 0.802 | 0.628 | 0.577 | 0.530 | 0.371 |
| BNN Full | binary_gt0_no_dead | 0.829 | 0.592 | 0.593 | 0.512 | 0.443 |
| VBLL | binary_gt0_no_dead | 0.761 | 0.624 | 0.578 | 0.521 | 0.441 |

### Key Findings

#### 1. All models work well with proper tuning and Y normalization

With pipeline-matching hyperparameters and Y-target normalization, every model achieves competitive baseline R²: XGBoost (0.895), RF (0.889), SVM (0.877), BNN Full (0.830), VBLL (0.774). This is consistent with the main pipeline results where binary PDV produces R² = 0.81–0.87 across all 14 ANOVA models.

An initial (v1) run without Y normalization and with default hyperparameters produced catastrophic failures for SVM (R²=0.14), BNN Full (R²=0.04), and VBLL (R²=-0.22). The two critical missing pieces were:
- **Y normalization**: QM9 homo-lumo gap has std=0.046 — without normalization, SVR's default epsilon=0.1 exceeds 2× the target standard deviation, and neural net gradients are tiny.
- **Tuned hyperparameters**: SVM on PDV uses a poly kernel (degree 2) rather than RBF; the DNN architecture for PDV is [64, 32] with tanh rather than the larger architectures used for other representations.

#### 2. XGBoost has the best baseline but is catastrophically non-robust

XGBoost achieves the highest baseline R² (0.895) but has extreme NDS values: -27 on continuous, -35 on binary_mean, -47 on binary_gt0_no_dead. At σ=0.2 it already produces negative R² (predicting worse than the mean). By σ=0.5, R² reaches -12 to -23. This is classic overfitting: XGBoost's aggressive gradient boosting memorizes training data so thoroughly that even moderate noise causes catastrophic extrapolation.

The pattern worsens with less informative representations: binary_gt0_no_dead (NDS=-47) is worse than binary_mean (-35), which is worse than continuous_raw (-27). Fewer informative features mean XGBoost overfits the remaining ones harder.

#### 3. BNN Full and VBLL are the most noise-robust — and it's genuine robustness

BNN Full (NDS ≈ -0.5 to -0.8) and VBLL (NDS ≈ -0.5 to -0.7) maintain R² ≈ 0.44–0.55 even at σ=0.5, where RF and XGBoost have already collapsed. Unlike the v1 results where near-zero NDS was an artifact of models that couldn't learn, these models achieve strong baselines (R² = 0.76–0.84) and degrade gracefully. The Bayesian posterior regularization and learned noise variance provide genuine noise resilience.

BNN Full slightly outperforms VBLL at baseline (0.83 vs 0.77 on continuous_raw) but their noise robustness profiles are similar. On binary_mean, BNN Full achieves the best balance: R² = 0.843 baseline with NDS = -0.60.

#### 4. SVM with tuned poly kernel is surprisingly robust on binary features

SVM (poly degree 2, C=2.04) shows an interesting pattern: on binary_gt0_no_dead, it achieves NDS=-0.87 while maintaining R²=0.802 — more robust than RF (NDS=-4.65) despite a lower baseline. On binary_mean, SVM reaches NDS=-1.58 with R²=0.836. The polynomial kernel on binary features creates implicit feature interactions (products of presence/absence indicators) that are more stable than the individual threshold splits used by tree methods.

On continuous features, SVM is less robust (NDS=-4.11), suggesting the poly kernel's stability advantage is specific to discrete/binary input spaces.

#### 5. The baseline vs robustness tradeoff is model-dependent

The RF finding that "binary is more noise-robust" does **not** generalize uniformly:

| Model | Best baseline variant | Most robust variant | Tradeoff? |
|---|---|---|---|
| RF | continuous_raw (0.889) | continuous_raw (-2.88) | No — continuous wins both |
| XGBoost | continuous_raw (0.895) | continuous_raw (-27.19) | No — continuous wins both (all terrible) |
| SVM | continuous_raw (0.877) | binary_gt0_no_dead (-0.87) | **Yes** — big tradeoff |
| BNN Full | binary_mean (0.843) | continuous_raw (-0.53) | Mild — binary_mean close (-0.60) |
| VBLL | binary_mean (0.789) | continuous_raw (-0.47) | Mild — binary_mean close (-0.52) |

For RF (with tuned parameters), continuous_raw actually wins on *both* baseline and robustness — the 5-iteration RF-only experiment showed a binary robustness advantage with default RF (500 trees, sqrt features), but the tuned RF (1552 trees, max_features=1.0) exploits continuous features more effectively.

For SVM, there is a genuine tradeoff: continuous gives 0.877 baseline but NDS=-4.11, while binary_gt0_no_dead gives 0.802 baseline but NDS=-0.87. The binary representation's regularizing effect is strongest for SVM.

For Bayesian models (BNN Full, VBLL), the differences are small — their built-in uncertainty quantification provides noise resilience regardless of representation.

## Bug Fix Note

A separate bug was found and fixed in `process_and_train.py` line 1230: `x_data = np.vstack(x_data).astype(np.uint8)` was applied unconditionally, which would truncate `continuous_pdv` float16 values to uint8 (0-255 with overflow). This has been fixed to use `float32` when `rep == "continuous_pdv"`. Any `continuous_pdv` experiments run before this fix have corrupted data.

## Methodology Note: v1 vs v2 Model Comparison

An initial model comparison (v1) was run without Y-target normalization and with default (untuned) hyperparameters. This produced near-zero R² for SVM, BNN Full, and VBLL — results that were misleading. The v1 results are superseded by the v2 results above, which match the main pipeline's preprocessing:

| Issue | v1 (incorrect) | v2 (correct) |
|---|---|---|
| Y normalization | None | z-score (mean/std from train) |
| SVM kernel | RBF, C=1.0 | poly deg2, C=2.04, coef0=1.91 |
| DNN architecture | [128, 64], ReLU | [64, 32], tanh |
| X normalization (binary) | z-score | None (raw {0,1}) |
| SVM R² (continuous) | 0.14 | **0.88** |
| BNN Full R² (continuous) | 0.04 | **0.83** |
| VBLL R² (continuous) | -0.22 | **0.77** |

The lesson: Y normalization is critical for SVM (epsilon is calibrated for unit variance) and neural networks (gradient magnitudes). Hyperparameter tuning is equally important — the main pipeline's Optuna-tuned parameters are model×representation-specific.

## Recommendation

For the paper, the binary PDV should be described accurately as a **binarized physiochemical descriptor vector** with a >0 threshold, and the existing binary PDV results are valid as-is. The `continuous_pdv` experiments (uint8 bug fixed, now running on server) will provide a direct comparison point in the main ANOVA framework.

Key points for the paper discussion:
1. Binarization costs ~4–9% R² depending on model, but provides implicit regularization that improves noise robustness for SVM.
2. Bayesian models (BNN Full, VBLL) are the most noise-robust regardless of representation — their posterior regularization and learned noise provide genuine resilience, not just inability to learn.
3. XGBoost achieves the best baseline R² but is catastrophically non-robust under noise injection — a finding that reinforces the paper's central message about the importance of noise robustness evaluation.
4. The binary/continuous tradeoff is model-dependent: SVM shows the strongest binary advantage, Bayesian models are representation-agnostic, and tree models depend on tuning.
