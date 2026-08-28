# Choosing the assessment noise level, and auditing the strategy cut-points

Analysis date: 2026-08-21. All numbers below were computed in this session from
`val_rerun.parquet` (48,510 rows; 3 datasets x 13 models x 4 reps x 6 strategies x 5 folds
x 11 sigma), `data/QM9/raw/gdb9.sdf.csv`, and the noise-injection source
`rust/src/main.rs` + `scripts/noise_strategy_params.json`.

Scripts: `sig_q0.py`, `sig_q1.py`, `sig_q1b.py`, `sig_q1c.py`, `sig_q1d.py`, `sig_q2.py`,
`sig_q2b.py`, `sig_q2c.py`, `sig_q2d.py`, `sig_q3.py`, `sig_q3b.py`, `sig_q4.py`,
`sig_q4b.py`, `sig_q4c.py`, `sig_q5.py`, `sig_q5b.py` (same directory).
Intermediate CSVs: `q1_cells.csv`, `q1_rank.csv`, `q1_drop.csv`, `q2_equal_damage.csv`,
`q2_legacy_equiv.csv`, `q2_rank_agreement.csv`, `q5_trio.csv`.

---

## 0. Preliminaries established from the data

### 0.1 The validation test labels are clean and on the raw scale

`Var(y_test) = rmse^2 / (1 - r2)` is **identical at every sigma** within each
(dataset, fold). That proves the test labels are never noised — noise is train-side only —
and that r2/rmse are on the raw label scale, not standardised.

```python
d['vary'] = d.rmse**2/(1-d.r2)
d.groupby(['dataset','fold','sigma']).vary.mean().unstack()
```

| dataset | label SD (pooled) | per-fold range |
|---|---|---|
| ChEMBL-hERG-Ki | **0.911** | 0.873 – 0.956 |
| OpenADMET-Caco2_Efflux | **0.444** | 0.408 – 0.494 |
| OpenADMET-LogD | **1.187** | 1.167 – 1.202 |

**sigma = 0.6 expressed in each dataset's own label SD:**

| dataset | sigma=0.6 in label-SD units |
|---|---|
| ChEMBL-hERG-Ki | 0.66 |
| OpenADMET-Caco2_Efflux | **1.35** |
| OpenADMET-LogD | 0.51 |

A single absolute sigma is a 2.7-fold different insult across the three datasets.

### 0.2 Divergence contaminates the raw R2 spread

6.45% of all rows have R2 < -1, down to -5914. It is **not** noise-induced: the rate is
already 5.19% at sigma = 0. It is concentrated in MLP-VBLL-Full (801 rows), MLP-BNN-Full
(630), VBLL-Full (574), BNN-Full (381) and in the MHG-GNN-pretrained rep (2,218 of 3,127).

Consequence: any "between-model spread" computed on raw R2 grows with sigma simply because
more models blow up. Every discrimination statistic below is therefore computed either on
R2 clipped at -1, or on ranks.

---

## 1. Is sigma = 0.6 the right assessment point?

### 1.1 Naive signal-to-noise (between-model SD / between-fold SD), raw R2

Per (dataset, rep, strategy, sigma): between-model SD of the 13 fold-averaged R2 values,
divided by the pooled between-fold SD. Median over the 24 (rep, strategy) cells.

```python
for (ds,rep,strat,sg),g in d.groupby(['dataset','rep','strategy','sigma']):
    mm = g.groupby('model').r2.mean()
    w  = g.groupby('model').r2.std(ddof=1)
    snr = mm.std(ddof=1)/np.sqrt((w**2).mean())
```

| sigma | hERG-Ki | Caco2 | LogD |
|---|---|---|---|
| 0.0 | 1.129 | 0.892 | 1.633 |
| 0.1 | 0.977 | 0.934 | 1.687 |
| 0.2 | 1.200 | 0.887 | 1.588 |
| 0.3 | 1.122 | 0.760 | 1.607 |
| 0.4 | 1.029 | 0.846 | 1.543 |
| 0.5 | 1.079 | 0.934 | 1.690 |
| **0.6** | **0.994** | **1.055** | **1.821** |
| 0.7 | 0.939 | 1.250 | 1.695 |
| 0.8 | 0.750 | 1.041 | 1.742 |
| 0.9 | 0.874 | 1.138 | 1.727 |
| 1.0 | 0.966 | 1.140 | 1.740 |

The ratio is essentially **flat** (0.75–1.82). No sigma stands out. LogD discriminates
about 1.7x better than the other two at every sigma, because its between-fold noise is much
smaller (within-fold SD ~0.058 vs ~0.24 for hERG).

### 1.2 Rank-based discrimination (immune to blow-ups)

Kendall's W across the 5 folds — how reproducibly the folds agree on the 13-model ordering.

| sigma | hERG-Ki | Caco2 | LogD | pooled |
|---|---|---|---|---|
| 0.0 | 0.656 | 0.683 | **0.864** | 0.728 |
| 0.1 | 0.649 | 0.714 | 0.870 | 0.740 |
| 0.2 | 0.632 | 0.665 | 0.856 | 0.711 |
| 0.3 | 0.649 | 0.585 | 0.853 | 0.697 |
| 0.4 | 0.665 | 0.628 | 0.842 | 0.688 |
| 0.5 | 0.648 | 0.692 | 0.820 | 0.741 |
| **0.6** | 0.615 | 0.747 | 0.826 | **0.749** |
| 0.7 | 0.642 | 0.766 | 0.810 | 0.758 |
| 0.8 | 0.651 | 0.783 | 0.817 | 0.750 |
| 0.9 | 0.660 | **0.800** | 0.800 | **0.771** |
| 1.0 | 0.662 | 0.795 | 0.798 | 0.756 |

Pooled range 0.688–0.771 across the whole grid. Reliability of the ranking barely depends
on sigma. Per dataset the trends **point in opposite directions**: LogD is best at sigma=0
and degrades; Caco2 improves monotonically; hERG is flat. That is exactly what section 0.1
predicts — Caco2's label SD is the smallest, so a given absolute sigma bites hardest there.

### 1.3 The measure that actually matters: is the sigma ranking *new* information?

Spearman between the model ranking at sigma and the ranking at sigma = 0 (median over cells):

| sigma | hERG-Ki | Caco2 | LogD | pooled |
|---|---|---|---|---|
| 0.1 | 0.944 | 0.963 | 0.967 | 0.963 |
| 0.2 | 0.916 | 0.945 | 0.972 | 0.948 |
| 0.3 | 0.902 | 0.775 | 0.956 | 0.907 |
| 0.4 | 0.896 | 0.726 | 0.960 | 0.904 |
| 0.5 | 0.877 | 0.715 | 0.940 | 0.874 |
| **0.6** | 0.838 | 0.555 | 0.945 | **0.843** |
| 0.7 | 0.803 | 0.481 | 0.920 | 0.821 |
| 0.8 | 0.724 | 0.404 | 0.905 | 0.783 |
| 0.9 | 0.678 | 0.431 | 0.859 | 0.758 |
| 1.0 | 0.500 | 0.352 | 0.833 | 0.672 |

And the paired-drop SNR — between-model SD of the per-fold drop (clean minus noisy),
over the within-fold SD of that drop. Pairing removes fold difficulty, so this is the
statistically efficient test of "do models differ in robustness":

| sigma | hERG-Ki | Caco2 | LogD | pooled |
|---|---|---|---|---|
| 0.1 | 0.358 | 0.388 | 0.435 | 0.391 |
| 0.2 | 0.457 | 0.465 | 0.511 | 0.466 |
| 0.3 | 0.383 | 0.609 | 0.501 | 0.493 |
| 0.4 | 0.482 | 0.659 | 0.466 | 0.549 |
| 0.5 | 0.501 | 0.728 | 0.553 | 0.598 |
| **0.6** | 0.524 | 1.264 | 0.636 | **0.710** |
| 0.7 | 0.563 | 1.399 | 0.604 | 0.662 |
| 0.8 | 0.653 | 1.113 | 0.698 | 0.773 |
| 0.9 | 0.765 | 1.428 | 0.717 | 0.888 |
| 1.0 | **0.858** | **1.457** | **0.737** | **1.030** |

Argmax over the 72 (dataset, rep, strategy) cells: sigma = 0.9 in 24 cells, sigma = 1.0 in
26 — i.e. **50 of 72 cells discriminate robustness best at the very top of the grid.**

### 1.4 Verdict on Q1

There is **no optimum at 0.6, and no optimum anywhere** — discrimination is flat, and the
two useful criteria pull in opposite directions:

- *Reliability of the ranking* (Kendall's W) is flat; the pooled maximum is 0.771 at
  sigma = 0.9 versus 0.749 at 0.6 — a difference of no practical consequence.
- *Distinctness from the clean ranking* rises monotonically. At 0.6 the ranking still
  shares rho = 0.843 with the clean ranking; 71% of the ordering is baseline skill, not
  robustness.
- *Power to separate models on robustness* (paired drop) rises monotonically and is
  1.45x higher at 1.0 than at 0.6.

sigma = 0.6 is a defensible compromise but it is **not** where discrimination is best, and
it does not mean the same thing on the three datasets: mean relative R2 drop at 0.6 is
0.082 on hERG, 0.057 on LogD, but 0.228 on Caco2.

---

## 2. Does the assessment sigma need to differ per strategy?

### 2.1 The doses, verified against source

`rust/src/main.rs` lines 309–420 plus `scripts/noise_strategy_params.json`:

| strategy | sigma_eff | notes |
|---|---|---|
| Legacy | `sigma` | Gaussian on all |
| ValueProportional | `sigma * (1 + 0.05*|y|)` | JSON overrides prop factor 0.10 -> 0.05; uses `value.abs()` |
| Quantile | `2*sigma` on y<=v10 or y>=v90, `0.1*sigma` on the middle 80% | deciles of the **whole** target vector |
| Threshold | `2*sigma` if y>=+1.0 or y<=-1.0, else `0.1*sigma` | absolute, unit-bearing cut-points |
| OutlierFocused | `3*sigma` if |z|>2, else `0.1*sigma` | mean/SD of the whole target vector |
| Heteroscedastic | `sigma * sqrt(0.1 + 0.05*|y|)` | alpha, beta both carry sigma^2 |

A caution for the write-up: the JSON pins ValueProportional's `proportionality_factor` to
0.05, not the 0.10 in the code default. Its `base_sigma` key is 0.1, but the empirical
curves are steeply sigma-dependent (mean R2 falls 0.38 -> 0.03 pooled), so the runs cannot
have used a frozen base_sigma; the sweep sigma was in force.

### 2.2 Equal-damage sigma (25% mean relative R2 drop)

Cells restricted to baseline clipped R2 > 0.1 (42,647 of 48,510 rows, 87.9%).
Mean relative drop `(R2(0)-R2(sigma))/R2(0)` pooled over datasets, linearly interpolated:

| strategy | sigma for a 25% drop | ratio to Legacy |
|---|---|---|
| Heteroscedastic | **0.829** | 1.75 |
| OutlierFocused | **0.544** | 1.15 |
| Legacy | **0.473** | 1.00 |
| Quantile | **0.390** | 0.82 |
| Threshold | **0.365** | 0.77 |
| ValueProportional | **0.244** | 0.52 |

At other targets: 10% drop -> hetero 0.149, legacy 0.231, outlier 0.174, quantile 0.181,
threshold 0.143, valprop 0.081. 40% drop -> legacy 0.609, outlier 0.765, quantile 0.647,
threshold 0.535, valprop 0.474, hetero never reaches it.

Per dataset (25% target) the spread is far wider still:

| dataset | hetero | legacy | outlier | quantile | threshold | valprop |
|---|---|---|---|---|---|---|
| hERG-Ki | 0.589 | 0.667 | 0.720 | 0.337 | 0.265 | 0.092 |
| Caco2 | 0.736 | 0.297 | 0.378 | 0.298 | 0.512 | 0.269 |
| LogD | never | 0.579 | never | 0.923 | 0.479 | 0.882 |

### 2.3 Legacy-equivalent dose multiplier

Inverting each strategy's damage onto the Legacy curve (Legacy at sigma has effective noise
SD exactly sigma), pooled, median over sigma 0.4–0.8:
hetero ~0.57, outlier ~0.90, quantile ~1.02, threshold ~1.24, valprop ~1.30.

Compare the *predicted* RMS doses on QM9 (section 3): hetero 0.671, outlier 0.531,
quantile 0.900, valprop 1.352. The ordering matches; outlier's empirical multiplier is
higher than its RMS dose because concentrating noise on 3–5% of points is more destructive
per unit of variance than spreading it.

### 2.4 Do rankings agree better at equal damage than at equal sigma?

180 strategy pairs (15 pairs x 3 datasets x 4 reps), Spearman of 13-model rankings.
Values obtained by linear interpolation of each model's mean R2 curve.

| comparison | mean pairwise Spearman |
|---|---|
| Clean (sigma = 0) — the ceiling | **0.963** |
| Equal sigma, all at 0.6 | **0.763** |
| Equal sigma at 0.438 (damage-matched control) | **0.825** |
| Equal damage (25% per-strategy sigmas) | **0.865** |

- Equal-damage vs equal-sigma-at-0.6: **+0.102**, improved in 134/180 pairs,
  Wilcoxon p = 1.0e-13.
- Against the fair control (a common sigma of 0.438, which delivers the same 25% damage
  *averaged over the six strategies*): **+0.039**, improved in 101/180 pairs,
  Wilcoxon p = 2.4e-4.

So roughly 60% of the naive gain is simply "less damage keeps you nearer the clean
ranking", but a real, statistically clear ~0.04 of dose-equalisation benefit survives.

Biggest gains are exactly where the dose mismatch is biggest:
hetero|threshold +0.273, hetero|valprop +0.269, hetero|quantile +0.151,
threshold|valprop +0.151. Only legacy|valprop got worse (-0.043).

Equal-sigma agreement as a function of the common sigma: 0.926 (0.2), 0.825 (0.4),
0.763 (0.6), 0.723 (0.8), 0.708 (1.0). Agreement falls monotonically with sigma —
the strategies genuinely diverge as damage accumulates, and part of that divergence is
an artefact of unequal dose.

---

## 3. Should the strategy cut-points change?

### 3.1 QM9 gap distribution (first 10,000 molecules, Hartree x 27.2114)

n = 10,000; mean 7.001 eV; SD 1.347; min 2.084; max 16.928.
Quantiles: 5% 4.890, 10% 5.358, 25% 6.019, 50% 6.890, 75% 8.025, 90% 8.838, 95% 9.143.
Skew +0.151, excess kurtosis -0.197.

### 3.2 Threshold — broken on QM9, and broken the other way on Caco2

- Fraction with y >= +1.0: **1.0000**. Fraction with y <= -1.0: **0.0000**.
  Mid band (which is supposed to receive 0.1*sigma): **0.0000**.
- **On QM9, Threshold is exactly Legacy at twice the sigma.** It contributes nothing
  beyond a doubled Gaussian dose and should not be described as a distinct mechanism.

On the validation sets there are no labels available locally, so the mid-band fraction was
inferred by inverting the legacy-equivalent multiplier through the RMS dose
`m = sqrt(4*(1-f) + 0.01*f)`. These are indirect estimates, flagged as such:

| dataset | threshold multiplier (median, sigma 0.4–0.8) | implied mid-band fraction |
|---|---|---|
| hERG-Ki | damage exceeds Legacy's whole range (multiplier >> 1; NaN by inversion) | ~0 — consistent with all pKi values > 1.0, so 2*sigma on everything |
| Caco2 | 0.65 | **~0.90** — Threshold is nearly *inert* here |
| LogD | 1.24 | ~0.62 |

Direct corroboration from the R2 tables: on hERG, Threshold at sigma = 0.6 (mean clipped
R2 0.080) is already worse than Legacy at sigma = 1.0 (0.111). On Caco2, Threshold at
sigma = 1.0 (-0.008) is roughly Legacy at sigma = 0.65 (0.042 at 0.6).

**The same fixed +/-1.0 cut-point means "noise everything" on QM9 and hERG and "noise almost
nothing" on Caco2.** That is the single clearest defect found.

Proposed quantile-based replacements, computed on QM9:

| rule | low cut (eV) | high cut (eV) | fraction affected | RMS dose |
|---|---|---|---|---|
| p = 0.10 (deciles) | 5.358 | 8.838 | 0.2005 | 0.900*sigma |
| p = 0.20 | 5.878 | 8.308 | 0.4007 | 1.268*sigma |
| **p = 0.25 (quartiles)** | **6.019** | **8.025** | **0.5011** | **1.418*sigma** |
| |z| > 1 | 5.654 | 8.348 | 0.3440 | — |

Note that p = 0.10 reproduces the Quantile strategy *exactly*. So Threshold must not be
re-specified at the deciles or the two strategies collapse into one. Quartiles (p = 0.25)
keep them distinct and affect half the molecules.

### 3.3 Quantile — sound as specified

Decile cuts on QM9 land at v10 = 5.358 eV and v90 = 8.838 eV; realised fractions 0.1003
low, 0.1002 high, 0.7995 middle. Exactly as intended. RMS dose 0.900*sigma. Empirical
legacy-equivalent multiplier ~1.02, consistent. **No change needed.**

### 3.4 Outlier — mildly under-triggered on QM9

|z| > 2 catches **3.03%** of QM9 molecules, not the 4.55% a Gaussian would give (the
distribution is slightly platykurtic, excess kurtosis -0.197), split 1.37% upper /
1.66% lower. RMS dose 0.531*sigma — the weakest nominal dose of the six after
Heteroscedastic. Inferred validation tail fractions: hERG ~0.119, Caco2 ~0.074,
LogD ~0.044. So the *label* "outliers" covers between 3% and 12% depending on dataset.
This is tolerable; if a fixed affected fraction is wanted, replace |z| > 2 with a
percentile rule (e.g. the outer 5%), which is distribution-free.

### 3.5 Heteroscedastic — very weak, and nearly flat

On QM9, sigma_eff = sigma*sqrt(0.1 + 0.05*|y|) gives an RMS dose of only **0.671*sigma**,
and the ratio of sigma_eff at max y to min y is only **2.15** — so it is barely
heteroscedastic. Empirically it is the weakest strategy by a wide margin: pooled mean
clipped R2 falls only 0.401 -> 0.317 across the whole sigma sweep, and it never reaches a
40% relative drop anywhere on the grid. Its 25% equal-damage sigma is 0.829, 1.75x Legacy's.

### 3.6 ValueProportional on QM9

RMS dose 1.352*sigma at prop = 0.05 (the JSON value), 1.705*sigma at prop = 0.10 (the code
default). It is the most damaging strategy per unit sigma. Because it uses `|y|` and QM9's
gap is bounded away from zero (min 2.08 eV), it behaves as a near-uniform 1.35x dose on
QM9 rather than a value-dependent one.

---

## 4. Is the sigma grid right?

### 4.1 Collapse to R2 < 0

Fraction of the 4,410 (dataset, model, rep, strategy, fold) curves with R2 < 0 **at** each
sigma:

| sigma | hERG-Ki | Caco2 | LogD | all |
|---|---|---|---|---|
| 0.0 | 11.2% | 9.0% | 6.2% | 8.8% |
| 0.2 | 12.7% | 9.9% | 7.1% | 9.9% |
| 0.4 | 14.5% | 10.7% | 6.6% | 10.6% |
| 0.6 | 17.1% | 20.7% | 7.5% | 15.1% |
| 0.8 | 25.2% | 29.5% | 7.9% | 20.9% |
| 1.0 | 29.3% | **44.5%** | **9.1%** | 27.6% |

Cumulative (ever gone negative at or before sigma): 8.8% at 0.0, 20.9% at 0.6, 35.4% at 1.0.

Note the 8.8% floor at sigma = 0 — that is the pre-existing divergence of section 0.2, not
noise.

### 4.2 Nothing is saturated at sigma = 1.0

Share of the total R2 decline (mean clipped R2, sigma 0 to 1) accumulated by each sigma:
0.028 (0.1), 0.110 (0.2), 0.185 (0.3), 0.243 (0.4), 0.370 (0.5), **0.485 (0.6)**,
0.560 (0.7), 0.713 (0.8), 0.883 (0.9), 1.000 (1.0).

**Only 48.5% of the degradation has happened by sigma = 0.6.** The curves are steepest at
the *top* of the grid, not the middle — mean |dR2/dsigma| over the last interval is 0.243
on LogD and 0.508 on Caco2. LogD is still at mean clipped R2 = 0.428 at sigma = 1.0 and
still falling; it has not begun to saturate.

The sigma > 0.6 half is not redundant: the model ranking from the mean of sigma <= 0.6
correlates only rho = 0.771 (mean) / 0.864 (median) with the ranking from sigma >= 0.7.

### 4.3 The three datasets collapse onto one curve in label-SD units

Retained fraction of clean R2, Legacy strategy only, on strictly matched support
(0.1–0.8 on each axis):

| axis value | hERG-Ki | Caco2 | LogD | spread |
|---|---|---|---|---|
| **sigma / label SD** | | | | |
| 0.1 | 0.957 | 0.960 | 0.923 | 0.037 |
| 0.2 | 0.966 | 0.920 | 0.921 | 0.046 |
| 0.3 | 0.912 | 0.898 | 0.920 | 0.023 |
| 0.4 | 0.858 | 0.882 | 0.866 | 0.024 |
| 0.5 | 0.841 | 0.846 | 0.733 | 0.112 |
| 0.6 | 0.828 | 0.789 | 0.840 | 0.051 |
| 0.7 | 0.777 | 0.741 | 0.812 | 0.071 |
| 0.8 | 0.704 | 0.716 | 0.760 | 0.056 |
| | | | | **mean 0.0525** |
| **nominal sigma** | | | | |
| 0.1 | 0.953 | 0.910 | 0.920 | 0.043 |
| 0.2 | 0.968 | 0.874 | 0.938 | 0.095 |
| 0.3 | 0.891 | 0.747 | 0.892 | 0.145 |
| 0.4 | 0.839 | 0.691 | 0.942 | 0.251 |
| 0.5 | 0.842 | 0.491 | 0.840 | 0.351 |
| 0.6 | 0.812 | 0.285 | 0.726 | 0.526 |
| 0.7 | 0.720 | 0.146 | 0.844 | 0.698 |
| 0.8 | 0.664 | 0.018 | 0.818 | 0.801 |
| | | | | **mean 0.3637** |

**Rescaling sigma by the label SD shrinks the across-dataset spread by 6.9x
(0.364 -> 0.053).** The three datasets are not behaving differently; they are being given
different doses. This is the single most useful result in the whole analysis.

Sigma needed to deliver 0.6 label-SD of noise: hERG 0.55, Caco2 0.27, LogD 0.71.

---

## 5. Do clean R2, R2 at sigma, and auc_norm measure different things?

auc_norm is recomputed here with the script's own definition
(`_retention_auc_norm`, `generate_paper_figures_v2.py` L1791):
`trapezoid(R2(sigma)/R2(0), sigma) / (sigma_max - sigma_min)`.

3,877 curves have clean R2 > 0.1; 3,734 survive a wide artifact band (-1 < auc_norm < 2),
matching the script's filtering intent.

### 5.1 Correlation structure (fold-level cells, n = 3,734)

| pair | Pearson | Spearman |
|---|---|---|
| clean R2 vs R2(0.6) | +0.709 | +0.813 |
| clean R2 vs auc_norm | **+0.336** | **+0.429** |
| R2(0.6) vs auc_norm | +0.731 | +0.764 |
| clean R2 vs retention(0.6) | +0.258 | +0.384 |
| retention(0.6) vs auc_norm | +0.741 | +0.854 |

Within dataset, clean-vs-auc_norm collapses further: hERG +0.030, Caco2 +0.122, LogD +0.255.

**When the question is which model to pick** (ranking models within a
dataset x rep x strategy cell, 72 cells), the mean Spearman is:

| pair | mean Spearman across 72 cells |
|---|---|
| clean R2 vs auc_norm | **-0.005** |
| clean R2 vs R2(0.6) | +0.591 |
| R2(0.6) vs auc_norm | +0.492 |

**Clean R2 and auc_norm are, for model selection, statistically orthogonal.** The script's
claim that auc_norm is "decoupled from baseline performance" is confirmed on the data.

### 5.2 Regression: does auc_norm add anything to the pair?

```
auc_norm = 0.6685 - 0.7563*clean + 1.2218*R2(0.6)
R2 = 0.6015,  residual SD = 0.2048,  auc_norm SD = 0.3243   ->  40% unexplained
```

The near-mirror coefficients (-0.76, +1.22) confirm auc_norm behaves like a *ratio*.
Adding retention(0.6) as a third predictor barely helps (R2 0.6015 -> 0.6062).

Each quantity regressed on the other two, fold-level and on cell means (folds averaged,
n = 814 cells):

| target | on | R2 (fold) | R2 (cell mean) | unexplained (cell mean) |
|---|---|---|---|---|
| auc_norm | clean + R2(0.6) | 0.601 | 0.540 | **46%** |
| R2(0.6) | clean + auc_norm | 0.777 | 0.720 | **28%** |
| clean | R2(0.6) + auc_norm | 0.574 | 0.503 | **50%** |

### 5.3 Is the 40–46% residual real, or replicate noise?

Between-fold (replicate) SD, pooled RMS within each (dataset, rep, strategy, model) cell:

| quantity | between-fold SD | total SD | ICC |
|---|---|---|---|
| clean R2 | 0.0744 | 0.1572 | 0.78 |
| R2(0.6) | 0.1417 | 0.2631 | 0.71 |
| auc_norm | 0.1777 | 0.3243 | 0.70 |

On cell means the auc_norm residual SD is 0.2301, while the standard error of a cell-mean
auc_norm is 0.1777/sqrt(5) = 0.0795.

**Ratio 2.89, i.e. the leftover variance is 8.4x the replicate variance.** The information
auc_norm carries beyond the pair is real signal, not noise.

### 5.4 Which single sigma's retention best proxies auc_norm?

| sigma | Pearson(retention, auc_norm) | Spearman |
|---|---|---|
| 0.3 | +0.547 | +0.748 |
| 0.5 | +0.733 | +0.844 |
| **0.6** | **+0.741** | **+0.854** |
| 0.8 | +0.807 | +0.879 |
| 0.9 | +0.795 | **+0.885** |
| 1.0 | +0.739 | +0.880 |

No single sigma reproduces auc_norm (best Spearman 0.885). The curve shape matters.

### 5.5 Verdict on Q5

None of the three is redundant, and they play distinct roles:

- **clean R2** — capability with perfect labels. Nearly orthogonal to auc_norm (rank
  correlation -0.005), so it cannot be inferred from the robustness metrics.
- **auc_norm** — robustness proper, a shape summary of the whole curve. 46% of its
  variance is unreachable from the pair, and that leftover is 8.4x the replicate noise.
  Caveat: it is the **least reproducible** of the three (ICC 0.70, and 55% of its SD is
  between-fold noise), so it needs the fold averaging it already gets.
- **R2 at sigma** — the most *redundant* of the three (only 28% unexplained by the other
  two on cell means), because it is a blend: it correlates +0.71 with clean and +0.73 with
  auc_norm. Its value is not statistical independence but interpretability — it is the one
  number a practitioner can read as "this is the accuracy you would actually get at this
  noise level".

---

## 6. What could not be determined

- **Validation-set label distributions.** The parquet carries no labels, and the raw
  hERG / LogD / Caco2 files are not present locally (`data/` holds only QM9 and
  `valid_qm9_indices.pth`). Label **SDs** were recovered exactly from `rmse^2/(1-r2)`,
  but means, skew, quantiles and the true fraction of molecules crossing the +/-1.0
  Threshold cut or the |z|>2 Outlier cut could not be measured directly — they were only
  inferred from the damage curves (section 3.2, 3.4) and are flagged as estimates.
- **QM9 per-sigma R2.** `results/paper_figures_v2/` carries auc_norm only, so none of the
  sigma-selection analysis could be repeated on QM9. Everything in sections 1, 2, 4 and 5
  is validation-set only.
- **Whether the ARC runs used `scripts/noise_strategy_params.json` or the code defaults.**
  The empirical curves rule out a frozen `base_sigma = 0.1` for ValueProportional, but
  prop = 0.05 vs 0.10 could not be distinguished from R2 alone.
