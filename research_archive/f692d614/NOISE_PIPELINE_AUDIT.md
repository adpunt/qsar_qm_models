# Label-noise pipeline audit (post-fix)

Date: 2026-08-21. Everything below was read in the working tree; nothing from memory.
`cargo check` passes on the edited Rust (warnings only).

## VERDICT

**Partly.** The specific bug that was fixed is genuinely fixed: held-out labels are
now clean, and R2 is now computed against clean held-out labels, which is the right
comparison for a training-noise study. But three things still leak or distort, and
one of them is a *new* consequence of the fix:

1. Seven model families concatenate the validation split into the training set
   (`x_train = np.vstack((x_train, x_val))`). Val is now clean, so ~11% of every
   "noisy" training set is noise-free. The injected noise level is silently diluted.
2. Val/test labels are standardised with the mean and SD of the **noisy** training
   labels. R2 is immune to this (affine on both sides), RMSE and MAE are not.
3. `injected_noise` in the uncertainty CSVs is now identically ~0 on the test split
   by construction, and `generate_paper_figures_v2.py` still correlates predicted
   uncertainty against it.

---

## 1. Noise generation — the design IS paired across sigma

`rust/src/main.rs`:

- `main` builds `noise_indices = (0..config.train_count)` only when `config.noise`
  is true; `config.noise` is set by `process_and_train.py` as `'noise': s > 0`, so
  sigma = 0.0 runs with an empty map and no noise at all.
- `generate_value_based_noise_map` sends `Legacy` to `generate_noise_by_indices` and
  everything else to `generate_adaptive_noise` after reading the training targets
  back off disk with `read_all_target_values`.
- Both samplers seed one `StdRng::seed_from_u64(seed)` and then walk `indices` in
  ascending order, drawing exactly one variate per index.

`seed` is `iteration_seed`, computed in the sigma loop of `process_and_train.py` as
`(args.random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF` — it depends on the
iteration but **not** on sigma. The scaffold split is also re-derived from the same
seeded RNG state, so molecule *i* is the same molecule at every sigma.

For the Gaussian branch — the only one that is ever used, see §7 — `Normal::new(0.0, sigma)`
draws a standard normal and multiplies by `sigma`; the underlying RNG consumption is
independent of `sigma`. Therefore

    noise_i(sigma) = sigma * z_i,   with the SAME z_i at every sigma.

**This is a paired design**, and that is the good outcome: the R2-versus-sigma curve
moves because sigma moves, not because a different noise draw happened. If each
sigma were an independent draw, the curve would carry the full between-draw variance
and the per-model robustness rankings would be far less stable across sigma.

One caveat that would break pairing if a strategy ever produced a non-positive sigma:
`sample_from_distribution` returns 0.0 for `sigma <= 0` **without consuming a variate**,
which desynchronises the stream from that index onward. With the current parameter
defaults every per-index sigma is strictly positive, so it does not fire today.

## 2. Noise application — the fix is complete and correctly positioned

`write_data` now takes `apply_noise: bool` as its final parameter, and the guard reads
`if config.noise && apply_noise`. The three call sites in `preprocess_data` are:

| split | `data_count` | `log_writes` | `apply_noise` |
|---|---|---|---|
| train | `config.train_count` | `config.logging` | `true` |
| val   | `config.val_count`   | `config.logging` | `false` |
| test  | `config.test_count`  | `config.logging` | `false` |

The signature ends `..., data_count: usize, log_writes: bool, apply_noise: bool`, so the
two adjacent bools are exactly the risk you flagged. I checked each call individually:
in all three, the positional argument before the new flag is `config.logging`, and the
new literal is last. **Positions are right.**

Other routes by which noise could reach val/test:

- **Statistics.** `generate_aggregate_stats` reads only the *train* mmap and adds the
  noise map before accumulating, so `mean`/`std_dev` are noisy-train statistics, and
  `write_data` applies `(property_value - mean)/std_dev` to all three splits. This is
  a real, still-open channel — see §3.
- **Merged validation split.** `x_train = np.vstack((x_train, x_val))` in
  `train_rf_model`, `train_svm_model`, `train_ngboost_model`, `train_xgboost_model`,
  `train_lgb_model`, `train_gauche_model`, and the gauche branch of
  `train_conformal_model`. Val is now clean, so these models train on 8 parts noisy
  plus 1 part clean. This is the reverse leak and it is new.
- `run_qm9_graph_model` re-reads the same three mmaps through `parse_mmap(..., "graph")`,
  so graph models inherit whatever the mmaps hold — no separate noise path.
- No other writer touches the label field.

## 3. Normalisation

`generate_aggregate_stats` accumulates `y_values` over the training file only,
`0..config.train_count`, adding `noise_map[index]` when `config.noise`. So mean and
SD come from **noisy training labels** — which is correct, because a real pipeline
cannot see clean labels.

Consequences for the now-clean held-out labels:

- **R2 is safe.** Predictions and targets live in the same standardised space, so the
  common factor cancels in `1 - SS_res/SS_tot`.
- **RMSE and MAE are not.** For zero-mean independent noise,
  `sd_noisy = sqrt(sd_clean^2 + sigma^2)`, and every reported error is divided by it.
  QM9 `homo_lumo_gap` is target index 4, which `torch_geometric/datasets/qm9.py`
  multiplies by `HAR2EV = 27.211386246`; the raw SD over the first 10,000 rows of
  `data/QM9/raw/gdb9.sdf.csv` is 0.04951 Hartree, i.e. **1.347 eV**. So the reported
  RMSE is deflated by

  | sigma | deflation factor |
  |---|---|
  | 0.2 | 0.978 |
  | 0.4 | 0.958 |
  | 0.6 | 0.913 |
  | 1.0 | 0.803 |

  A model whose true error is flat in sigma still shows RMSE falling ~20% from
  sigma 0 to 1. **Do not compare RMSE or MAE across sigma.** Multiplying by
  `sd_noisy` would restore comparability, but `sd_noisy` is not currently written out.
- **Zero-mean matters.** With a non-zero-mean noise distribution the test labels also
  pick up a constant offset of `-E[noise]/sd_noisy` relative to what the model learned,
  which shows up as pure bias in SS_res. The Gaussian branch is zero-mean, so this is
  latent — but see the left-tailed sampler in §7.

## 4. Index alignment — three real landmines, none currently firing

**a) `read_smiles_data` does not always consume one record.** It reads the
length-prefixed isomeric SMILES first and then `return None` if the string is shorter
than 5 characters, longer than 300, or contains U+FFFD / NUL / an apostrophe (the
character class lists U+FFFD twice, once escaped and once literal). On that path the
remaining fields of the record — canonical SMILES, the 4-byte target, and every
fixed-size descriptor block — are left in the stream, so the *next* call starts
mid-record and everything after it is garbage. The `if let Some(...)` in `write_data`
silently swallows the failure.

*Does it fire?* On QM9: no. PyG builds `data.smiles` with
`Chem.SDMolSupplier(..., removeHs=False)`, so hydrogens are explicit. Reproducing that
over all 133,885 SDF records gives max length 113 and **zero** records outside [5, 300]
— methane is `[H]C([H])([H])[H]`, 17 characters, not `C`. On the experimental sets the
SMILES come straight from the source file and are drug-sized; short or apostrophe-
bearing SMILES are implausible but nothing validates them. Latent, not active.

**b) The ECFP4 block's two early exits truncate the record.** In `write_data`, the
`if config.molecular_representations.contains("ecfp4")` block runs *last*. Both
`continue` statements — the `u64_vec.len() != 32` guard and the `Err(_)` arm of
`smiles_to_mol` — fire after the record's isomeric SMILES, canonical SMILES, raw
target, descriptor blocks, noisy normalised label, domain byte and SMILES one-hot have
already been written. **Exactly the 256-byte fingerprint is missing.** `parse_mmap`
reads 256 bytes unconditionally for `ecfp4`, so it would eat the first 256 bytes of the
following record and every subsequent record in the file would be misparsed; the
`except Exception: continue` in `parse_mmap` hides it. Not one lost row — the rest of
the split.

*Does it fire?* `rdk_fingerprint_mol` is RDKit's default 2048-bit fingerprint, so the
chunk count is always 32, and the SMILES being parsed were produced by RDKit. Latent.

**c) `read_all_target_values` builds a positional vector.** It loops
`for _index in 0..config.train_count` but only `push`es on `Some`, so one rejected
record shifts every later target by one position while the noise-map keys and
`write_data`'s counter — both plain loop counters — do not shift. Adaptive strategies
would then assign each molecule the sigma computed from a *different* molecule's value.
Same trigger as (a), so also latent. `generate_aggregate_stats` uses the loop index and
is aligned.

**d) Count drift.** `write_to_mmap` returns early without writing when a descriptor is
missing, but the QM9 loop appends to `successful_train_idx` regardless, and the Polaris
loop uses the splitter's indices directly. `config.train_count` can therefore exceed the
number of records on disk. That only causes the tail of the loop to hit EOF and stop —
order is preserved, so no misalignment — but the run silently uses fewer molecules than
it reports.

## 5. What the Python side receives

`parse_mmap` walks the Rust output field by field and keeps two labels per record:

- `y_data_original` <- the **raw, clean, unnormalised** target written near the start of
  the record (`# --- target value (raw) ---`). Rust always copies this straight from the
  input, so it was never noised even before the fix.
- `y_data` <- the **processed** target written after the descriptors: noisy for train
  (post-fix), clean for val/test, standardised in all three by noisy-train statistics.

These are on **different scales** — one raw eV, one standardised. Nothing reconstructs a
clean *training* label anywhere; `process_and_run` discards `y_train_original` with `_`.
That is fine for a training-noise study.

`save_uncertainty_values` in `scripts/utils.py` writes these columns:
`model, representation, sigma, iteration, file_no, sample_idx, y_pred_mean,
y_pred_std_uncalibrated, y_true_original, y_true_noisy, injected_noise,
y_pred_std_calibrated, temperature, epistemic_uncertainty, aleatoric_uncertainty`.

Provenance, traced through `models/models.py`: every caller passes
`y_true_original=y_test_original` (the raw clean label from `parse_mmap`) and
`y_true_noisy=y_test` (the processed test label). **The clean label is genuinely clean**
— and post-fix, so is `y_true_noisy`. Two consequences:

- `y_true_noisy` is now a misnomer: on the test split it is the clean label,
  standardised.
- `injected_noise` is computed as the residual of `linregress(y_orig, y_noisy)`. Since
  the test relation is now exactly affine, **the residual is ~0 for every test row at
  every sigma** (float32 rounding only). Any figure that correlates predicted
  uncertainty with `|injected_noise|` is now correlating against numerical dust.
  `generate_paper_figures_v2.py` does exactly that.
- `train_gnn` passes `test_targets` for *both* arguments, so `injected_noise` was
  already identically zero for graph models.

## 6. Metrics

`calculate_regression_metrics(y_test, prediction)` in `scripts/utils.py` computes MAE,
MSE, RMSE, R2 and Pearson r with scikit-learn. Every trainer calls it with `y_test` —
the processed test label — never with `y_test_original`. Post-fix that means
**predictions versus clean held-out labels, in standardised space. This is correct.**

Graph models do the same via `test_targets` built from `data.y_noisy`, which after the
fix holds clean standardised test labels despite the attribute name.

RMSE/MAE across sigma: not comparable, per §3. R2 and Pearson r are safe.

Note that the fix will move every published number: previously the test labels carried
their own noise, which inflated SS_res and depressed R2 at high sigma. Reported R2 will
rise, most at large sigma, so every robustness curve, AUC_norm value and ranking has to
be regenerated.

---

# DELETE LIST

Every symbol below was grepped across the whole repo before listing.

## rust/src/main.rs

| Lines | Delete | Why |
|---|---|---|
| 27 | `const DELIMITER` | Never referenced; `cargo check` flags it. |
| 78-82 | `#[derive(Serialize, Clone)] struct PlotPoint<T>` | Never constructed; `cargo check` flags it. |
| 866-870 | `fn tanimoto_distance` | Never called. |
| 885-907 | `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `r2_score` | Never called; metrics are computed in Python. Keeping a second, untested R2 invites someone to use it. |
| 211-234 | `NoiseDistribution::LeftTailed` arm of `generate_noise_by_indices` | Both branches return a non-positive value, so it is a one-sided downward shift, not noise: simulated at nominal sigma=0.6 it has mean **-0.677** and SD 0.430. Delete or replace with a real skew-normal. |
| 461-475 | the same LeftTailed arm duplicated in `sample_from_distribution` | Same defect, second copy. |
| 236-257 | `RightTailed` arm of `generate_noise_by_indices` | Not zero-mean and not sigma-calibrated: mean -0.118, SD 0.631 at nominal 0.6. |
| 477-490 | `RightTailed` arm of `sample_from_distribution` | Same defect, second copy. |
| 255-264 | `UShaped` arm of `generate_noise_by_indices` | Wrong constant. `k = sigma*2*sqrt(3)` with a Beta(0.5,0.5) whose variance is 1/8 gives SD `sqrt(6)*sigma` = **2.449x** the requested sigma (measured 1.469 at nominal 0.6). Correct constant is `k = sigma*sqrt(2)`. |
| 491-496 | `UShaped` arm of `sample_from_distribution` | Same defect, second copy. |
| 141-147, 420-439, 1219-1229 | `NoiseStrategy::ScaffoldBased` (variant, handler, CLI arm) | The handler calls `load_scaffold_assignments`, discards the result with `Ok(_) =>`, then applies `train_sigma` to every index. `test_sigma` and `val_sigma` are never read (`cargo check` flags both). It also `println!`s once per molecule. No slurm script passes `--noise_strategy scaffold`. Delete the variant and `load_scaffold_assignments` (281-293) with it. |
| 75, 589-591, 606, 733-740 | the `morgan` representation | Rust reads and rewrites 256 bytes for `morgan`, but Python's `write_to_mmap` never writes a morgan field and `parse_mmap` never reads one. Enabling `-r morgan` would desynchronise the record format. No slurm script uses it. |
| 193-279 | `generate_noise_by_indices` (whole function) | It is a byte-for-byte duplicate of `sample_from_distribution` used only by the `Legacy` arm. Route `Legacy` through `sample_from_distribution` so there is one sampler to fix. |

## scripts/utils.py

| Lines | Delete | Why |
|---|---|---|
| 62-85 | `decompose_uncertainty_sampling` | Hard-codes `aleatoric = None` and `total = epistemic`, so it is a dead end by construction — the `aleatoric_uncertainty` column is always NaN for every model that uses it. Its two correct callers (models.py ~2121 and ~2724, in `train_dnn_model` and `train_mlp_variant_model`) can call `preds.std(axis=0)` directly. Its two other callers are broken; see below. |

## models/models.py

| Lines | Delete | Why |
|---|---|---|
| 3133 | `epistemic, aleatoric = decompose_uncertainty_sampling(predictions_array)` | One argument passed to a two-argument function, two values unpacked from three returns. This **raises `TypeError`** every time a Bayesian graph model runs; the `except Exception` around `process_and_run` swallows it, so the run just produces no rows. |
| 3162 | `val_epistemic, _ = decompose_uncertainty_sampling(val_predictions_array)` | Same defect on the calibration path. |
| 3077 | the `decompose_uncertainty_sampling` name in `train_gnn`'s import | Follows from the above. |

## scripts/process_and_train.py

| Lines | Delete | Why |
|---|---|---|
| 260 | `--strategy-params` argparse option | `strategy_params` appears nowhere else in any Python file, and the subprocess call that launches the binary passes only `--seed`, `--model`, `--sigma`, `--noise_distribution`, `--noise_strategy`. The flag is inert. |

## scripts/noise_strategy_params.json

Delete the file, or start passing it. It is never read by Python and never reaches the
binary, so the Rust fallbacks are what actually ran, and two of them **disagree with the
file**: `value_proportional` used `base_sigma = sigma` (file says a fixed 0.1) and
`proportionality_factor = 0.1` (file says 0.05). The published value-proportional
results were produced with the Rust defaults, not with these numbers. The file also has
no `scaffold` entry at all.

## Also worth deleting or fixing (not noise, found en route)

- **`ecfp4` is not ECFP4.** `write_data` calls `rdk_fingerprint_mol`, which
  `~/.cargo/registry/.../rdkit-sys-0.4.12/wrapper/src/fingerprint.cc` line 7 implements as
  `RDKFingerprintMol` — RDKit's path-based topological fingerprint. The Morgan/circular
  binding `morgan_fingerprint_mol` sits three lines below it, unused. Everything the
  paper calls ECFP4 is an RDKit path fingerprint.
- **Graph models get the wrong molecules.** `split_qm9` rebinds its local parameter
  with `qm9 = qm9.index_select(indices)` after a `torch.randperm`, so the indices it
  returns are positions in the *shuffled* dataset. The mmaps are written from that
  shuffled copy, but the caller's `dataset` object is untouched, and
  `run_qm9_graph_model` indexes *that* object with the shuffled indices. Graph features
  and labels are decoupled.
- **`# ADD THIS` / `# CRITICAL:` / `# *** UPDATED:` comment litter** throughout
  `models/models.py` and `process_and_train.py`.

---

# SHORTEST PATH TO "YES"

1. Stop merging clean val into noisy train. Either drop the seven
   `np.vstack((x_train, x_val))` merges, or noise the val split with its own noise map
   keyed by val index (a second `generate_value_based_noise_map` call with a distinct
   seed) and keep test clean.
2. Emit `mean` and `sd_noisy` from Rust into `config.json` or a sidecar, and either
   report RMSE/MAE in raw label units or drop them from any cross-sigma comparison.
3. Delete the `injected_noise` column and every figure that correlates uncertainty with
   it; correlate uncertainty against `|y_pred_mean - y_true_noisy|` instead. Rename
   `y_true_noisy` to `y_true_eval` so the CSV stops lying.
4. Make `read_smiles_data` consume the whole record before rejecting it (or return a
   sentinel and skip in lockstep), and move the two ECFP4 `continue`s so they cannot
   truncate an already-started record — write the fingerprint block first or write 256
   zero bytes on failure.
5. Work the DELETE list above.
6. Regenerate everything. R2 at high sigma will rise across the board; every robustness
   number in the paper is stale.
