# Rust noise-injection audit (post-fix), 2026-08-21

Files: rust/src/main.rs; scripts/process_and_train.py; scripts/utils.py; models/models.py

## 1. Is the fix correct and complete?

CORRECT. `write_data` (L610-623) now ends `..., data_count: usize, log_writes: bool, apply_noise: bool`.
Noise is gated at L750: `if config.noise && apply_noise`. `preprocess_data` passes `true` (train, L1028),
`false` (val, L1055), `false` (test, L1082). No other write path touches the label.

Other paths noise could reach val/test:
- `generate_aggregate_stats` (L938-980) reads ONLY `train_{file_no}.mmap`, loops `0..config.train_count`,
  and applies `noise_map.get(&index)` before accumulating `y_values`. Index convention matches the train
  `write_data` call exactly. Mean/std therefore come from NOISY TRAINING labels only. CONFIRMED as intended.
- Consequence, not a bug but a reporting confound: `std_dev` grows as sqrt(var_y + sigma^2), and val/test
  labels are divided by it (L761). Nothing de-normalises (grep for denorm/inverse_transform returns
  nothing). R2 is invariant to a common affine map of y_true and y_pred, so R2-vs-sigma is safe. RMSE/MAE
  /y_pred_std ARE in units that shrink as sigma rises, attenuating the sigma trend.

## 2. Compilation

Signature = 13 params; all three call sites pass 13 args in the same order, bool last. `config.logging`
(bool) then literal bool. Types match (&Config, &HashMap, &HashMap). No arity/order/type error. Compiles.

## 3. Remaining noise-injection defects

D1 (CRITICAL, silent). `--strategy_params` is NEVER passed to the Rust binary. process_and_train.py
L1626-1637 builds argv with only --seed/--model/--sigma/--noise_distribution/--noise_strategy. The
argparse option exists (L260, default noise_strategy_params.json) and is dead. Rust falls back to
`serde_json::Map::new()`, every `params.get(..)` returns None, and ALL hardcoded defaults are used.
Downstream of that:
  - Threshold: high_threshold=1.0, low_threshold=-1.0. QM9 HOMO-LUMO gaps are all ~4-9 eV > 1.0, so every
    molecule takes the `value >= high_threshold` branch -> high_sigma = 2*sigma. Threshold is DEGENERATE:
    identical to Legacy at 2 sigma.
  - ScaffoldBased (L417-441) loads scaffold assignments, discards the result (`Ok(_) =>`), ignores
    test_sigma and val_sigma, and applies train_sigma = 0.1*sigma to every training index. Also
    println!s per molecule.

D2 (HIGH). U-shaped noise is 2.449x too strong. L253-259 / L491-496: `k = sigma*2*sqrt(3)`, value =
`(s-0.5)*2*k` in [-k,k] with s ~ Beta(0.5,0.5). Var = k^2 * Var(2(s-0.5)) = 12 sigma^2 * 0.5 = 6 sigma^2,
SD = 2.449 sigma. The comment claims variance = sigma^2. Correct arcsine constant is k = sigma*sqrt(2).
Uniform (a = sigma*sqrt(3), Var = a^2/3 = sigma^2) is right, so the two are not matched at equal sigma.

D3 (HIGH). LeftTailed is a pure downward SHIFT, not noise. L207-222 / L472-482: sample>=0 -> `-compressed`
(negative), sample<0 -> `sample*1.5` (negative). Every draw is <= 0. E[noise] << 0. RightTailed is signed
but non-linear in sigma: `x^1.5` for positives shrinks at sigma<1 and grows at sigma>1, `-sqrt(-x)` for
negatives INFLATES small draws (sigma=0.1 -> -0.32). For both, the sigma axis is non-linear and
non-monotonic in the induced SD. The shift is absorbed into the train mean but not the test labels:
y_test_norm = (y_test - (mean_y - b))/std, so the model's centred predictions are offset by b/std on test.
That is a shift artifact masquerading as noise sensitivity.

D4 (MEDIUM, scientific). Heteroscedastic and ValueProportional are nearly homoscedastic on QM9.
Heteroscedastic (L406-414 with L1214-1216): alpha = 0.1*sigma^2, beta = 0.05*sigma^2, so
sigma_i = sigma*sqrt(0.1 + 0.05|y|) — this DOES reduce as stated and DOES scale with sigma. But with
y in [4,9]: sigma_i ranges 0.55-0.74 sigma, ~+/-10% around 0.67 sigma. ValueProportional
(L309-317): base_sigma*(1 + 0.1|y|) = 1.4-1.9 * base_sigma, again ~+/-15%. Using |y| rather than a
centred value means the "heteroscedastic" condition is effectively Legacy Gaussian at a rescaled sigma,
and its distinctive level (0.67x vs 1.7x) is a nuisance scale difference, not heteroscedasticity.

D5 (MEDIUM, latent, same class as the bug just fixed). `read_all_target_values` (L173-190) PUSHES only
successfully-parsed records, while `write_data`/`generate_aggregate_stats` key on the LOOP COUNTER. One
skipped record and `target_values[idx]` is a different molecule than the one receiving noise[idx]. The
`if idx < target_values.len()` guard (every adaptive branch) then silently drops noise for the tail
indices, printing "No noise found for index N".

D6 (HIGH if triggered). Reader desynchronisation. `read_smiles_data` L531-534 returns None AFTER
consuming only isomeric_smiles when `len < 5 || len > 300 || contains replacement chars`. QM9 contains
SMILES shorter than 5 chars (C, N, O, CC, CO, C#N). No upstream length filter exists in
process_and_train.py. The remainder of that record (canonical SMILES, target, all fingerprint blocks) stays
in the stream and is misparsed as the next record's fields — every subsequent molecule, label and noise
index is garbage. Separately, `write_data`'s ECFP4 branch `continue`s (L829, L843) AFTER already writing
the label and all other representations, emitting a record short by 256 bytes; parse_mmap on the Python
side reads fixed widths and desyncs from there on with no error.

D7 (LOW). Quantile thresholds (L326-334): `(n * q) as usize` clamped to n-1; textbook index is
`(n-1)*q`. At most one element off — negligible at QM9 n. `sort_by(partial_cmp().unwrap())` panics on NaN.
Comparisons use `>=` high / `<=` low, so an exact tie counts as extreme. Acceptable.

## 4. RNG / paired design across sigma — GOOD

`StdRng::seed_from_u64(seed)`, indices iterated in ascending order (`(0..train_count).collect()`, L1145),
one draw per index. rand_distr scales AFTER sampling the standard variate, and the number of underlying
draws does not depend on sigma (Gaussian ziggurat, Uniform, Beta(0.5,0.5) all fixed-shape). So the same
molecule gets the same underlying z at every sigma, and for Gaussian/Uniform/UShaped noise_i = c*sigma*z_i:
a properly PAIRED design. It also pairs across strategies (same z stream). Two caveats: for
LeftTailed/RightTailed the map from z to noise is non-linear in sigma (D3), and `sample_from_distribution`
early-returns 0.0 without consuming the RNG when sigma <= 0, which would break stream alignment if any
per-index sigma multiplier were ever set to 0.

Strategies where sigma does NOT scale noise: none outright, but ValueProportional's `base_sigma` falls back
to CLI sigma ONLY because the params file is never passed (D1); if it were ever wired up with an explicit
`base_sigma`, the sigma axis would become inert for that strategy.

## 5. Python side — save_uncertainty_values (scripts/utils.py L200-267)

Columns written, in order:
model, representation, sigma, iteration, file_no, sample_idx, y_pred_mean, y_pred_std_uncalibrated,
y_true_original, y_true_noisy, injected_noise, y_pred_std_calibrated, temperature,
epistemic_uncertainty, aleatoric_uncertainty.

Arguments at every call site in models/models.py (L1403, 1555, 1793, 2155, 2272, 2494, 2761, 3196, 3397,
4000, 6851, 6977, 7183): `y_true_original=y_test_original`, `y_true_noisy=y_test`.
`y_test_original` comes from `parse_mmap`'s third return, `y_data_original`, which is the RAW
`target_value` field Rust re-writes verbatim (main.rs L739-741) — clean AND UNNORMALISED. `y_test` is the
processed field — normalised, and (post-fix) clean. So the two differ by an affine map plus, formerly,
noise; hence the linregress-residual trick at utils.py L218-221.

CONSEQUENCE OF THE FIX (must not be missed): with test noise removed, `y_true_noisy == affine(y_true_original)`
exactly, so `injected_noise` collapses to ~0 (f32 rounding) for every test row. Any figure or analysis
reading `injected_noise`, or contrasting aleatoric uncertainty against it, silently changes meaning after
the re-run. The 4000 call site passes `y_test_original.cpu().numpy().flatten()`; 6851/6977/7183 omit the
calibration and decomposition kwargs, so those rows get calibrated=uncalibrated and NaN decomposition.
