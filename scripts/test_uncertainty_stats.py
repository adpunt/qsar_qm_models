"""Tests for scripts/uncertainty_stats.py.

Every test EXECUTES the module against simulated data whose answer is known by
construction. Nothing here searches the source for a string.

Run it directly:

    python scripts/test_uncertainty_stats.py

or under pytest:

    pytest scripts/test_uncertainty_stats.py -q

The simulations contain no leakage: the predicted label is never a function of
the injected noise. That is the point of the permutation test below — the naive
null declares a leak on data that provably has none, and the correct null does
not.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from uncertainty_stats import (  # noqa: E402
    ConditioningError,
    UncertaintySchemaError,
    assert_single_cell,
    check_noise_scale_redundancy,
    check_pattern_invariance,
    confound_controlled_effect,
    load_uncertainty,
    permutation_null,
    q4_error_ratio,
    q4_plain_correlation,
    q5_mean_uncertainty,
    q6_error_ranking,
)

RESULTS = []


def _record(name, detail):
    RESULTS.append((name, detail))
    print(f"    {detail}")


# ---------------------------------------------------------------------------
# frame builders
# ---------------------------------------------------------------------------

def _cell(n, sigma, rng, *, condition='legacy', model='sim_model',
          rep='sim_rep', dataset='sim', fold='0', split='train_oof',
          model_error_sd=0.57, uncertainty=None, pattern=None,
          pattern_pred=None, eps=None):
    """One conditioned cell with no leakage: y_pred never sees the noise."""
    y = rng.normal(0.0, 3.0, n)
    if eps is None:
        eps = rng.normal(0.0, sigma, n) if sigma > 0 else np.zeros(n)
    y_pred = y + rng.normal(0.0, model_error_sd, n)
    if uncertainty is None:
        uncertainty = np.abs(rng.normal(0.0, 1.0, n)) + 0.1
    if pattern is None:
        pattern = np.ones(n)
    if pattern_pred is None:
        pattern_pred = np.ones(n)
    return pd.DataFrame({
        'dataset': dataset, 'model': model, 'rep': rep, 'condition': condition,
        'sigma': float(sigma), 'fold': str(fold), 'split': split,
        'mol_id': np.arange(n), 'sample_idx': np.arange(n),
        'y_true_clean': y, 'y_pred': y_pred, 'uncertainty': uncertainty,
        'injected_noise': eps,
        'noise_scale': float(sigma) * pattern,
        'noise_pattern': pattern,
        'noise_pattern_pred': pattern_pred,
        'oof_folds_ok': 5, 'source_file': 'simulated',
    })


# ---------------------------------------------------------------------------
# 1. no signal returns near zero
# ---------------------------------------------------------------------------

def test_no_signal_is_near_zero():
    rng = np.random.default_rng(11)
    df = _cell(3000, 1.0, rng)
    out = q4_plain_correlation(df)
    assert len(out) == 1
    rho = float(out['rho_raw'].iloc[0])
    _record('no_signal', f"q4_plain_correlation rho_raw = {rho:+.4f} "
                         f"(uncertainty independent of the injected noise)")
    assert abs(rho) < 0.06, rho

    # and the confound-controlled effect on a flat pattern is undefined, not a
    # number quietly close to zero
    out_b = confound_controlled_effect(df)
    assert bool(out_b['pattern_constant'].iloc[0]) is True
    assert np.isnan(out_b['rho_pattern'].iloc[0])
    _record('no_signal_pattern',
            "a constant noise_pattern gives NaN, not a spurious 0.0")


def test_q4_plain_correlation_names_itself_as_not_the_answer():
    rng = np.random.default_rng(12)
    df = pd.concat([_cell(600, 0.0, rng), _cell(600, 1.0, rng)],
                   ignore_index=True)
    out = q4_plain_correlation(df)
    assert set(out['statistic']) == {'q4_plain_correlation_NOT_THE_ANSWER'}
    zero_row = out[out['sigma'] == 0.0].iloc[0]
    one_row = out[out['sigma'] == 1.0].iloc[0]
    assert bool(zero_row['noise_size_constant']) is True
    assert np.isnan(zero_row['rho_raw'])
    # the zero-level subtraction is degenerate for this statistic and must not
    # silently become "minus zero"
    assert bool(one_row['baseline_defined']) is False
    assert np.isnan(one_row['rho_baselined'])
    assert np.isfinite(one_row['rho_raw'])
    _record('plain_baseline_degenerate',
            f"at level 0 injected_noise is constant -> baseline NaN, "
            f"rho_baselined NaN, rho_raw={one_row['rho_raw']:+.4f} still reported")


# ---------------------------------------------------------------------------
# 2. planted signal is recovered
# ---------------------------------------------------------------------------

def _confound_frames(rng, n=3000, detection_gain=0.0, track_prediction=False):
    """A quantile-style condition: molecules above the 75th percentile of the
    label get three times the noise. `noise_pattern` is that shape at reference
    level 1.0 and is identical at every level, including zero.

    The confound is planted deliberately: uncertainty always rises with |label|,
    which correlates with the pattern. Only `detection_gain` adds a genuine
    dependence on the pattern itself.
    """
    n_tot = n
    y = rng.normal(0.0, 3.0, n_tot)
    thr = np.quantile(y, 0.75)
    pattern = np.where(y > thr, 3.0, 1.0)
    y_pred = y + rng.normal(0.0, 1.0, n_tot)
    thr_p = np.quantile(y_pred, 0.75)
    pattern_pred = np.where(y_pred > thr_p, 3.0, 1.0)

    frames = []
    for sigma in (0.0, 1.0):
        eps = rng.normal(0.0, sigma, n_tot) * pattern if sigma > 0 else np.zeros(n_tot)
        base = np.abs(y) * 0.5 + 0.3 * np.abs(rng.normal(0.0, 1.0, n_tot))
        unc = base.copy()
        if sigma > 0:
            if track_prediction:
                unc = base + detection_gain * pattern_pred
            else:
                unc = base + detection_gain * pattern
        frames.append(pd.DataFrame({
            'dataset': 'sim', 'model': 'sim_model', 'rep': 'sim_rep',
            'condition': 'quantile', 'sigma': float(sigma), 'fold': '0',
            'split': 'train_oof', 'mol_id': np.arange(n_tot),
            'sample_idx': np.arange(n_tot),
            'y_true_clean': y, 'y_pred': y_pred, 'uncertainty': unc,
            'injected_noise': eps,
            'noise_scale': float(sigma) * pattern,
            'noise_pattern': pattern,
            'noise_pattern_pred': pattern_pred,
            'oof_folds_ok': 5, 'source_file': 'simulated',
        }))
    return pd.concat(frames, ignore_index=True)


def test_confound_control_recovers_planted_signal_and_rejects_the_confound():
    rng = np.random.default_rng(21)
    null_df = _confound_frames(rng, detection_gain=0.0)
    out = confound_controlled_effect(null_df)
    row = out[out['sigma'] == 1.0].iloc[0]
    _record('confound_null',
            f"no planted effect: rho={row['rho_pattern']:+.3f}, "
            f"rho at level 0={row['rho_pattern_at_sigma0']:+.3f}, "
            f"effect={row['effect']:+.3f}")
    # the raw correlation is large -- that is the confound, and it is exactly
    # what the zero-level subtraction is for
    assert row['rho_pattern'] > 0.3, row['rho_pattern']
    assert abs(row['effect']) < 0.05, row['effect']

    rng = np.random.default_rng(22)
    sig_df = _confound_frames(rng, detection_gain=1.5)
    out = confound_controlled_effect(sig_df)
    row = out[out['sigma'] == 1.0].iloc[0]
    _record('confound_signal',
            f"planted effect: rho={row['rho_pattern']:+.3f}, "
            f"rho at level 0={row['rho_pattern_at_sigma0']:+.3f}, "
            f"effect={row['effect']:+.3f}, "
            f"sham ceiling effect_pred={row['effect_pred']:+.3f}, "
            f"is_detection={bool(row['is_detection'])}")
    assert row['effect'] > 0.15, row['effect']
    assert bool(row['is_detection']) is True


def test_sham_ceiling_catches_a_model_tracking_its_own_prediction():
    rng = np.random.default_rng(23)
    df = _confound_frames(rng, detection_gain=1.5, track_prediction=True)
    out = confound_controlled_effect(df)
    row = out[out['sigma'] == 1.0].iloc[0]
    _record('sham_ceiling',
            f"uncertainty built from the PREDICTED shape: effect="
            f"{row['effect']:+.3f} vs effect_pred={row['effect_pred']:+.3f} "
            f"-> is_detection={bool(row['is_detection'])}")
    assert row['effect_pred'] > row['effect']
    assert bool(row['is_detection']) is False


def test_error_ratio_planted_signal_and_negative_control():
    """The uncertainty that IS the model's own error scale should improve the
    ranking of corrupted labels; an uncertainty that is pure noise should not."""
    rng = np.random.default_rng(31)
    n = 4000
    s = np.exp(rng.normal(0.0, 1.5, n))          # per-molecule model error scale
    y = rng.normal(0.0, 3.0, n)
    y_pred = y + s * rng.normal(0.0, 1.0, n)     # never sees the injected noise
    corrupted = rng.random(n) < 0.10
    eps = np.where(corrupted, 2.0 * rng.choice([-1.0, 1.0], n), 0.0)

    def frame(unc):
        return pd.DataFrame({
            'dataset': 'sim', 'model': 'm', 'rep': 'r', 'condition': 'outlier',
            'sigma': 1.0, 'fold': '0', 'split': 'train_oof',
            'mol_id': np.arange(n), 'sample_idx': np.arange(n),
            'y_true_clean': y, 'y_pred': y_pred, 'uncertainty': unc,
            'injected_noise': eps, 'noise_scale': np.nan,
            'noise_pattern': np.nan, 'noise_pattern_pred': np.nan,
            'oof_folds_ok': 5, 'source_file': 'simulated'})

    good = q4_error_ratio(frame(s)).iloc[0]
    _record('error_ratio_signal',
            f"uncertainty = true error scale: rho {good['rho_error']:+.3f} -> "
            f"{good['rho_ratio']:+.3f} (delta {good['rho_delta']:+.3f}); "
            f"AUC {good['auc_error']:.3f} -> {good['auc_ratio']:.3f} "
            f"(delta {good['auc_delta']:+.3f})")
    assert good['rho_delta'] > 0.01, good['rho_delta']
    assert good['auc_delta'] > 0.01, good['auc_delta']

    junk = np.abs(rng.normal(0.0, 1.0, n)) + 0.1
    bad = q4_error_ratio(frame(junk)).iloc[0]
    _record('error_ratio_control',
            f"uncertainty = independent noise: rho {bad['rho_error']:+.3f} -> "
            f"{bad['rho_ratio']:+.3f} (delta {bad['rho_delta']:+.3f}); "
            f"AUC {bad['auc_error']:.3f} -> {bad['auc_ratio']:.3f} "
            f"(delta {bad['auc_delta']:+.3f})")
    assert bad['rho_delta'] < 0.0, bad['rho_delta']
    assert bad['auc_delta'] < 0.0, bad['auc_delta']
    assert good['auc_delta'] > bad['auc_delta']


def test_error_ratio_is_nan_at_the_zero_level():
    rng = np.random.default_rng(32)
    out = q4_error_ratio(_cell(500, 0.0, rng)).iloc[0]
    assert np.isnan(out['rho_error']) and np.isnan(out['auc_error'])
    _record('error_ratio_zero_level',
            "at level 0 the target is constant, so every column is NaN "
            "(the negative control working)")


# ---------------------------------------------------------------------------
# 3. the permutation null — the naive one fires on clean data, the correct one
#    does not
# ---------------------------------------------------------------------------

def test_permutation_null_naive_versus_correct():
    rng = np.random.default_rng(41)
    n = 3000
    y = rng.normal(0.0, 3.0, n)
    eps = rng.normal(0.0, 1.0, n)
    # NO LEAKAGE by construction: the prediction is the clean label plus its own
    # independent error and never touches eps.
    y_pred = y + rng.normal(0.0, 0.57, n)
    df = pd.DataFrame({
        'dataset': 'sim', 'model': 'm', 'rep': 'r', 'condition': 'legacy',
        'sigma': 1.0, 'fold': '0', 'split': 'train_oof',
        'mol_id': np.arange(n), 'sample_idx': np.arange(n),
        'y_true_clean': y, 'y_pred': y_pred,
        'uncertainty': np.abs(rng.normal(0.0, 1.0, n)) + 0.1,
        'injected_noise': eps, 'noise_scale': np.nan, 'noise_pattern': np.nan,
        'noise_pattern_pred': np.nan, 'oof_folds_ok': 5,
        'source_file': 'simulated'})

    naive = permutation_null(df, 'error_noise_spearman', n_permutations=300,
                             recompute_error=False, seed=1).iloc[0]
    correct = permutation_null(df, 'error_noise_spearman', n_permutations=300,
                               recompute_error=True, seed=1).iloc[0]

    _record('perm_naive',
            f"NAIVE null (error left as computed): observed "
            f"{naive['observed']:+.3f}, band [{naive['null_lo']:+.3f}, "
            f"{naive['null_hi']:+.3f}], inside={bool(naive['observed_inside_null'])}, "
            f"p={naive['p_value']:.4f}  <- fires on data with no leakage")
    _record('perm_correct',
            f"CORRECT null (error recomputed from the permuted noise): observed "
            f"{correct['observed']:+.3f}, band [{correct['null_lo']:+.3f}, "
            f"{correct['null_hi']:+.3f}], inside="
            f"{bool(correct['observed_inside_null'])}, p={correct['p_value']:.4f}")

    # the observed value is the same statistic in both cases
    assert np.isclose(naive['observed'], correct['observed'])
    assert 0.50 < correct['observed'] < 0.75, correct['observed']

    # the naive null is centred on zero and excludes the observed value: a
    # false positive on a simulation that has no leakage at all
    assert abs(naive['null_mean']) < 0.02, naive['null_mean']
    assert naive['null_lo'] > -0.07 and naive['null_hi'] < 0.07
    assert not bool(naive['observed_inside_null'])
    assert naive['p_value'] < 0.01

    # the correct null is centred on the observed value and contains it
    assert abs(correct['null_mean'] - correct['observed']) < 0.05
    assert bool(correct['observed_inside_null'])
    assert correct['p_value'] > 0.05

    assert naive['null_kind'] == 'naive_UNSOUND'
    assert correct['null_kind'] == 'correct_recomputed'


def test_permutation_null_detects_real_leakage():
    """The correct null must still fire when the prediction really did see the
    noise, or it would be a null that never rejects anything."""
    rng = np.random.default_rng(42)
    n = 3000
    y = rng.normal(0.0, 3.0, n)
    eps = rng.normal(0.0, 1.0, n)
    y_pred = y + rng.normal(0.0, 0.57, n)
    unc = 0.2 + 3.0 * np.abs(eps)      # leakage: uncertainty knows the draw
    df = pd.DataFrame({
        'dataset': 'sim', 'model': 'm', 'rep': 'r', 'condition': 'legacy',
        'sigma': 1.0, 'fold': '0', 'split': 'train_oof',
        'mol_id': np.arange(n), 'sample_idx': np.arange(n),
        'y_true_clean': y, 'y_pred': y_pred, 'uncertainty': unc,
        'injected_noise': eps, 'noise_scale': np.nan, 'noise_pattern': np.nan,
        'noise_pattern_pred': np.nan, 'oof_folds_ok': 5,
        'source_file': 'simulated'})
    res = permutation_null(df, 'uncertainty_noise_spearman', n_permutations=300,
                           recompute_error=True, seed=2).iloc[0]
    _record('perm_leakage',
            f"planted leakage, uncertainty-vs-noise statistic: observed "
            f"{res['observed']:+.3f}, band [{res['null_lo']:+.3f}, "
            f"{res['null_hi']:+.3f}], p={res['p_value']:.4f}")
    assert res['observed'] > 0.9
    assert not bool(res['observed_inside_null'])
    assert res['p_value'] < 0.01


# ---------------------------------------------------------------------------
# 4. the conditioning assertion
# ---------------------------------------------------------------------------

def test_conditioning_assertion_raises_on_a_pooled_frame():
    rng = np.random.default_rng(51)
    a = _cell(400, 1.0, rng, model='model_a')
    b = _cell(400, 1.0, rng, model='model_b')
    pooled = pd.concat([a, b], ignore_index=True)

    raised = None
    try:
        assert_single_cell(pooled)
    except ConditioningError as exc:
        raised = str(exc)
    assert raised is not None and 'model' in raised
    _record('conditioning_pooled', f"pooled frame rejected: {raised[:110]}")

    # pooled across noise levels too
    pooled_sigma = pd.concat([_cell(400, 0.0, rng), _cell(400, 1.0, rng)],
                             ignore_index=True)
    try:
        assert_single_cell(pooled_sigma)
        raise AssertionError("pooling across the noise level was not caught")
    except ConditioningError as exc:
        assert 'sigma' in str(exc)
    _record('conditioning_sigma', "pooling across the noise level rejected")

    # a missing conditioning column is also a failure: without it the pooling
    # would be invisible
    try:
        assert_single_cell(a.drop(columns=['condition']))
        raise AssertionError("a missing conditioning column was not caught")
    except ConditioningError as exc:
        assert 'condition' in str(exc)
    _record('conditioning_missing_column',
            "a missing conditioning column is rejected, not assumed constant")

    # one cell passes
    assert assert_single_cell(a) is True


def test_extra_group_cols_conditions_on_the_fold_as_well():
    """A fold is a separate fit on its own molecules, so a caller must be able
    to check that a pooled-over-folds result survives being split by fold."""
    rng = np.random.default_rng(53)
    frames = []
    for fold in ('0', '1', '2'):
        for sigma in (0.0, 1.0):
            frames.append(_cell(400, sigma, rng, fold=fold))
    df = pd.concat(frames, ignore_index=True)

    pooled = q6_error_ranking(df)
    by_fold = q6_error_ranking(df, extra_group_cols=['fold'])
    assert len(pooled) == 2 and len(by_fold) == 6
    assert 'fold' in by_fold.columns and by_fold['fold'].nunique() == 3
    assert int(pooled['n'].iloc[0]) == 3 * int(by_fold['n'].iloc[0])

    # the zero-level subtraction must then match WITHIN a fold, not across
    cc = confound_controlled_effect(
        _confound_frames(np.random.default_rng(54), detection_gain=1.5),
        extra_group_cols=['fold'])
    assert 'fold' in cc.columns
    assert cc[cc['sigma'] == 1.0]['rho_pattern_at_sigma0'].notna().all()
    _record('extra_group_cols',
            f"q6 gives {len(pooled)} rows pooled over folds and {len(by_fold)} "
            f"split by fold; the zero-level baseline still matches within a fold")


def test_statistics_condition_before_correlating():
    """The public functions group first, so a pooled frame gives one row per
    cell rather than one pooled number."""
    rng = np.random.default_rng(52)
    pooled = pd.concat([_cell(400, 1.0, rng, model='model_a'),
                        _cell(400, 1.0, rng, model='model_b'),
                        _cell(400, 0.0, rng, model='model_a'),
                        _cell(400, 0.0, rng, model='model_b')],
                       ignore_index=True)
    out = q4_plain_correlation(pooled)
    assert len(out) == 4
    assert set(out['model']) == {'model_a', 'model_b'}
    assert set(out['sigma']) == {0.0, 1.0}
    _record('statistics_condition',
            f"a frame pooling 2 models x 2 levels gives {len(out)} conditioned "
            f"rows, not 1 pooled number")


# ---------------------------------------------------------------------------
# 5. questions 5 and 6
# ---------------------------------------------------------------------------

def test_q5_is_labelled_a_population_statement_and_tracks_the_level():
    rng = np.random.default_rng(61)
    frames = []
    for sigma in (0.0, 0.3, 0.6, 1.0):
        n = 500
        unc = 0.5 + 0.8 * sigma + np.abs(rng.normal(0.0, 0.1, n))
        frames.append(_cell(n, sigma, rng, uncertainty=unc))
    df = pd.concat(frames, ignore_index=True)
    out = q5_mean_uncertainty(df)
    assert set(out['level_of_inference']) == {'population_not_per_molecule'}
    assert len(out) == 4
    slope = float(out['slope_mean_unc_vs_sigma'].iloc[0])
    _record('q5', "mean uncertainty by level: "
                  + ", ".join(f"sigma={r.sigma:.1f} -> {r.mean_uncertainty:.3f}"
                              for r in out.sort_values('sigma').itertuples())
                  + f"; slope={slope:+.3f}")
    assert 0.7 < slope < 0.9
    assert float(out['rho_mean_unc_vs_sigma'].iloc[0]) > 0.99


def test_q6_uses_the_clean_label_and_is_computed_per_level():
    """Built so the two choices give visibly different answers: the uncertainty
    tracks the error against the CLEAN label and knows nothing about the
    injected noise."""
    rng = np.random.default_rng(62)
    n = 3000
    frames = []
    for sigma in (0.0, 1.5):
        y = rng.normal(0.0, 3.0, n)
        model_err = rng.normal(0.0, 1.0, n)
        y_pred = y + model_err
        eps = rng.normal(0.0, sigma, n) if sigma > 0 else np.zeros(n)
        frames.append(pd.DataFrame({
            'dataset': 'sim', 'model': 'm', 'rep': 'r', 'condition': 'legacy',
            'sigma': float(sigma), 'fold': '0', 'split': 'test',
            'mol_id': np.arange(n), 'sample_idx': np.arange(n),
            'y_true_clean': y, 'y_pred': y_pred,
            'uncertainty': np.abs(model_err),   # perfect knowledge of the clean error
            'injected_noise': eps, 'noise_scale': np.nan,
            'noise_pattern': np.nan, 'noise_pattern_pred': np.nan,
            'oof_folds_ok': -1, 'source_file': 'simulated'}))
    df = pd.concat(frames, ignore_index=True)
    out = q6_error_ranking(df)
    assert len(out) == 2
    assert set(out['error_reference']) == {'clean_label'}
    for _, r in out.iterrows():
        assert r['rho_unc_vs_clean_error'] > 0.99, r.to_dict()
    _record('q6_clean',
            "uncertainty = |clean error| gives rho "
            + ", ".join(f"{r.rho_unc_vs_clean_error:.4f} at sigma={r.sigma:.1f}"
                        for r in out.itertuples())
            + " at every level")

    # against the noisy label the same data gives a materially smaller number,
    # which is why the reference matters
    noisy = df.copy()
    noisy['y_true_clean'] = noisy['y_true_clean'] + noisy['injected_noise']
    out_noisy = q6_error_ranking(noisy)
    rho_noisy = float(out_noisy[out_noisy['sigma'] == 1.5]['rho_unc_vs_clean_error'].iloc[0])
    _record('q6_noisy_reference',
            f"the same rows scored against the NOISY label give "
            f"{rho_noisy:+.4f} instead of ~1.0 at sigma=1.5")
    assert rho_noisy < 0.85


# ---------------------------------------------------------------------------
# 6. the loader
# ---------------------------------------------------------------------------

def _write_qm9_file(path, n=60, sigmas=(0.0, 0.6), condition_column=True):
    rng = np.random.default_rng(71)
    rows = []
    smiles = [f"C{'C' * (i % 7)}O" for i in range(n)]
    pattern = np.where(np.arange(n) % 4 == 0, 3.0, 1.0)
    for sigma in sigmas:
        for split in ('test', 'train_oof'):
            y = rng.normal(0.0, 3.0, n)
            eps = (rng.normal(0.0, sigma, n) * pattern
                   if (sigma > 0 and split == 'train_oof') else np.zeros(n))
            rec = {
                'model': 'qrf', 'representation': 'pdv', 'sigma': sigma,
                'iteration': 0, 'file_no': 205052201,
                'sample_idx': np.arange(n),
                'y_pred_mean': y + rng.normal(0.0, 1.0, n),
                'y_pred_std_uncalibrated': np.abs(rng.normal(0.0, 1.0, n)) + 0.1,
                'y_true_original': y, 'y_true_noisy': y + eps,
                'injected_noise': eps,
                'y_pred_std_calibrated': np.nan, 'temperature': 1.0,
                'epistemic_uncertainty': np.nan, 'aleatoric_uncertainty': np.nan,
                'split': split, 'canonical_smiles': smiles,
                'noise_scale': sigma * pattern if split == 'train_oof' else 0.0,
                'noise_pattern': pattern,
                'noise_pattern_pred': pattern,
                'oof_folds_ok': 5 if split == 'train_oof' else -1,
            }
            if condition_column:
                rec['strategy'] = 'quantile'
            rows.append(pd.DataFrame(rec))
    df = pd.concat(rows, ignore_index=True)
    df['y_pred_std_calibrated'] = df['y_pred_std_uncalibrated'] * 2.0
    df.to_csv(path, index=False)
    return df


def _write_kirby_file(path, n=60):
    rng = np.random.default_rng(72)
    rows = []
    pattern = np.where(np.arange(n) % 4 == 0, 3.0, 1.0)
    for sigma in (0.0, 0.6):
        for split in ('test', 'train_oof'):
            y = rng.normal(0.0, 2.0, n)
            eps = (rng.normal(0.0, sigma, n) * pattern
                   if (sigma > 0 and split == 'train_oof') else np.zeros(n))
            rows.append(pd.DataFrame({
                'split': split, 'noise_type': 'outlier', 'sigma': sigma,
                'fold': 2, 'sample_idx': np.arange(n), 'mol_idx': np.arange(n),
                'y_true': y, 'y_pred': y + rng.normal(0.0, 0.8, n),
                'uncertainty': np.abs(rng.normal(0.0, 1.0, n)) + 0.1,
                'noise_scale': sigma * pattern if split == 'train_oof' else 0.0,
                'noise_pattern': pattern, 'noise_pattern_pred': pattern,
                'injected_noise': eps,
                'oof_folds_ok': 5 if split == 'train_oof' else -1,
                'dataset': 'herg_ki', 'model': 'NGBoost', 'rep': 'ECFP4'}))
    df = pd.concat(rows, ignore_index=True)
    df.to_csv(path, index=False)
    return df


def test_loader_reads_both_schemas_into_one_frame():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        qm9 = tmp / 'uncertainty_quantile_pdv_qrf_uncertainty_values.csv'
        kirby = tmp / 'NGBoost_ECFP4_uncertainty_values.csv'
        raw_qm9 = _write_qm9_file(qm9)
        raw_kirby = _write_kirby_file(kirby)

        df = load_uncertainty(tmp)
        assert len(df) == len(raw_qm9) + len(raw_kirby)
        assert set(df['dataset']) == {'QM9', 'herg_ki'}
        assert set(df['condition']) == {'quantile', 'outlier'}
        assert set(df['split']) == {'test', 'train_oof'}
        _record('loader_both',
                f"loaded {len(df)} rows from both schemas: "
                f"datasets {sorted(set(df['dataset']))}, "
                f"conditions {sorted(set(df['condition']))}, "
                f"splits {sorted(set(df['split']))}")

        # QM9's clean label is y_true_original, not y_true_noisy
        q = df[df['dataset'] == 'QM9']
        src = raw_qm9
        assert np.allclose(sorted(q['y_true_clean']), sorted(src['y_true_original']))
        # KIRBy's y_true is already the clean label
        k = df[df['dataset'] == 'herg_ki']
        assert np.allclose(sorted(k['y_true_clean']), sorted(raw_kirby['y_true']))
        # the QM9 molecule identifier survives; sample_idx alone would not link
        assert q['mol_id'].notna().all()
        assert q['mol_id'].nunique() < len(q)
        _record('loader_ids',
                f"canonical_smiles carried through as mol_id: "
                f"{q['mol_id'].nunique()} molecules across {len(q)} QM9 rows")

        # every statistic runs on the merged frame without any renaming
        for fn in (q4_plain_correlation, q4_error_ratio, q6_error_ranking,
                   q5_mean_uncertainty, confound_controlled_effect):
            res = fn(df)
            assert len(res) > 0, fn.__name__
        _record('loader_statistics',
                "all six public statistics run on the merged frame unchanged")

        # calibrated vs uncalibrated changes the scale but not any rank
        unc_a = load_uncertainty(qm9, uncertainty_column='uncalibrated')
        unc_b = load_uncertainty(qm9, uncertainty_column='calibrated')
        ra = q6_error_ranking(unc_a).sort_values(CELL := ['sigma', 'split'])
        rb = q6_error_ranking(unc_b).sort_values(CELL)
        assert np.allclose(ra['rho_unc_vs_clean_error'].to_numpy(),
                           rb['rho_unc_vs_clean_error'].to_numpy())
        assert not np.allclose(
            q5_mean_uncertainty(unc_a)['mean_uncertainty'].to_numpy(),
            q5_mean_uncertainty(unc_b)['mean_uncertainty'].to_numpy())
        _record('loader_calibration',
                "calibrated and uncalibrated give identical rank statistics and "
                "different means, as documented")


def test_loader_reads_the_condition_from_the_file_name_when_absent():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'uncertainty_hetero_ecfp4_ngboost_uncertainty_values.csv'
        _write_qm9_file(p, condition_column=False)
        df = load_uncertainty(p)
        assert set(df['condition']) == {'hetero'}
        _record('loader_condition_from_name',
                "a QM9 file with no condition column takes it from the file name")


def test_loader_refuses_a_file_predating_the_rewrite_and_names_the_column():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        p = tmp / 'uncertainty_legacy_pdv_qrf_uncertainty_values.csv'
        df = _write_qm9_file(p)
        df.drop(columns=['split', 'canonical_smiles', 'noise_pattern',
                         'noise_pattern_pred', 'noise_scale', 'oof_folds_ok']
                ).to_csv(p, index=False)
        raised = None
        try:
            load_uncertainty(p)
        except UncertaintySchemaError as exc:
            raised = str(exc)
        assert raised is not None
        for col in ('split', 'canonical_smiles', 'noise_pattern'):
            assert col in raised, raised
        _record('loader_old_qm9_refused',
                f"an old QM9 file is refused by name: ...{raised[-160:]}")

        # and the five-column experimental files that exist on disk today
        legacy = tmp / 'QRF_ECFP4_uncertainty_values.csv'
        pd.DataFrame({'sigma': [0.0] * 30, 'sample_idx': range(30),
                      'y_true': np.linspace(0, 3, 30),
                      'y_pred': np.linspace(0, 3, 30),
                      'uncertainty': np.linspace(0.1, 1, 30)}).to_csv(
            legacy, index=False)
        raised = None
        try:
            load_uncertainty(legacy)
        except UncertaintySchemaError as exc:
            raised = str(exc)
        assert raised is not None and 'split' in raised
        _record('loader_legacy_five_column',
                f"a five-column legacy file is refused: ...{raised[-140:]}")

        # non-strict loads it as test rows, and the statistics that need the
        # missing columns then refuse by name rather than inventing anything
        loose = load_uncertainty(legacy, strict=False)
        assert set(loose['split']) == {'test'}
        assert loose['model'].iloc[0] == 'QRF' and loose['rep'].iloc[0] == 'ECFP4'
        for fn in (q4_plain_correlation, q4_error_ratio):
            try:
                fn(loose)
                raise AssertionError(f"{fn.__name__} did not refuse")
            except UncertaintySchemaError as exc:
                assert 'injected_noise' in str(exc)
        _record('loader_legacy_non_strict',
                "strict=False loads the legacy file as test rows; the "
                "question-4 statistics then refuse, naming injected_noise")


def test_missing_column_is_named_not_reconstructed():
    rng = np.random.default_rng(81)
    df = _cell(200, 1.0, rng).drop(columns=['noise_pattern'])
    try:
        confound_controlled_effect(df)
        raise AssertionError("missing noise_pattern was not caught")
    except UncertaintySchemaError as exc:
        assert 'noise_pattern' in str(exc)
    _record('missing_column',
            "confound_controlled_effect names the missing column instead of "
            "reconstructing it")


# ---------------------------------------------------------------------------
# 7. the two diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics():
    rng = np.random.default_rng(91)
    pattern = np.where(np.arange(400) % 3 == 0, 3.0, 1.0)
    frames = [_cell(400, s, rng, condition='quantile', pattern=pattern)
              for s in (0.0, 0.6, 1.0)]
    df = pd.concat(frames, ignore_index=True)

    inv = check_pattern_invariance(df)
    assert bool(inv['invariant'].all())
    assert int(inv['n_levels'].max()) == 3
    _record('diag_pattern_invariance',
            f"noise_pattern identical across {int(inv['n_levels'].max())} levels "
            f"for all {len(inv)} molecules")

    red = check_noise_scale_redundancy(df)
    non_zero = red[red['sigma'] > 0]
    assert np.allclose(non_zero['rho_scale_vs_pattern'].to_numpy(), 1.0)
    assert float(red['max_abs_deviation'].max()) < 1e-12
    _record('diag_scale_redundancy',
            f"noise_scale == sigma * noise_pattern to "
            f"{float(red['max_abs_deviation'].max()):.1e}; Spearman 1.000 where "
            f"the pattern varies, so only the pattern is reported")

    # the diagnostic must actually fail when the premise fails
    broken = df.copy()
    broken.loc[broken['sigma'] == 1.0, 'noise_pattern'] = 5.0
    inv_bad = check_pattern_invariance(broken)
    assert not bool(inv_bad['invariant'].all())
    _record('diag_pattern_invariance_fails',
            "and it reports non-invariance when the pattern is corrupted")


# ---------------------------------------------------------------------------

def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = []
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            print("  PASS")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            failures.append((name, exc))
            print("  FAIL")
    print("\n" + "=" * 72)
    print(f"{len(tests) - len(failures)}/{len(tests)} tests passed")
    if failures:
        for name, exc in failures:
            print(f"  FAILED {name}: {exc}")
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
