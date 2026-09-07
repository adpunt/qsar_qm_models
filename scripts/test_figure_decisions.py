#!/usr/bin/env python
"""Each of the ten decision triggers fires on a planted answer, and stays quiet
on a null.

RERUN_PLAN.md 14.6 lists fourteen figures that exist only if the results say so.
A trigger that fires when it should not puts a figure in the paper that the data
does not support; a trigger that stays quiet when it should fire leaves one out.
Both are checked here, on frames whose answer is known before the test runs.

The hardest case, and the one that motivated this file: a statistic that was
never computed must not be reported as a null. "The uncertainty adds nothing
anywhere" and "nobody measured whether it does" produce the same empty column.

Run it directly:  python scripts/test_figure_decisions.py
"""
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figlib_config as C  # noqa: E402
import figlib_decisions as D  # noqa: E402
import figlib_metrics as M  # noqa: E402

MODELS = ['rf', 'svm', 'ngboost', 'xgboost', 'dnn', 'mlp']
REPS = ['ecfp4', 'pdv', 'chemberta']
CONDITIONS = ['gaussian', 'grouped_wider', 'grouped_shifted']


def summary(rows):
    """A robustness summary frame, the shape summarise_robustness returns."""
    df = pd.DataFrame(rows)
    df['dataset'] = df.get('dataset', 'qm9')
    for column, default in (('auc_norm_spread', 0.01), ('baseline_r2', 0.8),
                            ('n_replicates', 10)):
        if column not in df.columns:
            df[column] = default
    return df


def flat_summary(auc_of=None, spread=0.01):
    """Every model the same everywhere, unless `auc_of` says otherwise."""
    rows = []
    base = {'rf': 0.92, 'svm': 0.88, 'ngboost': 0.93, 'xgboost': 0.85,
            'dnn': 0.80, 'mlp': 0.76}
    for condition in CONDITIONS:
        for rep in REPS:
            for model in MODELS:
                value = (auc_of(model, rep, condition) if auc_of
                         else base[model])
                rows.append({'dataset': 'qm9', 'model': model, 'rep': rep,
                             'condition': condition, 'auc_norm': value,
                             'auc_norm_spread': spread})
    return summary(rows)


# ---------------------------------------------------------------------------

def d1_never_averages_over_models():
    """The model is the dominant source of variance, so a median over models
    describes the roster and not the representation.

    The first version of this table reported median_clean_r2 and
    median_auc_norm per representation, over DIFFERENT numbers of models and
    conditions -- 16 x 3 for one representation against 19 x 7 for another, in
    the same column.
    """
    rows = []
    for rep, models, conditions in (('ecfp4', MODELS, CONDITIONS),
                                    ('pdv', MODELS, CONDITIONS),
                                    ('sns', MODELS[:3], CONDITIONS[:1])):
        for condition in conditions:
            for i, model in enumerate(models):
                rows.append({'dataset': 'qm9', 'model': model, 'rep': rep,
                             'condition': condition,
                             'auc_norm': 0.95 - 0.03 * i,
                             'auc_norm_spread': 0.01, 'baseline_r2': 0.8})
    tables, v = D.d1_representation(summary(rows))
    table = tables['d1_representations']

    banned = [c for c in table.columns
              if c.startswith('median_') or 'mean' in c]
    assert not banned, (
        f'{banned} averages over models, which is what this test exists to '
        f'stop')
    assert 'NOT a median over models' in v['says'], v['says']
    print('    no column averages over models, and the verdict says why')

    # The paired evidence must be over ONE roster, or it compares different
    # experiments. sns has 3 models and 1 condition, so that is the common set.
    paired = tables['d1_auc_norm_by_model']
    assert set(paired['model']) == set(MODELS[:3]), sorted(set(paired['model']))
    assert set(paired['condition']) == set(CONDITIONS[:1]), \
        sorted(set(paired['condition']))
    assert {'ecfp4', 'pdv', 'sns'} <= set(paired.columns), list(paired.columns)
    print(f'    paired on the common roster only: {len(paired)} rows, '
          f'{paired["model"].nunique()} models every representation has')

    # And the coverage difference is stated rather than hidden inside a median.
    sns_row = table[table['rep'] == 'sns'].iloc[0]
    assert sns_row['n_models'] == 3 and sns_row['n_conditions'] == 1, dict(sns_row)
    assert sns_row['models_missing_vs_widest'] == len(MODELS) - 3
    print('    coverage is a column, not something folded into an average')


def d1_flags_replicates_that_do_not_vary():
    """A cell whose replicates are identical is a run that did not vary, not a
    model that is stable. On the first real table one representation reported a
    replicate spread of exactly 0.0."""
    rows = []
    for rep in ('ecfp4', 'sns'):
        for condition in CONDITIONS:
            for i, model in enumerate(MODELS):
                rows.append({'dataset': 'qm9', 'model': model, 'rep': rep,
                             'condition': condition,
                             'auc_norm': 0.95 - 0.03 * i,
                             'auc_norm_spread': 0.0 if rep == 'sns' else 0.02,
                             'baseline_r2': 0.8})
    tables, v = D.d1_representation(summary(rows))
    table = tables['d1_representations']
    assert int(table[table['rep'] == 'sns']['replicate_spread_zero_cells']) > 0
    assert int(table[table['rep'] == 'ecfp4']['replicate_spread_zero_cells']) == 0
    assert 'did not vary' in v['says'], v['says']
    print('    a zero replicate spread is named, not read as stability')


def d2_merges_identical_grids_and_keeps_different_ones():
    def auc(model, rep, condition):
        base = {'rf': 0.92, 'svm': 0.88, 'ngboost': 0.93, 'xgboost': 0.85,
                'dnn': 0.80, 'mlp': 0.76}[model]
        # grouped_wider is a copy of gaussian; grouped_shifted reverses the
        # order, so it cannot be a repeat of either.
        if condition == 'grouped_shifted':
            return 1.7 - base
        return base
    s = flat_summary(auc)
    per = pd.concat([s.assign(replicate=i) for i in range(10)],
                    ignore_index=True)
    per['auc_norm'] += np.random.default_rng(0).normal(0, 0.002, len(per))
    _, v = D.d2_grid_similarity(s, per)
    assert v['fired'], v['says']
    assert set(v['supplementary']) == {'grouped_wider'}, v['supplementary']
    assert 'gaussian' in v['main_text'] and 'grouped_shifted' in v['main_text']
    print(f"    main text {v['main_text']}, held back {v['supplementary']}")

    # And when every grid differs, every grid is a panel.
    def all_different(model, rep, condition):
        shift = CONDITIONS.index(condition) if condition in CONDITIONS else 0
        order = MODELS[shift:] + MODELS[:shift]
        return 0.95 - 0.03 * order.index(model)
    s2 = flat_summary(all_different)
    per2 = pd.concat([s2.assign(replicate=i) for i in range(10)],
                     ignore_index=True)
    _, v2 = D.d2_grid_similarity(s2, per2)
    assert not v2['supplementary'], v2['supplementary']
    assert 'largest figure' in v2['says'], v2['says']
    print('    every grid different -> every grid is a main-text panel')


def d3_fires_when_conditions_differ_and_not_when_they_do_not():
    identical = flat_summary()
    per = pd.concat([identical.assign(replicate=i) for i in range(10)],
                    ignore_index=True)
    _, v = D.d3_condition_separation(per, identical, 'pdv')
    assert not v['fired'], v['says']
    assert 'Only the AMOUNT matters' in v['says'], v['says']
    print(f"    identical conditions -> not fired, W = {v['kendall_w']:.3f}")

    def separated(model, rep, condition):
        base = {'rf': 0.92, 'svm': 0.88, 'ngboost': 0.93, 'xgboost': 0.85,
                'dnn': 0.80, 'mlp': 0.76}[model]
        return base - (0.25 if condition == 'grouped_shifted' else 0.0)
    s = flat_summary(separated)
    per2 = pd.concat([s.assign(replicate=i) for i in range(10)],
                     ignore_index=True)
    _, v2 = D.d3_condition_separation(per2, s, 'pdv')
    assert v2['fired'], v2['says']
    assert v2['n_significant'] > 0, v2
    print(f"    one condition 0.25 worse -> fired, "
          f"{v2['n_significant']} significant pair(s)")


def d5_finds_a_representation_outlier_and_ignores_a_baseline_one():
    def outlier(model, rep, condition):
        base = {'rf': 0.92, 'svm': 0.88, 'ngboost': 0.93, 'xgboost': 0.85,
                'dnn': 0.80, 'mlp': 0.76}[model]
        return base - (0.30 if (model == 'mlp' and rep == 'chemberta') else 0)
    _, v = D.d5_representation_outlier(flat_summary(outlier))
    assert v['fired'] and v['n_outliers'] == len(CONDITIONS), v
    assert 'NN-β' in v['says'] and 'ChemBERTa' in v['says'], v['says']
    print(f"    {v['n_outliers']} outlier cell(s), named in plain language")

    _, v2 = D.d5_representation_outlier(flat_summary())
    assert not v2['fired'], v2['says']
    print('    every representation alike -> not fired')

    # A model can be far worse on one representation and still not be an
    # AUC_norm outlier: the metric divides the clean baseline out, so a low
    # STARTING point does not move it. Guard 4 in one sentence.
    s = flat_summary()
    s.loc[(s['model'] == 'mlp') & (s['rep'] == 'chemberta'),
          'baseline_r2'] = 0.2
    _, v3 = D.d5_representation_outlier(s)
    assert not v3['fired'], (
        'a low clean baseline was read as a robustness outlier; AUC_norm '
        'divides the baseline out by construction')
    print('    a weak clean baseline alone is NOT a robustness outlier')


def d6_counts_cells_above_one_beside_their_baseline():
    s = flat_summary()
    s.loc[0, 'auc_norm'] = 1.21
    tables, v = D.d6_auc_above_one(s)
    assert v['fired'] and v['n_cell_medians'] == 1, v
    assert 'baseline_r2' in tables['d6_auc_above_one'].columns, (
        'a value above 1 must be printed beside what it started from')
    print(f"    1 cell above {C.AUC_NORM_IMPLAUSIBLE_HIGH}, with its baseline")
    _, v2 = D.d6_auc_above_one(flat_summary())
    assert not v2['fired'], v2['says']
    print('    nothing above 1 -> not fired')


def d6_does_not_report_zero_while_the_replicates_say_otherwise():
    """On the first real run the loader warned about 67 replicate values over
    the line and D6 reported "0 of 317 cells". Both were true -- one counts cell
    medians, the other counts replicates -- and together they read as a
    contradiction. That is failure mode 12 with two granularities instead of two
    names."""
    medians = flat_summary()                       # every median well under 1.05
    per_replicate = pd.concat(
        [medians.assign(replicate=i) for i in range(10)], ignore_index=True)
    per_replicate.loc[:6, 'auc_norm'] = 1.4        # a few replicates over it
    _, v = D.d6_auc_above_one([medians], [per_replicate])
    assert v['n_cell_medians'] == 0, v
    assert v['n_replicates'] == 7, v
    assert v['fired'], 'replicates over the line must still fire'
    assert 'run-to-run spread' in v['says'], v['says']
    print(f"    0 cell medians but {v['n_replicates']} replicates over the "
          f"line: both reported, and named as spread")


def d6_covers_every_dataset_not_only_qm9():
    """The assay side held 59 of the 67, and D6 was only ever handed QM9."""
    qm9 = flat_summary()
    assay = flat_summary()
    assay['dataset'] = 'logd'
    assay.loc[0, 'auc_norm'] = 1.30
    _, v = D.d6_auc_above_one([qm9, assay])
    assert v['n_cell_medians'] == 1, v
    print('    a cell over the line on an assay dataset is counted')


def a_missing_permutation_band_is_not_reported_as_a_null():
    """The failure this whole file exists for.

    Without a band every `auc_delta` looks the same as every other, so "the
    uncertainty adds nothing anywhere" and "nobody measured it" produce
    identical data. Reporting the first would put a null in the paper that no
    test produced.
    """
    no_band = pd.DataFrame({
        'dataset': 'qm9', 'model': ['qrf', 'ngboost'], 'rep': 'pdv',
        'condition': 'gaussian', 'sigma': 1.0, 'fold': '0',
        'auc_error': [0.6, 0.6], 'auc_ratio': [0.61, 0.59],
        'auc_delta': [0.01, -0.01], 'rho_delta': [0.01, -0.01]})
    _, v = D.d7_uncertainty_option(no_band, None, None)
    assert v['option'] == 'undecided', v
    assert not v['fired'], v
    assert 'about the analysis rather than about the models' in v['says'], \
        v['says']
    print('    no band -> "undecided", not "adds nothing"')

    inside = no_band.assign(observed_inside_null=True, outside_null=False,
                            null_lo=-0.05, null_hi=0.05, p_value=0.5)
    _, v2 = D.d7_uncertainty_option(inside, None, None)
    assert v2['option'] == '7C', v2
    assert v2['clean_null'], v2
    print('    band computed and nothing outside it -> 7C, the null as a figure')


def d7_picks_the_censoring_figure_when_censoring_fires():
    rows = pd.DataFrame({
        'dataset': 'qm9', 'model': ['qrf', 'qrf'], 'rep': 'pdv',
        'condition': ['gaussian', 'censoring'], 'sigma': 1.0, 'fold': '0',
        'auc_error': [0.60, 0.60], 'auc_ratio': [0.61, 0.82],
        'auc_delta': [0.01, 0.22], 'rho_delta': [0.01, 0.20],
        'observed_inside_null': [True, False],
        'outside_null': [False, True],
        'null_lo': -0.05, 'null_hi': 0.05, 'p_value': [0.5, 0.001]})
    _, v = D.d7_uncertainty_option(rows, None, None)
    assert v['option'] == '7B' and v['censoring_fires'], v
    assert 'paper' in v['says'], v['says']
    print("    censoring outside its band -> 7B, the enrichment curve")

    only_ranking = rows[rows['condition'] == 'gaussian'].copy()
    q6 = pd.DataFrame({'dataset': 'qm9', 'model': ['qrf'], 'rep': ['pdv'],
                       'condition': ['gaussian'],
                       'rho_unc_vs_clean_error': [0.55]})
    _, v2 = D.d7_uncertainty_option(only_ranking, q6, None)
    assert v2['option'] == '7A', v2
    print('    error ranking well above zero -> 7A, the retention curve')


def d10_pairs_on_the_replicate_and_never_across_representations():
    rng = np.random.default_rng(3)
    rows = []
    for rep in REPS:
        for replicate in range(10):
            for model, value in (('rf', 0.80), ('qrf', 0.86)):
                rows.append({'dataset': 'qm9', 'model': model, 'rep': rep,
                             'condition': 'gaussian', 'replicate': replicate,
                             'auc_norm': value + rng.normal(0, 0.005),
                             'baseline_r2': 0.8})
    table, v = D.d10_probabilistic(pd.DataFrame(rows))
    got = table['d10_probabilistic']
    assert set(got['rep']) == set(REPS), (
        'the comparison must be reported per representation, never pooled')
    assert (got['paired_on'] == 'replicate').all(), got['paired_on'].unique()
    assert (got['n_pairs'] == 10).all(), got['n_pairs'].unique()
    assert v['n_wins'] == len(REPS), v
    print(f"    {len(got)} comparisons, one per representation, "
          f"paired on the replicate, {v['n_wins']} win(s)")


def d0_says_when_the_grid_is_incomplete():
    rows = []
    for condition in ['gaussian']:
        for rep in REPS:
            for model in MODELS:
                levels = (C.expected_levels(condition)
                          if model != 'mlp' else [0.0, 0.2])
                for sigma in levels:
                    for replicate in range(10):
                        rows.append({'dataset': 'qm9', 'model': model,
                                     'rep': rep, 'condition': condition,
                                     'sigma': sigma, 'replicate': replicate,
                                     'r2': 0.8, 'gp_collapsed': 0})
    tables, v = D.d0_coverage(pd.DataFrame(rows), None)
    assert v['fired'], v['says']
    cover = tables['d0_coverage']
    partial = cover[cover['status'] == 'PARTIAL_LEVELS']
    assert set(partial['model']) == {'mlp'}, set(partial['model'])
    assert 'no headline may be quoted' in v['says'], v['says']
    print(f"    {len(partial)} cell(s) short of the ladder, and it says so")


def d9_promotes_the_table_when_the_ranking_does_not_transfer():
    qm9 = flat_summary()
    order = {m: 0.95 - 0.03 * i for i, m in enumerate(MODELS)}
    reverse = {m: 0.95 - 0.03 * i for i, m in enumerate(reversed(MODELS))}
    qm9['auc_norm'] = qm9['model'].map(order)
    assay = qm9.copy()
    assay['dataset'] = 'logd'
    assay['auc_norm'] = assay['model'].map(reverse)
    _, v = D.d9_rank_transfer(qm9, assay)
    assert v['fired'] and v['median_rho'] < 0, v
    assert 'promoted to a figure' in v['says'], v['says']
    print(f"    reversed ranking -> rho {v['median_rho']:.2f}, T7 promoted")

    agrees = qm9.copy(); agrees['dataset'] = 'logd'
    _, v2 = D.d9_rank_transfer(qm9, agrees)
    assert not v2['fired'] and v2['median_rho'] > 0.9, v2
    print(f"    same ranking -> rho {v2['median_rho']:.2f}, T7 stays a table")


def d4_promotes_simple_effects_only_when_the_pairing_wins():
    dominates = pd.DataFrame([{'condition': 'gaussian', 'eta2_model': 20.0,
                               'eta2_rep': 15.0, 'eta2_interaction': 50.0,
                               'eta2_residual': 15.0}])
    _, v = D.d4_interaction(dominates)
    assert v['fired'] and 'promoted' in v['says'], v
    print('    pairing above both main effects -> promoted to an F2 panel')
    model_wins = dominates.assign(eta2_model=60.0, eta2_interaction=10.0)
    _, v2 = D.d4_interaction(model_wins)
    assert not v2['fired'], v2['says']
    print('    model effect largest -> no extra panel')


def check(name, fn):
    print(f'  {name}')
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f'    FAIL: {type(exc).__name__}: {exc}')
        traceback.print_exc()
        return False
    print('    ok')
    return True


def main():
    print('the ten decision triggers (RERUN_PLAN.md 14.6, 14.9)')
    results = [
        check('D0 says when the grid is incomplete',
              d0_says_when_the_grid_is_incomplete),
        check('D1 never averages over models', d1_never_averages_over_models),
        check('D1 flags replicates that do not vary',
              d1_flags_replicates_that_do_not_vary),
        check('D2 merges identical grids, keeps different ones',
              d2_merges_identical_grids_and_keeps_different_ones),
        check('D3 fires when the conditions differ and not when they do not',
              d3_fires_when_conditions_differ_and_not_when_they_do_not),
        check('D4 promotes simple effects only when the pairing wins',
              d4_promotes_simple_effects_only_when_the_pairing_wins),
        check('D5 finds a representation outlier, ignores a baseline one',
              d5_finds_a_representation_outlier_and_ignores_a_baseline_one),
        check('D6 counts cells above 1 beside their baseline',
              d6_counts_cells_above_one_beside_their_baseline),
        check('D6 does not report zero while the replicates say otherwise',
              d6_does_not_report_zero_while_the_replicates_say_otherwise),
        check('D6 covers every dataset, not only QM9',
              d6_covers_every_dataset_not_only_qm9),
        check('a missing permutation band is NOT reported as a null',
              a_missing_permutation_band_is_not_reported_as_a_null),
        check('D7 picks the censoring figure when censoring fires',
              d7_picks_the_censoring_figure_when_censoring_fires),
        check('D9 promotes T7 when the ranking does not transfer',
              d9_promotes_the_table_when_the_ranking_does_not_transfer),
        check('D10 pairs on the replicate, never across representations',
              d10_pairs_on_the_replicate_and_never_across_representations),
    ]
    if not all(results):
        print('\nFAIL: a trigger fires on the wrong evidence')
        return 1
    print('\nOK: every trigger fires on its planted answer and on nothing else')
    return 0


if __name__ == '__main__':
    sys.exit(main())
