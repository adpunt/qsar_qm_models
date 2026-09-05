#!/usr/bin/env python
"""The averaging guard, and the five other assertions the figure script owns.

RERUN_PLAN.md 0.6 assigns failure modes 1, 3, 4, 8, 9 and 12 to this script.
Section 14.2 asks for one test by name: that REMOVING a declaration makes the
run fail. A guard you can silently opt out of would not have caught the line the
paper's cross-dataset claim rests on.

Everything here runs the real functions on real frames.

Run it directly:  python scripts/test_figure_guards.py
"""
import os
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figlib_config as C  # noqa: E402
import figlib_guard as G  # noqa: E402


def _val_frame():
    """The shape `fig_validation_combined` is handed: one row per
    (dataset, model, rep, condition)."""
    rows = []
    for dataset in ('logd', 'caco2', 'herg'):
        for model in ('rf', 'svm', 'ngboost'):
            for rep in ('ecfp4', 'pdv', 'chemberta'):
                for condition in ('gaussian', 'grouped_wider', 'grouped_shifted'):
                    rows.append({'dataset': dataset, 'model': model, 'rep': rep,
                                 'condition': condition,
                                 'auc_norm': 0.5 + 0.1 * (rep == 'pdv'),
                                 'baseline_r2': 0.7})
    return pd.DataFrame(rows)


def the_line_the_paper_rests_on_now_raises():
    """generate_paper_figures_v2.py:2494 in its own shape.

        model_ds_c = val_auc_df.pivot_table(values='auc_norm', index='model',
                                            columns='dataset', aggfunc='mean')

    Grouping by model and dataset only, so aggfunc='mean' averages over
    representation and noise type together -- up to 36 values per bar.
    """
    df = _val_frame()
    silently_averaged = df.pivot_table(values='auc_norm', index='model',
                                       columns='dataset', aggfunc='mean')
    assert silently_averaged.notna().all().all(), (
        'the old code path draws happily -- that is the point')

    try:
        G.declare(df, 'fig_validation_combined panel A',
                  fixed={'rep': 'pdv'}, varies=('model', 'dataset'))
    except G.GuardError as exc:
        text = str(exc)
        assert 'rep' in text, text
        assert 'condition' in text, (
            'the noise type is averaged over too and the message must say so')
        print('    raises, and names both averaged factors')
        return
    raise AssertionError(
        'the frame still held 3 representations and 3 noise types and the '
        'guard let it through')


def removing_a_declaration_makes_the_run_fail():
    """Section 14.2 asks for this test by name."""
    df = _val_frame()
    one = df[(df['rep'] == 'pdv') & (df['condition'] == 'gaussian')]

    fragment = G.declare(one, 'a complete declaration',
                         fixed={'rep': 'pdv', 'condition': 'gaussian'},
                         varies=('model', 'dataset'))
    print(f'    complete declaration passes, title reads: {fragment!r}')

    try:
        G.declare(one, 'a declaration with condition removed',
                  fixed={'rep': 'pdv'}, varies=('model', 'dataset'))
    except G.GuardError as exc:
        assert 'condition' in str(exc), str(exc)
        print('    same data, one factor left off the declaration: refused')
        return
    raise AssertionError(
        'a factor was left out of the declaration and nothing complained -- '
        'the guard can be silently opted out of')


def a_factor_may_not_be_declared_as_something_it_averages_over():
    df = _val_frame()
    try:
        G.declare(df, 'a figure averaging over representation',
                  fixed={}, varies=('model', 'dataset', 'condition'),
                  aggregates=('rep',))
    except G.GuardError as exc:
        assert 'FACTORS' in str(exc), str(exc)
        print('    "aggregates=(rep,)" refused: a representation is not a repeat')
        return
    raise AssertionError('a factor was accepted as a repeat axis')


def replicates_and_molecules_are_the_two_that_are_allowed():
    df = _val_frame()
    one = df[(df['rep'] == 'pdv') & (df['condition'] == 'gaussian')].copy()
    one['iteration'] = 0
    G.declare(one, 'a figure averaging replicates',
              fixed={'rep': 'pdv', 'condition': 'gaussian'},
              varies=('model', 'dataset'), aggregates=('iteration',))
    G.declare(one, 'a figure averaging molecules',
              fixed={'rep': 'pdv', 'condition': 'gaussian'},
              varies=('model', 'dataset'), aggregates=('molecule',))
    print('    replicates and molecules pass, as section 14.2 allows')


def the_title_is_generated_from_the_data_not_typed():
    df = _val_frame()
    one = df[(df['rep'] == 'pdv') & (df['condition'] == 'grouped_shifted')
             & (df['dataset'] == 'caco2')]
    fragment = G.declare(one, 'F8', fixed={'rep': None, 'condition': None,
                                           'dataset': None},
                         varies=('model',))
    assert 'PDV' in fragment, fragment
    assert 'Grouped, shifted' in fragment, fragment
    assert 'Caco-2' in fragment, fragment
    print(f'    {fragment!r}')

    # And a title that claims the wrong value is refused, not quietly corrected.
    try:
        G.declare(one, 'F8 mislabelled', fixed={'rep': 'ecfp4'},
                  varies=('model', 'condition', 'dataset'))
    except G.GuardError as exc:
        assert 'ecfp4' in str(exc) and 'pdv' in str(exc), str(exc)
        print('    a title naming the wrong representation is refused')
        return
    raise AssertionError('a figure claimed ECFP4 over PDV data and passed')


def a_ratio_is_never_printed_without_its_components():
    good = pd.DataFrame({'model': ['rf'], 'auc_norm': [0.82],
                         'baseline_r2': [0.71]})
    G.with_components(good, 'T4')

    bad = pd.DataFrame({'model': ['rf'], 'auc_norm': [0.82]})
    try:
        G.with_components(bad, 'T4')
    except G.GuardError as exc:
        assert 'baseline_r2' in str(exc), str(exc)
        print('    auc_norm without its clean baseline: refused')
    else:
        raise AssertionError('a retention fraction printed with no denominator')

    worse = pd.DataFrame({'model': ['qrf'], 'auc_delta': [0.04]})
    try:
        G.with_components(worse, 'T6')
    except G.GuardError as exc:
        assert 'auc_error' in str(exc), str(exc)
        print('    auc_delta without auc_error and auc_ratio: refused')
        return
    raise AssertionError('a delta printed with neither of its two halves')


def one_observation_per_cell_is_refused_before_a_decomposition():
    # The assay case: one fit per cell, seed pinned, five folds that are a
    # partition rather than repeats (RERUN_PLAN.md 3.2b).
    saturated = _val_frame()
    saturated = saturated[saturated['condition'] == 'gaussian']
    try:
        G.assert_replicates(saturated, ['dataset', 'model', 'rep'],
                            where='the assay ANOVA')
    except G.GuardError as exc:
        assert 'saturated' in str(exc), str(exc)
        print('    one row per cell: refused, residual would be arithmetically 0')
    else:
        raise AssertionError('a saturated fit was allowed into a decomposition')

    replicated = pd.concat(
        [saturated.assign(iteration=i) for i in range(10)], ignore_index=True)
    info = G.assert_replicates(replicated, ['dataset', 'model', 'rep'],
                               where='the QM9 ANOVA')
    assert info['median_per_cell'] == 10.0, info
    assert info['n_below_min'] == 0, info
    print(f"    ten replicates per cell: passes, median {info['median_per_cell']:g}")


def an_empty_condition_fails_loudly():
    try:
        G.assert_expected_rows(pd.DataFrame(), 350, where='censoring on QM9')
    except G.GuardError as exc:
        assert 'censoring on QM9' in str(exc), str(exc)
        print('    a condition that produced nothing is named, not skipped')
        return
    raise AssertionError('an empty condition passed silently')


def a_filter_that_flips_the_sign_is_refused():
    df = pd.DataFrame({'model': ['rf'] * 4 + ['svm'] * 4,
                       'r2': [0.8, 0.8, 0.8, -9.0, 0.1, 0.1, 0.1, 0.1]})
    floor = G.Filter('catastrophic', 'a replicate that did not train',
                     lambda d: d['r2'] > C.CATASTROPHIC_R2_THRESHOLD)

    def headline(frame):
        means = frame.groupby('model')['r2'].mean()
        return means.get('rf', np.nan) - means.get('svm', np.nan)

    try:
        G.headline_with_and_without(df, [floor], headline,
                                    where='RF minus SVM')
    except G.GuardError as exc:
        assert 'DIRECTION' in str(exc), str(exc)
        print('    a filter that reverses the comparison is refused')
    else:
        raise AssertionError('a sign-flipping filter passed')

    benign = df.copy()
    benign.loc[3, 'r2'] = 0.4
    out = G.headline_with_and_without(benign, [floor], headline,
                                      where='RF minus SVM')
    assert out['log'][0]['rows_dropped'] == 0, out['log']
    print('    a filter that changes only the size passes, and is logged')


def censoring_cannot_rank_models():
    try:
        G.refuse_ranking_axis('censoring', where='F4 bottom panel')
    except G.GuardError as exc:
        text = str(exc)
        assert 'F4 bottom panel' in text, text
        assert 'No claim about WHICH model' in text, (
            'the message must quote the registry, not paraphrase it')
        print('    censoring refused as a ranking axis, quoting its own scope block')
    else:
        raise AssertionError('censoring was accepted as a ranking axis')

    G.refuse_ranking_axis('gaussian')
    kept = G.ranking_conditions(C.SETTLED_CONDITIONS)
    assert 'censoring' not in kept, kept
    assert 'gaussian' in kept and 'laplace' in kept, kept
    print(f'    conditions a ranking may use: {kept}')


def one_number_never_gets_two_names():
    assert G.metric_label('auc_norm') == 'AUC$_{norm}$'
    try:
        G.metric_label('nds')
    except KeyError as exc:
        assert 'failure mode 12' in str(exc), str(exc)
        print('    a retired metric has no name to print under')
        return
    raise AssertionError('an unregistered metric produced a caption')


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
    print('the figure script\'s six guards (RERUN_PLAN.md 0.6, 14.2)')
    results = [
        check('the line the paper rests on now raises',
              the_line_the_paper_rests_on_now_raises),
        check('removing a declaration makes the run fail',
              removing_a_declaration_makes_the_run_fail),
        check('a factor may not be declared as a repeat axis',
              a_factor_may_not_be_declared_as_something_it_averages_over),
        check('replicates and molecules are the two that are allowed',
              replicates_and_molecules_are_the_two_that_are_allowed),
        check('the title is generated from the data, not typed',
              the_title_is_generated_from_the_data_not_typed),
        check('a ratio is never printed without its components',
              a_ratio_is_never_printed_without_its_components),
        check('one observation per cell is refused before a decomposition',
              one_observation_per_cell_is_refused_before_a_decomposition),
        check('an empty condition fails loudly', an_empty_condition_fails_loudly),
        check('a filter that flips the sign is refused',
              a_filter_that_flips_the_sign_is_refused),
        check('censoring cannot rank models', censoring_cannot_rank_models),
        check('one number never gets two names', one_number_never_gets_two_names),
    ]
    if not all(results):
        print('\nFAIL: a figure can still average away a factor')
        return 1
    print('\nOK: every factor must be declared, and the data must agree')
    return 0


if __name__ == '__main__':
    sys.exit(main())
