#!/usr/bin/env python
"""The loader rules, against figlib_load.

The same eight rules `scripts/test_figure_conditions.py` pins on
`generate_paper_figures_v2.py`. That file and its test stay on disk until the
new script has run on real cluster data, so this is the parallel check rather
than a replacement: both loaders are held to the same rules until one is
retired.

Every one of these is a bug that actually happened (RERUN_PLAN.md 2.11, 2.13).

Run it directly:  python scripts/test_figure_loading.py
"""
import os
import sys
import tempfile
import traceback

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figlib_config as C  # noqa: E402
import figlib_fixtures as F  # noqa: E402
import figlib_load as L  # noqa: E402

ROW = dict(sigma=0.0, iteration=0, model='rf', rep='ecfp4', sample_size=100,
           mae=0.1, mse=0.02, rmse=0.14, r2=0.9, pearson_corr=0.95,
           params_source='default', loss_function='mse', spec_version='1',
           spec_hash='abc', gp_fit_method='', gp_collapsed='')


def write(tmp, name, noise_type=None, **overrides):
    row = dict(ROW, **overrides)
    if noise_type is not None:
        row['noise_type'] = noise_type
    pd.DataFrame([row]).to_csv(os.path.join(tmp, name), index=False)


def two_conditions_do_not_collapse_onto_one_row():
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, 'anova_gaussian_ecfp4_rf.csv', 'gaussian', r2=0.90)
        write(tmp, 'anova_grouped_shifted_ecfp4_rf.csv', 'grouped_shifted',
              r2=0.40)
        df = L.load_qm9(tmp)
        assert df is not None and len(df) == 2, (
            f'two conditions for one (model, rep, level, replicate) came back '
            f'as {0 if df is None else len(df)} row(s)')
        assert sorted(df['condition']) == ['gaussian', 'grouped_shifted']
        print(f"    two conditions, two rows: {sorted(df['condition'])}")


def a_settled_name_is_not_read_as_the_retired_one_it_starts_with():
    """`outlier` was the value-proportional strategy and is a PREFIX of the
    settled `outlier_p10`. Pooling them puts two mechanisms under one name."""
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, 'anova_outlier_p10_ecfp4_rf.csv', None)
        got = L.load_qm9(tmp)['condition'].unique().tolist()
        assert got == ['outlier_p10'], got
        print(f'    anova_outlier_p10_... -> {got[0]}')


def a_file_naming_no_condition_is_never_left_blank():
    """A blank is treated as equal to every other blank by drop_duplicates,
    which is how six conditions became one row."""
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, 'anova_somethingelse_ecfp4_rf.csv', None)
        got = L.load_qm9(tmp)['condition'].tolist()
        assert all(str(v).startswith('unknown_') for v in got), got
        print(f'    unnamed file -> {got[0]}')


def censoring_levels_pair_into_one_condition():
    """QM9 writes the clipped percentage INSIDE the name; the assay runner
    writes plain `censoring` with the level in its own column. Left alone, each
    QM9 censoring level is a condition holding a single level, so every
    robustness function drops it for having no curve."""
    with tempfile.TemporaryDirectory() as tmp:
        for pct, sigma in ((0, 0.0), (25, 0.25), (50, 0.50)):
            write(tmp, f'anova_censoring_{pct}_ecfp4_rf.csv',
                  f'censoring_{pct}', sigma=sigma)
        df = L.load_qm9(tmp)
        assert sorted(df['condition'].unique()) == ['censoring'], \
            sorted(df['condition'].unique())
        assert len(df) == 3, len(df)
        print(f"    three censoring levels -> one condition, "
              f"{len(df)} levels on its own axis")


def sibling_files_are_not_read_as_results():
    """`anova_*.csv` matches three siblings the same run writes, and two of them
    carry the results columns -- so only the NAME rule rejects those."""
    with tempfile.TemporaryDirectory() as tmp:
        write(tmp, 'anova_gaussian_ecfp4_rf.csv', 'gaussian')
        pd.DataFrame([dict(ROW, model='manifest_row', file_no=1,
                           noise_type='gaussian')]).to_csv(
            os.path.join(tmp, 'anova_gaussian_ecfp4_rf_noise_manifest.csv'),
            index=False)
        pd.DataFrame([{'epoch': 1, 'train_loss': 0.4}]).to_csv(
            os.path.join(tmp, 'anova_gaussian_ecfp4_dnn_per_epoch.csv'),
            index=False)
        pd.DataFrame([{'sigma': 0.0, 'something': 1}]).to_csv(
            os.path.join(tmp, 'anova_gaussian_ecfp4_odd.csv'), index=False)
        df = L.load_qm9(tmp)
        assert len(df) == 1, f'{len(df)} rows from one results file + 3 siblings'
        assert 'manifest_row' not in set(df['model'])
        print('    manifest, per-epoch and a column-less file all skipped')


def the_replicate_is_in_the_deduplication_key():
    """Without it, appended runs of one cell overwrite each other."""
    with tempfile.TemporaryDirectory() as tmp:
        frame = pd.DataFrame([dict(ROW, iteration=i, r2=0.9 - 0.01 * i,
                                   noise_type='gaussian') for i in range(10)])
        frame.to_csv(os.path.join(tmp, 'anova_gaussian_ecfp4_rf.csv'),
                     index=False)
        df = L.load_qm9(tmp)
        assert len(df) == 10, f'{len(df)} of 10 replicates survived'
        print('    ten replicates of one cell survive as ten rows')


def the_fold_is_in_the_assay_deduplication_key():
    """The old loader deduplicated on dataset, model, rep, condition and level
    with NO fold, kept the first row, and discarded four fifths of the data."""
    with tempfile.TemporaryDirectory() as tmp:
        F.write_assay(tmp, models=['rf'], reps=['ecfp4'],
                      conditions=['gaussian'], datasets=('logd',), folds=5)
        df = L.load_assay_accuracy([tmp])
        per_level = df.groupby('sigma')['replicate'].nunique()
        assert (per_level == 5).all(), per_level.to_dict()
        assert set(df['replicate_kind']) == {'fold'}, set(df['replicate_kind'])
        print(f'    five folds survive at every level, labelled '
              f'{df["replicate_kind"].iloc[0]!r} and not "replicate"')


def the_uncertainty_column_is_the_one_the_spec_settles_on():
    both = pd.DataFrame({'y_pred_std_uncalibrated': [1.0, 2.0],
                         'y_pred_std_calibrated': [10.0, 20.0]})
    picked = L.uncertainty_column(both)
    expected = ('y_pred_std_uncalibrated' if C.UNCERTAINTY_PRIMARY == 'raw'
                else 'y_pred_std_calibrated')
    assert picked == expected, f'{picked!r} not {expected!r}'
    # The assay side calls it `uncertainty` and writes no calibrated column.
    assert L.uncertainty_column(pd.DataFrame({'uncertainty': [1.0]})) \
        == 'uncertainty'
    print(f'    spec settles on {C.UNCERTAINTY_PRIMARY!r}; with both columns '
          f'present the loader reads {picked!r}')


def streaming_gives_the_same_numbers_as_loading_everything():
    """Reading one file at a time must change no number.

    Loading every per-molecule row first is what a login node kills: each row is
    one molecule at one level in one fold, so the real grid is hundreds of
    millions and pandas dies allocating a few megabytes long after the cap was
    reached. Splitting by FILE is safe because a statistic is computed inside
    one cell -- dataset, model, rep, condition, sigma, fold, split -- and one
    file holds every level and fold for one (condition, representation, model),
    so no cell spans two files. This proves that rather than asserting it.
    """
    import figlib_uncertainty as U
    with tempfile.TemporaryDirectory() as tmp:
        F.write_per_molecule(tmp, models=['qrf', 'gauche_rbf'], reps=['ecfp4'],
                             conditions=['gaussian'], n_molecules=150, folds=2)
        files = U.discover([tmp])
        assert len(files) == 2, files

        whole = U.load(None, dataset_name='qm9', paths=files)
        batch_q6 = U.q6(whole).set_index(['model', 'rep', 'condition', 'sigma',
                                          'fold'])['rho_unc_vs_clean_error']
        batch_slopes = U.component_slopes(U.q5(whole))

        streamed = U.statistics([tmp], permutations=0, dataset_name='qm9',
                                progress_every=0)
        stream_q6 = streamed['q6'].set_index(
            ['model', 'rep', 'condition', 'sigma',
             'fold'])['rho_unc_vs_clean_error']

        shared = batch_q6.index.intersection(stream_q6.index)
        assert len(shared) == len(batch_q6), (
            f'{len(shared)} of {len(batch_q6)} cells survived streaming')
        worst = float((batch_q6.loc[shared] - stream_q6.loc[shared]).abs().max())
        assert worst < 1e-12, f'streaming moved a Q6 value by {worst}'

        key = ['model', 'rep', 'condition']
        b = batch_slopes.set_index(key)['verdict'].sort_index()
        s = streamed['slopes'].set_index(key)['verdict'].sort_index()
        assert b.equals(s), f'verdicts differ:\n{b}\n{s}'
        print(f'    {len(shared)} cells identical to {worst:.1e}, and every '
              f'decomposition verdict unchanged')


def a_broken_environment_explains_itself():
    """The one environment failure this hits, answered instead of raised.

    scipy's compiled parts are built against the conda environment's libstdc++,
    which is newer than the one in /lib64 on ARC. Without the environment ahead
    of the system, `from scipy import stats` dies forty lines deep inside
    scipy.optimize with GLIBCXX_3.4.30 not found -- which says nothing about
    what to do. setup.sh sets the path; a bare `conda activate` does not.
    """
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'scipy.py'), 'w') as fh:
            fh.write("raise ImportError(\"/lib64/libstdc++.so.6: version "
                     "`GLIBCXX_3.4.30' not found\")\n")
        env = dict(os.environ, PYTHONPATH=tmp, CONDA_PREFIX='/envs/env_test')
        got = subprocess.run(
            [sys.executable, os.path.join(here, 'run_paper_analysis.py'),
             '--qm9-dir', tmp],
            capture_output=True, text=True, env=env)
        assert got.returncode == 3, (
            f'exit {got.returncode}, expected 3 for an environment fault')
        assert 'This is the environment, not the analysis' in got.stderr
        assert '/envs/env_test/lib' in got.stderr, (
            'the fix must name the actual environment, not a placeholder')
        assert 'setup.sh' in got.stderr
        assert 'Traceback' not in got.stderr, (
            'a forty-line traceback is what this replaces')
        print('    a GLIBCXX failure prints the fix and exits 3, no traceback')

    # And an unrelated ImportError must NOT be swallowed by that handler.
    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'scipy.py'), 'w') as fh:
            fh.write("raise ImportError('no module named nonsense')\n")
        env = dict(os.environ, PYTHONPATH=tmp)
        got = subprocess.run(
            [sys.executable, os.path.join(here, 'run_paper_analysis.py'),
             '--qm9-dir', tmp], capture_output=True, text=True, env=env)
        assert got.returncode != 3, 'an unrelated import error was mislabelled'
        assert 'Traceback' in got.stderr, 'it should still raise normally'
        print('    an unrelated import error still raises normally')


def the_merge_step_is_not_a_prerequisite():
    """Each uncertainty task writes its own tables into its own directory.

    Requiring a collation pass before any number can be looked at would mean
    waiting on housekeeping to see results already on disk. `_merged/` is read
    when it is there and the task directories are read when it is not.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, 'uncertainty_rerun')
        for task in ('qrf__logd__ecfp4__gaussian', 'gp__logd__pdv__gaussian'):
            d = os.path.join(root, task)
            os.makedirs(d)
            pd.DataFrame([{'dataset': 'logd', 'model': 'QRF', 'rep': 'ECFP4',
                           'sigma': 0.0, 'fold': 0, 'r2': 0.5}]).to_csv(
                os.path.join(d, 'all_results.csv'), index=False)
        assert not os.path.isdir(os.path.join(root, '_merged'))
        got = L.load_merged_uncertainty([root])
        assert got['all_results'] is not None, (
            'nothing was read without a _merged/ directory')
        assert len(got['all_results']) == 2, len(got['all_results'])
        print('    two un-merged task directories read directly')


def partial_data_loads_and_says_what_is_short():
    """The screen lands in pieces. A quarter of the grid must produce a report,
    not an error and not a silent headline over whichever cells finished."""
    with tempfile.TemporaryDirectory() as tmp:
        F.write_qm9(tmp, models=['rf', 'svm'], reps=['ecfp4'],
                    conditions=['gaussian'], replicates=3)
        df = L.load_qm9(tmp)
        assert df is not None and len(df), 'partial data would not load'
        cover = L.coverage(df, 'partial')
        assert (cover['status'] == 'THIN_REPLICATES').all(), \
            cover['status'].unique()
        assert (cover['replicates_max'] == 3).all(), \
            cover['replicates_max'].unique()
        print(f'    {len(cover)} cell(s) at 3 replicates: loaded, and every '
              f'one flagged THIN rather than passed off as complete')


def the_reference_condition_is_never_the_whole_frame():
    """Every filter used to read `frame[frame.strategy == 'legacy'] if
    'strategy' in frame else frame`, so a frame with no condition column
    silently became every condition pooled under one name."""
    try:
        L.baseline_rows(pd.DataFrame({'model': ['rf'], 'auc_norm': [0.5]}),
                        'a test')
    except RuntimeError as exc:
        assert 'cannot be selected' in str(exc), str(exc)
        print('    a frame with no condition column is refused, not pooled')
        return
    raise AssertionError('a frame with no condition column was accepted')


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
    print('figlib_load: conditions, siblings, replicates (RERUN_PLAN.md 2.11, 2.13)')
    results = [
        check('two conditions do not collapse onto one row',
              two_conditions_do_not_collapse_onto_one_row),
        check('a settled name is not read as the retired name it starts with',
              a_settled_name_is_not_read_as_the_retired_one_it_starts_with),
        check('a file naming no condition is never left blank',
              a_file_naming_no_condition_is_never_left_blank),
        check('censoring levels pair into one condition',
              censoring_levels_pair_into_one_condition),
        check('sibling files are not read as results',
              sibling_files_are_not_read_as_results),
        check('the replicate is in the deduplication key',
              the_replicate_is_in_the_deduplication_key),
        check('the fold is in the assay deduplication key',
              the_fold_is_in_the_assay_deduplication_key),
        check('the uncertainty column is the one the spec settles on',
              the_uncertainty_column_is_the_one_the_spec_settles_on),
        check('the reference condition is never the whole frame',
              the_reference_condition_is_never_the_whole_frame),
        check('streaming gives the same numbers as loading everything',
              streaming_gives_the_same_numbers_as_loading_everything),
        check('a broken environment explains itself',
              a_broken_environment_explains_itself),
        check('the merge step is not a prerequisite',
              the_merge_step_is_not_a_prerequisite),
        check('partial data loads and says what is short',
              partial_data_loads_and_says_what_is_short),
    ]
    if not all(results):
        print('\nFAIL: the loader can still pool or discard rows')
        return 1
    print('\nOK: every row keeps its condition, its replicate and its scale')
    return 0


if __name__ == '__main__':
    sys.exit(main())
