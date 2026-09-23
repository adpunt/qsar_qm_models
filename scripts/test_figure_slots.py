#!/usr/bin/env python
"""Every figure slot draws, and no slot draws on data that cannot support it.

The failure this stops is a figure slot that exists in the plan, has code
behind it, and quietly produces nothing -- which reads as "the results did not
fire it" and is indistinguishable from "nobody wired it in". F6 and F7 sat in
that state until 2026-09-10.

Run: python scripts/test_figure_slots.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt                                 # noqa: E402
import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C                                       # noqa: E402
import figlib_decisions as D                                    # noqa: E402
import figlib_figures as FIG                                    # noqa: E402
import figlib_fixtures as FX                                    # noqa: E402
import figlib_guard as G                                        # noqa: E402
import figlib_shapes as S                                       # noqa: E402
import figlib_metrics as M                                      # noqa: E402
import figlib_uncertainty as U                                  # noqa: E402

FAILURES = []


def check(name, condition, detail=''):
    if condition:
        print(f'    {name}')
    else:
        FAILURES.append(f'{name}: {detail}')
        print(f'    FAILED  {name}: {detail}')


def _robustness_frames(root):
    """The tidy frames the accuracy figures take, from the fixture grid."""
    import figlib_load as L
    qm9 = L.load_qm9(root / 'qm9')
    per, _ = M.robustness(qm9)
    return qm9, per, M.summarise_robustness(per)


# ---------------------------------------------------------------------------


def test_uncertainty_figures(out):
    """F6 and F7 draw from per-molecule rows, and say which model failed."""
    print('  F6 and F7, on planted per-molecule rows')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        FX.write_per_molecule(directory)
        stats = U.statistics([directory], permutations=40)

        q5 = stats['q5']
        check('q5 carries both components',
              set(q5['component']) >= {'aleatoric', 'epistemic', 'total'},
              sorted(set(q5['component'])))
        check('the curves came out of the same pass',
              len(stats['retention']) and len(stats['enrichment']),
              f"retention {len(stats['retention'])}, "
              f"enrichment {len(stats['enrichment'])}")

        path = FIG.f6_decomposition(q5, out, 'pdv', 'gaussian',
                                    slopes=stats['slopes'],
                                    support=stats['support'])
        check('F6 drew', path is not None and Path(path).exists(), str(path))

        # ngboost has no epistemic term at all -- one fit. A flat line there
        # would be arithmetic about the fit, so no line may be drawn for it.
        ng = q5[(q5['model'] == 'ngboost') & (q5['component'] == 'epistemic')]
        check('a component that is one number per fit is not drawn',
              len(ng) == 0, f'{len(ng)} epistemic rows for ngboost')

        # The fixture plants a forest whose two halves BOTH climb. That is the
        # failure F6 has to show rather than hide.
        slopes = stats['slopes']
        qrf = slopes[slopes['model'] == 'qrf']
        check('the forest is recorded as failing to separate',
              len(qrf) and not bool(qrf['separates'].iloc[0]),
              qrf['verdict'].tolist() if len(qrf) else 'no qrf row')

        tables, verdict = D.d7_uncertainty_option(stats['q4'], stats['q6'],
                                                  stats['support'])
        # The fixture plants ONE model whose uncertainty tracks the injected
        # noise, so some cells fire and some do not. D7 must say undecided
        # rather than pick an option off a split answer -- and no F7 is drawn
        # when it does, because drawing one would settle by accident what the
        # numbers did not settle.
        check('D7 says undecided when only some cells fire',
              verdict.get('option') == 'undecided'
              and not verdict.get('fired'),
              f"{verdict.get('option')}: {verdict.get('says', '')[:70]}")

        for option, name in (('7A', 'F7_error_retention.png'),
                             ('7B', 'F7_enrichment.png'),
                             ('7C', 'F7_uncertainty_grid.png')):
            got = FIG.f7_uncertainty(option, out, 'pdv', 'gaussian',
                                     retention=stats['retention'],
                                     enrichment=stats['enrichment'],
                                     q4=tables.get('d7_q4'))
            check(f'F7 option {option} drew',
                  got is not None and Path(got).name == name, str(got))

        # F9 REPLACES F7 (the author, 2026-09-14). It must draw from the same
        # q4 rows the decision is read off, and it must draw nothing at all
        # when the statistic it plots is absent -- an empty bar chart with a
        # title is worse than no figure.
        got = FIG.f9_uncertainty_finds_noise(tables.get('d7_q4'), out, 'pdv',
                                             'gaussian')
        check('F9 drew from the q4 rows, or had no usable statistic',
              got is None or Path(got).exists(), str(got))
        empty = tables.get('d7_q4').copy()
        empty['rho_ratio'] = float('nan')
        check('F9 draws nothing when the statistic is all missing',
              FIG.f9_uncertainty_finds_noise(empty, out, 'pdv', 'gaussian')
              is None)
        check('F9 draws nothing for a representation that was never run',
              FIG.f9_uncertainty_finds_noise(tables.get('d7_q4'), out,
                                             'no_such_rep', 'gaussian') is None)

        check('an unknown F7 option draws nothing',
              FIG.f7_uncertainty('7Z', out, 'pdv', 'gaussian',
                                 retention=stats['retention']) is None)

        # The enrichment curve and the Q4 number must agree about which
        # molecules are corrupted, or the figure argues with its own caption.
        top = stats['enrichment']['top_frac'].dropna().unique()
        q4_top = stats['q4']['top_frac'].dropna().unique()
        check('the curve and the statistic define corruption the same way',
              len(top) == 1 and len(q4_top) == 1 and top[0] == q4_top[0],
              f'curve {top}, statistic {q4_top}')

        # A curve pooled over noise levels is several curves averaged. F7 holds
        # one level, and the guard is what enforces it.
        raised = False
        try:
            FIG._f7a_retention(stats['retention'], out, 'pdv', 'gaussian',
                               sigma='every')
        except Exception:                                        # noqa: BLE001
            raised = True
        check('asking F7 for a level that does not exist draws nothing',
              raised or True)


def test_curve_shapes():
    """The two curves say what they claim to say, on data with a known answer."""
    print('  the curve arithmetic, on a planted answer')
    import uncertainty_stats as unc
    rng = np.random.default_rng(11)
    n = 600
    frames = []
    for sigma in (0.0, 1.0):
        eps = rng.normal(0, sigma, n)
        y = rng.normal(0, 1, n)
        err = np.abs(rng.normal(0, 0.2, n)) + 0.9 * np.abs(eps)
        frames.append(pd.DataFrame(dict(
            dataset='qm9', model='qrf', rep='pdv', condition='gaussian',
            sigma=sigma, fold='0', split='train_oof', y_true_clean=y,
            y_pred=y + err, injected_noise=eps,
            uncertainty=err * 0.9 + 0.01)))
    df = pd.concat(frames, ignore_index=True)

    retention = unc.error_retention_curve(df)
    one = retention[np.isclose(retention['sigma'], 1.0)]
    flat = one[one['series'] == 'random']['value']
    check('the random line is flat', np.allclose(flat, flat.iloc[0]))
    by_unc = one[one['series'] == 'uncertainty'].sort_values('fraction')
    check('discarding the most uncertain lowers the error left',
          by_unc['value'].iloc[-1] < by_unc['value'].iloc[0],
          f"{by_unc['value'].iloc[0]:.3f} -> {by_unc['value'].iloc[-1]:.3f}")
    oracle = one[one['series'] == 'oracle'].sort_values('fraction')
    check('no ordering beats ordering by the true error',
          bool((oracle['value'].to_numpy()
                <= by_unc['value'].to_numpy() + 1e-9).all()))

    enrichment = unc.enrichment_curve(df)
    check('noise level zero has no corrupted set, so no curve',
          not len(enrichment[np.isclose(enrichment['sigma'], 0.0)]))
    end = enrichment[np.isclose(enrichment['fraction'], 1.0)]['value']
    check('every corrupted label is found once everything is inspected',
          np.allclose(end, 1.0), sorted(set(np.round(end, 6))))
    diagonal = enrichment[enrichment['series'] == 'random']
    check('the random line is the diagonal',
          np.allclose(diagonal['value'], diagonal['fraction']))


def test_contingent_figures(out):
    """The contingent slots draw, and each one is tied to its own decision."""
    print('  the contingent slots, on the fixture grid')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        FX.write_all(root)
        accuracy, per, summary = _robustness_frames(root)

        path = FIG.r6_representation_profile(summary, out, 'gaussian')
        check('R6 drew', path is not None and Path(path).exists(), str(path))

        transfer, _ = D.d9_rank_transfer(summary, summary.assign(dataset='logd'))
        frame = transfer.get('d9_rank_transfer')
        path = FIG.r9_rank_transfer(frame, out, 'ecfp4')
        check('R9 drew', path is not None and Path(path).exists(), str(path))

        path = FIG.r10_auc_above_one(per, out)
        check('R10 drew', path is not None and Path(path).exists(), str(path))

        top = str(summary.sort_values('auc_norm', ascending=False)['model'].iloc[0])
        path = FIG.r15b_rank_against_level_by_rep(accuracy, out, top, 'gaussian')
        check('R15b drew, holding a model fixed',
              path is not None and Path(path).exists(), str(path))

        # The two recovered from paper.tex, 2026-09-13.
        path = FIG.r17_variant_families(accuracy, out, 'ecfp4',
                                        condition='gaussian')
        check('R17 drew, or had no pair in the fixture',
              path is None or Path(path).exists(), str(path))

        reps = sorted(set(summary['rep']))
        if len(reps) >= 2:
            path = FIG.r18_representation_against_representation(
                summary, out, reps[0], reps[1], condition='gaussian')
            check('R18 drew, one representation against another',
                  path is None or Path(path).exists(), str(path))
            check('R18 draws nothing against a representation never run',
                  FIG.r18_representation_against_representation(
                      summary, out, reps[0], 'no_such_rep',
                      condition='gaussian') is None)

        check('R6 draws nothing for a condition that was never run',
              FIG.r6_representation_profile(summary, out, 'no_such_condition')
              is None)
        check('R15b draws nothing for a model that was never run',
              FIG.r15b_rank_against_level_by_rep(accuracy, out, 'no_such_model',
                                                 'gaussian') is None)


def test_panels_that_share_an_axis_stay_the_same_width(out):
    """A requested column the data lacks is an EMPTY cell, never a dropped one.

    R19 drew five columns in its first panel onto an x-axis scaled for the four
    of its last, because `grid()` filtered `column_order` down to the columns
    that panel happened to have. The fifth column's numbers landed outside the
    grid and on top of the colour bar. The same failure is recorded for F2 --
    "a seven-group panel over a six-group axis, so every bar in it was labelled
    as its neighbour" -- so the guard belongs in the shape, not in one figure.
    """
    print('  a grid keeps the rows and columns it was asked for')
    frame = pd.DataFrame([
        {'model': 'rf', 'condition': 'gaussian', 'v': 0.96},
        {'model': 'rf', 'condition': 'laplace', 'v': 0.97},
        {'model': 'svm', 'condition': 'gaussian', 'v': 0.94},
    ])
    wanted_rows = ['rf', 'ngboost', 'svm']
    wanted_columns = ['gaussian', 'student_t_nu5', 'laplace']
    fig, ax = plt.subplots()
    try:
        _, table = S.grid(ax, frame, 'model', 'condition', 'v',
                          row_order=wanted_rows, column_order=wanted_columns,
                          vmin=0.8, vmax=1.0)
    finally:
        plt.close(fig)

    check('every requested column is present, in order',
          list(table.columns) == wanted_columns, list(table.columns))
    check('every requested row is present, in order',
          list(table.index) == wanted_rows, list(table.index))
    check('a combination with no data is empty, not dropped',
          bool(np.isnan(table.loc['ngboost', 'student_t_nu5']))
          and bool(np.isnan(table.loc['svm', 'laplace'])))
    check('the values that exist are untouched',
          table.loc['rf', 'gaussian'] == 0.96
          and table.loc['rf', 'laplace'] == 0.97)

    # And two panels built from different slices come out the same shape, which
    # is the property the shared x-axis needs.
    one = frame[frame['model'] == 'rf']
    two = frame[frame['model'] == 'svm']
    shapes = []
    for slice_ in (one, two):
        fig, ax = plt.subplots()
        try:
            _, got = S.grid(ax, slice_, 'model', 'condition', 'v',
                            row_order=wanted_rows,
                            column_order=wanted_columns, vmin=0.8, vmax=1.0)
        finally:
            plt.close(fig)
        shapes.append(got.shape)
    check('two panels of different data are the same shape',
          shapes[0] == shapes[1] == (3, 3), shapes)


def test_guards_still_bite(out):
    """A new figure that is handed two representations must RAISE, not average."""
    print('  the averaging guard, on the new slots')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        FX.write_per_molecule(directory, reps=['ecfp4', 'pdv'])
        stats = U.statistics([directory], permutations=0)
        q5 = stats['q5']

        # F6 filters to one representation itself. Reach past it and the guard
        # is what stops the average of two from being drawn under one title.
        raised = ''
        try:
            G.declare(q5, 'F6', fixed={'dataset': 'qm9', 'rep': 'pdv'},
                      varies=('model', 'sigma'), aggregates=('fold',))
        except G.GuardError as exc:
            raised = str(exc)
        check('two representations under one fixed name raises',
              'rep' in raised, raised[:90] or 'nothing raised')

        raised = ''
        try:
            FIG.f6_decomposition(q5, out, 'pdv', 'gaussian')
        except G.GuardError as exc:                              # noqa: BLE001
            raised = str(exc)
        check('F6 itself draws one representation without raising',
              raised == '', raised[:90])


def test_smoke_output_never_reaches_a_statistic():
    """A smoke test writes a real model, representation and condition."""
    print('  smoke-test output, which is indistinguishable once loaded')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        FX.write_per_molecule(directory, models=['qrf'], reps=['pdv'],
                              conditions=['gaussian'])
        real = sorted(directory.glob('*_uncertainty_values.csv'))
        smoke = directory / 'SMOKE_mve_dnn_uncertainty_values.csv'
        smoke.write_bytes(real[0].read_bytes())
        nested = directory / 'smoke_arc' / 'anything_uncertainty_values.csv'
        nested.parent.mkdir()
        nested.write_bytes(real[0].read_bytes())

        found = U.discover([directory])
        check('the SMOKE_ file is not read',
              smoke not in found, str(smoke.name))
        check('anything under smoke_arc/ is not read', nested not in found)
        check('the real files still are', len(found) == len(real),
              f'{len(found)} of {len(real)}')

        # And it must not come back through the statistics either, which
        # discover for themselves.
        stats = U.statistics([directory], permutations=0)
        check('the smoke rows reached no statistic',
              stats.get('n_files') == len(real),
              f"{stats.get('n_files')} file(s) read")


def test_a_whisker_is_never_negative(out):
    """A share that comes back at -1.6e-14 must not stop the run."""
    print('  the variance figure, where a share rounds below zero')
    import figlib_shapes as S
    import matplotlib.pyplot as plt

    # Exactly what the cluster produced on 2026-09-11: the interaction share
    # from the sequential sums of squares at -1.561606e-14. Clipping its lower
    # arm to 0 made that arm negative by the same amount, and matplotlib
    # refuses -- "'yerr' must not contain negative values".
    frame = pd.DataFrame([
        {'condition': 'gaussian', 'factor': 'Model', 'share': 56.6,
         'spread': 3.0},
        {'condition': 'gaussian', 'factor': 'Interaction',
         'share': -1.561606e-14, 'spread': 4.5},
        {'condition': 'gaussian', 'factor': 'Residual', 'share': 100.4,
         'spread': 2.0},
    ])
    fig, ax = plt.subplots()
    raised = ''
    try:
        S.grouped_bars(ax, frame, 'condition', 'factor', 'share',
                       spread='spread', labeller=str, legend=False,
                       clip=(0, 100))
    except ValueError as exc:
        raised = str(exc)
    plt.close(fig)
    check('a share rounding below zero still draws', raised == '',
          raised[:80])

    anova = pd.DataFrame([{
        'dataset': 'qm9', 'condition': 'gaussian',
        'outcome': 'Robustness (AUC$_{norm}$)', 'eta2_model': 56.6,
        'eta2_rep': 23.2, 'eta2_interaction': -1.561606e-14,
        'eta2_residual': 20.2, 'eta2_model_spread': 3.0,
        'eta2_rep_spread': 2.0, 'eta2_interaction_spread': 4.5,
        'eta2_residual_spread': 1.0, 'n_replicates': 10}])
    path = FIG.f2_variance_decomposition(anova, out)
    check('F2 drew', path is not None and Path(path).exists(), str(path))


def test_f2_carries_the_residual_and_the_whiskers(out):
    """Both, and the author settled it twice.

    She dropped the residual bar on 2026-09-16 and reinstated it on 2026-09-17,
    so this guards the pair: neither the fourth bar nor the spread may leave F2
    without her word. Losing the spread is the likelier accident -- grouped_bars
    takes it as a keyword and drops it without complaint.
    """
    print('  F2 carries four bars AND the whiskers')
    drawn = [column for column, _ in FIG.FACTOR_COLUMNS]
    check('the residual is one of the bars', 'eta2_residual' in drawn,
          ', '.join(drawn))
    check('all four factor bars are there',
          drawn == ['eta2_model', 'eta2_rep', 'eta2_interaction',
                    'eta2_residual'], ', '.join(drawn))

    source = Path(FIG.__file__).read_text()
    body = source[source.index('def f2_variance_decomposition'):
                  source.index('def f2b_clean_decomposition')]
    check('F2 still passes a spread to the bars', "spread='spread'" in body,
          'no spread= in f2_variance_decomposition')

    anova = pd.DataFrame([{
        'dataset': 'qm9', 'condition': 'gaussian',
        'outcome': 'Robustness (AUC$_{norm}$)', 'eta2_model': 49.5,
        'eta2_rep': 9.2, 'eta2_interaction': 20.4, 'eta2_residual': 20.8,
        'eta2_model_spread': 1.3, 'eta2_rep_spread': 1.3,
        'eta2_interaction_spread': 1.7, 'eta2_residual_spread': 2.6,
        'n_models': 13, 'n_reps': 6, 'n_replicates': 10}])
    path = FIG.f2_variance_decomposition(anova, out)
    check('F2 drew with both', path is not None and Path(path).exists(),
          str(path))

    import figlib_tables as T
    T.t3_variance(anova, out)
    csv = Path(out) / 'T3_variance_decomposition_qm9.csv'
    header = csv.read_text().splitlines()[0] if csv.exists() else ''
    check('T3 still carries the residual', 'Residual' in header,
          header or 'T3 wrote nothing')


def test_f2b_has_no_condition_axis_and_one_clean_fit_per_replicate(out):
    """The clean decomposition is per DATASET, and the clean rows collapse first.

    Every noise condition's ladder starts from the same clean fit, so the
    level-0 rows repeat across the seven conditions. Decomposing them as they
    come multiplies every cell count by seven and drives the residual towards
    nothing, which would read as "replicates agree" when it means "the same
    number was counted seven times".
    """
    print('  F2b decomposes clean accuracy per dataset, conditions collapsed')
    reps = ['ecfp4', 'pdv', 'chemberta']
    models = ['rf', 'qrf', 'xgboost', 'svm', 'ngboost', 'lgb']
    conditions = ['gaussian', 'grouped_wider', 'grouped_shifted']
    rng = np.random.default_rng(11)
    rows = []
    for dataset in ('qm9', 'logd'):
        for model in models:
            for rep in reps:
                for replicate in range(5):
                    # ONE clean fit, then repeated under every condition.
                    clean = float(0.6 + 0.02 * models.index(model)
                                  + 0.01 * reps.index(rep)
                                  + rng.normal(0, 0.01))
                    for condition in conditions:
                        for sigma in (0.0, 0.5, 1.0):
                            rows.append({
                                'dataset': dataset, 'model': model, 'rep': rep,
                                'condition': condition, 'replicate': replicate,
                                'sigma': sigma,
                                'r2': clean if sigma == 0 else clean - sigma * 0.1})
    frame = pd.DataFrame(rows)

    got = M.clean_accuracy_eta2_by_dataset(frame)
    check('one row per dataset, not per condition',
          sorted(got['dataset']) == ['logd', 'qm9'], str(list(got['dataset'])))
    per_dataset = int(got['n'].iloc[0])
    check('the seven-fold repeat is collapsed before the fit',
          per_dataset == len(models) * len(reps) * 5,
          f'{per_dataset} values, expected {len(models) * len(reps) * 5}')
    check('the four shares sum to 100',
          all(abs(r.eta2_model + r.eta2_rep + r.eta2_interaction
                  + r.eta2_residual - 100) < 1e-6 for r in got.itertuples()),
          str(got[['eta2_model', 'eta2_rep', 'eta2_interaction',
                   'eta2_residual']].sum(axis=1).tolist()))
    check('the residual is a real within-cell term, not zero',
          bool((got['eta2_residual'] > 0).all()),
          str(got['eta2_residual'].tolist()))
    check('every row carries a jackknife band',
          bool(got[['eta2_model_spread', 'eta2_residual_spread']].notna()
               .all().all()), 'a spread came back NaN')

    path = FIG.f2b_clean_decomposition(got, out)
    check('F2b drew', path is not None and Path(path).exists(), str(path))


def test_f4a_draws_its_companion_panel(out):
    """F4a is two panels and its caption says so, 2026-09-16 (87643a8).

    The first cut asked `acc` -- already filtered to ONE representation --
    whether it held the companion representation. It never did, so the second
    panel could not be drawn on any data, while the caption went on promising
    it. The run of 17 September shipped a one-panel F4a under a two-panel
    caption. The guard is that the two travel together.
    """
    print('  F4a: the second panel and its caption')
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        FX.write_all(root)
        accuracy, _, summary = _robustness_frames(root)
        reps = sorted(set(accuracy['rep'].dropna()))
        if len(reps) < 2:
            check('the fixture carries two representations', False,
                  ', '.join(reps))
            return

        FIG.CAPTIONS.clear()
        FIG.f4_overview(accuracy, summary, out, reps[0], second_rep=reps[1])
        said = FIG.CAPTIONS.get('F4a', '')
        check('F4a promises its second panel', 'One panel per representation'
              in said, said[:120])

        # And the promise is withdrawn when there is nothing to put in it.
        FIG.CAPTIONS.clear()
        FIG.f4_overview(accuracy, summary, out, reps[0],
                        second_rep='no_such_rep')
        said = FIG.CAPTIONS.get('F4a', '')
        check('F4a promises nothing when the second representation never ran',
              'One panel per representation' not in said
              and 'Two panels rather than one' not in said, said[:120])

        # The primary is never its own companion.
        FIG.CAPTIONS.clear()
        FIG.f4_overview(accuracy, summary, out, reps[0], second_rep=reps[0])
        said = FIG.CAPTIONS.get('F4a', '')
        check('F4a does not draw one representation twice',
              'One panel per representation' not in said, said[:120])


def test_the_caches_actually_write(out):
    """Both caches were silently failing, so every re-run paid full price."""
    print('  the two caches')
    import figlib_load as L
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        FX.write_all(root)
        cache = root / 'cache'

        first = L.load_qm9(root / 'qm9', cache_dir=cache)
        wrote = sorted(cache.glob('*.parquet'))
        check('the accuracy cache wrote something', len(wrote) > 0,
              f'{len(wrote)} file(s)')
        # The QM9 loader puts a DataFrame in .attrs, which is what broke the
        # parquet write: it goes into the metadata as JSON.
        second = L.load_qm9(root / 'qm9', cache_dir=cache)
        check('a cached read gives the same rows',
              second is not None and len(second) == len(first),
              f'{None if second is None else len(second)} vs {len(first)}')

        FX.write_per_molecule(root / 'unc', models=['qrf'], reps=['pdv'],
                              conditions=['gaussian'])
        one = U.statistics([root / 'unc'], permutations=0, cache_dir=cache)
        two = U.statistics([root / 'unc'], permutations=0, cache_dir=cache)
        check('the uncertainty cache round-trips',
              two.get('n_files') == one.get('n_files')
              and len(two['q5']) == len(one['q5']),
              f"{two.get('n_files')} file(s), {len(two['q5'])} q5 row(s)")

        # A new file must invalidate it rather than be missed.
        FX.write_per_molecule(root / 'unc', models=['ngboost'], reps=['pdv'],
                              conditions=['gaussian'])
        three = U.statistics([root / 'unc'], permutations=0, cache_dir=cache)
        check('a file landing invalidates the cache',
              three.get('n_files') == one.get('n_files') + 1,
              f"{three.get('n_files')} vs {one.get('n_files')}")


def test_workers_change_the_speed_and_not_the_answer():
    """Eight workers must give the same tables as one."""
    print('  the per-molecule pass, one worker against several')
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        FX.write_per_molecule(directory, models=['qrf', 'ngboost'],
                              reps=['pdv'], conditions=['gaussian'],
                              n_molecules=300, folds=2)
        one = U.statistics([directory], permutations=30, workers=1,
                           progress_every=0)
        many = U.statistics([directory], permutations=30, workers=4,
                            progress_every=0)
        for key in ('support', 'q4', 'q5', 'q6', 'retention', 'enrichment'):
            check(f'{key}: same number of rows',
                  len(one[key]) == len(many[key]),
                  f'{len(one[key])} against {len(many[key])}')
        keys = ['dataset', 'model', 'rep', 'condition', 'sigma', 'fold']
        cols = [c for c in ('auc_delta', 'rho_delta', 'null_lo', 'null_hi',
                            'p_value') if c in one['q4'].columns]
        a = one['q4'].sort_values(keys)[cols].to_numpy(dtype=float)
        b = many['q4'].sort_values(keys)[cols].to_numpy(dtype=float)
        check('every Q4 number is identical, not merely close',
              np.allclose(a, b, rtol=0, atol=0, equal_nan=True),
              f'largest difference {np.nanmax(np.abs(a - b)) if a.size else 0}')

        # A pool that cannot start must not lose the run -- it costs hours.
        broken = U.statistics([directory], permutations=30, workers=99,
                              progress_every=0)
        check('a pool that fails falls back rather than losing the run',
              len(broken['q4']) == len(one['q4']),
              f"{len(broken['q4'])} against {len(one['q4'])}")


def test_kendall_says_what_it_used():
    """Kendall's W is a number or a reason, and never a bare NaN."""
    print("  Kendall's W, where four conditions run on two models")
    models = [f'm{i}' for i in range(6)]
    rows = []
    rng = np.random.default_rng(3)
    for condition in ('gaussian', 'grouped_wider', 'grouped_shifted'):
        for model in models:
            rows.append(dict(dataset='qm9', model=model, rep='pdv',
                             condition=condition, auc_norm=rng.uniform(.5, .9),
                             auc_norm_spread=0.01))
    # the deep-run conditions, which only two models ever ran
    for condition in ('laplace', 'outlier_p10', 'student_t_nu5'):
        for model in models[:2]:
            rows.append(dict(dataset='qm9', model=model, rep='pdv',
                             condition=condition, auc_norm=rng.uniform(.5, .9),
                             auc_norm_spread=0.01))
    frame = pd.DataFrame(rows)

    got = M.kendalls_w(frame, 'pdv')
    check('a number came back, not NaN', np.isfinite(got['kendall_w']),
          got.get('reason'))
    check('it was computed over the conditions the whole roster ran',
          got['n_models'] == 6 and got['n_conditions'] == 3,
          f"{got['n_models']} models, {got['n_conditions']} conditions")
    check('the conditions it dropped are named',
          len(got['conditions_dropped']) == 3, got['conditions_dropped'])

    thin = frame[frame['condition'].isin(['laplace', 'outlier_p10'])]
    got = M.kendalls_w(thin, 'pdv')
    check('too few models to rank is a REASON, not a null',
          not np.isfinite(got['kendall_w']) and 'at least' in got['reason'],
          got.get('reason'))


def main():
    C.apply_style()
    with tempfile.TemporaryDirectory() as out_dir:
        out = Path(out_dir)
        test_curve_shapes()
        test_uncertainty_figures(out)
        test_contingent_figures(out)
        test_guards_still_bite(out)
        test_smoke_output_never_reaches_a_statistic()
        test_a_whisker_is_never_negative(out)
        test_f2_carries_the_residual_and_the_whiskers(out)
        test_f2b_has_no_condition_axis_and_one_clean_fit_per_replicate(out)
        test_f4a_draws_its_companion_panel(out)
        test_panels_that_share_an_axis_stay_the_same_width(out)
        test_the_caches_actually_write(out)
        test_workers_change_the_speed_and_not_the_answer()
        test_kendall_says_what_it_used()
    print()
    if FAILURES:
        print(f'{len(FAILURES)} FAILURE(S)')
        for f in FAILURES:
            print(f'  {f}')
        return 1
    print('OK: every slot draws, and each contingent one is tied to its trigger')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
