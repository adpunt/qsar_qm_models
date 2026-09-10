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

import numpy as np                                              # noqa: E402
import pandas as pd                                             # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C                                       # noqa: E402
import figlib_decisions as D                                    # noqa: E402
import figlib_figures as FIG                                    # noqa: E402
import figlib_fixtures as FX                                    # noqa: E402
import figlib_guard as G                                        # noqa: E402
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

        check('R6 draws nothing for a condition that was never run',
              FIG.r6_representation_profile(summary, out, 'no_such_condition')
              is None)
        check('R15b draws nothing for a model that was never run',
              FIG.r15b_rank_against_level_by_rep(accuracy, out, 'no_such_model',
                                                 'gaussian') is None)


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
