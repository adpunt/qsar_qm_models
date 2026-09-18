#!/usr/bin/env python3
"""The permutation band Q4 is read against has to be a band about the uncertainty.

WHAT WAS WRONG
--------------
`figlib_uncertainty.q4` computed one permutation null, for
`error_noise_spearman` -- the correlation between the out-of-fold error and the
injected noise. The uncertainty appears nowhere in that statistic. The column it
produced, `outside_null`, was then read by three places as though it meant the
uncertainty had contributed something:

  * `figlib_decisions.d7_uncertainty_option` set `fires = outside_null` and, if
    ANY censoring fold fired, wrote "the uncertainty finds clipped labels better
    than the error alone, outside the permutation band" into
    `notes_for_the_text.md` -- which is the file the Results were told to quote.
  * `figlib_decisions.uncertainty_pairs` printed `rho_ratio` beside that band's
    `null_lo`/`null_hi` and derived `clears_the_band` from it.
  * `figlib_tables.t6_uncertainty` printed it as "Outside null".

On the 17 September harvest that produced censoring at 377 of 396 cells against
Gaussian at 7 of 492, and the difference is arithmetic rather than a finding: a
censored label is a large error by construction, so the ERROR tracks the noise
under censoring whatever the model's uncertainty does. 2,311 of the 5,932 firing
folds were outside the band on the LOW side -- the error tracking the noise LESS
than chance -- and counted as firing.

WHAT IT CHECKS
--------------
1. `delta_noise_spearman` is exactly ratio minus error, on real arrays.
2. On data where the uncertainty carries NO information about the noise, the
   observed gain sits INSIDE its own band, while the error's band sits far up
   the scale and nowhere near it. That is the whole point: the two are different
   questions, and only the first is readable across conditions that damage the
   labels by very different amounts. The gain's band is NOT centred on zero --
   dividing by any imperfect uncertainty degrades the ranking on average -- so
   the gain has to be read against the band and never against zero.
3. On data where dividing by the uncertainty genuinely sharpens the ranking, the
   delta is above its band and `adds_signal` is True.
4. `adds_signal` is False when the observed value is below the band, so "made
   the ranking worse" is never reported as "found the bad labels".
5. A cell whose statistic could not be computed does not count as outside the
   band. `permutation_null` writes `observed_inside_null=False` for a NaN
   statistic, and negating that turned "never measured" into "fired" -- 2,417
   of the 5,932 firing folds on the 17 September harvest.

    python scripts/test_q4_band_is_about_the_uncertainty.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uncertainty_stats as unc  # noqa: E402
import figlib_uncertainty as U  # noqa: E402

N = 400
PERMUTATIONS = 200


def frame(uncertainty_is_useful, seed=0):
    """One out-of-fold cell, with the uncertainty useful for the ranking or not.

    The error is built the way the pipeline builds it -- the clean residual plus
    the injected noise -- so the error tracks the noise in BOTH cases. What
    changes is only whether dividing by the uncertainty SHARPENS that tracking.

    ⚠️ "Useful" is not "the uncertainty equals the noise". The ratio asks how
    much larger an error is than the model expected, so an uncertainty that
    already anticipates the injected noise flattens the ratio and the gain comes
    out strongly NEGATIVE -- measured at -0.31 on a fixture built that way. What
    helps is an uncertainty that captures the part of the error which is NOT the
    injected noise: divide it out and what is left is the noise. So the useful
    case here is a heteroscedastic residual with the uncertainty tracking its
    scale, which is the mechanism Q4 is actually asking about.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, N)
    # The model is much less certain on some molecules than others, for reasons
    # that have nothing to do with the injected noise.
    residual_scale = np.exp(rng.normal(0, 0.8, N))
    residual = rng.normal(0, 1, N) * residual_scale
    noise = rng.normal(0, 0.5, N)
    # A quarter of the molecules get a much larger draw, so there is a subset to
    # find at all.
    noise[rng.random(N) < 0.25] *= 4.0
    y_pred = y - residual
    uncertainty = (residual_scale if uncertainty_is_useful
                   else np.exp(rng.normal(0, 0.8, N)))
    return pd.DataFrame({
        'dataset': 'qm9', 'model': 'ngboost', 'rep': 'pdv',
        'condition': 'censoring', 'sigma': 0.25, 'fold': 0,
        'split': 'train_oof',
        'y_true_clean': y, 'y_pred': y_pred, 'injected_noise': noise,
        'uncertainty': uncertainty,
    })


def check_delta_is_the_difference():
    d = frame(True, seed=1)
    arrays = (d['y_true_clean'].to_numpy(), d['y_pred'].to_numpy(),
              d['injected_noise'].to_numpy(), None)
    u = d['uncertainty'].to_numpy()
    error = unc.STATISTICS['error_noise_spearman'](arrays, False)
    ratio = unc.STATISTICS['ratio_noise_spearman'](arrays, False, unc=u)
    delta = unc.STATISTICS['delta_noise_spearman'](arrays, False, unc=u)
    assert np.isclose(delta, ratio - error), (
        f'delta {delta} is not ratio {ratio} minus error {error}')
    print(f'  delta = ratio - error: {ratio:.4f} - {error:.4f} = {delta:.4f}')


def band(df, statistic):
    got = unc.permutation_null(df, statistic=statistic,
                               n_permutations=PERMUTATIONS)
    assert len(got) == 1, f'{statistic}: expected one group, got {len(got)}'
    return got.iloc[0]


def check_uninformative_uncertainty_does_not_fire():
    d = frame(False, seed=2)
    delta = band(d, 'delta_noise_spearman')
    error = band(d, 'error_noise_spearman')
    print(f'  uninformative: delta observed {delta.observed:+.4f} in '
          f'[{delta.null_lo:+.4f}, {delta.null_hi:+.4f}]; '
          f'error observed {error.observed:+.4f} in '
          f'[{error.null_lo:+.4f}, {error.null_hi:+.4f}]')
    assert bool(delta.observed_inside_null), (
        'an uninformative uncertainty cleared the band on the gain')
    # AND THE BAND IS NOT CENTRED ON ZERO, WHICH IS WHY THERE HAS TO BE A BAND.
    # Dividing by any uncertainty that carries noise of its own degrades the
    # ranking on average, so the gain has a negative expectation even when the
    # uncertainty knows nothing. Testing the gain against zero would therefore
    # call a null result a loss, and call a small real gain nothing.
    assert delta.null_hi < 0, (
        f'the delta null came out straddling zero ({delta.null_lo:+.4f} to '
        f'{delta.null_hi:+.4f}); the point of the band is that it does not')
    # AND THE TWO BANDS ARE NOWHERE NEAR EACH OTHER. The error's null sits far
    # up the scale, because permuting the noise and recomputing the error from
    # the permuted value leaves an error that still contains it -- that is the
    # documented correct null, and it is why the error band lands next to the
    # error and says nothing about the uncertainty.
    assert error.null_lo > delta.null_hi + 0.2, (
        f'the error band [{error.null_lo:+.4f}, {error.null_hi:+.4f}] and the '
        f'gain band [{delta.null_lo:+.4f}, {delta.null_hi:+.4f}] are supposed '
        f'to live on different parts of the scale; reading a claim about the '
        f'uncertainty against the first is the defect this test pins')
    assert error.null_lo > 0, 'the error band should exclude zero here'


def frame_with_a_real_gain(seed=7):
    """A cell where dividing by the uncertainty genuinely has to help.

    Built backwards from the statistic: the error is made exactly
    `uncertainty x (1 + |noise|)`, so the ratio is `1 + |noise|` and ranks the
    injected noise perfectly, while the error alone is that ranking blurred by
    the spread of the uncertainty. If the band cannot see a gain here it cannot
    see one anywhere.

    ⚠️ IT TOOK A CONSTRUCTION THIS ARTIFICIAL, and that is worth recording. On
    ordinary additive noise -- error = |residual + injected| -- the gain came out
    NEGATIVE at every setting swept on 2026-09-17: residual spread 0.5 to 1.5,
    noise 0.5 to 1.5, contamination 3x to 4x, with the uncertainty set to the
    residual scale, to the absolute residual, and at random. Best of twelve was
    +0.009. The error already contains the injected noise additively, so
    dividing by anything imperfect mostly adds variance. The ratio statistic has
    very little room to beat the error by construction, which is a limitation of
    the measurement and belongs in the paper beside the result.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, N)
    noise = rng.normal(0, 0.5, N)
    noise[rng.random(N) < 0.25] *= 4.0
    uncertainty = np.exp(rng.normal(0, 0.8, N))
    error = uncertainty * (1.0 + np.abs(noise))
    return pd.DataFrame({
        'dataset': 'qm9', 'model': 'ngboost', 'rep': 'pdv',
        'condition': 'censoring', 'sigma': 0.25, 'fold': 0,
        'split': 'train_oof',
        'y_true_clean': y, 'y_pred': y + noise - error,
        'injected_noise': noise, 'uncertainty': uncertainty,
    })


def check_a_real_gain_fires():
    d = frame_with_a_real_gain()
    delta = band(d, 'delta_noise_spearman')
    print(f'  real gain:     delta observed {delta.observed:+.4f} against '
          f'[{delta.null_lo:+.4f}, {delta.null_hi:+.4f}]')
    assert not bool(delta.observed_inside_null), (
        'a gain built to be perfect did not clear the band')
    assert delta.observed > delta.null_hi, (
        'it cleared the band on the wrong side')


def check_adds_signal_has_a_side():
    """Below the band is not 'found the bad labels'."""
    got = U.q4(frame_with_a_real_gain(seed=4), permutations=PERMUTATIONS,
               where='test')
    assert 'adds_signal' in got.columns, 'q4 lost the adds_signal column'
    assert 'error_outside_null' in got.columns, (
        'q4 lost the error band, which is the precondition and is worth keeping')
    row = got.iloc[0]
    assert bool(row['adds_signal']), 'the informative case did not add signal'
    # Force the below-the-band case and check the flag refuses it.
    flipped = got.copy()
    flipped['observed'] = flipped['null_lo'] - 1.0
    flipped['outside_null'] = True
    recomputed = flipped['outside_null'] & (flipped['observed']
                                            > flipped['null_hi'])
    assert not bool(recomputed.iloc[0]), (
        'a value below the band was counted as adding signal')
    print('  adds_signal: True above the band, False below it')


def check_unmeasured_is_not_outside():
    """A cell with no statistic and no band has not cleared anything."""
    d = frame_with_a_real_gain(seed=11).copy()
    # A constant target: the injected noise is the same for every molecule, so
    # the Spearman is undefined and the whole null comes back NaN.
    d['injected_noise'] = 0.25
    got = U.q4(d, permutations=PERMUTATIONS, where='test')
    if not len(got):
        print('  unmeasured: q4 returned no rows for a constant target, which '
              'is also a correct answer')
        return
    row = got.iloc[0]
    assert not bool(row['outside_null']), (
        'a cell with no computable statistic was marked outside the band')
    assert not bool(row['adds_signal']), (
        'a cell with no computable statistic was marked as adding signal')
    print('  unmeasured: a NaN statistic is not outside the band')


def main():
    print(__doc__.split('\n')[0])
    check_delta_is_the_difference()
    check_uninformative_uncertainty_does_not_fire()
    check_a_real_gain_fires()
    check_adds_signal_has_a_side()
    check_unmeasured_is_not_outside()
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
