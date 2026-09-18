#!/usr/bin/env python3
"""The spread of the labels a model was actually shown, recorded at every level.

WHY THIS EXISTS
---------------
Under every noise condition but one, a model becomes less certain as the labels
are corrupted. Under censoring it becomes MORE certain, on every model and every
dataset, and the fall is monotone in the fraction of labels clipped. The obvious
reading is that censoring is the only condition that corrupts labels by REDUCING
their spread: additive noise widens the training distribution, while replacing
everything past a limit with the limit narrows it, and every uncertainty
estimator in the roster is fitted to the spread of the targets it is given.

That reading could not be checked. The only label column carried through the
statistics was `label_scale`, which is the CLEAN training spread and is the same
number at every level. So `q5_mean_uncertainty` now also records the spread of
the RECORDED labels, which is the clean label plus the amount injected — the
same quantity the error is built from everywhere else, so nothing new is needed
from the writer.

WHAT IT CHECKS
--------------
1. With no noise the recorded spread equals the clean spread, so the ratio is 1.
2. Additive noise WIDENS the recorded labels, and the ratio climbs above 1.
3. Censoring NARROWS them, and the ratio falls below 1 and keeps falling.
4. Both are reported in the label's own units, so `label_scale` is applied.
5. A frame without the label columns gets NaN rather than a wrong number.

    python scripts/test_recorded_label_spread.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uncertainty_stats as unc  # noqa: E402

N = 800


def cell(condition, level, label_scale=1.0, seed=0):
    """One out-of-fold cell at one noise level.

    `censoring` clips the top `level` fraction to the limit, which is what the
    injector does; `gaussian` adds a draw of that width.
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(0, 1, N)
    if condition == 'censoring':
        injected = (np.where(y > np.quantile(y, 1 - level),
                             np.quantile(y, 1 - level) - y, 0.0)
                    if level > 0 else np.zeros(N))
    else:
        injected = rng.normal(0, level, N) if level > 0 else np.zeros(N)
    return pd.DataFrame({
        'dataset': 'qm9', 'model': 'rf', 'rep': 'ecfp4', 'condition': condition,
        'sigma': level, 'fold': 0, 'split': 'train_oof',
        'label_scale': label_scale,
        'y_true_clean': y, 'injected_noise': injected,
        'y_pred': y + rng.normal(0, 0.1, N),
        'uncertainty': np.abs(rng.normal(0.3, 0.05, N)),
    })


def series(condition, levels, **kw):
    got = unc.q5_mean_uncertainty(
        pd.concat([cell(condition, l, **kw) for l in levels], ignore_index=True))
    return got.sort_values('sigma').set_index('sigma')


def check_clean_level_is_one():
    for condition in ('gaussian', 'censoring'):
        row = series(condition, [0.0]).iloc[0]
        assert abs(row['label_spread_ratio'] - 1.0) < 1e-9, (
            f'{condition} at level zero gave a ratio of '
            f'{row["label_spread_ratio"]}, and nothing was injected')
    print('  level zero: the recorded spread is the clean spread, both conditions')


def check_additive_noise_widens():
    got = series('gaussian', [0.0, 0.25, 0.5, 1.0])
    ratios = got['label_spread_ratio'].tolist()
    assert all(b > a for a, b in zip(ratios, ratios[1:])), (
        f'additive noise should widen the labels at every step, got {ratios}')
    assert ratios[-1] > 1.2, (
        f'at a level of 1.0 the recorded spread should be well above the clean '
        f'one, got a ratio of {ratios[-1]:.3f}')
    print('  additive noise widens: ratio '
          + ' -> '.join(f'{r:.3f}' for r in ratios))


def check_censoring_narrows():
    got = series('censoring', [0.0, 0.1, 0.25, 0.5])
    ratios = got['label_spread_ratio'].tolist()
    assert all(b < a for a, b in zip(ratios, ratios[1:])), (
        f'censoring should narrow the labels at every step, got {ratios}')
    assert ratios[-1] < 0.8, (
        f'at half the labels clipped the recorded spread should be well below '
        f'the clean one, got a ratio of {ratios[-1]:.3f}')
    print('  censoring narrows:    ratio '
          + ' -> '.join(f'{r:.3f}' for r in ratios))
    print('  those two lines are the mechanism behind the censoring result, '
          'measured rather than argued')


def check_label_units():
    plain = series('gaussian', [0.5]).iloc[0]
    scaled = series('gaussian', [0.5], label_scale=4.0).iloc[0]
    assert abs(scaled['recorded_label_sd'] - 4.0 * plain['recorded_label_sd']) < 1e-6, (
        'the recorded spread is not being put into the label\'s own units')
    assert abs(scaled['label_spread_ratio'] - plain['label_spread_ratio']) < 1e-9, (
        'the ratio should not depend on the units')
    print('  units: the spreads scale with label_scale and the ratio does not')


def check_missing_columns_give_nan():
    df = cell('gaussian', 0.5).drop(columns=['injected_noise'])
    got = unc.q5_mean_uncertainty(df)
    assert got['recorded_label_sd'].isna().all(), (
        'a frame with no injected-noise column produced a spread anyway')
    print('  a frame without the label columns gets NaN, not a wrong number')


def main():
    print(__doc__.split('\n')[0])
    check_clean_level_is_one()
    check_additive_noise_widens()
    check_censoring_narrows()
    check_label_units()
    check_missing_columns_give_nan()
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
