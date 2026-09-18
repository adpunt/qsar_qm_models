#!/usr/bin/env python3
"""A blank noise condition must never become its own cell.

WHY THIS EXISTS
---------------
`_pick` returns the CONDITION COLUMN when the file has one, and the loader's
`unspecified` fallback only fires when the column is missing altogether. So a
file that carries a `condition` column which is empty on some rows passed those
blanks through as NaN. `_cell_iter` groups with `dropna=False`, so every blank
became its own cell keyed on NaN.

That is 55 rows of `d7_q6.csv` in the 16 September harvest, all on QM9, across
thirteen models and all six representations. While every number in the
uncertainty table was a median over the seven noise conditions those rows
joined into each of them at once, and nothing complained, because a median does
not care where its inputs came from. 54 of the 55 are combinations that appear
under no named condition anywhere else in the file, so they are measurements
that lost their label rather than duplicates of a labelled row.

WHAT IT CHECKS
--------------
1. A blank condition column, on a file whose NAME names a condition, is filled
   from the name rather than left as NaN.
2. A blank condition column on a file whose name names nothing is refused under
   `strict`, because those rows cannot be told from any other noise type.
3. With `strict=False` the same file loads and the rows read `unspecified`,
   which is a value a group key can see.
4. No frame the loader returns ever carries a NaN condition.

    python scripts/test_blank_condition_is_not_a_cell.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uncertainty_stats as unc  # noqa: E402

N = 40


def frame(condition_values):
    """One small QM9-shaped uncertainty file, `condition` partly blank."""
    rng = np.random.default_rng(0)
    n = len(condition_values)
    return pd.DataFrame({
        'dataset': 'qm9', 'model': 'qrf', 'rep': 'pdv',
        'condition': condition_values,
        'sigma': np.tile([0.0, 0.5], n // 2),
        'fold': 0, 'split': 'test',
        'y_true_original': rng.normal(0, 1, n),
        'y_pred_mean': rng.normal(0, 1, n),
        'y_pred_std_uncalibrated': np.abs(rng.normal(0.3, 0.05, n)),
        'injected_noise': np.zeros(n),
        'canonical_smiles': [f'C{i % 7}' for i in range(n)],
        'noise_pattern': 1.0,
    })


def write(tmp, name, condition_values):
    path = Path(tmp) / name
    frame(condition_values).to_csv(path, index=False)
    return path


def load(path, strict=True):
    return unc.load_uncertainty(str(path), strict=strict,
                                pattern='*.csv', dataset_name='qm9')


def check_name_fills_the_blanks():
    values = ['gaussian'] * (N // 2) + [''] * (N // 2)
    with tempfile.TemporaryDirectory() as tmp:
        got = load(write(tmp, 'uncertainty_gaussian_r0_uncertainty_values.csv', values))
    assert got['condition'].isna().sum() == 0, (
        'a blank condition survived on a file whose name names one')
    assert set(got['condition']) == {'gaussian'}, (
        f'expected every row under gaussian, got {sorted(set(got["condition"]))}')
    print('  a blank column on a named file is filled from the name')


def check_strict_refuses_what_it_cannot_name():
    values = ['gaussian'] * (N // 2) + [''] * (N // 2)
    with tempfile.TemporaryDirectory() as tmp:
        path = write(tmp, 'plain_run_0_uncertainty_values.csv', values)
        try:
            load(path, strict=True)
        except unc.UncertaintySchemaError as error:
            assert '20' in str(error), (
                f'the message should count the blank rows, got: {error}')
            print('  strict refuses a blank column it cannot name, and counts them')
            return
    raise AssertionError(
        'strict accepted a file with 20 rows whose noise condition is unknown')


def check_lenient_marks_them_unspecified():
    values = ['gaussian'] * (N // 2) + [''] * (N // 2)
    with tempfile.TemporaryDirectory() as tmp:
        got = load(write(tmp, 'plain_run_0_uncertainty_values.csv', values), strict=False)
    assert got['condition'].isna().sum() == 0, 'a NaN condition got through'
    assert (got['condition'] == 'unspecified').sum() == N // 2, (
        'the unnameable rows should read unspecified, not something else')
    print('  lenient gives them `unspecified`, which a group key can see')


def check_no_nan_reaches_a_cell():
    """The failure this whole file is about, stated as the cell iterator sees it."""
    values = ['gaussian'] * (N // 2) + [None] * (N // 2)
    with tempfile.TemporaryDirectory() as tmp:
        got = load(write(tmp, 'uncertainty_gaussian_r0_uncertainty_values.csv', values))
    cells = {rec['condition'] for rec, _ in unc._cell_iter(got, min_n=1)}
    assert not any(pd.isna(c) for c in cells), (
        f'a cell keyed on a missing condition still exists: {cells}')
    print('  no cell the iterator yields is keyed on a missing condition')


def main():
    print(__doc__.split('\n')[0])
    check_name_fills_the_blanks()
    check_strict_refuses_what_it_cannot_name()
    check_lenient_marks_them_unspecified()
    check_no_nan_reaches_a_cell()
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
