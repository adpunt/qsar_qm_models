#!/usr/bin/env python
"""An accuracy file with no out-of-fold rows is not a landed cell.

    python scripts/test_check_runs_landed_oof.py

WHAT THIS CATCHES, AND WHAT IT COST.

`check_runs_landed.py` counted `anova_*.csv` and nothing else, and `load_qm9`
excludes every sibling suffix -- so the per-molecule uncertainty rows were never
looked at. The two files are not written together: the accuracy row for a
(level, replicate) is saved BEFORE the out-of-fold pass runs, and the pass can
fail on its own while the row it already wrote stands.

All nine `gauche_rbf` tasks of 12980590 failed inside that pass at every level and
every replicate, and `--stage 1` reported not one of their cells MISSING or
PARTIAL (RERUN_PLAN.md 13.27 D2). The per-molecule uncertainty is the entire
reason those pairs are settled.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import pandas as pd  # noqa: E402

import check_runs_landed as K  # noqa: E402

FAILS = []


def check(name, ok, detail=''):
    if not ok:
        FAILS.append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f'\n          {detail}' if detail and not ok else ''))


def write(path, splits):
    """One uncertainty file. `splits` is (split, sigma, iteration) triples."""
    pd.DataFrame(splits, columns=['split', 'sigma', 'iteration']).to_csv(
        path, index=False)


def main(tmp):
    want = K.qm9_oof_expected(1)
    check('the settled pairs are a SUBSET of the grid, not all of it',
          0 < len(want) < len(K.qm9_expected(1)[0]),
          f'{len(want)} out-of-fold cells against '
          f'{len(K.qm9_expected(1)[0])} accuracy cells')

    reps = {r for _, r, _ in want}
    check('only ECFP4, PDV and ChemBERTa run the pass',
          reps == {'ecfp4', 'pdv', 'chemberta'}, f'got {sorted(reps)}')
    check('gauche_rbf is among the models that run it',
          'gauche_rbf' in {m for _, _, m in want})

    gen = K._generator('qm9', 'slurm_scripts_qm9_rerun/generate_scripts.py')
    levels = len(str(gen.CONDITIONS['gaussian'][1]).split())
    reps_n = int(gen.STAGE_DEFAULTS[1]['replicates'])
    cells = [(lv, it) for lv in range(levels)
             for it in range(1, reps_n + 1)]

    stem = tmp / 'anova_gaussian_ecfp4_gauche_rbf_uncertainty_values.csv'

    # 1. THE REAL CASE: every accuracy row written, no out-of-fold row anywhere.
    write(stem, [('test', lv, it) for lv, it in cells])
    res = K.check_qm9_oof(tmp, 1)
    row = res['coverage']
    hit = row[(row['model'] == 'gauche_rbf') & (row['rep'] == 'ecfp4')
              & (row['condition'] == 'gaussian')]
    check('a file with test rows and NO train_oof rows is NO_OOF_ROWS',
          len(hit) == 1 and hit.iloc[0]['status'] == 'NO_OOF_ROWS',
          f'got {list(hit["status"])}')
    check('and it does not count as landed', res['ok'] < res['want'])

    # 2. Part-way through: some cells scored, not all.
    write(stem, [('test', lv, it) for lv, it in cells]
          + [('train_oof', lv, it) for lv, it in cells[:5]])
    row = K.check_qm9_oof(tmp, 1)['coverage']
    hit = row[(row['model'] == 'gauche_rbf') & (row['rep'] == 'ecfp4')
              & (row['condition'] == 'gaussian')]
    check('some cells scored and not others is PARTIAL_OOF',
          len(hit) == 1 and hit.iloc[0]['status'] == 'PARTIAL_OOF',
          f'got {list(hit["status"])}')

    # 3. Complete.
    write(stem, [('test', lv, it) for lv, it in cells]
          + [('train_oof', lv, it) for lv, it in cells])
    res = K.check_qm9_oof(tmp, 1)
    row = res['coverage']
    hit = row[(row['model'] == 'gauche_rbf') & (row['rep'] == 'ecfp4')
              & (row['condition'] == 'gaussian')]
    check('every (level, replicate) scored out of fold is OK, and drops off '
          'the list', len(hit) == 0)
    check('and it counts as landed', res['ok'] == 1, f"ok={res['ok']}")

    # 4. A missing sibling file is named, not skipped.
    stem.unlink()
    row = K.check_qm9_oof(tmp, 1)['coverage']
    hit = row[(row['model'] == 'gauche_rbf') & (row['rep'] == 'ecfp4')
              & (row['condition'] == 'gaussian')]
    check('no sibling file at all is MISSING_FILE',
          len(hit) == 1 and hit.iloc[0]['status'] == 'MISSING_FILE',
          f'got {list(hit["status"])}')

    print()
    if FAILS:
        print(f'  {len(FAILS)} failure(s): ' + ', '.join(FAILS))
        return 1
    print('  all checks passed.')
    return 0


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        sys.exit(main(Path(d)))
