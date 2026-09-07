#!/usr/bin/env python
"""The completeness check must count only what the generators asked for.

WHAT THIS CATCHES
-----------------
`--stage 2` restricts the expected set to `deep_run_pairs.json` -- six models on
three representations -- but the PARTIAL and THIN counts were taken over every
combination present on disk. On 2026-09-07 that printed 59 landed of 113 expected
beside 36 partial and 77 thin, and the thin list named Sort & Slice combinations
that no deep-run pair contains. The two halves of one line answered two different
questions, and the half that was wrong is the half that says how much work is left.

    python scripts/test_check_runs_landed_selection.py
"""
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import check_runs_landed as K                                       # noqa: E402
import figlib_config as C                                           # noqa: E402


def write_cell(directory, condition, rep, model, replicates):
    """One anova results file: every noise level, too few replicates.

    Every level, so the only thing wrong with it is the replicate count. A file
    short on levels reports PARTIAL, which would not tell the two counts apart.
    """
    rows = []
    for level in C.expected_levels(condition):
        for replicate in range(replicates):
            rows.append({'sigma': level, 'model': model, 'rep': rep,
                         'replicate': replicate, 'r2': 0.5, 'mae': 0.1,
                         'rmse': 0.2, 'dataset': 'qm9'})
    path = Path(directory) / f'anova_{condition}_{rep}_{model}.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main():
    selection = json.loads((ROOT / 'deep_run_pairs.json').read_text())
    models = selection['generator_labels']
    reps = selection['representations']

    inside_model, inside_rep = models[1], reps[1]
    outside_model, outside_rep = 'dnn', 'sns'
    if outside_model in models or outside_rep in reps:
        raise SystemExit(f'FAIL: this test needs a combination the deep run does '
                         f'not name; {outside_model} x {outside_rep} is in the file')

    with tempfile.TemporaryDirectory() as tmp:
        write_cell(tmp, 'gaussian', inside_rep, inside_model, replicates=2)
        write_cell(tmp, 'gaussian', outside_rep, outside_model, replicates=2)

        result = K.check_qm9(tmp, stage=2)
        if result is None:
            raise SystemExit('FAIL: check_qm9 returned nothing')

        cover = result['coverage']
        named = {(row.model, row.rep) for row in cover.itertuples()}

        checks = 0
        if (outside_model, outside_rep) in named:
            raise SystemExit(
                f'FAIL: {outside_model} x {outside_rep} is not in '
                f'deep_run_pairs.json and must not appear in the coverage table; '
                f'rows present: {sorted(named)}')
        checks += 1

        if (K.C.canonical_model(inside_model, "qm9"),
                K.C.canonical_rep(inside_rep)) not in named:
            raise SystemExit(
                f'FAIL: {inside_model} x {inside_rep} IS in deep_run_pairs.json '
                f'and must be reported; rows present: {sorted(named)}')
        checks += 1

        if result['thin'] != 1:
            raise SystemExit(
                f'FAIL: one combination the deep run names is short on '
                f'replicates, so thin must be 1, not {result["thin"]}')
        checks += 1

        if result['partial'] != 0:
            raise SystemExit(
                f'FAIL: neither combination is missing a level, so partial must '
                f'be 0, not {result["partial"]}')
        checks += 1

    print(f'OK: {checks} checks. The completeness check counts only the '
          f'combinations deep_run_pairs.json names.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
