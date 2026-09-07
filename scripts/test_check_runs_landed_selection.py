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

    checks += check_the_laboratory_runs_are_counted()
    checks += check_every_uncertainty_condition_is_counted()

    print(f'OK: {checks} checks. The completeness check counts only the '
          f'combinations deep_run_pairs.json names, and it counts every '
          f'laboratory and uncertainty cell that was submitted.')
    return 0


def check_the_laboratory_runs_are_counted():
    """`--stage 2` must ask about the laboratory depth run and censoring.

    WHAT THIS CATCHES. `check_assay` took no stage argument at all, so both
    `--stage 1` and `--stage 2` asked the same question -- the breadth grid --
    and the two laboratory runs submitted on 2026-09-06, 19 arrays and 327 tasks
    each, were counted NOWHERE. Not landed, not missing. The command HANDOFF.md
    names as the only proof that nothing is missing could not see 654 tasks.
    """
    gen = K._generator('val', 'slurm_scripts_validation_rerun/generate_scripts.py')
    breadth, _ = K.assay_expected(1)
    depth, _ = K.assay_expected(2)
    if not breadth or not depth:
        raise SystemExit('FAIL: the laboratory expected sets came back empty')

    n = 0
    if breadth == depth:
        raise SystemExit('FAIL: stage 1 and stage 2 ask the same question, which '
                         'is what left the laboratory depth run uncounted')
    n += 1

    conds_1 = {c for _, _, _, c in breadth}
    conds_2 = {c for _, _, _, c in depth}
    if conds_1 != set(gen.BREADTH_GRID):
        raise SystemExit(f'FAIL: stage 1 conditions are {sorted(conds_1)}, not the '
                         f'breadth grid {gen.BREADTH_GRID}')
    n += 1
    if conds_2 != set(gen.DEPTH_ONLY) | {'censoring'}:
        raise SystemExit(f'FAIL: stage 2 conditions are {sorted(conds_2)}, not the '
                         f'three depth-only ones plus censoring')
    n += 1

    # Censoring is five NAMED pairs, so it must not be a cross product.
    censoring = json.loads((ROOT / 'censoring_pairs.json').read_text())
    named = {(C.canonical_model(m, 'validation'), C.canonical_rep(r))
             for m, r in censoring['validation_pairs']}
    got = {(m, r) for _, m, r, c in depth if c == 'censoring'}
    if got != named:
        raise SystemExit(f'FAIL: censoring expects {sorted(got)}, not the five '
                         f'pairs {sorted(named)} the file names')
    n += 1
    return n


def check_every_uncertainty_condition_is_counted():
    """The uncertainty runs submit seven conditions; the check expected four.

    Reading only the main grid's four left the three depth-only conditions
    counted nowhere -- neither landed nor missing.

    The total is not pinned to a constant. It was 378 at six models, and adding
    `GP-Hetero` on 2026-09-07 made it 441; a number written here would have to be
    edited every time the roster moves, which is how a check stops describing the
    run. What IS pinned is that the two submissions add up to the whole: the
    first runs the conditions the generator defaults to and the second runs the
    rest, and no condition may be in both or in neither.
    """
    gen = K._generator('unc',
                       'slurm_scripts_uncertainty_rerun/generate_scripts.py')
    if gen is None:
        raise SystemExit('FAIL: could not read the uncertainty generator')
    per_condition = len(gen.DATASETS) * len(gen.MODELS) * len(gen.REPS)
    first = [c for c in gen.MAIN_GRID_CONDITIONS if c != 'censoring']
    second = ['censoring'] + list(gen.DEEP_RUN_CONDITIONS)
    if sorted(first + second) != sorted(gen.KNOWN_CONDITIONS):
        raise SystemExit(
            f'FAIL: the two submissions run {sorted(first + second)} between '
            f'them, against the {sorted(gen.KNOWN_CONDITIONS)} the completeness '
            f'check expects. A condition in neither is counted nowhere.')
    if set(first) & set(second):
        raise SystemExit(f'FAIL: {sorted(set(first) & set(second))} is in both '
                         f'submissions, so its cells are expected twice')
    total = per_condition * len(gen.KNOWN_CONDITIONS)
    if total != per_condition * len(first) + per_condition * len(second):
        raise SystemExit('FAIL: the submissions do not add to the expected grid')
    print(f'  uncertainty: {len(gen.MODELS)} models x {len(gen.DATASETS)} '
          f'datasets x {len(gen.REPS)} reps = {per_condition} tasks per '
          f'condition; {per_condition * len(first)} in the first submission, '
          f'{per_condition * len(second)} in the second, {total} expected cells')
    return 3


if __name__ == '__main__':
    sys.exit(main())
