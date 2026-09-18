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
    checks += check_every_model_the_files_name_is_expected()
    checks += check_a_producer_with_no_directory_is_not_skipped()

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

    # A model the generator queues on ONE representation must not be expected on
    # six. GP-Tanimoto is ECFP4 only -- a Tanimoto kernel is defined on binary
    # vectors -- and crossing blindly invented 45 laboratory cells nobody
    # submitted, so a complete breadth grid read as 981 landed of 1,026.
    pairs = {(m, r) for _, m, r, _ in breadth}
    tanimoto = {r for m, r in pairs
                if m == C.canonical_model('GP-Tanimoto', 'validation')}
    if tanimoto != {C.canonical_rep('ECFP4')}:
        raise SystemExit(
            f'FAIL: GP-Tanimoto is expected on {sorted(tanimoto)}; the generator '
            f'queues it on ECFP4 alone (its reps_for), so every other '
            f'representation is a cell nobody submitted')
    n += 1
    queued = sum(len(gen.reps_for(m, gen.ALL_REPS)) for m in gen.MODELS_ALL)
    if len(breadth) != queued * len(gen.BREADTH_GRID) * len(gen.DATASETS):
        raise SystemExit(
            f'FAIL: the breadth grid expects {len(breadth)} cells against the '
            f'{queued * len(gen.BREADTH_GRID) * len(gen.DATASETS)} the generator '
            f'queues')
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


def check_every_model_the_files_name_is_expected():
    """Every model in the two selection files must reach the expected set.

    WHAT THIS CATCHES. `qm9_expected` builds its set from the GENERATOR's MODELS
    table and then INTERSECTS it with `deep_run_pairs.json`. A model named in the
    file under a spelling the generator does not have -- a typo, or a model added
    to the file before it was added to the generator -- drops out of the
    intersection in silence. It is then queued nowhere and missed by nothing, so
    the check reports a complete deep run while that model has never run. The
    same holds on the laboratory side, where `validation_labels` is matched
    against the generator's MODELS_ALL.
    """
    deep = json.loads((ROOT / 'deep_run_pairs.json').read_text())
    censoring = json.loads((ROOT / 'censoring_pairs.json').read_text())
    n = 0

    want, _ = K.qm9_expected(2)
    have_m = {m for _, _, m in want}
    for label in deep['generator_labels']:
        if C.canonical_model(label, 'qm9') not in have_m:
            raise SystemExit(
                f'FAIL: deep_run_pairs.json names {label} and QM9 stage 2 '
                f'expects no cell for it. The QM9 job generator has no such key, '
                f'so the intersection dropped it without a word.')
    n += 1
    have_r = {r for _, r, _ in want}
    for rep in deep['representations']:
        if C.canonical_rep(rep) not in have_r:
            raise SystemExit(
                f'FAIL: deep_run_pairs.json names representation {rep} and QM9 '
                f'stage 2 expects no cell on it')
    n += 1
    for model, rep in censoring['generator_pairs']:
        cell = ('censoring', C.canonical_rep(rep), C.canonical_model(model, 'qm9'))
        if cell not in want:
            raise SystemExit(
                f'FAIL: censoring_pairs.json names {model} x {rep} and QM9 '
                f'stage 2 expects no censoring cell for it')
    n += 1

    assay, _ = K.assay_expected(2)
    have_am = {m for _, m, _, _ in assay}
    for label in deep['validation_labels']:
        if C.canonical_model(label, 'validation') not in have_am:
            raise SystemExit(
                f'FAIL: deep_run_pairs.json names {label} on the laboratory side '
                f'and the depth run expects no cell for it. The laboratory job '
                f'generator has no such name in MODELS_ALL.')
    n += 1
    return n


def check_a_producer_with_no_directory_is_not_skipped():
    """No --validation-dir must not mean the laboratory runs vanish.

    WHAT THIS CATCHES. `check_assay` and `check_uncertainty` returned None when
    no directory was passed, and `report` skips a None without printing a line.
    So `python scripts/check_runs_landed.py --stage 2` typed with no flags -- the
    spelling every document gives -- said nothing whatever about the laboratory
    depth run, censoring or the uncertainty runs, and still exited on the QM9
    counts alone. Neither landed nor missing is the one state this tool exists to
    make impossible.
    """
    n = 0
    for name, call, expected in (
            ('assay accuracy', lambda: K.check_assay(None, stage=2),
             len(K.assay_expected(2)[0])),
            ('uncertainty runs', lambda: K.check_uncertainty(None), None)):
        row = call()
        if row is None:
            raise SystemExit(
                f'FAIL: {name} with no directory returned None, and report() '
                f'skips a None, so those cells are counted nowhere')
        if 'NOT CHECKED' not in row['note']:
            raise SystemExit(
                f'FAIL: {name} with no directory must say NOT CHECKED and name '
                f'the flag; its note reads {row["note"]!r}')
        if row['missing'] != row['want'] or row['ok']:
            raise SystemExit(
                f'FAIL: {name} with no directory must count every cell '
                f'outstanding, not {row["ok"]} of {row["want"]}')
        if expected is not None and row['want'] != expected:
            raise SystemExit(
                f'FAIL: {name} with no directory expects {row["want"]} cells '
                f'against the {expected} it expects with one')
        n += 1
    return n


if __name__ == '__main__':
    sys.exit(main())
