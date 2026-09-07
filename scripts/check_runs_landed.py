#!/usr/bin/env python
"""Has the re-run landed? One command, all three producers.

    python scripts/check_runs_landed.py --stage 1

Exit code 0 means everything the generators asked for is on disk. Exit 1 means
it is still landing. So this can be waited on rather than watched:

    until python scripts/check_runs_landed.py --stage 1; do sleep 900; done

WHY IT READS THE ROSTERS RATHER THAN HOLDING THEM
-------------------------------------------------
The old completeness check globbed for six noise names that had been retired,
so it reported a complete grid for conditions nothing was running. Every roster
here is imported from the generator that queued the jobs -- models from each
generator's own MODELS, representations from its own list, conditions from
`noise_conditions.json`, levels from the one place they live. Nothing is typed
twice, so nothing can drift.

WHAT COUNTS AS LANDED
---------------------
Not "the file exists". A task can write a file and die at level three of seven,
and squeue will show it finished. A cell is landed when it has every level on
its ladder and every replicate that was asked for. The three states are counted
separately, because they need different actions: MISSING means resubmit that
index, PARTIAL means the task died part-way, THIN means it ran but the replicate
count is short of what the variance decomposition needs.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paper_analysis import _preflight  # noqa: E402

_preflight()

import pandas as pd  # noqa: E402

import figlib_config as C  # noqa: E402
import figlib_load as L  # noqa: E402

ROOT = C.ROOT


def _generator(tag, relative):
    """Import a job generator without running it."""
    path = ROOT / relative
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(f'_gen_{tag}', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[f'_gen_{tag}'] = module
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    except Exception as exc:  # pragma: no cover
        print(f'  could not read {relative}: {type(exc).__name__} {exc}')
        return None
    return module


def _pairs_file(name):
    path = ROOT / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


# ---------------------------------------------------------------------------

def qm9_expected(stage):
    """(condition, rep, model) for the stage, plus censoring on its named pairs."""
    gen = _generator('qm9', 'slurm_scripts_qm9_rerun/generate_scripts.py')
    if gen is None:
        return set(), 0
    conditions = list(gen.STAGE_DEFAULTS[stage]['conditions'])
    replicates = int(gen.STAGE_DEFAULTS[stage]['replicates'])
    # The generator's model keys are what the PIPELINE writes; the loader
    # canonicalises everything it reads. Comparing the two spellings directly
    # reports every cell missing while every cell is present.
    want = {(c, C.canonical_rep(r), C.canonical_model(m, 'qm9'))
            for m, entry in gen.MODELS.items()
            for r in entry[4]
            for c in conditions}

    # THE DEEP RUN IS A SELECTION, NOT THE WHOLE ROSTER. Every deep-run task reads
    # deep_run_pairs.json when it STARTS, and a task whose model or representation is
    # not listed exits 0 having fitted nothing -- that is the design, not a failure.
    # Crossing the generator's full MODELS table with every representation therefore
    # counts several hundred deliberate skips as missing cells and buries the handful
    # of real gaps. Six models on three representations do the work; the rest skip.
    if stage == 2:
        sel = _pairs_file('deep_run_pairs.json')
        if sel and sel.get('generator_labels') and sel.get('representations'):
            ok_m = {C.canonical_model(m, 'qm9')
                    for m in sel['generator_labels']}
            ok_r = {C.canonical_rep(r) for r in sel['representations']}
            want = {(c, r, m) for (c, r, m) in want if m in ok_m and r in ok_r}

    # Censoring is a pair subset, named outright rather than crossed.
    censoring = _pairs_file('censoring_pairs.json')
    if censoring:
        for model, rep in censoring.get('generator_pairs', []):
            want.add(('censoring', C.canonical_rep(rep),
                      C.canonical_model(model, 'qm9')))
    return want, replicates


def _only_expected(cover, want, keys):
    """Drop coverage rows for combinations nobody asked for.

    WHY: `want` is restricted -- the deep run is a selection of six models on
    three representations, and everything else on disk is the screen, whose gaps
    are not this stage's problem. The counts of PARTIAL and THIN were taken over
    the WHOLE coverage frame while `landed` and `missing` were taken against
    `want`, so one line of output mixed two different questions. On 2026-09-07 a
    `--stage 2` run reported 59 landed of 113 expected beside 36 partial and 77
    thin, and the thin list named Sort & Slice combinations that no deep-run pair
    contains. Restricting here fixes the counts and the --verbose listing at once,
    because both read this frame.
    """
    if cover is None or len(cover) == 0:
        return cover
    keep = [tuple(getattr(row, k, '') for k in keys) in want
            for row in cover.itertuples()]
    return cover[keep]


def check_qm9(directory, stage):
    want, replicates = qm9_expected(stage)
    if not want:
        return None
    frame = L.load_qm9(directory)
    if frame is None:
        return {'name': 'QM9', 'want': len(want), 'ok': 0, 'missing': len(want),
                'partial': 0, 'thin': 0, 'examples': sorted(want)[:6],
                'note': f'nothing in {directory}'}
    cover = L.coverage(frame, 'QM9')
    cover = _only_expected(cover, want, ('condition', 'rep', 'model'))
    have = {(row.condition, row.rep, row.model) for row in cover.itertuples()}
    landed = {(row.condition, row.rep, row.model) for row in cover.itertuples()
              if row.status == 'OK'}
    partial = cover[cover['status'].isin(['PARTIAL_LEVELS',
                                          'NO_CLEAN_BASELINE'])]
    thin = cover[(cover['status'] == 'THIN_REPLICATES')
                 | (cover['replicates_max'] < replicates)]
    missing = sorted(want - have)
    return {'name': 'QM9', 'want': len(want), 'ok': len(want & landed),
            'missing': len(missing), 'partial': int(len(partial)),
            'thin': int(len(thin)), 'examples': missing[:6],
            'note': f'{replicates} replicate(s) expected at stage {stage}',
            'coverage': cover}


def qm9_oof_expected(stage):
    """(condition, rep, model) for the cells that run the OUT-OF-FOLD pass.

    Generator labels, not canonical spellings, because these are read straight
    off the result FILE NAME, which the generator writes as
    `anova_<condition>_<representation>_<model>.csv`.

    Only the settled pairs run the pass -- `gen.UNCERTAINTY_PAIRS`, six models on
    three representations -- so this is a small subset of `qm9_expected`.
    """
    gen = _generator('qm9', 'slurm_scripts_qm9_rerun/generate_scripts.py')
    if gen is None:
        return set()
    conditions = list(gen.STAGE_DEFAULTS[stage]['conditions'])
    want = {(c, r, m)
            for m, entry in gen.MODELS.items()
            for r in entry[4]
            for c in conditions
            if r in (gen.UNCERTAINTY_PAIRS.get(m) or [])}
    if stage == 2:
        sel = _pairs_file('deep_run_pairs.json')
        if sel and sel.get('generator_labels') and sel.get('representations'):
            ok_m, ok_r = set(sel['generator_labels']), set(sel['representations'])
            want = {t for t in want if t[2] in ok_m and t[1] in ok_r}
    censoring = _pairs_file('censoring_pairs.json')
    if censoring:
        for model, rep in censoring.get('generator_pairs', []):
            if rep in (gen.UNCERTAINTY_PAIRS.get(model) or []):
                want.add(('censoring', rep, model))
    return want


def check_qm9_oof(directory, stage):
    """Do the settled pairs actually HAVE their out-of-fold training rows?

    WHY THIS IS A SEPARATE CHECK AND NOT PART OF check_qm9, ADDED 2026-09-07.

    `load_qm9` globs `anova_*.csv` and excludes every sibling suffix, so the
    accuracy file is the only thing `check_qm9` has ever counted. The two are not
    written together: the accuracy row for a (level, replicate) is saved BEFORE
    the out-of-fold pass runs, and the pass can then fail on its own while the
    row it already wrote stands.

    That is not hypothetical. All nine `gauche_rbf` tasks of `12980590` failed
    inside the out-of-fold pass at every level and replicate, and `--stage 1`
    reported not one of their cells MISSING or PARTIAL, because every accuracy row
    was there (RERUN_PLAN.md 13.27 D2). The per-molecule uncertainty is the whole
    reason those pairs are settled, and nothing was checking for it.

    A cell is landed here when every level on its ladder, at every replicate the
    stage asked for, has `train_oof` rows.
    """
    gen = _generator('qm9', 'slurm_scripts_qm9_rerun/generate_scripts.py')
    want = qm9_oof_expected(stage)
    if gen is None or not want:
        return None
    replicates = int(gen.STAGE_DEFAULTS[stage]['replicates'])
    directory = Path(directory)

    rows, ok = [], 0
    for condition, rep, model in sorted(want):
        path = directory / f'anova_{condition}_{rep}_{model}_uncertainty_values.csv'
        levels = len(str(gen.CONDITIONS[condition][1]).split())
        expected_cells = levels * replicates
        if not path.exists():
            status, seen = 'MISSING_FILE', 0
        else:
            try:
                frame = pd.read_csv(path, usecols=['split', 'sigma', 'iteration'])
            except Exception as exc:
                rows.append(('UNREADABLE', condition, rep, model, str(exc)[:60]))
                continue
            oof = frame[frame['split'] == 'train_oof']
            seen = len(oof.drop_duplicates(['sigma', 'iteration']))
            if seen == 0:
                status = 'NO_OOF_ROWS'
            elif seen < expected_cells:
                status = 'PARTIAL_OOF'
            else:
                status = 'OK'
                ok += 1
        if status != 'OK':
            rows.append((status, condition, rep, model,
                         f'{seen} of {expected_cells} (level, replicate) cells '
                         f'have train_oof rows'))

    cover = pd.DataFrame(rows, columns=['status', 'condition', 'rep', 'model',
                                        'detail'])
    return {'name': 'QM9 out-of-fold', 'want': len(want), 'ok': ok,
            'missing': int((cover['status'] == 'MISSING_FILE').sum())
                       + int((cover['status'] == 'NO_OOF_ROWS').sum()),
            'partial': int((cover['status'] == 'PARTIAL_OOF').sum()),
            'thin': int((cover['status'] == 'UNREADABLE').sum()),
            'examples': [],
            'note': (f'the settled pairs only (uncertainty_pairs.json); '
                     f'{replicates} replicate(s) expected at stage {stage}'),
            'coverage': cover}


def assay_expected(stage):
    """(dataset, model, rep, condition) for one stage of the laboratory runs.

    STAGE 1 is the breadth grid: every model on every representation, on the
    three conditions that are not a pair subset.

    STAGE 2 is the two runs that went out on 2026-09-06 and that nothing has
    ever checked. `check_assay` took no stage at all, so `--stage 2` reported
    the breadth grid a second time and the laboratory depth run and censoring
    were counted nowhere -- neither present nor missing, which is the state the
    coverage report exists to make impossible. Both are run-time selections:
    the depth run is `deep_run_pairs.json`'s models crossed with its
    representations, censoring is `censoring_pairs.json`'s five named pairs, and
    every task outside them skips by design.
    """
    gen = _generator('val', 'slurm_scripts_validation_rerun/generate_scripts.py')
    if gen is None:
        return set(), None
    datasets = [short for short, _ in gen.DATASETS]
    if stage != 2:
        want = {(d, C.canonical_model(m, 'validation'), C.canonical_rep(r), c)
                for d in datasets for m in gen.MODELS_ALL
                for r in gen.ALL_REPS for c in gen.BREADTH_GRID}
        return want, 'the breadth grid: every model on every representation'

    want = set()
    deep = _pairs_file('deep_run_pairs.json')
    if deep and deep.get('validation_labels') and deep.get('representations'):
        want |= {(d, C.canonical_model(m, 'validation'), C.canonical_rep(r), c)
                 for d in datasets
                 for m in deep['validation_labels']
                 for r in deep['representations']
                 for c in gen.DEPTH_ONLY}
    censoring = _pairs_file('censoring_pairs.json')
    if censoring:
        for model, rep in censoring.get('validation_pairs', []):
            for d in datasets:
                want.add((d, C.canonical_model(model, 'validation'),
                          C.canonical_rep(rep), 'censoring'))
    note = (f'{len(gen.DEPTH_ONLY)} depth condition(s) on the deep run pairs, '
            f'censoring on its own named pairs')
    return want, note


def check_assay(directories, stage=1):
    gen = _generator('val', 'slurm_scripts_validation_rerun/generate_scripts.py')
    if gen is None or not directories:
        return None
    want, note = assay_expected(stage)
    if not want:
        return None
    frame = L.load_assay_accuracy(directories)
    if frame is None:
        return {'name': 'assay accuracy', 'want': len(want), 'ok': 0,
                'missing': len(want), 'partial': 0, 'thin': 0,
                'examples': sorted(want)[:6],
                'note': 'nothing found -- these land in the KIRBy checkout, '
                        'not this one'}
    cover = L.coverage(frame, 'assay')
    cover = _only_expected(cover, want,
                           ('dataset', 'model', 'rep', 'condition'))
    have = {(row.dataset, row.model, row.rep, row.condition)
            for row in cover.itertuples()}
    landed = {(row.dataset, row.model, row.rep, row.condition)
              for row in cover.itertuples() if row.status == 'OK'}
    missing = sorted(want - have)
    partial = cover[cover['status'].isin(['PARTIAL_LEVELS',
                                          'NO_CLEAN_BASELINE'])]
    return {'name': 'assay accuracy', 'want': len(want),
            'ok': len(want & landed), 'missing': len(missing),
            'partial': int(len(partial)), 'thin': 0, 'examples': missing[:6],
            'note': f'five folds, no replicates -- a partition, not repeats; {note}',
            'coverage': cover}


def check_uncertainty(directories):
    gen = _generator('unc',
                     'slurm_scripts_uncertainty_rerun/generate_scripts.py')
    if gen is None or not directories:
        return None
    # ALL SEVEN CONDITIONS, not the main grid's four. The uncertainty runs went
    # out as two submissions -- the three the QM9 screen runs, then censoring and
    # the three depth-only ones -- so together they cover exactly
    # MAIN_GRID_CONDITIONS + DEEP_RUN_CONDITIONS, which is what the generator
    # calls KNOWN_CONDITIONS. Reading MAIN_GRID_CONDITIONS expected 216 cells of
    # the 378 submitted and left the 162 depth-only ones counted nowhere at all:
    # neither landed nor missing. Same failure the merge's own docstring warns
    # about, in the tool that decides whether the run is finished.
    conditions = list(gen.KNOWN_CONDITIONS)
    want = {(d, m, r, c) for d in gen.DATASETS for m in gen.MODELS
            for r in gen.REPS for c in conditions}
    merged = L.load_merged_uncertainty(directories)
    cover = merged.get('coverage')
    if cover is None or not len(cover):
        return {'name': 'uncertainty runs', 'want': len(want), 'ok': 0,
                'missing': len(want), 'partial': 0, 'thin': 0,
                'examples': sorted(want)[:6],
                'note': 'no _merged/coverage.csv -- run the merge step first'}
    status = cover['status'] if 'status' in cover.columns else pd.Series(dtype=str)
    ok = int((status == 'OK').sum())
    return {'name': 'uncertainty runs', 'want': len(want), 'ok': ok,
            'missing': max(0, len(want) - len(cover)),
            'partial': int(status.isin(['TRUNCATED_OOF', 'PARTIAL_FOLDS',
                                        'PARTIAL_LEVELS']).sum()),
            'thin': int(status.isin(['NO_OOF', 'OOF_ALL_NAN']).sum()),
            'examples': [],
            'note': (f"the merge step's own coverage.csv; "
                     f"{len(gen.MODELS)} models x {len(gen.DATASETS)} datasets x "
                     f"{len(gen.REPS)} representations x {len(conditions)} "
                     f"conditions"),
            'coverage': cover}


def report(results, verbose=False):
    print()
    print(f'{"":22s} {"landed":>14s} {"missing":>8s} {"partial":>8s} '
          f'{"thin":>6s}')
    complete = True
    for r in results:
        if r is None:
            continue
        share = f'{r["ok"]}/{r["want"]}'
        print(f'{r["name"]:22s} {share:>14s} {r["missing"]:>8d} '
              f'{r["partial"]:>8d} {r["thin"]:>6d}   {r["note"]}')
        if r['ok'] < r['want'] or r['partial'] or r['thin']:
            complete = False
        for example in r['examples']:
            print(f'    still missing: {example}')
        if verbose and 'coverage' in r:
            bad = r['coverage'][r['coverage']['status'] != 'OK']
            for row in bad.head(20).itertuples():
                bits = [str(getattr(row, k, '')) for k in
                        ('dataset', 'model', 'rep', 'condition', 'detail')]
                print(f'    {row.status:18s} {" ".join(b for b in bits if b)}')
    print()
    if complete:
        print('LANDED. Everything the generators asked for is on disk.')
    else:
        print('STILL LANDING -- and the analysis does NOT wait for this.')
        print('Run it now on what is there; D0 reports the gaps:')
        print('    python scripts/run_paper_analysis.py --qm9-dir results \\')
        print('        --output-dir results/decisions --permutations 0')
        print('This exits 1 while anything is short, so it can be looped, but '
              'nothing is blocked on it.')
    return 0 if complete else 1


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--qm9-dir', default=str(ROOT / 'results'))
    p.add_argument('--validation-dir', action='append', default=None)
    p.add_argument('--uncertainty-dir', action='append', default=None)
    p.add_argument('--stage', type=int, default=1, choices=[0, 1, 2],
                   help='0 = the screen, 1 = the main grid, 2 = the deep run')
    p.add_argument('--verbose', action='store_true',
                   help='list every incomplete cell, not just a count')
    args = p.parse_args(argv)

    print(f'checking against the generators\' own rosters, stage {args.stage}')
    results = [check_qm9(args.qm9_dir, args.stage),
               check_qm9_oof(args.qm9_dir, args.stage),
               check_assay(args.validation_dir, args.stage),
               check_uncertainty(args.uncertainty_dir)]
    return report(results, verbose=args.verbose)


if __name__ == '__main__':
    sys.exit(main())
