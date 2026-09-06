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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

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

    # Censoring is a pair subset, named outright rather than crossed.
    censoring = _pairs_file('censoring_pairs.json')
    if censoring:
        for model, rep in censoring.get('generator_pairs', []):
            want.add(('censoring', C.canonical_rep(rep),
                      C.canonical_model(model, 'qm9')))
    return want, replicates


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


def check_assay(directories):
    gen = _generator('val', 'slurm_scripts_validation_rerun/generate_scripts.py')
    if gen is None or not directories:
        return None
    datasets = [short for short, _ in gen.DATASETS]
    want = {(d, C.canonical_model(m, 'validation'), C.canonical_rep(r), c)
            for d in datasets for m in gen.MODELS_ALL
            for r in gen.ALL_REPS for c in gen.BREADTH_GRID}
    frame = L.load_assay_accuracy(directories)
    if frame is None:
        return {'name': 'assay accuracy', 'want': len(want), 'ok': 0,
                'missing': len(want), 'partial': 0, 'thin': 0,
                'examples': sorted(want)[:6],
                'note': 'nothing found -- these land in the KIRBy checkout, '
                        'not this one'}
    cover = L.coverage(frame, 'assay')
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
            'note': 'five folds, no replicates -- a partition, not repeats',
            'coverage': cover}


def check_uncertainty(directories):
    gen = _generator('unc',
                     'slurm_scripts_uncertainty_rerun/generate_scripts.py')
    if gen is None or not directories:
        return None
    want = {(d, m, r, c) for d in gen.DATASETS for m in gen.MODELS
            for r in gen.REPS for c in gen.MAIN_GRID_CONDITIONS}
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
            'examples': [], 'note': "the merge step's own coverage.csv",
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
                        ('dataset', 'model', 'rep', 'condition')]
                print(f'    {row.status:18s} {" ".join(b for b in bits if b)}')
    print()
    if complete:
        print('LANDED. Everything the generators asked for is on disk.')
    else:
        print('STILL LANDING. Re-run this when more arrives; exit code is 1 '
              'until it is all there.')
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
               check_assay(args.validation_dir),
               check_uncertainty(args.uncertainty_dir)]
    return report(results, verbose=args.verbose)


if __name__ == '__main__':
    sys.exit(main())
