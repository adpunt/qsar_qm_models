#!/usr/bin/env python
"""The laboratory merge finds the results wherever the jobs put them, and drops
pre-redesign rows rather than carrying them forward.

WHY THIS EXISTS. Two defects, both found on 2026-09-05 and both silent:

  1. `merge_results.py` held bare relative paths and its docstring said to run it from
     `<KIRBy>/tests`. The generated jobs `cd "$KIRBY_DIR"/tests` and pass
     `--results-root "../results/validation_rerun/..."`, so the new grid lands one
     directory ABOVE that, while the archive it merges into is below it. From `tests/`
     the new grid was invisible and the script printed "No rerun data found" and exited
     0; from the checkout root the archive was invisible and it would have written a
     fresh file with no history in it.

  2. The archive holds rows from the noise scheme that was replaced. Nothing dropped
     them, so they would have been carried forward into the file the figures read.

Everything below is built in a temporary directory. No cluster, no real results.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
MERGE = HERE / 'merge_results.py'
SETTLED = json.loads((HERE.parent / 'noise_conditions.json').read_text())
SETTLED_NAMES = [c['name'] for g in ('stage_1_full_grid', 'stage_2_depth_only')
                 for c in SETTLED.get(g, [])]
RETIRED = [c['name'] for c in SETTLED.get('not_run', [])] or ['skewed_draw']


def rows(model, rep, condition, n=4, dataset='logd'):
    return pd.DataFrame([
        dict(dataset=dataset, model=model, rep=rep, noise_type=condition,
             sigma=s, fold=0, r2=0.7 - 0.1 * s, rmse=1.0, mae=0.8)
        for s in [0.0, 0.2, 0.5, 1.0][:n]])


def build(kirby: Path, where: str):
    """One re-run cell for logD, written where `where` says, plus an archive."""
    d = kirby / where / 'rf_pdv_logd' / 'logd'
    d.mkdir(parents=True)
    rows('RF', 'PDV', 'gaussian').to_csv(d / 'all_results.csv', index=False)

    arch = kirby / 'tests' / 'results' / 'validation' / 'logd'
    arch.mkdir(parents=True)
    keep = rows('SVM', 'ECFP4', SETTLED_NAMES[0])
    drop = rows('SVM', 'ECFP4', RETIRED[0])
    pd.concat([keep, drop]).to_csv(arch / 'all_results.csv', index=False)
    return arch / 'all_results.csv'


def run(kirby, *extra):
    return subprocess.run(
        [sys.executable, str(MERGE), '--kirby-dir', str(kirby), *extra],
        capture_output=True, text=True)


def main():
    failures = []

    # 1. Found in EITHER place the jobs may have written.
    for where in ('results/validation_rerun', 'tests/results/validation_rerun'):
        with tempfile.TemporaryDirectory() as tmp:
            kirby = Path(tmp) / 'KIRBy'
            archive = build(kirby, where)
            proc = run(kirby, '--dry-run')
            if proc.returncode != 0:
                failures.append(f'{where}: exited {proc.returncode}\n'
                                f'      {proc.stdout[-400:]}{proc.stderr[-400:]}')
            # Only logD has a fixture, so caco2 and hERG say "No rerun data found"
            # legitimately. Read the logD section alone.
            elif 'No rerun data found' in proc.stdout.split('=== logd ===')[-1].split('===')[0]:
                failures.append(f'{where}: the merge did not find the re-run output '
                                f'that the jobs write there -- this is the defect, and '
                                f'it exits 0 while doing it')
            if '--dry-run' and 'Would write' not in proc.stdout and proc.returncode == 0:
                failures.append(f'{where}: --dry-run said nothing about what it would '
                                f'write')
            if archive.read_text().count(RETIRED[0]) == 0:
                failures.append(f'{where}: the fixture is wrong -- no retired rows')

    # 2. Retired conditions are dropped, and only on request kept.
    with tempfile.TemporaryDirectory() as tmp:
        kirby = Path(tmp) / 'KIRBy'
        archive = build(kirby, 'results/validation_rerun')
        proc = run(kirby)
        if proc.returncode != 0:
            failures.append(f'the real merge exited {proc.returncode}\n{proc.stderr[-400:]}')
        else:
            out = pd.read_csv(archive)
            if RETIRED[0] in set(out['noise_type']):
                failures.append(
                    f'{RETIRED[0]} survived the merge: pre-redesign rows are being '
                    f'carried into the file the figures are built from')
            if SETTLED_NAMES[0] not in set(out['noise_type']):
                failures.append('the settled archive rows were dropped too -- the '
                                'filter is removing more than the retired scheme')
            if 'gaussian' not in set(out['noise_type']):
                failures.append('the re-run rows did not reach the merged file')

    with tempfile.TemporaryDirectory() as tmp:
        kirby = Path(tmp) / 'KIRBy'
        archive = build(kirby, 'results/validation_rerun')
        proc = run(kirby, '--allow-retired-conditions')
        if proc.returncode == 0:
            out = pd.read_csv(archive)
            if RETIRED[0] not in set(out['noise_type']):
                failures.append('--allow-retired-conditions dropped them anyway, so '
                                'there is no way to get the old behaviour back')

    # 3. Nothing anywhere: it must say so and exit non-zero, not write an empty file.
    with tempfile.TemporaryDirectory() as tmp:
        kirby = Path(tmp) / 'KIRBy'
        kirby.mkdir(parents=True)
        proc = run(kirby)
        if proc.returncode == 0:
            failures.append('with no re-run output anywhere the merge exited 0; a '
                            'missing grid must not read as a finished one')

    if failures:
        print(f'FAIL — {len(failures)} problem(s):\n')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('PASS — the merge finds the re-run under either path, drops retired-scheme '
          'archive rows, keeps them on request, and refuses an empty run.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
