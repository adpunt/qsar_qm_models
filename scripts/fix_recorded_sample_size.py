#!/usr/bin/env python3
"""Correct the molecule count recorded by the three validation-dataset runs.

    python scripts/fix_recorded_sample_size.py --log-dir <where the run logs are>

WHAT WENT WRONG
---------------
`--sample-size` was not passed to the LogD, Caco-2 and hERG runs, because each of
those datasets is used whole rather than subsampled. The code did the right thing
-- it used every molecule -- but the column that records how many molecules were
used took the argument's DEFAULT of 10,000 and wrote that instead.

So the fits are correct and only the label is wrong. It still has to be fixed,
because the analysis refuses to read files that disagree on that column, which is
what stops results from different amounts of data being blended. Three files all
claiming 10,000 would pass that check while describing three different datasets
of 1,415, 4,000-odd and 5,000-odd molecules.

The true count comes from the run's own log, which prints the split it built:

    train 1182  validation 109  test 124

That line is the only record of what was actually used, so it is the source here.

SAFETY
------
A file still being written by a live run is left alone. Rewriting it underneath
the process would either be undone by the next flush or corrupt what is there --
the same class of collision that destroyed the timing results once already.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

SPLIT = re.compile(r'train\s+(\d+)\s+validation\s+(\d+)\s+test\s+(\d+)')


def true_count(log_path):
    """Molecules actually used, from the split the run printed."""
    if not os.path.exists(log_path):
        return None
    with open(log_path, errors='replace') as fh:
        for line in fh:
            m = SPLIT.search(line)
            if m:
                return sum(int(g) for g in m.groups())
    return None


def still_running(dataset):
    """Is a run for this dataset live?

    THIS CHECK FAILED ONCE AND REWROTE THREE LIVE FILES. The pattern began with
    two dashes, so pgrep read it as its own options, matched nothing, and
    reported every live run as finished. `--` ends pgrep's option parsing and the
    pattern no longer leads with a dash; both are needed, and the result is
    asserted below rather than trusted.
    """
    out = subprocess.run(['pgrep', '-f', '--', f'dataset {dataset}'],
                         capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise RuntimeError(
            f'pgrep failed ({out.returncode}): {out.stderr.strip()}. Refusing to '
            f'guess whether a run is live -- that guess is what corrupted the '
            f'files last time.')
    return bool(out.stdout.strip())



RESULT = re.compile(
    r'^\[\d+/\d+\] (\S+) x (\S+).*?(?:BLOCKED LOCALLY|'
    r'(default|screen_\d+|final_\d+|setting_\d+): R2=([+-][\d.]+)\s+([\d.]+)s\s+(\w+))')


def rebuild_from_log(log_path, out_path, n_molecules, seed=42):
    """Rebuild a results file from the run's log.

    The log records every fit as it happens, so it is a complete record even when
    the results file is not -- which is the case whenever a file has been replaced
    underneath a running process. Recovering from the log is how the timing
    results were saved after the same mistake.
    """
    rows = []
    with open(log_path, errors='replace') as fh:
        for line in fh:
            m = RESULT.match(line.rstrip())
            if not m:
                continue
            model, rep, setting, r2, secs, status = m.groups()
            if setting is None:
                rows.append(dict(model=model, rep=rep, setting='blocked', r2='',
                                 seconds='', status='blocked', detail=''))
            else:
                rows.append(dict(model=model, rep=rep, setting=setting, r2=r2,
                                 seconds=secs, status=status, detail=''))
    for r in rows:
        r.update(sample_size=n_molecules, seed=seed, written='rebuilt-from-log')
    fields = ['model', 'rep', 'setting', 'r2', 'seconds', 'status', 'detail',
              'sample_size', 'seed', 'written']
    tmp = out_path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out_path)
    return len(rows)

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--log-dir', required=True,
                    help="Where the runs' logs are, holding the split they built.")
    ap.add_argument('--datasets', nargs='*', default=['herg', 'logd', 'caco2'])
    ap.add_argument('--force', action='store_true',
                    help='Rewrite even a file whose run is still live. Do not.')
    args = ap.parse_args()

    changed = 0
    for ds in args.datasets:
        path = os.path.join(OUT_DIR, f'trials_{ds}.csv')
        if not os.path.exists(path):
            print(f'  {ds}: no results file yet')
            continue
        if still_running(ds) and not args.force:
            print(f'  {ds}: STILL RUNNING — left alone. Rewriting a file the '
                  f'run is appending to would be undone or would corrupt it.')
            continue

        n = true_count(os.path.join(args.log_dir, f'tune_{ds}.log'))
        if n is None:
            print(f'  {ds}: could not find the split in the log, so the true '
                  f'count is unknown. NOT guessing.')
            continue

        with open(path) as fh:
            reader = csv.DictReader(fh)
            fields, rows = reader.fieldnames, list(reader)
        was = {r['sample_size'] for r in rows}
        for r in rows:
            r['sample_size'] = n
        tmp = path + '.tmp'
        with open(tmp, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
        print(f'  {ds}: {len(rows)} rows, sample_size {sorted(was)} -> {n}')
        changed += 1

    print(f'\n{changed} file(s) corrected')
    return 0


if __name__ == '__main__':
    sys.exit(main())
