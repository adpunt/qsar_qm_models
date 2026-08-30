#!/usr/bin/env python3
"""Watch the tuning runs, and restart anything that dies before it finishes.

    nohup python3 -u scripts/watch_tuning_runs.py --log-dir <dir> &

WHY
---
Runs have died silently three times: an import collision killed three at once, a
duplicate process wrote to one file, and one restart never fired because of a
shell quoting mistake. Each time it was found by chance, hours later. Nothing was
watching.

This checks every interval: for each job it knows about, is the process alive,
and is its work complete? A job that is neither is restarted, and the restart is
recorded. It also refuses to start a second copy of a job that is already
running, which is what caused the duplicate.

It is deliberately dumb. It does not decide anything about the science; it only
notices that something that should be running is not.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')
STATUS = os.path.join(OUT_DIR, 'STATUS.md')
sys.path.insert(0, os.path.join(_ROOT, 'models'))

FITS_PER_PAIRING = 13   # the default plus twelve settings

# tag -> (the arguments that job runs with, the pairings it must complete)
JOBS = {}


def build_jobs():
    import tuning_rosters as R

    six = ['rf', 'xgboost', 'lgb', 'svm', 'dnn', 'mlp']
    reps = R.ALL_REPS

    def add(tag, args, models, only_reps=None):
        JOBS[tag] = {
            'args': args,
            'pairings': [(m, r) for m in models
                         for r in (only_reps or R.MODELS[m][4])],
        }

    # QM9, the parts still outstanding
    add('mlpbnn', ['--sample-size', '5000', '--screen-at', '800', '--promote', '4',
                   '--models', 'mlp_bnn_full'], ['mlp_bnn_full'])
    add('mlpvbll', ['--sample-size', '5000', '--screen-at', '800', '--promote', '4',
                    '--models', 'mlp_bnn_full_variational'],
        ['mlp_bnn_full_variational'])
    # PROMOTES TWO, NOT FOUR. Screening ranks settings by score and ignores what
    # they will cost at full size, so a 1024-wide Bayesian network -- cheap on
    # 800 molecules, hours on 4,000, because every weight is resampled on every
    # batch -- was promoted and then produced NO full-size fit in three and a half
    # hours. The top two almost always contain the winner; four was paying for
    # runners-up at the price of the whole job.
    add('dnnrest', ['--sample-size', '5000', '--screen-at', '800', '--promote', '2',
                    '--models', 'dnn_bnn_full', 'dnn_bnn_full_variational',
                    '--reps', 'sns', 'avalon', 'chemberta'],
        ['dnn_bnn_full', 'dnn_bnn_full_variational'], ['sns', 'avalon', 'chemberta'])
    # The Gaussian process is NOT here. Tuning was closed on 2026-08-30: four
    # settings and the default landed within 0.0005 of each other on all three
    # representations at full size, at 30-60 minutes a fit. Nothing to find.
    add('qrf', ['--sample-size', '5000', '--models', 'qrf'], ['qrf'])
    # the validation datasets
    add('caco22', ['--dataset', 'caco2', '--sample-size', '2161',
                   '--models'] + six, six)
    add('logd2', ['--dataset', 'logd', '--sample-size', '5039',
                  '--models'] + six, six)


def alive(tag):
    """Is a run with this tag live? The pattern must not begin with a dash --
    pgrep reads that as its own option and silently matches nothing, which is
    how three live files got overwritten."""
    out = subprocess.run(['pgrep', '-f', '--', f'tag {tag}'],
                         capture_output=True, text=True)
    if out.returncode not in (0, 1):
        raise RuntimeError(f'pgrep failed: {out.stderr.strip()}')
    return bool(out.stdout.strip())


def progress(tag, pairings):
    """How many of this job's pairings have all their fits."""
    counts = Counter()
    for path in glob.glob(os.path.join(OUT_DIR, f'trials_{tag}*.csv')):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                counts[(r['model'], r['rep'])] += 1
    done = sum(1 for p in pairings if counts.get(p, 0) >= FITS_PER_PAIRING)
    return done, len(pairings)


def slowest_fit(tag):
    """The longest single fit this job has recorded, in seconds.

    A fixed stall threshold does not work here. A screened job writes twelve
    cheap fits in a few minutes and then goes quiet for an hour while it refits
    the survivors at full size -- which looks identical to being stuck. The only
    honest reference is the job's own slowest fit so far.
    """
    worst = 0.0
    for path in glob.glob(os.path.join(OUT_DIR, f'trials_{tag}*.csv')):
        with open(path) as fh:
            for r in csv.DictReader(fh):
                try:
                    worst = max(worst, float(r['seconds']))
                except (TypeError, ValueError):
                    pass
    return worst


def fits_done(tag):
    """Total scored fits for this job, however many pairings they belong to."""
    n = 0
    for path in glob.glob(os.path.join(OUT_DIR, f'trials_{tag}*.csv')):
        with open(path) as fh:
            n += sum(1 for _ in csv.DictReader(fh))
    return n


def newest_row_age(tag):
    """Seconds since this job last wrote a result, or None if it never has."""
    newest = None
    for path in glob.glob(os.path.join(OUT_DIR, f'trials_{tag}*.csv')):
        m = os.path.getmtime(path)
        newest = m if newest is None else max(newest, m)
    return None if newest is None else time.time() - newest


def start(tag, args, log_dir):
    log = os.path.join(log_dir, f'run_{tag}.log')
    cmd = [sys.executable, '-u',
           os.path.join(_HERE, 'tune_hyperparameters.py'),
           '--sweep', '--settings', '12'] + args + ['--tag', tag]
    env = dict(os.environ, OMP_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE')
    with open(log, 'a') as fh:
        fh.write(f'\n=== restarted by the watcher at {datetime.now()} ===\n')
        fh.flush()
        subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                         cwd=_ROOT, start_new_session=True)
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-dir', required=True)
    ap.add_argument('--interval', type=int, default=600)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--stall-after', type=int, default=5400,
                    help='Seconds without a new result before a live run counts '
                         'as stalled (default 90 minutes; the slowest single fit '
                         'measured here was about 55).')
    args = ap.parse_args()
    stall_after = args.stall_after

    build_jobs()
    previous, stuck = {}, {}
    while True:
        stamp = datetime.now().strftime('%H:%M:%S')
        restarted = []
        lines = [f'# tuning status at {datetime.now():%Y-%m-%d %H:%M}', '']
        for tag, job in JOBS.items():
            done, total = progress(tag, job['pairings'])
            live = alive(tag)
            state = 'running' if live else ('complete' if done >= total else 'DEAD')

            # A LIVE PROCESS IS NOT THE SAME AS A WORKING ONE, and completed
            # PAIRINGS are too coarse to tell -- a job can be thirteen fits into
            # a pairing and still read as zero. Count the fits and watch whether
            # the number moves. That is throughput, and it is the only thing that
            # distinguishes slow from stuck.
            fits = fits_done(tag)
            moved = fits - previous.get(tag, 0)
            previous[tag] = fits
            if state == 'running':
                stuck[tag] = stuck.get(tag, 0) + 1 if moved == 0 else 0
                age = newest_row_age(tag)
                # Judged against this job's OWN slowest fit, with room for the
                # machine being shared, and never less than the flat floor.
                limit = max(stall_after, 4 * slowest_fit(tag))
                if stuck[tag] >= 2 and age and age > limit:
                    state = (f'NOT ADVANCING — {age/60:.0f} min with no new fit, '
                             f'against a slowest-ever fit of '
                             f'{slowest_fit(tag)/60:.0f} min')
                elif stuck[tag] >= 2:
                    state = (f'slow — {age/60:.0f} min since the last fit, but '
                             f'its slowest is {slowest_fit(tag)/60:.0f} min, so '
                             f'this is within range')
            else:
                stuck[tag] = 0

            print(f'{stamp}  {tag:10s} {done:2d}/{total:2d} pairings  '
                  f'{fits:4d} fits (+{moved})   {state}', flush=True)
            lines.append(f'- **{tag}**: {done}/{total} pairings, {fits} fits '
                         f'(+{moved} since the last check) — {state}')
            if state == 'DEAD':
                # Spacing the restarts apart: several interpreters importing the
                # same stack at once collide on a fixed temporary path and all
                # but one dies.
                time.sleep(45)
                start(tag, job['args'], args.log_dir)
                restarted.append(tag)
        if restarted:
            lines.append('')
            lines.append(f'RESTARTED this round: {", ".join(restarted)}')
            print(f'{stamp}  RESTARTED: {", ".join(restarted)}', flush=True)
        with open(STATUS, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        if args.once or all(progress(t, j['pairings'])[0] >= len(j['pairings'])
                            for t, j in JOBS.items()):
            print(f'{stamp}  every job complete' if not args.once else '',
                  flush=True)
            if not args.once:
                break
            break
        time.sleep(args.interval)
    return 0


if __name__ == '__main__':
    sys.exit(main())
