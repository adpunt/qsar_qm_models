#!/usr/bin/env python3
"""Watch everything this laptop is running and write a status anyone can read.

    nohup python3 -u scripts/watch_local_runs.py &

Writes results/tuning_local/STATUS.md every ten minutes and stops when nothing
is left running.

WHY IT LOOKS LIKE THIS
----------------------
Three traps, all hit before. A pattern that begins with a dash is read by pgrep
as its own option and silently matches nothing. A watcher whose own command line
contains the pattern it searches for counts itself, so it never sees the work
finish. And a live process is not a working one: a job can sit for an hour with
its process alive and nothing being written, so throughput is counted, not just
liveness.
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT = os.path.join(_ROOT, 'results', 'tuning_local')
STATUS = os.path.join(OUT, 'STATUS.md')

# name -> (file it writes, how many rows when finished, how it is described)
BNN = ['dnn_bnn_full', 'dnn_bnn_full_variational',
       'mlp_bnn_full', 'mlp_bnn_full_variational']
SIZE = {'herg': 1415, 'caco2': 2161, 'logd': 3000}


def sweep_cmd(dataset, size, rep, tag):
    """The clean search: one representation, all four models."""
    return [sys.executable, '-u', 'scripts/tune_hyperparameters.py', '--sweep',
            '--settings', '12', '--dataset', dataset, '--sample-size', str(size),
            '--screen-at', '800', '--promote', '1', '--models'] + BNN + \
           ['--reps', rep, '--tag', tag]


def pick_cmd(dataset, models, tag, reps):
    """The cross-representation test at noise 0.5 — what picks one setting
    per model."""
    return [sys.executable, '-u', 'scripts/pick_per_model_setting.py',
            '--dataset', dataset, '--sample-size', str(SIZE.get(dataset, 3000)),
            '--models'] + models + ['--reps'] + reps + \
           ['--seeds', '1', '--tag', tag]


# label, file, rows when finished, process pattern, how to restart it
JOBS = [
    # the clean search, still filling in the last representations
    ('search hERG', 'trials_bnnherg.csv', 312, 'tag bnnherg', None),
    ('search Caco-2', 'trials_bnncaco2.csv', 312, 'tag bnncaco2', None),
] + [
    (f'search LogD {r}', f'trials_bnnlogd_{r}.csv', 56, f'tag bnnlogd_{r}',
     sweep_cmd('logd', 3000, r, f'bnnlogd_{r}'))
    for r in ('pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns')
] + [
    # the cross-representation test. QM9 on all six; the three lab datasets on
    # PDV and ChemBERTa only, which is what the author asked for.
    ('QM9 pick, variational', 'per_model_varia.csv', 72, 'tag varia', None),
    ('QM9 pick, Bayesian a', 'per_model_bayes_a.csv', 42, 'tag bayes_a',
     pick_cmd('qm9', ['dnn_bnn_full'], 'bayes_a',
              ['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns'])),
    ('QM9 pick, Bayesian b', 'per_model_bayes_b.csv', 42, 'tag bayes_b',
     pick_cmd('qm9', ['mlp_bnn_full'], 'bayes_b',
              ['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns'])),
] + [
    (f'{ds} pick, 4 models', f'per_model_x_{ds}.csv', 56, f'tag x_{ds}',
     pick_cmd(ds, BNN, f'x_{ds}', ['pdv', 'chemberta']))
    for ds in ('herg', 'caco2', 'logd')
]

PATTERNS = ['scripts/tuned_under_noise', 'tune_hyperparameters.py --sweep',
            'pick_per_model_setting']


def running(pattern):
    """How many live processes match. The -- matters: without it a pattern
    starting with a dash is read as pgrep's own option."""
    r = subprocess.run(['pgrep', '-f', '--', pattern],
                       capture_output=True, text=True)
    me = str(os.getpid())
    pids = [p for p in r.stdout.split() if p != me]
    return len(pids)


def rows(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        return 0, None
    with open(path) as fh:
        n = sum(1 for _ in csv.reader(fh)) - 1
    return max(0, n), os.path.getmtime(path)


def main():
    previous, restarts = {}, {}
    while True:
        now = datetime.now()
        lines = [f'# what this laptop is running, {now:%Y-%m-%d %H:%M}', '']
        body, alive_any = [], False
        for label, fname, total, pattern, cmd in JOBS:
            n, mtime = rows(fname)
            moved = n - previous.get(fname, n)
            previous[fname] = n
            live = running(pattern) > 0
            done = n >= total

            if done:
                state = 'finished'
            elif live:
                state = 'running'
                alive_any = True
            else:
                # NOT FINISHED AND NOT ALIVE IS DEAD. Reported as 'never wrote'
                # before, which is what let a job killed by the KeOps import
                # race sit unnoticed for an hour on 2026-09-01.
                state = '**DEAD**'
                if cmd and restarts.get(fname, 0) < 3:
                    restarts[fname] = restarts.get(fname, 0) + 1
                    log = open(os.path.join(OUT, f'restart_{fname}.log'), 'a')
                    subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=_ROOT,
                                     env=dict(os.environ, OMP_NUM_THREADS='1',
                                              KMP_DUPLICATE_LIB_OK='TRUE'))
                    state = f'**DEAD — restarted (attempt {restarts[fname]})**'
                    alive_any = True
                elif cmd:
                    state = '**DEAD — 3 restarts failed, needs a person**'

            age = ('never' if mtime is None
                   else f'{(time.time() - mtime) / 60:.0f} min ago')
            body.append(f'| {label} | {n} | {total} | +{moved} | {age} | {state} |')

        lines.append('| job | done | of | moved | last wrote | state |')
        lines.append('|---|---|---|---|---|---|')
        lines += body
        lines.append('')
        if not alive_any:
            lines.append('**Everything has finished.**')
        with open(STATUS, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
        if not alive_any:
            return 0
        time.sleep(600)


if __name__ == '__main__':
    sys.exit(main())
