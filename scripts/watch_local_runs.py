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
JOBS = [
    ('noise test, ChemBERTa', 'noise_vs_tuned_cb.csv', 168,
     'tuned against default across seven noise levels'),
    ('noise test, LightGBM',  'noise_vs_tuned_lgb.csv', 56,
     'does LightGBM improve under noise on four more representations'),
    ('per-model pick at noise', 'per_model_varia.csv', 252,
     'one setting per model, chosen at noise 0.5 over three seeds'),
    ('Bayesian search, hERG',   'trials_bnnherg.csv',  312, '24 pairings'),
    ('Bayesian search, Caco-2', 'trials_bnncaco2.csv', 312, '24 pairings, screened'),
    ('Bayesian search, LogD',   'trials_bnnlogd.csv',  312, '24 pairings, screened'),
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
    previous = {}
    while True:
        live = sum(running(p) for p in PATTERNS)
        now = datetime.now()
        lines = [f'# what this laptop is running, {now:%Y-%m-%d %H:%M}', '',
                 f'{live} job(s) alive.', '']
        lines.append('| job | done | of | moved since last check | last wrote |')
        lines.append('|---|---|---|---|---|')
        for label, fname, total, _why in JOBS:
            n, mtime = rows(fname)
            moved = n - previous.get(fname, n)
            previous[fname] = n
            if mtime is None:
                age = 'never'
            else:
                mins = (time.time() - mtime) / 60
                age = f'{mins:.0f} min ago'
            done = 'finished' if n >= total else f'{n}'
            lines.append(f'| {label} | {done} | {total} | +{moved} | {age} |')
        lines.append('')
        if live == 0:
            lines.append('**Everything has finished.**')
        with open(STATUS, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
        if live == 0:
            return 0
        time.sleep(600)


if __name__ == '__main__':
    sys.exit(main())
