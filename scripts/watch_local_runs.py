#!/usr/bin/env python3
"""Watch every tuning run on this laptop, restart what dies, refresh the tables.

    nohup python3 -u scripts/watch_local_runs.py > results/tuning_local/watch.log 2>&1 &

Writes results/tuning_local/STATUS.md every five minutes, regenerates
CHOSEN_SETTINGS.md from whatever has landed, and stops when nothing is left.

WHY IT LOOKS LIKE THIS -- five traps, all of them hit
-----------------------------------------------------
1. A pgrep pattern that starts with a dash is read as pgrep's own option and
   silently matches nothing, so every job reads as dead.
2. A watcher whose own command line contains the pattern counts itself and never
   sees the work finish.
3. A live process is not a working one. A job can sit for an hour with its
   process alive and nothing written, so throughput is counted too.
4. HARD-CODED ROW TARGETS GO STALE. They were wrong twice: a finished job read
   as stalled, and a job that could never reach its target was restarted until
   it ran out of attempts. The target is now COMPUTED from the candidate pool
   the job will actually build.
5. RESTARTING A JOB THAT IS ALREADY ALIVE gives two processes appending to one
   file. That happened on 2026-09-01 when a relaunch and a restart collided. A
   restart now checks for a live process under the same tag first.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
OUT = os.path.join(_ROOT, 'results', 'tuning_local')
STATUS = os.path.join(OUT, 'STATUS.md')
PY = sys.executable

R6 = ['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns']
R2 = ['pdv', 'chemberta']
ALL4 = ['dnn_bnn_full', 'mlp_bnn_full',
        'dnn_bnn_full_variational', 'mlp_bnn_full_variational']
SIZE = {'qm9': 3000, 'herg': 1415, 'caco2': 2161, 'logd': 3000}

# tag, dataset, models, representations, noise level
#
# TWO EXPERIMENTS PER DATASET, both needed:
#   CLEAN  every candidate on all six representations. The main table.
#   NOISE  the same candidates, PDV and ChemBERTa only. The filter column.
RUNS = [
    ('c_ba', 'qm9', ['dnn_bnn_full'], R6, 0.0),
    ('c_bb', 'qm9', ['mlp_bnn_full'], R6, 0.0),
    ('c_va', 'qm9', ['dnn_bnn_full_variational'], R6, 0.0),
    ('c_vb', 'qm9', ['mlp_bnn_full_variational'], R6, 0.0),
    ('c_herg', 'herg', ALL4, R6, 0.0),
    ('c_caco2', 'caco2', ALL4, R6, 0.0),
    ('c_logd1', 'logd', ['dnn_bnn_full', 'mlp_bnn_full'], R6, 0.0),
    ('c_logd2', 'logd', ['dnn_bnn_full_variational',
                         'mlp_bnn_full_variational'], R6, 0.0),
    ('x_herg', 'herg', ALL4, R2, 0.5),
    ('x_caco2', 'caco2', ALL4, R2, 0.5),
    ('x_logd', 'logd', ALL4, R2, 0.5),
]
# QM9 at noise finished before this list existed; its files are read by the
# table script and there is nothing left to run.
FINISHED = [('QM9 noise, variational', 'per_model_varia.csv'),
            ('QM9 noise, Bayesian a', 'per_model_bayes_a.csv'),
            ('QM9 noise, Bayesian b', 'per_model_bayes_b.csv')]


def cmd_for(tag, dataset, models, reps, level):
    return [PY, '-u', 'scripts/pick_per_model_setting.py',
            '--dataset', dataset, '--sample-size', str(SIZE[dataset]),
            '--level', str(level), '--models'] + models + ['--reps'] + reps + \
           ['--seeds', '1', '--tag', tag]


def targets():
    """How many rows each run writes when it finishes, from the pool it builds.

    The pool is the model's own search winners across every representation the
    search covered, deduplicated, plus the default. Each is fitted on every
    representation the roster allows the model."""
    import report_tuning_results as RT
    import tuning_rosters as R
    out = {}
    for tag, dataset, models, reps, _lvl in RUNS:
        try:
            _d, cands, _s, _r = RT.load(dataset)
        except Exception:
            out[tag] = None
            continue
        n = 0
        for m in models:
            sigs = set()
            for (mm, rep) in cands:
                if mm != m:
                    continue
                c = cands.get((mm, rep))
                if c:
                    _sc, p = max(c, key=lambda t: t[0])
                    if p:
                        sigs.add(json.dumps(p, sort_keys=True))
            allowed = [r for r in reps if r in R.MODELS[m][4]]
            n += (len(sigs) + 1) * len(allowed)
        out[tag] = n
    return out


def live(tag):
    """Live processes for this tag, excluding this watcher."""
    r = subprocess.run(['pgrep', '-f', '--', f'--tag {tag}'],
                       capture_output=True, text=True)
    me = str(os.getpid())
    return [p for p in r.stdout.split() if p != me]


def rows(fname):
    path = os.path.join(OUT, fname)
    if not os.path.exists(path):
        return 0, None
    with open(path) as fh:
        n = sum(1 for _ in csv.reader(fh)) - 1
    return max(0, n), os.path.getmtime(path)


def said_it_finished(tag):
    """The job's own last word. A COMPUTED TARGET CAN STILL BE SLIGHTLY OFF --
    the search may add a candidate between the target being worked out and the
    job building its pool -- and a job one row short of its target would be
    restarted forever, appending duplicates each time. The script prints
    'wrote <path>' when it has finished, and that is definitive."""
    path = os.path.join(OUT, f'log_{tag}.txt')
    if not os.path.exists(path):
        return False
    with open(path, errors='replace') as fh:
        return 'wrote ' in fh.read()[-4000:]


def refresh_tables():
    try:
        subprocess.run([PY, 'scripts/write_chosen_settings.py'],
                       cwd=_ROOT, capture_output=True, text=True, timeout=300)
    except Exception as exc:
        print(f'table refresh failed: {exc}', flush=True)


def main():
    previous, restarts = {}, {}
    target = targets()
    while True:
        now = datetime.now()
        body, alive_any = [], False

        for tag, dataset, models, reps, level in RUNS:
            fname = f'per_model_{tag}.csv'
            n, mtime = rows(fname)
            moved = n - previous.get(tag, n)
            previous[tag] = n
            pids = live(tag)
            want = target.get(tag)
            done = said_it_finished(tag) or (want is not None and n >= want)

            if len(pids) > 1:
                # TWO PROCESSES ON ONE FILE. Keep the oldest, stop the rest.
                for p in pids[1:]:
                    subprocess.run(['kill', p])
                state = f'**duplicate processes — stopped {len(pids) - 1}**'
                alive_any = True
            elif done:
                state = 'finished'
            elif pids:
                state, alive_any = 'running', True
            else:
                # NOT FINISHED AND NOT ALIVE IS DEAD.
                state = '**DEAD**'
                if restarts.get(tag, 0) < 3:
                    restarts[tag] = restarts.get(tag, 0) + 1
                    log = open(os.path.join(OUT, f'log_{tag}.txt'), 'a')
                    subprocess.Popen(
                        cmd_for(tag, dataset, models, reps, level),
                        stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, cwd=_ROOT,
                        env=dict(os.environ, OMP_NUM_THREADS='1',
                                 MKL_NUM_THREADS='1', KMP_DUPLICATE_LIB_OK='TRUE'))
                    state = f'**DEAD — restarted ({restarts[tag]} of 3)**'
                    alive_any = True
                else:
                    state = '**DEAD — 3 restarts failed, needs a person**'

            age = ('never' if mtime is None
                   else f'{(time.time() - mtime) / 60:.0f} min ago')
            kind = 'clean' if level == 0.0 else 'noise'
            body.append(f'| {tag} | {dataset} {kind} | {n} | '
                        f'{want if want is not None else "?"} | +{moved} | '
                        f'{age} | {state} |')

        for label, fname in FINISHED:
            n, _m = rows(fname)
            body.append(f'| — | {label} | {n} | {n} | +0 | — | finished |')

        refresh_tables()
        lines = [f'# tuning runs on this laptop, {now:%Y-%m-%d %H:%M}', '',
                 '| tag | what | rows | of | moved | last wrote | state |',
                 '|---|---|---|---|---|---|---|'] + body + ['']
        if not alive_any:
            lines.append('**Everything has finished.** Tables are in '
                         '`results/tuning_local/CHOSEN_SETTINGS.md`.')
        with open(STATUS, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
        print('\n'.join(lines), flush=True)
        if not alive_any:
            return 0
        time.sleep(300)


if __name__ == '__main__':
    sys.exit(main())
