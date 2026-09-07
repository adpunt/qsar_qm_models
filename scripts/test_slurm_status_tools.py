#!/usr/bin/env python
"""The three cluster-status tools, against a synthetic sacct capture.

    python scripts/test_slurm_status_tools.py

Nothing here touches the cluster. It builds an sacct capture shaped like the real one
and checks the five things that were actually wrong on 2026-09-06, each of which
produced a confident wrong answer rather than an error:

  1. THE JOB-ID RANGES WERE GUESSED. Three tools carried the same hand-typed table of
     nine ranges and six of them were assumptions -- 13.18 records only three
     submissions. A range one job wide files tasks under the wrong submission in all
     three tools at once. Discovery now reads sacct.
  2. PENDING WAS INVISIBLE. A fully pending array is one bracketed sacct row and every
     tool dropped it, so "not submitted" and "nothing has started yet" looked the same.
  3. THE QM9 RESUBMISSION LINES NAMED FILES THAT DO NOT EXIST. The job name is
     `qm91_rf`; the script is `qm9_s1_rf.sh`.
  4. THE WALL BORROW READ ONLY COMPLETED TASKS. The completed set is the fast tail, so
     ngboost borrowed 18:02 while one of its own tasks was 46.7 h in -- a limit that
     kills the job at the wall.
  5. A SKIPPED TASK IS NOT A MEASUREMENT. The selection gate exits 0 in seconds, and a
     wall set from that kills the same task the day its pair is added.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import slurm_jobs as SJ  # noqa: E402

QM9 = ['rf', 'xgboost', 'lgb', 'svm', 'ngboost', 'dnn', 'mlp', 'dnn_bnn_full',
       'mlp_bnn_full', 'dnn_bnn_full_variational', 'mlp_bnn_full_variational',
       'heteroscedastic_gp', 'dnn_bnn_full_variational_hetero',
       'mlp_bnn_full_variational_hetero', 'dnn_bnn_full_mve', 'mlp_bnn_full_mve',
       'qrf', 'gauche_rbf', 'gauche']
LAB = ['rf', 'svm', 'xgboost', 'lightgbm', 'qrf', 'ngboost', 'dnn', 'mlp', 'bnn-full',
       'bnn-full-mve', 'mlp-bnn-full', 'mlp-bnn-full-mve', 'vbll-full',
       'vbll-full-hetero', 'mlp-vbll-full', 'mlp-vbll-full-hetero', 'gp', 'gp-hetero',
       'gp-tanimoto']
UNC = ['rf', 'ngboost', 'gp', 'vbll_full', 'bnn_full_mve', 'mlp_bnn_full_mve']

FAILS = []


def check(name, ok, detail=''):
    FAILS.append(name) if not ok else None
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f'\n          {detail}'
                                                     if detail and not ok else ''))


def row(jid, name, state, elapsed, limit, mem, submit, rss=''):
    return '|'.join([jid, name, state, elapsed, limit, mem, '0:0', submit, submit,
                     rss, '1'])


def capture(path):
    """One array per model per submission, in the shapes the run design fixes."""
    out = []

    def arrays(first, names, jobname, tasks, submit, limit, mem, state='COMPLETED',
               elapsed='0:10:00', short_model=None, short_tasks=None):
        for k, m in enumerate(names):
            n = short_tasks if (short_model and m == short_model) else tasks
            for i in range(n):
                out.append(row(f'{first + k}_{i}', jobname(m), state, elapsed, limit,
                               mem, submit))

    arrays(12971601, QM9, lambda m: f'qm90_{m}', 18, '2026-09-02T01:00:00',
           '19-20:59:00', '96Gn', short_model='gauche', short_tasks=3)
    arrays(12971620, LAB, lambda m: f'val_{m}', 18, '2026-09-02T02:00:00',
           '7:00:00', '128Gn', short_model='gp-tanimoto', short_tasks=3)
    arrays(12979965, ['bnn-full-mve', 'bnn-full', 'dnn', 'lightgbm', 'gp-tanimoto'],
           lambda m: f'val_{m}', 6, '2026-09-04T09:00:00', '7:00:00', '96Gn')
    arrays(12980573, QM9, lambda m: f'qm91_{m}', 18, '2026-09-04T18:00:00',
           '19-20:59:00', '96Gn', short_model='gauche', short_tasks=3)
    # The deep run and censoring went out MINUTES apart, both named qm92_.
    arrays(12986314, QM9, lambda m: f'qm92_{m}', 36, '2026-09-06T00:10:00',
           '22-01:59:00', '96Gn', elapsed='0:00:40',
           short_model='gauche', short_tasks=6)
    arrays(12986333, QM9, lambda m: f'qm92_{m}', 6, '2026-09-06T00:20:00',
           '22-01:59:00', '96Gn', elapsed='0:00:40',
           short_model='gauche', short_tasks=1)
    arrays(12986352, LAB, lambda m: f'val_{m}', 18, '2026-09-06T00:30:00',
           '7:00:00', '96Gn', short_model='gp-tanimoto', short_tasks=3)
    arrays(12986371, LAB, lambda m: f'val_{m}', 18, '2026-09-06T00:40:00',
           '7:00:00', '96Gn', short_model='gp-tanimoto', short_tasks=3)
    arrays(12986390, UNC, lambda m: f'unc_{m}', 27, '2026-09-06T00:50:00',
           '1-23:59:00', '96Gn')
    arrays(12986396, UNC, lambda m: f'unc_{m}', 36, '2026-09-06T01:00:00',
           '1-23:59:00', '96Gn')

    # The states that matter. ngboost: three completed short, one running long.
    out = [ln for ln in out if not ln.startswith(('12980577_', '12986318_'))]
    for i in range(3):
        out.append(row(f'12980577_{i}', 'qm91_ngboost', 'COMPLETED', '18:02:00',
                       '19-20:59:00', '96Gn', '2026-09-04T18:00:00'))
    out.append(row('12980577_4', 'qm91_ngboost', 'RUNNING', '1-22:39:39',
                   '19-20:59:00', '96Gn', '2026-09-04T18:00:00'))
    for i in range(3):
        out.append(row(f'12986318_{i}', 'qm92_ngboost', 'COMPLETED', '0:00:40',
                       '22-01:59:00', '96Gn', '2026-09-06T00:10:00'))
    # A fully PENDING array: one bracketed row for 33 tasks.
    out.append(row('12986318_[3-35%5]', 'qm92_ngboost', 'PENDING', '00:00:00',
                   '22-01:59:00', '96Gn', '2026-09-06T00:10:00'))

    # Sort & Slice: indices 5, 11 and 17 of every main-grid array, dead in seconds.
    fixed = []
    for ln in out:
        p = ln.split('|')
        if p[1].startswith('qm91_') and p[0].split('_')[1] in ('5', '11', '17'):
            p[2], p[3] = 'FAILED', '00:01:12'
        fixed.append('|'.join(p))
    out = fixed

    # KIRBy's other experiments, which must be dropped (13.20 item 7).
    for k, n in enumerate(('dta_esm', 'nuc_grid', 'pc_scan', 'graphinity_a', 'tune_x')):
        out.append(row(f'12990000_{k}', n, 'COMPLETED', '9:00:00', '10:00:00',
                       '256Gn', '2026-09-06T02:00:00'))

    # Step rows, where MaxRSS lives.
    out.append(row('12980573_0.batch', 'batch', 'COMPLETED', '0:10:00', '',
                   '64Gn', '2026-09-04T18:00:00', rss='2.86G'))
    path.write_text('\n'.join(out) + '\n')


def main():
    cap = Path('/tmp/test_slurm_status_capture.psv')
    capture(cap)
    rows = SJ.parse(SJ.run_sacct('2026-09-01', str(cap)))
    groups = SJ.group_submissions(rows)
    found = {sub.label: sorted(members) for sub, members in groups}

    # 1. Discovery, including the two that share a prefix and went out minutes apart.
    check('every submission in the roster is found, and no phantom one',
          sorted(found) == sorted(s.label for s in SJ.SUBMISSIONS),
          f'found {sorted(found)}')
    for label, first, last in [
            ('QM9 screen', 12971601, 12971619),
            ('laboratory breadth', 12971620, 12971638),
            ('laboratory hERG resubmits', 12979965, 12979969),
            ('QM9 main grid', 12980573, 12980591),
            ('QM9 deep run', 12986314, 12986332),
            ('QM9 censoring', 12986333, 12986351),
            ('laboratory depth', 12986352, 12986370),
            ('laboratory censoring', 12986371, 12986389),
            ('uncertainty, the three', 12986390, 12986395),
            ('uncertainty, the four', 12986396, 12986401)]:
        got = found.get(label)
        check(f'{label} is {first}-{last}',
              got is not None and got[0] == first and got[-1] == last,
              f'got {got[0] if got else None}-{got[-1] if got else None}')

    check("KIRBy's other work is not counted as this study's",
          not any(r['JobName'].startswith(('dta_', 'nuc_', 'pc_', 'graphinity_',
                                           'tune_')) for r in rows))

    # 2. A bracketed PENDING array is 33 tasks, not zero and not one.
    bracketed = [r for r in rows if r['pending_array']]
    check('a fully PENDING array is counted, not dropped',
          len(bracketed) == 1 and SJ.pending_count(bracketed[0]['JobID']) == 33,
          f'{len(bracketed)} bracketed row(s)')

    status = subprocess.run(
        [sys.executable, str(HERE / 'run_status.py'), '--sacct-file', str(cap)],
        capture_output=True, text=True)
    check('run_status.py runs and reports pending', status.returncode == 0
          and 'pend' in status.stdout and '  162' in status.stdout,
          status.stderr[-400:] or status.stdout[:400])
    check('run_status.py never says "not submitted" for a queued submission',
          'uncertainty, the four' not in status.stdout.split('NOT SUBMITTED')[-1]
          or 'NOT SUBMITTED' not in status.stdout)

    # 3. Every resubmission line names a script the generator actually writes.
    for sub in SJ.SUBMISSIONS:
        model = 'rf'
        script = sub.script_for(f'{sub.prefix}{model}')
        if sub.prefix.startswith('qm9'):
            stage = sub.prefix[3]
            check(f'{sub.label}: script for {sub.prefix}{model} is qm9_s{stage}_{model}.sh',
                  script == f'qm9_s{stage}_{model}.sh', f'got {script}')
        else:
            check(f'{sub.label}: script for {sub.prefix}{model} is {sub.prefix}{model}.sh',
                  script == f'{sub.prefix}{model}.sh', f'got {script}')
    gen = (HERE.parent / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py').read_text()
    check('the QM9 generator still writes qm9_s{stage}_{model}.sh and names the job '
          'qm9{stage}_{model}',
          "script_name = f'qm9_s{args.stage}_{model}.sh'" in gen
          and '--job-name=qm9{stage}_{jobslug}' in gen,
          'the naming rule changed -- slurm_jobs.py must change with it')

    ft = subprocess.run(
        [sys.executable, str(HERE / 'failed_tasks.py'), '--sacct-file', str(cap),
         '--no-logs', '--emit-sbatch'], capture_output=True, text=True)
    check('failed_tasks.py runs', ft.returncode == 0, ft.stderr[-400:])
    check('failed_tasks.py never emits a bare job name as a script',
          'sbatch --array' not in ft.stdout or 'qm91_rf.sh' not in ft.stdout,
          [ln for ln in ft.stdout.splitlines() if 'qm91_rf.sh' in ln])
    check('failed_tasks.py prints no resubmission line for a FAILED cause',
          'FAILED' in ft.stdout and 'the error text is above' in ft.stdout,
          ft.stdout[-600:])

    # 4. The borrow must not propose a wall below a running task.
    mw = subprocess.run(
        [sys.executable, str(HERE / 'measure_walls.py'), '--sacct-file', str(cap),
         '--emit-scontrol'], capture_output=True, text=True)
    check('measure_walls.py runs', mw.returncode == 0, mw.stderr[-400:])
    # THE INVARIANT, checked on every line rather than by naming a model: a proposed
    # TimeLimit must be above the longest task ALREADY RUNNING on that job id, or
    # scontrol ends it on the spot.
    longest_running = {}
    for r in rows:
        if r['State'].split()[0] == 'RUNNING':
            base = r['JobID'].split('_')[0]
            longest_running[base] = max(longest_running.get(base, 0),
                                        SJ.secs(r['Elapsed']) or 0)
    checked = 0
    for line in mw.stdout.splitlines():
        if 'TimeLimit=' not in line:
            continue
        base = line.split('JobId=')[1].split()[0]
        want = SJ.secs(line.split('TimeLimit=')[1].split()[0])
        used = longest_running.get(base, 0)
        if used:
            checked += 1
            check(f'{base}: proposed {SJ.hhmmss(want)} is above the '
                  f'{SJ.hhmmss(used)} already running',
                  want is not None and want > used,
                  f'{want}s proposed against {used}s already used')
    # And the borrow: qm92_ngboost has not started, so its wall must come from the
    # 46.7 h task still running on the main grid, not the 18:02 completed tail.
    borrowed = [ln for ln in mw.stdout.splitlines()
                if 'JobId=12986318' in ln and 'TimeLimit=' in ln]
    check('the deep run borrows ngboost from the RUNNING task, not the completed tail',
          len(borrowed) == 1
          and (SJ.secs(borrowed[0].split('TimeLimit=')[1].split()[0]) or 0)
          > 1 * 86400 + 22 * 3600 + 39 * 60 + 39,
          borrowed or 'no proposal for 12986318 at all')
    check('at least one running job was covered by the invariant', checked > 0)

    # 5. A skipped task is never measured.
    check('the deep run\'s skipped tasks are reported as skipped, not as a wall',
          'skipped)' in mw.stdout, mw.stdout[:600])
    check('no wall is proposed from the deep run\'s own 40-second tasks',
          not any('TimeLimit=1:00:00' in ln or 'TimeLimit=0:' in ln
                  for ln in mw.stdout.splitlines()))

    print(f"\n  {len(FAILS)} failure(s)" if FAILS else '\n  all checks passed.')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
