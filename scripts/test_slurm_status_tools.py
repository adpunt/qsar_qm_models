#!/usr/bin/env python
"""The three cluster-status tools, against a synthetic sacct capture.

    python scripts/test_slurm_status_tools.py

Nothing here touches the cluster. It builds an sacct capture shaped like the real one
and checks the seven things that were actually wrong, each of which produced a
confident wrong answer rather than an error:

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
  6. A FAILED TASK WHOSE CAUSE IS FIXED GOT NO RESUBMISSION LINE (2026-09-07). 104
     Sort & Slice tasks were fixed at 62f1fe2 and the tool printed nothing for them,
     which left typing an array range by hand as the only way forward. So: the fix is
     claimed in fixed_causes.json, and the claim is checked against git rather than
     believed. A task with a readable log is matched on its error text alone.
  7. THE RESUBMISSION LINES CARRIED NO ACCOUNT AND NO PARTITION (2026-09-07). The QM9
     and uncertainty generators keep both off the script on purpose, so a bare
     `sbatch qm9_s1_rf.sh` exits 2 at run time on the missing partition.
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
# THE ORDER IS THE SUBMIT ORDER, off submit_all.sh, because the job ids are assigned
# in it -- so index k here is job id first+k, exactly as on the cluster. Getting this
# wrong is not cosmetic: the grouping cuts a cluster where a model name repeats, so a
# model in the wrong slot splits a submission in two.
LAB = ['bnn-full-mve', 'bnn-full', 'dnn', 'gp-hetero', 'gp-tanimoto', 'gp',
       'lightgbm', 'mlp-bnn-full-mve', 'mlp-bnn-full', 'mlp-vbll-full-hetero',
       'mlp-vbll-full', 'mlp', 'ngboost', 'qrf', 'rf', 'svm', 'vbll-full-hetero',
       'vbll-full', 'xgboost']
UNC = ['qrf', 'ngboost', 'gp', 'vbll_full', 'bnn_full_mve', 'mlp_bnn_full_mve']

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
    # 12975687: val_lightgbm index 12, submitted ALONE on 2026-09-03 during the
    # confusion. One array, one task, matching no recorded range and no task count --
    # this is what took 'hERG resubmits' off the roster and shifted every laboratory
    # run one place along.
    out.append(row('12975687_12', 'val_lightgbm', 'COMPLETED', '0:04:00', '7:00:00',
                   '128Gn', '2026-09-03T14:00:00'))
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

    # LightGBM on the laboratory datasets genuinely finishes in under three minutes.
    # The breadth grid has no selection gate, so these must be MEASURED, not discarded
    # as skips -- doing that left val_lightgbm's wall resting on one task of twelve.
    lgb_id = 12971620 + LAB.index('lightgbm')                 # 12971626 on ARC
    out = [ln for ln in out if not ln.startswith(f'{lgb_id}_')]
    for i in range(18):
        out.append(row(f'{lgb_id}_{i}', 'val_lightgbm', 'COMPLETED', '0:02:20',
                       '7:00:00', '128Gn', '2026-09-02T02:00:00'))

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
    # qm91_qrf: 20:52 measured against a 1-02:59 request. 1.29x headroom, and three
    # replicates still to run. Only ever CUTS were proposed, so this came out nowhere.
    qrf = 12980573 + QM9.index('qrf')
    out = [ln for ln in out if not ln.startswith(f'{qrf}_')]
    for i in range(15):
        out.append(row(f'{qrf}_{i}', 'qm91_qrf', 'COMPLETED', '20:52:00',
                       '1-02:59:00', '64Gn', '2026-09-04T18:00:00'))

    # The screen's gauche_rbf: queued since 2026-09-02, not one task ever started, so
    # sacct holds nothing for it at all. It must still be named.
    screen_grbf = 12971601 + QM9.index('gauche_rbf')
    out = [ln for ln in out if not ln.startswith(f'{screen_grbf}_')]

    # gauche_rbf on the deep run: two completed short tasks, one still running, and a
    # 17-day request. Too little to propose from -- but it must be NAMED, not skipped
    # in silence, because it is the one model that has never finished anywhere.
    grbf = 12986314 + QM9.index('gauche_rbf')
    out = [ln for ln in out if not ln.startswith(f'{grbf}_')]
    for i in range(2):
        out.append(row(f'{grbf}_{i}', 'qm92_gauche_rbf', 'COMPLETED', '0:05:00',
                       '17-08:59:00', '96Gn', '2026-09-06T00:10:00'))
    out.append(row(f'{grbf}_2', 'qm92_gauche_rbf', 'RUNNING', '1:37:00',
                   '17-08:59:00', '96Gn', '2026-09-06T00:10:00'))

    # A fully PENDING array: one bracketed row for 33 tasks.
    out.append(row('12986318_[3-35%5]', 'qm92_ngboost', 'PENDING', '00:00:00',
                   '22-01:59:00', '96Gn', '2026-09-06T00:10:00'))

    # Sort & Slice: indices 5, 11 and 17 of every main-grid array, dead in seconds.
    # Those are the indices where `i % 6` is Sort & Slice's place in REPS, which is
    # how the generated script picks a representation.
    fixed = []
    for ln in out:
        p = ln.split('|')
        if p[1].startswith('qm91_') and p[0].split('_')[1] in ('5', '11', '17'):
            p[2], p[3] = 'FAILED', '00:01:12'
        fixed.append('|'.join(p))
    out = fixed

    # AND ONE FAILED TASK WITH NO FIXED CAUSE, at an index Sort & Slice does not own.
    # Without it, "every FAILED task got a line" and "the registry is matching the
    # right tasks" look the same.
    out = [ln for ln in out if not ln.startswith(f'{12980573 + QM9.index("dnn")}_2|')]
    out.append(row(f'{12980573 + QM9.index("dnn")}_2', 'qm91_dnn', 'FAILED',
                   '0:03:00', '19-20:59:00', '96Gn', '2026-09-04T18:00:00'))

    # KIRBy's other experiments, which must be dropped (13.20 item 7).
    for k, n in enumerate(('dta_esm', 'nuc_grid', 'pc_scan', 'graphinity_a', 'tune_x')):
        out.append(row(f'12990000_{k}', n, 'COMPLETED', '9:00:00', '10:00:00',
                       '256Gn', '2026-09-06T02:00:00'))

    # Step rows, where MaxRSS lives.
    out.append(row('12980573_0.batch', 'batch', 'COMPLETED', '0:10:00', '',
                   '64Gn', '2026-09-04T18:00:00', rss='2.86G'))
    path.write_text('\n'.join(out) + '\n')
    # squeue: the 29 deep-run ngboost tasks that have never started, which sacct does
    # not hold at all.
    path.with_suffix('.squeue').write_text(
        '12986318_[7-35%5]|qm92_ngboost|PD\n'
        '12986331_[8-35%5]|qm92_gauche_rbf|PD\n'
        '12971618_[0-17%4]|qm90_gauche_rbf|PD\n'
        '12980577_12|qm91_ngboost|PD\n'
        '12980577_13|qm91_ngboost|PD\n'
        '12986390_[0-26%6]|unc_qrf|PD\n')


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
            # Two sbatch submissions on two days, one results tree: 12975687 alone
            # on 2026-09-03, then 12979965-12979969 on the 4th.
            ('laboratory hERG resubmits', 12975687, 12979969),
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

    check('the lone recovery job 12975687 is filed with the hERG resubmits, not '
          'as a submission of its own',
          12975687 in found.get('laboratory hERG resubmits', []),
          f"12975687 landed in {[k for k, v in found.items() if 12975687 in v]}")
    check('laboratory depth and censoring are one each, not censoring twice',
          len(found.get('laboratory depth', [])) == 19
          and len(found.get('laboratory censoring', [])) == 19,
          f"depth {len(found.get('laboratory depth', []))}, "
          f"censoring {len(found.get('laboratory censoring', []))}")

    check("KIRBy's other work is not counted as this study's",
          not any(r['JobName'].startswith(('dta_', 'nuc_', 'pc_', 'graphinity_',
                                           'tune_')) for r in rows))

    # 2. A bracketed PENDING array is 33 tasks, not zero and not one.
    bracketed = [r for r in rows if r['pending_array']]
    check('a fully PENDING array is counted, not dropped',
          len(bracketed) == 1 and SJ.pending_count(bracketed[0]['JobID']) == 33,
          f'{len(bracketed)} bracketed row(s)')

    # squeue is where the queue is. sacct has none of these 29 + 28 + 27 tasks.
    q = SJ.squeue_pending(str(cap.with_suffix('.squeue')))
    check('pending tasks come from squeue, which is the only place they exist',
          q.get(12986318) == 29 and q.get(12986331) == 28 and q.get(12986390) == 27,
          str(q))

    status = subprocess.run(
        [sys.executable, str(HERE / 'run_status.py'), '--sacct-file', str(cap),
         '--squeue-file', str(cap.with_suffix('.squeue'))],
        capture_output=True, text=True)
    check('run_status.py runs and reports pending from the queue',
          status.returncode == 0 and 'pend' in status.stdout
          and 'NOTE: no queue reading' not in status.stdout,
          status.stderr[-400:] or status.stdout[:400])
    check('run_status.py says so plainly when it has NO queue reading',
          'NOTE: no queue reading' in subprocess.run(
              [sys.executable, str(HERE / 'run_status.py'), '--sacct-file', str(cap)],
              capture_output=True, text=True).stdout)
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

    # 11. THE ACCOUNT AND THE PARTITION. A QM9 script exits 2 at run time if it is
    #     submitted without a partition, and one submitted without an account bills a
    #     project this study does not use. The QM9 and uncertainty generators keep
    #     both off the script deliberately; the laboratory one writes them in. Both
    #     halves are checked against the generators, so a resubmission line cannot go
    #     stale against them.
    unc_gen = (HERE.parent / 'slurm_scripts_uncertainty_rerun'
               / 'generate_scripts.py').read_text()
    lab_gen = (HERE.parent / 'slurm_scripts_validation_rerun'
               / 'generate_scripts.py').read_text()
    check('the QM9 and uncertainty generators still put the account and the '
          'partition on the sbatch line, not in the script',
          'sbatch --account=$ACCT --partition=$PART ' in gen
          and 'sbatch --account=$ACCT --partition=$PART ' in unc_gen
          and '#SBATCH --account' not in gen and '#SBATCH --account' not in unc_gen)
    check('the laboratory generator still writes them into the script itself',
          '#SBATCH --account=stat-cadd' in lab_gen
          and '#SBATCH --partition=' in lab_gen)
    for sub in SJ.SUBMISSIONS:
        needs = sub.prefix.startswith(('qm9', 'unc_'))
        check(f'{sub.label}: a resubmission line '
              f'{"carries" if needs else "needs no"} account and partition',
              bool(sub.submit_flags) == needs
              and (not needs or ('--account=' in sub.submit_flags
                                 and '--partition=' in sub.submit_flags)),
              repr(sub.submit_flags))

    ft = subprocess.run(
        [sys.executable, str(HERE / 'failed_tasks.py'), '--sacct-file', str(cap),
         '--no-logs', '--emit-sbatch'], capture_output=True, text=True)
    check('every QM9 resubmission line names an account and a partition',
          all('--account=stat-cadd' in ln and '--partition=' in ln
              for ln in ft.stdout.splitlines()
              if 'sbatch ' in ln and '--array=' in ln and 'qm9_s' in ln),
          [ln for ln in ft.stdout.splitlines()
           if 'qm9_s' in ln and 'sbatch' in ln and '--account=' not in ln])
    check('failed_tasks.py runs', ft.returncode == 0, ft.stderr[-400:])
    check('failed_tasks.py never emits a bare job name as a script',
          '--array=' not in ft.stdout or 'qm91_rf.sh' not in ft.stdout,
          [ln for ln in ft.stdout.splitlines() if 'qm91_rf.sh' in ln])
    # 10. FAILED, BUT THE CAUSE IS FIXED. The old rule -- never print a line for a
    #     FAILED task -- was right in general and wrong for the 104 Sort & Slice
    #     tasks fixed at 62f1fe2, and the only way forward it left was typing an
    #     array range by hand, which has queued out-of-range tasks three times.
    dnn_id = str(12980573 + QM9.index('dnn'))
    sns_lines = [ln for ln in ft.stdout.splitlines()
                 if 'sbatch ' in ln and '--array=' in ln and 'fixed at' in ln]
    check('failed_tasks.py prints a resubmission line for a FAILED cause that is '
          'fixed at a commit in this checkout',
          bool(sns_lines) and all('5' in ln.split('--array=')[1].split('%')[0]
                                  for ln in sns_lines),
          ft.stdout[-900:])
    check('and every one of those lines names only the indices that representation '
          'owns',
          all(set(ln.split('--array=')[1].split('%')[0].split(',')) <= {'5', '11', '17'}
              for ln in sns_lines),
          [ln for ln in sns_lines
           if not set(ln.split('--array=')[1].split('%')[0].split(',')) <= {'5', '11', '17'}])
    # qm91_dnn index 2 failed on something the registry does not know. Its array is
    # resubmitted for Sort & Slice, so the test is that index 2 is not in that line.
    dnn_lines = [ln for ln in ft.stdout.splitlines()
                 if dnn_id in ln and 'sbatch ' in ln and '--array=' in ln]
    check('a FAILED task whose cause is NOT in the registry still gets no line',
          all('2' not in ln.split('--array=')[1].split('%')[0].split(',')
              for ln in dnn_lines) and 'no fixed cause' in ft.stdout,
          dnn_lines)

    # The registry itself, against the real repository: an entry that names a commit
    # this checkout does not carry must be refused, not believed.
    import failed_tasks as FT                                          # noqa: E402
    entries, err = FT.load_fixed_causes()
    check('every entry in fixed_causes.json names a commit that IS in this checkout',
          err is None and bool(entries) and all(e['verified'] for e in entries),
          err or [e['why'] for e in entries if not e['verified']])

    def stub_git(rc_exists, rc_ancestor):
        class R:
            def __init__(self, rc): self.returncode = rc
        return lambda *a: R(rc_exists if a[0] == 'cat-file' else rc_ancestor)

    reg = Path('/tmp/test_fixed_causes.json')
    reg.write_text('{"fixed_causes": [{"id": "x", "commit": "deadbee", "what": "w", '
                   '"proof": "p", "error_matches": ["boom"]}]}')
    unknown, _ = FT.load_fixed_causes(reg, stub_git(1, 1))
    notpulled, _ = FT.load_fixed_causes(reg, stub_git(0, 1))
    present, _ = FT.load_fixed_causes(reg, stub_git(0, 0))
    check('a commit git does not know is refused',
          not unknown[0]['verified'] and 'does not know' in unknown[0]['why'])
    check('a commit that exists but is not behind HEAD is refused, because that '
          'checkout would run code without the fix',
          not notpulled[0]['verified'] and 'NOT an ancestor' in notpulled[0]['why'])
    check('a commit behind HEAD is accepted', present[0]['verified'])

    # The match itself. A log that CAN be read is matched on its error text alone.
    sub_main = next(s for s in SJ.SUBMISSIONS if s.label == 'QM9 main grid')
    sns_err = 'ValueError: Sort & Slice produced an all-zero count vector for N'
    hit, how = FT.fixed_cause_for(entries, sub_main, 'qm91_rf', 5, sns_err)
    check('a failed task whose log carries the fixed error is matched on the text',
          hit is not None and 'log' in how, how)
    miss, _ = FT.fixed_cause_for(entries, sub_main, 'qm91_rf', 5,
                                 'RuntimeError: CUDA out of memory')
    check('a task at the SAME index that died on something else is not swept in',
          miss is None)
    idx_hit, how2 = FT.fixed_cause_for(entries, sub_main, 'qm91_rf', 17, None)
    idx_miss, _ = FT.fixed_cause_for(entries, sub_main, 'qm91_rf', 3, None)
    check('with no log at all, the index rule matches that representation only',
          idx_hit is not None and idx_miss is None, how2)
    check('and a laboratory job is never matched by a QM9 cause',
          FT.fixed_cause_for(entries, sub_main, 'val_rf', 5, sns_err)[0] is None)

    # 4. The borrow must not propose a wall below a running task.
    mw = subprocess.run(
        [sys.executable, str(HERE / 'measure_walls.py'), '--sacct-file', str(cap),
         '--squeue-file', str(cap.with_suffix('.squeue')), '--emit-scontrol'],
        capture_output=True, text=True)
    check('measure_walls.py runs', mw.returncode == 0, mw.stderr[-400:])
    # THE INVARIANT, checked on every line rather than by naming a model. Two halves:
    # a WHOLE-ARRAY cut must be above the longest task already running on that job id,
    # or scontrol ends it on the spot; and where anything is running, the cut should
    # not be whole-array at all -- only the queued elements, so the running task keeps
    # the limit it was admitted under and cannot be shortened out from under itself
    # later.
    longest_running = {}
    for r in rows:
        if r['State'].split()[0] == 'RUNNING':
            base = r['JobID'].split('_')[0]
            longest_running[base] = max(longest_running.get(base, 0),
                                        SJ.secs(r['Elapsed']) or 0)
    checked = whole = 0
    for line in mw.stdout.splitlines():
        if 'TimeLimit=' not in line or 'scontrol' not in line:
            continue
        target = line.split('JobId=')[1].split()[0]
        base = target.split('_')[0]
        want = SJ.secs(line.split('TimeLimit=')[1].split()[0])
        used = longest_running.get(base, 0)
        if not used:
            continue
        checked += 1
        if '_[' in target:
            check(f'{base}: cuts only the {target.split("_[")[1].rstrip("]")} queued '
                  f'elements, leaving the running task alone', True)
        else:
            whole += 1
            check(f'{base}: WHOLE array cut to {SJ.hhmmss(want)} with a task at '
                  f'{SJ.hhmmss(used)} already running',
                  want is not None and want > used,
                  f'{want}s proposed against {used}s already used')
    check('a partly-running array is cut per element, not whole', whole == 0,
          f'{whole} whole-array cut(s) on jobs that have a task running')
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

    # 8. An array where NOTHING has ever started is named, not left out of the table.
    screen_grbf = 12971601 + QM9.index('gauche_rbf')
    check('an array with no sacct rows at all is NAMED, not silently absent',
          'NOTHING HAS EVER STARTED' in mw.stdout
          and str(screen_grbf) in mw.stdout.split('NOTHING HAS EVER STARTED')[1][:500],
          mw.stdout[-1500:])

    # 9. A job with too little headroom is what may fail on time, and only ever
    #    proposing cuts means it comes out nowhere.
    check('a job whose request is under 2x its longest task is named as at risk',
          'TOO LITTLE HEADROOM' in mw.stdout
          and 'qm91_qrf' in mw.stdout.split('TOO LITTLE HEADROOM')[1][:800],
          mw.stdout.split('Walls proposed')[-1][:700])
    check('and no CUT is proposed for it, because it needs more not less',
          not any(str(12980573 + QM9.index('qrf')) in ln and 'TimeLimit=' in ln
                  for ln in mw.stdout.splitlines()),
          [ln for ln in mw.stdout.splitlines()
           if str(12980573 + QM9.index('qrf')) in ln and 'TimeLimit' in ln])

    # 6. A fast task on a grid with NO selection gate is a measurement.
    lgb = [ln for ln in mw.stdout.splitlines()
           if 'val_lightgbm' in ln and 'laboratory breadth' in ln]
    check('the breadth grid has no selection gate, so its 2-minute tasks are measured',
          bool(lgb) and 'skipped' not in lgb[0] and ' 18 ' in lgb[0],
          lgb or 'val_lightgbm missing from the breadth grid')

    # 7. The uncertainty runs keep their own memory floor (model_memory.json).
    unc_mem = [ln for ln in mw.stdout.splitlines()
               if 'MinMemoryNode' in ln and 'unc_' in ln]
    check('no uncertainty job is proposed below its own 96G floor',
          all(int(ln.split('MinMemoryNode=')[1].split()[0]) >= 96 * 1024
              for ln in unc_mem), unc_mem)

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
