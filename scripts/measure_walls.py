#!/usr/bin/env python
"""What the jobs ACTUALLY used, and the scontrol lines to re-request it.

    python scripts/measure_walls.py                 # report
    python scripts/measure_walls.py --emit-scontrol # + the commands to apply it
    python scripts/measure_walls.py --sacct-file sacct.psv   # offline

WHY THIS EXISTS, AND WHY IT IS URGENT RATHER THAN TIDY.

Every wall clock in this study is a guess with a wide margin, and several were never
timed on ARC at all. That was harmless while nothing was queued. It is not harmless
now: a 22-day request cannot backfill. SLURM fits pending work into the gaps it can
predict, and a job asking for three weeks fits almost no gap, so it waits for a drain
that may never come while 4-hour jobs run past it. The walls are the queue problem.

Cutting a TimeLimit on a PENDING job is allowed to the owner and keeps the submit time,
so it costs no queue position -- unlike cancel-and-resubmit. Raising one is not, so the
margins below are deliberately generous: a job killed at the wall has no partial credit.

MEMORY IS THE SAME QUESTION AND WAS SETTLED ON ONE DATA POINT. model_memory.json rests
on a single 61.2 GB peak from an old run, because that was all anybody had. There are
now hundreds of completed tasks from THIS pipeline, so the tiers can be read off rather
than argued about.

THE THREE WAYS THIS TOOL HAS BEEN WRONG, AND WHAT STOPS EACH ONE NOW
--------------------------------------------------------------------
1. It measured tasks that SKIPPED. A deep-run or censoring task whose pair is not in
   the selection exits 0 in seconds, and 36 of those read as "longest completed
   0:01:00" -- a one-hour wall for a model that needs a day the moment its pair is
   added. Anything under --min-elapsed is counted separately as skipped and never
   measured.

2. It would have KILLED A RUNNING TASK. `12980577_4` was 46.7 h in when the completed
   tasks suggested 37 h, and a TimeLimit cut below what a running task has already used
   ends it immediately. Running tasks are part of the measurement now, and nothing is
   ever proposed below one.

3. THE BORROW STILL IGNORED RUNNING TASKS -- fixed 2026-09-07. The deep run and
   censoring have not started, so their walls are borrowed from the same model in
   another submission. That borrow read only COMPLETED tasks, and the completed set is
   the FAST TAIL by construction: the slow ones have not finished to be counted. On
   `ngboost` the completed longest was 18:02 while a task was 46.7 h in and still
   going, so `qm92_ngboost` -- which is TEN replicates against the main grid's nine --
   would have been handed a 37-hour wall for work already known to take longer than
   that. It would have died at the wall with no partial credit: the exact failure this
   tool exists to prevent, one level removed. The borrow takes the longest of completed
   AND running now, and says which it used.

WHAT IT REFUSES TO DO. It never proposes a wall or a memory BELOW what was measured, it
never proposes memory under the author's 64G floor, and it says how many tasks each
figure rests on -- a median over two tasks is not a measurement.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slurm_jobs as SJ  # noqa: E402

BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE')


def model_of(sub, jobname):
    """(pipeline, model) -- what a measurement may be carried across.

    Only within a pipeline: QM9 from QM9, laboratory from laboratory, uncertainty from
    uncertainty. The datasets are different sizes and the uncertainty pass fits each
    model `1 + oof_folds` times per level, so nothing crosses those lines.
    """
    pipeline = {'qm90_': 'qm9', 'qm91_': 'qm9', 'qm92_': 'qm9',
                'val_': 'lab', 'unc_': 'unc'}[sub.prefix]
    return pipeline, jobname[len(sub.prefix):]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01')
    ap.add_argument('--sacct-file', help='a capture from slurm_jobs.py --save')
    ap.add_argument('--user')
    ap.add_argument('--wall-margin', type=float, default=2.0,
                    help='multiply the LONGEST observed task by this (default 2.0). '
                         'A killed job has no partial credit.')
    ap.add_argument('--mem-margin', type=float, default=1.6)
    ap.add_argument('--floor-gb', type=int, default=64, help="the author's 64G floor")
    ap.add_argument('--min-tasks', type=int, default=3,
                    help='do not propose anything from fewer observed tasks than this')
    ap.add_argument('--min-elapsed', type=int, default=180,
                    help='ignore tasks shorter than this many seconds (default 180). '
                         'A deep-run or censoring task whose pair is not in the '
                         'selection SKIPS and exits 0 in seconds.')
    ap.add_argument('--no-borrow', dest='borrow', action='store_false',
                    help='do not carry a measurement across from the same model in '
                         'another submission. Off by default: without it the deep run '
                         'and censoring -- the jobs that have not started and so are '
                         'the entire queue problem -- get no proposal at all.')
    ap.add_argument('--emit-scontrol', action='store_true')
    cli = ap.parse_args()

    rows = SJ.parse(SJ.run_sacct(cli.since, cli.sacct_file, cli.user))
    if not rows:
        print(f'  sacct knows no arrays of this study since {cli.since}.')
        return 1
    rss = SJ.max_rss_by_task(cli.since, cli.sacct_file, cli.user)

    per = {}
    for sub, members in SJ.group_submissions(rows):
        for base, rs in members.items():
            for r in rs:
                if r['pending_array']:
                    continue
                k = (sub, r['JobName'])
                d = per.setdefault(k, dict(elapsed=[], running=[], rss=[], req_t=None,
                                           req_m=None, jobs=set(), noop=0,
                                           bad=defaultdict(list)))
                d['jobs'].add(base)
                d['req_t'] = d['req_t'] or SJ.secs(r['Timelimit'])
                d['req_m'] = d['req_m'] or SJ.gb(r['ReqMem'])
                st = r['State'].split()[0]
                s = SJ.secs(r['Elapsed'])
                if st == 'COMPLETED':
                    if s is not None and s < cli.min_elapsed:
                        d['noop'] += 1
                        continue
                    if s:
                        d['elapsed'].append(s)
                    if r['JobID'] in rss:
                        d['rss'].append(rss[r['JobID']])
                elif st == 'RUNNING':
                    if s:
                        d['running'].append(s)
                    if r['JobID'] in rss:
                        d['rss'].append(rss[r['JobID']])
                elif st in BAD:
                    d['bad'][st].append(r['JobID'])

    # WHAT MAY BE BORROWED. The longest OBSERVED task per model, completed or still
    # running -- see point 3 in the header. A running task that is longer than every
    # completed one is not a stuck job, it is the measurement telling you the completed
    # set is the fast tail.
    best = {}
    for (sub, jn), dd in per.items():
        obs = dd['elapsed'] + dd['running']
        if obs:
            key = model_of(sub, jn)
            cand = max(obs)
            if cand > best.get(key, (0, ''))[0]:
                best[key] = (cand, 'still running' if max(dd['running'] or [0]) == cand
                             else 'completed')

    print(f"{'group / job':46s} {'n':>3s} {'longest':>10s} {'asked':>11s} "
          f"{'peak GB':>8s} {'asked':>7s}")
    print('-' * 96)
    scontrol = []
    for (sub, jname), d in sorted(per.items(), key=lambda t: (t[0][0].label, t[0][1])):
        n = len(d['elapsed'])
        longest = max(d['elapsed']) if d['elapsed'] else None
        longest_run = max(d['running']) if d['running'] else 0
        peak = max(d['rss']) if d['rss'] else None
        got = SJ.hhmmss(longest) if longest else '--'
        ask_t = SJ.hhmmss(d['req_t']) if d['req_t'] else '--'
        got_m = f'{peak:.1f}' if peak else '--'
        ask_m = f"{d['req_m']:.0f}" if d['req_m'] else '--'
        skipped = f"   ({d['noop']} skipped)" if d['noop'] else ''
        print(f"{sub.label[:22]:22s} {jname[:23]:23s} {n:3d} {got:>10s} "
              f"{ask_t:>11s} {got_m:>8s} {ask_m:>7s}{skipped}")
        for st, jids in sorted(d['bad'].items()):
            print(f"{'':46s} !! {len(jids):d} {st}: {', '.join(jids[:4])}"
                  + (' ...' if len(jids) > 4 else ''))
        if longest_run > (longest or 0):
            print(f"{'':46s} .. a task is STILL RUNNING at "
                  f"{SJ.hhmmss(longest_run)}, past every completed one -- the "
                  f"completed set is the fast tail")

        borrowed = None
        observed = max(longest or 0, longest_run)
        if (n < cli.min_tasks or not longest) and cli.borrow:
            b = best.get(model_of(sub, jname))
            if b and b[0] > observed:
                borrowed = b
                observed = b[0]
        if not observed:
            continue
        if borrowed is None and n < cli.min_tasks:
            continue

        # Never below what a running task has already used, or scontrol kills it.
        want_t = int(max(observed, longest_run) * cli.wall_margin) + 3600
        if d['req_t'] and want_t < d['req_t'] * 0.75:
            how = (f'BORROWED from the same model elsewhere, '
                   f'{SJ.hhmmss(borrowed[0])} ({borrowed[1]})' if borrowed
                   else f'longest of {n} was {SJ.hhmmss(longest)}')
            for j in sorted(d['jobs']):
                scontrol.append(f"scontrol update JobId={j} "
                                f"TimeLimit={SJ.hhmmss(want_t)}   # {jname}: {how}")
        if peak and d['req_m']:
            want_m = max(cli.floor_gb, int(peak * cli.mem_margin / 16 + 1) * 16)
            if want_m < d['req_m'] * 0.9:
                for j in sorted(d['jobs']):
                    scontrol.append(
                        f"scontrol update JobId={j} MinMemoryNode={want_m * 1024}"
                        f"   # {jname}: peak of {len(d['rss'])} was {peak:.1f} GB")

    print(f"\n  Walls proposed at {cli.wall_margin}x the LONGEST OBSERVED task "
          f"(completed or\n  running) plus an hour; memory at {cli.mem_margin}x the "
          f"peak, rounded up to 16 GB,\n  never below {cli.floor_gb}G. Nothing is "
          f"proposed from fewer than {cli.min_tasks} observed tasks,\n  and a task "
          f"that exited in under {cli.min_elapsed}s is not counted -- that is the "
          f"selection\n  gate skipping a pair, not a model that runs in a minute.")
    if scontrol:
        print(f"\n  {len(scontrol)} change(s) worth making. Cutting a TimeLimit on a "
              f"PENDING job keeps\n  its submit time, so it costs no queue position -- "
              f"and a smaller request is\n  what lets SLURM backfill it at all. "
              f"scontrol cannot RAISE a limit for you, so\n  anything measured as "
              f"UNDER-asked has to be regenerated and resubmitted.")
        if cli.emit_scontrol:
            print()
            for line in scontrol:
                print(f"  {line}")
        else:
            print("  Re-run with --emit-scontrol to print them.")
    else:
        print("\n  Nothing worth changing on the evidence so far.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
