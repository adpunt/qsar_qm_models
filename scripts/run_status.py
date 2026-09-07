#!/usr/bin/env python
"""How far along is the re-run, and what is the finish date actually waiting on.

    python scripts/run_status.py
    python scripts/run_status.py --sacct-file sacct.psv    # offline

Reads sacct AND squeue. Writes nothing.

WHY THIS EXISTS. "How far along is everything" was being answered by pasting squeue
into a chat and counting by eye, which is slow and gets the important part wrong: the
finish date is not set by how many tasks are left, it is set by the few longest ones. A
grid that is 95% complete finishes when its 22-day job finishes.

WHAT THIS FIXES, 2026-09-07
---------------------------
  * PENDING was invisible, and sacct is the wrong place to look for it. The old
    version dropped any sacct row whose id carried a bracket, which is what a fully
    pending array is -- so a submission where nothing had started printed "not
    submitted, or all pending", the one distinction that matters when the question is
    whether the queue is moving. But bracketed rows are not the whole answer either:
    an array element that has NEVER STARTED is not in the accounting database at all.
    On the real cluster that made the QM9 deep run look like 550 tasks against its 654.
    The pending column comes from squeue now, which is the only place the queue exists,
    and the tool says so plainly when it has no queue reading.
  * There was no denominator. done/running/failed with no total cannot answer "how far
    along"; a total and a percentage are now there.
  * The job-id ranges were a hand-typed guess, in three tools at once. They come from
    `slurm_jobs.py` now, which asks sacct what exists (see its header).
  * CANCELLED was counted as a failure. A task the operator cancelled is not a task
    that broke, and mixing them made the failure count unreadable.

WHAT IT DOES NOT KNOW. Requested wall clock is a ceiling, not a forecast -- the walls
in this study are deliberately generous because most were never timed on ARC. Where
tasks have finished, ELAPSED is what to believe. `measure_walls.py` turns that into
the scontrol lines; this only shows you where to look.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slurm_jobs as SJ  # noqa: E402

DONE = ('COMPLETED',)
BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_ME', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01', help='sacct -S')
    ap.add_argument('--sacct-file', help='a capture from slurm_jobs.py --save')
    ap.add_argument('--squeue-file',
                    help='the .squeue file beside it. Without a queue reading the '
                         'pending column is blank and the percentages are a fraction '
                         'of a moving denominator -- sacct does not know about a task '
                         'that has never started.')
    ap.add_argument('--user')
    ap.add_argument('--slowest', type=int, default=8,
                    help='how many of the longest-running tasks to name')
    cli = ap.parse_args()

    print(SJ.provenance())

    rows = SJ.parse(SJ.run_sacct(cli.since, cli.sacct_file, cli.user))
    if not rows:
        print(f'  sacct knows no arrays of this study since {cli.since}.')
        return 1
    groups = SJ.group_submissions(rows)
    queued = SJ.squeue_pending(cli.squeue_file, cli.user)
    if not queued:
        print('  NOTE: no queue reading. sacct does not hold a task that has never '
              'started,\n  so `pend` is what sacct happens to know and the '
              'percentages are of a\n  moving denominator. Run this on the cluster, '
              'or pass --squeue-file.\n')

    print(f"  {'submission':26s} {'done':>6s} {'run':>5s} {'pend':>6s} {'fail':>5s} "
          f"{'canc':>5s} {'total':>6s} {'%':>5s}  {'longest running':>15s}")
    print('  ' + '-' * 96)
    running_all = []
    grand = [0, 0, 0, 0, 0]
    for sub, members in groups:
        done = run = pend = fail = canc = 0
        longest = 0
        for base in members:
            pend += queued.get(base, 0)
        for rs in members.values():
            for r in rs:
                st = r['State'].split()[0]
                if r['pending_array']:
                    # squeue already counted it, and counted it right.
                    if not queued:
                        pend += SJ.pending_count(r['JobID'])
                    continue
                if st in DONE:
                    done += 1
                elif st == 'RUNNING':
                    run += 1
                    s = SJ.secs(r['Elapsed'])
                    if s:
                        running_all.append((s, r['JobID'], r['JobName'],
                                            r['Timelimit'], sub.label))
                        longest = max(longest, s)
                elif st == 'PENDING':
                    if not queued:
                        pend += 1
                elif st.startswith('CANCELLED'):
                    canc += 1
                elif st in BAD:
                    fail += 1
        total = done + run + pend + fail + canc
        for i, v in enumerate((done, run, pend, fail, canc)):
            grand[i] += v
        pct = f'{100.0 * done / total:.0f}%' if total else '--'
        print(f"  {sub.label:26s} {done:6d} {run:5d} {pend:6d} {fail:5d} {canc:5d} "
              f"{total:6d} {pct:>5s}  "
              f"{((str(round(longest / 3600, 1)) + ' h') if longest else '--'):>15s}")
    gt = sum(grand)
    print('  ' + '-' * 96)
    print(f"  {'ALL':26s} {grand[0]:6d} {grand[1]:5d} {grand[2]:6d} {grand[3]:5d} "
          f"{grand[4]:5d} {gt:6d} "
          f"{(f'{100.0 * grand[0] / gt:.0f}%' if gt else '--'):>5s}")

    missing = [s.label for s in SJ.SUBMISSIONS
               if s.label not in {g[0].label for g in groups}]
    if missing:
        print(f"\n  NOT SUBMITTED (sacct has no array of theirs since {cli.since}):"
              f"\n      {', '.join(missing)}")

    if grand[3]:
        print(f"\n  {grand[3]} FAILED task(s). Cause, and the line to put each one "
              f"back:\n      python scripts/failed_tasks.py")

    if running_all:
        running_all.sort(reverse=True)
        print(f"\n  THE FINISH DATE IS THESE, not the task count. Longest still "
              f"running:")
        for s, jid, jname, limit, group in running_all[:cli.slowest]:
            lim = SJ.secs(limit)
            frac = f'{100.0 * s / lim:3.0f}% of' if lim else 'of'
            print(f"      {s / 3600:7.1f} h {frac:>8s} {limit:>12s}   {jid:<16s} "
                  f"{jname:<26s} {group}")
        print("\n  A task a long way under its limit is the normal case -- the walls "
              "in this\n  study are ceilings, and most were never timed on ARC. One "
              "CLOSE to its limit\n  is the one to worry about: it dies there with no "
              "partial credit.\n  Over-asked walls are the queue problem, not the "
              "safe case:\n      python scripts/measure_walls.py")
    return 0


if __name__ == '__main__':
    sys.exit(main())
