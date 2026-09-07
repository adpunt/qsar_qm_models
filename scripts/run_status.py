#!/usr/bin/env python
"""How far along is the re-run, and what is the finish date actually waiting on.

    python scripts/run_status.py

Reads sacct and squeue. Nothing else, and it writes nothing.

WHY THIS EXISTS. "How far along is everything" was being answered by pasting squeue into
a chat and counting by eye, which is slow and gets the important part wrong: the finish
date is not set by how many tasks are left, it is set by the few longest ones. A grid
that is 95% complete finishes when its 22-day job finishes.

WHAT IT DOES NOT KNOW. Requested wall clock is a ceiling, not a forecast -- the walls in
this study are deliberately generous because most of them have never been timed on ARC.
So where tasks have finished, the ELAPSED column is what to believe, and the ratio beside
it says how far off the request was.
"""
import argparse
import subprocess
import sys
from collections import defaultdict

# The submissions, in the order RERUN_PLAN.md 13.19 makes them. Job ids come from
# 13.18 LAUNCH LOG; a group that has not been submitted simply reports nothing.
GROUPS = [
    ('QM9 screen',              12971601, 12971619),
    ('laboratory breadth',      12971620, 12971638),
    ('QM9 main grid',           12980573, 12980591),
    ('QM9 deep run',            12986314, 12986332),
    ('QM9 censoring',           12986333, 12986351),
    ('laboratory depth',        12986352, 12986370),
    ('laboratory censoring',    12986371, 12986389),
    ('uncertainty, the three',  12986390, 12986395),
    ('uncertainty, the four',   12986396, 12986401),
]


def seconds(elapsed):
    """SLURM Elapsed: [DD-]HH:MM:SS."""
    if not elapsed or elapsed in ('INVALID', 'UNKNOWN'):
        return None
    days, _, rest = elapsed.partition('-')
    if not rest:
        days, rest = '0', elapsed
    try:
        h, m, s = (int(x) for x in rest.split(':'))
    except ValueError:
        return None
    return int(days) * 86400 + h * 3600 + m * 60 + s


def sacct(first, last, since):
    ids = ','.join(str(j) for j in range(first, last + 1))
    try:
        out = subprocess.run(
            ['sacct', '-M', 'arc', '-S', since, '-j', ids, '-X', '-n', '-P',
             '--format=JobID,JobName,State,Elapsed,Timelimit'],
            capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit('no sacct on this machine -- run this on the cluster.')
    if out.returncode != 0:
        return []
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split('|')
        if len(parts) >= 5 and '_' in parts[0] and '[' not in parts[0]:
            rows.append(parts)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01', help='sacct -S')
    ap.add_argument('--slowest', type=int, default=8,
                    help='how many of the longest-running tasks to name')
    cli = ap.parse_args()

    print(f"{'submission':26s} {'done':>6s} {'run':>5s} {'fail':>5s} "
          f"{'longest so far':>15s}")
    print('  ' + '-' * 68)
    running_all = []
    for name, first, last in GROUPS:
        rows = sacct(first, last, cli.since)
        if not rows:
            print(f"  {name:26s} {'-- nothing in sacct; not submitted, or all pending --':>0s}")
            continue
        done = sum(1 for r in rows if r[2].startswith('COMPLETED'))
        run = sum(1 for r in rows if r[2].startswith('RUNNING'))
        fail = sum(1 for r in rows if r[2].split()[0] in
                   ('FAILED', 'TIMEOUT', 'OUT_OF_ME', 'NODE_FAIL', 'CANCELLED'))
        longest = 0
        for r in rows:
            if r[2].startswith('RUNNING'):
                s = seconds(r[3])
                if s:
                    running_all.append((s, r[0], r[1], r[4], name))
                    longest = max(longest, s)
        print(f"  {name:26s} {done:6d} {run:5d} {fail:5d} "
              f"{(str(round(longest / 3600, 1)) + ' h') if longest else '--':>15s}")

    if running_all:
        running_all.sort(reverse=True)
        print(f"\n  THE FINISH DATE IS THESE, not the task count. Longest still running:")
        for s, jid, jname, limit, group in running_all[:cli.slowest]:
            print(f"      {s / 3600:7.1f} h of {limit:>12s}   {jid:<16s} {jname:<26s} {group}")
        print(f"\n  A task that is a long way under its limit is the normal case -- the "
              f"walls\n  in this study are ceilings, and most were never timed on ARC. One "
              f"CLOSE to\n  its limit is the one to worry about: it dies there with no "
              f"partial credit.")

    print(f"\n  Everything still PENDING is invisible to sacct -X by design. "
          f"For the queue:\n      squeue -u $USER -o '%.18i %.9P %.24j %.2t %.11M %.11l %R'")
    return 0


if __name__ == '__main__':
    sys.exit(main())
