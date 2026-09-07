#!/usr/bin/env python
"""Which tasks failed, WHY, and the exact command to put each one back.

    python scripts/failed_tasks.py                  # triage
    python scripts/failed_tasks.py --emit-sbatch    # + the resubmission lines

WHY THIS EXISTS. `run_status.py` counts failures. A count is not actionable: 48 failed
QM9 tasks could be one bad node, or the whole grid dying at a memory limit, and those
need opposite responses. This separates them by cause, using the two things sacct knows
that the count throws away -- how the task ended, and how close it got to its limits.

  OUT OF MEMORY   MaxRSS at or near ReqMem, or the state says so outright. The QM9 main
                  grid went out at --mem=32G on 2026-09-04 and was only raised on
                  2026-09-05 (RERUN_PLAN.md 13.19 STEP 1), so anything that started in
                  that window ran under the old request. Resubmitting is enough; the
                  scripts on disk now carry the settled tiers.
  TIMEOUT         Elapsed at the limit. Resubmitting UNCHANGED just burns the wall
                  again -- raise the TimeLimit first.
  FAILED          The job ran and exited non-zero. Read its .out before resubmitting;
                  this is the only class where resubmission is not the answer.
  CANCELLED       Usually the operator, or a node going down. Safe to resubmit.

WHAT IT DOES NOT DO. It does not resubmit anything. Every line it prints is one you read
first. Nothing is deleted: the runner drops the rows for the combination it re-runs
before it writes, so a resubmitted task replaces its own rows.
"""
import argparse
import re
import subprocess
import sys
from collections import defaultdict

GROUPS = [
    ('QM9 screen', 12971601, 12971619, 'qm9'),
    ('laboratory breadth', 12971620, 12971638, 'lab'),
    ('QM9 main grid', 12980573, 12980591, 'qm9'),
    ('QM9 deep run', 12986314, 12986332, 'qm9'),
    ('QM9 censoring', 12986333, 12986351, 'qm9'),
    ('laboratory depth', 12986352, 12986370, 'lab'),
    ('laboratory censoring', 12986371, 12986389, 'lab'),
    ('uncertainty, the three', 12986390, 12986395, 'unc'),
    ('uncertainty, the four', 12986396, 12986401, 'unc'),
]
BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'CANCELLED', 'PREEMPTED')


def secs(text):
    if not text or text in ('INVALID', 'UNKNOWN', 'Partition_Limit', 'UNLIMITED'):
        return None
    d, _, rest = text.partition('-')
    if not rest:
        d, rest = '0', text
    try:
        parts = [int(x) for x in rest.split(':')]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def gb(text):
    if not text:
        return None
    m = re.match(r'^([\d.]+)\s*([KMGT]?)', text.strip())
    if not m:
        return None
    v = float(m.group(1))
    return {'K': v / 1048576, 'M': v / 1024, 'G': v, 'T': v * 1024,
            '': v / 1048576}[m.group(2)]


def sacct(ids, since, fields, extra):
    try:
        p = subprocess.run(['sacct', '-M', 'arc', '-S', since, '-j', ids, '-n', '-P',
                            f'--format={fields}', *extra],
                           capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit('no sacct on this machine -- run this on the cluster.')
    return p.stdout.splitlines() if p.returncode == 0 else []


def classify(state, elapsed, limit, rss, req):
    if state == 'OUT_OF_MEMORY':
        return 'OUT OF MEMORY'
    if state == 'TIMEOUT':
        return 'TIMEOUT'
    if rss and req and rss >= req * 0.92:
        return 'OUT OF MEMORY'          # killed before SLURM could label it
    if elapsed and limit and elapsed >= limit * 0.98:
        return 'TIMEOUT'
    if state in ('NODE_FAIL', 'PREEMPTED'):
        return 'NODE / PREEMPTED'
    if state == 'CANCELLED':
        return 'CANCELLED'
    return 'FAILED'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01')
    ap.add_argument('--emit-sbatch', action='store_true')
    ap.add_argument('--show', type=int, default=6, help='tasks to name per cause')
    cli = ap.parse_args()

    causes = defaultdict(list)
    by_script = defaultdict(lambda: defaultdict(set))
    for name, first, last, _kind in GROUPS:
        ids = ','.join(str(j) for j in range(first, last + 1))
        rss = {}
        for line in sacct(ids, cli.since, 'JobID,MaxRSS', []):
            p = line.split('|')
            if len(p) == 2 and '.' in p[0] and p[1]:
                base = p[0].rsplit('.', 1)[0]
                v = gb(p[1])
                if v is not None:
                    rss[base] = max(rss.get(base, 0), v)
        for line in sacct(ids, cli.since,
                          'JobID,JobName,State,Elapsed,Timelimit,ReqMem,ExitCode', ['-X']):
            p = line.split('|')
            if len(p) < 7 or '_' not in p[0] or '[' in p[0]:
                continue
            jid, jname, state, el, lim, req, code = p[:7]
            st = state.split()[0]
            if st not in BAD:
                continue
            cause = classify(st, secs(el), secs(lim), rss.get(jid), gb(req))
            causes[cause].append((name, jid, jname, el, lim, rss.get(jid), gb(req), code))
            by_script[cause][(name, jname, jid.split('_')[0])].add(int(jid.split('_')[1]))

    if not causes:
        print('  No failed tasks in any submission. Nothing to do.')
        return 0

    total = sum(len(v) for v in causes.values())
    print(f"  {total} failed task(s), by cause:\n")
    order = ['OUT OF MEMORY', 'TIMEOUT', 'FAILED', 'NODE / PREEMPTED', 'CANCELLED']
    for cause in order:
        rows = causes.get(cause)
        if not rows:
            continue
        print(f"  === {cause}: {len(rows)} task(s)")
        for group, jid, jname, el, lim, r, q, code in rows[:cli.show]:
            print(f"      {jid:<16s} {jname[:28]:28s} {group[:20]:20s} "
                  f"ran {el:>11s} of {lim:<12s} "
                  f"peak {(f'{r:.1f}' if r else '?'):>6s} of "
                  f"{(f'{q:.0f}' if q else '?'):>4s} GB  exit {code}")
        if len(rows) > cli.show:
            print(f"      ... and {len(rows) - cli.show} more")
        print()

    print("  WHAT TO DO WITH EACH")
    if 'OUT OF MEMORY' in causes:
        print("    OUT OF MEMORY  -- resubmit. The scripts on disk carry the settled "
              "memory tiers\n                      (model_memory.json); the failures "
              "predate them.")
    if 'TIMEOUT' in causes:
        print("    TIMEOUT        -- do NOT resubmit unchanged, it burns the wall "
              "again. Raise the\n                      limit first: "
              "scontrol cannot raise it on a job you own, so these\n"
              "                      have to be regenerated and resubmitted.")
    if 'FAILED' in causes:
        print("    FAILED         -- READ THE .out FIRST. This is the one class where "
              "resubmitting\n                      is not the answer, and 'it failed "
              "again' is the usual result.")
    if 'CANCELLED' in causes or 'NODE / PREEMPTED' in causes:
        print("    CANCELLED / NODE -- safe to resubmit as-is.")

    resub = {c: by_script[c] for c in ('OUT OF MEMORY', 'CANCELLED', 'NODE / PREEMPTED')
             if c in by_script}
    if resub:
        lines = []
        for cause, scripts in resub.items():
            for (group, jname, base), idx in sorted(scripts.items()):
                rng = ','.join(str(i) for i in sorted(idx))
                lines.append(f"sbatch --array={rng}%4 {jname}.sh"
                             f"   # {cause}, {group}, was {base}")
        print(f"\n  {len(lines)} resubmission line(s) for the safe causes. Run them from "
              f"the\n  directory that holds each script, after regenerating it.")
        if cli.emit_sbatch:
            print()
            for line in lines:
                print(f"    {line}")
        else:
            print("  Re-run with --emit-sbatch to print them.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
