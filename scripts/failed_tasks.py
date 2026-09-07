#!/usr/bin/env python
"""Which tasks failed, WHY, and the exact command to put each one back.

    python scripts/failed_tasks.py                  # triage, with the error text
    python scripts/failed_tasks.py --emit-sbatch    # + the resubmission lines
    python scripts/failed_tasks.py --sacct-file sacct.psv --no-logs   # offline

WHY THIS EXISTS. `run_status.py` counts failures. A count is not actionable: 48 failed
QM9 tasks could be one bad node, or the whole grid dying at a memory limit, and those
need opposite responses. This separates them by cause, using the three things a count
throws away -- how the task ended, how close it got to its limits, and what its log
says.

  OUT OF MEMORY   MaxRSS at or near ReqMem, or the state says so outright. The QM9 main
                  grid went out at --mem=32G on 2026-09-04 and was raised on 2026-09-05
                  (RERUN_PLAN.md 13.19 STEP 1), so anything started in that window ran
                  under the old request. Resubmitting is enough; the scripts on disk
                  carry the settled tiers.
  TIMEOUT         Elapsed at the limit. Resubmitting UNCHANGED burns the wall again --
                  raise the TimeLimit first, which means regenerating and resubmitting
                  because scontrol cannot raise a limit for the job's owner.
  FAILED          Ran and exited non-zero. The one class where resubmitting is not the
                  answer -- so this reads the .out files and groups the tasks by their
                  actual last error, rather than telling you to go and read them.
  CANCELLED       The operator, or a node going down. Safe to resubmit.

WHAT IT FIXES, 2026-09-07
-------------------------
  * EVERY QM9 RESUBMISSION LINE IT PRINTED WAS UNRUNNABLE. It built them from the JOB
    NAME, and on the QM9 side the job name is not the script name: the generator writes
    `qm9_s1_rf.sh` and names the job `qm91_rf`
    (slurm_scripts_qm9_rerun/generate_scripts.py:591, :1557). `sbatch qm91_rf.sh` is a
    file that does not exist. Scripts are resolved through `slurm_jobs.py` now, which
    also knows the DIRECTORY -- the QM9 deep run and censoring both emit `qm92_<model>`
    into two different directories, and so do the three laboratory runs.
  * It printed six rows and "read the .out first", which is where the diagnosis
    stalled: the cause reached the author by pasting a traceback into a chat. It reads
    the logs itself now and prints one block per distinct error with the count.
  * The job-id ranges were guessed. See `slurm_jobs.py`.

WHAT IT STILL DOES NOT DO. It does not resubmit anything. Nothing is deleted either:
the runner drops the rows for the combination it re-runs before it writes, so a
resubmitted task replaces its own rows.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slurm_jobs as SJ  # noqa: E402

BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE',
       'PREEMPTED')

# Lines that are the error rather than the frame it happened in. A traceback's last
# line is the exception; a job killed by the shell says so with `set -euo pipefail`.
_NOISE = re.compile(r'^\s*(File "|\s+\w|Traceback|\s*\^|During handling|The above)')


def classify(state, elapsed, limit, rss, req):
    if state == 'OUT_OF_MEMORY':
        return 'OUT OF MEMORY'
    if state == 'TIMEOUT':
        return 'TIMEOUT'
    if rss and req and rss >= req * 0.92:
        return 'OUT OF MEMORY'          # killed before SLURM could label it
    if elapsed and limit and elapsed >= limit * 0.98:
        return 'TIMEOUT'
    if state in ('NODE_FAIL', 'BOOT_FAIL', 'PREEMPTED'):
        return 'NODE / PREEMPTED'
    if state.startswith('CANCELLED'):
        return 'CANCELLED'
    return 'FAILED'


def log_path(sub, jobname, jid):
    """`qm91_rf_12980573_5.out` -- the generators all use `<jobname>_%A_%a.out`."""
    base, _, task = jid.partition('_')
    return Path(SJ.QSAR) / sub.directory / f'{jobname}_{base}_{task}.out'


def last_error(path, keep=12):
    """The error a log ended on, as one short block, or None.

    Walks back from the end for the first line that is an error rather than a frame,
    then returns it with a little context. Grouping on this is what turns "48 failed"
    into "48 failed, all the same ValueError".
    """
    try:
        text = path.read_text(errors='replace')
    except OSError:
        return None
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None
    for i in range(len(lines) - 1, -1, -1):
        ln = lines[i]
        if _NOISE.match(ln):
            continue
        if ln.startswith('=== finished') or ln.startswith('+ '):
            continue
        return '\n'.join(lines[max(0, i - 2):i + 1])[:keep * 100]
    return '\n'.join(lines[-keep:])[:keep * 100]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01')
    ap.add_argument('--sacct-file', help='a capture from slurm_jobs.py --save')
    ap.add_argument('--user')
    ap.add_argument('--emit-sbatch', action='store_true')
    ap.add_argument('--show', type=int, default=6, help='tasks to name per cause')
    ap.add_argument('--no-logs', dest='logs', action='store_false',
                    help='do not read the .out files (they are only on the cluster)')
    ap.add_argument('--throttle', type=int, default=4,
                    help='the %% on the resubmitted array (default 4)')
    cli = ap.parse_args()

    rows = SJ.parse(SJ.run_sacct(cli.since, cli.sacct_file, cli.user))
    if not rows:
        print(f'  sacct knows no arrays of this study since {cli.since}.')
        return 1
    rss = SJ.max_rss_by_task(cli.since, cli.sacct_file, cli.user)

    causes = defaultdict(list)
    by_script = defaultdict(lambda: defaultdict(set))
    for sub, members in SJ.group_submissions(rows):
        for base, rs in members.items():
            for r in rs:
                if r['pending_array']:
                    continue
                st = r['State'].split()[0]
                if not (st in BAD or st.startswith('CANCELLED')):
                    continue
                cause = classify(st, SJ.secs(r['Elapsed']), SJ.secs(r['Timelimit']),
                                 rss.get(r['JobID']), SJ.gb(r['ReqMem']))
                causes[cause].append((sub, r, rss.get(r['JobID'])))
                if r['task'] is not None:
                    by_script[cause][(sub, r['JobName'], base)].add(r['task'])

    if not causes:
        print('  No failed tasks in any submission. Nothing to do.')
        return 0

    total = sum(len(v) for v in causes.values())
    print(f"  {total} failed task(s), by cause:\n")
    order = ['OUT OF MEMORY', 'TIMEOUT', 'FAILED', 'NODE / PREEMPTED', 'CANCELLED']
    for cause in order:
        rows_c = causes.get(cause)
        if not rows_c:
            continue
        print(f"  === {cause}: {len(rows_c)} task(s)")
        for sub, r, peak in rows_c[:cli.show]:
            req = SJ.gb(r['ReqMem'])
            got = f'{peak:.1f}' if peak else '?'
            asked = f'{req:.0f}' if req else '?'
            print(f"      {r['JobID']:<16s} {r['JobName'][:28]:28s} "
                  f"{sub.label[:20]:20s} ran {r['Elapsed']:>11s} of "
                  f"{r['Timelimit']:<12s} peak {got:>6s} of {asked:>4s} GB"
                  f"  exit {r['ExitCode']}")
        if len(rows_c) > cli.show:
            print(f"      ... and {len(rows_c) - cli.show} more")

        # THE DIAGNOSIS, NOT A POINTER TO IT.
        if cli.logs and cause in ('FAILED', 'TIMEOUT'):
            seen = Counter()
            examples = {}
            unreadable = 0
            for sub, r, _peak in rows_c:
                err = last_error(log_path(sub, r['JobName'], r['JobID']))
                if err is None:
                    unreadable += 1
                    continue
                seen[err] += 1
                examples.setdefault(err, r['JobID'])
            for err, n in seen.most_common(4):
                print(f"\n      {n} of {len(rows_c)} ended on this "
                      f"(e.g. {examples[err]}):")
                for ln in err.splitlines():
                    print(f'        | {ln[:110]}')
            if unreadable:
                print(f"\n      {unreadable} log(s) not readable from here -- run this "
                      f"on the cluster,\n      or pass --no-logs to skip them.")
        print()

    print("  WHAT TO DO WITH EACH")
    if 'OUT OF MEMORY' in causes:
        print("    OUT OF MEMORY  -- resubmit. The scripts on disk carry the settled "
              "memory tiers\n                      (model_memory.json); the failures "
              "predate them.")
    if 'TIMEOUT' in causes:
        print("    TIMEOUT        -- do NOT resubmit unchanged, it burns the wall "
              "again. scontrol\n                      cannot RAISE a limit for you, so "
              "these must be regenerated\n                      and resubmitted.")
    if 'FAILED' in causes:
        print("    FAILED         -- the error text is above. Resubmitting an "
              "unfixed cause gets\n                      the same exit, which is why "
              "no sbatch line is printed for it.")
    if 'CANCELLED' in causes or 'NODE / PREEMPTED' in causes:
        print("    CANCELLED / NODE -- safe to resubmit as-is.")

    resub = {c: by_script[c] for c in ('OUT OF MEMORY', 'CANCELLED', 'NODE / PREEMPTED')
             if c in by_script}
    if resub:
        lines = []
        for cause, scripts in resub.items():
            for (sub, jname, base), idx in sorted(scripts.items(),
                                                  key=lambda t: (t[0][0].label, t[0][1])):
                rng = ','.join(str(i) for i in sorted(idx))
                script = sub.script_for(jname)
                lines.append(
                    f"(cd {sub.directory} && sbatch --array={rng}%{cli.throttle} "
                    f"{script})   # {cause}, {sub.label}, was {base}")
        print(f"\n  {len(lines)} resubmission line(s) for the safe causes. Paths are "
              f"relative to\n  the repository root; regenerate the scripts first if "
              f"the fix was in the generator.")
        if cli.emit_sbatch:
            print()
            for line in lines:
                print(f"    {line}")
        else:
            print("  Re-run with --emit-sbatch to print them.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
