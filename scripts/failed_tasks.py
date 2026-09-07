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
  FAILED          Ran and exited non-zero. Resubmitting an unfixed cause gets the same
                  exit -- so this reads the .out files and groups the tasks by their
                  actual last error, rather than telling you to go and read them. Where
                  the cause IS fixed, see `fixed_causes.json` below.
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
  * A FAILED TASK WHOSE CAUSE IS ALREADY FIXED GOT NO LINE EITHER. 104 Sort & Slice
    tasks were fixed at `62f1fe2` and this printed nothing for them, on the blanket
    rule that a FAILED cause is unfixed. `fixed_causes.json` is the exception, and it
    is not taken on trust: each entry names a commit, and a line is printed only when
    `git merge-base --is-ancestor` puts that commit behind the HEAD of the checkout
    this tool is running in. On the cluster that checkout is what the jobs run, so the
    claim is checked against the code. A commit that is not there prints a refusal.
    Where a log can be read the match is on the error text alone, so a task that died
    on something else is never swept in by its position in the array.

WHAT IT STILL DOES NOT DO. It does not resubmit anything. Nothing is deleted either:
the runner drops the rows for the combination it re-runs before it writes, so a
resubmitted task replaces its own rows.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import slurm_jobs as SJ  # noqa: E402

BAD = ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'BOOT_FAIL', 'DEADLINE',
       'PREEMPTED')

FIXED_CAUSES_FILE = Path(__file__).resolve().parent / 'fixed_causes.json'

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
        return '\n'.join(lines[max(0, i - 3):i + 1])
    return '\n'.join(lines[-keep:])


# ---------------------------------------------------------------------------
# FAILED, BUT THE CAUSE IS FIXED
#
# The rule this replaces was right in general and wrong in one case, and the wrong
# case cost the study 104 tasks: a cause fixed in code is not the same as an unfixed
# one, and refusing to print a line for it left the only way forward as typing an
# array range by hand, which has queued out-of-range tasks three times.
#
# A DOCUMENT CANNOT DECLARE A FIX. `fixed_causes.json` names a commit, and this
# checks that commit against the history of the checkout it is running in before
# anything is printed. On the cluster that checkout is what the jobs run, so the
# check is on the code and not on the claim. Three answers, all of them said out
# loud: the commit is behind HEAD, the commit exists but is not an ancestor of HEAD
# (a checkout that has not pulled), or git does not know it at all (a typo, or a
# commit that only ever existed on a laptop).
# ---------------------------------------------------------------------------
def _git(*args):
    try:
        return subprocess.run(['git', '-C', SJ.QSAR, *args],
                              capture_output=True, text=True)
    except OSError:
        return None


def load_fixed_causes(path=FIXED_CAUSES_FILE, git=_git):
    """The registry, every entry carrying whether its commit is in this checkout."""
    try:
        raw = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        return [], f'{path}: {exc}'
    out = []
    for entry in raw.get('fixed_causes', []):
        e = dict(entry)
        commit = e.get('commit', '')
        exists = git('cat-file', '-e', f'{commit}^{{commit}}') if commit else None
        if exists is None or exists.returncode != 0:
            e['verified'] = False
            e['why'] = (f'git does not know commit {commit or "(none named)"} -- so '
                        f'nothing here can say the cause is fixed')
        else:
            behind = git('merge-base', '--is-ancestor', commit, 'HEAD')
            if behind is None or behind.returncode != 0:
                e['verified'] = False
                e['why'] = (f'commit {commit} exists but is NOT an ancestor of HEAD. '
                            f'This checkout would run code without the fix -- run '
                            f'bash scripts/pull_safely.sh')
            else:
                e['verified'] = True
                e['why'] = f'commit {commit} is in this checkout, behind HEAD'
        out.append(e)
    return out, None


def rep_positions(rule, sub, jobname):
    """(how many representations an array cycles through, which of them this is).

    The QM9 script picks its representation with `REPS[$(( i % n_rep ))]`, so the
    tasks one representation owns are the indices with one remainder. The position is
    READ OFF the generated script wherever it is on disk -- `--reps` can change both
    numbers, and a script regenerated with four representations would make a typed 5
    point at the wrong one. The two numbers in the registry are the fallback for a
    laptop, where the generated scripts are not checked in.
    """
    if rule.get('kind') != 'qm9_representation':
        return None, set()
    want = rule.get('representation')
    try:
        text = (Path(SJ.QSAR) / sub.directory / sub.script_for(jobname)).read_text()
    except OSError:
        text = ''
    m = re.search(r'^REPS=\(([^)]*)\)', text, re.M)
    if m:
        reps = m.group(1).replace('"', '').replace("'", '').split()
        if want not in reps:
            return None, set()          # this script does not build it at all
        return len(reps), {reps.index(want)}
    return rule.get('assumed_modulus'), set(rule.get('assumed_residues', ()))


def fixed_cause_for(entries, sub, jobname, task, err):
    """The registry entry that explains one failed task, and how it was matched.

    A READABLE LOG IS MATCHED ON ITS ERROR TEXT AND NOTHING ELSE. Position in the
    array is the fallback for a log that cannot be read, so a task that died on
    something new is never resubmitted because it sits at an index that used to fail.
    """
    for e in entries:
        if not e.get('verified'):
            continue
        pattern = e.get('job_name_matches')
        if pattern and not re.search(pattern, jobname):
            continue
        if err is not None:
            if any(s in err for s in e.get('error_matches', ())):
                return e, 'its own log says so'
            continue
        mod, residues = rep_positions(e.get('task_rule') or {}, sub, jobname)
        if mod and task is not None and task % mod in residues:
            return e, (f'no log to read, so by position: one task in every {mod} is '
                       f'that representation')
    return None, ''


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
    ap.add_argument('--width', type=int, default=300,
                    help='characters of each error line to print (default 300). An '
                         'error cut off before the part that says what went wrong is '
                         'no better than no error.')
    ap.add_argument('--throttle', type=int, default=4,
                    help='the %% on the resubmitted array (default 4)')
    cli = ap.parse_args()

    print(SJ.provenance())

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

    # The logs are read ONCE, here, because two things need them: the grouped error
    # text printed below, and the match against fixed_causes.json.
    errs = {}
    if cli.logs:
        for cause in ('FAILED', 'TIMEOUT'):
            for sub, r, _peak in causes.get(cause, []):
                errs[r['JobID']] = last_error(log_path(sub, r['JobName'], r['JobID']))

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
            missing_paths = []
            for sub, r, _peak in rows_c:
                err = errs.get(r['JobID'])
                if err is None:
                    unreadable += 1
                    missing_paths.append(log_path(sub, r['JobName'], r['JobID']))
                    continue
                seen[err] += 1
                examples.setdefault(err, r['JobID'])
            for err, n in seen.most_common(4):
                print(f"\n      {n} of {len(rows_c)} ended on this "
                      f"(e.g. {examples[err]}):")
                for ln in err.splitlines():
                    print(f'        | {ln[:cli.width]}')
            if unreadable:
                print(f"\n      {unreadable} log(s) NOT FOUND. This is the path it "
                      f"looked for:")
                for path in missing_paths[:3]:
                    print(f'        {path}')
                print(f"      A missing log usually means the script was regenerated "
                      f"with a different\n      --output name after those tasks ran, "
                      f"or you are not on the cluster.")
        print()

    # Which FAILED tasks have a cause that is fixed in this checkout.
    entries, registry_error = load_fixed_causes()
    if registry_error:
        print(f"  fixed_causes.json could not be read: {registry_error}\n"
              f"  No FAILED task will get a resubmission line.\n")
    by_fixed = defaultdict(lambda: defaultdict(set))
    matched_how = defaultdict(Counter)
    still_unfixed = 0
    for sub, r, _peak in causes.get('FAILED', []):
        entry, how = fixed_cause_for(entries, sub, r['JobName'], r['task'],
                                     errs.get(r['JobID']))
        if entry is None or r['task'] is None:
            still_unfixed += 1
            continue
        by_fixed[entry['id']][(sub, r['JobName'], r['base'])].add(r['task'])
        matched_how[entry['id']][how] += 1

    refused = [e for e in entries if not e.get('verified')]
    if refused:
        print("  A FIXED CAUSE IS CLAIMED BUT NOT IN THIS CHECKOUT")
        for e in refused:
            print(f"    {e['id']}: {e['why']}")
        print("    No resubmission line is printed for it. Pull, then run this again.\n")

    if by_fixed:
        n = sum(len(t) for m in by_fixed.values() for t in m.values())
        print(f"  === FAILED, CAUSE FIXED: {n} task(s)")
        for cid, members in by_fixed.items():
            e = next(x for x in entries if x['id'] == cid)
            count = sum(len(t) for t in members.values())
            print(f"    {cid}: {count} task(s)")
            for how, k in matched_how[cid].most_common():
                print(f"      {k} matched because {how}")
            print(f"      what went wrong: {e['what']}")
            if e.get('fix'):
                print(f"      what changed:    {e['fix']}")
            print(f"      fixed at:        {e['why']}")
            print(f"      proved by:       {e['proof']}")
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
        if still_unfixed:
            print(f"    FAILED         -- {still_unfixed} of them have no fixed cause. "
                  f"The error text is\n                      above; resubmitting an "
                  f"unfixed cause gets the same exit, which\n                      is "
                  f"why no sbatch line is printed for those.")
        if by_fixed:
            print("    FAILED, FIXED  -- the lines below regenerate nothing. Rebuild "
                  "the scripts from\n                      the generator FIRST, so the "
                  "task runs the fixed code at the\n                      current wall "
                  "and memory request.")
    if 'CANCELLED' in causes or 'NODE / PREEMPTED' in causes:
        print("    CANCELLED / NODE -- safe to resubmit as-is.")

    resub = {c: by_script[c] for c in ('OUT OF MEMORY', 'CANCELLED', 'NODE / PREEMPTED')
             if c in by_script}
    for cid, members in by_fixed.items():
        resub[f'FAILED, fixed at {next(x for x in entries if x["id"] == cid)["commit"]}'
              ] = members
    if resub:
        lines = []
        for cause, scripts in resub.items():
            for (sub, jname, base), idx in sorted(scripts.items(),
                                                  key=lambda t: (t[0][0].label, t[0][1])):
                rng = ','.join(str(i) for i in sorted(idx))
                script = sub.script_for(jname)
                flags = (sub.submit_flags + ' ') if sub.submit_flags else ''
                lines.append(
                    f"(cd {sub.directory} && sbatch {flags}"
                    f"--array={rng}%{cli.throttle} "
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
