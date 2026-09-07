#!/usr/bin/env python
"""One place that knows what is on the cluster, shared by the three status tools.

    python scripts/slurm_jobs.py                    # what is actually submitted
    python scripts/slurm_jobs.py --emit-launch-log  # the table to paste into 13.18
    python scripts/slurm_jobs.py --save sacct.psv   # capture it for offline use

WHY THIS EXISTS.

`run_status.py`, `failed_tasks.py` and `measure_walls.py` each carried their own copy
of the same nine-row table of job-id ranges, and six of those nine rows were GUESSED.
`RERUN_PLAN.md` 13.18 records only three submissions -- 12971601-12971638, the hERG
resubmits 12979965-12979969, and the main grid 12980573-12980591. The deep run,
censoring, the laboratory depth and censoring runs and both uncertainty runs went out
afterwards and are in no document at all, so their ranges were written by assuming each
submission is exactly nineteen consecutive ids. A range that is one job wide of the
truth files tasks under the wrong submission or drops them, silently, in all three
tools at once.

So the ranges are no longer typed. This asks sacct which arrays exist, cuts them into
submissions, and names each one from the job-name prefix and its own task count. What
it finds is what gets recorded, rather than the other way round.

HOW A SUBMISSION IS RECOGNISED
------------------------------
Every array in this study is one model, and its name says which pipeline queued it:
`qm90_`/`qm91_`/`qm92_` for QM9 by run-design stage, `val_` for the laboratory, `unc_`
for the uncertainty pass. Everything else under this account belongs to KIRBy's other
work -- `dta_`, `nuc_`, `pc_`, `graphinity_`, `tune_` -- and is dropped (13.20 item 7).

That leaves three prefixes covering eight submissions, so two more things separate them:

  * a submission is one array per model, so each model name appears exactly once
    inside it. Walking a prefix's job ids in order and cutting where a model name
    comes round again separates submissions EXACTLY, however close together they were
    sent -- and the QM9 deep run and censoring went out minutes apart, inside any
    Submit-time threshold worth using. A time gap is kept only as a second cut, for a
    submission made twice with a partial roster;
  * the task count per array is fixed by the run design and differs between the
    submissions that share a prefix -- the QM9 deep run is 36 tasks an array and
    censoring 6; the uncertainty runs are 27 and 36.

A range 13.18 already records wins over both. Where it does not, the count names the
submission, and order is only the tie-break for the three laboratory runs, which are
18 tasks an array apiece.

OFFLINE
-------
`--save` writes the raw sacct output, and every tool takes `--sacct-file` to read one
back. The cluster is the only place these numbers exist and it is not reachable from
the laptop, so a captured file is the difference between checking an answer and
believing one.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

# The fields every tool asks for, in one order, so a saved file feeds all of them.
FIELDS = ('JobID,JobName,State,Elapsed,Timelimit,ReqMem,ExitCode,Submit,Start,'
          'MaxRSS,NNodes')

# ---------------------------------------------------------------------------
# What each submission is, and where its scripts live.
#
# THE SCRIPT NAME IS NOT THE JOB NAME ON THE QM9 SIDE. The generator writes
# `qm9_s{stage}_{model}.sh` and sets `--job-name=qm9{stage}_{model}`
# (slurm_scripts_qm9_rerun/generate_scripts.py:591, :1557), so a resubmission line
# built from the job name -- `sbatch qm91_rf.sh` -- names a file that does not exist.
# The laboratory and uncertainty generators do match, but only by accident of both
# being `val_{safe_name}` / `unc_{slug}`.
#
# AND THE SAME JOB NAME APPEARS IN TWO DIRECTORIES. The QM9 deep run and QM9
# censoring both emit `qm92_<model>`; the three laboratory runs all emit
# `val_<model>`. Only the submission tells them apart, which is why nothing here is
# resolved from a job name alone.
# ---------------------------------------------------------------------------
QSAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Submission:
    def __init__(self, label, prefix, directory, script_rule, tasks_per_array=None,
                 recorded=None, note=''):
        self.label = label
        self.prefix = prefix
        self.directory = directory
        self.script_rule = script_rule
        self.tasks_per_array = tasks_per_array
        self.recorded = recorded            # (first, last) if 13.18 records it
        self.note = note

    def script_for(self, jobname):
        return self.script_rule(jobname[len(self.prefix):])


def _qm9_script(stage):
    return lambda model: f'qm9_s{stage}_{model}.sh'


def _same(model):
    return None            # filled in per submission below


SUBMISSIONS = [
    Submission('QM9 screen', 'qm90_', 'slurm_scripts_qm9_rerun',
               _qm9_script(0), tasks_per_array=None,
               recorded=(12971601, 12971619),
               note='replicate 0, three conditions. 13.18 Submission 1'),
    Submission('QM9 main grid', 'qm91_', 'slurm_scripts_qm9_rerun',
               _qm9_script(1), tasks_per_array=None,
               recorded=(12980573, 12980591),
               note='replicates 1-9, three conditions. 13.18 Submission 4'),
    Submission('QM9 deep run', 'qm92_', 'slurm_scripts_qm9_rerun',
               _qm9_script(2), tasks_per_array=36,
               note='seven conditions on the selected pairs. 13.19 STEP 4'),
    Submission('QM9 censoring', 'qm92_', 'slurm_scripts_qm9_censoring',
               _qm9_script(2), tasks_per_array=6,
               note='censoring on the named pairs. 13.19 STEP 4'),
    Submission('laboratory breadth', 'val_', 'slurm_scripts_validation_rerun',
               lambda m: f'val_{m}.sh', tasks_per_array=18,
               recorded=(12971620, 12971638),
               note='19 models x 6 reps x 3 datasets. 13.18 Submission 2'),
    Submission('laboratory depth', 'val_', 'slurm_scripts_validation_depth',
               lambda m: f'val_{m}.sh', tasks_per_array=18,
               note='--include-depth-conditions. 13.19 STEP 5'),
    Submission('laboratory censoring', 'val_', 'slurm_scripts_validation_censoring',
               lambda m: f'val_{m}.sh', tasks_per_array=18,
               note='censoring on the named pairs. 13.19 STEP 5'),
    Submission('uncertainty, the three', 'unc_', 'slurm_scripts_uncertainty_rerun',
               lambda m: f'unc_{m}.sh', tasks_per_array=27,
               note='gaussian, grouped-wider, grouped-shifted. 13.19 STEP 6'),
    Submission('uncertainty, the four', 'unc_', 'slurm_scripts_uncertainty_depth',
               lambda m: f'unc_{m}.sh', tasks_per_array=36,
               note='censoring, student_t_nu5, outlier_p10, laplace. 13.19 STEP 6b'),
]

# 13.18 Submission 3: the 24 hERG tasks lost to the missing cache, resubmitted into
# the same scripts and the same results tree as the breadth grid. A separate sbatch, so
# a separate row -- folding it into the breadth grid would misreport that grid's range.
HERG_RESUBMITS = Submission(
    'laboratory hERG resubmits', 'val_', 'slurm_scripts_validation_rerun',
    lambda m: f'val_{m}.sh', tasks_per_array=6,
    recorded=(12979965, 12979969),
    note='the 24 tasks lost to the missing cache. 13.18 Submission 3')
SUBMISSIONS.insert(5, HERG_RESUBMITS)

PREFIXES = tuple(sorted({s.prefix for s in SUBMISSIONS}))


# ---------------------------------------------------------------------------
# sacct
# ---------------------------------------------------------------------------
def run_sacct(since, sacct_file=None, user=None, extra=('-X',)):
    """Every array this study owns, as raw pipe-separated sacct lines.

    -X gives one row per array TASK. Without it, sacct also emits the `.batch` and
    `.extern` steps, which is where MaxRSS lives -- so the callers that want memory
    ask twice rather than parsing a mixed table.
    """
    if sacct_file:
        with open(sacct_file) as f:
            return [ln for ln in f.read().splitlines() if ln.strip()]
    cmd = ['sacct', '-M', 'arc', '-S', since, '-n', '-P', f'--format={FIELDS}',
           '-u', user or os.environ.get('USER', ''), *extra]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit(
            'no sacct on this machine -- run this on the cluster, or capture it '
            'there with\n    python scripts/slurm_jobs.py --save sacct.psv\n'
            'and pass --sacct-file sacct.psv here.')
    if p.returncode != 0:
        raise SystemExit(f'sacct failed:\n{p.stderr.strip()}')
    return [ln for ln in p.stdout.splitlines() if ln.strip()]


def parse(lines):
    """Pipe-separated sacct rows -> dicts, keeping only this study's arrays.

    A fully PENDING array comes back as `12986399_[0-35%6]`, one row for the whole
    thing. It is kept: "nothing in sacct" and "every task still pending" are the two
    answers that most need telling apart, and dropping the bracketed row is what made
    them look the same.
    """
    names = FIELDS.split(',')
    rows = []
    for line in lines:
        p = line.split('|')
        if len(p) < len(names):
            continue
        r = dict(zip(names, p))
        jid = r['JobID']
        if '.' in jid:                      # a step row (.batch/.extern), not a task
            continue
        if not r['JobName'].startswith(PREFIXES):
            continue                        # KIRBy's other work (13.20 item 7)
        base, _, tail = jid.partition('_')
        if not base.isdigit():
            continue
        r['base'] = int(base)
        r['pending_array'] = tail.startswith('[')
        r['task'] = None if r['pending_array'] or not tail else _task_index(tail)
        rows.append(r)
    return rows


def _task_index(tail):
    m = re.match(r'^(\d+)', tail)
    return int(m.group(1)) if m else None


def pending_count(jid):
    """How many tasks a bracketed PENDING array still holds: `_[10-17%5]` -> 8."""
    m = re.search(r'\[([^\]]+)\]', jid)
    if not m:
        return 0
    total = 0
    for part in m.group(1).split('%')[0].split(','):
        if '-' in part:
            a, b = part.split('-')[:2]
            if a.isdigit() and b.isdigit():
                total += int(b) - int(a) + 1
        elif part.isdigit():
            total += 1
    return total


def _submit_epoch(text):
    for fmt in ('%Y-%m-%dT%H:%M:%S', '%Y-%m-%d %H:%M:%S'):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except (ValueError, TypeError):
            pass
    return None


def group_submissions(rows, gap_seconds=900):
    """Cluster arrays into submissions and name each one.

    Returns [(Submission, {base job id: [rows]})], in job-id order.

    THE SPLIT IS ON A REPEATED MODEL, NOT ON TIME. Every submission in this study is
    one array per model, submitted in a loop, so each model name appears exactly once
    inside it. Walking a prefix's base ids in order and starting a new cluster the
    moment a model name comes round again separates submissions exactly, however
    close together they were sent -- and the QM9 deep run and censoring went out
    minutes apart, which is inside any time threshold worth using. A Submit-time gap
    is kept as a second split for the case where a submission was made twice with a
    partial roster.

    Naming is by task count first -- the run design fixes it and it differs between
    the submissions sharing a prefix (QM9 deep run 36 an array against censoring 6;
    the uncertainty runs 27 against 36) -- and by submission order where the counts
    do not separate them, which is the three laboratory runs at 18 each.
    """
    by_base = defaultdict(list)
    for r in rows:
        by_base[r['base']].append(r)

    per_prefix = defaultdict(list)
    for base, rs in by_base.items():
        prefix = next(p for p in PREFIXES if rs[0]['JobName'].startswith(p))
        per_prefix[prefix].append(base)

    out = []
    for prefix, bases in per_prefix.items():
        bases.sort()
        clusters, current, seen = [], [], set()
        last = None
        for base in bases:
            model = by_base[base][0]['JobName'][len(prefix):]
            when = _submit_epoch(by_base[base][0].get('Submit', ''))
            gap = (current and last is not None and when is not None
                   and when - last > gap_seconds)
            if current and (model in seen or gap):
                clusters.append(current)
                current, seen = [], set()
            current.append(base)
            seen.add(model)
            if when is not None:
                last = when
        if current:
            clusters.append(current)

        candidates = [s for s in SUBMISSIONS if s.prefix == prefix]
        used = set()
        for cluster in clusters:
            counts = [_tasks_in(by_base[b]) for b in cluster]
            typical = max(set(counts), key=counts.count) if counts else None
            pick = None
            # A recorded range is the strongest evidence there is: 13.18 wrote it
            # down from the submission itself.
            for s in candidates:
                if s in used or s.recorded is None:
                    continue
                if s.recorded[0] <= cluster[0] <= s.recorded[1]:
                    pick = s
                    break
            if pick is None:
                for s in candidates:
                    if s in used or s.tasks_per_array is None:
                        continue
                    if s.tasks_per_array == typical:
                        pick = s
                        break
            if pick is None:
                pick = next((s for s in candidates if s not in used), candidates[-1])
            used.add(pick)
            out.append((pick, {b: by_base[b] for b in cluster}))
    out.sort(key=lambda t: min(t[1]))
    return out


def _tasks_in(rows):
    """Tasks in one array: counted rows, plus whatever a bracketed PENDING row holds."""
    n = sum(1 for r in rows if not r['pending_array'])
    n += sum(pending_count(r['JobID']) for r in rows if r['pending_array'])
    return n


def max_rss_by_task(since, sacct_file=None, user=None):
    """{'12980577_4': GB}. MaxRSS lives on the step rows, so this asks without -X."""
    out = {}
    for line in run_sacct(since, sacct_file, user, extra=()):
        p = line.split('|')
        if len(p) < len(FIELDS.split(',')):
            continue
        r = dict(zip(FIELDS.split(','), p))
        jid = r['JobID']
        if '.' not in jid:
            continue
        v = gb(r.get('MaxRSS'))
        if v is None:
            continue
        base = jid.rsplit('.', 1)[0]
        out[base] = max(out.get(base, 0.0), v)
    return out


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def secs(text):
    """SLURM [DD-]HH:MM:SS -> seconds, or None."""
    if not text or text.strip() in ('', 'INVALID', 'UNKNOWN', 'Partition_Limit',
                                    'UNLIMITED', 'None'):
        return None
    d, _, rest = text.partition('-')
    if not rest:
        d, rest = '0', text
    try:
        parts = [int(float(x)) for x in rest.split(':')]
    except ValueError:
        return None
    while len(parts) < 3:
        parts.insert(0, 0)
    return int(d) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def gb(text):
    """MaxRSS / ReqMem -> GB. Suffixes K, M, G, T; ReqMem may carry a trailing n or c."""
    if not text or not text.strip():
        return None
    m = re.match(r'^([\d.]+)\s*([KMGT]?)', text.strip())
    if not m:
        return None
    v = float(m.group(1))
    return {'K': v / 1048576, 'M': v / 1024, 'G': v, 'T': v * 1024,
            '': v / 1048576}[m.group(2)]


def hhmmss(s):
    """Seconds -> the [D-]H:MM:SS SLURM accepts, rounded UP to the minute.

    Up, not down: this feeds TimeLimit, and a limit rounded down is a limit that is
    below what was measured.
    """
    s = int(s) + 59
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m = r // 60
    return f'{d}-{h:02d}:{m:02d}:00' if d else f'{h}:{m:02d}:00'


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--since', default='2026-09-01')
    ap.add_argument('--sacct-file')
    ap.add_argument('--user')
    ap.add_argument('--save', help='write the raw sacct output here and stop')
    ap.add_argument('--emit-launch-log', action='store_true',
                    help='print the 13.18 rows for what was found')
    cli = ap.parse_args()

    if cli.save:
        lines = run_sacct(cli.since, cli.sacct_file, cli.user, extra=('-X',))
        lines += run_sacct(cli.since, cli.sacct_file, cli.user, extra=())
        with open(cli.save, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        print(f'  {len(lines)} sacct row(s) -> {cli.save}')
        print(f'  scp it back, then pass --sacct-file {cli.save} to any of the '
              f'status tools.')
        return 0

    rows = parse(run_sacct(cli.since, cli.sacct_file, cli.user))
    if not rows:
        print('  sacct knows no arrays of this study since ' + cli.since)
        return 1
    groups = group_submissions(rows)

    print(f"  {'submission':24s} {'job ids':>21s} {'arrays':>7s} {'tasks':>7s}  "
          f"{'recorded in 13.18?'}")
    print('  ' + '-' * 92)
    for sub, members in groups:
        bases = sorted(members)
        tasks = sum(_tasks_in(v) for v in members.values())
        rng = f'{bases[0]}-{bases[-1]}' if len(bases) > 1 else str(bases[0])
        if sub.recorded is None:
            state = 'NO -- add it'
        elif (bases[0], bases[-1]) == sub.recorded:
            state = 'yes'
        else:
            state = f'MISMATCH, 13.18 says {sub.recorded[0]}-{sub.recorded[1]}'
        print(f"  {sub.label:24s} {rng:>21s} {len(bases):7d} {tasks:7d}  {state}")

    missing = [s.label for s in SUBMISSIONS
               if s.label not in {g[0].label for g in groups}]
    if missing:
        print(f"\n  Not submitted, or outside --since {cli.since}: "
              f"{', '.join(missing)}")

    if cli.emit_launch_log:
        print('\n' + '=' * 78)
        print('  Paste into RERUN_PLAN.md 13.18. One row per array.\n')
        for sub, members in groups:
            if sub.recorded:
                continue
            print(f'#### {sub.label} — {sub.note}\n')
            print('| Job ID | Script | Tasks |')
            print('|---|---|---|')
            for base in sorted(members):
                rs = members[base]
                print(f'| {base} | `{sub.directory}/'
                      f'{sub.script_for(rs[0]["JobName"])}` | {_tasks_in(rs)} |')
            print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
