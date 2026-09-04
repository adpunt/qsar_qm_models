#!/usr/bin/env python
"""Which laboratory tasks ran the OLD noise draw, and what to do about each.

WHY THIS EXISTS. The laboratory noise draw changed on 2026-09-04 (RERUN_PLAN.md
3.3b): a molecule's corruption is now a property of the molecule rather than of
the fold it landed in. The first instinct was to cancel every laboratory job and
resubmit, which throws away the queue position of everything still waiting.

The author's question is the right one: **a queued task has not read the code
yet.** `val_*.sh` runs `python alternative_data_noise_robustness.py` at RUN time
from the KIRBy checkout, so a task that starts after the pull runs the new draw
with no resubmission at all. Only two groups are affected:

  * tasks that ALREADY FINISHED under the old code -- their rows are on disk and
    have to be replaced;
  * tasks RUNNING right now that started before the pull -- they loaded the old
    module at process start and will write old rows when they finish.

Everything PENDING is fine and must not be cancelled.

The cutoff is the modification time of the runner in the KIRBy checkout, which is
when the pull landed. A task whose Start is before it read the old code.

WHY NOTHING HAS TO BE DELETED. The runner removes the rows for the combinations
it is re-running before it writes (`alternative_data_noise_robustness.py`, the
"Merged with existing results" guard). Each task runs one model, one
representation and one dataset, so both filters are set and a resubmitted task
REPLACES its own old rows rather than appending beside them.

    # on the cluster
    python scripts/lab_tasks_on_old_noise.py

    # offline, against saved sacct output
    python scripts/lab_tasks_on_old_noise.py --sacct-file saved.psv --cutoff <epoch>
"""
import argparse
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime

RUNNER = '/data/stat-ecr/scat9264/KIRBy/tests/alternative_data_noise_robustness.py'
# Every laboratory submission made before the change. Ranges are expanded, never
# passed to sacct as a range -- sacct -j takes a comma-separated list and reports
# only the first job when given a hyphen (RERUN_PLAN.md 13.18).
DEFAULT_JOBS = ([str(j) for j in range(12971620, 12971639)]
                + ['12975687']
                + [str(j) for j in range(12979965, 12979970)])


def sacct(jobs, since):
    out = subprocess.run(
        ['sacct', '-M', 'arc', '-S', since, '-j', ','.join(jobs), '-X', '-n', '-P',
         '--format=JobID,JobName,State,Start'],
        capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"sacct failed: {out.stderr.strip()}")
    return out.stdout


def parse_start(text):
    """SLURM prints 2026-09-03T22:10:33; 'Unknown' means it never started."""
    if not text or text == 'Unknown':
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runner', default=RUNNER,
                    help='the KIRBy runner; its mtime is when the new code landed')
    ap.add_argument('--cutoff', type=float, default=None,
                    help='override the cutoff, as a unix timestamp')
    ap.add_argument('--jobs', nargs='+', default=None,
                    help='job ids (not ranges). Default: every laboratory job '
                         'submitted before the change.')
    ap.add_argument('--since', default='2026-09-02',
                    help="sacct -S. Note sacct cannot see a job that never started, "
                         "so PENDING arrays are absent here by design.")
    ap.add_argument('--sacct-file', default=None,
                    help='read sacct output from a file instead of running it')
    cli = ap.parse_args()

    if cli.cutoff is not None:
        cutoff = cli.cutoff
    else:
        if not os.path.exists(cli.runner):
            sys.exit(f"no runner at {cli.runner}. Pass --runner, or --cutoff if you "
                     f"are running this off the cluster.")
        cutoff = os.path.getmtime(cli.runner)
    print(f"  new code landed: {datetime.fromtimestamp(cutoff)}  "
          f"({cli.runner if cli.cutoff is None else 'given'})")
    print(f"  a task that STARTED before that read the old noise draw.\n")

    raw = (open(cli.sacct_file).read() if cli.sacct_file
           else sacct(cli.jobs or DEFAULT_JOBS, cli.since))

    resubmit = defaultdict(list)     # script -> array indices to re-run
    cancel = []                      # running on old code
    fine = 0
    for line in raw.splitlines():
        parts = line.split('|')
        if len(parts) < 4:
            continue
        jobid, name, state, start = parts[0], parts[1], parts[2], parts[3]
        if '_' not in jobid or '[' in jobid:
            continue                 # the pending array placeholder, not a task
        idx = jobid.rsplit('_', 1)[1]
        started = parse_start(start)
        if started is None or started >= cutoff:
            fine += 1
            continue
        if state.startswith('RUNNING'):
            # Cancelled AND resubmitted. Cancelling alone would leave the index
            # never run at all, which is worse than the old rows -- the first
            # version of this script said the resubmission covered them and it
            # did not.
            cancel.append(jobid)
            resubmit[f'{name}.sh'].append(int(idx))
        elif state.startswith('COMPLETED'):
            resubmit[f'{name}.sh'].append(int(idx))
        # FAILED/CANCELLED tasks wrote nothing worth replacing; a resubmission of
        # them is the hERG recovery, not this.

    print(f"  {fine} task(s) already run or will run the NEW draw — leave them alone.")
    print(f"  Everything still PENDING is invisible to sacct and is also fine: a "
          f"queued task\n  has not read the code yet.\n")

    if cancel:
        print(f"  {len(cancel)} task(s) are RUNNING on the old code. They will write "
              f"old rows when they\n  finish, so cancel them. They are also in the "
              f"resubmission below — cancelling alone\n  would leave those indices "
              f"never run:\n")
        print(f"    scancel {' '.join(cancel)}\n")

    if resubmit:
        total = sum(len(v) for v in resubmit.values())
        print(f"  {total} task(s) to re-run — finished under the old draw, or "
              f"cancelled above. Re-running\n  a task REPLACES its own rows: the "
              f"runner drops the combinations it is re-running\n  before it writes, "
              f"so nothing needs deleting:\n")
        for script in sorted(resubmit):
            idx = sorted(set(resubmit[script]))
            print(f"    sbatch --array={','.join(str(i) for i in idx)}%4 {script}")
        print()

    if not cancel and not resubmit:
        print("  Nothing to do: no task ran under the old draw.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
