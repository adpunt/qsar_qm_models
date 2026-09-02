#!/usr/bin/env python
"""RERUN_PLAN.md section 13.14 must agree with the generator that produces the runs.

WHY THIS EXISTS. That table is the one place the run design is stated in prose,
and it had drifted THREE times by 2026-09-02: it read 13 models/78 pairs against
a generator with 17, then 17 models/294 tasks against a generator with 19. Its
own note predicted the next drift and could not stop it -- "this section is not
check-locked, and has now drifted twice."

The runbook has been locked to the generator since 2026-08-28 by
slurm_scripts_qm9_rerun/test_runbook_matches_generator.py. This does the same for
the plan. It runs the generator and compares; it never edits.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py'
PLAN = ROOT / 'RERUN_PLAN.md'

failures = []


def emit(stage):
    """(scripts, tasks, training runs) the generator writes for one pass."""
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, str(GEN), '--stage', str(stage),
             '--out-dir', tmp, '--max-hours', '720'],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(f'the generator refused --stage {stage}:\n'
                             f'{r.stderr[-800:]}')
        head = re.search(r'Stage \d+: (\d+) array scripts, ([\d,]+) tasks', r.stdout)
        runs = re.search(r'grid total: ([\d,]+) training runs', r.stdout)
        if not head or not runs:
            raise SystemExit(f'cannot read the generator summary:\n{r.stdout[:600]}')
        return (int(head.group(1)),
                int(head.group(2).replace(',', '')),
                int(runs.group(1).replace(',', '')))


def check_no_restated_settings(plan):
    """Settings the generator owns must not be written out as a literal here.

    Section 2.29 said `OOF_FOLDS_SCORED = {'ngboost': 1}` for two days against a
    generator that had moved to 3, because the value was restated in prose instead
    of pointed at. The plan owns WHAT RUNS; a per-model cost setting is the
    generator's. Naming the constant is fine and is how a reader finds it -- giving
    it a value here is not.
    """
    for m in re.finditer(r'OOF_FOLDS_SCORED[^\n]*', plan):
        line = m.group(0)
        if re.search(r"OOF_FOLDS_SCORED\s*=|'ngboost'\s*:\s*\d|--oof-folds-scored\s+\d", line):
            failures.append(
                f'RERUN_PLAN.md states a value for OOF_FOLDS_SCORED: {line.strip()!r}. '
                f'That setting lives in slurm_scripts_qm9_rerun/generate_scripts.py '
                f'with its wall-clock table. Point at it, do not copy it.')


def main():
    plan = PLAN.read_text()
    row = re.compile(
        r'\|\s*\*\*The (screen|main grid)\*\*[^|]*\|[^|]*\|\s*(\d+)\s*\|'
        r'\s*([\d,]+)\s*\|\s*\*\*([\d,]+)\*\*\s*\|')
    stated = {m.group(1): (int(m.group(2)),
                           int(m.group(3).replace(',', '')),
                           int(m.group(4).replace(',', '')))
              for m in row.finditer(plan)}
    if set(stated) != {'screen', 'main grid'}:
        failures.append(
            f'section 13.14 does not have one row each for the screen and the '
            f'main grid; found {sorted(stated)}. If the table moved, move this '
            f'check with it rather than deleting it.')
        print('\n'.join('  FAIL  ' + f for f in failures))
        return 1

    for name, stage in (('screen', 0), ('main grid', 1)):
        want = emit(stage)
        got = stated[name]
        if want != got:
            failures.append(
                f'the {name}: the plan says {got[0]} scripts, {got[1]:,} tasks, '
                f'{got[2]:,} training runs; the generator writes {want[0]}, '
                f'{want[1]:,}, {want[2]:,}. Rebuild the table by re-running the '
                f'generator, never by editing the numbers.')
        else:
            print(f'  ok    the {name}: {got[0]} scripts, {got[1]:,} tasks, '
                  f'{got[2]:,} training runs')

    # The total row must be the sum, not a third independently typed number.
    tot = re.search(r'\*\*QM9 total, generated\*\*[^|]*\|[^|]*\|[^|]*\|'
                    r'\s*\*\*([\d,]+)\*\*\s*\|\s*\*\*([\d,]+)\*\*\s*\|', plan)
    if not tot:
        failures.append('section 13.14 has no "QM9 total, generated" row')
    else:
        t_tasks = int(tot.group(1).replace(',', ''))
        t_runs = int(tot.group(2).replace(',', ''))
        s_tasks = stated['screen'][1] + stated['main grid'][1]
        s_runs = stated['screen'][2] + stated['main grid'][2]
        if (t_tasks, t_runs) != (s_tasks, s_runs):
            failures.append(
                f'the total row says {t_tasks:,} tasks and {t_runs:,} runs; the '
                f'two rows above it sum to {s_tasks:,} and {s_runs:,}')
        else:
            print(f'  ok    the total is the sum: {t_tasks:,} tasks, '
                  f'{t_runs:,} training runs')

    check_no_restated_settings(plan)

    if failures:
        print()
        for f in failures:
            print('  FAIL  ' + f)
        print(f'\n{len(failures)} failure(s)')
        return 1
    print('\nPASSED -- RERUN_PLAN.md 13.14 agrees with the generator')
    return 0


if __name__ == '__main__':
    sys.exit(main())
