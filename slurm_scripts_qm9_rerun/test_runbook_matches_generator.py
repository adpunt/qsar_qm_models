#!/usr/bin/env python3
"""Every number and every `--array=` range in RUNBOOK.md must come from the generator.

WHY THIS EXISTS
---------------
The runbook is what an operator reads before spending cluster time, and on
2026-08-27 the close-out audit found it describing a run that no longer exists:
four conditions, 320 tasks, 22,400 training runs, `--array=0-23`. The generator
emits three conditions in the array and 240 tasks, so every `sbatch` line in it
would have mis-submitted -- six of every twenty-four indices exiting 2 on the
generator's own out-of-range guard, on every model, twice (RERUN_PLAN.md 13.12
A7). It also had no instruction for running censoring at all.

That is not a typo, it is drift: the design changed and the prose did not. Prose
cannot be kept in step by remembering to, on a document this long, in a project
where the condition set has moved four times. So the numbers are checked against
the generator instead, and the generator is the only place they are decided.

WHAT IT CHECKS
--------------
  * the condition names the grid table lists, against what the array actually runs
  * the script and task counts for the screen, the main grid and --include-excluded
  * the training-run totals
  * EVERY `--array=A-B` in an sbatch line, against the task count of the script
    that line names -- this is the one that mis-submits
  * that no sbatch line names a script the generator does not emit
  * that a pair-subset condition (censoring) has a command, with its --out-dir

WHAT IT DOES NOT CHECK
----------------------
Prose. Wall times, which the runbook itself says are arithmetic rather than
measurement, and the account and partition, which are live cluster state.

    python slurm_scripts_qm9_rerun/test_runbook_matches_generator.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNBOOK = HERE / 'RUNBOOK.md'

sys.path.insert(0, str(HERE))
import generate_scripts as gen                                   # noqa: E402

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def tasks_per_model(conditions, models):
    """{script label: array task count}, the generator's own arithmetic."""
    return {label: len(conditions) * len(reps)
            for label, (_, _, _, _, reps) in models.items()}


def generator_summary(*args):
    """Run the generator and return the summary line it prints.

    Asked of the generator rather than re-derived here. Two arithmetics for one
    number is how the runbook and the generator came apart in the first place,
    and a check that computes its own answer can be wrong in the same direction
    as the prose it is checking.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(HERE / 'generate_scripts.py'),
               '--out-dir', tmp, *args]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            fail(f'the generator refused `{" ".join(args)}`:\n{done.stdout}\n{done.stderr}')
            return None
        return done.stdout


def pass_shape(stage, include_excluded=False):
    """(conditions, models, replicates) for one pass, as main() would resolve them."""
    defaults = gen.STAGE_DEFAULTS[stage]
    models = dict(gen.MODELS)
    if include_excluded:
        models.update(gen.EXCLUDED_MODELS)
    return defaults['conditions'], models, defaults['replicates']


def check_conditions(text):
    """The grid table must name the conditions the array runs, and only those."""
    row = re.search(r'^\|\s*Noise conditions\s*\|(.+?)\|\s*$', text, re.M)
    if not row:
        fail('the grid table has no "Noise conditions" row')
        return
    listed = {c for c in re.findall(r'[a-z][a-z0-9_]+(?:-[a-z]+)?', row.group(1))
              if c.replace('-', '_') in gen.CONDITIONS}
    listed = {c.replace('-', '_') for c in listed}
    running = set(gen.STAGE_DEFAULTS[1]['conditions'])
    if listed != running:
        fail(f'the grid table lists conditions {sorted(listed)}; the array runs '
             f'{sorted(running)}. A condition in the table but not the array is one an '
             f'operator will look for and not find; the reverse is one nobody runs.')


def check_counts(text):
    """Script counts, task counts and training-run totals, for all three passes.

    Every figure comes from the generator's own printed summary.
    """
    for label, args in [
            ('screen', ('--stage', '0')),
            ('main grid', ('--stage', '1')),
            ('include-excluded', ('--stage', '0', '--include-excluded'))]:
        summary = generator_summary(*args)
        if summary is None:
            continue
        head = re.search(r'(\d+) array scripts, ([\d,]+) tasks', summary)
        runs = re.search(r'grid total: ([\d,]+) training runs', summary)
        if not (head and runs):
            fail(f'{label}: could not read the generator\'s own summary:\n{summary}')
            continue
        want_scripts = int(head.group(1))
        want_tasks = int(head.group(2).replace(',', ''))
        want_runs = int(runs.group(1).replace(',', ''))

        stated = {(int(a), int(b.replace(',', '')))
                  for a, b in re.findall(r'(\d+)\s+(?:array\s+)?scripts?,\s*([\d,]+)\s+tasks',
                                         text, re.I)}
        if (want_scripts, want_tasks) not in stated:
            fail(f'{label}: the generator emits {want_scripts} scripts and {want_tasks} '
                 f'tasks; the runbook states {sorted(stated) or "no such pair"}')

        stated_runs = {int(n.replace(',', ''))
                       for n in re.findall(r'([\d,]{3,})\s+training runs', text)}
        if want_runs not in stated_runs:
            fail(f'{label}: the generator prints {want_runs:,} training runs; the runbook '
                 f'states {sorted(stated_runs) or "none"}')


def check_array_ranges(text):
    """The range every sbatch line asks for, against the script it names.

    This is the one that costs cluster time: an index past the end exits 2 on the
    generator's own guard, so a range that is too wide mis-submits silently and a
    range that is too narrow drops whole cells of the grid.
    """
    per_model = {}
    for stage in (0, 1):
        conditions, models, _ = pass_shape(stage)
        for label, count in tasks_per_model(conditions, models).items():
            per_model[f'qm9_s{stage}_{label}.sh'] = count

    # `--array=0-23%5 qm9_s0_rf.sh`, and the loop form where the script name is
    # built from a shell variable: `qm9_s0_$s.sh` preceded by `for s in a b c`.
    seen = 0
    for line_no, line in enumerate(text.splitlines(), 1):
        m = re.search(r'--array=(\d+)-(\d+)', line)
        if not m:
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        names = re.findall(r'(qm9_s\d+_[\w$]+\.sh)', line)
        if not names:
            continue
        for name in names:
            if '$' in name:
                continue                       # resolved below, from the enclosing loop
            seen += 1
            if name not in per_model:
                fail(f'RUNBOOK.md:{line_no} submits {name}, which the generator does '
                     f'not emit')
            elif (lo, hi) != (0, per_model[name] - 1):
                fail(f'RUNBOOK.md:{line_no} submits {name} as --array={lo}-{hi}; it has '
                     f'{per_model[name]} tasks, so the range is 0-{per_model[name] - 1}')

    # The loop form: pair each `for s in ...` with the --array= line that follows it.
    lines = text.splitlines()
    for i, line in enumerate(lines):
        loop = re.match(r'\s*for s in (.+?); do\s*$', line)
        if not loop:
            continue
        labels = loop.group(1).split()
        for follow in lines[i + 1:i + 4]:
            m = re.search(r'--array=(\d+)-(\d+)', follow)
            tmpl = re.search(r'(qm9_s\d+)_\$s\.sh', follow)
            if not (m and tmpl):
                continue
            lo, hi = int(m.group(1)), int(m.group(2))
            for label in labels:
                name = f'{tmpl.group(1)}_{label}.sh'
                seen += 1
                if name not in per_model:
                    fail(f'RUNBOOK.md:{i + 1} loops over {label}, which the generator '
                         f'does not emit as {name}')
                elif (lo, hi) != (0, per_model[name] - 1):
                    fail(f'RUNBOOK.md:{i + 1} submits {name} as --array={lo}-{hi}; it has '
                         f'{per_model[name]} tasks, so the range is 0-{per_model[name] - 1}')
            break

    if seen == 0:
        fail('no sbatch --array= line found in RUNBOOK.md at all')
    return seen


def check_pair_subset_has_a_command(text):
    """Censoring runs on a subset of pairs and is not in the array. It needs its own line.

    The audit's second half: `grep -i censor` returned one hit, so an operator
    following this file would never run the condition at all -- and generating it
    without --out-dir overwrites the main-grid scripts (13.12 A4).
    """
    # A shell command may be split over several lines with a trailing backslash;
    # join those first or --out-dir on the continuation line reads as absent.
    joined = re.sub(r'\\\n\s*', ' ', text)
    for name in sorted(gen.PAIR_SUBSET_CONDITIONS):
        commands = [ln for ln in joined.splitlines()
                    if 'generate_scripts.py' in ln and name in ln]
        if not commands:
            fail(f'{name} runs on a subset of pairs and is not in the array, and the '
                 f'runbook has no generate_scripts.py command for it')
            continue
        if not any('--out-dir' in ln for ln in commands):
            fail(f'the {name} command has no --out-dir, so it would be written into the '
                 f"generator's own directory and overwrite the main-grid scripts")


def check_no_retired_flag(text):
    """A COMMAND spelled with a flag that no longer exists.

    Only commands: the runbook explains the retired flags in prose on purpose,
    and that prose is the reason the reader knows to look for them.
    """
    commands = ('generate_scripts.py', 'process_and_train.py', 'sbatch ', 'rust_processor')
    for flag in ('--sigma', '--distribution', '--noise-strategy', '--strategy-params',
                 '--bootstrapping'):
        for line_no, line in enumerate(text.splitlines(), 1):
            if not any(c in line for c in commands):
                continue
            if re.search(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])', line):
                fail(f'RUNBOOK.md:{line_no} runs {flag}, which is retired: {line.strip()}')


def main():
    text = RUNBOOK.read_text()
    check_conditions(text)
    check_counts(text)
    ranges = check_array_ranges(text)
    check_pair_subset_has_a_command(text)
    check_no_retired_flag(text)

    if FAILURES:
        print(f'FAIL — {len(FAILURES)} disagreement(s) between RUNBOOK.md and the generator:\n')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    print(f'PASS — conditions, counts, {ranges} array range(s) and the pair-subset command '
          f'all agree with generate_scripts.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
