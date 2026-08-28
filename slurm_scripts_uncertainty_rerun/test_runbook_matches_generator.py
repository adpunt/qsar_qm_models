#!/usr/bin/env python3
"""Every count and every `--array=` range in RUNBOOK.md must come from the generator.

WHY THIS EXISTS
---------------
The QM9 side has had this check since 2026-08-27 and this side did not, and on
2026-08-28 the difference showed. The generator emits 10 scripts of 84 tasks;
the runbook described 7 scripts of 60 tasks and every `sbatch` line in it read
`--array=0-59`. Inside a generated script the index maps datasets in blocks of
28 (`ds = DATASETS[i / (n_cond * n_rep)]`), so `0-59` reaches 4 of hERG's 28
tasks and none of the rest.

That failure is silent in both directions. A short array range is a legitimate
request -- SLURM runs it, every task it does run succeeds, and the missing 86%
of hERG shows up only in `coverage.csv`, after the queue has been spent. Three
of the ten scripts had no submit line at all.

Prose cannot be kept in step by remembering to. So the numbers are checked
against the generator, and the generator is the only place they are decided.

WHAT IT CHECKS
--------------
  * the condition table names exactly the conditions the array runs
  * the script count and the per-script task count, from the generator's own
    printed summary
  * the total task count and the total fit count
  * EVERY `--array=A-B` line, against the task count of the script it names --
    this is the one that mis-submits
  * that every script the generator emits has a submit line somewhere, and that
    no submit line names a script the generator does not emit

WHAT IT DOES NOT CHECK
----------------------
Prose. Wall times, which the runbook itself flags as arithmetic rather than
measurement. The account and partition, which are live cluster state.

    python slurm_scripts_uncertainty_rerun/test_runbook_matches_generator.py
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


def generator_summary(*args):
    """Run the generator and return what it printed.

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


def script_names():
    """{script filename: model name}, the generator's own naming rule."""
    return {f'unc_{m.lower().replace("-", "_")}.sh': m for m in gen.MODELS}


def check_conditions(text):
    """The condition table must name what the array runs, and only that."""
    running = set(gen.MAIN_GRID_CONDITIONS + gen.DEEP_RUN_CONDITIONS)
    listed = set()
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        cell = line.split('|')[1].strip().strip('`')
        if cell in running or cell in gen.KNOWN_CONDITIONS:
            listed.add(cell)
    if listed != running:
        fail(f'the condition table lists {sorted(listed)}; the array runs '
             f'{sorted(running)}. A condition in the table but not the array is one '
             f'an operator will look for and not find; the reverse is one nobody runs.')


def check_counts(text, summary):
    """Script count, per-script task count, total tasks and total fits."""
    head = re.search(r'Wrote (\d+) array scripts, (\d+) tasks each '
                     r'\(([\d,]+) tasks total\)', summary)
    if not head:
        fail(f"could not read the generator's own summary:\n{summary}")
        return
    n_scripts = int(head.group(1))
    per_script = int(head.group(2))
    total = int(head.group(3).replace(',', ''))

    stated = {(int(a), int(b)) for a, b in
              re.findall(r'(\d+) array scripts, (\d+) tasks each', text)}
    if (n_scripts, per_script) not in stated:
        fail(f'the generator emits {n_scripts} scripts of {per_script} tasks; the '
             f'runbook states {sorted(stated) or "no such pair"}')

    if not re.search(r'\*\*' + f'{total:,}' + r' tasks\*\*', text):
        fail(f'the generator emits {total:,} tasks in total; the runbook does not '
             f'state that figure in bold anywhere')

    # Fits per task: noise levels x scaffold folds x (1 + out-of-fold folds).
    # The level grid is the runner's, read here rather than restated.
    levels = _noise_levels()
    if levels is None:
        return
    folds = 5
    per_task = levels * folds * (1 + _default_oof())
    fits = per_task * total
    stated_fits = {int(n.replace(',', ''))
                   for n in re.findall(r'([\d,]{3,}) (?:model )?fits', text)}
    if fits not in stated_fits:
        fail(f'{levels} levels x {folds} folds x (1 + {_default_oof()}) fits x '
             f'{total:,} tasks = {fits:,} fits; the runbook states '
             f'{sorted(stated_fits) or "none"}')


def _default_oof():
    for action in _generator_actions():
        if action.dest == 'oof_folds':
            return action.default
    fail('the generator has no --oof-folds default to read')
    return 5


def _generator_actions():
    """The generator's own argparse actions, without running main()."""
    import argparse
    ap = argparse.ArgumentParser()
    src = (HERE / 'generate_scripts.py').read_text()
    # Cheap and exact: the default is a literal in the source, and importing
    # main() would generate scripts as a side effect.
    m = re.search(r"--oof-folds', type=int, default=(\d+)", src)
    if m:
        ap.add_argument('--oof-folds', type=int, default=int(m.group(1)))
    return ap._actions


def _noise_levels():
    """How many noise levels the runner sweeps. Read from the runner, not restated."""
    runner = Path(gen.KIRBY_DIR)
    for candidate in (runner, Path.home() / 'repos' / 'KIRBy'):
        f = candidate / 'tests' / 'alternative_data_noise_robustness.py'
        if f.exists():
            m = re.search(r'^NOISE_LEVELS = \[(.+?)\]', f.read_text(), re.M)
            if m:
                return len([x for x in m.group(1).split(',') if x.strip()])
    fail('could not find NOISE_LEVELS in the runner — pass a KIRBy checkout this '
         'host can see, or the fit count cannot be checked')
    return None


def check_array_ranges(text, per_script):
    """The range every sbatch line asks for, against the script it names.

    This is the one that costs cluster time: a range that is too narrow drops
    whole datasets, and does it without an error anywhere.
    """
    known = script_names()
    submitted, seen = set(), 0
    for line_no, line in enumerate(text.splitlines(), 1):
        names = re.findall(r'(unc_[a-z0-9_]+\.sh)', line)
        if not names:
            continue
        m = re.search(r'--array=(\d+)-(\d+)', line)
        if not m:
            continue
        lo, hi = int(m.group(1)), int(m.group(2))
        for name in names:
            seen += 1
            submitted.add(name)
            if name not in known:
                fail(f'RUNBOOK.md:{line_no} submits {name}, which the generator does '
                     f'not emit')
            elif (lo, hi) != (0, per_script - 1):
                fail(f'RUNBOOK.md:{line_no} submits {name} as --array={lo}-{hi}; it has '
                     f'{per_script} tasks, so the range is 0-{per_script - 1}')

    if seen == 0:
        fail('no sbatch --array= line found in RUNBOOK.md at all')
    for name, model in sorted(known.items()):
        if name not in submitted:
            fail(f'the generator emits {name} ({model}) and no sbatch line in '
                 f'RUNBOOK.md submits it — an operator following this file would '
                 f'never run that model')
    return seen


def main():
    text = RUNBOOK.read_text()
    summary = generator_summary()
    if summary is None:
        print('FAIL — the generator would not run; nothing could be checked')
        return 1

    head = re.search(r'Wrote \d+ array scripts, (\d+) tasks each', summary)
    per_script = int(head.group(1)) if head else 0

    check_conditions(text)
    check_counts(text, summary)
    ranges = check_array_ranges(text, per_script) if per_script else 0

    if FAILURES:
        print(f'FAIL — {len(FAILURES)} disagreement(s) between RUNBOOK.md and the '
              f'generator:\n')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    print(f'PASS — conditions, counts and {ranges} array range(s) all agree with '
          f'generate_scripts.py')
    return 0


if __name__ == '__main__':
    sys.exit(main())
