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

THE SCRIPTS ARE NO LONGER ALL THE SAME LENGTH (2026-08-29)
----------------------------------------------------------
The first version of this file read one number, "N tasks each", out of the
generator's summary and compared every `--array=` line in the runbook against
it. The screen of 2026-08-28 then narrowed VBLL to ChemBERTa alone, so the
roster became three scripts of 63 tasks and one of 21, the summary stopped
printing "tasks each" at all, and this check failed with "could not read the
generator's own summary" -- which is the check going blind, not the runbook
being right. Preflight section 0b calls this file and aborts on a non-zero exit,
so for as long as that lasted the lab queue was blocked on an unreadable
message.

Every range is now read PER SCRIPT out of the generator's own tier listing, so a
roster where one model runs fewer representations than the others is checked
rather than being the thing that breaks the checker.

WHAT IT CHECKS
--------------
  * the condition table names exactly the conditions the array runs
  * the script count, from the generator's own printed summary
  * the total task count and the total fit count
  * EVERY `--array=A-B` line, against the task count of the SCRIPT IT NAMES --
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


def generator_ranges(summary):
    """{script filename: (model, n_tasks)} out of the generator's tier listing.

    The listing is the generator's own answer to "how long is this array", the
    same string an operator reads off the screen, so the check and the operator
    cannot be looking at different numbers.
    """
    out = {}
    for name, model, lo, hi in re.findall(
            r'(unc_[a-z0-9_]+\.sh)\s+(\S+)\s+--array=(\d+)-(\d+)', summary):
        if int(lo) != 0:
            fail(f'the generator prints {name} as --array={lo}-{hi}; every array is '
                 f'expected to start at 0')
        out[name] = (model, int(hi) + 1)
    return out


def check_counts(text, summary, ranges):
    """Script count, total tasks and total fits.

    There is deliberately no "tasks each" check any more: with VBLL narrowed to
    one representation the scripts have two different lengths, and a single
    figure for all of them is exactly what mis-submits.
    """
    # The generator also writes submit_all.sh, so the line reads "N array scripts +
    # submit_all.sh, M tasks total". Match the counts, not the prose between them.
    head = re.search(r'Wrote (\d+) array scripts[^,]*, ([\d,]+) tasks total', summary)
    if not head:
        fail(f"could not read the generator's own summary:\n{summary}")
        return
    n_scripts = int(head.group(1))
    total = int(head.group(2).replace(',', ''))

    if n_scripts != len(ranges):
        fail(f'the generator says it wrote {n_scripts} scripts and lists {len(ranges)} '
             f'in its tier breakdown: {sorted(ranges)}')
    listed_total = sum(n for _, n in ranges.values())
    if listed_total != total:
        fail(f'the generator says {total} tasks in total, and its per-script ranges '
             f'add up to {listed_total}')

    stated = {int(n) for n in re.findall(r'(\d+)\s+array\s+scripts', text)}
    if n_scripts not in stated:
        fail(f'the generator emits {n_scripts} array scripts; the runbook states '
             f'{sorted(stated) or "no such figure"}')

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


def check_array_ranges(text, ranges):
    """The range every sbatch line asks for, against the script it names.

    This is the one that costs cluster time: a range that is too narrow drops
    whole datasets, and does it without an error anywhere. Each script is
    checked against ITS OWN length -- VBLL runs one representation and the rest
    run three, so 0-62 on the VBLL line and 0-20 on a tier 1 line are both
    wrong, in opposite directions.
    """
    submitted, seen = set(), 0
    for line_no, line in enumerate(text.splitlines(), 1):
        if not line.lstrip().startswith('sbatch'):
            continue
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
            if name not in ranges:
                fail(f'RUNBOOK.md:{line_no} submits {name}, which the generator does '
                     f'not emit')
                continue
            model, n_tasks = ranges[name]
            if (lo, hi) != (0, n_tasks - 1):
                fail(f'RUNBOOK.md:{line_no} submits {name} ({model}) as '
                     f'--array={lo}-{hi}; it has {n_tasks} tasks, so the range is '
                     f'0-{n_tasks - 1}')

    if seen == 0:
        fail('no sbatch --array= line found in RUNBOOK.md at all')
    for name, (model, _) in sorted(ranges.items()):
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

    emitted = generator_ranges(summary)
    if not emitted:
        print('FAIL — the generator printed no per-script --array= ranges; nothing '
              'could be checked. Its tier listing is where this file reads them.')
        return 1

    check_conditions(text)
    check_counts(text, summary, emitted)
    ranges = check_array_ranges(text, emitted)

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
