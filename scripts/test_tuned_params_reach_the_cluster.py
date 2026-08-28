#!/usr/bin/env python3
"""A generated job script must be able to deliver a tuned hyperparameter.

    python scripts/test_tuned_params_reach_the_cluster.py

WHAT THIS GUARDS
----------------
`scripts/test_tuned_params_reach_models.py` proves a tuned value reaches the
estimator once `--use-best-params` is on. This proves the flag is on.

Until 2026-08-28 it was not, in any script any generator wrote. All twelve places
`models/models.py` reads the tuned files sit behind that one flag, so the whole
chain was intact and disconnected at the last link: the sweep could finish, the
winners could be written to `results/master_tuned_hyperparameters.json`, the
decisions file could say USE_TUNED, and every job on the grid would still have
trained on the shared defaults. Nothing would have looked wrong. The rows would
have said `params_source=default` and no one reads that column expecting a
surprise.

This is a generated-artefact check, not a source check: it runs the generator
into a temporary directory and reads what it wrote, because the flag has to
survive the template, not merely appear in it.

It also refuses the spelling that does not work. `--use_best_params True` is not
an option: argparse declares `--use-best-params` with `action='store_true'`
(`process_and_train.py`), so the underscored form is unrecognised and the value
after it would be swallowed as a positional.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

GENERATORS = [
    (_ROOT / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py', ['--stage', '0']),
]

FLAG = '--use-best-params'
BAD_SPELLINGS = ('--use_best_params', '--use-best-params True',
                 '--use_best_params True')

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f'  FAIL  {msg}')


def ok(msg):
    print(f'  ok    {msg}')


def check_argparse_accepts_the_flag():
    """The flag this file insists on must be one process_and_train.py declares."""
    src = (_ROOT / 'scripts' / 'process_and_train.py').read_text()
    m = re.search(r"add_argument\(\s*'--use-best-params'\s*,\s*action='store_true'",
                  src)
    if not m:
        fail("process_and_train.py no longer declares --use-best-params as "
             "action='store_true'. If it moved or changed shape, every "
             "generated script's flag is now wrong and this check is stale.")
    else:
        ok("process_and_train.py declares --use-best-params (store_true)")


def check_generated_scripts():
    for gen, extra in GENERATORS:
        if not gen.exists():
            fail(f'{gen} does not exist')
            continue
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, str(gen), '--out-dir', tmp, *extra],
                capture_output=True, text=True, cwd=str(gen.parent))
            if proc.returncode != 0:
                fail(f'{gen.name} exited {proc.returncode}:\n'
                     f'{(proc.stdout + proc.stderr)[-800:]}')
                continue
            scripts = sorted(Path(tmp).glob('*.sh'))
            if not scripts:
                fail(f'{gen.name} wrote no scripts, so nothing was checked')
                continue

            without = [s.name for s in scripts if FLAG not in s.read_text()]
            if without:
                fail(f'{gen.name}: {len(without)} of {len(scripts)} generated '
                     f'script(s) never pass {FLAG}, so no tuned hyperparameter '
                     f'can reach the model. First few: {without[:4]}')
            else:
                ok(f'{gen.name}: all {len(scripts)} generated script(s) mention '
                   f'{FLAG}')

            # MENTIONING THE FLAG IS NOT PASSING IT.
            #
            # The QM9 template decides the flag once at task start and passes it
            # through $TUNED_FLAG, so the literal appears in an assignment. A
            # containment test is satisfied by that assignment even if the
            # variable is never referenced on the command line -- which is this
            # project's recurring failure: a check that searches the file as text
            # and passes for the wrong reason (RERUN_PLAN.md 2.23).
            #
            # So: find the invocation itself and require that the flag reaches it,
            # either literally or through a variable that is set to it.
            for s in scripts:
                check_flag_reaches_the_invocation(s)

            for s in scripts:
                text = s.read_text()
                for bad in BAD_SPELLINGS:
                    # The good spelling contains the bad one as a prefix, so
                    # only flag the bad one where it is not that.
                    for hit in re.finditer(re.escape(bad), text):
                        tail = text[hit.end():hit.end() + 1]
                        if bad == '--use_best_params' or tail in (' ', '\n'):
                            if bad == FLAG:
                                continue
                            fail(f'{s.name} uses {bad!r}. argparse declares '
                                 f'{FLAG} as store_true, so that spelling is '
                                 f'either unrecognised or swallows the next '
                                 f'token as a positional.')
                            break



def check_flag_reaches_the_invocation(script):
    """The flag must be ON THE COMMAND LINE, not merely somewhere in the file.

    Accepts the literal, or a shell variable that the script sets to the literal
    and then references in the call. Anything else means the tuned path is dead
    however many times the file names it.
    """
    text = script.read_text()
    m = re.search(r'python -u process_and_train\.py(.*?)\n\nstatus=', text, re.S)
    if not m:
        m = re.search(r'python -u process_and_train\.py(.*?)-f "\$OUT"', text, re.S)
    if not m:
        fail(f'{script.name}: no process_and_train.py invocation found, so '
             f'nothing could be checked')
        return
    call = m.group(1)

    if FLAG in call:
        return

    # A variable on the command line: $NAME or ${NAME}
    for var in set(re.findall(r'\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?', call)):
        # ...that the script assigns the flag to, anywhere above.
        if re.search(rf'^\s*{var}=.*{re.escape(FLAG)}', text, re.M):
            # And prove the two branches behave, by running the decision itself.
            _check_decision_runs_both_ways(script, var)
            return

    fail(f'{script.name}: {FLAG} appears in the file but never reaches the '
         f'process_and_train.py command line, directly or through a variable. '
         f'The tuned path is dead in this script.')


def _check_decision_runs_both_ways(script, var):
    """Execute the script's own decision block and assert both outcomes.

    The block tests for the two tuned files relative to the working directory the
    job runs in (`scripts/`), so it is exercised in a stub with and without them.
    """
    text = script.read_text()
    block = re.search(rf'({re.escape(var)}=""\nif .*?\nfi)\n', text, re.S)
    if not block:
        fail(f'{script.name}: {var} is used on the command line but its decision '
             f'block could not be located, so the two branches were not proven')
        return
    snippet = block.group(1)

    for present in (True, False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'scripts').mkdir()
            (root / 'results').mkdir()
            if present:
                for name in ('master_tuned_hyperparameters.json',
                             'hyperparameter_decisions.json'):
                    (root / 'results' / name).write_text('{}')
            proc = subprocess.run(
                # The block echoes a progress line; send it away so stdout
                # carries only the value being asserted.
                ['bash', '-c',
                 f'set -uo pipefail\n{{\n{snippet}\n}} >/dev/null 2>&1\n'
                 f'printf "%s" "${var}"'],
                capture_output=True, text=True, cwd=str(root / 'scripts'))
            got = proc.stdout.strip()
            want = FLAG if present else ''
            if got != want:
                fail(f'{script.name}: with the tuned files '
                     f'{"present" if present else "absent"} the decision block set '
                     f'{var}={got!r}, expected {want!r}')
                return
    ok(f'{script.name}: the flag is decided once at task start and both branches '
       f'behave')


def main():
    print('the flag argparse declares')
    check_argparse_accepts_the_flag()
    print('\nwhat the generators actually write')
    check_generated_scripts()
    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s)')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
