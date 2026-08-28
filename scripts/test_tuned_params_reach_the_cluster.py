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
                ok(f'{gen.name}: all {len(scripts)} generated script(s) pass '
                   f'{FLAG}')

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
