#!/usr/bin/env python3
"""The val_*.sh sitting in this directory must be what the generator writes today.

WHY THIS EXISTS
---------------
The same check already guards `slurm_scripts_qm9_rerun` and
`slurm_scripts_uncertainty_rerun`. This directory had no such check and it is
the one that had actually drifted: on 2026-08-30 it held 92 `val_*.sh`
against 129 the generator writes, and the 42 missing ones were EVERY Avalon and
EVERY ChemBERTa pairing -- the two representations added to the settled set on
2026-08-26. Submitting what was on disk would have run four of the six settled
representations on the three laboratory datasets and left no error behind,
because a script that does not exist cannot fail.

This directory is also where the failure has form: three scripts sat here for
weeks with the dead `micromamba` lines and no activation guard, looking as
sanctioned as their 85 siblings, because the generator only overwrites the
scripts it still emits and leaves the rest (RERUN_PLAN.md 13.2, chat D).

WHAT IT CHECKS
--------------
Byte equality between each `val_*.sh` on disk and the same file generated fresh,
plus the two set differences: a script on disk the generator no longer writes (a
pairing an operator could still submit), and a script the generator writes that
is not on disk (a pairing that would simply never run).

`submit_all.sh` and the three `check_*.sh` are not generated per pairing and are
not compared here; `submit_all.sh` IS compared, because the generator writes it
and it is what an operator runs.

A directory with no `val_*.sh` in it is NOT a failure -- that is a fresh clone.

    python slurm_scripts_validation_rerun/test_generated_scripts_match_generator.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def main():
    on_disk = {p.name: p for p in sorted(HERE.glob('val_*.sh'))}
    sub = HERE / 'submit_all.sh'
    if sub.exists():
        on_disk[sub.name] = sub

    with tempfile.TemporaryDirectory() as tmp:
        fresh_dir = Path(tmp)
        # No arguments: the roster and every narrowing decision lives inside the
        # generator rather than on its command line.
        done = subprocess.run(
            [sys.executable, str(HERE / 'generate_scripts.py'),
             '--out-dir', str(fresh_dir)],
            capture_output=True, text=True)
        if done.returncode != 0:
            print('FAIL — the generator would not run; nothing could be compared\n')
            print(done.stdout, done.stderr)
            return 1
        fresh = {p.name: p.read_bytes()
                 for p in sorted(fresh_dir.glob('*.sh'))
                 if p.name.startswith('val_') or p.name == 'submit_all.sh'}

        if not on_disk:
            print(f'note: no val_*.sh in {HERE.name}, so there is nothing to have '
                  f'drifted. The generator writes {len(fresh)} of them.')

        for name in sorted(set(on_disk) & set(fresh)):
            if on_disk[name].read_bytes() != fresh[name]:
                fail(f'{name} on disk is not what the generator writes today.')
        for name in sorted(set(on_disk) - set(fresh)):
            fail(f'{name} is on disk and the generator does not write it. It names a '
                 f'pairing this run no longer includes, and an operator can still '
                 f'submit it. Delete it.')
        if on_disk:
            missing = sorted(set(fresh) - set(on_disk))
            for name in missing:
                fail(f'the generator writes {name} and it is not on disk, so that '
                     f'pairing would simply not be submitted.')

    if FAILURES:
        print(f'FAIL — {len(FAILURES)} disagreement(s) between the val_*.sh on disk '
              f'and generate_scripts.py:\n')
        for f in FAILURES[:12]:
            print(f'  - {f}')
        if len(FAILURES) > 12:
            print(f'  ... and {len(FAILURES) - 12} more')
        print('\n  Fix: `cd slurm_scripts_validation_rerun && rm -f val_*.sh && '
              'python generate_scripts.py`')
        return 1
    if not on_disk:
        print('PASS — nothing on disk to compare, and the generator runs clean')
        return 0
    print(f'PASS — all {len(on_disk)} generated .sh on disk are byte-identical to '
          f'what generate_scripts.py writes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
