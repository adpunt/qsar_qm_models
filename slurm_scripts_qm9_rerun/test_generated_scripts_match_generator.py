#!/usr/bin/env python3
"""The .sh sitting in this directory must be what the generator writes today.

WHY THIS EXISTS
---------------
The .sh are generated output and left git on 2026-08-28, on the argument that
they are rebuilt from the generator every time. Nothing checked that they had
been. The cluster keeps its own copies at these paths, and a copy written a week
ago looks exactly like a copy written this morning -- same name, same shape,
`sbatch` accepts it, every task it starts succeeds.

Measured on the last set that WAS committed, against a generation from the
current source, the drift was not cosmetic:

  * `--oof-folds 5` was missing from all 16 scripts whose model reports an
    uncertainty. Without it a QM9 uncertainty row is a TEST row, and a QM9 test
    label is never corrupted by design -- so the question "does the uncertainty
    find the corrupted labels?" had no data behind it at all.
  * the representation was named `continuous_pdv`, which the reader now refuses
    by name, so every task would have died at argument parsing.
  * the tuned-hyperparameter block was absent, so a task could not have picked
    up `results/master_tuned_hyperparameters.json` even once it exists.
  * the two start-up refusals were absent: the one for an environment setup.sh
    would not repair, and the one for a missing `data/QM9/processed/data_v3.pt`
    that stops 294 tasks building the same cache into the same path at once.
  * three models had no script at all.

The wall clock moved with the fits -- 4:59 to 23:59 on the cross-fitted
scripts -- so a stale script would also have been killed part way through.

WHAT IT CHECKS
--------------
Byte equality between each .sh on disk and the same file generated fresh, plus
the two set differences: a script on disk the generator no longer writes (a
retired model an operator could still submit), and a script the generator writes
that is not on disk.

A directory with no .sh in it is NOT a failure -- that is a fresh clone, and the
runbook's step 4 regenerates them. It says so and passes.

    python slurm_scripts_qm9_rerun/test_generated_scripts_match_generator.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNBOOK = HERE / 'RUNBOOK.md'

# The generator refuses stage 1 under a wall-clock ceiling it cannot meet, so
# the runbook submits it with an explicit one. These are the runbook's own
# commands, and the first check below is that they are still its own commands --
# a test that invented its own arguments would compare against scripts nobody
# generates.
STAGE_COMMANDS = {
    '0': ['--stage', '0'],
    '1': ['--stage', '1', '--max-hours', '720'],
}

FAILURES = []


def fail(msg):
    FAILURES.append(msg)


def check_runbook_still_says_this():
    text = RUNBOOK.read_text()
    for stage, args in sorted(STAGE_COMMANDS.items()):
        wanted = 'generate_scripts.py ' + ' '.join(args)
        if wanted not in text:
            fail(f'this file regenerates stage {stage} as `{wanted}`, and RUNBOOK.md '
                 f'does not contain that command. One of the two is out of date, and '
                 f'the comparison below would be against scripts nobody generates.')


def generate(stage, args, into):
    done = subprocess.run(
        [sys.executable, str(HERE / 'generate_scripts.py'), *args, '--out-dir', str(into)],
        capture_output=True, text=True)
    if done.returncode != 0:
        fail(f'the generator refused stage {stage} (`{" ".join(args)}`):\n'
             f'{done.stdout}\n{done.stderr}')
        return False
    return True


def main():
    on_disk = {p.name: p for p in sorted(HERE.glob('*.sh'))}

    check_runbook_still_says_this()

    with tempfile.TemporaryDirectory() as tmp:
        fresh_dir = Path(tmp)
        for stage, args in sorted(STAGE_COMMANDS.items()):
            if not generate(stage, args, fresh_dir):
                print('FAIL — the generator would not run; nothing could be compared')
                return 1
        fresh = {p.name: p.read_bytes() for p in sorted(fresh_dir.glob('*.sh'))}

        if not on_disk:
            print(f'note: no .sh in {HERE.name}, so there is nothing to have drifted. '
                  f'The generator writes {len(fresh)} of them; RUNBOOK.md step 4 is '
                  f'where they get written.')

        for name in sorted(set(on_disk) & set(fresh)):
            if on_disk[name].read_bytes() != fresh[name]:
                fail(f'{name} on disk is not what the generator writes today. '
                     f'Regenerate before submitting — `sbatch` cannot tell the '
                     f'difference and neither can the job.')
        for name in sorted(set(on_disk) - set(fresh)):
            fail(f'{name} is on disk and the generator does not write it. It is a '
                 f'model or a stage this run no longer includes, and an operator can '
                 f'still submit it. Delete it.')
        if on_disk:
            for name in sorted(set(fresh) - set(on_disk)):
                fail(f'the generator writes {name} and it is not on disk, so that '
                     f'model would simply not be submitted.')

    if FAILURES:
        print(f'FAIL — {len(FAILURES)} disagreement(s) between the .sh on disk and '
              f'generate_scripts.py:\n')
        for f in FAILURES:
            print(f'  - {f}')
        return 1
    if not on_disk:
        print('PASS — nothing on disk to compare, and the generator runs clean')
        return 0
    print(f'PASS — all {len(on_disk)} .sh on disk are byte-identical to what '
          f'generate_scripts.py writes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
