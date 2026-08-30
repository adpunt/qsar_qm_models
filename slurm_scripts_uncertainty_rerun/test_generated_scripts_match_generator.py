#!/usr/bin/env python3
"""The unc_*.sh sitting in this directory must be what the generator writes today.

WHY THIS EXISTS
---------------
The .sh are generated output and left git on 2026-08-28, on the argument that
they are rebuilt from the generator every time. Nothing checked that they had
been, on either side of the study, and this side is where it showed: on
2026-08-29 this directory held ten scripts written on 2026-08-28 at 08:57 from a
roster of ten models on four representations, while the generator writes four
scripts on three representations. Six of the ten name models the screen of
2026-08-28 dropped, and `sbatch unc_gp_hetero.sh` would still have queued one.

It is worse than a wasted queue slot, because two other things READ these files
rather than the generator. `preflight.sh` section 4b takes the run's condition
list off the first unc_*.sh it finds, and `merge_results.py` takes the expected
condition list off them too -- so a stale script does not merely run, it decides
what the coverage report expects.

WHAT IT CHECKS
--------------
Byte equality between each unc_*.sh on disk and the same file generated fresh,
plus the two set differences: a script on disk the generator no longer writes (a
retired model an operator could still submit), and a script the generator writes
that is not on disk (a model that would simply never run).

A directory with no unc_*.sh in it is NOT a failure -- that is a fresh clone, and
the runbook's step 1 regenerates them. It says so and passes.

    python slurm_scripts_uncertainty_rerun/test_generated_scripts_match_generator.py
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
    on_disk = {p.name: p for p in sorted(HERE.glob('unc_*.sh'))}

    with tempfile.TemporaryDirectory() as tmp:
        fresh_dir = Path(tmp)
        # No arguments: the runbook's step 1 is a bare `python generate_scripts.py`,
        # and every roster and narrowing decision lives inside the generator rather
        # than on its command line.
        done = subprocess.run(
            [sys.executable, str(HERE / 'generate_scripts.py'),
             '--out-dir', str(fresh_dir)],
            capture_output=True, text=True)
        if done.returncode != 0:
            print('FAIL — the generator would not run; nothing could be compared\n')
            print(done.stdout, done.stderr)
            return 1
        fresh = {p.name: p.read_bytes() for p in sorted(fresh_dir.glob('unc_*.sh'))}

        if not on_disk:
            print(f'note: no unc_*.sh in {HERE.name}, so there is nothing to have '
                  f'drifted. The generator writes {len(fresh)} of them; RUNBOOK.md '
                  f'step 1 is where they get written.')

        for name in sorted(set(on_disk) & set(fresh)):
            if on_disk[name].read_bytes() != fresh[name]:
                fail(f'{name} on disk is not what the generator writes today. '
                     f'preflight section 4b and merge_results.py both read the run '
                     f'design out of these files, so a stale one is not just a stale '
                     f'job — it moves what the coverage report expects.')
        for name in sorted(set(on_disk) - set(fresh)):
            fail(f'{name} is on disk and the generator does not write it. It names a '
                 f'model this run no longer includes, and an operator can still '
                 f'submit it. Delete it.')
        if on_disk:
            for name in sorted(set(fresh) - set(on_disk)):
                fail(f'the generator writes {name} and it is not on disk, so that '
                     f'model would simply not be submitted.')

    if FAILURES:
        print(f'FAIL — {len(FAILURES)} disagreement(s) between the unc_*.sh on disk '
              f'and generate_scripts.py:\n')
        for f in FAILURES:
            print(f'  - {f}')
        print('\n  Fix: `cd slurm_scripts_uncertainty_rerun && rm -f unc_*.sh && '
              'python generate_scripts.py`')
        return 1
    if not on_disk:
        print('PASS — nothing on disk to compare, and the generator runs clean')
        return 0
    print(f'PASS — all {len(on_disk)} unc_*.sh on disk are byte-identical to what '
          f'generate_scripts.py writes')
    return 0


if __name__ == '__main__':
    sys.exit(main())
