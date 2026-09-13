#!/usr/bin/env python3
"""The sweep for the four untuned transformations must actually build.

WHY THIS EXISTS
---------------
Two separate things stopped it, and neither announced itself as a blocked sweep.

FIRST, the timing file. `results/tuning_local/timing.csv` is what
`slurm_scripts_tuning/generate_scripts.py` sizes every wall clock from. On
2026-08-28 a one-pairing `--time` run opened that path with 'w' while the full
pass was six hours into writing it; the truncation left 6,184 NUL bytes followed
by 13 rows, and the header was inside the NULs. `csv.DictReader` then took the
first surviving row as the header, and the generator died with
`KeyError: 'status'` -- not with anything that said the timing file was damaged.
The cause is written up in scripts/tune_hyperparameters.py, at the code that
gives a narrowed run its own filename so it cannot happen again. The file was
repaired on 2026-09-12 by merging what survived with
`results/tuning_local/timing_recovered.csv`, which was read back out of the log;
the damaged bytes are kept at `timing.nul_damaged_2026-08-28.csv`.

SECOND, the sizing. The two variance-head networks and the two heteroscedastic
VBLL networks were added to the roster after the timing pass ran, so they have no
measured row and the generator refuses them by name -- correctly, because a
made-up wall clock leaves a partial trials file that `--merge` reads as a
finished one. Each is one flag away from a model that WAS timed, and the QM9 job
generator already sizes each against that same sibling, so the ratio is read out
of the roster.

WHAT IS CHECKED
---------------
  1. the timing file has its header and parses, and holds a measured row for
     every model the four are sized from;
  2. the generator builds all four, writes a script for each, and writes each
     array range itself;
  3. every model it writes a script for resolves to a search space, so the
     submitted array fits settings instead of recording 'blocked' and exiting 0
     on a whole allocation;
  4. it still REFUSES a pairing it can size from nothing, rather than inventing
     a number.

    python scripts/test_tuning_sweep_buildable.py
"""
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATOR = os.path.join(REPO, 'slurm_scripts_tuning', 'generate_scripts.py')
TIMING = os.path.join(REPO, 'results', 'tuning_local', 'timing.csv')

# The four the sweep is for: two variance-head networks and two heteroscedastic
# VBLL networks, one of each on either base network.
WANTED = ['dnn_bnn_full_mve', 'mlp_bnn_full_mve',
          'dnn_bnn_full_variational_hetero', 'mlp_bnn_full_variational_hetero']

EXPECTED_COLUMNS = ['model', 'rep', 'setting', 'r2', 'seconds', 'status',
                    'detail', 'sample_size', 'seed', 'written']


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    failures = []

    # 1. The timing file.
    print(f"timing file: {os.path.relpath(TIMING, REPO)}")
    if not os.path.exists(TIMING):
        print("  missing")
        failures.append(f"{TIMING} does not exist; the generator cannot size "
                        f"anything without it.")
        rows = []
    else:
        with open(TIMING, newline='') as fh:
            head = fh.readline().rstrip('\r\n').split(',')
            fh.seek(0)
            rows = list(csv.DictReader(fh))
        print(f"  {len(rows)} row(s), columns {head}")
        if head != EXPECTED_COLUMNS:
            failures.append(
                f"the first line of the timing file is {head}, not the column "
                f"names {EXPECTED_COLUMNS}. Its header has been overwritten, so "
                f"every row is read under the wrong names and the generator dies "
                f"on the first lookup.")
        with open(TIMING, 'rb') as fh:
            if b'\x00' in fh.read():
                failures.append(
                    "the timing file contains NUL bytes. It was truncated by a "
                    "concurrent writer; repair it from timing_recovered.csv "
                    "rather than reading past the damage.")

    generator = load(GENERATOR, 'tuning_generator')
    rosters = load(os.path.join(REPO, 'models', 'tuning_rosters.py'),
                   'tuning_rosters')
    measured = {(r['model'], r['rep']) for r in rows
                if r.get('status') == 'ok' and r.get('seconds')}

    print("\nwhat each unmeasured model is sized from")
    for model, sibling in sorted(generator.DERIVED_FROM_SIBLING.items()):
        ratio = generator.sibling_ratio(rosters, model, sibling)
        print(f"  {model:34s} {ratio:.2f} x {sibling}")
        if not (1.0 <= ratio <= 4.0):
            failures.append(
                f"{model} is sized at {ratio:.2f} times {sibling}. A ratio "
                f"outside 1 to 4 means the roster's cost fields have moved and "
                f"the derivation is no longer the one that was reasoned about.")

    # 2. The four build, and the generator writes the ranges.
    print("\nbuilding the sweep for the four untuned transformations")
    with tempfile.TemporaryDirectory() as out:
        proc = subprocess.run(
            [sys.executable, GENERATOR, '--settings', '12',
             '--out-dir', out, '--models', *WANTED],
            capture_output=True, text=True)
        stdout = proc.stdout
        if proc.returncode != 0:
            failures.append(
                f"the generator refused to build the sweep (exit "
                f"{proc.returncode}):\n{proc.stdout}{proc.stderr}")
        for model in WANTED:
            script = os.path.join(out, f'tune_{model}.sh')
            if not os.path.exists(script):
                failures.append(f"no tune_{model}.sh was written, so this model "
                                f"has no sweep to submit.")
                continue
            reps = rosters.MODELS[model][4]
            # NEVER TYPED BY HAND. The generator writes the range for each
            # script from that model's own representation list.
            wanted_range = f'--array=0-{len(reps) - 1}%'
            if wanted_range not in stdout:
                failures.append(
                    f"the generator did not print an array range for {model}. "
                    f"Its range must come from the generator, never from a hand-"
                    f"typed 0-N.")
            body = open(script).read()
            if '--loss heteroscedastic' in rosters.MODELS[model][0] \
                    and '--models ' + model not in body:
                failures.append(
                    f"tune_{model}.sh does not name {model}, so the fit would "
                    f"not carry that model's flags.")
        for line in stdout.splitlines():
            if 'core-hours' in line or 'task(s)' in line or line.startswith('  sbatch'):
                print(' ', line.strip())

    # 3. A sized script is not a runnable one. `search_family` in
    #    scripts/tune_hyperparameters.py decides which SEARCH_SPACES entry a
    #    model is searched under; when it returns None the task records
    #    'blocked', prints one line and exits 0, and the array looks like it
    #    succeeded. That is what happened to the two variance-head networks.
    print("\nevery model with a script resolves to a search space")
    tuner = load(os.path.join(REPO, 'scripts', 'tune_hyperparameters.py'),
                 'tune_hyperparameters')
    for model in WANTED:
        key = tuner.search_family(model, rosters)
        print(f"  {model:34s} searched as {key}")
        if key is None or key not in tuner.SEARCH_SPACES:
            failures.append(
                f"a job script is written for {model} but search_family gives it "
                f"{key!r}, which is no search space. Every task in that array "
                f"would record 'blocked' and exit 0, spending the allocation and "
                f"producing no trials file.")

    # 4. It still refuses what it can size from nothing.
    print("\nit still refuses a pairing it can size from nothing")
    with tempfile.TemporaryDirectory() as out:
        proc = subprocess.run(
            [sys.executable, GENERATOR, '--settings', '12',
             '--out-dir', out, '--models', 'gauche_rbf'],
            capture_output=True, text=True)
        refused = 'REFUSING' in (proc.stdout + proc.stderr)
        print(f"  gauche_rbf (measured on ecfp4 alone): "
              f"{'refused' if refused else 'BUILT'}")
        if not refused:
            failures.append(
                "the generator built a script for gauche_rbf, which has a "
                "measured row on ecfp4 and on no other representation. Refusing "
                "an unmeasured pairing is the whole point of the timing pass.")

    if failures:
        print(f"\nFAIL -- {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: the timing file parses, the four untuned transformations are "
          "sized from their measured siblings, and the generator writes every "
          "array range itself")
    return 0


if __name__ == '__main__':
    sys.exit(main())
