#!/usr/bin/env python3
"""Job scripts for the hyperparameter sweep, sized from measured seconds.

    python slurm_scripts_tuning/generate_scripts.py --settings 12

WHY THIS EXISTS
---------------
The sweep does not fit on the laptop, and the rule is that it does not get cut
down to make it fit. Measured 2026-08-27 with
`scripts/tune_hyperparameters.py --time` at the production sample size of
10,000 molecules: **one fit of each pairing at its default** already costs more
than three hours, before the neural families and the Gaussian processes. A
sweep is that cost multiplied by (settings + 1). So it goes to the cluster with
every model, every representation and every parameter range intact.

HOW IT IS SIZED
---------------
Not by guessing. `results/tuning_local/timing.csv` holds one measured fit per
pairing, at the same sample size and the same split the sweep will use, so each
task's wall clock is that pairing's own seconds x (settings + 1) x a headroom
factor. A pairing with no measured row **stops this generator by name** rather
than being given a number someone made up -- the whole point of the timing pass
was to stop that happening.

The quantile forest is the one exception and it is stated, not silent: it cannot
be fitted on the laptop at all (quantile_forest 1.4.1 against scikit-learn 1.3.2
raises "Invalid parameter 'monotonic_cst'"), so it has no measured row. The
cluster environment does not have that fault -- the roster audit fits it there
(RERUN_PLAN.md 2.8i) -- so its tasks are sized from the plain forest, which is
the same algorithm with more trees, scaled by the tree ratio in the shared spec.

ONE TASK = ONE PAIRING
----------------------
Every task tunes one model on one representation and writes its own
trials_<model>_<rep>.csv. A fixed output name would have every array task
writing the same path, and the last to finish would be the only one that
survived -- silently. `--tag` gives each its own; `--merge` reads them back and
names any pairing that produced no rows at all.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent

QSAR_DIR = '/data/stat-cadd/scat9264/qsar_qm_models'
TIMING = _ROOT / 'results' / 'tuning_local' / 'timing.csv'

# Headroom on the measured time. The cluster nodes are not this laptop and the
# search visits settings with more trees and more epochs than the default, so
# the measured fit is a floor rather than a typical case.
HEADROOM = 3.0

# The quantile forest has no measured row (see the docstring). 300 trees against
# the forest's 100, from models/model_defaults.py, and the quantile predict pass
# on top.
QRF_FROM_RF = 3.5

# Which partition a script names. The author confirmed 2026-08-28 that a 60-hour
# request is fine on `long`, so NGBoost is left as one array rather than split by
# representation to dodge its wall clock.
#
# 48 is an ASSUMPTION about where `medium` stops, not something read from the
# cluster. It only decides which partition the generated header suggests, and
# --medium-max-hours corrects it without touching anything else.
MEDIUM_MAX_HOURS = 48


def _rosters():
    spec = importlib.util.spec_from_file_location(
        'tuning_rosters', _ROOT / 'models' / 'tuning_rosters.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def measured_seconds():
    """One measured fit per pairing, from the timing pass."""
    if not TIMING.exists():
        raise SystemExit(
            f'{TIMING} does not exist. Run\n'
            f'    OMP_NUM_THREADS=1 python scripts/tune_hyperparameters.py --time\n'
            f'first. Sizing a cluster request without it means inventing the '
            f'numbers, which is what the timing pass exists to prevent.')
    out = {}
    with open(TIMING) as fh:
        for row in csv.DictReader(fh):
            if row['status'] == 'ok' and row['seconds']:
                for rep in _rep_spellings(row['rep']):
                    out[(row['model'], rep)] = float(row['seconds'])
    return out


# The descriptor vector is being renamed from `continuous_pdv` to `pdv`
# (RERUN_PLAN.md 5.6), and the timing pass was started before the rename landed
# in the job generator. Its rows therefore carry the old spelling while the
# roster now carries the new one, and a plain lookup would report the pairing as
# unmeasured and refuse to size it. Both spellings mean the CONTINUOUS vector --
# the binary variant that used to own the short name is being deleted, not kept
# -- so a measured row is filed under both.
_PDV_SPELLINGS = ('pdv', 'continuous_pdv')


def _rep_spellings(rep):
    return _PDV_SPELLINGS if rep in _PDV_SPELLINGS else (rep,)


SCRIPT = '''#!/bin/bash
# ============================================================================
# Hyperparameter sweep: {model} on {n_rep} representation(s).
#
#   array task -> representation
#   each task  -> 1 default fit + {settings} random settings, scored on the
#                 VALIDATION split of one scaffold split of {sample_size}
#                 molecules. No folds. Clean labels.
#
# Wall clock is {hours}h on `{partition}`, from the MEASURED seconds for this model's slowest
# representation ({slowest_rep}, {slowest_s:.0f}s per fit) x {n_fits} fits x {headroom}
# headroom. Nothing here is a guess; see results/tuning_local/timing.csv.
#
# --account and --partition are LIVE STATE; pass them at submit time:
#   sbatch --account=stat-cadd --partition={partition} --array=0-{last}%{throttle} {script_name}
#
# Afterwards, and NOT optional -- a winner picked on the split that chose it
# proves nothing (RERUN_PLAN.md 5.7i):
#   python scripts/tune_hyperparameters.py --merge
#   python scripts/tune_hyperparameters.py --confirm
#   python scripts/tune_hyperparameters.py --write-master --margin 0.01
#   python scripts/confirm_tuned_on_validation_datasets.py --run
#   python scripts/confirm_tuned_on_validation_datasets.py --prune
# ============================================================================
#SBATCH --job-name=tune_{model}
#SBATCH --output=tune_{model}_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours}:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

cd {qsar_dir}
. setup.sh

# An unactivated task falls through to the system Anaconda, which has none of
# the model packages: the job runs, finds nothing to do, and writes no rows.
# That is what happened to 12822693 and 12822694 (RERUN_PLAN.md 2.8d).
if [ -z "${{CONDA_PREFIX:-}}" ]; then
    echo "ERROR: setup.sh did not activate an environment (CONDA_PREFIX unset)."
    exit 2
fi
PY_PATH="$(command -v python)"
case "$PY_PATH" in
    "$CONDA_PREFIX"/*) : ;;
    *) echo "ERROR: python is $PY_PATH, outside $CONDA_PREFIX"; exit 2 ;;
esac
if [ "$(basename "$CONDA_PREFIX")" != "env_test" ]; then
    echo "ERROR: active environment is $(basename "$CONDA_PREFIX"), not env_test"
    exit 2
fi
echo "=== interpreter: $PY_PATH  (CONDA_PREFIX=$CONDA_PREFIX)"

# Private scratch per task: joblib, matplotlib and the HuggingFace cache all
# honour TMPDIR, and a fixed-name temp file shared by concurrent array tasks is
# the defect class that produced the config.json race (RERUN_PLAN.md 2.8a).
export TMPDIR="${{TMPDIR:-/tmp}}/tune_${{SLURM_JOB_ID:-$$}}_${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# The interpreter has to be able to BUILD what this job is about to ask for.
# Seconds here against a whole wasted allocation later.
python scripts/check_environment.py --models {check_model} || {{
    echo "ERROR: this interpreter cannot build model '{model}'."
    exit 2
}}

REPS=({reps})
n_tasks=${{#REPS[@]}}
i="${{SLURM_ARRAY_TASK_ID:-0}}"
if [ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
rep="${{REPS[$i]}}"

echo "=== task $i: model={model} rep=$rep"
echo "=== settings: {settings} random, plus the shared default"
echo "=== started: $(date)"

# One output file per task. Merged afterwards by --merge, which also names any
# pairing that produced no rows.
python -u scripts/tune_hyperparameters.py --sweep \\
    --settings {settings} \\
    --sample-size {sample_size} \\
    --seed {seed} \\
    --models {model} \\
    --reps "$rep" \\
    --tag "{model}_$rep"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
'''


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--settings', type=int, required=True,
                    help='Random settings per pairing, on top of the default. '
                         'This is decision 7(a) in RERUN_PLAN.md 4 and the cost '
                         'is linear in it.')
    ap.add_argument('--sample-size', type=int, default=10000,
                    help='Molecules before the 80/10/10 scaffold split. Must '
                         'match the timing pass, or the sizing means nothing.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--cpus', type=int, default=8)
    ap.add_argument('--mem', default='32G')
    ap.add_argument('--throttle', type=int, default=6)
    ap.add_argument('--medium-max-hours', type=int, default=MEDIUM_MAX_HOURS,
                    help=f'Longest request the medium partition takes; anything '
                         f'above it is written for long (default '
                         f'{MEDIUM_MAX_HOURS}, an assumption, not read from the '
                         f'cluster).')
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--qsar-dir', default=QSAR_DIR)
    ap.add_argument('--out-dir', default=str(_HERE))
    args = ap.parse_args()

    rosters = _rosters()
    measured = measured_seconds()
    n_fits = args.settings + 1

    chosen = [m for m in rosters.MODELS
              if not args.models or m in args.models]
    unknown = [m for m in (args.models or []) if m not in rosters.MODELS]
    if unknown:
        raise SystemExit(f'unknown model label(s) {unknown}. '
                         f'Known: {sorted(rosters.MODELS)}')

    out_dir = Path(args.out_dir)
    written, total_core_hours, unmeasured = [], 0.0, []

    for model in chosen:
        reps = rosters.MODELS[model][4]
        seconds = {}
        for rep in reps:
            if (model, rep) in measured:
                seconds[rep] = measured[(model, rep)]
            elif model == 'qrf' and ('rf', rep) in measured:
                seconds[rep] = measured[('rf', rep)] * QRF_FROM_RF
            else:
                unmeasured.append((model, rep))
        if len(seconds) != len(reps):
            continue

        slowest_rep = max(seconds, key=seconds.get)
        slowest_s = seconds[slowest_rep]
        hours = max(1, int((slowest_s * n_fits * HEADROOM) // 3600) + 1)
        # Every task in the array gets the array's wall clock, so the request is
        # the slowest representation's. The cost estimate below is the honest
        # per-representation sum, not that number times the task count.
        total_core_hours += sum(seconds.values()) * n_fits / 3600

        partition = 'medium' if hours <= args.medium_max_hours else 'long'
        name = f'tune_{model}.sh'
        (out_dir / name).write_text(SCRIPT.format(
            model=model,
            check_model=model,
            reps=' '.join(f'"{r}"' for r in reps),
            n_rep=len(reps),
            last=len(reps) - 1,
            throttle=args.throttle,
            settings=args.settings,
            sample_size=args.sample_size,
            seed=args.seed,
            cpus=args.cpus,
            mem=args.mem,
            hours=hours,
            headroom=HEADROOM,
            n_fits=n_fits,
            slowest_rep=slowest_rep,
            slowest_s=slowest_s,
            script_name=name,
            partition=partition,
            qsar_dir=args.qsar_dir,
        ))
        os.chmod(out_dir / name, 0o755)
        written.append((model, len(reps), hours, partition, slowest_rep, slowest_s))

    if unmeasured:
        raise SystemExit(
            'REFUSING to write scripts for pairings with no measured fit:\n' +
            '\n'.join(f'    {m} x {r}' for m, r in unmeasured) +
            '\n\nFinish the timing pass first:\n'
            '    OMP_NUM_THREADS=1 python scripts/tune_hyperparameters.py --time\n'
            'Sizing these by hand is exactly what the timing pass exists to '
            'prevent, and a task that runs out of wall clock leaves a partial '
            'trials file that --merge would read as a finished one.')

    print(f'{len(written)} script(s) in {out_dir}')
    print(f'  {args.settings} settings + 1 default = {n_fits} fits per pairing')
    print(f'  sample size {args.sample_size}, seed {args.seed}\n')
    for model, n_rep, hours, partition, srep, ss in written:
        print(f'  {model:28s} {n_rep} task(s)  --time={hours}:59:00  '
              f'--partition={partition:6s} (slowest {srep} at {ss:.0f}s/fit)')
    print(f'\n  total compute across every task: '
          f'{total_core_hours:.1f} core-hours before headroom')
    long_ones = [m for m, _, _, p, _, _ in written if p == 'long']
    if long_ones:
        print(f'  needs the long partition: {", ".join(long_ones)}')
    print(f'\nSubmit each with the partition named in its own header:')
    for model, n_rep, hours, partition, _, _ in written:
        print(f'  sbatch --account=stat-cadd --partition={partition} '
              f'--array=0-{n_rep - 1}%{args.throttle} tune_{model}.sh')


if __name__ == '__main__':
    main()
