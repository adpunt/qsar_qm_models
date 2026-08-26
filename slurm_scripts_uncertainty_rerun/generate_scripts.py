#!/usr/bin/env python3
"""Generate the SLURM job-array scripts for the uncertainty re-run.

WHAT THIS RUN IS FOR
--------------------
Two questions, one set of jobs:

  (A) Do the molecules whose labels were corrupted come back as the uncertain
      ones?  Measured on TRAINING molecules, scored OUT-OF-FOLD so that no
      molecule is judged by a model that fitted its own corrupted label.
      Needs  --oof-folds N.

  (B) Does the model learn WHERE the data is unreliable?  Measured on TEST
      molecules against the noise scale their region of the label distribution
      receives.  Five of the six strategies corrupt some molecules far more
      than others, so there is a pattern to learn; Gaussian ('legacy') hits
      every molecule equally and is therefore the control.
      Needs  --unc-strategies all.

DESIGN NOTES
------------
* One array TASK per (dataset, representation, strategy); one SCRIPT per model.
* Every task writes to its OWN --results-root.  The pipeline merges results by
  read-modify-write, so two concurrent tasks sharing a directory would race and
  silently lose rows.  merge_results.py stitches them back together afterwards.
* Cross-fitting is applied only to models that emit a per-molecule uncertainty
  (the pipeline enforces this too, via UNCERTAINTY_MODELS).
* GP must be told which reps to run on (--gp-reps); it defaults to PDV only.
"""
import argparse
from pathlib import Path

# Models that emit a per-molecule uncertainty. Must match UNCERTAINTY_MODELS in
# KIRBy/tests/alternative_data_noise_robustness.py.
MODELS = {
    # name          : (tier, cpus, mem,  hours, note)
    # Memory matches the working reference (slurm_scripts_validation_rerun uses
    # 128G for these same models on these same datasets); this run additionally
    # holds the per-molecule uncertainty frames in memory, so do not go lower.
    # Wall times are deliberately generous: the out-of-fold pass multiplies the
    # fit count by (1 + oof_folds) and nothing here has been timed on ARC.
    'QRF':            (1, 8,  '128G', 36, 'quantile spread; strongest error-ranker in existing results'),
    'NGBoost':        (1, 8,  '128G', 47, '500 estimators; slowest of the tree models by far'),
    'GP':             (1, 8,  '128G', 47, 'gauche ExactGP, RBF kernel, uncapped (GP_MAX_N from the shared model spec)'),
    'BNN-Full':       (2, 8,  '128G', 47, '100 stochastic forward passes'),
    'VBLL-Full':      (2, 8,  '128G', 47, '100 stochastic forward passes'),
    'MLP-BNN-Full':   (2, 8,  '128G', 47, '100 stochastic forward passes'),
    'MLP-VBLL-Full':  (2, 8,  '128G', 47, '100 stochastic forward passes'),
}
DATASETS = ['logd', 'caco2', 'herg_ki']
REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']
STRATEGIES = ['legacy', 'outlier', 'quantile', 'hetero', 'threshold', 'valprop']

KIRBY_DIR = '/data/stat-cadd/scat9264/KIRBy'
QSAR_DIR = '/data/stat-cadd/scat9264/qsar_qm_models'
RESULTS_ROOT = 'results/uncertainty_rerun'

TEMPLATE = '''#!/bin/bash
# ============================================================================
# Uncertainty re-run — model: {model}
# {note}
# ============================================================================
# Array task -> (dataset, representation, strategy).
#   {n_ds} datasets x {n_rep} reps x {n_st} strategies = {n_tasks} tasks
#
# --account and --partition are LIVE STATE: pass them at submit time. Confirm
# with  bash tests/slurm_scripts/where_to_submit.sh  in the KIRBy repo first.
#
#   sbatch --account=<acct> --partition=medium --array=0-{last}%{throttle} \\
#          {script_name}
#
# Every task writes its own --results-root, so tasks never race on a shared
# file. Run merge_results.py afterwards.
#
# Resubmit only the failed indices, e.g.:
#   sbatch --account=<acct> --partition=medium --array=3,17,40 {script_name}
# ============================================================================
#SBATCH --job-name=unc_{jobslug}
#SBATCH --output=unc_{jobslug}_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours}:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

cd {kirby_dir}
. {qsar_dir}/setup.sh

# Activation is not optional. micromamba has never worked on this cluster, so
# the `export MAMBA_EXE=...` lines that used to sit above `. setup.sh` pointed
# at a file that does not exist -- and nothing checked. setup.sh falls through
# to its conda branch; if that also fails, the task carries on in the system
# Anaconda at /apps/system/..., which has no gpytorch, no quantile_forest and
# no ngboost. The job then runs, finds nothing to do, and writes no rows. That
# is what happened to 12822693 and 12822694 (RERUN_PLAN.md section 2.8d).
if [ -z "${{CONDA_PREFIX:-}}" ]; then
    echo "ERROR: setup.sh did not activate an environment (CONDA_PREFIX unset)."
    exit 2
fi
PY_PATH="$(command -v python)"
case "$PY_PATH" in
    "$CONDA_PREFIX"/*) : ;;
    *)
        echo "ERROR: python is $PY_PATH, which is not inside the activated"
        echo "       environment ($CONDA_PREFIX)."
        case "$PY_PATH" in
            /apps/system/*)
                echo "       That is the system Anaconda. It has no gpytorch, no"
                echo "       quantile_forest and no ngboost, so this job would run,"
                echo "       find nothing to do, and write no rows." ;;
        esac
        exit 2 ;;
esac
# The test above cannot fail once setup.sh has run: setup.sh:83 prepends
# $CONDA_PREFIX/bin to PATH, so python resolves inside the prefix whichever
# environment was activated. It still catches an unset CONDA_PREFIX, which is
# the case that actually bit us -- but on its own it would pass a task that
# activated the WRONG environment, so the name is checked too. This literal
# must track ENV_NAME in setup.sh.
if [ "$(basename "$CONDA_PREFIX")" != "env_test" ]; then
    echo "ERROR: the active environment is $(basename "$CONDA_PREFIX"), not env_test"
    echo "       (CONDA_PREFIX=$CONDA_PREFIX). setup.sh activated the wrong one."
    exit 2
fi
echo "=== interpreter: $PY_PATH  (CONDA_PREFIX=$CONDA_PREFIX)"

# A private scratch directory per task.
#
# Hygiene, not a fix for any known defect here: joblib, matplotlib, numba and
# the HuggingFace cache all honour TMPDIR, and a fixed-name temp file shared by
# concurrent array tasks is the defect class that produced the config.json race.
# This closes the whole class cheaply.
#
# It does NOT fix the one instance found while testing: keopscore (gpytorch ->
# pykeops) hardcodes /tmp/compiler_version.txt and /tmp/brew_prefix.txt, writing
# and deleting them during import, so two simultaneous imports race and one dies
# with FileNotFoundError. TMPDIR is ignored by that code. It is guarded by
# platform.system() == "Darwin", so it cannot fire on this cluster -- it is a
# macOS-only problem, and it is why the two-task half of
# scripts/test_config_isolation.py skips on a laptop.
export TMPDIR="${{TMPDIR:-/tmp}}/qsar_${{SLURM_JOB_ID:-$$}}_${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT


# Guarded: under `set -u` an unset CONDA_PREFIX aborts the shell here, before
# python is ever reached.
export LD_LIBRARY_PATH="${{CONDA_PREFIX:-}}/lib:${{LD_LIBRARY_PATH:-}}"

cd tests

DATASETS=({datasets})
REPS=({reps})
STRATS=({strategies})

n_st=${{#STRATS[@]}}
n_rep=${{#REPS[@]}}
n_tasks=$(( ${{#DATASETS[@]}} * n_rep * n_st ))

# Guard: under `set -u` an unset SLURM_ARRAY_TASK_ID would abort with a cryptic
# message. This makes a non-array invocation run task 0 and say so.
i="${{SLURM_ARRAY_TASK_ID:-0}}"
if [ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
st="${{STRATS[$(( i % n_st ))]}}"
rep="${{REPS[$(( (i / n_st) % n_rep ))]}}"
ds="${{DATASETS[$(( i / (n_st * n_rep) ))]}}"

# One directory per task — no cross-task write races.
rep_slug=$(echo "$rep" | tr 'A-Z' 'a-z' | tr -d '-')
OUT="{results_root}/{model_slug}__${{ds}}__${{rep_slug}}__${{st}}"

if [ -z "${{SLURM_JOB_PARTITION:-}}" ]; then
    echo "ERROR: no partition. Submit with --partition=medium (see RUNBOOK step 4)."; exit 2
fi

echo "=== task $i: model={model} dataset=$ds rep=$rep strategy=$st"
echo "=== out: $OUT"
echo "=== started: $(date)"

{gp_line}python -u alternative_data_noise_robustness.py \\
    --datasets "$ds" \\
    --models "{model}" \\
    --reps "$rep" \\
    --strategies "$st" \\
    --unc-strategies all \\
    --oof-folds {oof} \\
    {extra_args}{gp_args}--results-root "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--oof-folds', type=int, default=5,
                    help='Cross-fitting folds for the training-set uncertainty (default 5). '
                         'Cost is (1 + this) fits per noise level.')
    ap.add_argument('--oof-outer-folds', type=int, default=None,
                    help='Cross-fit only the first N of the 5 scaffold folds. All 5 folds are '
                         'trained regardless, so TEST-side uncertainty is free for all of them; '
                         'this only limits the expensive out-of-fold TRAINING pass. '
                         '1 cuts the added cost about threefold.')
    ap.add_argument('--threshold-quantile', type=float, default=None,
                    help='Make threshold noise cut at quantiles of the training labels instead '
                         'of the absolute +/-1.0. Without this, threshold is CONSTANT on hERG '
                         '(every pChEMBL value clears 1.0) and that arm cannot answer question B. '
                         'Turning it on changes the injected noise, so those results are not '
                         'comparable with runs made without it.')
    ap.add_argument('--drop-strategies', nargs='+', default=[],
                    help='Strategies to leave out. Note heteroscedastic and value-proportional '
                         'rank molecules IDENTICALLY (Spearman 1.000), so for the uncertainty '
                         'questions they are one arm: dropping hetero reclaims 84 tasks and '
                         'loses no rank information.')
    ap.add_argument('--throttle', type=int, default=6,
                    help='Max concurrent tasks per array (the %%N in --array).')
    ap.add_argument('--out-dir', default=str(Path(__file__).parent))
    args = ap.parse_args()

    strategies = [x for x in STRATEGIES if x not in set(args.drop_strategies)]
    if args.drop_strategies:
        print(f"Dropping strategies: {args.drop_strategies} -> {len(strategies)} remain")
    out = Path(args.out_dir)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
    n_tasks = len(DATASETS) * len(REPS) * len(strategies)

    # Optional flags, built as ONE string each carrying its own line
    # continuation, so that when nothing optional is set the line collapses
    # entirely. An empty placeholder on its own line leaves a dangling
    # backslash followed by a whitespace-only line, which bash reads as the end
    # of the command -- it then tries to run `--results-root` as a program.
    # `bash -n` does not catch this.
    extra_bits = []
    if args.oof_outer_folds:
        extra_bits.append(f'--oof-outer-folds {args.oof_outer_folds}')
    if args.threshold_quantile:
        extra_bits.append(f'--threshold-quantile {args.threshold_quantile:g}')
    extra_args = ''.join(f'{b} \\\n    ' for b in extra_bits)

    written = []
    for model, (tier, cpus, mem, hours, note) in MODELS.items():
        slug = model.lower().replace('-', '_')
        script_name = f'unc_{slug}.sh'
        # GP is gated to PDV only unless told otherwise, and the kernel has to
        # be RBF for it to be valid on every representation (Tanimoto is only
        # defined on binary fingerprints).
        gp_args = '--gp-reps "$rep" --gp-kernel rbf \\\n    ' if model == 'GP' else ''
        body = TEMPLATE.format(
            model=model, note=note, jobslug=slug, model_slug=slug,
            cpus=cpus, mem=mem, hours=hours, oof=args.oof_folds,
            kirby_dir=KIRBY_DIR, qsar_dir=QSAR_DIR, results_root=RESULTS_ROOT,
            datasets=' '.join(DATASETS),
            reps=' '.join(f'"{r}"' for r in REPS),
            strategies=' '.join(strategies),
            n_ds=len(DATASETS), n_rep=len(REPS), n_st=len(strategies),
            n_tasks=n_tasks, last=n_tasks - 1, throttle=args.throttle,
            script_name=script_name, gp_args=gp_args, gp_line='',
            extra_args=extra_args)
        (out / script_name).write_text(body)
        (out / script_name).chmod(0o755)
        written.append((tier, script_name, model, hours))

    print(f"Wrote {len(written)} array scripts, {n_tasks} tasks each "
          f"({len(written) * n_tasks} tasks total), oof-folds={args.oof_folds}, "
          f"oof-outer-folds={args.oof_outer_folds or 'all 5'}")
    for tier in (1, 2):
        print(f"\n  Tier {tier}:")
        for t, name, model, hours in written:
            if t == tier:
                print(f"    {name:26s} {model:16s} --time={hours}:00:00")


if __name__ == '__main__':
    main()
