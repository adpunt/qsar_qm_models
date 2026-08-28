#!/bin/bash
# ============================================================================
# Hyperparameter sweep: qrf on 6 representation(s).
#
#   array task -> representation
#   each task  -> 1 default fit + 12 random settings, scored on the
#                 VALIDATION split of one scaffold split of 10000
#                 molecules. No folds. Clean labels.
#
# Wall clock is 15h, from the MEASURED seconds for this model's slowest
# representation (chemberta, 1319s per fit) x 13 fits x 3.0
# headroom. Nothing here is a guess; see results/tuning_local/timing.csv.
#
# --account and --partition are LIVE STATE; pass them at submit time:
#   sbatch --account=stat-cadd --partition=medium --array=0-5%6 tune_qrf.sh
#
# Afterwards, and NOT optional -- a winner picked on the split that chose it
# proves nothing (RERUN_PLAN.md 5.7i):
#   python scripts/tune_hyperparameters.py --merge
#   python scripts/tune_hyperparameters.py --confirm
#   python scripts/tune_hyperparameters.py --write-master --margin 0.01
#   python scripts/confirm_tuned_on_validation_datasets.py --run
#   python scripts/confirm_tuned_on_validation_datasets.py --prune
# ============================================================================
#SBATCH --job-name=tune_qrf
#SBATCH --output=tune_qrf_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=15:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh

# An unactivated task falls through to the system Anaconda, which has none of
# the model packages: the job runs, finds nothing to do, and writes no rows.
# That is what happened to 12822693 and 12822694 (RERUN_PLAN.md 2.8d).
if [ -z "${CONDA_PREFIX:-}" ]; then
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
export TMPDIR="${TMPDIR:-/tmp}/tune_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# The interpreter has to be able to BUILD what this job is about to ask for.
# Seconds here against a whole wasted allocation later.
python scripts/check_environment.py --models qrf || {
    echo "ERROR: this interpreter cannot build model 'qrf'."
    exit 2
}

REPS=("ecfp4" "pdv" "mhggnn" "avalon" "chemberta" "sns")
n_tasks=${#REPS[@]}
i="${SLURM_ARRAY_TASK_ID:-0}"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
rep="${REPS[$i]}"

echo "=== task $i: model=qrf rep=$rep"
echo "=== settings: 12 random, plus the shared default"
echo "=== started: $(date)"

# One output file per task. Merged afterwards by --merge, which also names any
# pairing that produced no rows.
python -u scripts/tune_hyperparameters.py --sweep \
    --settings 12 \
    --sample-size 10000 \
    --seed 42 \
    --models qrf \
    --reps "$rep" \
    --tag "qrf_$rep"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
