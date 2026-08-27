#!/bin/bash
# ============================================================================
# Uncertainty re-run — model: NGBoost
# 500 estimators; slowest of the tree models by far
# ============================================================================
# Array task -> (dataset, representation, noise condition).
#   3 datasets x 4 reps x 5 conditions = 60 tasks
#
# Conditions: gaussian, grouped_wider, grouped_shifted, censoring, outlier_p10
# These are the settled set (noise_conditions.json), inherited from the main
# grid. Every one of them delivers the SAME amount of noise at a given level, so
# a difference between them is a difference of shape, not of dose.
#
# Levels are NOT passed: the runner anchors them per dataset to published assay
# error, and sweeps censoring on its own axis (the fraction of labels clipped).
#
# --account and --partition are LIVE STATE: pass them at submit time. Confirm
# with  bash tests/slurm_scripts/where_to_submit.sh  in the KIRBy repo first.
#
#   sbatch --account=<acct> --partition=medium --array=0-59%6 \
#          unc_ngboost.sh
#
# Every task writes its own --results-root, so tasks never race on a shared
# file. Run merge_results.py afterwards.
#
# Resubmit only the failed indices, e.g.:
#   sbatch --account=<acct> --partition=medium --array=3,17,40 unc_ngboost.sh
# ============================================================================
#SBATCH --job-name=unc_ngboost
#SBATCH --output=unc_ngboost_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

KIRBY_DIR="/data/stat-cadd/scat9264/KIRBy"

# Which KIRBy checkout this is, and whether it carries the redesigned runner.
#
# There are two checkouts on the cluster. 125 of the 127 job scripts in the
# KIRBy repository itself use /data/stat-ecr/scat9264/KIRBy -- the move was made
# on 2026-05-07 when stat-cadd hit 99.9% of its quota -- and two use stat-cadd,
# which is what this generator has always pointed at (RERUN_PLAN.md 2.8b). That
# cannot be settled from a laptop, so it is settled here, at the top of the job:
# a checkout without the redesigned command line is refused by name rather than
# producing 336 tasks' worth of results from the wrong code.
if [ ! -d "$KIRBY_DIR" ]; then
    echo "ERROR: no KIRBy checkout at $KIRBY_DIR."
    echo "       The other checkout is /data/stat-ecr/scat9264/KIRBy, which is what"
    echo "       125 of KIRBy's own 127 job scripts use. Regenerate with"
    echo "       --kirby-dir <path> rather than editing this file."
    exit 2
fi
RUNNER="$KIRBY_DIR/tests/alternative_data_noise_robustness.py"
if [ ! -f "$RUNNER" ]; then
    echo "ERROR: $RUNNER does not exist."; exit 2
fi
if ! grep -q -- "'--conditions'" "$RUNNER"; then
    echo "ERROR: $RUNNER has no --conditions flag, so this checkout predates the"
    echo "       noise redesign (noiseInject 1.0.0, 2026-08-26). Running it would"
    echo "       produce a full set of results from the old six strategies."
    echo "       Pull it, or point --kirby-dir at the other checkout."
    exit 2
fi
echo "=== KIRBy: $KIRBY_DIR  ($(git -C "$KIRBY_DIR" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"

cd "$KIRBY_DIR"
. /data/stat-cadd/scat9264/qsar_qm_models/setup.sh

# Activation is not optional. micromamba has never worked on this cluster, so
# the `export MAMBA_EXE=...` lines that used to sit above `. setup.sh` pointed
# at a file that does not exist -- and nothing checked. setup.sh falls through
# to its conda branch; if that also fails, the task carries on in the system
# Anaconda at /apps/system/..., which has no gpytorch, no quantile_forest and
# no ngboost. The job then runs, finds nothing to do, and writes no rows. That
# is what happened to 12822693 and 12822694 (RERUN_PLAN.md section 2.8d).
if [ -z "${CONDA_PREFIX:-}" ]; then
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
# The test above cannot fail once setup.sh has run: setup.sh:124 prepends
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

# The injector must be the redesigned one, and it must be the checkout that was
# pulled rather than a stale copy on the path. A task that runs the old injector
# writes results that look exactly like the new ones and are a different
# experiment.
python - <<'PYCHECK' || exit 2
import sys, inspect
try:
    import noiseInject
    from noiseInject import CONDITIONS
except Exception as exc:
    print(f"ERROR: noiseInject does not import: {type(exc).__name__}: {exc}")
    sys.exit(1)
print(f"=== noiseInject: {inspect.getfile(noiseInject)} "
      f"version {getattr(noiseInject, '__version__', 'unknown')}")
missing = [c for c in ['gaussian', 'grouped_wider', 'grouped_shifted', 'censoring', 'outlier_p10'] if c not in CONDITIONS]
if missing:
    print(f"ERROR: this noiseInject does not know {missing}. Known: {sorted(CONDITIONS)}.")
    print("       That is the pre-1.0.0 injector -- the six deleted strategies.")
    print("       pip install --no-deps -e <the NoiseInject checkout you pulled>")
    sys.exit(1)
PYCHECK

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
export TMPDIR="${TMPDIR:-/tmp}/qsar_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT


# Guarded: under `set -u` an unset CONDA_PREFIX aborts the shell here, before
# python is ever reached.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

# Can this interpreter actually build the model this array runs?
#
# The activation guard above proves an environment is active and that it is
# named env_test; the check above that proves noiseInject is the redesigned
# one. Neither proves the MODEL's backend is installed. Jobs 12822693 and
# 12822694 ran to completion and wrote nothing because gpytorch was missing,
# and the runner is built to SKIP a model whose backend will not import (the
# HAS_* flags at alternative_data_noise_robustness.py:253-333) rather than to
# stop -- so a missing package is silent by design, and this array would burn
# 60 tasks finding nothing to do. Seconds once, before the task loop.
python "/data/stat-cadd/scat9264/qsar_qm_models/scripts/check_environment.py" --validation-models "NGBoost" || {
    echo "ERROR: this interpreter cannot build NGBoost. See above."
    exit 2
}

cd tests

DATASETS=(logd caco2 herg_ki)
REPS=("ECFP4" "PDV" "SNS" "MHG-GNN-pretrained")
CONDS=(gaussian grouped_wider grouped_shifted censoring outlier_p10)

n_cond=${#CONDS[@]}
n_rep=${#REPS[@]}
n_tasks=$(( ${#DATASETS[@]} * n_rep * n_cond ))

# Guard: under `set -u` an unset SLURM_ARRAY_TASK_ID would abort with a cryptic
# message. This makes a non-array invocation run task 0 and say so.
i="${SLURM_ARRAY_TASK_ID:-0}"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
cond="${CONDS[$(( i % n_cond ))]}"
rep="${REPS[$(( (i / n_cond) % n_rep ))]}"
ds="${DATASETS[$(( i / (n_cond * n_rep) ))]}"

# One directory per task — no cross-task write races. merge_results.py splits
# this name back apart on the DOUBLE underscore, so the condition's own single
# underscores are safe.
rep_slug=$(echo "$rep" | tr 'A-Z' 'a-z' | tr -d '-')
OUT="results/uncertainty_rerun/ngboost__${ds}__${rep_slug}__${cond}"

if [ -z "${SLURM_JOB_PARTITION:-}" ]; then
    echo "ERROR: no partition. Submit with --partition=medium (see RUNBOOK step 4)."; exit 2
fi

echo "=== task $i: model=NGBoost dataset=$ds rep=$rep condition=$cond"
echo "=== out: $OUT"
echo "=== started: $(date)"

python -u alternative_data_noise_robustness.py \
    --datasets "$ds" \
    --models "NGBoost" \
    --reps "$rep" \
    --conditions "$cond" \
    --unc-conditions all \
    --oof-folds 5 \
    --results-root "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
