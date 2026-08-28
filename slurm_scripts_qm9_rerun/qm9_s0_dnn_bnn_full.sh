#!/bin/bash
# ============================================================================
# QM9 re-run — stage 0 — model: dnn_bnn_full
# BNN-alpha
# ============================================================================
# Array task -> (noise condition, representation).
#   3 conditions x 6 representations = 18 tasks
#   each task = 7 noise levels x 1 replicate(s) = 7 training runs
#
# Conditions in this stage: gaussian grouped_wider grouped_shifted
# Replicates: 1, numbered 0..0. The replicate seed
# depends only on the replicate index, so these are exchangeable with a run made
# at any other time and append to the same file.
#
# REQUIRES the rebuilt rust binary (the held-out-noise fix). Run
# `cargo build --release` in /data/stat-cadd/scat9264/qsar_qm_models/rust FIRST -- see RUNBOOK.
#
# --account and --partition are LIVE STATE; pass them at submit time:
#   sbatch --account=<acct> --partition=medium --array=0-17%5 qm9_s0_dnn_bnn_full.sh
#
# Resubmit only failed indices:
#   sbatch --account=<acct> --partition=medium --array=3,17 qm9_s0_dnn_bnn_full.sh
# ============================================================================
#SBATCH --job-name=qm90_dnn_bnn_full
#SBATCH --output=qm90_dnn_bnn_full_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

# micromamba has never worked on this cluster -- setup.sh has always fallen
# through to its conda branch. The two `export MAMBA_EXE=...` lines that used to
# sit here were dead: the hook failed, and because this script runs under
# `set -uo pipefail` with no `-e`, the failure did not stop anything. The task
# simply carried on unactivated, in whatever python was on PATH.
cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh

# Activation is not optional, and until now nothing checked it. An unactivated
# task falls through to the system Anaconda at /apps/system/..., which has no
# gpytorch, no quantile_forest and no ngboost -- so the job runs, finds nothing
# to do, and produces no rows. That is what happened to 12822693 and 12822694
# (RERUN_PLAN.md §2.8d). Confirmed on the cluster 2026-08-26.
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


# The binary carries the held-out-noise fix. Refuse to run without it rather
# than silently regenerating the same invalid results.
if [ ! -x rust/target/release/rust_processor ]; then
    echo "ERROR: rust/target/release/rust_processor missing. Run:"
    echo "  cd /data/stat-cadd/scat9264/qsar_qm_models/rust && cargo build --release"
    exit 2
fi

cd scripts

# The interpreter has to be able to BUILD what this job is about to ask for.
# Jobs 12822693 and 12822694 (2026-08-19) ran to completion and produced nothing
# because gpytorch was missing: the experiment list came out empty, five folds
# looped over nothing, and the job crashed reading its own empty output. This
# costs seconds and answers that before the queue does (RERUN_PLAN.md §2.8d).
python check_environment.py --models dnn_bnn_full || {
    echo "ERROR: this interpreter cannot build model 'dnn_bnn_full'. See the output above."
    exit 2
}

CONDS=(gaussian grouped_wider grouped_shifted)
REPS=(ecfp4 continuous_pdv mhggnn avalon chemberta sns)

n_rep=${#REPS[@]}
n_tasks=$(( ${#CONDS[@]} * n_rep ))
i="${SLURM_ARRAY_TASK_ID:-0}"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
if [ -z "${SLURM_JOB_PARTITION:-}" ]; then
    echo "ERROR: no partition. Submit with --partition=medium (see RUNBOOK)."; exit 2
fi

rep="${REPS[$(( i % n_rep ))]}"
cond="${CONDS[$(( i / n_rep ))]}"

# Each condition is a (shape, targeting) pair plus whatever that pair needs, and
# its own level grid -- censoring is a fraction of labels clipped, not a dose, so
# it cannot share the others' axis. Only flags that differ from the
# process_and_train.py default are passed; the injector echoes the full parameter
# set on every call, so the run log carries the provenance.
#
# A case statement rather than an associative array: associative arrays need
# bash 4, and this file is also smoke-tested on macOS, where /bin/bash is 3.2.
case "$cond" in
  # The reference case, and what both direct predecessors used, so the results stay comparable to theirs.
  gaussian) NOISE_FLAGS="--noise-shape gaussian --noise-targeting uniform"; LEVELS="0.0 0.2 0.3 0.5 0.75 1.0 1.5" ;;
  # Whole scaffold families get a larger error, still centred on the truth. Indistinguishable from Gaussian on its own -- it is here because it is the comparator that makes grouped_shifted interpretable. The pair is the claim; neither half is.
  grouped_wider) NOISE_FLAGS="--noise-shape gaussian --noise-targeting grouped_wide"; LEVELS="0.2 0.3 0.5 0.75 1.0 1.5" ;;
  # Whole scaffold families pushed in one direction. The ONLY zero-mean condition that separates: against grouped_wider at level 1.5 it costs 0.0963 R2 on the random forest and 0.1419 on LightGBM, at 2.52x and 2.71x the run-to-run spread, paired t p = 0.0031 and 0.0002. The wider range quoted before 2026-08-27 (0.096 to 0.314 at up to 7.4x) included a Ridge row, and Ridge is not in the study. Same amount of noise, same families, same targeting -- the only difference is direction.
  grouped_shifted) NOISE_FLAGS="--noise-shape gaussian --noise-targeting grouped_shift"; LEVELS="0.2 0.3 0.5 0.75 1.0 1.5" ;;
  *) echo "ERROR: unknown noise condition '$cond'"; exit 2 ;;
esac

OUT="../results/anova_${cond}_${rep}_dnn_bnn_full.csv"

echo "=== task $i: model=dnn_bnn_full condition=$cond rep=$rep"
echo "=== noise: $NOISE_FLAGS"
echo "=== levels: $LEVELS"
echo "=== replicates: 1 starting at 0"
echo "=== out: $OUT"
echo "=== started: $(date)"

python -u process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn --bayesian-transformation full -u True \
    -r "$rep" \
    --noise-level $LEVELS \
    --dose-units spread \
    $NOISE_FLAGS \
    -n 10000 \
    --repetitions 1 \
    --start-iteration 0 \
    -s scaffold \
    --normalize True \
    -f "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
