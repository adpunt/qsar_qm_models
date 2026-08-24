#!/bin/bash
# ============================================================================
# QM9 re-run — model: lgb
# LightGBM
# ============================================================================
# Array task -> (strategy, representation).
#   6 strategies x 5 representations = 30 tasks
#   each task = 11 noise levels x 10 replicates = 110 training runs
#
# REQUIRES the rebuilt rust binary (the held-out-noise fix). Run
# `cargo build --release` in /data/stat-cadd/scat9264/qsar_qm_models/rust FIRST -- see RUNBOOK.
#
# --account and --partition are LIVE STATE; pass them at submit time:
#   sbatch --account=<acct> --partition=medium --array=0-29%5 qm9_lgb.sh
#
# Resubmit only failed indices:
#   sbatch --account=<acct> --partition=medium --array=3,17 qm9_lgb.sh
# ============================================================================
#SBATCH --job-name=qm9_lgb
#SBATCH --output=qm9_lgb_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=23:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
# Guarded: under `set -u` an unset CONDA_PREFIX aborts here, before python.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh

# The binary carries the held-out-noise fix. Refuse to run without it rather
# than silently regenerating the same invalid results.
if [ ! -x rust/target/release/rust_processor ]; then
    echo "ERROR: rust/target/release/rust_processor missing. Run:"
    echo "  cd /data/stat-cadd/scat9264/qsar_qm_models/rust && cargo build --release"
    exit 2
fi

cd scripts

STRATS=(legacy value_proportional quantile threshold heteroscedastic outlier)
REPS=(ecfp4 continuous_pdv smiles mhggnn mol2vec)

n_rep=${#REPS[@]}
n_tasks=$(( ${#STRATS[@]} * n_rep ))
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
strat="${STRATS[$(( i / n_rep ))]}"
# Filename tag the figure script globs for (it differs from the CLI spelling).
case "$strat" in
  value_proportional) tag=valprop ;;
  heteroscedastic)    tag=hetero ;;
  *)                  tag="$strat" ;;
esac

OUT="../results/anova_${tag}_${rep}_lgb.csv"

echo "=== task $i: model=lgb strategy=$strat rep=$rep"
echo "=== out: $OUT"
echo "=== started: $(date)"

python -u process_and_train.py -d QM9 -t homo_lumo_gap \
    -m lgb \
    -r "$rep" \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy "$strat" \
    -n 10000 \
    -b 10 \
    -s scaffold \
    --normalize True \
    -f "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
