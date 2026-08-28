#!/bin/bash
# ============================================================================
# Uncertainty re-run — model: VBLL-Full
# 30 stochastic forward passes
# ============================================================================
# Array task -> (dataset, representation, strategy).
#   3 datasets x 4 reps x 6 strategies = 60 tasks
#
# --account and --partition are LIVE STATE: pass them at submit time. Confirm
# with  bash tests/slurm_scripts/where_to_submit.sh  in the KIRBy repo first.
#
#   sbatch --account=<acct> --partition=medium --array=0-59%6 \
#          unc_vbll_full.sh
#
# Every task writes its own --results-root, so tasks never race on a shared
# file. Run merge_results.py afterwards.
#
# Resubmit only the failed indices, e.g.:
#   sbatch --account=<acct> --partition=medium --array=3,17,40 unc_vbll_full.sh
# ============================================================================
#SBATCH --job-name=unc_vbll_full
#SBATCH --output=unc_vbll_full_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=47:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
# Guarded: under `set -u` an unset CONDA_PREFIX aborts the shell here, before
# python is ever reached. The reference scripts get away with the unguarded
# form only because they do not set -u.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

cd /data/stat-cadd/scat9264/KIRBy
. /data/stat-cadd/scat9264/qsar_qm_models/setup.sh
cd tests

DATASETS=(logd caco2 herg_ki)
REPS=("ECFP4" "PDV" "SNS" "MHG-GNN-pretrained")
STRATS=(legacy outlier quantile threshold valprop)

n_st=${#STRATS[@]}
n_rep=${#REPS[@]}
n_tasks=$(( ${#DATASETS[@]} * n_rep * n_st ))

# Guard: under `set -u` an unset SLURM_ARRAY_TASK_ID would abort with a cryptic
# message. This makes a non-array invocation run task 0 and say so.
i="${SLURM_ARRAY_TASK_ID:-0}"
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
st="${STRATS[$(( i % n_st ))]}"
rep="${REPS[$(( (i / n_st) % n_rep ))]}"
ds="${DATASETS[$(( i / (n_st * n_rep) ))]}"

# One directory per task — no cross-task write races.
rep_slug=$(echo "$rep" | tr 'A-Z' 'a-z' | tr -d '-')
OUT="results/uncertainty_rerun/vbll_full__${ds}__${rep_slug}__${st}"

if [ -z "${SLURM_JOB_PARTITION:-}" ]; then
    echo "ERROR: no partition. Submit with --partition=medium (see RUNBOOK step 4)."; exit 2
fi

echo "=== task $SLURM_ARRAY_TASK_ID: model=VBLL-Full dataset=$ds rep=$rep strategy=$st"
echo "=== out: $OUT"
echo "=== started: $(date)"

python -u alternative_data_noise_robustness.py \
    --datasets "$ds" \
    --models "VBLL-Full" \
    --reps "$rep" \
    --strategies "$st" \
    --unc-strategies all \
    --oof-folds 5 \
    --oof-outer-folds 1 \
    --results-root "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
