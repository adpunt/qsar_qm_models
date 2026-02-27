#!/bin/bash
#SBATCH --job-name=svm_tune_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:59:00
#SBATCH --partition=short
#SBATCH --mem=32G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# Standalone SVM hyperparameter tuning for all reps (except PDV, already tuned)
# 200 Optuna trials per rep, scaffold split, Y z-score normalized
# Results saved to pdv_investigation/svm_tuning_results.json for review
# Does NOT auto-update master_tuned_hyperparameters.json

python tune_svm_all_reps.py --n-samples 10000 --n-trials 200 --seed 42
