#!/bin/bash
#SBATCH --job-name=pdv_invest_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:59:00
#SBATCH --partition=short
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# PDV Investigation v3: re-run with corrected XGBoost params
# XGBoost now uses master_tuned_hyperparameters.json values:
#   max_depth=18, lr=0.021, n_estimators=695, subsample=0.821
# (v2 used fig6a values which were different)
# Everything else unchanged from v2.

python investigate_pdv.py --model-comparison --n-iterations 5
