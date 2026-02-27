#!/bin/bash
#SBATCH --job-name=pdv_invest
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

# PDV Binarization Investigation — full run
# Part 1: RF × 8 PDV variants (binary_gt0, continuous_raw, continuous_zscore,
#          continuous_minmax, binary_mean, binary_median, binary_nonzero,
#          binary_gt0_no_dead) — 5 iterations, 5 sigma levels
# Part 2: Model comparison (RF, XGBoost, SVM, BNN Full, VBLL) × 3 PDV variants
#          (continuous_raw, binary_mean, binary_gt0_no_dead) — 1 iteration
#          with Y normalization and tuned hyperparameters (matching main pipeline)

python investigate_pdv.py --model-comparison --n-iterations 5
