#!/bin/bash
#SBATCH --job-name=svm_sns
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

python tune_svm_all_reps.py --n-samples 5000 --n-trials 200 --seed 42 --reps sns --output-suffix _5k_sns
