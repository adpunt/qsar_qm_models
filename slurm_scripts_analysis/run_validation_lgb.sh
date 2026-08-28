#!/bin/bash
#SBATCH --job-name=val_lgb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. ../qsar_qm_models/setup.sh

cd tests
git pull

python alternative_data_noise_robustness.py \
    --datasets all \
    --models LightGBM \
    --results-root results/alternative_full

echo "Done: validation LightGBM"
