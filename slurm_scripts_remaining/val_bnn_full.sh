#!/bin/bash
#SBATCH --job-name=val_bnn_full
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=71:59:00
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

# Full-BNN validation on all 3 external datasets
python alternative_data_noise_robustness.py \
    --datasets all \
    --models Full-BNN \
    --results-root results/alternative_full

echo "Done: validation Full-BNN (all datasets)"
