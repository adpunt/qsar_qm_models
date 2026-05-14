#!/bin/bash
#SBATCH --job-name=val_VBLL_pdv_herg
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. ../qsar_qm_models/setup.sh
cd tests

python alternative_data_noise_robustness.py \
    --datasets herg_ki \
    --models VBLL-Full \
    --reps PDV \
    --results-root results/validation_rerun/pdv_herg

echo "Done: VBLL-Full x PDV x herg_ki"
