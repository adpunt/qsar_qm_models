#!/bin/bash
#SBATCH --job-name=val_FULL_BNN_mhg_gnn_pretrained_herg
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
. setup.sh
cd tests

python alternative_data_noise_robustness.py \
    --datasets herg_ki \
    --models BNN-Full \
    --reps MHG-GNN-pretrained \
    --results-root results/validation_rerun/mhg_gnn_pretrained_herg

echo "Done: BNN-Full x MHG-GNN-pretrained x herg"
