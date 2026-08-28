#!/bin/bash
#SBATCH --job-name=mlp_threshold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# mlp/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_mlp.csv

# mlp/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_mlp.csv

# mlp/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_mlp.csv

# mlp/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_mlp.csv

# mlp/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_mlp.csv
