#!/bin/bash
#SBATCH --job-name=gauche_hetero
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

# gauche/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy heteroscedastic \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_hetero_ecfp4_gauche.csv

# gauche/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy heteroscedastic \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_hetero_pdv_gauche.csv

# gauche/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy heteroscedastic \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_hetero_sns_gauche.csv

# gauche/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy heteroscedastic \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_hetero_smiles_gauche.csv
