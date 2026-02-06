#!/bin/bash
#SBATCH --job-name=rsmiles_legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# Randomized SMILES for legacy

# rf/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m rf \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_rf.csv

# xgboost/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m xgboost \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_xgboost.csv

# qrf/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m qrf \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_qrf.csv

# ngboost/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_ngboost.csv

# dnn/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_dnn.csv

# mlp/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_mlp.csv

# gauche/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_gauche.csv

# lgb/randomized_smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m lgb \
    -r randomized_smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_randomized_smiles_lgb.csv

