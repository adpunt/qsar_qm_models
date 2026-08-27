#!/bin/bash
#SBATCH --job-name=svm_legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=71:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# svm/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m svm \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_ecfp4_svm.csv

# svm/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m svm \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_pdv_svm.csv

# svm/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m svm \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_sns_svm.csv

# svm/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m svm \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_smiles_svm.csv

# svm/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m svm \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_legacy_mhggnn_svm.csv
