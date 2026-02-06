#!/bin/bash
#SBATCH --job-name=conf_threshold
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

# Conformal prediction for threshold

# conformal_rf/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model rf \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_ecfp4_conformal_rf.csv

# conformal_rf/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model rf \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_pdv_conformal_rf.csv

# conformal_rf/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model rf \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_sns_conformal_rf.csv

# conformal_rf/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model rf \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_smiles_conformal_rf.csv

# conformal_qrf/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model qrf \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_ecfp4_conformal_qrf.csv

# conformal_qrf/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model qrf \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_pdv_conformal_qrf.csv

# conformal_qrf/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model qrf \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_sns_conformal_qrf.csv

# conformal_qrf/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model qrf \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_smiles_conformal_qrf.csv

# conformal_dnn/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model dnn \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_ecfp4_conformal_dnn.csv

# conformal_dnn/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model dnn \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_pdv_conformal_dnn.csv

# conformal_dnn/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model dnn \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_sns_conformal_dnn.csv

# conformal_dnn/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model dnn \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    -u True \
    -f ../results/anova_threshold_smiles_conformal_dnn.csv

