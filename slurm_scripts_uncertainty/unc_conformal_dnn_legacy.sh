#!/bin/bash
#SBATCH --job-name=unc_conf_dnn_legacy
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# Uncertainty: conformal_dnn - legacy

# conformal_dnn/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model dnn \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 1 \
    -u True \
    --normalize True \
    -f ../results/uncertainty_legacy_pdv_conformal_dnn.csv

# conformal_dnn/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m conformal \
    --cp-base-model dnn \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy legacy \
    -n 10000 \
    -b 1 \
    -u True \
    --normalize True \
    -f ../results/uncertainty_legacy_sns_conformal_dnn.csv

