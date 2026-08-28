#!/bin/bash
#SBATCH --job-name=m2v_ngb_missing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# ngboost/mol2vec/outlier (missing)
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r mol2vec \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy outlier \
    -n 10000 \
    -b 10 \
    --normalize True \
    -f ../results/anova_outlier_mol2vec_ngboost.csv

# ngboost/mol2vec/hetero (missing)
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r mol2vec \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy heteroscedastic \
    -n 10000 \
    -b 10 \
    --normalize True \
    -f ../results/anova_hetero_mol2vec_ngboost.csv
