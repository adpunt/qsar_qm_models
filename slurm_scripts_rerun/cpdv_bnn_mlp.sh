#!/bin/bash
#SBATCH --job-name=cpdv_bnn_mlp
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
git pull
. setup.sh
cd scripts

# MLP-BNN Full and MLP-BNN Last × continuous_pdv × all 6 strategies — ANOVA only (no uncertainty)
# 2 transforms × 6 strategies = 12 runs, 10 bootstrap iterations each

for STRATEGY in legacy quantile threshold outlier heteroscedastic value_proportional; do
    for TRANSFORM in full last; do
        echo "=== mlp_bnn_${TRANSFORM} / continuous_pdv / ${STRATEGY} ==="
        python process_and_train.py -d QM9 -t homo_lumo_gap \
            -m mlp \
            -r continuous_pdv \
            --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
            --noise-strategy $STRATEGY \
            -n 10000 \
            -b 10 \
            --normalize True \
            --bayesian-transformation $TRANSFORM \
            -f ../results/anova_${STRATEGY}_continuous_pdv_mlp_bnn_${TRANSFORM}.csv
    done
done
