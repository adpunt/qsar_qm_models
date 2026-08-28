#!/bin/bash
#SBATCH --job-name=vbll_dnn_r5
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
git pull
. setup.sh
cd scripts

for STRATEGY in legacy quantile threshold outlier heteroscedastic value_proportional; do
    echo "=== dnn_vbll / continuous_pdv / ${STRATEGY} / rep 5 ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m dnn \
        -r continuous_pdv \
        --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 1 \
        --start-iteration 5 \
        --normalize True \
        --bayesian-transformation full_variational \
        -f ../results/anova_${STRATEGY}_continuous_pdv_dnn_bnn_full_variational.csv
done
