#!/bin/bash
#SBATCH --job-name=morgan_unc_ngb
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
git pull
. setup.sh
cd scripts

# NGBoost × morgan (ECFP4 radius=2) × all 6 strategies — WITH UNCERTAINTY

for STRATEGY in legacy quantile threshold outlier heteroscedastic value_proportional; do
    echo "=== ngboost / morgan / ${STRATEGY} (uncertainty) ==="
    python process_and_train.py -d QM9 -t homo_lumo_gap \
        -m ngboost \
        -r morgan \
        --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
        --noise-strategy $STRATEGY \
        -n 10000 \
        -b 1 \
        -u True \
        --normalize True \
        -f ../results/uncertainty_${STRATEGY}_morgan_ngboost.csv
done
