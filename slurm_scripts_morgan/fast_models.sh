#!/bin/bash
#SBATCH --job-name=morgan_fast
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
git pull
. setup.sh
cd scripts

# Fast models × morgan (ECFP4 radius=2) × all 6 strategies
# Models: RF, XGBoost, LightGBM, SVM, NGBoost

for STRATEGY in legacy quantile threshold outlier heteroscedastic value_proportional; do
    for MODEL in rf xgboost lgb svm ngboost; do
        echo "=== ${MODEL} / morgan / ${STRATEGY} ==="
        python process_and_train.py -d QM9 -t homo_lumo_gap \
            -m $MODEL \
            -r morgan \
            --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
            --noise-strategy $STRATEGY \
            -n 10000 \
            -b 10 \
            -u True \
            --normalize True \
            -f ../results/anova_${STRATEGY}_morgan_${MODEL}.csv
    done
done
