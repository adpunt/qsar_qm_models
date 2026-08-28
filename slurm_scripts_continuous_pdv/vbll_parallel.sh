#!/bin/bash
#SBATCH --job-name=cpdv_vbll
#SBATCH --array=0-11
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

# 12 parallel tasks: 6 strategies × 2 models
STRATEGIES=(legacy quantile threshold outlier heteroscedastic value_proportional)
MODELS=(dnn mlp)

STRAT_IDX=$((SLURM_ARRAY_TASK_ID / 2))
MODEL_IDX=$((SLURM_ARRAY_TASK_ID % 2))

STRATEGY=${STRATEGIES[$STRAT_IDX]}
MODEL=${MODELS[$MODEL_IDX]}

echo "=== ${MODEL}_vbll / continuous_pdv / ${STRATEGY} (ANOVA, 10 reps) ==="
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m $MODEL \
    -r continuous_pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy $STRATEGY \
    -n 10000 \
    -b 10 \
    --normalize True \
    --bayesian-transformation full_variational \
    -f ../results/anova_${STRATEGY}_continuous_pdv_${MODEL}_bnn_full_variational.csv
