#!/bin/bash
#SBATCH --job-name=cpdv_bmlp
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:59:00
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

SIGMAS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}

for STRATEGY in legacy quantile threshold outlier heteroscedastic value_proportional; do
    for BAYESIAN in full last; do
        python process_and_train.py -d QM9 -t homo_lumo_gap \
            -m mlp --bayesian $BAYESIAN \
            -r continuous_pdv \
            --sigma $SIGMA \
            --noise-strategy $STRATEGY \
            -n 10000 -b 10 --normalize True \
            -f ../results/anova_${STRATEGY}_continuous_pdv_mlp_bnn_${BAYESIAN}.csv
    done
done
