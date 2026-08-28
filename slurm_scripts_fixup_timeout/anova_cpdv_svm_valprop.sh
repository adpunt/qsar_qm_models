#!/bin/bash
#SBATCH --job-name=cpdv_svm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:59:00
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

# Missing sigmas 0.4-1.0 for SVM × continuous_pdv × value_proportional
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m svm \
    -r continuous_pdv \
    --sigma 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy value_proportional \
    -n 10000 -b 10 --normalize True \
    -f ../results/anova_value_proportional_continuous_pdv_svm.csv
