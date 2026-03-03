#!/bin/bash
#SBATCH --job-name=pdv_invest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:59:00
#SBATCH --partition=long
#SBATCH --mem=64G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
git pull
. setup.sh
cd scripts

echo "=== PDV Binarization Investigation (default hyperparameters, 11 sigma levels) ==="
python investigate_pdv.py --n-samples 10000 --n-iterations 5 --seed 42

echo ""
echo "=== Model Comparison v4: RF/XGB/SVM/BNN/VBLL/Gauche(Tanimoto+RBF) ==="
python investigate_pdv.py --n-samples 10000 --seed 42 --skip-rf --skip-analysis --model-comparison
