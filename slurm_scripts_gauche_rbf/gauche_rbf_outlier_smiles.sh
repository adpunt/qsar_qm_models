#!/bin/bash
#SBATCH --job-name=gauche_rbf_outlier_smiles
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2-00:00:00
#SBATCH --partition=medium
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
# To bill to stat-cadd instead of the default account, uncomment:
# #SBATCH --account=stat-cadd

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# GP (gauche) with RBF kernel -> saved as 'gauche_rbf'. Completes RBF-GP across
# all ANOVA reps so a single consistent GP model can enter the cross-rep ANOVA.
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche --kernel rbf \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy outlier \
    -n 10000 \
    -b 10 \
    --normalize True \
    -f ../results/anova_outlier_smiles_gauche_rbf.csv
