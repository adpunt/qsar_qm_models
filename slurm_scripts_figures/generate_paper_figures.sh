#!/bin/bash
#SBATCH --job-name=gen_paper_figures
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
git pull
. setup.sh

# Consolidate per-job validation CSVs first (idempotent — safe to re-run)
cd /data/stat-cadd/scat9264/KIRBy/tests
python /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_validation_rerun/merge_results.py

# Generate figures. Both --qm9-dir and --validation-dir given as absolute
# paths — the script's default --qm9-dir='../results' assumes cwd=scripts/,
# which we don't honor here (we run python from the repo root).
cd /data/stat-cadd/scat9264/qsar_qm_models
python scripts/generate_paper_figures.py \
    --qm9-dir /data/stat-cadd/scat9264/qsar_qm_models/results \
    --validation-dir /data/stat-cadd/scat9264/KIRBy/tests/results/validation

echo "Done: generate_paper_figures"
