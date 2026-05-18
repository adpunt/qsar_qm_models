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

# Generate figures. All three paths given as absolute — the script's
# defaults --qm9-dir='../results' and --output-dir='../results/paper_figures'
# assume cwd=scripts/, but we run python from the repo root, so without
# explicit paths --output-dir resolves to /data/stat-cadd/scat9264/results/
# paper_figures (the parent dir), NOT qsar_qm_models/results/paper_figures.
cd /data/stat-cadd/scat9264/qsar_qm_models
python scripts/generate_paper_figures.py \
    --qm9-dir /data/stat-cadd/scat9264/qsar_qm_models/results \
    --validation-dir /data/stat-cadd/scat9264/KIRBy/tests/results/validation \
    --output-dir /data/stat-cadd/scat9264/qsar_qm_models/results/paper_figures

echo "Done: generate_paper_figures"
