#!/bin/bash
# Regenerates all paper figures. Runs on the long partition (short is heavily
# backlogged; long shares the same nodes and a 2h job fits its 30-day cap).
#SBATCH --job-name=paper_figs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --partition=long
#SBATCH --mem=128G
#SBATCH --output=/data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_analysis/figs-%j.out
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

git pull

python generate_paper_figures.py --qm9-dir ../results \
  --validation-dir /data/stat-cadd/scat9264/KIRBy/tests/results/alternative_full
