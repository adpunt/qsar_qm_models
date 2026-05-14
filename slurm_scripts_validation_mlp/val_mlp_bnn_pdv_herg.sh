#!/bin/bash
#SBATCH --job-name=val_mlp_bnn_pdv_herg
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=long
#SBATCH --time=16:00:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
git pull
. setup.sh
cd /data/stat-cadd/scat9264/KIRBy
git pull
cd tests

# Defensive env top-up — only runs if package is missing in env_test.
python -c "import noiseInject" 2>/dev/null || \
    pip install --no-deps -e /data/stat-cadd/scat9264/NoiseInject 2>/dev/null || \
    pip install --no-deps -e /data/stat-ecr/scat9264/NoiseInject 2>/dev/null || \
    echo "WARN: could not locate NoiseInject source — install manually"
python -c "from kirby.representations.molecular import create_pdv" 2>/dev/null || \
    pip install --no-deps -e /data/stat-cadd/scat9264/KIRBy 2>/dev/null

python alternative_data_noise_robustness.py \
    --datasets herg_ki \
    --models MLP-BNN-Full \
    --reps PDV \
    --results-root results/validation_rerun/pdv_herg

echo "Done: MLP-BNN-Full x PDV x herg_ki"
