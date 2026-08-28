#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=36:00:00              # Request 36 hour runtime 
#SBATCH --partition=medium           # Choose the appropriate partition
#SBATCH --mem=16G                    # Allocate 16GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts
git checkout bayesian_transformation

# Bayesian test:
python3 process_and_train.py -t homo_lumo_gap -m mlp dnn mtl residual_mlp factorization_mlp -r ecfp4 -n 20000 -f ../results/bayesianBaseline.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json -b 5

python3 process_and_train.py -t homo_lumo_gap -m mlp dnn mtl residual_mlp factorization_mlp -r ecfp4 -n 20000 -f ../results/bayesianApplied.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation true -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json -b 5
