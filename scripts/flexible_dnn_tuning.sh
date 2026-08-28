#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=64:00:00              # Request 64 hour runtime 
#SBATCH --partition=long           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 16GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts

git checkout main

python3 process_and_train.py -t homo_lumo_gap -m flexible_dnn -r  ecfp4 sns smiles randomized_smiles -n 15000 -f ../results/tuning.csv --n-trials 150 --tuning true >> ../results/netTuning.txt

python3 process_and_train.py -t homo_lumo_gap -m flexible_dnn -r  ecfp4 sns smiles randomized_smiles -n 15000 -f ../results/scaffoldTuning.csv --n-trials 150 --tuning true --split scaffold >> ../results/netscaffoldTuning.txt
