#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=00:30:00              # Request 30 min runtime 
#SBATCH --partition=short           # Choose the appropriate partition
#SBATCH --mem=16G                    # Allocate 16GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts
git checkout loss_landscape

# Loss landscape:
python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 smiles -n 3000 -f small_test.csv --loss-landscape true -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json
