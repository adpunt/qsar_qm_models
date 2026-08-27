#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=12:00:00              # Request 50 hour runtime 
#SBATCH --partition=medium           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 128GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts

# Mu and alpha
# Tuning
git checkout main
python3 process_and_train.py -t mu -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/muScaffold.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuningMu.json --split scaffold
git checkout main
python3 process_and_train.py -t alpha -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/alphaScaffold.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuningAlpha.json --split scaffold


# Mu and alpha
# No tuning
git checkout main
python3 process_and_train.py -t mu -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/muNT.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 --split scaffold
git checkout main
python3 process_and_train.py -t alpha -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/alphaNT.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 --split scaffold
