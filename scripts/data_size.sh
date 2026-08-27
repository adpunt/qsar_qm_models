#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --partition=long           # Choose the appropriate partition
#SBATCH --mem=164G                    # Allocate 128GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd qsar_qm_models/ 
. setup.sh

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts
python3 precompute_distances

# Data size:
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche -r ecfp4 smiles -n 1000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m gin graph_gp -r graph -n 1000 -f ../results/dataSizeGraph.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche  -r ecfp4 smiles -n 5000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m gin graph_gp -r graph -n 5000 -f ../results/dataSizeGraph.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche -r ecfp4 smiles -n 10000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m gin grap9894251h_gp -r graph -n 10000 -f ../results/dataSizeGraph.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche  -r ecfp4 smiles -n 20000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -3 --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m gin graph_gp -r graph -n 20000 -f ../results/dataSize.csv --sigma 0.25 0.5 0.75 1.0 1.25 -b 3 --split scaffold

# Timed out - may not need the full set, same for graphs going to 20k with a day