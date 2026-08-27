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

# Non-Gaussian noise: 
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionGaussianNT.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionLeftNT.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution left-tailed --split scaffold
git checkout ma
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionRightNT.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution right-tailed --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionUNT.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution u-shaped --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionUniformNT.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution uniform --split scaffold
