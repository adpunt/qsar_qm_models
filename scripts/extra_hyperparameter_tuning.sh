#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=42:00:00              # Request 42 hour runtime 
#SBATCH --partition=medium           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 16GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts

git checkout main

# Standard + extra reps:
# Didn't work for this but worked for everything else???
python3 process_and_train.py -t homo_lumo_gap -m xgboost rf -r randomized_smiles -n 15000 -f ../results/tuning.csv  --tuning true --n-trials 100 >> ../results/randomizedSmilesTuning.txt

python3 process_and_train.py -t homo_lumo_gap -m dnn mlp lgb rnn gru mtl residual_mlp -r randomized_smiles -n 15000 -f ../results/tuning.csv --n-trials 150 --tuning true >> ../results/extraSnsTuning.txt

# python3 process_and_train.py -t homo_lumo_gap -m  gru -r smiles -n 30000 -f ../results/tuning.csv --n-trials 150 --tuning true >> ../results/tuningStandardReps.txt

# python3 process_and_train.py -t homo_lumo_gap -m gauche -r sns -n 15000 -f ../results/tuning.csv --n-trials 20 --tuning true >> ../results/tuningHighComputation.txt

python3 process_and_train.py -t homo_lumo_gap -m svm gauche -r randomized_smiles -n 15000 -f ../results/tuning.csv --n-trials 20 --tuning true >> ../results/tuningHighComputation.txt

# python3 process_and_train.py -t mu -m gauche -r ecfp4 sns smiles -n 15000 -f ../results/tuningMu.csv  --tuning true --n-trials 20 >> ../results/tuningMuHighComputation.txt 

# python3 process_and_train.py -t alpha -m svm gauche -r ecfp4 sns smiles -n 15000 -f ../results/tuningAlpha.csv  --tuning true --n-trials 20  >> ../results/tuningAlphaHighComputation.txt 

# python3 process_and_train.py -t alpha -m svm gauche -r randomized_smiles -n 15000 -f ../results/tuningAlpha.csv  --tuning true --n-trials 20  >> ../results/tuningAlphaHighComputation.txt 



# randomized_smiles still not working, randomized_smiles should be in the regular one
# gru not working

python3 process_and_train.py -t homo_lumo_gap -m xgboost rf -r randomized_smiles -n 1500 

python3 process_and_train.py -t homo_lumo_gap -m dnn -r ecfp4 smiles -n 8000 --sigma 0.0 1.0 --loss-landscape true


python3 process_and_train.py -t homo_lumo_gap -m xgboost rf -r randomized_smiles -n 1500 
