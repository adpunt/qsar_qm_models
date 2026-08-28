#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=164:00:00              # Request 64 hour runtime 
#SBATCH --partition=long           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 16GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd qsar_qm_models/ 
. setup.sh

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts

# Tuning
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m xgboost rf qrt ngboost -r ecfp4 sns smiles randomized_smiles pdv -n 5000 -f ../results/scaffoldTuning.csv  --tuning true --n-trials 100 --split scaffold >> ../results/standardscaffoldTuning2.txt
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m dnn mlp lgb rnn gru mtl residual_mlp -r  ecfp4 sns smiles randomized_smiles pdv -n 5000 -f ../results/scaffoldTuning.csv --n-trials 150 --tuning true --split scaffold >> ../results/netscaffoldTuning.txt
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m svm gauche -r  ecfp4 sns smiles randomized_smiles pdv -n 5000 -f ../results/scaffoldTuning.csv --n-trials 50 --tuning true --split scaffold >> ../results/highComputationscaffoldTuning.txt

# SHAP (temporary but I want these experiments done)
git checkout shap
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles sns pdv -n 5000 -f ../results/shap.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --shap true --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout shap
python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 sns pdv -n 3000 -f ../results/shapSVM.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --shap true --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json

# Tuning
git checkout main
python3 process_and_train.py -t mu -m xgboost rf -r ecfp4 smiles pdv -n 5000 -f ../results/scaffoldTuningMu.csv  --tuning true --n-trials 100 --split scaffold >> ../results/muscaffoldTuning.txt
git checkout main
python3 process_and_train.py -t alpha -m xgboost rf -r ecfp4 smiles pdv -n 5000 -f ../results/scaffoldTuningAlpha.csv  --tuning true --n-trials 100 --split scaffold >> ../results/alphascaffoldTuning.txt
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m graph_gp -r graph pdv -n 5000 -f ../results/scaffoldTuning.csv --n-trials 150 --tuning true --split scaffold >> ../results/graphGpscaffoldTuning.txt
