#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=24:00:00              # Request 50 hour runtime 
#SBATCH --partition=medium           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 128GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts

git checkout shap
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles sns -n 5000 -f ../results/shap.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --shap true --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json

python3 process_and_train.py -t homo_lumo_gap -m svm -r ecfp4 sns -n 5000 -f ../results/shapSVM.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 --shap true --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
