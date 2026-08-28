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

git checkout bayesian_transformation

# Tuning, scaffold
python3 process_and_train.py -t homo_lumo_gap -m gauche dnn --bayesian-transformation full -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation last_layer -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json

git checkout bayesian_transformation
# Tuning, no scaffold
python3 process_and_train.py -t homo_lumo_gap -m gauche dnn --bayesian-transformation full -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation last_layer -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json

git checkout bayesian_transformation
# No tuning, scaffold
python3 process_and_train.py -t homo_lumo_gap -m gauche dnn --bayesian-transformation full -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation last_layer -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0  --split scaffold

git checkout bayesian_transformation
# No tuning, no scaffold
python3 process_and_train.py -t homo_lumo_gap -m gauche dnn --bayesian-transformation full -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation last_layer -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m dnn --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 

# Need to add for graph_gp
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m graph_gp --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m graph_gp --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --split scaffold 
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m graph_gp --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m graph_gp --bayesian-transformation variational -r ecfp4 smiles -n 15000 -f ../results/uncertainty_test.csv  -u true --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 
