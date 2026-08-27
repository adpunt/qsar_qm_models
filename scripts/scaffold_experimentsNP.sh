#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=48:00:00              # Request 50 hour runtime 
#SBATCH --partition=long           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 128GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts

# Without tuning
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche svm mlp dnn mtl residual_mlp factorization_mlp flexible_dnn -r ecfp4 smiles sns -n 15000 -f ../results/scaffoldNT.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m gin gcn graph_gp -r graph -n 15000 -f ../results/scaffoldGraphNT.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 --split scaffold

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

# Mu and alpha
git checkout main
python3 process_and_train.py -t mu -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/muNT.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 --split scaffold
git checkout main
python3 process_and_train.py -t alpha -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/alphaNT.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 --split scaffold

# Bayesian transformation 
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 -n 15000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -f ../results/bayesianBaselineNT.csv -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json -b 10 --split scaffold
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 -n 15000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation full -f ../results/bayesianFullNT.csv -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json -b 10 --split scaffold
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 -n 15000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation last_layer -f ../results/bayesianLastLayerNT.csv -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json -b 10 --split scaffold
git checkout bayesian_transformation
python3 process_and_train.py -t homo_lumo_gap -m  dnn -r ecfp4 -n 15000 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation variational -f ../results/bayesianVariationalNT.csv -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json -b 10 --split scaffold

# Loss landscape:
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m dnn mlp rnn gru mtl factorization_mlp residual_mlp flexible_dnn -r ecfp4 smiles -n 15000 -f ../results/lossLandscapeBaselineNT.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 --split scaffold -b 5
git checkout loss_landscape
python3 process_and_train.py -t homo_lumo_gap -m dnn mlp rnn gru mtl factorization_mlp residual_mlp flexible_dnn -r ecfp4 smiles -n 15000 -f ../results/lossLandscapeAppliedNT.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 --loss-landscape true --split scaffold -b 5



