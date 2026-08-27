#!/bin/bash
#SBATCH --job-name=setup_noise_env  # Name of the job
#SBATCH --nodes=1                   # Request 1 node
#SBATCH --ntasks=1                   # Request 1 task
#SBATCH --cpus-per-task=4            # Number of CPU cores per task
#SBATCH --time=52:00:00              # Request 52 hour runtime 
#SBATCH --partition=long           # Choose the appropriate partition
#SBATCH --mem=128G                    # Allocate 128GB of memory
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk  # Notification email
#SBATCH --output=slurm-%j.out        # Save standard output to slurm-<jobID>.out

export MAMBA_EXE="$(pwd)/bin/micromamba"
eval "$($MAMBA_EXE shell hook --shell bash)"
micromamba activate py_rust_env
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models/scripts
# git checkout bayesian_transformation

# Bayesian test:
# python3 process_and_train.py -t homo_lumo_gap -m mlp dnn mtl residual_mlp factorization_mlp -r ecfp4 -n 20000 -f ../results/bayesianBaseline.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json -b 5

# python3 process_and_train.py -t homo_lumo_gap -m mlp dnn mtl residual_mlp factorization_mlp -r ecfp4 -n 20000 -f ../results/bayesianApplied.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 --bayesian-transformation true -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json -b 5

# cd /data/stat-cadd/scat9264/qsar_qm_models/scripts
# git checkout loss_landscape

# Loss landscape:
# python3 process_and_train.py -t homo_lumo_gap -m dnn mlp rnn gru mtl factorization_mlp residual_mlp -r ecfp4 smiles -n 30000 -f ../results/lossLandscapeBaseline.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json
# git branch
# git checkout loss_landscape
# python3 process_and_train.py -t homo_lumo_gap -m dnn mlp rnn gru mtl factorization_mlp residual_mlp -r ecfp4 smiles -n 30000 -f ../results/lossLandscapeApplied.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 --loss-landscape true -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json


# Baseline takes 4 hours

git checkout main

# Line plots:
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/linePlot.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json
# This timed out at 12 hours
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche dnn -r randomized_smiles -n 15000 -f ../results/linePlot.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# git checkout main
python3 process_and_train.py -t homo_lumo_gap -m gin gcn -r graph -n 15000 -f ../results/linePlotGraph.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 10

# NEED TO ADD BACK IN randomized_smiles here!!!
# # Extra line plots:
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost svm gauche dnn mtl residual_mlp factorization_mlp mlp rnn gru -r ecfp4 sns smiles -n 15000 -f ../results/linePlotExtra.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# # Data size:
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles -n 1000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 1000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles -n 5000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 5000 -f ../results/dataSize.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles -n 10000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 10000 -f ../results/dataSize.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles -n 20000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 20000 -f ../results/dataSize.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles -n 30000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 30000 -f ../results/dataSize.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 smiles -n 50000 -f ../results/dataSize.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m gin -r graph -n 50000 -f ../results/dataSize.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json


# # Non-Gaussian noise: 
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionGaussian.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionLeft.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution left-tailed -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionRight.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution right-tailed -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionU.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution u-shaped -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json

# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 sns smiles -n 15000 -f ../results/distributionUniform.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --distribution uniform -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json


# # Scaffold splitting 
# # Can compare to distributionGaussian
# git checkout main
python3 process_and_train.py -t homo_lumo_gap -m rf xgboost gauche dnn -r ecfp4 smiles -n 15000 -f scaffold.csv  --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 --split scaffold -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuning.json
# TODO: SNS and ranodomised SMILES 

# # # Mu and alpha
# git checkout main
# python3 process_and_train.py -t homo_lumo_gap -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/gap.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuningMu.json
# git checkout main
# python3 process_and_train.py -t mu -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/mu.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuningMu.json
# git checkout main
# python3 process_and_train.py -t alpha -m rf xgboost -r ecfp4 sns smiles -n 20000 -f ../results/alpha.csv -b 5 --sigma 0.0 0.25 0.5 0.75 1.0 1.25 1.5 1.75 2.0 -b 5 -p /data/stat-cadd/scat9264/qsar_qm_models/results/tuningAlpha.json



# Next up:
# - Basic line plots for graphs didn't work 
# - Bayesian test with scaffold splits
# - Data size with scaffold 
# - Re-run for other distributions - find a way to compare fairly, see how they do it elsewhere