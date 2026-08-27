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

# Loss landscape:
git checkout main
python3 process_and_train.py -t homo_lumo_gap -m dnn mlp rnn gru mtl factorization_mlp residual_mlp flexible_dnn -r ecfp4 smiles -n 15000 -f ../results/lossLandscapeBaselineScaffold.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json --split scaffold -b 5
git checkout loss_landscape
python3 process_and_train.py -t homo_lumo_gap -m dnn mlp rnn gru mtl factorization_mlp residual_mlp flexible_dnn -r ecfp4 smiles -n 15000 -f ../results/lossLandscapeAppliedScaffold.csv --sigma 0.0 0.25 0.5 0.75 1.0 1.25 --loss-landscape true -p /data/stat-cadd/scat9264/qsar_qm_models/results/scaffoldTuning.json --split scaffold -b 5

