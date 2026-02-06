#!/bin/bash
#SBATCH --job-name=ext_threshold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=47:59:00
#SBATCH --partition=long
#SBATCH --mem=256G
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/qsar_qm_models
. setup.sh
cd scripts

# Extended sigmas (0.7-1.0) for threshold

# rf/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m rf \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_rf.csv

# rf/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m rf \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_rf.csv

# rf/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m rf \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_rf.csv

# rf/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m rf \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_rf.csv

# rf/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m rf \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_rf.csv

# xgboost/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m xgboost \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_xgboost.csv

# xgboost/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m xgboost \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_xgboost.csv

# xgboost/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m xgboost \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_xgboost.csv

# xgboost/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m xgboost \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_xgboost.csv

# xgboost/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m xgboost \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_xgboost.csv

# qrf/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m qrf \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_qrf.csv

# qrf/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m qrf \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_qrf.csv

# qrf/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m qrf \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_qrf.csv

# qrf/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m qrf \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_qrf.csv

# qrf/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m qrf \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_qrf.csv

# ngboost/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_ngboost.csv

# ngboost/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_ngboost.csv

# ngboost/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_ngboost.csv

# ngboost/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_ngboost.csv

# ngboost/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m ngboost \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_ngboost.csv

# dnn/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_dnn.csv

# dnn/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_dnn.csv

# dnn/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_dnn.csv

# dnn/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_dnn.csv

# dnn/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m dnn \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_dnn.csv

# mlp/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_mlp.csv

# mlp/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_mlp.csv

# mlp/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_mlp.csv

# mlp/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_mlp.csv

# mlp/mhggnn
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m mlp \
    -r mhggnn \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_mhggnn_mlp.csv

# gauche/ecfp4
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r ecfp4 \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_ecfp4_gauche.csv

# gauche/pdv
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r pdv \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_pdv_gauche.csv

# gauche/sns
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r sns \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_sns_gauche.csv

# gauche/smiles
python process_and_train.py -d QM9 -t homo_lumo_gap \
    -m gauche \
    -r smiles \
    --sigma 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --noise-strategy threshold \
    -n 10000 \
    -b 10 \
    --normalize True \
    \
    -f ../results/anova_threshold_smiles_gauche.csv

