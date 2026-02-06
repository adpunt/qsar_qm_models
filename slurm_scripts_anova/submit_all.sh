#!/bin/bash
# Master script to submit all ANOVA gap-filling jobs
# Review individual scripts before running!

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_anova

echo "Submitting extended sigma scripts..."
sbatch ext_sigma_legacy.sh
sbatch ext_sigma_valprop.sh
sbatch ext_sigma_quantile.sh
sbatch ext_sigma_threshold.sh
sbatch ext_sigma_outlier.sh
sbatch ext_sigma_hetero.sh

echo 'Submitting LightGBM scripts...'
sbatch lgb_legacy.sh
sbatch lgb_valprop.sh
sbatch lgb_quantile.sh
sbatch lgb_threshold.sh
sbatch lgb_outlier.sh
sbatch lgb_hetero.sh

echo 'Submitting randomized SMILES scripts...'
sbatch rsmiles_legacy.sh
sbatch rsmiles_valprop.sh
sbatch rsmiles_quantile.sh
sbatch rsmiles_threshold.sh
sbatch rsmiles_outlier.sh
sbatch rsmiles_hetero.sh

echo 'Submitting BNN last_layer scripts...'
sbatch bnn_last_legacy.sh
sbatch bnn_last_valprop.sh
sbatch bnn_last_quantile.sh
sbatch bnn_last_threshold.sh
sbatch bnn_last_outlier.sh
sbatch bnn_last_hetero.sh

echo 'Submitting BNN variational scripts...'
sbatch bnn_var_legacy.sh
sbatch bnn_var_valprop.sh
sbatch bnn_var_quantile.sh
sbatch bnn_var_threshold.sh
sbatch bnn_var_outlier.sh
sbatch bnn_var_hetero.sh

echo 'Submitting flexible DNN scripts...'
sbatch flex_dnn_legacy.sh
sbatch flex_dnn_valprop.sh
sbatch flex_dnn_quantile.sh
sbatch flex_dnn_threshold.sh
sbatch flex_dnn_outlier.sh
sbatch flex_dnn_hetero.sh

echo 'Submitting conformal scripts...'
sbatch conformal_legacy.sh
sbatch conformal_valprop.sh
sbatch conformal_quantile.sh
sbatch conformal_threshold.sh
sbatch conformal_outlier.sh
sbatch conformal_hetero.sh

echo 'All jobs submitted!'
