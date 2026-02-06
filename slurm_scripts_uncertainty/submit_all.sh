#!/bin/bash
# Master script to submit all uncertainty experiments
# Limited to PDV and SNS representations

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty

echo "Submitting native uncertainty models (QRF, NGBoost, Gauche)..."
sbatch unc_qrf_legacy.sh
sbatch unc_qrf_valprop.sh
sbatch unc_qrf_quantile.sh
sbatch unc_qrf_threshold.sh
sbatch unc_qrf_outlier.sh
sbatch unc_qrf_hetero.sh
sbatch unc_ngboost_legacy.sh
sbatch unc_ngboost_valprop.sh
sbatch unc_ngboost_quantile.sh
sbatch unc_ngboost_threshold.sh
sbatch unc_ngboost_outlier.sh
sbatch unc_ngboost_hetero.sh
sbatch unc_gauche_legacy.sh
sbatch unc_gauche_valprop.sh
sbatch unc_gauche_quantile.sh
sbatch unc_gauche_threshold.sh
sbatch unc_gauche_outlier.sh
sbatch unc_gauche_hetero.sh

echo 'Submitting BNN variants...'
sbatch unc_dnn_bnn_full_legacy.sh
sbatch unc_dnn_bnn_full_valprop.sh
sbatch unc_dnn_bnn_full_quantile.sh
sbatch unc_dnn_bnn_full_threshold.sh
sbatch unc_dnn_bnn_full_outlier.sh
sbatch unc_dnn_bnn_full_hetero.sh
sbatch unc_dnn_bnn_last_legacy.sh
sbatch unc_dnn_bnn_last_valprop.sh
sbatch unc_dnn_bnn_last_quantile.sh
sbatch unc_dnn_bnn_last_threshold.sh
sbatch unc_dnn_bnn_last_outlier.sh
sbatch unc_dnn_bnn_last_hetero.sh
sbatch unc_dnn_bnn_var_legacy.sh
sbatch unc_dnn_bnn_var_valprop.sh
sbatch unc_dnn_bnn_var_quantile.sh
sbatch unc_dnn_bnn_var_threshold.sh
sbatch unc_dnn_bnn_var_outlier.sh
sbatch unc_dnn_bnn_var_hetero.sh
sbatch unc_mlp_bnn_full_legacy.sh
sbatch unc_mlp_bnn_full_valprop.sh
sbatch unc_mlp_bnn_full_quantile.sh
sbatch unc_mlp_bnn_full_threshold.sh
sbatch unc_mlp_bnn_full_outlier.sh
sbatch unc_mlp_bnn_full_hetero.sh
sbatch unc_mlp_bnn_last_legacy.sh
sbatch unc_mlp_bnn_last_valprop.sh
sbatch unc_mlp_bnn_last_quantile.sh
sbatch unc_mlp_bnn_last_threshold.sh
sbatch unc_mlp_bnn_last_outlier.sh
sbatch unc_mlp_bnn_last_hetero.sh
sbatch unc_mlp_bnn_var_legacy.sh
sbatch unc_mlp_bnn_var_valprop.sh
sbatch unc_mlp_bnn_var_quantile.sh
sbatch unc_mlp_bnn_var_threshold.sh
sbatch unc_mlp_bnn_var_outlier.sh
sbatch unc_mlp_bnn_var_hetero.sh

echo 'Submitting conformal models...'
sbatch unc_conformal_rf_legacy.sh
sbatch unc_conformal_rf_valprop.sh
sbatch unc_conformal_rf_quantile.sh
sbatch unc_conformal_rf_threshold.sh
sbatch unc_conformal_rf_outlier.sh
sbatch unc_conformal_rf_hetero.sh
sbatch unc_conformal_qrf_legacy.sh
sbatch unc_conformal_qrf_valprop.sh
sbatch unc_conformal_qrf_quantile.sh
sbatch unc_conformal_qrf_threshold.sh
sbatch unc_conformal_qrf_outlier.sh
sbatch unc_conformal_qrf_hetero.sh
sbatch unc_conformal_dnn_legacy.sh
sbatch unc_conformal_dnn_valprop.sh
sbatch unc_conformal_dnn_quantile.sh
sbatch unc_conformal_dnn_threshold.sh
sbatch unc_conformal_dnn_outlier.sh
sbatch unc_conformal_dnn_hetero.sh

echo 'All uncertainty jobs submitted!'
echo 'Total jobs: 72'
