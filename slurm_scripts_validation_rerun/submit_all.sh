#!/bin/bash
# Submit all validation re-run jobs
COUNT=0

sbatch val_dnn_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_dnn_sns_logd.sh
COUNT=$((COUNT + 1))
sbatch val_gp_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_gp_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_gp_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_lightgbm_sns_logd.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_ngboost_sns_logd.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_qrf_sns_logd.sh
COUNT=$((COUNT + 1))
sbatch val_rf_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_rf_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_rf_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_rf_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_rf_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_rf_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_rf_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_rf_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_rf_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_rf_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_rf_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_rf_sns_logd.sh
COUNT=$((COUNT + 1))
sbatch val_svm_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_svm_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_svm_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_svm_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_svm_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_svm_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_svm_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_svm_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_svm_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_svm_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_svm_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_svm_sns_logd.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_ecfp4_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_ecfp4_herg.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_ecfp4_logd.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_mhg_gnn_pretrained_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_mhg_gnn_pretrained_herg.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_mhg_gnn_pretrained_logd.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_pdv_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_pdv_herg.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_pdv_logd.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_sns_caco2.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_sns_herg.sh
COUNT=$((COUNT + 1))
sbatch val_xgboost_sns_logd.sh
COUNT=$((COUNT + 1))

echo "Submitted $COUNT jobs (expected 87)"

