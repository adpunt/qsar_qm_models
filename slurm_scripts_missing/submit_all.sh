#!/bin/bash
# Submit all missing ANOVA jobs

sbatch dnn_legacy.sh
sbatch dnn_valprop.sh
sbatch dnn_quantile.sh
sbatch dnn_threshold.sh
sbatch dnn_outlier.sh
sbatch dnn_hetero.sh

sbatch mlp_legacy.sh
sbatch mlp_valprop.sh
sbatch mlp_quantile.sh
sbatch mlp_threshold.sh
sbatch mlp_outlier.sh
sbatch mlp_hetero.sh

sbatch gauche_legacy.sh
sbatch gauche_valprop.sh
sbatch gauche_quantile.sh
sbatch gauche_threshold.sh
sbatch gauche_outlier.sh
sbatch gauche_hetero.sh

sbatch lgb_legacy.sh
sbatch lgb_valprop.sh
sbatch lgb_quantile.sh
sbatch lgb_threshold.sh
sbatch lgb_outlier.sh
sbatch lgb_hetero.sh

sbatch dnn_bnn_full_legacy.sh
sbatch dnn_bnn_full_valprop.sh
sbatch dnn_bnn_full_quantile.sh
sbatch dnn_bnn_full_threshold.sh
sbatch dnn_bnn_full_outlier.sh
sbatch dnn_bnn_full_hetero.sh

sbatch mlp_bnn_full_legacy.sh
sbatch mlp_bnn_full_valprop.sh
sbatch mlp_bnn_full_quantile.sh
sbatch mlp_bnn_full_threshold.sh
sbatch mlp_bnn_full_outlier.sh
sbatch mlp_bnn_full_hetero.sh

