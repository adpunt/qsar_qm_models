#!/bin/bash
# Submit all missing valprop + hetero ANOVA jobs
# 11 models x 2 strategies = 22 jobs

sbatch rf_valprop.sh
sbatch rf_hetero.sh

sbatch xgboost_valprop.sh
sbatch xgboost_hetero.sh

sbatch ngboost_valprop.sh
sbatch ngboost_hetero.sh

sbatch svm_valprop.sh
sbatch svm_hetero.sh

sbatch flexible_dnn_valprop.sh
sbatch flexible_dnn_hetero.sh

sbatch flexible_dnn_256_128_64_valprop.sh
sbatch flexible_dnn_256_128_64_hetero.sh

sbatch flexible_dnn_512_256_valprop.sh
sbatch flexible_dnn_512_256_hetero.sh

sbatch dnn_bnn_last_valprop.sh
sbatch dnn_bnn_last_hetero.sh

sbatch dnn_bnn_variational_valprop.sh
sbatch dnn_bnn_variational_hetero.sh

sbatch mlp_bnn_last_valprop.sh
sbatch mlp_bnn_last_hetero.sh

sbatch mlp_bnn_variational_valprop.sh
sbatch mlp_bnn_variational_hetero.sh

