#!/bin/bash
# Submit all mol2vec ANOVA jobs

sbatch mol2vec_rf.sh
sbatch mol2vec_xgboost.sh
sbatch mol2vec_lgb.sh
sbatch mol2vec_svm.sh
sbatch mol2vec_ngboost.sh
sbatch mol2vec_dnn.sh
sbatch mol2vec_mlp.sh
sbatch mol2vec_flexible_dnn.sh
sbatch mol2vec_flexible_dnn_256_128_64.sh
sbatch mol2vec_flexible_dnn_512_256.sh
sbatch mol2vec_dnn_bnn_last.sh
sbatch mol2vec_dnn_bnn_variational.sh
sbatch mol2vec_mlp_bnn_last.sh
sbatch mol2vec_mlp_bnn_variational.sh
sbatch mol2vec_dnn_bnn_full.sh
sbatch mol2vec_mlp_bnn_full.sh
sbatch mol2vec_gauche.sh

