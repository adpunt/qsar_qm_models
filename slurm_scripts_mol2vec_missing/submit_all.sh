#!/bin/bash
# Submit all missing mol2vec experiments
# 15 missing combinations total:
#   dnn_bnn_last/mol2vec: 6 strategies (will likely be excluded - bad baseline)
#   mlp_bnn_last/mol2vec: 6 strategies (will likely be excluded - bad baseline)
#   ngboost/mol2vec: outlier + hetero (real data expected)
#   gauche/mol2vec: hetero (real data expected)

sbatch mol2vec_dnn_bnn_last.sh
sbatch mol2vec_mlp_bnn_last.sh
sbatch mol2vec_ngboost_missing.sh
sbatch mol2vec_gauche_missing.sh

echo "Submitted 4 jobs for 15 missing mol2vec combinations"
