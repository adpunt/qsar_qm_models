#!/bin/bash
# Submit ONLY the truly missing ANOVA jobs
# - dnn/mlp/gauche/dnn_bnn_full/mlp_bnn_full: only valprop + hetero (strategy name bug)
# - lgb: all 6 strategies (train_lgb_model was undefined)

# dnn: valprop + hetero only
sbatch dnn_valprop.sh
sbatch dnn_hetero.sh

# mlp: valprop + hetero only
sbatch mlp_valprop.sh
sbatch mlp_hetero.sh

# gauche: valprop + hetero only
sbatch gauche_valprop.sh
sbatch gauche_hetero.sh

# lgb: ALL strategies (no valid data exists)
sbatch lgb_legacy.sh
sbatch lgb_valprop.sh
sbatch lgb_quantile.sh
sbatch lgb_threshold.sh
sbatch lgb_outlier.sh
sbatch lgb_hetero.sh

# dnn_bnn_full: valprop + hetero only
sbatch dnn_bnn_full_valprop.sh
sbatch dnn_bnn_full_hetero.sh

# mlp_bnn_full: valprop + hetero only
sbatch mlp_bnn_full_valprop.sh
sbatch mlp_bnn_full_hetero.sh
