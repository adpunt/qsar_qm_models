#!/bin/bash
# Submit all deeper DNN architecture experiments
# 2 architectures × 6 strategies = 12 jobs

cd /data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_deeper_dnn

echo 'Submitting deep DNN scripts...'
sbatch fdnn_deep_legacy.sh
sbatch fdnn_deep_valprop.sh
sbatch fdnn_deep_quantile.sh
sbatch fdnn_deep_threshold.sh
sbatch fdnn_deep_outlier.sh
sbatch fdnn_deep_hetero.sh

echo 'Submitting wide DNN scripts...'
sbatch fdnn_wide_legacy.sh
sbatch fdnn_wide_valprop.sh
sbatch fdnn_wide_quantile.sh
sbatch fdnn_wide_threshold.sh
sbatch fdnn_wide_outlier.sh
sbatch fdnn_wide_hetero.sh

echo 'All deeper DNN jobs submitted!'
