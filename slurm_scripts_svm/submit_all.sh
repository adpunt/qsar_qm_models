#!/bin/bash
# Submit all SVM ANOVA experiment jobs

sbatch svm_legacy.sh
sbatch svm_valprop.sh
sbatch svm_quantile.sh
sbatch svm_threshold.sh
sbatch svm_outlier.sh
sbatch svm_hetero.sh

echo "Submitted 6 SVM jobs (6 strategies x 5 representations each)"
