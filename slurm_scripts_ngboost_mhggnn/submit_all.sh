#!/bin/bash
# Submit ngboost+mhggnn reruns (72h)

sbatch ngboost_mhggnn_legacy.sh
sbatch ngboost_mhggnn_valprop.sh
sbatch ngboost_mhggnn_quantile.sh
sbatch ngboost_mhggnn_threshold.sh
sbatch ngboost_mhggnn_outlier.sh
sbatch ngboost_mhggnn_hetero.sh

