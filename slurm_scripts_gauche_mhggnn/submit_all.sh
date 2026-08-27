#!/bin/bash
# Submit gauche+mhggnn jobs (extra resources)

sbatch gauche_mhggnn_legacy.sh
sbatch gauche_mhggnn_valprop.sh
sbatch gauche_mhggnn_quantile.sh
sbatch gauche_mhggnn_threshold.sh
sbatch gauche_mhggnn_outlier.sh
sbatch gauche_mhggnn_hetero.sh

