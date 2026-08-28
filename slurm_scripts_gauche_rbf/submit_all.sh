#!/bin/bash
# Submit all RBF-GP (gauche_rbf) ANOVA-completion jobs: 6 strategies x 4 reps = 24.
# PDV already has gauche_rbf; these add ecfp4/smiles/mhggnn/mol2vec.
set -e
sbatch gauche_rbf_legacy_ecfp4.sh
sbatch gauche_rbf_legacy_smiles.sh
sbatch gauche_rbf_legacy_mhggnn.sh
sbatch gauche_rbf_legacy_mol2vec.sh
sbatch gauche_rbf_valprop_ecfp4.sh
sbatch gauche_rbf_valprop_smiles.sh
sbatch gauche_rbf_valprop_mhggnn.sh
sbatch gauche_rbf_valprop_mol2vec.sh
sbatch gauche_rbf_quantile_ecfp4.sh
sbatch gauche_rbf_quantile_smiles.sh
sbatch gauche_rbf_quantile_mhggnn.sh
sbatch gauche_rbf_quantile_mol2vec.sh
sbatch gauche_rbf_threshold_ecfp4.sh
sbatch gauche_rbf_threshold_smiles.sh
sbatch gauche_rbf_threshold_mhggnn.sh
sbatch gauche_rbf_threshold_mol2vec.sh
sbatch gauche_rbf_outlier_ecfp4.sh
sbatch gauche_rbf_outlier_smiles.sh
sbatch gauche_rbf_outlier_mhggnn.sh
sbatch gauche_rbf_outlier_mol2vec.sh
sbatch gauche_rbf_hetero_ecfp4.sh
sbatch gauche_rbf_hetero_smiles.sh
sbatch gauche_rbf_hetero_mhggnn.sh
sbatch gauche_rbf_hetero_mol2vec.sh
