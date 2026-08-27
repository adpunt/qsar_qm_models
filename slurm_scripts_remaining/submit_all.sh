#!/bin/bash
# Submit all remaining data generation jobs

echo "=== QRF missing strategies (QM9 ANOVA) ==="
sbatch qrf_hetero.sh
sbatch qrf_valprop.sh

echo "=== NGBoost validation (split by dataset, 72h walltime) ==="
sbatch val_ngboost_herg.sh
sbatch val_ngboost_caco2.sh
sbatch val_ngboost_logd.sh

echo "=== BNN validation (DNN-based, all 3 external datasets) ==="
sbatch val_bnn_full.sh
sbatch val_bnn_last.sh

echo ""
echo "Total: 7 jobs submitted"
echo ""
echo "NOT covered (need KIRBy code changes first):"
echo "  - MLP, MLP-BNN Full/Last/VBLL: KIRBy has no MLP models"
echo "  - DNN-VBLL, MLP-VBLL: VBLL not implemented in KIRBy"
echo ""
echo "Use 'squeue -u scat9264' to monitor"
