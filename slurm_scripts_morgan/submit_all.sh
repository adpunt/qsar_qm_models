#!/bin/bash
# Submit all morgan (ECFP4 radius=2) experiments
# 14 ANOVA models × 6 strategies × 11 sigmas × 10 bootstraps
# + uncertainty extraction for NGBoost and QRF

echo "=== Morgan (ECFP4) ANOVA experiments ==="

echo "Fast models (RF, XGBoost, LightGBM, SVM, NGBoost × 6 strategies)..."
sbatch fast_models.sh

echo "NN models (DNN, MLP × 6 strategies)..."
sbatch nn_models.sh

echo "Gauche GP (× 6 strategies, with uncertainty)..."
sbatch gauche.sh

echo "DNN-BNN Full/Last (× 6 strategies, with uncertainty)..."
sbatch bnn_dnn.sh

echo "MLP-BNN Full/Last (× 6 strategies, with uncertainty)..."
sbatch bnn_mlp.sh

echo "DNN-VBLL/MLP-VBLL (× 6 strategies, with uncertainty)..."
sbatch vbll.sh

echo ""
echo "=== Morgan (ECFP4) UNCERTAINTY experiments ==="

echo "NGBoost uncertainty (× 6 strategies, -b 1 -u True)..."
sbatch unc_ngboost.sh

echo "QRF uncertainty (× 6 strategies, -b 1 -u True)..."
sbatch unc_qrf.sh

echo ""
echo "Total: 8 jobs submitted"
echo "  ANOVA (10 bootstraps each):"
echo "    fast_models.sh: 5 models × 6 strategies = 30 runs"
echo "    nn_models.sh:   2 models × 6 strategies = 12 runs"
echo "    gauche.sh:      1 model  × 6 strategies =  6 runs (+ uncertainty)"
echo "    bnn_dnn.sh:     2 models × 6 strategies = 12 runs (+ uncertainty)"
echo "    bnn_mlp.sh:     2 models × 6 strategies = 12 runs (+ uncertainty)"
echo "    vbll.sh:        2 models × 6 strategies = 12 runs (+ uncertainty)"
echo "  Uncertainty only (1 bootstrap each):"
echo "    unc_ngboost.sh: 1 model × 6 strategies =  6 runs"
echo "    unc_qrf.sh:     1 model × 6 strategies =  6 runs"
echo "  ---"
echo "  Total: 84 ANOVA runs + 12 uncertainty-only runs"
echo ""
echo "Output CSVs:"
echo "  ANOVA:       ../results/anova_{strategy}_morgan_{model}.csv"
echo "  Uncertainty: ../results/uncertainty_{strategy}_morgan_{model}.csv"
echo ""
echo "Use 'squeue -u scat9264' to monitor"
