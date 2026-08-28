#!/bin/bash
# Submit all SVM + continuous_pdv experiments

echo "=== SVM ANOVA (all reps except continuous_pdv) ==="
echo "  SVM × {ecfp4, smiles, mhggnn, mol2vec} × 6 strategies = 24 runs"
sbatch svm_all_reps.sh

echo ""
echo "=== continuous_pdv ANOVA (all 15 models) ==="

echo "  Fast models (RF, QRF, XGB, LGB, SVM, NGBoost × 6 strategies = 36 runs)..."
sbatch cpdv_fast.sh

echo "  NN models (DNN, MLP × 6 strategies = 12 runs)..."
sbatch cpdv_nn.sh

echo "  Gauche GP (× 6 strategies = 6 runs)..."
sbatch cpdv_gauche.sh

echo "  DNN-BNN Full/Last (× 6 strategies = 12 runs)..."
sbatch cpdv_bnn_dnn.sh

echo "  MLP-BNN Full/Last (× 6 strategies = 12 runs)..."
sbatch cpdv_bnn_mlp.sh

echo "  DNN-VBLL/MLP-VBLL (× 6 strategies = 12 runs)..."
sbatch cpdv_vbll.sh

echo ""
echo "=== continuous_pdv UNCERTAINTY (9 models, 1 iteration) ==="

echo "  QRF + NGBoost (× 6 strategies = 12 runs)..."
sbatch unc_cpdv_fast.sh

echo "  Gauche GP (× 6 strategies = 6 runs)..."
sbatch unc_cpdv_gauche.sh

echo "  DNN-BNN Full/Last (× 6 strategies = 12 runs)..."
sbatch unc_cpdv_bnn_dnn.sh

echo "  MLP-BNN Full/Last (× 6 strategies = 12 runs)..."
sbatch unc_cpdv_bnn_mlp.sh

echo "  DNN-VBLL/MLP-VBLL (× 6 strategies = 12 runs)..."
sbatch unc_cpdv_vbll.sh

echo ""
echo "=== SUMMARY ==="
echo "Total: 12 jobs submitted"
echo ""
echo "  SVM ANOVA:                24 runs (4 reps × 6 strategies × 10 iters)"
echo "  continuous_pdv ANOVA:     90 runs (15 models × 6 strategies × 10 iters)"
echo "  continuous_pdv Uncertainty: 54 runs (9 models × 6 strategies × 1 iter)"
echo "  ---"
echo "  Total experiment runs: 168"
echo ""
echo "  ANOVA output:       ../results/anova_{strategy}_*_{model}.csv"
echo "  Uncertainty output:  ../results/uncertainty_{strategy}_continuous_pdv_{model}.csv"
echo ""
echo "Use 'squeue -u scat9264' to monitor"
