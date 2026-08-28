#!/bin/bash
# Check all expected ANOVA files exist
cd /data/stat-cadd/scat9264/qsar_qm_models/results

missing=0
total=0

STRATEGIES="legacy quantile threshold outlier valprop hetero"
REPS="ecfp4 pdv smiles mhggnn mol2vec"
MODELS="rf xgboost ngboost svm dnn mlp lgb gauche flexible_dnn flexible_dnn_256_128_64 flexible_dnn_512_256 dnn_bnn_full dnn_bnn_last dnn_bnn_variational mlp_bnn_full mlp_bnn_last mlp_bnn_variational dnn_bnn_full_variational mlp_bnn_full_variational"

for strat in $STRATEGIES; do
    for rep in $REPS; do
        for model in $MODELS; do
            # Skip gauche+mhggnn (incompatible)
            if [ "$model" = "gauche" ] && [ "$rep" = "mhggnn" ]; then
                continue
            fi
            total=$((total+1))
            f="anova_${strat}_${rep}_${model}.csv"
            if [ ! -f "$f" ]; then
                echo "MISSING: $f"
                missing=$((missing+1))
            fi
        done
    done
done

echo ""
echo "=== SUMMARY ==="
echo "Total expected: $total"
echo "Total missing: $missing"
echo "Total present: $((total-missing))"
echo ""

echo "=== CONFORMAL + QRF (legacy only, ANOVA-excluded) ==="
for rep in $REPS; do
    for model in conformal_rf conformal_qrf conformal_dnn qrf; do
        f="anova_legacy_${rep}_${model}.csv"
        if [ ! -f "$f" ]; then
            echo "MISSING: $f"
        fi
    done
done

echo ""
echo "=== VALIDATION ==="
summary="/data/stat-cadd/scat9264/KIRBy/tests/results/alternative_full/combined_summary.csv"
if [ -f "$summary" ]; then
    echo "Models in combined_summary:"
    awk -F',' 'NR>1{print $2}' "$summary" | sort -u
    echo ""
    echo "Datasets:"
    awk -F',' 'NR>1{print $1}' "$summary" | sort -u
    echo ""
    echo "Row count: $(wc -l < "$summary")"
else
    echo "combined_summary.csv not found"
fi

echo ""
echo "Per-dataset NGBoost check:"
for d in caco2 herg logd; do
    f="/data/stat-cadd/scat9264/KIRBy/tests/results/alternative_full/${d}/summary.csv"
    if [ -f "$f" ]; then
        ngb=$(grep -c "NGBoost" "$f" 2>/dev/null || echo 0)
        echo "  $d: $ngb NGBoost rows"
    else
        echo "  $d: summary.csv not found"
    fi
done
