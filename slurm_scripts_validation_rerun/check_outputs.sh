#!/bin/bash
# Verify all expected output files exist in validation_rerun/
# Run from: /data/stat-cadd/scat9264/KIRBy/tests

RERUN=results/validation_rerun
TOTAL=0
OK=0
MISSING=0
EMPTY=0

for REP_DS in ecfp4_logd ecfp4_caco2 ecfp4_herg sns_logd sns_caco2 sns_herg mhg_gnn_pretrained_logd mhg_gnn_pretrained_caco2 mhg_gnn_pretrained_herg pdv_logd pdv_caco2 pdv_herg; do
    DS=$(echo $REP_DS | rev | cut -d_ -f1 | rev)
    DIR="$RERUN/$REP_DS/$DS"

    if [ ! -f "$DIR/all_results.csv" ]; then
        # Check partial
        if [ -f "$DIR/all_results_partial.csv" ]; then
            ROWS=$(wc -l < "$DIR/all_results_partial.csv")
            echo "PARTIAL: $REP_DS — $ROWS rows (partial only)"
            MISSING=$((MISSING + 1))
        else
            echo "MISSING: $REP_DS — no output at all"
            MISSING=$((MISSING + 1))
        fi
    else
        ROWS=$(wc -l < "$DIR/all_results.csv")
        MODELS=$(cut -d, -f6 "$DIR/all_results.csv" | sort -u | grep -v model | tr '\n' ' ')
        TOTAL=$((TOTAL + 1))

        if [ "$ROWS" -le 1 ]; then
            echo "EMPTY:   $REP_DS — $ROWS rows"
            EMPTY=$((EMPTY + 1))
        else
            echo "OK:      $REP_DS — $ROWS rows — models: $MODELS"
            OK=$((OK + 1))
        fi
    fi
done

echo ""
echo "=== Summary ==="
echo "OK:      $OK / 12 rep×dataset combos"
echo "Missing: $MISSING"
echo "Empty:   $EMPTY"

if [ $OK -eq 12 ]; then
    echo ""
    echo "=== ALL OUTPUTS PRESENT ==="
    echo "Ready to merge."
fi
