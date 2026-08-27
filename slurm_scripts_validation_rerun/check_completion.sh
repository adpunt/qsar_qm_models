#!/bin/bash
# Check which validation rerun jobs completed successfully
# Run from: /data/stat-cadd/scat9264/KIRBy/tests

RERUN_DIR="results/validation_rerun"
TOTAL=0
COMPLETE=0
MISSING=0

echo "=== Checking validation rerun completion ==="

for combo_dir in "$RERUN_DIR"/*/; do
    combo=$(basename "$combo_dir")
    # Extract dataset from dir name (last component after final _)
    dataset=$(echo "$combo" | rev | cut -d_ -f1 | rev)

    results_file="$combo_dir/$dataset/all_results.csv"
    TOTAL=$((TOTAL + 1))

    if [ -f "$results_file" ]; then
        rows=$(wc -l < "$results_file")
        if [ "$rows" -gt 1 ]; then
            COMPLETE=$((COMPLETE + 1))
        else
            echo "EMPTY: $combo ($results_file has $rows lines)"
            MISSING=$((MISSING + 1))
        fi
    else
        echo "MISSING: $combo (no $results_file)"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "=== Summary ==="
echo "Total expected: $TOTAL"
echo "Complete: $COMPLETE"
echo "Missing/empty: $MISSING"

if [ "$MISSING" -eq 0 ] && [ "$TOTAL" -gt 0 ]; then
    echo ""
    echo "All jobs complete. Safe to run merge_results.py"
fi
