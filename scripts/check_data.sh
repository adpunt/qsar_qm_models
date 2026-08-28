#!/bin/bash
# Diagnostic script to check existing data files and identify issues

echo "=========================================="
echo "DIAGNOSTIC: Checking Existing Data Files"
echo "=========================================="
echo ""

# Check if results directory exists
if [ ! -d "results" ]; then
    echo "ERROR: results/ directory not found!"
    echo "Please run this script from the project root directory"
    exit 1
fi

echo "=== Figure 1c Files ==="
echo "Checking for Figure 1c data files..."
echo ""

fig1c_files=(
    "results/fig1c_nn.csv"
    "results/fig1c_graph_nn.csv"
    "results/fig1c_bnn_full.csv"
    "results/fig1c_bnn_last.csv"
    "results/fig1c_bnn_var.csv"
    "results/fig1c_graph_bnn_full.csv"
    "results/fig1c_graph_bnn_last.csv"
    "results/fig1c_graph_bnn_var.csv"
)

for file in "${fig1c_files[@]}"; do
    if [ -f "$file" ]; then
        rows=$(wc -l < "$file")
        size=$(du -h "$file" | cut -f1)
        echo "✓ Found: $file"
        echo "  Rows: $rows, Size: $size"
        
        # Show unique model names
        if [ "$rows" -gt 1 ]; then
            echo "  Models:"
            tail -n +2 "$file" | cut -d',' -f4 | sort | uniq | sed 's/^/    - /'
            echo "  Reps:"
            tail -n +2 "$file" | cut -d',' -f5 | sort | uniq | sed 's/^/    - /'
        fi
        echo ""
    else
        echo "✗ Missing: $file"
        echo ""
    fi
done

echo "=== Figure 5b Files ==="
echo "Checking for Figure 5b data files..."
echo ""

# Check existing Fig 5 files
ls -lh results/fig5*.csv 2>/dev/null | awk '{print $9, "("$5")"}'
echo ""

# Check for Bayesian transformation files with different sizes
echo "Looking for Bayesian transformations with different sample sizes:"
for trans in "full" "last" "var"; do
    echo "  Transformation: $trans"
    found=0
    for n in 50 100 200 500; do
        file="results/fig5b_bnn_${trans}_n${n}.csv"
        if [ -f "$file" ]; then
            echo "    ✓ n=$n: $file"
            found=$((found + 1))
        else
            echo "    ✗ n=$n: MISSING"
        fi
    done
    if [ $found -eq 0 ]; then
        echo "    >> NO FILES FOUND FOR THIS TRANSFORMATION <<"
    fi
    echo ""
done

echo "=== Figure 6a Files ==="
echo "Checking for Figure 6a tuning files..."
echo ""

# Check tuned files
tuned_files=$(ls results/fig6a*tuned.csv 2>/dev/null)
if [ -z "$tuned_files" ]; then
    echo "No tuned files found!"
else
    for file in $tuned_files; do
        rows=$(wc -l < "$file")
        size=$(du -h "$file" | cut -f1)
        echo "Found: $(basename $file)"
        echo "  Rows: $rows, Size: $size"
        
        # Check for suspicious R² values
        if [ "$rows" -gt 1 ]; then
            echo "  Sample R² values:"
            tail -n +2 "$file" | cut -d',' -f9 | head -5 | sed 's/^/    /'
            
            # Check for extremely negative values
            bad_r2=$(tail -n +2 "$file" | cut -d',' -f9 | awk '$1 < -100 {count++} END {print count+0}')
            if [ "$bad_r2" -gt 0 ]; then
                echo "  ⚠️  WARNING: Found $bad_r2 rows with R² < -100!"
            fi
        fi
        echo ""
    done
fi

# Check default (non-tuned) files for Fig 6a
echo "Checking corresponding default (non-tuned) files:"
default_files=$(ls results/fig6a*.csv 2>/dev/null | grep -v tuned)
if [ -z "$default_files" ]; then
    echo "No default files found!"
else
    for file in $default_files; do
        rows=$(wc -l < "$file")
        size=$(du -h "$file" | cut -f1)
        echo "Found: $(basename $file)"
        echo "  Rows: $rows, Size: $size"
        
        if [ "$rows" -gt 1 ]; then
            echo "  Sample R² values:"
            tail -n +2 "$file" | cut -d',' -f9 | head -5 | sed 's/^/    /'
            
            # Check for extremely negative values
            bad_r2=$(tail -n +2 "$file" | cut -d',' -f9 | awk '$1 < -100 {count++} END {print count+0}')
            if [ "$bad_r2" -gt 0 ]; then
                echo "  ⚠️  WARNING: Found $bad_r2 rows with R² < -100!"
            fi
        fi
        echo ""
    done
fi

echo "=========================================="
echo "DIAGNOSTIC COMPLETE"
echo "=========================================="
echo ""
echo "Summary:"
echo "1. Check if all expected Fig 1c files exist above"
echo "2. Note which Fig 5b Bayesian files are missing"
echo "3. Check for suspicious R² values in Fig 6a files"
echo ""
echo "See debugging_analysis.md for detailed explanations and fixes"