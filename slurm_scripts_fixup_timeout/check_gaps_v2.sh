#!/bin/bash
# Complete gap check aligned with generate_paper_figures.py data requirements
# Checks ALL data needed for every figure and table in the paper
#
# Data requirements (from figure gen trace):
#   ANOVA data (anova_*.csv):  10 reps × 11 sigmas = 110 rows per (strategy × model)
#     → TABLE 1 (η²), TABLE 2 (NDS), TABLE 3 (Wilcoxon), TABLE 5 (sigma ranks),
#       TABLE 6 (Kendall W), TABLE 7 (sensitivity), FIG 1-5, FIG interaction
#   Uncertainty data (*_uncertainty_values.csv): 1+ reps, 100+ samples
#     → TABLE 4 (uncertainty metrics), Uncertainty figure
#
# NAMING CONVENTIONS:
#   Old scripts used short strategy names: hetero, valprop
#   New scripts use full names: heteroscedastic, value_proportional
#   generate_paper_figures.py handles both (STRATEGY_NORMALIZE)
#   This script checks BOTH naming conventions

RESULTS="/data/stat-cadd/scat9264/qsar_qm_models/results"
ANOVA_THRESHOLD=100   # expect 110 rows (10 reps × 11 sigmas), allow some slack
UNC_THRESHOLD=10      # uncertainty needs samples, not many rows

check() {
    local FILE="$1"
    local LABEL="$2"
    local THRESH="$3"
    if [ ! -f "$FILE" ]; then
        echo "  MISSING      ${LABEL}"
        return 1
    fi
    DATA_ROWS=$(($(wc -l < "$FILE") - 1))
    if [ "$DATA_ROWS" -ge "$THRESH" ]; then
        return 0
    else
        echo "  INCOMPLETE   ${LABEL} (${DATA_ROWS} rows, need >=${THRESH})"
        return 1
    fi
}

# Check with fallback: try primary filename, then alternate naming convention
check_with_fallback() {
    local FILE1="$1"
    local FILE2="$2"
    local LABEL="$3"
    local THRESH="$4"
    if [ -f "$FILE1" ]; then
        check "$FILE1" "$LABEL" "$THRESH"
        return $?
    elif [ -f "$FILE2" ]; then
        check "$FILE2" "$LABEL" "$THRESH"
        return $?
    else
        echo "  MISSING      ${LABEL}"
        return 1
    fi
}

# =========================================================================
# ANOVA models (14 total) — matches generate_paper_figures.py after exclusions
# =========================================================================
ANOVA_MODELS="rf xgboost lgb svm ngboost gauche dnn mlp dnn_bnn_full dnn_bnn_last dnn_bnn_full_variational mlp_bnn_full mlp_bnn_last mlp_bnn_full_variational"

# Models that produce uncertainty values (Bayesian/ensemble models with -u True)
UNC_MODELS="ngboost gauche dnn_bnn_full dnn_bnn_last dnn_bnn_full_variational mlp_bnn_full mlp_bnn_last mlp_bnn_full_variational"

# Strategies: use full names as primary, short names as fallback
STRATEGIES="legacy quantile threshold outlier heteroscedastic value_proportional"

# ANOVA reps (5 total)
ANOVA_REPS="ecfp4 continuous_pdv smiles mhggnn mol2vec"

PRIMARY_REP="continuous_pdv"

# Map full strategy name to short name for fallback filename check
strategy_short() {
    case "$1" in
        heteroscedastic) echo "hetero" ;;
        value_proportional) echo "valprop" ;;
        *) echo "$1" ;;
    esac
}

echo "========================================================"
echo "  COMPLETE DATA GAP CHECK"
echo "  Aligned with generate_paper_figures.py requirements"
echo "  $(date)"
echo "========================================================"

# =========================================================================
# SECTION 1: ANOVA data — needed for TABLE 1,2,3,5,6,7 and FIG 1-5
# =========================================================================
echo ""
echo "============================================"
echo "  SECTION 1: ANOVA DATA (14 models × 5 reps × 6 strategies)"
echo "  Needed for: TABLE 1 (η²), TABLE 2 (NDS), TABLE 3 (Wilcoxon),"
echo "  TABLE 5 (sigma ranks), TABLE 6 (Kendall W), TABLE 7 (sensitivity),"
echo "  FIG 1 (overview), FIG 2 (ANOVA), FIG 3 (rankings),"
echo "  FIG 4 (DNN family), FIG 5 (MLP/RF family), FIG interaction"
echo "============================================"

TOTAL_ANOVA_PROBLEMS=0

for REP in $ANOVA_REPS; do
    echo ""
    echo "--- ${REP} ---"
    REP_PROBLEMS=0
    for STRATEGY in $STRATEGIES; do
        SHORT=$(strategy_short "$STRATEGY")
        for MODEL in $ANOVA_MODELS; do
            check_with_fallback \
                "$RESULTS/anova_${STRATEGY}_${REP}_${MODEL}.csv" \
                "$RESULTS/anova_${SHORT}_${REP}_${MODEL}.csv" \
                "${STRATEGY} × ${MODEL}" \
                $ANOVA_THRESHOLD || ((REP_PROBLEMS++))
        done
    done
    if [ "$REP_PROBLEMS" -eq 0 ]; then
        echo "  All OK (${REP})"
    else
        echo "  ${REP_PROBLEMS} problems (${REP})"
    fi
    TOTAL_ANOVA_PROBLEMS=$((TOTAL_ANOVA_PROBLEMS + REP_PROBLEMS))
done

echo ""
if [ "$TOTAL_ANOVA_PROBLEMS" -eq 0 ]; then
    echo ">>> ANOVA: ALL COMPLETE (14 models × 5 reps × 6 strategies)"
else
    echo ">>> ANOVA: ${TOTAL_ANOVA_PROBLEMS} TOTAL PROBLEMS"
fi

# =========================================================================
# SECTION 2: Uncertainty values from ANOVA runs (-u True)
# =========================================================================
echo ""
echo "============================================"
echo "  SECTION 2: UNCERTAINTY DATA (from ANOVA runs with -u True)"
echo "  Needed for: TABLE 4 (ECE, coverage, correlations),"
echo "  Uncertainty figure (mean unc vs sigma)"
echo "  Checking PRIMARY_REP (${PRIMARY_REP}) only"
echo "============================================"

UNC_PROBLEMS=0
for STRATEGY in $STRATEGIES; do
    SHORT=$(strategy_short "$STRATEGY")
    for MODEL in $UNC_MODELS; do
        check_with_fallback \
            "$RESULTS/anova_${STRATEGY}_${PRIMARY_REP}_${MODEL}_uncertainty_values.csv" \
            "$RESULTS/anova_${SHORT}_${PRIMARY_REP}_${MODEL}_uncertainty_values.csv" \
            "unc: ${STRATEGY} × ${MODEL}" \
            $UNC_THRESHOLD || ((UNC_PROBLEMS++))
    done
done

if [ "$UNC_PROBLEMS" -eq 0 ]; then
    echo "  All uncertainty files OK"
else
    echo "  ${UNC_PROBLEMS} uncertainty problems"
fi

# =========================================================================
# SECTION 3: Dedicated uncertainty CSVs (QRF, NGBoost with -b 1)
# =========================================================================
echo ""
echo "============================================"
echo "  SECTION 3: DEDICATED UNCERTAINTY (qrf, ngboost × -b 1)"
echo "============================================"

DED_PROBLEMS=0
for STRATEGY in $STRATEGIES; do
    SHORT=$(strategy_short "$STRATEGY")
    for MODEL in ngboost qrf; do
        check_with_fallback \
            "$RESULTS/uncertainty_${STRATEGY}_${PRIMARY_REP}_${MODEL}.csv" \
            "$RESULTS/uncertainty_${SHORT}_${PRIMARY_REP}_${MODEL}.csv" \
            "ded_unc: ${STRATEGY} × ${MODEL}" \
            $UNC_THRESHOLD || ((DED_PROBLEMS++))
        check_with_fallback \
            "$RESULTS/uncertainty_${STRATEGY}_${PRIMARY_REP}_${MODEL}_uncertainty_values.csv" \
            "$RESULTS/uncertainty_${SHORT}_${PRIMARY_REP}_${MODEL}_uncertainty_values.csv" \
            "ded_unc_vals: ${STRATEGY} × ${MODEL}" \
            $UNC_THRESHOLD || ((DED_PROBLEMS++))
    done
done

if [ "$DED_PROBLEMS" -eq 0 ]; then
    echo "  All dedicated uncertainty files OK"
else
    echo "  ${DED_PROBLEMS} dedicated uncertainty problems"
fi

# =========================================================================
# SUMMARY
# =========================================================================
echo ""
echo "========================================================"
echo "  SUMMARY"
echo "========================================================"
TOTAL=$((TOTAL_ANOVA_PROBLEMS + UNC_PROBLEMS + DED_PROBLEMS))
echo "  ANOVA:                ${TOTAL_ANOVA_PROBLEMS} problems"
echo "  Uncertainty (ANOVA):  ${UNC_PROBLEMS} problems"
echo "  Uncertainty (dedicated): ${DED_PROBLEMS} problems"
echo "  TOTAL:                ${TOTAL} problems"
if [ "$TOTAL" -eq 0 ]; then
    echo ""
    echo "  >>> ALL DATA COMPLETE — ready to run generate_paper_figures.py"
fi
echo "========================================================"
