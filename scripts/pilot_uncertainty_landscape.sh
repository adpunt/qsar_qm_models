#!/usr/bin/env bash
# A small uncertainty landscape on QM9, run locally, before the grid is sized.
#
# WHAT IT IS FOR
# The uncertainty grid is 420 tasks and 88,200 model fits. Three things about it
# are open -- how many representations it carries, whether cross-fitting can be
# cut, and whether all five noise types earn their place -- and none can be
# settled by argument. This is the look at the data that settles them.
#
# It is deliberately narrow: two representations, two models, one replicate. It
# answers "does uncertainty find corrupted labels at all, and how much
# cross-fitting does that take". It says NOTHING about Avalon or ChemBERTa, and
# nothing about whether the answer carries to logd/caco2/hERG.
#
# WHICH INTERPRETER, AND WHY IT IS NOT THE DEFAULT ONE
# The laptop's base Anaconda has scikit-learn 1.3.2 while pip-constraints.txt
# pins 1.6.1, and quantile-forest 1.4.x needs the newer one -- so QRF, the
# strongest error-ranker in the existing results, raises
# `Invalid parameter 'monotonic_cst'` there and always has.
#
# The fix is NOT to pip install scikit-learn into the base environment. That is
# precisely the failure RERUN_PLAN.md 2.8i identified: a PyPI wheel installed
# over a conda package brings its own OpenMP runtime, and four runtimes in one
# interpreter is what makes LightGBM and the Gaussian process segfault with no
# traceback. env_test is already built from conda-forge with the pinned versions
# -- scikit-learn 1.6.1, quantile-forest 1.4.0, torch 2.5.1 -- and it is on this
# laptop. Use it and change nothing.
#
# WHY NGBOOST IS NOT HERE
# Measured on this laptop, one (model, rep, condition, level) cell at n=5000
# with --oof-folds 3: BNN 96 s, QRF 656 s, NGBoost 956 s. All three would be
# about 9.4 hours. The approved plan says to drop NGBoost first if the probe
# lands over four hours, and it did. QRF covers the tree-based uncertainty case
# and is the better error-ranker of the two.
#
#   bash scripts/pilot_uncertainty_landscape.sh
#   PILOT_MODELS="qrf" bash scripts/pilot_uncertainty_landscape.sh    # one model
#
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

PY="${PILOT_PYTHON:-/usr/local/Cellar/micromamba/2.0.8/envs/env_test/bin/python}"
if [ ! -x "$PY" ]; then
    echo "ERROR: no interpreter at $PY"
    echo "       Set PILOT_PYTHON to an env_test with scikit-learn 1.6.1 and"
    echo "       quantile-forest installed. The base Anaconda will not do: QRF"
    echo "       raises Invalid parameter 'monotonic_cst' there."
    exit 2
fi
"$PY" - <<'PYCHECK' || exit 2
import sys
import sklearn, quantile_forest
if tuple(int(x) for x in sklearn.__version__.split('.')[:2]) < (1, 4):
    print(f"ERROR: scikit-learn {sklearn.__version__} is too old for "
          f"quantile-forest {quantile_forest.__version__} — QRF will raise "
          f"Invalid parameter 'monotonic_cst' on every fit.")
    sys.exit(1)
print(f"=== interpreter OK: scikit-learn {sklearn.__version__}, "
      f"quantile-forest {quantile_forest.__version__}")
PYCHECK

OUT="${PILOT_OUT:-$REPO/results/pilot_uncertainty}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

SAMPLE=5000
OOF=3
REPS="ecfp4 continuous_pdv"
MODELS="${PILOT_MODELS:-qrf dnn_bnn_full}"

# The flag string for each model, as the QM9 job generator writes it
# (slurm_scripts_qm9_rerun/generate_scripts.py MODELS), so the pilot asks for
# exactly what the real run asks for.
model_flags() {
    case "$1" in
        qrf)          echo "-m qrf -u True" ;;
        ngboost)      echo "-m ngboost -u True" ;;
        dnn_bnn_full) echo "-m dnn --bayesian-transformation full -u True" ;;
        *) echo "unknown model $1" >&2; return 1 ;;
    esac
}

# The five noise types, translated the same way the QM9 job generator does
# (CONDITION_FLAGS). Levels: 0 and the reporting level 1.5. Censoring is swept on
# its own axis -- the fraction of labels clipped -- so its levels are fractions.
#
# Level 0 is run INSIDE every noise type, not once and copied. copy_zero_rows.py
# deliberately skips uncertainty files, because `noise_pattern` is the noise
# type's own shape taken at a fixed reference level and is NOT zero at level 0.
# That column is the entire basis of the level-0 subtraction.
condition_flags() {
    case "$1" in
        gaussian)        echo "--noise-shape gaussian --noise-targeting uniform" ;;
        grouped_wider)   echo "--noise-shape gaussian --noise-targeting grouped_wide" ;;
        grouped_shifted) echo "--noise-shape gaussian --noise-targeting grouped_shift" ;;
        censoring)       echo "--noise-targeting censoring --censor-side upper" ;;
        outlier_p10)     echo "--noise-shape gaussian --noise-targeting outlier --outlier-p 0.1" ;;
        *) echo "unknown condition $1" >&2; return 1 ;;
    esac
}
condition_levels() {
    case "$1" in
        censoring) echo "0.0 0.30" ;;
        *)         echo "0.0 1.5" ;;
    esac
}

CONDITIONS="gaussian grouped_wider grouped_shifted censoring outlier_p10"

echo "=== pilot: QM9 n=$SAMPLE, 1 replicate, --oof-folds $OOF"
echo "    models:     $MODELS"
echo "    reps:       $REPS"
echo "    conditions: $CONDITIONS"
echo "    out:        $OUT"
echo

started=$(date +%s)
fail=0
for model in $MODELS; do
    for cond in $CONDITIONS; do
        levels="$(condition_levels "$cond")"
        log="$LOGS/${model}__${cond}.log"
        echo "--- $model / $cond (levels $levels) -> $log"
        cell_start=$(date +%s)
        # shellcheck disable=SC2086
        "$PY" scripts/process_and_train.py \
            -d QM9 -n "$SAMPLE" -b 1 \
            $(model_flags "$model") \
            -r $REPS \
            --oof-folds "$OOF" \
            --noise-level $levels \
            $(condition_flags "$cond") \
            -f "$OUT/${model}__${cond}.csv" \
            > "$log" 2>&1
        status=$?
        cell_end=$(date +%s)
        if [ $status -ne 0 ]; then
            fail=$((fail + 1))
            echo "    FAILED (exit $status) after $(( cell_end - cell_start ))s — see $log"
            grep -E "^ERROR|RuntimeError|ValueError" "$log" | head -3 | sed 's/^/      /'
        else
            echo "    ok, $(( cell_end - cell_start ))s"
        fi
    done
done

echo
echo "=== finished in $(( ($(date +%s) - started) / 60 )) minutes, $fail cell(s) failed"
echo "    uncertainty rows:"
find "$OUT" -name '*_uncertainty_values.csv' | wc -l
echo
echo "Next: python scripts/uncertainty_stats.py over $OUT"
[ "$fail" -eq 0 ]
