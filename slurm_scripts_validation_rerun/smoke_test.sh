#!/bin/bash
#SBATCH --job-name=val_smoke
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=short
#SBATCH --time=1:00:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# Smoke test: two models writing to same rep_dataset dir
# Verifies merge logic keeps both models' data

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. ../qsar_qm_models/setup.sh
cd tests

TESTDIR=results/validation_smoke_test/ecfp4_herg

echo "=== STEP 1: RF x ECFP4 x herg (sigma=0.0 only) ==="
python alternative_data_noise_robustness.py \
    --datasets herg \
    --models RF \
    --reps ECFP4 \
    --sigmas 0.0 \
    --results-root $TESTDIR

if [ ! -f $TESTDIR/herg/all_results.csv ]; then
    echo "FAIL: no output"
    exit 1
fi
echo "Rows: $(wc -l < $TESTDIR/herg/all_results.csv)"
echo "Models: $(cut -d, -f6 $TESTDIR/herg/all_results.csv | sort -u | grep -v model)"

echo ""
echo "=== STEP 2: SVM x ECFP4 x herg (sigma=0.0, should merge with RF) ==="
python alternative_data_noise_robustness.py \
    --datasets herg \
    --models SVM \
    --reps ECFP4 \
    --sigmas 0.0 \
    --results-root $TESTDIR

echo "Rows: $(wc -l < $TESTDIR/herg/all_results.csv)"
MODELS=$(cut -d, -f6 $TESTDIR/herg/all_results.csv | sort -u | grep -v model)
echo "Models: $MODELS"

if echo "$MODELS" | grep -q RF && echo "$MODELS" | grep -q SVM; then
    echo ""
    echo "=== SMOKE TEST PASSED ==="
else
    echo ""
    echo "=== FAIL: RF was overwritten by SVM ==="
    exit 1
fi

rm -rf results/validation_smoke_test
