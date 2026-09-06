#!/bin/bash
# The decision report: every open figure choice in RERUN_PLAN.md 14.6 and 14.9,
# with the number that settles it. Stage 1 -- no figures yet, by design.
#
#   sbatch run_paper_analysis.sh
#
# Two things this does NOT do, and the old run_figures_v2.sh does both:
#   * it does not `eval` a micromamba shell hook. micromamba has never worked on
#     ARC and is not on PATH there, so setup.sh falls through to its conda
#     branch. The eval was silently a no-op.
#   * it does not point at /data/stat-cadd/.../KIRBy. KIRBy moved to stat-ecr on
#     2026-05-07 when stat-cadd hit quota; the old path is a stale checkout.
#SBATCH --job-name=paper_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --account=stat-cadd
#SBATCH --mem=128G
#SBATCH --output=/data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_analysis/paper_analysis-%j.out
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk

set -euo pipefail

QSAR=/data/stat-cadd/scat9264/qsar_qm_models
KIRBY=/data/stat-ecr/scat9264/KIRBy

cd "$QSAR"
. setup.sh

# setup.sh:234 sets this, and every generated job script sets it again anyway.
# Belt and braces, because without it scipy's compiled parts fail against the
# system libstdc++ in /lib64 forty lines into an import.
export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"

python -c "import sys; print('interpreter:', sys.executable)"

# Provenance in the log: which commit produced these numbers.
echo "branch: $(git rev-parse --abbrev-ref HEAD)"
echo "commit: $(git rev-parse --short HEAD)"

# A progress note in the log, nothing more. The analysis below runs on whatever
# has landed -- it does not wait for a complete grid and there is no collation
# step before it. This exits non-zero while anything is short, hence `|| true`.
python scripts/check_runs_landed.py \
  --qm9-dir "$QSAR/results" \
  --validation-dir "$KIRBY/results/validation_rerun" \
  --validation-dir "$KIRBY/tests/results/validation_rerun" \
  --uncertainty-dir "$KIRBY/results/uncertainty_rerun" \
  --uncertainty-dir "$KIRBY/tests/results/uncertainty_rerun" \
  --stage "${STAGE:-1}" || true

# The permutation band is the slow part. --permutations 0 skips it, and every
# Q4 number is then unreadable against a band, so the report says undecided
# rather than reporting a null nothing measured.
python scripts/run_paper_analysis.py \
  --qm9-dir "$QSAR/results" \
  --validation-dir "$KIRBY/results/validation_rerun" \
  --validation-dir "$KIRBY/tests/results/validation_rerun" \
  --uncertainty-dir "$KIRBY/results/uncertainty_rerun" \
  --uncertainty-dir "$KIRBY/tests/results/uncertainty_rerun" \
  --output-dir "$QSAR/results/decisions" \
  --cache-dir "$QSAR/results/.figcache" \
  --permutations "${PERMUTATIONS:-200}" \
  --only decisions

echo "done. read $QSAR/results/decisions/DECISIONS.md"
