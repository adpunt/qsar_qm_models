#!/bin/bash
# The decision report AND the figures and tables: every open figure choice in
# RERUN_PLAN.md 14.6 and 14.9, with the number that settles it, then every slot
# the data supports drawn from those numbers.
#
#   sbatch run_paper_analysis.sh
#
# ONLY=decisions gets the report alone, which is what this did before F6 and F7
# existed (2026-09-10, section 14.11). Nothing drawn is a taste: a contingent
# figure appears only when its decision fired, and F7 draws whichever of its
# three options D7 chose.
#
# Two things this does NOT do, and the old run_figures_v2.sh does both:
#   * it does not `eval` a micromamba shell hook. micromamba has never worked on
#     ARC and is not on PATH there, so setup.sh falls through to its conda
#     branch. The eval was silently a no-op.
#   * it does not point at /data/stat-cadd/.../KIRBy. KIRBy moved to stat-ecr on
#     2026-05-07 when stat-cadd hit quota; the old path is a stale checkout.
# WHY THIS ASKS SO LITTLE, and it is the reason 13105402 sat on Priority
# overnight while four-hour jobs ran past it.
#
#   memory   128G was never measured. The worst peak across ~1,400 finished
#            tasks of this whole study is 4.1 GB (13.23 B3), and this pass holds
#            ONE file per worker -- eight files, not the set. 32G is eight times
#            the largest file this has ever been pointed at.
#   time     12:00:00 predates the 2026-09-12 speed work: the permutation band
#            was 72% of the run and the pass used one of eight cores (14.11).
#            About 7x faster now, and the caches mean a second run reads
#            neither the grid nor the per-molecule files.
#   long     asks for a 30-day partition to do a job of hours. `short` caps at
#            12:00:00, which is four times this wall, and a three-hour job is
#            what fits a backfill window -- measured between 6 and 19 hours
#            (13.27 D2w). Read `sinfo -s` before overriding.
#
# All three are overridable at submit time, because the file count only grows.
# On the COMMAND LINE, not as environment variables -- these lines are read by
# sbatch before any shell runs, so nothing in the script's own environment can
# reach them, and a flag on the command line wins over the directive here:
#   sbatch --mem=64G --time=06:00:00 --partition=medium \
#       slurm_scripts_analysis/run_paper_analysis.sh
#SBATCH --job-name=paper_analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --partition=short
#SBATCH --account=stat-cadd
#SBATCH --mem=32G
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

# The uncertainty statistics read every per-molecule file, one at a time -- the
# whole set is hundreds of millions of rows and will not fit in memory. Set
# SKIP_UNCERTAINTY=1 for the accuracy half alone.
#
# The permutation band is the slow part. PERMUTATIONS=0 skips it, and every Q4
# number is then unreadable against a band, so the report says undecided rather
# than reporting a null nothing measured.
python scripts/run_paper_analysis.py \
  --qm9-dir "$QSAR/results" \
  --validation-dir "$KIRBY/results/validation_rerun" \
  --validation-dir "$KIRBY/tests/results/validation_rerun" \
  --uncertainty-dir "$KIRBY/results/uncertainty_rerun" \
  --uncertainty-dir "$KIRBY/tests/results/uncertainty_rerun" \
  --output-dir "$QSAR/results/decisions" \
  --cache-dir "$QSAR/results/.figcache" \
  --permutations "${PERMUTATIONS:-200}" \
  ${SKIP_UNCERTAINTY:+--skip-uncertainty} \
  --only "${ONLY:-all}"

echo "done in ${SECONDS}s. read $QSAR/results/decisions/DECISIONS.md"
echo "figures in $QSAR/results/decisions/figures, tables in .../tables"
echo
echo "what this cost, for the next submission:"
sacct -j "${SLURM_JOB_ID:-0}" --format=JobID,Elapsed,MaxRSS,ReqMem,State -P 2>/dev/null \
    || echo "  (sacct not available here; read it after the job ends)"
