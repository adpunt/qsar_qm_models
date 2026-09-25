#!/bin/bash
# EVERY figure and table in the paper, from one command, on ARC.
#
#     bash scripts/regenerate_figures.sh
#     bash scripts/regenerate_figures.sh --dry-run    # says what each step would do
#
# Run it from the login node, from anywhere. It does the four things that used
# to be four commands in three directories, in the order they have to happen:
#
#   1  pull, via scripts/pull_safely.sh, because a fix that is not on the
#      cluster does not exist
#   2  fill in the clean row for every noise condition that did not run one.
#      The job scripts run noise level 0 under Gaussian only and copy it across,
#      because at level 0 the fit is bit-identical whichever condition labels
#      it. AUC_norm divides each condition's curve by that condition's own clean
#      accuracy, so a condition with no clean row produces nothing at all
#   3  merge the laboratory runs into the archive the assay figures read. The
#      jobs write one directory per model, representation and dataset; nothing
#      downstream reads those
#   4  submit the analysis, which writes the decision report, every figure and
#      every table
#
# Steps 2 and 3 are idempotent and refuse to overwrite a row a job computed, so
# running this twice is safe and running it on a half-finished grid is safe --
# the analysis draws whatever has landed and the report says what is short.
#
# Environment, all optional and all passed straight through:
#   ONLY=decisions        the report alone, no figures or tables
#   SKIP_UNCERTAINTY=1    the accuracy half only, which is the fast half
#   PERMUTATIONS=0        skip the permutation band
#   STAGE=2               which stage check_runs_landed.py reports against
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
QSAR="$(pwd)"
KIRBY="${KIRBY_DIR:-/data/stat-ecr/scat9264/KIRBy}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

step() { echo ""; echo "=== $* ==="; }
run()  { if [ "$DRY" = 1 ]; then echo "would run: $*"; else "$@"; fi; }

echo "repo:  $QSAR"
echo "KIRBy: $KIRBY"
[ "$DRY" = 1 ] && echo "DRY RUN -- nothing will be changed or submitted"

step "1/4  pull"
if [ "$DRY" = 1 ]; then DRY_RUN=1 bash scripts/pull_safely.sh; else bash scripts/pull_safely.sh; fi

step "2/4  clean rows for the conditions that borrow one"
run python "$QSAR/slurm_scripts_qm9_rerun/copy_zero_rows.py" \
    --results "$QSAR/results" ${DRY:+}$([ "$DRY" = 1 ] && echo --dry-run)

step "3/4  merge the laboratory runs"
run python "$QSAR/slurm_scripts_validation_rerun/merge_results.py" \
    --kirby-dir "$KIRBY" $([ "$DRY" = 1 ] && echo --dry-run)

step "4/4  submit the analysis"
if [ "$DRY" = 1 ]; then
    echo "would run: sbatch $QSAR/slurm_scripts_analysis/run_paper_analysis.sh"
    exit 0
fi
if ! command -v sbatch >/dev/null 2>&1; then
    echo "no sbatch here, so this is not ARC. Steps 1 to 3 are done; run step 4 on the cluster."
    exit 1
fi
JOB=$(sbatch --parsable "$QSAR/slurm_scripts_analysis/run_paper_analysis.sh") || exit 1
echo "submitted $JOB"
echo ""
echo "watch it:   squeue -j $JOB"
echo "log:        $QSAR/slurm_scripts_analysis/paper_analysis-$JOB.out"
echo "when done:  $QSAR/results/decisions/DECISIONS.md"
echo "            $QSAR/results/decisions/figures, $QSAR/results/decisions/tables"
echo ""
echo "to bring it home, from the laptop:"
echo "  rsync -av arc:$QSAR/results/decisions/ results/decisions_arc/"
