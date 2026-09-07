#!/usr/bin/env python3
"""Generate the SLURM job-array scripts for the uncertainty re-run.

WHAT THIS RUN IS FOR
--------------------
Two questions, one set of jobs:

  (A) Do the molecules whose labels were corrupted come back as the uncertain
      ones?  Measured on TRAINING molecules, scored OUT-OF-FOLD so that no
      molecule is judged by a model that fitted its own corrupted label.
      Needs  --oof-folds N.

  (B) Does the model learn WHERE the data is unreliable?  Measured on TEST
      molecules against the noise scale their region of the label distribution
      receives.  Only some conditions give different molecules different
      amounts, so only those have a pattern to learn -- see the next section.
      Needs  --unc-conditions all.

WHAT CHANGED ON 2026-08-27, AND WHY THE OLD SCRIPTS COULD NOT RUN
-----------------------------------------------------------------
This generator was written against the noise scheme that has since been
replaced (NOISE_DESIGN.md). It listed six noise strategies -- legacy, outlier,
quantile, hetero, threshold, valprop -- which were deleted in noiseInject 1.0.0
because they were one strategy at six doses: at a common nominal setting they
delivered between 0.49x and 2.00x the same amount of noise, and their whole
apparent severity ordering was that. It emitted `--strategies`,
`--unc-strategies` and `--threshold-quantile`, none of which the runner has any
more; it now takes `--conditions` and `--unc-conditions`, and `--conditions`
carries `choices=`, so a job asking for a deleted name dies at argument
parsing. Every script this file used to write would have failed there.

WHICH CONDITIONS THIS RUN USES, AND WHY IT DOES NOT CHOOSE THEM ITSELF
----------------------------------------------------------------------
noise_conditions.json is the settled set. It is read by tests on both
injectors, by the QM9 job generator, and here -- restating it is how the two
injectors drifted apart for the life of the project.

This run uses ALL SEVEN settled conditions: gaussian, grouped_wider,
grouped_shifted, censoring, student_t_nu5, outlier_p10 and laplace. The author's
decision, 2026-08-28. It used to be five.

WHY ALL SEVEN, AND WHAT IT BUYS
-------------------------------
Three questions are asked here, not two.

  A. Does a model become less sure of itself as its training labels get worse?
  B. Can a model's uncertainty point at WHICH labels are bad?
  C. Does uncertainty track some KINDS of noise better than others?

C is the reason for all seven, and it is a result either way it comes out. The
conditions were chosen for the main grid on ACCURACY -- which of them changes
R2. A kind of noise can barely move accuracy and still be the one a model is
best, or worst, at noticing. That is a different property and nothing has
measured it.

WHAT EACH CONDITION CAN AND CANNOT ANSWER
-----------------------------------------
For B to mean anything, some molecules must be more corrupted than others. Under
a condition that gives every molecule the SAME amount, B is undefined -- not
zero, undefined.

  gaussian          same amount for every molecule. A's condition, and the
                    leakage check.
  laplace           same amount for every molecule, heavier tails.
  student_t_nu5     same amount for every molecule, heavier tails still.
  grouped_shifted   same amount for every molecule -- whole scaffold families
                    pushed one way, by a constant.
  grouped_wider     whole scaffold families get a LARGER amount. A real
                    per-molecule pattern, keyed to the scaffold.
  censoring         values past the assay limit recorded as the limit, so the
                    damage is keyed to the label. A real per-molecule pattern.
  outlier_p10       a tenth of the molecules take nearly all of the noise. The
                    most concentrated pattern in the study.

So B is answered by three of the seven. All seven answer A and C.

Both grouped conditions are keyed to something a scaffold split holds out
whole, which is RERUN_PLAN.md 3.1d: on HELD-OUT molecules the grouped pattern
is flat, truthfully, and the predicted-label control is degenerate for it. That
is a Methods sentence, not a defect here. Censoring and outlier are keyed to the
label and to the draw, so neither is affected.

The levels are NOT passed. The runner sweeps one shared grid in fractions of
each dataset's own clean training label spread -- 0, 0.2, 0.3, 0.5, 0.75, 1.0,
1.5, the same grid QM9 runs, so the same number means the same relative
corruption everywhere (author's decision, 2026-08-27). Censoring runs on its own
axis, the fraction of labels clipped, because it has no variance parameter and
cannot be dose-matched. Passing --sigmas would override both.

DESIGN NOTES
------------
* One array TASK per (dataset, representation, condition); one SCRIPT per model.
* Every task writes to its OWN --results-root.  The pipeline merges results by
  read-modify-write, so two concurrent tasks sharing a directory would race and
  silently lose rows.  merge_results.py stitches them back together afterwards.
* Cross-fitting is applied only to models that emit a per-molecule uncertainty
  (the pipeline enforces this too, via UNCERTAINTY_MODELS).
* GP must be told which reps to run on (--gp-reps); it defaults to PDV only.
* One replicate, plus a permutation null (RERUN_PLAN.md 13.1 item 3, the
  recorded default). The runner has no replicate axis; the five scaffold folds
  are the only repeat, and the null is computed afterwards by
  scripts/uncertainty_stats.py.

Checked by scripts/test_uncertainty_job_scripts.py, which generates real
scripts and runs the command line each one emits through the runner's own
argument parser.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

# Models that emit a per-molecule uncertainty. Must match UNCERTAINTY_MODELS in
# KIRBy/tests/alternative_data_noise_robustness.py.
MODELS = {
    # name          : (tier, cpus, note)   -- memory is in model_memory.json,
    #                                        the wall clock is COMPUTED (wall_clock below)
    #
    # FOUR, not seven, by the AUTHOR'S DECISION of 2026-08-28 (RERUN_PLAN.md chat N).
    # The list is chosen, not computed: quantile forest, NGBoost, Gaussian process
    # and VBLL, with VBLL on ChemBERTa alone. Every
    # model here was measured on QM9 against all six representations: how well its
    # predicted uncertainty tracks its own error, and how often the truth falls
    # inside the range it states. The three that were dropped -- BNN-Full,
    # MLP-BNN-Full, MLP-VBLL-Full -- track their error at between -0.10 and +0.19,
    # which is nothing, on every representation, and are overconfident everywhere.
    # Their jobs would have produced rows nobody could read.
    #
    # MEMORY IS NOT IN THIS TABLE ANY MORE. It comes from model_memory.json, which
    # the other two generators read as well, and it carries a FLOOR for this
    # pipeline specifically: the tree tier that the grid gets is deliberately not
    # applied here. The author, 2026-09-04: "I swear to fucking god its different
    # with uncertainty" -- and it is. A grid task fits a model once per level; a
    # task here fits it (1 + oof_folds) times per level and holds a per-molecule
    # uncertainty for every TRAINING molecule at every level. Nothing has measured
    # this pipeline, because it has not run once since the redesign.
    # THE WALL CLOCK IS NOT IN THIS TABLE ANY MORE EITHER, from 2026-09-07. It was
    # five hand-typed constants -- 36 for the quantile forest and 47 for the other
    # five -- and "deliberately generous" is not what they were. Nothing scaled
    # them: the same numbers came out for three conditions and for four, and they
    # did not move with --oof-folds, which is the flag that multiplies the work.
    # Priced against the laboratory generator's own per-fit table, three of the six
    # were TOO SMALL on logD, NGBoost by a factor of three. See wall_clock below.
    'QRF':            (1, 8, 'first on BOTH measures on all six representations, and the cheapest: tracks its error 0.25-0.35, truth inside 1 sd 0.70-0.83 against a target of 0.68'),
    'NGBoost':        (1, 8, 'second on four representations of six (0.09-0.30), mildly overconfident (0.53-0.66). Expensive -- 7.4x the forest on the screen -- and kept because it is the noise-robust model the study highlights'),
    'GP':             (1, 8, 'the only non-tree model that shows anything, and it depends on the representation: 0.28 on PDV, 0.19 on ChemBERTa, 0.06 on ECFP4. gauche ExactGP, RBF kernel'),
    'VBLL-Full':      (2, 8, 'the variational network. Badly overconfident -- truth inside 1 sd 0.27-0.51 against a target of 0.68 -- which is itself the finding. On ALL THREE representations from 2026-09-01: the ChemBERTa restriction is lifted'),
    # THE VARIANCE-HEAD NETWORKS, added 2026-09-01 on the author's decision.
    # Kendall & Gal eq. 6 -- one network predicts the value and its own
    # observation noise, with the weights sampled for the model term. The only
    # models on either pipeline whose ALEATORIC term varies per molecule while
    # the two halves come from different mechanisms, so the only ones that can be
    # asked the per-molecule decomposition question at all (RERUN_PLAN.md 2.32).
    # Wall clock from their plain Bayesian siblings, which is what QM9 derived
    # theirs from.
    'BNN-Full-MVE':     (2, 8, 'a Bayesian network with a VARIANCE HEAD -- the literature flagship case, and the only network whose aleatoric term varies per molecule'),
    'MLP-BNN-Full-MVE': (2, 8, 'the same variance head on the NN-beta base, so the finding does not rest on one architecture'),
    # THE SEVENTH MODEL, added 2026-09-07 on the author's decision.
    #
    # Of the six above, three report a data-noise value AND a model-uncertainty
    # value for each molecule separately: the quantile forest and the two
    # variance-head networks. NGBoost has no model-uncertainty term at all -- one
    # fit, nothing to disagree with (RERUN_PLAN.md 2.30). The plain Gaussian
    # process and the variational network each fit ONE observation-noise number
    # for the whole model, so their data-noise half is identical on every
    # molecule and its correlation with per-molecule injected noise is zero
    # however good the model is.
    #
    # This is the same exact Gaussian process with a small network predicting the
    # observation noise for each molecule, so its two halves come apart. It is
    # the only route to that in the Gaussian-process family, and it is the model
    # whose separation measured cleanest in the roster (5.5i). Without it the
    # per-molecule question is answered by a forest and two networks and by no
    # kernel model.
    #
    # It costs nothing already queued: 3 datasets x 3 representations x the
    # conditions of each submission, so 27 tasks in the first and 36 in the
    # second, added as one more array to each. Nothing is removed.
    #
    # The wall and the memory need no new numbers. `GP-Hetero` is already in the
    # laboratory generator's per-fit table at 2339.0 seconds per 1,000 molecules
    # -- twice the plain Gaussian process, for the noise network and its Adam
    # epochs -- and `gp_args` below gates it by the FAMILY, `model.startswith
    # ('GP-')`, so it inherits --gp-reps and the RBF kernel without a change.
    'GP-Hetero':      (1, 8, 'the same exact Gaussian process with a network predicting the observation noise per molecule, so its data-noise and model-uncertainty halves come apart. The cleanest separation measured in the roster (5.5i), and the only kernel model that can be asked the per-molecule question'),
}
# MEMORY COMES FROM model_memory.json, WITH THIS PIPELINE'S OWN FLOOR ON TOP.
#
# The grid generators put the trees on a 64G tier, on the evidence that every row
# belonging to this study in the author's sacct sweep is a neural network. That tier is
# deliberately NOT carried into this run: an uncertainty task fits its model
# (1 + oof_folds) times per level and holds a per-molecule uncertainty for every training
# molecule at every level, and nothing has measured it because it has not run once since
# the redesign. The author said so before anyone checked, 2026-09-04.
MODEL_MEMORY_FILE = Path(__file__).resolve().parent.parent / 'model_memory.json'
_MEMORY_SPEC = json.loads(MODEL_MEMORY_FILE.read_text())
_CANONICAL = json.loads(
    (Path(__file__).resolve().parent.parent / 'model_names.json').read_text())['validation']
_BY_CANONICAL = {m: tier for tier, spec in _MEMORY_SPEC['tiers'].items()
                 for m in spec['models']}
_PIPELINE_FLOOR = _MEMORY_SPEC.get('pipeline_overrides', {}).get(
    'uncertainty_runs', {}).get('floor')


def _gb(mem):
    return int(str(mem).upper().rstrip('G'))


def memory_for(model):
    """The memory request for one uncertainty-run model, never below this run's floor."""
    tier = _BY_CANONICAL.get(_CANONICAL.get(model, model), _MEMORY_SPEC['default'])
    if _PIPELINE_FLOOR and _gb(tier) < _gb(_PIPELINE_FLOOR):
        return _PIPELINE_FLOOR
    return tier


DATASETS = ['logd', 'caco2', 'herg_ki']
# THREE, by the author's decision of 2026-08-28 (RERUN_PLAN.md chat N): ECFP4,
# PDV and ChemBERTa. Informed by measurement rather than
# assumed. Sort & Slice is out because nothing distinguishes any model on it;
# Avalon is out because it behaves like ECFP4 for the tree models and is the worst
# of the six for the Bayesian networks; MHG-GNN is out on the author's call -- it
# was the most expensive representation to build, 10m43s against 2m24s for ECFP4
# on the screen's own timing, and the forest's ordering does not change without it.
# ChemBERTa is IN, and it was in neither list before: it is where the Bayesian
# networks show what little they have, and the tree models do as well on it as
# anywhere.
REPS = ['ECFP4', 'PDV', 'ChemBERTa']

# Representations for ONE model, where the full list would buy nothing. VBLL
# tracks its error at 0.25 on ChemBERTa and at 0.011 on ECFP4 and 0.152 on PDV,
# so two of its three representations would produce rows nobody can use.
MODEL_REPS = {
    # EMPTY from 2026-09-01. The variational network was restricted to ChemBERTa
    # on a roster screen that measured it tracking its own error at 0.25 there
    # against 0.01 to 0.15 elsewhere -- but that screen's neural numbers predate
    # the label-scale defect (RERUN_PLAN.md 2.31), and a restriction resting on
    # them is not safe. The author lifted it. Every model runs on every
    # representation in REPS.
}

# ---------------------------------------------------------------------------
# Noise conditions -- read, never restated
# ---------------------------------------------------------------------------
NOISE_CONDITIONS_FILE = Path(__file__).resolve().parent.parent / 'noise_conditions.json'
_SETTLED = json.loads(NOISE_CONDITIONS_FILE.read_text())

# The four the main grid runs, and the three that are depth-only. The JSON keys
# are the settled file's own; the names used here are the run design's.
MAIN_GRID_CONDITIONS = [c['name'] for c in _SETTLED['stage_1_full_grid']]
DEEP_RUN_CONDITIONS = [c['name'] for c in _SETTLED['stage_2_depth_only']]
KNOWN_CONDITIONS = MAIN_GRID_CONDITIONS + DEEP_RUN_CONDITIONS

# The one depth-only condition this run adds to the inherited four, because it
# is the only one of the three that can answer question B at all
# (RERUN_PLAN.md 13.1 item 2). Named rather than derived from FLAT_BY_DESIGN so
# that a new depth-only condition does not silently join the run.
ADDED_FOR_QUESTION_B = ['outlier_p10']

# Conditions that give EVERY molecule the same amount. Question B has nothing to
# find in them -- the correlation is undefined, not zero. Printed at generate
# time so the split is visible before the queue is spent, and cross-checked
# against the real injector by preflight section 4b.
FLAT_BY_DESIGN = {'gaussian', 'laplace', 'grouped_shifted',
                  'student_t_nu5', 'student_t_nu3', 'student_t_nu10'}

# Question A needs a condition whose noise is even across molecules; dropping
# every one of them leaves the run unable to answer it.
QUESTION_A_CONDITION = 'gaussian'

# ===========================================================================
# THE WALL CLOCK, COMPUTED. Added 2026-09-07.
# ===========================================================================
#
# WHAT WAS THERE BEFORE. Five constants typed into the MODELS table: 36 hours for
# the quantile forest and 47 for the other five. Nothing computed them and nothing
# scaled them. Generating the three-condition submission and the four-condition
# one emitted IDENTICAL walls, and --oof-folds -- the flag that multiplies the fit
# count -- did not move them at all.
#
# WHY THAT MATTERED. Priced against the laboratory generator's own per-fit table,
# three of the six were TOO SMALL on logD: NGBoost needs about three times the 47
# it asked, and the two variational and variance-head networks more than they
# asked as well. All 378 tasks of this run have sat in the queue since 2026-09-06
# without one starting, and the two arrays that are over-asked on hERG and Caco-2
# are the reason a scheduler cannot fit them.
#
# WHAT ONE TASK DOES. One dataset, one representation, one condition, sweeping
# that condition's levels. Each level is CV_FOLDS outer scaffold folds, and each
# outer fold that is cross-fitted refits the model --oof-folds times more on top:
#
#     fits = levels x (outer_folds + cross_fitted_outer_folds x oof_folds)
#
# At the settled seven levels, five outer folds and five out-of-fold folds that is
# 7 x (5 + 25) = 210 fits, against the 35 a laboratory grid task does.
#
# WHERE THE PER-FIT SECONDS COME FROM. The laboratory generator, imported rather
# than retyped -- same model names, same three datasets, same cubic branch for the
# exact Gaussian processes. A second copy of that table here is a copy that drifts.
#
# THE MARGIN IS THE LABORATORY GENERATOR'S OWN 1.5, imported, not a new number.
# It multiplies the same per-fit table, on the same three datasets, for the same
# models, and that table has been checked against real tasks: on 2026-09-07 every
# laboratory job except one had 4.9x headroom or more between its longest task and
# its request. What is different here is the fit COUNT, and the count is arithmetic
# -- levels, outer folds, out-of-fold folds -- not an estimate. Inventing a wider
# margin for a number that is counted rather than guessed would only add queue time
# nobody can use. Re-price the per-fit table from real tasks once these have run.
_LAB = Path(__file__).resolve().parent.parent / 'slurm_scripts_validation_rerun' / 'generate_scripts.py'
_spec = importlib.util.spec_from_file_location('_validation_generator', _LAB)
_lab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lab)

WALL_MARGIN = _lab.WALL_MARGIN
WALL_FLOOR_HOURS = _lab.WALL_FLOOR_HOURS

# The wall a `medium` partition can hold, so a run that cannot start there says so
# at generate time rather than at submit time. From the sinfo capture in
# RERUN_PLAN.md, "`short` had no idle nodes" (2026-08-28): short 12:00:00,
# medium 2 days, long 30 days.
MEDIUM_PARTITION_HOURS = 48

# The runner's dataset labels are the --datasets values; the laboratory tables are
# keyed by path label, and hERG is spelled differently in the two. One map, here,
# rather than a second TRAIN_N.
DATASET_PATH_LABEL = {cli: path for path, cli in _lab.DATASETS}


def wall_clock(model, dataset, oof_folds, oof_outer_folds, conditions):
    """Hours to request for one uncertainty array, on its worst dataset.

    ONE TASK RUNS ONE CONDITION -- the array index picks it -- so the wall is the
    WORST single condition, never the sum of them. Censoring sweeps a different
    number of levels from the dose ladder, which is why the condition list is
    passed in at all rather than a count.
    """
    outer = _lab.CV_FOLDS
    cross_fitted = min(oof_outer_folds or outer, outer)
    fits = max(_lab.levels_for(c) for c in conditions) * (
        outer + cross_fitted * oof_folds)
    seconds = _lab.seconds_per_fit(model, DATASET_PATH_LABEL[dataset])
    return max(WALL_FLOOR_HOURS,
               math.ceil(seconds * fits / 3600.0 * WALL_MARGIN))


# CORRECTED 2026-09-01, same as the accuracy generator. This said stat-cadd,
# the checkout KIRBy moved AWAY from when that filesystem hit its quota; 125
# of KIRBy's own 127 job scripts use stat-ecr. The guard below already names
# the other checkout in its error message -- it was pointing at the wrong one
# as the default and telling the operator the right one was the alternative.
KIRBY_DIR = '/data/stat-ecr/scat9264/KIRBy'
QSAR_DIR = '/data/stat-cadd/scat9264/qsar_qm_models'
RESULTS_ROOT = 'results/uncertainty_rerun'

TEMPLATE = '''#!/bin/bash
# ============================================================================
# Uncertainty re-run — model: {model}
# {note}
# ============================================================================
# Array task -> (dataset, representation, noise condition).
#   {n_ds} datasets x {n_rep} reps x {n_cond} conditions = {n_tasks} tasks
#
# Conditions: {condition_list}
# These are the settled set (noise_conditions.json), inherited from the main
# grid. Every one of them delivers the SAME amount of noise at a given level, so
# a difference between them is a difference of shape, not of dose.
#
# Levels are NOT passed: the runner anchors them per dataset to published assay
# error, and sweeps censoring on its own axis (the fraction of labels clipped).
#
# --account and --partition are LIVE STATE: pass them at submit time.
#
# THE PARTITION IS `long`, corrected 2026-09-07. This run asks {hours} hours, and
# `medium` stops at 48. Five of the six models in this pipeline are past that
# once the wall is computed from the fit count rather than typed: NGBoost 193,
# VBLL-Full 91, MLP-BNN-Full-MVE 78, GP 68, BNN-Full-MVE 51. Only the quantile
# forest at 40 would fit. The twelve arrays submitted on 2026-09-06 went to
# `medium` under the old hand-typed 47:59 (RERUN_PLAN.md 13.27 D1).
#
#   sbatch --account=<acct> --partition=long --array=0-{last}%{throttle} \\
#          {script_name}
#
# Every task writes its own --results-root, so tasks never race on a shared
# file. Run merge_results.py afterwards.
#
# Resubmit only the failed indices, e.g.:
#   sbatch --account=<acct> --partition=long --array=3,17,40 {script_name}
# ============================================================================
#SBATCH --job-name=unc_{jobslug}
#SBATCH --output=unc_{jobslug}_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={hours}:59:00
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=FAIL

set -uo pipefail

# ---------------------------------------------------------------------------
# GIVE THIS TASK ITS OWN KeOps CACHE AND ITS OWN SCRATCH.
#
# This replaced a random wait of up to ten minutes at the top of every task
# (removed 2026-09-01, the same change both sister generators got).
#
# The wait guarded a race over /tmp/compiler_version.txt, which KeOps writes at
# import. Checked in the library rather than assumed: both hard-coded /tmp paths
# in keopscore/config/base_config.py sit inside `if platform.system() ==
# "Darwin"`. On Linux neither line runs. What IS shared on Linux is the compile
# cache, $KEOPS_CACHE_FOLDER, whose default path contains the node name -- so the
# clash is between tasks on one node, which is what an array produces. Each task
# now compiles into its own scratch, removed on exit.
export TMPDIR="${{TMPDIR:-/tmp}}/unc_${{SLURM_JOB_ID:-$$}}_${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export KEOPS_CACHE_FOLDER="$TMPDIR/keops"
mkdir -p "$KEOPS_CACHE_FOLDER"

KIRBY_DIR="{kirby_dir}"

# Which KIRBy checkout this is, and whether it carries the redesigned runner.
#
# There are two checkouts on the cluster. 125 of the 127 job scripts in the
# KIRBy repository itself use /data/stat-ecr/scat9264/KIRBy -- the move was made
# on 2026-05-07 when stat-cadd hit 99.9% of its quota -- and two use stat-cadd,
# which is what this generator has always pointed at (RERUN_PLAN.md 2.8b). That
# cannot be settled from a laptop, so it is settled here, at the top of the job:
# a checkout without the redesigned command line is refused by name rather than
# producing 336 tasks' worth of results from the wrong code.
if [ ! -d "$KIRBY_DIR" ]; then
    echo "ERROR: no KIRBy checkout at $KIRBY_DIR."
    echo "       The other checkout is /data/stat-cadd/scat9264/KIRBy, which is"
    echo "       the one KIRBy moved away from. Regenerate with"
    echo "       --kirby-dir <path> rather than editing this file."
    exit 2
fi
RUNNER="$KIRBY_DIR/tests/alternative_data_noise_robustness.py"
if [ ! -f "$RUNNER" ]; then
    echo "ERROR: $RUNNER does not exist."; exit 2
fi
if ! grep -q -- "'--conditions'" "$RUNNER"; then
    echo "ERROR: $RUNNER has no --conditions flag, so this checkout predates the"
    echo "       noise redesign (noiseInject 1.0.0, 2026-08-26). Running it would"
    echo "       produce a full set of results from the old six strategies."
    echo "       Pull it, or point --kirby-dir at the other checkout."
    exit 2
fi
echo "=== KIRBy: $KIRBY_DIR  ($(git -C "$KIRBY_DIR" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"

cd "$KIRBY_DIR"
. {qsar_dir}/setup.sh

# Activation is not optional. micromamba has never worked on this cluster, so
# the `export MAMBA_EXE=...` lines that used to sit above `. setup.sh` pointed
# at a file that does not exist -- and nothing checked. setup.sh falls through
# to its conda branch; if that also fails, the task carries on in the system
# Anaconda at /apps/system/..., which has no gpytorch, no quantile_forest and
# no ngboost. The job then runs, finds nothing to do, and writes no rows. That
# is what happened to 12822693 and 12822694 (RERUN_PLAN.md section 2.8d).
if [ -z "${{CONDA_PREFIX:-}}" ]; then
    echo "ERROR: setup.sh did not activate an environment (CONDA_PREFIX unset)."
    exit 2
fi
PY_PATH="$(command -v python)"
case "$PY_PATH" in
    "$CONDA_PREFIX"/*) : ;;
    *)
        echo "ERROR: python is $PY_PATH, which is not inside the activated"
        echo "       environment ($CONDA_PREFIX)."
        case "$PY_PATH" in
            /apps/system/*)
                echo "       That is the system Anaconda. It has no gpytorch, no"
                echo "       quantile_forest and no ngboost, so this job would run,"
                echo "       find nothing to do, and write no rows." ;;
        esac
        exit 2 ;;
esac
# The test above cannot fail once setup.sh has run: setup.sh:124 prepends
# $CONDA_PREFIX/bin to PATH, so python resolves inside the prefix whichever
# environment was activated. It still catches an unset CONDA_PREFIX, which is
# the case that actually bit us -- but on its own it would pass a task that
# activated the WRONG environment, so the name is checked too. This literal
# must track ENV_NAME in setup.sh.
if [ "$(basename "$CONDA_PREFIX")" != "env_test" ]; then
    echo "ERROR: the active environment is $(basename "$CONDA_PREFIX"), not env_test"
    echo "       (CONDA_PREFIX=$CONDA_PREFIX). setup.sh activated the wrong one."
    exit 2
fi
echo "=== interpreter: $PY_PATH  (CONDA_PREFIX=$CONDA_PREFIX)"

# WHICH qsar_qm_models CHECKOUT THE SHARED SPEC COMES FROM. Say it, do not let
# it be guessed.
#
# `alternative_data_noise_robustness.py` loads three files from this repository
# rather than holding copies: models/model_defaults.py (every hyperparameter),
# scripts/uncertainty_decomposition.py (the aleatoric/epistemic split and its
# guard) and noise_conditions.json (the settled condition set). It finds them by
# trying $QSAR_QM_MODELS_ROOT first and then WALKING UP FROM THE KIRBy CHECKOUT
# to a sibling named qsar_qm_models.
#
# That walk is right only while both checkouts sit under the same parent. There
# are two KIRBy checkouts on this cluster -- stat-cadd and stat-ecr, and 125 of
# KIRBy's own 127 job scripts use stat-ecr (RERUN_PLAN.md 2.8b) -- so a task
# regenerated with --kirby-dir pointing at the other one would take its spec
# from a sibling of THAT directory while setup.sh and check_environment.py above
# used this one. Two copies of one specification, drifting, with nothing saying
# so: failure mode 10 of RERUN_PLAN.md 0.6.
#
# Setting it explicitly costs nothing and removes the walk. The runner prints
# each file it loaded and the spec hash; those lines are the receipt.
export QSAR_QM_MODELS_ROOT="{qsar_dir}"
if [ ! -f "$QSAR_QM_MODELS_ROOT/models/model_defaults.py" ]; then
    echo "ERROR: no shared spec at $QSAR_QM_MODELS_ROOT/models/model_defaults.py."
    echo "       Regenerate with --qsar-dir <path> rather than editing this file."
    exit 2
fi
echo "=== shared spec: $QSAR_QM_MODELS_ROOT ($(git -C "$QSAR_QM_MODELS_ROOT" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"

# The injector must be the redesigned one, and it must be the checkout that was
# pulled rather than a stale copy on the path. A task that runs the old injector
# writes results that look exactly like the new ones and are a different
# experiment.
python - <<'PYCHECK' || exit 2
import sys, inspect
try:
    import noiseInject
    from noiseInject import CONDITIONS
except Exception as exc:
    print(f"ERROR: noiseInject does not import: {{type(exc).__name__}}: {{exc}}")
    sys.exit(1)
print(f"=== noiseInject: {{inspect.getfile(noiseInject)}} "
      f"version {{getattr(noiseInject, '__version__', 'unknown')}}")
missing = [c for c in {condition_list_py} if c not in CONDITIONS]
if missing:
    print(f"ERROR: this noiseInject does not know {{missing}}. Known: {{sorted(CONDITIONS)}}.")
    print("       That is the pre-1.0.0 injector -- the six deleted strategies.")
    print("       pip install --no-deps -e <the NoiseInject checkout you pulled>")
    sys.exit(1)
PYCHECK

# A private scratch directory per task.
#
# Hygiene, not a fix for any known defect here: joblib, matplotlib, numba and
# the HuggingFace cache all honour TMPDIR, and a fixed-name temp file shared by
# concurrent array tasks is the defect class that produced the config.json race.
# This closes the whole class cheaply.
#
# It does NOT fix the one instance found while testing: keopscore (gpytorch ->
# pykeops) hardcodes /tmp/compiler_version.txt and /tmp/brew_prefix.txt, writing
# and deleting them during import, so two simultaneous imports race and one dies
# with FileNotFoundError. TMPDIR is ignored by that code. It is guarded by
# platform.system() == "Darwin", so it cannot fire on this cluster -- it is a
# macOS-only problem, and it is why the two-task half of
# scripts/test_config_isolation.py skips on a laptop.
export TMPDIR="${{TMPDIR:-/tmp}}/qsar_${{SLURM_JOB_ID:-$$}}_${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT


# Guarded: under `set -u` an unset CONDA_PREFIX aborts the shell here, before
# python is ever reached.
export LD_LIBRARY_PATH="${{CONDA_PREFIX:-}}/lib:${{LD_LIBRARY_PATH:-}}"

# Can this interpreter actually build the model this array runs?
#
# The activation guard above proves an environment is active and that it is
# named env_test; the check above that proves noiseInject is the redesigned
# one. Neither proves the MODEL's backend is installed. Jobs 12822693 and
# 12822694 ran to completion and wrote nothing because gpytorch was missing,
# and the runner is built to SKIP a model whose backend will not import (the
# HAS_* flags at alternative_data_noise_robustness.py:253-333) rather than to
# stop -- so a missing package is silent by design, and this array would burn
# {n_tasks} tasks finding nothing to do. Seconds once, before the task loop.
python "{qsar_dir}/scripts/check_environment.py" --validation-models "{model}" || {{
    echo "ERROR: this interpreter cannot build {model}. See above."
    exit 2
}}

cd tests

DATASETS=({datasets})
REPS=({reps})
CONDS=({conditions})

n_cond=${{#CONDS[@]}}
n_rep=${{#REPS[@]}}
n_tasks=$(( ${{#DATASETS[@]}} * n_rep * n_cond ))

# Guard: under `set -u` an unset SLURM_ARRAY_TASK_ID would abort with a cryptic
# message. This makes a non-array invocation run task 0 and say so.
i="${{SLURM_ARRAY_TASK_ID:-0}}"
if [ -z "${{SLURM_ARRAY_TASK_ID:-}}" ]; then
    echo "WARNING: not an array job — running task 0 only. Submit with --array=0-$(( n_tasks - 1 ))%N"
fi
if [ "$i" -ge "$n_tasks" ]; then
    echo "ERROR: task $i is out of range (0..$(( n_tasks - 1 )))"; exit 2
fi
cond="${{CONDS[$(( i % n_cond ))]}}"
rep="${{REPS[$(( (i / n_cond) % n_rep ))]}}"
ds="${{DATASETS[$(( i / (n_cond * n_rep) ))]}}"

# One directory per task — no cross-task write races. merge_results.py splits
# this name back apart on the DOUBLE underscore, so the condition's own single
# underscores are safe.
rep_slug=$(echo "$rep" | tr 'A-Z' 'a-z' | tr -d '-')
OUT="{results_root}/{model_slug}__${{ds}}__${{rep_slug}}__${{cond}}"

if [ -z "${{SLURM_JOB_PARTITION:-}}" ]; then
    echo "ERROR: no partition. Submit with --partition=long (see RUNBOOK step 4)."; exit 2
fi

echo "=== task $i: model={model} dataset=$ds rep=$rep condition=$cond"
echo "=== out: $OUT"
echo "=== started: $(date)"

{gp_line}python -u alternative_data_noise_robustness.py \\
    --datasets "$ds" \\
    --models "{model}" \\
    --reps "$rep" \\
    --conditions "$cond" \\
    --unc-conditions all \\
    --oof-folds {oof} \\
    {extra_args}{gp_args}--results-root "$OUT"

status=$?
echo "=== finished: $(date)  exit=$status"
exit $status
'''


def main():
    ap = argparse.ArgumentParser(
        description='Generate the uncertainty re-run job arrays.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Conditions come from noise_conditions.json; this file does not choose them.')
    ap.add_argument('--oof-folds', type=int, default=5,
                    help='Cross-fitting folds for the training-set uncertainty (default 5). '
                         'Cost is (1 + this) fits per noise level. The runner refuses 1: with '
                         'one fold the fit set and the scored set are the same molecules.')
    ap.add_argument('--oof-outer-folds', type=int, default=None,
                    help='Cross-fit only the first N of the 5 scaffold folds. All 5 folds are '
                         'trained regardless, so TEST-side uncertainty is free for all of them; '
                         'this only limits the expensive out-of-fold TRAINING pass. '
                         '1 cuts the added cost about threefold.')
    ap.add_argument('--conditions', nargs='+', default=None, choices=KNOWN_CONDITIONS,
                    help='Run exactly these noise conditions instead of all seven. Names come '
                         'from noise_conditions.json.')
    ap.add_argument('--include-deep-conditions', action='store_true',
                    help='Accepted and ignored. Every settled condition is the DEFAULT here '
                         'since 2026-08-28, so this flag no longer adds anything. Kept so a '
                         'command written before that date still runs and still means what it '
                         'said.')
    ap.add_argument('--drop-conditions', nargs='+', default=[], choices=KNOWN_CONDITIONS,
                    help='Conditions to leave out. gaussian is the one condition that spreads '
                         'the noise evenly across molecules, which is what makes it question '
                         "A's condition and the leakage check; grouped_wider and censoring are "
                         'the only two with a per-molecule pattern for question B. Dropping any '
                         'of the three removes a question, not a duplicate.')
    ap.add_argument('--reps', nargs='+', default=None,
                    help=f'Representations to run (default: {" ".join(REPS)}). Must be names the '
                         f"runner's --reps accepts.")
    ap.add_argument('--kirby-dir', default=KIRBY_DIR,
                    help=f'KIRBy checkout on the cluster (default {KIRBY_DIR}). There are two; '
                         f'the generated scripts refuse one that predates the noise redesign '
                         f'rather than running it (RERUN_PLAN.md 2.8b).')
    ap.add_argument('--qsar-dir', default=QSAR_DIR,
                    help=f'This repository on the cluster (default {QSAR_DIR}), the checkout '
                         f'whose setup.sh activates the environment. Changing it is how the '
                         f'smoke test runs a real generated script off the cluster.')
    ap.add_argument('--throttle', type=int, default=6,
                    help='Max concurrent tasks per array (the %%N in --array).')
    ap.add_argument('--out-dir', default=str(Path(__file__).parent))
    args = ap.parse_args()

    if args.conditions:
        conditions = list(dict.fromkeys(args.conditions))
        source = 'named on the command line'
    else:
        # THE SAME THREE THE QM9 SCREEN RUNS, and for the same reason.
        #
        # Superseding the all-seven default of 2026-08-28. Author's decision,
        # 2026-09-01: "this should apply to both qm9 and lab. Lab needs to follow
        # suit in waiting for the decision. These should be identical."
        #
        # QM9 answers the uncertainty question on three conditions in the screen
        # and on the other four afterwards, on a named subset of model-and-
        # representation pairs CHOSEN FROM THE SCREEN. Running all seven here
        # while QM9 runs three meant the two halves of the study answered the
        # same question on different noise, and could not be compared until the
        # QM9 follow-ups landed.
        #
        # The four that wait are censoring -- which is a pair-subset condition on
        # both sides, and sweeps a clipped fraction on its own axis rather than a
        # dose -- and the three depth-only ones. Ask for them by name with
        # --conditions, or for the depth three with --include-deep-conditions,
        # once the pairs are settled.
        # Censoring is named here, not derived from the settled file's
        # pair-subset marking. When censoring runs on THIS side it runs at full
        # breadth, on every settled pair -- the author's decision of 2026-08-27
        # (RERUN_PLAN.md 13.1 item 6), which scripts/test_noise_conditions.py
        # enforces by refusing to let this file read that marking at all. What is
        # deferred below is WHEN censoring runs, never how wide.
        conditions = [c for c in MAIN_GRID_CONDITIONS if c != 'censoring']
        source = (f'the three the QM9 screen runs ({NOISE_CONDITIONS_FILE.name}) '
                  f'— censoring and the three depth-only conditions follow on a '
                  f'named subset of pairs, exactly as they do on QM9')
    dropped = set(args.drop_conditions)
    conditions = [c for c in conditions if c not in dropped]
    if not conditions:
        raise SystemExit('every condition was dropped — there is nothing to run.')

    reps = list(args.reps) if args.reps else list(REPS)
    out = Path(args.out_dir)
    (out / 'logs').mkdir(parents=True, exist_ok=True)

    def reps_for(model):
        """The representations one model runs on.

        MODEL_REPS narrows a model that the screen measured as useful on some
        representations and useless on the others; --reps on the command line
        overrides everything, because a deliberate override should not be
        silently narrowed. A name in MODEL_REPS that is not in the run's own
        list is dropped, so narrowing can never ADD a representation.
        """
        if args.reps:
            return reps
        narrowed = MODEL_REPS.get(model)
        if not narrowed:
            return reps
        kept = [r for r in narrowed if r in reps]
        return kept or reps

    n_tasks = len(DATASETS) * len(reps) * len(conditions)

    print(f"Conditions: {source}")
    for c in conditions:
        role = ('no per-molecule pattern — question A and the leakage check'
                if c in FLAT_BY_DESIGN else
                'a per-molecule pattern — question B can be asked here')
        print(f"    {c:18s} {role}")
    if dropped:
        print(f"  Dropped: {', '.join(sorted(dropped))}")
    if QUESTION_A_CONDITION not in conditions:
        print(f"  WARNING: {QUESTION_A_CONDITION} is not in this run. It is the only condition "
              f"that spreads the noise evenly across molecules, so question A has no clean "
              f"reference and the leakage check cannot be made.")
    if not any(c not in FLAT_BY_DESIGN for c in conditions):
        print("  WARNING: every condition in this run gives every molecule the same amount of "
              "noise, so question B is undefined throughout — the run can only answer A.")

    # Optional flags, built as ONE string each carrying its own line
    # continuation, so that when nothing optional is set the line collapses
    # entirely. An empty placeholder on its own line leaves a dangling
    # backslash followed by a whitespace-only line, which bash reads as the end
    # of the command -- it then tries to run `--results-root` as a program.
    # `bash -n` does not catch this.
    extra_bits = []
    if args.oof_outer_folds:
        extra_bits.append(f'--oof-outer-folds {args.oof_outer_folds}')
    extra_args = ''.join(f'{b} \\\n    ' for b in extra_bits)

    written = []
    total_tasks = 0
    for model, (tier, cpus, note) in MODELS.items():
        mem = memory_for(model)
        # ONE WALL COVERS THE WHOLE ARRAY, so it is the worst dataset in it.
        # logD is nearly four times hERG's training set and the Gaussian
        # processes are cubic in that, so the three differ by far more than a
        # margin absorbs.
        hours = max(wall_clock(model, d, args.oof_folds,
                               args.oof_outer_folds, conditions)
                    for d in DATASETS)
        model_reps = reps_for(model)
        model_tasks = len(DATASETS) * len(model_reps) * len(conditions)
        total_tasks += model_tasks
        slug = model.lower().replace('-', '_')
        script_name = f'unc_{slug}.sh'
        # GP is gated to PDV only unless told otherwise, and the kernel has to
        # be RBF for it to be valid on every representation (Tanimoto is only
        # defined on binary fingerprints).
        #
        # The gate is the FAMILY, not the exact name. GP-Hetero is registered
        # inside the runner's `rname in gp_rep_set` block
        # (alternative_data_noise_robustness.py:2570-2594), so it inherits the
        # same PDV-only default. Testing `model == 'GP'` sent it out with no
        # --gp-reps at all: five of its six representations would have produced
        # nothing, silently, and it is the one model on this roster added
        # specifically to separate the two halves of an uncertainty.
        gp_args = ('--gp-reps "$rep" --gp-kernel rbf \\\n    '
                   if model == 'GP' or model.startswith('GP-') else '')
        body = TEMPLATE.format(
            model=model, note=note, jobslug=slug, model_slug=slug,
            cpus=cpus, mem=mem, hours=hours, oof=args.oof_folds,
            kirby_dir=args.kirby_dir, qsar_dir=args.qsar_dir, results_root=RESULTS_ROOT,
            datasets=' '.join(DATASETS),
            reps=' '.join(f'"{r}"' for r in model_reps),
            conditions=' '.join(conditions),
            condition_list=', '.join(conditions),
            condition_list_py=repr(conditions),
            n_ds=len(DATASETS), n_rep=len(model_reps), n_cond=len(conditions),
            n_tasks=model_tasks, last=model_tasks - 1, throttle=args.throttle,
            script_name=script_name, gp_args=gp_args, gp_line='',
            extra_args=extra_args)
        (out / script_name).write_text(body)
        (out / script_name).chmod(0o755)
        written.append((tier, script_name, model, hours, model_tasks, model_reps))

    # THE SUBMITTER, SO NOBODY EVER TYPES AN ARRAY RANGE AGAIN.
    #
    # The range is on the sbatch line, not in the script, and it CHANGES with the
    # condition set: three conditions is 27 tasks, the four that follow are 36, and
    # narrowing a model's representations changes its count alone. The runbook said
    # 0-62 against scripts that held 27 for long enough to be found by accident, and
    # 36 indices per array would have queued, started and exited on the out-of-range
    # guard. The generator knows each count; the operator should never have to.
    # The QM9 and laboratory generators both emit one of these.
    # THE DEFAULT PARTITION FOLLOWS THE WALLS THIS RUN ASKS FOR.
    #
    # It was `medium` whatever the walls said. The generator PRINTED a warning at
    # the end saying to submit with PART=long, and that warning is on the screen
    # of whoever regenerated -- not in the file the operator pastes. With any
    # array over medium's 48 hours, every one of those sbatch lines is refused at
    # submit time, and the run reads as "submitted 1 of 7".
    _over = [h for _, _, _, h, _, _ in written if h > MEDIUM_PARTITION_HOURS]
    _part = 'long' if _over else 'medium'
    _why = (f'# PARTITION: {len(_over)} of {len(written)} arrays ask more than '
            f"medium's {MEDIUM_PARTITION_HOURS}h, so the default here is `long`."
            if _over else
            f"# PARTITION: every array fits medium's {MEDIUM_PARTITION_HOURS}h.")
    submit = ['#!/bin/bash',
              f'# Submit the uncertainty runs: one array per model, each at its OWN range.',
              f'# {len(written)} arrays, {total_tasks} tasks, conditions:'
              f' {" ".join(conditions)}.',
              '# Written by generate_scripts.py -- regenerate rather than edit.',
              '# The scripts carry no partition and refuse to run without one.',
              _why,
              'ACCT=${ACCT:-stat-cadd}',
              'PART=${PART:-' + _part + '}',
              'ok=0; bad=0',
              'echo "submitting to account=$ACCT partition=$PART"',
              '']
    for _tier, _name, _model, _hours, _tasks, _mreps in written:
        submit.append(f'# {_name}: {_model}, {_tasks} tasks, --time={_hours}:59:00')
        submit.append(
            f'if sbatch --account=$ACCT --partition=$PART '
            f'--array=0-{_tasks - 1}%{args.throttle} {_name}; '
            f'then ok=$((ok+1)); else bad=$((bad+1)); fi')
    submit += ['',
               f'echo "submitted $ok of {len(written)}; $bad failed"',
               'if [ "$bad" -gt 0 ]; then exit 1; fi']
    (out / 'submit_all.sh').write_text('\n'.join(submit) + '\n')
    (out / 'submit_all.sh').chmod(0o755)

    print(f"\nWrote {len(written)} array scripts + submit_all.sh, "
          f"{total_tasks} tasks total, "
          f"oof-folds={args.oof_folds}, "
          f"oof-outer-folds={args.oof_outer_folds or 'all 5'}")
    print(f"  {len(DATASETS)} datasets x {len(conditions)} conditions, "
          f"representations per model:")
    for _, _, model, _, model_tasks, model_reps in written:
        note = '' if model_reps == reps else '   (narrowed by the author)'
        print(f"    {model:16s} {len(model_reps)} reps x {len(DATASETS)} x "
              f"{len(conditions)} = {model_tasks:3d} tasks   "
              f"{', '.join(model_reps)}{note}")
    for tier in (1, 2):
        print(f"\n  Tier {tier}:")
        for t, name, model, hours, model_tasks, _ in written:
            if t == tier:
                print(f"    {name:26s} {model:16s} --array=0-{model_tasks - 1} "
                      f"--time={hours}:59:00")

    # SAY WHICH PARTITION THESE CAN ACTUALLY START ON.
    #
    # The wall clocks used to be five constants, the largest 47:59, and 47:59 fits
    # `medium`. So the runbook sent this run to medium and nothing contradicted it.
    # Computed against the laboratory generator's per-fit table, every one of these
    # arrays needs more than medium's two days on logD, NGBoost by a factor of four
    # against what it was asking. A job whose wall exceeds the partition limit is
    # refused at submit time -- or, worse, silently capped and killed later.
    too_big = sorted({(m, h) for _, _, m, h, _, _ in written
                      if h > MEDIUM_PARTITION_HOURS})
    if too_big:
        print(f"\n  ⚠ {len(too_big)} of these cannot run on `medium` "
              f"({MEDIUM_PARTITION_HOURS}h): "
              + ', '.join(f'{m} {h}h' for m, h in too_big))
        print(f"    Submit this run with PART=long. One task is one dataset, one "
              f"representation and one condition,")
        print(f"    sweeping {max(_lab.levels_for(c) for c in conditions)} levels "
              f"x {_lab.CV_FOLDS} outer folds x "
              f"(1 + {args.oof_folds}) fits = "
              f"{max(_lab.levels_for(c) for c in conditions) * (_lab.CV_FOLDS + min(args.oof_outer_folds or _lab.CV_FOLDS, _lab.CV_FOLDS) * args.oof_folds)} fits, "
              f"against 35 for a grid task.")
        print(f"    --oof-outer-folds 1 cuts the out-of-fold half about threefold "
              f"if the queue will not take these.")


if __name__ == '__main__':
    main()
