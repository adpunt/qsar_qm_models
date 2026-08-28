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
import json
from pathlib import Path

# Models that emit a per-molecule uncertainty. Must match UNCERTAINTY_MODELS in
# KIRBy/tests/alternative_data_noise_robustness.py.
MODELS = {
    # name          : (tier, cpus, mem,  hours, note)
    # Memory matches the working reference (slurm_scripts_validation_rerun uses
    # 128G for these same models on these same datasets); this run additionally
    # holds the per-molecule uncertainty frames in memory, so do not go lower.
    # Wall times are deliberately generous: the out-of-fold pass multiplies the
    # fit count by (1 + oof_folds) and nothing here has been timed on ARC. They
    # were set against the old eleven-level ladder and are left alone now that
    # it is six levels -- a task that finishes early costs nothing, one that is
    # killed at the wall costs the whole task.
    'QRF':            (1, 8,  '128G', 36, 'quantile spread; strongest error-ranker in existing results'),
    'NGBoost':        (1, 8,  '128G', 47, '500 estimators; slowest of the tree models by far'),
    'GP':             (1, 8,  '128G', 47, 'gauche ExactGP, RBF kernel, uncapped (GP_MAX_N from the shared model spec)'),
    'BNN-Full':       (2, 8,  '128G', 47, '100 stochastic forward passes'),
    'VBLL-Full':      (2, 8,  '128G', 47, '100 stochastic forward passes'),
    'MLP-BNN-Full':   (2, 8,  '128G', 47, '100 stochastic forward passes'),
    'MLP-VBLL-Full':  (2, 8,  '128G', 47, '100 stochastic forward passes'),
}
DATASETS = ['logd', 'caco2', 'herg_ki']
# A subset of the runner's ALL_REPS. The four the uncertainty work has always
# used; the two the study added later (Avalon, ChemBERTa) are a 50% cost
# increase on this grid and belong to the same open question as which models and
# representations go deep (RERUN_PLAN.md 13.1 item 4). Pass --reps to change it.
REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']

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

KIRBY_DIR = '/data/stat-cadd/scat9264/KIRBy'
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
# --account and --partition are LIVE STATE: pass them at submit time. Confirm
# with  bash tests/slurm_scripts/where_to_submit.sh  in the KIRBy repo first.
#
#   sbatch --account=<acct> --partition=medium --array=0-{last}%{throttle} \\
#          {script_name}
#
# Every task writes its own --results-root, so tasks never race on a shared
# file. Run merge_results.py afterwards.
#
# Resubmit only the failed indices, e.g.:
#   sbatch --account=<acct> --partition=medium --array=3,17,40 {script_name}
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
    echo "       The other checkout is /data/stat-ecr/scat9264/KIRBy, which is what"
    echo "       125 of KIRBy's own 127 job scripts use. Regenerate with"
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
    echo "ERROR: no partition. Submit with --partition=medium (see RUNBOOK step 4)."; exit 2
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
        # ALL SEVEN, by the author's decision of 2026-08-28. See the header.
        conditions = MAIN_GRID_CONDITIONS + DEEP_RUN_CONDITIONS
        source = (f'every settled condition ({NOISE_CONDITIONS_FILE.name}) — '
                  f'the run answers whether uncertainty tracks some kinds of noise '
                  f'better than others, which needs all of them')
    dropped = set(args.drop_conditions)
    conditions = [c for c in conditions if c not in dropped]
    if not conditions:
        raise SystemExit('every condition was dropped — there is nothing to run.')

    reps = list(args.reps) if args.reps else list(REPS)
    out = Path(args.out_dir)
    (out / 'logs').mkdir(parents=True, exist_ok=True)
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
    for model, (tier, cpus, mem, hours, note) in MODELS.items():
        slug = model.lower().replace('-', '_')
        script_name = f'unc_{slug}.sh'
        # GP is gated to PDV only unless told otherwise, and the kernel has to
        # be RBF for it to be valid on every representation (Tanimoto is only
        # defined on binary fingerprints).
        gp_args = '--gp-reps "$rep" --gp-kernel rbf \\\n    ' if model == 'GP' else ''
        body = TEMPLATE.format(
            model=model, note=note, jobslug=slug, model_slug=slug,
            cpus=cpus, mem=mem, hours=hours, oof=args.oof_folds,
            kirby_dir=args.kirby_dir, qsar_dir=args.qsar_dir, results_root=RESULTS_ROOT,
            datasets=' '.join(DATASETS),
            reps=' '.join(f'"{r}"' for r in reps),
            conditions=' '.join(conditions),
            condition_list=', '.join(conditions),
            condition_list_py=repr(conditions),
            n_ds=len(DATASETS), n_rep=len(reps), n_cond=len(conditions),
            n_tasks=n_tasks, last=n_tasks - 1, throttle=args.throttle,
            script_name=script_name, gp_args=gp_args, gp_line='',
            extra_args=extra_args)
        (out / script_name).write_text(body)
        (out / script_name).chmod(0o755)
        written.append((tier, script_name, model, hours))

    print(f"\nWrote {len(written)} array scripts, {n_tasks} tasks each "
          f"({len(written) * n_tasks} tasks total), oof-folds={args.oof_folds}, "
          f"oof-outer-folds={args.oof_outer_folds or 'all 5'}")
    print(f"  {len(DATASETS)} datasets x {len(reps)} reps x {len(conditions)} conditions")
    for tier in (1, 2):
        print(f"\n  Tier {tier}:")
        for t, name, model, hours in written:
            if t == tier:
                print(f"    {name:26s} {model:16s} --time={hours}:00:00")


if __name__ == '__main__':
    main()
