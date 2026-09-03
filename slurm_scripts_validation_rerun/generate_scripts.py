#!/usr/bin/env python3
"""Generate individual SLURM scripts for validation re-run.

Each script runs ONE model x ONE rep x ONE dataset.
Writes to isolated dir: results/validation_rerun/{rep}_{dataset}/
This avoids race conditions from parallel jobs writing the same file.

After all jobs complete, run merge_results.py to combine into results/validation/.
"""
import argparse
import json
import math
import re
import os
from pathlib import Path

# THE SAME NINETEEN MODELS QM9 RUNS. Author's decision, 2026-09-01: "accuracy
# roster should match as well. this was an oversight."
#
# It was eight against QM9's nineteen, so the cross-model comparison existed on
# QM9 and did not exist on the three laboratory datasets -- every Bayesian and
# variational network, both variance-head networks, the Tanimoto and
# heteroscedastic Gaussian processes and the plain MLP were absent. The runner
# has had all of them for some time; only this list was short.
#
# The names are the laboratory spellings from model_names.json, which is the one
# place the correspondence between the two pipelines is written down. Every one
# of QM9's nineteen screen scripts maps to exactly one name here -- checked, no
# gaps -- so every row can be joined across the four datasets on the canonical
# name.
MODELS_ALL = [
    # tree and deterministic (QM9 tier 1)
    'RF', 'QRF', 'XGBoost', 'LightGBM', 'SVM', 'NGBoost', 'DNN', 'MLP',
    # Bayesian and variational networks (QM9 tier 2)
    'BNN-Full', 'MLP-BNN-Full', 'VBLL-Full', 'MLP-VBLL-Full',
    # the variance-head pair and the heteroscedastic variational pair
    'BNN-Full-MVE', 'MLP-BNN-Full-MVE',
    'VBLL-Full-Hetero', 'MLP-VBLL-Full-Hetero',
    # the three Gaussian processes
    'GP', 'GP-Tanimoto', 'GP-Hetero',
]
# All SIX of the study's representations, matching the QM9 grid (author, 2026-08-28:
# "it should run all 6 this is a mistake"). Avalon and ChemBERTa were absent while the
# rest of the study moved to six, so every representation claim on logD, Caco-2 and hERG
# would have rested on four of six while QM9 used all six -- and the representation
# half of the model-versus-representation split is the paper's spine. The runner has
# accepted all six since the storage fix; its --reps carries choices=ALL_REPS, so a name
# this file emits that the runner does not know stops the task at argument parsing.
ALL_REPS = ['ECFP4', 'SNS', 'MHG-GNN-pretrained', 'PDV', 'Avalon', 'ChemBERTa']

# CORRECTED 2026-09-01. This said /data/stat-cadd/scat9264/KIRBy, which is the
# checkout KIRBy moved AWAY from when stat-cadd hit its quota; 125 of KIRBy's own
# 127 job scripts use the stat-ecr path (RERUN_PLAN.md 0.4). The `cd` below had no
# guard and the script had no `set -e`, so all 129 jobs would have carried on past
# a failed `cd` and died later saying they could not open a python file -- or, if a
# stale second checkout happened to be there, run an OLD runner and write a full
# set of results from the retired noise scheme. The sister uncertainty generator
# was fixed for exactly this; its guard is copied into the preamble below.
KIRBY_DIR = '/data/stat-ecr/scat9264/KIRBy'
QSAR_DIR = '/data/stat-cadd/scat9264/qsar_qm_models'

# The hERG set is spelled two different ways and both spellings are load-bearing.
#
#   on the command line   'herg_ki'  -- alternative_data_noise_robustness.py's
#                                       --datasets carries choices=['logd',
#                                       'caco2', 'herg_ki', 'all'] (:2870), so
#                                       'herg' is rejected by argparse and the
#                                       task dies before it loads anything.
#   in every path         'herg'     -- the runner writes to
#                                       Path(results_root) / 'herg' (:3037), and
#                                       merge_results.py matches directories by
#                                       the '_{dataset}' suffix.
#
# Collapsing them to one name breaks one end or the other. Two of these scripts
# were hand-edited to 'herg_ki' at some point and the rest were left; that is
# the drift this table exists to stop.
DATASETS = [
    # (path label, --datasets value)
    ('logd', 'logd'),
    ('caco2', 'caco2'),
    ('herg', 'herg_ki'),
]

# ---------------------------------------------------------------------------
# Noise conditions -- read, never restated
# ---------------------------------------------------------------------------
# These jobs used to pass no --conditions at all, so they inherited the runner's
# own NOISE_CONDITIONS literal (alternative_data_noise_robustness.py:168). That
# literal contains outlier_p05, which noise_conditions.json lists under not_run:
# it was retired on 2026-08-27 in favour of outlier_p10. Every one of these 87
# scripts would have run a retired setting and skipped the settled one, silently,
# because a condition name is not something a result file makes you look at.
#
# The fix is the rule the rest of the project already follows: read the settled
# file, state the conditions on the command line, never restate them here. The
# runner's --conditions carries choices=sorted(CONDITIONS), so a name this file
# emits that the injector does not know stops the task at argument parsing
# instead of part way through.
# THE NOISE LADDER, READ FROM THE QM9 GENERATOR RATHER THAN RETYPED.
#
# NOISE_DESIGN.md section 6.4 is the one place the levels live and says so:
# "one shared ladder ... Applies to QM9, logD, Caco-2 and hERG". The QM9
# generator holds the operative copy as DOSE_LEVELS. This file used to hold none
# at all and passed none on the command line, so the laboratory ladder came from
# a default inside the other repository and nothing here could state it. It is
# lifted, not copied, so the two cannot drift (2026-09-01).
def _dose_levels():
    src = (Path(__file__).resolve().parent.parent
           / 'slurm_scripts_qm9_rerun' / 'generate_scripts.py').read_text()
    m = re.search(r"^DOSE_LEVELS\s*=\s*'([^']+)'", src, re.M)
    if not m:
        raise SystemExit(
            "ERROR: cannot find DOSE_LEVELS in the QM9 generator. The two "
            "pipelines share one ladder (NOISE_DESIGN.md 6.4) and this file "
            "reads it from there rather than holding a second copy.")
    return [float(x) for x in m.group(1).split()]


DOSE_LEVELS = _dose_levels()

NOISE_CONDITIONS_FILE = Path(__file__).resolve().parent.parent / 'noise_conditions.json'
_SETTLED = json.loads(NOISE_CONDITIONS_FILE.read_text())
FULL_GRID = [c['name'] for c in _SETTLED['stage_1_full_grid']]
DEPTH_ONLY = [c['name'] for c in _SETTLED['stage_2_depth_only']]
# Membership test for the deep-run guard below, kept beside the list it derives from.
DEEP_ONLY_SET = set(DEPTH_ONLY)
RETIRED = [c['name'] for c in _SETTLED['not_run']]

# Conditions that run on a NAMED SUBSET of model-and-representation pairs rather
# than the whole grid. Censoring is one, settled by the author 2026-08-27: the
# question it answers is how big the effect is, not which model resists it best,
# and it is the same ruling that already applies on QM9. Which pairs comes from
# the screen. This generator therefore refuses to put censoring in a script
# unless the models and representations are named, exactly as the QM9 generator
# does -- the failure to avoid is running it across the whole grid by accident.
PAIR_SUBSET = {c['name']: c['scope']
               for g in ('stage_1_full_grid', 'stage_2_depth_only')
               for c in _SETTLED[g]
               if c.get('scope', {}).get('mode') == 'pair_subset'
               and 'validation_robustness' in c['scope'].get('applies_to', [])}
BREADTH_GRID = [c for c in FULL_GRID if c not in PAIR_SUBSET]

# WALL CLOCKS, COMPUTED FROM MEASUREMENT, NOT TYPED.
#
# This used to be eight round numbers with no provenance -- RF 8h, NGBoost 24h,
# GP 47:59 and so on -- one per model, the same for all three datasets. Two of
# them were too small (found 2026-09-01, the same defect the QM9 generator had):
# NGBoost on logD needs 64 hours and asked for 24, and the quantile forest needs
# 13 and asked for 8. A job killed at the wall leaves a partial results file
# holding the low noise levels and nothing above them, which merges in looking
# like a finished condition sweep with the top of the ladder missing.
#
# SECONDS PER FIT PER 1,000 TRAINING MOLECULES, worst representation, default
# settings, from results/tuning_local/timing*.csv and trials*.csv -- the same
# sweeps that sized the QM9 generator, normalised by the sample size each was
# measured at. The tree and network models are close to linear in the number of
# molecules at a fixed round count; the Gaussian process is cubic and is treated
# as such below.
SECONDS_PER_FIT_PER_1K = {
    # Measured, from results/tuning_local.
    'RF': 37.7, 'QRF': 112.3, 'XGBoost': 31.6, 'LightGBM': 37.4,
    'SVM': 15.5, 'NGBoost': 544.4, 'DNN': 38.2, 'GP': 1169.5,
    # DERIVED for the eleven models added 2026-09-01, from the QM9 measurement of
    # the model each one extends, scaled by that model's own laboratory number.
    # QM9's hours-per-110-runs, measured: dnn 12, mlp 11, dnn_bnn_full 30,
    # mlp_bnn_full 46, dnn_bnn_full_variational 81, mlp_bnn_full_variational 79.
    # DNN is 38.2 here against 12 there, a factor of 3.2, applied to the rest.
    'MLP': 35.0,                    # mlp is 11/12 of dnn on QM9
    'BNN-Full': 95.5,               # 30/12 x 38.2
    'MLP-BNN-Full': 146.4,          # 46/12 x 38.2
    'VBLL-Full': 257.9,             # 81/12 x 38.2
    'MLP-VBLL-Full': 251.5,         # 79/12 x 38.2
    # The variance-head and heteroscedastic variants: 1.5x their plain sibling,
    # the same margin the QM9 generator carries for them.
    'BNN-Full-MVE': 143.3,
    'MLP-BNN-Full-MVE': 219.6,
    'VBLL-Full-Hetero': 386.9,
    'MLP-VBLL-Full-Hetero': 377.3,
    # The other two Gaussian processes: the same exact GP at the same capped
    # size, so the same per-fit cost as 'GP'. The heteroscedastic one carries a
    # noise network and 100 Adam epochs on top, hence twice.
    'GP-Tanimoto': 1169.5,
    'GP-Hetero': 2339.0,
}

# Training molecules per dataset (RERUN_PLAN.md 13.14: logD 4,031, Caco-2 1,729,
# hERG 1,132). The three differ almost fourfold, so one wall clock per MODEL was
# always wrong for two of them.
TRAIN_N = {'logd': 4031, 'caco2': 1729, 'herg': 1132}

# Fits per script: conditions x levels x folds. The folds are the runner's
# GroupKFold(n_splits=5) over scaffold groups.
CV_FOLDS = 5

# Margin over the computed need. The QM9 generator uses 1.25; this side uses 1.5
# because its per-fit numbers are normalised across sample sizes rather than
# measured at the size that will run, so they carry more uncertainty.
WALL_MARGIN = 1.5


BS = chr(92)   # one backslash: the shell's line continuation


def gp_flags_for(model, rep):
    """The two flags the Gaussian process needs, and nothing for other models.

    The runner runs its Gaussian process on PDV alone unless --gp-reps says
    otherwise, so a GP job for any other representation builds the features,
    finds no experiment to run, and stops with "No experiments to run". Naming
    the representation makes the script's filename and what it actually runs the
    same thing.
    """
    if model != 'GP':
        return ''
    return ("    --gp-kernel rbf " + BS + "\n"
            "    --gp-reps " + rep + " " + BS + "\n")


def wall_clock(model, dataset, n_conditions, n_levels):
    """Hours to request for one laboratory job, from the measurements above."""
    n = TRAIN_N[dataset]
    per_fit = SECONDS_PER_FIT_PER_1K[model]
    if model.startswith('GP'):
        # startswith, not == 'GP'. With eight models this only had to catch one
        # name; at nineteen it must also catch GP-Tanimoto and GP-Hetero, which
        # are the same exact process with a different kernel and a noise network.
        # Scaling those linearly asked 207 and 413 hours against a true 34 and 68
        # (found 2026-09-01, the day the roster grew).
        # An exact Gaussian process factorises an n x n matrix, so it is cubic in
        # the training set rather than linear. The basis above was measured at
        # 10,000 molecules, hence the /10 twice.
        seconds = per_fit * (n / 1000.0) ** 3 / 100.0
    else:
        seconds = per_fit * n / 1000.0
    hours = seconds * n_conditions * n_levels * CV_FOLDS / 3600.0
    return max(1, math.ceil(hours * WALL_MARGIN))

PREAMBLE = """# GIVE THIS JOB ITS OWN KeOps CACHE AND ITS OWN SCRATCH.
#
# This replaced a random wait of up to ten minutes at the top of every job
# (removed 2026-09-01, the same change the QM9 generator got the day before).
#
# WHAT THE WAIT WAS FOR. KeOps arrives with the Gaussian-process stack, and the
# runner imports gpytorch at module scope, so every job here pulls it in --
# a random-forest job on logD too. On MACOS it runs `c++ --version` at import,
# writes the answer to the hard-coded /tmp/compiler_version.txt, reads it back
# and deletes it; two processes importing together race over that file.
#
# WHY IT CANNOT HAPPEN HERE, checked in the library rather than assumed. Both
# hard-coded /tmp paths in keopscore/config/base_config.py sit inside
# `if platform.system() == "Darwin"`. On Linux neither line runs. The wait was
# guarding a macOS bug on a Linux cluster and cost five minutes a job on average.
#
# WHAT IS ACTUALLY SHARED ON LINUX. KeOps compiles into $KEOPS_CACHE_FOLDER,
# which defaults to a path containing the NODE name -- so the clash is between
# jobs on one node, and submit_all.sh fires 129 at once with no throttle. Each
# job now compiles into its own scratch, removed on exit. Nothing shared,
# nothing waits.
export TMPDIR="${{TMPDIR:-/tmp}}/val_${{SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
export KEOPS_CACHE_FOLDER="$TMPDIR/keops"
mkdir -p "$KEOPS_CACHE_FOLDER"

KIRBY_DIR="{kirby_dir}"
QSAR_DIR="{qsar_dir}"

# THE CHECKOUT MUST BE THERE, AND IT MUST BE CURRENT.
#
# Copied from slurm_scripts_uncertainty_rerun/generate_scripts.py, where it has
# been proven. Without it a wrong path is silent: the cd fails, the environment
# guards below still pass because they use the absolute $QSAR_DIR, and the job
# dies much later saying it cannot open a python file -- after the queue wait.
if [ ! -d "$KIRBY_DIR" ]; then
    echo "ERROR: no KIRBy checkout at $KIRBY_DIR."
    echo "       The other checkout is /data/stat-cadd/scat9264/KIRBy, which is"
    echo "       the one KIRBy moved away from. Regenerate with --kirby-dir <path>"
    echo "       rather than editing this file."
    exit 2
fi
RUNNER="$KIRBY_DIR/tests/alternative_data_noise_robustness.py"
if [ ! -f "$RUNNER" ]; then
    echo "ERROR: $RUNNER does not exist."; exit 2
fi
if ! grep -q -- "'--conditions'" "$RUNNER"; then
    echo "ERROR: $RUNNER has no --conditions flag, so this checkout predates the"
    echo "       noise redesign (noiseInject 1.0.0, 2026-08-26). Running it would"
    echo "       produce a full set of results from the retired six strategies."
    echo "       Pull it, or point --kirby-dir at the other checkout."
    exit 2
fi
echo "=== KIRBy: $KIRBY_DIR  ($(git -C "$KIRBY_DIR" log --oneline -1 2>/dev/null || echo 'not a git checkout'))"

cd "$KIRBY_DIR" || {{ echo "ERROR: cannot enter $KIRBY_DIR"; exit 2; }}
. "$QSAR_DIR/setup.sh"

# setup.sh refuses to build the environment or install the extras from inside an
# ARRAY task -- but these 129 are separate jobs, not an array, so neither refusal
# can see them. SLURM_ARRAY_TASK_ID is exported here for the same reason: a
# hundred concurrent jobs rebuilding one shared environment is the failure those
# refusals exist to prevent, and it does not care whether the jobs came from one
# array or from a submit script (found 2026-09-01).

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

# THE NOISE LADDER MUST BE THE SETTLED ONE, AND THIS SIDE NEVER STATES IT.
#
# The command line below passes --conditions and NOT --sigmas, deliberately:
# censoring is a clipped fraction on its own axis, and overriding the ladder
# would dose it on the wrong one. So the seven levels come from a default inside
# the runner, in the OTHER repository, and nothing here could say what they were.
# The last completed laboratory results carry ELEVEN levels, 0.0 to 1.0 by 0.1 --
# the retired ladder -- so this is not hypothetical.
#
# NOISE_DESIGN.md section 6.4 is the one place the levels live. This asks the
# runner what it would use and refuses if it disagrees, rather than discovering
# it in the results (found 2026-09-01).
# The heredoc delimiter is QUOTED so the regex backslashes below survive the
# shell -- which also means $KIRBY_DIR is NOT expanded inside it. The path is
# passed as an argument instead. Getting that wrong made every one of these jobs
# exit 2 on a file called literally '$KIRBY_DIR/tests/...' (found and fixed the
# same day, 2026-09-01).
python - "$KIRBY_DIR" <<'PYLEVELS' || exit 2
import json, re, sys, pathlib
want = {levels_json}
src = pathlib.Path(sys.argv[1], 'tests',
                   'alternative_data_noise_robustness.py').read_text()
m = re.search(r"^\s*(?:DEFAULT_)?(?:SIGMAS|NOISE_LEVELS|LEVELS)\s*=\s*(\[[^\]]*\])",
              src, re.M)
if not m:
    print("ERROR: cannot find the runner's default noise ladder. It is passed")
    print("       nowhere on the command line, so it cannot be confirmed. Look for")
    print("       the sigma list in alternative_data_noise_robustness.py by hand")
    print("       and compare it with NOISE_DESIGN.md section 6.4:")
    print("       " + " ".join(str(x) for x in want))
    sys.exit(2)
got = [float(x) for x in re.findall(r"-?\d*\.?\d+", m.group(1))]
if got != [float(x) for x in want]:
    print("ERROR: the runner's noise ladder is not the settled one.")
    print("       runner:  " + " ".join(str(x) for x in got))
    print("       settled: " + " ".join(str(x) for x in want))
    print("       Pull KIRBy, or the whole degradation curve is on a retired axis.")
    sys.exit(2)
print("=== noise ladder: " + " ".join(str(x) for x in got) + "  (matches NOISE_DESIGN 6.4)")
PYLEVELS

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


export LD_LIBRARY_PATH="${{CONDA_PREFIX:-}}/lib:${{LD_LIBRARY_PATH:-}}"

# Can this interpreter actually build the model this task runs?
#
# The activation guard above proves an environment is active and that it is
# named env_test. It does not prove the packages are in it. Jobs 12822693 and
# 12822694 ran to completion and wrote nothing because gpytorch was missing,
# and the KIRBy runner is built to SKIP a model whose backend will not import
# (the HAS_* flags at alternative_data_noise_robustness.py:253-333) rather than
# to stop -- so a missing package here is silent by design. This is seconds per
# task and turns that into an exit 2 before any data is loaded.
python "$QSAR_DIR/scripts/check_environment.py" --validation-models {model} || {{
    echo "ERROR: this interpreter cannot build {model}. See above."
    exit 2
}}

# The injector must be the redesigned one.
#
# The runner does `from noiseInject import CONDITIONS` at module scope, so a
# stale checkout does not fail -- it runs the pre-1.0.0 scheme, where the six
# strategies were one strategy at six doses and a level meant something else
# entirely. The results look exactly like the new ones and are a different
# experiment. The uncertainty jobs have carried this check since 2026-08-27;
# these did not, and they use the same runner and the same injector.
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
"""

# The header and the body are formatted SEPARATELY and concatenated, never
# formatted together. PREAMBLE contains shell brace-expansions written as `{{`
# for .format(); once it has been formatted those are real braces, and running
# .format() over them a second time -- which is what putting {preamble} inside
# one big template would do -- fails with "Single '{' encountered".
SLURM_HEADER = """#!/bin/bash
#SBATCH --job-name=val_{safe_name}
#SBATCH --account=stat-cadd
#SBATCH --output=val_{safe_name}_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

# `set -u` catches an unset variable rather than expanding it to nothing, and
# `pipefail` makes a failing command in a pipeline fail the pipeline. NOT
# `set -e`: sourcing setup.sh legitimately returns non-zero on some paths and
# would kill the job before it started. The exit status is carried by hand at
# the end of the body instead, which is what the QM9 side does.
set -uo pipefail

"""

SLURM_BODY = """
# ONE ARRAY PER MODEL, NOT ONE JOB PER COMBINATION.
#
# This directory used to emit one script per (model, representation, dataset) --
# 144 separate sbatch submissions, against QM9's 19 array scripts for 327 tasks.
# The author's call, 2026-09-01: the same four families should each be a handful
# of arrays. 8 models x 18 tasks = the same 144 units of work, submitted as 8.
#
# The index picks the pair, the same arithmetic the QM9 template uses.
CONDS_UNUSED=""
REPS=({reps_list})
REPS_SAFE=({reps_safe_list})
DATASETS=({datasets_list})
n_rep=${{#REPS[@]}}
i="${{SLURM_ARRAY_TASK_ID:-0}}"
n_task=$(( n_rep * ${{#DATASETS[@]}} ))
if [ "$i" -ge "$n_task" ]; then
    echo "ERROR: array index $i is past the end; this script has $n_task tasks (0-$(( n_task - 1 )))."
    exit 2
fi
rep="${{REPS[$(( i % n_rep ))]}}"
rep_safe="${{REPS_SAFE[$(( i % n_rep ))]}}"
dataset="${{DATASETS[$(( i / n_rep ))]}}"

# The runner spells hERG 'herg_ki' on the command line and 'herg' in its output
# directory; both spellings are load-bearing and neither can be derived from the
# other, so the pair is carried explicitly.
case "$dataset" in
{dataset_cases}
  *) echo "ERROR: unknown dataset '$dataset'"; exit 2 ;;
esac

# THE DATASET'S CACHE MUST BE ON DISK BEFORE THE MODEL IS IMPORTED.
#
# `tests/data_cache/*.csv` is gitignored in KIRBy, so a fresh clone or a checkout
# on a different filesystem has none of it. The two OpenADMET endpoints survive
# that -- `download_openadmet` refetches them with no guard -- but hERG does NOT:
# `fetch_chembl_herg_ki` REFUSES to fetch without KIRBY_ALLOW_CHEMBL_FETCH=1,
# deliberately, because ChEMBL grows and today's release is not the dataset any
# existing result came from.
#
# That refusal cost the whole hERG third of the 2026-09-02 launch: every task at
# index 12-17 of every array that got that far died with exit 1 in under two
# minutes, and nothing said why until a log was read (RERUN_PLAN.md 13.18). It is
# the last third of the array, so it fails AFTER logd and caco2 have succeeded
# and the run looks healthy.
#
# Checked here rather than left to the runner, because the runner raises after
# importing a torch backend -- one to two minutes per task, times eighteen tasks,
# times nineteen arrays.
CACHE_DIR="$KIRBY_DIR/tests/data_cache"
case "$dataset" in
  herg)
    if [ ! -s "$CACHE_DIR/chembl_herg_ki.csv" ]; then
        echo "ERROR: $CACHE_DIR/chembl_herg_ki.csv is missing or empty, and the"
        echo "       runner will not fetch hERG live -- ChEMBL today is a different"
        echo "       dataset from the one every existing result was produced from."
        echo "       COPY the cached file in; do not set KIRBY_ALLOW_CHEMBL_FETCH."
        echo "       The other checkout is the likely source:"
        echo "         cp /data/stat-cadd/scat9264/KIRBy/tests/data_cache/chembl_herg_ki.csv $CACHE_DIR/"
        echo "       Bring chembl_herg_ki.provenance.json with it if it exists, or"
        echo "       which ChEMBL release the labels came from goes unrecorded."
        exit 2
    fi
    _n_herg=$(wc -l < "$CACHE_DIR/chembl_herg_ki.csv")
    echo "=== hERG cache: $CACHE_DIR/chembl_herg_ki.csv, $_n_herg lines"
    ;;
  *)
    # Not fatal: the runner refetches these. But eighteen tasks fetching one file
    # at once is the pattern the QM9 runbook warms caches to avoid.
    [ -s "$CACHE_DIR/openadmet_train.csv" ] || \
        echo "WARNING: no $CACHE_DIR/openadmet_train.csv -- this task will download it."
    ;;
esac

# The Gaussian process runs on PDV alone unless --gp-reps says otherwise, so a GP
# task for any other representation would build its features, find no experiment
# to run, and stop. The kernel is stated rather than inherited: --gp-kernel
# defaults to rbf, but Tanimoto is defined on binary vectors only and four of the
# six representations are not binary.
GP_FLAGS=""
case "{model}" in
  GP|GP-Hetero)                   GP_FLAGS="--gp-kernel rbf --gp-reps $rep" ;;
  GP-Tanimoto|GP-Tanimoto-Hetero) GP_FLAGS="--gp-kernel tanimoto --gp-reps $rep" ;;
esac

# THE MODEL AND THE REPRESENTATION ARE BOTH IN THE OUTPUT PATH. Until 2026-09-01
# the model was not, so seven or eight scripts sharing a representation and a
# dataset wrote into ONE directory. The runner appends with no lock, so rows were
# lost or torn and the merge could not tell.
OUT_ROOT="results/validation_rerun/{model_lower}_${{rep_safe}}_${{dataset}}"

echo "=== task $i: model={model} rep=$rep dataset=$dataset"
echo "=== out: $OUT_ROOT"
echo "=== started: $(date)"

cd tests || {{ echo "ERROR: no tests/ under $KIRBY_DIR"; exit 2; }}
[ -f alternative_data_noise_robustness.py ] || {{
    echo "ERROR: the runner is not in $KIRBY_DIR/tests"; exit 2; }}

python -u alternative_data_noise_robustness.py \
    --datasets "$dataset_cli" \
    --models {model} \
    --reps "$rep" \
    --conditions {conditions} \
    $GP_FLAGS \
    --results-root "../$OUT_ROOT"
status=$?

# CARRY THE EXIT STATUS. The last statement used to be an echo, which always
# returns 0, so a job whose python run died was recorded by sacct as COMPLETED
# with exit 0 and mailed an END rather than a FAIL. That is the shape of a run
# that looks finished and has no results.
echo "=== finished: $(date)  exit=$status"
exit $status
"""

# PREFLIGHT. Costs nothing, submits nothing, runs on a login node.
#
# Written 2026-09-03, after all 26 failures in the first laboratory launch turned
# out to be one missing file: tests/data_cache/*.csv is gitignored in KIRBy, and
# hERG is the one dataset the runner REFUSES to refetch (RERUN_PLAN.md 13.18).
# The in-script guard added the same day stops a task early; this answers the
# question before a task is submitted at all, which is what the author actually
# needed and did not have.
PREFLIGHT_BODY = """#!/bin/bash
# Will an hERG task run? Answers it without submitting anything.
#
#   bash preflight.sh [path-to-KIRBy]
#
# Exit 0 means an hERG task will get past the point where all 26 tasks failed on
# 2026-09-02. It does NOT promise the fit succeeds -- only that the data loads.
set -u
KIRBY_DIR="${{1:-{kirby_dir}}}"
CACHE="$KIRBY_DIR/tests/data_cache"
fail=0
warn=0

echo "=== KIRBy:  $KIRBY_DIR"
echo "=== cache:  $CACHE"
echo

if [ ! -d "$KIRBY_DIR" ]; then
    echo "FAIL  no KIRBy checkout at $KIRBY_DIR"
    echo "      Pass the right path as the first argument."
    exit 1
fi

# 1. THE FILE THAT WAS MISSING.
if [ ! -s "$CACHE/chembl_herg_ki.csv" ]; then
    echo "FAIL  $CACHE/chembl_herg_ki.csv is missing or empty."
    echo "      This is what killed every hERG task. The runner will not fetch it:"
    echo "      ChEMBL today is a different dataset from the one this study is built"
    echo "      on, so fetching would silently change the labels."
    echo
    echo "      Copy it from the other checkout:"
    echo "        cp /data/stat-cadd/scat9264/KIRBy/tests/data_cache/chembl_herg_ki.csv $CACHE/"
    echo "      Do NOT set KIRBY_ALLOW_CHEMBL_FETCH."
    fail=1
else
    n_lines=$(wc -l < "$CACHE/chembl_herg_ki.csv")
    echo "ok    chembl_herg_ki.csv present, $n_lines lines"
    if [ "$n_lines" -ne 1416 ]; then
        echo "WARN  expected 1416 lines -- 1,415 molecules plus a header, which is the"
        echo "      hERG count the reporting level in RERUN_PLAN.md 13.16 rests on."
        echo "      A different count means a different extraction; check before running."
        warn=1
    fi
fi

# 2. WHICH ChEMBL RELEASE. A warning, never a failure -- an unstamped cache still
#    trains, it just cannot be cited.
if [ -s "$CACHE/chembl_herg_ki.provenance.json" ]; then
    echo "ok    provenance stamp present"
else
    echo "WARN  no chembl_herg_ki.provenance.json beside it, so which ChEMBL release"
    echo "      produced these labels is unrecorded. Copy it across if it exists."
    warn=1
fi

# 3. THE OTHER TWO DATASETS. The runner refetches these with no guard, so a miss
#    is slow rather than fatal -- but eighteen tasks fetching one file at once is
#    the pattern the QM9 runbook warms caches to avoid.
if [ -s "$CACHE/openadmet_train.csv" ]; then
    echo "ok    openadmet_train.csv present (logD and Caco-2)"
else
    echo "WARN  no openadmet_train.csv -- every logD and Caco-2 task will download it."
    warn=1
fi

echo
if [ "$fail" -ne 0 ]; then
    echo "PREFLIGHT FAILED -- fix the above and run this again. Submit nothing yet."
    exit 1
fi

# 4. THE ACTUAL PROOF: ask the runner's own loader for the data. Everything above
#    is about a file on disk; this is the thing a task really does. Import failure
#    is reported as its own outcome, because that is an environment problem and
#    not the missing-cache problem this script is about.
echo "=== loading hERG through the runner's own loader (a minute or so) ..."
python - "$KIRBY_DIR" <<'PYLOAD'
import sys, pathlib, importlib.util, traceback
root = pathlib.Path(sys.argv[1], 'tests')
sys.path.insert(0, str(root))
try:
    spec = importlib.util.spec_from_file_location(
        'kirby_runner', root / 'alternative_data_noise_robustness.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except Exception:
    traceback.print_exc()
    print()
    print("INCONCLUSIVE: the runner would not import. That is an environment")
    print("              problem, not the missing hERG cache. Check that env_test")
    print("              is activated -- '. setup.sh' in an allocation.")
    sys.exit(2)
try:
    smiles, labels = mod.load_chembl_herg()
except Exception:
    traceback.print_exc()
    print()
    print("FAIL: the loader raised. If the message mentions KIRBY_ALLOW_CHEMBL_FETCH")
    print("      the cache is still not where the runner looks for it.")
    sys.exit(1)
n = len(labels)
print(f"ok    hERG loaded: {{n}} molecules")
if n != 1415:
    print(f"WARN  expected 1415 -- the count every existing hERG result and the")
    print(f"      reporting level rest on. Got {{n}}. Do not resubmit until you know why.")
    sys.exit(3)
PYLOAD
rc=$?

echo
case "$rc" in
  0) if [ "$warn" -ne 0 ]; then
         echo "PREFLIGHT PASSED, with warnings above. An hERG task will now run."
     else
         echo "PREFLIGHT PASSED. An hERG task will now run."
     fi
     echo "Next: ONE task, not the grid --  sbatch --array=12 val_lightgbm.sh"
     ;;
  2) echo "PREFLIGHT INCONCLUSIVE -- the environment, not the cache. See above." ;;
  3) echo "PREFLIGHT PASSED THE FILE CHECKS but the molecule count is wrong. See above." ;;
  *) echo "PREFLIGHT FAILED -- see above. Submit nothing." ;;
esac
exit $rc
"""


# The smoke test: two models writing into ONE rep_dataset directory, checking
# that the second does not overwrite the first. Generated here rather than kept
# by hand -- it was the last script in this directory still carrying the dead
# micromamba hook, because hand-written files do not get regenerated.
SMOKE_BODY = """
cd tests

TESTDIR=results/validation_smoke_test/ecfp4_herg

echo "=== STEP 1: RF x ECFP4 x herg (level 0.0 only) ==="
python alternative_data_noise_robustness.py \\
    --datasets {dataset_cli} \\
    --models RF \\
    --reps ECFP4 \\
    --sigmas 0.0 \\
    --results-root $TESTDIR

if [ ! -f $TESTDIR/{dataset}/all_results.csv ]; then
    echo "FAIL: no output"
    exit 1
fi
echo "Rows: $(wc -l < $TESTDIR/{dataset}/all_results.csv)"

echo ""
echo "=== STEP 2: SVM x ECFP4 x herg (level 0.0, should merge with RF) ==="
python alternative_data_noise_robustness.py \\
    --datasets {dataset_cli} \\
    --models SVM \\
    --reps ECFP4 \\
    --sigmas 0.0 \\
    --results-root $TESTDIR

echo "Rows: $(wc -l < $TESTDIR/{dataset}/all_results.csv)"

# The model column by NAME, not by position. `cut -d, -f6` was reading whatever
# column happened to be sixth, and it silently reads the wrong one the moment a
# column is added.
MODELS=$(python -c "import pandas,sys; \\
print(' '.join(sorted(pandas.read_csv(sys.argv[1])['model'].unique())))" \\
    $TESTDIR/{dataset}/all_results.csv)
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
"""


def safe_name(rep):
    return rep.replace('-', '_').lower()


def main():
    ap = argparse.ArgumentParser(
        description='Generate the validation re-run job scripts.',
        epilog='Conditions come from noise_conditions.json; this file does not choose them.')
    ap.add_argument('--out-dir', default=None,
                    help='Where to write the scripts (default: this directory). A test needs '
                         'this: without it the only way to see what the generator emits is to '
                         'run it, which overwrites the committed scripts. Probing it that way '
                         'is what silently rewrote all 87 of them on 2026-08-27.')
    ap.add_argument('--models', nargs='+', default=None, metavar='NAME',
                    help=f'Only these models (default: all {len(MODELS_ALL)}). Required, with '
                         f'--reps, to run a pair-subset condition.')
    ap.add_argument('--reps', nargs='+', default=None, metavar='NAME',
                    help=f'Only these representations (default: all {len(ALL_REPS)}).')
    ap.add_argument('--conditions', nargs='+', default=None, metavar='NAME',
                    help=f'Override the conditions. Default: {" ".join(BREADTH_GRID)}. '
                         f'The pair-subset conditions ({", ".join(PAIR_SUBSET) or "none"}) are '
                         f'not in the default and need --models and --reps.')
    ap.add_argument('--kirby-dir', default=KIRBY_DIR,
                    help=f'The KIRBy checkout the jobs cd into (default: '
                         f'{KIRBY_DIR}). The preamble has told the operator to '
                         f'"regenerate with --kirby-dir" since it was written; '
                         f'until 2026-09-01 the flag did not exist.')
    ap.add_argument('--qsar-dir', default=QSAR_DIR,
                    help=f'This checkout, which the runner loads the shared spec '
                         f'from (default: {QSAR_DIR}).')
    ap.add_argument('--include-depth-conditions', action='store_true',
                    help=f'The DEEP RUN: also run the depth-only conditions '
                         f'({", ".join(DEPTH_ONLY)}), on a named subset of pairs. Requires '
                         f'--models and --reps, exactly as the QM9 deep run does -- the author '
                         f'ruled on 2026-08-28 that the validation datasets get the same noise '
                         f'as QM9, and QM9 runs these on a dozen pairs rather than the whole '
                         f'grid. Which pairs comes from the screen (RERUN_PLAN.md 13.17 B).')
    args = ap.parse_args()

    if args.conditions:
        conditions = list(args.conditions)
    else:
        conditions = list(BREADTH_GRID) + (
            list(DEPTH_ONLY) if args.include_depth_conditions else [])

    bad = [c for c in conditions if c in RETIRED]
    if bad:
        ap.error(f"{', '.join(bad)} is retired -- {NOISE_CONDITIONS_FILE.name} lists it under "
                 f"not_run. It cannot go in a job script.")
    unknown = [c for c in conditions
               if c not in FULL_GRID + DEPTH_ONLY]
    if unknown:
        ap.error(f"unknown condition(s) {', '.join(unknown)}; "
                 f"{NOISE_CONDITIONS_FILE.name} knows {', '.join(FULL_GRID + DEPTH_ONLY)}")

    # The deep run is a SUBSET run, on both pipelines. Asking for the depth-only
    # conditions across the whole grid is the accident this guards: on QM9 the deep
    # run is about a dozen pairs and the generator refuses `--stage 2` without
    # --models and --reps rather than inventing a default. The validation datasets
    # now get the same noise as QM9 (author, 2026-08-28), which means the same
    # conditions on the same shape of run -- not these three across 8 models x 6
    # representations, which is 3 x the breadth grid's cost and was never the design.
    deep = [c for c in conditions if c in DEEP_ONLY_SET]
    if deep and not (args.models and args.reps):
        ap.error(
            f"the deep run ({', '.join(deep)}) goes on a named subset of "
            f"model-and-representation pairs, not the full grid, so it needs --models and "
            f"--reps -- the same rule the QM9 generator applies to its own deep run. Which "
            f"pairs comes from the screen; see RERUN_PLAN.md 13.17 B. To run the breadth grid "
            f"without them, take the default: {' '.join(BREADTH_GRID)}.")

    # A pair-subset condition across the whole grid is the accident this guards.
    restricted = [c for c in conditions if c in PAIR_SUBSET]
    if restricted and not (args.models and args.reps):
        n = PAIR_SUBSET[restricted[0]]['n_pairs']
        ap.error(
            f"{', '.join(restricted)} runs on about {n} model-and-representation pairs, not the "
            f"full grid, so it needs --models and --reps. Which pairs comes from the screen; see "
            f"RERUN_PLAN.md 13.13. To run the rest without it, take the default: "
            f"{' '.join(BREADTH_GRID)}.")
    if restricted and args.models and args.reps:
        n_pairs = len(args.models) * len(args.reps)
        want = PAIR_SUBSET[restricted[0]]['n_pairs']
        if n_pairs > 2 * want:
            ap.error(
                f"--models x --reps is {n_pairs} pairs, and {', '.join(restricted)} is meant to "
                f"run on about {want}. If that is deliberate, say so in RERUN_PLAN.md 13.13 "
                f"first and raise n_pairs in {NOISE_CONDITIONS_FILE.name} -- the file is where "
                f"the decision lives.")

    models = args.models or MODELS_ALL
    reps = args.reps or ALL_REPS
    condition_args = ' '.join(conditions)

    print(f"Conditions ({len(conditions)}, from {NOISE_CONDITIONS_FILE.name}): "
          f"{condition_args}")
    if not args.conditions:
        if PAIR_SUBSET:
            print(f"  pair subset, NOT in the default: {', '.join(PAIR_SUBSET)} "
                  f"(needs --conditions with --models and --reps)")
        if not args.include_depth_conditions:
            print(f"  depth-only, NOT run: {', '.join(DEPTH_ONLY)}  "
                  f"(--include-depth-conditions)")
    print(f"  retired, never run:  {', '.join(RETIRED)}")
    print(f"  {len(models)} model(s) x {len(reps)} representation(s)")

    output_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    scripts = []

    for model in models:
        # ONE ARRAY PER MODEL. Its tasks are (representation x dataset), and the
        # wall clock is the WORST dataset in it, because one clock covers the
        # whole array.
        # THE TANIMOTO GAUSSIAN PROCESS RUNS ON ECFP4 ALONE. Its kernel is a
        # ratio of set overlaps, defined on BINARY vectors. Sort & Slice is built
        # with sub_counts=True so its features are small integers, and the other
        # four are continuous; the fit is refused at run time. Without this, 5 of
        # every 6 tasks queue, start, raise on every noise level and write an
        # empty file -- the identical defect the QM9 generator carried until
        # 2026-08-31 (RERUN_PLAN.md 2.33c), and QM9 restricts it the same way.
        model_reps = (['ECFP4'] if model.startswith('GP-Tanimoto')
                      else list(reps))
        n_tasks = len(model_reps) * len(DATASETS)
        hours = max(wall_clock(model, d, len(conditions), len(DOSE_LEVELS))
                    for d, _ in DATASETS)
        cases = '\n'.join(
            f'  {d}) dataset_cli="{cli}" ;;' for d, cli in DATASETS)
        content = (
            SLURM_HEADER.format(safe_name=f'{model.lower()}'[:30], mem='128G',
                                partition='long', time_limit=f'{hours}:00:00')
            + PREAMBLE.format(kirby_dir=args.kirby_dir,
                              qsar_dir=args.qsar_dir,
                              model=model,
                              levels_json=repr(DOSE_LEVELS),
                              condition_list_py=repr(conditions))
            + SLURM_BODY.format(model=model, model_lower=model.lower(),
                                reps_list=' '.join(model_reps),
                                reps_safe_list=' '.join(safe_name(r) for r in model_reps),
                                datasets_list=' '.join(d for d, _ in DATASETS),
                                dataset_cases=cases,
                                conditions=condition_args)
        )
        filename = f"val_{model.lower()}.sh"
        with open(os.path.join(output_dir, filename), 'w') as f:
            f.write(content)
        scripts.append((filename, n_tasks, hours))

    # The smoke test runs RF and SVM, so its guard has to cover both.
    herg_path, herg_cli = next(d for d in DATASETS if d[0] == 'herg')
    smoke = (
        SLURM_HEADER.format(safe_name='smoke', mem='128G', partition='short',
                            time_limit='1:00:00')
        + PREAMBLE.format(kirby_dir=args.kirby_dir, qsar_dir=args.qsar_dir,
                          model='RF SVM', levels_json=repr(DOSE_LEVELS),
                          condition_list_py=repr(conditions))
        + SMOKE_BODY.format(dataset=herg_path, dataset_cli=herg_cli)
    )
    with open(os.path.join(output_dir, 'smoke_test.sh'), 'w') as f:
        f.write(smoke)

    # The preflight is NOT a SLURM script and deliberately has no header: its
    # whole value is that it answers the hERG question on a login node, before
    # any task is queued.
    with open(os.path.join(output_dir, 'preflight.sh'), 'w') as f:
        f.write(PREFLIGHT_BODY.format(kirby_dir=args.kirby_dir))

    # SUBMIT SCRIPT. It used to print "Submitted N jobs" from a counter it
    # incremented itself, so it said 144 even if every sbatch had been rejected.
    # It now reads sbatch's own exit status and refuses to claim a submission
    # that did not happen.
    total = sum(n for _, n, _ in scripts)
    lines = ["#!/bin/bash",
             "# Submit the laboratory accuracy grid: one array per model.",
             f"# {len(scripts)} arrays, {total} tasks. The account and the wall clock",
             "# are in the scripts; the throttle is here because it is queue state.",
             "THROTTLE=${THROTTLE:-4}",
             "ok=0; bad=0", ""]
    for name, n, hours in sorted(scripts):
        lines.append(f"# {name}: {n} tasks, --time={hours}:00:00")
        lines.append(f'if sbatch --array=0-{n - 1}%$THROTTLE {name}; then '
                     f'ok=$((ok+1)); else bad=$((bad+1)); fi')
    lines += ["",
              f'echo "submitted $ok of {len(scripts)} arrays'
              f' ({total} tasks); $bad rejected"',
              'if [ "$bad" -ne 0 ]; then exit 1; fi', ""]

    with open(os.path.join(output_dir, 'submit_all.sh'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Generated {len(scripts)} array scripts, {total} tasks total, "
          f"+ submit_all.sh")
    for name, n, hours in sorted(scripts):
        print(f"    {name:24s} --array=0-{n - 1}  --time={hours}:00:00")


if __name__ == '__main__':
    main()
