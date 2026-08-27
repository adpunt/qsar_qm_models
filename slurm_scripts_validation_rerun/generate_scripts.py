#!/usr/bin/env python3
"""Generate individual SLURM scripts for validation re-run.

Each script runs ONE model x ONE rep x ONE dataset.
Writes to isolated dir: results/validation_rerun/{rep}_{dataset}/
This avoids race conditions from parallel jobs writing the same file.

After all jobs complete, run merge_results.py to combine into results/validation/.
"""
import argparse
import json
import os
from pathlib import Path

MODELS_ALL = ['RF', 'QRF', 'XGBoost', 'DNN', 'GP', 'NGBoost', 'SVM', 'LightGBM']
ALL_REPS = ['ECFP4', 'SNS', 'MHG-GNN-pretrained', 'PDV']

KIRBY_DIR = '/data/stat-cadd/scat9264/KIRBy'
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
NOISE_CONDITIONS_FILE = Path(__file__).resolve().parent.parent / 'noise_conditions.json'
_SETTLED = json.loads(NOISE_CONDITIONS_FILE.read_text())
FULL_GRID = [c['name'] for c in _SETTLED['stage_1_full_grid']]
DEPTH_ONLY = [c['name'] for c in _SETTLED['stage_2_depth_only']]
RETIRED = [c['name'] for c in _SETTLED['not_run']]

TIME_LIMITS = {
    'RF': '8:00:00',
    'QRF': '8:00:00',
    'XGBoost': '8:00:00',
    'LightGBM': '8:00:00',
    'SVM': '16:00:00',
    'NGBoost': '24:00:00',
    'DNN': '24:00:00',
    'GP': '47:59:00',
}

PREAMBLE = """KIRBY_DIR="{kirby_dir}"
QSAR_DIR="{qsar_dir}"

cd "$KIRBY_DIR"
. "$QSAR_DIR/setup.sh"

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
#SBATCH --output=slurm-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --partition={partition}
#SBATCH --time={time_limit}
#SBATCH --mail-user=adelaide.punt@stcatz.ox.ac.uk
#SBATCH --mail-type=END,FAIL

"""

SLURM_BODY = """
cd tests

python alternative_data_noise_robustness.py \\
    --datasets {dataset_cli} \\
    --models {model} \\
    --reps {rep} \\
    --conditions {conditions} \\
    --results-root results/validation_rerun/{rep_safe}_{dataset}

echo "Done: {model} x {rep} x {dataset}"
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
    ap.add_argument('--include-depth-conditions', action='store_true',
                    help=f'Also run the depth-only conditions ({", ".join(DEPTH_ONLY)}). '
                         f'Off by default: RERUN_PLAN.md 6.3 puts them in the depth run, so '
                         f'they enter the experimental datasets only if that is run here too. '
                         f'Adds {len(DEPTH_ONLY)} of {len(FULL_GRID) + len(DEPTH_ONLY)} '
                         f'conditions, so roughly {100 * len(DEPTH_ONLY) // len(FULL_GRID)}% '
                         f'more compute.')
    args = ap.parse_args()

    conditions = list(FULL_GRID) + (list(DEPTH_ONLY) if args.include_depth_conditions else [])
    bad = [c for c in conditions if c in RETIRED]
    assert not bad, f"retired condition(s) reached the job scripts: {bad}"
    condition_args = ' '.join(conditions)

    print(f"Conditions ({len(conditions)}, from {NOISE_CONDITIONS_FILE.name}): "
          f"{condition_args}")
    if not args.include_depth_conditions:
        print(f"  depth-only, NOT run: {', '.join(DEPTH_ONLY)}  (--include-depth-conditions)")
    print(f"  retired, never run:  {', '.join(RETIRED)}")

    output_dir = args.out_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    scripts = []

    for model in MODELS_ALL:
        for rep in ALL_REPS:
            # GP is PDV-only in the code
            if model == 'GP' and rep != 'PDV':
                continue
            for dataset, dataset_cli in DATASETS:
                sn = f"{model}_{safe_name(rep)}_{dataset}"
                content = (
                    SLURM_HEADER.format(safe_name=sn[:30], mem='128G',
                                        partition='long',
                                        time_limit=TIME_LIMITS[model])
                    + PREAMBLE.format(kirby_dir=KIRBY_DIR, qsar_dir=QSAR_DIR,
                                      model=model,
                                      condition_list_py=repr(conditions))
                    + SLURM_BODY.format(dataset=dataset, dataset_cli=dataset_cli,
                                        model=model, rep=rep,
                                        rep_safe=safe_name(rep),
                                        conditions=condition_args)
                )
                filename = f"val_{model.lower()}_{safe_name(rep)}_{dataset}.sh"
                with open(os.path.join(output_dir, filename), 'w') as f:
                    f.write(content)
                scripts.append(filename)

    # The smoke test runs RF and SVM, so its guard has to cover both.
    herg_path, herg_cli = next(d for d in DATASETS if d[0] == 'herg')
    smoke = (
        SLURM_HEADER.format(safe_name='smoke', mem='128G', partition='short',
                            time_limit='1:00:00')
        + PREAMBLE.format(kirby_dir=KIRBY_DIR, qsar_dir=QSAR_DIR, model='RF SVM',
                          condition_list_py=repr(conditions))
        + SMOKE_BODY.format(dataset=herg_path, dataset_cli=herg_cli)
    )
    with open(os.path.join(output_dir, 'smoke_test.sh'), 'w') as f:
        f.write(smoke)

    lines = ["#!/bin/bash", "# Submit all validation re-run jobs", "COUNT=0", ""]
    for s in sorted(scripts):
        lines.append(f"sbatch {s}")
        lines.append("COUNT=$((COUNT + 1))")
    lines += ["", f'echo "Submitted $COUNT jobs (expected {len(scripts)})"', ""]

    with open(os.path.join(output_dir, 'submit_all.sh'), 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Generated {len(scripts)} SLURM scripts + submit_all.sh")


if __name__ == '__main__':
    main()
