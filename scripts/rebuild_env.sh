#!/bin/bash
# =============================================================================
# Rebuild env_test on the cluster, then prove it, in ONE submission.
#
#     cd /data/stat-cadd/scat9264/qsar_qm_models
#     git pull
#     sbatch scripts/rebuild_env.sh
#
# then send back  ~/env_rebuild_report.txt.
#
# To see what it would do first, changing nothing:
#     REBUILD_DRY_RUN=1 bash scripts/rebuild_env.sh
#
# It runs on a login node. It builds with micromamba, whose solver fits inside
# the per-user memory cap that killed conda's on 2026-08-27; conda does
# everything else and the result is an ordinary conda environment. If micromamba
# cannot be fetched it puts itself in a small allocation instead.
#
# WHAT IT DOES, in order:
#   0  refuses to run while any of your jobs are queued or running
#   1  records what the OLD environment was, before destroying it
#   2  rebuilds it IN THE SAME PREFIX from env.yml + pip-constraints.txt
#   3  the gate: check_environment.py --deep --validation
#   4  the two named blockers, separately, for the record
#   5  the noise half: both injectors, and the KIRBy pipeline's own import
#   6  one verdict at the end
#
# Background: RERUN_PLAN.md section 2.8i.
# =============================================================================
#SBATCH --job-name=rebuild_env
#SBATCH --account=stat-cadd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out

# No `set -e`: every check must run even if an earlier one fails. One paste,
# one report, no half answers.

REPORT="$HOME/env_rebuild_report.txt"
: > "$REPORT"
say() { echo "$@" | tee -a "$REPORT"; }
section() { say ""; say "=============================================================================="
            say "$@"; say "=============================================================================="; }

REPO="${QSAR_QM_MODELS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO" || { echo "cannot cd to $REPO"; exit 1; }

# --- a solver small enough for a login node ---------------------------------
# Measured 2026-08-27: on a login node the conda solve for env.yml is killed
# part way through "Collecting package metadata", with no message but "Killed".
# That is the memory cap, and it is conda's own solver that hits it -- conda
# 4.12 parses the whole conda-forge index into memory. micromamba solves the
# same file in a few hundred megabytes.
#
# So the build goes through micromamba and EVERYTHING ELSE still goes through
# conda: what it writes is an ordinary conda environment, `conda activate` reads
# it identically, and no job script changes. This is not the micromamba that has
# never worked here -- that was `micromamba activate` needing a shell hook. This
# binary is used for one command, `create`, and is never activated through.
MM=""
if [ "${REBUILD_NO_MICROMAMBA:-0}" != "1" ]; then
    MM_DIR="${REBUILD_MM_DIR:-$(dirname "$REPO")/.micromamba}"
    if [ ! -x "$MM_DIR/bin/micromamba" ]; then
        echo "Fetching micromamba (one file, ~5 MB) into $MM_DIR"
        mkdir -p "$MM_DIR" && ( cd "$MM_DIR" && \
            curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
            | tar -xj bin/micromamba ) 2>&1 | tail -3
    fi
    if [ -x "$MM_DIR/bin/micromamba" ]; then
        MM="$MM_DIR/bin/micromamba"
        export SETUP_CREATE_WITH="$MM"
        export MAMBA_ROOT_PREFIX="$MM_DIR/root"
        echo "Building with $MM ($("$MM" --version 2>/dev/null))"
        echo "  -- conda does everything else, and the result is an ordinary"
        echo "     conda environment."
    else
        echo "micromamba could not be fetched; falling back to conda's solver."
    fi
fi

# Only if that failed does this need a node at all. The ask is deliberately
# small so it schedules quickly: the solve needs memory, not cores or hours.
if [ -z "$MM" ] && [ -z "${SLURM_JOB_ID:-}" ] && [ "${REBUILD_NO_SRUN:-0}" != "1" ] \
   && command -v srun &>/dev/null; then
    echo "Not inside an allocation, and the conda solve needs more memory than a"
    echo "login node allows. Re-running this script inside one:"
    echo "  --partition=${REBUILD_PARTITION:-short} --cpus-per-task=4 --mem=32G --time=01:30:00"
    echo "If it sits here for a while that is the wait for a node. Leave it."
    echo "To pick your own allocation instead:"
    echo "  REBUILD_NO_SRUN=1 bash scripts/rebuild_env.sh   (inside your own srun)"
    echo ""
    exec srun --account="${REBUILD_ACCOUNT:-stat-cadd}" \
              --partition="${REBUILD_PARTITION:-short}" --nodes=1 --ntasks=1 \
              --cpus-per-task=4 --mem=32G --time=01:30:00 \
              --job-name=rebuild_env bash "$0" "$@"
fi

section "ENVIRONMENT REBUILD  --  $(date)  --  $(hostname)"
say "user: $(whoami)"
say "repo: $REPO"
say "git:  $(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format=%s 2>/dev/null)"
say "job:  ${SLURM_JOB_ID:-<not a slurm job>}"

# --- 0. never during a run --------------------------------------------------
section "0. is anything running? (a rebuild changes numbers, so it must not be)"
# What this guard is actually protecting: no job may WRITE RESULT ROWS across
# the rebuild, because rows from two environments in one file, with nothing
# recording which, is unreadable afterwards.
#
# So it blocks on this study's training jobs and on nothing else. Two lessons
# from 2026-08-27, when it refused on thirteen jobs and every one was a false
# alarm: the account runs other projects (nine of them were), and audits,
# preflights and smoke tests write no rows at all (the other three were).
JOB_PATTERN="${REBUILD_JOB_PATTERN:-^(qm9|unc_|val_|qsar|anova|vbll|gauche|mol2vec|ngboost|conformal|svm|dnn|mlp|rf_|lgb|xgb|graph)}"
# Names that produce no result rows, whatever else they match.
JOB_HARMLESS="${REBUILD_JOB_HARMLESS:-(audit|check|preflight|smoke|test|figures|analysis|merge)}"
if command -v squeue &>/dev/null; then
    ALL=$(squeue -u "$(whoami)" -h -o "%i %T %j" 2>/dev/null \
          | grep -v "^${SLURM_JOB_ID:-__none__} " | grep -v "rebuild_env")
    MINE=$(echo "$ALL"  | awk -v pat="$JOB_PATTERN" -v ok="$JOB_HARMLESS" \
                              'NF && $3 ~ pat && $3 !~ ok')
    OTHER=$(echo "$ALL" | awk -v pat="$JOB_PATTERN" -v ok="$JOB_HARMLESS" \
                              'NF && ($3 !~ pat || $3 ~ ok)')
    if [ -n "$OTHER" ]; then
        say "  note: ignoring $(echo "$OTHER" | wc -l | tr -d ' ') job(s) belonging to other work --"
        say "$OTHER" | sed 's/^/          /'
        say "        Either they belong to other work, or they write no result rows"
        say "        (audits, preflights, smoke tests). Neither can be corrupted by this."
        say ""
    fi
    if [ -n "$MINE" ]; then
        say "  REFUSING: this study has jobs in the queue."
        say "$MINE" | sed 's/^/    /'
        say ""
        say "  A rebuild changes library versions and therefore numbers. Anything"
        say "  running now would produce rows under one environment and rows under"
        say "  another, in the same results file, with nothing recording which."
        say "  Let them finish or cancel them, then rerun."
        say "  To override deliberately:  FORCE_REBUILD=1 bash scripts/rebuild_env.sh"
        say "  If the pattern is wrong:   REBUILD_JOB_PATTERN='<regex>' bash scripts/rebuild_env.sh"
        if [ "${FORCE_REBUILD:-0}" != "1" ]; then exit 2; fi
        say "  FORCE_REBUILD=1 given -- continuing anyway."
    else
        say "  OK   none of this study's jobs are queued or running"
    fi
else
    say "  ---  no squeue here; cannot check. Make sure nothing is running."
fi

# A login-node build is no longer fatal: the old problem was CUDA libraries that
# would not map under the per-user memory cap, and torch is a CPU build now.
# Warn, do not block -- being stopped twice by a preflight is worse than a slow
# build.
if [ -z "${SLURM_JOB_ID:-}" ]; then
    say ""
    say "  WARNING: not inside a Slurm allocation, so this is the login node. It"
    say "  will work, but it is slow and shares memory with everyone. Better:"
    say "    srun --account=stat-cadd --partition=short --cpus-per-task=4 --mem=32G --time=01:30:00 --pty bash"
    say "  then rerun this script."
fi

# --- conda ------------------------------------------------------------------
# micromamba has never worked on this cluster; activate through conda, which is
# what setup.sh falls through to and what every job script uses.
if ! command -v conda &>/dev/null; then
    say "FATAL: conda is not on PATH. Nothing else in this report would mean anything."
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

# --- 1. what the OLD environment was ---------------------------------------
section "1. the OLD environment, recorded before it is destroyed"
OLD_PREFIX="$(conda env list | awk '$1=="env_test"{print $NF}')"
if [ -z "$OLD_PREFIX" ]; then
    OLD_PREFIX="$(conda env list | grep -E '/env_test$' | awk '{print $NF}' | head -1)"
fi
# env_test may not be there to ask -- the 2026-08-27 build removed it and then
# died. Its location is recorded in the archive this script wrote before doing
# so, and building by NAME instead would drop several gigabytes into the home
# quota, so recover the path rather than guess.
if [ -z "$OLD_PREFIX" ]; then
    OLD_PREFIX="${REBUILD_ENV_PREFIX:-$(grep -h '^# prefix:' \
        "$REPO"/research_archive/env_test_before_rebuild_*.txt 2>/dev/null \
        | tail -1 | awk '{print $3}')}"
    [ -n "$OLD_PREFIX" ] && say "  env_test is not registered; using the last recorded prefix"
fi
say "  current prefix: ${OLD_PREFIX:-<env_test not found>}"

# The package cache defaults to ~/.conda/pkgs when the conda install itself is
# not writable, and a full solve pulls down several gigabytes -- straight into
# the home quota. Keep it beside the environment.
if [ -z "${CONDA_PKGS_DIRS:-}" ] && [ -n "$OLD_PREFIX" ]; then
    export CONDA_PKGS_DIRS="$(dirname "$OLD_PREFIX")/conda_pkgs"
    mkdir -p "$CONDA_PKGS_DIRS"
    say "  package cache: $CONDA_PKGS_DIRS  (off the home quota)"
fi

ARCHIVE="$REPO/research_archive/env_test_before_rebuild_$(date +%Y-%m-%d).txt"
if [ -n "$OLD_PREFIX" ] && [ -x "$OLD_PREFIX/bin/python" ]; then
    mkdir -p "$(dirname "$ARCHIVE")"
    {
        echo "# env_test as it stood before the 2026-08-27 rebuild"
        echo "# prefix: $OLD_PREFIX"
        echo "# host:   $(hostname)   date: $(date)"
        echo "# Rollback, if it is ever needed:"
        echo "#   conda create --prefix $OLD_PREFIX --file <this file>"
        echo "#   then rerun setup.sh's extras."
        conda list --prefix "$OLD_PREFIX" --explicit 2>/dev/null
    } > "$ARCHIVE"
    say "  written: $ARCHIVE  ($(wc -l < "$ARCHIVE") lines)"
    say "  the versions that mattered:"
    "$OLD_PREFIX/bin/python" - <<'PY' 2>&1 | tee -a "$REPORT"
import importlib.metadata as md
for p in ("torch","scikit-learn","lightgbm","xgboost","numpy","scipy",
          "gpytorch","botorch","quantile-forest","ngboost","transformers",
          "torch-geometric","noiseInject","kirby"):
    try:
        print(f"    {p:20s} {md.version(p)}")
    except Exception:
        print(f"    {p:20s} -- ABSENT --")
PY
    # This is the whole point of the pin decision: a "+cu121" style local
    # version tag means a PyPI wheel replaced the conda package.
    say ""
    say "  distinct OpenMP runtimes in the OLD environment:"
    "$OLD_PREFIX/bin/python" scripts/check_environment.py --models lgb 2>&1 \
        | sed -n '/threading runtimes/,/^$/p' | sed 's/^/    /' | tee -a "$REPORT"
else
    say "  env_test was not found -- this will be a first build, not a rebuild."
fi

# --- 2. rebuild -------------------------------------------------------------
if [ "${REBUILD_DRY_RUN:-0}" = "1" ]; then
    section "DRY RUN -- stopping before anything is changed"
    say "  Everything above is what would be recorded. Nothing has been removed,"
    say "  built or installed. Rerun without REBUILD_DRY_RUN=1 to do it for real."
    say ""
    say "written to: $REPORT"
    exit 0
fi

section "2. rebuild, in the same prefix"
if [ -n "$OLD_PREFIX" ]; then
    export ENV_TEST_PREFIX="$OLD_PREFIX"
    say "  rebuilding in place: $ENV_TEST_PREFIX"
    say "  (in place on purpose -- creating by NAME would put a multi-gigabyte"
    say "   environment in whichever envs_dir comes first, which on this cluster"
    say "   can be your home quota)"
fi
say ""
# NOT `. ./setup.sh | tee`: the left-hand side of a pipe runs in a subshell, so
# the activation it performs is thrown away and everything below this line runs
# against whatever python was already on PATH -- which on 2026-08-27 was the
# system Anaconda's 3.9. Process substitution keeps the source in this shell.
SETUP_REBUILD=1 . ./setup.sh > >(tee -a "$REPORT") 2>&1
BUILD_RC=$?
say ""
say "  interpreter now: $(command -v python)"
say "  python:          $(python -c 'import sys; print(sys.version.split()[0])' 2>&1)"

# If the build did not produce the environment, stop here. Six cascading
# failures against the wrong interpreter say nothing that this line does not.
if [ "$BUILD_RC" -ne 0 ] || [ -z "${CONDA_PREFIX:-}" ] \
   || [ "$(basename "${CONDA_PREFIX:-none}")" != "env_test" ]; then
    section "STOPPING -- the environment was not built"
    say "  setup.sh exit code: $BUILD_RC"
    say "  active prefix:      ${CONDA_PREFIX:-<none>}"
    say ""
    say "  Nothing below would be a test of env_test, so nothing below was run."
    say "  The previous environment, if there was one, has been left in place or"
    say "  put back -- see the build output above."
    say ""
    say "written to: $REPORT"
    exit 1
fi

# --- 3. the gate ------------------------------------------------------------
section "3. THE GATE -- check_environment.py --deep --validation"
say "  This is the one that has to exit 0. It constructs every model in BOTH"
say "  rosters, imports models/models.py for real, checks env.yml against what"
say "  is installed, checks noiseInject and kirby import, counts the DISTINCT"
say "  OpenMP runtime files both statically and in /proc/self/maps, and runs"
say "  both blockers under both of the thread settings the pipelines use."
say ""
python scripts/check_environment.py --deep --validation 2>&1 | tee -a "$REPORT"
GATE_RC=${PIPESTATUS[0]}
say ""
say "  gate exit code: $GATE_RC"

# --- 4. the two named blockers, on their own -------------------------------
section "4. the two blockers again, standalone, for the record"
python -c "import torch, lightgbm as lgb, numpy as np
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=400, n_features=512, random_state=0)
lgb.LGBMRegressor(n_estimators=15, verbose=-1).fit(X, y); print('LightGBM OK')" 2>&1 | tee -a "$REPORT"
LGB_RC=${PIPESTATUS[0]}
say "  lightgbm probe exit: $LGB_RC"

python -c "
import lightgbm, xgboost, numpy as np, torch, gpytorch
class G(gpytorch.models.ExactGP):
    def __init__(s, x, y, l):
        super().__init__(x, y, l)
        s.m = gpytorch.means.ConstantMean()
        s.c = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(s, x):
        return gpytorch.distributions.MultivariateNormal(s.m(x), s.c(x))
r = np.random.RandomState(0)
X = r.normal(size=(900, 208)); y = X[:, 0] * 2 + r.normal(scale=.3, size=900)
xt = torch.from_numpy(X); yt = torch.from_numpy(y)
l = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3); m = G(xt, yt, l)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(l, m)
m.train(); l.train()
o = torch.optim.Adam(m.parameters(), lr=0.1)
for _ in range(20):
    o.zero_grad(); (-mll(m(xt), yt)).backward(); o.step()
print('GP OK')" 2>&1 | tee -a "$REPORT"
GP_RC=${PIPESTATUS[0]}
say "  gaussian-process probe exit: $GP_RC   (139 or 134 = segfault)"

# --- 5. the other two code bases -------------------------------------------
section "5. the noise injector, and the KIRBy validation pipeline"
say "  -- the settled conditions resolve on both sides"
python scripts/test_noise_conditions.py 2>&1 | tail -8 | tee -a "$REPORT"
NOISE_RC=${PIPESTATUS[0]}

say ""
say "  -- the two injectors agree (Rust for QM9, Python for validation)"
if [ -x "$REPO/rust/target/release/rust_processor" ]; then
    python scripts/crosscheck_injectors.py 2>&1 | tail -6 | tee -a "$REPORT"
    CROSS_RC=${PIPESTATUS[0]}
else
    say "     SKIPPED: rust/target/release/rust_processor is not built."
    say "     Build it and rerun this check:  cd rust && cargo build --release"
    CROSS_RC=-1
fi

say ""
say "  -- the KIRBy validation pipeline imports and builds its parser"
KIRBY=""
for c in "$HOME/repos/KIRBy" /data/stat-cadd/scat9264/KIRBy /data/stat-ecr/scat9264/KIRBy; do
    [ -f "$c/tests/alternative_data_noise_robustness.py" ] && KIRBY="$c" && break
done
if [ -n "$KIRBY" ]; then
    say "     checkout: $KIRBY"
    ( cd "$KIRBY" && python tests/alternative_data_noise_robustness.py --help >/dev/null 2>&1 )
    KIRBY_RC=$?
    say "     import + parser exit: $KIRBY_RC"
else
    say "     SKIPPED: no KIRBy checkout found."
    KIRBY_RC=-1
fi

# --- 6. verdict -------------------------------------------------------------
section "6. VERDICT"
ok=1
verdict() { # verdict "<label>" <rc> [<rc that means skipped>]
    if [ "$2" -eq 0 ]; then say "  PASS  $1"
    elif [ "$2" -eq -1 ]; then say "  ----  $1 (skipped, see above)"
    else say "  FAIL  $1  (exit $2)"; ok=0; fi
}
verdict "check_environment.py --deep --validation" "$GATE_RC"
verdict "LightGBM fits" "$LGB_RC"
verdict "Gaussian process fits after the boosting libraries" "$GP_RC"
verdict "noise conditions resolve on both sides" "$NOISE_RC"
verdict "the two injectors agree" "$CROSS_RC"
verdict "KIRBy validation pipeline imports" "$KIRBY_RC"
say ""
if [ "$ok" -eq 1 ] && [ "$GATE_RC" -eq 0 ]; then
    say "  ALL CLEAR. env_test holds one threading runtime, both rosters build, and"
    say "  both of the failures that forced this rebuild are gone in ONE environment."
    say ""
    say "  Next, and only now: GP_DEFAULTS['single_thread_fit'] in"
    say "  models/model_defaults.py can go to False. It costs 11% on the Gaussian"
    say "  process alone and it was the net under exactly this failure."
else
    say "  NOT CLEAR. Do not submit anything against this interpreter."
    say "  Send this file back -- every check above says what it saw."
fi
say ""
say "written to: $REPORT"
