#!/bin/bash
# =============================================================================
# ONE script that closes every server-side question in the plan.
#
# Everything in RERUN_PLAN.md that says "check on the cluster" is in here. It
# runs in about five minutes on a login node, changes nothing, submits nothing,
# and writes ONE file:
#
#     ~/server_audit_report.txt
#
# HOW TO RUN IT
#
#     cd /data/stat-cadd/scat9264/qsar_qm_models
#     git pull
#     bash scripts/server_audit.sh
#
# then send back ~/server_audit_report.txt. Every open server-side item is
# answered by that one file.
#
# IT SOURCES setup.sh, because that is what every job script does. Auditing a
# bare `conda activate` instead would test an environment nobody runs in -- and
# that is precisely how jobs 12822693 and 12822694 died: activation failed
# silently, the job ran in the system Anaconda, found no gpytorch, and wrote
# nothing. Since 2026-08-27 setup.sh only installs anything when a hash of
# env.yml + pip-constraints.txt has changed, so on an environment that is
# already built this is activation and nothing else. If the recipe HAS changed
# it will install; that is a rebuild, and a rebuild belongs before a launch and
# never during one (RERUN_PLAN.md 2.8i).
#
#   bash scripts/server_audit.sh --no-setup
#
# skips it and audits a bare activation instead: faster, changes nothing, but
# NOT what the jobs get. Use it only for a quick look.
#
# If your site objects to a five-minute login-node job, submit it instead:
#     sbatch --account=stat-cadd --partition=short --time=00:20:00 \
#            --output=$HOME/server_audit_%j.out scripts/server_audit.sh
#
# WHAT IT ANSWERS, and where each question comes from
#   1  Which KIRBy checkout is live -- stat-cadd or stat-ecr        (2.8b)
#   2  Are the two interpreters the same, package for package       (runbook 1b)
#   3  Can every model in the roster actually be constructed        (2.8d)
#   4  Does the quantile forest fit, or does it raise on contact    (3.4.4d)
#   5  Does the Gaussian process segfault once the boosting
#      libraries are loaded -- the silent job-killer                (2.8e)
#   6  Do the two pipelines agree, parameter for parameter          (3.4.5)
#   7  Is the Rust noise binary built, and does it pass its gates   (chat A)
#   8  Is torch_geometric importable -- needed for the caching work
#
# THE GATE THIS SCRIPT FEEDS. Questions 3, 4 and 5 are now also answered, in one
# command and in one environment, by
#     python scripts/check_environment.py --deep --validation
# which additionally counts the DISTINCT OpenMP runtime files a job would load
# -- the root cause of question 5 and of the LightGBM hang (RERUN_PLAN.md 2.8i).
# This script stays because it answers the cluster-shaped questions (1, 2, 6, 7)
# that a single interpreter cannot.
# =============================================================================
#SBATCH --job-name=server_audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# No `set -e`: every check must run even if an earlier one fails. That is the
# point -- one paste, one report, no half answers.
set -uo pipefail

REPORT="${HOME}/server_audit_report.txt"
: > "$REPORT"

# Both known cluster paths are checked. Pre-set QSAR_QM_MODELS_ROOT or KIRBY_ROOT
# to point somewhere else -- that is also how this script is tested off-cluster.
QSAR_CANDIDATES=(
  ${QSAR_QM_MODELS_ROOT:-}
  /data/stat-cadd/scat9264/qsar_qm_models
  /data/stat-ecr/scat9264/qsar_qm_models
)
KIRBY_CANDIDATES=(
  ${KIRBY_ROOT:-}
  /data/stat-cadd/scat9264/KIRBy
  /data/stat-ecr/scat9264/KIRBy
)
# Drop the empty entry when the variables are unset.
QSAR_CANDIDATES=($(printf '%s\n' "${QSAR_CANDIDATES[@]}" | grep -v '^$'))
KIRBY_CANDIDATES=($(printf '%s\n' "${KIRBY_CANDIDATES[@]}" | grep -v '^$'))

say()     { echo "$@" | tee -a "$REPORT"; }
section() { say ""; say "==============================================================================";
            say "$@"; say "=============================================================================="; }

section "SERVER AUDIT  --  $(date)  --  $(hostname)"
say "user: $(whoami)"

# -----------------------------------------------------------------------------
section "1. WHICH CHECKOUT IS LIVE   (RERUN_PLAN.md 2.8b)"
# The uncertainty job generator hard-codes stat-cadd; 125 of 127 job scripts in
# KIRBy itself use stat-ecr. If the wrong one is live, 504 tasks run against an
# old checkout and produce a full set of results from unpatched code.
# -----------------------------------------------------------------------------
QSAR_ROOT=""
for d in "${QSAR_CANDIDATES[@]}"; do
  if [ -d "$d/.git" ]; then
    say ""
    say "  qsar_qm_models at $d"
    say "    last commit : $(git -C "$d" log -1 --format='%h %ad %s' --date=short 2>&1 | head -c 140)"
    say "    branch      : $(git -C "$d" rev-parse --abbrev-ref HEAD 2>&1)"
    say "    uncommitted : $(git -C "$d" status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
    say "    has the shared settings file: $([ -f "$d/models/model_defaults.py" ] && echo YES || echo 'NO  <-- this checkout is behind')"
    [ -z "$QSAR_ROOT" ] && [ -f "$d/models/model_defaults.py" ] && QSAR_ROOT="$d"
  else
    say "  qsar_qm_models at $d : ABSENT"
  fi
done
[ -z "$QSAR_ROOT" ] && QSAR_ROOT="${QSAR_CANDIDATES[0]}"

KIRBY_ROOT=""
for d in "${KIRBY_CANDIDATES[@]}"; do
  if [ -d "$d/.git" ]; then
    say ""
    say "  KIRBy at $d"
    say "    last commit : $(git -C "$d" log -1 --format='%h %ad %s' --date=short 2>&1 | head -c 140)"
    say "    uncommitted : $(git -C "$d" status --porcelain 2>/dev/null | wc -l | tr -d ' ') file(s)"
    if [ -f "$d/tests/alternative_data_noise_robustness.py" ]; then
      say "    imports the shared settings file: $(grep -c '_load_model_defaults' "$d/tests/alternative_data_noise_robustness.py" 2>/dev/null | tr -d ' ') hit(s) — 0 means this checkout is behind"
    fi
    [ -z "$KIRBY_ROOT" ] && KIRBY_ROOT="$d"
  else
    say "  KIRBy at $d : ABSENT"
  fi
done
[ -z "$KIRBY_ROOT" ] && KIRBY_ROOT="${KIRBY_CANDIDATES[0]}"

say ""
say "  Using for the rest of this audit:"
say "    QSAR_QM_MODELS_ROOT=$QSAR_ROOT"
say "    KIRBY_ROOT=$KIRBY_ROOT"
export QSAR_QM_MODELS_ROOT="$QSAR_ROOT"
export KIRBY_ROOT="$KIRBY_ROOT"

# -----------------------------------------------------------------------------
section "2. THE INTERPRETERS   (runbook 1b)"
# Two Gaussian-process jobs once ran to completion and produced nothing because
# the interpreter was missing gpytorch. There are two interpreters here and they
# are not the same one.
# -----------------------------------------------------------------------------
# EVERY CHECK RUNS THROUGH `. setup.sh`, because that is what the job scripts
# do. Testing a bare `conda activate` instead would audit an environment nobody
# runs in -- which is the exact class of mistake that produced the two dead
# jobs, and it is a mistake this script made until 2026-08-26.
#
# setup.sh is NOT read-only. It will create the conda environment if missing,
# conda-install libstdcxx-ng, and pip-install several packages. On a machine
# where jobs have already run those are satisfied and it is quick, but be aware
# this step can reach the network. Pass --no-setup to skip it and audit a bare
# activation instead; that is faster and changes nothing, but it is NOT what the
# jobs get.
#
# It is sourced in a CHILD shell. setup.sh calls `exit 1` when activation fails,
# and a sourced exit would kill this audit part-way through with no report.
USE_SETUP=1
[ "${1:-}" = "--no-setup" ] && USE_SETUP=0

# Is there an environment to activate at all?
#
# On 2026-08-27 the environment rebuild deleted env_test and its solve was then
# killed, so setup.sh found nothing to activate and began BUILDING one -- a conda
# solve plus several pip installs. Inside a 15-minute job that is not an audit,
# it is a rebuild that gets terminated at the wall clock, and the report stops at
# this section's heading with no explanation. The comment above said setup.sh can
# do this; nothing checked, which is the same shape as every other defect in this
# project's history.
#
# So: look first, and never let this step run unbounded.
# setup.sh calls the environment env_test and resolves conda by `command -v`.
# Inside a compute job conda may not be on PATH yet, so the known install roots
# are checked too rather than relying on one lookup.
ENV_PREFIX=""
for base in "$(conda info --base 2>/dev/null)" \
            "${CONDA_EXE%/bin/conda}" \
            "$HOME/miniconda3" "$HOME/anaconda3" \
            /data/stat-cadd/scat9264/miniconda3 \
            /data/stat-cadd/scat9264/anaconda3 \
            /apps/system/easybuild/software/Anaconda3; do
  [ -n "$base" ] && [ -d "$base/envs/env_test" ] && ENV_PREFIX="$base/envs/env_test" && break
done
# And accept an env_test that is already active, whatever root it came from.
[ -z "$ENV_PREFIX" ] && [ "$(basename "${CONDA_PREFIX:-}")" = "env_test" ] && ENV_PREFIX="$CONDA_PREFIX"
# setup.sh is how this project's environment becomes usable. It has always been
# the path, the job scripts source it, and auditing a bare activation instead
# audits an environment nobody runs in -- the mistake this script made until
# 2026-08-26 and then made again on 2026-08-28.
#
# So it is still sourced. What is added is a warning, because when env_test is
# absent setup.sh BUILDS it -- a conda solve plus several pip installs -- and a
# fifteen-minute job is not long enough for that. On 2026-08-27 the audit was
# killed at its wall clock part-way through a build, and the report stopped at
# section 2's heading saying nothing about why.
ENV_MISSING=0
if [ -z "$ENV_PREFIX" ]; then
  ENV_MISSING=1
  if [ "$USE_SETUP" = "1" ]; then
    say ""
    say "  ⚠️  env_test does not exist yet, so setup.sh is going to BUILD it rather"
    say "     than activate it: a conda solve and several pip installs. That is the"
    say "     right thing to happen -- it is how this environment is made -- but it"
    say "     takes tens of minutes, not the five this audit usually needs."
    say ""
    say "     If this run is inside a job, give it at least an hour or it will be"
    say "     killed part-way through the build with an unfinished report."
    say "     For a quick look that builds nothing, re-run with --no-setup; it can"
    say "     only answer sections 1, 2 and 7."
    say ""
    say "     The rebuild belongs elsewhere -- to env.yml, pip-constraints.txt"
    say "     and the deep environment probe, not to this script:"
    say "         python scripts/check_environment.py --deep --validation"
    say ""
  fi
fi
[ "${FORCE_SETUP:-0}" = "1" ] && USE_SETUP=1

# The shell preamble that puts a child into the job environment. Resolution order
# mirrors setup.sh's own (micromamba, then mamba, then conda by `command -v`) --
# NOT a hard-coded micromamba path, which is what this script wrongly used and
# which does not exist on this cluster at all.
if [ "$USE_SETUP" = "1" ]; then
  # No timeout wrapper here, deliberately: `timeout` runs a subprocess, and an
  # environment sourced inside a subprocess does not reach this child shell. The
  # guard is the existence check above -- with env_test present, setup.sh
  # activates rather than builds, which is the case that ran long.
  ENV_PREAMBLE="cd '$QSAR_ROOT' && . setup.sh >/dev/null 2>&1 || true"
else
  ENV_PREAMBLE='
    if command -v micromamba >/dev/null 2>&1; then eval "$(micromamba shell hook --shell bash)"; micromamba activate env_test
    elif command -v mamba >/dev/null 2>&1; then . "$(mamba info --base)/etc/profile.d/conda.sh"; conda activate env_test
    elif command -v conda >/dev/null 2>&1; then . "$(conda info --base)/etc/profile.d/conda.sh"; conda activate env_test
    fi
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"'
fi
[ -n "${AUDIT_PYTHON:-}" ] && ENV_PREAMBLE="true"

# job_exec <args...>  -- run python <args> in the job environment, PRESERVING its
# exit code. That matters: a segfault shows up as 139 and as nothing else.
job_exec() {
  local py="${AUDIT_PYTHON:-python}"
  bash -c "$ENV_PREAMBLE
exec \"\$0\" \"\$@\"" "$py" "$@"
}

# run_job_env <<'PY' ... PY   -- run stdin as python, in the job environment.
run_job_env() { local c; c="$(cat)"; printf '%s' "$c" | job_exec - 2>&1; }

# Which interpreter does that actually resolve to?
JOB_PY="$(printf 'import sys;print(sys.executable)' | run_job_env | tail -1)"
KIRBY_PY="/data/stat-cadd/scat9264/py311-kirby/bin/python"

say ""
say "  setup.sh sourced for every check below: $([ "$USE_SETUP" = 1 ] && echo YES || echo 'NO (--no-setup)')"
say "  (a) the environment the job scripts activate : ${JOB_PY:-COULD NOT RESOLVE}"
say "  (b) the one the two dead jobs actually used  : $([ -x "$KIRBY_PY" ] && echo "$KIRBY_PY" || echo 'ABSENT')"
case "$JOB_PY" in
  /apps/system/*)
    say ""
    say "  🔴 THAT IS THE SYSTEM PYTHON, NOT THE PROJECT ENVIRONMENT."
    say "     Activation silently failed. This is what killed jobs 12822693 and"
    say "     12822694: no gpytorch, no quantile_forest, no ngboost, an empty"
    say "     roster, and a job that exits having written nothing."
    ;;
esac

# The job environment is checked by running python through run_job_env; the
# second interpreter is checked by its absolute path.
RUNNERS=("job")
[ -x "$KIRBY_PY" ] && RUNNERS+=("kirby")
if [ -z "$JOB_PY" ] && [ ! -x "$KIRBY_PY" ]; then
  say ""
  say "  NO USABLE INTERPRETER FOUND. Everything below is skipped."
  say "  Try: cd $QSAR_ROOT && . setup.sh   and read what it prints."
  say ""
  say "Report written to $REPORT"
  exit 1
fi

# run_py <runner> <<'PY' ... PY
run_py() {
  local who="$1"; local code; code="$(cat)"
  if [ "$who" = "job" ]; then printf '%s' "$code" | run_job_env
  else printf '%s' "$code" | "$KIRBY_PY" - 2>&1; fi
}

# -----------------------------------------------------------------------------
# Everything from here runs under EVERY interpreter found, and the answers are
# compared. A check that passes in one and fails in the other is the failure
# mode that produced the two dead jobs.
# -----------------------------------------------------------------------------
# Without an interpreter that has the modelling packages, sections 3 to 6 all
# fail for one reason and report it as six. Say it once.
if [ "$ENV_MISSING" = "1" ] && [ "$USE_SETUP" = "0" ]; then
  section "3-6. MODELS, QUANTILE FOREST, GAUSSIAN PROCESS, PIPELINE PARITY"
  say ""
  say "  NOT CHECKED. env_test does not exist and --no-setup was passed, so nothing"
  say "  built it. Running these anyway reports eight launch blockers that are all"
  say "  the same missing environment, which is worse than reporting nothing."
  say ""
  say "  Run this WITHOUT --no-setup and setup.sh will build the environment, which"
  say "  is how it is meant to be made. Allow at least an hour if it is in a job."
  # Only the job environment is dropped. The second interpreter is a separate
  # install and may still be usable, so it is left in rather than assumed dead.
  KEPT=()
  for r in ${RUNNERS[@]+"${RUNNERS[@]}"}; do [ "$r" = "job" ] || KEPT+=("$r"); done
  RUNNERS=(${KEPT[@]+"${KEPT[@]}"})
  if [ "${#RUNNERS[@]}" -gt 0 ]; then
    say ""
    say "  The second interpreter IS present, so sections 3 to 6 run under it below."
  fi
fi

for WHO in ${RUNNERS[@]+"${RUNNERS[@]}"}; do

if [ "$WHO" = "job" ]; then
  section "INTERPRETER: the job environment$([ "$USE_SETUP" = 1 ] && echo ' (via . setup.sh)')"
else
  section "INTERPRETER: $KIRBY_PY"
fi
say "  python $(printf 'import sys;print(sys.version.split()[0])' | run_py "$WHO" | tail -1)"

# --- 3. can every model be constructed -------------------------------------
say ""
say "  --- 3. can every model in the roster be CONSTRUCTED (not just imported)?"
run_py "$WHO" <<'PY' | tee -a "$REPORT"
import importlib
need = ['sklearn', 'xgboost', 'lightgbm', 'ngboost', 'quantile_forest',
        'gpytorch', 'gauche', 'botorch', 'torch', 'torchbnn', 'torchhk',
        'rdkit', 'numpy', 'pandas', 'scipy', 'torch_geometric']
missing = []
for n in need:
    try:
        m = importlib.import_module(n)
        print(f"      {n:18s} {getattr(m, '__version__', '(no version)')}")
    except Exception as e:
        missing.append(n)
        print(f"      {n:18s} MISSING ({type(e).__name__})")
if missing:
    print(f"      >>> {len(missing)} MISSING: {missing}")
    print("      >>> every job requesting those models dies, or silently runs an empty roster")
PY

# --- 4. quantile forest ----------------------------------------------------
say ""
say "  --- 4. does the quantile forest FIT? (it imports fine and fails on contact)"
run_py "$WHO" <<'PY' | tee -a "$REPORT"
try:
    import numpy as np
    from quantile_forest import RandomForestQuantileRegressor
    from sklearn.datasets import make_regression
    import sklearn
    X, y = make_regression(n_samples=120, n_features=6, random_state=0)
    m = RandomForestQuantileRegressor(n_estimators=8, random_state=0).fit(X, y)
    q = m.predict(X[:4], quantiles=[0.16, 0.5, 0.84])
    assert q.shape[1] == 3
    print(f"      OK   quantile_forest fits (scikit-learn {sklearn.__version__})")
except Exception as e:
    import sklearn
    print(f"      FAIL {type(e).__name__}: {e}")
    print(f"           scikit-learn {sklearn.__version__}")
    print("           >>> LAUNCH BLOCKER: every quantile-forest task dies on contact.")
PY

# --- 5. the Gaussian-process segfault --------------------------------------
say ""
say "  --- 5. does the Gaussian process SEGFAULT once the boosting libraries are loaded?"
say "         (this kills a job with NO error message at all, so it looks like a task"
say "          that simply stopped -- see RERUN_PLAN.md 2.8e)"
GPCODE=$(cat <<'PY'
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
print("ok")
PY
)
# Test the thread settings the JOBS actually run under, by name, rather than
# whatever happens to be in the shell running this audit. The two halves of the
# study differ here: the QM9 jobs source setup.sh, which sets no thread count at
# all, while the experimental module pins both to 4 at import. Testing only the
# ambient setting is how this same check reported OK here and SEGFAULT three
# lines later in the parity audit -- same question, opposite answers, because
# they inherited different environments.
gp_probe() {   # gp_probe "<label>" <env args...>
  # NOTE: use `-u VAR` to UNSET, never `VAR=` -- setting a thread count to the
  # empty string is not the same as leaving it unset, and some numerical
  # libraries fail outright on an empty value. That produced a spurious "exit 1"
  # in this very check, which would have reported a fault the jobs do not have.
  local label="$1"; shift
  if [ "$WHO" = "job" ]; then env "$@" bash -c "$ENV_PREAMBLE
exec python -c \"\$0\"" "$GPCODE" >/dev/null 2>&1
  else env "$@" "$KIRBY_PY" -c "$GPCODE" >/dev/null 2>&1; fi
  local rc=$?
  if [ $rc -eq 0 ]; then say "      OK   $label"
  elif [ $rc -eq 139 ] || [ $rc -eq 134 ]; then say "      FAIL $label -- SEGFAULT (exit $rc)"
  else say "      FAIL $label -- exit $rc"; fi
  return $rc
}

gp_probe "as the QM9 jobs run it (no thread count set)" -u OMP_NUM_THREADS -u MKL_NUM_THREADS
QM9_RC=$?
gp_probe "as the experimental jobs run it (both pinned to 4)" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
EXP_RC=$?

if [ $QM9_RC -ne 0 ] || [ $EXP_RC -ne 0 ]; then
  say "           >>> LAUNCH BLOCKER: those Gaussian-process tasks die with NO traceback,"
  say "               so in a job array they look like tasks that simply stopped."
  gp_probe "with both pinned to 1" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
  if [ $? -eq 0 ]; then
    say "           >>> CURED by pinning both to 1. That is a stop-gap: it costs every"
    say "               tree fit its parallelism across the whole grid. The real fix is to"
    say "               install xgboost and lightgbm from the same channel as pytorch so"
    say "               only ONE threading runtime is loaded."
  else
    say "           >>> NOT cured by pinning threads. The environment has to be rebuilt"
    say "               so that only one threading runtime is present."
  fi
fi

# --- 6. do the two pipelines agree -----------------------------------------
say ""
say "  --- 6. do the two pipelines build the SAME models?"
if [ -f "$QSAR_ROOT/scripts/audit_pipeline_parity.py" ]; then
  if [ "$WHO" = "job" ]; then
    job_exec "$QSAR_ROOT/scripts/audit_pipeline_parity.py" --strict >>"$REPORT" 2>&1
  else
    "$KIRBY_PY" "$QSAR_ROOT/scripts/audit_pipeline_parity.py" --strict >>"$REPORT" 2>&1
  fi
  RC=$?
  say "      audit exit code: $RC   (0 = the two pipelines agree and this environment can build every model)"
else
  say "      SKIPPED: $QSAR_ROOT/scripts/audit_pipeline_parity.py not found -- this checkout is behind, git pull"
fi

done   # per-interpreter loop

# -----------------------------------------------------------------------------
section "7. THE RUST NOISE BINARY   (chat A)"
# -----------------------------------------------------------------------------
RUSTBIN="$QSAR_ROOT/rust/target/release/rust_processor"
if [ -x "$RUSTBIN" ]; then
  say "  built: $RUSTBIN"
  say "  built at: $(date -r "$RUSTBIN" 2>/dev/null || stat -c %y "$RUSTBIN" 2>/dev/null)"
  say ""
  say "  Its command-line flags CHANGED. The old ones are refused by name:"
  "$RUSTBIN" --help 2>&1 | grep -E '^\s+--(noise-level|dose-units|noise-shape|noise-targeting)' | tee -a "$REPORT"
  say ""
  say "  self-test (this is the gate that says the noise doses are right):"
  say "    the binary needs a labels file and a scaffold file. Run it against real"
  say "    training labels once the split exists:"
  say "      $RUSTBIN --self-test <labels.csv> --scaffold-file <groups.json>"
else
  say "  NOT BUILT at $RUSTBIN"
  say "  >>> nothing on the QM9 side can run. Build it:"
  say "        cd $QSAR_ROOT/rust && cargo build --release"
fi

# -----------------------------------------------------------------------------
section "8. THE QM9 JOB GENERATOR   (chat M)"
# -----------------------------------------------------------------------------
# The scripts are not in version control -- they are rebuilt from the generator
# every time -- so the generator is the only thing worth auditing, and the only
# question worth asking about it is whether the commands it emits are ones THIS
# interpreter's pipeline accepts. It emitted --sigma and --noise-strategy for
# weeks after both were refused by name, because nobody had run its output.
GENTEST="$QSAR_ROOT/slurm_scripts_qm9_rerun/test_generate_scripts.py"
if [ "$ENV_MISSING" = "1" ] && [ "$USE_SETUP" = "0" ]; then
  say "  NOT CHECKED. This test feeds every command the generator writes through"
  say "  the training program's own settings reader, which means importing the"
  say "  pipeline. With env_test absent the import fails, and a failed import is"
  say "  NOT evidence that the generator is wrong. Reporting it as such is how a"
  say "  missing environment gets recorded as a code fault."
elif [ -f "$GENTEST" ]; then
  say "  running $GENTEST (this imports the whole pipeline; about a minute)"
  say ""
  GENOUT="$(cd "$QSAR_ROOT" && printf '%s' "import runpy,sys; sys.argv=['t']; runpy.run_path('$GENTEST', run_name='__main__')" | run_job_env 2>&1)"
  GEN_RC=$?
  printf '%s\n' "$GENOUT" | grep -E "^(stage|guards|the real|  checked|PASS|FAIL|  - )" | head -40 | tee -a "$REPORT"
  if printf '%s' "$GENOUT" | grep -q '^PASS'; then
    say ""
    say "  PASS: every command line the generator emits is accepted by this pipeline."
  else
    say ""
    say "  FAIL (exit $GEN_RC): the generator emits commands this pipeline refuses."
    say "  >>> Every job it writes would die at argument parsing. Do not submit."
  fi
else
  say "  MISSING: $GENTEST"
  say "  >>> This checkout is behind. git pull before generating any job script."
fi

# --- 9. the noise half ------------------------------------------------------
section "9. THE NOISE CONDITIONS AND THE TWO INJECTORS"
say "  Both of these need the environment: the first imports the Python injector,"
say "  the second runs the Rust binary against it on real QM9 labels."
say ""
COND_RC=-1
NOISE_CROSS_RC=-1
if [ "$ENV_MISSING" = "1" ] && [ "$USE_SETUP" = "0" ]; then
  say "  NOT CHECKED: no environment (see section 2)."
else
  if [ -f "$QSAR_ROOT/scripts/test_noise_conditions.py" ]; then
    ( cd "$QSAR_ROOT" && job_exec scripts/test_noise_conditions.py ) 2>&1 \
        | tail -6 | sed 's/^/    /' | tee -a "$REPORT"
    COND_RC=${PIPESTATUS[0]}
    say "  conditions exit: $COND_RC"
  else
    say "  MISSING: scripts/test_noise_conditions.py -- this checkout is behind."
  fi
  say ""
  if [ -x "$QSAR_ROOT/rust/target/release/rust_processor" ]; then
    ( cd "$QSAR_ROOT" && job_exec scripts/crosscheck_injectors.py ) 2>&1 \
        | tail -5 | sed 's/^/    /' | tee -a "$REPORT"
    NOISE_CROSS_RC=${PIPESTATUS[0]}
    say "  cross-check exit: $NOISE_CROSS_RC"
  else
    say "  SKIPPED: rust/target/release/rust_processor is not built."
    say "  Build it INSIDE the environment, so it links against its RDKit:"
    say "    . ./setup.sh && cd rust && cargo build --release"
  fi
fi

# --- 10. does a real task write a real row ----------------------------------
section "10. END TO END -- a real training task, on this node, writing a real row"
say "  Everything above says the parts are present. This says the thing works."
say "  It is a genuine run of the QM9 pipeline -- same program, same arguments"
say "  shape, same noise machinery as a submitted task -- shrunk to 300 molecules"
say "  and one repetition so it finishes in a couple of minutes."
say ""
say "  It is the check nothing else makes. The failure this whole section exists"
say "  to stop does not appear at import: it is a HANG partway through fitting,"
say "  which only a fit can find."
say ""
E2E_RC=-1
E2E_OUT="${TMPDIR:-/tmp}/audit_e2e_$$.csv"
if [ "$ENV_MISSING" = "1" ] && [ "$USE_SETUP" = "0" ]; then
  say "  NOT CHECKED: no environment (see section 2)."
elif [ ! -f "$QSAR_ROOT/scripts/process_and_train.py" ]; then
  say "  MISSING: scripts/process_and_train.py"
else
  say "  running (timeout 20 minutes; lgb on ECFP4, 300 molecules, two levels)..."
  ( cd "$QSAR_ROOT/scripts" && timeout 1200 \
      bash -c "$ENV_PREAMBLE
exec python -u process_and_train.py -d QM9 -t homo_lumo_gap -m lgb -r ecfp4 \
    --noise-level 0.0 0.4 --dose-units spread --noise-shape gaussian \
    -n 300 --repetitions 1 --start-iteration 0 -s scaffold \
    --normalize True -f '$E2E_OUT'" ) 2>&1 | tail -12 | sed 's/^/    /' | tee -a "$REPORT"
  E2E_RC=${PIPESTATUS[0]}
  say ""
  say "  exit: $E2E_RC   (124 = TIMED OUT, which is the hang, not a crash)"
  if [ -s "$E2E_OUT" ]; then
    E2E_ROWS=$(( $(wc -l < "$E2E_OUT") - 1 ))
    say "  rows written: $E2E_ROWS   -> $E2E_OUT"
    head -2 "$E2E_OUT" | sed 's/^/    /' | tee -a "$REPORT"
    [ "$E2E_ROWS" -lt 1 ] && E2E_RC=1
  else
    say "  NO OUTPUT FILE. The task wrote nothing, which is the failure mode that"
    say "  looks like a queue problem: allocation spent, no rows, no error."
    E2E_RC=1
  fi
fi

# -----------------------------------------------------------------------------
section "WHAT THIS MEANS"
say ""
say "  Read the exit codes above. In order of what stops a launch:"
say ""
if [ "$ENV_MISSING" = "1" ] && [ "$USE_SETUP" = "0" ]; then
  say "   * env_test DOES NOT EXIST and --no-setup stopped anything building it."
  say "                        That is the only finding this run can make."
  say "                        Sections 3 to 6 and 8 all need an interpreter with the"
  say "                        modelling packages, so none of them ran. Re-run without"
  say "                        --no-setup and setup.sh will build it; nothing else here"
  say "                        is evidence of anything until it exists."
  say ""
fi
say "   * Section 5 FAIL  -> every Gaussian-process task dies silently. Fix before submitting"
say "                        any GP job. This is the one with no error message."
say "   * Section 4 FAIL  -> every quantile-forest task dies on contact."
say "   * Section 3 MISSING -> jobs for those models produce nothing."
say "   * Section 6 non-zero -> the two pipelines are NOT training the same models, so"
say "                        results from them cannot be compared."
say "   * Section 8 FAIL  -> every QM9 job dies at argument parsing before it trains"
say "                        anything. The generator, not the scripts, is what to fix."
say "   * Section 1 shows a checkout WITHOUT the shared settings file -> it is behind;"
say "                        git pull it before anything is submitted from it."
say "   * Section 9 non-zero -> the two injectors disagree, or the conditions do not"
say "                        resolve. Noise written by one and read by the other is"
say "                        not the same noise, so no result is comparable."
say "   * Section 10 exit 124 -> THE HANG. A real fit did not finish on this node."
say "                        Nothing may be submitted: every task would burn its"
say "                        whole allocation and write no rows and no error."
say "   * Section 10 zero rows -> the pipeline runs and produces nothing, which is"
say "                        how jobs 12822693 and 12822694 looked from the queue."
say ""
say "  Section 10 is the one to read first. Sections 1-9 say the parts are there;"
say "  section 10 is the only one where a fit actually happens on this node."
say ""
say "  If sections 3-6 pass under BOTH interpreters, the environment side is clear."
say ""
say "Report written to $REPORT"
echo "Report written to $REPORT"
