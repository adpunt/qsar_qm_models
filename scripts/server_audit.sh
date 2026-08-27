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

# The shell preamble that puts a child into the job environment. Resolution order
# mirrors setup.sh's own (micromamba, then mamba, then conda by `command -v`) --
# NOT a hard-coded micromamba path, which is what this script wrongly used and
# which does not exist on this cluster at all.
if [ "$USE_SETUP" = "1" ]; then
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
for WHO in "${RUNNERS[@]}"; do

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
if [ -f "$GENTEST" ]; then
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

# -----------------------------------------------------------------------------
section "WHAT THIS MEANS"
say ""
say "  Read the exit codes above. In order of what stops a launch:"
say ""
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
say ""
say "  If sections 3-6 pass under BOTH interpreters, the environment side is clear."
say ""
say "Report written to $REPORT"
echo "Report written to $REPORT"
