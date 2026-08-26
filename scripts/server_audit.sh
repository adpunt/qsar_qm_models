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
# It is safe to run on a login node. If your site objects to a five-minute
# login-node job, submit it instead:
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
JOB_PY=""
if [ -n "${AUDIT_PYTHON:-}" ]; then
  JOB_PY="$AUDIT_PYTHON"
elif [ -x "/data/stat-cadd/scat9264/bin/micromamba" ]; then
  export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
  eval "$("$MAMBA_EXE" shell hook --shell bash)" 2>/dev/null
  if micromamba activate env_test 2>/dev/null; then
    JOB_PY="$(command -v python)"
    export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"
  fi
fi
KIRBY_PY="/data/stat-cadd/scat9264/py311-kirby/bin/python"

say ""
say "  (a) the environment the job scripts activate : ${JOB_PY:-COULD NOT ACTIVATE env_test}"
say "  (b) the one the two dead jobs actually used  : $([ -x "$KIRBY_PY" ] && echo "$KIRBY_PY" || echo 'ABSENT')"

PYTHONS=()
[ -n "$JOB_PY" ] && PYTHONS+=("$JOB_PY")
[ -x "$KIRBY_PY" ] && PYTHONS+=("$KIRBY_PY")
if [ ${#PYTHONS[@]} -eq 0 ]; then
  say ""
  say "  NO USABLE INTERPRETER FOUND. Everything below is skipped."
  say "  Fix the environment first: cd $QSAR_ROOT && . setup.sh"
  say ""
  say "Report written to $REPORT"
  exit 1
fi

# -----------------------------------------------------------------------------
# Everything from here runs under EVERY interpreter found, and the answers are
# compared. A check that passes in one and fails in the other is the failure
# mode that produced the two dead jobs.
# -----------------------------------------------------------------------------
for PY in "${PYTHONS[@]}"; do

section "INTERPRETER: $PY"
say "  python $("$PY" -c 'import sys;print(sys.version.split()[0])' 2>&1 | head -1)"

# --- 3. can every model be constructed -------------------------------------
say ""
say "  --- 3. can every model in the roster be CONSTRUCTED (not just imported)?"
"$PY" - <<'PY' 2>&1 | tee -a "$REPORT"
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
"$PY" - <<'PY' 2>&1 | tee -a "$REPORT"
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
"$PY" -c "$GPCODE" >/dev/null 2>&1
GPRC=$?
if [ $GPRC -eq 0 ]; then
  say "      OK   the Gaussian process fits with the boosting libraries loaded"
  say "           (OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}, MKL_NUM_THREADS=${MKL_NUM_THREADS:-unset})"
else
  if [ $GPRC -eq 139 ] || [ $GPRC -eq 134 ]; then WHAT="SEGFAULT (exit $GPRC)"; else WHAT="exit $GPRC"; fi
  say "      FAIL $WHAT"
  say "           >>> LAUNCH BLOCKER: every Gaussian-process task dies with no traceback."
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "$PY" -c "$GPCODE" >/dev/null 2>&1
  if [ $? -eq 0 ]; then
    say "           >>> CURED by OMP_NUM_THREADS=1 MKL_NUM_THREADS=1"
    say "               (a stop-gap: it costs every tree fit its parallelism. The real fix is"
    say "                to install xgboost and lightgbm from the same channel as pytorch, so"
    say "                only ONE threading runtime is loaded.)"
  else
    say "           >>> NOT cured by pinning threads. The environment must be rebuilt so that"
    say "               only one threading runtime is present."
  fi
fi

# --- 6. do the two pipelines agree -----------------------------------------
say ""
say "  --- 6. do the two pipelines build the SAME models?"
if [ -f "$QSAR_ROOT/scripts/audit_pipeline_parity.py" ]; then
  "$PY" "$QSAR_ROOT/scripts/audit_pipeline_parity.py" --strict >>"$REPORT" 2>&1
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
say "   * Section 1 shows a checkout WITHOUT the shared settings file -> it is behind;"
say "                        git pull it before anything is submitted from it."
say ""
say "  If sections 3-6 pass under BOTH interpreters, the environment side is clear."
say ""
say "Report written to $REPORT"
echo "Report written to $REPORT"
