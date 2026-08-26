#!/bin/bash
# ============================================================================
# PREFLIGHT — run this ONCE on the server BEFORE submitting anything.
# ============================================================================
# It costs ~2 minutes and catches the failures that would otherwise waste days
# of queue time. Two of these were caught locally and are real:
#
#   * quantile_forest vs scikit-learn version clash. On this laptop
#     RandomForestQuantileRegressor.fit() raises
#     "Invalid parameter 'monotonic_cst'". If the server env has the same
#     clash, every QRF job fails on contact.
#   * The cached hERG CSV has been written with different label column names by
#     different versions. The loader now accepts pKi / pChEMBL / pchembl_value,
#     but the cache still has to exist and parse.
#
#   sbatch --account=<acct> --partition=short preflight.sh
#   # or just run it in an interactive session:
#   #   bash preflight.sh
# ============================================================================
#SBATCH --job-name=unc_preflight
#SBATCH --output=logs/preflight_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -uo pipefail
OOF=3        # fold count used by the section-5 smoke run; checks below compare against it

cd /data/stat-cadd/scat9264/KIRBy
. /data/stat-cadd/scat9264/qsar_qm_models/setup.sh

# Activation is not optional. micromamba has never worked on this cluster, so
# the `export MAMBA_EXE=...` lines that used to sit above `. setup.sh` pointed
# at a file that does not exist -- and nothing checked. setup.sh falls through
# to its conda branch; if that also fails, the task carries on in the system
# Anaconda at /apps/system/..., which has no gpytorch, no quantile_forest and
# no ngboost. The job then runs, finds nothing to do, and writes no rows. That
# is what happened to 12822693 and 12822694 (RERUN_PLAN.md section 2.8d).
if [ -z "${CONDA_PREFIX:-}" ]; then
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
# The test above cannot fail once setup.sh has run: setup.sh:83 prepends
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
export TMPDIR="${TMPDIR:-/tmp}/qsar_${SLURM_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT


export LD_LIBRARY_PATH="${CONDA_PREFIX:-}/lib:${LD_LIBRARY_PATH:-}"   # guarded: set -u is on
cd tests

echo "############ 0. the two pipelines build the SAME models"
# Both pipelines read qsar_qm_models/models/model_defaults.py. This confirms
# they resolve the SAME copy of it, that every model can still be constructed
# with the libraries installed here, and that no estimator default has drifted
# under a library upgrade. It is the check that would have caught XGBoost
# training at three times the step size on one side.
export QSAR_QM_MODELS_ROOT=/data/stat-cadd/scat9264/qsar_qm_models
export KIRBY_ROOT=/data/stat-cadd/scat9264/KIRBy
python -u "$QSAR_QM_MODELS_ROOT/scripts/audit_pipeline_parity.py" --strict \
  || { echo "FAIL: the two pipelines do not agree, or this environment cannot build a model."
       echo "      Nothing below matters until this passes. Do NOT submit."; exit 1; }

echo
echo "############ 1. the patched pipeline imports, and has the new flags"
python -u alternative_data_noise_robustness.py --help 2>&1 \
  | grep -E -- "--conditions|--unc-conditions|--oof-folds" \
  || { echo "FAIL: patched flags missing — is KIRBy up to date on this host?"; exit 1; }

echo
echo "############ 2. NoiseInject exposes the new recording API"
python - <<'PY'
import sys, numpy as np
try:
    from noiseInject import NoiseInjectorRegression
except Exception as e:
    print("FAIL: cannot import noiseInject:", e); sys.exit(1)
try:
    from noiseInject import CONDITIONS, dose_tolerance   # 1.0.0 or newer
except Exception as e:
    print("FAIL: NoiseInject is the OLD version (pre-1.0.0):", e)
    print("      pip install --no-deps -e <path-to-NoiseInject>")
    sys.exit(1)
for gone in ('legacy', 'quantile', 'threshold', 'hetero', 'valprop'):
    if gone in CONDITIONS:
        print(f"FAIL: the deleted condition {gone!r} is still reachable"); sys.exit(1)
inj = NoiseInjectorRegression.from_condition('gaussian', random_state=42)
missing = [m for m in ('noise_scale', 'inject_verbose', 'scale_map', 'unit_dose')
           if not hasattr(inj, m)]
if missing:
    print("FAIL: NoiseInject is missing:", missing); sys.exit(1)
y = np.random.RandomState(0).normal(size=2000)
r = inj.inject_verbose(y, 0.6)
assert np.array_equal(r.y_clean + r.epsilon, r.y_noisy), \
    "epsilon does not reconstruct the noisy label exactly"
assert np.array_equal(
    NoiseInjectorRegression.from_condition('gaussian', random_state=42).inject(y, 0.6),
    r.y_noisy), "inject_verbose does not match inject"
z = inj.inject_verbose(y, 0.0)
assert np.array_equal(z.epsilon, np.zeros(len(y))), "level 0 is not EXACTLY zero"
for col in ('noise_type', 'unit_dose_g', 'realised_dose_label_units',
            'affected_molecule_fraction', 'effective_n', 'seed'):
    assert r.as_row()[col] is not None, f"provenance column {col} is blank"
print("OK: the redesigned API is present, the deleted conditions are gone,")
print("    epsilon reconstructs the label exactly and level 0 is a true zero")
PY
[ $? -ne 0 ] && exit 1

echo
echo "############ 2b. the two injectors agree — RERUN_PLAN.md gate 2"
# The noise scheme is implemented twice: Rust for QM9, Python for these three
# datasets and every uncertainty number. They drifted apart once before and
# nothing noticed. This is the check that fails when they do.
#
# It needs the reference implementation built; if cargo is not on the server,
# that is a REAL gap and it says so rather than passing quietly.
QSAR=/data/stat-cadd/scat9264/qsar_qm_models
if [ ! -f "$QSAR/scripts/crosscheck_injectors.py" ]; then
    echo "FAIL: crosscheck_injectors.py is missing from $QSAR/scripts"
    exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "FAIL: cargo is not available, so the reference implementation cannot be"
    echo "      built and the two injectors CANNOT be compared. Run the check on a"
    echo "      machine that has it and paste the result, or install rust here."
    exit 1
fi
( cd "$QSAR" && python scripts/crosscheck_injectors.py --seeds 20 )
[ $? -ne 0 ] && exit 1

echo
echo "############ 3. every uncertainty model can actually be fitted"
python - <<'PY'
import numpy as np, sys
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=8, noise=0.3, random_state=0)
bad = []

def probe(name, fn):
    try:
        fn(); print(f"  OK    {name}")
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}"); bad.append(name)

def _qrf():
    from quantile_forest import RandomForestQuantileRegressor
    m = RandomForestQuantileRegressor(n_estimators=10, random_state=42).fit(X, y)
    q = m.predict(X[:5], quantiles=[0.16, 0.5, 0.84])
    assert q.shape[1] == 3
probe("QRF (quantile_forest)", _qrf)

def _ngb():
    from ngboost import NGBRegressor
    m = NGBRegressor(n_estimators=20, verbose=False, random_state=42).fit(X, y)
    d = m.pred_dist(X[:5]); assert len(d.scale) == 5
probe("NGBoost", _ngb)

def _gp():
    import gpytorch, gauche  # noqa
probe("GP (gpytorch + gauche)", _gp)

def _bnn():
    import torchbnn  # noqa
    from torchhk import transform_model  # noqa
probe("BNN/VBLL (torchbnn + torchhk)", _bnn)

if bad:
    print("\nFAIL: do NOT submit jobs for:", ", ".join(bad))
    sys.exit(1)
print("\nOK: all uncertainty models fit")
PY
[ $? -ne 0 ] && exit 1

echo
echo "############ 4. all three datasets load"
python - <<'PY'
import sys, importlib.util
spec = importlib.util.spec_from_file_location('p', 'alternative_data_noise_robustness.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
bad = []
try:
    df = m.download_openadmet()
    for col_key in ('LogD', 'Caco'):
        c = next((c for c in df.columns if col_key in c), None)
        print(f"  {'OK   ' if c else 'FAIL '} openadmet column for {col_key}: {c}")
        if not c: bad.append(col_key)
except Exception as e:
    print("  FAIL  openadmet:", e); bad.append('openadmet')
try:
    s, l = m.load_chembl_herg(); print(f"  OK    hERG: {len(s)} molecules")
except Exception as e:
    print("  FAIL  hERG:", type(e).__name__, e); bad.append('herg')
sys.exit(1 if bad else 0)
PY
[ $? -ne 0 ] && exit 1

echo
echo "############ 4b. which (dataset, condition) pairs are DEGENERATE"
python - <<'PY'
import sys, importlib.util, numpy as np
sys.path.insert(0, '/data/stat-cadd/scat9264/NoiseInject')
spec = importlib.util.spec_from_file_location('p', 'alternative_data_noise_robustness.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from noiseInject import NoiseInjectorRegression as NI

DATASETS = ['logd', 'caco2', 'herg_ki']
REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']
# Read the condition list from the GENERATED scripts, not from the module: after
# --drop-strategies the two differ and the printed indices would be wrong.
import re as _re, glob as _glob
_sh = sorted(_glob.glob('/data/stat-cadd/scat9264/qsar_qm_models/slurm_scripts_uncertainty_rerun/unc_*.sh'))
STRATS = m.STRATEGIES
if _sh:
    _mm = _re.search(r'^STRATS=\((.*?)\)', open(_sh[0]).read(), _re.M)
    if _mm:
        STRATS = _mm.group(1).split()
        print(f"  (condition list read from {_sh[0].split('/')[-1]}: {STRATS})")

labels = {}
df = m.download_openadmet()
c = next((c for c in df.columns if 'LogD' in c), None)
labels['logd'] = m.load_openadmet_endpoint(df, c, log_transform=False)[1]
c = next((c for c in df.columns if 'Caco' in c and 'Efflux' in c), None)
labels['caco2'] = m.load_openadmet_endpoint(df, c, log_transform=True)[1]
labels['herg_ki'] = m.load_chembl_herg()[1]

print("  distinct noise scales per (dataset, condition) -- 1 means it is CONSTANT,")
print("  so 'which molecules are unreliable' is UNDEFINED there, not zero")
print(f"  {'dataset':9s} " + " ".join(f"{s:>10s}" for s in STRATS))
degenerate = []
for ds in DATASETS:
    y = labels[ds]
    row = []
    for st in STRATS:
        n = len(np.unique(np.round(
            NI.from_condition(st, random_state=0).noise_scale(y, 1.0, reference=y), 12)))
        row.append(f"{n:>10d}")
        # Constant BY DESIGN for the shape-only conditions and grouped-shifted:
        # every molecule gets the same scale there, so question B is undefined
        # rather than answerable. Only flag a condition that should have a pattern.
        _flat_by_design = st in ('gaussian', 'laplace', 'grouped_shifted') or \
            st.startswith('student_t')
        if n == 1 and not _flat_by_design:
            degenerate.append((ds, st))
    print(f"  {ds:9s} " + " ".join(row))

print()
print("  The shape-only conditions (gaussian, student_t_*, laplace) and grouped_shifted")
print("  are constant everywhere BY DESIGN -- every molecule gets the same scale, so")
print("  'which molecules are unreliable' is UNDEFINED there, not zero. Those answer")
print("  question A. grouped_wider and censoring are the ones with a pattern to find.")
if degenerate:
    print()
    print("  DEGENERATE ARMS (constant noise scale -> question B undefined):")
    for ds, st in degenerate:
        print(f"    {ds} x {st}")
    n_st, n_rep = len(STRATS), len(REPS)
    idx = []
    for ds, st in degenerate:
        d_i, s_i = DATASETS.index(ds), STRATS.index(st)
        for r_i in range(n_rep):
            idx.append(d_i * (n_st * n_rep) + r_i * n_st + s_i)
    print()
    print("  Either skip these array indices in EVERY unc_*.sh:")
    print("    " + ",".join(str(i) for i in sorted(idx)))
    print("  or regenerate with --threshold-quantile 0.1 so the cut-points come from the")
    print("  label distribution (note: that changes the injected noise).")
else:
    print("  No degenerate arms.")
PY

echo
echo "############ 4c. the conditions differ in SHAPE at a matched amount"
python - <<'PYEOF'
# This replaced a check that heteroscedastic and value-proportional ranked
# molecules identically. They did -- which is why they, and the two other
# value-keyed conditions, were deleted in noiseInject 1.0.0. The property that
# has to hold now is the opposite one: every condition delivers the same
# AMOUNT, so any difference between them is a difference of shape.
import sys, numpy as np
sys.path.insert(0, '/data/stat-cadd/scat9264/NoiseInject')
from noiseInject import NoiseInjectorRegression as NI, CONDITIONS, dose_tolerance
y = np.random.RandomState(0).normal(2.0, 1.2, 20000)
g = np.random.RandomState(1).randint(0, 400, 20000)
tau = 0.5 * float(y.std())
off, tails = [], {}
for cond, spec in CONDITIONS.items():
    if spec['strategy'] == 'censoring':
        continue
    r = NI.from_condition(cond, random_state=0).inject_verbose(y, tau, groups=g)
    tol = dose_tolerance(r.epsilon, r.effective_n, nu=spec.get('nu'))
    if abs(r.realised_dose_label_units / tau - 1) > tol:
        off.append("%s %+.1f%% (tolerance %.1f%%)" % (
            cond, 100 * (r.realised_dose_label_units / tau - 1), 100 * tol))
    tails[cond] = float(np.mean(np.abs(r.epsilon) > 3 * tau))
if off:
    print("FAIL: these conditions did not deliver the amount asked for: " + "; ".join(off))
    sys.exit(1)
print("  OK    every condition delivers the amount it was asked for.")
print("        The six it replaced spanned 0.49x to 2.00x at one setting.")
print("  labels off by more than 3x the dose, at IDENTICAL total noise:")
for cond, t in sorted(tails.items(), key=lambda kv: kv[1]):
    print("    %-18s %5.2f%%" % (cond, 100 * t))
if tails.get('student_t_nu3', 0) <= 3 * tails.get('gaussian', 1):
    print("FAIL: the conditions no longer differ in shape - nothing to compare")
    sys.exit(1)
PYEOF
[ $? -ne 0 ] && exit 1

echo "############ 5. one real task, smallest dataset, 2 sigmas — end to end"
OUT=$(mktemp -d)
python -u alternative_data_noise_robustness.py \
    --datasets herg_ki --models NGBoost --reps ECFP4 \
    --conditions grouped_wider --sigmas 0.0 1.0 \
    --unc-conditions all --oof-folds "$OOF" \
    --results-root "$OUT" || { echo "FAIL: end-to-end run errored"; exit 1; }

python - "$OUT" "$OOF" <<'PY'
import sys, glob, pandas as pd
root = sys.argv[1]
f = glob.glob(f"{root}/**/*_uncertainty_values.csv", recursive=True)
if not f:
    print("FAIL: no uncertainty file written"); sys.exit(1)
d = pd.read_csv(f[0])
need = {'split','noise_type','sigma','fold','uncertainty','noise_scale',
        'noise_pattern','injected_noise','oof_folds_ok'}
miss = need - set(d.columns)
if miss: print("FAIL: missing columns", miss); sys.exit(1)
print(f"  OK    {f[0].split('/')[-1]}: {len(d)} rows")
print(f"  splits    : {sorted(d['split'].unique())}")
print(f"  conditions: {sorted(d['noise_type'].unique())}")
print(f"  folds     : {sorted(d['fold'].unique())}  (must be all 5, not just one)")
if 'train_oof' not in set(d['split']):
    print("FAIL: no out-of-fold training rows — question A cannot be answered"); sys.exit(1)
tr = d[d.split=='train_oof']
z = tr[tr.sigma==0.0]
if len(z) and z['injected_noise'].abs().max() != 0:
    print("FAIL: sigma=0 is not a clean control"); sys.exit(1)
print("  OK    sigma=0 control is exactly zero; out-of-fold training rows present")
pat = tr['noise_pattern']
# A constant pattern is EXPECTED for the shape-only conditions and for
# grouped-shifted: every molecule gets the same noise scale there, so question B
# is undefined rather than answerable. It is a defect only for a condition that
# is supposed to have a pattern.
_flat_by_design = {'gaussian', 'laplace', 'grouped_shifted'} | {
    c for c in d['noise_type'].unique() if str(c).startswith('student_t')}
if pat.nunique() <= 1 and d['noise_type'].iloc[0] not in _flat_by_design:
    print(f"FAIL: noise_pattern is constant for {d['noise_type'].iloc[0]} - the "
          f"question-B confound control is unusable"); sys.exit(1)
if tr['uncertainty'].notna().sum() == 0:
    print("FAIL: every out-of-fold uncertainty is blank"); sys.exit(1)
want = int(sys.argv[2])
if (tr['oof_folds_ok'] < want).any():
    print(f"WARN: some out-of-fold passes were truncated "
          f"(min {int(tr['oof_folds_ok'].min())}/{want} inner folds)")
print(f"  OK    noise_pattern varies ({pat.nunique()} distinct); "
      f"{tr['uncertainty'].notna().sum()} finite out-of-fold uncertainties")
PY
status=$?
rm -rf "$OUT"
[ $status -ne 0 ] && exit 1

echo
echo "############ ALL PREFLIGHT CHECKS PASSED — safe to submit"
