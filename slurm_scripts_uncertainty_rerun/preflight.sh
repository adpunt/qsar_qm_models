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

export MAMBA_EXE="/data/stat-cadd/scat9264/bin/micromamba"
eval "$("$MAMBA_EXE" shell hook --shell bash)"
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /data/stat-cadd/scat9264/KIRBy
. /data/stat-cadd/scat9264/qsar_qm_models/setup.sh
cd tests

echo "############ 1. the patched pipeline imports, and has the new flags"
python -u alternative_data_noise_robustness.py --help 2>&1 \
  | grep -E -- "--strategies|--unc-strategies|--oof-folds" \
  || { echo "FAIL: patched flags missing — is KIRBy up to date on this host?"; exit 1; }

echo
echo "############ 2. NoiseInject exposes the new recording API"
python - <<'PY'
import sys, numpy as np
try:
    from noiseInject import NoiseInjectorRegression
except Exception as e:
    print("FAIL: cannot import noiseInject:", e); sys.exit(1)
inj = NoiseInjectorRegression(strategy='quantile', random_state=42)
missing = [m for m in ('noise_scale', 'inject_verbose') if not hasattr(inj, m)]
if missing:
    print("FAIL: NoiseInject is the OLD version, missing:", missing)
    print("      pip install --no-deps -e <path-to-NoiseInject>")
    sys.exit(1)
y = np.random.RandomState(0).normal(size=200)
yn, sc, eps = inj.inject_verbose(y, 0.6)
assert np.allclose(yn, y + eps), "epsilon does not reconstruct the noisy label"
# and it must reproduce the old draw exactly
a = NoiseInjectorRegression(strategy='quantile', random_state=42).inject(y, 0.6)
assert np.array_equal(a, yn), "inject_verbose does not match inject"
print("OK: noise_scale + inject_verbose present and consistent")
PY
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
echo "############ 4b. which (dataset, strategy) arms are DEGENERATE"
python - <<'PY'
import sys, importlib.util, numpy as np
sys.path.insert(0, '/data/stat-cadd/scat9264/NoiseInject')
spec = importlib.util.spec_from_file_location('p', 'alternative_data_noise_robustness.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from noiseInject import NoiseInjectorRegression as NI

DATASETS = ['logd', 'caco2', 'herg_ki']
REPS = ['ECFP4', 'PDV', 'SNS', 'MHG-GNN-pretrained']
STRATS = m.STRATEGIES

labels = {}
df = m.download_openadmet()
c = next((c for c in df.columns if 'LogD' in c), None)
labels['logd'] = m.load_openadmet_endpoint(df, c, log_transform=False)[1]
c = next((c for c in df.columns if 'Caco' in c and 'Efflux' in c), None)
labels['caco2'] = m.load_openadmet_endpoint(df, c, log_transform=True)[1]
labels['herg_ki'] = m.load_chembl_herg()[1]

print("  distinct noise scales per (dataset, strategy) -- 1 means the arm is CONSTANT")
print(f"  {'dataset':9s} " + " ".join(f"{s:>10s}" for s in STRATS))
degenerate = []
for ds in DATASETS:
    y = labels[ds]
    row = []
    for st in STRATS:
        n = len(np.unique(np.round(NI(strategy=st, random_state=0).noise_scale(y, 1.0), 12)))
        row.append(f"{n:>10d}")
        if n == 1 and st != 'legacy':
            degenerate.append((ds, st))
    print(f"  {ds:9s} " + " ".join(row))

print()
print("  'legacy' is constant everywhere by design -- Gaussian gives every molecule the")
print("  same dose. That arm answers question A, not question B.")
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
echo "############ 4c. heteroscedastic vs value-proportional are ONE arm"
python - <<'PY'
import sys, numpy as np
sys.path.insert(0, '/data/stat-cadd/scat9264/NoiseInject')
from noiseInject import NoiseInjectorRegression as NI
from scipy.stats import spearmanr
y = np.random.RandomState(0).normal(2.0, 1.2, 3000)
a = NI(strategy='hetero', random_state=0).noise_scale(y, 1.0)
b = NI(strategy='valprop', random_state=0).noise_scale(y, 1.0)
r = spearmanr(a, b).correlation
print(f"  Spearman(hetero scale, valprop scale) = {r:.8f}")
print("  Both are strictly increasing in |y|, so they rank molecules IDENTICALLY.")
print("  For the uncertainty questions they are ONE arm -- never cite them as")
print("  independent replication. Dropping one reclaims 84 tasks:")
print("    python generate_scripts.py --drop-strategies hetero")
PY

echo
echo "############ 5. one real task, smallest dataset, 2 sigmas — end to end"
OUT=$(mktemp -d)
python -u alternative_data_noise_robustness.py \
    --datasets herg_ki --models NGBoost --reps ECFP4 \
    --strategies outlier --sigmas 0.0 1.0 \
    --unc-strategies all --oof-folds 3 \
    --results-root "$OUT" || { echo "FAIL: end-to-end run errored"; exit 1; }

python - "$OUT" <<'PY'
import sys, glob, pandas as pd
root = sys.argv[1]
f = glob.glob(f"{root}/**/*_uncertainty_values.csv", recursive=True)
if not f:
    print("FAIL: no uncertainty file written"); sys.exit(1)
d = pd.read_csv(f[0])
need = {'split','strategy','sigma','fold','uncertainty','noise_scale',
        'noise_pattern','injected_noise','oof_folds_ok'}
miss = need - set(d.columns)
if miss: print("FAIL: missing columns", miss); sys.exit(1)
print(f"  OK    {f[0].split('/')[-1]}: {len(d)} rows")
print(f"  splits    : {sorted(d['split'].unique())}")
print(f"  strategies: {sorted(d['strategy'].unique())}")
print(f"  folds     : {sorted(d['fold'].unique())}  (must be all 5, not just one)")
if 'train_oof' not in set(d['split']):
    print("FAIL: no out-of-fold training rows — question A cannot be answered"); sys.exit(1)
tr = d[d.split=='train_oof']
z = tr[tr.sigma==0.0]
if len(z) and z['injected_noise'].abs().max() != 0:
    print("FAIL: sigma=0 is not a clean control"); sys.exit(1)
print("  OK    sigma=0 control is exactly zero; out-of-fold training rows present")
pat = tr['noise_pattern']
if pat.nunique() <= 1 and d['strategy'].iloc[0] != 'legacy':
    print("FAIL: noise_pattern is constant - the question-B confound control is unusable"); sys.exit(1)
if tr['uncertainty'].notna().sum() == 0:
    print("FAIL: every out-of-fold uncertainty is blank"); sys.exit(1)
if (tr['oof_folds_ok'] < 5).any():
    print(f"WARN: some out-of-fold passes were truncated "
          f"(min {int(tr['oof_folds_ok'].min())}/5 inner folds)")
print(f"  OK    noise_pattern varies ({pat.nunique()} distinct); "
      f"{tr['uncertainty'].notna().sum()} finite out-of-fold uncertainties")
PY
status=$?
rm -rf "$OUT"
[ $status -ne 0 ] && exit 1

echo
echo "############ ALL PREFLIGHT CHECKS PASSED — safe to submit"
