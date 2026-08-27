import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

# ---- real molecules, real scaffolds -------------------------------------
df = pd.read_csv('/Users/apunt/repos/KIRBy/data/herg.tab', sep='\t')
smis = df['Drug'].tolist()
gen = AllChem.GetMorganFingerprintAsBitVect
X, scaf, keep = [], [], []
for s in smis:
    m = Chem.MolFromSmiles(s)
    if m is None: continue
    fp = gen(m, 2, nBits=1024)
    X.append(np.array(fp)); keep.append(s)
    try: scaf.append(MurckoScaffold.MurckoScaffoldSmiles(mol=m))
    except Exception: scaf.append('')
X = np.array(X, float); scaf = np.array(scaf)
print(f"n={len(X)} molecules, {len(set(scaf))} Murcko scaffolds")

# synthetic but chemically-structured label
rs = np.random.RandomState(0)
w = rs.randn(X.shape[1]) * 0.15
y = X @ w; y = (y - y.mean())/y.std() * 1.2 + 3.0   # logD-like, ~[0,6]
n = len(y)

# ---------- EXACT reimplementation of the shipped _oof_predict ------------
def oof_predict(X, y_noisy, n_folds, seed=42, groups=None):
    oof_m = np.full(n, np.nan); oof_u = np.full(n, np.nan)
    if groups is None:
        order = np.random.RandomState(seed).permutation(n)      # <-- as shipped
        folds = np.array_split(order, n_folds)
    else:                                                        # scaffold variant
        from sklearn.model_selection import GroupKFold
        folds = [te for _, te in GroupKFold(n_folds).split(X, y_noisy, groups)]
    for held in folds:
        keep_i = np.setdiff1d(np.arange(n), held)
        m = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)
        m.fit(X[keep_i], y_noisy[keep_i])
        per_tree = np.stack([t.predict(X[held]) for t in m.estimators_])
        q16,q50,q84 = np.percentile(per_tree,[16,50,84],axis=0)
        oof_m[held]=q50; oof_u[held]=(q84-q16)/2
    return oof_m, oof_u

SIG = 0.6
inj = NoiseInjectorRegression('legacy', 42)
y_noisy, scale, eps = inj.inject_verbose(y, SIG)
print(f"\nGaussian (legacy) at sigma={SIG}: noise_scale is constant = {np.unique(scale)}")

for label, g in [('RANDOM OOF (as shipped)', None), ('SCAFFOLD OOF (paper standard)', scaf)]:
    m,u = oof_predict(X, y_noisy, 5, groups=g)
    r_eps  = spearmanr(u, np.abs(eps)).correlation
    r_res  = spearmanr(np.abs(y_noisy-m), np.abs(eps)).correlation
    rmse   = np.sqrt(np.mean((m-y)**2))
    # permutation null for r_eps
    null=[spearmanr(u, np.abs(eps[np.random.RandomState(k).permutation(n)])).correlation for k in range(200)]
    lo,hi=np.percentile(null,[2.5,97.5])
    print(f"\n{label}")
    print(f"  OOF RMSE vs CLEAN y      = {rmse:.3f}   mean uncertainty = {u.mean():.3f}")
    print(f"  QUESTION A: spearman(uncertainty, |epsilon|) = {r_eps:+.4f}"
          f"   permutation null 95% CI [{lo:+.4f},{hi:+.4f}]  -> {'INSIDE NULL' if lo<r_eps<hi else 'outside'}")
    print(f"  TRAP:       spearman(|y_noisy - oof_pred|, |epsilon|) = {r_res:+.4f}   <- trivially large, contains eps by construction")
