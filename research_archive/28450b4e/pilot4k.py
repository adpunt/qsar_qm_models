"""PILOT of the real experiment: do the proposed noise arms produce different
model behaviour at MATCHED dose?

Real task: QM9 HOMO-LUMO gap from RDKit physicochemical descriptors.
Real protocol: scaffold split, noise on TRAIN ONLY, evaluate on CLEAN test labels.

If the arms all give the same curve, the experiment is not worth running.
If they separate, it is.
"""
import numpy as np, pandas as pd, sys, time
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')
rng = np.random.default_rng(11)
N_MOL = 4000
t0 = time.time()

# ---------------------------------------------------------- load molecules
supp = Chem.SDMolSupplier('data/QM9/raw/gdb9.sdf', removeHs=False, sanitize=True)
props = pd.read_csv('data/QM9/raw/gdb9.sdf.csv')
gap_by_id = dict(zip(props.mol_id, props.gap * 27.211386))

mols, ys, scaffs = [], [], []
for i, m in enumerate(supp):
    if m is None:
        continue
    name = m.GetProp('_Name') if m.HasProp('_Name') else None
    if name not in gap_by_id:
        continue
    try:
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        continue
    mols.append(m); ys.append(gap_by_id[name]); scaffs.append(sc)
    if len(mols) >= N_MOL:
        break
print(f"loaded {len(mols)} molecules in {time.time()-t0:.0f}s", flush=True)

y = np.array(ys)
names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
X = np.array([calc.CalcDescriptors(m) for m in mols], dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
keep = X.std(axis=0) > 0
X = X[:, keep]
print(f"descriptors: {X.shape}  gap SD = {y.std():.4f} eV   ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------- scaffold split 80/20
uniq = pd.Series(scaffs).value_counts().index.tolist()
rng.shuffle(uniq)
test_sc, n_t = set(), 0
for s in uniq:
    if n_t >= 0.2 * len(y):
        break
    test_sc.add(s); n_t += scaffs.count(s) if False else 0
# faster: build group sizes once
sizes = pd.Series(scaffs).value_counts().to_dict()
test_sc, n_t = set(), 0
for s in uniq:
    if n_t >= 0.2 * len(y):
        break
    test_sc.add(s); n_t += sizes[s]
is_test = np.array([s in test_sc for s in scaffs])
Xtr, Xte = X[~is_test], X[is_test]
ytr, yte = y[~is_test], y[is_test]
sc_tr = [s for s, t in zip(scaffs, is_test) if not t]
print(f"scaffold split: train {len(ytr)}  test {len(yte)}", flush=True)

sd_tr = ytr.std()

# ---------------------------------------------------------- noise arms
def gaussian(y, tgt, rng):
    return rng.normal(0, tgt, len(y))

def student_t(y, tgt, nu, rng):
    return rng.standard_t(nu, len(y)) * tgt * np.sqrt((nu - 2) / nu)

def grouped(y, tgt, lam, fgrp, groups, rng):
    g = np.array(groups)
    uq = np.unique(g)
    bad = set(rng.choice(uq, max(1, int(round(fgrp * len(uq)))), replace=False).tolist())
    isbad = np.array([x in bad for x in g])
    f = isbad.mean()
    s0 = tgt / np.sqrt(1 - f + f * lam**2)
    return rng.normal(0, 1, len(y)) * np.where(isbad, lam * s0, s0)

def contaminated(y, tgt, p, lam, rng):
    s0 = tgt / np.sqrt(1 + p * (lam**2 - 1))
    hit = rng.random(len(y)) < p
    return rng.normal(0, 1, len(y)) * np.where(hit, lam * s0, s0)

def laplace(y, tgt, rng):
    b = tgt/np.sqrt(2)
    return rng.laplace(0, b, len(y))

def censor(y, p_hi):
    c = np.quantile(y, 1-p_hi)
    return np.minimum(y, c) - y

ARMS = {
    'A Gaussian':          lambda t, r: gaussian(ytr, t, r),
    'B Student-t nu=10':   lambda t, r: student_t(ytr, t, 10, r),
    'B Student-t nu=5':    lambda t, r: student_t(ytr, t, 5, r),
    'B Student-t nu=3':    lambda t, r: student_t(ytr, t, 3, r),
    'C Laplace':           lambda t, r: laplace(ytr, t, r),
    'D Grouped lam=3':     lambda t, r: grouped(ytr, t, 3.0, 0.2, sc_tr, r),
    'E Outlier p=0.05':    lambda t, r: contaminated(ytr, t, 0.05, 3.0, r),
    'E Outlier p=0.10':    lambda t, r: contaminated(ytr, t, 0.10, 3.0, r),
}
CENSOR_FRACS = [0.10, 0.25, 0.40]

def models():
    return {
        'RF':    RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=0),
        'LGBM':  lgb.LGBMRegressor(n_estimators=300, verbose=-1, random_state=0),
        'Ridge': Ridge(alpha=1.0),
    }

scaler = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

# baseline
print("\n--- baseline (clean labels) ---", flush=True)
base = {}
for mn, mdl in models().items():
    Xa, Xb = (Xtr_s, Xte_s) if mn == 'Ridge' else (Xtr, Xte)
    mdl.fit(Xa, ytr)
    base[mn] = r2_score(yte, mdl.predict(Xb))
    print(f"  {mn:6s} R2 = {base[mn]:.4f}", flush=True)

# ---------------------------------------------------------- run arms
REPS = 3
rows = []
for k in [0.25, 0.5, 1.0]:
    tgt = k * sd_tr
    for arm, fn in ARMS.items():
        for rep in range(REPS):
            r = np.random.default_rng(1000 + rep)
            noise = fn(tgt, r)
            ynoisy = ytr + noise
            dose = np.sqrt(np.mean(noise**2)) / sd_tr
            for mn, mdl in models().items():
                Xa, Xb = (Xtr_s, Xte_s) if mn == 'Ridge' else (Xtr, Xte)
                mdl.fit(Xa, ynoisy)
                rows.append(dict(k=k, arm=arm, rep=rep, model=mn, dose=dose,
                                 r2=r2_score(yte, mdl.predict(Xb)), base=base[mn]))
        print(f"  k={k} {arm:20s} done ({time.time()-t0:.0f}s)", flush=True)

# Censoring: not dose-matched, swept on its own axis
for pc in CENSOR_FRACS:
    noise = censor(ytr, pc)
    ynoisy = ytr + noise
    dose = np.sqrt(np.mean(noise**2))/sd_tr
    for mn, mdl in models().items():
        Xa, Xb = (Xtr_s, Xte_s) if mn=='Ridge' else (Xtr, Xte)
        mdl.fit(Xa, ynoisy)
        rows.append(dict(k=np.nan, arm=f'F Censor {int(pc*100)}%', rep=0, model=mn,
                         dose=dose, r2=r2_score(yte, mdl.predict(Xb)), base=base[mn]))
    print(f"  censor {pc} done ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
df['ret'] = df.r2 / df.base
out = '/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/28450b4e-4a9e-4197-993c-548e1dd48b09/scratchpad/pilot4k_results.csv'
df.to_csv(out, index=False)

print("\n" + "=" * 100)
print("R2 on CLEAN test labels, mean over 3 reps. All arms at MATCHED dose.")
print("=" * 100)
for k in sorted(df.k.unique()):
    sub = df[df.k == k]
    print(f"\n--- noise = {k} x label SD  (realised dose {sub.dose.mean():.3f}) ---")
    piv = sub.pivot_table(index='arm', columns='model', values='r2', aggfunc='mean')
    sdv = sub.pivot_table(index='arm', columns='model', values='r2', aggfunc='std')
    for arm in piv.index:
        cells = "  ".join(f"{m}={piv.loc[arm,m]:.4f}+-{sdv.loc[arm,m]:.4f}" for m in piv.columns)
        print(f"  {arm:20s} {cells}")
    print(f"  {'SPREAD across arms':20s} " +
          "  ".join(f"{m}={piv[m].max()-piv[m].min():.4f}" for m in piv.columns))
print(f"\nsaved {out}   total {time.time()-t0:.0f}s")
