"""RANGE-FINDING run: what noise levels should the real experiment use?

Fine grid around realistic assay-error levels, on QM9 N=4000.
Also finds the censoring knee, which the coarse run bracketed only between 25% and 40%.

Output feeds directly into the experimental design: we pick levels where the
curve actually moves, not an arbitrary 0..1 ladder.
"""
import numpy as np, pandas as pd, time
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit.Chem import Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import lightgbm as lgb

RDLogger.DisableLog('rdApp.*')
N_MOL = 4000
t0 = time.time()

supp = Chem.SDMolSupplier('data/QM9/raw/gdb9.sdf', removeHs=False, sanitize=True)
props = pd.read_csv('data/QM9/raw/gdb9.sdf.csv')
gap = dict(zip(props.mol_id, props.gap * 27.211386))
mols, ys, scaffs = [], [], []
for m in supp:
    if m is None: continue
    nm = m.GetProp('_Name') if m.HasProp('_Name') else None
    if nm not in gap: continue
    try: sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception: continue
    mols.append(m); ys.append(gap[nm]); scaffs.append(sc)
    if len(mols) >= N_MOL: break
y = np.array(ys)
names = [d[0] for d in Descriptors._descList]
calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
X = np.array([calc.CalcDescriptors(m) for m in mols], dtype=float)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = X[:, X.std(axis=0) > 0]
print(f"loaded {len(y)} mols, {X.shape[1]} descriptors, label SD={y.std():.4f} eV ({time.time()-t0:.0f}s)", flush=True)

rng = np.random.default_rng(11)
sizes = pd.Series(scaffs).value_counts().to_dict()
uniq = list(sizes); rng.shuffle(uniq)
test_sc, n_t = set(), 0
for s in uniq:
    if n_t >= 0.2*len(y): break
    test_sc.add(s); n_t += sizes[s]
is_test = np.array([s in test_sc for s in scaffs])
Xtr, Xte, ytr, yte = X[~is_test], X[is_test], y[~is_test], y[is_test]
sc_tr = [s for s,t in zip(scaffs, is_test) if not t]
sd = ytr.std()
scaler = StandardScaler().fit(Xtr); Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
print(f"train {len(ytr)} test {len(yte)}  train label SD={sd:.4f}", flush=True)

def models():
    return {'RF': RandomForestRegressor(n_estimators=150, n_jobs=-1, random_state=0),
            'LGBM': lgb.LGBMRegressor(n_estimators=300, verbose=-1, random_state=0),
            'Ridge': Ridge(alpha=1.0)}

base = {}
for mn, md in models().items():
    Xa, Xb = (Xtr_s, Xte_s) if mn=='Ridge' else (Xtr, Xte)
    md.fit(Xa, ytr); base[mn] = r2_score(yte, md.predict(Xb))
print("clean baseline:", {k: round(v,4) for k,v in base.items()}, flush=True)

def gauss(t, r): return r.normal(0, t, len(ytr))
def tdist(t, nu, r): return r.standard_t(nu, len(ytr)) * t*np.sqrt((nu-2)/nu)
def grouped(t, lam, f, r):
    g = np.array(sc_tr); u = np.unique(g)
    bad = set(r.choice(u, max(1,int(round(f*len(u)))), replace=False).tolist())
    ib = np.array([x in bad for x in g]); fr = ib.mean()
    s0 = t/np.sqrt(1-fr+fr*lam**2)
    return r.normal(0,1,len(ytr))*np.where(ib, lam*s0, s0)
def outlier(t, p, lam, r):
    s0 = t/np.sqrt(1+p*(lam**2-1)); hit = r.random(len(ytr))<p
    return r.normal(0,1,len(ytr))*np.where(hit, lam*s0, s0)

TYPES = {'Gaussian':      lambda t,r: gauss(t,r),
         'Student-t nu=3':lambda t,r: tdist(t,3,r),
         'Grouped':       lambda t,r: grouped(t,3.0,0.2,r),
         'Outlier p=0.05':lambda t,r: outlier(t,0.05,3.0,r)}

KS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
rows = []
for k in KS:
    t = k*sd
    for name, fn in TYPES.items():
        for rep in range(3):
            r = np.random.default_rng(500+rep)
            n = fn(t, r); yn = ytr + n
            for mn, md in models().items():
                Xa, Xb = (Xtr_s, Xte_s) if mn=='Ridge' else (Xtr, Xte)
                md.fit(Xa, yn)
                rows.append(dict(kind='noise', level=k, type=name, rep=rep, model=mn,
                                 dose=np.sqrt(np.mean(n**2))/sd,
                                 r2=r2_score(yte, md.predict(Xb)), base=base[mn]))
    print(f"  k={k} done ({time.time()-t0:.0f}s)", flush=True)

# censoring knee
for pc in [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50]:
    cut = np.quantile(ytr, 1-pc); n = np.minimum(ytr,cut)-ytr; yn = ytr+n
    for mn, md in models().items():
        Xa, Xb = (Xtr_s, Xte_s) if mn=='Ridge' else (Xtr, Xte)
        md.fit(Xa, yn)
        rows.append(dict(kind='censor', level=pc, type='Censoring', rep=0, model=mn,
                         dose=np.sqrt(np.mean(n**2))/sd,
                         r2=r2_score(yte, md.predict(Xb)), base=base[mn]))
print(f"  censoring done ({time.time()-t0:.0f}s)", flush=True)

df = pd.DataFrame(rows)
out='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/28450b4e-4a9e-4197-993c-548e1dd48b09/scratchpad/ranges_results.csv'
df.to_csv(out, index=False)

print("\n=== NOISE: R2 retained (mean of 3 reps) ===")
p = df[df.kind=='noise'].pivot_table(index='level', columns=['model','type'], values='r2')
for k in KS:
    cells = "  ".join(f"{m}:{df[(df.kind=='noise')&(df.level==k)&(df.model==m)].r2.mean():.3f}" for m in ['LGBM','RF','Ridge'])
    spread = df[(df.kind=='noise')&(df.level==k)].groupby(['model','type']).r2.mean().groupby('model').agg(lambda v:v.max()-v.min()).mean()
    print(f"  k={k:<5} {cells}   spread across noise types={spread:.4f}")

print("\n=== CENSORING: finding the knee ===")
for pc in [0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50]:
    s = df[(df.kind=='censor')&(df.level==pc)]
    cells = "  ".join(f"{r.model}:{r.r2:.3f}" for _,r in s.iterrows())
    print(f"  censor {int(pc*100):>2}%  dose={s.dose.iloc[0]:.3f}   {cells}")
print(f"\nsaved {out}  total {time.time()-t0:.0f}s")
