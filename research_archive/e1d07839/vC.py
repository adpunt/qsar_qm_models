import numpy as np, pandas as pd, sys
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from scipy.stats import spearmanr
from noiseInject.core import NoiseInjectorRegression as N
d=pd.read_csv('/Users/apunt/repos/KIRBy/tests/data_cache/chembl_herg_ki.csv')
mols=[Chem.MolFromSmiles(s) for s in d['SMILES']]
k=[i for i,m in enumerate(mols) if m is not None]; d=d.iloc[k].reset_index(drop=True); mols=[mols[i] for i in k]
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
X=np.array([gen.GetFingerprintAsNumPy(m) for m in mols],dtype=np.float32)
y=d['pChEMBL'].values.astype(float)
groups=pd.factorize(pd.Series([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]))[0]
tr,te=next(iter(GroupKFold(5).split(X,y,groups)))
ytr,yte=y[tr],y[te]
def unc(yn):
    rf=RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=-1).fit(X[tr],yn)
    P=np.stack([t.predict(X[te]) for t in rf.estimators_]); return P.std(0)
u0=unc(ytr)
print(f"{'strategy':10s} {'RMSdose@0.6':>11s} {'rho@0.6':>9s} {'rho@0':>8s} {'DELTA':>8s}")
for st in ['hetero','valprop']:
    inj=N(st,random_state=42); yn,sc,eps=inj.inject_verbose(ytr,0.6)
    tgt=inj.noise_scale(yte,1.0,reference=ytr)
    u=unc(yn)
    r=spearmanr(u,tgt).correlation; r0=spearmanr(u0,tgt).correlation
    print(f"{st:10s} {np.sqrt(np.mean(eps**2)):11.3f} {r:+9.4f} {r0:+8.4f} {r-r0:+8.4f}")
