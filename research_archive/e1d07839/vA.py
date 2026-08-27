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
k=[i for i,m in enumerate(mols) if m is not None]
d=d.iloc[k].reset_index(drop=True); mols=[mols[i] for i in k]
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
X=np.array([gen.GetFingerprintAsNumPy(m) for m in mols],dtype=np.float32)
y=d['pChEMBL'].values.astype(float)
scaf=[MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
groups=pd.factorize(scaf)[0]
print("n=",len(y),"scaffolds=",len(set(scaf)),flush=True)
inj=N('legacy',random_state=42); yn,sc,eps=inj.inject_verbose(y,0.6)
print("legacy scale unique:",np.unique(sc),flush=True)
def oof(folds):
    om=np.full(len(y),np.nan); ou=np.full(len(y),np.nan)
    for held in folds:
        ki=np.setdiff1d(np.arange(len(y)),held)
        rf=RandomForestRegressor(n_estimators=150,random_state=0,n_jobs=-1).fit(X[ki],yn[ki])
        P=np.stack([t.predict(X[held]) for t in rf.estimators_])
        om[held]=P.mean(0); ou[held]=P.std(0)
    return om,ou
rand=np.array_split(np.random.RandomState(42).permutation(len(y)),5)
scafF=[te for _,te in GroupKFold(5).split(X,yn,groups)]
for nm,f in [('RANDOM(shipped)',rand),('SCAFFOLD',scafF)]:
    om,ou=oof(f)
    r=spearmanr(ou,np.abs(eps)).correlation
    null=[spearmanr(ou,np.random.RandomState(kk).permutation(np.abs(eps))).correlation for kk in range(200)]
    lo,hi=np.percentile(null,[2.5,97.5])
    print(f"{nm}: rho(unc,|eps|)={r:+.4f} null95=[{lo:+.4f},{hi:+.4f}] {'INSIDE' if lo<r<hi else 'OUTSIDE'}")
    print(f"  OOF RMSE vs clean={np.sqrt(np.nanmean((om-y)**2)):.4f} mean_unc={np.nanmean(ou):.4f}")
    print(f"  TRAP rho(|y_noisy-oofpred|,|eps|)={spearmanr(np.abs(yn-om),np.abs(eps)).correlation:+.4f}")
    print(f"  rho(|y_clean-oofpred|,|eps|)={spearmanr(np.abs(y-om),np.abs(eps)).correlation:+.4f}",flush=True)
