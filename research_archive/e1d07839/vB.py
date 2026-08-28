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
groups=pd.factorize(pd.Series([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]))[0]
tr,te=next(iter(GroupKFold(5).split(X,y,groups)))
ytr,yte=y[tr],y[te]
rf=RandomForestRegressor(n_estimators=200,random_state=0,n_jobs=-1).fit(X[tr],ytr)   # SIGMA=0, clean labels
P=np.stack([t.predict(X[te]) for t in rf.estimators_]); pred=P.mean(0); unc=P.std(0)
print("hERG fold0  n_test=",len(te)," corr(y,ypred)=",round(np.corrcoef(yte,pred)[0,1],3))
print(f"{'strategy':10s} {'rho(REAL unc,target)':>22s} {'rho(SHAM h(ypred),target)':>26s}")
for st in ['legacy','outlier','quantile','hetero','threshold','valprop']:
    inj=N(st,random_state=42)
    tgt=inj.noise_scale(yte,1.0,reference=ytr)
    sham=inj.noise_scale(pred,1.0,reference=ytr)
    if np.std(tgt)==0:
        print(f"{st:10s} {'TARGET CONSTANT -> nan':>22s} uniq={np.unique(tgt)}"); continue
    r1=spearmanr(unc,tgt); r2=spearmanr(sham,tgt)
    print(f"{st:10s} {r1.correlation:+22.4f} {r2.correlation:+26.4f}   (p_real={r1.pvalue:.2g})")
