import numpy as np, sys, random
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests')
import alternative_data_noise_robustness as M
from ngboost import NGBRegressor as _NGB
def fast(**kw):
    kw['n_estimators']=25; kw['verbose']=False
    return _NGB(**kw)          # real class -> str(type(mdl)) still NGBRegressor
M.NGBRegressor = fast
from pathlib import Path
from rdkit import Chem
rng=random.Random(0)
frag=['C','N','O','CC','CO','CN','c1ccccc1','C1CCCCC1','CCO','C(=O)O','CF','CCl','CBr','C#N','CS']
sm=[];seen=set()
while len(sm)<300:
    s=''.join(rng.choice(frag) for _ in range(rng.randint(2,5)))
    m=Chem.MolFromSmiles(s)
    if m is None: continue
    cs=Chem.MolToSmiles(m)
    if cs in seen: continue
    seen.add(cs); sm.append(cs)
smiles=np.array(sm); labels=np.random.RandomState(1).normal(3,1.2,size=len(sm))
out=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/e2e_ngb2')
M.run_dataset('TINY', smiles, labels, out, model_filter=['NGBoost'], rep_filter=['PDV'],
              sigma_levels=[0.0,0.6], unc_strategies='all', oof_folds=5, strategies=['quantile'])
import pandas as pd
f=sorted(out.glob('*uncertainty_values.csv')); print("FILES:",[x.name for x in f])
d=pd.read_csv(f[0]); print("cols:",list(d.columns))
print(d.groupby(['split','sigma']).size())
print("folds:",sorted(d.fold.unique()),"strategies:",d.strategy.unique())
t=d[(d.split=='train_oof')]
print("oof unc nan:", t.uncertainty.isna().sum(),"/",len(t))
print("per-fold train rows:", t.groupby(['fold','sigma']).size().to_dict())
