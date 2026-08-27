import sys, shutil, numpy as np, pandas as pd
from pathlib import Path
sys.argv=['x']
import importlib.util
from sklearn.ensemble import RandomForestRegressor
spec=importlib.util.spec_from_file_location('anr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

class QuantileShim:
    def __init__(self,**kw): self.f=RandomForestRegressor(n_estimators=30,random_state=42)
    def fit(self,X,y): self.f.fit(X,y); return self
    def predict(self,X,quantiles=None):
        P=np.stack([t.predict(X) for t in self.f.estimators_])
        return np.quantile(P,quantiles,axis=0).T
m.RandomForestQuantileRegressor=QuantileShim
m.HAS_QRF=True

rng=np.random.RandomState(0); N=300
smiles=np.array(['c1ccccc1'+'C'*(1+i%7) for i in range(N)])
labels=rng.normal(3.0,1.2,N)
def fake_reps(sl, rep_filter=None):
    X=np.random.RandomState(1).normal(0,1,(len(sl),16))
    X=np.hstack([X,(labels*0.7+np.random.RandomState(2).normal(0,0.5,len(sl))).reshape(-1,1)])
    return {name:X for name in (rep_filter or ['ECFP4'])}
m.generate_representations=fake_reps
m.assign_scaffold_groups=lambda s:(np.arange(len(s))%15, 15)
m.N_FOLDS=3

D=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/smokeout')
if D.exists(): shutil.rmtree(D)
RES=D/'results/uncertainty_rerun/qrf__logd__ecfp4__outlier'
m.run_dataset('OpenADMET-LogD', smiles, labels, RES/'logd',
              model_filter=['QRF'], rep_filter=['ECFP4'], sigma_levels=[0.0,0.5,1.0],
              unc_strategies='all', oof_folds=3, strategies=['outlier'])
print("\n== files ==")
for f in sorted((RES/'logd').glob('*.csv')): print("  ",f.name, sum(1 for _ in open(f))-1,"rows")
u=pd.read_csv(RES/'logd'/'QRF_ECFP4_uncertainty_values.csv')
print(u.groupby(['split','strategy']).agg(folds=('fold','nunique'),n=('fold','size'),sig=('sigma','nunique')))
print("cols:",list(u.columns))
print("test noise_scale nunique per sigma:", u[u.split=='test'].groupby('sigma')['noise_scale'].nunique().to_dict())
print("oof injected_noise nonzero frac:", (u[u.split=='train_oof']['injected_noise']!=0).mean())
print("oof uncertainty NaN frac:", u[u.split=='train_oof']['uncertainty'].isna().mean())
print("oof y_pred NaN frac:", u[u.split=='train_oof']['y_pred'].isna().mean())
print("\n== now simulate a SECOND strategy writing into the SAME dir (what would happen without per-task dirs) ==")
m.run_dataset('OpenADMET-LogD', smiles, labels, RES/'logd',
              model_filter=['QRF'], rep_filter=['ECFP4'], sigma_levels=[0.0,0.5,1.0],
              unc_strategies='all', oof_folds=3, strategies=['hetero'])
u2=pd.read_csv(RES/'logd'/'QRF_ECFP4_uncertainty_values.csv')
print(u2.groupby(['split','strategy']).size())
ar=pd.read_csv(RES/'logd'/'all_results.csv'); print("all_results strategies:", ar.groupby('strategy').size().to_dict())
sm=pd.read_csv(RES/'logd'/'summary.csv'); print("summary rows:",len(sm), sm[['model','rep','strategy']].to_dict('records'))
