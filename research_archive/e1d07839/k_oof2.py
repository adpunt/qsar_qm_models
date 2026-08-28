import sys; sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
import importlib.util, numpy as np, pandas as pd
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from sklearn.ensemble import RandomForestRegressor

class QuantileStub:
    """type name contains 'Quantile' -> hits the QRF branch of _tree_mean_std"""
    def __init__(self): self.f=RandomForestRegressor(n_estimators=20,random_state=42,n_jobs=1)
    def fit(self,X,y): self.f.fit(X,y); return self
    def predict(self,X,quantiles=None):
        per=np.stack([t.predict(X) for t in self.f.estimators_])
        return np.quantile(per,quantiles,axis=0).T

rs=np.random.RandomState(0); n_tr,n_te,d=300,80,12
Xtr=rs.normal(size=(n_tr,d)); Xte=rs.normal(size=(n_te,d)); w=rs.normal(size=d)
ytr=Xtr@w+rs.normal(0,.3,n_tr)+7.0; yte=Xte@w+rs.normal(0,.3,n_te)+7.0

for strat in ['legacy','quantile','threshold','outlier','hetero','valprop']:
    p,u,ex=m.run_tree_experiment(Xtr,ytr,Xte,yte,QuantileStub,strat,[0.0,0.5,1.0],oof_folds=5)
    for sig in [0.0,0.5]:
        om=ex['oof_mean'][sig]; ou=ex['oof_unc'][sig]
        ts=ex['test_noise_scale'][sig]; tp=ex['test_noise_pattern'][sig]
        trs=ex['train_noise_scale'][sig]; eps=ex['train_epsilon'][sig]
        print(f"{strat:9s} s={sig}: oofNaN={np.isnan(om).sum()} uncNaN={np.isnan(ou).sum()} "
              f"te_scale_u={len(np.unique(ts))} te_pat_u={len(np.unique(tp))} tr_scale_u={len(np.unique(trs))} "
              f"eps_rms={np.sqrt((eps**2).mean()):.3f} len(ts)={len(ts)} len(trs)={len(trs)}")
    # is test_noise_scale == sigma * test_noise_pattern ?
    for sig in [0.5,1.0]:
        a=ex['test_noise_scale'][sig]; b=sig*ex['test_noise_pattern'][sig]
        print(f"    scale==sigma*pattern at {sig}: {np.allclose(a,b)}")
