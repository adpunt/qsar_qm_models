import sys; sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
import importlib.util, numpy as np
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
from sklearn.ensemble import RandomForestRegressor
rs=np.random.RandomState(0); Xtr=rs.normal(size=(200,8)); Xte=rs.normal(size=(50,8)); w=rs.normal(size=8)
ytr=Xtr@w+7; yte=Xte@w+7
# oof_folds=0
p,u,ex=m.run_tree_experiment(Xtr,ytr,Xte,yte,lambda: RandomForestRegressor(n_estimators=10,random_state=42),'quantile',[0.0,0.5],oof_folds=0)
print("oof=0 -> oof_mean dict:",ex['oof_mean'],"truthy:",bool(ex['oof_unc']),
      "| train_noise_scale keys:",sorted(ex['train_noise_scale']))
# oof_folds=5 with a None-uncertainty model (what happens if UNCERTAINTY_MODELS ever drifts)
p,u,ex=m.run_tree_experiment(Xtr,ytr,Xte,yte,lambda: RandomForestRegressor(n_estimators=10,random_state=42),'quantile',[0.0,0.5],oof_folds=5)
print("oof=5 None-unc model -> u[0.0]:",u[0.0]," oof_unc all-NaN:",np.all(np.isnan(ex['oof_unc'][0.5])),
      " dict truthy:",bool(ex['oof_unc']))
# oof_folds=1 (the >1 guard)
p,u,ex=m.run_tree_experiment(Xtr,ytr,Xte,yte,lambda: RandomForestRegressor(n_estimators=10,random_state=42),'quantile',[0.5],oof_folds=1)
print("oof=1 -> oof dicts empty:",ex['oof_unc']=={} )
# all-folds-fail case
def bad(): raise RuntimeError('boom')
class B:
    def fit(self,X,y): raise RuntimeError('boom')
p,u,ex=m.run_tree_experiment(Xtr,ytr,Xte,yte,lambda: RandomForestRegressor(n_estimators=10,random_state=42),'quantile',[0.5],oof_folds=5)
