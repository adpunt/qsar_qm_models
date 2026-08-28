import sys, types, numpy as np
from sklearn.ensemble import RandomForestRegressor
class RandomForestQuantileRegressor:
    def __init__(self,**kw): self.rf=RandomForestRegressor(n_estimators=kw.get('n_estimators',100),random_state=kw.get('random_state',42),n_jobs=-1)
    def fit(self,X,y): self.rf.fit(X,y); return self
    def predict(self,X,quantiles=None):
        P=np.stack([t.predict(X) for t in self.rf.estimators_])
        if quantiles is None: return P.mean(0)
        return np.stack([np.quantile(P,q,axis=0) for q in quantiles],axis=1)
m=types.ModuleType('quantile_forest'); m.RandomForestQuantileRegressor=RandomForestQuantileRegressor
sys.modules['quantile_forest']=m
sys.argv=['x','--datasets','herg_ki','--models','QRF','--reps','ECFP4','--strategies','valprop',
          '--unc-strategies','all','--oof-folds','2','--sigmas','0.0','0.6',
          '--results-root','/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/verif_run3']
import importlib.util
spec=importlib.util.spec_from_file_location('__main__x','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
mod.main()
