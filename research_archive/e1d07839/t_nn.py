import numpy as np, sys
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests')
import alternative_data_noise_robustness as M
X=np.random.RandomState(0).normal(size=(300,20)).astype(np.float32); y=X[:,0]*2+np.random.RandomState(1).normal(0,.3,300)
Xv=X[:60]; yv=y[:60]; Xtr=X[60:]; ytr=y[60:]
Xt=np.random.RandomState(2).normal(size=(40,20)).astype(np.float32); yt=Xt[:,0]*2
for mt in ['full-bnn','full-vbll','mlp-full-bnn','mlp-full-vbll','deterministic']:
    p,u,ex=M.run_neural_experiment(Xtr,ytr,Xv,yv,Xt,yt,mt,'outlier',[0.0,0.6],oof_folds=5)
    ou=ex['oof_unc'].get(0.6)
    print(mt, "u0 None?",u.get(0.0) is None, "oof_unc nan:", (int(np.isnan(ou).sum()) if ou is not None else 'NA'), "/", len(ou) if ou is not None else 0,
          "oof_mean nan:", int(np.isnan(ex['oof_mean'][0.6]).sum()), flush=True)
