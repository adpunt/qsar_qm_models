import sys; sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
import importlib.util, numpy as np
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

rs=np.random.RandomState(0)
n_tr, n_te, d = 300, 80, 12
Xtr=rs.normal(size=(n_tr,d)); Xte=rs.normal(size=(n_te,d))
w=rs.normal(size=d)
ytr=Xtr@w+rs.normal(0,.3,n_tr)+7.0
yte=Xte@w+rs.normal(0,.3,n_te)+7.0

from sklearn.ensemble import RandomForestRegressor
try:
    from quantile_forest import RandomForestQuantileRegressor as QRF
    HAS=True
except Exception as e:
    print('no qrf',e); HAS=False

# --- 1) oof index coverage / leakage check
seen=[]
def fp(Xf,yf,Xs):
    return np.zeros(len(Xs)), np.ones(len(Xs))
# instrument
orig=m._oof_predict
n=n_tr
order=np.random.RandomState(42).permutation(n)
folds=np.array_split(order,5)
allheld=np.concatenate(folds)
print("oof: all indices held exactly once:", np.array_equal(np.sort(allheld), np.arange(n)))
for held in folds:
    keep=np.setdiff1d(order,held,assume_unique=False)
    assert len(np.intersect1d(keep,held))==0
    print("  fold size", len(held), "keep", len(keep), "overlap", len(np.intersect1d(keep,held)))

# --- 2) run_tree_experiment with QRF + oof
if HAS:
    for strat in ['legacy','quantile','threshold','outlier','hetero','valprop']:
        p,u,ex = m.run_tree_experiment(Xtr,ytr,Xte,yte,
            lambda: QRF(n_estimators=30,random_state=42,n_jobs=1),
            strat,[0.0,0.5],oof_folds=5)
        for sig in [0.0,0.5]:
            om=ex['oof_mean'][sig]; ou=ex['oof_unc'][sig]
            ts=ex['test_noise_scale'][sig]; tp=ex['test_noise_pattern'][sig]
            trs=ex['train_noise_scale'][sig]; eps=ex['train_epsilon'][sig]
            print(f"{strat:10s} sig={sig}: oof nan={np.isnan(om).sum()}/{len(om)} unc nan={np.isnan(ou).sum()} "
                  f"| te_scale uniq={len(np.unique(ts))} te_pat uniq={len(np.unique(tp))} "
                  f"| tr_scale uniq={len(np.unique(trs))} eps_rms={np.sqrt((eps**2).mean()):.4f} "
                  f"| lens {len(ts)}=={n_te} {len(trs)}=={n_tr}")

# --- 3) model with no uncertainty under oof
p,u,ex = m.run_tree_experiment(Xtr,ytr,Xte,yte,
    lambda: RandomForestRegressor(n_estimators=20,random_state=42,n_jobs=1),
    'legacy',[0.0,0.5],oof_folds=5)
print("RF: uncertainties[0.0] is", u[0.0], "oof_unc all-nan:", np.all(np.isnan(ex['oof_unc'][0.5])),
      "oof_unc dict truthy:", bool(ex['oof_unc']))
