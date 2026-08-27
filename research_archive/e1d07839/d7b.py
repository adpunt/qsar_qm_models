import sys, os, numpy as np
os.chdir('/Users/apunt/repos/KIRBy/tests')
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests'); sys.path.insert(0,'/Users/apunt/repos/KIRBy/src'); sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
import importlib.util
spec = importlib.util.spec_from_file_location("pipe","/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py")
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
from sklearn.ensemble import RandomForestRegressor

S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/'
y=np.load(S+'herg.npy')
rng=np.random.RandomState(0)
n=len(y); perm=rng.permutation(n)
tr,va,te = perm[:800], perm[800:1000], perm[1000:]
y_tr,y_va,y_te = y[tr],y[va],y[te]
X=rng.randn(n,8); X_tr,X_va,X_te=X[tr],X[va],X[te]

q=0.1
sp={'high_threshold':float(np.quantile(y,1-q)),'low_threshold':float(np.quantile(y,q))}
print("sp =",sp)
sig=[0.0,0.5]

# ---- TREE PATH ----
for label, spx in [("NO --threshold-quantile", None), ("WITH --threshold-quantile 0.1", sp)]:
    p,u,ex = pipe.run_tree_experiment(X_tr,y_tr,X_te,y_te,
        lambda: RandomForestRegressor(n_estimators=5,random_state=0), 'threshold', sig,
        oof_folds=0, strategy_params=spx)
    tp=ex['test_noise_pattern'][0.5]; ts=ex['test_noise_scale'][0.5]; trp=ex['train_noise_pattern'][0.5]; trs=ex['train_noise_scale'][0.5]
    print(f"\nTREE {label}")
    print("  train_noise_pattern uniq", np.unique(trp), " train_noise_scale(0.5) uniq", np.unique(trs))
    print("  test_noise_pattern  uniq", np.unique(tp),  " test_noise_scale(0.5) uniq", np.unique(ts))

# ---- NEURAL PATH: capture y_val_used ----
cap={}
orig = pipe.train_neural_regression
def fake(Xtr,ytr,Xva,yva,Xte, model_type=None, epochs=None):
    cap.setdefault('val',[]).append(np.array(yva))
    cap.setdefault('trn',[]).append(np.array(ytr))
    return np.zeros(len(Xte)), np.ones(len(Xte))
pipe.train_neural_regression = fake
for label, spx in [("NO --threshold-quantile", None), ("WITH --threshold-quantile 0.1", sp)]:
    cap.clear()
    p,u,ex = pipe.run_neural_experiment(X_tr,y_tr,X_va,y_va,X_te,y_te,'DNN','threshold',sig,
        oof_folds=0, noise_validation=True, strategy_params=spx)
    yv_noisy = cap['val'][1]          # sigma=0.5 call
    yv_eps = yv_noisy - y_va
    # infer applied sigma_i on validation from |eps| grouped by which side of cut
    hi = sp['high_threshold']; lo = sp['low_threshold']
    grpQ = np.where((y_va>=hi)|(y_va<=lo), 'extreme','mid')
    grpA = np.where((y_va>=1.0)|(y_va<=-1.0), 'extreme','mid')
    print(f"\nNEURAL {label}  (sigma=0.5)")
    for gname,g in [('quantile-cutpoints',grpQ),('absolute-cutpoints',grpA)]:
        for lev in ['extreme','mid']:
            m=g==lev
            if m.sum(): print(f"   {gname:20s} {lev:8s} n={m.sum():4d} RMS(eps)={np.sqrt(np.mean(yv_eps[m]**2)):.4f}  (expect 1.0 if 2*sigma, 0.05 if 0.1*sigma)")
    # exact check: reproduce with an independent injector using sp
    ind = pipe.NoiseInjectorRegression(strategy='threshold', random_state=1337)
    _ = None
    exp = ind.inject(y_va, 0.5, **(spx or {}))  # sigma=0.5 is 2nd draw? no: fresh rng -> first draw
    # the pipeline's val injector draws only at sigma>0, so sigma=0.5 is its FIRST draw
    print("   val labels bit-identical to independent injector(seed1337, sp) :", np.allclose(yv_noisy, exp, atol=0, rtol=0), " max|diff|=", np.max(np.abs(yv_noisy-exp)))
pipe.train_neural_regression = orig
