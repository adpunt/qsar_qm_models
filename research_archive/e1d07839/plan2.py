import sys, os, numpy as np
os.chdir('/Users/apunt/repos/KIRBy/tests')
for p in ['/Users/apunt/repos/KIRBy/tests','/Users/apunt/repos/KIRBy/src','/Users/apunt/repos/NoiseInject']: sys.path.insert(0,p)
import importlib.util
spec=importlib.util.spec_from_file_location("pipe","/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py")
pipe=importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
from sklearn.ensemble import RandomForestRegressor
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/'
y=np.load(S+'logd.npy')[:1200]
rng=np.random.RandomState(0); n=len(y); X=rng.randn(n,12)
tr=np.arange(900); te=np.arange(900,n)
sig=[0.0,0.3,0.7]
res={}
for st in pipe.STRATEGIES:
    p,u,ex=pipe.run_tree_experiment(X[tr],y[tr],X[te],y[te],
        lambda: RandomForestRegressor(n_estimators=30,random_state=42,n_jobs=1), st, sig, oof_folds=0)
    res[st]=(p,u,ex)

print("CHECK 1: is the sigma=0 fit identical across all six strategies?")
base=res['legacy'][0][0.0]; baseu=res['legacy'][1][0.0]
for st in pipe.STRATEGIES:
    print(f"   {st:9s} pred identical={np.array_equal(res[st][0][0.0],base)}  unc identical={np.array_equal(res[st][1][0.0],baseu)}")

print()
print("CHECK 2: is  noise_scale == sigma * noise_pattern  exactly, for every strategy/sigma?")
for st in pipe.STRATEGIES:
    ex=res[st][2]
    ok=[]
    for s in sig:
        ts=ex['test_noise_scale'][s]; tp=ex['test_noise_pattern'][s]
        ok.append(np.allclose(ts, s*tp, rtol=0, atol=1e-12))
    print(f"   {st:9s} test: {dict(zip(sig,ok))}")

print()
print("CHECK 3: at a FIXED sigma>0, spearman(noise_scale, noise_pattern) on test rows")
from scipy.stats import spearmanr
for st in pipe.STRATEGIES:
    ex=res[st][2]; ts=ex['test_noise_scale'][0.7]; tp=ex['test_noise_pattern'][0.7]
    if len(np.unique(tp))==1: print(f"   {st:9s} undefined (constant)"); continue
    print(f"   {st:9s} rho = {spearmanr(ts,tp).correlation:.12f}")
