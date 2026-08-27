import sys, numpy as np
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject import NoiseInjectorRegression
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/'
STR=['legacy','outlier','quantile','hetero','threshold','valprop']

# ---------- DEMO A: permutation null for question A ----------
print("="*78)
print("DEMO A  permutation null on |resid| vs |eps|, GAUSSIAN, cross-fitted, NO leakage")
rng=np.random.RandomState(0)
n=2000; d=10
X=rng.randn(n,d); f=X[:,0]*2+np.sin(X[:,1]*3)
inj=NoiseInjectorRegression('legacy',42)
eps_all={}
y_noisy=None
for sig in [0.0,0.5]:
    if sig==0.0:
        yn=f.copy(); eps=np.zeros(n)
    else:
        yn,sc,eps=inj.inject_verbose(f,sig)
    eps_all[sig]=(yn,eps)
yn,eps=eps_all[0.5]
# honest 5-fold cross-fitting
oof=np.full(n,np.nan)
idx=rng.permutation(n); parts=np.array_split(idx,5)
for h in parts:
    k=np.setdiff1d(idx,h)
    m=RandomForestRegressor(n_estimators=100,random_state=0,n_jobs=1).fit(X[k],yn[k])
    oof[h]=m.predict(X[h])
resid=np.abs(f+eps-oof)
obs=spearmanr(resid,np.abs(eps)).correlation
# null 1: RUNBOOK as literally written -- shuffle injected_noise, statistic recomputed? ambiguous
null_fixed=[]   # residual held fixed (eps shuffled only in the statistic's 2nd arg)
null_recalc=[]  # residual recomputed from the shuffled eps
r0=np.abs(f+eps-oof)
for b in range(300):
    p=rng.permutation(n); e=eps[p]
    null_fixed.append(spearmanr(r0,np.abs(e)).correlation)
    null_recalc.append(spearmanr(np.abs(f+e-oof),np.abs(e)).correlation)
nf=np.array(null_fixed); nr=np.array(null_recalc)
print(f"  observed rho(|resid|,|eps|)            = {obs:+.4f}")
print(f"  null A (residual held FIXED)   mean={nf.mean():+.4f} 95%=[{np.percentile(nf,2.5):+.4f},{np.percentile(nf,97.5):+.4f}]  -> observed inside? {np.percentile(nf,2.5)<=obs<=np.percentile(nf,97.5)}")
print(f"  null B (residual RECOMPUTED)   mean={nr.mean():+.4f} 95%=[{np.percentile(nr,2.5):+.4f},{np.percentile(nr,97.5):+.4f}]  -> observed inside? {np.percentile(nr,2.5)<=obs<=np.percentile(nr,97.5)}")
print("  (there is NO leakage in this simulation by construction)")

# ---------- DEMO B: sham ceiling discrimination ----------
print()
print("="*78)
print("DEMO B  sham ceiling: spearman(pattern(y_true), pattern(y_pred))  [logd labels, R2~0.6]")
y=np.load(S+'logd.npy')
rs=np.random.RandomState(1)
for r2 in [0.5,0.6,0.8]:
    resid_sd=np.std(y)*np.sqrt(1-r2)
    ypred=y+rs.normal(0,resid_sd,len(y))
    row=[]
    for st in STR:
        i=NoiseInjectorRegression(st,0)
        pt=i.noise_scale(y,1.0,reference=y); ps=i.noise_scale(ypred,1.0,reference=y)
        if len(np.unique(pt))==1 or len(np.unique(ps))==1: row.append(f"{st}=undef"); continue
        row.append(f"{st}={spearmanr(pt,ps).correlation:.3f}")
    print(f"  R2={r2}: "+"  ".join(row))

# ---------- DEMO C: GP 2000-subsample reference shift ----------
print()
print("="*78)
print("DEMO C  GP is capped at 2000 train pts -> quantile/outlier cut-points differ from other models")
for name in ['logd','caco2','herg']:
    yy=np.load(S+name+'.npy')
    ntr=int(len(yy)*0.8)
    tr=np.random.RandomState(7).choice(len(yy),ntr,replace=False)
    te=np.setdiff1d(np.arange(len(yy)),tr)
    ytr=yy[tr]; yte=yy[te]
    sub=np.random.RandomState(42).choice(len(ytr),min(2000,len(ytr)),replace=False)
    for st in ['quantile','outlier']:
        i=NoiseInjectorRegression(st,0)
        full=i.noise_scale(yte,1.0,reference=ytr)
        gp=i.noise_scale(yte,1.0,reference=ytr[sub])
        diff=int((full!=gp).sum())
        print(f"  {name:6s} n_train={ntr:5d} capped={len(sub):5d}  {st:9s}: {diff}/{len(yte)} test molecules get a DIFFERENT noise_pattern value ({100*diff/len(yte):.2f}%)")

# ---------- DEMO D: how many independent arms ----------
print()
print("="*78)
print("DEMO D  cross-strategy Spearman of noise_pattern (are the 6 arms independent?)")
for name in ['logd','caco2','herg']:
    yy=np.load(S+name+'.npy')
    pats={st:NoiseInjectorRegression(st,0).noise_scale(yy,1.0,reference=yy) for st in STR}
    print(f"  {name}:")
    hdr="          "+"".join(f"{s[:8]:>10s}" for s in STR); print(hdr)
    for a in STR:
        cells=[]
        for b in STR:
            if len(np.unique(pats[a]))==1 or len(np.unique(pats[b]))==1: cells.append(f"{'undef':>10s}")
            else: cells.append(f"{spearmanr(pats[a],pats[b]).correlation:>10.3f}")
        print(f"  {a:>8s}"+"".join(cells))
