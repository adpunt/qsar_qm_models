import importlib.util, sys, numpy as np
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); sys.modules['adnr']=m; spec.loader.exec_module(m)

def check(n, groups, n_folds, label):
    X = np.arange(n, dtype=float).reshape(-1,1)   # col0 == global row index
    y = np.arange(n, dtype=float)*10.0
    seen = []
    def fp(Xf, yf, Xs):
        seen.append((Xf[:,0].astype(int).copy(), Xs[:,0].astype(int).copy()))
        return Xs[:,0].copy(), Xs[:,0].copy()+0.5
    om, ou, n_ok = m._oof_predict(fp, X, y, n_folds, groups=groups)
    prob=[]
    # 1. every molecule scored exactly once
    cnt = np.zeros(n, int)
    for keep, held in seen:
        cnt[held]+=1
        if len(np.intersect1d(keep, held)): prob.append('LEAK keep&held overlap')
        if len(keep)+len(held)!=n: prob.append(f'keep+held={len(keep)+len(held)} != n={n}')
        if groups is not None and len(np.unique(groups))>=n_folds:
            gk=set(np.asarray(groups)[keep]); gh=set(np.asarray(groups)[held])
            if gk & gh: prob.append(f'SCAFFOLD SHARED {sorted(gk&gh)[:5]}')
    if not (cnt==1).all():
        prob.append(f'coverage counts: min={cnt.min()} max={cnt.max()} n_zero={(cnt==0).sum()} n_dup={(cnt>1).sum()}')
    # 2. identity: oof_mean[i] must be i  (off-by-one / mis-assignment detector)
    if not np.array_equal(om, np.arange(n,dtype=float)): 
        bad=np.flatnonzero(om!=np.arange(n)); prob.append(f'MISASSIGNED rows {bad[:10]} got {om[bad[:10]]}')
    if not np.array_equal(ou, np.arange(n,dtype=float)+0.5):
        prob.append('unc misassigned')
    if n_ok!=len(seen): prob.append(f'n_ok {n_ok} != folds run {len(seen)}')
    print(f"{label:55s} n={n:5d} folds_run={len(seen)} n_ok={n_ok}  {'OK' if not prob else 'FAIL: '+ '; '.join(prob)}")
    return prob

rng=np.random.RandomState(0)
fails=0
# A. plenty of scaffolds, uneven sizes
for n,ng in [(100,37),(1000,250),(53,11),(2000,1999),(101,5)]:
    g=rng.randint(0,ng,n); g[:ng]=np.arange(ng)   # ensure all groups present
    fails+=len(check(n,g,5,f'GroupKFold n={n} ngroups={ng}'))
# B. one giant scaffold (pathological but legal)
g=np.zeros(200,int); g[:5]=np.arange(5)
fails+=len(check(200,g,5,'GroupKFold 1 huge group + 4 singletons'))
# C. exactly n_folds scaffolds
g=rng.randint(0,5,300); g[:5]=np.arange(5)
fails+=len(check(300,g,5,'GroupKFold exactly 5 scaffolds'))
# D. fallback: fewer scaffolds than folds
for ng in [1,2,4]:
    g=rng.randint(0,ng,257); g[:ng]=np.arange(ng)
    fails+=len(check(257,g,5,f'FALLBACK ngroups={ng}'))
# E. groups=None
fails+=len(check(257,None,5,'groups=None'))
# F. n not divisible by folds
fails+=len(check(7,None,5,'groups=None n=7 folds=5'))
# G. n < n_folds  (fallback, empty parts)
fails+=len(check(3,None,5,'groups=None n=3 folds=5'))
print('TOTAL PROBLEMS:', fails)
