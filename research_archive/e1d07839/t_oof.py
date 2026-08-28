import numpy as np, sys
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests')
# import just the function without running module-level heavy imports
import importlib.util, types
src = open('/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py').read()
start = src.index('def _oof_predict')
end = src.index('def _tree_mean_std')
ns = {'np': np}
exec(src[start:end], ns)
_oof_predict = ns['_oof_predict']

for n in (7, 100, 2001):
    for k in (5,):
        # synthetic model: prediction encodes the row index of the scored row.
        # We cheat by encoding index in X: X[:,0] = index
        X = np.arange(n, dtype=float).reshape(-1,1)
        y = np.arange(n, dtype=float)*1000
        seen = []
        def fp(Xf, yf, Xs):
            seen.append((Xf[:,0].astype(int).copy(), Xs[:,0].astype(int).copy()))
            return Xs[:,0].copy(), Xs[:,0].copy()+0.5
        m,u = _oof_predict(fp, X, y, k, seed=42)
        assert not np.isnan(m).any(), (n,'nan in mean')
        ok = np.array_equal(m, np.arange(n,dtype=float))
        ok2 = np.array_equal(u, np.arange(n,dtype=float)+0.5)
        # partition check
        held_all = np.concatenate([s[1] for s in seen])
        part_ok = np.array_equal(np.sort(held_all), np.arange(n))
        overlap = any(len(np.intersect1d(f,h))>0 for f,h in seen)
        cover = all(len(f)+len(h)==n for f,h in seen)
        print(f"n={n} k={k} index_ok={ok} unc_ok={ok2} partition={part_ok} keep/held_overlap={overlap} keep+held==n={cover}")
