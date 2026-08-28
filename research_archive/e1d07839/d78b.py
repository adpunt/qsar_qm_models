import sys, numpy as np
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject import NoiseInjectorRegression
from scipy.stats import spearmanr
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/'
D={n:np.load(S+n+'.npy') for n in ['logd','caco2','herg']}

print("="*70); print("DEFECT 7a: threshold noise scale, DEFAULT (absolute +/-1.0)")
for n,y in D.items():
    inj=NoiseInjectorRegression('threshold',0)
    sc=inj.noise_scale(y,1.0)
    u=np.unique(np.round(sc,12))
    print(f"  {n:6s} n={len(y):5d} range=[{y.min():.3f},{y.max():.3f}]  unique sigma_i = {u}  n_unique={len(u)}  CONSTANT={len(u)==1}")

print()
print("DEFECT 7a: threshold with --threshold-quantile 0.1 (cut-points from labels)")
for n,y in D.items():
    q=0.1
    sp={'high_threshold':float(np.quantile(y,1-q)),'low_threshold':float(np.quantile(y,q))}
    inj=NoiseInjectorRegression('threshold',0)
    sc=inj.noise_scale(y,1.0,**sp)
    u,c=np.unique(np.round(sc,12),return_counts=True)
    print(f"  {n:6s} cut={sp}  unique={u} counts={c}  n_unique={len(u)}")

print()
print("="*70); print("DEFECT 8: hetero vs valprop rank identity (spearman of sigma_i)")
for n,y in D.items():
    h=NoiseInjectorRegression('hetero',0).noise_scale(y,1.0)
    v=NoiseInjectorRegression('valprop',0).noise_scale(y,1.0)
    rho=spearmanr(h,v).correlation
    # exact rank identity
    rh=np.argsort(np.argsort(h)); rv=np.argsort(np.argsort(v))
    # tie-robust: compare via ranking of |y|
    ay=np.abs(y)
    rho_hy=spearmanr(h,ay).correlation; rho_vy=spearmanr(v,ay).correlation
    print(f"  {n:6s} spearman(hetero_sigma, valprop_sigma) = {rho:.12f}   "
          f"spearman(hetero,|y|)={rho_hy:.6f} spearman(valprop,|y|)={rho_vy:.6f}  "
          f"argsort_identical={np.array_equal(rh,rv)}")
