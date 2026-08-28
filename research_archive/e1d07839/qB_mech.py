import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression
df = pd.read_csv('/Users/apunt/repos/qsar_qm_models/results/validation_full/openadmet_logd/QRF_PDV_uncertainty_values.csv')
y = df[df.sigma==0.0].y_true.values.astype(float)

print("1) Is noise_scale exactly proportional to sigma?  (=> rank target identical at every sigma)")
for s in ['legacy','outlier','quantile','hetero','threshold','valprop']:
    inj=NoiseInjectorRegression(s,42)
    a=inj.noise_scale(y,0.1,reference=y); b=inj.noise_scale(y,1.0,reference=y)
    prop = np.allclose(b, a*10)
    rho = spearmanr(a,b).correlation if np.std(a)>0 else float('nan')
    print(f"   {s:10s} scale(1.0)==10*scale(0.1): {prop}   spearman(scale@0.1,scale@1.0)={rho}")

print("\n2) hetero vs valprop: are they the same RANKING?")
h=NoiseInjectorRegression('hetero',42).noise_scale(y,1.0)
v=NoiseInjectorRegression('valprop',42).noise_scale(y,1.0)
print(f"   spearman(hetero_scale, valprop_scale) = {spearmanr(h,v).correlation:.6f}")
print(f"   both are monotone in |y|: spearman(|y|,hetero)={spearmanr(np.abs(y),h).correlation:.6f}"
      f"  spearman(|y|,valprop)={spearmanr(np.abs(y),v).correlation:.6f}")

print("\n3) class balance of the categorical targets (logD, cutpoints as coded)")
for s in ['outlier','quantile','threshold']:
    sc=NoiseInjectorRegression(s,42).noise_scale(y,1.0,reference=y)
    u,c=np.unique(sc,return_counts=True)
    print(f"   {s:10s} {dict(zip(np.round(u,3), c))}   n={len(y)}")

print("\n4) hERG-like labels (pKi ~4-11): what does 'threshold' produce?")
pki = np.random.RandomState(0).uniform(4,11,2000)
sc=NoiseInjectorRegression('threshold',42).noise_scale(pki,1.0,reference=pki)
print(f"   unique scales = {np.unique(sc)}  -> std={np.std(sc)}  => target CONSTANT, rho undefined")
sc=NoiseInjectorRegression('valprop',42).noise_scale(pki,1.0)
print(f"   valprop on pKi: scale range {sc.min():.3f}-{sc.max():.3f} (ratio {sc.max()/sc.min():.2f}) and strictly monotone in y")
