import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression
P='/Users/apunt/repos/qsar_qm_models/results/validation_full/openadmet_logd/QRF_PDV_uncertainty_values.csv'
df=pd.read_csv(P)
d0=df[df.sigma==0.0]
y=d0.y_true.values.astype(float); yp=d0.y_pred.values.astype(float); u=d0.uncertainty.values.astype(float)
print(f"sigma=0 (CLEAN LABELS, no noise ever injected). QRF/PDV/logD n={len(y)}  R2-ish corr(y,ypred)={np.corrcoef(y,yp)[0,1]:.3f}\n")
print(f"{'strategy':10s} {'rho(REAL unc, target)':>22s} {'rho(SHAM unc = h(y_pred))':>26s}")
for s in ['outlier','quantile','hetero','threshold','valprop']:
    inj=NoiseInjectorRegression(s,42)
    target = inj.noise_scale(y, 1.0, reference=y)      # exactly what the run writes for test rows
    sham   = inj.noise_scale(yp, 1.0, reference=y)     # a model with ZERO noise-awareness: just predicts y
    if np.std(target)==0: print(f"{s:10s} target constant -> nan"); continue
    r_real=spearmanr(u,target).correlation
    r_sham=spearmanr(sham,target).correlation
    print(f"{s:10s} {r_real:+22.3f} {r_sham:+26.3f}")
print("\n=> A model that only predicts y, with NO uncertainty head at all and NO noise in the data,")
print("   scores far higher on question B than any real uncertainty estimate.")
