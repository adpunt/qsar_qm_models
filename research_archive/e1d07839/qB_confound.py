import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, '/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression

STRATS = ['legacy','outlier','quantile','hetero','threshold','valprop']
files = [
 ('logd','QRF_PDV','results/validation_full/openadmet_logd/QRF_PDV_uncertainty_values.csv'),
 ('logd','QRF_ECFP4','results/validation_full/openadmet_logd/QRF_ECFP4_uncertainty_values.csv'),
 ('logd','GP_PDV','results/validation_full/openadmet_logd/GP_PDV_uncertainty_values.csv'),
 ('caco2','QRF_ECFP4','results/validation_full/openadmet_caco2/QRF_ECFP4_uncertainty_values.csv'),
]
print("Spearman( sigma=0 uncertainty , noise_scale target ) -- NO NOISE WAS INJECTED")
print("If this is far from 0, question B's headline is reproducible with zero noise-learning.\n")
for ds, tag, path in files:
    try: df = pd.read_csv('/Users/apunt/repos/qsar_qm_models/'+path)
    except Exception as e: print(ds,tag,'SKIP',e); continue
    d0 = df[df.sigma==0.0]
    y = d0.y_true.values.astype(float); u = d0.uncertainty.values.astype(float)
    print(f"--- {ds} {tag}  n={len(y)}  y range [{y.min():.2f},{y.max():.2f}]")
    for s in STRATS:
        inj = NoiseInjectorRegression(strategy=s, random_state=42)
        # the exact target this run writes for TEST rows, at sigma=1.0
        h = inj.noise_scale(y, 1.0, reference=y)
        if np.nanstd(h) == 0:
            print(f"    {s:10s} target CONSTANT (std=0) -> rho undefined (nan). uniq={np.unique(h)}")
            continue
        rho, p = spearmanr(u, h)
        # what fraction of molecules sit in the 'high noise' class
        frac = (h > np.median(h)).mean()
        print(f"    {s:10s} rho={rho:+.3f}  p={p:.2g}   target uniq={len(np.unique(h))} frac>med={frac:.2f}")
