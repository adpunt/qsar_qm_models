import sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression
df=pd.read_csv('/Users/apunt/repos/qsar_qm_models/results/validation_full/openadmet_logd/QRF_PDV_uncertainty_values.csv')
y0=df[df.sigma==0.0].y_true.values.astype(float)
tg={s:NoiseInjectorRegression(s,42).noise_scale(y0,1.0,reference=y0) for s in ['outlier','quantile','hetero','threshold','valprop']}
print("These uncertainties come from models trained under LEGACY (Gaussian) noise ONLY.")
print("There is therefore NO valprop/hetero/quantile pattern in the training labels at any sigma.")
print("Any sigma-trend below is pure artefact.\n")
print(f"{'sigma':>6} " + " ".join(f"{s:>10s}" for s in tg))
for sg in sorted(df.sigma.unique()):
    d=df[df.sigma==sg]; u=d.uncertainty.values.astype(float); y=d.y_true.values.astype(float)
    assert np.allclose(y,y0)
    row=[]
    for s,t in tg.items():
        row.append(f"{spearmanr(u,t).correlation:+10.3f}" if np.std(t)>0 else f"{'nan':>10s}")
    print(f"{sg:>6.1f} " + " ".join(row))
