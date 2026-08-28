import pandas as pd, numpy as np
from scipy import stats
np.set_printoptions(suppress=True)

for name in ['dnn_lastlayer_bnn','dnn_full_bnn']:
    f=f'results/phase1_continuous_pdv_{name}_uncertainty_values.csv'
    df=pd.read_csv(f).rename(columns={'representation':'rep'})
    d0=df[np.isclose(df['sigma'],0.0)]
    print(f"\n### {name}: sigma=0 rows={len(d0)}")
    print("  iterations at sig0:", sorted(d0['iteration'].unique()))
    print("  raw injected_noise at sig0: mean=%.4e std=%.4e min=%.4e max=%.4e" % (d0['injected_noise'].mean(),d0['injected_noise'].std(),d0['injected_noise'].min(),d0['injected_noise'].max()))
    # per group, run regression, look at residual spread
    gc=['model','rep','sigma','iteration']
    resid_all=[]
    for gk,g in d0.groupby(gc):
        yn=g['y_true_noisy'].values; yo=g['y_true_original'].values
        m=np.isfinite(yn)&np.isfinite(yo)
        if m.sum()<10: continue
        s,i,r,_,_=stats.linregress(yo[m],yn[m])
        res=yn-(s*yo+i)
        resid_all.append((gk[3], s, i, r**2, np.abs(res).max(), np.std(res)))
    rr=pd.DataFrame(resid_all,columns=['iter','slope','intercept','r2','max_abs_res','std_res'])
    print(rr.to_string(index=False))
