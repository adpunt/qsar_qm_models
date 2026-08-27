import pandas as pd, numpy as np
from scipy import stats

files = {
 'gp':'results/phase1_continuous_pdv_gp_uncertainty_values.csv',
 'qrf':'results/phase1_continuous_pdv_qrf_uncertainty_values.csv',
 'ngboost':'results/phase1_continuous_pdv_ngboost_uncertainty_values.csv',
 'dnn_full_bnn':'results/phase1_continuous_pdv_dnn_full_bnn_uncertainty_values.csv',
 'dnn_lastlayer_bnn':'results/phase1_continuous_pdv_dnn_lastlayer_bnn_uncertainty_values.csv',
 'dnn_var_bnn':'results/phase1_continuous_pdv_dnn_var_bnn_uncertainty_values.csv',
}

def fix_injected_noise(df):
    required=['model','rep','sigma','iteration']
    group_cols=[c for c in required if c in df.columns]
    corrected=df['injected_noise'].copy()
    n=0
    for gk,gidx in df.groupby(group_cols).groups.items():
        g=df.loc[gidx]; yn=g['y_true_noisy'].values; yo=g['y_true_original'].values
        mask=np.isfinite(yn)&np.isfinite(yo)
        if mask.sum()<10: continue
        s,i,_,_,_=stats.linregress(yo[mask],yn[mask])
        corrected.loc[gidx]=yn-(s*yo+i); n+=1
    df=df.copy(); df['injected_noise']=corrected
    return df,n

for name,f in files.items():
    df=pd.read_csv(f)
    df=df.rename(columns={'representation':'rep'})
    modelcol=df['model'].iloc[0]
    df,ngrp=fix_injected_noise(df)
    unc=df['y_pred_std_calibrated'].values
    noise_mag=np.abs(df['injected_noise'].values)
    sig=df['sigma'].values
    print(f"=== {name} (model col={modelcol}) groups_fixed={ngrp} sigmas={sorted(df['sigma'].unique())}")
    for s in [0.0,0.3,0.6]:
        m=np.isclose(sig,s,atol=1e-6)&np.isfinite(noise_mag)&np.isfinite(unc)
        n=int(m.sum())
        if n>100:
            rho,p=stats.spearmanr(unc[m],noise_mag[m])
            # also stats on fixed noise at sigma
            nm=noise_mag[m]
            print(f"   sig={s} n={n} rho={rho:.4f} p={p:.2e} | |noise| mean={nm.mean():.4f} std={nm.std():.4f} max={nm.max():.4f} | unc mean={unc[m].mean():.4f}")
        else:
            print(f"   sig={s} n={n} (skipped, <=100)")
