import pandas as pd, numpy as np
from scipy import stats

def fixed_resid(g):
    yn=g['y_true_noisy'].values; yo=g['y_true_original'].values
    m=np.isfinite(yn)&np.isfinite(yo)
    s,i,_,_,_=stats.linregress(yo[m],yn[m])
    return yn-(s*yo+i)

for name in ['dnn_lastlayer_bnn','dnn_var_bnn','dnn_full_bnn','gp']:
    f=f'results/phase1_continuous_pdv_{name}_uncertainty_values.csv'
    df=pd.read_csv(f).rename(columns={'representation':'rep'})
    d0=df[np.isclose(df['sigma'],0.0)].copy()
    res=np.concatenate([fixed_resid(g) for _,g in d0.groupby(['model','rep','sigma','iteration'])])
    d0=d0.sort_values(['model','rep','sigma','iteration']) # align order not needed; recompute per group order
    # rebuild aligned
    res_list=[]; unc_list=[]; ymag_list=[]
    for _,g in d0.groupby(['model','rep','sigma','iteration']):
        res_list.append(fixed_resid(g)); unc_list.append(g['y_pred_std_calibrated'].values); ymag_list.append(np.abs(g['y_true_noisy'].values))
    res=np.concatenate(res_list); unc=np.concatenate(unc_list); ymag=np.concatenate(ymag_list)
    absres=np.abs(res)
    r_ru,_=stats.spearmanr(unc,absres)          # uncertainty vs |float-residual|  (the reported sig0 rho)
    r_ry,_=stats.spearmanr(absres,ymag)         # |float-residual| vs |y|  (precision leakage)
    r_uy,_=stats.spearmanr(unc,ymag)            # uncertainty vs |y|  (does model unc track |y|?)
    print(f"{name:20s} rho(unc,|res|)={r_ru:+.4f}  rho(|res|,|y|)={r_ry:+.4f}  rho(unc,|y|)={r_uy:+.4f}  |res|max={absres.max():.2e}")
