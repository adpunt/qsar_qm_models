import pandas as pd, numpy as np
from scipy import stats
g=pd.read_csv('per_fold_gated.csv'); gb=g[(g.auc_norm>=0)&(g.r2_06>=-0.5)]
cfg=gb.groupby(['dataset','rep','strategy','model']).agg(auc=('auc_norm','mean'),r06=('r2_06','mean'),
        base=('baseline','mean'),nf=('fold','count')).reset_index()
cfg=cfg[cfg.nf>=3]
rows=[]
for k,s in cfg.groupby(['dataset','rep','strategy']):
    if len(s)<5: continue
    rho=stats.spearmanr(s.auc,s.r06).statistic
    rows.append(dict(dataset=k[0],rep=k[1],strategy=k[2],n=len(s),rho=rho,
        top_auc=s.loc[s.auc.idxmax(),'model'], top_r06=s.loc[s.r06.idxmax(),'model'],
        top_base=s.loc[s.base.idxmax(),'model'],
        rho_auc_base=stats.spearmanr(s.auc,s.base).statistic))
r=pd.DataFrame(rows)
print('cells:',len(r))
print('Spearman(auc_norm, R2@0.6) across models: median %.2f, IQR %.2f-%.2f, min %.2f, max %.2f'%(
    r.rho.median(),r.rho.quantile(.25),r.rho.quantile(.75),r.rho.min(),r.rho.max()))
print('same top model (retention vs R2@0.6): %d/%d = %.0f%%'%((r.top_auc==r.top_r06).sum(),len(r),100*(r.top_auc==r.top_r06).mean()))
print('same top model (R2@0.6 vs clean R2): %d/%d = %.0f%%'%((r.top_r06==r.top_base).sum(),len(r),100*(r.top_r06==r.top_base).mean()))
print('Spearman(auc_norm, clean R2): median %.2f'%r.rho_auc_base.median())
print()
print(r.groupby('dataset').agg(rho_med=('rho','median'),agree=('rho',lambda x: np.nan)).round(2))
print(r.groupby('dataset').apply(lambda s: pd.Series({'rho_med':s.rho.median(),'top_agree':(s.top_auc==s.top_r06).mean()})).round(2))
print()
print('top-model counts, retention:'); print(r.top_auc.value_counts().to_string())
print('top-model counts, R2@0.6:'); print(r.top_r06.value_counts().to_string())
r.to_csv('tension.csv',index=False)
