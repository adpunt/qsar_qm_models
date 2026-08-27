import pandas as pd, numpy as np
from scipy import stats
g=pd.read_csv('per_fold_gated.csv'); gb=g[(g.auc_norm>=0)&(g.r2_06>=-0.5)]
# model:rep ratio
o=pd.read_csv('val_eta2_filtered.csv')
for m in ['auc_norm','r2_06']:
    s=o[o.metric==m]; r=100*s['model']/(s['model']+s['rep'])
    print(m,'model share of (model+rep): min %.1f max %.1f median %.1f'%(r.min(),r.max(),r.median()))
    print('   rep eta2 range %.1f-%.1f ; model eta2 range %.1f-%.1f'%(s.rep.min(),s.rep.max(),s.model.min(),s.model.max()))
print()
# per-strategy damage vs model eta2
cell=gb.groupby(['dataset','strategy']).agg(mean_ret=('auc_norm','mean'),
        spread=('auc_norm',lambda x: x.quantile(.9)-x.quantile(.1))).reset_index()
mm=o[o.metric=='auc_norm'][['dataset','strategy','model']].rename(columns={'model':'model_eta2'})
cell=cell.merge(mm,on=['dataset','strategy'])
print(cell.round(3).to_string(index=False))
print('\nSpearman(mean retention, model eta2) within dataset:')
for ds,s in cell.groupby('dataset'):
    print(' ',ds, round(stats.spearmanr(s.mean_ret,s.model_eta2).statistic,3))
