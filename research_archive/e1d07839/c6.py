import pandas as pd, numpy as np, itertools
from scipy import stats
g=pd.read_csv('per_fold_gated.csv'); gb=g[(g.auc_norm>=0)&(g.r2_06>=-0.5)]
cfg=gb.groupby(['dataset','rep','strategy','model']).agg(auc=('auc_norm','mean'),r06=('r2_06','mean'),n=('fold','count')).reset_index()
cfg=cfg[cfg.n>=3]
for metric in ['auc','r06']:
    vals=[]
    for (ds,st),s in cfg.groupby(['dataset','strategy']):
        p=s.pivot(index='model',columns='rep',values=metric)
        for a,b in itertools.combinations(p.columns,2):
            sub=p[[a,b]].dropna()
            if len(sub)>=5: vals.append(stats.spearmanr(sub[a],sub[b]).statistic)
    v=np.array(vals)
    print(f'{metric}: cross-representation model-ranking Spearman, n={len(v)} rep-pairs x dataset x strategy; median {np.median(v):.2f}, IQR {np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}, frac>0.7 {np.mean(v>0.7):.2f}, min {v.min():.2f}')
# cross-strategy transfer, per dataset x rep
for metric in ['auc','r06']:
    vals=[]
    for (ds,rp),s in cfg.groupby(['dataset','rep']):
        p=s.pivot(index='model',columns='strategy',values=metric)
        for a,b in itertools.combinations(p.columns,2):
            sub=p[[a,b]].dropna()
            if len(sub)>=5: vals.append(stats.spearmanr(sub[a],sub[b]).statistic)
    v=np.array(vals)
    print(f'{metric}: cross-STRATEGY model-ranking Spearman, n={len(v)}; median {np.median(v):.2f}, IQR {np.percentile(v,25):.2f}-{np.percentile(v,75):.2f}, min {v.min():.2f}')
