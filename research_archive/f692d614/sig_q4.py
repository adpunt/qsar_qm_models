import pandas as pd, numpy as np
from scipy.stats import spearmanr
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
print("=== fraction of (dataset,model,rep,strategy,fold) curves with R2 < 0 AT each sigma ===")
t=d.assign(neg=d.r2<0).pivot_table(index='sigma',columns='dataset',values='neg',aggfunc='mean')
t['ALL']=d.assign(neg=d.r2<0).groupby('sigma').neg.mean()
print((t*100).round(1).to_string())
print("\n=== fraction with R2 < 0 AT OR BEFORE each sigma (cumulative, per curve) ===")
key=['dataset','model','rep','strategy','fold']
rows=[]
for sg in sorted(d.sigma.unique()):
    sub=d[d.sigma<=sg].assign(neg=lambda x:x.r2<0)
    ever=sub.groupby(key).neg.max()
    rows.append(dict(sigma=sg,frac=ever.mean()))
print(pd.DataFrame(rows).assign(pct=lambda x:(x.frac*100).round(1))[['sigma','pct']].to_string(index=False))

print("\n=== how much of the total R2 decline has happened by each sigma (mean clipped R2, normalised) ===")
d['r2c']=d.r2.clip(lower=-1)
mc=d.groupby('sigma').r2c.mean()
print(((mc.iloc[0]-mc)/(mc.iloc[0]-mc.iloc[-1])).round(3).to_string())

print("\n=== per-dataset mean clipped R2 and monotone check ===")
print(d.pivot_table(index='sigma',columns='dataset',values='r2c',aggfunc='mean').round(4).to_string())

print("\n=== does the sigma>0.6 half add ranking information? ===")
cell=d.groupby(['dataset','rep','strategy','model','sigma']).r2c.mean().reset_index()
res=[]
for (ds,rep,st),g in cell.groupby(['dataset','rep','strategy']):
    p=g.pivot_table(index='sigma',columns='model',values='r2c').sort_index()
    lo=p.loc[:0.6].mean(); hi=p.loc[0.7:].mean()
    res.append(dict(dataset=ds,rho=spearmanr(lo,hi).correlation))
R=pd.DataFrame(res)
print("Spearman(model ranking from sigma<=0.6 mean, ranking from sigma>=0.7 mean): mean %.3f median %.3f"%(R.rho.mean(),R.rho.median()))
print(R.groupby('dataset').rho.mean().round(3).to_string())

print("\n=== curvature: where is the R2 curve steepest? mean |dR2/dsigma| per interval ===")
p=d.pivot_table(index='sigma',columns='dataset',values='r2c',aggfunc='mean')
print((-p.diff()/0.1).round(3).to_string())
