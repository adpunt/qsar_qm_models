import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet'); d['r2c']=d.r2.clip(lower=-1)
d['vary']=d.rmse**2/(1-d.r2)
sd=d[d.sigma==0].groupby('dataset').vary.mean().pow(0.5)
print("label SD:",sd.round(3).to_dict())
d['nsr']=d.sigma/d.dataset.map(sd)
key=['dataset','rep','strategy','model','fold']
d=d.join(d[d.sigma==0].set_index(key).r2c.rename('b'),on=key)
u=d[d.b>0.1].copy(); u['rel']=u.r2c/u.b     # retained fraction of clean R2

print("\n=== retained fraction of clean R2 vs NOMINAL sigma (legacy/Gaussian only) ===")
g=u[u.strategy=='legacy']
print(g.pivot_table(index='sigma',columns='dataset',values='rel',aggfunc='mean').round(3).to_string())
print("  spread across datasets at each sigma (max-min):")
p=g.pivot_table(index='sigma',columns='dataset',values='rel',aggfunc='mean')
print((p.max(1)-p.min(1)).round(3).to_string())
print("  MEAN across-dataset spread (nominal sigma): %.4f"%(p.max(1)-p.min(1)).mean())

print("\n=== retained fraction vs sigma / label SD (legacy), on a common grid ===")
grid=np.arange(0.1,1.31,0.1)
out={}
for ds,gg in g.groupby('dataset'):
    c=gg.groupby('nsr').rel.mean().sort_index()
    out[ds]=np.interp(grid,c.index.values,c.values,left=np.nan,right=np.nan)
P=pd.DataFrame(out,index=np.round(grid,2))
print(P.round(3).to_string())
sp=(P.max(1)-P.min(1)).dropna()
print("  spread across datasets:"); print(sp.round(3).to_string())
print("  MEAN across-dataset spread (SD units, common support): %.4f"%sp.mean())
# matched comparison on the same rows of support
common=P.dropna().index
p2=p.copy(); p2.index=np.round(p2.index,2)
print("\nlike-for-like: nominal-sigma spread over its full grid %.4f vs SD-unit spread %.4f"%(
    (p.max(1)-p.min(1)).mean(), sp.mean()))
