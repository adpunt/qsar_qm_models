import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet'); d['r2c']=d.r2.clip(lower=-1)
d['vary']=d.rmse**2/(1-d.r2)
sd=d[d.sigma==0].groupby('dataset').vary.mean().pow(0.5)
d['nsr']=d.sigma/d.dataset.map(sd)
key=['dataset','rep','strategy','model','fold']
d=d.join(d[d.sigma==0].set_index(key).r2c.rename('b'),on=key)
u=d[d.b>0.1].copy(); u['rel']=u.r2c/u.b
g=u[u.strategy=='legacy']
grid=np.round(np.arange(0.1,0.81,0.1),2)   # COMMON support in SD units for all 3
A={}; B={}
for ds,gg in g.groupby('dataset'):
    c=gg.groupby('nsr').rel.mean().sort_index()
    A[ds]=np.interp(grid,c.index.values,c.values)
    cn=gg.groupby('sigma').rel.mean().sort_index()
    B[ds]=np.interp(grid,cn.index.values,cn.values)   # same nominal grid values 0.1..0.8
PA=pd.DataFrame(A,index=grid); PB=pd.DataFrame(B,index=grid)
print("SD-UNIT axis (sigma/labelSD), common support 0.1-0.8:")
print(PA.round(3).to_string()); print("spread:",(PA.max(1)-PA.min(1)).round(3).to_dict())
print("mean across-dataset spread = %.4f"%(PA.max(1)-PA.min(1)).mean())
print("\nNOMINAL sigma axis, same 0.1-0.8 range:")
print(PB.round(3).to_string()); print("spread:",(PB.max(1)-PB.min(1)).round(3).to_dict())
print("mean across-dataset spread = %.4f"%(PB.max(1)-PB.min(1)).mean())
print("\nreduction factor: %.1fx"%((PB.max(1)-PB.min(1)).mean()/(PA.max(1)-PA.min(1)).mean()))
print("\nsigma=0.6 in label-SD units per dataset:", (0.6/sd).round(2).to_dict())
print("sigma needed for 0.6 label-SD per dataset:", (0.6*sd).round(2).to_dict())
