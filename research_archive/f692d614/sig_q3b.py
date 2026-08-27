import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet'); d['r2c']=d.r2.clip(lower=-1.0)
key=['dataset','rep','strategy','model','fold']
d=d.join(d[d.sigma==0].set_index(key).r2c.rename('b'),on=key)
u=d[d.b>0.1].copy(); u['reldrop']=(u.b-u.r2c)/u.b
def inv(x,y,t):
    for i in range(1,len(x)):
        if y[i]>=t:
            if y[i]==y[i-1]: return x[i]
            return x[i-1]+(t-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1])
    return np.nan
print("=== legacy-equivalent dose multiplier per DATASET x strategy (sigma_legacy giving the same damage / sigma) ===")
for ds,g in u.groupby('dataset'):
    c=g.pivot_table(index='sigma',columns='strategy',values='reldrop',aggfunc='mean')
    xs=c.index.values; leg=c['legacy'].values
    print("\n"+ds)
    out={}
    for st in c.columns:
        row={}
        for i,sg in enumerate(xs):
            if sg==0: continue
            e=inv(xs,leg,c[st].values[i])
            row[sg]=e/sg if e==e else np.nan
        out[st]=row
    O=pd.DataFrame(out)
    print(O.round(2).to_string())
    m=O.loc[0.4:0.8].median()
    print("median multiplier over sigma 0.4-0.8:", m.round(2).to_dict())
    f=((4-m['threshold']**2)/3.99)
    print("  => implied THRESHOLD mid-band fraction (RMS-dose inversion): %.2f"%np.clip(f,0,1))
    fo=((9-m['outlier']**2)/8.99)
    print("  => implied OUTLIER non-tail fraction: %.2f  (i.e. tail fraction %.3f)"%(np.clip(fo,0,1),1-np.clip(fo,0,1)))
