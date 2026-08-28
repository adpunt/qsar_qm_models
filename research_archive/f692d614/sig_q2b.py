import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
d['r2c']=d.r2.clip(lower=-1.0)
key=['dataset','rep','strategy','model','fold']
base=d[d.sigma==0].set_index(key).r2c.rename('b')
d=d.join(base,on=key)
# restrict to cells that actually learn something clean
use=d[d.b>0.1].copy()
print("cells kept (baseline clipped R2>0.1): %d of %d rows (%.1f%%)"%(len(use),len(d),100*len(use)/len(d)))
use['reldrop']=(use.b-use.r2c)/use.b

def invert(x,y,target):
    """first sigma where mean curve y crosses target, linear interp"""
    for i in range(1,len(x)):
        if y[i]>=target:
            if y[i]==y[i-1]: return x[i]
            return x[i-1]+(target-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1])
    return np.nan

print("\n=== mean relative R2 drop vs sigma, by strategy (pooled datasets) ===")
c=use.pivot_table(index='sigma',columns='strategy',values='reldrop',aggfunc='mean')
print(c.round(4).to_string())
print("\n=== sigma for a 25% mean relative drop (equal-damage sigma) ===")
res={}
for st in c.columns:
    res[st]=invert(c.index.values,c[st].values,0.25)
eq=pd.Series(res).round(3)
print(eq.to_string())
print("\nratio to legacy:", (eq/eq['legacy']).round(2).to_dict())

print("\n=== equal-damage sigma (25%) per dataset x strategy ===")
tab={}
for ds,g in use.groupby('dataset'):
    cc=g.pivot_table(index='sigma',columns='strategy',values='reldrop',aggfunc='mean')
    tab[ds]={st:invert(cc.index.values,cc[st].values,0.25) for st in cc.columns}
T=pd.DataFrame(tab).T
print(T.round(3).to_string())
print("\n=== also 10% and 40% targets (pooled) ===")
for tg in [0.10,0.40]:
    print(tg, {st:round(invert(c.index.values,c[st].values,tg),3) if not np.isnan(invert(c.index.values,c[st].values,tg)) else None for st in c.columns})

# --- legacy-equivalent noise: invert each strategy's damage onto the legacy curve ---
print("\n=== legacy-equivalent sigma: sigma_legacy giving the same mean rel drop ===")
leg=c['legacy'].values; xs=c.index.values
out={}
for st in c.columns:
    row={}
    for i,sg in enumerate(xs):
        if sg==0: continue
        row[sg]=invert(xs,leg,c[st].values[i])
    out[st]=row
L=pd.DataFrame(out)
print(L.round(3).to_string())
print("\nlegacy-equivalent sigma / nominal sigma (the effective dose multiplier):")
print((L.div(pd.Series(L.index,index=L.index),axis=0)).round(2).to_string())
eq.to_csv(S+'q2_equal_damage.csv')
L.to_csv(S+'q2_legacy_equiv.csv')
