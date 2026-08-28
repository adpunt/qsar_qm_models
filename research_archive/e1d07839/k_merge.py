import pandas as pd, numpy as np, io
def merge(prev, unc_all):
    keys=[c for c in ('split','strategy','fold') if c in prev.columns and c in unc_all.columns]
    if keys:
        new_keys=set(map(tuple, unc_all[keys].drop_duplicates().values))
        prev=prev[~prev[keys].apply(tuple,axis=1).isin(new_keys)]
    return pd.concat([prev,unc_all],ignore_index=True)

def mk(split,strat,fold,n=3,tag=0):
    return pd.DataFrame({'split':[split]*n,'strategy':[strat]*n,'fold':[fold]*n,
                         'sample_idx':range(n),'uncertainty':[tag]*n})

# case 1: rerun of the same strategy -> old rows must be dropped
prev=pd.concat([mk('test','legacy',f,tag=1) for f in range(5)],ignore_index=True)
new =pd.concat([mk('test','legacy',f,tag=2) for f in range(5)],ignore_index=True)
r=merge(prev.copy(),new); print("case1 rerun same strategy: rows",len(r),"tags",sorted(r['uncertainty'].unique()))

# case 2: different strategies written into the same dir
prev=pd.concat([mk('test','legacy',f,tag=1) for f in range(5)],ignore_index=True)
new =pd.concat([mk('test','outlier',f,tag=2) for f in range(5)],ignore_index=True)
r=merge(prev.copy(),new); print("case2 different strategy: rows",len(r),"strategies",sorted(r['strategy'].unique()))

# case 3: SAME strategy, DIFFERENT sigma subset  (keys have no sigma)
prev=mk('test','legacy',0,tag=1).assign(sigma=0.0)
new =mk('test','legacy',0,tag=2).assign(sigma=0.5)
r=merge(prev.copy(),new); print("case3 different sigma same (split,strategy,fold): rows",len(r),
      "sigmas kept",sorted(r['sigma'].unique()))

# case 4: prev has dtype str fold read back from csv
buf=io.StringIO(); prev=mk('test','legacy',0,tag=1); prev.to_csv(buf,index=False)
prev2=pd.read_csv(io.StringIO(buf.getvalue()))
new=mk('test','legacy',0,tag=2)
r=merge(prev2,new); print("case4 csv-roundtrip dedupe: rows",len(r),"tags",sorted(r['uncertainty'].unique()))
