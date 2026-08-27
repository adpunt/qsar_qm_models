import pandas as pd, numpy as np, io
# simulate: previous file written by task A (strategy 'legacy', folds 0-4, splits test/train_oof)
def mk(strategy, folds, n=3):
    rows=[]
    for sp in ('test','train_oof'):
        for f in folds:
            for i in range(n):
                rows.append(dict(split=sp,strategy=strategy,sigma=0.0,fold=f,sample_idx=i,
                                 y_true=1.0,y_pred=1.0,uncertainty=0.1,noise_scale=0.0,injected_noise=0.0))
    return pd.DataFrame(rows)

prev_df = mk('legacy',[0,1,2,3,4])
prev_df.to_csv('prev.csv',index=False)

# new run: SAME strategy, rerun (should REPLACE prev rows, not duplicate)
unc_all = mk('legacy',[0,1,2,3,4])
prev = pd.read_csv('prev.csv')
keys=[c for c in ('split','strategy','fold') if c in prev.columns and c in unc_all.columns]
new_keys=set(map(tuple, unc_all[keys].drop_duplicates().values))
print("dtypes prev:", dict(prev[keys].dtypes))
print("dtypes new :", dict(unc_all[keys].dtypes))
print("sample new_key:", list(new_keys)[0], [type(x) for x in list(new_keys)[0]])
mask = prev[keys].apply(tuple,axis=1).isin(new_keys)
print("prev rows:",len(prev),"matched for removal:",mask.sum())
kept = prev[~mask]
out = pd.concat([kept, unc_all], ignore_index=True)
print("RESULT rows:",len(out), "(expected 30 if dedup works, 60 if duplicated)")
