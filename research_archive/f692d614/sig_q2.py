import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
d['r2c']=d.r2.clip(lower=-1.0)
print("=== mean clipped R2 vs sigma, by strategy x dataset ===")
for ds,g in d.groupby('dataset'):
    print("\n"+ds)
    print(g.pivot_table(index='sigma',columns='strategy',values='r2c',aggfunc='mean').round(4).to_string())
print("\n=== pooled mean clipped R2 vs sigma by strategy ===")
print(d.pivot_table(index='sigma',columns='strategy',values='r2c',aggfunc='mean').round(4).to_string())
print("\n=== median R2 (unclipped) vs sigma by strategy, pooled ===")
print(d.pivot_table(index='sigma',columns='strategy',values='r2',aggfunc='median').round(4).to_string())
