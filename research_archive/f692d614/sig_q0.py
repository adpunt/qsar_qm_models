import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
d['vary']=d.rmse**2/(1-d.r2)
lab=d[d.sigma==0].groupby('dataset').vary.mean().pow(0.5)
print("label SD (pooled over folds, from rmse^2/(1-r2) at sigma=0):")
print(lab.round(4).to_string())
per=d[d.sigma==0].groupby(['dataset','fold']).vary.mean().pow(0.5)
print(per.round(4).to_string())
print("\nsigma=0.6 as multiple of label SD:")
print((0.6/lab).round(3).to_string())
