import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
bad=d[d.r2<-1]
print("rows with R2 < -1:", len(bad), "of", len(d), f"({100*len(bad)/len(d):.2f}%)")
print("\nby model:"); print(bad.model.value_counts().to_string())
print("\nby rep:"); print(bad.rep.value_counts().to_string())
print("\nby dataset:"); print(bad.dataset.value_counts().to_string())
print("\nR2<-1 rate by sigma:")
print((d.assign(b=d.r2<-1).groupby('sigma').b.mean()*100).round(2).to_string())
print("\nR2<-1 rate by model x (sigma bucket):")
d['b']=d.r2<-1
print((d.pivot_table(index='model',columns='sigma',values='b',aggfunc='mean')*100).round(1).to_string())
