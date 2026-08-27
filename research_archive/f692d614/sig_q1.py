import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
print("r2 range:", round(d.r2.min(),3), round(d.r2.max(),3))
print("quantiles:", d.r2.quantile([0.001,.01,.05,.5,.95,1.0]).round(3).to_dict())

rows=[]
for (ds,rep,strat,sg),g in d.groupby(['dataset','rep','strategy','sigma']):
    mm=g.groupby('model').r2.mean()
    between_sd=mm.std(ddof=1)
    rng=mm.max()-mm.min()
    # within: pooled between-fold SD
    w=g.groupby('model').r2.std(ddof=1)
    within_sd=np.sqrt((w**2).mean())
    nf=g.groupby('model').size().mean()
    # one-way ANOVA F (model factor)
    F=(nf*between_sd**2)/(within_sd**2) if within_sd>0 else np.nan
    rows.append(dict(dataset=ds,rep=rep,strategy=strat,sigma=sg,
                     mean_r2=g.r2.mean(),between_sd=between_sd,range=rng,
                     within_sd=within_sd,snr=between_sd/within_sd,F=F))
R=pd.DataFrame(rows)
R.to_csv(S+'q1_cells.csv',index=False)

print("\n=== Q1a: median across (rep,strategy) of between-model SD / within-fold SD, by dataset x sigma ===")
t=R.pivot_table(index='sigma',columns='dataset',values='snr',aggfunc='median')
print(t.round(3).to_string())
print("\n=== between-model SD (median) ===")
print(R.pivot_table(index='sigma',columns='dataset',values='between_sd',aggfunc='median').round(4).to_string())
print("\n=== within-fold SD (median) ===")
print(R.pivot_table(index='sigma',columns='dataset',values='within_sd',aggfunc='median').round(4).to_string())
print("\n=== range of model means (median) ===")
print(R.pivot_table(index='sigma',columns='dataset',values='range',aggfunc='median').round(4).to_string())
print("\n=== mean R2 (median over cells) ===")
print(R.pivot_table(index='sigma',columns='dataset',values='mean_r2',aggfunc='median').round(4).to_string())
print("\n=== F statistic (median) ===")
print(R.pivot_table(index='sigma',columns='dataset',values='F',aggfunc='median').round(2).to_string())
print("\n=== SNR pooled over all datasets (median, mean) ===")
print(R.groupby('sigma').snr.agg(['median','mean']).round(3).to_string())
print("\n=== SNR by strategy x sigma (median over dataset,rep) ===")
print(R.pivot_table(index='sigma',columns='strategy',values='snr',aggfunc='median').round(3).to_string())
print("\n=== argmax sigma per (dataset,rep,strategy) by snr ===")
best=R.loc[R.groupby(['dataset','rep','strategy']).snr.idxmax()]
print(best.groupby('dataset').sigma.value_counts().unstack(fill_value=0).to_string())
print("\noverall argmax sigma counts:"); print(best.sigma.value_counts().sort_index().to_string())
