import pandas as pd, numpy as np
from scipy.stats import spearmanr
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
d['r2c']=d.r2.clip(lower=-1.0)
base=d[d.sigma==0][['dataset','rep','strategy','model','fold','r2c']].rename(columns={'r2c':'b'})
m=d.merge(base,on=['dataset','rep','strategy','model','fold'])
m['drop']=m.b-m.r2c          # paired per-fold drop

rows=[]
for (ds,rep,strat,sg),g in m.groupby(['dataset','rep','strategy','sigma']):
    pv=g.pivot_table(index='fold',columns='model',values='r2c')
    pb=g.pivot_table(index='fold',columns='model',values='b')
    pd_=g.pivot_table(index='fold',columns='model',values='drop')
    # 1. does sigma-ranking differ from clean ranking?
    rho_base=spearmanr(pv.mean(0).values, pb.mean(0).values).correlation
    # 2. paired-drop SNR: between-model SD of mean drop / within-fold SD of drop
    mm=pd_.mean(0); w=pd_.std(0,ddof=1)
    snr_drop=mm.std(ddof=1)/np.sqrt((w**2).mean()) if (w**2).mean()>0 else np.nan
    rows.append(dict(dataset=ds,rep=rep,strategy=strat,sigma=sg,
                     rho_vs_clean=rho_base,snr_drop=snr_drop,mean_drop=mm.mean()))
R=pd.DataFrame(rows); R.to_csv(S+'q1_drop.csv',index=False)
print("=== Spearman(model ranking at sigma, model ranking at sigma=0), median ===")
print(R.pivot_table(index='sigma',columns='dataset',values='rho_vs_clean',aggfunc='median').round(3).to_string())
print("pooled:", R.groupby('sigma').rho_vs_clean.median().round(3).to_dict())
print("\n=== paired-drop SNR (between-model SD of drop / within-fold SD of drop), median ===")
print(R.pivot_table(index='sigma',columns='dataset',values='snr_drop',aggfunc='median').round(3).to_string())
print("pooled:", R.groupby('sigma').snr_drop.median().round(3).to_dict())
print("\n=== paired-drop SNR by strategy (median over dataset,rep) ===")
print(R.pivot_table(index='sigma',columns='strategy',values='snr_drop',aggfunc='median').round(3).to_string())
print("\n=== mean drop (clipped R2 units), median over cells ===")
print(R.pivot_table(index='sigma',columns='dataset',values='mean_drop',aggfunc='median').round(3).to_string())
print("\n=== argmax sigma of paired-drop SNR ===")
b=R[R.sigma>0].loc[R[R.sigma>0].groupby(['dataset','rep','strategy']).snr_drop.idxmax()]
print(b.groupby('dataset').sigma.value_counts().unstack(fill_value=0).to_string())
print("overall:", b.sigma.value_counts().sort_index().to_dict())
