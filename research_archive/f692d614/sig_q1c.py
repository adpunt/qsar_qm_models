import pandas as pd, numpy as np
from scipy.stats import spearmanr, rankdata
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
d['r2c']=d.r2.clip(lower=-1.0)

def kendallW(M):   # M: folds x models matrix of values
    R=np.apply_along_axis(rankdata,1,M)   # rank models within each fold
    m,n=R.shape
    Rsum=R.sum(0)
    S_=((Rsum-Rsum.mean())**2).sum()
    return 12*S_/(m**2*(n**3-n))

rows=[]
for (ds,rep,strat,sg),g in d.groupby(['dataset','rep','strategy','sigma']):
    piv=g.pivot_table(index='fold',columns='model',values='r2')
    if piv.isna().any().any(): continue
    M=piv.values
    W=kendallW(M)
    # mean pairwise Spearman between folds
    ps=[]
    for i in range(M.shape[0]):
        for j in range(i+1,M.shape[0]):
            ps.append(spearmanr(M[i],M[j]).correlation)
    # clipped SNR
    pc=g.pivot_table(index='fold',columns='model',values='r2c')
    mm=pc.mean(0); w=pc.std(0,ddof=1)
    snr_c=mm.std(ddof=1)/np.sqrt((w**2).mean())
    rows.append(dict(dataset=ds,rep=rep,strategy=strat,sigma=sg,W=W,
                     meanSpear=np.mean(ps),snr_clip=snr_c,
                     mean_r2c=pc.values.mean()))
R=pd.DataFrame(rows); R.to_csv(S+'q1_rank.csv',index=False)

for v,lab in [('W',"Kendall's W (fold agreement on model ranking)"),
              ('meanSpear','mean pairwise between-fold Spearman of model ranking'),
              ('snr_clip','between-model SD / within-fold SD, R2 clipped at -1')]:
    print(f"\n=== {lab} : median over (rep,strategy), by dataset x sigma ===")
    print(R.pivot_table(index='sigma',columns='dataset',values=v,aggfunc='median').round(3).to_string())
    print("pooled median:", R.groupby('sigma')[v].median().round(3).to_dict())

print("\n=== W by strategy x sigma (median) ===")
print(R.pivot_table(index='sigma',columns='strategy',values='W',aggfunc='median').round(3).to_string())
print("\n=== W by rep x sigma (median) ===")
print(R.pivot_table(index='sigma',columns='rep',values='W',aggfunc='median').round(3).to_string())
print("\n=== argmax sigma by W, per (dataset,rep,strategy) ===")
b=R.loc[R.groupby(['dataset','rep','strategy']).W.idxmax()]
print(b.groupby('dataset').sigma.value_counts().unstack(fill_value=0).to_string())
