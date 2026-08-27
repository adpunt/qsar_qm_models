import pandas as pd, numpy as np, itertools
SP='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(SP+'val_rerun.parquet')
print('missing model x rep cells:')
full=set(itertools.product(d.model.unique(),d.rep.unique()))
have=set(map(tuple,d[['model','rep']].drop_duplicates().values))
print(sorted(full-have))
print()

def auc_norm(sig,r2,base):
    idx=np.argsort(sig); sig=np.asarray(sig)[idx]; r2=np.asarray(r2)[idx]
    ret=r2/base
    return float(np.trapz(ret,sig)/(sig.max()-sig.min()))

GATE=0.3
rows=[]
for (ds,strat,m,r,f),g in d.groupby(['dataset','strategy','model','rep','fold']):
    g=g.sort_values('sigma')
    b=g.loc[np.isclose(g.sigma,0.0),'r2']
    if len(b)==0: continue
    base=float(b.iloc[0])
    rows.append(dict(dataset=ds,strategy=strat,model=m,rep=r,fold=f,
                     baseline_r2=base, r2_s06=float(g.loc[np.isclose(g.sigma,0.6),'r2'].iloc[0]),
                     auc_norm=auc_norm(g.sigma.values,g.r2.values,base) if base>=GATE else np.nan))
fold_df=pd.DataFrame(rows)
fold_df.to_csv(SP+'val_fold_metrics.csv',index=False)
print('per-fold metrics rows:',len(fold_df))
print('gated out (baseline<0.3):',fold_df.auc_norm.isna().sum(),'of',len(fold_df))
print()
print('baseline pass-rate by dataset x model (fraction of folds with baseline>=0.3, legacy):')
sub=fold_df[fold_df.strategy=='legacy']
print((sub.assign(ok=sub.baseline_r2>=GATE).pivot_table(index='model',columns='dataset',values='ok')).round(2).to_string())
