import pandas as pd, numpy as np
SP='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
fd=pd.read_csv(SP+'val_fold_metrics.csv')

def eta2(df,resp,f1='model',f2='rep'):
    d=df.dropna(subset=[resp,f1,f2]); y=d[resp].to_numpy(float); n=len(y)
    if n<6: return None
    tss=float(((y-y.mean())**2).sum())
    if tss==0: return None
    def rss(*facs):
        X=np.ones((n,1))
        for f in facs:
            dm=pd.get_dummies(f,drop_first=True).to_numpy(float)
            if dm.shape[1]: X=np.hstack([X,dm])
        b,*_=np.linalg.lstsq(X,y,rcond=None); return float(((y-X@b)**2).sum())
    m=d[f1].astype(str); r=d[f2].astype(str); cell=m.str.cat(r,sep='|')
    rm=rss(m); rmr=rss(m,r); rf=rss(cell)
    return dict(Model=(tss-rm)/tss*100, Rep=(rmr and (rm-rmr))/tss*100,
                Interaction=(rmr-rf)/tss*100, Residual=rf/tss*100,
                n=n, n_models=d[f1].nunique(), n_reps=d[f2].nunique())

MINF=3
def balanced(df,resp):
    d=df.dropna(subset=[resp]).copy()
    reps=sorted(d.rep.unique()); sizes=d.groupby(['model','rep']).size()
    keep=[m for m in d.model.unique() if all(sizes.get((m,r),0)>=MINF for r in reps)]
    return d[d.model.isin(keep)], sorted(set(d.model.unique())-set(keep))

for REPSET,tag in [(['PDV','ECFP4','MHG-GNN-pretrained'],'3 reps (SNS dropped, QM9 convention)'),
                   (['PDV','ECFP4','MHG-GNN-pretrained','SNS'],'4 reps (all)')]:
  for resp in ['auc_norm','baseline_r2','r2_s06']:
    print('#'*100); print(f'## {resp}   |  {tag}')
    out=[]
    for ds in ['OpenADMET-LogD','OpenADMET-Caco2_Efflux','ChEMBL-hERG-Ki']:
        for st in ['legacy','quantile','threshold','hetero','valprop','outlier']:
            sub=fd[(fd.dataset==ds)&(fd.strategy==st)&(fd.rep.isin(REPSET))&(fd.model!='GP')]
            b,dropped=balanced(sub,resp)
            e=eta2(b,resp)
            if e: out.append(dict(dataset=ds.replace('OpenADMET-','').replace('ChEMBL-',''),strategy=st,**e,dropped=len(dropped)))
    o=pd.DataFrame(out)
    print(o.round(1).to_string(index=False))
    print()
