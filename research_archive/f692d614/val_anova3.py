import pandas as pd, numpy as np
SP='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
fd=pd.read_csv(SP+'val_fold_metrics.csv')
REPS=['PDV','ECFP4','MHG-GNN-pretrained']

def _rss(y,*facs):
    n=len(y); X=np.ones((n,1))
    for f in facs:
        dm=pd.get_dummies(f,drop_first=True).to_numpy(float)
        if dm.shape[1]: X=np.hstack([X,dm])
    b,*_=np.linalg.lstsq(X,y,rcond=None); return float(((y-X@b)**2).sum())

def eta3(d,resp):
    d=d.dropna(subset=[resp]); y=d[resp].to_numpy(float); n=len(y)
    if n<8: return None
    tss=float(((y-y.mean())**2).sum())
    m=d['model'].astype(str); r=d['rep'].astype(str); fo=d['fold'].astype(str)
    cell=m.str.cat(r,sep='|')
    r_f  =_rss(y,fo)             # fold block first
    r_fm =_rss(y,fo,m)
    r_fmr=_rss(y,fo,m,r)
    r_full=_rss(y,fo,cell)
    return dict(Fold=(tss-r_f)/tss*100, Model=(r_f-r_fm)/tss*100, Rep=(r_fm-r_fmr)/tss*100,
                Interaction=(r_fmr-r_full)/tss*100, Residual=r_full/tss*100, n=n, n_models=d.model.nunique())

MINF=3
def bal(df,resp):
    d=df.dropna(subset=[resp]).copy(); reps=sorted(d.rep.unique()); sz=d.groupby(['model','rep']).size()
    keep=[m for m in d.model.unique() if all(sz.get((m,r),0)>=MINF for r in reps)]
    return d[d.model.isin(keep)]

for resp in ['auc_norm','baseline_r2','r2_s06']:
    print('='*95); print(f'## {resp}  — fold entered FIRST as a blocking factor (3 reps, GP excluded)')
    out=[]
    for ds in ['OpenADMET-LogD','OpenADMET-Caco2_Efflux','ChEMBL-hERG-Ki']:
        for st in ['legacy','quantile','threshold','hetero','valprop','outlier']:
            sub=fd[(fd.dataset==ds)&(fd.strategy==st)&(fd.rep.isin(REPS))&(fd.model!='GP')]
            e=eta3(bal(sub,resp),resp)
            if e: out.append(dict(dataset=ds.split('-')[-1],strategy=st,**e))
    print(pd.DataFrame(out).round(1).to_string(index=False)); print()
