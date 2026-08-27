import pandas as pd, numpy as np, itertools
base="/Users/apunt/repos/KIRBy/tests/results/validation/%s/all_results.csv"
NEURAL={'DNN','MLP','BNN-Full','MLP-BNN-Full','VBLL-Full','MLP-VBLL-Full'}
def eta2(df,y='r2'):
    # balanced-ish two-way with replication, Type-I on Model then Rep then Inter
    d=df.dropna(subset=[y])
    gm=d[y].mean(); sst=((d[y]-gm)**2).sum()
    cm=d.groupby(['model','rep'])[y].mean()
    n=d.groupby(['model','rep'])[y].size()
    ssm=sum(len(g)* (g[y].mean()-gm)**2 for _,g in d.groupby('model'))
    ssr=sum(len(g)* (g[y].mean()-gm)**2 for _,g in d.groupby('rep'))
    sscell=sum(len(g)*(g[y].mean()-gm)**2 for _,g in d.groupby(['model','rep']))
    ssi=sscell-ssm-ssr
    sse=sst-sscell
    return [100*x/sst for x in (ssm,ssr,ssi,sse)]+[len(d)]

for ds in ['logd','caco2','herg']:
    df=pd.read_csv(base%ds); df=df[df.model!='GP'].copy()
    d0=df[df.sigma==0].copy()
    # identical-run check
    g=d0.groupby(['model','rep','fold'])['r2']
    sd=g.std()
    print(ds,"median across-strategy SD in (model,rep,fold):",round(sd.median(),6),
          " frac cells with SD==0:",round((sd==0).mean(),3), " max SD:",round(sd.max(),2))
    # A3 split decompositions at sigma=0, r2 floored at 0
    for name,models in [('nonneural',[m for m in d0.model.unique() if m not in NEURAL]),
                        ('neural',[m for m in d0.model.unique() if m in NEURAL]),
                        ('all',list(d0.model.unique()))]:
        s=d0[d0.model.isin(models)].copy(); s['r2']=s.r2.clip(lower=0)
        e=eta2(s); print("   ",ds,name,"M/R/I/Res:",[round(x,1) for x in e[:4]],"n",e[4])
    # A2: model/rep/inter eta2 per strategy per sigma
    rows=[]
    for strat in sorted(df.strategy.unique()):
        for sg in [0.0,0.3,0.6,1.0]:
            s=df[(df.strategy==strat)&(np.isclose(df.sigma,sg))].copy()
            s['r2']=s.r2.clip(lower=0)
            e=eta2(s); rows.append((strat,sg,*[round(x,1) for x in e[:4]],e[4]))
    r=pd.DataFrame(rows,columns=['strat','sigma','M','R','I','Res','n'])
    print(r.to_string(index=False))
    print("   A2 check: Model largest in all?",(r.M>r[['R','I','Res']].max(axis=1)).all(),
          "M range",r.M.min(),r.M.max(),"I range",r.I.min(),r.I.max(),"R range",r.R.min(),r.R.max())
