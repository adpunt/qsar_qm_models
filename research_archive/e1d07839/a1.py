import pandas as pd, numpy as np
base="/Users/apunt/repos/KIRBy/tests/results/validation/%s/all_results.csv"
NEURAL={'DNN','MLP','BNN-Full','MLP-BNN-Full','VBLL-Full','MLP-VBLL-Full'}
for ds in ['logd','caco2','herg']:
    df=pd.read_csv(base%ds)
    print("==",ds,df.shape, sorted(df.model.unique()), sorted(df.rep.unique()))
    d0=df[df.sigma==0]
    print(" sigma0 rows total",len(d0), " GP rows",(d0.model=='GP').sum())
    dg=d0[d0.model!='GP']
    print(" non-GP sigma0 rows",len(dg))
    dg=dg.copy(); dg['fam']=np.where(dg.model.isin(NEURAL),'neural','nonneural')
    t=dg.groupby(['fam','rep']).apply(lambda g: pd.Series({'n':len(g),'neg':(g.r2<0).sum()}))
    print(t)
    # GP failures
    gp=d0[d0.model=='GP']
    if len(gp): print(" GP: n",len(gp),"neg",(gp.r2<0).sum(),"median r2",gp.r2.median())
    # per-model worst
    print(" worst run:",dg.r2.min())
    # per model median by rep
    piv=dg.pivot_table(index='model',columns='rep',values='r2',aggfunc='median').round(3)
    print(piv)
