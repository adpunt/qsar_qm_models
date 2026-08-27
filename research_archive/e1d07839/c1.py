import pandas as pd, numpy as np
from scipy.integrate import trapezoid
df = pd.read_pickle('val.pkl')
rows=[]
for k,g in df.groupby(['dataset','model','rep','strategy','fold']):
    g=g.sort_values('sigma')
    sig=g.sigma.values.astype(float); r2=g.r2.values.astype(float)
    b=r2[np.isclose(sig,0.0)]
    if len(b)==0 or len(sig)<3: continue
    base=float(b[0])
    r06=r2[np.isclose(sig,0.6)]
    rows.append(dict(dataset=k[0],model=k[1],rep=k[2],strategy=k[3],fold=k[4],
        baseline=base, auc_norm=float(trapezoid(r2/base,sig)/(sig.max()-sig.min())) if base!=0 else np.nan,
        r2_06=float(r06[0]) if len(r06) else np.nan))
per=pd.DataFrame(rows)
per.to_csv('per_fold.csv',index=False)
print(len(per), 'cells; gated (baseline>=0.3):', (per.baseline>=0.3).sum())
g=per[per.baseline>=0.3].copy()
g.to_csv('per_fold_gated.csv',index=False)
print(g.groupby('dataset').size())
