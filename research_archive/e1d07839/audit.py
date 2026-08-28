import pandas as pd, numpy as np, itertools
from scipy.integrate import trapezoid
from scipy.stats import spearmanr

base='/Users/apunt/repos/KIRBy/tests/results/validation/'
dfs=[pd.read_csv(base+d+'/all_results.csv') for d in ['caco2','herg','logd']]
df=pd.concat(dfs,ignore_index=True)
print("rows",len(df),"models",df.model.nunique(),"reps",df.rep.nunique(),"strats",df.strategy.nunique(),"folds",sorted(df.fold.unique()))
print(sorted(df.model.unique()))
print(sorted(df.rep.unique()))

rows=[]
for keys,g in df.groupby(['dataset','model','rep','strategy','fold']):
    g=g.sort_values('sigma')
    sig=g.sigma.values.astype(float); r2=g.r2.values.astype(float)
    b=r2[np.isclose(sig,0.0)]
    if len(b)==0 or len(sig)<3: continue
    baseline=float(b[0])
    r206=r2[np.isclose(sig,0.6)]
    rows.append(dict(zip(['dataset','model','rep','strategy','fold'],keys),
        baseline_r2=baseline,
        auc_norm=float(trapezoid(r2/baseline,sig)/(sig.max()-sig.min())) if baseline!=0 else np.nan,
        r2_06=float(r206[0]) if len(r206) else np.nan))
cells=pd.DataFrame(rows)
print("\ntotal cells",len(cells))
gated=cells[cells.baseline_r2>=0.3].copy()
print("after baseline>=0.3 gate:",len(gated))
div=(gated.auc_norm<0)|(gated.r2_06<-0.5)
print("divergent removed:",div.sum(),"kept:",(~div).sum())
K=gated[~div].copy()
K.to_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/kirby_cells.csv',index=False)
