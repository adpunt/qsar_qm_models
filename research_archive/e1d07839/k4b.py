import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from scipy.integrate import trapezoid
base='/Users/apunt/repos/KIRBy/tests/results/validation/'
df=pd.concat([pd.read_csv(base+d+'/all_results.csv') for d in ['caco2','herg','logd']],ignore_index=True)
rows=[]
for keys,g in df.groupby(['dataset','model','rep','strategy','fold']):
    g=g.sort_values('sigma'); sig=g.sigma.values.astype(float); r2=g.r2.values.astype(float)
    b=r2[np.isclose(sig,0.0)]
    if len(b)==0: continue
    baseline=float(b[0]); r206=r2[np.isclose(sig,0.6)]
    rows.append(dict(zip(['dataset','model','rep','strategy','fold'],keys),baseline_r2=baseline,
        auc_norm=float(trapezoid(r2/baseline,sig)/(sig.max()-sig.min())) if baseline!=0 else np.nan,
        r2_06=float(r206[0]) if len(r206) else np.nan))
CELLS=pd.DataFrame(rows); G=CELLS[CELLS.baseline_r2>=0.3].copy()  # gate only, NO divergent removal
G['m']=G.model.str.replace('-','_'); G['r']=G.rep.str.replace('-','_')
res=[]
for (ds,st),g in G.groupby(['dataset','strategy']):
    gg=g.dropna(subset=['auc_norm'])
    mod=smf.ols('auc_norm ~ C(m)*C(r)',data=gg).fit(); a=anova_lm(mod,typ=1); t=a['sum_sq'].sum()
    res.append(dict(dataset=ds,strategy=st,model=100*a['sum_sq'].iloc[0]/t,rep=100*a['sum_sq'].iloc[1]/t,
                    inter=100*a['sum_sq'].iloc[2]/t,resid=100*a['sum_sq'].iloc[-1]/t))
R=pd.DataFrame(res)
print("NO divergent filter, retention eta2:")
print(R.round(1).to_string(index=False))
print(f"model {R.model.min():.1f}-{R.model.max():.1f}  rep {R.rep.min():.1f}-{R.rep.max():.1f}  inter {R.inter.min():.1f}-{R.inter.max():.1f}")
