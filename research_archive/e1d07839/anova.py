import pandas as pd, numpy as np
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
K=pd.read_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/kirby_cells.csv')
K['m']=K.model.str.replace('-','_'); K['r']=K.rep.str.replace('-','_')
out=[]
for (ds,st),g in K.groupby(['dataset','strategy']):
    for resp,lbl in [('baseline_r2','cleanR2'),('r2_06','R2@0.6'),('auc_norm','retention')]:
        gg=g.dropna(subset=[resp])
        if gg.m.nunique()<2 or gg.r.nunique()<2: continue
        mod=smf.ols(f'{resp} ~ C(m)*C(r)',data=gg).fit()
        a=anova_lm(mod,typ=1)
        tot=a['sum_sq'].sum()
        out.append(dict(dataset=ds,strategy=st,resp=lbl,
            model=100*a['sum_sq'].iloc[0]/tot, rep=100*a['sum_sq'].iloc[1]/tot,
            inter=100*a['sum_sq'].iloc[2]/tot, resid=100*a['sum_sq'].iloc[-1]/tot, n=len(gg)))
A=pd.DataFrame(out)
pd.set_option('display.width',200)
for lbl in ['cleanR2','R2@0.6','retention']:
    s=A[A.resp==lbl]
    print(f"\n=== {lbl} ===  rep eta2 range {s.rep.min():.1f}-{s.rep.max():.1f} | model eta2 range {s.model.min():.1f}-{s.model.max():.1f}")
    print(s[['dataset','strategy','model','rep','inter','resid','n']].round(1).to_string(index=False))
A.to_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/anova_out.csv',index=False)
print("\nLogD only:")
for lbl in ['cleanR2','R2@0.6','retention']:
    s=A[(A.resp==lbl)&(A.dataset=='OpenADMET-LogD')]
    print(lbl,"rep",round(s.rep.min(),1),"-",round(s.rep.max(),1),"model",round(s.model.min(),1),"-",round(s.model.max(),1))
