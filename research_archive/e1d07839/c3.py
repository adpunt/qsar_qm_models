import pandas as pd, numpy as np
import statsmodels.formula.api as smf, statsmodels.api as sm
g=pd.read_csv('per_fold_gated.csv')
def eta2(sub, metric):
    sub=sub.dropna(subset=[metric])
    while True:
        mc=sub.groupby('model')['rep'].nunique(); rc=sub.groupby('rep')['model'].nunique()
        sub2=sub[sub.model.isin(mc[mc>=2].index)&sub.rep.isin(rc[rc>=2].index)]
        if len(sub2)==len(sub) or len(sub2)==0: sub=sub2; break
        sub=sub2
    if len(sub)<10 or sub.model.nunique()<2 or sub.rep.nunique()<2: return None
    m=smf.ols(f'{metric} ~ C(model)*C(rep)',data=sub).fit(); a=sm.stats.anova_lm(m,typ=1); t=a['sum_sq'].sum()
    return dict(model=100*a['sum_sq']['C(model)']/t, rep=100*a['sum_sq']['C(rep)']/t,
                inter=100*a['sum_sq']['C(model):C(rep)']/t, resid=100*a['sum_sq']['Residual']/t,
                n=len(sub), nm=sub.model.nunique(), nr=sub.rep.nunique())
# variant B: drop diverged folds (any r2 below -0.5 in the curve proxied by auc_norm<0)
gb=g[(g.auc_norm>=0)&(g.r2_06>=-0.5)]
print('rows kept',len(gb),'of',len(g))
out=[]
for (ds,st),sub in gb.groupby(['dataset','strategy']):
    for metric in ['auc_norm','r2_06']:
        r=eta2(sub.copy(),metric)
        if r: out.append(dict(dataset=ds,strategy=st,metric=metric,**r))
o=pd.DataFrame(out); pd.set_option('display.width',200)
for m in ['auc_norm','r2_06']:
    print('===',m,'(divergent folds removed)')
    print(o[o.metric==m][['dataset','strategy','model','rep','inter','resid','n','nm','nr']].round(1).to_string(index=False))
o.to_csv('val_eta2_filtered.csv',index=False)
