import pandas as pd, numpy as np, itertools
g=pd.read_csv('per_fold_gated.csv')

def eta2(sub, metric):
    # balanced-ish two-way Type I: model, rep, model:rep, residual on folds
    sub=sub.dropna(subset=[metric])
    # keep only models present in >=2 reps and reps present in >=2 models
    while True:
        mc=sub.groupby('model')['rep'].nunique(); rc=sub.groupby('rep')['model'].nunique()
        keep_m=mc[mc>=2].index; keep_r=rc[rc>=2].index
        n0=len(sub); sub=sub[sub.model.isin(keep_m)&sub.rep.isin(keep_r)]
        if len(sub)==n0 or len(sub)==0: break
    if len(sub)<10 or sub.model.nunique()<2 or sub.rep.nunique()<2: return None
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    m=smf.ols(f'{metric} ~ C(model)*C(rep)', data=sub).fit()
    a=sm.stats.anova_lm(m, typ=1)
    tot=a['sum_sq'].sum()
    return dict(model=100*a['sum_sq']['C(model)']/tot, rep=100*a['sum_sq']['C(rep)']/tot,
                inter=100*a['sum_sq']['C(model):C(rep)']/tot, resid=100*a['sum_sq']['Residual']/tot,
                n=len(sub), nm=sub.model.nunique(), nr=sub.rep.nunique())

out=[]
for (ds,st),sub in g.groupby(['dataset','strategy']):
    for metric in ['auc_norm','r2_06','baseline']:
        r=eta2(sub.copy(),metric)
        if r: out.append(dict(dataset=ds,strategy=st,metric=metric,**r))
o=pd.DataFrame(out)
pd.set_option('display.width',200)
for metric in ['auc_norm','r2_06','baseline']:
    print('===',metric)
    print(o[o.metric==metric][['dataset','strategy','model','rep','inter','resid','n','nm','nr']].round(1).to_string(index=False))
o.to_csv('val_eta2.csv',index=False)
