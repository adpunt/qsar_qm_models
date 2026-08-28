import pandas as pd, numpy as np, itertools
from scipy.stats import spearmanr, wilcoxon
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet'); d['r2c']=d.r2.clip(lower=-1.0)
key=['dataset','rep','strategy','model','fold']
d=d.join(d[d.sigma==0].set_index(key).r2c.rename('b'),on=key)
u=d[d.b>0.1].copy(); u['reldrop']=(u.b-u.r2c)/u.b
c=u.pivot_table(index='sigma',columns='strategy',values='reldrop',aggfunc='mean')
avg=c.mean(1)
print("mean relative drop AVERAGED over the six strategies, by sigma:")
print(avg.round(4).to_string())
ctrl=np.interp(0.25,avg.values,avg.index.values)
print("\ncommon sigma delivering the same 25%% AVERAGE damage: %.3f"%ctrl)

cell=d.groupby(['dataset','rep','strategy','model','sigma']).r2c.mean().reset_index()
EQ={'hetero':0.829,'legacy':0.473,'outlier':0.544,'quantile':0.390,'threshold':0.365,'valprop':0.244}
def curve_at(sub,t):
    p=sub.pivot_table(index='sigma',columns='model',values='r2c').sort_index()
    return pd.Series({m:np.interp(t,p.index.values,p[m].values) for m in p.columns})
def agree(getter):
    per=[]
    for (ds,rep),g in cell.groupby(['dataset','rep']):
        v={st:getter(gs,st) for st,gs in g.groupby('strategy')}
        for a,b in itertools.combinations(sorted(v),2):
            x,y=v[a].align(v[b]); per.append(dict(dataset=ds,rep=rep,pair=f"{a}|{b}",rho=spearmanr(x,y).correlation))
    return pd.DataFrame(per)
A=agree(lambda g,st: curve_at(g,ctrl))
B=agree(lambda g,st: curve_at(g,EQ[st]))
print("\nEQUAL-SIGMA at the damage-matched control sigma %.3f : mean rho = %.4f"%(ctrl,A.rho.mean()))
print("EQUAL-DAMAGE (per-strategy sigmas)                    : mean rho = %.4f"%B.rho.mean())
m=A.merge(B,on=['dataset','rep','pair'],suffixes=('_ctrl','_eqdam'))
print("difference %+.4f, improved in %d/%d pairs, Wilcoxon p=%.3g"%(
   (m.rho_eqdam-m.rho_ctrl).mean(),((m.rho_eqdam-m.rho_ctrl)>0).sum(),len(m),
   wilcoxon(m.rho_eqdam,m.rho_ctrl).pvalue))
print("\nby dataset:"); print(m.groupby('dataset')[['rho_ctrl','rho_eqdam']].mean().round(3).to_string())
