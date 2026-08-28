import pandas as pd, numpy as np, itertools
from scipy.stats import spearmanr
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet'); d['r2c']=d.r2.clip(lower=-1.0)
cell=d.groupby(['dataset','rep','strategy','model','sigma']).r2c.mean().reset_index()

EQ={'hetero':0.829,'legacy':0.473,'outlier':0.544,'quantile':0.390,'threshold':0.365,'valprop':0.244}

def curve_at(sub,tgt):
    p=sub.pivot_table(index='sigma',columns='model',values='r2c').sort_index()
    return pd.Series({m:np.interp(tgt,p.index.values,p[m].values) for m in p.columns})

def agreement(getter,label):
    per=[]
    for (ds,rep),g in cell.groupby(['dataset','rep']):
        vecs={st:getter(gs,st) for st,gs in g.groupby('strategy')}
        sts=sorted(vecs)
        for a,b in itertools.combinations(sts,2):
            v1,v2=vecs[a].align(vecs[b])
            per.append(dict(dataset=ds,rep=rep,pair=f"{a}|{b}",rho=spearmanr(v1,v2).correlation))
    P=pd.DataFrame(per)
    print(f"\n--- {label}: pairwise Spearman of 13-model rankings between strategies ---")
    print("mean %.4f   median %.4f   min %.3f   n=%d"%(P.rho.mean(),P.rho.median(),P.rho.min(),len(P)))
    print("by dataset:"); print(P.groupby('dataset').rho.agg(['mean','median']).round(3).to_string())
    return P

Pa=agreement(lambda g,st: curve_at(g,0.6), "EQUAL SIGMA (all at 0.6)")
Pb=agreement(lambda g,st: curve_at(g,EQ[st]), "EQUAL DAMAGE (25% rel drop, pooled sigmas)")
Pc=agreement(lambda g,st: curve_at(g,0.0), "CLEAN (sigma=0) reference ceiling")

m=Pa.merge(Pb,on=['dataset','rep','pair'],suffixes=('_eqsig','_eqdam'))
m['delta']=m.rho_eqdam-m.rho_eqsig
print("\n=== change per strategy pair (equal-damage minus equal-sigma), mean over dataset x rep ===")
print(m.groupby('pair').delta.mean().sort_values().round(3).to_string())
print("\noverall mean improvement: %+.4f  (pairs improved: %d/%d)"%(m.delta.mean(),(m.delta>0).sum(),len(m)))
from scipy.stats import wilcoxon
print("Wilcoxon signed-rank on the paired change: p=%.3g"%wilcoxon(m.rho_eqdam,m.rho_eqsig).pvalue)
m.to_csv(S+'q2_rank_agreement.csv',index=False)

# sensitivity: sweep a common sigma and see agreement at each
print("\n=== equal-sigma agreement as a function of the common sigma ===")
for sg in [0.2,0.4,0.6,0.8,1.0]:
    rr=[]
    for (ds,rep),g in cell.groupby(['dataset','rep']):
        vecs={st:curve_at(gs,sg) for st,gs in g.groupby('strategy')}
        for a,b in itertools.combinations(sorted(vecs),2):
            v1,v2=vecs[a].align(vecs[b]); rr.append(spearmanr(v1,v2).correlation)
    print(f"  sigma={sg}: mean rho={np.mean(rr):.4f}")
