import pandas as pd, numpy as np
from scipy.integrate import trapezoid
from scipy.stats import spearmanr, pearsonr
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
d=pd.read_parquet(S+'val_rerun.parquet')
key=['dataset','rep','strategy','model','fold']
rows=[]
for k,g in d.groupby(key):
    g=g.sort_values('sigma'); s=g.sigma.values; r=g.r2.values
    base=r[0]
    if base<=0.1: continue
    ret=r/base
    an=trapezoid(ret,s)/(s.max()-s.min())
    rows.append(dict(zip(key,k))|{'clean':base,'r2_06':g.loc[g.sigma==0.6,'r2'].iloc[0],
        'ret_06':g.loc[g.sigma==0.6,'r2'].iloc[0]/base,'auc_norm':an})
M=pd.DataFrame(rows)
print("cells with clean R2>0.1: %d (of %d curves)"%(len(M),d.groupby(key).ngroups))
# match the script's artifact filter band
print("auc_norm quantiles:",{k:round(v,3) for k,v in M.auc_norm.quantile([0,.01,.25,.5,.75,.99,1]).to_dict().items()})
M=M[(M.auc_norm>-1)&(M.auc_norm<2)].copy()   # the script filters extremes; use a wide band
print("after wide artifact filter: %d"%len(M))
M.to_csv(S+'q5_trio.csv',index=False)

print("\n=== Pearson / Spearman among the three headline quantities (all cells) ===")
for a,b in [('clean','r2_06'),('clean','auc_norm'),('r2_06','auc_norm')]:
    print("  %-18s pearson %+.3f   spearman %+.3f"%(a+" vs "+b,pearsonr(M[a],M[b])[0],spearmanr(M[a],M[b])[0]))
print("\n  and the retention form: ret_06 = R2(0.6)/R2(0)")
for a,b in [('clean','ret_06'),('ret_06','auc_norm')]:
    print("  %-18s pearson %+.3f   spearman %+.3f"%(a+" vs "+b,pearsonr(M[a],M[b])[0],spearmanr(M[a],M[b])[0]))

print("\n=== within dataset ===")
for ds,g in M.groupby('dataset'):
    print(" ",ds, {f"{a}~{b}":round(pearsonr(g[a],g[b])[0],3) for a,b in [('clean','r2_06'),('clean','auc_norm'),('r2_06','auc_norm'),('ret_06','auc_norm')]})

def reg(y,X,names):
    X=np.column_stack([np.ones(len(y))]+X)
    beta,*_=np.linalg.lstsq(X,y,rcond=None)
    pred=X@beta; res=y-pred
    r2=1-res.var()/y.var()
    return beta,r2,res.std(ddof=X.shape[1])

print("\n=== regression of auc_norm on (clean R2, R2 at 0.6) ===")
y=M.auc_norm.values
b,r2,rs=reg(y,[M.clean.values,M.r2_06.values],['clean','r2_06'])
print("  auc_norm = %.4f + %.4f*clean + %.4f*R2(0.6)"%tuple(b))
print("  R2 of fit = %.4f   residual SD = %.4f   (auc_norm SD = %.4f)"%(r2,rs,y.std()))
print("  => %.1f%% of auc_norm variance is NOT explained by the pair"%(100*(1-r2)))

print("\n=== same, adding the retention form R2(0.6)/R2(0) ===")
b2,r22,rs2=reg(y,[M.clean.values,M.r2_06.values,M.ret_06.values],['clean','r2_06','ret_06'])
print("  R2 of fit = %.4f   residual SD = %.4f  => unexplained %.1f%%"%(r22,rs2,100*(1-r22)))

print("\n=== auc_norm on the retention at 0.6 alone ===")
b3,r23,rs3=reg(y,[M.ret_06.values],['ret_06'])
print("  auc_norm = %.4f + %.4f*ret(0.6);  R2 = %.4f  residual SD = %.4f"%(b3[0],b3[1],r23,rs3))

print("\n=== which single sigma's retention best predicts auc_norm? ===")
full=d.merge(M[key],on=key)
for sg in sorted(d.sigma.unique()):
    if sg==0: continue
    t=full[full.sigma==sg].merge(M[key+['clean','auc_norm']],on=key)
    ret=t.r2/t.clean
    print("   sigma=%.1f : pearson(ret, auc_norm) = %+.3f ; spearman = %+.3f"%(sg,pearsonr(ret,t.auc_norm)[0],spearmanr(ret,t.auc_norm)[0]))

print("\n=== do clean R2 and auc_norm rank models differently? (per dataset x rep x strategy) ===")
rr=[]
for k,g in M.groupby(['dataset','rep','strategy']):
    mm=g.groupby('model')[['clean','r2_06','auc_norm']].mean()
    if len(mm)<5: continue
    rr.append(dict(clean_auc=spearmanr(mm.clean,mm.auc_norm)[0],
                   clean_r206=spearmanr(mm.clean,mm.r2_06)[0],
                   r206_auc=spearmanr(mm.r2_06,mm.auc_norm)[0]))
RR=pd.DataFrame(rr)
print(RR.mean().round(3).to_string()); print("n cells:",len(RR))
