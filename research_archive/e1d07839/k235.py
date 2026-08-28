import pandas as pd, numpy as np, itertools
from scipy.stats import spearmanr
K=pd.read_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/kirby_cells.csv')
A=pd.read_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/anova_out.csv')

print("== K1 check: is rep eta2 > model eta2 anywhere on cleanR2? ==")
c=A[A.resp=='cleanR2']
print((c.rep>c.model).sum(),"of",len(c))
print("cleanR2 rep eta2 by dataset:\n",c.groupby('dataset').rep.agg(['min','max']).round(1))
print("cleanR2 model eta2 by dataset:\n",c.groupby('dataset').model.agg(['min','max']).round(1))
print("retention interaction eta2 by dataset:\n",A[A.resp=='retention'].groupby('dataset').inter.agg(['min','max']).round(1))
print("retention: model>rep in", (A[A.resp=='retention'].model>A[A.resp=='retention'].rep).sum(),"of 18")
print("retention: model>interaction in",(A[A.resp=='retention'].model>A[A.resp=='retention'].inter).sum(),"of 18")

print("\n== K2: Spearman(mean retention, model eta2) per dataset across 6 strategies ==")
ret=A[A.resp=='retention']
for ds,g in ret.groupby('dataset'):
    mr=K[K.dataset==ds].groupby('strategy').auc_norm.mean()
    g=g.set_index('strategy')
    common=[s for s in mr.index if s in g.index]
    rho,p=spearmanr(mr[common],g.loc[common,'model'])
    rho2,_=spearmanr(mr[common],g.loc[common,'resid'])
    print(f"{ds}: rho(retention,model_eta2)={rho:.3f} p={p:.3f} | rho(retention,resid)={rho2:.3f}")
    print("   ",{s:round(mr[s],3) for s in common},{s:round(g.loc[s,'model'],1) for s in common})

print("\n== K3: best-by-retention vs best-by-R2@0.6, per dataset x rep x strategy ==")
agree=0;tot=0;perds={}
lgbwin_acc=0;lgbwin_ret=0;ngb_ret=0;ngb_acc=0
for keys,g in K.groupby(['dataset','rep','strategy']):
    m=g.groupby('model')[['auc_norm','r2_06']].mean().dropna()
    if len(m)<2: continue
    tot+=1
    b1=m.auc_norm.idxmax(); b2=m.r2_06.idxmax()
    if b1==b2: agree+=1
    perds[keys[0]]=perds.get(keys[0],[0,0]); perds[keys[0]][1]+=1; perds[keys[0]][0]+= (b1==b2)
    if b2=='LightGBM': lgbwin_acc+=1
    if b1=='LightGBM': lgbwin_ret+=1
    if b1=='NGBoost': ngb_ret+=1
    if b2=='NGBoost': ngb_acc+=1
print(f"agree {agree}/{tot}", perds)
print("LightGBM top-acc",lgbwin_acc,"top-ret",lgbwin_ret,"| NGBoost top-ret",ngb_ret,"top-acc",ngb_acc)
rhos=[]
for keys,g in K.groupby(['dataset','rep','strategy']):
    m=g.groupby('model')[['auc_norm','baseline_r2']].mean().dropna()
    if len(m)>=4: rhos.append(spearmanr(m.auc_norm,m.baseline_r2)[0])
print("median Spearman(retention, cleanR2) across models:",round(np.median(rhos),3),"n=",len(rhos))
# LogD worked example
l=K[(K.dataset=='OpenADMET-LogD')].groupby('model')[['auc_norm','r2_06','baseline_r2']].mean()
print("\nLogD model means (avg over reps+strats+folds):\n",l.round(3).sort_values('auc_norm',ascending=False))
print("hERG GP clean R2:",K[(K.dataset=='ChEMBL-hERG-Ki')&(K.model=='GP')].baseline_r2.mean())

print("\n== K5: strategy-pair vs rep-pair rank agreement ==")
sp=[];rp=[]
for (ds,rep),g in K.groupby(['dataset','rep']):
    piv=g.groupby(['strategy','model']).auc_norm.mean().unstack()
    for a,b in itertools.combinations(piv.index,2):
        x=piv.loc[a].dropna(); y=piv.loc[b].dropna()
        c=x.index.intersection(y.index)
        if len(c)>=4: sp.append(spearmanr(x[c],y[c])[0])
for (ds,st),g in K.groupby(['dataset','strategy']):
    piv=g.groupby(['rep','model']).auc_norm.mean().unstack()
    for a,b in itertools.combinations(piv.index,2):
        x=piv.loc[a].dropna(); y=piv.loc[b].dropna()
        c=x.index.intersection(y.index)
        if len(c)>=4: rp.append(spearmanr(x[c],y[c])[0])
sp=np.array(sp);rp=np.array(rp)
print(f"strategy-pairs n={len(sp)} median={np.median(sp):.3f} IQR={np.percentile(sp,25):.2f}-{np.percentile(sp,75):.2f} min={sp.min():.3f}")
print(f"rep-pairs      n={len(rp)} median={np.median(rp):.3f} IQR={np.percentile(rp,25):.2f}-{np.percentile(rp,75):.2f} min={rp.min():.3f}")
