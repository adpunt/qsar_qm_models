import glob, numpy as np, pandas as pd
from scipy import stats
files = sorted(glob.glob('/Users/apunt/repos/KIRBy/tests/results_server/validation_rerun/*/*/*_uncertainty_values.csv'))
rows=[]
percfg={}
for f in files:
    d = pd.read_csv(f)
    parts=f.split('/'); ds=parts[-2]; name=parts[-1].replace('_uncertainty_values.csv','')
    mdl,rep=name.split('_',1)
    # bit-identical y_true check
    gs={s:g.sort_values('sample_idx')['y_true'].values for s,g in d.groupby('sigma')}
    ks=sorted(gs); ident=all(np.array_equal(gs[ks[0]],gs[k]) for k in ks)
    rec={'ds':ds,'model':mdl,'rep':rep,'n_sigma':len(ks),'n':len(gs[ks[0]]),'ytrue_identical':ident}
    per=[]
    for s in ks:
        g=d[d.sigma==s]
        err=np.abs(g.y_true-g.y_pred); unc=g.uncertainty.values
        r2=1-((g.y_true-g.y_pred)**2).sum()/((g.y_true-g.y_true.mean())**2).sum()
        rho=stats.spearmanr(unc,err)
        cov1=float((err<=unc).mean()); cov2=float((err<=2*unc).mean())
        per.append(dict(sigma=s,mean_unc=unc.mean(),mean_err=err.mean(),r2=r2,rho=rho.correlation,p=rho.pvalue,cov1=cov1,cov2=cov2))
    p=pd.DataFrame(per); percfg[(ds,mdl,rep)]=p
    b=p[p.sigma==0.0].iloc[0]; t=p[p.sigma==1.0].iloc[0]
    rec.update(unc_ratio=t.mean_unc/b.mean_unc, err_ratio=t.mean_err/b.mean_err,
        rho0=b.rho,rho1=t.rho,p1=t.p,cov1_0=b.cov1,cov1_1=t.cov1,cov2_0=b.cov2,cov2_1=t.cov2,
        r2_0=b.r2,r2_1=t.r2,r2_min=p.r2.min(),
        sp_unc=stats.spearmanr(p.sigma,p.mean_unc).correlation,
        sp_rho=stats.spearmanr(p.sigma,p.rho).correlation)
    rows.append(rec)
D=pd.DataFrame(rows)
pd.set_option('display.width',250,'display.max_columns',50)
print(D[['ds','model','rep','n_sigma','n','ytrue_identical']].to_string())
print("\n== U1 ==")
print("unc_ratio median %.3f range %.3f-%.3f"%(D.unc_ratio.median(),D.unc_ratio.min(),D.unc_ratio.max()))
print("err_ratio median %.3f range %.3f-%.3f"%(D.err_ratio.median(),D.err_ratio.min(),D.err_ratio.max()))
print("unc>err in %d/%d"%((D.unc_ratio>D.err_ratio).sum(),len(D)))
print("wilcoxon", stats.wilcoxon(D.unc_ratio,D.err_ratio))
print("monotone unc spearman==1 in %d/%d"%((D.sp_unc>0.999).sum(),len(D)))
print("\n== U2 ==")
print("rho0 median %.3f range %.3f-%.3f"%(D.rho0.median(),D.rho0.min(),D.rho0.max()))
print("rho1 median %.3f range %.3f-%.3f"%(D.rho1.median(),D.rho1.min(),D.rho1.max()))
print("declining %d/%d ; median sp_rho %.3f"%((D.rho1<D.rho0).sum(),len(D),D.sp_rho.median()))
print("wilcoxon", stats.wilcoxon(D.rho0,D.rho1))
H=D[D.r2_min>=0.3]; print("healthy n=%d rho0 med %.3f rho1 med %.3f decline %d/%d"%(len(H),H.rho0.median(),H.rho1.median(),(H.rho1<H.rho0).sum(),len(H)))
print("\n== U3 ==")
print("cov1 s0 med %.3f range %.3f-%.3f ; s1 med %.3f range %.3f-%.3f ; rising %d/%d"%(D.cov1_0.median(),D.cov1_0.min(),D.cov1_0.max(),D.cov1_1.median(),D.cov1_1.min(),D.cov1_1.max(),(D.cov1_1>D.cov1_0).sum(),len(D)))
print("cov2 s0 med %.3f ; s1 med %.3f"%(D.cov2_0.median(),D.cov2_1.median()))
print("healthy cov1 %.3f -> %.3f"%(H.cov1_0.median(),H.cov1_1.median()))
print("\nneg r2 at s=1:"); print(D[D.r2_1<0][['ds','model','rep','r2_0','r2_1','cov1_1']].to_string())
D.to_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/D.csv',index=False)
print("\nper-model U1/U2 breakdown")
print(D.groupby('model')[['unc_ratio','err_ratio','rho0','rho1','cov1_0','cov1_1']].median().to_string())
print("\nper-dataset")
print(D.groupby('ds')[['unc_ratio','err_ratio','rho0','rho1','cov1_0','cov1_1']].median().to_string())
