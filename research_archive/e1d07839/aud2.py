import glob, numpy as np, pandas as pd
from scipy import stats
files=sorted(glob.glob('/Users/apunt/repos/KIRBy/tests/results_server/validation_rerun/*/*/*_uncertainty_values.csv'))
R=[]
for f in files:
    d=pd.read_csv(f); p=f.split('/'); ds=p[-2]; m,rep=p[-1].replace('_uncertainty_values.csv','').split('_',1)
    for s,g in d.groupby('sigma'):
        err=np.abs(g.y_true-g.y_pred).values; unc=g.uncertainty.values
        R.append(dict(ds=ds,model=m,rep=rep,sigma=s,mu=unc.mean(),me=err.mean(),
            cv_unc=unc.std()/unc.mean(), iqr_ratio=(np.percentile(unc,75)-np.percentile(unc,25))/np.median(unc),
            ratio=unc.mean()/err.mean(),
            r2=1-((g.y_true-g.y_pred)**2).sum()/((g.y_true-g.y_true.mean())**2).sum()))
R=pd.DataFrame(R)
pd.set_option('display.width',250)
print("mean_unc / mean_|err| by sigma (median over 27 cfgs):")
print(R.groupby('sigma')[['ratio','cv_unc','iqr_ratio']].median().to_string())
print("\nratio at s=0 range: %.3f-%.3f ; at s=1: %.3f-%.3f"%(R[R.sigma==0].ratio.min(),R[R.sigma==0].ratio.max(),R[R.sigma==1].ratio.min(),R[R.sigma==1].ratio.max()))
print("\ncv_unc s0 vs s1 per model:")
print(R[R.sigma.isin([0.0,1.0])].groupby(['model','sigma'])[['cv_unc','iqr_ratio']].median().to_string())
print("\ncv_unc declines s0->s1 in %d/27"%sum(R[(R.ds==a)&(R.model==b)&(R.rep==c)&(R.sigma==1.0)].cv_unc.iloc[0] < R[(R.ds==a)&(R.model==b)&(R.rep==c)&(R.sigma==0.0)].cv_unc.iloc[0] for a,b,c in R[['ds','model','rep']].drop_duplicates().values))
print("\nerr_ratio near 1 cfgs:")
D=pd.read_csv('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/D.csv')
print(D.nsmallest(5,'err_ratio')[['ds','model','rep','unc_ratio','err_ratio','r2_0','r2_1']].to_string())
print("\nGP only (3 cfgs):"); print(D[D.model=='GP'][['ds','rep','unc_ratio','err_ratio','rho0','rho1','cov1_0','cov1_1','r2_0','r2_1']].to_string())
