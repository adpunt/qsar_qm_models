import pandas as pd, numpy as np
q=pd.read_csv('/Users/apunt/repos/qsar_qm_models/data/QM9/raw/gdb9.sdf.csv')
g=q['gap'].values[:10000]*27.2114
print("QM9 gap (eV), first 10000 molecules: n=%d mean=%.3f sd=%.3f min=%.3f max=%.3f"%(len(g),g.mean(),g.std(ddof=0),g.min(),g.max()))
print("quantiles:", {k:round(v,3) for k,v in q['gap'][:10000].mul(27.2114).quantile([0,.05,.1,.25,.5,.75,.9,.95,1]).to_dict().items()})

print("\n--- THRESHOLD, current cut-points high=+1.0 low=-1.0 (2*sigma outside, 0.1*sigma inside) ---")
print("  fraction with y >= +1.0 : %.4f"%np.mean(g>=1.0))
print("  fraction with y <= -1.0 : %.4f"%np.mean(g<=-1.0))
print("  fraction in the mid band (gets 0.1*sigma): %.4f"%np.mean((g<1.0)&(g>-1.0)))
print("  => effective dose is a FLAT 2*sigma on 100% of molecules; the strategy is Gaussian at double sigma.")

print("\n--- QUANTILE, deciles (top 10% and bottom 10% get 2*sigma, mid 80% gets 0.1*sigma) ---")
lo,hi=np.quantile(g,[0.1,0.9])
print("  decile cut values: v10=%.3f eV  v90=%.3f eV"%(lo,hi))
print("  fraction <= v10: %.4f ; >= v90: %.4f ; mid: %.4f"%(np.mean(g<=lo),np.mean(g>=hi),np.mean((g>lo)&(g<hi))))
print("  RMS effective dose = %.3f * sigma"%np.sqrt(np.mean(np.where((g<=lo)|(g>=hi),4.0,0.01))))

print("\n--- OUTLIER, |z|>2 gets 3*sigma, rest 0.1*sigma ---")
z=(g-g.mean())/g.std(ddof=0)
print("  fraction |z|>2: %.4f (Gaussian expectation 0.0455)"%np.mean(np.abs(z)>2))
print("  of those, upper tail %.4f, lower tail %.4f"%(np.mean(z>2),np.mean(z<-2)))
print("  skew=%.3f kurtosis(excess)=%.3f"%(pd.Series(g).skew(),pd.Series(g).kurtosis()))
print("  RMS effective dose = %.3f * sigma"%np.sqrt(np.mean(np.where(np.abs(z)>2,9.0,0.01))))

print("\n--- proposed quantile-based THRESHOLD replacements on QM9 ---")
for p in [0.1,0.2,0.25]:
    lo_,hi_=np.quantile(g,[p,1-p])
    frac=np.mean((g<=lo_)|(g>=hi_))
    print("  p=%.2f -> low=%.3f eV high=%.3f eV, affected=%.4f, RMS dose=%.3f*sigma"%(
        p,lo_,hi_,frac,np.sqrt(frac*4.0+(1-frac)*0.01)))
print("  z-based (|z|>1): low=%.3f high=%.3f affected=%.4f"%(g.mean()-g.std(ddof=0),g.mean()+g.std(ddof=0),np.mean(np.abs(z)>1)))

print("\n--- VALPROP / HETERO effective dose on QM9 (uses |y|) ---")
vp=np.sqrt(np.mean((1+0.05*np.abs(g))**2)); vp2=np.sqrt(np.mean((1+0.1*np.abs(g))**2))
print("  ValueProportional RMS dose: prop=0.05 -> %.3f*sigma ; prop=0.10 -> %.3f*sigma"%(vp,vp2))
he=np.sqrt(np.mean(0.1+0.05*np.abs(g)))
print("  Heteroscedastic RMS dose: sqrt(0.1+0.05|y|) -> %.3f*sigma"%he)
print("  ratio of hetero sigma at max vs min y: %.2f"%(np.sqrt(0.1+0.05*g.max())/np.sqrt(0.1+0.05*g.min())))
