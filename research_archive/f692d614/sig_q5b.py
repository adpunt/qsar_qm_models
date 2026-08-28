import pandas as pd, numpy as np
S='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
M=pd.read_csv(S+'q5_trio.csv')
M=M[(M.auc_norm>-1)&(M.auc_norm<2)]
def reg(y,cols):
    X=np.column_stack([np.ones(len(y))]+[c for c in cols])
    b,*_=np.linalg.lstsq(X,y,rcond=None); r=y-X@b
    return 1-r.var()/y.var(), r.std(ddof=X.shape[1])
print("=== each headline quantity regressed on the other two (fold-level cells) ===")
for tgt,others in [('auc_norm',['clean','r2_06']),('r2_06',['clean','auc_norm']),('clean',['r2_06','auc_norm'])]:
    r2,rs=reg(M[tgt].values,[M[c].values for c in others])
    print("  %-9s ~ %-22s R2=%.3f  residual SD=%.4f  (own SD=%.4f)  unexplained %.0f%%"%(
        tgt,"+".join(others),r2,rs,M[tgt].std(),100*(1-r2)))

print("\n=== replicate (between-fold) noise floor for each quantity ===")
cellkey=['dataset','rep','strategy','model']
for c in ['clean','r2_06','auc_norm']:
    w=M.groupby(cellkey)[c].std(ddof=1)
    print("  %-9s between-fold SD (pooled RMS) = %.4f ; total SD = %.4f"%(c,np.sqrt((w**2).mean()),M[c].std()))

print("\n=== repeat the regression on CELL MEANS (fold noise averaged out, n=5) ===")
C=M.groupby(cellkey)[['clean','r2_06','auc_norm','ret_06']].mean().reset_index()
print("  n cells =",len(C))
for tgt,others in [('auc_norm',['clean','r2_06']),('r2_06',['clean','auc_norm']),('clean',['r2_06','auc_norm'])]:
    r2,rs=reg(C[tgt].values,[C[c].values for c in others])
    print("  %-9s ~ %-22s R2=%.3f  residual SD=%.4f  (own SD=%.4f)  unexplained %.0f%%"%(
        tgt,"+".join(others),r2,rs,C[tgt].std(),100*(1-r2)))
fold_floor=np.sqrt((M.groupby(cellkey)['auc_norm'].std(ddof=1)**2).mean())/np.sqrt(5)
r2,rs=reg(C.auc_norm.values,[C.clean.values,C.r2_06.values])
print("\n  auc_norm cell-mean residual SD = %.4f"%rs)
print("  standard error of a cell-mean auc_norm (fold noise / sqrt5) = %.4f"%fold_floor)
print("  => signal-to-noise of the leftover: %.2f  (residual variance is %.1fx the replicate variance)"%(rs/fold_floor,(rs/fold_floor)**2))
