import os, glob, hashlib
import pandas as pd, numpy as np
roots = ["/Users/apunt/repos/KIRBy", "/Users/apunt/repos/qsar_qm_models"]
files=[]
for r in roots:
    for dp,dn,fn in os.walk(r):
        for f in fn:
            if f.endswith("_uncertainty_values.csv") and not f.startswith("._"):
                files.append(os.path.join(dp,f))
files.sort()
rows=[]
for p in files:
    try:
        df=pd.read_csv(p)
    except Exception as e:
        rows.append((p,"READ_ERROR:"+str(e),"","","",""))
        continue
    cols=",".join(df.columns)
    n=len(df)
    sig = sorted(df['sigma'].unique().tolist()) if 'sigma' in df else None
    mdl = sorted(df['model'].unique().tolist()) if 'model' in df else None
    rep = sorted(df['representation'].unique().tolist()) if 'representation' in df else None
    extra=""
    # y_true consistency across sigma
    ycol = 'y_true' if 'y_true' in df else ('y_true_original' if 'y_true_original' in df else None)
    idcol = 'sample_idx' if 'sample_idx' in df else ('sample_id' if 'sample_id' in df else None)
    ychk=""
    if ycol and idcol and sig and len(sig)>1:
        piv = df.pivot_table(index=idcol, columns='sigma', values=ycol, aggfunc='first')
        same = bool(piv.nunique(axis=1).max()==1)
        ychk = f"y_true_const_across_sigma={same}"
        if not same:
            ychk += f" (maxrange={float((piv.max(axis=1)-piv.min(axis=1)).max()):.4g})"
        # per-sigma counts
        cnt = df.groupby('sigma').size().unique().tolist()
        ychk += f"; rows_per_sigma={cnt}"
        # duplicated ids within a sigma -> folds/reps
        dup = df.groupby(['sigma',idcol]).size().max()
        ychk += f"; max_dup_per(sigma,id)={int(dup)}"
    if 'y_true_noisy' in df and 'y_true_original' in df:
        d=(df['y_true_noisy']-df['y_true_original']).abs()
        ychk += f"; noisy!=orig_frac={float((d>1e-12).mean()):.3f}"
    rows.append((p,cols,n,sig,mdl,rep,ychk))
for r in rows:
    print("FILE:",r[0])
    print("  cols:",r[1])
    print("  nrows:",r[2])
    print("  sigma:",r[3])
    print("  model:",r[4],"| rep:",r[5])
    print("  chk:",r[6])
