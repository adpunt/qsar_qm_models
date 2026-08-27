import sys, importlib.util, numpy as np
sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
from noiseInject.core import NoiseInjectorRegression as New
spec=importlib.util.spec_from_file_location("oldcore","old_core.py"); old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
Old=old.NoiseInjectorRegression
SIG=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
rs=np.random.RandomState(1); ytr=rs.normal(7,1.2,800); yte=rs.normal(7,1.2,200)
for s in ['legacy','quantile','threshold','outlier','hetero','valprop']:
    o=Old(strategy=s,random_state=42); n=New(strategy=s,random_state=42)
    same=True
    for sig in SIG:
        yo = ytr if sig==0.0 else o.inject(ytr,sig)
        if sig==0.0:
            yn=ytr
        else:
            yn,ts,eps = n.inject_verbose(ytr,sig)
            _ = n.noise_scale(yte,sig,reference=ytr)   # interleaved call
        if not np.array_equal(yo,yn): same=False; print("  MISMATCH at sigma",sig)
    # also pattern column
    n2=New(strategy=s,random_state=42)
    pat_tr=n2.noise_scale(ytr,1.0,reference=ytr); pat_te=n2.noise_scale(yte,1.0,reference=ytr)
    print(f"{s:10s} y_noisy identical to old pipeline: {same} | te_pattern unique vals={len(np.unique(pat_te))} tr_pattern unique={len(np.unique(pat_tr))}")
