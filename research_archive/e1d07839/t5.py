import importlib.util, sys, numpy as np, pandas as pd, shutil
from pathlib import Path
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); sys.modules['adnr']=m; spec.loader.exec_module(m)

cores=['c1ccccc1','c1ccncc1','c1ccc2ccccc2c1','C1CCCCC1','C1CCNCC1','c1cc[nH]c1','C1CCOC1','c1ccsc1']
N=300; smiles=[cores[i%len(cores)]+'C'*(1+i//len(cores)%6) for i in range(N)]
rng=np.random.RandomState(7); labels=rng.normal(5,1.2,N)
X=np.column_stack([rng.normal(size=N), np.arange(N,dtype=float), rng.normal(size=N)])
m.GP_MAX_N=40; m.HAS_GP=True
m.generate_representations=lambda s, rep_filter=None: {'PDV': X}

FAIL_FIRST=[0]; CALLS=[0]
class GaussianProcessStub:
    def __init__(self,*a,**k): pass
    def fit(self,Xf,yf):
        if len(yf)<40:                       # inner (out-of-fold) fit
            CALLS[0]+=1
            if CALLS[0]<=FAIL_FIRST[0]:
                raise RuntimeError(f'simulated inner-fold failure #{CALLS[0]}')
        self.n=len(yf); return self
    def predict(self,Xs,return_std=False):
        pm=np.asarray(Xs)[:,1].astype(float).copy()
        return (pm, np.full(len(Xs),1.0)) if return_std else pm
m.GaussianProcessGauche=GaussianProcessStub

def run(k, tag):
    FAIL_FIRST[0]=k; CALLS[0]=0
    out=Path(f'/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/out5_{tag}')
    if out.exists(): shutil.rmtree(out)
    import io, contextlib
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):
        m.run_dataset('t5', smiles, labels, out, model_filter=['GP'], rep_filter=['PDV'],
                      sigma_levels=[0.0], gp_reps=['PDV'], unc_strategies='all',
                      oof_folds=5, strategies=['legacy'], oof_outer_folds=1)
    log=buf.getvalue()
    f=out/'GP_PDV_uncertainty_values.csv'
    d=pd.read_csv(f); tr=d[d.split=='train_oof']
    print(f'--- {tag}: {k}/5 inner folds forced to fail')
    for L in log.splitlines():
        if '[oof]' in L or 'ERROR' in L: print('   LOG:', L.strip())
    print(f'   train_oof rows written : {len(tr)}')
    if len(tr):
        print(f'   oof_folds_ok values    : {sorted(tr.oof_folds_ok.unique())}')
        print(f'   NaN y_pred / uncertainty: {tr.y_pred.isna().sum()} / {tr.uncertainty.isna().sum()} of {len(tr)}')
    print(f'   test rows written      : {(d.split=="test").sum()}')
    print(f'   oof_folds_ok on TEST rows: {d[d.split=="test"].oof_folds_ok.unique() if "oof_folds_ok" in d else "n/a"}')
    return d

run(0,'ok')
run(2,'partial')
run(5,'all')
