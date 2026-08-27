import importlib.util, sys, os, numpy as np, pandas as pd, shutil
from pathlib import Path
from sklearn.model_selection import GroupKFold
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); sys.modules['adnr']=m; spec.loader.exec_module(m)

SCR=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/out4')
if SCR.exists(): shutil.rmtree(SCR)

cores=['c1ccccc1','c1ccncc1','c1ccc2ccccc2c1','C1CCCCC1','C1CCNCC1','c1cc[nH]c1','C1CCOC1','c1ccsc1']
N=300
smiles=[cores[i%len(cores)]+'C'*(1+i//len(cores)%6) for i in range(N)]
rng=np.random.RandomState(7)
labels=rng.normal(5,1.2,N)
X=np.column_stack([rng.normal(size=N), np.arange(N,dtype=float), rng.normal(size=N)])

# --- stub GP: name must contain 'GaussianProcess' so _tree_mean_std uses (mean,std)
class GaussianProcessStub:
    def __init__(self,*a,**k): pass
    def fit(self,Xf,yf): self.n=len(yf); self.ym=float(np.mean(yf)); return self
    def predict(self,Xs,return_std=False):
        pm=np.asarray(Xs)[:,1].astype(float).copy()      # scaled molecule id
        ps=np.full(len(Xs), float(self.n))               # = size of the fit set
        return (pm,ps) if return_std else pm
m.GaussianProcessGauche=GaussianProcessStub
m.HAS_GP=True
m.GP_MAX_N=40
m.generate_representations=lambda smiles_list, rep_filter=None: {'PDV': X}

SIG=[0.0,0.5]
res=m.run_dataset('t4', smiles, labels, SCR, model_filter=['GP'], rep_filter=['PDV'],
                  sigma_levels=SIG, gp_reps=['PDV'], unc_strategies='all',
                  oof_folds=5, strategies=['valprop'], oof_outer_folds=1)

f=SCR/'GP_PDV_uncertainty_values.csv'
df=pd.read_csv(f)
print('\n=== written', f, df.shape, sorted(df.split.unique()))
tr=df[df.split=='train_oof']
print('train_oof rows', len(tr), 'folds', sorted(tr.fold.unique()), 'sigmas', sorted(tr.sigma.unique()))

# --- recompute the exact splits the runner used -------------------------
groups,_=m.assign_scaffold_groups(smiles)
gkf=GroupKFold(n_splits=m.N_FOLDS)
fold0_train,_=list(gkf.split(smiles,labels,groups))[0]
n_val=len(fold0_train)//5
tl=np.arange(n_val,len(fold0_train))
y_train=labels[fold0_train][tl]                 # FULL training labels (post val carve)
X_train=X[fold0_train][tl]
gp_row_idx=np.random.RandomState(42).choice(len(X_train), m.GP_MAX_N, replace=False)
print('len(X_train)=',len(X_train),' GP_MAX_N=',m.GP_MAX_N,' subsample=',len(gp_row_idx))
X_gp=X_train[gp_row_idx]; y_gp=y_train[gp_row_idx]
mu1,s1=X_gp[:,1].mean(), X_gp[:,1].std()

probs=[]
for sig in SIG:
    b=tr[(tr.sigma==sig)&(tr.fold==0)].sort_values('sample_idx')
    if len(b)==0: probs.append(f'sigma {sig}: NO BLOCK'); continue
    si=b.sample_idx.values
    if len(si)!=m.GP_MAX_N: probs.append(f'sigma {sig}: {len(si)} rows != GP_MAX_N {m.GP_MAX_N}')
    if len(set(si))!=len(si): probs.append(f'sigma {sig}: DUPLICATE sample_idx')
    if si.max()<m.GP_MAX_N: probs.append(f'sigma {sig}: max sample_idx {si.max()} < GP_MAX_N -> indices are into the SUBSAMPLE')
    if not set(si)==set(gp_row_idx): probs.append(f'sigma {sig}: sample_idx set != gp_row_idx set')
    # y_true must be the clean label of the FULL-train row named by sample_idx
    bad=np.flatnonzero(~np.isclose(b.y_true.values, y_train[si]))
    if len(bad): probs.append(f'sigma {sig}: y_true mismatch on {len(bad)} rows e.g. idx {si[bad[:3]]}')
    # y_pred (= scaled column 1 of the scored molecule) must decode to sample_idx
    dec=np.rint(b.y_pred.values*s1+mu1).astype(int)
    # column1 of X is the GLOBAL molecule id; map sample_idx -> global id
    glob=X_train[si,1].astype(int)
    bad2=np.flatnonzero(dec!=glob)
    if len(bad2): probs.append(f'sigma {sig}: y_pred decodes to the WRONG molecule for {len(bad2)} rows: got {dec[bad2[:5]]} expected {glob[bad2[:5]]}')
    print(f'  sigma={sig}: rows={len(si)} sample_idx[min,max]=({si.min()},{si.max()}) '
          f'unique={len(set(si))} oof_folds_ok={sorted(b.oof_folds_ok.unique())} '
          f'y_pred-decode-matches={len(bad2)==0}')

# --- noise arrays must belong to the SAME molecules ---------------------
inj=m.NoiseInjectorRegression(strategy='valprop', random_state=42)
_ = inj.noise_scale(y_gp,1.0,reference=y_gp)   # tr_pattern: consumes no randomness
_ = inj.noise_scale(labels[list(gkf.split(smiles,labels,groups))[0][1]] if False else y_gp,1.0,reference=y_gp)
yn, sc, eps = inj.inject_verbose(y_gp, 0.5)
b=tr[(tr.sigma==0.5)&(tr.fold==0)]
pos={int(v):i for i,v in enumerate(gp_row_idx)}          # full-train idx -> subsample position
p=np.array([pos[int(v)] for v in b.sample_idx.values])
d_eps=np.abs(b.injected_noise.values-eps[p]).max()
d_sc =np.abs(b.noise_scale.values-sc[p]).max()
print(f'  max|injected_noise - eps[subsample pos]| = {d_eps:.3e}')
print(f'  max|noise_scale    - sc [subsample pos]| = {d_sc:.3e}')
if d_eps>1e-9: probs.append(f'injected_noise misaligned (max diff {d_eps})')
if d_sc >1e-9: probs.append(f'noise_scale misaligned (max diff {d_sc})')
# and prove it would FAIL if the arrays had been indexed by full-train index
alt=np.abs(b.injected_noise.values-eps[np.arange(len(b))]).max()
print(f'  (control) same check against eps[0..n) instead: {alt:.3e}  -> test is discriminating: {alt>1e-6}')

print('\nDEFECT-4 PROBLEMS:', probs if probs else 'NONE')
