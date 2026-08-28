import importlib.util, sys, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); sys.modules['adnr']=m; spec.loader.exec_module(m)
SCR=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/out4')
cores=['c1ccccc1','c1ccncc1','c1ccc2ccccc2c1','C1CCCCC1','C1CCNCC1','c1cc[nH]c1','C1CCOC1','c1ccsc1']
N=300; smiles=[cores[i%len(cores)]+'C'*(1+i//len(cores)%6) for i in range(N)]
rng=np.random.RandomState(7); labels=rng.normal(5,1.2,N)
X=np.column_stack([rng.normal(size=N), np.arange(N,dtype=float), rng.normal(size=N)])
groups,_=m.assign_scaffold_groups(smiles)
tr_idx,_=list(GroupKFold(n_splits=5).split(smiles,labels,groups))[0]
n_val=len(tr_idx)//5; tl=np.arange(n_val,len(tr_idx))
y_train=labels[tr_idx][tl]
GP_MAX_N=40
gp_row_idx=np.random.RandomState(42).choice(len(y_train),GP_MAX_N,replace=False)
y_gp=y_train[gp_row_idx]

df=pd.read_csv(SCR/'GP_PDV_uncertainty_values.csv')
b=df[(df.split=='train_oof')&(df.sigma==0.5)&(df.fold==0)]
si=b.sample_idx.values.astype(int)
print('row order == gp_row_idx order:', np.array_equal(si, gp_row_idx))
print('CHECK  max|y_true - y_train[sample_idx]|      =', np.abs(b.y_true.values-y_train[si]).max())
print('CTRL-A max|y_true - y_train[0..n)|  (buggy)   =', np.abs(b.y_true.values-y_train[:GP_MAX_N]).max())
print('CTRL-B max|y_true - y_gp[position]| (=CHECK)  =', np.abs(b.y_true.values-y_gp).max())
# noise_scale must be the valprop scale of THIS row's own label
inj=m.NoiseInjectorRegression(strategy='valprop',random_state=42)
sc_from_ytrue=inj.noise_scale(b.y_true.values,0.5,reference=y_gp)
print('CHECK  max|noise_scale - scale(y_true_of_row)| =', np.abs(b.noise_scale.values-sc_from_ytrue).max())
sc_wrong=inj.noise_scale(y_train[:GP_MAX_N],0.5,reference=y_gp)
print('CTRL   max|noise_scale - scale(y_train[0..n))| =', np.abs(b.noise_scale.values-sc_wrong).max())
# injected_noise must be consistent with the scale: |eps| ~ scale
r=np.abs(b.injected_noise.values)/np.maximum(b.noise_scale.values,1e-12)
print('|eps|/scale  min/median/max =', r.min(), np.median(r), r.max())
