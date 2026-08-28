import sys, shutil, numpy as np, pandas as pd
from pathlib import Path
sys.argv=['x']
import importlib.util
spec=importlib.util.spec_from_file_location('anr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

rng=np.random.RandomState(0)
N=300
# 30 distinct scaffolds x 10 substituent variants
cores=['c1ccccc1','c1ccncc1','C1CCCCC1','c1ccc2ccccc2c1','c1cc2ccccc2[nH]1','C1CCNCC1',
       'c1ccc(-c2ccccc2)cc1','c1csc(n1)','O=C1CCCN1','c1ccc2[nH]ncc2c1']
smiles=[]
for i in range(N):
    c=cores[i%len(cores)]
    tail='C'*(1+(i//len(cores))%5)
    smiles.append(c.replace('c1cc','c1cc',1)+tail if not c.endswith('(n1)') else 'c1csc(n1)'+tail)
smiles=np.array(smiles)
labels=rng.normal(3.0,1.2,N)

D=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/smokeout')
if D.exists(): shutil.rmtree(D)

# tiny cheap representation, bypass RDKit descriptor generation
def fake_reps(smiles_list, rep_filter=None):
    r=np.random.RandomState(1).normal(0,1,(len(smiles_list),16))
    y=labels.reshape(-1,1)
    X=np.hstack([r, y*0.7+np.random.RandomState(2).normal(0,0.5,(len(smiles_list),1))])
    out={}
    for name in (rep_filter or ['ECFP4']):
        out[name]=X
    return out
m.generate_representations=fake_reps
m.N_FOLDS=3

RES=D/'qrf__logd__ecfp4__outlier'
s=m.run_dataset('OpenADMET-LogD', smiles, labels, RES/'logd',
                model_filter=['QRF'], rep_filter=['ECFP4'],
                sigma_levels=[0.0,0.5,1.0],
                unc_strategies='all', oof_folds=3, strategies=['outlier'])
print("== files ==")
for f in sorted((RES/'logd').glob('*.csv')): print("  ",f.name, sum(1 for _ in open(f))-1,"rows")
u=pd.read_csv(RES/'logd'/'QRF_ECFP4_uncertainty_values.csv')
print(u.groupby(['split','strategy'])['fold'].agg(['nunique','count']))
print("sigmas:",sorted(u['sigma'].unique()))
print("cols:",list(u.columns))
print("test noise_scale distinct:", u[u.split=='test'].groupby('sigma')['noise_scale'].nunique().to_dict())
print("oof injected_noise nonzero frac:", (u[u.split=='train_oof']['injected_noise']!=0).mean())
print("oof uncertainty NaN frac:", u[u.split=='train_oof']['uncertainty'].isna().mean())
