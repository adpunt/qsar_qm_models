import numpy as np, pandas as pd, shutil
from pathlib import Path
S=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad')
root=S/'mergeroot'
if root.exists(): shutil.rmtree(root)
d=root/'gp__logd__pdv__legacy'/'logd'; d.mkdir(parents=True)
rows=[]
sig=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
rng=np.random.RandomState(0)
for f in range(5):
    for s in sig:
        for i in range(20):
            rows.append(dict(split='test',strategy='legacy',sigma=s,fold=f,sample_idx=i,
                             y_true=1.0,y_pred=1.0,uncertainty=0.3,noise_scale=s,
                             noise_pattern=1.0,injected_noise=0.0,dataset='logd',model='GP',rep='PDV'))
# out-of-fold only on outer fold 0, and only 3 of 5 inner folds succeeded ->
# 40% of the rows are NaN, oof_folds_ok = 3
for s in sig:
    for i in range(50):
        nan = i>=30
        rows.append(dict(split='train_oof',strategy='legacy',sigma=s,fold=0,oof_folds_ok=3,
                         sample_idx=i,y_true=1.0,
                         y_pred=(np.nan if nan else 1.0),
                         uncertainty=(np.nan if nan else 0.3),
                         noise_scale=s,noise_pattern=1.0,injected_noise=0.0,
                         dataset='logd',model='GP',rep='PDV'))
pd.DataFrame(rows).to_csv(d/'GP_PDV_uncertainty_values.csv',index=False)
print('wrote', d)
