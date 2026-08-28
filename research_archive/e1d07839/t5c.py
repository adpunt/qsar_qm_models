import numpy as np, pandas as pd, shutil
from pathlib import Path
S=Path('/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad')
root=S/'mergeroot2'
if root.exists(): shutil.rmtree(root)
sig=[0.0,0.5]

def test_rows(model):
    return [dict(split='test',strategy='legacy',sigma=s,fold=f,sample_idx=i,y_true=1.0,
                 y_pred=2.0,uncertainty=0.3,noise_scale=s,noise_pattern=1.0,
                 injected_noise=0.0,dataset='logd',model=model,rep='PDV')
            for f in range(2) for s in sig for i in range(3)]
def oof_rows(model):
    return [dict(split='train_oof',strategy='legacy',sigma=s,fold=0,oof_folds_ok=5,
                 sample_idx=i,y_true=1.0,y_pred=2.0,uncertainty=0.3,noise_scale=s,
                 noise_pattern=1.0,injected_noise=0.9,dataset='logd',model=model,rep='PDV')
            for s in sig for i in range(3)]

# task A: healthy -> has oof_folds_ok  (column order exactly as _flush_uncertainties produces)
a=pd.concat([pd.DataFrame(test_rows('QRF')),pd.DataFrame(oof_rows('QRF'))],ignore_index=True)
# task B: every inner fold failed -> the oof block was skipped -> NO oof_folds_ok column
b=pd.DataFrame(test_rows('GP'))
for name,df,mdl in [('qrf__logd__pdv__legacy',a,'QRF'),('gp__logd__pdv__legacy',b,'GP')]:
    d=root/name/'logd'; d.mkdir(parents=True)
    df.to_csv(d/f'{mdl}_PDV_uncertainty_values.csv',index=False)
    print(name,'columns:',list(df.columns))
