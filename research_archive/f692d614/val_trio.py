import pandas as pd, numpy as np
SP='/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/f692d614-388b-4b8a-a48f-22b1897f7dae/scratchpad/'
fd=pd.read_csv(SP+'val_fold_metrics.csv')
pd.set_option('display.width',300)
# PDV only (PRIMARY_REP analogue), all 6 strategies, folds as replicates
for ds in ['OpenADMET-LogD','OpenADMET-Caco2_Efflux','ChEMBL-hERG-Ki']:
    d=fd[(fd.dataset==ds)&(fd.rep=='PDV')]
    t=d.groupby(['model','strategy'])[['baseline_r2','r2_s06','auc_norm']].mean().reset_index()
    print('#'*100); print('##',ds,' — PDV representation, mean over 5 folds')
    for col,lab in [('baseline_r2','clean R2'),('r2_s06','R2 @ sigma=0.6'),('auc_norm','auc_norm')]:
        p=t.pivot_table(index='model',columns='strategy',values=col)
        p=p[['legacy','quantile','threshold','hetero','valprop','outlier']]
        print(f'-- {lab}')
        print(p.round(3).sort_values('legacy',ascending=False).to_string()); print()
