import pandas as pd, numpy as np, itertools, shutil
from pathlib import Path
ROOT = Path('faketree/results/uncertainty_rerun')
if ROOT.exists(): shutil.rmtree('faketree')
MODELS = {'QRF':'qrf','NGBoost':'ngboost','GP':'gp','BNN-Full':'bnn_full','VBLL-Full':'vbll_full',
          'MLP-BNN-Full':'mlp_bnn_full','MLP-VBLL-Full':'mlp_vbll_full'}
DATASETS=['logd','caco2','herg_ki']
DSDIR={'logd':'logd','caco2':'caco2','herg_ki':'herg'}   # <-- as run_dataset writes them
DSNAME={'logd':'OpenADMET-LogD','caco2':'OpenADMET-Caco2','herg_ki':'ChEMBL-hERG-Ki'}
REPS=['ECFP4','PDV','SNS','MHG-GNN-pretrained']
STRATS=['legacy','outlier','quantile','hetero','threshold','valprop']
SIG=[0.0,0.5,1.0]
NF=5
for model,slug in MODELS.items():
    for ds in DATASETS:
        for rep in REPS:
            rep_slug=rep.lower().replace('-','')
            for st in STRATS:
                d=ROOT/f"{slug}__{ds}__{rep_slug}__{st}"/DSDIR[ds]
                d.mkdir(parents=True,exist_ok=True)
                rows=[]
                for f in range(NF):
                    for s in SIG:
                        rows.append(dict(sigma=s,r2=0.5,rmse=1.0,mae=0.8,spearman=0.6,
                                         model=model,rep=rep,strategy=st,fold=f,dataset=DSNAME[ds]))
                pd.DataFrame(rows).to_csv(d/'all_results.csv',index=False)
                pd.DataFrame([dict(dataset=DSNAME[ds],model=model,rep=rep,strategy=st,
                                   baseline_r2=0.5,baseline_r2_std=0.01,baseline_rmse=1.0,
                                   auc_norm=0.7,weibull_tau=1.0,weibull_beta=1.0,
                                   **{'retention_0.5':0.8,'retention_1.0':0.6})]).to_csv(d/'summary.csv',index=False)
                u=[]
                for sp in ('test','train_oof'):
                    for f in range(NF):
                        for s in SIG:
                            for i in range(4):
                                u.append(dict(split=sp,strategy=st,sigma=s,fold=f,sample_idx=i,
                                              y_true=1.0,y_pred=1.0,uncertainty=0.2,
                                              noise_scale=0.1,injected_noise=0.0,
                                              dataset=DSNAME[ds],model=model,rep=rep))
                fname=f"{model.replace('-','')}_{rep.replace('-','')}_uncertainty_values.csv"
                pd.DataFrame(u).to_csv(d/fname,index=False)
print("tree built")
