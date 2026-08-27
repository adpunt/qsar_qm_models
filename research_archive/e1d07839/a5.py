import pandas as pd, numpy as np
p="/Users/apunt/repos/qsar_qm_models/results/paper_figures_v2/table_validation_auc_full.csv"
df=pd.read_csv(p)
print(df.shape, df.columns.tolist())
print("datasets",df.dataset.unique())
print("models",sorted(df.model.unique()))
print("reps",sorted(df.rep.unique()))
print("strategies",sorted(df.strategy.unique()))
EXC={'sns','randomized_smiles','random_smiles','pdv','morgan'}
for ds in df.dataset.unique():
    s=df[(df.dataset==ds)&(df.strategy=='legacy')&(~df.rep.isin(EXC))]
    print(ds,"n rows",len(s),"unique cells",s.groupby(['model','rep']).ngroups,"max cell size",s.groupby(['model','rep']).size().max())
    print("   reps used:",sorted(s.rep.unique()),"models:",sorted(s.model.unique()))
    print("   baseline_r2 dnn:",s[s.model=='dnn'][['rep','baseline_r2','auc_norm']].to_string(index=False))
