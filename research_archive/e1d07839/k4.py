import pandas as pd, numpy as np
base='/Users/apunt/repos/KIRBy/tests/results/validation/'
df=pd.concat([pd.read_csv(base+d+'/all_results.csv') for d in ['caco2','herg','logd']],ignore_index=True)
c=df[np.isclose(df.sigma,0.0)]
# clean R2 per dataset x model x rep, folds averaged
m=c.groupby(['dataset','model','rep']).r2.mean().reset_index()
neural={'DNN','MLP','BNN-Full','VBLL-Full','MLP-BNN-Full','MLP-VBLL-Full'}
trees={'RF','QRF','XGBoost','LightGBM','NGBoost'}
for name,s in [('neural',neural),('trees',trees),('SVM',{'SVM'}),('GP',{'GP'})]:
    g=m[m.model.isin(s)]
    print(f"{name}: cells={len(g)} below0.3={(g.r2<0.3).sum()} below0={(g.r2<0).sum()} min={g.r2.min():.3f}")
print("\nworst neural cells:\n", m[m.model.isin(neural)].nsmallest(8,'r2').to_string(index=False))
print("\nhERG neural by rep:\n", m[(m.dataset=='ChEMBL-hERG-Ki')&(m.model.isin(neural))].pivot(index='model',columns='rep',values='r2').round(2).to_string())
print("\nLogD neural by rep:\n", m[(m.dataset=='OpenADMET-LogD')&(m.model.isin(neural))].pivot(index='model',columns='rep',values='r2').round(2).to_string())
print("\nCaco2 neural by rep:\n", m[(m.dataset=='OpenADMET-Caco2_Efflux')&(m.model.isin(neural))].pivot(index='model',columns='rep',values='r2').round(2).to_string())
print("\nGP cells:\n", m[m.model=='GP'].to_string(index=False))
print("\ndataset N check - unique rows per dataset in source? n/a. sigma0 rows:",c.groupby('dataset').size().to_dict())
