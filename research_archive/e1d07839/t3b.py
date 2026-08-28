import importlib.util, sys, numpy as np, os
from pathlib import Path
from sklearn.model_selection import GroupKFold
os.chdir('/Users/apunt/repos/KIRBy/tests')
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); sys.modules['adnr']=m; spec.loader.exec_module(m)

datasets={}
smi,lab=m.load_chembl_herg(); datasets['herg_ki']=(smi,lab)
df=m.download_openadmet()
for name,key,lt in [('logd','LogD',False),('caco2','Caco',True)]:
    col=next(c for c in df.columns if key in c and (('Efflux' in c) if name=='caco2' else True))
    s,l=m.load_openadmet_endpoint(df,col,log_transform=lt); datasets[name]=(s,l)

for name,(smi,lab) in datasets.items():
    g,ng=m.assign_scaffold_groups(smi)
    print(f'\n### {name}: N={len(smi)} scaffolds={ng}')
    gkf=GroupKFold(n_splits=5)
    for fi,(tr,te) in enumerate(gkf.split(smi,lab,g)):
        gf=np.asarray(g)[tr]
        n_val=len(tr)//5
        gv=gf[:n_val]; gt=gf[n_val:]
        n_sc_train=len(np.unique(gt))
        shared=len(set(gv)&set(gt))
        frac_train_mols_sharing_val_scaffold=np.isin(gt,list(set(gv))).mean()
        print(f'  fold{fi}: n_train={len(gt)} inner_scaffolds={n_sc_train} '
              f'(>=5 so GroupKFold path: {n_sc_train>=5}) | val scaffolds={len(set(gv))} '
              f'shared with train={shared} -> {frac_train_mols_sharing_val_scaffold*100:.1f}% of '
              f'training molecules have their scaffold present in the early-stopping val set')
        if fi==0:
            # how much of each INNER held-out fold shares a scaffold with the val set
            fr=[]
            for k,(kp,hd) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(gt)),np.zeros(len(gt)),gt)):
                fr.append(np.isin(gt[hd],list(set(gv))).mean())
            print(f'          inner held-out folds: fraction sharing a scaffold with val = '
                  f'{[f"{x*100:.0f}%" for x in fr]}')
