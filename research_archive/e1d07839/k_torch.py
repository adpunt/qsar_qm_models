import sys; sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
import importlib.util, numpy as np, torch
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)

rs=np.random.RandomState(0); d=8
Xtr=rs.normal(size=(240,d)); Xv=rs.normal(size=(60,d)); Xte=rs.normal(size=(50,d))
w=rs.normal(size=d)
ytr=Xtr@w+7; yv=Xv@w+7; yte=Xte@w+7
SIG=[0.0,0.3]

def run(oof):
    torch.manual_seed(0); np.random.seed(0)
    p,u,ex=m.run_neural_experiment(Xtr,ytr,Xv,yv,Xte,yte,'full-vbll','legacy',SIG,oof_folds=oof)
    return p,u,ex

p0,u0,_=run(0)
p5,u5,e5=run(5)
for s in SIG:
    print(f"sigma={s}: preds identical with/without oof: {np.array_equal(p0[s],p5[s])}  "
          f"max|diff|={np.abs(p0[s]-p5[s]).max():.6g}   unc identical: {np.array_equal(u0[s],u5[s])}")
print("oof_unc nan frac at 0.3:", np.isnan(e5['oof_unc'][0.3]).mean())
