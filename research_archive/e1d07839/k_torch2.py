import sys; sys.path.insert(0,'/Users/apunt/repos/NoiseInject')
import importlib.util, numpy as np, torch
spec=importlib.util.spec_from_file_location('adnr','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rs=np.random.RandomState(0); d=6
Xtr=rs.normal(size=(120,d)); Xv=rs.normal(size=(40,d)); Xte=rs.normal(size=(30,d)); w=rs.normal(size=d)
ytr=Xtr@w+7; yv=Xv@w+7
T=lambda X,y: m.train_neural_regression(X,y,Xv,yv,Xte,model_type='mlp-full-bnn',epochs=5)
# A: sigma-0 fit then sigma-1 fit  (no oof)
torch.manual_seed(0); np.random.seed(0)
a0=T(Xtr,ytr); a1=T(Xtr,ytr+0.3*rs.normal(size=120)*0)  # same data, second call
# B: sigma-0 fit, then 5 "oof" fits, then sigma-1 fit  (oof on)
torch.manual_seed(0); np.random.seed(0)
b0=T(Xtr,ytr)
for _ in range(5): T(Xtr[:96],ytr[:96])
b1=T(Xtr,ytr)
print("first fit identical:", np.array_equal(a0[0],b0[0]))
print("second fit identical (oof off vs on):", np.array_equal(a1[0],b1[0]),
      " max|diff|=", np.abs(a1[0]-b1[0]).max())
