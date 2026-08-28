import sys, numpy as np, torch
sys.path.insert(0,'/Users/apunt/repos/KIRBy/tests')
import importlib.util
spec=importlib.util.spec_from_file_location('m','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
print('HAS_BAYESIAN_TORCH', m.HAS_BAYESIAN_TORCH)
rng=np.random.RandomState(0)
X=rng.randn(120,6).astype(np.float32); y=(X[:,0]*2+rng.randn(120)*.1).astype(np.float64)
Xv,yv=X[:24],y[:24]
def T(Xa,ya): 
    p,u=m.train_neural_regression(Xa,ya,Xv,yv,X,model_type='mlp-full-bnn',epochs=5)
    return np.asarray(p)
torch.manual_seed(0); a0=T(X,y); a1=T(X,y)
torch.manual_seed(0); b0=T(X,y); [T(X[:96],y[:96]) for _ in range(5)]; b1=T(X,y)
print('first fit identical:', np.array_equal(a0,b0))
print('second fit identical (oof off vs on):', np.array_equal(a1,b1), 'max|diff|=', np.abs(a1-b1).max())
