import importlib.util, sys, numpy as np, torch, random
spec = importlib.util.spec_from_file_location('kb','/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m = importlib.util.module_from_spec(spec); sys.modules['kb']=m; spec.loader.exec_module(m)
rs=np.random.RandomState(0); n_tr,n_va,n_te,d=120,40,50,8
Xtr=rs.randn(n_tr,d); ytr=Xtr[:,0]*2+rs.randn(n_tr)*0.3
Xva=rs.randn(n_va,d); yva=Xva[:,0]*2+rs.randn(n_va)*0.3
Xte=rs.randn(n_te,d); yte=Xte[:,0]*2+rs.randn(n_te)*0.3
g=rs.randint(0,12,n_tr)
def run(oof):
    torch.manual_seed(12345); np.random.seed(999); random.seed(777)
    m.run_neural_experiment(Xtr,ytr,Xva,yva,Xte,yte,'deterministic','legacy',[0.0,0.3,0.6],
                            oof_folds=oof, train_groups=g)
    return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())
a=run(0); b=run(3)
print("torch global state identical after run:", torch.equal(a[0],b[0]))
print("numpy global state identical after run:", all(np.array_equal(np.asarray(x,dtype=object),np.asarray(y,dtype=object)) if isinstance(x,tuple) else np.array_equal(x,y) for x,y in zip(a[1],b[1])))
print("python random state identical after run:", a[2]==b[2])
