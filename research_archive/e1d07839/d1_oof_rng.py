import importlib.util, sys, numpy as np, torch, random
spec = importlib.util.spec_from_file_location('kb', '/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py')
m = importlib.util.module_from_spec(spec); sys.modules['kb'] = m; spec.loader.exec_module(m)

rs = np.random.RandomState(0)
n_tr, n_va, n_te, d = 120, 40, 50, 8
Xtr = rs.randn(n_tr, d); ytr = Xtr[:, 0] * 2 + rs.randn(n_tr) * 0.3
Xva = rs.randn(n_va, d); yva = Xva[:, 0] * 2 + rs.randn(n_va) * 0.3
Xte = rs.randn(n_te, d); yte = Xte[:, 0] * 2 + rs.randn(n_te) * 0.3
groups = rs.randint(0, 12, n_tr)
sigmas = [0.0, 0.3, 0.6]

MT = sys.argv[1] if len(sys.argv) > 1 else 'full-vbll'

def run(oof):
    torch.manual_seed(12345)
    np.random.seed(999)
    random.seed(777)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(12345)
    return m.run_neural_experiment(Xtr, ytr, Xva, yva, Xte, yte,
                                   MT, 'legacy', sigmas,
                                   oof_folds=oof, train_groups=groups)

p0, u0, e0 = run(0)
p3, u3, e3 = run(3)

print(f"model_type={MT}")
ok = True
for s in sigmas:
    dp = np.max(np.abs(np.asarray(p0[s], float) - np.asarray(p3[s], float)))
    same_p = np.array_equal(np.asarray(p0[s]), np.asarray(p3[s]))
    if u0[s] is None:
        du, same_u = 'n/a', True
    else:
        du = np.max(np.abs(np.asarray(u0[s], float) - np.asarray(u3[s], float)))
        same_u = np.array_equal(np.asarray(u0[s]), np.asarray(u3[s]))
    print(f"  sigma={s}: pred bitidentical={same_p} maxabsdiff={dp:.3e} | unc bitidentical={same_u} maxabsdiff={du}")
    ok &= same_p and same_u
    # training corruption must be identical too
    print(f"           train_epsilon identical={np.array_equal(e0['train_epsilon'][s], e3['train_epsilon'][s])}")
print("OOF RNG-NEUTRAL:", ok)
