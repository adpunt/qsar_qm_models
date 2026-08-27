import importlib.util, sys, numpy as np, torch, random

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod); return mod

NEW = '/Users/apunt/repos/KIRBy/tests/alternative_data_noise_robustness.py'
OLD = '/private/tmp/claude-501/-Users-apunt-repos-qsar-qm-models/e1d07839-2376-4458-891b-01d0a9e0433b/scratchpad/kirby_prefix9.py'
m = load(NEW, 'kb_new')
o = load(OLD, 'kb_old')

rs = np.random.RandomState(0)
n_tr, n_va, n_te, d = 120, 40, 50, 8
Xtr = rs.randn(n_tr, d); ytr = Xtr[:, 0] * 2 + rs.randn(n_tr) * 0.3
Xva = rs.randn(n_va, d); yva = Xva[:, 0] * 2 + rs.randn(n_va) * 0.3
Xte = rs.randn(n_te, d); yte = Xte[:, 0] * 2 + rs.randn(n_te) * 0.3
sigmas = [0.0, 0.3, 0.6]
STRATS = ['legacy', 'quantile', 'threshold', 'outlier', 'hetero', 'valprop']
MT = 'deterministic'

def instrument(mod):
    orig = mod.train_neural_regression
    log = []
    def wrapped(X_train, y_train, X_val, y_val, X_test, **kw):
        log.append((np.array(y_train, float, copy=True), np.array(y_val, float, copy=True)))
        return orig(X_train, y_train, X_val, y_val, X_test, **kw)
    mod.train_neural_regression = wrapped
    return log, orig

def seed():
    torch.manual_seed(12345); np.random.seed(999); random.seed(777)

allok = True
for strat in STRATS:
    log, orig = instrument(m)
    seed(); pT, uT, eT = m.run_neural_experiment(Xtr, ytr, Xva, yva, Xte, yte, MT, strat, sigmas,
                                                 oof_folds=0, noise_validation=True)
    logT = list(log); log.clear()
    seed(); pF, uF, eF = m.run_neural_experiment(Xtr, ytr, Xva, yva, Xte, yte, MT, strat, sigmas,
                                                 oof_folds=0, noise_validation=False)
    logF = list(log)
    m.train_neural_regression = orig

    # OLD behaviour (pre-fix-9 snapshot)
    logO, origO = instrument(o)
    seed(); pO, uO, eO = o.run_neural_experiment(Xtr, ytr, Xva, yva, Xte, yte, MT, strat, sigmas,
                                                 oof_folds=0)
    logOl = list(logO); o.train_neural_regression = origO

    print(f"--- strategy={strat}")
    for i, s in enumerate(sigmas):
        ytrT, yvaT = logT[i]; ytrF, yvaF = logF[i]; ytrO, yvaO = logOl[i]
        train_same = np.array_equal(ytrT, ytrF)
        train_same_old = np.array_equal(ytrT, ytrO)
        val_corrupted = not np.array_equal(yvaT, np.asarray(yva, float))
        val_clean_off = np.array_equal(yvaF, np.asarray(yva, float))
        val_off_eq_old = np.array_equal(yvaF, yvaO)
        vd = float(np.max(np.abs(yvaT - np.asarray(yva, float))))
        print(f"  s={s}: train(on)==train(off) {train_same} | train(on)==train(OLD) {train_same_old}"
              f" | val corrupted(on) {val_corrupted} (max|dy|={vd:.4g})"
              f" | val clean(off) {val_clean_off} | val(off)==val(OLD) {val_off_eq_old}")
        if s > 0:
            allok &= train_same and train_same_old and val_corrupted and val_clean_off and val_off_eq_old
        else:
            allok &= train_same and train_same_old and val_clean_off
    # end-to-end: --no-noise-validation must reproduce OLD test predictions exactly
    for s in sigmas:
        eq = np.array_equal(np.asarray(pF[s], float), np.asarray(pO[s], float))
        if not eq:
            print(f"  !! s={s} predictions(off) != predictions(OLD) maxdiff="
                  f"{np.max(np.abs(np.asarray(pF[s],float)-np.asarray(pO[s],float))):.3e}")
        allok &= eq
    print(f"  e2e preds(off)==preds(OLD) for all sigma: "
          f"{all(np.array_equal(np.asarray(pF[s],float), np.asarray(pO[s],float)) for s in sigmas)}")

print("DEFECT9 ALL OK:", allok)
