#!/usr/bin/env python3
"""Does NGBoost's stage selection change the answer on QM9?

WHY THIS EXISTS
---------------
`models/model_defaults.py` makes `n_estimators` 500 a CAP and reads the real stage
count off a validation curve, because that is what NGBoost's own paper does
(Duan et al., ICML 2020, section 4). QM9 has applied that since 2026-08-28 and
the experimental pipeline since KIRBy `6e7b860` on the same day. Nothing had
measured whether selecting the stage count changes any answer, so RERUN_PLAN.md
5.7l's parity note could not say what the remaining difference between the two
sides is worth. This measures it. Findings are in RERUN_PLAN.md 5.7q.

Three fits per cell, all built from the shared spec:

  stopped      stop on the held-out validation split, predict at the optimum
               -- what models/models.py train_ngboost_model does.
  all fitted   the SAME model, predicting at every stage it fitted
               -- what ngboost's own predict() does with no max_iter.
  to the cap   no stopping, the full 500 stages
               -- what the experimental pipeline did before 2026-08-28.

USAGE
-----
    python scripts/check_ngboost_stage_selection.py            # 5,000 molecules
    N_MOL=10000 REPS=pdv python scripts/check_ngboost_stage_selection.py

`N_MOL` is the QM9 sample size, so the training block is 80% of it -- and the
whole finding is that the effect depends on that number. `REPS` is a comma list.
`OUT` names the results file. Roughly six minutes per fit at 8,000 training rows.
"""
import json, os, sys, time
import numpy as np

# Anchored to THIS FILE, not the working directory. process_and_train resolves
# its data directory as "../data", so the process has to sit in scripts/.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', 'models'))
os.chdir(_HERE)

import process_and_train as pt
from model_defaults import SKLEARN_DEFAULTS
from noiseInject import CONDITIONS as INJ_CONDITIONS, NoiseInjectorRegression

from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE
from sklearn.tree import DecisionTreeRegressor

N = int(os.environ.get('N_MOL', 5000))
SEED = 42
REPS = tuple(os.environ.get('REPS', 'pdv,ecfp4').split(','))
CONDITIONS = (('clean', 0.0), ('gaussian', 0.5))

SPEC = SKLEARN_DEFAULTS['ngboost']

# ---------------------------------------------------------------- data
import torch
torch.manual_seed(SEED)
np.random.seed(SEED)

qm9 = pt.load_qm9('homo_lumo_gap')
perm = torch.randperm(len(qm9))
qm9 = qm9.index_select(perm)
sub = qm9[:N]
smiles = [d.smiles for d in sub]
y_all = np.array([float(d.y) for d in sub], dtype=float)

np.random.seed(SEED)
train_idx, val_idx, test_idx = pt.scaffold_split_indices(smiles, 0.8, 0.1, 0.1)
canon = [pt.Chem.MolToSmiles(pt.Chem.MolFromSmiles(s), isomericSmiles=False) for s in smiles]
_assign = pt.build_scaffold_groups(canon)
groups_all = np.array([_assign[c] for c in canon])
print(f"{N} QM9 molecules | train {len(train_idx)} val {len(val_idx)} test {len(test_idx)}"
      f" | {len(np.unique(groups_all))} scaffold groups", flush=True)

def featurise(rep):
    if rep == 'pdv':
        return np.stack([pt.rdkit_mol_descriptors_from_smiles(s) for s in smiles]).astype(np.float32)
    if rep == 'ecfp4':
        return np.stack([np.unpackbits(pt.ecfp4_fingerprint(s), bitorder='little')
                         for s in smiles]).astype(np.float32)
    raise ValueError(rep)

def noisy(cond, level, y, groups, spread, seed):
    if level == 0.0 or cond == 'clean':
        return y.copy()
    ref = 1.0 if INJ_CONDITIONS[cond]['strategy'] == 'censoring' else spread
    inj = NoiseInjectorRegression.from_condition(cond, random_state=seed)
    return np.asarray(inj.inject_verbose(y, level * ref, groups=groups).y_noisy, dtype=float)

def build():
    return NGBRegressor(
        Dist=Normal, Score=MLE,
        Base=DecisionTreeRegressor(max_depth=SPEC['base_max_depth'],
                                   criterion=SPEC['base_criterion'], random_state=SEED),
        natural_gradient=SPEC['natural_gradient'],
        n_estimators=SPEC['n_estimators'],
        learning_rate=SPEC['learning_rate'],
        minibatch_frac=SPEC['minibatch_frac'],
        col_sample=SPEC['col_sample'],
        verbose=False, random_state=SEED)

def score(model, xt, yt, max_iter):
    mu = model.predict(xt, max_iter=max_iter)
    d = model.pred_dist(xt, max_iter=max_iter)
    sd = np.asarray(d.scale, dtype=float)
    rmse = float(np.sqrt(np.mean((mu - yt) ** 2)))
    r2 = float(1 - np.sum((mu - yt) ** 2) / np.sum((yt - yt.mean()) ** 2))
    nll = float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + (yt - mu) ** 2 / (2 * sd ** 2)))
    return dict(rmse=rmse, r2=r2, nll=nll, mean_sd=float(sd.mean()), mu=mu)

rows = []
for rep in REPS:
    x_all = featurise(rep)
    for cond, level in CONDITIONS:
        # labels: clean training mean/SD standardisation, as the pipeline does
        y_tr_clean = y_all[train_idx]
        mu0, sd0 = y_tr_clean.mean(), y_tr_clean.std()
        z = (y_all - mu0) / sd0
        spread = float(z[train_idx].std())          # the CLEAN training spread
        y_tr = noisy(cond, level, z[train_idx], groups_all[train_idx], spread, SEED)
        y_va = noisy(cond, level, z[val_idx], groups_all[val_idx], spread, SEED + 1)
        y_te = z[test_idx]                          # test labels are never noised

        x_tr, x_va, x_te = x_all[train_idx], x_all[val_idx], x_all[test_idx]
        if rep == 'pdv':                            # per-feature, on training only
            m, s = x_tr.mean(0), x_tr.std(0)
            s[s == 0] = 1.0
            x_tr, x_va, x_te = (x_tr - m) / s, (x_va - m) / s, (x_te - m) / s
        x_tr = np.nan_to_num(x_tr, nan=0.0, posinf=0.0, neginf=0.0)
        x_va = np.nan_to_num(x_va, nan=0.0, posinf=0.0, neginf=0.0)
        x_te = np.nan_to_num(x_te, nan=0.0, posinf=0.0, neginf=0.0)

        t0 = time.time()
        m_es = build()
        m_es.fit(x_tr, y_tr, X_val=x_va, Y_val=y_va,
                 early_stopping_rounds=SPEC['early_stopping_rounds'])
        best = getattr(m_es, 'best_val_loss_itr', None)
        fitted = len(m_es.scalings)
        t_es = time.time() - t0

        t0 = time.time()
        m_cap = build()
        m_cap.fit(x_tr, y_tr)
        t_cap = time.time() - t0

        A = score(m_es, x_te, y_te, best)
        Ap = score(m_es, x_te, y_te, None)
        B = score(m_cap, x_te, y_te, None)

        row = dict(rep=rep, condition=cond, level=level,
                   best_iter=(None if best is None else int(best)),
                   stages_fitted=int(fitted), cap=SPEC['n_estimators'],
                   seconds_es=round(t_es, 1), seconds_cap=round(t_cap, 1),
                   pred_shift_A_vs_Aprime=float(np.max(np.abs(A['mu'] - Ap['mu']))),
                   pred_shift_A_vs_B=float(np.max(np.abs(A['mu'] - B['mu']))))
        for name, res in (('A_optimum', A), ('Aprime_all_fitted', Ap), ('B_no_stopping', B)):
            for k in ('r2', 'rmse', 'nll', 'mean_sd'):
                row[f'{name}_{k}'] = round(res[k], 6)
        rows.append(row)
        print(json.dumps(row, indent=None), flush=True)

out = os.environ.get('OUT', 'ngb_es_test_results.json')
with open(out, 'w') as f:
    json.dump(rows, f, indent=2)
print(f"\nwrote {out}")
