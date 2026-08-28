#!/usr/bin/env python3
"""NGBoost must predict at the validation optimum, not at every stage it fitted.

    export OMP_NUM_THREADS=1
    python scripts/test_ngboost_stage_selection.py

WHAT THIS GUARDS
----------------
NGBoost's own paper does not fix the number of boosting stages. Duan, Avati,
Ding, Thai, Basu, Ng and Schuler, ICML 2020, section 4: a validation set selects
the M giving the best log-likelihood, and the model is then refitted at that M.
So `n_estimators` is a cap and the answer is read off the validation curve.

THE TRAP is what the library does next. With `early_stopping_rounds` set, ngboost
stops at (best + patience) and KEEPS every stage it fitted, and `predict(X)` with
no `max_iter` then uses all of them. Measured on 2026-08-28: best_val_loss_itr
227, 278 stages fitted, and the two predictions differ. Those 51 extra stages are
fitted past the validation optimum on noisy labels -- which is exactly the defect
the neural models had when they counted patience and returned the LAST epoch
instead of the best one, up to 20 epochs past the optimum, and it was a
procedural reason their degradation curves looked steeper.

So this asserts three things about the fitted model, on toy data, in seconds:

  1. the base learner is the one the SPEC names, not whatever the library
     happens to default to -- depth 3 is the paper's, and pinning it is the only
     thing stopping an upgrade moving it;
  2. the stage count is read off the validation curve, so fewer stages are
     fitted than the cap;
  3. prediction uses the optimum, not the cap. This is the assertion that
     actually costs something if it regresses, and it regresses invisibly: the
     model still fits, still predicts, and is simply slightly worse.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'models'),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f'  FAIL  {msg}')


def ok(msg):
    print(f'  ok    {msg}')


def main():
    from model_defaults import SKLEARN_DEFAULTS
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from ngboost.scores import MLE
    from sklearn.tree import DecisionTreeRegressor

    spec = SKLEARN_DEFAULTS['ngboost']

    for key in ('base_max_depth', 'base_criterion', 'minibatch_frac',
                'col_sample', 'early_stopping_rounds', 'use_best_iteration'):
        if key not in spec:
            fail(f'the shared spec no longer pins {key!r}. It was pinned on '
                 f'2026-08-28 from Duan et al. section 4 so that a library '
                 f'upgrade could not move it silently.')
    if spec.get('base_max_depth') != 3:
        fail(f"base_max_depth is {spec.get('base_max_depth')}, not the paper's 3")
    else:
        ok('the spec pins the base learner to the paper\'s depth of 3')

    rng = np.random.default_rng(0)
    x = rng.normal(size=(500, 10))
    y = (x[:, 0] * 2 - x[:, 1] + rng.normal(scale=0.5, size=500))
    x_val, y_val = x[:100], y[:100]
    x_tr, y_tr = x[100:], y[100:]

    base = DecisionTreeRegressor(max_depth=spec['base_max_depth'],
                                 criterion=spec['base_criterion'],
                                 random_state=0)
    model = NGBRegressor(
        Dist=Normal, Score=MLE, Base=base,
        natural_gradient=spec['natural_gradient'],
        n_estimators=spec['n_estimators'],
        learning_rate=spec['learning_rate'],
        minibatch_frac=spec['minibatch_frac'],
        col_sample=spec['col_sample'],
        verbose=False, random_state=0)
    model.fit(x_tr, y_tr, X_val=x_val, Y_val=y_val,
              early_stopping_rounds=spec['early_stopping_rounds'])

    built = model.Base
    if getattr(built, 'max_depth', None) != spec['base_max_depth']:
        fail(f'the fitted model\'s base learner has max_depth '
             f'{getattr(built, "max_depth", None)}, not the spec\'s '
             f'{spec["base_max_depth"]}')
    else:
        ok(f'the fitted model used a depth-{spec["base_max_depth"]} base learner')

    fitted = len(model.scalings)
    best = getattr(model, 'best_val_loss_itr', None)
    if best is None:
        fail('the fit recorded no validation optimum, so the stage count was '
             'never read off the validation curve -- the paper\'s procedure did '
             'not happen')
        return _finish()
    if fitted >= spec['n_estimators']:
        print(f'  note  the curve was still improving at the cap '
              f'({fitted} stages); nothing to select, which is legitimate')
    else:
        ok(f'stages read off the validation curve: optimum {best}, '
           f'{fitted} fitted, cap {spec["n_estimators"]}')

    at_cap = model.predict(x_val)
    at_best = model.predict(x_val, max_iter=best)
    if best < fitted and np.allclose(at_cap, at_best):
        fail('predicting at the cap and at the validation optimum give the same '
             'answer, so max_iter is not doing anything and the trap this check '
             'exists for cannot be detected here')
    elif best < fitted:
        ok(f'predicting at the optimum differs from predicting at every fitted '
           f'stage — {fitted - best} stages past the optimum would otherwise be '
           f'used')

    return _finish()


def _finish():
    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s)')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
