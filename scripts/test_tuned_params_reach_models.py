#!/usr/bin/env python3
"""A tuned value must change the model that gets built.

    export OMP_NUM_THREADS=1
    python scripts/test_tuned_params_reach_models.py

WHAT THIS GUARDS
----------------
Writing a parameter into results/master_tuned_hyperparameters.json is not the
same as that parameter reaching a model. Between the file and the estimator sit
a decisions file, a key lookup, and a builder that reads some of the dict and
ignores the rest. Every one of those has already failed at least once:

  * results/hyperparameter_decisions.json did not exist at all, so the tuned
    branch could never fire. Until 2026-08-27 it fell through to the defaults in
    silence; it now says which file is missing, by name.
  * the master file on disk carried `use_default_max_depth`, which is an Optuna
    bookkeeping flag and not an argument of RandomForestRegressor. Under
    --use-best-params the forest would have raised TypeError on construction --
    that entry was never a working setting, it was a crash waiting for the flag
    that switches it on.
  * four of the values the Optuna path suggested never reached a model at all
    (models.py:1660 quantile, :4255 alpha, :4256 predictor_type, :2561 the loss
    parameters), and nothing said so.

So this asserts the whole path, twice over:

  REAL     every parameter in the master file on disk, if it exists, reaches
           the estimator the pipeline builds for that model.
  SYNTHETIC every tuned key that addresses exactly one model accepts a value
           chosen to be unmistakable, and that value arrives.

The estimator is caught at construction: the class the builder looks up is
replaced by a recorder that stores the keyword arguments and then builds the
real thing. For the three families that build a torch object rather than take
constructor keywords -- the Gaussian process and the two neural bases -- there
is nothing to intercept, so the assertion is behavioural instead: the tuned
setting must produce different predictions from the default. A parameter that
changes neither the constructor call nor the predictions has not reached a
model, whatever the file says.
"""
from __future__ import annotations

import os
import sys

_THREAD_VARS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
for _v in _THREAD_VARS:
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# LightGBM must be imported BEFORE torch and tensorflow arrive, and this is the
# first chance. Measured 2026-08-27, three runs, same machine, same environment:
#
#   lightgbm imported first, fitted with its default thread count   works
#   lightgbm imported after process_and_train, fitted with n_jobs=1 works
#   lightgbm imported after process_and_train, default thread count SEGFAULT
#
# Exit 139, no traceback, nothing in the log -- the same threading-runtime
# collision as RERUN_PLAN.md 2.8e, reached by a different route. models.py does
# `import lightgbm as lgb` inside train_lgb_model, so in the pipeline it is
# always the late one; importing it here puts its runtime in first.
import lightgbm as _lightgbm_first  # noqa: F401  (imported for its side effect)

import contextlib
import io
import json
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MASTER = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters.json')
DECISIONS = os.path.join(_ROOT, 'results', 'hyperparameter_decisions.json')

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f'  FAIL  {msg}')


def ok(msg):
    print(f'  ok    {msg}')


# ---------------------------------------------------------------------------
# Where each builder looks its estimator class up. A function-local import means
# the attribute has to be replaced on the module, not in models' namespace.
# ---------------------------------------------------------------------------
CONSTRUCTOR_SITES = {
    'rf':      ('models', 'RandomForestRegressor'),
    'svm':     ('models', 'SVR'),
    'xgboost': ('xgboost', 'XGBRegressor'),
    'lgb':     ('lightgbm', 'LGBMRegressor'),
    'ngboost': ('ngboost', 'NGBRegressor'),
}

# Values chosen to be unmistakable: nothing else in the pipeline would set them.
SYNTHETIC = {
    'rf':      {'n_estimators': 7, 'max_depth': 3, 'min_samples_leaf': 3,
                'min_samples_split': 6, 'max_features': 0.4, 'bootstrap': False},
    'svm':     {'C': 7.5, 'gamma': 0.017, 'kernel': 'rbf'},
    'xgboost': {'n_estimators': 7, 'max_depth': 3, 'learning_rate': 0.077,
                'subsample': 0.7, 'colsample_bytree': 0.7,
                'colsample_bylevel': 0.7, 'min_child_weight': 3,
                'gamma': 0.3, 'reg_alpha': 0.3, 'reg_lambda': 0.3},
    'lgb':     {'n_estimators': 7, 'learning_rate': 0.077, 'num_leaves': 7,
                'max_depth': 3, 'subsample': 0.7, 'colsample_bytree': 0.7,
                'min_child_samples': 7, 'reg_alpha': 0.3, 'reg_lambda': 0.3},
    'ngboost': {'n_estimators': 7, 'learning_rate': 0.077,
                'natural_gradient': False},
    # No constructor to intercept: asserted on the predictions instead, so the
    # values have to be far enough from the defaults to move them.
    #
    # kernel_name is NOT optional. train_gauche_model reads params['kernel_name']
    # with no fallback the moment the dict comes from the tuned file, so an entry
    # without it raises KeyError rather than falling back to --kernel. It is
    # pinned, never searched -- see models/tuning_rosters.py on why the RBF and
    # Tanimoto Gaussian processes must not share a tuned kernel.
    'gauche':  {'kernel_name': 'RBF', 'outputscale': 9.0,
                'likelihood_noise': 0.09},
    'dnn':     {'hidden_size1': 8, 'hidden_size2': 8, 'activation': 'tanh'},
    'mlp':     {'hidden_size': 8, 'num_hidden_layers': 1, 'dropout_rate': 0.45,
                'lr': 0.0077},
}

BEHAVIOURAL = {'gauche', 'gauche_rbf', 'dnn', 'mlp'}


def behavioural_keys(rosters):
    """Keys checked by whether the FIT MOVES, not by intercepting a constructor.

    A network is built by hand from torch layers, so there is no single class to
    record; the check is that the same data fitted with and without the setting
    gives a different score. DERIVED FROM THE ROSTER rather than hard-coded:
    when every model got its own tuned key on 2026-08-31 the set still named
    only 'dnn' and 'mlp', so a master file carrying 'dnn_bnn_full' fell through
    to the constructor branch and this test died with KeyError instead of
    checking anything.
    """
    out = set(BEHAVIOURAL)
    for label, key in rosters.TUNED_KEY.items():
        if key and (label.startswith('dnn') or label.startswith('mlp')
                    or label.startswith('gauche')):
            out.add(key)
    return out


def toy_data(n=120, d=12, seed=0, binary=False):
    """A small regression problem.

    `binary` gives a 0/1 matrix instead of Gaussian columns. The fingerprint
    kernels -- Tanimoto and its family -- are ratios of set overlaps and are
    defined on binary vectors ONLY; models.py refuses them on anything else, so
    fitting the Tanimoto Gaussian process on Gaussian columns raises before it
    reaches a single hyperparameter. That is what this fixture used to do, and
    it made the one check that proves a tuned value reaches the Gaussian process
    fail for a reason that had nothing to do with tuned values.
    """
    rng = np.random.default_rng(seed)
    if binary:
        x = (rng.random((n, d)) < 0.35).astype(np.float32)
    else:
        x = rng.normal(size=(n, d)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 1] + rng.normal(scale=0.1, size=n)).astype(np.float32)
    cut = int(n * 0.75)
    return x[:cut], y[:cut], x[cut:], y[cut:]


@contextlib.contextmanager
def recording(module_name, attr):
    """Replace an estimator class with one that records its keyword arguments."""
    import importlib

    module = importlib.import_module(module_name)
    original = getattr(module, attr)
    seen = []

    # A factory, not a subclass. scikit-learn reads an estimator's parameter
    # names off its __init__ signature and REFUSES one with *args, so a
    # subclass with (*a, **kw) makes get_params() raise -- which XGBoost and
    # LightGBM both call inside fit. The factory records and then builds the
    # real class, so the object the builder gets is untouched.
    def recorder(*a, **kw):
        seen.append(dict(kw))
        return original(*a, **kw)

    setattr(module, attr, recorder)
    try:
        yield seen
    finally:
        setattr(module, attr, original)


def build_with(tuner, model_label, rep, params, data):
    """One fit through the pipeline's own builder, with `params` delivered the
    way --use-best-params delivers them."""
    import models as M
    import tuning_rosters as rosters
    import process_and_train as pat

    x_train, y_train, x_val, y_val = data
    with tempfile.TemporaryDirectory() as tmp:
        scratch = os.path.join(tmp, 'rows.csv')
        args = tuner.make_args(pat, model_label, rep, len(x_train), scratch,
                               rosters)
        args.use_best_params = params is not None
        sink = io.StringIO()
        with tuner.candidate(M, params), contextlib.redirect_stdout(sink):
            common = dict(x_train=x_train, y_train=y_train,
                          x_test=x_val, y_test=y_val,
                          x_val=x_val, y_val=y_val,
                          args=args, s=0.0, rep=rep, iteration=0,
                          iteration_seed=0, train_noise=None)
            mt = args.models[0]
            if mt in ('rf', 'qrf'):
                return M.train_rf_model(model_type=mt, file_no=0,
                                        y_test_original=y_val, **common)
            if mt == 'svm':
                return M.train_svm_model(**common)
            if mt == 'xgboost':
                return M.train_xgboost_model(**common)
            if mt == 'lgb':
                return M.train_lgb_model(**common)
            if mt == 'ngboost':
                return M.train_ngboost_model(file_no=0, y_test_original=y_val,
                                             **common)
            if mt == 'gauche':
                return M.train_gauche_model(file_no=0, y_test_original=y_val,
                                            **common)
            if mt == 'dnn':
                return M.train_dnn_model(file_no=0, y_test_original=y_val,
                                         **common)
            if mt == 'mlp':
                return M.train_mlp_variant_model(model_type='mlp', file_no=0,
                                                 y_test_original=y_val, **common)
            raise KeyError(mt)


def a_model_for(key, rosters):
    """A model label that reads this tuned key, and a representation it runs on."""
    for label, k in rosters.TUNED_KEY.items():
        if k == key:
            return label, rosters.MODELS[label][4][0]
    raise KeyError(key)


def check_key_reaches(key, params, tuner, rosters, why):
    """Assert that every parameter in `params` reaches the built model."""
    label, rep = a_model_for(key, rosters)
    # A fingerprint kernel needs binary features or models.py refuses it before
    # any hyperparameter is read. The Tanimoto Gaussian process is the only
    # model here that asks for one; the RBF process, which the roster spells
    # gauche_rbf, is happy either way and shares this builder.
    needs_binary = rosters.MODELS[label][0].split().count('tanimoto') > 0
    data = toy_data(binary=needs_binary)

    if key in behavioural_keys(rosters):
        try:
            import torch
            torch.manual_seed(0)
            base = build_with(tuner, label, rep, None, data)
            torch.manual_seed(0)
            tuned = build_with(tuner, label, rep, params, data)
        except Exception as exc:
            fail(f'{why}: {key} could not be fitted ({type(exc).__name__}: {exc})')
            return
        if base == tuned or (np.isnan(base) and np.isnan(tuned)):
            fail(f'{why}: {key} scored identically with and without the tuned '
                 f'setting ({base}), so the setting changed nothing that the '
                 f'model does')
        else:
            ok(f'{why}: {key} — tuned setting moves the fit '
               f'(R2 {base:+.4f} -> {tuned:+.4f})')
        return

    if key not in CONSTRUCTOR_SITES:
        fail(f'{why}: {key} has no constructor site here and is not checked '
             f'behaviourally, so nothing proves a tuned value reaches it. Add '
             f'it to CONSTRUCTOR_SITES or to behavioural_keys().')
        return
    module_name, attr = CONSTRUCTOR_SITES[key]
    try:
        with recording(module_name, attr) as seen:
            build_with(tuner, label, rep, params, data)
    except Exception as exc:
        fail(f'{why}: {key} could not be fitted with those parameters '
             f'({type(exc).__name__}: {exc})')
        return
    if not seen:
        fail(f'{why}: {key} built no {attr}, so nothing received the setting')
        return

    kwargs = seen[0]
    missing = [p for p in params if p not in kwargs]
    wrong = [(p, params[p], kwargs[p]) for p in params
             if p in kwargs and kwargs[p] != params[p]]
    if missing:
        fail(f'{why}: {key} — {sorted(missing)} never reached {attr}. '
             f'It was written to the tuned file and the builder ignores it.')
    if wrong:
        fail(f'{why}: {key} — {attr} was built with '
             f'{[(p, f"{g!r} not {w!r}") for p, w, g in wrong]}')
    if not missing and not wrong:
        ok(f'{why}: {key} — all {len(params)} value(s) reached {attr}')


def check_file_contract(rosters):
    """The two files, read the way load_best_hyperparameters reads them."""
    import models as M

    with tempfile.TemporaryDirectory() as tmp:
        with open(os.path.join(tmp, 'master_tuned_hyperparameters.json'), 'w') as fh:
            json.dump({'rf': {'ecfp4': {'n_estimators': 7}}}, fh)
        with open(os.path.join(tmp, 'hyperparameter_decisions.json'), 'w') as fh:
            json.dump({'rf': 'USE_TUNED'}, fh)
        got = M.load_best_hyperparameters('rf', 'ecfp4', results_dir=tmp)
        if got != {'n_estimators': 7}:
            fail(f'the reader did not return the tuned dict; got {got!r}')
        else:
            ok('the reader returns the tuned dict for a USE_TUNED model')

        with open(os.path.join(tmp, 'hyperparameter_decisions.json'), 'w') as fh:
            json.dump({'rf': 'USE_DEFAULT'}, fh)
        if M.load_best_hyperparameters('rf', 'ecfp4', results_dir=tmp) is not None:
            fail('a model not marked USE_TUNED still got tuned parameters')
        else:
            ok('a model not marked USE_TUNED gets none')

    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            got = M.load_best_hyperparameters('rf', 'ecfp4', results_dir=tmp)
        if got is not None:
            fail('a missing file still produced tuned parameters')
        elif 'master_tuned_hyperparameters.json' not in buf.getvalue():
            fail('a missing file was not named on the way past — that silence '
                 'is what hid the tuned branch never firing')
        else:
            ok('a missing file is reported by name, not skipped silently')


def main():
    import tuning_rosters as rosters
    import tune_hyperparameters as tuner

    print('file contract')
    check_file_contract(rosters)

    print('\nsynthetic: every writable key, with a value nothing else would set')
    # ORDER MATTERS, and it is an environment fact rather than a preference.
    # LightGBM segfaults -- exit 139, no traceback -- if it fits in a process that
    # has already fitted a gpytorch model, even at one thread and even with
    # KMP_DUPLICATE_LIB_OK set. Same threading-runtime collision as RERUN_PLAN.md
    # 2.8e. The boosting and forest fits therefore run first, and the three that
    # go through torch run last, with the Gaussian process last of all.
    keys = ['rf', 'svm', 'xgboost', 'lgb', 'ngboost', 'dnn', 'mlp', 'gauche']
    # None means the model has NO tuned path at all -- its builder never calls
    # load_best_hyperparameters, so no entry in either JSON file can reach it
    # (the heteroscedastic Gaussian process). There is no key to check, which is
    # different from a key that is missing, so it is dropped here rather than
    # sorted with the strings -- which is what it used to do, and it raised
    # TypeError comparing None with a str.
    # ONE KEY PER BUILDER, NOT ONE PER MODEL.
    #
    # Every model got its own tuned key on 2026-08-31, so there are sixteen keys
    # where there were eight. They do NOT go through sixteen builders: a
    # Bayesian network is `train_dnn_model` with a flag, the heteroscedastic
    # variational network is the same builder with two flags, the quantile
    # forest is `train_rf_model` with a different model_type, and the Tanimoto
    # Gaussian process is `train_gauche_model` with a different kernel. Fitting
    # all sixteen would fit the same eight functions twice over and double an
    # already slow check for nothing.
    #
    # So this fits one key per BUILDER, and asserts that every other key reaches
    # one of those builders. That every key is produced by a real run's
    # arguments is checked separately and cheaply, in
    # scripts/test_tuning_rosters.py, which drives roster_label directly.
    BUILDER_OF = {
        'rf': 'rf', 'qrf': 'rf',
        'svm': 'svm', 'xgboost': 'xgboost', 'lgb': 'lgb', 'ngboost': 'ngboost',
        'gauche': 'gauche', 'gauche_rbf': 'gauche',
        'dnn': 'dnn', 'dnn_bnn_full': 'dnn',
        'dnn_bnn_full_variational': 'dnn',
        'dnn_bnn_full_variational_hetero': 'dnn',
        'mlp': 'mlp', 'mlp_bnn_full': 'mlp',
        'mlp_bnn_full_variational': 'mlp',
        'mlp_bnn_full_variational_hetero': 'mlp',
        # The variance-head networks are the same two builders with a
        # likelihood loss, so a tuned entry under their own key would reach
        # them. Nothing has scored them yet, so they train at their defaults.
        'dnn_bnn_full_mve': 'dnn',
        'mlp_bnn_full_mve': 'mlp',
    }
    known = {k for k in rosters.TUNED_KEY.values() if k is not None}
    unmapped = sorted(known - set(BUILDER_OF))
    assert not unmapped, (
        f'tuned key(s) {unmapped} reach no builder listed here. Add them to '
        f'BUILDER_OF, and if they are a NEW builder add them to `keys` above '
        f'so the fit is actually exercised.')
    builders = {BUILDER_OF[k] for k in known}
    assert set(keys) == builders, (
        f'the fitted key list is stale: builders are {sorted(builders)}, '
        f'fitted are {sorted(keys)}')
    for key in keys:
        check_key_reaches(key, SYNTHETIC[key], tuner, rosters, 'synthetic')

    print('\nthe master file on disk')
    if not os.path.exists(MASTER):
        print(f'  skip  {MASTER} does not exist yet')
    else:
        with open(MASTER) as fh:
            master = json.load(fh)
        known_keys = {rosters.TUNED_KEY[m] for m in rosters.TUNED_KEY}
        for key, by_rep in sorted(master.items()):
            if key not in known_keys:
                fail(f'master file: {key!r} is not a key any builder asks for, '
                     f'so nothing will ever read it')
                continue
            for rep, params in sorted(by_rep.items()):
                if rep not in rosters.ALL_REPS:
                    fail(f'master file: {key}/{rep} — {rep!r} is not on the '
                         f'representation roster')
                    continue
                check_key_reaches(key, params, tuner, rosters,
                                  f'master file {key}/{rep}')

    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s)')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
