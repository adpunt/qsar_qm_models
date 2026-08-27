#!/usr/bin/env python3
"""The Bayesian networks must be trained on the loss the transformation wrapped.

    export OMP_NUM_THREADS=1
    python scripts/test_bnn_criterion_order.py

WHAT THIS GUARDS
----------------
Both neural bases wrap their loss when a Bayesian transformation is applied:
`bnn_elbo_criterion` for the BNN variants, `VBLLLoss` for the variational ones.
NN-alpha built the base loss before the transformation and kept the wrapper.
NN-beta built it AFTER, and then overwrote it, so on 2026-08-27:

  * MLP-BNN-Full raised UnboundLocalError on `criterion` and produced no rows;
  * and had it not raised, the line after the transformation would have replaced
    the ELBO wrapper with a plain loss, training NN-beta's Bayesian variants with
    no KL term at all -- the exact defect the KL term was added to fix, back
    again by ordering.

So the loss that reaches the trainer is what this checks, for both bases and
every transformation. Nothing else in the pipeline records which loss was used,
so a silent revert here would show up only as NN-beta's Bayesian variants
looking better calibrated than they are.
"""
from __future__ import annotations

import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
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
import tempfile
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

FAILURES = []

# (model label, what the criterion handed to the trainer must be)
CASES = [
    ('dnn',                      'plain'),
    ('mlp',                      'plain'),
    ('dnn_bnn_full',             'elbo'),
    ('mlp_bnn_full',             'elbo'),
    ('dnn_bnn_full_variational', 'vbll'),
    ('mlp_bnn_full_variational', 'vbll'),
]


def classify(criterion, M):
    if isinstance(criterion, M.VBLLLoss):
        return 'vbll'
    # bnn_elbo_criterion returns a closure over the base loss; it is a plain
    # function, where every unwrapped loss is an nn.Module.
    if isinstance(criterion, types.FunctionType):
        return 'elbo'
    return 'plain'


def main():
    import models as M
    import process_and_train as pat
    import tuning_rosters as rosters
    import tune_hyperparameters as tuner
    import torch

    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 12)).astype(np.float32)
    y = (x[:, 0] * 2.0 - x[:, 1]).astype(np.float32)
    x_train, y_train, x_val, y_val = x[:90], y[:90], x[90:], y[90:]

    seen = {}
    original_train_nn = M.train_nn

    def spy(model, train_loader, val_loader, criterion, *a, **kw):
        seen['criterion'] = criterion
        # Stop before the epochs: the loss object is the whole question here and
        # training it would make this check cost minutes instead of seconds.
        raise _Stop()

    class _Stop(Exception):
        pass

    for label, expected in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            args = tuner.make_args(pat, label, 'ecfp4', 90,
                                   os.path.join(tmp, 'rows.csv'), rosters)
            seen.clear()
            M.train_nn = spy
            torch.manual_seed(0)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    common = dict(x_train=x_train, y_train=y_train,
                                  x_test=x_val, y_test=y_val,
                                  x_val=x_val, y_val=y_val, args=args, s=0.0,
                                  rep='ecfp4', iteration=0, iteration_seed=0,
                                  file_no=0, y_test_original=y_val,
                                  train_noise=None)
                    if args.models[0] == 'dnn':
                        M.train_dnn_model(**common)
                    else:
                        M.train_mlp_variant_model(model_type='mlp', **common)
            except _Stop:
                pass
            except Exception as exc:
                FAILURES.append(label)
                print(f'  FAIL  {label}: {type(exc).__name__}: {exc}')
                M.train_nn = original_train_nn
                continue
            finally:
                M.train_nn = original_train_nn

        if 'criterion' not in seen:
            FAILURES.append(label)
            print(f'  FAIL  {label}: the trainer was never reached')
            continue
        got = classify(seen['criterion'], M)
        if got != expected:
            FAILURES.append(label)
            print(f'  FAIL  {label}: trained on the {got} loss, expected '
                  f'the {expected} one')
        else:
            print(f'  ok    {label}: trained on the {expected} loss')

    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s): {FAILURES}')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
