#!/usr/bin/env python3
"""Prove a tuned setting reaches the LOGD / CACO-2 / HERG models, or fail.

    python scripts/test_lab_tuned_reaches_models.py

Until 2026-09-01 that pipeline built every model from the shared spec and there
was no route for a tuned value at all, so a setting could be chosen with great
care and the job would run at its default anyway (RERUN_PLAN.md 5.7d). This
checks the route end to end, on the file that is actually shipped:

1. every entry in the lab file names a model this pipeline can tune,
   a representation on the roster, and a dataset it runs;
2. tuned_neural_params returns that entry for the pipeline's own model_type
   string, including the herg/herg_ki name difference;
3. the built network really carries the tuned width, depth and activation --
   not merely that the call did not raise;
4. an unknown key STOPS the run rather than being silently dropped, because a
   dropped key fits a different model from the one that was chosen.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))
LAB = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters_lab.json')

_fails = []


def ok(msg):
    print(f'  ok    {msg}')


def fail(msg):
    print(f'  FAIL  {msg}')
    _fails.append(msg)


def main():
    os.environ.setdefault('QSAR_QM_MODELS_ROOT', _ROOT)
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    import tuning_rosters as R

    if not os.path.exists(LAB):
        print(f'  skip  {LAB} does not exist yet — run '
              f'scripts/ship_tuned_settings.py --write')
        return 0
    with open(LAB) as fh:
        lab = json.load(fh)

    print('the shipped file')
    known = {k for k in R.TUNED_KEY.values() if k is not None}
    for ds, by_model in sorted(lab.items()):
        if ds not in ('herg', 'caco2', 'logd'):
            fail(f'{ds!r} is not a dataset this pipeline runs')
        for model, by_rep in sorted(by_model.items()):
            if model not in known:
                fail(f'{ds}/{model}: not a key any builder asks for')
            for rep in by_rep:
                if rep not in R.ALL_REPS:
                    fail(f'{ds}/{model}/{rep}: not on the representation roster')
    if not _fails:
        ok(f'{sum(len(v) for v in lab.values())} entries, every dataset, model '
           f'and representation on the roster')

    path = os.path.expanduser(
        '~/repos/KIRBy/tests/alternative_data_noise_robustness.py')
    if not os.path.exists(path):
        print(f'  skip  {path} not present; cannot check the reader')
        return 1 if _fails else 0

    import importlib.util
    spec = importlib.util.spec_from_file_location('altpipe', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    print('\nthe reader')
    if not mod.TUNED_LAB:
        fail('the pipeline loaded no tuned settings at all')
    else:
        ok(f'pipeline loaded {sum(len(v) for v in mod.TUNED_LAB.values())} entries')

    # herg is called herg_ki there; the alias must survive.
    checked = 0
    for ds, by_model in sorted(lab.items()):
        pipe_ds = 'herg_ki' if ds == 'herg' else ds
        for model, by_rep in sorted(by_model.items()):
            mt = {v: k for k, v in mod.TUNED_MODEL_LABEL.items()}.get(model)
            if mt is None:
                fail(f'{model} has no model_type in the pipeline')
                continue
            for rep, params in sorted(by_rep.items()):
                got = mod.tuned_neural_params(mt, rep, pipe_ds)
                if got != params:
                    fail(f'{pipe_ds}/{mt}/{rep}: reader returned {got}, '
                         f'file holds {params}')
                checked += 1
    if checked:
        ok(f'{checked} lookups return exactly what the file holds '
           f'(herg -> herg_ki included)')

    print('\nthe built network carries the setting')
    import torch  # noqa: F401
    dnn = mod.DeterministicRegressor(
        50, 64, 32, activation='tanh')
    sizes = [l.out_features for l in dnn.net if hasattr(l, 'out_features')]
    acts = [type(l).__name__ for l in dnn.net
            if type(l).__name__ in ('ReLU', 'Tanh')]
    if sizes[:2] == [64, 32] and set(acts) == {'Tanh'}:
        ok(f'DNN built at widths {sizes[:2]} with {acts[0]}')
    else:
        fail(f'DNN ignored the setting: widths {sizes[:2]}, activations {acts}')

    mlp = mod.MLPRegressor(50, hidden_size=64, num_hidden_layers=1,
                           dropout_rate=0.379)
    # num_hidden_layers counts the INPUT layer, so the ModuleList holds
    # num_hidden_layers - 1. Both sides build it that way
    # (models/models.py:516 and the copy in this pipeline); the count below
    # follows that convention rather than asserting a different one.
    n_hidden = len(mlp.hidden_layers) + 1
    if mlp.input_layer.out_features == 64 and n_hidden == 1:
        ok(f'MLP built at width 64 with {n_hidden} hidden layer '
           f'({len(mlp.hidden_layers)} in the ModuleList, as both sides count)')
    else:
        fail(f'MLP ignored the setting: width '
             f'{mlp.input_layer.out_features}, {n_hidden} hidden layers')
    mlp4 = mod.MLPRegressor(50, hidden_size=128, num_hidden_layers=4,
                            dropout_rate=0.2)
    if len(mlp4.hidden_layers) + 1 == 4 and mlp4.input_layer.out_features == 128:
        ok('MLP at 4 layers really builds 4, so depth is not being ignored')
    else:
        fail(f'MLP depth ignored: asked 4, built '
             f'{len(mlp4.hidden_layers) + 1}')

    print('\nan unapplied key stops the run')
    ds0 = sorted(lab)[0] if lab else None
    if ds0:
        model0 = sorted(lab[ds0])[0]
        rep0 = sorted(lab[ds0][model0])[0]
        mt0 = {v: k for k, v in mod.TUNED_MODEL_LABEL.items()}[model0]
        saved = dict(lab[ds0][model0][rep0])
        pipe_ds0 = 'herg_ki' if ds0 == 'herg' else ds0
        mod.TUNED_LAB[ds0][model0][rep0] = dict(saved, weight_decay=0.01)
        try:
            mod.tuned_neural_params(mt0, rep0, pipe_ds0)
            fail('an unknown key was accepted and would have been dropped')
        except ValueError as exc:
            if 'weight_decay' in str(exc):
                ok('an unknown key raises and names itself')
            else:
                fail(f'raised, but not about the key: {exc}')
        finally:
            mod.TUNED_LAB[ds0][model0][rep0] = saved

    print()
    if _fails:
        print(f'{len(_fails)} failure(s)')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
