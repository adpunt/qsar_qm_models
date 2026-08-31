#!/usr/bin/env python3
"""The names in the tuned-parameter files must be the roster's names.

    python scripts/test_tuning_rosters.py

WHAT THIS GUARDS
----------------
results/master_tuned_hyperparameters.json is keyed by model and then by
representation, and nothing validates either key. The file on disk before this
check existed was keyed by `pdv`, `smiles`, `randomized_smiles` and `graph`, and
carried `gcn`, `gin`, `residual_mlp`, `factorization_mlp`, `mtl`, `flexible_dnn`
and `conformal` -- a representation set and a model set from before the roster
was settled. None of those keys can ever be read by a QM9 job on the current
grid, so the file looked like a set of tuned parameters and delivered nothing.

A name that is wrong here does not raise. `load_best_hyperparameters` looks the
key up, misses, and returns None, and the job trains on the library defaults
having been asked for tuned ones. So the check is the only thing between a
misspelling and a grid that quietly ignores the entire tuning run.

It also checks the three name systems against their sources, so the map in
models/tuning_rosters.py cannot drift:

    the QM9 roster        imported from generate_scripts.py, never retyped
    the tuned-file key    the literal string at each load_best_hyperparameters
                          call site in models.py
    the experimental name the display names in KIRBy's alternative_data_
                          noise_robustness.py

The experimental pipeline has NO reader for these files -- it builds every model
from sklearn_params(...) and NEURAL_DEFAULTS -- so the name map is checked here
against what that pipeline calls things, and is not yet used to deliver anything
to it.
"""
from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (os.path.join(_ROOT, 'models'),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tuning_rosters as rosters

MASTER = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters.json')
DECISIONS = os.path.join(_ROOT, 'results', 'hyperparameter_decisions.json')
MODELS_PY = os.path.join(_ROOT, 'models', 'models.py')

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f'  FAIL  {msg}')


def ok(msg):
    print(f'  ok    {msg}')


def check_map_covers_roster():
    """Every model and representation on the roster has an experimental name,
    and nothing else does."""
    roster_models = set(rosters.MODELS)
    mapped = set(rosters.MODEL_NAME_MAP)
    if roster_models - mapped:
        fail(f'no experimental name for {sorted(roster_models - mapped)}')
    if mapped - roster_models:
        fail(f'experimental names for models not on the roster: '
             f'{sorted(mapped - roster_models)}')
    if roster_models == mapped:
        ok(f'all {len(mapped)} roster models have an experimental name')

    roster_reps = set(rosters.ALL_REPS)
    mapped_reps = set(rosters.REP_NAME_MAP)
    if roster_reps != mapped_reps:
        fail(f'representation map does not match the roster: '
             f'missing {sorted(roster_reps - mapped_reps)}, '
             f'extra {sorted(mapped_reps - roster_reps)}')
    else:
        ok(f'all {len(mapped_reps)} roster representations have an '
           f'experimental name')

    if set(rosters.TUNED_KEY) != roster_models:
        fail(f'TUNED_KEY does not cover the roster: '
             f'missing {sorted(roster_models - set(rosters.TUNED_KEY))}, '
             f'extra {sorted(set(rosters.TUNED_KEY) - roster_models)}')
    else:
        ok('every roster model has a tuned-file key')


def check_tuned_keys_against_models_py():
    """Every literal key this map claims must appear at a real call site."""
    src = open(MODELS_PY).read()
    literals = set(re.findall(
        r"load_best_hyperparameters\(\s*'([a-z_0-9]+)'", src))
    # THE CALL SITES NO LONGER PASS LITERALS, AND THAT IS THE POINT.
    #
    # Until 2026-08-31 each builder passed a hard-coded family name, so four
    # networks read 'dnn', four read 'mlp', the quantile forest read 'rf' and
    # both Gaussian processes read 'gauche'. Thirteen of seventeen models could
    # not be addressed on their own. The builders now derive the label from the
    # run's own arguments through rosters.roster_label(), so what has to be
    # checked is that the function produces every roster label and that the
    # builders actually call it.
    if 'roster_label(' not in src:
        fail('models.py no longer calls roster_label() — the tuned settings '
             'would go back to being shared between models')
        return
    n_dynamic = src.count('load_best_hyperparameters(_label, rep)')
    if n_dynamic < 4:
        fail(f'only {n_dynamic} call site(s) pass the derived label; the forest, '
             f'the Gaussian process, the dnn family and the mlp family all need '
             f'one, or their models silently share settings again')

    # Every label the roster knows must be produced by roster_label from some
    # argument combination -- otherwise a tuned entry written under that label
    # is unreachable, which is the defect this whole change exists to remove.
    class _Args:
        kernel = None
        bayesian_transformation = None
        heteroscedastic_vbll = False

    def _args(**kw):
        a = _Args()
        for k, v in kw.items():
            setattr(a, k, v)
        return a

    ARG_CASES = {
        'rf': ('rf', {}), 'qrf': ('qrf', {}), 'svm': ('svm', {}),
        'xgboost': ('xgboost', {}), 'lgb': ('lgb', {}), 'ngboost': ('ngboost', {}),
        'gauche_rbf': ('gauche', {'kernel': 'rbf'}),
        'gauche': ('gauche', {'kernel': 'tanimoto'}),
        'heteroscedastic_gp': ('het_gp', {'kernel': 'rbf'}),
        'dnn': ('dnn', {}), 'mlp': ('mlp', {}),
        'dnn_bnn_full': ('dnn', {'bayesian_transformation': 'full'}),
        'mlp_bnn_full': ('mlp', {'bayesian_transformation': 'full'}),
        'dnn_bnn_full_variational':
            ('dnn', {'bayesian_transformation': 'full_variational'}),
        'mlp_bnn_full_variational':
            ('mlp', {'bayesian_transformation': 'full_variational'}),
        'dnn_bnn_full_variational_hetero':
            ('dnn', {'bayesian_transformation': 'full_variational',
                     'heteroscedastic_vbll': True}),
        'mlp_bnn_full_variational_hetero':
            ('mlp', {'bayesian_transformation': 'full_variational',
                     'heteroscedastic_vbll': True}),
    }
    missing_case = [l for l in rosters.TUNED_KEY if l not in ARG_CASES]
    if missing_case:
        fail(f'no argument case here for roster model(s) {sorted(missing_case)} '
             f'— add one, or a tuned entry for it can never be shown to arrive')
    wrong = []
    for label, (model_type, kw) in sorted(ARG_CASES.items()):
        got = rosters.roster_label(model_type, _args(**kw))
        if got != label:
            wrong.append(f'{model_type} {kw} -> {got!r}, expected {label!r}')
    if wrong:
        fail('roster_label does not produce the roster label:\n      '
             + '\n      '.join(wrong))
    else:
        ok(f'roster_label produces all {len(ARG_CASES)} roster labels from the '
           f'run arguments')

    # A label recorded as having no tuned path must really have none.
    for label, key in sorted(rosters.TUNED_KEY.items()):
        if key is not None:
            continue
        if label in literals:
            fail(f'{label!r} is recorded as having no tuned path, but '
                 f'load_best_hyperparameters({label!r}, ...) is called in '
                 f'models.py — it does have one')
    claimed = {k for k in rosters.TUNED_KEY.values() if k is not None}
    unbacked = {k for k in claimed if k not in ARG_CASES}
    if unbacked:
        fail(f'tuned key(s) {sorted(unbacked)} appear at no call site in '
             f'models.py; nothing would ever read them')
    else:
        ok(f'every tuned key is reachable from a real run '
           f'({len(rosters.TUNED_KEY)} roster models, {n_dynamic} builders '
           f'deriving their label)')


def _code_only(src):
    """The source with comments removed, so a name can only be found in code.

    This check searches for the quoted name. Until 2026-08-30 it searched the
    raw text, and 'GP-Hetero' was satisfied by a COMMENT at line 296 while the
    name itself was assembled with an f-string and existed nowhere as a literal
    -- a check that passes on prose passes on stale prose. The names are now
    spelled out in GP_DISPLAY_NAMES on that side, and this side no longer reads
    comments.
    """
    import io
    import tokenize
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type != tokenize.COMMENT:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return src        # unparseable: fall back rather than pass everything
    return '\n'.join(out)


def check_experimental_names_exist():
    """Every experimental name must be a name that pipeline actually uses."""
    path = rosters.KIRBY_PIPELINE_PATH
    if not os.path.exists(path):
        print(f'  skip  {path} not on this machine')
        return
    src = _code_only(open(path).read())
    for qm9_name, exp_name in sorted(rosters.MODEL_NAME_MAP.items()):
        if f"'{exp_name}'" not in src:
            fail(f'the experimental pipeline has no model called {exp_name!r} '
                 f'(mapped from {qm9_name!r})')
    for qm9_rep, exp_rep in sorted(rosters.REP_NAME_MAP.items()):
        if f"'{exp_rep}'" not in src:
            fail(f'the experimental pipeline has no representation called '
                 f'{exp_rep!r} (mapped from {qm9_rep!r})')
    if not FAILURES:
        ok(f'all {len(rosters.MODEL_NAME_MAP)} model and '
           f'{len(rosters.REP_NAME_MAP)} representation names exist in '
           f'{os.path.basename(path)}')


def check_master_file():
    if not os.path.exists(MASTER):
        print(f'  skip  {MASTER} does not exist yet')
        return {}
    with open(MASTER) as fh:
        master = json.load(fh)

    known_keys = set(rosters.TUNED_KEY.values())
    collapsed = rosters.collapsed_models()
    shared = {rosters.TUNED_KEY[m] for m in collapsed}

    for key, by_rep in sorted(master.items()):
        if key not in known_keys:
            fail(f'master file: model key {key!r} is not read by any model on '
                 f'the roster. Known keys: {sorted(known_keys)}')
            continue
        if key in shared:
            sharers = sorted(m for m in rosters.TUNED_KEY
                             if rosters.TUNED_KEY[m] == key)
            fail(f'master file: {key!r} is read by {sharers}, so a value here '
                 f'changes models it was not measured on')
        for rep in sorted(by_rep):
            if rep not in rosters.ALL_REPS:
                fail(f'master file: {key}/{rep} — {rep!r} is not on the '
                     f'representation roster {rosters.ALL_REPS}')
    if not FAILURES:
        ok(f'master file: {len(master)} model key(s), every name on the roster')
    return master


def check_decisions_file(master):
    if not os.path.exists(DECISIONS):
        print(f'  skip  {DECISIONS} does not exist yet')
        return
    with open(DECISIONS) as fh:
        decisions = json.load(fh)

    known_keys = set(rosters.TUNED_KEY.values())
    for key, value in sorted(decisions.items()):
        if key not in known_keys:
            fail(f'decisions file: {key!r} is not read by any model on the '
                 f'roster')
        if value != 'USE_TUNED':
            fail(f'decisions file: {key} is {value!r}. The reader compares to '
                 f'the exact string "USE_TUNED"; anything else silently means '
                 f'defaults.')
    if master:
        no_decision = sorted(set(master) - set(decisions))
        if no_decision:
            fail(f'{no_decision} have tuned parameters and no USE_TUNED entry, '
                 f'so those parameters are never read')
        no_params = sorted(k for k, v in decisions.items()
                           if v == 'USE_TUNED' and k not in master)
        if no_params:
            fail(f'{no_params} are marked USE_TUNED with no entry in the '
                 f'master file')
    if not FAILURES:
        ok(f'decisions file: {len(decisions)} entries, all USE_TUNED, all '
           f'matched to the master file')


def main():
    print('the name map against the rosters')
    check_map_covers_roster()
    check_tuned_keys_against_models_py()

    print('\nthe experimental pipeline\'s names')
    check_experimental_names_exist()

    print('\nthe two files the pipeline reads')
    master = check_master_file()
    check_decisions_file(master)

    print()
    if FAILURES:
        print(f'{len(FAILURES)} failure(s)')
        return 1
    print('all checks passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
