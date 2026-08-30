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
    if not literals:
        fail('no literal load_best_hyperparameters call site found in models.py '
             '— the reader may have been renamed or removed')
        return
    # 'mlp' arrives as model_type rather than a literal; assert the dynamic call
    # site exists instead of pretending it is a literal.
    dynamic = 'load_best_hyperparameters(model_type, rep)' in src
    # None means the model has NO tuned path -- its builder never calls the
    # reader, so no entry in either JSON file can reach it. That is a fact about
    # the model, not a missing key, and it is recorded in TUNED_KEY rather than
    # left out (leaving it out made this gate report the model as missing, which
    # reads as an oversight). Check it is true rather than taking it on trust.
    for label, key in sorted(rosters.TUNED_KEY.items()):
        if key is not None:
            continue
        if label in literals:
            fail(f'{label!r} is recorded as having no tuned path, but '
                 f'load_best_hyperparameters({label!r}, ...) is called in '
                 f'models.py — it does have one')
    claimed = {k for k in rosters.TUNED_KEY.values() if k is not None}
    unbacked = {k for k in claimed if k not in literals
                and not (k == 'mlp' and dynamic)}
    if unbacked:
        fail(f'tuned key(s) {sorted(unbacked)} appear at no call site in '
             f'models.py; nothing would ever read them')
    else:
        ok(f'every tuned key is a real call site '
           f'({len(literals)} literal, model_type dynamic: {dynamic})')


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
