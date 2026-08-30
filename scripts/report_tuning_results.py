#!/usr/bin/env python3
"""The tuning results, in the shape the author asked for. Regenerate any time.

    python scripts/report_tuning_results.py

Writes results/tuning_local/TUNING_RESULTS.md and prints the same thing.

WHY THIS IS A SCRIPT AND NOT A PASTED TABLE
-------------------------------------------
The searches finish at different times and a table pasted into a chat is stale
the moment the next pairing lands. This reads whatever result files exist and
rebuilds the whole report, so the answer to "where are the results" is one
command rather than a scroll through a conversation.

THE SHAPE, WHICH IS NOT NEGOTIABLE (RERUN_PLAN.md 5.7p)
-------------------------------------------------------
Per dataset, four groups and nothing else:

  1. no value at an end AND beats the default   -> RECORD, used going forward
  2. one value at an end, and NOT depth          -> second pass over alternatives
  3. two or more values at an end                -> discuss alternatives
  4. depth at an end                             -> discuss, whatever else it is

One value at an end is acceptable on its own. Depth is its own group because a
network at its maximum layer count or a tree at its maximum depth is saying the
SHAPE is constrained, not that a regularisation setting wants turning.

Then, per model family, the table the author asked for: one row per pairing,
columns model / representation / default / best / one column per hyperparameter,
with values at an end of their range STARRED. The search ranges are printed once
above each table. Then a second table of the runner-up settings, same columns,
containing only the rows that had a star.

WHAT NOT TO ADD
---------------
No invented columns. Earlier versions of this report carried "best interior" and
"cost", which the author had to ask about because they meant nothing, and a
position-in-range percentage that was a log-scale artefact. Raw values, the
default beside them, and a star. Nothing else.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')
REPORT = os.path.join(OUT_DIR, 'TUNING_RESULTS.md')

sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))

DEPTH = {'num_hidden_layers', 'max_depth'}

# Which result files belong to which dataset. QM9 is everything that is not one
# of the three named ones.
VALIDATION = ('herg', 'logd', 'caco2')

# The parameters each family is searched on, so a table has one column each.
FAMILIES = [
    ('network alpha', ['dnn', 'dnn_bnn_full', 'dnn_bnn_full_variational'],
     'dnn', ['hidden_size1', 'hidden_size2', 'activation']),
    ('network beta', ['mlp', 'mlp_bnn_full', 'mlp_bnn_full_variational'],
     'mlp', ['hidden_size', 'num_hidden_layers', 'dropout_rate', 'lr']),
    ('random forest', ['rf', 'qrf'], 'rf',
     ['n_estimators', 'max_depth', 'max_features', 'min_samples_leaf',
      'min_samples_split', 'bootstrap']),
    ('XGBoost', ['xgboost'], 'xgboost',
     ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
      'colsample_bytree', 'colsample_bylevel', 'min_child_weight', 'gamma',
      'reg_alpha', 'reg_lambda']),
    ('LightGBM', ['lgb'], 'lgb',
     ['n_estimators', 'learning_rate', 'num_leaves', 'max_depth', 'subsample',
      'colsample_bytree', 'min_child_samples', 'reg_alpha', 'reg_lambda']),
    ('support vector machine', ['svm'], 'svm', ['C', 'gamma', 'kernel']),
    ('NGBoost', ['ngboost'], 'ngboost',
     ['n_estimators', 'learning_rate', 'natural_gradient']),
    ('Gaussian process', ['gauche_rbf', 'gauche'], 'gauche',
     ['outputscale', 'likelihood_noise']),
]


def load(dataset):
    """Every scored fit for one dataset, with its setting."""
    import analyse_tuned_settings as A
    import tune_hyperparameters as T
    import tuning_rosters as R

    files = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'trials_*.csv'))):
        name = os.path.basename(path)
        is_val = any(v in name for v in VALIDATION)
        if (dataset == 'qm9') == (not is_val) and (
                dataset == 'qm9' or dataset in name):
            files.append(path)
    rows = []
    for path in files:
        with open(path) as fh:
            rows += [r for r in csv.DictReader(fh) if r['status'] == 'ok']
    if not rows:
        return None, None, None, None

    sizes = {r['sample_size'] for r in rows if r.get('sample_size')}
    final = {(r['model'], r['rep']) for r in rows
             if r['setting'].startswith('final_')}
    default, cands = {}, defaultdict(list)

    # Settings are recorded in the detail column where the run wrote them. Runs
    # rebuilt from a log have no detail, so the same draws are reproduced from
    # the seed the run used -- the generator is deterministic and the roster
    # order is fixed, so this reconstructs exactly what was fitted.
    rng = random.Random(42)
    drawn = {}
    models_here = sorted({r['model'] for r in rows})
    for m, rep in R.qm9_pairings():
        if m not in models_here:
            continue
        drawn[(m, rep)] = [T.sample_setting(R.TUNED_KEY[m], rng, m)
                           for _ in range(12)]

    for r in rows:
        k = (r['model'], r['rep'])
        if r['setting'] == 'default':
            default[k] = float(r['r2'])
            continue
        if r['setting'].startswith('screen_') and k in final:
            continue
        params = json.loads(r['detail']) if r['detail'] else None
        if params is None and '_' in r['setting'] and k in drawn:
            n = int(r['setting'].rsplit('_', 1)[1])
            if 1 <= n <= len(drawn[k]):
                params = drawn[k][n - 1]
        cands[k].append((float(r['r2']), params or {}))
    return default, cands, sorted(sizes), A.observed_ranges()


def fmt(v):
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f'{v:.4g}'
    return str(v)


def report(dataset, out):
    import analyse_tuned_settings as A
    import tuning_rosters as R

    default, cands, sizes, ranges = load(dataset)
    if not cands:
        out.append(f'\n## {dataset} — no results yet\n')
        return

    out.append(f'\n## {dataset}\n')
    out.append(f'Molecules: {", ".join(sizes)}. '
               f'{len(cands)} pairings with at least one scored setting.\n')

    groups = {'record': [], 'worse': [], 'one': [], 'two': [], 'depth': []}
    for k, c in sorted(cands.items()):
        sp = ranges.get(R.TUNED_KEY.get(k[0], k[0]), {})
        if not sp:
            continue
        score, params = max(c, key=lambda t: t[0])
        d = default.get(k)
        if d is None or not params:
            continue
        e = A.edges_in(params, sp)
        de = [x for x in e if x in DEPTH]
        if not e:
            groups['record' if score > d else 'worse'].append((k, d, score, params, e))
        elif de:
            groups['depth'].append((k, d, score, params, e))
        elif len(e) == 1:
            groups['one'].append((k, d, score, params, e))
        else:
            groups['two'].append((k, d, score, params, e))

    titles = [
        ('record', 'RECORD — no value at an end, beats the default. USE THESE.'),
        ('worse', 'KEEP THE DEFAULT — no value at an end but the default wins'),
        ('one', 'ONE VALUE AT AN END, NOT DEPTH — needs a second pass'),
        ('two', 'TWO OR MORE VALUES AT AN END — discuss alternatives'),
        ('depth', 'DEPTH AT AN END — discuss alternatives'),
    ]
    for key, title in titles:
        rowsx = groups[key]
        out.append(f'\n### {title}  ({len(rowsx)})\n')
        if not rowsx:
            out.append('none\n')
            continue
        out.append('| model | representation | default | tuned | at an end |')
        out.append('|---|---|---|---|---|')
        for (m, rep), d, s, p, e in rowsx:
            out.append(f'| {m} | {rep} | {d:+.4f} | {s:+.4f} | '
                       f'{", ".join(e) if e else "—"} |')
        out.append('')

    # Per family: best, then runner-ups for the starred rows.
    for fam_name, models, key, cols in FAMILIES:
        present = [k for k in cands if k[0] in models]
        if not present:
            continue
        sp = ranges.get(key, {})
        out.append(f'\n### {fam_name}\n')
        rng_txt = ' · '.join(
            f'{c} {A.describe(sp[c])[0]}' for c in cols if c in sp)
        out.append(f'Search ranges: {rng_txt}\n')

        for label, want_star in (('Best', None), ('Runner-up, starred rows only', True)):
            lines = []
            for k in sorted(present):
                ranked = sorted(cands[k], key=lambda t: -t[0])
                best_e = A.edges_in(ranked[0][1], sp)
                if want_star and not best_e:
                    continue
                if want_star:
                    clean = [(s, p) for s, p in ranked if not A.edges_in(p, sp)]
                    if not clean:
                        lines.append(f'| {k[0]} | {k[1]} | '
                                     f'{default.get(k, float("nan")):+.4f} | — | '
                                     + ' | '.join(['—'] * len(cols)) + ' |')
                        continue
                    score, params = clean[0]
                else:
                    score, params = ranked[0]
                e = A.edges_in(params, sp)
                cells = []
                for c in cols:
                    v = fmt(params.get(c, '—'))
                    cells.append(f'**{v}\\***' if c in e else v)
                lines.append(f'| {k[0]} | {k[1]} | '
                             f'{default.get(k, float("nan")):+.4f} | {score:+.4f} | '
                             + ' | '.join(cells) + ' |')
            if not lines:
                continue
            out.append(f'\n**{label}**\n')
            out.append('| model | rep | default | score | ' + ' | '.join(cols) + ' |')
            out.append('|---' * (4 + len(cols)) + '|')
            out += lines
            out.append('')


def main():
    out = ['# Hyperparameter tuning results',
           '',
           'Regenerate with `python scripts/report_tuning_results.py`.',
           'Raw fits are in `results/tuning_local/trials_*.csv`; the settings to '
           'use are in `recorded_for_<dataset>.json`.',
           '',
           'A starred value sits at an end of the range the search could draw '
           'from, which means the range chose it rather than the data.',
           '']
    for ds in ('qm9', 'herg', 'logd', 'caco2'):
        report(ds, out)
    text = '\n'.join(out) + '\n'
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(REPORT, 'w') as fh:
        fh.write(text)
    print(text)
    print(f'\nwritten to {REPORT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
