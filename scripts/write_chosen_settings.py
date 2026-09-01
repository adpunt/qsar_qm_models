#!/usr/bin/env python3
"""The chosen setting per model, PER MODEL not per representation.

    python scripts/write_chosen_settings.py

Writes results/tuning_local/CHOSEN_SETTINGS.json and .md.

WHY THE SEARCH ALONE CANNOT ANSWER THIS
----------------------------------------
Each representation drew its own twelve candidates and scored them only on
itself. No setting was ever tried on a representation other than the one that
drew it. So the search cannot say how a model's setting behaves across
representations, and a per-model choice needs exactly that.

scripts/pick_per_model_setting.py supplies it: every candidate fitted on all six
representations with training labels noised to level 0.5, scored on a clean test
split. This reads those results.

THE RULES, APPLIED PER MODEL
-----------------------------
Candidates are ranked by their WORST representation -- not the average, because
a setting that is excellent on five and ruinous on one must not win on the
strength of the five.

Then walk DOWN that ranking and take the first candidate that passes:

  1. EXTREME   no value at the end of its range that makes the model bigger or
               slower. A width at its smallest option is not flagged.
  2. COMPUTE   no more than 2x the default's measured fit time.
  3. BEATS THE DEFAULT   its worst representation beats the default's worst
               representation. Everything here is already measured at noise 0.5,
               so this is the noise test as well.

STOP when a candidate no longer beats the default: everything below it in the
ranking is worse still, so there is nothing left to assess. If that happens
before anything passes, the model keeps its default.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')
JSON_OUT = os.path.join(OUT_DIR, 'CHOSEN_SETTINGS.json')
MD_OUT = os.path.join(OUT_DIR, 'CHOSEN_SETTINGS.md')

REPS = ['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns']
NICE = {'dnn_bnn_full': 'Bayesian alpha', 'mlp_bnn_full': 'Bayesian beta',
        'dnn_bnn_full_variational': 'variational alpha',
        'mlp_bnn_full_variational': 'variational beta'}


def load():
    """model -> setting source -> representation -> [scores, one per seed]."""
    by = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    params = collections.defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'per_model_*.csv'))):
        for r in csv.DictReader(open(path)):
            if r['status'] != 'ok':
                continue
            by[r['model']][r['from_rep']][r['applied_to']].append(float(r['r2']))
            if r['detail'] and r['from_rep'] not in params[r['model']]:
                params[r['model']][r['from_rep']] = json.loads(r['detail'])
    return by, params


def main():
    import decide_tuned_settings as D
    import tune_hyperparameters as T
    import tuning_rosters as R
    import report_tuning_results as RT

    by, params = load()
    if not by:
        print('no per-model results yet', file=sys.stderr)
        return 1
    _d, _c, _s, ranges = RT.load('qm9')
    trials = D.trial_rows()

    chosen, md = {}, []
    md.append('# The chosen setting, one per model')
    md.append('')
    md.append('Regenerate with `python scripts/write_chosen_settings.py`. '
              'Machine-readable copy in `CHOSEN_SETTINGS.json`.')
    md.append('')
    md.append('R-squared on a clean held-out test set, training labels noised to '
              'level 0.5. QM9, 3,000 molecules. Every candidate is fitted on '
              'ALL SIX representations, so these are per-model numbers.')
    md.append('')
    md.append('Candidates are ranked by their WORST representation. Walking down '
              'that ranking, the first one that passes the extreme and compute '
              'filters and beats the default is taken. The walk stops at the '
              'first candidate that no longer beats the default — everything '
              'below it is worse still.')
    md.append('')

    for model in sorted(by, key=lambda m: NICE.get(m, m)):
        fam = T.search_family(model, R) or model
        sp = ranges.get(fam, {})
        rows = []
        for src, d in by[model].items():
            vals = [statistics.median(d[r]) for r in REPS if r in d]
            if len(vals) != len(REPS):
                continue
            rows.append((min(vals), src, vals))
        if not rows:
            continue
        rows.sort(reverse=True)
        dflt = next((w for w, s2, _v in rows if s2 == 'DEFAULT'), None)

        md.append(f'\n## {NICE.get(model, model)}  (`{model}`)\n')
        md.append('| rank | setting from | ' + ' | '.join(REPS)
                  + ' | worst | extreme | compute | beats default | verdict |')
        md.append('|---' * (len(REPS) + 6) + '|')

        picked = None
        stopped = False
        for rank, (worst, src, vals) in enumerate(rows, 1):
            if src == 'DEFAULT':
                md.append(f'| {rank} | DEFAULT | '
                          + ' | '.join(f'{v:.3f}' for v in vals)
                          + f' | **{worst:.3f}** | — | — | — | the baseline |')
                continue
            p = params[model].get(src, {})
            ex = not D.significant(p, sp, fam)
            rr = trials.get(('qm9', model, src), [])
            b = next((x for l, _r, x in rr if l == 'default'), None)
            w = max((x for l, _r, x in rr if l != 'default'), default=None)
            ratio = (w / b) if (b and w and b > 0) else None
            co = not (ratio and ratio >= 2.0)
            beats = dflt is None or worst > dflt

            if stopped:
                verdict = 'not assessed'
            elif not beats:
                verdict = '**STOP — no longer beats the default**'
                stopped = True
            elif not ex:
                verdict = 'rejected: extreme'
            elif not co:
                verdict = 'rejected: too slow'
            elif picked is None:
                verdict = '**CHOSEN**'
                picked = (src, p, worst, vals)
            else:
                verdict = 'passes, but ranked below the chosen'
            md.append(f'| {rank} | {src} | '
                      + ' | '.join(f'{v:.3f}' for v in vals)
                      + f' | **{worst:.3f}** | {"pass" if ex else "FAIL"} | '
                      f'{"pass" if co else "FAIL"} | '
                      f'{"yes" if beats else "no"} | {verdict} |')
        md.append('')

        if picked is None:
            md.append('**Keeps the default** — every candidate above the default '
                      'was rejected by a filter, and the walk stopped where they '
                      'stopped beating it.')
            chosen[model] = {'keeps_default': True}
        else:
            src, p, worst, vals = picked
            md.append('**Chosen setting:**')
            md.append('')
            md.append('```json')
            md.append(json.dumps(p, indent=2, sort_keys=True))
            md.append('```')
            md.append('')
            if dflt is not None:
                md.append(f'Worst representation {worst:.3f} against the '
                          f'default\'s {dflt:.3f}, a gain of {worst - dflt:+.3f}.')
            chosen[model] = {
                'keeps_default': False, 'drawn_on': src, 'setting': p,
                'worst_rep_r2_at_noise_0.5': round(worst, 4),
                'default_worst_rep': None if dflt is None else round(dflt, 4),
                'per_representation_r2': {r: round(v, 4)
                                          for r, v in zip(REPS, vals)}}
        md.append('')

    with open(JSON_OUT, 'w') as fh:
        json.dump(chosen, fh, indent=2, sort_keys=True)
    text = '\n'.join(md) + '\n'
    with open(MD_OUT, 'w') as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
