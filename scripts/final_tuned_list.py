#!/usr/bin/env python3
"""The final list: one setting per model per dataset, from the search alone.

    python scripts/final_tuned_list.py

Writes results/tuning_local/FINAL_LIST.md and .json.

HOW A SETTING IS CHOSEN -- NO EXTRA FITTING
--------------------------------------------
Each model's six per-representation winners are put through the three filters:

  1. no value at the expensive end of its range
  2. not two times slower to train than the default
  3. beats its own default

Filtering usually leaves more than one survivor -- variational alpha passes on
all six representations on QM9 -- so a tiebreak is needed. The tiebreak is the
setting's GAIN OVER ITS OWN DEFAULT on the representation it was drawn on. That
number is a difference measured on one representation against its own baseline,
so it is comparable across representations in a way the raw scores are not, and
it needs no new fitting.

A model where nothing survives keeps its default, recorded as a decision.
"""
from __future__ import annotations

import json
import os
import sys

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import lightgbm as _first  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_ROOT, 'models')):
    if _p not in sys.path:
        sys.path.insert(0, _p)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

MODELS = [('dnn_bnn_full', 'Bayesian alpha'),
          ('mlp_bnn_full', 'Bayesian beta'),
          ('dnn_bnn_full_variational', 'variational alpha'),
          ('mlp_bnn_full_variational', 'variational beta')]
REPS = ['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns']
SLOW = 2.0


def main():
    import decide_tuned_settings as D
    import report_tuning_results as RT
    import tune_hyperparameters as T
    import tuning_rosters as R

    trials = D.trial_rows()
    out, md = {}, ['# The final tuned settings', '',
                   'Regenerate with `python scripts/final_tuned_list.py`. '
                   'Machine-readable copy in `FINAL_LIST.json`.', '',
                   'One setting per model per dataset. Chosen from the search '
                   'results by the three filters, with ties broken on gain over '
                   'the setting\'s own default. No extra fitting.', '',
                   'Every model not listed here keeps its default.', '']

    for ds in ('qm9', 'herg', 'caco2', 'logd'):
        default, cands, _s, ranges = RT.load(ds)
        if not cands:
            md.append(f'\n## {ds}\n\nno search results yet\n')
            continue
        md.append(f'\n## {ds}\n')
        md.append('| model | setting drawn on | gain over its own default | '
                  'settings that passed | chosen setting |')
        md.append('|---|---|---|---|---|')
        for m, nice in MODELS:
            fam = T.search_family(m, R) or m
            sp = ranges.get(fam, {})
            surv = []
            for rep in REPS:
                c = cands.get((m, rep))
                if not c:
                    continue
                score, params = max(c, key=lambda t: t[0])
                d = default.get((m, rep))
                if d is None or not params or score <= d:
                    continue
                if D.significant(params, sp, fam):
                    continue
                rows = trials.get((ds, m, rep), [])
                b = next((x for l, _r, x in rows if l == 'default'), None)
                w = D.seconds_for(rows, score)
                if b and w and b > 0 and w / b >= SLOW:
                    continue
                surv.append((score - d, rep, params))
            if not surv:
                md.append(f'| {nice} | — | — | 0 | **keeps the default** |')
                out.setdefault(ds, {})[m] = {'keeps_default': True}
                continue
            surv.sort(key=lambda t: -t[0])
            gain, rep, params = surv[0]
            md.append(f'| {nice} | {rep} | {gain:+.4f} | {len(surv)} of 6 | '
                      f'`{json.dumps(params, sort_keys=True)}` |')
            out.setdefault(ds, {})[m] = {
                'keeps_default': False, 'drawn_on': rep,
                'gain_over_own_default': round(gain, 4),
                'settings_that_passed': len(surv), 'setting': params}
        md.append('')

    with open(os.path.join(OUT_DIR, 'FINAL_LIST.json'), 'w') as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    text = '\n'.join(md) + '\n'
    with open(os.path.join(OUT_DIR, 'FINAL_LIST.md'), 'w') as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
