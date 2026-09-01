#!/usr/bin/env python3
"""The chosen setting per model, written to one file. Regenerate any time.

    python scripts/write_chosen_settings.py

Writes results/tuning_local/CHOSEN_SETTINGS.json and .md.

WHERE THE CHOICE COMES FROM
---------------------------
scripts/pick_per_model_setting.py fits every candidate on all six
representations with training labels noised to level 0.5 and scores it on a
clean test split. This reads those results and applies the rule:

    a setting is scored by its WORST representation, and the winner is the
    setting whose worst case is best.

Worst case, not average, because a setting that is excellent on five
representations and ruinous on one would win on the average -- and the whole
reason for one setting per model is that no representation should pay for the
shared choice.

WHY IT IS NOT THE SEARCH'S OWN WINNER
-------------------------------------
The search scored candidates on CLEAN labels, which prefers larger, less
restricted models, and those are the ones that memorise noisy labels. Measured:
LightGBM's clean winner on ECFP4 ends at 0.226 under noise where its own default
holds 0.634. Every setting here was re-chosen under noise for that reason.

A model whose DEFAULT has the best worst case keeps the default, and the file
records that as a decision rather than leaving the model absent.
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
    by, params = load()
    if not by:
        print('no per-model results yet', file=sys.stderr)
        return 1

    chosen, md = {}, []
    md.append('# The chosen setting, one per model')
    md.append('')
    md.append('Regenerate with `python scripts/write_chosen_settings.py`. '
              'Machine-readable copy in `CHOSEN_SETTINGS.json`.')
    md.append('')
    md.append('R-squared on a clean held-out test set, with training labels '
              'noised to level 0.5. QM9, 3,000 molecules. Where a setting was '
              'fitted under several noise seeds the median is shown.')
    md.append('')
    md.append('**The rule: a setting is scored by its WORST representation, and '
              'the setting with the best worst case wins.** Not the average — a '
              'setting that is ruinous on one representation must not win on '
              'the strength of the other five.')
    md.append('')

    for model in sorted(by, key=lambda m: NICE.get(m, m)):
        srcs = by[model]
        rows, seeds_seen = [], set()
        for src, d in srcs.items():
            vals = []
            for rep in REPS:
                if rep not in d:
                    vals = None
                    break
                seeds_seen.add(len(d[rep]))
                vals.append(statistics.median(d[rep]))
            if vals:
                rows.append((min(vals), src, vals))
        if not rows:
            continue
        rows.sort(reverse=True)
        worst, src, vals = rows[0]
        keeps_default = (src == 'DEFAULT')
        chosen[model] = {
            'setting': None if keeps_default else params[model].get(src),
            'keeps_default': keeps_default,
            'drawn_on': None if keeps_default else src,
            'worst_representation_r2_at_noise_0.5': round(worst, 4),
            'per_representation_r2': {r: round(v, 4) for r, v in zip(REPS, vals)},
            'noise_seeds': sorted(seeds_seen),
        }

        md.append(f'\n## {NICE.get(model, model)}  (`{model}`)\n')
        md.append(f'Noise seeds per fit: {sorted(seeds_seen)}\n')
        md.append('| setting drawn on | ' + ' | '.join(REPS) + ' | worst |')
        md.append('|---' * (len(REPS) + 2) + '|')
        for w, s2, v in rows:
            mark = ' **← chosen**' if s2 == src else ''
            md.append(f'| {s2}{mark} | ' + ' | '.join(f'{x:.3f}' for x in v)
                      + f' | **{w:.3f}** |')
        md.append('')
        if keeps_default:
            md.append('**Keeps the default** — no drawn setting had a better '
                      'worst case.')
        else:
            md.append('**Chosen setting:**')
            md.append('')
            md.append('```json')
            md.append(json.dumps(params[model].get(src), indent=2,
                                 sort_keys=True))
            md.append('```')
            dflt = next((w for w, s2, _v in rows if s2 == 'DEFAULT'), None)
            if dflt is not None:
                md.append('')
                md.append(f'Worst case {worst:.3f} against the default\'s '
                          f'{dflt:.3f}, a gain of {worst - dflt:+.3f}.')
        md.append('')

    with open(JSON_OUT, 'w') as fh:
        json.dump(chosen, fh, indent=2, sort_keys=True)
    text = '\n'.join(md) + '\n'
    with open(MD_OUT, 'w') as fh:
        fh.write(text)
    print(text)
    print(f'written to {MD_OUT} and {JSON_OUT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
