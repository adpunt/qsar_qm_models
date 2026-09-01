#!/usr/bin/env python3
"""The final list: one setting per model per dataset. THE RULES, IN ONE PLACE.

    python scripts/final_tuned_list.py

Writes results/tuning_local/FINAL_LIST.md and .json.

WHICH MODELS ARE TUNED
----------------------
Four, and no others: the two Bayesian networks and the two variational ones.
Every other model keeps its default. Measured over all four datasets, these four
gain a median +0.19 in validation R-squared and win on 24 of 24 pairings, where
every other family gains 0.02 to 0.04. Their defaults do not merely underperform
-- the Bayesian alpha network scores -0.5568 on QM9 PDV untuned.

ONE SETTING PER MODEL, NOT PER REPRESENTATION
---------------------------------------------
The headline analysis splits variance over model and representation and reports
their interaction. If each cell carried its own hyperparameters they would vary
along the same axis as that interaction, and the tuning could never be separated
from it. One setting per model per dataset.

THE THREE FILTERS -- a setting must pass all three
---------------------------------------------------
1. EXTREME. No value at the end of its range that makes the model bigger or
   slower. A width at its SMALLEST option is at an end too and is not flagged:
   it asks for a cheaper model.
2. COMPUTE. Not two times slower to train than the default, measured from the
   seconds each fit recorded.
3. NOISE. R-squared at noise level 0.5 not below the default's. Applied where a
   noise test exists, which is PDV and ChemBERTa. A setting drawn on an untested
   representation is not failed for lack of a test.

Plus the obvious: it must beat its own default on clean labels.

THE TIEBREAK
------------
Filtering usually leaves more than one survivor -- variational alpha passes on
all six representations on QM9 -- so one number has to choose. It is the
setting's GAIN OVER ITS OWN DEFAULT on the representation it was drawn on: a
difference against its own baseline, comparable across representations in a way
the raw scores are not, and needing no new fitting.

A model where nothing survives keeps its default, recorded as a decision.
"""
from __future__ import annotations

import csv
import glob
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

    # R-squared at noise level 0.5, per (model, representation). QM9 only, and
    # only PDV and ChemBERTa were tested -- a setting drawn elsewhere is not
    # failed for lack of a test.
    noise = {}
    for f in glob.glob(os.path.join(OUT_DIR, 'noise_vs_tuned_*.csv')):
        for r in csv.DictReader(open(f)):
            if r['status'] == 'ok' and abs(float(r['level']) - 0.5) < 1e-9:
                noise.setdefault((r['model'], r['rep']), {})[r['setting']] = \
                    float(r['r2'])

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

        # THE FULL TABLE FIRST. The chosen row alone hides the evidence, and the
        # noise columns on it are usually empty because the winning setting
        # tends to come from a representation the noise test did not cover.
        md.append('### every pairing, and how it did against each filter\n')
        md.append('| model | rep | R2 clean: default | R2 clean: tuned | '
                  'R2 noise 0.5: default | R2 noise 0.5: tuned | extreme | '
                  'compute | noise |')
        md.append('|---|---|---|---|---|---|---|---|---|')
        for m, nice in MODELS:
            fam = T.search_family(m, R) or m
            sp = ranges.get(fam, {})
            for rep in REPS:
                c = cands.get((m, rep))
                if not c:
                    continue
                score, params = max(c, key=lambda t: t[0])
                d = default.get((m, rep))
                if d is None or not params:
                    continue
                ex = 'FAIL' if D.significant(params, sp, fam) else 'pass'
                rr = trials.get((ds, m, rep), [])
                b = next((x for l, _r, x in rr if l == 'default'), None)
                w = D.seconds_for(rr, score)
                ratio = (w / b) if (b and w and b > 0) else None
                co = 'FAIL' if (ratio and ratio >= SLOW) else 'pass'
                nz = noise.get((m, rep), {}) if ds == 'qm9' else {}
                nd = nz.get('default')
                nt = next((v for k, v in nz.items() if k != 'default'), None)
                if nd is None or nt is None:
                    ns, nds, nts = 'not tested', '—', '—'
                else:
                    ns = 'FAIL' if nt < nd else 'pass'
                    nds, nts = f'{nd:.3f}', f'{nt:.3f}'
                beat = '' if score > d else ' (loses on clean)'
                md.append(f'| {nice} | {rep} | {d:.3f} | {score:.3f}{beat} | '
                          f'{nds} | {nts} | {ex} | {co} | {ns} |')
        md.append('')
        md.append('### the chosen setting\n')
        md.append('| model | drawn on | R2 clean: default | R2 clean: tuned | '
                  'R2 noise 0.5: default | R2 noise 0.5: tuned | passed | '
                  'chosen setting |')
        md.append('|---|---|---|---|---|---|---|---|')
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
                nz = noise.get((m, rep), {}) if ds == 'qm9' else {}
                nd = nz.get('default')
                nt = next((v for k, v in nz.items() if k != 'default'), None)
                if nd is not None and nt is not None and nt < nd:
                    continue          # filter 3: worse at noise 0.5
                surv.append((score - d, rep, params, d, score, nd, nt))
            if not surv:
                md.append(f'| {nice} | — | — | — | — | — | 0 of 6 | '
                          f'**keeps the default** |')
                out.setdefault(ds, {})[m] = {'keeps_default': True}
                continue
            surv.sort(key=lambda t: -t[0])
            gain, rep, params, cd, ct, nd, nt = surv[0]
            f2 = lambda v: '—' if v is None else f'{v:.3f}'
            md.append(f'| {nice} | {rep} | {cd:.3f} | {ct:.3f} | {f2(nd)} | '
                      f'{f2(nt)} | {len(surv)} of 6 | '
                      f'`{json.dumps(params, sort_keys=True)}` |')
            out.setdefault(ds, {})[m] = {
                'keeps_default': False, 'drawn_on': rep,
                'r2_clean_default': round(cd, 4), 'r2_clean_tuned': round(ct, 4),
                'r2_noise05_default': None if nd is None else round(nd, 4),
                'r2_noise05_tuned': None if nt is None else round(nt, 4),
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
