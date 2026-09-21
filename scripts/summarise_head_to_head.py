#!/usr/bin/env python3
"""Read the head-to-head files and say which setting wins, per model.

    python scripts/summarise_head_to_head.py

One row of the printed table is one model on one dataset. The columns are the
candidate settings, and each cell is R-squared on the clean test split. The
winner is the setting with the highest R-squared summed over the noise levels
fitted, which is stated per table rather than assumed.
"""
from __future__ import annotations

import argparse
import collections
import csv
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', default='head_to_head_*.csv')
    cli = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(OUT_DIR, cli.pattern))):
        if 'smoke' in os.path.basename(p):
            continue
        with open(p) as fh:
            rows += [r for r in csv.DictReader(fh) if r['status'] == 'ok']
    if not rows:
        raise SystemExit('no finished fits found')

    by = collections.defaultdict(dict)
    levels = sorted({float(r['level']) for r in rows})
    for r in rows:
        by[(r['dataset'], r['rep'], r['model'])][(r['setting'], float(r['level']))] = float(r['r2'])

    settings_of = collections.defaultdict(list)
    for r in rows:
        fam = ('forest' if r['model'] in ('rf', 'qrf')
               else 'alpha' if r['model'].startswith('dnn') else 'beta')
        if r['setting'] not in settings_of[fam]:
            settings_of[fam].append(r['setting'])

    for dataset in sorted({r['dataset'] for r in rows}):
        for rep in sorted({r['rep'] for r in rows if r['dataset'] == dataset}):
            print(f'\n{dataset}  {rep}   '
                  f'R-squared on the clean test split, one row per model, '
                  f'one column per candidate setting and noise level')
            for fam in ('alpha', 'beta', 'forest'):
                names = settings_of[fam]
                if not names:
                    continue
                models = sorted({m for (d, rp, m) in by
                                 if d == dataset and rp == rep
                                 and (('forest' if m in ('rf', 'qrf')
                                       else 'alpha' if m.startswith('dnn')
                                       else 'beta') == fam)})
                if not models:
                    continue
                head = '  '.join(f'{n[:12]:>13s}' for n in names)
                print(f'  level {"":34s}{head}')
                for m in models:
                    cells = by[(dataset, rep, m)]
                    for lv in levels:
                        line = '  '.join(
                            (f'{cells[(n, lv)]:+13.4f}' if (n, lv) in cells
                             else f'{"—":>13s}') for n in names)
                        best = max(((n, cells[(n, lv)]) for n in names
                                    if (n, lv) in cells),
                                   key=lambda t: t[1], default=(None, None))
                        tag = f'  best {best[0]}' if best[0] else ''
                        print(f'  {lv:<5} {m:34s}{line}{tag}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
