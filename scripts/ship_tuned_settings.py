#!/usr/bin/env python3
"""Write the chosen settings into the two files the cluster reads. One command.

    python scripts/ship_tuned_settings.py          # show what it would write
    python scripts/ship_tuned_settings.py --write  # write them

Then:  git add results/master_tuned_hyperparameters.json \\
                results/hyperparameter_decisions.json
       git commit -m "tuned settings for the four Bayesian families"
       git push

WHAT IT WRITES
--------------
results/master_tuned_hyperparameters.json   keyed [model][representation]
results/hyperparameter_decisions.json       {model: "USE_TUNED"}

Both are read by load_best_hyperparameters in models/models.py, and BOTH must
say yes: a model with settings in the master file and no USE_TUNED entry trains
on the defaults, silently.

ONE SETTING PER MODEL, WRITTEN UNDER EVERY REPRESENTATION. The file's shape is
per representation, the decision is per model, so the same setting is written
six times. That is deliberate and is what per-model tuning means here.

WHY THESE FILES ARE NOT GITIGNORED WHEN EVERYTHING ELSE UNDER results/ IS
--------------------------------------------------------------------------
They are not run output. They are a decision the cluster has to read, and
`results/` being ignored meant they could not be committed or pulled at all --
so a tuned setting could never reach a job however carefully it was chosen.
Two negations in .gitignore, added 2026-09-01.

TWO FILES, because the two pipelines ask different questions.
`master_tuned_hyperparameters.json` is keyed by model and representation and is
read by QM9. `master_tuned_hyperparameters_lab.json` is keyed by DATASET first,
then model and representation, and is read by
alternative_data_noise_robustness.py for LogD, Caco-2 and hERG -- which choose
their own setting per dataset, so one flat file cannot hold both.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))

CHOSEN = os.path.join(_ROOT, 'results', 'tuning_local', 'CHOSEN_SETTINGS.json')
MASTER = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters.json')
DECISIONS = os.path.join(_ROOT, 'results', 'hyperparameter_decisions.json')
# LogD, Caco-2 and hERG choose their own setting per dataset, so the QM9 file --
# keyed by model and representation, with no room for a dataset -- cannot carry
# them. They get their own file, keyed by dataset first, read by
# alternative_data_noise_robustness.py.
LAB = os.path.join(_ROOT, 'results', 'master_tuned_hyperparameters_lab.json')
LAB_DATASETS = ['herg', 'caco2', 'logd']
REPS = ['ecfp4', 'pdv', 'mhggnn', 'avalon', 'chemberta', 'sns']


def main():
    write = '--write' in sys.argv
    if not os.path.exists(CHOSEN):
        print(f'{CHOSEN} does not exist. Run:\n'
              f'  python scripts/write_chosen_settings.py', file=sys.stderr)
        return 1
    chosen = json.load(open(CHOSEN))

    import tuning_rosters as R
    # CHOSEN_SETTINGS.json is keyed 'dataset/model' and holds the parameters
    # themselves, or null where the model keeps its default. The master file the
    # QM9 pipeline reads is keyed by tuned-file key and repeats the one setting
    # under every representation -- that is the point of the whole exercise: one
    # setting per model, shared across representations.
    master, decisions, skipped = {}, {}, []
    for pair, setting in sorted(chosen.items()):
        dataset, _, model = pair.partition('/')
        if dataset != 'qm9':
            continue
        if not setting:
            skipped.append((model, 'keeps the default by rule'))
            continue
        key = R.TUNED_KEY.get(model)
        if key is None:
            skipped.append((model, 'its builder never reads a tuned file'))
            continue
        master[key] = {rep: setting for rep in REPS}
        decisions[key] = 'USE_TUNED'
    if not any(k.startswith('qm9/') for k in chosen):
        print('no QM9 rows in CHOSEN_SETTINGS.json — nothing to ship',
              file=sys.stderr)
        return 1

    print(f'{len(master)} model(s) would be written, each under '
          f'{len(REPS)} representations:\n')
    for k, v in sorted(master.items()):
        print(f'  {k}')
        print(f'      {json.dumps(v[REPS[0]], sort_keys=True)}')
    for m, why in skipped:
        print(f'  {m}: SKIPPED — {why}')

    if not write:
        print('\nnothing written. Add --write to write it.')
        return 0

    lab = {}
    for pair, setting in sorted(chosen.items()):
        dataset, _, model = pair.partition('/')
        if dataset not in LAB_DATASETS or not setting:
            continue
        key = R.TUNED_KEY.get(model)
        if key is None:
            continue
        lab.setdefault(dataset, {})[key] = {rep: setting for rep in REPS}

    with open(MASTER, 'w') as fh:
        json.dump(master, fh, indent=2, sort_keys=True)
    with open(LAB, 'w') as fh:
        json.dump(lab, fh, indent=2, sort_keys=True)
    print(f'\nwrote {LAB}')
    for ds in LAB_DATASETS:
        got = sorted(lab.get(ds, {}))
        print(f'  {ds}: {len(got)} model(s) tuned' +
              (f' — {", ".join(got)}' if got else ' — all keep their defaults'))
    with open(DECISIONS, 'w') as fh:
        json.dump(decisions, fh, indent=2, sort_keys=True)
    print(f'\nwrote {MASTER}\nwrote {DECISIONS}')
    print('\nNow:\n  git add results/master_tuned_hyperparameters.json '
          'results/hyperparameter_decisions.json\n'
          '  git commit -m "tuned settings"\n  git push')
    return 0


if __name__ == '__main__':
    sys.exit(main())
