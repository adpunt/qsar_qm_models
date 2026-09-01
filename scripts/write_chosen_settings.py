#!/usr/bin/env python3
"""One table per dataset per model, and the setting each one picks.

    python scripts/write_chosen_settings.py

Writes results/tuning_local/CHOSEN_SETTINGS.md (the tables) and
results/tuning_local/CHOSEN_SETTINGS.json (the answer, machine readable).

WHAT IS BEING CHOSEN
--------------------
ONE setting per MODEL per DATASET. Not per representation. The models are
tested across six representations, but the six share a single setting, because
the two-way decomposition over model and representation cannot be read if each
cell carries its own hyperparameters -- a difference between two cells would
then be part model, part representation, part tuning, with no way to separate
them.

Four models are tuned: the two plain Bayesian networks and the two variational
ones. Every other model keeps its default, because measured over the search
those four gain a median +0.19 validation R2 and win every pairing, and no other
family gains more than 0.04.

HOW A SETTING IS RANKED
-----------------------
By the MEAN OF ITS CHANGES FROM THE DEFAULT, one change per representation.

If ECFP4's default scores 0.6 and this setting scores 0.7, that is +0.1. If
PDV's default scores 0.7 and this setting scores 0.5, that is -0.2. The setting
is ranked on mean(+0.1, -0.2) = -0.05.

Ranking on the raw scores instead would rank the representations, not the
settings -- a setting scored mostly on easy representations would beat a better
one scored on hard ones. Subtracting each representation's own default removes
that.

THE THREE FILTERS
-----------------
extreme  A value at the end of its range that makes the model bigger or slower:
         width at 1024 or 512, or four hidden layers. Dropout and learning rate
         are never flagged; they cost nothing at a fixed number of epochs.
compute  The setting's slowest representation takes more than twice as long as
         the default's slowest. Measured across all six, not one.
noise    Refitted with training labels noised to half the clean training label
         spread, scored on a clean test split, on PDV and ChemBERTa only. If the
         setting is worse than the default there, it fails. Clean scoring
         prefers larger, less restricted models, and those are the ones that
         memorise noisy labels -- so a clean winner is not adopted until it has
         been seen under noise.

THE WALK
--------
Start at rank 1. If it passes all three filters, it is chosen. If it fails any,
reject it and try the next. Stop at the first setting whose mean change is not
positive: from there down nothing beats the default, so the default is kept.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

DATASETS = ['qm9', 'herg', 'caco2', 'logd']
REPS = ['pdv', 'chemberta', 'ecfp4', 'avalon', 'mhggnn', 'sns']
NOISE_REPS = ['pdv', 'chemberta']
MODELS = {'dnn_bnn_full': 'Bayesian alpha',
          'mlp_bnn_full': 'Bayesian beta',
          'dnn_bnn_full_variational': 'variational alpha',
          'mlp_bnn_full_variational': 'variational beta'}

# A value at the end of its range that makes the model BIGGER OR SLOWER.
# The ranges are the search spaces in scripts/tune_hyperparameters.py.
EXTREME_AT = {'hidden_size1': 1024, 'hidden_size2': 512,
              'hidden_size': 512, 'num_hidden_layers': 4}
COMPUTE_MULTIPLE = 2.0


def is_extreme(params):
    return sorted(k for k, v in (params or {}).items()
                  if k in EXTREME_AT and v == EXTREME_AT[k])


def load():
    """(dataset, level, model) -> setting -> representation -> [scores],
    and the same shape for seconds, and setting -> parameters."""
    scores, secs, params = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'per_model_*.csv'))):
        tag = os.path.basename(path)[len('per_model_'):-len('.csv')]
        with open(path) as fh:
            for r in csv.DictReader(fh):
                # GUARD. Files written before 'dataset' and 'level' existed were
                # appended to by restarts that wrote the wider row, so some rows
                # are shifted and name a level or a dataset where the model goes.
                # Those rows are not repairable and are dropped here.
                if r.get('model') not in MODELS or r.get('status') != 'ok':
                    continue
                try:
                    r2 = float(r['r2'])
                except (TypeError, ValueError):
                    continue
                ds = r.get('dataset') or 'qm9'
                if ds not in DATASETS:
                    continue
                lvl = r.get('level')
                lvl = float(lvl) if lvl else (0.0 if tag.startswith('c_') else 0.5)
                key = (ds, 'clean' if lvl == 0.0 else 'noise', r['model'])
                sig = 'DEFAULT' if r['from_rep'] == 'DEFAULT' else r['detail']
                if sig != 'DEFAULT':
                    try:
                        params[sig] = json.loads(r['detail'])
                    except ValueError:
                        continue
                scores.setdefault(key, {}).setdefault(sig, {}) \
                      .setdefault(r['applied_to'], []).append(r2)
                try:
                    secs.setdefault(key, {}).setdefault(sig, {}) \
                        .setdefault(r['applied_to'], []).append(float(r['seconds']))
                except (TypeError, ValueError):
                    pass
    return scores, secs, params


def med(xs):
    return statistics.median(xs) if xs else None


def deltas(tbl, reps):
    """setting -> (mean change, {rep: change}) against the default."""
    base = tbl.get('DEFAULT')
    if not base:
        return {}, {}
    baseline = {rep: med(base.get(rep, [])) for rep in reps}
    out = {}
    for sig, per_rep in tbl.items():
        if sig == 'DEFAULT':
            continue
        d = {}
        for rep in reps:
            here, there = med(per_rep.get(rep, [])), baseline.get(rep)
            if here is not None and there is not None:
                d[rep] = here - there
        out[sig] = (statistics.mean(d.values()) if d else None, d)
    return out, baseline


def cell(v):
    return '—' if v is None else f'{v:+.3f}'


def main():
    scores, secs, params = load()
    lines = ['# One setting per model per dataset', '',
             'Generated by `scripts/write_chosen_settings.py`. Every number is a',
             'change from that representation\'s own default. `—` means the fit has',
             'not been recorded yet.', '']
    chosen = {}

    for ds in DATASETS:
        lines += [f'## {ds}', '']
        for model, nice in MODELS.items():
            ck, nk = (ds, 'clean', model), (ds, 'noise', model)
            clean = scores.get(ck, {})
            if not clean or 'DEFAULT' not in clean:
                lines += [f'### {nice}  (`{model}`)', '',
                          '_no clean results recorded yet_', '']
                continue

            reps = [r for r in REPS
                    if any(r in v for v in clean.values())] or REPS
            dl, baseline = deltas(clean, reps)
            nd, _ = deltas(scores.get(nk, {}), NOISE_REPS)

            dsec = secs.get(ck, {}).get('DEFAULT', {})

            rows = sorted(((s, m, d) for s, (m, d) in dl.items() if m is not None),
                          key=lambda t: -t[1])

            lines += [f'### {nice}  (`{model}`)', '',
                      'default R2: ' + '  '.join(
                          f'{r} {baseline[r]:.3f}' for r in reps
                          if baseline.get(r) is not None), '',
                      '| rank | ' + ' | '.join(f'D {r}' for r in reps) +
                      ' | MEAN D (n reps) | extreme | compute | noise | verdict |',
                      '|---' * (len(reps) + 6) + '|']

            picked, stopped = None, False
            for i, (sig, mean_d, d) in enumerate(rows, 1):
                ext = is_extreme(params.get(sig))
                # LIKE FOR LIKE. Compare the slowest fit of each, over the
                # representations BOTH were timed on. Taking each one's own
                # worst across whatever it happened to have finished failed
                # settings smaller than the default, purely because the
                # candidate had reached a slow representation and the default
                # had not.
                ssec = secs.get(ck, {}).get(sig, {})
                shared = [r for r in reps if ssec.get(r) and dsec.get(r)]
                if shared:
                    worst = max(med(ssec[r]) for r in shared)
                    base = max(med(dsec[r]) for r in shared)
                    slow = bool(base) and worst > COMPUTE_MULTIPLE * base
                    # The seconds are shown, not just the verdict. For these
                    # four models the default is sometimes fast BECAUSE IT
                    # FAILS TO TRAIN -- Bayesian alpha's default fits PDV in
                    # 36s and scores -0.826 -- so twice the default can be a
                    # very low bar, and the reader has to be able to see that.
                    comp = f'{"FAIL" if slow else "pass"} {worst:.0f}/{base:.0f}s'
                else:
                    comp = '—'
                nmean = nd.get(sig, (None, {}))[0]
                noi = '—' if nmean is None else ('FAIL' if nmean < 0
                                                 else f'pass {nmean:+.3f}')

                if stopped:
                    verdict = 'not assessed'
                elif mean_d <= 0:
                    verdict = '**STOP — no longer beats the default**'
                    stopped = True
                elif picked is not None:
                    verdict = 'below the chosen'
                elif ext:
                    verdict = f'rejected: extreme ({", ".join(ext)})'
                elif comp.startswith('FAIL'):
                    verdict = 'rejected: too slow'
                elif noi == 'FAIL':
                    verdict = 'rejected: worse under noise'
                elif comp == '—' or noi == '—':
                    verdict = 'held: filter data missing'
                else:
                    verdict, picked = '**CHOSEN**', sig

                # HOW MANY REPRESENTATIONS THE MEAN IS OVER. If a setting
                # failed to fit somewhere, its mean is over fewer than the rest
                # and is not comparable to them. Shown rather than hidden.
                lines.append(
                    f'| {i} | ' + ' | '.join(cell(d.get(r)) for r in reps) +
                    f' | **{mean_d:+.3f}** ({len(d)}) | '
                    f'{"FAIL" if ext else "pass"} | {comp} | {noi} | {verdict} |')

            lines.append('')
            if picked:
                chosen[f'{ds}/{model}'] = params[picked]
                lines += ['chosen setting: `' + json.dumps(params[picked],
                                                           sort_keys=True) + '`', '']
            else:
                chosen[f'{ds}/{model}'] = None
                lines += ['**no setting chosen — the default is kept**', '']

            for i, (sig, mean_d, _d) in enumerate(rows, 1):
                lines.append(f'- rank {i}: `' +
                             json.dumps(params.get(sig, {}), sort_keys=True) + '`')
            lines.append('')

    with open(os.path.join(OUT_DIR, 'CHOSEN_SETTINGS.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    with open(os.path.join(OUT_DIR, 'CHOSEN_SETTINGS.json'), 'w') as fh:
        json.dump(chosen, fh, indent=2, sort_keys=True)

    done = sum(1 for v in chosen.values() if v is not None)
    print(f'{len(chosen)} of 16 model-and-dataset pairs assessed, '
          f'{done} chose a tuned setting')
    return 0


if __name__ == '__main__':
    sys.exit(main())
