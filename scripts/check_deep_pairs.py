#!/usr/bin/env python
"""Is the deep-run selection the one the screen actually supports?

    python scripts/check_deep_pairs.py --qm9-dir results

Reads the landed screen with the CURRENT metric -- AUC_norm integrated per
replicate, not averaged first -- and prints the ranking the selection rule is
supposed to be read off, with the models currently in deep_run_pairs.json
marked. It changes nothing.

THE RULE, quoted from RERUN_PLAN.md 13.17 B:

    "Take the widest spread of behaviour the screen shows -- the most and least
     noise-tolerant model, plus one from each remaining family -- and the
     representations that span fingerprint, descriptor and learned embedding.
     NGBoost is locked on by a check that refuses to build without it."

WHAT IT COSTS TO CHANGE, and this is why it is worth checking now rather than
later. The pair file is read WHEN A TASK STARTS, so it can be edited up to the
moment one begins. Narrowing is free: a task no longer listed skips and exits 0.
Widening is NOT: a task that has already skipped stays skipped, so a model added
later needs its array indices resubmitted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_paper_analysis import _preflight  # noqa: E402

_preflight()

import figlib_config as C  # noqa: E402
import figlib_load as L  # noqa: E402
import figlib_metrics as M  # noqa: E402

#: Which family each model belongs to, so "one from each remaining family" can
#: be checked rather than eyeballed.
FAMILY = {
    'rf': 'trees', 'qrf': 'trees', 'xgboost': 'boosting', 'lgb': 'boosting',
    'ngboost': 'boosting', 'svm': 'kernel',
    'gauche': 'gaussian process', 'gauche_rbf': 'gaussian process',
    'het_gp_rbf': 'gaussian process',
    'dnn': 'neural α', 'dnn_bnn_full': 'neural α', 'dnn_vbll': 'neural α',
    'dnn_bnn_full_mve': 'neural α', 'dnn_vbll_hetero': 'neural α',
    'mlp': 'neural β', 'mlp_bnn_full': 'neural β', 'mlp_vbll': 'neural β',
    'mlp_bnn_full_mve': 'neural β', 'mlp_vbll_hetero': 'neural β',
}


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--qm9-dir', default=str(C.ROOT / 'results'))
    p.add_argument('--condition', default='gaussian',
                   help='the condition the ranking is read at')
    p.add_argument('--pairs-file', default=str(C.ROOT / 'deep_run_pairs.json'))
    args = p.parse_args(argv)

    selected = json.loads(Path(args.pairs_file).read_text())
    chosen = set(selected.get('models', []))
    chosen_reps = list(selected.get('representations', []))

    frame = L.load_qm9(args.qm9_dir)
    if frame is None:
        print('nothing landed yet; nothing to read the rule off')
        return 1
    for filt in (L.catastrophic_filter(), L.collapsed_gp_filter()):
        frame, _ = filt.apply(frame)
    per, _ = M.robustness(frame)
    summary = M.summarise_robustness(per)
    summary = summary[summary['condition'] == args.condition]
    if not len(summary):
        print(f'no rows at condition {args.condition!r}')
        return 1

    # WERE THESE FITTED UNDER THE SAME SETTINGS? The tuned hyperparameters
    # landed on 2026-09-01 and --use-best-params re-reads that file inside every
    # training run, so a task that started before it fitted at the shared
    # defaults and one that started after fitted at the tuned setting. A cell
    # containing both is not one experiment, and ranking models on it compares
    # some tuned fits against some untuned ones.
    if 'params_source' in frame.columns:
        counts = frame['params_source'].value_counts(dropna=False)
        print('\nHYPERPARAMETER SOURCE across the rows being ranked')
        for source, n in counts.items():
            print(f'  {str(source):24s} {n:>7,} rows  '
                  f'({n / len(frame):.0%})')
        cell = [c for c in ('model', 'rep', 'condition') if c in frame.columns]
        mixed = frame.groupby(cell, dropna=False)['params_source'].nunique()
        n_mixed = int((mixed > 1).sum())
        if n_mixed:
            print(f'  ⚠ {n_mixed} of {len(mixed)} cells contain BOTH. Those '
                  f'cells mix tuned and untuned fits, so their replicates are '
                  f'not repeats of one experiment and the ranking below '
                  f'compares some tuned models against some untuned ones.')
            worst = mixed[mixed > 1].head(6)
            for keys, _ in worst.items():
                print('      ' + ' / '.join(str(k) for k in
                                            (keys if isinstance(keys, tuple)
                                             else (keys,))))
            if n_mixed > 6:
                print(f'      ... and {n_mixed - 6} more')
        else:
            print('  every cell is internally consistent')
        # WHICH MODELS are tuned. Cells being internally consistent is not the
        # end of it: if some models were tuned and others were not, the ranking
        # below is partly a ranking of who got tuned. Only a few models in this
        # study can be handed a setting that reaches them alone (RERUN_PLAN.md
        # 5.7a), so an unequal split is expected -- and it lands on the model at
        # the top of the ranking.
        by_model = (frame.groupby('model')['params_source']
                    .agg(lambda s: '/'.join(sorted(set(s.dropna().astype(str))))))
        tuned = sorted(m for m, v in by_model.items() if 'tuned' in v)
        untuned = sorted(m for m, v in by_model.items() if 'tuned' not in v)
        if tuned and untuned:
            print(f'  ⚠ {len(tuned)} model(s) were fitted at TUNED settings and '
                  f'{len(untuned)} at the shared defaults:')
            print(f'      tuned  : {[C.model_label(m) for m in tuned]}')
            print(f'      default: {[C.model_label(m) for m in untuned][:8]}'
                  + (' ...' if len(untuned) > 8 else ''))
            print(f'    The ranking below therefore compares tuned models '
                  f'against untuned ones. A model near the top that is in the '
                  f'tuned list may be there because it was tuned.')

    print(f'\nTHE SCREEN, at {C.condition_label(args.condition)}, '
          f'AUC_norm integrated per replicate then medianed.')
    print('IN = currently in deep_run_pairs.json\n')

    for rep in chosen_reps:
        sub = summary[summary['rep'] == rep].sort_values('auc_norm',
                                                         ascending=False)
        if not len(sub):
            print(f'  {C.rep_label(rep)}: nothing landed')
            continue
        print(f'  {C.rep_label(rep)} — {len(sub)} models')
        for position, row in enumerate(sub.itertuples()):
            edge = ''
            if position == 0:
                edge = '  <- MOST noise-tolerant'
            elif position == len(sub) - 1:
                edge = '  <- LEAST noise-tolerant'
            mark = 'IN ' if row.model in chosen else '   '
            print(f'    {mark} {C.model_label(row.model):22s} '
                  f'{row.auc_norm:.4f}  (clean {row.baseline_r2:.3f}, '
                  f'spread {row.auc_norm_spread:.3f}){edge}')
        print()

    # Does the current selection satisfy the rule?
    print('AGAINST THE RULE')
    for rep in chosen_reps:
        sub = summary[summary['rep'] == rep].sort_values('auc_norm',
                                                         ascending=False)
        if not len(sub):
            continue
        top, bottom = sub.iloc[0]['model'], sub.iloc[-1]['model']
        print(f'  {C.rep_label(rep):12s} most: {C.model_label(top):22s}'
              f'{"in" if top in chosen else "NOT IN":>8s}   '
              f'least: {C.model_label(bottom):22s}'
              f'{"in" if bottom in chosen else "NOT IN":>8s}')

    families = {FAMILY.get(m, 'unknown') for m in chosen}
    all_families = {FAMILY.get(m, 'unknown')
                    for m in summary['model'].unique()}
    print(f'\n  families covered : {sorted(families)}')
    missing = sorted(all_families - families)
    if missing:
        print(f'  families MISSING : {missing}')
    if 'ngboost' not in chosen:
        print('  ⚠ NGBoost is not in the file and the generator refuses to '
              'build a deep run without it')

    print(f'\n  currently selected: {sorted(chosen)}')
    print(f'  that is {len(chosen)} models x {len(chosen_reps)} '
          f'representations = {len(chosen) * len(chosen_reps)} pairs')
    print('\nTO CHANGE IT: edit deep_run_pairs.json now. Removing a model is '
          'free -- its tasks skip and exit 0. ADDING one means resubmitting '
          'that model\'s array indices, because a task that already skipped '
          'stays skipped.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
