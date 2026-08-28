#!/usr/bin/env python3
"""Do the validation molecules replace the out-of-fold refitting?

WHAT THIS DECIDES
-----------------
One uncertainty task is (levels) x (folds) x (1 fit + `--oof-folds` cross-fits).
With --oof-folds 5, five of every six model fits in the uncertainty runs are the
cross-fit. If the validation molecules answer the same question, that cost
disappears.

A scored molecule has to satisfy two things: no model fitted it, and its
injected noise is known exactly. Both now hold for validation molecules -- since
2026-08-27 nothing stacks validation into training, and validation carries its
own independently drawn noise, recorded per molecule. So the question is no
longer whether the route is legitimate; it is whether it gives the same answer.

THE DECISION RULE, FIXED BEFORE ANY NUMBER CAME BACK
----------------------------------------------------
Written into RERUN_PLAN.md 13 chat O before this script existed, and this script
was committed before the run it reads finished. It must not be revised now.

  Rank the models by `rho_delta` under each route.

  DROP THE CROSS-FIT EVERYWHERE   if the two rankings agree at Spearman >= 0.8
                                  AND every model's validation value falls
                                  inside the 300-resample interval of its
                                  out-of-fold value.
  KEEP IT FOR THE NEURAL FAMILIES if that holds for the trees and the Gaussian
  ONLY                            process and fails for the networks.
  KEEP IT EVERYWHERE              otherwise, and write down what disagreed.

TWO READINGS THIS SCRIPT FIXES, ALSO BEFORE THE NUMBERS
--------------------------------------------------------
1. WHICH LEVEL. `rho_delta` is NaN at level 0 by construction -- every molecule
   gets exactly zero noise there, so the target is constant. The rule is applied
   at the reporting level, 1.5.

2. WHICH REPRESENTATION. The rule says "the two rankings", singular, and this
   project never pools across representations. So it is applied WITHIN each
   representation and the verdict is the conjunction: the cross-fit is dropped
   only if every representation passes. Each representation's own numbers are
   printed, so a split verdict is visible rather than averaged away.

THE UNCALIBRATED COLUMN, AND WHY
---------------------------------
Temperature calibration is one multiplier over every molecule
(`models/models.py`, the calibration branch in each trainer), so it cannot
change any ranking, and every statistic here is a rank correlation. It is also
fitted ON the validation molecules for several families, which is exactly the
kind of thing that would make this comparison unfair if it mattered. It does
not, and this reads `y_pred_std_uncalibrated` -- the loader's default -- so the
question does not arise.

THE ONE LEAK THAT IS REAL
-------------------------
The four neural families choose when to stop training by watching the validation
molecules, so their error on them is optimistic. That is not a flaw in the test:
it is the thing the test is looking for, and it is why the rule has a middle
verdict that keeps the cross-fit for those four alone.

EVERY STATISTIC COMES FROM A MODULE THAT ALREADY EXISTED
---------------------------------------------------------
`q4_error_ratio` and `q6_error_ranking` from `scripts/uncertainty_stats.py`, and
`bootstrap_cis` from `scripts/uncertainty_screen_tables.py`, which is where the
300-resample interval lives. Nothing is invented here.

USAGE
-----
    python scripts/compare_validation_vs_oof.py \
        --run-dir /Volumes/seagate/chatO_screen \
        --out-dir results/validation_vs_oof
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uncertainty_stats as us  # noqa: E402
from uncertainty_screen_tables import (  # noqa: E402
    MODEL_LABELS, REP_LABELS, REPORT_LEVEL, bootstrap_cis,
)

#: The rule's threshold, and the resample count it names.
RANK_AGREEMENT_THRESHOLD = 0.8
N_BOOT = 300

#: The four families that early-stop on the validation molecules. The rule's
#: middle verdict is about exactly these.
NEURAL_FAMILIES = {
    'dnn_bnn_full', 'dnn_bnn_full_variational',
    'mlp_bnn_full', 'mlp_bnn_full_variational',
}


def spearman(a, b):
    """Rank correlation, with the module's own guards."""
    return us._spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def load(run_dir):
    paths = sorted(Path(run_dir).glob('*_uncertainty_values.csv'))
    if not paths:
        raise SystemExit(f"no *_uncertainty_values.csv under {run_dir}")
    print(f"reading {len(paths)} uncertainty files from {run_dir}")
    # uncertainty_column defaults to 'uncalibrated'. Named here so the choice is
    # in the record rather than in a default.
    df = us.load_uncertainty(paths, uncertainty_column='uncalibrated')

    unscored = (df['split'].eq('train_oof')
                & (~np.isfinite(df['y_pred']) | ~np.isfinite(df['uncertainty'])))
    if unscored.any():
        print(f"  dropping {int(unscored.sum()):,} training rows with no "
              f"out-of-fold score")
        df = df[~unscored].copy()
    counts = df.groupby(['split', 'sigma']).size()
    print(f"  {len(df):,} rows")
    print(counts.to_string())
    for needed in ('train_oof', 'validation'):
        if needed not in set(df['split']):
            raise SystemExit(
                f"the run wrote no split == {needed!r} rows, so the two routes "
                f"cannot be compared. Re-run with --oof-folds and "
                f"--score-validation together.")
    return df


def per_route_table(df, level):
    """One row per (model, rep, split) at the reporting level, from the
    statistics that already exist."""
    frames = []
    for split in ('train_oof', 'validation'):
        ratio = us.q4_error_ratio(df, split=split)
        rank = us.q6_error_ranking(df, split=split)
        keys = ['dataset', 'model', 'rep', 'condition', 'sigma']
        t = (ratio[keys + ['n', 'rho_error', 'rho_ratio', 'rho_delta',
                           'auc_error', 'auc_ratio', 'auc_delta']]
             .merge(rank[keys + ['rho_unc_vs_clean_error']], on=keys, how='outer'))
        t['split'] = split
        frames.append(t)
    out = pd.concat(frames, ignore_index=True)
    return out[np.isclose(out['sigma'].astype(float), level)].copy()


def verdict(df, level=REPORT_LEVEL):
    """Apply the rule, per representation, and return the per-representation
    findings plus the overall verdict."""
    tab = per_route_table(df, level)
    boot = bootstrap_cis(df, split='train_oof', n_boot=N_BOOT)
    boot = boot[np.isclose(boot['sigma'].astype(float), level)]

    oof = tab[tab['split'] == 'train_oof']
    val = tab[tab['split'] == 'validation']
    keys = ['model', 'rep']
    merged = (oof[keys + ['n', 'rho_delta', 'rho_error', 'rho_ratio',
                          'rho_unc_vs_clean_error']]
              .rename(columns={c: c + '_oof' for c in
                               ('n', 'rho_delta', 'rho_error', 'rho_ratio',
                                'rho_unc_vs_clean_error')})
              .merge(val[keys + ['n', 'rho_delta', 'rho_error', 'rho_ratio',
                                 'rho_unc_vs_clean_error']]
                     .rename(columns={c: c + '_val' for c in
                                      ('n', 'rho_delta', 'rho_error',
                                       'rho_ratio', 'rho_unc_vs_clean_error')}),
                     on=keys, how='outer')
              .merge(boot[keys + ['rho_delta_lo', 'rho_delta_hi']],
                     on=keys, how='left'))
    merged['inside_interval'] = (
        (merged['rho_delta_val'] >= merged['rho_delta_lo'])
        & (merged['rho_delta_val'] <= merged['rho_delta_hi']))
    merged['is_neural'] = merged['model'].isin(NEURAL_FAMILIES)
    merged['model_label'] = merged['model'].map(lambda m: MODEL_LABELS.get(m, m))
    merged['rep_label'] = merged['rep'].map(lambda r: REP_LABELS.get(r, r))

    findings = []
    for repname, cell in merged.groupby('rep', sort=True):
        cell = cell.sort_values('model')
        rho = spearman(cell['rho_delta_oof'], cell['rho_delta_val'])
        trees = cell[~cell['is_neural']]
        nets = cell[cell['is_neural']]
        findings.append({
            'rep': repname,
            'rep_label': REP_LABELS.get(repname, repname),
            'n_models': int(len(cell)),
            'rank_agreement': rho,
            'ranks_agree': bool(np.isfinite(rho)
                                and rho >= RANK_AGREEMENT_THRESHOLD),
            'n_inside': int(cell['inside_interval'].sum()),
            'all_inside': bool(cell['inside_interval'].all()),
            'trees_gp_all_inside': bool(trees['inside_interval'].all())
                                   if len(trees) else False,
            'networks_all_inside': bool(nets['inside_interval'].all())
                                   if len(nets) else False,
            'outside': ', '.join(sorted(
                cell.loc[~cell['inside_interval'], 'model_label'])) or '-',
        })
    fin = pd.DataFrame(findings)

    everywhere = bool(fin['ranks_agree'].all() and fin['all_inside'].all())
    trees_ok = bool(fin['ranks_agree'].all() and fin['trees_gp_all_inside'].all())
    nets_fail = not bool(fin['networks_all_inside'].all())

    if everywhere:
        call = 'DROP THE CROSS-FIT EVERYWHERE'
    elif trees_ok and nets_fail:
        call = 'KEEP THE CROSS-FIT FOR THE FOUR NEURAL FAMILIES ONLY'
    else:
        call = 'KEEP THE CROSS-FIT EVERYWHERE'
    return merged, fin, call


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--out-dir', default='results/validation_vs_oof')
    ap.add_argument('--level', type=float, default=REPORT_LEVEL)
    args = ap.parse_args()

    df = load(args.run_dir)
    merged, fin, call = verdict(df, level=args.level)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_dir / f'per_model_level{args.level}.csv', index=False)
    fin.to_csv(out_dir / f'per_representation_level{args.level}.csv', index=False)

    cols = ['model_label', 'n_oof', 'n_val', 'rho_delta_oof', 'rho_delta_lo',
            'rho_delta_hi', 'rho_delta_val', 'inside_interval',
            'rho_unc_vs_clean_error_oof', 'rho_unc_vs_clean_error_val']
    for repname, cell in merged.groupby('rep', sort=True):
        print(f"\n=== {REP_LABELS.get(repname, repname)} "
              f"— noise level {args.level}, uncalibrated uncertainty ===")
        print(cell.sort_values('model')[cols].to_string(index=False))

    print("\n=== the rule, per representation ===")
    print(fin[['rep_label', 'n_models', 'rank_agreement', 'ranks_agree',
               'n_inside', 'all_inside', 'trees_gp_all_inside',
               'networks_all_inside', 'outside']].to_string(index=False))

    print(f"\nVERDICT: {call}")
    print(f"wrote {out_dir}/per_model_level{args.level}.csv and "
          f"per_representation_level{args.level}.csv")
    return 0


if __name__ == '__main__':
    sys.exit(main())
