#!/usr/bin/env python
"""Variance shares by model family and representation group, with the split as a block.

A standalone test (the author, 2026-09-28), kept out of run_paper_analysis.py
until the grouping is settled. It answers three things on QM9:

1. How much of the variance in each outcome is model family, representation
   group, their pairing, and the split, with bootstrap intervals over splits.
2. How much is left between models INSIDE a family, and between representations
   inside a group. Those are their own nested terms, so a family's members
   differing is reported rather than pushed into the residual.
3. Whether the grouping holds up: Tukey HSD between the members of each family,
   one test per representation (never pooled across representations), and
   between the members of each representation group, one test per model.

THE DESIGN. One row is one (model, representation, split). QM9's replicate seed
depends only on the replicate number, so split r is the same data split and the
same noise draw for every model and representation. Split is therefore a block,
entered first.

Terms, entered in this order (Type I, sequential):
    split, family, model within family, representation group,
    representation within group, family x representation group,
    the rest of the model x representation pairing, residual.

The outcomes: clean R2 (no noise added), and AUC_norm under each noise condition
that every representation ran, each decomposed on its own.

Usage:
    python scripts/family_anova.py --qm9-dir results/qm9_arc --grouping A
    python scripts/family_anova.py --qm9-dir results/qm9_arc --grouping B
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_config as C  # noqa: E402
import figlib_load as L  # noqa: E402
import figlib_metrics as M  # noqa: E402

# ---------------------------------------------------------------------------
# The groupings under test. A model or representation not named is its own
# group. Only base models enter (figlib_config.VARIANT_MODELS stay out).
# ---------------------------------------------------------------------------

#: A -- the groups the author named as certain, 2026-09-28.
FAMILIES_A = {
    'rf': 'Forest', 'qrf': 'Forest',
    'dnn': 'NN-α', 'dnn_bnn_full': 'NN-α', 'dnn_vbll': 'NN-α',
    'mlp': 'NN-β', 'mlp_bnn_full': 'NN-β', 'mlp_vbll': 'NN-β',
}
REP_GROUPS_A = {}

#: B -- A plus the two groups the author asked to try.
FAMILIES_B = {**FAMILIES_A,
              'xgboost': 'Boosting', 'lgb': 'Boosting', 'ngboost': 'Boosting'}
REP_GROUPS_B = {'ecfp4': 'ECFP4 / Sort & Slice', 'sns': 'ECFP4 / Sort & Slice'}

#: C -- from the pair check (family_anova pairs.csv, 2026-09-28): a pair is
#: grouped only if it is one method by construction AND its two members differ
#: by less than their split-to-split SD in most (representation, outcome) cells.
#: VBLL left the NN families (it differs from NN and BNN of its own backbone);
#: NGBoost left Boosting; ECFP4 / Sort & Slice failed on clean R2 (0 of 13).
FAMILIES_C = {
    'rf': 'Forest', 'qrf': 'Forest',
    'xgboost': 'Boosting', 'lgb': 'Boosting',
    'dnn': 'NN-α', 'dnn_bnn_full': 'NN-α',
    'mlp': 'NN-β', 'mlp_bnn_full': 'NN-β',
    'dnn_vbll': 'VBLL', 'mlp_vbll': 'VBLL',
}
#: D -- C with the two plain/Bayesian network pairs as one family.
FAMILIES_D = {**FAMILIES_C, 'dnn': 'NN', 'dnn_bnn_full': 'NN',
              'mlp': 'NN', 'mlp_bnn_full': 'NN'}

#: E -- the networks grouped by METHOD instead of by architecture.
FAMILIES_E = {'rf': 'Forest', 'qrf': 'Forest',
              'xgboost': 'Boosting', 'lgb': 'Boosting',
              'dnn': 'NN', 'mlp': 'NN',
              'dnn_bnn_full': 'BNN', 'mlp_bnn_full': 'BNN',
              'dnn_vbll': 'VBLL', 'mlp_vbll': 'VBLL'}

GROUPINGS = {'E': (FAMILIES_E, {}), 'A': (FAMILIES_A, REP_GROUPS_A), 'B': (FAMILIES_B, REP_GROUPS_B),
             'C': (FAMILIES_C, {}), 'D': (FAMILIES_D, {})}

TERMS = ['split', 'family', 'model_in_family', 'rep_group', 'rep_in_group',
         'family_x_rep_group', 'rest_of_pairing', 'residual']


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def per_split(qm9_dir, cache_dir=None):
    """One row per (model, rep, condition, split): AUC_norm and clean R2."""
    import run_paper_analysis as R
    qm9 = L.load_qm9(qm9_dir, cache_dir=cache_dir)
    qm9, _ = R.apply_declared_filters(qm9, 'QM9')
    per, _ = M.robustness(qm9)
    per = C.cross_model(per, 'family ANOVA')
    return per.rename(columns={'replicate': 'split'})


def outcomes(per):
    """{name: frame with columns model, rep, split, y}."""
    out = {}
    clean = per[per['condition'] == 'gaussian']
    out['clean R2'] = clean[['model', 'rep', 'split']].assign(
        y=clean['baseline_r2'].to_numpy())
    reps_everywhere = per['rep'].nunique()
    for condition, group in per.groupby('condition'):
        if group['rep'].nunique() < reps_everywhere:
            continue            # ran on too few representations to decompose
        out[f'AUC_norm, {C.condition_label(condition)}'] = \
            group[['model', 'rep', 'split']].assign(
                y=group['auc_norm'].to_numpy())
    return out


# ---------------------------------------------------------------------------
# The decomposition
# ---------------------------------------------------------------------------

def _dummies(labels):
    return pd.get_dummies(pd.Series(labels), drop_first=True).to_numpy(float)


def shares(frame, families, rep_groups):
    """Type I eta-squared, in percent, for every term in TERMS."""
    d = frame.dropna(subset=['y'])
    y = d['y'].to_numpy(float)
    total = float(((y - y.mean()) ** 2).sum())
    if total == 0:
        return None
    model = d['model'].astype(str).to_numpy().astype(str)
    rep = d['rep'].astype(str).to_numpy().astype(str)
    family = np.array([families.get(m, m) for m in model], dtype=str)
    group = np.array([rep_groups.get(r, r) for r in rep], dtype=str)
    steps = [
        ('split', d['split'].astype(str).to_numpy().astype(str)),
        ('family', family),
        ('model_in_family', model),
        ('rep_group', group),
        ('rep_in_group', rep),
        ('family_x_rep_group', np.char.add(np.char.add(family, '|'), group)),
        ('rest_of_pairing', np.char.add(np.char.add(model, '|'), rep)),
    ]
    X = np.ones((len(y), 1))
    previous = total
    result = {}
    for name, labels in steps:
        X = np.hstack([X, _dummies(labels)])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = float(((y - X @ beta) ** 2).sum())
        result[name] = (previous - rss) / total * 100
        previous = rss
    result['residual'] = previous / total * 100
    return result


def bootstrap(frame, families, rep_groups, n_boot=1000, seed=0):
    """Percentile 95% intervals, resampling whole splits with replacement."""
    rng = np.random.default_rng(seed)
    splits = np.array(sorted(frame['split'].unique()))
    by_split = {s: frame[frame['split'] == s] for s in splits}
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(splits, size=len(splits), replace=True)
        # A split drawn twice is two blocks, not one, so it gets a new label.
        sample = pd.concat([by_split[s].assign(split=f'{s}#{i}')
                            for i, s in enumerate(picked)], ignore_index=True)
        got = shares(sample, families, rep_groups)
        if got is not None:
            draws.append(got)
    draws = pd.DataFrame(draws)
    return draws.quantile(0.025), draws.quantile(0.975)


# ---------------------------------------------------------------------------
# Tukey HSD: does the grouping hold up?
# ---------------------------------------------------------------------------

def _tukey(values, labels):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    res = pairwise_tukeyhsd(values, labels)
    table = pd.DataFrame(res.summary().data[1:],
                         columns=res.summary().data[0])
    return table


def tukey_within(frame, members, across, within):
    """Tukey between the members of one group, once per level of `within`.

    The split block is taken out first: each value minus the mean of its split
    at that level of `within`, over the members being compared.
    """
    rows = []
    for level, sub in frame[frame[across].isin(members)].groupby(within):
        if sub[across].nunique() < 2:
            continue
        sub = sub.dropna(subset=['y']).copy()
        sub['y_blocked'] = sub['y'] - sub.groupby('split')['y'].transform('mean')
        table = _tukey(sub['y_blocked'].to_numpy(float),
                       sub[across].astype(str).to_numpy())
        for r in table.itertuples(index=False):
            rows.append({within: level, 'a': r[0], 'b': r[1],
                         'difference': float(r[2]), 'p_adj': float(r[3]),
                         'differs': bool(r[6])})
    return pd.DataFrame(rows)


def tukey_checks(frame, families, rep_groups):
    out = []
    for name, members in _groups(families).items():
        t = tukey_within(frame, members, 'model', 'rep')
        if len(t):
            out.append(t.assign(group=name, kind='models in a family'))
    for name, members in _groups(rep_groups).items():
        t = tukey_within(frame, members, 'rep', 'model')
        if len(t):
            out.append(t.assign(group=name, kind='representations in a group'))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _groups(mapping):
    groups = {}
    for member, group in mapping.items():
        groups.setdefault(group, []).append(member)
    return {g: m for g, m in groups.items() if len(m) > 1}


# ---------------------------------------------------------------------------

def run(per, grouping, output_dir, n_boot=1000):
    families, rep_groups = GROUPINGS[grouping]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    share_rows, tukey_frames = [], []
    for name, frame in outcomes(per).items():
        point = shares(frame, families, rep_groups)
        if point is None:
            continue
        lo, hi = bootstrap(frame, families, rep_groups, n_boot=n_boot)
        for term in TERMS:
            share_rows.append({'outcome': name, 'term': term,
                               'share': point[term], 'lo': lo[term],
                               'hi': hi[term]})
        t = tukey_checks(frame, families, rep_groups)
        if len(t):
            tukey_frames.append(t.assign(outcome=name))
    table = pd.DataFrame(share_rows)
    tukey = (pd.concat(tukey_frames, ignore_index=True) if tukey_frames
             else pd.DataFrame())
    table.to_csv(output_dir / f'family_anova_shares_{grouping}.csv', index=False)
    tukey.to_csv(output_dir / f'family_anova_tukey_{grouping}.csv', index=False)
    return table, tukey


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--qm9-dir', required=True)
    p.add_argument('--cache-dir', default=None)
    p.add_argument('--grouping', choices=sorted(GROUPINGS), default='A')
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--output-dir', default=str(C.ROOT / 'results' / 'family_anova'))
    args = p.parse_args(argv)
    pickled = Path(args.output_dir) / 'per.pkl'
    if pickled.exists():
        per = pd.read_pickle(pickled)
    else:
        per = per_split(args.qm9_dir, args.cache_dir)
    table, tukey = run(per, args.grouping, args.output_dir, args.bootstrap)
    pd.set_option('display.width', 200)
    wide = table.assign(cell=table.apply(
        lambda r: f"{r['share']:.1f} [{r['lo']:.1f}, {r['hi']:.1f}]", axis=1))
    print(wide.pivot(index='term', columns='outcome', values='cell')
          .reindex(TERMS).to_string())
    if len(tukey):
        print('\nTukey HSD, pairs that differ at the adjusted 0.05 level, '
              'out of the representations (or models) tested:')
        print(tukey.groupby(['outcome', 'kind', 'group', 'a', 'b'])['differs']
              .agg(lambda s: f'{int(s.sum())} of {len(s)}').to_string())


if __name__ == '__main__':
    main()
