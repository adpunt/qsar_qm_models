#!/usr/bin/env python
"""Read the deep-run and censoring pairs off the screen, from whatever has landed.

WHY THIS EXISTS
---------------
RERUN_PLAN.md 13.17 B defers three choices with a rule fixed in advance, to be
read off the screen when it lands. On 2026-09-04 the screen was landing in
pieces -- nine of nineteen QM9 arrays complete, the rest queued behind a brutal
wait -- and the author's call was to choose from what exists and revise as more
arrives. That is a reading, not a new decision, so this applies the stated rule
rather than inventing one.

WHAT IT DOES NOT DO
-------------------
It computes no metric of its own. `load_anova_data` and `calculate_robustness`
are imported from `generate_paper_figures_v2.py`, which is the one place
auc_norm is defined -- two implementations of one statistic is the drift this
project keeps paying for (RERUN_PLAN.md 0.6, failure mode 10).

It never pools across representations. Every table is one representation, named
in its title, and a model is ranked within a representation and a condition --
never on an average over them.

THE RULE, QUOTED FROM 13.17 B
-----------------------------
  deep run: "Take the widest spread of behaviour the screen shows -- the most and
  least noise-tolerant model, plus one from each remaining family -- and the
  representations that span fingerprint, descriptor and learned embedding.
  NGBoost is locked on by a check that refuses to build without it."

  censoring: "Chosen from the screen on interest and clean performance, at least
  two of them models that report a per-molecule uncertainty. One selection, used
  on QM9 and all three validation datasets."

    python scripts/select_deep_run_pairs.py --results-dir results
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

import generate_paper_figures_v2 as fig                              # noqa: E402

# The three representations the rule asks for, one from each kind. They are the
# same three the uncertainty pass already runs on (uncertainty_pairs.json), so
# the deep run and the uncertainty work land on one set of features rather than
# two. Override with --reps.
SPANNING_REPS = ['ecfp4', 'pdv', 'chemberta']
REP_KIND = {
    'ecfp4': 'fingerprint', 'sns': 'fingerprint', 'avalon': 'fingerprint',
    'pdv': 'descriptor',
    'mhg_gnn_pretrained': 'learned', 'chemberta': 'learned',
}

# Families, keyed on the CANONICAL names in model_names.json. A model that is in
# no family is a model the rule cannot place, so the coverage assertion below
# fails rather than letting it drop silently out of the selection.
FAMILIES = {
    'tree':                 ['rf', 'qrf', 'xgboost', 'lgb', 'ngboost'],
    'kernel':               ['svm'],
    'gaussian_process':     ['gauche', 'gauche_rbf', 'het_gp_rbf', 'het_gp_tanimoto'],
    'neural_plain':         ['dnn', 'mlp'],
    'neural_bayesian':      ['dnn_bnn_full', 'mlp_bnn_full', 'dnn_bnn_last',
                             'mlp_bnn_last', 'dnn_bnn_variational',
                             'mlp_bnn_variational'],
    'neural_variational':   ['dnn_vbll', 'mlp_vbll', 'dnn_vbll_hetero',
                             'mlp_vbll_hetero'],
    'neural_variance_head': ['dnn_bnn_full_mve', 'mlp_bnn_full_mve'],
}
FAMILY_OF = {m: f for f, members in FAMILIES.items() for m in members}

# 13.17 B: "NGBoost is locked on by a check that refuses to build without it."
LOCKED_ON = ['ngboost']


def canonical_names():
    """The settled roster, from the one file that holds the spellings."""
    spec = json.loads((ROOT / 'model_names.json').read_text())
    return list(spec.get('canonical', []))


def uncertainty_models():
    """The models that emit an uncertainty at all, from the settled pairs file."""
    spec = json.loads((ROOT / 'uncertainty_pairs.json').read_text())
    return [m['canonical'] for m in spec.get('models', [])]


def check_family_map():
    missing = [m for m in canonical_names() if m not in FAMILY_OF]
    if missing:
        sys.exit(f"FAIL: {', '.join(missing)} are in model_names.json's canonical "
                 f"roster and in no family here. A model the rule cannot place "
                 f"would be dropped from the selection without a word. Add it to "
                 f"FAMILIES.")


def coverage(df, conditions):
    """What the screen has actually produced, and what is still missing."""
    have = df.groupby(['model', 'rep'])['strategy'].nunique()
    models = sorted(df['model'].unique())
    reps = sorted(df['rep'].unique())
    print(f"\n  models present: {len(models)} -- {', '.join(models)}")
    absent = [m for m in canonical_names() if m not in models]
    if absent:
        print(f"  models ABSENT:  {len(absent)} -- {', '.join(absent)}")
    print(f"  representations present: {', '.join(reps)}")
    partial = have[have < len(conditions)]
    if len(partial):
        print(f"  {len(partial)} model-and-representation cells have fewer than "
              f"{len(conditions)} conditions:")
        for (m, r), n in partial.items():
            print(f"      {m:28s} {r:20s} {n} of {len(conditions)}")
    return models, reps, absent


def per_rep_tables(robust, reps, conditions):
    """One table per representation. Never one table across them."""
    for rep in reps:
        sub = robust[robust['rep'] == rep]
        if sub.empty:
            continue
        wide = sub.pivot_table(index='model', columns='strategy', values='auc_norm')
        base = sub.groupby('model')['baseline_r2'].first()
        wide = wide.reindex(fig.sort_models_by_family(list(wide.index)))
        print(f"\n  auc_norm on {rep} -- higher is more noise-tolerant. "
              f"Clean R2 beside it, because a ratio without its denominator is "
              f"failure mode 4.")
        header = '      ' + f"{'model':28s}" + ''.join(f"{c:>18s}" for c in wide.columns) + f"{'clean R2':>10s}"
        print(header)
        for m, row in wide.iterrows():
            cells = ''.join(f"{row[c]:>18.4f}" if pd.notna(row[c]) else f"{'--':>18s}"
                            for c in wide.columns)
            print(f"      {m:28s}{cells}{base.get(m, float('nan')):>10.3f}")


def extremes(robust, reps, verbose=True):
    """Most and least noise-tolerant, within each representation and condition.

    Ranked inside a cell, never on an average across cells. The union is what
    the rule's "widest spread of behaviour" means when the screen shows more
    than one representation.
    """
    most, least = {}, {}
    if verbose:
        print("\n  the widest spread the screen shows, per representation and "
              "condition:")
    for rep in reps:
        for cond, g in robust[robust['rep'] == rep].groupby('strategy'):
            if len(g) < 2:
                continue
            hi = g.loc[g['auc_norm'].idxmax()]
            lo = g.loc[g['auc_norm'].idxmin()]
            most[hi['model']] = most.get(hi['model'], 0) + 1
            least[lo['model']] = least.get(lo['model'], 0) + 1
            if verbose:
                print(f"      {rep:20s} {cond:18s} most {hi['model']:24s} "
                      f"{hi['auc_norm']:.4f}   least {lo['model']:24s} "
                      f"{lo['auc_norm']:.4f}")
    return most, least


def select(robust, reps, n_models, verbose=True):
    """Apply the rule. Returns the chosen models and why each one is there."""
    most, least = extremes(robust, reps, verbose=verbose)
    chosen, why = [], {}

    for m in LOCKED_ON:
        if m in set(robust['model']):
            chosen.append(m)
            why[m] = 'locked on by 13.17 B; the generator refuses to build without it'
        else:
            why[m] = ('LOCKED ON but NOT IN THE SCREEN YET -- it is in the deep run '
                      'regardless, and its rank is unknown until its array lands')
            chosen.append(m)

    for pool, label in ((most, 'most noise-tolerant'), (least, 'least noise-tolerant')):
        for m, _ in sorted(pool.items(), key=lambda kv: -kv[1]):
            if m not in chosen:
                chosen.append(m)
                why[m] = f'{label} in {pool[m]} of the screen\'s cells'
                break

    covered = {FAMILY_OF.get(m) for m in chosen}
    for m in fig.sort_models_by_family(list(robust['model'].unique())):
        if len(chosen) >= n_models:
            break
        fam = FAMILY_OF.get(m)
        if fam not in covered:
            chosen.append(m)
            why[m] = f'one from the {fam} family, which nothing above covers'
            covered.add(fam)
    return chosen, why


def floor_disagrees(df, conditions, reps, n_models):
    """Guard 8: the accuracy floor is an undeclared filter, so run it both ways.

    A configuration whose clean R2 is under the threshold is dropped from the
    robustness table entirely, and that bias runs toward unstable models. If the
    selection changes when the floor is lifted, the selection is resting on the
    filter and not on the data.
    """
    with_floor, _ = fig.calculate_robustness(df)
    without, _ = fig.calculate_robustness(df, baseline_threshold=-np.inf)
    if with_floor.empty or without.empty:
        return None
    a, _ = select(with_floor, reps, n_models, verbose=False)
    b, _ = select(without, reps, n_models, verbose=False)
    return (sorted(a), sorted(b)) if sorted(a) != sorted(b) else None


def generator_labels(chosen):
    """Translate results-side names into the labels --models will accept.

    The results carry CANONICAL names (model_names.json); the generator is keyed
    on its own labels, and for the variational networks the two differ --
    `dnn_vbll` in a results row is `dnn_bnn_full_variational` to the generator.
    Printing a command with the wrong one exits on the generator's unknown-label
    guard, which is a wasted round trip on a queue this slow. Refuses rather than
    guesses.
    """
    sys.path.insert(0, str(ROOT / 'slurm_scripts_qm9_rerun'))
    import generate_scripts as gen
    qm9_map = json.loads((ROOT / 'model_names.json').read_text()).get('qm9', {})
    out, unresolved = [], []
    for m in chosen:
        if m in gen.MODELS:
            out.append(m)
            continue
        cands = [k for k, v in qm9_map.items() if v == m and k in gen.MODELS]
        if cands:
            out.append(sorted(cands, key=len)[-1])
        else:
            unresolved.append(m)
            out.append(m)
    if unresolved:
        print(f"\n  WARNING {', '.join(unresolved)} is not a label the QM9 generator "
              f"knows, and no entry in model_names.json maps to one. The command "
              f"below will not run as printed.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results-dir', default=str(ROOT / 'results'))
    ap.add_argument('--reps', nargs='+', default=SPANNING_REPS,
                    help='the representations the deep run uses. Default spans '
                         'fingerprint, descriptor and learned embedding.')
    ap.add_argument('--n-models', type=int, default=4,
                    help='4 models x 3 representations is the dozen pairs 13.14 '
                         'prices the deep run at.')
    ap.add_argument('--out', default=None,
                    help='write the selection here as JSON (default: '
                         '<results-dir>/deep_run_pairs.json)')
    cli = ap.parse_args()

    check_family_map()
    results_dir = Path(cli.results_dir)
    print(f"=== reading the screen from {results_dir}")
    df = fig.load_anova_data(results_dir)
    if df is None or len(df) == 0:
        sys.exit("FAIL: no anova_*.csv rows. Is this the right --results-dir?")

    conditions = sorted(df['strategy'].unique())
    print(f"  {len(df):,} rows, conditions: {', '.join(conditions)}")
    models, reps_present, absent = coverage(df, conditions)

    robust, excluded = fig.calculate_robustness(df)
    if robust.empty:
        sys.exit("FAIL: no configuration cleared the baseline floor, so nothing "
                 "can be ranked. Check the clean level landed.")
    if len(excluded):
        print(f"\n  {len(excluded)} configuration(s) dropped by the clean-R2 floor "
              f"of {fig.ROBUSTNESS_BASELINE_THRESHOLD} -- declared, not silent:")
        for _, r in excluded.iterrows():
            print(f"      {r['model']:28s} {r['rep']:20s} {r['strategy']:18s} "
                  f"clean R2 {r['baseline']:.3f}")

    per_rep_tables(robust, [r for r in cli.reps if r in reps_present], conditions)

    usable = [r for r in cli.reps if r in reps_present]
    missing_reps = [r for r in cli.reps if r not in reps_present]
    chosen, why = select(robust, usable, cli.n_models)

    print("\n=== THE SELECTION, on what has landed")
    for m in chosen:
        print(f"  {m:28s} {why[m]}")
    for r in cli.reps:
        kind = REP_KIND.get(r, 'UNCLASSIFIED')
        mark = '' if r in usable else '   <- NOT IN THE SCREEN YET'
        print(f"  {r:28s} {kind}{mark}")

    disagree = floor_disagrees(df, conditions, usable, cli.n_models)
    if disagree:
        print(f"\n  WARNING the clean-R2 floor changes the selection:")
        print(f"      with the floor:    {', '.join(disagree[0])}")
        print(f"      without the floor: {', '.join(disagree[1])}")
        print(f"      The choice is resting on the filter. Say which in the Methods, "
              f"or pick from the union.")
    else:
        print("\n  ok  the selection is the same with and without the clean-R2 floor.")

    unc = set(uncertainty_models())
    n_unc = len([m for m in chosen if m in unc])
    print(f"\n=== CENSORING uses the same selection (13.13: one selection, QM9 and "
          f"all three laboratory datasets).")
    if n_unc >= 2:
        print(f"  ok  {n_unc} of the {len(chosen)} report a per-molecule uncertainty, "
              f"and the rule asks for at least two.")
    else:
        print(f"  WARNING only {n_unc} of the {len(chosen)} report an uncertainty; the "
              f"rule asks for at least two. Add one of: "
              f"{', '.join(sorted(unc - set(chosen)))}")

    n_pairs = len(chosen) * len(cli.reps)
    print(f"\n=== SIZE")
    print(f"  {len(chosen)} models x {len(cli.reps)} representations = {n_pairs} pairs")
    print(f"  The training-run total is NOT computed here. Only gaussian runs the "
          f"clean level and the depth conditions carry six levels rather than "
          f"seven, so the obvious multiplication overstates it -- this script "
          f"said 5,880 against the generator's 4,440 the first time it ran. The "
          f"generator prints its own total; the commands below are how you get it.")

    labels = generator_labels(chosen)
    print(f"\n=== THE FLAGS, in the generator's own labels")
    print(f"  QM9:        python generate_scripts.py --stage 2 \\")
    print(f"                --models {' '.join(labels)} \\")
    print(f"                --reps {' '.join(cli.reps)}")
    print(f"  censoring:  python generate_scripts.py --stage 2 --conditions censoring \\")
    print(f"                --models {' '.join(labels)} \\")
    print(f"                --reps {' '.join(cli.reps)} --out-dir <its own directory>")

    if absent or missing_reps:
        print(f"\n=== PROVISIONAL. This is a reading of a screen that is not finished.")
        if absent:
            print(f"  {len(absent)} model(s) have not landed and could not be ranked: "
                  f"{', '.join(absent)}")
        if missing_reps:
            print(f"  representation(s) not in the screen yet: {', '.join(missing_reps)}")
        print(f"  Re-run this when they land. The rule does not change; only what it "
              f"can see does.")

    out = Path(cli.out) if cli.out else results_dir / 'deep_run_pairs.json'
    out.write_text(json.dumps({
        'what_this_is': 'the deep-run and censoring selection, read off the screen',
        'rule': 'RERUN_PLAN.md 13.17 B',
        'provisional': bool(absent or missing_reps),
        'models_absent_from_the_screen': absent,
        'models': chosen,
        'generator_labels': labels,
        'why': why,
        'representations': cli.reps,
        'n_pairs': n_pairs,
        'conditions_in_the_screen': conditions,
    }, indent=2) + '\n')
    print(f"\n  written: {out}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
