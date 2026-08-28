#!/usr/bin/env python3
"""
Compute ICC (Intraclass Correlation) per proposed model family
to check whether family groupings are valid for ANOVA.

ICC(1,1) = (BMS - WMS) / (BMS + (k-1)*WMS)
where BMS = between-group mean square, WMS = within-group mean square,
k = number of observations per group.

High ICC (>0.7) = family members behave similarly → valid grouping
Low ICC (<0.3) = family members are as different from each other as from other families → invalid
"""

import pandas as pd
import numpy as np

# Load NDS data
df = pd.read_csv('../results/paper_figures/table2_supp_nds_all_reps.csv')

print("=" * 80)
print("ICC ANALYSIS FOR MODEL FAMILY GROUPINGS")
print("=" * 80)
print(f"\nData: {len(df)} model×rep configurations")
print(f"Models: {sorted(df['model'].unique())}")
print(f"Reps: {sorted(df['rep'].unique())}")

# Proposed families
families = {
    'tree': ['rf', 'xgboost', 'qrf', 'ngboost'],  # lgb not in data yet
    'nn': ['flexible_dnn', 'bnn_last', 'bnn_variational', 'mlp_bnn_last', 'mlp_bnn_variational'],
    # dnn, mlp partial; dnn_bnn_full, mlp_bnn_full missing
    'kernel': ['gauche'],  # svm not in data yet — can't compute ICC with 1 model
}

# Use MEAN NDS (across strategies) as the response
# Only include reps where ALL family members have data
strategies = ['hetero', 'legacy', 'outlier', 'quantile', 'threshold', 'valprop']

print("\n" + "=" * 80)
print("FAMILY-LEVEL ICC ANALYSIS")
print("=" * 80)

for family_name, members in families.items():
    print(f"\n{'─' * 60}")
    print(f"FAMILY: {family_name}")
    print(f"Members: {members}")

    # Filter to family members
    family_df = df[df['model'].isin(members)].copy()

    if len(family_df) < 2:
        print(f"  Only {len(family_df)} configs — cannot compute ICC")
        continue

    # Find reps where ALL family members have data
    rep_coverage = family_df.groupby('rep')['model'].apply(set)
    complete_reps = [rep for rep, models in rep_coverage.items()
                     if set(members).issubset(models)]

    print(f"Reps with all members: {complete_reps}")

    if len(complete_reps) < 2:
        print(f"  Need ≥2 complete reps for meaningful ICC, have {len(complete_reps)}")
        # Show what's available
        for rep in sorted(family_df['rep'].unique()):
            models_present = sorted(family_df[family_df['rep'] == rep]['model'].unique())
            missing = set(members) - set(models_present)
            print(f"    {rep}: {models_present}" + (f" MISSING: {missing}" if missing else " ✓"))
        continue

    # Compute ICC using MEAN NDS
    balanced = family_df[family_df['rep'].isin(complete_reps)][['model', 'rep', 'MEAN']].copy()
    balanced = balanced.dropna(subset=['MEAN'])

    # Pivot: rows = reps (subjects), columns = models (raters/targets)
    pivot = balanced.pivot(index='rep', columns='model', values='MEAN')
    print(f"\nNDS matrix (reps × models):")
    print(pivot.round(4).to_string())

    n = len(pivot)       # number of reps
    k = len(pivot.columns)  # number of models in family

    if n < 2 or k < 2:
        print(f"  Need ≥2 reps and ≥2 models, have {n} reps, {k} models")
        continue

    # ICC(1,1): one-way random effects
    # Each rep is rated by different models
    grand_mean = pivot.values.mean()

    # Between-subjects (between reps) mean square
    row_means = pivot.mean(axis=1)
    BMS = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)

    # Within-subjects (within reps) mean square
    WMS = 0
    for _, row in pivot.iterrows():
        row_mean = row.mean()
        WMS += ((row - row_mean) ** 2).sum()
    WMS = WMS / (n * (k - 1))

    # ICC(1,1)
    icc = (BMS - WMS) / (BMS + (k - 1) * WMS)

    print(f"\nICC(1,1) = {icc:.4f}")
    print(f"  BMS (between-rep variance) = {BMS:.6f}")
    print(f"  WMS (within-rep variance, i.e., between-model) = {WMS:.6f}")

    if icc > 0.75:
        print(f"  → EXCELLENT agreement: family members behave very similarly")
    elif icc > 0.5:
        print(f"  → MODERATE agreement: family members share patterns but differ in magnitude")
    elif icc > 0.25:
        print(f"  → POOR agreement: family members show different patterns")
    else:
        print(f"  → NO agreement: family grouping is not valid")

    # Also show within-family spread per rep
    print(f"\n  Within-family spread per rep:")
    for rep in complete_reps:
        vals = pivot.loc[rep]
        print(f"    {rep}: range [{vals.min():.4f}, {vals.max():.4f}], spread = {vals.max() - vals.min():.4f}")

    # Show per-model means across reps
    print(f"\n  Model means across reps:")
    for model in pivot.columns:
        vals = pivot[model]
        print(f"    {model}: mean = {vals.mean():.4f}, std = {vals.std():.4f}")

    # Also compute ICC per strategy (not just MEAN)
    print(f"\n  ICC per strategy:")
    for strategy in strategies:
        strat_data = family_df[family_df['rep'].isin(complete_reps)][['model', 'rep', strategy]].copy()
        strat_data = strat_data.dropna(subset=[strategy])
        if len(strat_data) < 4:
            print(f"    {strategy}: insufficient data")
            continue

        try:
            strat_pivot = strat_data.pivot(index='rep', columns='model', values=strategy)
            strat_pivot = strat_pivot.dropna()

            if len(strat_pivot) < 2 or len(strat_pivot.columns) < 2:
                print(f"    {strategy}: insufficient complete data")
                continue

            n_s = len(strat_pivot)
            k_s = len(strat_pivot.columns)
            gm = strat_pivot.values.mean()
            rm = strat_pivot.mean(axis=1)
            bms_s = k_s * ((rm - gm) ** 2).sum() / (n_s - 1)
            wms_s = sum(((strat_pivot.loc[idx] - strat_pivot.loc[idx].mean()) ** 2).sum()
                        for idx in strat_pivot.index) / (n_s * (k_s - 1))

            if bms_s + (k_s - 1) * wms_s == 0:
                print(f"    {strategy}: zero variance")
                continue

            icc_s = (bms_s - wms_s) / (bms_s + (k_s - 1) * wms_s)
            print(f"    {strategy}: ICC = {icc_s:.4f}")
        except Exception as e:
            print(f"    {strategy}: error - {e}")


# Also check: do NNs split into "deterministic" vs "bayesian" subgroups?
print(f"\n{'=' * 80}")
print("NN SUBGROUP ANALYSIS")
print("=" * 80)

nn_det = ['flexible_dnn']  # dnn, mlp are partial
nn_bnn = ['bnn_last', 'bnn_variational', 'mlp_bnn_last', 'mlp_bnn_variational']

# For BNN subgroup only
bnn_df = df[df['model'].isin(nn_bnn)]
bnn_reps = bnn_df.groupby('rep')['model'].apply(set)
bnn_complete = [rep for rep, models in bnn_reps.items() if set(nn_bnn).issubset(models)]

print(f"\nBNN subgroup: {nn_bnn}")
print(f"Complete reps: {bnn_complete}")

if len(bnn_complete) >= 2:
    bnn_balanced = bnn_df[bnn_df['rep'].isin(bnn_complete)][['model', 'rep', 'MEAN']].dropna(subset=['MEAN'])
    bnn_pivot = bnn_balanced.pivot(index='rep', columns='model', values='MEAN')
    print(f"\nBNN NDS matrix:")
    print(bnn_pivot.round(4).to_string())

    n = len(bnn_pivot)
    k = len(bnn_pivot.columns)
    gm = bnn_pivot.values.mean()
    rm = bnn_pivot.mean(axis=1)
    BMS = k * ((rm - gm) ** 2).sum() / (n - 1)
    WMS = sum(((bnn_pivot.loc[idx] - bnn_pivot.loc[idx].mean()) ** 2).sum()
              for idx in bnn_pivot.index) / (n * (k - 1))

    icc = (BMS - WMS) / (BMS + (k - 1) * WMS)
    print(f"\nBNN subgroup ICC = {icc:.4f}")

    print(f"\n  Within-BNN spread per rep:")
    for rep in bnn_complete:
        vals = bnn_pivot.loc[rep]
        print(f"    {rep}: range [{vals.min():.4f}, {vals.max():.4f}], spread = {vals.max() - vals.min():.4f}")


# Cross-family comparison: how does between-family variance compare to within-family?
print(f"\n{'=' * 80}")
print("BETWEEN vs WITHIN FAMILY VARIANCE")
print("=" * 80)

# Use reps where BOTH tree and nn families have complete data
tree_members = ['rf', 'xgboost', 'qrf', 'ngboost']
nn_members = ['flexible_dnn', 'bnn_last', 'bnn_variational', 'mlp_bnn_last', 'mlp_bnn_variational']

tree_df = df[df['model'].isin(tree_members)]
nn_df_sub = df[df['model'].isin(nn_members)]

tree_complete = set(r for r, m in tree_df.groupby('rep')['model'].apply(set).items() if set(tree_members).issubset(m))
nn_complete = set(r for r, m in nn_df_sub.groupby('rep')['model'].apply(set).items() if set(nn_members).issubset(m))
shared_reps = sorted(tree_complete & nn_complete)

print(f"\nReps with complete data for both tree and NN families: {shared_reps}")

if shared_reps:
    print(f"\nPer rep: family means and within-family std")
    print(f"{'Rep':<20} {'Tree mean':>12} {'Tree std':>12} {'NN mean':>12} {'NN std':>12} {'Gap':>12}")

    for rep in shared_reps:
        tree_vals = df[(df['model'].isin(tree_members)) & (df['rep'] == rep)]['MEAN'].values
        nn_vals = df[(df['model'].isin(nn_members)) & (df['rep'] == rep)]['MEAN'].values

        print(f"{rep:<20} {np.mean(tree_vals):>12.4f} {np.std(tree_vals):>12.4f} "
              f"{np.mean(nn_vals):>12.4f} {np.std(nn_vals):>12.4f} "
              f"{abs(np.mean(tree_vals) - np.mean(nn_vals)):>12.4f}")

    # Overall
    all_tree = df[(df['model'].isin(tree_members)) & (df['rep'].isin(shared_reps))]['MEAN'].values
    all_nn = df[(df['model'].isin(nn_members)) & (df['rep'].isin(shared_reps))]['MEAN'].values

    print(f"\nOverall:")
    print(f"  Tree family: mean = {np.mean(all_tree):.4f}, std = {np.std(all_tree):.4f}")
    print(f"  NN family:   mean = {np.mean(all_nn):.4f}, std = {np.std(all_nn):.4f}")
    print(f"  Between-family gap: {abs(np.mean(all_tree) - np.mean(all_nn)):.4f}")
    print(f"  Tree within-family std: {np.std(all_tree):.4f}")
    print(f"  NN within-family std:   {np.std(all_nn):.4f}")

    if abs(np.mean(all_tree) - np.mean(all_nn)) > max(np.std(all_tree), np.std(all_nn)):
        print(f"  → Between-family gap > within-family spread: families are distinguishable")
    else:
        print(f"  → Between-family gap ≤ within-family spread: families overlap substantially")
