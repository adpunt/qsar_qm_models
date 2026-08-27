#!/usr/bin/env python3
"""
Compute pairwise correlations between models and between reps
to identify redundant levels for the ANOVA.

If two models have nearly identical NDS profiles across reps (rho > 0.9),
including both inflates the model factor. Same logic for reps.
"""

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('../results/paper_figures/table2_supp_nds_all_reps.csv')

print("=" * 80)
print("REDUNDANCY CHECK: PAIRWISE CORRELATIONS")
print("=" * 80)

# ============================================================
# MODEL-MODEL correlations (NDS profile across reps × strategies)
# ============================================================
print("\n" + "=" * 80)
print("MODEL-MODEL CORRELATIONS")
print("Using MEAN NDS across strategies as the response")
print("=" * 80)

# We need models that share reps to compare them
# Use the MEAN column
models = sorted(df['model'].unique())

# Build model profiles: for each model, its NDS on each rep
model_profiles = {}
for model in models:
    model_data = df[df['model'] == model][['rep', 'MEAN']].dropna(subset=['MEAN'])
    model_profiles[model] = dict(zip(model_data['rep'], model_data['MEAN']))

# Pairwise correlations
print(f"\nPairwise Spearman correlations (models with ≥3 shared reps):")
print(f"{'Model A':<25} {'Model B':<25} {'N reps':>8} {'rho':>8} {'p':>10} {'Note'}")
print("-" * 90)

high_corr_pairs = []
for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        shared_reps = sorted(set(model_profiles[m1].keys()) & set(model_profiles[m2].keys()))
        if len(shared_reps) < 3:
            continue

        vals1 = [model_profiles[m1][r] for r in shared_reps]
        vals2 = [model_profiles[m2][r] for r in shared_reps]

        rho, p = stats.spearmanr(vals1, vals2)

        note = ""
        if abs(rho) > 0.95:
            note = "REDUNDANT"
            high_corr_pairs.append((m1, m2, rho, len(shared_reps)))
        elif abs(rho) > 0.85:
            note = "HIGH"

        if abs(rho) > 0.8 or len(shared_reps) >= 4:
            print(f"{m1:<25} {m2:<25} {len(shared_reps):>8} {rho:>8.4f} {p:>10.4f} {note}")

# Also compute using all 6 strategy columns independently
print(f"\n\n{'=' * 80}")
print("MODEL-MODEL CORRELATIONS (using all strategy NDS values, not just MEAN)")
print("=" * 80)

strategies = ['hetero', 'legacy', 'outlier', 'quantile', 'threshold', 'valprop']

# Build full profiles: for each model, a vector of (rep, strategy) NDS values
def get_full_profile(model_name):
    """Get all available NDS values for a model as a dict of (rep, strategy) -> NDS"""
    model_data = df[df['model'] == model_name]
    profile = {}
    for _, row in model_data.iterrows():
        for strat in strategies:
            if pd.notna(row[strat]):
                profile[(row['rep'], strat)] = row[strat]
    return profile

print(f"\nPairwise correlations using full (rep × strategy) vectors:")
print(f"{'Model A':<25} {'Model B':<25} {'N pts':>8} {'rho':>8} {'Note'}")
print("-" * 80)

full_profiles = {m: get_full_profile(m) for m in models}

for i, m1 in enumerate(models):
    for m2 in models[i+1:]:
        shared_keys = sorted(set(full_profiles[m1].keys()) & set(full_profiles[m2].keys()))
        if len(shared_keys) < 6:
            continue

        vals1 = [full_profiles[m1][k] for k in shared_keys]
        vals2 = [full_profiles[m2][k] for k in shared_keys]

        rho, p = stats.spearmanr(vals1, vals2)

        note = ""
        if abs(rho) > 0.95:
            note = "REDUNDANT"
        elif abs(rho) > 0.85:
            note = "HIGH"

        if abs(rho) > 0.8 or len(shared_keys) >= 20:
            print(f"{m1:<25} {m2:<25} {len(shared_keys):>8} {rho:>8.4f} {note}")

# ============================================================
# REP-REP correlations (NDS profile across models)
# ============================================================
print(f"\n\n{'=' * 80}")
print("REP-REP CORRELATIONS")
print("Using MEAN NDS, comparing which models rank similarly across reps")
print("=" * 80)

reps = sorted(df['rep'].unique())

# Already have this from per_configuration_analysis.md but let's compute fresh
rep_profiles = {}
for rep in reps:
    rep_data = df[df['rep'] == rep][['model', 'MEAN']].dropna(subset=['MEAN'])
    rep_profiles[rep] = dict(zip(rep_data['model'], rep_data['MEAN']))

print(f"\nPairwise Spearman correlations (reps with ≥3 shared models):")
print(f"{'Rep A':<22} {'Rep B':<22} {'N models':>10} {'rho':>8} {'p':>10} {'Note'}")
print("-" * 85)

for i, r1 in enumerate(reps):
    for r2 in reps[i+1:]:
        shared_models = sorted(set(rep_profiles[r1].keys()) & set(rep_profiles[r2].keys()))
        if len(shared_models) < 3:
            continue

        vals1 = [rep_profiles[r1][m] for m in shared_models]
        vals2 = [rep_profiles[r2][m] for m in shared_models]

        rho, p = stats.spearmanr(vals1, vals2)

        note = ""
        if abs(rho) > 0.95:
            note = "REDUNDANT"
        elif abs(rho) > 0.85:
            note = "HIGH"

        print(f"{r1:<22} {r2:<22} {len(shared_models):>10} {rho:>8.4f} {p:>10.4f} {note}")


# ============================================================
# SUMMARY: What to keep / drop
# ============================================================
print(f"\n\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)

print(f"\nModels with ≥3 shared reps and rho > 0.9 (candidates for dropping one):")
if high_corr_pairs:
    for m1, m2, rho, n in high_corr_pairs:
        print(f"  {m1} ↔ {m2}: rho = {rho:.4f} (on {n} shared reps)")
else:
    print("  None found (using MEAN NDS)")

print(f"\nRep pairs with rho > 0.85 (potentially redundant):")
for i, r1 in enumerate(reps):
    for r2 in reps[i+1:]:
        shared_models = sorted(set(rep_profiles[r1].keys()) & set(rep_profiles[r2].keys()))
        if len(shared_models) < 3:
            continue
        vals1 = [rep_profiles[r1][m] for m in shared_models]
        vals2 = [rep_profiles[r2][m] for m in shared_models]
        rho, _ = stats.spearmanr(vals1, vals2)
        if abs(rho) > 0.85:
            print(f"  {r1} ↔ {r2}: rho = {rho:.4f} (on {len(shared_models)} models)")

# Count what a balanced design would look like
print(f"\n\nCurrent model×rep coverage:")
for model in models:
    reps_available = sorted(df[df['model'] == model]['rep'].unique())
    print(f"  {model:<25} {len(reps_available)} reps: {reps_available}")
