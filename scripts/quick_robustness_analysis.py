#!/usr/bin/env python3
"""
Quick analysis of existing ANOVA data to identify robustness patterns.
Helps decide which models to investigate with uncertainty analysis.

Run locally: python quick_robustness_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

RESULTS_DIR = Path(__file__).parent.parent / "results"
BASELINE_THRESHOLD = 0.5  # Models with R² < this on clean data are excluded

# Track excluded configurations for transparency
EXCLUDED = []

def load_anova_data():
    """Load all anova_*.csv files."""
    all_data = []

    for f in RESULTS_DIR.glob("anova_*.csv"):
        if '_uncertainty_values' in f.name:
            continue

        try:
            df = pd.read_csv(f)
            # Parse filename: anova_{strategy}_{rep}_{model}.csv
            parts = f.stem.split('_')
            if len(parts) >= 4:
                df['strategy'] = parts[1]
                df['source_file'] = f.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded {len(combined)} rows from {len(all_data)} files")
        return combined
    return None


def calculate_nds(group, model, rep, strategy):
    """Calculate Noise Degradation Slope for a model-rep-strategy combination.

    Models with baseline R² < BASELINE_THRESHOLD are excluded from robustness
    comparisons since they failed to learn the task on clean data.
    """
    group = group.sort_values('sigma')

    if len(group) < 3:
        return None

    # Get baseline (R² at sigma=0)
    baseline = group[group['sigma'] == 0.0]['r2'].mean()

    if pd.isna(baseline):
        return None

    if baseline < BASELINE_THRESHOLD:
        EXCLUDED.append({
            'model': model,
            'rep': rep,
            'strategy': strategy,
            'baseline_r2': baseline,
        })
        return None

    # Linear regression of R² vs sigma
    try:
        slope, intercept, r_val, p_val, std_err = stats.linregress(
            group['sigma'], group['r2']
        )
        return {
            'nds': slope,
            'baseline_r2': baseline,
        }
    except:
        return None


def create_strategy_table(nds_df, rep_filter=None, title="ALL REPRESENTATIONS"):
    """Create a table with models as rows, strategies as columns."""

    df = nds_df.copy()
    if rep_filter:
        df = df[df['rep'].isin(rep_filter)]

    if len(df) == 0:
        return None

    # Pivot: models as rows, strategies as columns
    pivot = df.pivot_table(
        values='nds',
        index='model',
        columns='strategy',
        aggfunc='mean'
    ).round(3)

    # Add mean across strategies
    pivot['MEAN'] = pivot.mean(axis=1).round(3)

    # Sort by mean (most robust first)
    pivot = pivot.sort_values('MEAN', ascending=False)

    # Add baseline R² info
    baseline_pivot = df.pivot_table(
        values='baseline_r2',
        index='model',
        columns='strategy',
        aggfunc='mean'
    ).round(2)

    return pivot, baseline_pivot


def main():
    print("=" * 80)
    print("ROBUSTNESS ANALYSIS - NOISE DEGRADATION SLOPES (NDS)")
    print("=" * 80)
    print(f"\nNote: Models with baseline R² < {BASELINE_THRESHOLD} on clean data are EXCLUDED")
    print("      (they failed to learn the task, robustness comparison meaningless)")

    df = load_anova_data()
    if df is None:
        print("No data found!")
        return

    print(f"\nStrategies: {sorted(df['strategy'].unique())}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Representations: {sorted(df['rep'].unique())}")
    print(f"Sigma range: {df['sigma'].min()} - {df['sigma'].max()}")

    # Calculate NDS for each model-rep-strategy
    nds_results = []

    for (model, rep, strategy), group in df.groupby(['model', 'rep', 'strategy']):
        # Average across iterations first
        avg_group = group.groupby('sigma')['r2'].mean().reset_index()
        avg_group['sigma'] = avg_group['sigma'].astype(float)

        result = calculate_nds(avg_group, model, rep, strategy)
        if result:
            result['model'] = model
            result['rep'] = rep
            result['strategy'] = strategy
            nds_results.append(result)

    nds_df = pd.DataFrame(nds_results)
    print(f"\nCalculated NDS for {len(nds_df)} valid model-rep-strategy combinations")
    print(f"Excluded {len(EXCLUDED)} combinations due to low baseline R²")

    # =========================================================================
    # EXCLUDED CONFIGURATIONS
    # =========================================================================
    if EXCLUDED:
        print("\n" + "=" * 80)
        print(f"EXCLUDED CONFIGURATIONS (baseline R² < {BASELINE_THRESHOLD})")
        print("=" * 80)
        excluded_df = pd.DataFrame(EXCLUDED)
        excluded_summary = excluded_df.groupby(['model', 'rep']).agg({
            'baseline_r2': 'mean',
            'strategy': 'count'
        }).round(3)
        excluded_summary.columns = ['mean_baseline_r2', 'n_strategies_excluded']
        print("\n" + excluded_summary.to_string())

    # =========================================================================
    # TABLE 1: ALL REPRESENTATIONS
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL ROBUSTNESS BY NOISE STRATEGY - ALL REPRESENTATIONS")
    print("(NDS values; higher/less negative = more robust)")
    print("=" * 80)

    pivot_all, baseline_all = create_strategy_table(nds_df)
    if pivot_all is not None:
        print("\n" + pivot_all.to_string())

    # =========================================================================
    # TABLE 2: PDV ONLY
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL ROBUSTNESS BY NOISE STRATEGY - PDV ONLY")
    print("=" * 80)

    pivot_pdv, _ = create_strategy_table(nds_df, rep_filter=['pdv'])
    if pivot_pdv is not None:
        print("\n" + pivot_pdv.to_string())
    else:
        print("No PDV data")

    # =========================================================================
    # TABLE 3: SNS ONLY
    # =========================================================================
    print("\n" + "=" * 80)
    print("MODEL ROBUSTNESS BY NOISE STRATEGY - SNS ONLY")
    print("=" * 80)

    pivot_sns, _ = create_strategy_table(nds_df, rep_filter=['sns'])
    if pivot_sns is not None:
        print("\n" + pivot_sns.to_string())
    else:
        print("No SNS data")


    # =========================================================================
    # BASELINE R² TABLE (sanity check)
    # =========================================================================
    print("\n" + "=" * 80)
    print("BASELINE R² (σ=0) BY MODEL AND STRATEGY - ALL REPS")
    print("(Confirms models learned the task before noise added)")
    print("=" * 80)

    if baseline_all is not None:
        baseline_all['MEAN'] = baseline_all.mean(axis=1).round(2)
        baseline_all = baseline_all.sort_values('MEAN', ascending=False)
        print("\n" + baseline_all.to_string())

    # =========================================================================
    # RANKING ANALYSIS
    # =========================================================================
    print("\n" + "=" * 80)
    print("RANKING ANALYSIS")
    print("=" * 80)

    if pivot_all is not None:
        # Get rankings (1 = best/most robust)
        strategies_only = [c for c in pivot_all.columns if c != 'MEAN']
        rankings = pivot_all[strategies_only].rank(ascending=False)  # Higher NDS = rank 1

        # Average rank across strategies
        rankings['AVG_RANK'] = rankings.mean(axis=1)
        rankings['RANK_STD'] = rankings[strategies_only].std(axis=1)
        rankings = rankings.sort_values('AVG_RANK')

        print("\nModel Rankings by Strategy (1=best, lower=more robust):")
        print(rankings.round(2).to_string())

        # Kendall's W (coefficient of concordance)
        # W = 12 * S / (k^2 * (n^3 - n)) where S = sum of squared deviations from mean rank
        k = len(strategies_only)  # number of rankers (strategies)
        n = len(rankings)  # number of items (models)

        # Only use models with complete data
        complete_models = rankings[strategies_only].dropna().index
        if len(complete_models) >= 3:
            ranks_matrix = rankings.loc[complete_models, strategies_only].values
            mean_ranks = ranks_matrix.mean(axis=1)
            S = np.sum((ranks_matrix.sum(axis=1) - ranks_matrix.sum(axis=1).mean()) ** 2)
            W = 12 * S / (k**2 * (n**3 - n)) if n > 1 else 0

            print(f"\nKendall's W (ranking concordance): {W:.3f}")
            print("  W=1: perfect agreement across strategies")
            print("  W=0: no agreement")
            if W > 0.7:
                print("  → Strong agreement: rankings are consistent")
            elif W > 0.5:
                print("  → Moderate agreement")
            else:
                print("  → Weak agreement: rankings vary by strategy")

        # Most robust model per strategy
        print("\nMost robust model per strategy:")
        for col in strategies_only:
            valid = pivot_all[col].dropna()
            if len(valid) > 0:
                best = valid.idxmax()
                val = valid.loc[best]
                print(f"  {col:12} -> {best:20} (NDS={val:+.3f})")

        # Least robust model per strategy
        print("\nLeast robust model per strategy:")
        for col in strategies_only:
            valid = pivot_all[col].dropna()
            if len(valid) > 0:
                worst = valid.idxmin()
                val = valid.loc[worst]
                print(f"  {col:12} -> {worst:20} (NDS={val:+.3f})")

    # =========================================================================
    # RANKINGS BY NOISE LEVEL (SIGMA)
    # =========================================================================
    print("\n" + "=" * 80)
    print("RANKINGS BY NOISE LEVEL (σ)")
    print("Do rankings hold as noise increases?")
    print("=" * 80)

    # Get R² at each sigma level, averaged across strategies and reps
    sigma_performance = df.groupby(['model', 'sigma'])['r2'].mean().unstack()

    # Filter to models with baseline > threshold
    good_baseline = sigma_performance[0.0] >= BASELINE_THRESHOLD
    sigma_performance = sigma_performance[good_baseline]

    if len(sigma_performance) > 0:
        # Show performance table
        print("\nMean R² by model and sigma (averaged across strategies & reps):")
        print(sigma_performance.round(3).to_string())

        # Rankings at each sigma
        sigma_rankings = sigma_performance.rank(ascending=False)
        print("\nRankings at each sigma (1=best):")
        print(sigma_rankings.round(1).to_string())

        # Track rank changes
        print("\nRank trajectory (does top performer stay on top?):")
        sigmas = sorted([c for c in sigma_rankings.columns if isinstance(c, (int, float))])
        for model in sigma_rankings.index:
            ranks = [sigma_rankings.loc[model, s] for s in sigmas if s in sigma_rankings.columns]
            rank_str = " → ".join([f"{r:.0f}" for r in ranks])
            change = ranks[-1] - ranks[0] if len(ranks) > 1 else 0
            direction = "↓" if change > 0 else ("↑" if change < 0 else "=")
            print(f"  {model:20} {rank_str}  ({direction}{abs(change):.0f})")

    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    output_dir = RESULTS_DIR / "robustness_analysis"
    output_dir.mkdir(exist_ok=True)

    if pivot_all is not None:
        pivot_all.to_csv(output_dir / "nds_by_strategy_all_reps.csv")
    if pivot_pdv is not None:
        pivot_pdv.to_csv(output_dir / "nds_by_strategy_pdv.csv")
    if pivot_sns is not None:
        pivot_sns.to_csv(output_dir / "nds_by_strategy_sns.csv")

    nds_df.to_csv(output_dir / "all_nds_values.csv", index=False)

    if EXCLUDED:
        pd.DataFrame(EXCLUDED).to_csv(output_dir / "excluded_low_baseline.csv", index=False)

    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
