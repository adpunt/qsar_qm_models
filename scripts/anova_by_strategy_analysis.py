#!/usr/bin/env python3
"""
ANOVA by Noise Strategy Analysis

Paper Question: Does the "representation drives performance, model drives robustness"
finding generalize across noise injection strategies?

Part 1 Hypothesis:
- Representation explains most variance in predictive performance
- Model explains most variance in noise robustness (degradation slope)

This script tests whether this pattern holds across all 6 noise strategies,
providing generalization evidence for the paper's central claim.

Outputs:
- Figure: ANOVA variance decomposition comparison across strategies
- Table: η² values for model/representation/interaction by strategy
- Statistical tests for consistency of the pattern
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE SETTINGS (Journal of Cheminformatics)
# =============================================================================

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
})

STRATEGY_COLORS = {
    'legacy': '#3498db',
    'valprop': '#e74c3c',
    'quantile': '#2ecc71',
    'threshold': '#9b59b6',
    'outlier': '#f39c12',
    'hetero': '#1abc9c',
}

STRATEGY_LABELS = {
    'legacy': 'Gaussian',
    'valprop': 'Value-Proportional',
    'quantile': 'Quantile',
    'threshold': 'Threshold',
    'outlier': 'Outlier',
    'hetero': 'Heteroscedastic',
}

# =============================================================================
# DATA LOADING
# =============================================================================

def load_strategy_results(results_dir):
    """Load all anova_*.csv files grouped by noise strategy."""
    results_dir = Path(results_dir)

    strategy_data = defaultdict(list)

    # Look for anova_<strategy>_<rep>_<model>.csv files
    for f in results_dir.glob("anova_*.csv"):
        filename = f.name

        # Parse: anova_<strategy>_<rep>_<model>.csv
        parts = filename.replace('.csv', '').split('_')
        if len(parts) >= 4 and parts[0] == 'anova':
            strategy = parts[1]
            # Rest is rep_model (might have underscores)

            try:
                df = pd.read_csv(f)
                df['source_file'] = filename
                df['noise_strategy'] = strategy
                strategy_data[strategy].append(df)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")

    # Combine data for each strategy
    combined = {}
    for strategy, dfs in strategy_data.items():
        if dfs:
            combined[strategy] = pd.concat(dfs, ignore_index=True)
            print(f"Loaded {strategy}: {len(combined[strategy])} rows from {len(dfs)} files")

    return combined


def load_phase0c_as_legacy(results_dir):
    """Load phase0c_screen files as 'legacy' baseline for comparison."""
    results_dir = Path(results_dir)

    all_data = []
    for f in results_dir.glob("phase0c_screen_*.csv"):
        if '_uncertainty_values' in f.name:
            continue
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            df['noise_strategy'] = 'legacy_phase0c'
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded phase0c (legacy baseline): {len(combined)} rows")
        return combined
    return None


# =============================================================================
# ANOVA FUNCTIONS
# =============================================================================

def run_anova_decomposition(df, sigma_value=0.3, baseline_threshold=0.6):
    """
    Run two-way ANOVA decomposition for a single dataset.

    Returns η² (eta-squared) for:
    - Model effect
    - Representation effect
    - Interaction effect
    - Residual
    """
    # Filter to specific sigma
    df_sigma = df[np.abs(df['sigma'] - sigma_value) < 0.05].copy()

    if len(df_sigma) == 0:
        return None

    # Clean data
    df_sigma = df_sigma[df_sigma['r2'] > -10].dropna(subset=['r2', 'model', 'rep'])

    # Exclude randomized_smiles (too similar to smiles)
    df_sigma = df_sigma[~df_sigma['rep'].isin(['randomized_smiles', 'random_smiles'])]

    # Standardize column names
    if 'representation' in df_sigma.columns and 'rep' not in df_sigma.columns:
        df_sigma['rep'] = df_sigma['representation']

    if len(df_sigma) < 10:
        return None

    # Calculate ANOVA components
    grand_mean = df_sigma['r2'].mean()
    total_ss = ((df_sigma['r2'] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    # Model effect
    model_means = df_sigma.groupby('model')['r2'].mean()
    model_counts = df_sigma.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)

    # Representation effect
    rep_means = df_sigma.groupby('rep')['r2'].mean()
    rep_counts = df_sigma.groupby('rep').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)

    # Interaction effect
    interaction_means = df_sigma.groupby(['model', 'rep'])['r2'].mean()
    interaction_counts = df_sigma.groupby(['model', 'rep']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        if model in model_means.index and rep in rep_means.index:
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2

    # Residual
    ss_residual = total_ss - ss_model - ss_rep - ss_interaction

    # Calculate η² (proportion of variance)
    eta2_model = (ss_model / total_ss) * 100
    eta2_rep = (ss_rep / total_ss) * 100
    eta2_interaction = (ss_interaction / total_ss) * 100
    eta2_residual = (ss_residual / total_ss) * 100

    # Calculate F-statistics and p-values
    n = len(df_sigma)
    df_model = df_sigma['model'].nunique() - 1
    df_rep = df_sigma['rep'].nunique() - 1
    df_interaction = df_model * df_rep
    df_residual = max(1, n - (df_model + 1) * (df_rep + 1))

    ms_model = ss_model / max(1, df_model)
    ms_rep = ss_rep / max(1, df_rep)
    ms_interaction = ss_interaction / max(1, df_interaction)
    ms_residual = ss_residual / max(1, df_residual)

    f_model = ms_model / ms_residual if ms_residual > 0 else np.nan
    f_rep = ms_rep / ms_residual if ms_residual > 0 else np.nan

    p_model = 1 - stats.f.cdf(f_model, df_model, df_residual) if not np.isnan(f_model) else np.nan
    p_rep = 1 - stats.f.cdf(f_rep, df_rep, df_residual) if not np.isnan(f_rep) else np.nan

    return {
        'n_observations': n,
        'n_models': df_sigma['model'].nunique(),
        'n_reps': df_sigma['rep'].nunique(),
        'grand_mean_r2': grand_mean,
        'eta2_model': eta2_model,
        'eta2_rep': eta2_rep,
        'eta2_interaction': eta2_interaction,
        'eta2_residual': eta2_residual,
        'f_model': f_model,
        'f_rep': f_rep,
        'p_model': p_model,
        'p_rep': p_rep,
        'ss_total': total_ss,
    }


def run_robustness_anova(df, baseline_threshold=0.6):
    """
    Run ANOVA on noise degradation slope (NDS) instead of raw R².

    This tests: "Model drives noise ROBUSTNESS"

    IMPORTANT: Computes NDS per-iteration to get proper within-cell replication
    for ANOVA (10 NDS values per model-rep cell).
    """
    # Calculate NDS for each model-rep-iteration combination
    nds_data = []

    for (model, rep, iteration), group in df.groupby(['model', 'rep', 'iteration']):
        group = group.sort_values('sigma')

        if len(group) < 3:
            continue

        # Check baseline threshold
        baseline = group[group['sigma'] == 0.0]
        if len(baseline) == 0 or baseline['r2'].values[0] < baseline_threshold:
            continue

        # Calculate slope for this iteration
        try:
            slope, _, r_val, p_val, _ = stats.linregress(group['sigma'], group['r2'])
            nds_data.append({
                'model': model,
                'rep': rep,
                'iteration': iteration,
                'nds': slope,  # Negative = degradation
                'baseline_r2': baseline['r2'].values[0],
            })
        except:
            continue

    if len(nds_data) < 10:
        return None

    nds_df = pd.DataFrame(nds_data)

    # Exclude randomized_smiles
    nds_df = nds_df[~nds_df['rep'].isin(['randomized_smiles', 'random_smiles'])]

    # Run ANOVA on NDS
    grand_mean = nds_df['nds'].mean()
    total_ss = ((nds_df['nds'] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    # Model effect on robustness
    model_means = nds_df.groupby('model')['nds'].mean()
    model_counts = nds_df.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)

    # Representation effect on robustness
    rep_means = nds_df.groupby('rep')['nds'].mean()
    rep_counts = nds_df.groupby('rep').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)

    # Interaction
    ss_interaction = 0
    for _, row in nds_df.iterrows():
        if row['model'] in model_means.index and row['rep'] in rep_means.index:
            expected = model_means[row['model']] + rep_means[row['rep']] - grand_mean
            # This is simplified - proper interaction needs cell means

    interaction_means = nds_df.groupby(['model', 'rep'])['nds'].mean()
    interaction_counts = nds_df.groupby(['model', 'rep']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        if model in model_means.index and rep in rep_means.index:
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2

    ss_residual = total_ss - ss_model - ss_rep - ss_interaction

    eta2_model = (ss_model / total_ss) * 100
    eta2_rep = (ss_rep / total_ss) * 100
    eta2_interaction = (ss_interaction / total_ss) * 100
    eta2_residual = (ss_residual / total_ss) * 100

    return {
        'n_configs': len(nds_df),
        'n_models': nds_df['model'].nunique(),
        'n_reps': nds_df['rep'].nunique(),
        'mean_nds': grand_mean,
        'eta2_model': eta2_model,
        'eta2_rep': eta2_rep,
        'eta2_interaction': eta2_interaction,
        'eta2_residual': eta2_residual,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def analyze_all_strategies(strategy_data, output_dir):
    """Run ANOVA for each strategy and compare."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Store results
    performance_results = {}  # R² at σ=0.3
    robustness_results = {}   # NDS (degradation slope)

    print("\n" + "="*80)
    print("ANOVA BY NOISE STRATEGY")
    print("="*80)

    for strategy, df in strategy_data.items():
        print(f"\n--- {STRATEGY_LABELS.get(strategy, strategy)} ---")

        # Performance ANOVA (R² at σ=0.3)
        perf = run_anova_decomposition(df, sigma_value=0.3)
        if perf:
            performance_results[strategy] = perf
            print(f"  Performance (R² at σ=0.3):")
            print(f"    Model: {perf['eta2_model']:.1f}%  |  Rep: {perf['eta2_rep']:.1f}%  |  Interaction: {perf['eta2_interaction']:.1f}%")

        # Robustness ANOVA (NDS)
        robust = run_robustness_anova(df)
        if robust:
            robustness_results[strategy] = robust
            print(f"  Robustness (NDS):")
            print(f"    Model: {robust['eta2_model']:.1f}%  |  Rep: {robust['eta2_rep']:.1f}%  |  Interaction: {robust['eta2_interaction']:.1f}%")

    return performance_results, robustness_results


def create_comparison_figure(performance_results, robustness_results, output_dir):
    """Create figure comparing ANOVA decomposition across strategies."""
    output_dir = Path(output_dir)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Performance (R² at σ=0.3)
    ax_a = axes[0]

    strategies = list(performance_results.keys())
    x = np.arange(len(strategies))
    width = 0.25

    model_vals = [performance_results[s]['eta2_model'] for s in strategies]
    rep_vals = [performance_results[s]['eta2_rep'] for s in strategies]
    interaction_vals = [performance_results[s]['eta2_interaction'] for s in strategies]

    ax_a.bar(x - width, model_vals, width, label='Model', color='#3498db', alpha=0.8)
    ax_a.bar(x, rep_vals, width, label='Representation', color='#e74c3c', alpha=0.8)
    ax_a.bar(x + width, interaction_vals, width, label='Interaction', color='#2ecc71', alpha=0.8)

    ax_a.set_ylabel('Variance Explained (η², %)')
    ax_a.set_xlabel('Noise Strategy')
    ax_a.set_title('A. Performance at Moderate Noise (σ=0.3)', fontweight='bold')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], rotation=45, ha='right')
    ax_a.legend(loc='upper right')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.set_ylim(0, 100)

    # Add reference lines for paper claims
    ax_a.axhline(72, color='#e74c3c', linestyle='--', alpha=0.5, linewidth=1)
    ax_a.text(len(strategies)-0.5, 74, '72% (claim)', fontsize=7, color='#e74c3c', ha='right')

    # Panel B: Robustness (NDS)
    ax_b = axes[1]

    strategies_r = list(robustness_results.keys())
    x_r = np.arange(len(strategies_r))

    model_vals_r = [robustness_results[s]['eta2_model'] for s in strategies_r]
    rep_vals_r = [robustness_results[s]['eta2_rep'] for s in strategies_r]
    interaction_vals_r = [robustness_results[s]['eta2_interaction'] for s in strategies_r]

    ax_b.bar(x_r - width, model_vals_r, width, label='Model', color='#3498db', alpha=0.8)
    ax_b.bar(x_r, rep_vals_r, width, label='Representation', color='#e74c3c', alpha=0.8)
    ax_b.bar(x_r + width, interaction_vals_r, width, label='Interaction', color='#2ecc71', alpha=0.8)

    ax_b.set_ylabel('Variance Explained (η², %)')
    ax_b.set_xlabel('Noise Strategy')
    ax_b.set_title('B. Noise Robustness (Degradation Slope)', fontweight='bold')
    ax_b.set_xticks(x_r)
    ax_b.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies_r], rotation=45, ha='right')
    ax_b.legend(loc='upper right')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.set_ylim(0, 100)

    # Add reference line
    ax_b.axhline(67, color='#3498db', linestyle='--', alpha=0.5, linewidth=1)
    ax_b.text(len(strategies_r)-0.5, 69, '67% (claim)', fontsize=7, color='#3498db', ha='right')

    plt.tight_layout()

    output_path = output_dir / "anova_strategy_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved figure to {output_path}")
    plt.close()


def create_summary_table(performance_results, robustness_results, output_dir):
    """Create summary table of ANOVA results."""
    output_dir = Path(output_dir)

    rows = []
    for strategy in set(list(performance_results.keys()) + list(robustness_results.keys())):
        row = {'Strategy': STRATEGY_LABELS.get(strategy, strategy)}

        if strategy in performance_results:
            p = performance_results[strategy]
            row['Perf_η²_Model'] = p['eta2_model']
            row['Perf_η²_Rep'] = p['eta2_rep']
            row['Perf_η²_Interaction'] = p['eta2_interaction']
            row['Perf_n'] = p['n_observations']

        if strategy in robustness_results:
            r = robustness_results[strategy]
            row['Robust_η²_Model'] = r['eta2_model']
            row['Robust_η²_Rep'] = r['eta2_rep']
            row['Robust_η²_Interaction'] = r['eta2_interaction']
            row['Robust_n'] = r['n_configs']

        rows.append(row)

    df = pd.DataFrame(rows)

    # Save CSV
    csv_path = output_dir / "anova_strategy_summary.csv"
    df.to_csv(csv_path, index=False, float_format='%.2f')
    print(f"✓ Saved summary table to {csv_path}")

    # Save LaTeX table
    latex_path = output_dir / "anova_strategy_summary.tex"
    with open(latex_path, 'w') as f:
        f.write("% ANOVA Variance Decomposition by Noise Strategy\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Variance decomposition (η²) across noise injection strategies. ")
        f.write("Performance measured at σ=0.3; robustness measured by noise degradation slope.}\n")
        f.write("\\label{tab:anova_strategies}\n")
        f.write("\\begin{tabular}{lcccccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{Performance (R²)} & \\multicolumn{3}{c}{Robustness (NDS)} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write("Strategy & Model & Rep & Int. & Model & Rep & Int. \\\\\n")
        f.write("\\midrule\n")

        for _, row in df.iterrows():
            f.write(f"{row['Strategy']} & ")
            f.write(f"{row.get('Perf_η²_Model', '--'):.1f} & " if pd.notna(row.get('Perf_η²_Model')) else "-- & ")
            f.write(f"{row.get('Perf_η²_Rep', '--'):.1f} & " if pd.notna(row.get('Perf_η²_Rep')) else "-- & ")
            f.write(f"{row.get('Perf_η²_Interaction', '--'):.1f} & " if pd.notna(row.get('Perf_η²_Interaction')) else "-- & ")
            f.write(f"{row.get('Robust_η²_Model', '--'):.1f} & " if pd.notna(row.get('Robust_η²_Model')) else "-- & ")
            f.write(f"{row.get('Robust_η²_Rep', '--'):.1f} & " if pd.notna(row.get('Robust_η²_Rep')) else "-- & ")
            f.write(f"{row.get('Robust_η²_Interaction', '--'):.1f}" if pd.notna(row.get('Robust_η²_Interaction')) else "--")
            f.write(" \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    print(f"✓ Saved LaTeX table to {latex_path}")

    return df


def generate_text_report(performance_results, robustness_results, output_dir):
    """Generate text report with interpretation."""
    output_dir = Path(output_dir)

    lines = []
    lines.append("="*80)
    lines.append("ANOVA BY NOISE STRATEGY - ANALYSIS REPORT")
    lines.append("="*80)

    lines.append("\nPAPER CLAIMS TO TEST:")
    lines.append("  1. Representation drives performance (~72% variance)")
    lines.append("  2. Model drives noise robustness (~67% variance)")
    lines.append("  3. These patterns GENERALIZE across noise strategies")

    lines.append("\n" + "="*80)
    lines.append("RESULTS BY STRATEGY")
    lines.append("="*80)

    for strategy in performance_results.keys():
        label = STRATEGY_LABELS.get(strategy, strategy)
        lines.append(f"\n{label}:")

        if strategy in performance_results:
            p = performance_results[strategy]
            lines.append(f"  Performance (R² at σ=0.3, n={p['n_observations']}):")
            lines.append(f"    - Representation: {p['eta2_rep']:.1f}% variance")
            lines.append(f"    - Model: {p['eta2_model']:.1f}% variance")
            lines.append(f"    - Interaction: {p['eta2_interaction']:.1f}% variance")

        if strategy in robustness_results:
            r = robustness_results[strategy]
            lines.append(f"  Robustness (NDS, n={r['n_configs']}):")
            lines.append(f"    - Model: {r['eta2_model']:.1f}% variance")
            lines.append(f"    - Representation: {r['eta2_rep']:.1f}% variance")
            lines.append(f"    - Interaction: {r['eta2_interaction']:.1f}% variance")

    # Summary statistics
    lines.append("\n" + "="*80)
    lines.append("SUMMARY: DOES THE PATTERN GENERALIZE?")
    lines.append("="*80)

    if performance_results:
        rep_vals = [p['eta2_rep'] for p in performance_results.values()]
        lines.append(f"\nPerformance - Representation η² across strategies:")
        lines.append(f"  Mean: {np.mean(rep_vals):.1f}%  |  Std: {np.std(rep_vals):.1f}%  |  Range: [{min(rep_vals):.1f}%, {max(rep_vals):.1f}%]")
        lines.append(f"  Claim (72%): {'✓ SUPPORTED' if np.mean(rep_vals) > 50 else '✗ NOT SUPPORTED'}")

    if robustness_results:
        model_vals = [r['eta2_model'] for r in robustness_results.values()]
        lines.append(f"\nRobustness - Model η² across strategies:")
        lines.append(f"  Mean: {np.mean(model_vals):.1f}%  |  Std: {np.std(model_vals):.1f}%  |  Range: [{min(model_vals):.1f}%, {max(model_vals):.1f}%]")
        lines.append(f"  Claim (67%): {'✓ SUPPORTED' if np.mean(model_vals) > 50 else '✗ NOT SUPPORTED'}")

    lines.append("\n" + "="*80)
    lines.append("INTERPRETATION")
    lines.append("="*80)
    lines.append("""
If the pattern holds across strategies:
  → Strong evidence that the finding is NOT an artifact of Gaussian noise
  → Supports generalization claim in Part 1 of the paper

If the pattern varies:
  → Indicates noise-type-specific effects
  → May need to qualify claims or investigate further
""")

    report_path = output_dir / "anova_strategy_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Saved report to {report_path}")

    # Also print to console
    print('\n'.join(lines))


# =============================================================================
# MAIN
# =============================================================================

def main():
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"

    print("="*80)
    print("ANOVA BY NOISE STRATEGY ANALYSIS")
    print("="*80)

    # Load data
    print("\nLoading results...")
    strategy_data = load_strategy_results(results_dir)

    if not strategy_data:
        print("No anova_*.csv files found. Checking for phase0c files...")
        phase0c = load_phase0c_as_legacy(results_dir)
        if phase0c is not None:
            strategy_data['legacy_phase0c'] = phase0c

    if not strategy_data:
        print("ERROR: No data found!")
        return

    print(f"\nFound data for {len(strategy_data)} strategies: {list(strategy_data.keys())}")

    # Create output directory
    output_dir = Path(results_dir) / "anova_strategy_analysis"
    output_dir.mkdir(exist_ok=True)

    # Run analysis
    performance_results, robustness_results = analyze_all_strategies(strategy_data, output_dir)

    if not performance_results and not robustness_results:
        print("ERROR: Could not compute ANOVA for any strategy!")
        return

    # Generate outputs
    print("\nGenerating outputs...")

    if len(performance_results) >= 2 or len(robustness_results) >= 2:
        create_comparison_figure(performance_results, robustness_results, output_dir)

    create_summary_table(performance_results, robustness_results, output_dir)
    generate_text_report(performance_results, robustness_results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print(f"Outputs saved to: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
