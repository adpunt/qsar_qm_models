"""
Phase 5 Analysis - Noise Strategy Generalization
Generates Figure 8 and Supplementary S8

Evaluates model robustness across different noise injection strategies:
- Gaussian (homoscedastic)
- Heteroscedastic
- Quantile-based
- Outlier injection
- Value-proportional

Key metrics:
- NSI (Noise Sensitivity Index): slope of R² vs σ
- Retention %: (R²_high / R²_baseline) × 100
- Baseline R² at σ=0
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# JOURNAL OF CHEMINFORMATICS STYLE
# ============================================================================

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Clean display names
NOISE_STRATEGY_LABELS = {
    'gaussian': 'Gaussian',
    'legacy': 'Legacy (Gaussian)',
    'hetero': 'Heteroscedastic',
    'heteroscedastic': 'Heteroscedastic',
    'value_proportional': 'Value-proportional',
    'proportional': 'Proportional',
    'quantile': 'Quantile-based',
    'outlier': 'Outlier',
    'threshold': 'Threshold',
    'uniform': 'Uniform',
}

NOISE_STRATEGY_COLORS = {
    'gaussian': '#3498db',
    'legacy': '#2ecc71',
    'hetero': '#e74c3c',
    'heteroscedastic': '#e74c3c',
    'value_proportional': '#9b59b6',
    'proportional': '#9b59b6',
    'quantile': '#f39c12',
    'outlier': '#e67e22',
    'threshold': '#1abc9c',
    'uniform': '#27ae60',
}

MODEL_LABELS = {
    'rf': 'RF',
    'qrf': 'QRF',
    'xgboost': 'XGBoost',
    'ngboost': 'NGBoost',
    'dnn': 'DNN',
    'bnn': 'BNN',
    'gauche': 'GAUCHE',
    'gp': 'GP',
}

REP_LABELS = {
    'ecfp4': 'ECFP4',
    'pdv': 'PDV',
    'sns': 'SNS',
    'smiles': 'SMILES',
    'graph': 'Graph',
}


def get_clean_label(value, label_dict):
    """Get clean display label from dictionary (case-insensitive)"""
    if value is None:
        return 'Unknown'
    value_lower = value.lower()
    return label_dict.get(value_lower, value.upper())


def get_color(value, color_dict, default='#888888'):
    """Get color from dictionary (case-insensitive)"""
    if value is None:
        return default
    value_lower = value.lower()
    return color_dict.get(value_lower, default)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_phase5_results(results_dir):
    """
    Load Phase 5 noise strategy results + Phase 0 as "legacy" baseline.
    
    Phase 5 format: phase5_{model}_{noise_strategy}.csv
    Phase 0 format: phase0_{rep}_{model}.csv (standard Gaussian noise)
    
    Only includes Phase 0 data for model/rep combinations that exist in Phase 5.
    """
    print("\n" + "=" * 80)
    print("LOADING PHASE 5 NOISE STRATEGY DATA")
    print("=" * 80)
    
    results_dir = Path(results_dir)
    
    # Exclude size experiment files (contain n followed by digits like n500, n1000)
    import re
    size_pattern = re.compile(r'_n\d+[_.]')
    
    # Load Phase 5 files first
    phase5_files = list(results_dir.glob("phase5*.csv"))
    phase5_files = [f for f in phase5_files if '_uncertainty' not in f.name.lower()]
    phase5_files = [f for f in phase5_files if not size_pattern.search(f.name)]
    
    print(f"Found {len(phase5_files)} Phase 5 result files")
    
    phase5_dfs = []
    for filepath in phase5_files:
        df = pd.read_csv(filepath)
        df['source_file'] = filepath.name
        df['phase'] = 'phase5'
        # Standardize column name immediately
        if 'rep' in df.columns and 'representation' not in df.columns:
            df = df.rename(columns={'rep': 'representation'})
        phase5_dfs.append(df)
    
    if not phase5_dfs:
        raise FileNotFoundError(f"No phase5*.csv files found in {results_dir}")
    
    phase5_df = pd.concat(phase5_dfs, ignore_index=True)
    
    # Filter out excluded representations before getting combos
    exclude_reps = ['graph', 'mhggnn']
    phase5_df_filtered = phase5_df[~phase5_df['representation'].str.lower().isin(exclude_reps)]
    
    # Get unique model/rep combinations from Phase 5
    phase5_combos = set(zip(phase5_df_filtered['model'].str.lower(), phase5_df_filtered['representation'].str.lower()))
    print(f"Phase 5 model/rep combinations: {phase5_combos}")
    
    # Load Phase 0 files as "legacy" noise strategy
    phase0_files = list(results_dir.glob("phase0*.csv"))
    phase0_files = [f for f in phase0_files if '_uncertainty' not in f.name.lower()]
    print(f"Found {len(phase0_files)} Phase 0 files")
    
    phase0_dfs = []
    for filepath in phase0_files:
        df = pd.read_csv(filepath)
        df['source_file'] = filepath.name
        df['phase'] = 'phase0'
        df['noise_strategy'] = 'legacy'
        
        # Standardize column name immediately
        if 'rep' in df.columns and 'representation' not in df.columns:
            df = df.rename(columns={'rep': 'representation'})
        
        # Only keep rows matching Phase 5 combinations
        if 'model' in df.columns and 'representation' in df.columns:
            df['_combo'] = list(zip(df['model'].str.lower(), df['representation'].str.lower()))
            df = df[df['_combo'].isin(phase5_combos)]
            df = df.drop(columns=['_combo'])
        
        if len(df) > 0:
            phase0_dfs.append(df)
    
    # Combine all
    all_dfs = phase5_dfs + phase0_dfs
    results_df = pd.concat(all_dfs, ignore_index=True)
    
    phase0_rows = len(results_df[results_df['phase'] == 'phase0'])
    print(f"Included {phase0_rows} Phase 0 rows matching Phase 5 configurations")
    print(f"Loaded {len(results_df)} total rows")
    
    return results_df
    
    return results_df


def parse_phase5_filenames(df):
    """
    Parse noise_strategy from filenames for Phase 5 data.
    Phase 0 rows already have noise_strategy='legacy' assigned.
    
    Phase 5 filename format: phase5_{prefix}_{noise_strategy}.csv
    """
    # Parse noise_strategy from filename for Phase 5 rows only
    def extract_noise_strategy(row):
        if pd.notna(row.get('noise_strategy')):
            return row['noise_strategy']  # Already set (e.g., phase0 legacy)
        
        filename = row['source_file']
        name = filename.replace('.csv', '')
        parts = name.split('_')
        
        # phase5_prefix_noisestrategy or phase5_prefix_noise_strategy
        if len(parts) >= 3:
            return '_'.join(parts[2:])
        return None
    
    df['noise_strategy'] = df.apply(extract_noise_strategy, axis=1)
    
    print(f"Noise strategies: {sorted(df['noise_strategy'].dropna().unique())}")
    print(f"Models: {sorted(df['model'].dropna().unique())}")
    print(f"Representations: {sorted(df['representation'].dropna().unique())}")
    
    # Drop rows with missing required columns
    required_cols = ['model', 'representation', 'noise_strategy', 'sigma', 'r2']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}")
        return pd.DataFrame()
    
    before = len(df)
    df = df.dropna(subset=required_cols)
    after = len(df)
    if before != after:
        print(f"Dropped {before - after} rows with missing values")
    
    # Filter out rows with invalid R² values (failed runs)
    before = len(df)
    df = df[(df['r2'] >= -2) & (df['r2'] <= 1.01)]
    after = len(df)
    if before != after:
        print(f"Filtered out {before - after} rows with invalid R² values (outside [-2, 1])")
    
    # Filter out GNN models, conformal methods, and graph representations
    exclude_model_patterns = ['gin', 'gcn', 'conformal', 'graph']
    exclude_reps = ['graph', 'mhggnn']
    
    before = len(df)
    model_mask = ~df['model'].str.lower().str.contains('|'.join(exclude_model_patterns), regex=True)
    rep_mask = ~df['representation'].str.lower().isin(exclude_reps)
    df = df[model_mask & rep_mask]
    after = len(df)
    if before != after:
        print(f"Filtered out {before - after} rows with GNN/conformal/graph models")
    
    return df


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_robustness_metrics(df, sigma_high=0.5):
    """
    Calculate robustness metrics for each model/representation/noise_strategy.
    
    Returns DataFrame with:
    - baseline_r2: R² at σ=0
    - r2_high: R² at σ=sigma_high
    - retention_pct: (r2_high / baseline_r2) × 100
    - nsi: slope of R² vs σ (Noise Sensitivity Index)
    """
    print(f"\nCalculating robustness metrics (σ_high = {sigma_high})")
    
    if len(df) == 0:
        print("ERROR: No data to calculate metrics from!")
        return pd.DataFrame()
    
    if 'sigma' not in df.columns:
        print(f"ERROR: 'sigma' column not found. Available columns: {list(df.columns)}")
        return pd.DataFrame()
    
    available_sigmas = sorted(df['sigma'].dropna().unique())
    print(f"Available σ values: {available_sigmas}")
    
    if len(available_sigmas) == 0:
        print("ERROR: No valid sigma values found!")
        return pd.DataFrame()
    
    # Find closest sigma to requested high value
    if sigma_high not in available_sigmas:
        sigma_high = min(available_sigmas, key=lambda x: abs(x - sigma_high))
        print(f"Using closest available σ_high: {sigma_high}")
    
    metrics_list = []
    groups = df.groupby(['model', 'representation', 'noise_strategy'])
    
    for (model, rep, noise_strat), group in groups:
        if len(group) < 3:
            continue
        
        # Average across iterations at each sigma
        avg_by_sigma = group.groupby('sigma')['r2'].mean().reset_index()
        avg_by_sigma = avg_by_sigma.sort_values('sigma')
        
        metrics = {
            'model': model,
            'representation': rep,
            'noise_strategy': noise_strat,
        }
        
        # Baseline at σ=0
        baseline = avg_by_sigma[avg_by_sigma['sigma'] == 0.0]
        metrics['baseline_r2'] = baseline['r2'].values[0] if len(baseline) > 0 else np.nan
        
        # Performance at high noise
        high_noise = avg_by_sigma[avg_by_sigma['sigma'] == sigma_high]
        metrics['r2_high'] = high_noise['r2'].values[0] if len(high_noise) > 0 else np.nan
        
        # Retention percentage
        if pd.notna(metrics['baseline_r2']) and pd.notna(metrics['r2_high']):
            if metrics['baseline_r2'] > 0:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            else:
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # NSI (slope of R² vs σ)
        if len(avg_by_sigma) >= 3:
            slope, _, _, _, _ = stats.linregress(avg_by_sigma['sigma'], avg_by_sigma['r2'])
            metrics['nsi'] = slope
        else:
            metrics['nsi'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    print(f"Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


# ============================================================================
# FIGURE 8: NOISE STRATEGY GENERALIZATION
# ============================================================================

def create_figure8(df, metrics_df, output_dir):
    """
    Figure 8: Noise strategy generalization (3 panels)
    
    A: R² degradation curves by noise strategy
    B: Robustness ranking across strategies
    C: Model × Strategy performance heatmap
    """
    print("\n" + "=" * 80)
    print("GENERATING FIGURE 8: NOISE STRATEGY GENERALIZATION")
    print("=" * 80)
    
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.8, 1.2],
                          left=0.06, right=0.98, top=0.88, bottom=0.18,
                          wspace=0.35)
    
    # ========================================================================
    # PANEL A: R² degradation by noise strategy
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    strategies = df['noise_strategy'].unique()
    
    for strategy in sorted(strategies):
        strategy_data = df[df['noise_strategy'] == strategy]
        avg_curve = strategy_data.groupby('sigma')['r2'].mean().reset_index()
        
        if len(avg_curve) > 2:
            color = get_color(strategy, NOISE_STRATEGY_COLORS)
            label = get_clean_label(strategy, NOISE_STRATEGY_LABELS)
            
            ax_a.plot(avg_curve['sigma'], avg_curve['r2'],
                     marker='o', linewidth=2, markersize=5,
                     color=color, label=label, alpha=0.9)
    
    ax_a.set_xlabel('Noise level (σ)')
    ax_a.set_ylabel('Mean R² (across all models)')
    ax_a.set_title('A. Degradation by noise strategy', fontweight='bold', pad=10)
    ax_a.legend(fontsize=7, loc='lower left', frameon=True, framealpha=0.95)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_a.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL B: Robustness ranking by strategy
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    strategy_stats = metrics_df.groupby('noise_strategy').agg({
        'retention_pct': ['mean', 'std']
    }).reset_index()
    strategy_stats.columns = ['noise_strategy', 'retention_mean', 'retention_std']
    strategy_stats = strategy_stats.sort_values('retention_mean', ascending=True)
    
    y_pos = np.arange(len(strategy_stats))
    colors = [get_color(s, NOISE_STRATEGY_COLORS) for s in strategy_stats['noise_strategy']]
    labels = [get_clean_label(s, NOISE_STRATEGY_LABELS) for s in strategy_stats['noise_strategy']]
    
    ax_b.barh(y_pos, strategy_stats['retention_mean'],
             xerr=strategy_stats['retention_std'],
             color=colors, alpha=0.85, height=0.65,
             edgecolor='black', linewidth=0.5, capsize=3)
    
    ax_b.axvline(100, color='#666666', linestyle='--', linewidth=1, alpha=0.6)
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(labels, fontsize=7)
    ax_b.set_xlabel('Retention % (±std)')
    ax_b.set_title('B. Strategy difficulty ranking', fontweight='bold', pad=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    ax_b.set_xlim(0, max(110, strategy_stats['retention_mean'].max() + 10))
    
    # ========================================================================
    # PANEL C: Model × Strategy heatmap
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Pivot: rows = model/rep, columns = noise strategy
    pivot_data = metrics_df.pivot_table(
        index=['model', 'representation'],
        columns='noise_strategy',
        values='retention_pct',
        aggfunc='mean'
    )
    
    # Sort by mean retention across strategies
    pivot_data['_mean'] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values('_mean', ascending=False).drop('_mean', axis=1)
    
    # Take top 12 configurations
    pivot_data = pivot_data.head(12)
    
    # Clean labels
    row_labels = [f"{get_clean_label(m, MODEL_LABELS)}/{get_clean_label(r, REP_LABELS)}" 
                  for m, r in pivot_data.index]
    col_labels = [get_clean_label(s, NOISE_STRATEGY_LABELS) for s in pivot_data.columns]
    
    im = ax_c.imshow(pivot_data.values, aspect='auto', cmap='RdYlGn',
                    vmin=0, vmax=100, interpolation='nearest')
    
    ax_c.set_xticks(np.arange(len(col_labels)))
    ax_c.set_yticks(np.arange(len(row_labels)))
    ax_c.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=7)
    ax_c.set_yticklabels(row_labels, fontsize=7)
    
    # Text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = pivot_data.iloc[i, j]
            if pd.notna(val):
                color = 'white' if val < 50 else 'black'
                ax_c.text(j, i, f'{val:.0f}', ha='center', va='center',
                         color=color, fontsize=6, fontweight='medium')
    
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label('Retention (%)', fontsize=8, rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=7)
    
    ax_c.set_title('C. Model performance by strategy', fontweight='bold', pad=10)
    ax_c.set_xlabel('Noise strategy')
    ax_c.set_ylabel('Model / Representation')
    
    # Save
    output_path = Path(output_dir) / "figure8_noise_strategy_generalization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S8: RANKING CONCORDANCE
# ============================================================================

def create_supplementary_s8(metrics_df, output_dir):
    """
    Supplementary S8: Ranking concordance across noise strategies
    
    Shows Spearman correlation of model rankings between different strategies
    """
    print("\n" + "=" * 80)
    print("GENERATING SUPPLEMENTARY S8: RANKING CONCORDANCE")
    print("=" * 80)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Pivot to get retention by model/rep for each strategy
    pivot_data = metrics_df.pivot_table(
        index=['model', 'representation'],
        columns='noise_strategy',
        values='retention_pct',
        aggfunc='mean'
    )
    
    strategies = list(pivot_data.columns)
    n = len(strategies)
    
    if n < 2:
        ax.text(0.5, 0.5, 'Insufficient strategies for concordance analysis',
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        ax.axis('off')
    else:
        # Compute Spearman correlation matrix
        corr_matrix = np.ones((n, n))
        
        for i, strat1 in enumerate(strategies):
            for j, strat2 in enumerate(strategies):
                if i != j:
                    data1 = pivot_data[strat1].dropna()
                    data2 = pivot_data[strat2].dropna()
                    common = data1.index.intersection(data2.index)
                    
                    if len(common) >= 3:
                        corr, _ = spearmanr(data1[common], data2[common])
                        corr_matrix[i, j] = corr
        
        # Plot
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1,
                      aspect='equal', interpolation='nearest')
        
        labels = [get_clean_label(s, NOISE_STRATEGY_LABELS) for s in strategies]
        
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        
        # Annotations
        for i in range(n):
            for j in range(n):
                val = corr_matrix[i, j]
                color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                       color=color, fontsize=9, fontweight='medium')
        
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Spearman ρ', fontsize=10, rotation=270, labelpad=15)
        
        ax.set_title('Ranking concordance across noise strategies\n(Spearman correlation of model rankings)',
                    fontsize=11, fontweight='bold', pad=12)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "supplementary_s8_ranking_concordance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Generate summary tables for Phase 5"""
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY TABLES")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Robustness by noise strategy
    table1 = metrics_df.groupby('noise_strategy').agg({
        'baseline_r2': 'mean',
        'retention_pct': ['mean', 'std'],
        'nsi': lambda x: np.abs(x).mean()
    }).reset_index()
    
    table1.columns = ['Noise Strategy', 'Mean Baseline R²', 'Mean Retention %', 
                      'Std Retention %', 'Mean |NSI|']
    table1['Noise Strategy'] = table1['Noise Strategy'].apply(
        lambda x: get_clean_label(x, NOISE_STRATEGY_LABELS))
    table1 = table1.sort_values('Mean Retention %', ascending=False)
    
    table1.to_csv(output_dir / "table_phase5_by_noise_strategy.csv", index=False)
    print(f"✓ Saved: table_phase5_by_noise_strategy.csv")
    
    # Table 2: Top configurations
    table2 = metrics_df.groupby(['model', 'representation']).agg({
        'retention_pct': 'mean',
        'nsi': lambda x: np.abs(x).mean(),
        'baseline_r2': 'mean'
    }).reset_index()
    
    table2['Model'] = table2['model'].apply(lambda x: get_clean_label(x, MODEL_LABELS))
    table2['Representation'] = table2['representation'].apply(lambda x: get_clean_label(x, REP_LABELS))
    table2 = table2[['Model', 'Representation', 'retention_pct', 'nsi', 'baseline_r2']]
    table2.columns = ['Model', 'Representation', 'Mean Retention %', 'Mean |NSI|', 'Mean Baseline R²']
    table2 = table2.sort_values('Mean Retention %', ascending=False).head(15)
    
    table2.to_csv(output_dir / "table_phase5_top_configs.csv", index=False)
    print(f"✓ Saved: table_phase5_top_configs.csv")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution"""
    print("=" * 80)
    print("PHASE 5 ANALYSIS: NOISE STRATEGY GENERALIZATION")
    print("=" * 80)
    
    # Load and parse data
    df = load_phase5_results(results_dir)
    df = parse_phase5_filenames(df)
    
    if len(df) == 0:
        print("\nERROR: No valid data after parsing. Check file format.")
        return
    
    print(f"\nExperiment summary:")
    print(f"  Total rows: {len(df)}")
    print(f"  Models: {len(df['model'].unique())} unique")
    print(f"  Representations: {', '.join(sorted(df['representation'].unique()))}")
    print(f"  Noise strategies: {', '.join(sorted(df['noise_strategy'].unique()))}")
    print(f"  σ range: {df['sigma'].min()} - {df['sigma'].max()}")
    print(f"  R² range: {df['r2'].min():.3f} - {df['r2'].max():.3f}")
    
    # Show counts per strategy
    print(f"\n  Rows per strategy:")
    for strat, count in df.groupby('noise_strategy').size().items():
        print(f"    {strat}: {count}")
    
    # Calculate metrics
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.5)
    
    if len(metrics_df) == 0:
        print("\nERROR: No metrics could be calculated. Check data format.")
        return
    
    # Create output directory
    output_dir = Path(results_dir) / "phase5_figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase5_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics: {output_dir / 'phase5_robustness_metrics.csv'}")
    
    # Generate outputs
    create_figure8(df, metrics_df, output_dir)
    create_supplementary_s8(metrics_df, output_dir)
    create_summary_tables(metrics_df, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    
    print("\nNoise strategy difficulty (hardest first):")
    difficulty = metrics_df.groupby('noise_strategy')['retention_pct'].mean().sort_values()
    for i, (strategy, retention) in enumerate(difficulty.items(), 1):
        label = get_clean_label(strategy, NOISE_STRATEGY_LABELS)
        print(f"  {i}. {label}: {retention:.1f}% retention")
    
    print("\nMost robust configurations:")
    best = metrics_df.groupby(['model', 'representation'])['retention_pct'].mean().sort_values(ascending=False)
    for i, ((model, rep), retention) in enumerate(best.head(5).items(), 1):
        m_label = get_clean_label(model, MODEL_LABELS)
        r_label = get_clean_label(rep, REP_LABELS)
        print(f"  {i}. {m_label}/{r_label}: {retention:.1f}% retention")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)