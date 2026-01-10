"""
Phase 4 Analysis - Alternative Noise Strategies and Targets
Generates Figure 8 and Supplementary S8

Based on the detailed outline:
- Figure 8: Generalisation across noise strategies and properties (3 panels)
- Supplementary S8: Ranking concordance plots

Key metrics used (NO AUC as primary robustness metric):
- NSI (Noise Sensitivity Index): slope of R² vs σ
- Retention percentage: (R²_high / R²_baseline) * 100
- Baseline R² at σ=0

Heavily based on existing phase4_analysis.py generalization code
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

# Color palettes
NOISE_STRATEGY_COLORS = {
    'gaussian': '#3498db',
    'hetero': '#e74c3c',
    'legacy': '#2ecc71',
    'valprop': '#9b59b6',
    'quantile': '#f39c12',
    'outlier': '#e67e22',
}

MODEL_COLORS = {
    'rf': '#3498db',
    'qrf': '#16a085',
    'xgboost': '#e74c3c',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'gauche': '#9b59b6',
}

# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================

def load_phase4_results(results_dir="../results"):
    """
    Load Phase 4 generalization testing results
    
    Expected format: phase4X_target_rep_model_noisestrategy.csv
    Columns: sigma, iteration, r2, rmse, mae
    """
    print("\n" + "="*80)
    print("LOADING PHASE 4 GENERALIZATION DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Find all phase4 files
    phase4_files = list(results_dir.glob("phase4*.csv"))
    phase4_files = [f for f in phase4_files if '_uncertainty_values' not in f.name]
    
    if not phase4_files:
        print("ERROR: No phase4*.csv files found!")
        return pd.DataFrame()
    
    print(f"Found {len(phase4_files)} Phase 4 results files")
    
    all_results = []
    for filepath in phase4_files:
        try:
            df = pd.read_csv(filepath)
            df['source_file'] = filepath.name
            all_results.append(df)
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    print(f"\nLoaded {len(results_df)} result rows")
    
    return results_df


def parse_phase4_info(df):
    """
    Parse model, rep, noise strategy, target from filenames
    
    Format: phase4X_target_rep_model_noisestrategy.csv
    Example: phase4a_homolumo_pdv_rf_gaussian.csv
    """
    
    def extract_info(row):
        filename = row['source_file']
        
        # Remove .csv and split
        name = filename.replace('.csv', '')
        parts = name.split('_')
        
        # phase4X_target_rep_model_noisestrategy
        if len(parts) >= 5:
            subphase = parts[0]       # phase4a, phase4b, etc.
            target = parts[1]          # homolumo, alpha, etc.
            rep = parts[2]             # ecfp4, pdv, sns
            model = parts[3]           # rf, qrf, dnn
            noise_strategy = parts[4]  # gaussian, hetero, etc.
            
            return pd.Series({
                'subphase': subphase,
                'target': target,
                'representation': rep,
                'model': model,
                'noise_strategy': noise_strategy
            })
        
        return pd.Series({
            'subphase': None,
            'target': None,
            'representation': None,
            'model': None,
            'noise_strategy': None
        })
    
    info = df.apply(extract_info, axis=1)
    df[['subphase', 'target', 'representation', 'model', 'noise_strategy']] = info
    
    return df


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_robustness_metrics(df, sigma_high=0.5):
    """
    Calculate robustness metrics for each model/rep/noise_strategy/target
    
    Metrics:
    - baseline_r2: R² at σ=0
    - r2_high: R² at σ=sigma_high (or closest available)
    - retention_pct: (r2_high / baseline_r2) * 100
    - nsi_r2: slope of R² vs σ
    """
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS (σ_high = {sigma_high})")
    print("="*80)
    
    # Debug: Check what we're grouping by
    print(f"\nGrouping by: model, representation, noise_strategy, target")
    print(f"Total rows before grouping: {len(df)}")
    
    # Check available sigma values
    available_sigmas = sorted(df['sigma'].unique())
    print(f"Available sigma values: {available_sigmas}")
    
    # Find closest sigma to sigma_high
    if sigma_high not in available_sigmas:
        closest_sigma = min(available_sigmas, key=lambda x: abs(x - sigma_high))
        print(f"⚠️  Requested sigma_high={sigma_high} not found, using closest: {closest_sigma}")
        sigma_high = closest_sigma
    
    groups = df.groupby(['model', 'representation', 'noise_strategy', 'target'])
    print(f"Number of groups found: {len(groups)}")
    
    if len(groups) == 0:
        print("⚠️  No groups found! Checking for null values in grouping columns...")
        print(df[['model', 'representation', 'noise_strategy', 'target']].isnull().sum())
        return pd.DataFrame()
    
    # Show first few groups
    print(f"\nFirst 3 groups:")
    for i, ((model, rep, noise_strat, target), group) in enumerate(groups):
        if i >= 3:
            break
        print(f"  {i+1}. {model}/{rep}/{noise_strat}/{target}: {len(group)} rows, sigma range: {group['sigma'].min()}-{group['sigma'].max()}")
    
    metrics_list = []
    
    for (model, rep, noise_strat, target), group in groups:
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        # Average across iterations
        group_avg = group.groupby('sigma').agg({
            'r2': 'mean',
            'rmse': 'mean',
            'mae': 'mean'
        }).reset_index()
        
        metrics = {
            'model': model,
            'representation': rep,
            'noise_strategy': noise_strat,
            'target': target,
        }
        
        # Baseline at σ=0
        sigma_0 = group_avg[group_avg['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # Performance at high noise - use exact match or closest
        sigma_h = group_avg[group_avg['sigma'] == sigma_high]
        if len(sigma_h) == 0:
            # Try closest within 0.1
            sigma_h = group_avg[np.abs(group_avg['sigma'] - sigma_high) < 0.1]
        
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Retention
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            if metrics['baseline_r2'] != 0:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            else:
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # NSI
        if len(group_avg) >= 3:
            try:
                slope_r2, _, _, _, _ = stats.linregress(group_avg['sigma'], group_avg['r2'])
                metrics['nsi_r2'] = slope_r2
            except:
                metrics['nsi_r2'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    print(f"\nCalculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


# ============================================================================
# FIGURE 8: GENERALIZATION ACROSS NOISE STRATEGIES AND PROPERTIES
# ============================================================================

def create_figure8_generalization(df, metrics_df, output_dir):
    """
    Figure 8: Generalisation across noise strategies and properties
    
    Panel A: Noise strategy difficulty across models
    Panel B: Robustness ranking by strategy
    Panel C: Cross-property generalisation heatmap
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 8: GENERALIZATION")
    print("="*80)
    
    df = parse_phase4_info(df)
    
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: Noise strategy difficulty across models
    # ========================================================================
    
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Select top models from Phase 0-3 (you can adjust this list)
    available_models = df['model'].unique()
    available_reps = df['representation'].unique()
    
    # Use top 3-5 model-rep combinations
    top_configs = []
    for model in ['qrf', 'ngboost', 'gauche', 'rf', 'dnn']:
        for rep in ['pdv', 'sns']:
            if model in available_models and rep in available_reps:
                top_configs.append((model, rep))
                if len(top_configs) >= 5:
                    break
        if len(top_configs) >= 5:
            break
    
    # Categorize noise strategies as easy vs hard
    # Calculate average degradation for each strategy
    noise_difficulty = metrics_df.groupby('noise_strategy').agg({
        'retention_pct': 'mean',
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    noise_difficulty['difficulty'] = 100 - noise_difficulty['retention_pct']
    noise_difficulty = noise_difficulty.sort_values('difficulty', ascending=False)
    
    # Plot R² curves for each noise strategy (averaged across top models)
    for noise_strat in noise_difficulty['noise_strategy'].head(6):  # Top 6 strategies
        # Filter for this strategy and top models
        strategy_data = df[df['noise_strategy'] == noise_strat]
        
        # Average across top models
        avg_by_sigma = strategy_data.groupby('sigma')['r2'].mean().reset_index()
        
        if len(avg_by_sigma) > 2:
            color = NOISE_STRATEGY_COLORS.get(noise_strat, '#999999')
            ax_a.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker='o', linewidth=2, markersize=5, alpha=0.9,
                     label=noise_strat, color=color)
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=9)
    ax_a.set_ylabel('Average R² (top models)', fontsize=9)
    ax_a.set_title('A. Noise Strategy Difficulty', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=6.5, loc='best', ncol=1, frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL B: Robustness ranking by strategy
    # ========================================================================
    
    ax_b = fig.add_subplot(gs[0, 1])
    
    # For each noise strategy, show average retention across top models
    strategy_robustness = metrics_df.groupby('noise_strategy').agg({
        'retention_pct': ['mean', 'std'],
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    strategy_robustness.columns = ['noise_strategy', 'retention_mean', 'retention_std', 'abs_nsi']
    strategy_robustness = strategy_robustness.sort_values('retention_mean', ascending=True)
    
    y_pos = np.arange(len(strategy_robustness))
    colors = [NOISE_STRATEGY_COLORS.get(s, '#999999') for s in strategy_robustness['noise_strategy']]
    
    ax_b.barh(y_pos, strategy_robustness['retention_mean'], 
             xerr=strategy_robustness['retention_std'],
             color=colors, alpha=0.8, height=0.7,
             edgecolor='black', linewidth=0.5, capsize=3)
    
    ax_b.axvline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                label='No degradation')
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels(strategy_robustness['noise_strategy'], fontsize=7)
    ax_b.set_xlabel('Average Retention % (±std)', fontsize=9)
    ax_b.set_title('B. Robustness Ranking by Strategy', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    ax_b.set_xlim(0, 110)
    
    # ========================================================================
    # PANEL C: Cross-property generalisation heatmap
    # ========================================================================
    
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Create heatmap: rows = model/rep configs, columns = targets
    # Value = average retention_pct across noise strategies
    
    # Select top configurations
    top_n = 10
    overall_best = metrics_df.groupby(['model', 'representation']).agg({
        'retention_pct': 'mean'
    }).reset_index().nlargest(top_n, 'retention_pct')
    
    # Create pivot table
    heatmap_data = []
    for _, row in overall_best.iterrows():
        model, rep = row['model'], row['representation']
        config_label = f"{model}/{rep}"
        
        target_performance = {}
        for target in metrics_df['target'].unique():
            target_data = metrics_df[
                (metrics_df['model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['target'] == target)
            ]
            
            if len(target_data) > 0:
                target_performance[target] = target_data['retention_pct'].mean()
            else:
                target_performance[target] = np.nan
        
        target_performance['config'] = config_label
        heatmap_data.append(target_performance)
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('config')
        
        # Plot heatmap
        im = ax_c.imshow(heatmap_df.values, aspect='auto', cmap='RdYlGn',
                        vmin=0, vmax=100, interpolation='nearest')
        
        ax_c.set_xticks(np.arange(len(heatmap_df.columns)))
        ax_c.set_yticks(np.arange(len(heatmap_df.index)))
        ax_c.set_xticklabels(heatmap_df.columns, rotation=45, ha='right', fontsize=7)
        ax_c.set_yticklabels(heatmap_df.index, fontsize=7)
        
        # Add text annotations
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                value = heatmap_df.iloc[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < 50 else 'black'
                    ax_c.text(j, i, f'{value:.0f}', ha='center', va='center',
                            color=text_color, fontsize=6)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label('Retention (%)', fontsize=8, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=7)
        
        ax_c.set_title('C. Cross-Property Generalisation\n(Top 10 configs)', 
                      fontsize=10, fontweight='bold', pad=10)
        ax_c.set_xlabel('Target Property', fontsize=9)
        ax_c.set_ylabel('Model/Representation', fontsize=9)
    else:
        ax_c.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax_c.transAxes, fontsize=10)
        ax_c.axis('off')
    
    # ========================================================================
    # Save
    # ========================================================================
    
    output_path = Path(output_dir) / "figure8_generalization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 8 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S8: RANKING CONCORDANCE
# ============================================================================

def create_supplementary_s8(metrics_df, output_dir):
    """
    Supplementary S8: Ranking concordance plots
    
    Show Spearman rank correlation between robustness rankings 
    under different noise strategies and targets
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S8: RANKING CONCORDANCE")
    print("="*80)
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, hspace=0.25, wspace=0.30,
                          left=0.08, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: Concordance matrix across noise strategies
    # ========================================================================
    
    ax_a = fig.add_subplot(gs[0, 0])
    
    # For a fixed target, calculate rank correlation between strategies
    targets = metrics_df['target'].unique()
    main_target = targets[0] if len(targets) > 0 else None
    
    if main_target:
        target_data = metrics_df[metrics_df['target'] == main_target]
        
        # Create rankings for each noise strategy
        strategies = target_data['noise_strategy'].unique()
        
        if len(strategies) >= 2:
            # Pivot to get retention by model/rep for each strategy
            pivot_data = target_data.pivot_table(
                index=['model', 'representation'],
                columns='noise_strategy',
                values='retention_pct',
                aggfunc='mean'
            )
            
            # Calculate Spearman rank correlations
            n_strategies = len(strategies)
            corr_matrix = np.ones((n_strategies, n_strategies))
            
            for i, strat1 in enumerate(strategies):
                for j, strat2 in enumerate(strategies):
                    if i != j and strat1 in pivot_data.columns and strat2 in pivot_data.columns:
                        # Remove NaN pairs
                        data1 = pivot_data[strat1].dropna()
                        data2 = pivot_data[strat2].dropna()
                        common_idx = data1.index.intersection(data2.index)
                        
                        if len(common_idx) >= 3:
                            corr, _ = spearmanr(data1[common_idx], data2[common_idx])
                            corr_matrix[i, j] = corr
            
            # Plot heatmap
            im = ax_a.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest')
            
            ax_a.set_xticks(np.arange(n_strategies))
            ax_a.set_yticks(np.arange(n_strategies))
            ax_a.set_xticklabels(strategies, rotation=45, ha='right', fontsize=7)
            ax_a.set_yticklabels(strategies, fontsize=7)
            
            # Add text annotations
            for i in range(n_strategies):
                for j in range(n_strategies):
                    value = corr_matrix[i, j]
                    text_color = 'white' if value < 0.5 else 'black'
                    ax_a.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=6)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
            cbar.set_label('Spearman ρ', fontsize=8, rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=7)
            
            ax_a.set_title(f'A. Strategy Concordance ({main_target})\nSpearman rank correlation', 
                          fontsize=10, fontweight='bold', pad=10)
        else:
            ax_a.text(0.5, 0.5, 'Need ≥2 strategies', ha='center', va='center',
                     transform=ax_a.transAxes, fontsize=10)
            ax_a.axis('off')
    else:
        ax_a.text(0.5, 0.5, 'No target data', ha='center', va='center',
                 transform=ax_a.transAxes, fontsize=10)
        ax_a.axis('off')
    
    # ========================================================================
    # PANEL B: Concordance matrix across targets
    # ========================================================================
    
    ax_b = fig.add_subplot(gs[0, 1])
    
    # For a fixed noise strategy, calculate rank correlation between targets
    if len(targets) >= 2:
        # Use the most common noise strategy
        main_strategy = metrics_df['noise_strategy'].value_counts().index[0]
        
        strategy_data = metrics_df[metrics_df['noise_strategy'] == main_strategy]
        
        # Pivot to get retention by model/rep for each target
        pivot_data = strategy_data.pivot_table(
            index=['model', 'representation'],
            columns='target',
            values='retention_pct',
            aggfunc='mean'
        )
        
        # Calculate Spearman rank correlations
        n_targets = len(targets)
        corr_matrix = np.ones((n_targets, n_targets))
        
        for i, target1 in enumerate(targets):
            for j, target2 in enumerate(targets):
                if i != j and target1 in pivot_data.columns and target2 in pivot_data.columns:
                    data1 = pivot_data[target1].dropna()
                    data2 = pivot_data[target2].dropna()
                    common_idx = data1.index.intersection(data2.index)
                    
                    if len(common_idx) >= 3:
                        corr, _ = spearmanr(data1[common_idx], data2[common_idx])
                        corr_matrix[i, j] = corr
        
        # Plot heatmap
        im = ax_b.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1,
                        aspect='auto', interpolation='nearest')
        
        ax_b.set_xticks(np.arange(n_targets))
        ax_b.set_yticks(np.arange(n_targets))
        ax_b.set_xticklabels(targets, rotation=45, ha='right', fontsize=7)
        ax_b.set_yticklabels(targets, fontsize=7)
        
        # Add text annotations
        for i in range(n_targets):
            for j in range(n_targets):
                value = corr_matrix[i, j]
                text_color = 'white' if value < 0.5 else 'black'
                ax_b.text(j, i, f'{value:.2f}', ha='center', va='center',
                        color=text_color, fontsize=6)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
        cbar.set_label('Spearman ρ', fontsize=8, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=7)
        
        ax_b.set_title(f'B. Target Concordance ({main_strategy})\nSpearman rank correlation', 
                      fontsize=10, fontweight='bold', pad=10)
    else:
        ax_b.text(0.5, 0.5, 'Need ≥2 targets', ha='center', va='center',
                 transform=ax_b.transAxes, fontsize=10)
        ax_b.axis('off')
    
    # ========================================================================
    # Save
    # ========================================================================
    
    output_path = Path(output_dir) / "supplementary_s8_ranking_concordance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S8 to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Create summary tables for Phase 4"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Robustness by noise strategy
    table1 = metrics_df.groupby('noise_strategy').agg({
        'baseline_r2': 'mean',
        'retention_pct': ['mean', 'std'],
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    table1.columns = ['Noise Strategy', 'Mean Baseline R²', 'Mean Retention %', 'Std Retention %', 'Mean |NSI|']
    table1 = table1.sort_values('Mean Retention %', ascending=False)
    table1 = table1.round(4)
    
    table1.to_csv(output_dir / "table_phase4_by_noise_strategy.csv", index=False)
    
    with open(output_dir / "table_phase4_by_noise_strategy.tex", 'w') as f:
        f.write(table1.to_latex(index=False, float_format="%.4f"))
    
    print(f"✓ Saved noise strategy table")
    
    # Table 2: Robustness by target
    table2 = metrics_df.groupby('target').agg({
        'baseline_r2': 'mean',
        'retention_pct': ['mean', 'std'],
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    table2.columns = ['Target', 'Mean Baseline R²', 'Mean Retention %', 'Std Retention %', 'Mean |NSI|']
    table2 = table2.sort_values('Mean Retention %', ascending=False)
    table2 = table2.round(4)
    
    table2.to_csv(output_dir / "table_phase4_by_target.csv", index=False)
    
    with open(output_dir / "table_phase4_by_target.tex", 'w') as f:
        f.write(table2.to_latex(index=False, float_format="%.4f"))
    
    print(f"✓ Saved target table")
    
    # Table 3: Top configurations across all conditions
    table3 = metrics_df.groupby(['model', 'representation']).agg({
        'retention_pct': 'mean',
        'nsi_r2': lambda x: np.abs(x).mean(),
        'baseline_r2': 'mean'
    }).reset_index()
    
    table3.columns = ['Model', 'Representation', 'Mean Retention %', 'Mean |NSI|', 'Mean Baseline R²']
    table3 = table3.sort_values('Mean Retention %', ascending=False).head(20)
    table3 = table3.round(4)
    
    table3.to_csv(output_dir / "table_phase4_top_configs.csv", index=False)
    
    with open(output_dir / "table_phase4_top_configs.tex", 'w') as f:
        f.write(table3.to_latex(index=False, float_format="%.4f"))
    
    print(f"✓ Saved top configurations table")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 4 ANALYSIS - GENERALIZATION")
    print("Journal of Cheminformatics Style")
    print("="*80)
    
    # Load data
    df = load_phase4_results(results_dir)
    if len(df) == 0:
        print("ERROR: No Phase 4 data loaded!")
        return
    
    # Parse info
    df = parse_phase4_info(df)
    
    # Debug: Check what columns we have
    print(f"\nColumns after parsing: {df.columns.tolist()}")
    print(f"\nSample of parsed data:")
    print(df[['model', 'representation', 'noise_strategy', 'target', 'sigma']].head(10) if len(df) > 0 else "Empty dataframe")
    print(f"\nData shape: {df.shape}")
    print(f"\nNull counts:")
    print(df[['model', 'representation', 'noise_strategy', 'target']].isnull().sum())
    
    print(f"\nExperiments found:")
    print(f"  Targets: {', '.join(sorted(df['target'].dropna().unique()))}")
    print(f"  Models: {', '.join(sorted(df['model'].dropna().unique()))}")
    print(f"  Representations: {', '.join(sorted(df['representation'].dropna().unique()))}")
    print(f"  Noise strategies: {', '.join(sorted(df['noise_strategy'].dropna().unique()))}")
    
    # Calculate metrics
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.5)
    
    print(f"\n✓ Calculated metrics for {len(metrics_df)} configurations")
    
    # Debug: Check metrics
    if len(metrics_df) > 0:
        print(f"\nMetrics sample:")
        print(metrics_df[['model', 'representation', 'noise_strategy', 'target', 
                         'baseline_r2', 'r2_high', 'retention_pct', 'nsi_r2']].head())
        print(f"\nNull counts in metrics:")
        print(metrics_df[['retention_pct', 'nsi_r2']].isnull().sum())
    else:
        print("\n⚠️  WARNING: No metrics calculated! This means groupby found no valid groups.")
        print("    Check that model/representation/noise_strategy/target columns are populated.")
    
    # Create output directory
    output_dir = Path(results_dir) / "phase4_figures_v2"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase4_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase4_robustness_metrics.csv'}")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure8_generalization(df, metrics_df, output_dir)
    create_supplementary_s8(metrics_df, output_dir)
    
    # Generate tables
    create_summary_tables(metrics_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 4 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Figures:")
    print("    - figure8_generalization.png")
    print("    - supplementary_s8_ranking_concordance.png")
    print("  Tables:")
    print("    - table_phase4_by_noise_strategy.csv/.tex")
    print("    - table_phase4_by_target.csv/.tex")
    print("    - table_phase4_top_configs.csv/.tex")
    print("  Data:")
    print("    - phase4_robustness_metrics.csv")
    
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    # Hardest noise strategies
    print("\nHardest Noise Strategies:")
    hardest = metrics_df.groupby('noise_strategy')['retention_pct'].mean().sort_values()
    for idx, (strategy, retention) in enumerate(hardest.head(5).items(), 1):
        print(f"  {idx}. {strategy}: {retention:.1f}% average retention")
    
    # Best models overall
    print("\nMost Robust Configurations (across all conditions):")
    best = metrics_df.groupby(['model', 'representation'])['retention_pct'].mean().sort_values(ascending=False)
    for idx, ((model, rep), retention) in enumerate(best.head(5).items(), 1):
        print(f"  {idx}. {model}/{rep}: {retention:.1f}% average retention")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)