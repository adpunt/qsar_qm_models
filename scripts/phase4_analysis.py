"""
Phase 4 Analysis - Alternative Noise Strategies
Generates Figure 8 and S8

Based on the original working script with these changes:
- Retention metric REMOVED
- NSI renamed to Noise Degradation Slope (NDS) with threshold
- Target field parsed but IGNORED in analysis (aggregated across targets)
- "Supplementary" removed from figure names

Key metrics:
- NDS (Noise Degradation Slope): slope of R² vs σ (negative = degradation)
- Baseline R² at σ=0
- NDS is thresholded: only calculated for configs with baseline R² > 0.6
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
    'heteroscedastic': '#e74c3c',
    'legacy': '#2ecc71',
    'valprop': '#9b59b6',
    'quantile': '#f39c12',
    'outlier': '#e67e22',
    'threshold': '#7f8c8d',
    'uniform': '#1abc9c',
    'laplace': '#34495e',
}

MODEL_COLORS = {
    'rf': '#3498db',
    'qrf': '#16a085',
    'xgboost': '#e74c3c',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'mlp': '#2c3e50',
    'gauche': '#9b59b6',
}

REPRESENTATION_COLORS = {
    'pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'smiles': '#CA3542',
    'mhggnn': '#CC79A7',
}


def format_model(model):
    """Format model names for display"""
    if model is None:
        return 'Unknown'
    mapping = {
        'rf': 'RF',
        'qrf': 'QRF',
        'xgboost': 'XGBoost',
        'ngboost': 'NGBoost',
        'dnn': 'DNN',
        'mlp': 'MLP',
        'gauche': 'GP',
    }
    return mapping.get(model.lower(), model.upper())


def format_representation(rep):
    """Format representation names for display"""
    if rep is None:
        return 'Unknown'
    mapping = {
        'pdv': 'PDV',
        'sns': 'SNS',
        'ecfp4': 'ECFP4',
        'smiles': 'SMILES',
        'mhggnn': 'MHGGNN',
    }
    return mapping.get(rep.lower(), rep.upper())


def format_noise_strategy(strategy):
    """Format noise strategy names for display"""
    if strategy is None:
        return 'Unknown'
    mapping = {
        'gaussian': 'Gaussian',
        'hetero': 'Heteroscedastic',
        'heteroscedastic': 'Heteroscedastic',
        'legacy': 'Legacy',
        'valprop': 'Value-Proportional',
        'quantile': 'Quantile',
        'outlier': 'Outlier',
        'uniform': 'Uniform',
        'laplace': 'Laplace',
        'threshold': 'Threshold',
    }
    return mapping.get(strategy.lower(), strategy.capitalize())


# ============================================================================
# DATA LOADING AND PARSING - ORIGINAL WORKING VERSION
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
    for filepath in sorted(phase4_files):
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
    
    ORIGINAL WORKING VERSION - positional parsing
    """
    
    def extract_info(row):
        filename = row['source_file']
        
        # Remove .csv and split
        name = filename.replace('.csv', '')
        parts = name.split('_')
        
        # phase4X_target_rep_model_noisestrategy
        if len(parts) >= 5:
            subphase = parts[0]        # phase4a, phase4b, etc.
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

def calculate_robustness_metrics(df, baseline_threshold=0.6):
    """
    Calculate robustness metrics for each model/rep/noise_strategy
    (aggregated across targets)
    
    Metrics:
    - baseline_r2: R² at σ=0
    - nds_r2: Noise Degradation Slope (slope of R² vs σ)
    - nds_thresholded: NDS only for configs with baseline R² > threshold
    """
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS")
    print(f"  Baseline threshold for NDS: R² > {baseline_threshold}")
    print("="*80)
    
    # Check available sigma values
    available_sigmas = sorted(df['sigma'].unique())
    print(f"Available sigma values: {available_sigmas}")
    
    # Group by model/rep/noise_strategy (aggregate across targets)
    groups = df.groupby(['model', 'representation', 'noise_strategy'])
    print(f"Number of groups found: {len(groups)}")
    
    if len(groups) == 0:
        print("⚠️  No groups found!")
        return pd.DataFrame()
    
    metrics_list = []
    
    for (model, rep, noise_strat), group in groups:
        if model is None or rep is None or noise_strat is None:
            continue
            
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        # Average across iterations AND targets
        group_avg = group.groupby('sigma').agg({
            'r2': 'mean',
            'rmse': 'mean',
            'mae': 'mean'
        }).reset_index()
        
        metrics = {
            'model': model,
            'representation': rep,
            'noise_strategy': noise_strat,
        }
        
        # Baseline at σ=0
        sigma_0 = group_avg[group_avg['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # High noise performance (σ=0.5 or closest)
        sigma_high = 0.5
        sigma_h = group_avg[np.abs(group_avg['sigma'] - sigma_high) < 0.1]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
        else:
            metrics['r2_high'] = np.nan
        
        # Calculate NDS (Noise Degradation Slope) for ALL configs
        if len(group_avg) >= 3:
            try:
                slope_r2, intercept, r_val, p_val, _ = stats.linregress(
                    group_avg['sigma'], group_avg['r2']
                )
                metrics['nds_r2'] = slope_r2
                metrics['nds_r2_pval'] = p_val
                metrics['nds_r2_r'] = r_val
            except:
                metrics['nds_r2'] = np.nan
        else:
            metrics['nds_r2'] = np.nan
        
        # Thresholded NDS: only valid if baseline R² > threshold
        if not np.isnan(metrics.get('baseline_r2', np.nan)) and metrics['baseline_r2'] > baseline_threshold:
            metrics['nds_thresholded'] = metrics['nds_r2']
            metrics['meets_baseline_threshold'] = True
        else:
            metrics['nds_thresholded'] = np.nan
            metrics['meets_baseline_threshold'] = False
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Summary
    n_total = len(metrics_df)
    n_meets_threshold = metrics_df['meets_baseline_threshold'].sum()
    
    print(f"\nCalculated metrics for {n_total} configurations")
    print(f"  Configs meeting baseline threshold (R² > {baseline_threshold}): {n_meets_threshold}")
    print(f"  Configs below threshold: {n_total - n_meets_threshold}")
    
    return metrics_df


# ============================================================================
# FIGURE 8: GENERALIZATION ACROSS NOISE STRATEGIES
# ============================================================================

def create_figure8_generalization(df, metrics_df, output_dir):
    """
    Figure 8: Generalisation across noise strategies
    
    Panel A: R² degradation curves by noise strategy
    Panel B: Noise Degradation Slope ranking by strategy
    Panel C: Model robustness across strategies (heatmap)
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 8: NOISE STRATEGY GENERALIZATION")
    print("="*80)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    if len(metrics_thresh) == 0:
        print("⚠️  No configurations meet the baseline threshold!")
        return
    
    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: R² degradation curves by noise strategy
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Get noise strategies
    noise_strategies = sorted(df['noise_strategy'].dropna().unique())
    
    for noise_strat in noise_strategies:
        strategy_data = df[df['noise_strategy'] == noise_strat]
        
        # Average across all models/reps/targets
        avg_by_sigma = strategy_data.groupby('sigma')['r2'].mean().reset_index()
        
        if len(avg_by_sigma) > 2:
            color = NOISE_STRATEGY_COLORS.get(noise_strat, '#999999')
            ax_a.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker='o', linewidth=2, markersize=5, alpha=0.9,
                     label=format_noise_strategy(noise_strat), color=color)
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=9)
    ax_a.set_ylabel('Average R² (all configs)', fontsize=9)
    ax_a.set_title('A. R² Degradation by Noise Strategy', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=6.5, loc='lower left', ncol=1, frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_a.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL B: Noise Degradation Slope ranking by strategy
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # For each noise strategy, show average NDS across configs
    strategy_robustness = metrics_thresh.groupby('noise_strategy').agg({
        'nds_thresholded': ['mean', 'std'],
        'baseline_r2': 'mean'
    }).reset_index()
    
    strategy_robustness.columns = ['noise_strategy', 'nds_mean', 'nds_std', 'baseline_mean']
    strategy_robustness['abs_nds'] = strategy_robustness['nds_mean'].abs()
    strategy_robustness = strategy_robustness.sort_values('abs_nds', ascending=True)
    
    y_pos = np.arange(len(strategy_robustness))
    colors = [NOISE_STRATEGY_COLORS.get(s, '#999999') for s in strategy_robustness['noise_strategy']]
    
    # Plot NDS (negative values, so more negative = worse)
    ax_b.barh(y_pos, strategy_robustness['nds_mean'], 
             xerr=strategy_robustness['nds_std'].fillna(0),
             color=colors, alpha=0.8, height=0.7,
             edgecolor='black', linewidth=0.5, capsize=3)
    
    ax_b.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([format_noise_strategy(s) for s in strategy_robustness['noise_strategy']], fontsize=7)
    ax_b.set_xlabel('Mean Noise Degradation Slope (±std)', fontsize=9)
    ax_b.set_title('B. Noise Degradation Slope by Strategy\n(closer to 0 = more stable)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL C: Model robustness across strategies (heatmap)
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Create heatmap: rows = model/rep configs, columns = noise strategies
    # Value = NDS (thresholded)
    
    # Get unique model/rep combinations
    configs = metrics_thresh.groupby(['model', 'representation']).agg({
        'nds_thresholded': lambda x: x.abs().mean()
    }).reset_index()
    configs.columns = ['model', 'representation', 'mean_abs_nds']
    configs = configs.nsmallest(10, 'mean_abs_nds')  # Top 10 most robust
    
    # Create pivot table
    heatmap_data = []
    for _, row in configs.iterrows():
        model, rep = row['model'], row['representation']
        config_label = f"{format_model(model)}/{format_representation(rep)}"
        
        strategy_performance = {}
        for strategy in metrics_thresh['noise_strategy'].unique():
            strat_data = metrics_thresh[
                (metrics_thresh['model'] == model) &
                (metrics_thresh['representation'] == rep) &
                (metrics_thresh['noise_strategy'] == strategy)
            ]
            
            if len(strat_data) > 0:
                strategy_performance[strategy] = strat_data['nds_thresholded'].mean()
            else:
                strategy_performance[strategy] = np.nan
        
        strategy_performance['config'] = config_label
        heatmap_data.append(strategy_performance)
    
    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data)
        heatmap_df = heatmap_df.set_index('config')
        
        # Rename columns for display
        heatmap_df.columns = [format_noise_strategy(c) for c in heatmap_df.columns]
        
        # Determine color scale (NDS is negative, so more negative = red)
        vmin = heatmap_df.min().min()
        vmax = min(0, heatmap_df.max().max())  # Cap at 0
        
        # Plot heatmap
        im = ax_c.imshow(heatmap_df.values, aspect='auto', cmap='RdYlGn',
                        vmin=vmin, vmax=vmax, interpolation='nearest')
        
        ax_c.set_xticks(np.arange(len(heatmap_df.columns)))
        ax_c.set_yticks(np.arange(len(heatmap_df.index)))
        ax_c.set_xticklabels(heatmap_df.columns, rotation=45, ha='right', fontsize=7)
        ax_c.set_yticklabels(heatmap_df.index, fontsize=7)
        
        # Add text annotations
        for i in range(len(heatmap_df.index)):
            for j in range(len(heatmap_df.columns)):
                value = heatmap_df.iloc[i, j]
                if not np.isnan(value):
                    text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                    ax_c.text(j, i, f'{value:.2f}', ha='center', va='center',
                            color=text_color, fontsize=6)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
        cbar.set_label('Noise Degradation Slope', fontsize=8, rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=7)
        
        ax_c.set_title('C. Model Robustness Across Strategies\n(Top 10 configs, closer to 0 = better)', 
                      fontsize=10, fontweight='bold', pad=10)
        ax_c.set_xlabel('Noise Strategy', fontsize=9)
        ax_c.set_ylabel('Model/Representation', fontsize=9)
    else:
        ax_c.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                 transform=ax_c.transAxes, fontsize=10)
        ax_c.axis('off')
    
    # ========================================================================
    # Save
    # ========================================================================
    
    output_path = Path(output_dir) / "figure8_noise_strategy_generalization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 8 to {output_path}")
    plt.close()


# ============================================================================
# S8: RANKING CONCORDANCE
# ============================================================================

def create_s8_ranking_concordance(df, metrics_df, output_dir):
    """
    S8: Ranking concordance plots
    
    Panel A: Spearman correlation matrix between noise strategies
    Panel B: Scatter plot of rankings between most/least difficult strategies
    """
    print("\n" + "="*80)
    print("GENERATING S8: RANKING CONCORDANCE")
    print("="*80)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    if len(metrics_thresh) == 0:
        print("⚠️  No configurations meet the baseline threshold!")
        return
    
    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2, hspace=0.25, wspace=0.30,
                          left=0.08, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: Concordance matrix across noise strategies
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    strategies = sorted(metrics_thresh['noise_strategy'].dropna().unique())
    
    if len(strategies) >= 2:
        # Pivot to get NDS by model/rep for each strategy
        pivot_data = metrics_thresh.pivot_table(
            index=['model', 'representation'],
            columns='noise_strategy',
            values='nds_thresholded',
            aggfunc='mean'
        )
        
        # Need at least 3 configs per strategy for meaningful correlation
        valid_strategies = [s for s in strategies if s in pivot_data.columns and pivot_data[s].notna().sum() >= 3]
        
        if len(valid_strategies) >= 2:
            # Calculate Spearman rank correlations
            n_strategies = len(valid_strategies)
            corr_matrix = np.ones((n_strategies, n_strategies))
            
            for i, strat1 in enumerate(valid_strategies):
                for j, strat2 in enumerate(valid_strategies):
                    if i != j:
                        data1 = pivot_data[strat1].dropna()
                        data2 = pivot_data[strat2].dropna()
                        common_idx = data1.index.intersection(data2.index)
                        
                        if len(common_idx) >= 3:
                            corr, _ = spearmanr(data1[common_idx], data2[common_idx])
                            corr_matrix[i, j] = corr
                        else:
                            corr_matrix[i, j] = np.nan
            
            # Plot heatmap
            im = ax_a.imshow(corr_matrix, cmap='RdYlGn', vmin=0, vmax=1,
                           aspect='auto', interpolation='nearest')
            
            ax_a.set_xticks(np.arange(n_strategies))
            ax_a.set_yticks(np.arange(n_strategies))
            ax_a.set_xticklabels([format_noise_strategy(s) for s in valid_strategies], rotation=45, ha='right', fontsize=7)
            ax_a.set_yticklabels([format_noise_strategy(s) for s in valid_strategies], fontsize=7)
            
            # Add text annotations
            for i in range(n_strategies):
                for j in range(n_strategies):
                    value = corr_matrix[i, j]
                    if not np.isnan(value):
                        text_color = 'white' if value < 0.5 else 'black'
                        ax_a.text(j, i, f'{value:.2f}', ha='center', va='center',
                                color=text_color, fontsize=7)
            
            # Colorbar
            cbar = plt.colorbar(im, ax=ax_a, fraction=0.046, pad=0.04)
            cbar.set_label('Spearman ρ', fontsize=8, rotation=270, labelpad=15)
            cbar.ax.tick_params(labelsize=7)
            
            ax_a.set_title('A. Strategy Ranking Concordance\n(Spearman correlation of Noise Degradation Slope rankings)', 
                          fontsize=10, fontweight='bold', pad=10)
        else:
            ax_a.text(0.5, 0.5, f'Need ≥2 strategies with ≥3 configs\nFound: {len(valid_strategies)}', 
                     ha='center', va='center', transform=ax_a.transAxes, fontsize=10)
            ax_a.axis('off')
    else:
        ax_a.text(0.5, 0.5, 'Need ≥2 strategies', ha='center', va='center',
                 transform=ax_a.transAxes, fontsize=10)
        ax_a.axis('off')
    
    # ========================================================================
    # PANEL B: Scatter plot comparing easiest vs hardest strategy
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    if len(strategies) >= 2 and 'pivot_data' in dir() and len(pivot_data) > 0:
        # Find easiest and hardest strategies by mean |NDS|
        strategy_difficulty = metrics_thresh.groupby('noise_strategy')['nds_thresholded'].apply(
            lambda x: x.abs().mean()
        ).sort_values()
        
        easiest = strategy_difficulty.index[0]
        hardest = strategy_difficulty.index[-1]
        
        # Get NDS values for both strategies
        if easiest in pivot_data.columns and hardest in pivot_data.columns:
            easy_data = pivot_data[easiest].dropna()
            hard_data = pivot_data[hardest].dropna()
            common_idx = easy_data.index.intersection(hard_data.index)
            
            if len(common_idx) >= 3:
                x = easy_data[common_idx].values
                y = hard_data[common_idx].values
                
                # Color by model
                colors = []
                for idx in common_idx:
                    model, rep = idx
                    color = MODEL_COLORS.get(model, '#999999')
                    colors.append(color)
                
                ax_b.scatter(x, y, c=colors, s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
                
                # Add diagonal line (perfect concordance)
                min_val = min(x.min(), y.min())
                max_val = max(x.max(), y.max())
                ax_b.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5)
                
                # Calculate correlation
                corr, pval = spearmanr(x, y)
                ax_b.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {pval:.4f}', 
                         transform=ax_b.transAxes, fontsize=8, va='top',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax_b.set_xlabel(f'Noise Degradation Slope ({format_noise_strategy(easiest)})', fontsize=9)
                ax_b.set_ylabel(f'Noise Degradation Slope ({format_noise_strategy(hardest)})', fontsize=9)
                ax_b.set_title(f'B. Easiest vs Hardest Strategy\n({format_noise_strategy(easiest)} vs {format_noise_strategy(hardest)})', 
                              fontsize=10, fontweight='bold', pad=10)
                
                # Legend for models
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor=MODEL_COLORS.get(m, '#999999'), 
                                        edgecolor='black', label=format_model(m))
                                  for m in sorted(set([idx[0] for idx in common_idx]))]
                if legend_elements:
                    ax_b.legend(handles=legend_elements, fontsize=7, loc='lower right', framealpha=0.9)
                
                ax_b.spines['top'].set_visible(False)
                ax_b.spines['right'].set_visible(False)
                ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            else:
                ax_b.text(0.5, 0.5, f'Insufficient common data\n({len(common_idx)} configs)', 
                         ha='center', va='center', transform=ax_b.transAxes, fontsize=10)
                ax_b.axis('off')
        else:
            ax_b.text(0.5, 0.5, 'Strategy data not found', ha='center', va='center',
                     transform=ax_b.transAxes, fontsize=10)
            ax_b.axis('off')
    else:
        ax_b.text(0.5, 0.5, 'Need ≥2 strategies with data', ha='center', va='center',
                 transform=ax_b.transAxes, fontsize=10)
        ax_b.axis('off')
    
    # ========================================================================
    # Save
    # ========================================================================
    
    output_path = Path(output_dir) / "s8_ranking_concordance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved S8 to {output_path}")
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
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    if len(metrics_thresh) == 0:
        print("⚠️  No thresholded data for tables")
        return
    
    # Table 1: Robustness by noise strategy
    table1 = metrics_thresh.groupby('noise_strategy').agg({
        'baseline_r2': 'mean',
        'nds_thresholded': ['mean', 'std'],
    }).reset_index()
    
    table1.columns = ['Noise Strategy', 'Mean Baseline R²', 'Mean Noise Degradation Slope', 'Std Noise Degradation Slope']
    table1['|Mean NDS|'] = table1['Mean Noise Degradation Slope'].abs()
    table1 = table1.sort_values('|Mean NDS|', ascending=True)
    table1 = table1.round(4)
    
    table1.to_csv(output_dir / "table_phase4_by_noise_strategy.csv", index=False)
    print(f"✓ Saved noise strategy table")
    
    # Table 2: Top configurations across all strategies
    table2 = metrics_thresh.groupby(['model', 'representation']).agg({
        'nds_thresholded': ['mean', 'std'],
        'baseline_r2': 'mean'
    }).reset_index()
    
    table2.columns = ['Model', 'Representation', 'Mean Noise Degradation Slope', 'Std Noise Degradation Slope', 'Mean Baseline R²']
    table2['|Mean NDS|'] = table2['Mean Noise Degradation Slope'].abs()
    table2 = table2.sort_values('|Mean NDS|', ascending=True).head(20)
    table2 = table2.round(4)
    
    table2.to_csv(output_dir / "table_phase4_top_configs.csv", index=False)
    print(f"✓ Saved top configurations table")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 4 ANALYSIS - NOISE STRATEGY GENERALIZATION")
    print("="*80)
    print("\nKey changes in this version:")
    print("  - Retention metric REMOVED")
    print("  - NSI renamed to Noise Degradation Slope (NDS)")
    print("  - NDS thresholded: only calculated for baseline R² > 0.6")
    print("  - Target field parsed but aggregated across (not used for grouping)")
    print("  - Using ORIGINAL parsing logic")
    print("="*80)
    
    # Load data
    df = load_phase4_results(results_dir)
    if len(df) == 0:
        print("ERROR: No Phase 4 data loaded!")
        return
    
    # Parse info - ORIGINAL VERSION
    df = parse_phase4_info(df)
    
    print(f"\nExperiments found:")
    print(f"  Targets: {sorted(df['target'].dropna().unique())}")
    print(f"  Models: {sorted(df['model'].dropna().unique())}")
    print(f"  Representations: {sorted(df['representation'].dropna().unique())}")
    print(f"  Noise strategies: {sorted(df['noise_strategy'].dropna().unique())}")
    
    # Calculate metrics (aggregated across targets)
    metrics_df = calculate_robustness_metrics(df, baseline_threshold=0.6)
    
    if len(metrics_df) == 0:
        print("ERROR: No metrics calculated!")
        return
    
    # Create output directory
    output_dir = Path(results_dir) / "phase4_figures_v3"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase4_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase4_robustness_metrics.csv'}")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES (using thresholded NDS)")
    print("="*80)
    
    create_figure8_generalization(df, metrics_df, output_dir)
    create_s8_ranking_concordance(df, metrics_df, output_dir)
    
    # Generate tables
    create_summary_tables(metrics_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True]
    
    print(f"\nTotal configs: {len(metrics_df)}")
    print(f"Configs meeting threshold (R² > 0.6): {len(metrics_thresh)}")
    
    if len(metrics_thresh) > 0:
        # Noise strategy rankings
        print("\nNoise Strategy Rankings (by |Noise Degradation Slope|, lower = more stable):")
        strategy_ranking = metrics_thresh.groupby('noise_strategy')['nds_thresholded'].apply(
            lambda x: x.abs().mean()
        ).sort_values()
        for idx, (strategy, val) in enumerate(strategy_ranking.items(), 1):
            print(f"  {idx}. {format_noise_strategy(strategy)}: |NDS|={val:.4f}")
        
        # Best models
        print("\nMost Robust Configurations (across all strategies):")
        best = metrics_thresh.groupby(['model', 'representation'])['nds_thresholded'].apply(
            lambda x: x.abs().mean()
        ).sort_values()
        for idx, ((model, rep), val) in enumerate(best.head(5).items(), 1):
            print(f"  {idx}. {format_model(model)}/{format_representation(rep)}: |NDS|={val:.4f}")
    
    print("\n" + "="*80)
    print("PHASE 4 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)