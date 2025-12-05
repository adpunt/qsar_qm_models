#!/usr/bin/env python3
"""
ENHANCED Comprehensive Noise Robustness Analysis
Implements all 12 research directions from the analysis plan:

1. METHOD CATEGORIZATION & COMPARISON
2. HET_GP INVESTIGATION
3. LOSS × METHOD INTERACTION
4. DISTANCE METRIC DRILL-DOWN
5. UNCERTAINTY METHOD VALIDATION
6. KERNEL ANALYSIS
7. TOP PERFORMER DEEP DIVE
8. VARIANCE DECOMPOSITION (ANOVA)
9. FAILURE ANALYSIS
10. PHASE 0C GAP ANALYSIS
11. CLUSTERING/PATTERN FINDING
12. NOISE LEVEL SENSITIVITY

Each analysis generates focused figures and statistical insights.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Suppress matplotlib font warnings
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# JoC style
sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 7,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'legend.frameon': False,
    'legend.fontsize': 6,
})

def load_comprehensive_results(results_dir):
    """Load all comprehensive noise study results"""
    all_data = []
    
    result_files = []
    result_files.extend(list(Path(results_dir).glob("meta_weight_net_*.csv")))
    result_files.extend(list(Path(results_dir).glob("dividemix_*.csv")))
    result_files.extend(list(Path(results_dir).glob("early_learning_*.csv")))
    result_files.extend(list(Path(results_dir).glob("mixup_*.csv")))
    result_files.extend(list(Path(results_dir).glob("sam_*.csv")))
    result_files.extend(list(Path(results_dir).glob("multistage_cleaning_*.csv")))
    result_files.extend(list(Path(results_dir).glob("uncertainty_curriculum_*.csv")))
    result_files.extend(list(Path(results_dir).glob("conformal_hetero*.csv")))
    result_files.extend(list(Path(results_dir).glob("confident_learning_*.csv")))
    result_files.extend(list(Path(results_dir).glob("small_loss_*.csv")))
    result_files.extend(list(Path(results_dir).glob("mentornet_*.csv")))
    result_files.extend(list(Path(results_dir).glob("contrast_divide_*.csv")))
    result_files.extend(list(Path(results_dir).glob("distance_select_*.csv")))
    result_files.extend(list(Path(results_dir).glob("het_gp_*.csv")))
    result_files.extend(list(Path(results_dir).glob("evidential_kernel_*.csv")))
    result_files.extend(list(Path(results_dir).glob("ntk_gnn*.csv")))
    
    result_files = list(set(result_files))
    
    if not result_files:
        print(f"No comprehensive noise study CSV files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(result_files)} comprehensive study result files")
    
    for filepath in result_files:
        try:
            df = pd.read_csv(filepath)
            required_cols = ['sigma', 'model', 'rep', 'r2', 'rmse']
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {filepath.name}: missing required columns")
                continue
            df['source_file'] = filepath.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Filter catastrophic failures
    initial_count = len(combined_df)
    combined_df = combined_df[combined_df['r2'] > -10]
    filtered_count = initial_count - len(combined_df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with catastrophic R² values (< -10)")
    
    # Filter to Phase 0C sigma values
    valid_sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combined_df = combined_df[combined_df['sigma'].isin(valid_sigmas)]
    print(f"Filtered to Phase 0C sigma values: {valid_sigmas}")
    
    # Aggregate
    results = combined_df.groupby(['model', 'rep', 'sigma', 'loss_function']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'rep': 'representation', 'iteration': 'n_iterations'}, inplace=True)
    
    print(f"\nLoaded {len(results)} unique model/representation/sigma/loss combinations")
    print(f"Models: {results['model'].nunique()}")
    print(f"Representations: {results['representation'].nunique()}")
    print(f"Loss functions: {results['loss_function'].nunique()}")
    
    return results

def parse_model_variants(df):
    """Extract model variants from model names"""
    
    def get_distance_metric(model_name):
        for dist in ['tanimoto', 'euclidean', 'mahalanobis']:
            if dist in model_name.lower():
                return dist
        return 'none'
    
    def get_kernel(model_name):
        for kernel in ['tanimoto', 'rbf', 'matern']:
            if kernel in model_name.lower():
                return kernel
        return 'none'
    
    def get_base_model(model_name):
        base = model_name.lower()
        for suffix in ['_mse', '_heteroscedastic', '_evidential', '_huber', '_cauchy',
                       '_tanimoto', '_euclidean', '_mahalanobis', '_rbf', '_matern']:
            base = base.replace(suffix, '')
        return base
    
    df['distance_metric'] = df['model'].apply(get_distance_metric)
    df['kernel'] = df['model'].apply(get_kernel)
    df['base_model'] = df['model'].apply(get_base_model)
    df['uses_distance'] = df['distance_metric'] != 'none'
    
    return df

def categorize_models(df):
    """Categorize models into research-relevant groups"""
    
    def get_group(base_model):
        if base_model in ['meta_weight_net', 'dividemix', 'early_learning']:
            return 'sample_reweighting'
        elif base_model in ['mixup']:
            return 'data_augmentation'
        elif base_model in ['sam']:
            return 'robust_optimization'
        elif base_model in ['small_loss', 'confident_learning', 'mentornet']:
            return 'sample_selection'
        elif base_model in ['multistage_cleaning', 'uncertainty_curriculum']:
            return 'cleaning_curriculum'
        elif base_model in ['distance_select', 'contrast_divide']:
            return 'distance_based'
        elif base_model in ['conformal_hetero']:
            return 'uncertainty_conformal'
        elif base_model in ['het_gp']:
            return 'uncertainty_heteroscedastic'
        elif base_model in ['evidential_kernel']:
            return 'uncertainty_evidential'
        elif base_model in ['ntk_gnn']:
            return 'graph_based'
        else:
            return 'other'
    
    df['model_group'] = df['base_model'].apply(get_group)
    
    return df

def calculate_robustness_metrics(df, baseline_threshold=0.1):
    """Calculate comprehensive robustness metrics"""
    robustness_metrics = []
    
    groupby_cols = ['model', 'representation', 'loss_function', 'distance_metric', 'kernel']
    
    for group_key, group in df.groupby(groupby_cols):
        model, rep, loss, dist, kernel = group_key
        
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) == 0:
            continue
        
        r2_baseline = sigma_0['r2'].values[0]
        
        if r2_baseline < baseline_threshold:
            continue
        
        metrics = {
            'model': model,
            'base_model': group['base_model'].iloc[0],
            'model_group': group['model_group'].iloc[0],
            'representation': rep,
            'loss_function': loss,
            'distance_metric': dist,
            'kernel': kernel,
            'uses_distance': group['uses_distance'].iloc[0],
            'n_sigma_values': len(group)
        }
        
        # Values at key sigmas
        for sigma_val in [0.0, 0.3, 0.6, 1.0]:
            sigma_data = group[group['sigma'] == sigma_val]
            metrics[f'r2_at_{sigma_val}'] = sigma_data['r2'].values[0] if len(sigma_data) > 0 else np.nan
            metrics[f'rmse_at_{sigma_val}'] = sigma_data['rmse'].values[0] if len(sigma_data) > 0 else np.nan
        
        # NSI (Noise Sensitivity Index)
        if len(group) >= 2:
            slope_r2, intercept_r2, r_val_r2, p_val_r2, _ = stats.linregress(group['sigma'], group['r2'])
            metrics['nsi_r2'] = slope_r2
            metrics['nsi_r2_intercept'] = intercept_r2
            metrics['nsi_r2_r'] = r_val_r2
            metrics['nsi_r2_pval'] = p_val_r2
            
            if metrics['r2_at_0.0'] != 0 and not np.isnan(metrics['r2_at_0.0']):
                metrics['nsi_r2_relative'] = slope_r2 / abs(metrics['r2_at_0.0'])
            else:
                metrics['nsi_r2_relative'] = np.nan
            
            slope_rmse, _, _, _, _ = stats.linregress(group['sigma'], group['rmse'])
            metrics['nsi_rmse'] = slope_rmse
        else:
            metrics['nsi_r2'] = np.nan
            metrics['nsi_r2_relative'] = np.nan
            metrics['nsi_rmse'] = np.nan
        
        # Performance retention
        if not np.isnan(metrics['r2_at_0.0']) and metrics['r2_at_0.0'] != 0:
            for sigma_val in [0.3, 0.6, 1.0]:
                if not np.isnan(metrics[f'r2_at_{sigma_val}']):
                    metrics[f'retention_{sigma_val}'] = (metrics[f'r2_at_{sigma_val}'] / metrics['r2_at_0.0']) * 100
                else:
                    metrics[f'retention_{sigma_val}'] = np.nan
        else:
            for sigma_val in [0.3, 0.6, 1.0]:
                metrics[f'retention_{sigma_val}'] = np.nan
        
        # Absolute drops
        if not np.isnan(metrics['r2_at_0.0']):
            metrics['r2_drop_0_to_0.6'] = metrics['r2_at_0.0'] - metrics['r2_at_0.6'] if not np.isnan(metrics['r2_at_0.6']) else np.nan
            metrics['r2_drop_0_to_1.0'] = metrics['r2_at_0.0'] - metrics['r2_at_1.0'] if not np.isnan(metrics['r2_at_1.0']) else np.nan
        
        robustness_metrics.append(metrics)
    
    return pd.DataFrame(robustness_metrics)


# ============================================================================
# ANALYSIS 1: METHOD CATEGORIZATION & COMPARISON
# ============================================================================

def analysis_1_method_categorization(robustness_df, output_dir):
    """Compare R² at 0.3 across method categories"""
    print("\n" + "="*80)
    print("ANALYSIS 1: METHOD CATEGORIZATION & COMPARISON")
    print("="*80)
    
    # Group by category and get summary stats
    category_summary = robustness_df.groupby('model_group').agg({
        'r2_at_0.3': ['mean', 'std', 'max', 'count'],
        'retention_0.6': ['mean', 'std'],
        'nsi_r2': ['mean', 'std']
    }).round(4)
    
    print("\nCategory Performance Summary (R² at σ=0.3):")
    print(category_summary)
    
    # Statistical test
    categories = robustness_df['model_group'].unique()
    if len(categories) > 2:
        groups = [robustness_df[robustness_df['model_group'] == cat]['r2_at_0.3'].dropna() 
                  for cat in categories]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\nANOVA: F={f_stat:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("SIGNIFICANT differences between categories!")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Box plot
    robustness_df.boxplot(column='r2_at_0.3', by='model_group', ax=ax1)
    ax1.set_title('R² at σ=0.3 by Method Category', fontsize=8)
    ax1.set_xlabel('Method Category', fontsize=7)
    ax1.set_ylabel('R² at σ=0.3', fontsize=7)
    ax1.tick_params(axis='x', rotation=45, labelsize=5)
    plt.suptitle('')
    
    # Mean with error bars
    means = robustness_df.groupby('model_group')['r2_at_0.3'].mean().sort_values(ascending=False)
    stds = robustness_df.groupby('model_group')['r2_at_0.3'].std()
    
    ax2.barh(range(len(means)), means.values, xerr=stds[means.index].values, 
             capsize=3, alpha=0.7, color='steelblue')
    ax2.set_yticks(range(len(means)))
    ax2.set_yticklabels(means.index, fontsize=6)
    ax2.set_xlabel('Mean R² at σ=0.3', fontsize=7)
    ax2.set_title('Category Ranking', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis1_method_categorization.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis1_method_categorization.png'}")
    plt.close()
    
    # Find most promising category
    best_category = means.idxmax()
    print(f"\n🎯 MOST PROMISING CATEGORY: {best_category}")
    print(f"   Mean R² at σ=0.3: {means[best_category]:.4f}")


# ============================================================================
# ANALYSIS 2: HET_GP INVESTIGATION
# ============================================================================

def analysis_2_het_gp_investigation(df, robustness_df, output_dir):
    """Deep dive into het_gp performance vs other losses"""
    print("\n" + "="*80)
    print("ANALYSIS 2: HET_GP INVESTIGATION")
    print("="*80)
    
    # Get het_gp configs
    het_gp_data = robustness_df[robustness_df['base_model'] == 'het_gp']
    
    if len(het_gp_data) == 0:
        print("No het_gp data found!")
        return
    
    print(f"\nFound {len(het_gp_data)} het_gp configurations")
    
    # Compare het_gp loss to MSE within same models
    # For models that have both het_gp and mse variants
    comparison_results = []
    
    for rep in het_gp_data['representation'].unique():
        het_gp_rep = het_gp_data[het_gp_data['representation'] == rep]
        
        # Find corresponding non-het_gp configs (ideally same base model with MSE)
        other_losses = robustness_df[
            (robustness_df['representation'] == rep) &
            (robustness_df['base_model'] != 'het_gp') &
            (robustness_df['loss_function'] == 'mse')
        ]
        
        if len(other_losses) > 0:
            for _, het_config in het_gp_rep.iterrows():
                for _, mse_config in other_losses.iterrows():
                    comparison_results.append({
                        'representation': rep,
                        'het_gp_r2_0.3': het_config['r2_at_0.3'],
                        'mse_r2_0.3': mse_config['r2_at_0.3'],
                        'het_gp_retention': het_config['retention_0.6'],
                        'mse_retention': mse_config['retention_0.6'],
                        'het_gp_nsi': het_config['nsi_r2'],
                        'mse_nsi': mse_config['nsi_r2']
                    })
    
    if comparison_results:
        comp_df = pd.DataFrame(comparison_results)
        print("\nHet_GP vs MSE Comparison:")
        print(f"  Mean het_gp R² at 0.3: {comp_df['het_gp_r2_0.3'].mean():.4f}")
        print(f"  Mean MSE R² at 0.3: {comp_df['mse_r2_0.3'].mean():.4f}")
        print(f"  Improvement: {(comp_df['het_gp_r2_0.3'].mean() - comp_df['mse_r2_0.3'].mean()):.4f}")
        
        # Statistical significance
        t_stat, p_val = stats.ttest_rel(comp_df['het_gp_r2_0.3'], comp_df['mse_r2_0.3'])
        print(f"  Paired t-test: t={t_stat:.2f}, p={p_val:.4f}")
        if p_val < 0.05:
            print("  ✓ SIGNIFICANT improvement!")
    
    # Plot degradation curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(het_gp_data)))
    
    for idx, (_, config) in enumerate(het_gp_data.iterrows()):
        config_df = df[
            (df['model'] == config['model']) &
            (df['representation'] == config['representation']) &
            (df['loss_function'] == config['loss_function'])
        ].sort_values('sigma')
        
        if len(config_df) > 0:
            label = f"{config['representation']}/{config['kernel']} (ret={config['retention_0.6']:.0f}%)"
            ax1.plot(config_df['sigma'], config_df['r2'],
                    marker='o', label=label, color=colors[idx],
                    linewidth=1.5, markersize=3)
            ax2.plot(config_df['sigma'], config_df['rmse'],
                    marker='o', color=colors[idx],
                    linewidth=1.5, markersize=3)
    
    ax1.set_xlabel('σ', fontsize=7)
    ax1.set_ylabel('R²', fontsize=7)
    ax1.set_title('HET_GP: R² Degradation', fontsize=8)
    ax1.legend(fontsize=5, loc='best')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.grid(True, alpha=0.2)
    
    ax2.set_xlabel('σ', fontsize=7)
    ax2.set_ylabel('RMSE', fontsize=7)
    ax2.set_title('HET_GP: RMSE vs σ', fontsize=8)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis2_het_gp_investigation.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis2_het_gp_investigation.png'}")
    plt.close()
    
    # Verdict
    best_het_gp = het_gp_data.loc[het_gp_data['r2_at_0.3'].idxmax()]
    print(f"\n🎯 BEST HET_GP CONFIG: {best_het_gp['representation']}/{best_het_gp['kernel']}")
    print(f"   R² at σ=0.3: {best_het_gp['r2_at_0.3']:.4f}")
    print(f"   Retention at σ=0.6: {best_het_gp['retention_0.6']:.1f}%")
    if best_het_gp['r2_at_0.3'] > 0.6:
        print("   ✓ PAPER-WORTHY!")


# ============================================================================
# ANALYSIS 3: LOSS × METHOD INTERACTION
# ============================================================================

def analysis_3_loss_method_interaction(robustness_df, output_dir):
    """Heatmap: method category × loss function"""
    print("\n" + "="*80)
    print("ANALYSIS 3: LOSS × METHOD INTERACTION")
    print("="*80)
    
    # Create pivot table
    pivot = robustness_df.pivot_table(
        values='r2_at_0.3',
        index='model_group',
        columns='loss_function',
        aggfunc='mean'
    )
    
    print("\nMean R² at σ=0.3 by Category × Loss:")
    print(pivot.round(4))
    
    # ANOVA within each category
    print("\nANOVA within each category:")
    for category in robustness_df['model_group'].unique():
        cat_data = robustness_df[robustness_df['model_group'] == category]
        losses = cat_data['loss_function'].unique()
        
        if len(losses) > 1:
            groups = [cat_data[cat_data['loss_function'] == loss]['r2_at_0.3'].dropna() 
                      for loss in losses]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups)
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {category}: F={f_stat:.2f}, p={p_val:.4f} {sig}")
    
    # Heatmap
    fig, ax = plt.subplots(figsize=(6, 4))
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=pivot.values.mean(), ax=ax, cbar_kws={'label': 'R² at σ=0.3'})
    ax.set_title('Loss × Method Interaction (R² at σ=0.3)', fontsize=8)
    ax.set_xlabel('Loss Function', fontsize=7)
    ax.set_ylabel('Method Category', fontsize=7)
    ax.tick_params(axis='both', labelsize=6)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis3_loss_method_interaction.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis3_loss_method_interaction.png'}")
    plt.close()
    
    # Find where loss matters most
    loss_variance = pivot.var(axis=1).sort_values(ascending=False)
    print(f"\n🎯 LOSS MATTERS MOST FOR: {loss_variance.idxmax()}")
    print(f"   Variance across losses: {loss_variance.max():.4f}")


# ============================================================================
# ANALYSIS 4: DISTANCE METRIC DRILL-DOWN
# ============================================================================

def analysis_4_distance_drill_down(df, robustness_df, output_dir):
    """Understand when and why distance metrics hurt/help"""
    print("\n" + "="*80)
    print("ANALYSIS 4: DISTANCE METRIC DRILL-DOWN")
    print("="*80)
    
    # Compare configs with/without distance
    with_dist = robustness_df[robustness_df['uses_distance'] == True]
    without_dist = robustness_df[robustness_df['uses_distance'] == False]
    
    print(f"\nConfigs with distance: {len(with_dist)}")
    print(f"Configs without distance: {len(without_dist)}")
    
    if len(with_dist) > 0:
        print(f"\nWith distance - Mean R² at σ=0.3: {with_dist['r2_at_0.3'].mean():.4f}")
        print(f"Without distance - Mean R² at σ=0.3: {without_dist['r2_at_0.3'].mean():.4f}")
        
        # Statistical test
        t_stat, p_val = stats.ttest_ind(with_dist['r2_at_0.3'].dropna(), 
                                        without_dist['r2_at_0.3'].dropna())
        print(f"T-test: t={t_stat:.2f}, p={p_val:.4f}")
        
        # By distance type
        print("\nPerformance by distance metric:")
        for dist in ['tanimoto', 'euclidean', 'mahalanobis']:
            dist_data = robustness_df[robustness_df['distance_metric'] == dist]
            if len(dist_data) > 0:
                print(f"  {dist}: mean R² at 0.3 = {dist_data['r2_at_0.3'].mean():.4f} (n={len(dist_data)})")
        
        # By representation
        print("\nDistance × Representation interaction:")
        for rep in robustness_df['representation'].unique():
            rep_with = with_dist[with_dist['representation'] == rep]
            rep_without = without_dist[without_dist['representation'] == rep]
            
            if len(rep_with) > 0 and len(rep_without) > 0:
                diff = rep_with['r2_at_0.3'].mean() - rep_without['r2_at_0.3'].mean()
                print(f"  {rep}: Δ = {diff:+.4f} (with={rep_with['r2_at_0.3'].mean():.4f}, without={rep_without['r2_at_0.3'].mean():.4f})")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    
    # Box plot comparison
    ax1 = axes[0, 0]
    data_for_box = []
    labels_for_box = []
    if len(with_dist) > 0:
        data_for_box.append(with_dist['r2_at_0.3'].dropna())
        labels_for_box.append('With Distance')
    if len(without_dist) > 0:
        data_for_box.append(without_dist['r2_at_0.3'].dropna())
        labels_for_box.append('Without Distance')
    
    if len(data_for_box) > 0:
        ax1.boxplot(data_for_box, labels=labels_for_box)
        ax1.set_ylabel('R² at σ=0.3', fontsize=7)
        ax1.set_title('Distance Impact on Performance', fontsize=8)
        ax1.tick_params(labelsize=6)
    
    # By distance type
    ax2 = axes[0, 1]
    if len(with_dist) > 0:
        dist_means = with_dist.groupby('distance_metric')['r2_at_0.3'].mean().sort_values()
        ax2.barh(range(len(dist_means)), dist_means.values, color='coral')
        ax2.set_yticks(range(len(dist_means)))
        ax2.set_yticklabels(dist_means.index, fontsize=6)
        ax2.set_xlabel('Mean R² at σ=0.3', fontsize=7)
        ax2.set_title('By Distance Type', fontsize=8)
    
    # Degradation curves with vs without
    ax3 = axes[1, 0]
    if len(with_dist) > 0:
        for _, config in with_dist.head(5).iterrows():
            config_df = df[
                (df['model'] == config['model']) &
                (df['representation'] == config['representation'])
            ].sort_values('sigma')
            if len(config_df) > 0:
                ax3.plot(config_df['sigma'], config_df['r2'], 
                        marker='o', linewidth=1, markersize=2, alpha=0.6, color='red')
    
    if len(without_dist) > 0:
        for _, config in without_dist.head(5).iterrows():
            config_df = df[
                (df['model'] == config['model']) &
                (df['representation'] == config['representation'])
            ].sort_values('sigma')
            if len(config_df) > 0:
                ax3.plot(config_df['sigma'], config_df['r2'],
                        marker='o', linewidth=1, markersize=2, alpha=0.6, color='blue')
    
    ax3.set_xlabel('σ', fontsize=7)
    ax3.set_ylabel('R²', fontsize=7)
    ax3.set_title('Degradation Curves (red=with, blue=without)', fontsize=8)
    ax3.grid(True, alpha=0.2)
    
    # Retention comparison
    ax4 = axes[1, 1]
    if len(with_dist) > 0 and len(without_dist) > 0:
        ax4.scatter(with_dist['retention_0.6'], with_dist['r2_at_0.3'],
                   alpha=0.5, s=30, label='With Distance', color='red')
        ax4.scatter(without_dist['retention_0.6'], without_dist['r2_at_0.3'],
                   alpha=0.5, s=30, label='Without Distance', color='blue')
        ax4.set_xlabel('Retention % at σ=0.6', fontsize=7)
        ax4.set_ylabel('R² at σ=0.3', fontsize=7)
        ax4.set_title('Performance vs Retention', fontsize=8)
        ax4.legend(fontsize=5)
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis4_distance_drill_down.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis4_distance_drill_down.png'}")
    plt.close()
    
    # Verdict
    if len(with_dist) > 0 and len(without_dist) > 0:
        if with_dist['r2_at_0.3'].mean() < without_dist['r2_at_0.3'].mean():
            print("\n🎯 VERDICT: Distance metrics generally HURT performance")
        else:
            print("\n🎯 VERDICT: Distance metrics can HELP in specific cases")


# ============================================================================
# ANALYSIS 5: UNCERTAINTY METHOD VALIDATION
# ============================================================================

def analysis_5_uncertainty_validation(df, robustness_df, output_dir):
    """Compare uncertainty quantification approaches"""
    print("\n" + "="*80)
    print("ANALYSIS 5: UNCERTAINTY METHOD VALIDATION")
    print("="*80)
    
    # Define UQ categories
    uq_categories = {
        'conformal': 'uncertainty_conformal',
        'heteroscedastic': 'uncertainty_heteroscedastic',
        'evidential': 'uncertainty_evidential',
    }
    
    uq_data = robustness_df[robustness_df['model_group'].isin(uq_categories.values())]
    
    if len(uq_data) == 0:
        print("No UQ methods found!")
        return
    
    print(f"\nFound {len(uq_data)} UQ configurations:")
    for cat_name, cat_group in uq_categories.items():
        n = len(uq_data[uq_data['model_group'] == cat_group])
        if n > 0:
            mean_r2 = uq_data[uq_data['model_group'] == cat_group]['r2_at_0.3'].mean()
            print(f"  {cat_name}: n={n}, mean R² at 0.3 = {mean_r2:.4f}")
    
    # Compare UQ approaches
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    
    # Performance comparison
    ax1 = axes[0, 0]
    uq_data.boxplot(column='r2_at_0.3', by='model_group', ax=ax1)
    ax1.set_title('UQ Methods: R² at σ=0.3', fontsize=8)
    ax1.set_xlabel('UQ Approach', fontsize=7)
    ax1.set_ylabel('R² at σ=0.3', fontsize=7)
    ax1.tick_params(axis='x', rotation=45, labelsize=5)
    plt.suptitle('')
    
    # Retention comparison
    ax2 = axes[0, 1]
    uq_data.boxplot(column='retention_0.6', by='model_group', ax=ax2)
    ax2.set_title('UQ Methods: Retention at σ=0.6', fontsize=8)
    ax2.set_xlabel('UQ Approach', fontsize=7)
    ax2.set_ylabel('Retention %', fontsize=7)
    ax2.tick_params(axis='x', rotation=45, labelsize=5)
    plt.suptitle('')
    
    # Degradation curves
    ax3 = axes[1, 0]
    colors = {'uncertainty_conformal': 'red', 
              'uncertainty_heteroscedastic': 'blue',
              'uncertainty_evidential': 'green'}
    
    for _, config in uq_data.iterrows():
        config_df = df[
            (df['model'] == config['model']) &
            (df['representation'] == config['representation'])
        ].sort_values('sigma')
        
        if len(config_df) > 0:
            color = colors.get(config['model_group'], 'gray')
            ax3.plot(config_df['sigma'], config_df['r2'],
                    marker='o', linewidth=1, markersize=2, 
                    alpha=0.6, color=color)
    
    ax3.set_xlabel('σ', fontsize=7)
    ax3.set_ylabel('R²', fontsize=7)
    ax3.set_title('UQ Degradation Curves', fontsize=8)
    ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax3.grid(True, alpha=0.2)
    
    # NSI comparison
    ax4 = axes[1, 1]
    uq_data.boxplot(column='nsi_r2', by='model_group', ax=ax4)
    ax4.set_title('UQ Methods: NSI (lower=better)', fontsize=8)
    ax4.set_xlabel('UQ Approach', fontsize=7)
    ax4.set_ylabel('NSI', fontsize=7)
    ax4.tick_params(axis='x', rotation=45, labelsize=5)
    ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.suptitle('')
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis5_uncertainty_validation.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis5_uncertainty_validation.png'}")
    plt.close()
    
    # Best UQ approach
    best_uq = uq_data.loc[uq_data['r2_at_0.3'].idxmax()]
    print(f"\n🎯 BEST UQ APPROACH: {best_uq['model_group']}")
    print(f"   Config: {best_uq['base_model']}/{best_uq['representation']}")
    print(f"   R² at σ=0.3: {best_uq['r2_at_0.3']:.4f}")


# ============================================================================
# ANALYSIS 6: KERNEL ANALYSIS
# ============================================================================

def analysis_6_kernel_analysis(df, robustness_df, output_dir):
    """Compare kernels for GP methods"""
    print("\n" + "="*80)
    print("ANALYSIS 6: KERNEL ANALYSIS (GP METHODS ONLY)")
    print("="*80)
    
    kernel_data = robustness_df[robustness_df['kernel'] != 'none']
    
    if len(kernel_data) == 0:
        print("No kernel-based methods found!")
        return
    
    print(f"\nFound {len(kernel_data)} kernel configurations")
    
    # Summary by kernel
    print("\nPerformance by kernel:")
    for kernel in kernel_data['kernel'].unique():
        k_data = kernel_data[kernel_data['kernel'] == kernel]
        print(f"  {kernel}: mean R² at 0.3 = {k_data['r2_at_0.3'].mean():.4f} (n={len(k_data)})")
    
    # ANOVA
    kernels = kernel_data['kernel'].unique()
    if len(kernels) > 1:
        groups = [kernel_data[kernel_data['kernel'] == k]['r2_at_0.3'].dropna() for k in kernels]
        f_stat, p_val = stats.f_oneway(*groups)
        print(f"\nANOVA: F={f_stat:.2f}, p={p_val:.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    
    # Box plot
    ax1 = axes[0, 0]
    kernel_data.boxplot(column='r2_at_0.3', by='kernel', ax=ax1)
    ax1.set_title('Kernel Comparison: R² at σ=0.3', fontsize=8)
    ax1.set_xlabel('Kernel', fontsize=7)
    ax1.set_ylabel('R² at σ=0.3', fontsize=7)
    ax1.tick_params(labelsize=6)
    plt.suptitle('')
    
    # NSI
    ax2 = axes[0, 1]
    kernel_data.boxplot(column='nsi_r2', by='kernel', ax=ax2)
    ax2.set_title('Kernel Comparison: NSI', fontsize=8)
    ax2.set_xlabel('Kernel', fontsize=7)
    ax2.set_ylabel('NSI', fontsize=7)
    ax2.tick_params(labelsize=6)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.suptitle('')
    
    # Degradation curves
    ax3 = axes[1, 0]
    colors = {'tanimoto': 'red', 'rbf': 'blue', 'matern': 'green'}
    
    for _, config in kernel_data.iterrows():
        config_df = df[
            (df['model'] == config['model']) &
            (df['representation'] == config['representation']) &
            (df['kernel'] == config['kernel'])
        ].sort_values('sigma')
        
        if len(config_df) > 0:
            color = colors.get(config['kernel'], 'gray')
            ax3.plot(config_df['sigma'], config_df['r2'],
                    marker='o', linewidth=1, markersize=2,
                    alpha=0.6, color=color)
    
    ax3.set_xlabel('σ', fontsize=7)
    ax3.set_ylabel('R²', fontsize=7)
    ax3.set_title('Degradation by Kernel', fontsize=8)
    ax3.grid(True, alpha=0.2)
    
    # Retention
    ax4 = axes[1, 1]
    kernel_data.boxplot(column='retention_0.6', by='kernel', ax=ax4)
    ax4.set_title('Retention % at σ=0.6', fontsize=8)
    ax4.set_xlabel('Kernel', fontsize=7)
    ax4.set_ylabel('Retention %', fontsize=7)
    ax4.tick_params(labelsize=6)
    plt.suptitle('')
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis6_kernel_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis6_kernel_analysis.png'}")
    plt.close()
    
    # Best kernel
    best_kernel = kernel_data.loc[kernel_data['r2_at_0.3'].idxmax()]
    print(f"\n🎯 BEST KERNEL: {best_kernel['kernel']}")
    print(f"   R² at σ=0.3: {best_kernel['r2_at_0.3']:.4f}")


# ============================================================================
# ANALYSIS 7: TOP PERFORMER DEEP DIVE
# ============================================================================

def analysis_7_top_performer_deep_dive(robustness_df, output_dir):
    """Understand what makes top performers special"""
    print("\n" + "="*80)
    print("ANALYSIS 7: TOP PERFORMER DEEP DIVE")
    print("="*80)
    
    top_n = 20
    top_configs = robustness_df.nlargest(top_n, 'r2_at_0.3')
    
    print(f"\nTop {top_n} configurations:")
    for idx, (_, row) in enumerate(top_configs.iterrows(), 1):
        print(f"{idx:2d}. {row['base_model']}/{row['representation']}/{row['loss_function']} "
              f"- R²@0.3={row['r2_at_0.3']:.4f}, ret={row['retention_0.6']:.1f}%")
    
    # Pattern analysis
    print("\nPatterns in top performers:")
    print(f"  Most common category: {top_configs['model_group'].mode().values[0]}")
    print(f"  Most common representation: {top_configs['representation'].mode().values[0]}")
    print(f"  Most common loss: {top_configs['loss_function'].mode().values[0]}")
    print(f"  Use distance? {(top_configs['uses_distance'].sum() / len(top_configs) * 100):.0f}%")
    
    # Distribution visualization
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    
    # Category distribution
    ax1 = axes[0, 0]
    top_configs['model_group'].value_counts().plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Count', fontsize=7)
    ax1.set_title('Top 20: Category Distribution', fontsize=8)
    ax1.tick_params(labelsize=6)
    
    # Representation distribution
    ax2 = axes[0, 1]
    top_configs['representation'].value_counts().plot(kind='barh', ax=ax2, color='coral')
    ax2.set_xlabel('Count', fontsize=7)
    ax2.set_title('Top 20: Representation Distribution', fontsize=8)
    ax2.tick_params(labelsize=6)
    
    # Loss distribution
    ax3 = axes[1, 0]
    top_configs['loss_function'].value_counts().plot(kind='barh', ax=ax3, color='lightgreen')
    ax3.set_xlabel('Count', fontsize=7)
    ax3.set_title('Top 20: Loss Distribution', fontsize=8)
    ax3.tick_params(labelsize=6)
    
    # Distance usage
    ax4 = axes[1, 1]
    dist_counts = top_configs['uses_distance'].value_counts()
    ax4.pie(dist_counts.values, labels=['No Distance', 'Uses Distance'], 
            autopct='%1.0f%%', startangle=90)
    ax4.set_title('Top 20: Distance Usage', fontsize=8)
    
    for ax in axes.flat[:3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis7_top_performer_deep_dive.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis7_top_performer_deep_dive.png'}")
    plt.close()
    
    # Statistical comparison: top vs rest
    rest_configs = robustness_df[~robustness_df.index.isin(top_configs.index)]
    
    print("\nTop 20 vs Rest:")
    print(f"  Top 20 mean R² at 0: {top_configs['r2_at_0.0'].mean():.4f}")
    print(f"  Rest mean R² at 0: {rest_configs['r2_at_0.0'].mean():.4f}")
    print(f"  Top 20 mean retention: {top_configs['retention_0.6'].mean():.1f}%")
    print(f"  Rest mean retention: {rest_configs['retention_0.6'].mean():.1f}%")
    
    print("\n🎯 WINNING FORMULA:")
    print(f"   Category: {top_configs['model_group'].mode().values[0]}")
    print(f"   Representation: {top_configs['representation'].mode().values[0]}")
    print(f"   Loss: {top_configs['loss_function'].mode().values[0]}")


# ============================================================================
# ANALYSIS 8: VARIANCE DECOMPOSITION (ANOVA)
# ============================================================================

def analysis_8_variance_decomposition(robustness_df, output_dir):
    """ANOVA: what factors matter most?"""
    print("\n" + "="*80)
    print("ANALYSIS 8: VARIANCE DECOMPOSITION (ANOVA)")
    print("="*80)
    
    # Prepare data
    data = robustness_df.dropna(subset=['r2_at_0.3'])
    
    # One-way ANOVA for each factor
    factors = {
        'model_group': 'Method Category',
        'loss_function': 'Loss Function',
        'uses_distance': 'Uses Distance',
        'representation': 'Representation',
        'kernel': 'Kernel'
    }
    
    results = []
    
    print("\nOne-way ANOVA results:")
    for factor, label in factors.items():
        if factor in data.columns:
            groups = data.groupby(factor)['r2_at_0.3'].apply(list)
            
            if len(groups) > 1:
                f_stat, p_val = stats.f_oneway(*groups.values)
                
                # Calculate effect size (eta-squared)
                grand_mean = data['r2_at_0.3'].mean()
                ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups.values)
                ss_total = sum((x - grand_mean)**2 for g in groups.values for x in g)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0
                
                results.append({
                    'factor': label,
                    'f_stat': f_stat,
                    'p_val': p_val,
                    'eta_squared': eta_sq,
                    'n_levels': len(groups)
                })
                
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                print(f"  {label:20s}: F={f_stat:6.2f}, p={p_val:.4f} {sig:3s}, η²={eta_sq:.3f}")
    
    results_df = pd.DataFrame(results).sort_values('eta_squared', ascending=False)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Effect sizes
    ax1.barh(range(len(results_df)), results_df['eta_squared'], color='steelblue')
    ax1.set_yticks(range(len(results_df)))
    ax1.set_yticklabels(results_df['factor'], fontsize=7)
    ax1.set_xlabel('Effect Size (η²)', fontsize=7)
    ax1.set_title('Variance Explained by Each Factor', fontsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # P-values (log scale)
    ax2.barh(range(len(results_df)), -np.log10(results_df['p_val']), color='coral')
    ax2.set_yticks(range(len(results_df)))
    ax2.set_yticklabels(results_df['factor'], fontsize=7)
    ax2.set_xlabel('-log10(p-value)', fontsize=7)
    ax2.set_title('Statistical Significance', fontsize=8)
    ax2.axvline(x=-np.log10(0.05), color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis8_variance_decomposition.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis8_variance_decomposition.png'}")
    plt.close()
    
    # Save results
    results_df.to_csv(output_dir / "analysis8_anova_results.csv", index=False)
    
    print(f"\n🎯 MOST IMPORTANT FACTOR: {results_df.iloc[0]['factor']}")
    print(f"   Explains {results_df.iloc[0]['eta_squared']*100:.1f}% of variance")


# ============================================================================
# ANALYSIS 9: FAILURE ANALYSIS
# ============================================================================

def analysis_9_failure_analysis(robustness_df, output_dir):
    """Understand what makes configs fail"""
    print("\n" + "="*80)
    print("ANALYSIS 9: FAILURE ANALYSIS")
    print("="*80)
    
    bottom_n = 20
    failures = robustness_df.nsmallest(bottom_n, 'r2_at_0.3')
    
    print(f"\nBottom {bottom_n} configurations:")
    for idx, (_, row) in enumerate(failures.iterrows(), 1):
        print(f"{idx:2d}. {row['base_model']}/{row['representation']}/{row['loss_function']} "
              f"- R²@0.3={row['r2_at_0.3']:.4f}, R²@0={row['r2_at_0.0']:.4f}")
    
    # Pattern analysis
    print("\nPatterns in failures:")
    print(f"  Most common category: {failures['model_group'].mode().values[0]}")
    print(f"  Most common representation: {failures['representation'].mode().values[0]}")
    print(f"  Most common loss: {failures['loss_function'].mode().values[0]}")
    print(f"  Use distance? {(failures['uses_distance'].sum() / len(failures) * 100):.0f}%")
    
    # Compare to successful configs
    successes = robustness_df.nlargest(bottom_n, 'r2_at_0.3')
    
    print("\nFailures vs Successes:")
    print(f"  Failures mean R² at 0: {failures['r2_at_0.0'].mean():.4f}")
    print(f"  Successes mean R² at 0: {successes['r2_at_0.0'].mean():.4f}")
    print(f"  Failures mean NSI: {failures['nsi_r2'].mean():.4f}")
    print(f"  Successes mean NSI: {successes['nsi_r2'].mean():.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    
    # Category distribution
    ax1 = axes[0, 0]
    failures['model_group'].value_counts().plot(kind='barh', ax=ax1, color='darkred')
    ax1.set_xlabel('Count', fontsize=7)
    ax1.set_title('Bottom 20: Category Distribution', fontsize=8)
    ax1.tick_params(labelsize=6)
    
    # R² at 0 vs R² at 0.3
    ax2 = axes[0, 1]
    ax2.scatter(failures['r2_at_0.0'], failures['r2_at_0.3'], 
               alpha=0.6, s=50, color='darkred', label='Failures')
    ax2.scatter(successes['r2_at_0.0'], successes['r2_at_0.3'],
               alpha=0.6, s=50, color='darkgreen', label='Successes')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.5)
    ax2.set_xlabel('R² at σ=0', fontsize=7)
    ax2.set_ylabel('R² at σ=0.3', fontsize=7)
    ax2.set_title('Baseline vs Noisy Performance', fontsize=8)
    ax2.legend(fontsize=6)
    
    # NSI distribution
    ax3 = axes[1, 0]
    ax3.hist([failures['nsi_r2'].dropna(), successes['nsi_r2'].dropna()],
            bins=20, alpha=0.6, color=['darkred', 'darkgreen'],
            label=['Failures', 'Successes'])
    ax3.set_xlabel('NSI', fontsize=7)
    ax3.set_ylabel('Count', fontsize=7)
    ax3.set_title('NSI Distribution', fontsize=8)
    ax3.legend(fontsize=6)
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    # Retention
    ax4 = axes[1, 1]
    data_for_box = [failures['retention_0.6'].dropna(), successes['retention_0.6'].dropna()]
    ax4.boxplot(data_for_box, labels=['Failures', 'Successes'])
    ax4.set_ylabel('Retention % at σ=0.6', fontsize=7)
    ax4.set_title('Retention Comparison', fontsize=8)
    ax4.tick_params(labelsize=6)
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis9_failure_analysis.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis9_failure_analysis.png'}")
    plt.close()
    
    print("\n🎯 FAILURE PATTERNS:")
    print(f"   Avoid: {failures['model_group'].mode().values[0]} category")
    print(f"   Poor baseline (R² < {failures['r2_at_0.0'].mean():.2f}) is a red flag")


# ============================================================================
# ANALYSIS 10: PHASE 0C GAP ANALYSIS
# ============================================================================

def analysis_10_phase0c_gap(robustness_df, phase0c_dir, output_dir):
    """When is sophistication worth it?"""
    print("\n" + "="*80)
    print("ANALYSIS 10: PHASE 0C GAP ANALYSIS")
    print("="*80)
    
    if phase0c_dir is None:
        print("No Phase 0C directory provided, skipping")
        return
    
    # Load Phase 0C data (simplified version)
    print(f"\nLoading Phase 0C baseline data from {phase0c_dir}...")
    
    all_data = []
    screening_files = list(Path(phase0c_dir).glob("phase0c_screen_*.csv"))
    
    if not screening_files:
        print(f"No phase0c_screen_*.csv files found")
        return
    
    for filepath in screening_files:
        try:
            df = pd.read_csv(filepath)
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    if not all_data:
        return
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['model'] = combined_df['model'].str.replace('_split', '', regex=False)
    combined_df = combined_df[combined_df['r2'] > -10]
    
    valid_sigmas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    combined_df = combined_df[combined_df['sigma'].isin(valid_sigmas)]
    
    phase0c_results = combined_df.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean'
    }).reset_index()
    phase0c_results.rename(columns={'rep': 'representation'}, inplace=True)
    
    # Get R² at 0.3 for Phase 0C
    phase0c_at_03 = phase0c_results[phase0c_results['sigma'] == 0.3].groupby(['model', 'representation'])['r2'].mean()
    
    # Find best Phase 0C baseline
    best_phase0c_r2 = phase0c_at_03.max()
    best_phase0c_config = phase0c_at_03.idxmax()
    
    print(f"\nBest Phase 0C baseline:")
    print(f"  {best_phase0c_config[0]}/{best_phase0c_config[1]}")
    print(f"  R² at σ=0.3: {best_phase0c_r2:.4f}")
    
    # Compare comprehensive study
    comprehensive_at_03 = robustness_df['r2_at_0.3']
    
    # Configs that beat Phase 0C
    beats_phase0c = robustness_df[robustness_df['r2_at_0.3'] > best_phase0c_r2]
    loses_to_phase0c = robustness_df[robustness_df['r2_at_0.3'] < best_phase0c_r2]
    
    print(f"\nComprehensive study vs Phase 0C:")
    print(f"  Configs beating Phase 0C: {len(beats_phase0c)} ({len(beats_phase0c)/len(robustness_df)*100:.1f}%)")
    print(f"  Configs losing to Phase 0C: {len(loses_to_phase0c)} ({len(loses_to_phase0c)/len(robustness_df)*100:.1f}%)")
    
    if len(beats_phase0c) > 0:
        print(f"\n  Best improvement: {(beats_phase0c['r2_at_0.3'].max() - best_phase0c_r2):.4f}")
        print(f"  Mean improvement: {(beats_phase0c['r2_at_0.3'].mean() - best_phase0c_r2):.4f}")
        
        print("\n  Top 5 configs beating Phase 0C:")
        for idx, (_, row) in enumerate(beats_phase0c.nlargest(5, 'r2_at_0.3').iterrows(), 1):
            improvement = row['r2_at_0.3'] - best_phase0c_r2
            print(f"    {idx}. {row['base_model']}/{row['representation']}/{row['loss_function']} "
                  f"+{improvement:.4f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    # Distribution comparison
    ax1.hist([loses_to_phase0c['r2_at_0.3'], beats_phase0c['r2_at_0.3']],
            bins=20, alpha=0.6, color=['coral', 'steelblue'],
            label=['Worse than Phase 0C', 'Better than Phase 0C'])
    ax1.axvline(x=best_phase0c_r2, color='red', linestyle='--', linewidth=2, 
               label=f'Best Phase 0C ({best_phase0c_r2:.3f})')
    ax1.set_xlabel('R² at σ=0.3', fontsize=7)
    ax1.set_ylabel('Count', fontsize=7)
    ax1.set_title('Comprehensive vs Phase 0C', fontsize=8)
    ax1.legend(fontsize=6)
    
    # Category breakdown
    if len(beats_phase0c) > 0:
        ax2 = beats_phase0c['model_group'].value_counts().plot(kind='barh', ax=ax2, color='steelblue')
        ax2.set_xlabel('Count', fontsize=7)
        ax2.set_title('Categories Beating Phase 0C', fontsize=8)
        ax2.tick_params(labelsize=6)
    
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis10_phase0c_gap.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis10_phase0c_gap.png'}")
    plt.close()
    
    if len(beats_phase0c) > 0:
        print("\n🎯 SOPHISTICATION WORTH IT FOR:")
        print(f"   {beats_phase0c['model_group'].mode().values[0]} category")
        print(f"   Potential gain: up to +{(beats_phase0c['r2_at_0.3'].max() - best_phase0c_r2):.4f} R²")
    else:
        print("\n🎯 WARNING: No configs beat best Phase 0C baseline!")


# ============================================================================
# ANALYSIS 11: CLUSTERING/PATTERN FINDING
# ============================================================================

def analysis_11_clustering(robustness_df, output_dir):
    """Find config archetypes"""
    print("\n" + "="*80)
    print("ANALYSIS 11: CLUSTERING/PATTERN FINDING")
    print("="*80)
    
    # Prepare features for clustering
    features = ['r2_at_0.0', 'r2_at_0.3', 'r2_at_0.6', 'retention_0.6', 'nsi_r2']
    
    cluster_data = robustness_df[features].dropna()
    
    if len(cluster_data) < 10:
        print("Not enough data for clustering")
        return
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_data)
    
    # K-means with 4 clusters (archetypes)
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_data['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels back to main df
    robustness_df_clustered = robustness_df.loc[cluster_data.index].copy()
    robustness_df_clustered['cluster'] = cluster_data['cluster']
    
    # Characterize each cluster
    print(f"\nFound {n_clusters} config archetypes:")
    
    cluster_names = {}
    for cluster_id in range(n_clusters):
        cluster_configs = robustness_df_clustered[robustness_df_clustered['cluster'] == cluster_id]
        
        mean_baseline = cluster_configs['r2_at_0.0'].mean()
        mean_retention = cluster_configs['retention_0.6'].mean()
        mean_r2_03 = cluster_configs['r2_at_0.3'].mean()
        
        # Name the cluster
        if mean_baseline > 0.6 and mean_retention > 70:
            name = "🌟 High Baseline, High Retention"
        elif mean_baseline > 0.6 and mean_retention < 70:
            name = "⚠️ High Baseline, Poor Retention"
        elif mean_baseline < 0.6 and mean_retention > 70:
            name = "💪 Lower Baseline, Great Retention"
        else:
            name = "❌ Low Performance Overall"
        
        cluster_names[cluster_id] = name
        
        print(f"\nCluster {cluster_id}: {name}")
        print(f"  Size: {len(cluster_configs)}")
        print(f"  Mean R² at 0: {mean_baseline:.4f}")
        print(f"  Mean R² at 0.3: {mean_r2_03:.4f}")
        print(f"  Mean retention: {mean_retention:.1f}%")
        print(f"  Top category: {cluster_configs['model_group'].mode().values[0]}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(7, 6))
    
    # Scatter: baseline vs retention
    ax1 = axes[0, 0]
    for cluster_id in range(n_clusters):
        cluster_configs = robustness_df_clustered[robustness_df_clustered['cluster'] == cluster_id]
        ax1.scatter(cluster_configs['r2_at_0.0'], cluster_configs['retention_0.6'],
                   alpha=0.6, s=30, label=f"C{cluster_id}")
    ax1.set_xlabel('R² at σ=0 (baseline)', fontsize=7)
    ax1.set_ylabel('Retention % at σ=0.6', fontsize=7)
    ax1.set_title('Cluster Scatter: Baseline vs Retention', fontsize=8)
    ax1.legend(fontsize=6)
    ax1.grid(True, alpha=0.2)
    
    # Scatter: R² at 0.3 vs NSI
    ax2 = axes[0, 1]
    for cluster_id in range(n_clusters):
        cluster_configs = robustness_df_clustered[robustness_df_clustered['cluster'] == cluster_id]
        ax2.scatter(cluster_configs['nsi_r2'], cluster_configs['r2_at_0.3'],
                   alpha=0.6, s=30, label=f"C{cluster_id}")
    ax2.set_xlabel('NSI (slope)', fontsize=7)
    ax2.set_ylabel('R² at σ=0.3', fontsize=7)
    ax2.set_title('Cluster Scatter: NSI vs Performance', fontsize=8)
    ax2.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.legend(fontsize=6)
    ax2.grid(True, alpha=0.2)
    
    # Cluster sizes
    ax3 = axes[1, 0]
    cluster_sizes = robustness_df_clustered['cluster'].value_counts().sort_index()
    cluster_labels = [f"C{i}: {cluster_names[i][:20]}..." for i in cluster_sizes.index]
    ax3.barh(range(len(cluster_sizes)), cluster_sizes.values, color='steelblue')
    ax3.set_yticks(range(len(cluster_sizes)))
    ax3.set_yticklabels(cluster_labels, fontsize=6)
    ax3.set_xlabel('Number of Configs', fontsize=7)
    ax3.set_title('Cluster Sizes', fontsize=8)
    
    # Mean performance by cluster
    ax4 = axes[1, 1]
    cluster_means = robustness_df_clustered.groupby('cluster')['r2_at_0.3'].mean().sort_values(ascending=False)
    ax4.barh(range(len(cluster_means)), cluster_means.values, color='coral')
    ax4.set_yticks(range(len(cluster_means)))
    ax4.set_yticklabels([f"C{i}" for i in cluster_means.index], fontsize=6)
    ax4.set_xlabel('Mean R² at σ=0.3', fontsize=7)
    ax4.set_title('Cluster Performance Ranking', fontsize=8)
    
    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis11_clustering.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis11_clustering.png'}")
    plt.close()
    
    # Save cluster assignments
    cluster_output = robustness_df_clustered[['model', 'representation', 'loss_function', 
                                              'r2_at_0.0', 'r2_at_0.3', 'retention_0.6', 
                                              'nsi_r2', 'cluster']].copy()
    cluster_output['cluster_name'] = cluster_output['cluster'].map(cluster_names)
    cluster_output.to_csv(output_dir / "analysis11_cluster_assignments.csv", index=False)
    
    print("\n🎯 PICK YOUR TRADEOFF:")
    best_cluster = cluster_means.idxmax()
    print(f"   Best overall: {cluster_names[best_cluster]}")


# ============================================================================
# ANALYSIS 12: NOISE LEVEL SENSITIVITY
# ============================================================================

def analysis_12_noise_sensitivity(df, robustness_df, output_dir):
    """Which methods maintain performance longest?"""
    print("\n" + "="*80)
    print("ANALYSIS 12: NOISE LEVEL SENSITIVITY")
    print("="*80)
    
    # Get top 10 configs
    top_configs = robustness_df.nlargest(10, 'r2_at_0.3')
    
    # Plot degradation curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_configs)))
    
    for idx, (_, config) in enumerate(top_configs.iterrows()):
        config_df = df[
            (df['model'] == config['model']) &
            (df['representation'] == config['representation']) &
            (df['loss_function'] == config['loss_function'])
        ].sort_values('sigma')
        
        if len(config_df) > 0:
            label = f"{config['base_model']}/{config['representation'][:3]}"
            
            ax1.plot(config_df['sigma'], config_df['r2'],
                    marker='o', label=label, color=colors[idx],
                    linewidth=1.5, markersize=3)
            
            # Normalized to baseline
            r2_baseline = config_df[config_df['sigma'] == 0]['r2'].values[0]
            if r2_baseline != 0:
                r2_normalized = (config_df['r2'] / r2_baseline) * 100
                ax2.plot(config_df['sigma'], r2_normalized,
                        marker='o', label=label, color=colors[idx],
                        linewidth=1.5, markersize=3)
    
    ax1.set_xlabel('σ (noise level)', fontsize=7)
    ax1.set_ylabel('R²', fontsize=7)
    ax1.set_title('Top 10: Absolute R² Degradation', fontsize=8)
    ax1.legend(fontsize=5, loc='best', ncol=2)
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax1.grid(True, alpha=0.2)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    ax2.set_xlabel('σ (noise level)', fontsize=7)
    ax2.set_ylabel('% of Baseline R²', fontsize=7)
    ax2.set_title('Top 10: Normalized Performance', fontsize=8)
    ax2.axhline(y=100, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax2.axhline(y=70, color='orange', linestyle=':', linewidth=0.5, alpha=0.3)
    ax2.grid(True, alpha=0.2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "analysis12_noise_sensitivity.png", dpi=300, bbox_inches='tight')
    print(f"Saved to {output_dir / 'analysis12_noise_sensitivity.png'}")
    plt.close()
    
    # Find critical thresholds
    print("\nCritical noise thresholds (where R² drops below 70% of baseline):")
    for _, config in top_configs.head(5).iterrows():
        config_df = df[
            (df['model'] == config['model']) &
            (df['representation'] == config['representation']) &
            (df['loss_function'] == config['loss_function'])
        ].sort_values('sigma')
        
        if len(config_df) > 0:
            baseline = config_df[config_df['sigma'] == 0]['r2'].values[0]
            threshold_70 = baseline * 0.7
            
            # Find first sigma where R² < 70% baseline
            below_threshold = config_df[config_df['r2'] < threshold_70]
            if len(below_threshold) > 0:
                critical_sigma = below_threshold.iloc[0]['sigma']
                print(f"  {config['base_model']}/{config['representation']}: σ={critical_sigma:.1f}")
            else:
                print(f"  {config['base_model']}/{config['representation']}: σ>1.0 (robust!)")
    
    print("\n🎯 OPERATING RANGES:")
    best_config = top_configs.iloc[0]
    print(f"   Most robust: {best_config['base_model']}/{best_config['representation']}")
    print(f"   Maintains {best_config['retention_0.6']:.0f}% at σ=0.6")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main(results_dir, phase0c_dir=None):
    """Run all 12 analyses"""
    
    print("="*80)
    print("ENHANCED COMPREHENSIVE NOISE ROBUSTNESS ANALYSIS")
    print("Running all 12 research directions")
    print("="*80)
    
    # Load data
    print("\n[SETUP] Loading comprehensive study results...")
    df = load_comprehensive_results(results_dir)
    
    if len(df) == 0:
        print("No data found!")
        return
    
    print("\n[SETUP] Parsing model variants...")
    df = parse_model_variants(df)
    df = categorize_models(df)
    
    print("\n[SETUP] Calculating robustness metrics...")
    robustness_df = calculate_robustness_metrics(df)
    print(f"   Computed metrics for {len(robustness_df)} configurations")
    
    # Create output directory
    output_dir = Path(results_dir) / "comprehensive_analysis_enhanced"
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"\n[SETUP] Saving outputs to {output_dir}")
    
    # Run all analyses
    analysis_1_method_categorization(robustness_df, output_dir)
    analysis_2_het_gp_investigation(df, robustness_df, output_dir)
    analysis_3_loss_method_interaction(robustness_df, output_dir)
    analysis_4_distance_drill_down(df, robustness_df, output_dir)
    analysis_5_uncertainty_validation(df, robustness_df, output_dir)
    analysis_6_kernel_analysis(df, robustness_df, output_dir)
    analysis_7_top_performer_deep_dive(robustness_df, output_dir)
    analysis_8_variance_decomposition(robustness_df, output_dir)
    analysis_9_failure_analysis(robustness_df, output_dir)
    
    if phase0c_dir:
        analysis_10_phase0c_gap(robustness_df, phase0c_dir, output_dir)
    
    analysis_11_clustering(robustness_df, output_dir)
    analysis_12_noise_sensitivity(df, robustness_df, output_dir)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"\nGenerated:")
    print(f"  - 12 focused analysis plots")
    print(f"  - Statistical tables and results")
    print(f"  - Cluster assignments")
    print(f"  - ANOVA results")
    
    # Top-level insights
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    best_overall = robustness_df.nlargest(1, 'r2_at_0.3').iloc[0]
    print(f"\n🏆 BEST OVERALL CONFIG:")
    print(f"   {best_overall['base_model']}/{best_overall['representation']}/{best_overall['loss_function']}")
    print(f"   R² at σ=0.3: {best_overall['r2_at_0.3']:.4f}")
    print(f"   Retention: {best_overall['retention_0.6']:.1f}%")
    
    print(f"\n📊 DATASET SUMMARY:")
    print(f"   Total configs analyzed: {len(robustness_df)}")
    print(f"   Method categories: {robustness_df['model_group'].nunique()}")
    print(f"   Representations: {robustness_df['representation'].nunique()}")
    print(f"   Loss functions: {robustness_df['loss_function'].nunique()}")
    
    print("\n✅ Check individual analysis plots for detailed insights!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python analyze_comprehensive_noise_study_enhanced.py <results_dir> [phase0c_dir]")
        print("Example: python analyze_comprehensive_noise_study_enhanced.py ~/results/comprehensive_noise_study ~/results")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    phase0c_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(results_dir, phase0c_dir)