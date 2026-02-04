"""
Phase 0 Analysis - Global Screening of Model-Representation Pairs
Generates Figures 1-2 and Supplementary S1-S3

Based on the detailed outline:
- Figure 1: Global noise robustness landscape (4 panels)
- Figure 2: Representation and noise-type effects (4 panels)
- Supplementary S1-S3: Additional analyses

Key metrics used:
- NDS (Noise Degradation Slope): slope of R² vs σ (negative = degradation)
- Baseline R² at σ=0
- NDS is thresholded: only calculated for configs with baseline R² > 0.6
  (except for one comparison showing the threshold importance)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib.patches import Patch
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
REPRESENTATION_COLORS = {
    'pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'smiles': '#CA3542',
    'graph': '#949494',
    'pdv-3d': '#56B4E9',
    'random_smiles': '#756bb1',
    'mhggnn': '#CC79A7',
}

MODEL_COLORS = {
    'rf': '#3498db',
    'xgboost': '#e74c3c',
    'gauche': '#9b59b6',
    'qrf': '#16a085',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'mlp': '#2c3e50',
    'dnn_bnn_full': '#e67e22',
    'dnn_bnn_last': '#95a5a6',
    'dnn_bnn_variational': '#d35400',
    'mlp_bnn_full': '#8e44ad',
    'mlp_bnn_last': '#7f8c8d',
    'mlp_bnn_variational': '#c0392b',
}

# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_representation(rep):
    """Format representation names for display"""
    rep_map = {
        'ecfp4': 'ECFP4',
        'pdv': 'PDV',
        'pdv-3d': 'PDV-3D',
        'sns': 'SNS',
        'smiles': 'SMILES',
        'randomized_smiles': 'R-SMILES',
        'random_smiles': 'R-SMILES',
        'graph': 'Graph',
        'mhggnn': 'MHGGNN',
    }
    return rep_map.get(rep.lower(), rep.upper())


def format_model(model):
    """Format model names for display"""
    model_lower = model.lower()
    
    model_map = {
        'rf': 'RF',
        'qrf': 'QRF',
        'xgboost': 'XGBoost',
        'ngboost': 'NGBoost',
        'dnn': 'DNN',
        'mlp': 'MLP',
        'gauche': 'GP',
        'svm': 'SVM',
        'gcn': 'GCN',
        'gin': 'GIN',
        'gat': 'GAT',
        'dnn_bnn_full': 'DNN-BNN-Full',
        'dnn_bnn_last': 'DNN-BNN-Last',
        'dnn_bnn_variational': 'DNN-BNN-Var',
        'mlp_bnn_full': 'MLP-BNN-Full',
        'mlp_bnn_last': 'MLP-BNN-Last',
        'mlp_bnn_variational': 'MLP-BNN-Var',
        'mtl': 'MTL',
        'mtl_bnn_full': 'MTL-BNN-Full',
        'mtl_bnn_last': 'MTL-BNN-Last',
        'mtl_bnn_variational': 'MTL-BNN-Var',
        'flexible_dnn': 'Flex-DNN',
        'flexible_bnn_full': 'Flex-BNN-Full',
        'flexible_bnn_last': 'Flex-BNN-Last',
        'flexible_bnn_variational': 'Flex-BNN-Var',
        'residual_mlp': 'Res-MLP',
        'residual_mlp_bnn_full': 'Res-MLP-BNN-Full',
        'residual_mlp_bnn_last': 'Res-MLP-BNN-Last',
        'residual_mlp_bnn_variational': 'Res-MLP-BNN-Var',
        'conformal_rf': 'Conf-RF',
        'conformal_qrf': 'Conf-QRF',
        'conformal_xgboost': 'Conf-XGB',
        'conformal_gauche': 'Conf-GP',
        'conformal_dnn': 'Conf-DNN',
    }
    
    if model_lower in model_map:
        return model_map[model_lower]
    
    result = model
    result = result.replace('_', '-')
    result = result.replace('bnn-full', 'BNN-Full')
    result = result.replace('bnn-last', 'BNN-Last')
    result = result.replace('bnn-variational', 'BNN-Var')
    
    return result

# ============================================================================
# DATA LOADING AND METRICS CALCULATION
# ============================================================================

def load_screening_results(results_dir="../results"):
    """Load Phase 0c screening results including MHGGNN files"""
    print("\n" + "="*80)
    print("LOADING PHASE 0C SCREENING DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    screening_files = list(results_dir.glob("phase0c_screen_*.csv"))
    mhggnn_files = list(results_dir.glob("phase0_mhggnn_*.csv"))
    
    all_files = screening_files + mhggnn_files
    
    if not all_files:
        print("ERROR: No phase0c_screen_*.csv or phase0_mhggnn_*.csv files found!")
        return pd.DataFrame()
    
    print(f"Found {len(screening_files)} standard screening files")
    print(f"Found {len(mhggnn_files)} MHGGNN files")
    print(f"Total: {len(all_files)} files")
    
    all_data = []
    for filepath in all_files:
        try:
            df = pd.read_csv(filepath)
            df['source_file'] = filepath.name
            
            if 'mhggnn' in filepath.name.lower():
                if 'rep' in df.columns:
                    df['rep'] = 'mhggnn'
                elif 'representation' in df.columns:
                    df['representation'] = 'mhggnn'
                else:
                    df['rep'] = 'mhggnn'
            
            if 'representation' in df.columns and 'rep' not in df.columns:
                df['rep'] = df['representation']
            
            all_data.append(df)
        except Exception as e:
            print(f"Warning: {filepath.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['model'] = combined_df['model'].str.replace('_split', '', regex=False)
    
    print(f"\nRaw data loaded: {len(combined_df)} rows")
    
    print("Filtering catastrophic failures...")
    print(f"  Before: {len(combined_df)} rows")
    combined_df = combined_df[combined_df['r2'] > -10]
    print(f"  After R² > -10 filter: {len(combined_df)} rows")
    
    baseline_check = combined_df[combined_df['sigma'] == 0.0].groupby(['model', 'rep'])['r2'].mean()
    bad_configs = baseline_check[baseline_check < 0.1].index
    
    if len(bad_configs) > 0:
        print(f"  Removing {len(bad_configs)} configs with baseline R² < 0.1")
        for model, rep in bad_configs:
            combined_df = combined_df[~((combined_df['model'] == model) & (combined_df['rep'] == rep))]
        print(f"  After baseline filter: {len(combined_df)} rows")
    
    results = combined_df.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'rep': 'representation', 'iteration': 'n_seeds'}, inplace=True)
    
    print("\nFiltering out problematic data...")
    before_filter = len(results)
    results = results[~results['representation'].isin(['graph'])]
    results = results[~results['model'].str.lower().isin(['gcn', 'gin', 'gat'])]
    print(f"  Removed {before_filter - len(results)} rows (graph rep, GNN models)")
    
    print(f"\nFinal aggregated data: {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    print(f"Representations found: {sorted(results['representation'].unique())}")
    print(f"Sigma values: {sorted(results['sigma'].unique())}")
    
    return results


def calculate_robustness_metrics(df, baseline_threshold=0.6):
    """
    Calculate robustness metrics for each model-representation pair.
    
    DS (Degradation Slope) is calculated for ALL configs initially,
    but ds_thresholded is only set for configs with baseline R² > threshold.
    """
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS")
    print(f"  Baseline threshold for DS: R² > {baseline_threshold}")
    print("="*80)
    
    metrics_list = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'model': model,
            'representation': rep,
        }
        
        # Baseline performance at σ=0
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # Performance at high noise (σ=0.6)
        sigma_h = group[np.abs(group['sigma'] - 0.6) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Calculate DS (Degradation Slope) for ALL configs
        if len(group) >= 3:
            try:
                slope_r2, intercept_r2, r_val_r2, p_val_r2, _ = stats.linregress(
                    group['sigma'], group['r2']
                )
                metrics['ds_r2'] = slope_r2  # Unthresholded DS
                metrics['ds_r2_pval'] = p_val_r2
                metrics['ds_r2_r'] = r_val_r2
                
                slope_rmse, _, r_val_rmse, p_val_rmse, _ = stats.linregress(
                    group['sigma'], group['rmse']
                )
                metrics['ds_rmse'] = slope_rmse
                metrics['ds_rmse_pval'] = p_val_rmse
                metrics['ds_rmse_r'] = r_val_rmse
                
            except Exception as e:
                metrics['ds_r2'] = np.nan
                metrics['ds_rmse'] = np.nan
        else:
            metrics['ds_r2'] = np.nan
            metrics['ds_rmse'] = np.nan
        
        # Thresholded DS: only valid if baseline R² > threshold
        if not np.isnan(metrics['baseline_r2']) and metrics['baseline_r2'] > baseline_threshold:
            metrics['ds_thresholded'] = metrics['ds_r2']
            metrics['meets_baseline_threshold'] = True
        else:
            metrics['ds_thresholded'] = np.nan
            metrics['meets_baseline_threshold'] = False
        
        # Performance at intermediate sigmas
        for sig in [0.2, 0.3, 0.4]:
            sigma_val = group[np.abs(group['sigma'] - sig) < 0.05]
            if len(sigma_val) > 0:
                metrics[f'r2_s{sig}'] = sigma_val['r2'].values[0]
                metrics[f'rmse_s{sig}'] = sigma_val['rmse'].values[0]
            else:
                metrics[f'r2_s{sig}'] = np.nan
                metrics[f'rmse_s{sig}'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Summary statistics
    n_total = len(metrics_df)
    n_meets_threshold = metrics_df['meets_baseline_threshold'].sum()
    n_below_threshold = n_total - n_meets_threshold
    
    print(f"\nCalculated metrics for {n_total} configurations")
    print(f"  Configs meeting baseline threshold (R² > {baseline_threshold}): {n_meets_threshold}")
    print(f"  Configs below threshold: {n_below_threshold}")
    
    print(f"\nBaseline R² statistics (all configs):")
    print(f"  Mean: {metrics_df['baseline_r2'].mean():.4f}")
    print(f"  Median: {metrics_df['baseline_r2'].median():.4f}")
    print(f"  Range: [{metrics_df['baseline_r2'].min():.4f}, {metrics_df['baseline_r2'].max():.4f}]")
    
    print(f"\nDS (R²) statistics - ALL configs (unthresholded):")
    print(f"  Mean: {metrics_df['ds_r2'].mean():.4f}")
    print(f"  Median: {metrics_df['ds_r2'].median():.4f}")
    
    print(f"\nDS (R²) statistics - THRESHOLDED (baseline R² > {baseline_threshold}):")
    thresholded_ds = metrics_df['ds_thresholded'].dropna()
    if len(thresholded_ds) > 0:
        print(f"  Mean: {thresholded_ds.mean():.4f}")
        print(f"  Median: {thresholded_ds.median():.4f}")
    else:
        print("  No configs meet threshold")
    
    return metrics_df


def define_robustness_score(metrics_df):
    """
    Define composite robustness score based on:
    - Baseline R² (higher is better)
    - DS magnitude (less negative / closer to 0 is better)
    
    Only uses thresholded DS for the score.
    """
    # Filter to only configs that meet threshold
    valid_mask = metrics_df['meets_baseline_threshold'] == True
    
    # For configs meeting threshold, calculate robustness score
    # Normalize baseline R² (0-1 scale)
    baseline_min = metrics_df.loc[valid_mask, 'baseline_r2'].min()
    baseline_max = metrics_df.loc[valid_mask, 'baseline_r2'].max()
    baseline_range = baseline_max - baseline_min if baseline_max != baseline_min else 1
    
    baseline_normalized = (metrics_df['baseline_r2'] - baseline_min) / baseline_range
    
    # Normalize DS (less negative is better, so we flip it)
    # DS is typically negative (performance degrades), so more negative = worse
    ds_vals = metrics_df.loc[valid_mask, 'ds_thresholded'].dropna()
    if len(ds_vals) > 0:
        ds_min = ds_vals.min()  # Most negative (worst)
        ds_max = ds_vals.max()  # Least negative (best)
        ds_range = ds_max - ds_min if ds_max != ds_min else 1
        
        # Higher score = better = less negative DS
        ds_normalized = (metrics_df['ds_thresholded'] - ds_min) / ds_range
    else:
        ds_normalized = pd.Series(np.nan, index=metrics_df.index)
    
    # Composite score: weighted average (equal weights)
    metrics_df['robustness_score'] = (baseline_normalized + ds_normalized) / 2
    
    # Set score to NaN for configs that don't meet threshold
    metrics_df.loc[~valid_mask, 'robustness_score'] = np.nan
    
    return metrics_df


def perform_variance_decomposition(df, output_dir):
    """ANOVA variance decomposition"""
    print("\n" + "="*80)
    print("PERFORMING ANOVA VARIANCE DECOMPOSITION")
    print("="*80)
    
    df_03 = df[np.abs(df['sigma'] - 0.3) < 0.05].copy()
    
    if len(df_03) == 0:
        print("⚠️  No data at σ=0.3 for ANOVA")
        return
    
    df_03 = df_03[df_03['r2'] > -10].dropna(subset=['r2', 'model', 'representation'])
    df_03 = df_03[~df_03['representation'].isin(['random_smiles', 'randomized_smiles'])].copy()
    
    print(f"\nData for ANOVA:")
    print(f"  {len(df_03)} observations")
    print(f"  {df_03['model'].nunique()} models")
    print(f"  {df_03['representation'].nunique()} representations")
    
    grand_mean = df_03['r2'].mean()
    total_ss = ((df_03['r2'] - grand_mean) ** 2).sum()
    
    model_means = df_03.groupby('model')['r2'].mean()
    model_counts = df_03.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)
    
    rep_means = df_03.groupby('representation')['r2'].mean()
    rep_counts = df_03.groupby('representation').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)
    
    interaction_means = df_03.groupby(['model', 'representation'])['r2'].mean()
    interaction_counts = df_03.groupby(['model', 'representation']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        cell_mean = interaction_means[(model, rep)]
        expected = model_means[model] + rep_means[rep] - grand_mean
        ss_interaction += count * (cell_mean - expected) ** 2
    
    ss_residual = total_ss - ss_model - ss_rep - ss_interaction
    
    n = len(df_03)
    df_model = df_03['model'].nunique() - 1
    df_rep = df_03['representation'].nunique() - 1
    df_interaction = df_model * df_rep
    df_residual = n - (df_model + 1) * (df_rep + 1)
    
    ms_model = ss_model / df_model if df_model > 0 else 0
    ms_rep = ss_rep / df_rep if df_rep > 0 else 0
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0
    ms_residual = ss_residual / df_residual if df_residual > 0 else 0
    
    f_model = ms_model / ms_residual if ms_residual > 0 else np.nan
    f_rep = ms_rep / ms_residual if ms_residual > 0 else np.nan
    f_interaction = ms_interaction / ms_residual if ms_residual > 0 else np.nan
    
    p_model = 1 - stats.f.cdf(f_model, df_model, df_residual) if not np.isnan(f_model) else np.nan
    p_rep = 1 - stats.f.cdf(f_rep, df_rep, df_residual) if not np.isnan(f_rep) else np.nan
    p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_residual) if not np.isnan(f_interaction) else np.nan
    
    eta2_model = ss_model / total_ss * 100
    eta2_rep = ss_rep / total_ss * 100
    eta2_interaction = ss_interaction / total_ss * 100
    eta2_residual = ss_residual / total_ss * 100
    
    print("\n" + "="*80)
    print("ANOVA TABLE (R² at σ=0.3)")
    print("="*80)
    print(f"{'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10} {'η² (%)':>10}")
    print("-"*80)
    print(f"{'Model':<20} {ss_model:>12.4f} {df_model:>6} {ms_model:>12.4f} {f_model:>10.2f} {p_model:>10.4f} {eta2_model:>10.2f}")
    print(f"{'Representation':<20} {ss_rep:>12.4f} {df_rep:>6} {ms_rep:>12.4f} {f_rep:>10.2f} {p_rep:>10.4f} {eta2_rep:>10.2f}")
    print(f"{'Interaction':<20} {ss_interaction:>12.4f} {df_interaction:>6} {ms_interaction:>12.4f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.2f}")
    print(f"{'Residual':<20} {ss_residual:>12.4f} {df_residual:>6} {ms_residual:>12.4f} {'':>10} {'':>10} {eta2_residual:>10.2f}")
    print("-"*80)
    print(f"{'Total':<20} {total_ss:>12.4f} {n-1:>6} {'':>12} {'':>10} {'':>10} {100.0:>10.2f}")
    print("="*80)
    
    output_path = Path(output_dir) / "anova_variance_decomposition.txt"
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("ANOVA VARIANCE DECOMPOSITION\n")
        f.write("Dependent Variable: R² at σ=0.3\n")
        f.write("="*80 + "\n\n")
        f.write(f"Data: {len(df_03)} observations, {df_03['model'].nunique()} models, {df_03['representation'].nunique()} representations\n\n")
        f.write("ANOVA TABLE\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Source':<20} {'SS':>12} {'df':>6} {'MS':>12} {'F':>10} {'p-value':>10} {'η² (%)':>10}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {ss_model:>12.4f} {df_model:>6} {ms_model:>12.4f} {f_model:>10.2f} {p_model:>10.4f} {eta2_model:>10.2f}\n")
        f.write(f"{'Representation':<20} {ss_rep:>12.4f} {df_rep:>6} {ms_rep:>12.4f} {f_rep:>10.2f} {p_rep:>10.4f} {eta2_rep:>10.2f}\n")
        f.write(f"{'Interaction':<20} {ss_interaction:>12.4f} {df_interaction:>6} {ms_interaction:>12.4f} {f_interaction:>10.2f} {p_interaction:>10.4f} {eta2_interaction:>10.2f}\n")
        f.write(f"{'Residual':<20} {ss_residual:>12.4f} {df_residual:>6} {ms_residual:>12.4f} {'':>10} {'':>10} {eta2_residual:>10.2f}\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Total':<20} {total_ss:>12.4f} {n-1:>6} {'':>12} {'':>10} {'':>10} {100.0:>10.2f}\n")
        f.write("="*80 + "\n\n")
        f.write("INTERPRETATION:\n")
        f.write(f"  - Model architecture explains {eta2_model:.1f}% of variance\n")
        f.write(f"  - Representation explains {eta2_rep:.1f}% of variance\n")
        f.write(f"  - Interaction explains {eta2_interaction:.1f}% of variance\n")
        f.write(f"  - Residual (unexplained) is {eta2_residual:.1f}% of variance\n")
    
    print(f"\n✓ Saved ANOVA results to {output_path}")
    
    return {
        'eta2_model': eta2_model,
        'eta2_representation': eta2_rep,
        'eta2_interaction': eta2_interaction,
        'eta2_residual': eta2_residual
    }


# ============================================================================
# THRESHOLD COMPARISON ANALYSIS (ONE-TIME)
# ============================================================================

def create_threshold_comparison_figure(metrics_df, output_dir):
    """
    ONE-TIME figure showing why baseline threshold matters for NDS.
    Highlights configs like NGBoost/SMILES with good NDS but poor baseline.
    """
    print("\n" + "="*80)
    print("GENERATING THRESHOLD COMPARISON FIGURE")
    print("(Showing why baseline R² > 0.6 threshold matters)")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Baseline R² vs NDS (unthresholded) - all configs
    ax_a = axes[0]
    
    # Color by whether meets threshold
    meets_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True]
    below_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == False]
    
    ax_a.scatter(below_thresh['baseline_r2'], below_thresh['ds_r2'],
                 alpha=0.6, s=60, c='#e74c3c', label='Below threshold (R² ≤ 0.6)',
                 edgecolors='black', linewidth=0.5, marker='x')
    ax_a.scatter(meets_thresh['baseline_r2'], meets_thresh['ds_r2'],
                 alpha=0.7, s=60, c='#2ecc71', label='Meets threshold (R² > 0.6)',
                 edgecolors='black', linewidth=0.5, marker='o')
    
    # Highlight problematic configs (good NDS but poor baseline)
    # These are configs with NDS > -0.5 (relatively flat) but baseline < 0.6
    problematic = below_thresh[(below_thresh['ds_r2'] > -0.5) & (below_thresh['baseline_r2'] < 0.6)]
    
    if len(problematic) > 0:
        ax_a.scatter(problematic['baseline_r2'], problematic['ds_r2'],
                     alpha=1.0, s=150, facecolors='none', edgecolors='#9b59b6',
                     linewidth=2.5, marker='o', label='Misleading: low degradation, poor baseline')
        
        # Annotate a few key examples
        for idx, (_, row) in enumerate(problematic.head(5).iterrows()):
            label = f"{format_model(row['model'])}/{format_representation(row['representation'])}"
            ax_a.annotate(label, (row['baseline_r2'], row['ds_r2']),
                         xytext=(10, 5), textcoords='offset points',
                         fontsize=7, color='#9b59b6', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    ax_a.axvline(0.6, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Threshold (R² = 0.6)')
    ax_a.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    ax_a.set_xlabel('Baseline R² (σ=0)', fontsize=10)
    ax_a.set_ylabel('Noise Degradation Slope', fontsize=10)
    ax_a.set_title('Why Threshold Matters: Baseline vs Noise Degradation Slope', fontsize=11, fontweight='bold', pad=12)
    ax_a.legend(fontsize=8, loc='lower left', frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Remove panel B (we'll output LaTeX table instead)
    ax_b = axes[1]
    ax_b.axis('off')
    ax_b.text(0.5, 0.5, 'See LaTeX table:\nthreshold_excluded_configs.tex', 
              ha='center', va='center', fontsize=12, style='italic',
              transform=ax_b.transAxes)
    
    # Create table data for LaTeX
    problematic_sorted = metrics_df[metrics_df['meets_baseline_threshold'] == False].copy()
    problematic_sorted['abs_ds'] = problematic_sorted['ds_r2'].abs()
    problematic_sorted = problematic_sorted.nsmallest(15, 'abs_ds')  # Top 15 least degradation
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "threshold_comparison_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved threshold comparison scatter to {output_path}")
    plt.close()
    
    # Generate LaTeX table
    latex_path = Path(output_dir) / "threshold_excluded_configs.tex"
    with open(latex_path, 'w') as f:
        f.write("% Configurations excluded from main analysis (baseline R² ≤ 0.6)\n")
        f.write("% These show low noise degradation but poor baseline performance\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Model-representation configurations excluded from noise robustness analysis due to poor baseline performance (R$^2 \\leq 0.6$). These configurations exhibit low noise degradation slopes but fail to achieve adequate predictive performance even under clean conditions. Note: Some excluded configurations (e.g., NGBoost/SMILES) may still provide valuable uncertainty estimates despite poor point predictions.}\n")
        f.write("\\label{tab:excluded_configs}\n")
        f.write("\\begin{tabular}{llcc}\n")
        f.write("\\toprule\n")
        f.write("Model & Representation & Baseline R$^2$ & Noise Degradation Slope \\\\\n")
        f.write("\\midrule\n")
        
        for _, row in problematic_sorted.iterrows():
            model_fmt = format_model(row['model'])
            rep_fmt = format_representation(row['representation'])
            f.write(f"{model_fmt} & {rep_fmt} & {row['baseline_r2']:.3f} & {row['ds_r2']:.4f} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table to {latex_path}")
    
    # Also save a text summary
    summary_path = Path(output_dir) / "threshold_comparison_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BASELINE THRESHOLD ANALYSIS\n")
        f.write("Why we require baseline R² > 0.6 for Noise Degradation Slope analysis\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total configurations: {len(metrics_df)}\n")
        f.write(f"Configs meeting threshold (R² > 0.6): {meets_thresh.shape[0]}\n")
        f.write(f"Configs below threshold: {below_thresh.shape[0]}\n\n")
        
        f.write("PROBLEM: Some configs show low degradation (slope close to 0) but have poor\n")
        f.write("baseline performance. Without thresholding, these would appear 'robust'\n")
        f.write("when in reality they simply never performed well to begin with.\n\n")
        
        f.write("NOTABLE EXAMPLES (excluded from main analysis):\n")
        f.write("-"*80 + "\n")
        for _, row in problematic_sorted.iterrows():
            f.write(f"  {format_model(row['model'])}/{format_representation(row['representation'])}: ")
            f.write(f"Baseline R²={row['baseline_r2']:.3f}, Slope={row['ds_r2']:.4f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("NOTE: These configurations may still be valuable for uncertainty quantification\n")
        f.write("(see Phase 2 analysis). NGBoost/SMILES in particular provides excellent\n")
        f.write("uncertainty estimates despite poor point predictions.\n")
    
    print(f"✓ Saved threshold summary to {summary_path}")


# ============================================================================
# FIGURE 1: GLOBAL NOISE ROBUSTNESS LANDSCAPE (THRESHOLDED)
# ============================================================================

def create_figure1_global_landscape(df, metrics_df, output_dir):
    """Figure 1: Global noise robustness landscape (using thresholded NDS)"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 1: GLOBAL ROBUSTNESS LANDSCAPE (VERTICAL LAYOUT)")
    print("="*80)
    
    # Filter to only configs meeting threshold
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    if len(metrics_thresh) == 0:
        print("ERROR: No configurations meet the baseline threshold!")
        return
    
    print(f"Using {len(metrics_thresh)} configurations (baseline R² > 0.6)")
    
    fig = plt.figure(figsize=(12, 18))
    gs = fig.add_gridspec(3, 1, hspace=0.35, wspace=0.20,
                          left=0.10, right=0.95, top=0.96, bottom=0.04)
    
    # PANEL A: Top 20 by robustness score
    ax_a = fig.add_subplot(gs[0, 0])
    top_20 = metrics_thresh.nlargest(20, 'robustness_score').sort_values('robustness_score')
    y_pos = np.arange(len(top_20))
    bar_colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in top_20['representation']]
    
    ax_a.barh(y_pos, top_20['robustness_score'], 
              color=bar_colors, alpha=0.8, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (idx, row) in enumerate(top_20.iterrows()):
        text = f"R²₀={row['baseline_r2']:.2f}, NDS={row['ds_thresholded']:.3f}"
        ax_a.text(row['robustness_score'] + 0.01, i, text, va='center', fontsize=7, color='black')
    
    ax_a.set_yticks(y_pos)
    labels = [f"{format_model(row['model'])}/{format_representation(row['representation'])}" 
              for _, row in top_20.iterrows()]
    ax_a.set_yticklabels(labels, fontsize=8)
    ax_a.set_xlabel('Robustness Score', fontsize=10)
    ax_a.set_title('A. Top 20 Most Noise-Robust Configurations (baseline R² > 0.6)', fontsize=11, fontweight='bold', pad=12)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.set_xlim(0, max(top_20['robustness_score']) * 1.25)
    ax_a.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # PANEL B: Bottom 20
    ax_b = fig.add_subplot(gs[1, 0])
    bottom_20 = metrics_thresh.nsmallest(20, 'robustness_score').sort_values('robustness_score', ascending=False)
    y_pos = np.arange(len(bottom_20))
    bar_colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in bottom_20['representation']]
    
    ax_b.barh(y_pos, bottom_20['robustness_score'], 
              color=bar_colors, alpha=0.8, height=0.75, edgecolor='black', linewidth=0.5)
    
    for i, (idx, row) in enumerate(bottom_20.iterrows()):
        text = f"R²₀={row['baseline_r2']:.2f}, NDS={row['ds_thresholded']:.3f}"
        ax_b.text(row['robustness_score'] + 0.01, i, text, va='center', fontsize=7, color='black')
    
    ax_b.set_yticks(y_pos)
    labels = [f"{format_model(row['model'])}/{format_representation(row['representation'])}" 
              for _, row in bottom_20.iterrows()]
    ax_b.set_yticklabels(labels, fontsize=8)
    ax_b.set_xlabel('Robustness Score', fontsize=10)
    ax_b.set_title('B. Bottom 20 Least Robust Configurations (baseline R² > 0.6)', fontsize=11, fontweight='bold', pad=12)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.set_xlim(0, max(bottom_20['robustness_score']) * 1.25)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # PANEL C: Heatmap of DS (thresholded)
    ax_c = fig.add_subplot(gs[2, 0])
    
    # Use absolute DS for heatmap (smaller = better)
    heatmap_data = metrics_thresh.pivot_table(
        index='model', columns='representation', values='ds_thresholded', aggfunc='mean'
    )
    # Sort by mean |DS| (least degradation first)
    heatmap_data = heatmap_data.loc[heatmap_data.abs().mean(axis=1).sort_values().index]
    
    formatted_models = [format_model(m) for m in heatmap_data.index]
    formatted_reps = [format_representation(r) for r in heatmap_data.columns]
    
    # Diverging colormap: red (more negative = worse) to green (closer to 0 = better)
    vmin = heatmap_data.min().min()
    vmax = 0  # DS should be ≤ 0
    
    im = ax_c.imshow(heatmap_data.values, aspect='auto', cmap='RdYlGn', 
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    
    ax_c.set_xticks(np.arange(len(heatmap_data.columns)))
    ax_c.set_yticks(np.arange(len(heatmap_data.index)))
    ax_c.set_xticklabels(formatted_reps, rotation=45, ha='right', fontsize=9)
    ax_c.set_yticklabels(formatted_models, fontsize=9)
    
    cbar = plt.colorbar(im, ax=ax_c, fraction=0.03, pad=0.04)
    cbar.set_label('Noise Degradation Slope', fontsize=9, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=8)
    
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < (vmin + vmax) / 2 else 'black'
                ax_c.text(j, i, f'{value:.2f}', ha='center', va='center',
                         color=text_color, fontsize=7, fontweight='bold')
    
    ax_c.set_title('C. Global Heatmap: Noise Degradation Slope (baseline R² > 0.6)', fontsize=10, fontweight='bold', pad=10)
    ax_c.set_xlabel('Representation', fontsize=9)
    ax_c.set_ylabel('Model', fontsize=9)
    
    output_path = Path(output_dir) / "figure1_global_robustness_landscape.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 1 to {output_path}")
    plt.close()


def create_s3_degradation_curves(df, metrics_df, output_dir):
    """
    S3: R² degradation curves for top 5 vs bottom 5 (thresholded)
    """
    print("\n" + "="*80)
    print("GENERATING S3: DEGRADATION CURVES")
    print("="*80)
    
    # Use only thresholded configs
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # Select top 5 and bottom 5 by robustness score
    top_5 = metrics_thresh.nlargest(5, 'robustness_score')
    bottom_5 = metrics_thresh.nsmallest(5, 'robustness_score')
    
    # Plot top configurations (solid lines)
    for idx, (_, row) in enumerate(top_5.iterrows()):
        model, rep = row['model'], row['representation']
        pair_data = df[(df['model'] == model) & (df['representation'] == rep)].sort_values('sigma')
        
        if len(pair_data) == 0:
            continue
        
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        label = f"{format_model(model)}/{format_representation(rep)} (top #{idx+1})"
        ax.plot(pair_data['sigma'], pair_data['r2'], 
                marker='o', markersize=5, linewidth=2.5, alpha=0.9,
                label=label, color=color, linestyle='-')
        
        # Annotate noise degradation slope on the line
        mid_idx = len(pair_data) // 2
        if mid_idx < len(pair_data):
            ax.text(pair_data.iloc[mid_idx]['sigma'], pair_data.iloc[mid_idx]['r2'] + 0.02, 
                    f"NDS={row['ds_thresholded']:.3f}", fontsize=7, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              alpha=0.8, edgecolor=color, linewidth=1))
    
    # Plot bottom configurations (dashed lines)
    for idx, (_, row) in enumerate(bottom_5.iterrows()):
        model, rep = row['model'], row['representation']
        pair_data = df[(df['model'] == model) & (df['representation'] == rep)].sort_values('sigma')
        
        if len(pair_data) == 0:
            continue
        
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        label = f"{format_model(model)}/{format_representation(rep)} (bottom #{idx+1})"
        ax.plot(pair_data['sigma'], pair_data['r2'], 
                marker='s', markersize=5, linewidth=2, alpha=0.7,
                label=label, color=color, linestyle='--')
        
        # Annotate noise degradation slope
        mid_idx = len(pair_data) // 2
        if mid_idx < len(pair_data):
            ax.text(pair_data.iloc[mid_idx]['sigma'], pair_data.iloc[mid_idx]['r2'] - 0.03, 
                    f"NDS={row['ds_thresholded']:.3f}", fontsize=7, color=color, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                              alpha=0.8, edgecolor=color, linewidth=1))
    
    ax.set_xlabel('Noise level (σ)', fontsize=10)
    ax.set_ylabel('R² score', fontsize=10)
    ax.set_title('R² Degradation Curves: Top 5 vs Bottom 5 Configurations\n(Ranked by robustness score, baseline R² > 0.6)', 
                 fontsize=11, fontweight='bold', pad=12)
    ax.legend(fontsize=7, loc='lower left', ncol=2, frameon=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax.set_ylim(bottom=0)
    
    output_path = Path(output_dir) / "s3_degradation_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved S3 degradation curves to {output_path}")
    plt.close()


def create_figure2_representation_effects(df, metrics_df, output_dir):
    """Figure 2: Representation and noise-type effects on robustness"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 2: REPRESENTATION EFFECTS")
    print("="*80)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.35,
                          left=0.08, right=0.98, top=0.96, bottom=0.06)
    
    # PANEL A: Performance at σ=0.3 by representation
    ax_a = fig.add_subplot(gs[0, 0])
    
    r2_at_03 = []
    for (model, rep), group in df.groupby(['model', 'representation']):
        # Only include if this config meets threshold
        if not ((metrics_df['model'] == model) & (metrics_df['representation'] == rep) & 
                (metrics_df['meets_baseline_threshold'] == True)).any():
            continue
        sigma_03 = group[np.abs(group['sigma'] - 0.3) < 0.05]
        if len(sigma_03) > 0:
            r2_at_03.append({'model': model, 'representation': rep, 'r2_at_03': sigma_03['r2'].values[0]})
    r2_at_03_df = pd.DataFrame(r2_at_03)
    
    if len(r2_at_03_df) > 0:
        rep_order = r2_at_03_df.groupby('representation')['r2_at_03'].median().sort_values(ascending=False).index
        
        valid_reps = []
        valid_data = []
        for rep in rep_order:
            data = r2_at_03_df[r2_at_03_df['representation'] == rep]['r2_at_03'].dropna()
            if len(data) >= 2:
                valid_reps.append(rep)
                valid_data.append(data)
        
        if len(valid_data) > 0:
            parts = ax_a.violinplot(valid_data, positions=range(len(valid_reps)), widths=0.7, showmeans=True, showmedians=True)
            
            for i, pc in enumerate(parts['bodies']):
                rep = valid_reps[i]
                color = REPRESENTATION_COLORS.get(rep, '#999999')
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(0.8)
            
            ax_a.set_xticks(range(len(valid_reps)))
            ax_a.set_xticklabels([format_representation(r) for r in valid_reps], rotation=45, ha='right')
    
    ax_a.set_ylabel('R² at σ=0.3', fontsize=9)
    ax_a.set_title('A. Performance at Moderate Noise', fontsize=10, fontweight='bold', pad=10)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    ax_a.set_ylim(bottom=0)
    
    # PANEL B: DS by representation (thresholded)
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Sort by median |DS| (least degradation first)
    rep_order_ds = metrics_thresh.groupby('representation')['ds_thresholded'].apply(lambda x: x.abs().median()).sort_values().index
    
    valid_reps_ds = []
    valid_data_ds = []
    for rep in rep_order_ds:
        data = metrics_thresh[metrics_thresh['representation'] == rep]['ds_thresholded'].dropna()
        if len(data) >= 2:
            valid_reps_ds.append(rep)
            valid_data_ds.append(data)
    
    if len(valid_data_ds) > 0:
        parts = ax_b.violinplot(valid_data_ds, positions=range(len(valid_reps_ds)), widths=0.7, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            rep = valid_reps_ds[i]
            color = REPRESENTATION_COLORS.get(rep, '#999999')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        
        ax_b.set_xticks(range(len(valid_reps_ds)))
        ax_b.set_xticklabels([format_representation(r) for r in valid_reps_ds], rotation=45, ha='right')
    
    ax_b.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax_b.set_ylabel('Noise Degradation Slope', fontsize=9)
    ax_b.set_title('B. Noise Degradation Slope by Representation', fontsize=10, fontweight='bold', pad=10)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # PANEL C: |DS| ranking (degradation rate)
    ax_c = fig.add_subplot(gs[0, 2])
    
    ds_ranking = metrics_thresh.groupby('representation').agg({
        'ds_thresholded': lambda x: np.abs(x).mean()
    }).reset_index()
    ds_ranking.columns = ['representation', 'mean_abs_ds']
    ds_ranking = ds_ranking.sort_values('mean_abs_ds', ascending=True)  # Least degradation first
    
    colors = [REPRESENTATION_COLORS.get(rep, '#999999') for rep in ds_ranking['representation']]
    ax_c.barh(range(len(ds_ranking)), ds_ranking['mean_abs_ds'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.8)
    ax_c.set_yticks(range(len(ds_ranking)))
    ax_c.set_yticklabels([format_representation(r) for r in ds_ranking['representation']])
    ax_c.set_xlabel('Average |Noise Degradation Slope| (lower = more stable)', fontsize=9)
    ax_c.set_title('C. Representation Stability Ranking', fontsize=10, fontweight='bold', pad=10)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # PANEL D: Noise-by-representation interaction
    ax_d = fig.add_subplot(gs[1, 0])
    
    # Only plot representations that have configs meeting threshold
    valid_reps_for_curves = metrics_thresh['representation'].unique()
    
    for rep in sorted(valid_reps_for_curves):
        # Get models that meet threshold for this rep
        valid_models = metrics_thresh[metrics_thresh['representation'] == rep]['model'].unique()
        rep_data = df[(df['representation'] == rep) & (df['model'].isin(valid_models))]
        avg_by_sigma = rep_data.groupby('sigma')['r2'].mean().reset_index()
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax_d.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'], marker='o', markersize=4, 
                  linewidth=2, alpha=0.8, label=format_representation(rep), color=color)
    
    ax_d.set_xlabel('Noise level (σ)', fontsize=9)
    ax_d.set_ylabel('Average R² (threshold-meeting configs)', fontsize=9)
    ax_d.set_title('D. Representation Performance Across Noise Levels', fontsize=10, fontweight='bold', pad=10)
    ax_d.legend(fontsize=7, loc='best', frameon=True, fancybox=True, framealpha=0.9, ncol=2)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # PANEL E: SNS Top Performers
    ax_e = fig.add_subplot(gs[1, 1])
    target_models = ['xgboost', 'ngboost', 'rf', 'gauche']
    
    for model in target_models:
        # Check if this model/sns meets threshold
        if not ((metrics_df['model'] == model) & (metrics_df['representation'] == 'sns') & 
                (metrics_df['meets_baseline_threshold'] == True)).any():
            continue
        model_data = df[(df['model'] == model) & (df['representation'] == 'sns')]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#999999')
            ax_e.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'], marker='o', markersize=5, 
                      linewidth=2.5, alpha=0.9, label=format_model(model), color=color)
    
    ax_e.set_xlabel('Noise level (σ)', fontsize=9)
    ax_e.set_ylabel('R²', fontsize=9)
    ax_e.set_title('E. SNS: Top Performer Comparison', fontsize=10, fontweight='bold', pad=10)
    ax_e.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_e.set_ylim(bottom=0)
    
    # PANEL F: PDV Top Performers
    ax_f = fig.add_subplot(gs[1, 2])
    
    for model in target_models:
        # Check if this model/pdv meets threshold
        if not ((metrics_df['model'] == model) & (metrics_df['representation'] == 'pdv') & 
                (metrics_df['meets_baseline_threshold'] == True)).any():
            continue
        model_data = df[(df['model'] == model) & (df['representation'] == 'pdv')]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#999999')
            ax_f.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'], marker='o', markersize=5, 
                      linewidth=2.5, alpha=0.9, label=format_model(model), color=color)
    
    ax_f.set_xlabel('Noise level (σ)', fontsize=9)
    ax_f.set_ylabel('R²', fontsize=9)
    ax_f.set_title('F. PDV: Top Performer Comparison', fontsize=10, fontweight='bold', pad=10)
    ax_f.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_f.set_ylim(bottom=0)
    
    output_path = Path(output_dir) / "figure2_representation_effects.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 2 to {output_path}")
    plt.close()


def create_s1_baseline_vs_nds(metrics_df, output_dir):
    """S1: Baseline R² vs NDS scatter (thresholded only)"""
    print("\nGENERATING S1: BASELINE VS NDS")
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    representations = metrics_thresh['representation'].unique()
    for rep in representations:
        rep_data = metrics_thresh[metrics_thresh['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['ds_thresholded'],
                  alpha=0.7, s=60, color=color, label=format_representation(rep),
                  edgecolors='black', linewidth=0.5)
    
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0.6, color='red', linestyle=':', linewidth=1, alpha=0.5, label='Threshold')
    
    ax.set_xlabel('Baseline R² (σ=0)', fontsize=10)
    ax.set_ylabel('Noise Degradation Slope', fontsize=10)
    ax.set_title('Baseline Performance vs Noise Degradation Slope\n(baseline R² > 0.6 only)', 
                 fontsize=11, fontweight='bold', pad=15)
    ax.legend(fontsize=8, loc='best', ncol=2, frameon=True, fancybox=True, framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "s1_baseline_vs_nds.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved S1 to {output_path}")
    plt.close()


def create_supplementary_nds_distributions(metrics_df, output_dir):
    """S2: NDS distributions (thresholded)"""
    print("\nGENERATING S2: NDS DISTRIBUTIONS")
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    
    # Panel A: Histogram of Noise Degradation Slope
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.hist(metrics_thresh['ds_thresholded'].dropna(), bins=30, color='#3498db', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_a.axvline(metrics_thresh['ds_thresholded'].median(), color='red', linestyle='--', linewidth=2, 
                 label=f'Median = {metrics_thresh["ds_thresholded"].median():.4f}')
    ax_a.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_a.set_xlabel('Noise Degradation Slope', fontsize=9)
    ax_a.set_ylabel('Frequency', fontsize=9)
    ax_a.set_title('A. Distribution of Noise Degradation Slope', fontsize=10, fontweight='bold')
    ax_a.legend(fontsize=8)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel B: DS by representation
    ax_b = fig.add_subplot(gs[0, 1])
    rep_order = metrics_thresh.groupby('representation')['ds_thresholded'].apply(lambda x: np.abs(x).median()).sort_values().index
    
    valid_reps = []
    valid_data = []
    for rep in rep_order:
        data = metrics_thresh[metrics_thresh['representation'] == rep]['ds_thresholded'].dropna()
        if len(data) >= 2:
            valid_reps.append(rep)
            valid_data.append(data)
    
    if len(valid_data) > 0:
        parts = ax_b.violinplot(valid_data, positions=range(len(valid_reps)), widths=0.7, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            rep = valid_reps[i]
            color = REPRESENTATION_COLORS.get(rep, '#999999')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        
        ax_b.set_xticks(range(len(valid_reps)))
        ax_b.set_xticklabels([format_representation(r) for r in valid_reps], rotation=45, ha='right')
    
    ax_b.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_b.set_ylabel('Noise Degradation Slope', fontsize=9)
    ax_b.set_title('B. Noise Degradation Slope by Representation', fontsize=10, fontweight='bold')
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel C: Histogram of Noise Degradation Slope (RMSE)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.hist(metrics_thresh['ds_rmse'].dropna(), bins=30, color='#e74c3c', alpha=0.7, edgecolor='black', linewidth=0.8)
    ax_c.axvline(metrics_thresh['ds_rmse'].median(), color='blue', linestyle='--', linewidth=2, 
                 label=f'Median = {metrics_thresh["ds_rmse"].median():.4f}')
    ax_c.set_xlabel('Noise Degradation Slope (RMSE)', fontsize=9)
    ax_c.set_ylabel('Frequency', fontsize=9)
    ax_c.set_title('C. Distribution of RMSE Noise Degradation Slope', fontsize=10, fontweight='bold')
    ax_c.legend(fontsize=8)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Panel D: DS by model (top 10)
    ax_d = fig.add_subplot(gs[1, 1])
    model_order = metrics_thresh.groupby('model')['ds_thresholded'].apply(lambda x: np.abs(x).median()).sort_values().index[:10]
    
    valid_models = []
    valid_data_models = []
    for model in model_order:
        data = metrics_thresh[metrics_thresh['model'] == model]['ds_thresholded'].dropna()
        if len(data) >= 2:
            valid_models.append(model)
            valid_data_models.append(data)
    
    if len(valid_data_models) > 0:
        parts = ax_d.violinplot(valid_data_models, positions=range(len(valid_models)), widths=0.7, showmeans=True, showmedians=True)
        
        for i, pc in enumerate(parts['bodies']):
            model = valid_models[i]
            color = MODEL_COLORS.get(model, '#999999')
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(0.8)
        
        ax_d.set_xticks(range(len(valid_models)))
        ax_d.set_xticklabels([format_model(m) for m in valid_models], rotation=45, ha='right', fontsize=7)
    
    ax_d.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax_d.set_ylabel('Noise Degradation Slope', fontsize=9)
    ax_d.set_title('D. Noise Degradation Slope by Model (Top 10 most stable)', fontsize=10, fontweight='bold')
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "s2_nds_distributions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved S2 to {output_path}")
    plt.close()


def create_summary_tables(metrics_df, output_dir):
    """Create summary tables"""
    print("\nGENERATING SUMMARY TABLES")
    output_dir = Path(output_dir)
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    # Table 1: Top 20 (thresholded)
    table1 = metrics_thresh.nlargest(20, 'robustness_score')[
        ['model', 'representation', 'baseline_r2', 'r2_high', 'ds_thresholded', 'robustness_score']
    ].copy()
    table1.columns = ['Model', 'Representation', 'Baseline R²', 'R² (high noise)', 'Noise Degradation Slope', 'Robustness Score']
    table1.to_csv(output_dir / "table1_top20_robust_configs.csv", index=False, float_format='%.4f')
    print(f"✓ Saved Table 1")
    
    # Table 2: By representation (thresholded)
    table2 = metrics_thresh.groupby('representation').agg({
        'baseline_r2': ['mean', 'std', 'median'],
        'ds_thresholded': ['mean', 'std', 'median'],
        'robustness_score': ['mean', 'std']
    }).round(4)
    table2.to_csv(output_dir / "table2_performance_by_representation.csv")
    print(f"✓ Saved Table 2")
    
    # Table 3: By model (thresholded)
    table3 = metrics_thresh.groupby('model').agg({
        'baseline_r2': ['mean', 'std', 'median'],
        'ds_thresholded': ['mean', 'std', 'median'],
        'robustness_score': ['mean', 'std']
    }).round(4)
    table3 = table3.sort_values(('robustness_score', 'mean'), ascending=False).head(20)
    table3.to_csv(output_dir / "table3_performance_by_model.csv")
    print(f"✓ Saved Table 3")
    
    # Table 4: All configs with threshold status
    table4 = metrics_df[['model', 'representation', 'baseline_r2', 'ds_r2', 'ds_thresholded', 
                         'meets_baseline_threshold', 'robustness_score']].copy()
    table4.to_csv(output_dir / "table4_all_configs_with_threshold.csv", index=False, float_format='%.4f')
    print(f"✓ Saved Table 4 (all configs)")


def perform_statistical_tests(df, metrics_df, output_dir):
    """Perform statistical tests"""
    print("\nPERFORMING STATISTICAL TESTS")
    output_dir = Path(output_dir)
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    results_text = ["STATISTICAL COMPARISONS - PHASE 0", "="*80, ""]
    results_text.append("Note: All comparisons use thresholded data (baseline R² > 0.6)")
    results_text.append("")
    
    results_text.append("TEST 1: Representation Comparisons (Noise Degradation Slope)")
    results_text.append("-"*80)
    
    reps = metrics_thresh['representation'].value_counts().head(5).index
    
    for i in range(len(reps)):
        for j in range(i+1, len(reps)):
            rep1, rep2 = reps[i], reps[j]
            data1 = metrics_thresh[metrics_thresh['representation'] == rep1]['ds_thresholded'].dropna()
            data2 = metrics_thresh[metrics_thresh['representation'] == rep2]['ds_thresholded'].dropna()
            
            if len(data1) >= 3 and len(data2) >= 3:
                from scipy.stats import mannwhitneyu
                stat, p_val = mannwhitneyu(data1.abs(), data2.abs(), alternative='two-sided')
                mean1, mean2 = data1.abs().mean(), data2.abs().mean()
                
                results_text.append(f"{format_representation(rep1)} vs {format_representation(rep2)}:")
                results_text.append(f"  Mean |Noise Degradation Slope|: {mean1:.4f} vs {mean2:.4f}")
                results_text.append(f"  Mann-Whitney U: stat={stat:.2f}, p={p_val:.4f}")
                
                if p_val < 0.05:
                    winner = rep1 if mean1 < mean2 else rep2  # Lower |DS| is better
                    results_text.append(f"  → Significant difference, {format_representation(winner)} more stable")
                else:
                    results_text.append(f"  → No significant difference")
                results_text.append("")
    
    output_path = output_dir / "statistical_tests_phase0.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_text))
    print(f"✓ Saved statistical tests to {output_path}")


def perform_nds_anova(metrics_df, output_dir):
    """ANOVA variance decomposition for Noise Degradation Slope"""
    print("\n" + "="*80)
    print("ANOVA VARIANCE DECOMPOSITION - NOISE DEGRADATION SLOPE")
    print("="*80)
    
    # Use thresholded data
    analysis_df = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    analysis_df = analysis_df.dropna(subset=['ds_thresholded', 'model', 'representation'])
    analysis_df = analysis_df[~analysis_df['representation'].isin(['random_smiles', 'randomized_smiles'])].copy()
    analysis_df['abs_ds'] = analysis_df['ds_thresholded'].abs()
    
    print(f"\nData for ANOVA: {len(analysis_df)} observations (thresholded)")
    
    dependent_vars = {
        'Noise Degradation Slope': 'ds_thresholded',
        '|Noise Degradation Slope|': 'abs_ds'
    }
    
    all_results = {}
    output_text = ["="*80, "ANOVA VARIANCE DECOMPOSITION - NOISE DEGRADATION SLOPE METRICS", 
                   "(Baseline R² > 0.6 threshold applied)", "="*80, ""]
    
    for metric_name, metric_col in dependent_vars.items():
        y_values = analysis_df[metric_col]
        grand_mean = y_values.mean()
        total_ss = ((y_values - grand_mean) ** 2).sum()
        
        model_means = analysis_df.groupby('model')[metric_col].mean()
        model_counts = analysis_df.groupby('model').size()
        ss_model = sum(model_counts * (model_means - grand_mean) ** 2)
        
        rep_means = analysis_df.groupby('representation')[metric_col].mean()
        rep_counts = analysis_df.groupby('representation').size()
        ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)
        
        interaction_means = analysis_df.groupby(['model', 'representation'])[metric_col].mean()
        interaction_counts = analysis_df.groupby(['model', 'representation']).size()
        ss_interaction = 0
        for (model, rep), count in interaction_counts.items():
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2
        
        ss_residual = total_ss - ss_model - ss_rep - ss_interaction
        
        eta2_model = (ss_model / total_ss * 100) if total_ss > 0 else 0
        eta2_rep = (ss_rep / total_ss * 100) if total_ss > 0 else 0
        eta2_interaction = (ss_interaction / total_ss * 100) if total_ss > 0 else 0
        eta2_residual = (ss_residual / total_ss * 100) if total_ss > 0 else 0
        
        all_results[metric_name] = {
            'eta2_model': eta2_model,
            'eta2_representation': eta2_rep,
            'eta2_interaction': eta2_interaction,
            'eta2_residual': eta2_residual
        }
        
        output_text.append(f"\nMETRIC: {metric_name}")
        output_text.append(f"  Model: {eta2_model:.1f}%")
        output_text.append(f"  Representation: {eta2_rep:.1f}%")
        output_text.append(f"  Interaction: {eta2_interaction:.1f}%")
        output_text.append(f"  Residual: {eta2_residual:.1f}%")
    
    output_path = Path(output_dir) / "anova_nds_decomposition.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(output_text))
    print(f"✓ Saved Noise Degradation Slope ANOVA to {output_path}")
    
    return all_results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(results_dir="../results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 0 ANALYSIS - GLOBAL SCREENING")
    print("Journal of Cheminformatics Style")
    print("="*80)
    print("\nKey changes in this version:")
    print("  - Retention metric REMOVED")
    print("  - NSI renamed to Noise Degradation Slope (NDS)")
    print("  - NDS thresholded: only calculated for baseline R² > 0.6")
    print("="*80)
    
    df = load_screening_results(results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    metrics_df = calculate_robustness_metrics(df, baseline_threshold=0.6)
    metrics_df = define_robustness_score(metrics_df)
    
    output_dir = Path(results_dir) / "phase0_figures_v3"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_df.to_csv(output_dir / "phase0_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase0_robustness_metrics.csv'}")
    
    # ONE-TIME: Create threshold comparison figure
    create_threshold_comparison_figure(metrics_df, output_dir)
    
    perform_variance_decomposition(df, output_dir)
    perform_nds_anova(metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("GENERATING FIGURES (using thresholded NDS)")
    print("="*80)
    
    create_figure1_global_landscape(df, metrics_df, output_dir)
    create_s3_degradation_curves(df, metrics_df, output_dir)
    create_figure2_representation_effects(df, metrics_df, output_dir)
    create_s1_baseline_vs_nds(metrics_df, output_dir)
    create_supplementary_nds_distributions(metrics_df, output_dir)
    create_summary_tables(metrics_df, output_dir)
    perform_statistical_tests(df, metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True]
    
    print(f"\nTotal configs: {len(metrics_df)}")
    print(f"Configs meeting threshold (R² > 0.6): {len(metrics_thresh)}")
    
    print("\nTop 10 Most Robust Configurations (thresholded):")
    for idx, (_, row) in enumerate(metrics_thresh.nlargest(10, 'robustness_score').iterrows(), 1):
        print(f"  {idx}. {format_model(row['model'])}/{format_representation(row['representation'])}: "
              f"Score={row['robustness_score']:.3f}, R²₀={row['baseline_r2']:.3f}, NDS={row['ds_thresholded']:.4f}")
    
    print("\nRepresentation Rankings (by median |NDS|, lower = more stable):")
    rep_ranking = metrics_thresh.groupby('representation')['ds_thresholded'].apply(lambda x: x.abs().median()).sort_values()
    for idx, (rep, val) in enumerate(rep_ranking.items(), 1):
        print(f"  {idx}. {format_representation(rep)}: |NDS|={val:.4f}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)