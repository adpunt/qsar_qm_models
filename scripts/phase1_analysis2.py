"""
Phase 1 Analysis - Deterministic vs Probabilistic Models
=========================================================

Pulls data from Phase 0 screening results and analyzes:
- Deterministic models: RF, XGBoost, DNN, MLP
- Probabilistic models: QRF, NGBoost, GP/GAUCHE, BNN variants

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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# JOURNAL OF CHEMINFORMATICS STYLE
# ============================================================================

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
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

COLORS = {
    'deterministic': '#0173B2',
    'probabilistic': '#DE8F05',
}

REPRESENTATION_COLORS = {
    'pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'smiles': '#CA3542',
    'mhggnn': '#CC79A7',
    'graph': '#949494',
}

MODEL_COLORS = {
    'rf': '#3498db',
    'qrf': '#2ecc71',
    'xgboost': '#e74c3c',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'mlp': '#2c3e50',
    'gauche': '#9b59b6',
    'dnn_bnn_full': '#e67e22',
    'dnn_bnn_last': '#95a5a6',
    'dnn_bnn_variational': '#d35400',
    'mlp_bnn_full': '#8e44ad',
    'mlp_bnn_last': '#7f8c8d',
    'mlp_bnn_variational': '#c0392b',
}

# ============================================================================
# FORMATTING
# ============================================================================

def format_representation(rep):
    mapping = {
        'pdv': 'PDV',
        'sns': 'SNS',
        'ecfp4': 'ECFP4',
        'smiles': 'SMILES',
        'randomized_smiles': 'R-SMILES',
        'random_smiles': 'R-SMILES',
        'mhggnn': 'MHGGNN',
        'graph': 'Graph',
    }
    return mapping.get(rep.lower(), rep.upper())


def format_model(model):
    """Format model names for display"""
    model_lower = model.lower()
    
    mapping = {
        'rf': 'RF',
        'qrf': 'QRF',
        'xgboost': 'XGBoost',
        'ngboost': 'NGBoost',
        'dnn': 'DNN',
        'mlp': 'MLP',
        'gauche': 'GP',
        'svm': 'SVM',
        # BNN variants
        'dnn_bnn_full': 'DNN-BNN-Full',
        'dnn_bnn_last': 'DNN-BNN-Last',
        'dnn_bnn_variational': 'DNN-BNN-Var',
        'mlp_bnn_full': 'MLP-BNN-Full',
        'mlp_bnn_last': 'MLP-BNN-Last',
        'mlp_bnn_variational': 'MLP-BNN-Var',
        # MTL variants
        'mtl': 'MTL',
        'mtl_bnn_full': 'MTL-BNN-Full',
        'mtl_bnn_last': 'MTL-BNN-Last',
        'mtl_bnn_variational': 'MTL-BNN-Var',
        # Flexible variants
        'flexible_dnn': 'Flex-DNN',
        'flexible_bnn_full': 'Flex-BNN-Full',
        'flexible_bnn_last': 'Flex-BNN-Last',
        'flexible_bnn_variational': 'Flex-BNN-Var',
        # Residual variants
        'residual_mlp': 'Res-MLP',
        'residual_mlp_bnn_full': 'Res-MLP-BNN-Full',
        'residual_mlp_bnn_last': 'Res-MLP-BNN-Last',
        'residual_mlp_bnn_variational': 'Res-MLP-BNN-Var',
        # Conformal variants
        'conformal_rf': 'Conf-RF',
        'conformal_qrf': 'Conf-QRF',
        'conformal_xgboost': 'Conf-XGB',
        'conformal_gauche': 'Conf-GP',
        'conformal_dnn': 'Conf-DNN',
    }
    
    if model_lower in mapping:
        return mapping[model_lower]
    
    # Handle any remaining patterns
    result = model.replace('_', '-')
    return result


def get_model_type(model_name):
    """Classify model as deterministic or probabilistic"""
    model_lower = model_name.lower()
    
    # Probabilistic models
    prob_keywords = ['qrf', 'ngboost', 'gauche', 'bnn', 'gp']
    if any(kw in model_lower for kw in prob_keywords):
        return 'probabilistic'
    
    return 'deterministic'


def get_model_pair(model_name):
    """Get the deterministic/probabilistic pair for a model"""
    model_lower = model_name.lower()
    
    pairs = {
        'rf': ('rf', 'qrf'),
        'qrf': ('rf', 'qrf'),
        'xgboost': ('xgboost', 'ngboost'),
        'ngboost': ('xgboost', 'ngboost'),
        'dnn': ('dnn', 'dnn_bnn'),
        'mlp': ('mlp', 'mlp_bnn'),
    }
    
    for key, pair in pairs.items():
        if key in model_lower:
            return pair
    
    return (model_name, None)

# ============================================================================
# DATA LOADING - FROM PHASE 0 DATA
# ============================================================================

def load_phase0_data(results_dir="../results"):
    """Load Phase 0 screening data for Phase 1 analysis"""
    print("\n" + "="*80)
    print("LOADING PHASE 0 DATA FOR PHASE 1 ANALYSIS")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Load all Phase 0 file patterns
    screening_files = list(results_dir.glob("phase0c_screen_*.csv"))
    mhggnn_files = list(results_dir.glob("phase0_mhggnn_*.csv"))
    continuous_pdv_files = list(results_dir.glob("phase0_continuous_pdv_*.csv"))
    
    all_files = screening_files + mhggnn_files + continuous_pdv_files
    
    if not all_files:
        print("ERROR: No phase0 files found!")
        return pd.DataFrame()
    
    print(f"Found {len(screening_files)} standard screening files")
    print(f"Found {len(mhggnn_files)} MHGGNN files")
    print(f"Found {len(continuous_pdv_files)} continuous PDV files")
    print(f"Total: {len(all_files)} files")
    
    all_data = []
    for filepath in all_files:
        try:
            df = pd.read_csv(filepath)
            df['source_file'] = filepath.name
            
            # For MHGGNN files, set representation to 'mhggnn'
            if 'mhggnn' in filepath.name.lower():
                if 'rep' in df.columns:
                    df['rep'] = 'mhggnn'
                elif 'representation' in df.columns:
                    df['representation'] = 'mhggnn'
                else:
                    df['rep'] = 'mhggnn'
            
            # For continuous PDV files, set representation to 'pdv'
            if 'continuous_pdv' in filepath.name.lower():
                if 'rep' in df.columns:
                    df['rep'] = 'pdv'
                elif 'representation' in df.columns:
                    df['representation'] = 'pdv'
                else:
                    df['rep'] = 'pdv'
            
            # Ensure 'rep' column exists (standardize from 'representation' if needed)
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
    
    # Filter catastrophic failures
    combined_df = combined_df[combined_df['r2'] > -10]
    print(f"After R² > -10 filter: {len(combined_df)} rows")
    
    # Add model_type classification
    combined_df['model_type'] = combined_df['model'].apply(get_model_type)
    
    # Aggregate across iterations using 'rep' column (same as phase0)
    results = combined_df.groupby(['model', 'rep', 'sigma', 'model_type']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    # Now rename 'rep' to 'representation' for consistency
    results.rename(columns={'rep': 'representation', 'iteration': 'n_seeds'}, inplace=True)
    
    # Filter out problematic data
    print("\nFiltering out problematic data...")
    before_filter = len(results)
    results = results[~results['representation'].isin(['graph'])]
    results = results[~results['model'].str.lower().isin(['gcn', 'gin', 'gat'])]
    print(f"  Removed {before_filter - len(results)} rows (graph rep, GNN models)")
    
    print(f"\nFinal aggregated data: {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    print(f"Representations: {sorted(results['representation'].unique())}")
    
    # Model type breakdown
    det_count = len(results[results['model_type'] == 'deterministic']['model'].unique())
    prob_count = len(results[results['model_type'] == 'probabilistic']['model'].unique())
    print(f"\nModel types:")
    print(f"  Deterministic: {det_count} models")
    print(f"  Probabilistic: {prob_count} models")
    
    return results

# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_robustness_metrics(df, baseline_threshold=0.6):
    """
    Calculate robustness metrics for each model-representation pair.
    
    NDS (Noise Degradation Slope) is calculated for ALL configs initially,
    but nds_thresholded is only set for configs with baseline R² > threshold.
    """
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS")
    print(f"  Baseline threshold for NDS: R² > {baseline_threshold}")
    print("="*80)
    
    metrics_list = []
    
    for (model, rep, model_type), group in df.groupby(['model', 'representation', 'model_type']):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'model': model,
            'representation': rep,
            'model_type': model_type,
        }
        
        # Baseline at σ=0
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # High noise (σ=0.6)
        sigma_h = group[np.abs(group['sigma'] - 0.6) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Calculate NDS (Noise Degradation Slope) for ALL configs
        if len(group) >= 3:
            try:
                slope_r2, intercept, r_val, p_val, _ = stats.linregress(group['sigma'], group['r2'])
                metrics['nds_r2'] = slope_r2  # Unthresholded NDS
                metrics['nds_r2_pval'] = p_val
                metrics['nds_r2_r'] = r_val
                
                slope_rmse, _, _, _, _ = stats.linregress(group['sigma'], group['rmse'])
                metrics['nds_rmse'] = slope_rmse
            except:
                metrics['nds_r2'] = np.nan
                metrics['nds_rmse'] = np.nan
        else:
            metrics['nds_r2'] = np.nan
            metrics['nds_rmse'] = np.nan
        
        # Thresholded NDS: only valid if baseline R² > threshold
        if not np.isnan(metrics['baseline_r2']) and metrics['baseline_r2'] > baseline_threshold:
            metrics['nds_thresholded'] = metrics['nds_r2']
            metrics['meets_baseline_threshold'] = True
        else:
            metrics['nds_thresholded'] = np.nan
            metrics['meets_baseline_threshold'] = False
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Summary statistics
    n_total = len(metrics_df)
    n_meets_threshold = metrics_df['meets_baseline_threshold'].sum()
    n_below_threshold = n_total - n_meets_threshold
    
    print(f"\nCalculated metrics for {n_total} configurations")
    print(f"  Configs meeting baseline threshold (R² > {baseline_threshold}): {n_meets_threshold}")
    print(f"  Configs below threshold: {n_below_threshold}")
    
    # Model type breakdown
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_df[metrics_df['model_type'] == model_type]
        n_type = len(subset)
        n_thresh = subset['meets_baseline_threshold'].sum()
        print(f"\n  {model_type.capitalize()}:")
        print(f"    Total: {n_type}, Meeting threshold: {n_thresh}")
    
    return metrics_df


def define_robustness_score(metrics_df):
    """
    Define composite robustness score based on:
    - Baseline R² (higher is better)
    - NDS magnitude (less negative / closer to 0 is better)
    
    Only uses thresholded NDS for the score.
    """
    # Filter to only configs that meet threshold
    valid_mask = metrics_df['meets_baseline_threshold'] == True
    
    # For configs meeting threshold, calculate robustness score
    baseline_min = metrics_df.loc[valid_mask, 'baseline_r2'].min()
    baseline_max = metrics_df.loc[valid_mask, 'baseline_r2'].max()
    baseline_range = baseline_max - baseline_min if baseline_max != baseline_min else 1
    
    baseline_normalized = (metrics_df['baseline_r2'] - baseline_min) / baseline_range
    
    # Normalize NDS (less negative is better)
    nds_vals = metrics_df.loc[valid_mask, 'nds_thresholded'].dropna()
    if len(nds_vals) > 0:
        nds_min = nds_vals.min()
        nds_max = nds_vals.max()
        nds_range = nds_max - nds_min if nds_max != nds_min else 1
        
        nds_normalized = (metrics_df['nds_thresholded'] - nds_min) / nds_range
    else:
        nds_normalized = pd.Series(np.nan, index=metrics_df.index)
    
    # Composite score
    metrics_df['robustness_score'] = (baseline_normalized + nds_normalized) / 2
    metrics_df.loc[~valid_mask, 'robustness_score'] = np.nan
    
    return metrics_df

# ============================================================================
# FIGURE 3: DETERMINISTIC VS PROBABILISTIC
# ============================================================================

def create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir):
    """Figure 3: Deterministic vs probabilistic models comparison"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: DETERMINISTIC VS PROBABILISTIC")
    print("="*80)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30,
                          left=0.08, right=0.98, top=0.94, bottom=0.08)
    
    # Find best representation for comparison
    available_reps = df['representation'].unique()
    primary_rep = 'pdv' if 'pdv' in available_reps else available_reps[0]
    
    # ========================================================================
    # PANEL A: RF vs QRF
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    for model, color, style, marker in [('rf', COLORS['deterministic'], '--', 'o'),
                                         ('qrf', COLORS['probabilistic'], '-', 's')]:
        model_data = df[(df['model'] == model) & (df['representation'] == primary_rep)]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            label = f"{format_model(model)} ({'det' if model == 'rf' else 'prob'})"
            ax_a.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker=marker, linestyle=style, linewidth=2, alpha=0.9,
                     label=label, color=color, markersize=6)
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=9)
    ax_a.set_ylabel('R² score', fontsize=9)
    ax_a.set_title(f'A. RF → QRF ({format_representation(primary_rep)})', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_a.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL B: MLP vs MLP-BNN variants
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    mlp_variants = ['mlp', 'mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational']
    colors_mlp = [COLORS['deterministic'], '#e67e22', '#27ae60', '#8e44ad']
    styles = ['-', '--', '-.', ':']
    
    for model, color, style in zip(mlp_variants, colors_mlp, styles):
        model_data = df[(df['model'] == model) & (df['representation'] == primary_rep)]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            ax_b.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker='o', linestyle=style, linewidth=2, alpha=0.9,
                     label=format_model(model), color=color, markersize=5)
    
    ax_b.set_xlabel('Noise level (σ)', fontsize=9)
    ax_b.set_ylabel('R² score', fontsize=9)
    ax_b.set_title(f'B. MLP → BNN Variants ({format_representation(primary_rep)})', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_b.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL C: Baseline vs NDS scatter (thresholded)
    # ========================================================================
    ax_c = fig.add_subplot(gs[1, 0])
    
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_thresh[metrics_thresh['model_type'] == model_type]
        if len(subset) > 0:
            color = COLORS[model_type]
            marker = 'o' if model_type == 'deterministic' else 's'
            ax_c.scatter(subset['baseline_r2'], subset['nds_thresholded'],
                        alpha=0.7, s=60, color=color, marker=marker,
                        label=model_type.capitalize(),
                        edgecolors='black', linewidth=0.5)
    
    ax_c.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_c.set_xlabel('Baseline R² (σ=0)', fontsize=9)
    ax_c.set_ylabel('Noise Degradation Slope', fontsize=9)
    ax_c.set_title('C. Baseline vs Noise Degradation Slope', fontsize=10, fontweight='bold', pad=10)
    ax_c.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL D: NDS by model type (box plot)
    # ========================================================================
    ax_d = fig.add_subplot(gs[1, 1])
    
    det_nds = metrics_thresh[metrics_thresh['model_type'] == 'deterministic']['nds_thresholded'].dropna()
    prob_nds = metrics_thresh[metrics_thresh['model_type'] == 'probabilistic']['nds_thresholded'].dropna()
    
    if len(det_nds) > 0 and len(prob_nds) > 0:
        bp = ax_d.boxplot([det_nds, prob_nds],
                         labels=['Deterministic', 'Probabilistic'],
                         patch_artist=True, widths=0.6)
        
        bp['boxes'][0].set_facecolor(COLORS['deterministic'])
        bp['boxes'][1].set_facecolor(COLORS['probabilistic'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        # Add individual points
        for i, (data, color) in enumerate([(det_nds, COLORS['deterministic']),
                                            (prob_nds, COLORS['probabilistic'])]):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax_d.scatter(x, data, alpha=0.4, s=20, color=color, zorder=3)
        
        # Statistical test
        stat, p_val = stats.mannwhitneyu(det_nds.abs(), prob_nds.abs(), alternative='two-sided')
        sig_text = f"p = {p_val:.4f}" + (" *" if p_val < 0.05 else "")
        ax_d.text(0.5, 0.95, sig_text, transform=ax_d.transAxes, ha='center', fontsize=8)
    
    ax_d.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_d.set_ylabel('Noise Degradation Slope', fontsize=9)
    ax_d.set_title('D. Noise Degradation Slope Distribution by Model Type', fontsize=10, fontweight='bold', pad=10)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "figure3_deterministic_vs_probabilistic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 to {output_path}")
    plt.close()


def create_figure4_bayesian_transformations(df, metrics_df, output_dir):
    """Figure 4: Bayesian transformation comparisons (architecturally valid pairs)"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: BAYESIAN TRANSFORMATIONS")
    print("="*80)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.30,
                          left=0.08, right=0.98, top=0.94, bottom=0.06)
    
    # Define valid Bayesian transformation pairs (DNN didn't run, so only RF and MLP)
    transformations = [
        {
            'base': 'rf',
            'variants': ['qrf'],
            'title': 'RF → QRF',
            'description': 'Tree-based'
        },
        {
            'base': 'mlp',
            'variants': ['mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational'],
            'title': 'MLP → BNN',
            'description': 'MLP'
        },
    ]
    
    available_reps = df['representation'].unique()
    primary_rep = 'pdv' if 'pdv' in available_reps else available_reps[0]
    
    # Row 1: Degradation curves for each transformation
    for idx, trans in enumerate(transformations):
        ax = fig.add_subplot(gs[0, idx])
        
        base = trans['base']
        variants = trans['variants']
        
        # Plot base model
        base_data = df[(df['model'] == base) & (df['representation'] == primary_rep)]
        if len(base_data) > 0:
            avg = base_data.groupby('sigma')['r2'].mean().reset_index()
            ax.plot(avg['sigma'], avg['r2'],
                   marker='o', linestyle='-', linewidth=2.5, alpha=0.9,
                   label=format_model(base), color=COLORS['deterministic'], markersize=6)
        
        # Plot variants
        variant_colors = ['#e67e22', '#95a5a6', '#d35400']
        variant_styles = ['--', '-.', ':']
        
        for i, variant in enumerate(variants):
            var_data = df[(df['model'] == variant) & (df['representation'] == primary_rep)]
            if len(var_data) > 0:
                avg = var_data.groupby('sigma')['r2'].mean().reset_index()
                ax.plot(avg['sigma'], avg['r2'],
                       marker='s', linestyle=variant_styles[i % len(variant_styles)], 
                       linewidth=2, alpha=0.8,
                       label=format_model(variant), color=variant_colors[i % len(variant_colors)], 
                       markersize=5)
        
        ax.set_xlabel('Noise level (σ)', fontsize=9)
        ax.set_ylabel('R² score', fontsize=9)
        ax.set_title(f'A{idx+1}. {trans["title"]} ({format_representation(primary_rep)})', 
                     fontsize=10, fontweight='bold', pad=10)
        ax.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_ylim(bottom=0)
    
    # Row 2: Baseline comparison (base vs best variant) across representations
    representations = sorted(df['representation'].unique())
    
    for idx, trans in enumerate(transformations):
        ax = fig.add_subplot(gs[1, idx])
        
        base = trans['base']
        variants = trans['variants']
        
        base_baselines = []
        variant_baselines = []
        rep_labels = []
        
        for rep in representations:
            base_met = metrics_thresh[(metrics_thresh['model'] == base) & (metrics_thresh['representation'] == rep)]
            
            # Get best variant for this representation (by baseline R²)
            best_var_baseline = None
            for var in variants:
                var_met = metrics_thresh[(metrics_thresh['model'] == var) & (metrics_thresh['representation'] == rep)]
                if len(var_met) > 0:
                    var_baseline = var_met['baseline_r2'].mean()
                    if best_var_baseline is None or var_baseline > best_var_baseline:
                        best_var_baseline = var_baseline
            
            if len(base_met) > 0 and best_var_baseline is not None:
                base_baselines.append(base_met['baseline_r2'].mean())
                variant_baselines.append(best_var_baseline)
                rep_labels.append(format_representation(rep))
        
        if base_baselines:
            x = np.arange(len(rep_labels))
            width = 0.35
            
            ax.bar(x - width/2, base_baselines, width, label=format_model(base),
                  color=COLORS['deterministic'], alpha=0.8, edgecolor='black', linewidth=0.5)
            ax.bar(x + width/2, variant_baselines, width, label='Best BNN',
                  color=COLORS['probabilistic'], alpha=0.8, edgecolor='black', linewidth=0.5)
            
            ax.set_xticks(x)
            ax.set_xticklabels(rep_labels, rotation=45, ha='right', fontsize=8)
        
        ax.set_ylabel('Baseline R²', fontsize=9)
        ax.set_title(f'B{idx+1}. {trans["title"]} Baseline by Rep', fontsize=10, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Row 3: NDS improvement (base - variant, positive = variant better)
    for idx, trans in enumerate(transformations):
        ax = fig.add_subplot(gs[2, idx])
        
        base = trans['base']
        variants = trans['variants']
        
        improvements = []
        labels = []
        colors_list = []
        
        for rep in representations:
            base_met = metrics_thresh[(metrics_thresh['model'] == base) & (metrics_thresh['representation'] == rep)]
            
            for var in variants:
                var_met = metrics_thresh[(metrics_thresh['model'] == var) & (metrics_thresh['representation'] == rep)]
                
                if len(base_met) > 0 and len(var_met) > 0:
                    # NDS improvement: less negative is better, so var - base
                    # If var NDS is -0.3 and base NDS is -0.4, improvement is 0.1 (positive = good)
                    base_nds = base_met['nds_thresholded'].mean()
                    var_nds = var_met['nds_thresholded'].mean()
                    improvement = var_nds - base_nds  # Less negative = positive improvement
                    
                    improvements.append(improvement)
                    labels.append(f"{format_model(var)}\n{format_representation(rep)}")
                    colors_list.append(COLORS['probabilistic'] if improvement > 0 else COLORS['deterministic'])
        
        if improvements:
            y_pos = np.arange(len(improvements))
            ax.barh(y_pos, improvements, color=colors_list, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=6)
            ax.set_xlabel('Noise Degradation Slope improvement (less negative = better)', fontsize=9)
        
        ax.set_title(f'C{idx+1}. {trans["title"]} Noise Degradation Slope Change', fontsize=10, fontweight='bold', pad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "figure4_bayesian_transformations.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 4 to {output_path}")
    plt.close()


def create_summary_tables(metrics_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    # Table 1: By model type (thresholded)
    table1 = metrics_thresh.groupby('model_type').agg({
        'baseline_r2': ['mean', 'std'],
        'nds_thresholded': ['mean', 'std'],
    }).round(4)
    table1.to_csv(output_dir / "table_phase1_by_model_type.csv")
    print(f"✓ Saved model type summary")
    
    # Table 2: Full breakdown (thresholded)
    table2 = metrics_thresh.groupby(['model', 'representation', 'model_type']).agg({
        'baseline_r2': 'mean',
        'r2_high': 'mean',
        'nds_thresholded': 'mean',
    }).round(4)
    table2.to_csv(output_dir / "table_phase1_full_breakdown.csv")
    print(f"✓ Saved full breakdown table")
    
    # Table 3: Bayesian transformation comparisons
    transformations = [
        ('rf', ['qrf']),
        ('mlp', ['mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational']),
    ]
    pair_results = []
    
    for base, variants in transformations:
        for var in variants:
            for rep in metrics_thresh['representation'].unique():
                base_data = metrics_thresh[(metrics_thresh['model'] == base) & (metrics_thresh['representation'] == rep)]
                var_data = metrics_thresh[(metrics_thresh['model'] == var) & (metrics_thresh['representation'] == rep)]
                
                if len(base_data) > 0 and len(var_data) > 0:
                    pair_results.append({
                        'base_model': base,
                        'bayesian_variant': var,
                        'representation': rep,
                        'base_baseline': base_data['baseline_r2'].mean(),
                        'var_baseline': var_data['baseline_r2'].mean(),
                        'base_nds': base_data['nds_thresholded'].mean(),
                        'var_nds': var_data['nds_thresholded'].mean(),
                        'nds_improvement': var_data['nds_thresholded'].mean() - base_data['nds_thresholded'].mean(),
                    })
    
    if pair_results:
        table3 = pd.DataFrame(pair_results).round(4)
        table3.to_csv(output_dir / "table_phase1_bayesian_transformations.csv", index=False)
        print(f"✓ Saved Bayesian transformations table")
    
    # Table 4: All configs with threshold status
    table4 = metrics_df[['model', 'representation', 'model_type', 'baseline_r2', 
                         'nds_r2', 'nds_thresholded', 'meets_baseline_threshold']].copy()
    table4.to_csv(output_dir / "table_phase1_all_configs.csv", index=False, float_format='%.4f')
    print(f"✓ Saved all configs table")


def perform_statistical_tests(metrics_df, output_dir):
    """Perform statistical tests"""
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL TESTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Use thresholded metrics
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True].copy()
    
    results_text = ["STATISTICAL COMPARISONS - PHASE 1", "="*80, ""]
    results_text.append("Note: All comparisons use thresholded data (baseline R² > 0.6)")
    results_text.append("")
    
    # Overall comparison
    results_text.append("OVERALL: DETERMINISTIC vs PROBABILISTIC")
    results_text.append("-"*80)
    
    det_nds = metrics_thresh[metrics_thresh['model_type'] == 'deterministic']['nds_thresholded'].dropna()
    prob_nds = metrics_thresh[metrics_thresh['model_type'] == 'probabilistic']['nds_thresholded'].dropna()
    
    if len(det_nds) >= 3 and len(prob_nds) >= 3:
        # Compare absolute NDS (lower is better)
        stat, p_val = stats.mannwhitneyu(det_nds.abs(), prob_nds.abs(), alternative='two-sided')
        results_text.append(f"Noise Degradation Slope (|NDS|) comparison:")
        results_text.append(f"  Deterministic: mean |NDS|={det_nds.abs().mean():.4f}, n={len(det_nds)}")
        results_text.append(f"  Probabilistic: mean |NDS|={prob_nds.abs().mean():.4f}, n={len(prob_nds)}")
        results_text.append(f"  Mann-Whitney U: stat={stat:.2f}, p={p_val:.6f}")
        if p_val < 0.05:
            winner = 'Probabilistic' if prob_nds.abs().mean() < det_nds.abs().mean() else 'Deterministic'
            results_text.append(f"  → Significant (p<0.05): {winner} models show better noise stability")
        else:
            results_text.append(f"  → No significant difference")
    
    results_text.append("")
    
    # Bayesian transformation comparisons
    transformations = [
        ('rf', 'qrf', 'Random Forest → QRF'),
        ('mlp', 'mlp_bnn_full', 'MLP → MLP-BNN-Full'),
        ('mlp', 'mlp_bnn_last', 'MLP → MLP-BNN-Last'),
        ('mlp', 'mlp_bnn_variational', 'MLP → MLP-BNN-Var'),
    ]
    
    for base, var, name in transformations:
        results_text.append(f"\n{name.upper()}")
        results_text.append("-"*80)
        
        base_data = metrics_thresh[metrics_thresh['model'] == base]['nds_thresholded'].dropna()
        var_data = metrics_thresh[metrics_thresh['model'] == var]['nds_thresholded'].dropna()
        
        if len(base_data) >= 3 and len(var_data) >= 3:
            stat, p_val = stats.mannwhitneyu(base_data.abs(), var_data.abs(), alternative='two-sided')
            results_text.append(f"  {base}: mean |NDS|={base_data.abs().mean():.4f}, n={len(base_data)}")
            results_text.append(f"  {var}: mean |NDS|={var_data.abs().mean():.4f}, n={len(var_data)}")
            results_text.append(f"  Mann-Whitney U: p={p_val:.6f}")
            if p_val < 0.05:
                winner = var if var_data.abs().mean() < base_data.abs().mean() else base
                results_text.append(f"  → Significant: {winner} more stable")
            else:
                results_text.append(f"  → No significant difference")
        else:
            results_text.append(f"  Insufficient data for comparison")
    
    output_path = output_dir / "statistical_tests_phase1.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_text))
    print(f"✓ Saved statistical tests to {output_path}")

# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution"""
    print("="*80)
    print("PHASE 1 ANALYSIS - DETERMINISTIC VS PROBABILISTIC")
    print("="*80)
    print("\nKey changes in this version:")
    print("  - Retention metric REMOVED")
    print("  - NSI renamed to Noise Degradation Slope (NDS)")
    print("  - NDS thresholded: only calculated for baseline R² > 0.6")
    print("="*80)
    
    df = load_phase0_data(results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    metrics_df = calculate_robustness_metrics(df, baseline_threshold=0.6)
    metrics_df = define_robustness_score(metrics_df)
    
    output_dir = Path(results_dir) / "phase1_figures_v3"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_df.to_csv(output_dir / "phase1_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase1_robustness_metrics.csv'}")
    
    print("\n" + "="*80)
    print("GENERATING FIGURES (using thresholded NDS)")
    print("="*80)
    
    create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir)
    create_figure4_bayesian_transformations(df, metrics_df, output_dir)
    create_summary_tables(metrics_df, output_dir)
    perform_statistical_tests(metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    metrics_thresh = metrics_df[metrics_df['meets_baseline_threshold'] == True]
    
    print(f"\nTotal configs: {len(metrics_df)}")
    print(f"Configs meeting threshold (R² > 0.6): {len(metrics_thresh)}")
    
    print("\nModel Type Comparison (thresholded):")
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_thresh[metrics_thresh['model_type'] == model_type]
        if len(subset) > 0:
            print(f"  {model_type.capitalize()}:")
            print(f"    Models: {subset['model'].nunique()}")
            print(f"    Mean |NDS|: {subset['nds_thresholded'].abs().mean():.4f}")
    
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)