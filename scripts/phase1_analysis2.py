"""
Phase 1 Analysis - Deterministic vs Probabilistic Models
=========================================================

Pulls data from Phase 0 screening results and analyzes:
- Deterministic models: RF, XGBoost, DNN, MLP
- Probabilistic models: QRF, NGBoost, GP/GAUCHE, BNN variants

Key metrics (NO AUC):
- NSI (Noise Sensitivity Index): slope of R² vs σ
- Retention percentage: (R²_high / R²_baseline) * 100
- Baseline R² at σ=0
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
    
    # Load both file patterns (same as phase0_analysis.py)
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
            
            # For MHGGNN files, set representation to 'mhggnn'
            if 'mhggnn' in filepath.name.lower():
                if 'rep' in df.columns:
                    df['rep'] = 'mhggnn'
                elif 'representation' in df.columns:
                    df['representation'] = 'mhggnn'
                else:
                    df['rep'] = 'mhggnn'
            
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

def calculate_robustness_metrics(df, sigma_high=0.6):
    """Calculate robustness metrics for each model-representation pair"""
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS (σ_high = {sigma_high})")
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
        
        # High noise
        sigma_h = group[np.abs(group['sigma'] - sigma_high) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
            metrics['rmse_high'] = sigma_h['rmse'].values[0]
        else:
            metrics['r2_high'] = np.nan
            metrics['rmse_high'] = np.nan
        
        # Retention percentage
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            if metrics['baseline_r2'] != 0:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            else:
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # NSI (slope)
        if len(group) >= 3:
            try:
                slope_r2, intercept, r_val, p_val, _ = stats.linregress(group['sigma'], group['r2'])
                metrics['nsi_r2'] = slope_r2
                metrics['nsi_r2_pval'] = p_val
                metrics['nsi_r2_r'] = r_val
                
                slope_rmse, _, _, _, _ = stats.linregress(group['sigma'], group['rmse'])
                metrics['nsi_rmse'] = slope_rmse
            except:
                metrics['nsi_r2'] = np.nan
                metrics['nsi_rmse'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
            metrics['nsi_rmse'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    # Filter outliers
    print("\nFiltering outliers...")
    before_count = len(metrics_df)
    
    metrics_df = metrics_df[
        (metrics_df['baseline_r2'] >= 0.1) &
        (metrics_df['retention_pct'] >= -50) &
        (metrics_df['retention_pct'] <= 150)
    ].copy()
    
    print(f"  Before: {before_count}, After: {len(metrics_df)}")
    print(f"  Excluded: {before_count - len(metrics_df)} outliers")
    
    return metrics_df

# ============================================================================
# FIGURE 3: DETERMINISTIC VS PROBABILISTIC
# ============================================================================

def create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir):
    """Figure 3: Deterministic vs probabilistic models comparison"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: DETERMINISTIC VS PROBABILISTIC")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.08)
    
    # ========================================================================
    # PANEL A: Degradation curves - RF vs QRF
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Find best representation for comparison
    available_reps = df['representation'].unique()
    primary_rep = 'pdv' if 'pdv' in available_reps else available_reps[0]
    
    # RF vs QRF
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
    ax_a.set_title(f'A. RF vs QRF ({format_representation(primary_rep)})', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_a.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL B: XGBoost vs NGBoost
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    for model, color, style, marker in [('xgboost', COLORS['deterministic'], '--', 'o'),
                                         ('ngboost', COLORS['probabilistic'], '-', 's')]:
        model_data = df[(df['model'] == model) & (df['representation'] == primary_rep)]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            label = f"{format_model(model)} ({'det' if model == 'xgboost' else 'prob'})"
            ax_b.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker=marker, linestyle=style, linewidth=2, alpha=0.9,
                     label=label, color=color, markersize=6)
    
    ax_b.set_xlabel('Noise level (σ)', fontsize=9)
    ax_b.set_ylabel('R² score', fontsize=9)
    ax_b.set_title(f'B. XGBoost vs NGBoost ({format_representation(primary_rep)})', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_b.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL C: DNN vs BNN variants
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    dnn_variants = ['dnn', 'dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_variational']
    colors_dnn = ['#34495e', '#e67e22', '#95a5a6', '#d35400']
    styles = ['-', '--', '-.', ':']
    
    for model, color, style in zip(dnn_variants, colors_dnn, styles):
        model_data = df[(df['model'] == model) & (df['representation'] == primary_rep)]
        if len(model_data) > 0:
            avg_by_sigma = model_data.groupby('sigma')['r2'].mean().reset_index()
            ax_c.plot(avg_by_sigma['sigma'], avg_by_sigma['r2'],
                     marker='o', linestyle=style, linewidth=2, alpha=0.9,
                     label=format_model(model), color=color, markersize=5)
    
    ax_c.set_xlabel('Noise level (σ)', fontsize=9)
    ax_c.set_ylabel('R² score', fontsize=9)
    ax_c.set_title(f'C. DNN vs BNN Variants ({format_representation(primary_rep)})', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_c.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_c.set_ylim(bottom=0)
    
    # ========================================================================
    # PANEL D: Baseline vs Retention scatter
    # ========================================================================
    ax_d = fig.add_subplot(gs[1, 0])
    
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_df[metrics_df['model_type'] == model_type]
        if len(subset) > 0:
            color = COLORS[model_type]
            marker = 'o' if model_type == 'deterministic' else 's'
            ax_d.scatter(subset['baseline_r2'], subset['retention_pct'],
                        alpha=0.7, s=60, color=color, marker=marker,
                        label=model_type.capitalize(),
                        edgecolors='black', linewidth=0.5)
    
    ax_d.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_d.set_xlabel('Baseline R² (σ=0)', fontsize=9)
    ax_d.set_ylabel('Retention at high noise (%)', fontsize=9)
    ax_d.set_title('D. Baseline vs Robustness', fontsize=10, fontweight='bold', pad=10)
    ax_d.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
    ax_d.spines['top'].set_visible(False)
    ax_d.spines['right'].set_visible(False)
    ax_d.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL E: Retention by model type (box plot)
    # ========================================================================
    ax_e = fig.add_subplot(gs[1, 1])
    
    det_retention = metrics_df[metrics_df['model_type'] == 'deterministic']['retention_pct'].dropna()
    prob_retention = metrics_df[metrics_df['model_type'] == 'probabilistic']['retention_pct'].dropna()
    
    if len(det_retention) > 0 and len(prob_retention) > 0:
        bp = ax_e.boxplot([det_retention, prob_retention],
                         labels=['Deterministic', 'Probabilistic'],
                         patch_artist=True, widths=0.6)
        
        bp['boxes'][0].set_facecolor(COLORS['deterministic'])
        bp['boxes'][1].set_facecolor(COLORS['probabilistic'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        # Add individual points
        for i, (data, color) in enumerate([(det_retention, COLORS['deterministic']),
                                            (prob_retention, COLORS['probabilistic'])]):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax_e.scatter(x, data, alpha=0.4, s=20, color=color, zorder=3)
        
        # Statistical test
        stat, p_val = stats.mannwhitneyu(det_retention, prob_retention, alternative='two-sided')
        sig_text = f"p = {p_val:.4f}" + (" *" if p_val < 0.05 else "")
        ax_e.text(0.5, 0.95, sig_text, transform=ax_e.transAxes, ha='center', fontsize=8)
    
    ax_e.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_e.set_ylabel('Retention at high noise (%)', fontsize=9)
    ax_e.set_title('E. Retention Distribution by Model Type', fontsize=10, fontweight='bold', pad=10)
    ax_e.spines['top'].set_visible(False)
    ax_e.spines['right'].set_visible(False)
    ax_e.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL F: NSI comparison
    # ========================================================================
    ax_f = fig.add_subplot(gs[1, 2])
    
    det_nsi = metrics_df[metrics_df['model_type'] == 'deterministic']['nsi_r2'].dropna()
    prob_nsi = metrics_df[metrics_df['model_type'] == 'probabilistic']['nsi_r2'].dropna()
    
    if len(det_nsi) > 0 and len(prob_nsi) > 0:
        bp = ax_f.boxplot([det_nsi, prob_nsi],
                         labels=['Deterministic', 'Probabilistic'],
                         patch_artist=True, widths=0.6)
        
        bp['boxes'][0].set_facecolor(COLORS['deterministic'])
        bp['boxes'][1].set_facecolor(COLORS['probabilistic'])
        for box in bp['boxes']:
            box.set_alpha(0.7)
        
        for i, (data, color) in enumerate([(det_nsi, COLORS['deterministic']),
                                            (prob_nsi, COLORS['probabilistic'])]):
            x = np.random.normal(i+1, 0.04, size=len(data))
            ax_f.scatter(x, data, alpha=0.4, s=20, color=color, zorder=3)
        
        stat, p_val = stats.mannwhitneyu(det_nsi, prob_nsi, alternative='two-sided')
        sig_text = f"p = {p_val:.4f}" + (" *" if p_val < 0.05 else "")
        ax_f.text(0.5, 0.95, sig_text, transform=ax_f.transAxes, ha='center', fontsize=8)
    
    ax_f.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax_f.set_ylabel('NSI (R²) - Degradation slope', fontsize=9)
    ax_f.set_title('F. NSI Distribution by Model Type', fontsize=10, fontweight='bold', pad=10)
    ax_f.spines['top'].set_visible(False)
    ax_f.spines['right'].set_visible(False)
    ax_f.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "figure3_deterministic_vs_probabilistic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 to {output_path}")
    plt.close()


def create_figure4_paired_comparisons(df, metrics_df, output_dir):
    """Figure 4: Paired model comparisons across representations"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: PAIRED COMPARISONS")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.08)
    
    pairs = [
        ('rf', 'qrf', 'RF vs QRF'),
        ('xgboost', 'ngboost', 'XGBoost vs NGBoost'),
        ('dnn', 'gauche', 'DNN vs GP'),
    ]
    
    representations = sorted(df['representation'].unique())
    
    # Row 1: Retention difference by representation
    for idx, (det, prob, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[0, idx])
        
        differences = []
        rep_labels = []
        colors_list = []
        
        for rep in representations:
            det_data = metrics_df[(metrics_df['model'] == det) & (metrics_df['representation'] == rep)]
            prob_data = metrics_df[(metrics_df['model'] == prob) & (metrics_df['representation'] == rep)]
            
            if len(det_data) > 0 and len(prob_data) > 0:
                det_ret = det_data['retention_pct'].mean()
                prob_ret = prob_data['retention_pct'].mean()
                diff = prob_ret - det_ret  # Positive = probabilistic better
                
                differences.append(diff)
                rep_labels.append(format_representation(rep))
                colors_list.append(COLORS['probabilistic'] if diff > 0 else COLORS['deterministic'])
        
        if differences:
            y_pos = np.arange(len(differences))
            ax.barh(y_pos, differences, color=colors_list, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(rep_labels, fontsize=8)
            ax.set_xlabel('Retention difference (% points)', fontsize=9)
            ax.set_title(f'A{idx+1}. {title}', fontsize=10, fontweight='bold', pad=10)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # Row 2: Performance across noise levels for each pair
    for idx, (det, prob, title) in enumerate(pairs):
        ax = fig.add_subplot(gs[1, idx])
        
        # Average across all representations
        det_data = df[df['model'] == det]
        prob_data = df[df['model'] == prob]
        
        if len(det_data) > 0:
            det_avg = det_data.groupby('sigma')['r2'].mean().reset_index()
            ax.plot(det_avg['sigma'], det_avg['r2'],
                   marker='o', linestyle='--', linewidth=2, alpha=0.9,
                   label=format_model(det), color=COLORS['deterministic'], markersize=6)
        
        if len(prob_data) > 0:
            prob_avg = prob_data.groupby('sigma')['r2'].mean().reset_index()
            ax.plot(prob_avg['sigma'], prob_avg['r2'],
                   marker='s', linestyle='-', linewidth=2, alpha=0.9,
                   label=format_model(prob), color=COLORS['probabilistic'], markersize=6)
        
        ax.set_xlabel('Noise level (σ)', fontsize=9)
        ax.set_ylabel('R² (avg across reps)', fontsize=9)
        ax.set_title(f'B{idx+1}. {title} Degradation', fontsize=10, fontweight='bold', pad=10)
        ax.legend(fontsize=8, loc='best', frameon=True, framealpha=0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
        ax.set_ylim(bottom=0)
    
    output_path = Path(output_dir) / "figure4_paired_comparisons.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 4 to {output_path}")
    plt.close()


def create_summary_tables(metrics_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: By model type
    table1 = metrics_df.groupby('model_type').agg({
        'baseline_r2': ['mean', 'std'],
        'retention_pct': ['mean', 'std'],
        'nsi_r2': lambda x: np.abs(x).mean(),
    }).round(4)
    table1.to_csv(output_dir / "table_phase1_by_model_type.csv")
    print(f"✓ Saved model type summary")
    
    # Table 2: Full breakdown
    table2 = metrics_df.groupby(['model', 'representation', 'model_type']).agg({
        'baseline_r2': 'mean',
        'r2_high': 'mean',
        'retention_pct': 'mean',
        'nsi_r2': 'mean',
    }).round(4)
    table2.to_csv(output_dir / "table_phase1_full_breakdown.csv")
    print(f"✓ Saved full breakdown table")
    
    # Table 3: Paired comparisons
    pairs = [('rf', 'qrf'), ('xgboost', 'ngboost'), ('dnn', 'gauche')]
    pair_results = []
    
    for det, prob in pairs:
        for rep in metrics_df['representation'].unique():
            det_data = metrics_df[(metrics_df['model'] == det) & (metrics_df['representation'] == rep)]
            prob_data = metrics_df[(metrics_df['model'] == prob) & (metrics_df['representation'] == rep)]
            
            if len(det_data) > 0 and len(prob_data) > 0:
                pair_results.append({
                    'pair': f"{det} vs {prob}",
                    'representation': rep,
                    'det_baseline': det_data['baseline_r2'].mean(),
                    'prob_baseline': prob_data['baseline_r2'].mean(),
                    'det_retention': det_data['retention_pct'].mean(),
                    'prob_retention': prob_data['retention_pct'].mean(),
                    'retention_diff': prob_data['retention_pct'].mean() - det_data['retention_pct'].mean(),
                })
    
    if pair_results:
        table3 = pd.DataFrame(pair_results).round(4)
        table3.to_csv(output_dir / "table_phase1_paired_comparisons.csv", index=False)
        print(f"✓ Saved paired comparisons table")


def perform_statistical_tests(metrics_df, output_dir):
    """Perform statistical tests"""
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL TESTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    results_text = ["STATISTICAL COMPARISONS - PHASE 1", "="*80, ""]
    
    # Overall comparison
    results_text.append("OVERALL: DETERMINISTIC vs PROBABILISTIC")
    results_text.append("-"*80)
    
    det_ret = metrics_df[metrics_df['model_type'] == 'deterministic']['retention_pct'].dropna()
    prob_ret = metrics_df[metrics_df['model_type'] == 'probabilistic']['retention_pct'].dropna()
    
    if len(det_ret) >= 3 and len(prob_ret) >= 3:
        stat, p_val = stats.mannwhitneyu(det_ret, prob_ret, alternative='two-sided')
        results_text.append(f"Retention % comparison:")
        results_text.append(f"  Deterministic: mean={det_ret.mean():.2f}%, n={len(det_ret)}")
        results_text.append(f"  Probabilistic: mean={prob_ret.mean():.2f}%, n={len(prob_ret)}")
        results_text.append(f"  Mann-Whitney U: stat={stat:.2f}, p={p_val:.6f}")
        if p_val < 0.05:
            winner = 'Probabilistic' if prob_ret.mean() > det_ret.mean() else 'Deterministic'
            results_text.append(f"  → Significant (p<0.05): {winner} models show better retention")
        else:
            results_text.append(f"  → No significant difference")
    
    results_text.append("")
    
    # Paired comparisons
    pairs = [('rf', 'qrf', 'Random Forest'), ('xgboost', 'ngboost', 'Gradient Boosting')]
    
    for det, prob, name in pairs:
        results_text.append(f"\n{name.upper()}: {det} vs {prob}")
        results_text.append("-"*80)
        
        det_data = metrics_df[metrics_df['model'] == det]['retention_pct'].dropna()
        prob_data = metrics_df[metrics_df['model'] == prob]['retention_pct'].dropna()
        
        if len(det_data) >= 3 and len(prob_data) >= 3:
            stat, p_val = stats.mannwhitneyu(det_data, prob_data, alternative='two-sided')
            results_text.append(f"  {det}: mean={det_data.mean():.2f}%, n={len(det_data)}")
            results_text.append(f"  {prob}: mean={prob_data.mean():.2f}%, n={len(prob_data)}")
            results_text.append(f"  Mann-Whitney U: p={p_val:.6f}")
            if p_val < 0.05:
                winner = prob if prob_data.mean() > det_data.mean() else det
                results_text.append(f"  → Significant: {winner} superior")
            else:
                results_text.append(f"  → No significant difference")
    
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
    
    df = load_phase0_data(results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)
    
    output_dir = Path(results_dir) / "phase1_figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    metrics_df.to_csv(output_dir / "phase1_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase1_robustness_metrics.csv'}")
    
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir)
    create_figure4_paired_comparisons(df, metrics_df, output_dir)
    create_summary_tables(metrics_df, output_dir)
    perform_statistical_tests(metrics_df, output_dir)
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print("\nModel Type Comparison:")
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_df[metrics_df['model_type'] == model_type]
        if len(subset) > 0:
            print(f"  {model_type.capitalize()}:")
            print(f"    Models: {subset['model'].nunique()}")
            print(f"    Mean retention: {subset['retention_pct'].mean():.1f}%")
            print(f"    Mean |NSI|: {subset['nsi_r2'].abs().mean():.4f}")
    
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)