"""
Phase 1 Analysis - Deterministic vs Probabilistic Counterparts
Generates Figure 3 and Supplementary S4

Based on the detailed outline:
- Figure 3: Deterministic vs probabilistic models under noise (4 panels)
- Supplementary S4: Pairwise difference plots

Key metrics used (NO AUC as primary robustness metric):
- NSI (Noise Sensitivity Index): slope of R² vs σ
- Retention percentage: (R²_high / R²_baseline) * 100
- Baseline R² at σ=0
- Trade-off between clean accuracy and robustness
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
COLORS = {
    'deterministic': '#0173B2',
    'probabilistic': '#DE8F05',
    'full_bnn': '#e74c3c',
    'last_layer': '#95a5a6',
    'variational': '#d35400',
}

REPRESENTATION_COLORS = {
    'pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
}

# ============================================================================
# DATA LOADING AND PARSING
# ============================================================================

def load_phase1_results(results_dir="../results"):
    """Load Phase 1 results files"""
    print("\n" + "="*80)
    print("LOADING PHASE 1 DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Find all phase1 files
    phase1_files = {
        '1a': list(results_dir.glob("phase1a_*.csv")),
        '1b': list(results_dir.glob("phase1b_*.csv")),
        '1c': list(results_dir.glob("phase1c_*.csv"))
    }
    
    print(f"Found files:")
    for phase, files in phase1_files.items():
        print(f"  Phase {phase}: {len(files)} files")
    
    all_results = []
    
    for phase, files in phase1_files.items():
        for filepath in files:
            # Skip per_epoch and uncertainty files for now
            if 'per_epoch' in filepath.name or 'uncertainty' in filepath.name:
                continue
            
            try:
                df = pd.read_csv(filepath)
                df['phase'] = phase
                df['source_file'] = filepath.name
                all_results.append(df)
            except Exception as e:
                print(f"Warning: Could not read {filepath.name}: {e}")
    
    if not all_results:
        print("ERROR: No Phase 1 results files loaded!")
        return pd.DataFrame()
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    print(f"\nLoaded {len(results_df)} result rows")
    print(f"Phases: {results_df['phase'].unique()}")
    
    return results_df


def parse_model_info(df):
    """
    Parse model information from filenames
    
    Expected filename format:
    phase1X_REPRESENTATION_BASEMODEL_TRANSFORMATION.csv
    
    Examples:
    - phase1a_pdv_rf_baseline.csv
    - phase1a_pdv_rf.csv (deterministic, no transformation)
    - phase1b_pdv_xgboost_baseline.csv
    - phase1b_pdv_ngboost.csv (probabilistic counterpart)
    - phase1c_pdv_dnn_full.csv (full BNN)
    """
    
    def extract_info(row):
        filename = row['source_file']
        parts = filename.replace('.csv', '').replace('phase1a_', '').replace('phase1b_', '').replace('phase1c_', '').split('_')
        
        if len(parts) >= 2:
            rep = parts[0]
            
            # Handle different formats
            if len(parts) == 2:
                # e.g., pdv_rf or pdv_qrf
                base_model = parts[1]
                transformation = 'deterministic'
            elif len(parts) >= 3:
                base_model = parts[1]
                transformation = '_'.join(parts[2:])
            else:
                base_model = 'unknown'
                transformation = 'unknown'
            
            # Normalize transformation names
            if transformation in ['baseline', 'deterministic']:
                transformation = 'deterministic'
            elif transformation == 'full':
                transformation = 'full_bnn'
            elif transformation in ['lastlayer', 'last']:
                transformation = 'last_layer'
            elif transformation == 'variational':
                transformation = 'variational'
            
            # Determine if probabilistic
            probabilistic_models = ['qrf', 'ngboost', 'gauche']
            probabilistic_transforms = ['full_bnn', 'last_layer', 'variational']
            
            if base_model in probabilistic_models or transformation in probabilistic_transforms:
                model_type = 'probabilistic'
            else:
                model_type = 'deterministic'
            
            return pd.Series({
                'representation': rep,
                'base_model': base_model,
                'transformation': transformation,
                'model_type': model_type,
                'model_full': f"{base_model}_{transformation}" if transformation != 'deterministic' else base_model
            })
        
        return pd.Series({
            'representation': None,
            'base_model': None,
            'transformation': None,
            'model_type': None,
            'model_full': None
        })
    
    info = df.apply(extract_info, axis=1)
    df[['representation', 'base_model', 'transformation', 'model_type', 'model_full']] = info
    
    return df


def calculate_robustness_metrics(df, sigma_high=0.6):
    """
    Calculate robustness metrics for each configuration
    
    Metrics:
    - baseline_r2: R² at σ=0
    - r2_high: R² at σ=sigma_high
    - retention_pct: (r2_high / baseline_r2) * 100
    - nsi_r2: slope of R² vs σ
    """
    print("\n" + "="*80)
    print(f"CALCULATING ROBUSTNESS METRICS (σ_high = {sigma_high})")
    print("="*80)
    
    df = parse_model_info(df)
    
    metrics_list = []
    
    for (base, rep, transform, phase), group in df.groupby(['base_model', 'representation', 'transformation', 'phase']):
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
            'base_model': base,
            'representation': rep,
            'transformation': transform,
            'phase': phase,
            'model_type': group['model_type'].iloc[0],
        }
        
        # Baseline at σ=0
        sigma_0 = group_avg[group_avg['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
            metrics['baseline_rmse'] = sigma_0['rmse'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
            metrics['baseline_rmse'] = np.nan
        
        # Performance at high noise
        sigma_h = group_avg[np.abs(group_avg['sigma'] - sigma_high) < 0.05]
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
                slope_r2, intercept_r2, r_val, p_val, _ = stats.linregress(
                    group_avg['sigma'], group_avg['r2']
                )
                metrics['nsi_r2'] = slope_r2
                metrics['nsi_r2_pval'] = p_val
                
                if intercept_r2 != 0:
                    metrics['nsi_r2_relative'] = slope_r2 / abs(intercept_r2)
                else:
                    metrics['nsi_r2_relative'] = np.nan
                
            except:
                metrics['nsi_r2'] = np.nan
                metrics['nsi_r2_relative'] = np.nan
        else:
            metrics['nsi_r2'] = np.nan
            metrics['nsi_r2_relative'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    
    print(f"Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


# ============================================================================
# FIGURE 3: DETERMINISTIC VS PROBABILISTIC MODELS
# ============================================================================

def create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir):
    """
    Figure 3: Deterministic vs probabilistic models under noise
    
    Panel A: Degradation curves for paired deterministic vs probabilistic models
    Panel B: Baseline vs Retention scatter (shows no systematic advantage)
    Panel C: Full Bayesian vs approximate transformations in DNNs
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 3: DETERMINISTIC VS PROBABILISTIC (3-PANEL)")
    print("="*80)
    
    df = parse_model_info(df)
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.12)
    
    # ========================================================================
    # PANEL A: Degradation curves for paired models
    # ========================================================================
    
    # Define pairs
    pairs = [
        ('rf', 'qrf', 'RF vs QRF'),
        ('xgboost', 'ngboost', 'XGBoost vs NGBoost'),
        ('gauche', 'gauche', 'GP (deterministic vs heteroscedastic)'),  # If available
    ]
    
    # Check what representations are available
    available_reps = df['representation'].unique()
    if 'pdv' in available_reps:
        primary_rep = 'pdv'
    elif 'sns' in available_reps:
        primary_rep = 'sns'
    else:
        primary_rep = available_reps[0]
    
    print(f"Using representation: {primary_rep.upper()}")
    
    ax_a = fig.add_subplot(gs[0, 0])
    
    for det_model, prob_model, label in pairs:
        # Deterministic
        det_data = df[(df['base_model'] == det_model) & 
                      (df['representation'] == primary_rep) &
                      (df['model_type'] == 'deterministic')]
        
        if len(det_data) > 0:
            det_avg = det_data.groupby('sigma')['r2'].mean().reset_index()
            ax_a.plot(det_avg['sigma'], det_avg['r2'],
                     marker='o', linestyle='--', linewidth=2, alpha=0.7,
                     label=f'{det_model} (det)', color=COLORS['deterministic'])
        
        # Probabilistic
        prob_data = df[(df['base_model'] == prob_model) & 
                       (df['representation'] == primary_rep) &
                       (df['model_type'] == 'probabilistic')]
        
        if len(prob_data) > 0:
            prob_avg = prob_data.groupby('sigma')['r2'].mean().reset_index()
            ax_a.plot(prob_avg['sigma'], prob_avg['r2'],
                     marker='s', linestyle='-', linewidth=2, alpha=0.9,
                     label=f'{prob_model} (prob)', color=COLORS['probabilistic'])
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=9)
    ax_a.set_ylabel('R² score', fontsize=9)
    ax_a.set_title(f'A. Degradation Curves ({primary_rep.upper()})\nDeterministic vs Probabilistic', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=7, loc='best', ncol=2, frameon=True, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL B: Baseline vs Retention scatter (shows no systematic advantage)
    # ========================================================================
    
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Scatter: x = baseline R², y = retention %
    for model_type in ['deterministic', 'probabilistic']:
        subset = metrics_df[metrics_df['model_type'] == model_type]
        
        if len(subset) > 0:
            color = COLORS[model_type]
            marker = 'o' if model_type == 'deterministic' else 's'
            
            ax_b.scatter(subset['baseline_r2'], subset['retention_pct'],
                        alpha=0.7, s=80, color=color, marker=marker,
                        label=model_type.capitalize(),
                        edgecolors='black', linewidth=0.8)
    
    # Add reference line for no degradation
    ax_b.axhline(100, color='gray', linestyle='--', linewidth=1, alpha=0.5,
                label='No degradation')
    
    # Annotate a few key points
    top_det = metrics_df[metrics_df['model_type'] == 'deterministic'].nlargest(2, 'retention_pct')
    top_prob = metrics_df[metrics_df['model_type'] == 'probabilistic'].nlargest(2, 'retention_pct')
    
    for _, row in top_det.iterrows():
        ax_b.annotate(f"{row['base_model']}/{row['representation']}", 
                     (row['baseline_r2'], row['retention_pct']),
                     fontsize=6, alpha=0.8,
                     xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.7, edgecolor=COLORS['deterministic'], linewidth=0.5))
    
    for _, row in top_prob.iterrows():
        ax_b.annotate(f"{row['base_model']}/{row['representation']}", 
                     (row['baseline_r2'], row['retention_pct']),
                     fontsize=6, alpha=0.8,
                     xytext=(5, -15), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             alpha=0.7, edgecolor=COLORS['probabilistic'], linewidth=0.5))
    
    ax_b.set_xlabel('Baseline R² (σ=0)', fontsize=9)
    ax_b.set_ylabel('Retention at high noise (%)', fontsize=9)
    ax_b.set_title('B. Baseline vs Robustness\nNo Systematic Advantage', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL C: Full Bayesian vs approximate transformations (DNNs)
    # ========================================================================
    
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Find DNN/MLP data with different transformations
    dnn_data = metrics_df[metrics_df['base_model'].isin(['dnn', 'mlp'])]
    
    if len(dnn_data) > 0:
        # Group by transformation
        transforms = ['deterministic', 'full_bnn', 'last_layer', 'variational']
        transform_labels = ['Deterministic', 'Full BNN', 'Last-layer', 'Variational']
        
        metrics_to_plot = ['baseline_r2', 'r2_high', 'retention_pct']
        metric_labels = ['Baseline R²', 'R² (high noise)', 'Retention (%)']
        
        x_pos = np.arange(len(transforms))
        width = 0.25
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            values = []
            for transform in transforms:
                subset = dnn_data[dnn_data['transformation'] == transform]
                if len(subset) > 0:
                    if metric == 'retention_pct':
                        values.append(subset[metric].mean())
                    else:
                        values.append(subset[metric].mean())
                else:
                    values.append(0)
            
            # Normalize retention to same scale as R²
            if metric == 'retention_pct':
                values = [v/100 for v in values]
            
            ax_c.bar(x_pos + i*width, values, width, label=label,
                    alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax_c.set_xticks(x_pos + width)
        ax_c.set_xticklabels(transform_labels, rotation=45, ha='right', fontsize=7)
        ax_c.set_ylabel('Score (normalized)', fontsize=9)
        ax_c.set_title('C. Full Bayesian vs Approximate\nTransforms (DNN)', 
                      fontsize=10, fontweight='bold', pad=10)
        ax_c.legend(fontsize=7, loc='best', frameon=True, framealpha=0.9)
        ax_c.spines['top'].set_visible(False)
        ax_c.spines['right'].set_visible(False)
        ax_c.grid(True, axis='y', alpha=0.3, linestyle=':', linewidth=0.5)
    else:
        ax_c.text(0.5, 0.5, 'No DNN transformation data available',
                 ha='center', va='center', transform=ax_c.transAxes,
                 fontsize=10, style='italic')
        ax_c.axis('off')
    
    # ========================================================================
    # Save
    # ========================================================================
    
    output_path = Path(output_dir) / "figure3_deterministic_vs_probabilistic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3 (3-panel) to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S4: PAIRWISE DIFFERENCE PLOTS
# ============================================================================

def create_supplementary_s4(metrics_df, output_dir):
    """
    Supplementary S4: Pairwise difference plots
    Forest plot showing probabilistic gain per configuration
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S4: PAIRWISE DIFFERENCES")
    print("="*80)
    
    # Define pairs
    pairs = [
        ('rf', 'qrf'),
        ('xgboost', 'ngboost'),
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    for idx, (ax, metric) in enumerate(zip(axes, ['retention_pct', 'nsi_r2'])):
        differences = []
        labels = []
        colors_list = []
        
        for det_model, prob_model in pairs:
            for rep in metrics_df['representation'].unique():
                # Deterministic
                det = metrics_df[(metrics_df['base_model'] == det_model) &
                                (metrics_df['representation'] == rep) &
                                (metrics_df['model_type'] == 'deterministic')]
                
                # Probabilistic
                prob = metrics_df[(metrics_df['base_model'] == prob_model) &
                                 (metrics_df['representation'] == rep) &
                                 (metrics_df['model_type'] == 'probabilistic')]
                
                if len(det) > 0 and len(prob) > 0:
                    det_val = det[metric].mean()
                    prob_val = prob[metric].mean()
                    
                    if metric == 'nsi_r2':
                        # For NSI, positive difference means probabilistic degrades slower (better)
                        diff = det_val - prob_val  # More negative NSI = better for prob
                    else:
                        # For retention, positive difference means probabilistic retains more
                        diff = prob_val - det_val
                    
                    differences.append(diff)
                    labels.append(f'{det_model} vs {prob_model}\n{rep}')
                    
                    # Color by whether probabilistic is better
                    if diff > 0:
                        colors_list.append(COLORS['probabilistic'])
                    else:
                        colors_list.append(COLORS['deterministic'])
        
        if differences:
            y_pos = np.arange(len(differences))
            
            ax.barh(y_pos, differences, color=colors_list, alpha=0.8,
                   edgecolor='black', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
            
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=7)
            
            if metric == 'retention_pct':
                ax.set_xlabel('Retention difference (% points)', fontsize=9)
                ax.set_title('Retention % Gain\n(Probabilistic - Deterministic)', 
                           fontsize=10, fontweight='bold')
            else:
                ax.set_xlabel('NSI difference (absolute)', fontsize=9)
                ax.set_title('NSI Improvement\n(lower |NSI| = better)', 
                           fontsize=10, fontweight='bold')
            
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "supplementary_s4_pairwise_differences.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S4 to {output_path}")
    plt.close()


# ============================================================================
# STATISTICAL COMPARISONS
# ============================================================================

def perform_statistical_comparisons(df, metrics_df, output_dir):
    """
    Statistical comparisons between deterministic and probabilistic variants
    """
    print("\n" + "="*80)
    print("PERFORMING STATISTICAL COMPARISONS")
    print("="*80)
    
    df = parse_model_info(df)
    
    results_text = []
    results_text.append("STATISTICAL COMPARISONS - PHASE 1")
    results_text.append("="*80)
    results_text.append("")
    results_text.append("Comparing deterministic vs probabilistic variants")
    results_text.append("Tests: Wilcoxon signed-rank (paired samples across noise levels)")
    results_text.append("")
    
    # Define pairs
    pairs = [
        ('rf', 'qrf', 'Random Forest'),
        ('xgboost', 'ngboost', 'Gradient Boosting'),
    ]
    
    for det_model, prob_model, desc in pairs:
        results_text.append(f"\n{'='*80}")
        results_text.append(f"{desc.upper()}: {det_model} vs {prob_model}")
        results_text.append(f"{'='*80}")
        
        for rep in df['representation'].unique():
            results_text.append(f"\nRepresentation: {rep.upper()}")
            results_text.append("-"*80)
            
            # Get data for both
            det_data = df[(df['base_model'] == det_model) &
                         (df['representation'] == rep) &
                         (df['model_type'] == 'deterministic')]
            
            prob_data = df[(df['base_model'] == prob_model) &
                          (df['representation'] == rep) &
                          (df['model_type'] == 'probabilistic')]
            
            if len(det_data) == 0 or len(prob_data) == 0:
                results_text.append("  No data available for comparison")
                continue
            
            # Align by sigma
            det_avg = det_data.groupby('sigma')['r2'].mean()
            prob_avg = prob_data.groupby('sigma')['r2'].mean()
            
            common_sigmas = set(det_avg.index) & set(prob_avg.index)
            
            if len(common_sigmas) < 3:
                results_text.append("  Insufficient overlapping sigma values")
                continue
            
            det_vals = [det_avg[s] for s in sorted(common_sigmas)]
            prob_vals = [prob_avg[s] for s in sorted(common_sigmas)]
            
            # Wilcoxon signed-rank test
            try:
                stat, p_val = stats.wilcoxon(det_vals, prob_vals)
                
                mean_det = np.mean(det_vals)
                mean_prob = np.mean(prob_vals)
                
                results_text.append(f"  Mean R² (deterministic): {mean_det:.4f}")
                results_text.append(f"  Mean R² (probabilistic): {mean_prob:.4f}")
                results_text.append(f"  Wilcoxon statistic: {stat:.2f}")
                results_text.append(f"  p-value: {p_val:.6f}")
                
                if p_val < 0.05:
                    winner = 'probabilistic' if mean_prob > mean_det else 'deterministic'
                    results_text.append(f"  → Significant difference (p<0.05), {winner} superior")
                else:
                    results_text.append(f"  → No significant difference")
                
            except Exception as e:
                results_text.append(f"  Error in statistical test: {e}")
    
    # Save
    output_path = Path(output_dir) / "statistical_comparisons_phase1.txt"
    with open(output_path, 'w') as f:
        f.write('\n'.join(results_text))
    
    print(f"✓ Saved statistical comparisons to {output_path}")


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Deterministic vs Probabilistic comparison
    table1 = metrics_df.groupby(['base_model', 'representation', 'model_type']).agg({
        'baseline_r2': 'mean',
        'r2_high': 'mean',
        'retention_pct': 'mean',
        'nsi_r2': lambda x: np.abs(x).mean()
    }).reset_index()
    
    table1 = table1.round(4)
    table1.to_csv(output_dir / "table_phase1_summary.csv", index=False)
    
    with open(output_dir / "table_phase1_summary.tex", 'w') as f:
        f.write(table1.to_latex(index=False, float_format="%.4f"))
    
    print(f"✓ Saved summary table")
    
    # Table 2: DNN transformations (if available)
    dnn_data = metrics_df[metrics_df['base_model'].isin(['dnn', 'mlp'])]
    
    if len(dnn_data) > 0:
        table2 = dnn_data.groupby(['base_model', 'representation', 'transformation']).agg({
            'baseline_r2': 'mean',
            'r2_high': 'mean',
            'retention_pct': 'mean',
            'nsi_r2': lambda x: np.abs(x).mean()
        }).reset_index()
        
        table2 = table2.round(4)
        table2.to_csv(output_dir / "table_phase1_dnn_transforms.csv", index=False)
        
        with open(output_dir / "table_phase1_dnn_transforms.tex", 'w') as f:
            f.write(table2.to_latex(index=False, float_format="%.4f"))
        
        print(f"✓ Saved DNN transformation table")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 1 ANALYSIS - DETERMINISTIC VS PROBABILISTIC")
    print("Journal of Cheminformatics Style")
    print("="*80)
    
    # Load data
    df = load_phase1_results(results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    # Calculate metrics
    metrics_df = calculate_robustness_metrics(df, sigma_high=0.6)
    
    # Create output directory
    output_dir = Path(results_dir) / "phase1_figures_v2"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase1_robustness_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase1_robustness_metrics.csv'}")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure3_deterministic_vs_probabilistic(df, metrics_df, output_dir)
    create_supplementary_s4(metrics_df, output_dir)
    
    # Generate tables
    create_summary_tables(metrics_df, output_dir)
    
    # Statistical tests
    perform_statistical_comparisons(df, metrics_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 1 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  Figures:")
    print("    - figure3_deterministic_vs_probabilistic.png")
    print("    - supplementary_s4_pairwise_differences.png")
    print("  Tables:")
    print("    - table_phase1_summary.csv/.tex")
    print("    - table_phase1_dnn_transforms.csv/.tex (if applicable)")
    print("  Data:")
    print("    - phase1_robustness_metrics.csv")
    print("    - statistical_comparisons_phase1.txt")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)