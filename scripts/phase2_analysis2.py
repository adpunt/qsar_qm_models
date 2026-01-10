"""
Phase 2 Uncertainty Analysis - Unified Version

Combines:
- Correct data loading from original script
- Clean publication-quality visualization
- Comprehensive uncertainty analysis

Usage:
    python phase2_analysis_unified.py results/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import sys
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

# ============================================================================
# MODEL AND REPRESENTATION CONFIGURATION
# ============================================================================

MODEL_COLORS = {
    'qrf': '#3498db',
    'ngboost': '#e74c3c',
    'bnn_full': '#2ecc71',
    'bnn_last': '#27ae60',
    'bnn_variational': '#9b59b6',
    'gauche': '#f39c12',
    'mlp_bnn_full': '#1abc9c',
    'mlp_bnn_last': '#34495e',
    'mlp_bnn_variational': '#d35400',
}

MODEL_MARKERS = {
    'qrf': 'o',
    'ngboost': 's',
    'bnn_full': '^',
    'bnn_last': 'v',
    'bnn_variational': 'D',
    'gauche': 'p',
    'mlp_bnn_full': 'o',
    'mlp_bnn_last': 'v',
    'mlp_bnn_variational': 'D',
}

MODEL_DISPLAY = {
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    'bnn_full': 'BNN-Full',
    'bnn_last': 'BNN-Last',
    'bnn_variational': 'BNN-Var',
    'gauche': 'GAUCHE',
    'mlp_bnn_full': 'MLP-BNN-Full',
    'mlp_bnn_last': 'MLP-BNN-Last',
    'mlp_bnn_variational': 'MLP-BNN-Var',
}

REP_DISPLAY = {
    'continuous_pdv': 'PDV',
    'pdv': 'PDV',
    'sns': 'SNS',
    'smiles': 'SMILES',
    'mhggnn': 'MHGGNN',
}

# Core models to prioritize in figures
CORE_MODELS = ['qrf', 'ngboost', 'bnn_full', 'gauche']
EXTENDED_MODELS = ['qrf', 'ngboost', 'bnn_full', 'bnn_variational', 'gauche', 
                   'mlp_bnn_full', 'mlp_bnn_variational']


def get_display_name(model):
    return MODEL_DISPLAY.get(model, model)


def get_rep_display(rep):
    return REP_DISPLAY.get(rep, rep)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_uncertainty_data(results_dir):
    """Load all phase2*_uncertainty_values.csv files"""
    print("\n" + "="*80)
    print("LOADING UNCERTAINTY DATA")
    print("="*80)
    
    results_dir = Path(results_dir).resolve()  # Convert to absolute path
    print(f"Looking in: {results_dir}")
    
    # Try the glob pattern
    files = list(results_dir.glob("phase2*_uncertainty_values.csv"))
    
    # Debug: show what's in the directory
    if not files:
        print(f"\nDirectory contents:")
        all_files = list(results_dir.glob("*.csv"))
        phase2_files = [f for f in all_files if f.name.startswith('phase2')]
        print(f"  Total CSV files: {len(all_files)}")
        print(f"  Files starting with 'phase2': {len(phase2_files)}")
        if phase2_files[:5]:
            print(f"  Examples: {[f.name for f in phase2_files[:5]]}")
        
        # Try alternative pattern (in case of naming variations)
        files = list(results_dir.glob("phase2_*_uncertainty_values.csv"))
        print(f"  Trying 'phase2_*_uncertainty_values.csv': found {len(files)}")
    
    if not files:
        raise FileNotFoundError(f"No phase2*_uncertainty_values.csv files in {results_dir}")
    
    print(f"\nFound {len(files)} files")
    
    all_data = []
    for filepath in sorted(files):
        df = pd.read_csv(filepath)
        df['source_file'] = filepath.name
        
        # Standardize column names
        if 'model' in df.columns and 'model_name' not in df.columns:
            df['model_name'] = df['model']
        
        required = ['model_name', 'representation', 'sigma', 'y_pred_mean',
                   'y_pred_std_calibrated', 'y_true_noisy', 'y_true_original']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            print(f"⚠️  {filepath.name}: missing {missing}")
            continue
        
        models = sorted(df['model_name'].unique())
        reps = sorted(df['representation'].unique())
        sigmas = sorted(df['sigma'].unique())
        
        print(f"✓ {filepath.name}:")
        print(f"    {len(df):,} rows | models={models}")
        print(f"    reps={reps} | σ={sigmas}")
        
        all_data.append(df)
    
    if not all_data:
        raise ValueError("No valid data loaded")
    
    combined = pd.concat(all_data, ignore_index=True)
    
    print(f"\n{'='*80}")
    print("COMBINED DATA")
    print(f"{'='*80}")
    print(f"Total rows: {len(combined):,}")
    print(f"Models: {sorted(combined['model_name'].unique())}")
    print(f"Representations: {sorted(combined['representation'].unique())}")
    print(f"Sigma levels: {sorted(combined['sigma'].unique())}")
    
    has_decomp = ('epistemic_uncertainty' in combined.columns and
                  'aleatoric_uncertainty' in combined.columns)
    print(f"Decomposition: {'✓ YES' if has_decomp else '✗ NO'}")
    
    return combined


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_ece(uncertainties, errors, n_bins=10):
    """Calculate Expected Calibration Error"""
    if len(uncertainties) < n_bins:
        return np.nan
    
    bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[-1] += 1e-8
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            expected = uncertainties[in_bin].mean()
            observed = errors[in_bin].mean()
            ece += (in_bin.sum() / len(uncertainties)) * abs(expected - observed)
    
    return ece


def calculate_metrics(uncertainty_df):
    """Calculate comprehensive metrics for each model/rep/sigma"""
    print("\n" + "="*80)
    print("CALCULATING METRICS")
    print("="*80)
    
    metrics = []
    
    for (model, rep, sigma), group in uncertainty_df.groupby(['model_name', 'representation', 'sigma']):
        errors = np.abs(group['y_true_noisy'] - group['y_pred_mean'])
        uncertainties = group['y_pred_std_calibrated']
        
        if uncertainties.isna().all() or len(uncertainties) < 10:
            continue
        
        valid_mask = ~(uncertainties.isna() | errors.isna())
        uncertainties = uncertainties[valid_mask]
        errors = errors[valid_mask]
        
        if len(uncertainties) < 10:
            continue
        
        # Correlation
        if len(errors) > 1 and uncertainties.std() > 0:
            correlation, p_value = stats.pearsonr(uncertainties, errors)
        else:
            correlation, p_value = np.nan, np.nan
        
        # ECE
        ece = calculate_ece(uncertainties.values, errors.values)
        
        # Coverage
        coverage_1std = np.mean(errors <= uncertainties)
        coverage_2std = np.mean(errors <= 2 * uncertainties)
        
        # MAE
        mae = errors.mean()
        
        metrics.append({
            'model_name': model,
            'representation': rep,
            'sigma': sigma,
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'mean_uncertainty': uncertainties.mean(),
            'std_uncertainty': uncertainties.std(),
            'mean_absolute_error': mae,
            'ece': ece,
            'coverage_1std': coverage_1std,
            'coverage_2std': coverage_2std,
            'n_samples': len(group),
            'median_uncertainty': uncertainties.median(),
            'median_error': errors.median()
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    print(f"✓ Calculated metrics for {len(metrics_df)} configurations")
    print(f"  Models: {len(metrics_df['model_name'].unique())}")
    print(f"  Representations: {len(metrics_df['representation'].unique())}")
    print(f"  Sigma levels: {len(metrics_df['sigma'].unique())}")
    
    return metrics_df


def calculate_decomposition(uncertainty_df):
    """Calculate epistemic/aleatoric decomposition"""
    print("\n" + "="*80)
    print("CALCULATING DECOMPOSITION")
    print("="*80)
    
    if 'epistemic_uncertainty' not in uncertainty_df.columns:
        print("⚠️  No epistemic/aleatoric columns found")
        return pd.DataFrame()
    
    decomp_metrics = []
    
    for (model, rep, sigma), group in uncertainty_df.groupby(['model_name', 'representation', 'sigma']):
        epistemic = group['epistemic_uncertainty']
        aleatoric = group['aleatoric_uncertainty']
        
        if epistemic.isna().all() or aleatoric.isna().all():
            continue
        
        valid_mask = ~(epistemic.isna() | aleatoric.isna())
        epistemic = epistemic[valid_mask]
        aleatoric = aleatoric[valid_mask]
        
        if len(epistemic) < 10:
            continue
        
        mean_epistemic = epistemic.mean()
        mean_aleatoric = aleatoric.mean()
        total = np.sqrt(epistemic**2 + aleatoric**2).mean()
        ratio = mean_epistemic / mean_aleatoric if mean_aleatoric > 0 else np.nan
        
        decomp_metrics.append({
            'model_name': model,
            'representation': rep,
            'sigma': sigma,
            'mean_epistemic': mean_epistemic,
            'mean_aleatoric': mean_aleatoric,
            'mean_total': total,
            'epistemic_aleatoric_ratio': ratio,
            'n_samples': len(epistemic)
        })
    
    decomp_df = pd.DataFrame(decomp_metrics)
    print(f"✓ Calculated decomposition for {len(decomp_df)} configurations")
    
    return decomp_df


def get_models_for_rep(metrics_df, rep, max_models=5):
    """Get available models for a representation, prioritizing core models"""
    rep_data = metrics_df[metrics_df['representation'] == rep]
    available = set(rep_data['model_name'].unique())
    
    # Filter to models with at least 2 sigma levels
    good_models = []
    for model in available:
        model_data = rep_data[rep_data['model_name'] == model]
        if len(model_data['sigma'].unique()) >= 2:
            good_models.append(model)
    
    available = set(good_models)
    
    # Prioritize core models
    result = [m for m in EXTENDED_MODELS if m in available]
    
    # Add any remaining
    for m in sorted(available):
        if m not in result:
            result.append(m)
    
    return result[:max_models]


# ============================================================================
# FIGURE 4: UNCERTAINTY-ERROR RELATIONSHIPS
# ============================================================================

def create_figure4(uncertainty_df, metrics_df, output_dir):
    """
    Figure 4: Uncertainty-Error Relationships
    
    For each representation:
    - Panel A: Scatter plot at σ=0.3 (best 2 models)
    - Panel B: Correlation across σ levels
    - Panel C: Uncertainty inflation
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: Uncertainty-Error")
    print("="*80)
    
    reps = sorted(metrics_df['representation'].unique())
    n_rows = len(reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    # Global y-axis limits for correlation
    valid_corr = metrics_df['correlation'].dropna()
    if len(valid_corr) > 0:
        corr_min = max(-0.2, valid_corr.min() - 0.05)
        corr_max = min(1.0, valid_corr.max() + 0.1)
    else:
        corr_min, corr_max = -0.2, 0.6
    
    fig = plt.figure(figsize=(16, 4*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.28,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        models = get_models_for_rep(metrics_df, rep)
        rep_name = get_rep_display(rep)
        
        if len(models) == 0:
            continue
        
        print(f"  {rep}: {models}")
        
        # ====================================================================
        # Panel A: Scatter at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        # Pick best 2 models by correlation at σ=0.3
        sigma_03 = rep_metrics[np.abs(rep_metrics['sigma'] - 0.3) < 0.1]
        if len(sigma_03) > 0:
            scatter_models = sigma_03.nlargest(2, 'correlation')['model_name'].tolist()
        else:
            scatter_models = models[:2]
        
        for model in scatter_models:
            data = uncertainty_df[
                (uncertainty_df['model_name'] == model) &
                (uncertainty_df['representation'] == rep) &
                (np.abs(uncertainty_df['sigma'] - 0.3) < 0.1)
            ]
            
            if len(data) < 50:
                continue
            
            if len(data) > 2000:
                data = data.sample(2000, random_state=42)
            
            errors = np.abs(data['y_true_noisy'] - data['y_pred_mean']).values
            uncertainties = data['y_pred_std_calibrated'].values
            
            valid = ~(np.isnan(errors) | np.isnan(uncertainties))
            errors, uncertainties = errors[valid], uncertainties[valid]
            
            if len(errors) < 30:
                continue
            
            color = MODEL_COLORS.get(model, '#999999')
            ax.scatter(uncertainties, errors, s=6, alpha=0.25, color=color,
                      edgecolors='none', rasterized=True, label=get_display_name(model))
        
        # Perfect calibration line
        if len(ax.collections) > 0:
            all_vals = []
            for col in ax.collections:
                offsets = col.get_offsets()
                if len(offsets) > 0:
                    all_vals.extend(offsets[:, 0])
                    all_vals.extend(offsets[:, 1])
            if all_vals:
                max_val = max(all_vals) * 1.05
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5,
                       label='y=x', zorder=10)
        
        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Uncertainty vs Error (σ=0.3)',
                    fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel B: Correlation across σ
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['correlation'],
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.axhline(0.3, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Uncertainty-Error Correlation')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: UQ Quality', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
        ax.set_ylim(corr_min, corr_max)
        ax.set_xlim(-0.02, max(metrics_df['sigma']) + 0.05)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel C: Uncertainty inflation
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        for model in models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_uncertainty'],
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        # Ideal inflation line
        sigma_max = metrics_df['sigma'].max()
        sigma_range = np.linspace(0, sigma_max, 20)
        baseline_data = rep_metrics[np.abs(rep_metrics['sigma']) < 0.05]
        if len(baseline_data) > 0:
            baseline = baseline_data['mean_uncertainty'].median()
            if not np.isnan(baseline):
                ax.plot(sigma_range, baseline + sigma_range, 'k--',
                       linewidth=1.5, alpha=0.5, label='Ideal (+σ)')
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Uncertainty Inflation', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
        ax.set_xlim(-0.02, sigma_max + 0.05)
        sns.despine(ax=ax)
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure4_uncertainty_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: CALIBRATION ANALYSIS
# ============================================================================

def create_figure5(uncertainty_df, metrics_df, output_dir):
    """
    Figure 5: Calibration Analysis
    
    For each representation:
    - Panel A: Reliability diagram at σ=0.3
    - Panel B: Coverage across σ
    - Panel C: ECE heatmap
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 5: Calibration")
    print("="*80)
    
    reps = sorted(metrics_df['representation'].unique())
    n_rows = len(reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    # ECE range for color scaling
    ece_max = metrics_df['ece'].max()
    ece_vmax = max(0.5, np.ceil(ece_max * 10) / 10)
    
    fig = plt.figure(figsize=(16, 4*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.40, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        models = get_models_for_rep(metrics_df, rep)
        rep_name = get_rep_display(rep)
        
        if len(models) == 0:
            continue
        
        # ====================================================================
        # Panel A: Reliability diagram
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        plotted_any = False
        for model in models[:4]:  # Limit to 4 for clarity
            data = uncertainty_df[
                (uncertainty_df['model_name'] == model) &
                (uncertainty_df['representation'] == rep) &
                (np.abs(uncertainty_df['sigma'] - 0.3) < 0.1)
            ]
            
            if len(data) < 50:
                continue
            
            errors = np.abs(data['y_true_noisy'] - data['y_pred_mean']).values
            uncertainties = data['y_pred_std_calibrated'].values
            
            valid = ~(np.isnan(errors) | np.isnan(uncertainties))
            errors, uncertainties = errors[valid], uncertainties[valid]
            
            if len(errors) < 30:
                continue
            
            # Binned reliability
            n_bins = 10
            try:
                bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8
            except Exception:
                continue
            
            bin_centers, bin_rmse = [], []
            for i in range(n_bins):
                in_bin = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
                if in_bin.sum() > 5:
                    bin_centers.append(uncertainties[in_bin].mean())
                    bin_rmse.append(np.sqrt(np.mean(errors[in_bin]**2)))
            
            if len(bin_centers) >= 3:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(bin_centers, bin_rmse, marker=marker, linewidth=2,
                       markersize=6, color=color, alpha=0.8, label=get_display_name(model))
                plotted_any = True
        
        # Perfect calibration line
        if plotted_any and len(ax.lines) > 0:
            all_vals = []
            for line in ax.lines:
                xdata, ydata = line.get_xdata(), line.get_ydata()
                if len(xdata) > 0:
                    all_vals.extend(xdata)
                    all_vals.extend(ydata)
            if all_vals:
                max_val = max(all_vals) * 1.05
                ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5, alpha=0.5, label='Perfect')
        
        ax.set_xlabel('Mean Predicted Uncertainty')
        ax.set_ylabel('Observed RMSE')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Reliability (σ=0.3)', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel B: Coverage across σ
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                coverage = model_data['coverage_1std'] * 100
                
                ax.plot(model_data['sigma'], coverage,
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        ax.axhline(68, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (68%)')
        ax.axhspan(0, 30, alpha=0.08, color='red', zorder=0)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Coverage at 1σ (%)')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Coverage', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
        ax.set_xlim(-0.02, metrics_df['sigma'].max() + 0.05)
        ax.set_ylim(0, 105)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel C: ECE heatmap
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        pivot = rep_metrics.pivot_table(
            values='ece',
            index='model_name',
            columns='sigma',
            aggfunc='mean'
        )
        
        if len(pivot) > 0:
            # Reorder by priority
            ordered = [m for m in models if m in pivot.index]
            other = [m for m in pivot.index if m not in ordered]
            new_order = ordered + other
            pivot = pivot.reindex([m for m in new_order if m in pivot.index])
            
            im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=ece_vmax)
            
            # Annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > ece_vmax * 0.5 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=7, color=text_color)
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([f'{s:.1f}' for s in pivot.columns], fontsize=7)
            ax.set_yticklabels([get_display_name(m) for m in pivot.index], fontsize=7)
            
            ax.set_xlabel('Noise Level (σ)')
            ax.set_ylabel('Model')
            ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: ECE', fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('ECE (lower = better)', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure5_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 6: MODEL COMPARISON SUMMARY
# ============================================================================

def create_figure6(metrics_df, output_dir):
    """
    Figure 6: Cross-representation model comparison
    
    - Panel A: MAE by representation and model
    - Panel B: Correlation by representation and model  
    - Panel C: Coverage bar chart at σ=0.3
    - Panel D: Overall ranking heatmap
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 6: Model Comparison")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    reps = sorted(metrics_df['representation'].unique())
    all_models = sorted(metrics_df['model_name'].unique())
    
    rep_colors = {rep: plt.cm.Set2(i/len(reps)) for i, rep in enumerate(reps)}
    
    # ====================================================================
    # Panel A: MAE degradation by model (averaged across reps)
    # ====================================================================
    ax = axes[0, 0]
    
    for model in EXTENDED_MODELS:
        if model not in all_models:
            continue
        model_data = metrics_df[metrics_df['model_name'] == model]
        avg = model_data.groupby('sigma')['mean_absolute_error'].agg(['mean', 'std']).reset_index()
        
        if len(avg) >= 2:
            color = MODEL_COLORS.get(model, '#999999')
            marker = MODEL_MARKERS.get(model, 'o')
            ax.plot(avg['sigma'], avg['mean'], marker=marker, linewidth=2,
                   markersize=5, label=get_display_name(model), color=color)
            ax.fill_between(avg['sigma'], avg['mean'] - avg['std'],
                           avg['mean'] + avg['std'], alpha=0.15, color=color)
    
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('MAE (avg across representations)')
    ax.set_title('A. Prediction Error Degradation', fontweight='bold')
    ax.legend(fontsize=7, loc='upper left', framealpha=0.9, ncol=2)
    ax.set_xlim(-0.02, metrics_df['sigma'].max() + 0.05)
    sns.despine(ax=ax)
    
    # ====================================================================
    # Panel B: Correlation by model (averaged)
    # ====================================================================
    ax = axes[0, 1]
    
    for model in EXTENDED_MODELS:
        if model not in all_models:
            continue
        model_data = metrics_df[metrics_df['model_name'] == model]
        avg = model_data.groupby('sigma')['correlation'].agg(['mean', 'std']).reset_index()
        
        if len(avg) >= 2:
            color = MODEL_COLORS.get(model, '#999999')
            marker = MODEL_MARKERS.get(model, 'o')
            ax.plot(avg['sigma'], avg['mean'], marker=marker, linewidth=2,
                   markersize=5, label=get_display_name(model), color=color)
            ax.fill_between(avg['sigma'], avg['mean'] - avg['std'],
                           avg['mean'] + avg['std'], alpha=0.15, color=color)
    
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Noise Level (σ)')
    ax.set_ylabel('Uncertainty-Error Correlation')
    ax.set_title('B. UQ Quality (avg across representations)', fontweight='bold')
    ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
    ax.set_xlim(-0.02, metrics_df['sigma'].max() + 0.05)
    sns.despine(ax=ax)
    
    # ====================================================================
    # Panel C: Coverage bar chart at σ=0.3
    # ====================================================================
    ax = axes[1, 0]
    
    sigma_03 = metrics_df[np.abs(metrics_df['sigma'] - 0.3) < 0.1]
    if len(sigma_03) > 0:
        # Group by model
        cov_by_model = sigma_03.groupby('model_name')['coverage_1std'].mean() * 100
        cov_by_model = cov_by_model.sort_values(ascending=False)
        
        colors = [MODEL_COLORS.get(m, '#999999') for m in cov_by_model.index]
        bars = ax.bar(range(len(cov_by_model)), cov_by_model.values, color=colors, alpha=0.85)
        
        ax.axhline(68, color='#c0392b', linestyle='--', linewidth=2, alpha=0.7, label='Target (68%)')
        
        ax.set_xticks(range(len(cov_by_model)))
        ax.set_xticklabels([get_display_name(m) for m in cov_by_model.index],
                         rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Coverage at 1σ (%)')
        ax.set_title('C. Coverage at σ=0.3', fontweight='bold')
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7, framealpha=0.9)
    
    sns.despine(ax=ax)
    
    # ====================================================================
    # Panel D: Summary heatmap (rank by correlation + coverage)
    # ====================================================================
    ax = axes[1, 1]
    
    # Create summary metric: normalized correlation + normalized coverage
    summary = metrics_df.groupby(['model_name', 'representation']).agg({
        'correlation': 'mean',
        'coverage_1std': 'mean',
        'ece': 'mean'
    }).reset_index()
    
    if len(summary) > 0:
        pivot = summary.pivot_table(
            values='correlation',
            index='model_name',
            columns='representation',
            aggfunc='mean'
        )
        
        if len(pivot) > 0:
            # Order models by mean correlation
            model_order = pivot.mean(axis=1).sort_values(ascending=False).index
            pivot = pivot.reindex(model_order)
            
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto', vmin=-0.1, vmax=0.5)
            
            # Annotations
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if abs(val) < 0.15 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                               fontsize=8, color=text_color)
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([get_rep_display(r) for r in pivot.columns], fontsize=8)
            ax.set_yticklabels([get_display_name(m) for m in pivot.index], fontsize=8)
            
            ax.set_xlabel('Representation')
            ax.set_ylabel('Model')
            ax.set_title('D. Correlation by Model × Representation', fontweight='bold')
            
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation', fontsize=7)
            cbar.ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "figure6_model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 7: DECOMPOSITION (if available)
# ============================================================================

def create_figure7(decomp_df, output_dir):
    """
    Figure 7: Epistemic/Aleatoric Decomposition
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 7: Decomposition")
    print("="*80)
    
    if len(decomp_df) == 0:
        print("⚠️  No decomposition data")
        return
    
    reps = sorted(decomp_df['representation'].unique())
    n_rows = len(reps)
    
    # Global limits
    global_epist_max = decomp_df['mean_epistemic'].max() * 1.15
    global_alea_max = decomp_df['mean_aleatoric'].max() * 1.15
    
    fig = plt.figure(figsize=(14, 4*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.40, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(reps):
        rep_data = decomp_df[decomp_df['representation'] == rep]
        
        # Get models with data
        model_counts = rep_data.groupby('model_name')['sigma'].nunique()
        available = [m for m in model_counts[model_counts >= 2].index]
        
        models = [m for m in EXTENDED_MODELS if m in available][:5]
        rep_name = get_rep_display(rep)
        
        if len(models) == 0:
            continue
        
        # ====================================================================
        # Panel A: Epistemic
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 0])
        
        for model in models:
            model_data = rep_data[rep_data['model_name'] == model].sort_values('sigma')
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_epistemic'],
                       marker=marker, linewidth=2, markersize=5,
                       label=get_display_name(model), color=color)
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Epistemic Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Epistemic', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_xlim(-0.02, decomp_df['sigma'].max() + 0.05)
        ax.set_ylim(0, global_epist_max)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel B: Aleatoric
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 1])
        
        for model in models:
            model_data = rep_data[rep_data['model_name'] == model].sort_values('sigma')
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_aleatoric'],
                       marker=marker, linewidth=2, markersize=5,
                       label=get_display_name(model), color=color)
        
        # Ideal line
        sigma_max = decomp_df['sigma'].max()
        ax.plot([0, sigma_max], [0, sigma_max], 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Aleatoric Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Aleatoric', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9)
        ax.set_xlim(-0.02, sigma_max + 0.05)
        ax.set_ylim(0, global_alea_max)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # ====================================================================
        # Panel C: Stacked bar at σ=0.3
        # ====================================================================
        ax = fig.add_subplot(gs[row_idx, 2])
        
        sigma_03 = rep_data[np.abs(rep_data['sigma'] - 0.3) < 0.1]
        
        if len(sigma_03) > 0:
            models_with_data = [m for m in models if m in sigma_03['model_name'].values]
            
            if len(models_with_data) > 0:
                x = np.arange(len(models_with_data))
                
                epist = [sigma_03[sigma_03['model_name'] == m]['mean_epistemic'].mean()
                        for m in models_with_data]
                alea = [sigma_03[sigma_03['model_name'] == m]['mean_aleatoric'].mean()
                       for m in models_with_data]
                ratios = [sigma_03[sigma_03['model_name'] == m]['epistemic_aleatoric_ratio'].mean()
                         for m in models_with_data]
                
                ax.bar(x, alea, label='Aleatoric', alpha=0.85, color='#3498db', edgecolor='white')
                ax.bar(x, epist, bottom=alea, label='Epistemic', alpha=0.85, color='#e74c3c', edgecolor='white')
                
                # Ratio annotations
                for i, (e, a, r) in enumerate(zip(epist, alea, ratios)):
                    if not np.isnan(r):
                        ax.text(i, e + a + 0.02, f'{r:.2f}',
                               ha='center', va='bottom', fontsize=7, fontweight='bold')
                
                ax.set_xticks(x)
                ax.set_xticklabels([get_display_name(m) for m in models_with_data],
                                  rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Uncertainty')
                ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Decomposition (σ=0.3)', fontweight='bold')
                ax.legend(fontsize=7, loc='upper right', framealpha=0.9)
                
                ax.text(0.98, 0.98, 'E/A ratio', transform=ax.transAxes,
                       ha='right', va='top', fontsize=6, style='italic', color='gray')
        
        sns.despine(ax=ax)
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure7_decomposition.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# TABLES
# ============================================================================

def create_tables(metrics_df, decomp_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Overall summary
    table1 = metrics_df.groupby(['model_name', 'representation']).agg({
        'correlation': ['mean', 'std'],
        'ece': ['mean', 'std'],
        'coverage_1std': ['mean', 'std'],
        'mean_absolute_error': ['mean', 'std'],
        'n_samples': 'sum'
    }).round(4)
    table1.columns = ['_'.join(col).strip() for col in table1.columns.values]
    table1.to_csv(output_dir / "table1_model_summary.csv")
    print(f"✓ table1_model_summary.csv")
    
    # Table 2: Per sigma level
    for sigma in sorted(metrics_df['sigma'].unique()):
        sigma_data = metrics_df[np.abs(metrics_df['sigma'] - sigma) < 0.05]
        if len(sigma_data) > 0:
            table = sigma_data.pivot_table(
                values=['correlation', 'ece', 'coverage_1std', 'mean_absolute_error'],
                index='model_name',
                columns='representation',
                aggfunc='mean'
            ).round(4)
            table.to_csv(output_dir / f"table2_sigma{sigma:.1f}.csv")
            print(f"✓ table2_sigma{sigma:.1f}.csv")
    
    # Table 3: Decomposition
    if len(decomp_df) > 0:
        ratio_table = decomp_df.groupby(['model_name', 'representation']).agg({
            'epistemic_aleatoric_ratio': ['mean', 'std'],
            'mean_epistemic': 'mean',
            'mean_aleatoric': 'mean'
        }).round(4)
        ratio_table.columns = ['_'.join(col).strip() for col in ratio_table.columns.values]
        ratio_table.to_csv(output_dir / "table3_decomposition.csv")
        print(f"✓ table3_decomposition.csv")
    
    # Table 4: MAE degradation
    mae_pivot = metrics_df.pivot_table(
        values='mean_absolute_error',
        index='model_name',
        columns='sigma',
        aggfunc='mean'
    ).round(4)
    mae_pivot.to_csv(output_dir / "table4_mae_degradation.csv")
    print(f"✓ table4_mae_degradation.csv")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="results"):
    """Main analysis"""
    print("="*80)
    print("PHASE 2: UNCERTAINTY ANALYSIS (UNIFIED)")
    print("="*80)
    
    # Load data
    uncertainty_df = load_uncertainty_data(results_dir)
    if len(uncertainty_df) == 0:
        raise ValueError("No data loaded")
    
    # Calculate metrics
    metrics_df = calculate_metrics(uncertainty_df)
    if len(metrics_df) == 0:
        raise ValueError("No metrics calculated")
    
    # Calculate decomposition
    decomp_df = calculate_decomposition(uncertainty_df)
    
    # Output directory
    output_dir = Path(results_dir) / "phase2_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\n✓ Saved metrics.csv")
    
    if len(decomp_df) > 0:
        decomp_df.to_csv(output_dir / "decomposition.csv", index=False)
        print(f"✓ Saved decomposition.csv")
    
    # Generate figures
    create_figure4(uncertainty_df, metrics_df, output_dir)
    create_figure5(uncertainty_df, metrics_df, output_dir)
    create_figure6(metrics_df, output_dir)
    
    if len(decomp_df) > 0:
        create_figure7(decomp_df, output_dir)
    
    # Generate tables
    create_tables(metrics_df, decomp_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nData: {len(uncertainty_df):,} samples")
    print(f"Configurations: {len(metrics_df)}")
    print(f"Models: {sorted(metrics_df['model_name'].unique())}")
    print(f"Representations: {sorted(metrics_df['representation'].unique())}")
    
    print("\n" + "-"*40)
    print("KEY METRICS BY MODEL")
    print("-"*40)
    
    summary = metrics_df.groupby('model_name').agg({
        'correlation': 'mean',
        'coverage_1std': lambda x: x.mean() * 100,
        'ece': 'mean',
        'mean_absolute_error': 'mean'
    }).round(3)
    summary.columns = ['Corr', 'Cov%', 'ECE', 'MAE']
    print(summary.sort_values('Corr', ascending=False).to_string())
    
    # Highlight best performers
    print("\n" + "-"*40)
    print("TOP PERFORMERS")
    print("-"*40)
    print(f"Best UQ correlation: {summary['Corr'].idxmax()} ({summary['Corr'].max():.3f})")
    print(f"Best coverage: {summary['Cov%'].idxmax()} ({summary['Cov%'].max():.1f}%)")
    print(f"Lowest ECE: {summary['ECE'].idxmin()} ({summary['ECE'].min():.3f})")
    print(f"Lowest MAE: {summary['MAE'].idxmin()} ({summary['MAE'].min():.3f})")
    
    # Warnings
    low_coverage = summary[summary['Cov%'] < 50]
    if len(low_coverage) > 0:
        print(f"\n⚠️  Low coverage (<50%): {list(low_coverage.index)}")
    
    low_corr = summary[summary['Corr'] < 0.1]
    if len(low_corr) > 0:
        print(f"⚠️  Poor UQ correlation (<0.1): {list(low_corr.index)}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs: {output_dir}")
    print("\nGenerated files:")
    print("  - metrics.csv")
    print("  - figure4_uncertainty_error.png")
    print("  - figure5_calibration.png")
    print("  - figure6_model_comparison.png")
    if len(decomp_df) > 0:
        print("  - figure7_decomposition.png")
    print("  - table1_model_summary.csv")
    print("  - table2_sigma*.csv")
    if len(decomp_df) > 0:
        print("  - table3_decomposition.csv")
    print("  - table4_mae_degradation.csv")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    main(results_dir)