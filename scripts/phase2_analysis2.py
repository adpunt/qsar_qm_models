"""
Phase 2 Uncertainty Analysis - Enhanced Version with Noise-Robustness Integration
==================================================================================

Core analysis:
- Uncertainty-Error relationships (does UQ predict errors?)
- Calibration analysis (ECE, coverage)
- Model comparison

NEW - Noise-Robustness Integration:
- Uncertainty response to noise (does uncertainty scale with σ?)
- Aleatoric tracking of injected noise
- Cross-phase analysis: UQ quality vs Noise Degradation Slope
- Identifying configurations where uncertainty predicts noise-induced errors

This is central to the thesis: good uncertainty quantification should help
identify when predictions become unreliable under noise.

Usage:
    python phase2_analysis_v3.py results/
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
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

REPRESENTATION_COLORS = {
    'pdv': '#0173B2',
    'continuous_pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'smiles': '#CA3542',
    'mhggnn': '#CC79A7',
}

MODEL_DISPLAY = {
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    'bnn_full': 'BNN-Full',
    'bnn_last': 'BNN-Last',
    'bnn_variational': 'BNN-Var',
    'gauche': 'GP',
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
    'ecfp4': 'ECFP4',
}

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
    
    results_dir = Path(results_dir).resolve()
    print(f"Looking in: {results_dir}")
    
    files = list(results_dir.glob("phase2*_uncertainty_values.csv"))
    
    if not files:
        print(f"\nSearching for alternative patterns...")
        files = list(results_dir.glob("phase2_*_uncertainty_values.csv"))
    
    if not files:
        raise FileNotFoundError(f"No phase2*_uncertainty_values.csv files in {results_dir}")
    
    print(f"\nFound {len(files)} files")
    
    all_data = []
    for filepath in sorted(files):
        df = pd.read_csv(filepath)
        df['source_file'] = filepath.name
        
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


def load_phase0_metrics(results_dir):
    """Load Phase 0 robustness metrics for cross-phase analysis"""
    print("\n" + "="*80)
    print("LOADING PHASE 0 ROBUSTNESS METRICS")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Try different possible locations
    possible_paths = [
        results_dir / "phase0_figures_v3" / "phase0_robustness_metrics.csv",
        results_dir / "phase0_figures" / "phase0_robustness_metrics.csv",
        results_dir / "phase0_robustness_metrics.csv",
    ]
    
    for path in possible_paths:
        if path.exists():
            print(f"✓ Found Phase 0 metrics at {path}")
            df = pd.read_csv(path)
            
            # Standardize column names
            if 'ds_thresholded' in df.columns:
                df['nds'] = df['ds_thresholded']
            elif 'nds_thresholded' in df.columns:
                df['nds'] = df['nds_thresholded']
            elif 'nsi_r2' in df.columns:
                df['nds'] = df['nsi_r2']
            
            print(f"  Loaded {len(df)} configurations")
            return df
    
    print("⚠️  Phase 0 metrics not found - cross-phase analysis will be limited")
    return pd.DataFrame()


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
    print("CALCULATING UNCERTAINTY METRICS")
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
        
        # Correlation between uncertainty and error
        if len(errors) > 1 and uncertainties.std() > 0:
            correlation, p_value = pearsonr(uncertainties, errors)
        else:
            correlation, p_value = np.nan, np.nan
        
        # ECE
        ece = calculate_ece(uncertainties.values, errors.values)
        
        # Coverage
        coverage_1std = np.mean(errors <= uncertainties)
        coverage_2std = np.mean(errors <= 2 * uncertainties)
        
        # MAE and RMSE
        mae = errors.mean()
        rmse = np.sqrt((errors**2).mean())
        
        # R² (from original values if available)
        if 'y_true_original' in group.columns:
            y_true = group.loc[valid_mask.index[valid_mask], 'y_true_original']
            y_pred = group.loc[valid_mask.index[valid_mask], 'y_pred_mean']
            ss_res = ((y_true - y_pred)**2).sum()
            ss_tot = ((y_true - y_true.mean())**2).sum()
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
        else:
            r2 = np.nan
        
        metrics.append({
            'model_name': model,
            'representation': rep,
            'sigma': sigma,
            'correlation': correlation,
            'correlation_pvalue': p_value,
            'mean_uncertainty': uncertainties.mean(),
            'std_uncertainty': uncertainties.std(),
            'mean_absolute_error': mae,
            'rmse': rmse,
            'r2': r2,
            'ece': ece,
            'coverage_1std': coverage_1std,
            'coverage_2std': coverage_2std,
            'n_samples': len(uncertainties),
            'median_uncertainty': uncertainties.median(),
            'median_error': errors.median()
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    print(f"✓ Calculated metrics for {len(metrics_df)} configurations")
    print(f"  Models: {len(metrics_df['model_name'].unique())}")
    print(f"  Representations: {len(metrics_df['representation'].unique())}")
    print(f"  Sigma levels: {len(metrics_df['sigma'].unique())}")
    
    return metrics_df


def calculate_uncertainty_noise_response(uncertainty_df):
    """
    Calculate how uncertainty responds to noise level.
    
    Key metrics:
    - Uncertainty inflation slope: d(mean_uncertainty)/d(sigma)
    - Ideal inflation: uncertainty should increase ~linearly with sigma
    - Uncertainty-sigma correlation: does uncertainty track noise?
    """
    print("\n" + "="*80)
    print("CALCULATING UNCERTAINTY-NOISE RESPONSE")
    print("="*80)
    
    response_metrics = []
    
    for (model, rep), group in uncertainty_df.groupby(['model_name', 'representation']):
        # Get mean uncertainty at each sigma level
        sigma_means = group.groupby('sigma').agg({
            'y_pred_std_calibrated': 'mean',
            'y_pred_mean': 'count'
        }).reset_index()
        sigma_means.columns = ['sigma', 'mean_uncertainty', 'n_samples']
        
        if len(sigma_means) < 3:
            continue
        
        # Calculate uncertainty inflation slope
        try:
            slope, intercept, r_val, p_val, _ = stats.linregress(
                sigma_means['sigma'], sigma_means['mean_uncertainty']
            )
        except:
            continue
        
        # Baseline uncertainty (at sigma=0)
        baseline = sigma_means[sigma_means['sigma'] == 0]['mean_uncertainty']
        baseline_unc = baseline.values[0] if len(baseline) > 0 else sigma_means['mean_uncertainty'].min()
        
        # Maximum uncertainty
        max_sigma = sigma_means['sigma'].max()
        max_unc = sigma_means[sigma_means['sigma'] == max_sigma]['mean_uncertainty'].values[0]
        
        # Ideal slope would be ~1 (uncertainty increases 1:1 with noise)
        # But this depends on the scale, so we use relative inflation
        relative_inflation = (max_unc - baseline_unc) / max_sigma if max_sigma > 0 else np.nan
        
        response_metrics.append({
            'model_name': model,
            'representation': rep,
            'uncertainty_inflation_slope': slope,
            'uncertainty_inflation_r2': r_val**2,
            'uncertainty_inflation_pval': p_val,
            'baseline_uncertainty': baseline_unc,
            'max_uncertainty': max_unc,
            'relative_inflation': relative_inflation,
            'inflation_intercept': intercept,
        })
    
    response_df = pd.DataFrame(response_metrics)
    print(f"✓ Calculated uncertainty-noise response for {len(response_df)} configurations")
    
    return response_df


def calculate_decomposition_response(uncertainty_df):
    """
    Calculate how epistemic and aleatoric uncertainty respond to noise.
    
    Key insight: Aleatoric should track injected noise, epistemic should be stable.
    """
    print("\n" + "="*80)
    print("CALCULATING DECOMPOSITION RESPONSE TO NOISE")
    print("="*80)
    
    if 'epistemic_uncertainty' not in uncertainty_df.columns:
        print("⚠️  No decomposition data available")
        return pd.DataFrame()
    
    decomp_response = []
    
    for (model, rep), group in uncertainty_df.groupby(['model_name', 'representation']):
        # Get means at each sigma
        sigma_means = group.groupby('sigma').agg({
            'epistemic_uncertainty': 'mean',
            'aleatoric_uncertainty': 'mean',
        }).reset_index()
        
        if len(sigma_means) < 3:
            continue
        
        # Aleatoric inflation slope (should be ~1 ideally)
        try:
            alea_slope, alea_int, alea_r, alea_p, _ = stats.linregress(
                sigma_means['sigma'], sigma_means['aleatoric_uncertainty']
            )
        except:
            alea_slope, alea_r, alea_p = np.nan, np.nan, np.nan
        
        # Epistemic stability (should be ~0 ideally)
        try:
            epist_slope, epist_int, epist_r, epist_p, _ = stats.linregress(
                sigma_means['sigma'], sigma_means['epistemic_uncertainty']
            )
        except:
            epist_slope, epist_r, epist_p = np.nan, np.nan, np.nan
        
        # Ratio at different noise levels
        baseline = sigma_means[sigma_means['sigma'] == 0]
        high_noise = sigma_means[sigma_means['sigma'] == sigma_means['sigma'].max()]
        
        if len(baseline) > 0 and len(high_noise) > 0:
            baseline_ratio = (baseline['epistemic_uncertainty'].values[0] / 
                            baseline['aleatoric_uncertainty'].values[0]
                            if baseline['aleatoric_uncertainty'].values[0] > 0 else np.nan)
            high_ratio = (high_noise['epistemic_uncertainty'].values[0] / 
                         high_noise['aleatoric_uncertainty'].values[0]
                         if high_noise['aleatoric_uncertainty'].values[0] > 0 else np.nan)
        else:
            baseline_ratio, high_ratio = np.nan, np.nan
        
        decomp_response.append({
            'model_name': model,
            'representation': rep,
            'aleatoric_slope': alea_slope,
            'aleatoric_r2': alea_r**2 if not np.isnan(alea_r) else np.nan,
            'aleatoric_tracks_noise': alea_slope > 0.5 and alea_r**2 > 0.8 if not np.isnan(alea_slope) else False,
            'epistemic_slope': epist_slope,
            'epistemic_r2': epist_r**2 if not np.isnan(epist_r) else np.nan,
            'epistemic_stable': abs(epist_slope) < 0.1 if not np.isnan(epist_slope) else False,
            'baseline_epist_alea_ratio': baseline_ratio,
            'high_noise_epist_alea_ratio': high_ratio,
        })
    
    decomp_df = pd.DataFrame(decomp_response)
    print(f"✓ Calculated decomposition response for {len(decomp_df)} configurations")
    
    return decomp_df


def calculate_cross_phase_metrics(metrics_df, phase0_df):
    """
    Merge Phase 2 UQ metrics with Phase 0 NDS metrics.
    
    Key question: Do models with better UQ also have better noise robustness?
    """
    print("\n" + "="*80)
    print("CALCULATING CROSS-PHASE METRICS (UQ vs NDS)")
    print("="*80)
    
    if len(phase0_df) == 0:
        print("⚠️  No Phase 0 data - skipping cross-phase analysis")
        return pd.DataFrame()
    
    # Aggregate Phase 2 metrics across sigma levels
    phase2_agg = metrics_df.groupby(['model_name', 'representation']).agg({
        'correlation': 'mean',
        'ece': 'mean',
        'coverage_1std': 'mean',
        'mean_uncertainty': 'mean',
        'mean_absolute_error': 'mean',
    }).reset_index()
    
    phase2_agg.columns = ['model_name', 'representation', 'mean_uq_correlation', 
                          'mean_ece', 'mean_coverage', 'mean_uncertainty', 'mean_mae']
    
    # Standardize Phase 0 column names
    phase0_clean = phase0_df.copy()
    if 'model' in phase0_clean.columns:
        phase0_clean = phase0_clean.rename(columns={'model': 'model_name'})
    
    # Merge
    cross_df = phase2_agg.merge(
        phase0_clean[['model_name', 'representation', 'baseline_r2', 'nds', 'meets_baseline_threshold']],
        on=['model_name', 'representation'],
        how='inner'
    )
    
    print(f"✓ Merged {len(cross_df)} configurations with both UQ and NDS data")
    
    if len(cross_df) > 3:
        # Calculate correlation between UQ quality and NDS
        valid = cross_df.dropna(subset=['mean_uq_correlation', 'nds'])
        if len(valid) > 3:
            corr, pval = spearmanr(valid['mean_uq_correlation'], valid['nds'].abs())
            print(f"  UQ-Error Correlation vs |NDS|: ρ={corr:.3f}, p={pval:.4f}")
    
    return cross_df


# ============================================================================
# FIGURE 4: UNCERTAINTY-ERROR RELATIONSHIPS (EXISTING)
# ============================================================================

def create_figure4(uncertainty_df, metrics_df, output_dir):
    """Figure 4: Uncertainty-Error Relationships"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 4: Uncertainty-Error")
    print("="*80)
    
    reps = sorted(metrics_df['representation'].unique())
    n_rows = len(reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    fig = plt.figure(figsize=(16, 4*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.35, wspace=0.28,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        models = _get_models_for_rep(metrics_df, rep)
        rep_name = get_rep_display(rep)
        
        if len(models) == 0:
            continue
        
        # Panel A: Scatter at σ=0.3
        ax = fig.add_subplot(gs[row_idx, 0])
        sigma_03 = rep_metrics[np.abs(rep_metrics['sigma'] - 0.3) < 0.1]
        scatter_models = sigma_03.nlargest(2, 'correlation')['model_name'].tolist() if len(sigma_03) > 0 else models[:2]
        
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
        
        if len(ax.collections) > 0:
            all_vals = []
            for col in ax.collections:
                offsets = col.get_offsets()
                if len(offsets) > 0:
                    all_vals.extend(offsets[:, 0])
                    all_vals.extend(offsets[:, 1])
            if all_vals:
                max_val = max(all_vals) * 1.05
                ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.6, linewidth=1.5, label='y=x', zorder=10)
        
        ax.set_xlabel('Predicted Uncertainty')
        ax.set_ylabel('Absolute Error')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Uncertainty vs Error (σ=0.3)', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left', framealpha=0.9)
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # Panel B: Correlation across σ
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
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Uncertainty-Error Correlation')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: UQ Quality Across Noise', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # Panel C: Uncertainty inflation
        ax = fig.add_subplot(gs[row_idx, 2])
        for model in models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                ax.plot(model_data['sigma'], model_data['mean_uncertainty'],
                       marker=marker, linewidth=2, markersize=5, alpha=0.9,
                       label=get_display_name(model), color=color)
        
        sigma_max = metrics_df['sigma'].max()
        sigma_range = np.linspace(0, sigma_max, 20)
        baseline_data = rep_metrics[np.abs(rep_metrics['sigma']) < 0.05]
        if len(baseline_data) > 0:
            baseline = baseline_data['mean_uncertainty'].median()
            if not np.isnan(baseline):
                ax.plot(sigma_range, baseline + sigma_range, 'k--', linewidth=1.5, alpha=0.5, label='Ideal (+σ)')
        
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Mean Predicted Uncertainty')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Uncertainty Inflation', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
        sns.despine(ax=ax)
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure4_uncertainty_error.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 5: CALIBRATION ANALYSIS (EXISTING)
# ============================================================================

def create_figure5(uncertainty_df, metrics_df, output_dir):
    """Figure 5: Calibration Analysis"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 5: Calibration")
    print("="*80)
    
    reps = sorted(metrics_df['representation'].unique())
    n_rows = len(reps)
    
    if n_rows == 0:
        print("⚠️  No data available")
        return
    
    fig = plt.figure(figsize=(16, 4*n_rows))
    gs = fig.add_gridspec(n_rows, 3, hspace=0.40, wspace=0.30,
                          left=0.06, right=0.98, top=0.94, bottom=0.06)
    
    panel_idx = 0
    
    for row_idx, rep in enumerate(reps):
        rep_metrics = metrics_df[metrics_df['representation'] == rep]
        models = _get_models_for_rep(metrics_df, rep)
        rep_name = get_rep_display(rep)
        
        if len(models) == 0:
            continue
        
        # Panel A: Reliability diagram
        ax = fig.add_subplot(gs[row_idx, 0])
        for model in models[:4]:
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
            
            n_bins = 10
            try:
                bin_edges = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8
            except:
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
        
        if len(ax.lines) > 0:
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
        
        # Panel B: Coverage across σ
        ax = fig.add_subplot(gs[row_idx, 1])
        for model in models:
            model_data = rep_metrics[rep_metrics['model_name'] == model].sort_values('sigma')
            if len(model_data) >= 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = MODEL_MARKERS.get(model, 'o')
                coverage = model_data['coverage_1std'] * 100
                ax.plot(model_data['sigma'], coverage, marker=marker, linewidth=2, markersize=5,
                       alpha=0.9, label=get_display_name(model), color=color)
        
        ax.axhline(68, color='#c0392b', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (68%)')
        ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('Coverage at 1σ (%)')
        ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: Coverage', fontweight='bold')
        ax.legend(fontsize=7, loc='best', framealpha=0.9, ncol=2)
        ax.set_ylim(0, 105)
        sns.despine(ax=ax)
        panel_idx += 1
        
        # Panel C: ECE heatmap
        ax = fig.add_subplot(gs[row_idx, 2])
        pivot = rep_metrics.pivot_table(values='ece', index='model_name', columns='sigma', aggfunc='mean')
        
        if len(pivot) > 0:
            ordered = [m for m in models if m in pivot.index]
            pivot = pivot.reindex([m for m in ordered if m in pivot.index])
            
            im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.5)
            
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        text_color = 'white' if val > 0.25 else 'black'
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=7, color=text_color)
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels([f'{s:.1f}' for s in pivot.columns], fontsize=7)
            ax.set_yticklabels([get_display_name(m) for m in pivot.index], fontsize=7)
            ax.set_xlabel('Noise Level (σ)')
            ax.set_ylabel('Model')
            ax.set_title(f'{chr(65 + panel_idx)}. {rep_name}: ECE', fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('ECE (lower = better)', fontsize=7)
        
        panel_idx += 1
    
    output_path = Path(output_dir) / "figure5_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# NEW FIGURE 6: UNCERTAINTY RESPONSE TO NOISE
# ============================================================================

def create_figure6_uncertainty_noise_response(uncertainty_df, metrics_df, response_df, output_dir):
    """
    Figure 6: How Uncertainty Responds to Noise
    
    Panel A: Uncertainty inflation curves by model
    Panel B: Inflation slope ranking (bar chart)
    Panel C: Ideal vs actual inflation scatter
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 6: Uncertainty Response to Noise")
    print("="*80)
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: Uncertainty inflation curves
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Average across representations for each model
    model_curves = metrics_df.groupby(['model_name', 'sigma']).agg({
        'mean_uncertainty': 'mean'
    }).reset_index()
    
    models = sorted(model_curves['model_name'].unique())
    for model in models:
        model_data = model_curves[model_curves['model_name'] == model].sort_values('sigma')
        if len(model_data) >= 2:
            color = MODEL_COLORS.get(model, '#999999')
            marker = MODEL_MARKERS.get(model, 'o')
            ax_a.plot(model_data['sigma'], model_data['mean_uncertainty'],
                     marker=marker, linewidth=2, markersize=5, alpha=0.9,
                     label=get_display_name(model), color=color)
    
    # Ideal line
    sigma_max = metrics_df['sigma'].max()
    baseline = model_curves[model_curves['sigma'] == 0]['mean_uncertainty'].mean()
    ax_a.plot([0, sigma_max], [baseline, baseline + sigma_max], 'k--', 
             linewidth=2, alpha=0.5, label='Ideal (slope=1)')
    
    ax_a.set_xlabel('Noise Level (σ)', fontsize=9)
    ax_a.set_ylabel('Mean Predicted Uncertainty', fontsize=9)
    ax_a.set_title('A. Uncertainty Inflation by Model', fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=6, loc='upper left', ncol=2, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL B: Inflation slope ranking
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Average slope across representations
    model_slopes = response_df.groupby('model_name').agg({
        'uncertainty_inflation_slope': 'mean',
        'uncertainty_inflation_r2': 'mean',
    }).reset_index()
    model_slopes = model_slopes.sort_values('uncertainty_inflation_slope', ascending=True)
    
    y_pos = np.arange(len(model_slopes))
    colors = [MODEL_COLORS.get(m, '#999999') for m in model_slopes['model_name']]
    
    bars = ax_b.barh(y_pos, model_slopes['uncertainty_inflation_slope'],
                    color=colors, alpha=0.8, height=0.7, edgecolor='black', linewidth=0.5)
    
    # Add ideal slope line
    ax_b.axvline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (slope=1)')
    
    ax_b.set_yticks(y_pos)
    ax_b.set_yticklabels([get_display_name(m) for m in model_slopes['model_name']], fontsize=8)
    ax_b.set_xlabel('Uncertainty Inflation Slope (d(unc)/d(σ))', fontsize=9)
    ax_b.set_title('B. Uncertainty Response to Noise\n(closer to 1 = better tracking)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL C: Actual vs ideal inflation scatter
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # For each config, plot (ideal inflation, actual inflation)
    # Ideal = increase of σ_max from baseline
    for _, row in response_df.iterrows():
        model = row['model_name']
        rep = row['representation']
        actual_slope = row['uncertainty_inflation_slope']
        r2 = row['uncertainty_inflation_r2']
        
        if np.isnan(actual_slope):
            continue
        
        color = MODEL_COLORS.get(model, '#999999')
        marker = MODEL_MARKERS.get(model, 'o')
        
        # Size by R² (how linear the response is)
        size = 50 + 100 * r2 if not np.isnan(r2) else 50
        
        ax_c.scatter(1.0, actual_slope, s=size, alpha=0.7, color=color, marker=marker,
                    edgecolors='black', linewidth=0.5)
    
    # Add diagonal line
    ax_c.plot([0, 2], [0, 2], 'k--', linewidth=1.5, alpha=0.5, label='Perfect tracking')
    
    # Add horizontal band for "good" tracking
    ax_c.axhspan(0.8, 1.2, alpha=0.1, color='green', label='Good tracking (0.8-1.2)')
    
    ax_c.set_xlabel('Ideal Inflation Slope (=1)', fontsize=9)
    ax_c.set_ylabel('Actual Inflation Slope', fontsize=9)
    ax_c.set_title('C. Tracking Quality\n(point size = linearity R²)', fontsize=10, fontweight='bold', pad=10)
    ax_c.legend(fontsize=7, loc='upper left', framealpha=0.9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    ax_c.set_xlim(0.5, 1.5)
    
    # Add model legend
    legend_handles = [plt.Line2D([0], [0], marker=MODEL_MARKERS.get(m, 'o'), color='w',
                                 markerfacecolor=MODEL_COLORS.get(m, '#999999'),
                                 markersize=8, label=get_display_name(m))
                     for m in sorted(response_df['model_name'].unique())]
    ax_c.legend(handles=legend_handles, fontsize=6, loc='lower right', ncol=2, framealpha=0.9)
    
    output_path = Path(output_dir) / "figure6_uncertainty_noise_response.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# NEW FIGURE 7: CROSS-PHASE ANALYSIS (UQ vs NDS)
# ============================================================================

def create_figure7_cross_phase(cross_df, output_dir):
    """
    Figure 7: Cross-Phase Analysis - UQ Quality vs Noise Robustness
    
    Panel A: UQ-Error Correlation vs |NDS| scatter
    Panel B: Coverage vs |NDS| scatter
    Panel C: Combined ranking comparison
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 7: UQ Quality vs Noise Robustness")
    print("="*80)
    
    if len(cross_df) == 0:
        print("⚠️  No cross-phase data available")
        return
    
    # Filter to threshold-meeting configs
    if 'meets_baseline_threshold' in cross_df.columns:
        cross_df = cross_df[cross_df['meets_baseline_threshold'] == True].copy()
    
    if len(cross_df) < 3:
        print("⚠️  Insufficient data for cross-phase analysis")
        return
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: UQ-Error Correlation vs |NDS|
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    valid = cross_df.dropna(subset=['mean_uq_correlation', 'nds'])
    
    if len(valid) > 0:
        for _, row in valid.iterrows():
            model = row['model_name']
            rep = row['representation']
            color = REPRESENTATION_COLORS.get(rep, MODEL_COLORS.get(model, '#999999'))
            marker = MODEL_MARKERS.get(model, 'o')
            
            ax_a.scatter(row['mean_uq_correlation'], abs(row['nds']),
                        s=80, alpha=0.7, color=color, marker=marker,
                        edgecolors='black', linewidth=0.5)
        
        # Add correlation line if significant
        if len(valid) > 5:
            corr, pval = spearmanr(valid['mean_uq_correlation'], valid['nds'].abs())
            if pval < 0.1:
                z = np.polyfit(valid['mean_uq_correlation'], valid['nds'].abs(), 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid['mean_uq_correlation'].min(), valid['mean_uq_correlation'].max(), 100)
                ax_a.plot(x_line, p(x_line), 'r--', linewidth=2, alpha=0.7)
            
            ax_a.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {pval:.4f}',
                     transform=ax_a.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_a.set_xlabel('Mean Uncertainty-Error Correlation', fontsize=9)
    ax_a.set_ylabel('|Noise Degradation Slope|', fontsize=9)
    ax_a.set_title('A. UQ Quality vs Noise Robustness\n(lower |NDS| = more stable)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL B: Coverage vs |NDS|
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    valid = cross_df.dropna(subset=['mean_coverage', 'nds'])
    
    if len(valid) > 0:
        for _, row in valid.iterrows():
            model = row['model_name']
            rep = row['representation']
            color = REPRESENTATION_COLORS.get(rep, MODEL_COLORS.get(model, '#999999'))
            marker = MODEL_MARKERS.get(model, 'o')
            
            ax_b.scatter(row['mean_coverage'] * 100, abs(row['nds']),
                        s=80, alpha=0.7, color=color, marker=marker,
                        edgecolors='black', linewidth=0.5)
        
        # Ideal coverage line
        ax_b.axvline(68, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Ideal coverage (68%)')
        
        if len(valid) > 5:
            corr, pval = spearmanr(valid['mean_coverage'], valid['nds'].abs())
            ax_b.text(0.05, 0.95, f'Spearman ρ = {corr:.3f}\np = {pval:.4f}',
                     transform=ax_b.transAxes, fontsize=8, va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax_b.set_xlabel('Mean Coverage at 1σ (%)', fontsize=9)
    ax_b.set_ylabel('|Noise Degradation Slope|', fontsize=9)
    ax_b.set_title('B. Calibration vs Noise Robustness', fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=7, loc='upper right', framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL C: Combined ranking heatmap
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Create normalized scores
    valid = cross_df.dropna(subset=['mean_uq_correlation', 'nds', 'mean_coverage']).copy()
    
    if len(valid) > 3:
        # Normalize each metric (higher = better)
        valid['uq_score'] = (valid['mean_uq_correlation'] - valid['mean_uq_correlation'].min()) / \
                           (valid['mean_uq_correlation'].max() - valid['mean_uq_correlation'].min())
        
        # For NDS, less negative (closer to 0) is better
        valid['nds_score'] = 1 - (valid['nds'].abs() - valid['nds'].abs().min()) / \
                            (valid['nds'].abs().max() - valid['nds'].abs().min())
        
        # Coverage: closer to 0.68 is better
        valid['cov_score'] = 1 - np.abs(valid['mean_coverage'] - 0.68) / 0.68
        valid['cov_score'] = valid['cov_score'].clip(0, 1)
        
        # Combined score
        valid['combined_score'] = (valid['uq_score'] + valid['nds_score'] + valid['cov_score']) / 3
        
        # Top 10
        top10 = valid.nlargest(10, 'combined_score')
        
        # Create heatmap data
        heatmap_data = top10[['uq_score', 'nds_score', 'cov_score', 'combined_score']].values
        labels = [f"{get_display_name(row['model_name'])}/{get_rep_display(row['representation'])}" 
                 for _, row in top10.iterrows()]
        
        im = ax_c.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        ax_c.set_xticks([0, 1, 2, 3])
        ax_c.set_xticklabels(['UQ\nCorr', 'Noise\nRobust', 'Coverage', 'Combined'], fontsize=7)
        ax_c.set_yticks(range(len(labels)))
        ax_c.set_yticklabels(labels, fontsize=7)
        
        # Add text annotations
        for i in range(len(labels)):
            for j in range(4):
                val = heatmap_data[i, j]
                text_color = 'white' if val < 0.5 else 'black'
                ax_c.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6, color=text_color)
        
        cbar = plt.colorbar(im, ax=ax_c, shrink=0.8)
        cbar.set_label('Normalized Score', fontsize=7)
        
        ax_c.set_title('C. Top 10 Configurations\n(Combined UQ + Robustness)', fontsize=10, fontweight='bold', pad=10)
    else:
        ax_c.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax_c.transAxes)
        ax_c.axis('off')
    
    output_path = Path(output_dir) / "figure7_uq_vs_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# NEW FIGURE 8: DECOMPOSITION RESPONSE TO NOISE
# ============================================================================

def create_figure8_decomposition_response(uncertainty_df, decomp_response_df, output_dir):
    """
    Figure 8: How Epistemic and Aleatoric Uncertainty Respond to Noise
    
    Panel A: Aleatoric inflation (should track noise)
    Panel B: Epistemic stability (should be constant)
    Panel C: Aleatoric slope vs ideal
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 8: Decomposition Response")
    print("="*80)
    
    if len(decomp_response_df) == 0:
        print("⚠️  No decomposition data available")
        return
    
    has_decomp = 'epistemic_uncertainty' in uncertainty_df.columns
    if not has_decomp:
        print("⚠️  No decomposition columns in uncertainty data")
        return
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, hspace=0.25, wspace=0.30,
                          left=0.06, right=0.98, top=0.88, bottom=0.15)
    
    # ========================================================================
    # PANEL A: Aleatoric inflation curves
    # ========================================================================
    ax_a = fig.add_subplot(gs[0, 0])
    
    # Average aleatoric by model across sigma
    alea_curves = uncertainty_df.groupby(['model_name', 'sigma']).agg({
        'aleatoric_uncertainty': 'mean'
    }).reset_index()
    
    for model in sorted(alea_curves['model_name'].unique()):
        model_data = alea_curves[alea_curves['model_name'] == model].sort_values('sigma')
        if len(model_data) >= 2:
            color = MODEL_COLORS.get(model, '#999999')
            marker = MODEL_MARKERS.get(model, 'o')
            ax_a.plot(model_data['sigma'], model_data['aleatoric_uncertainty'],
                     marker=marker, linewidth=2, markersize=5, alpha=0.9,
                     label=get_display_name(model), color=color)
    
    # Ideal line (aleatoric should equal sigma)
    sigma_max = uncertainty_df['sigma'].max()
    ax_a.plot([0, sigma_max], [0, sigma_max], 'k--', linewidth=2, alpha=0.5, label='Ideal (=σ)')
    
    ax_a.set_xlabel('Injected Noise Level (σ)', fontsize=9)
    ax_a.set_ylabel('Mean Aleatoric Uncertainty', fontsize=9)
    ax_a.set_title('A. Aleatoric Tracks Injected Noise\n(should follow diagonal)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_a.legend(fontsize=6, loc='upper left', ncol=2, framealpha=0.9)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    ax_a.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL B: Epistemic stability
    # ========================================================================
    ax_b = fig.add_subplot(gs[0, 1])
    
    epist_curves = uncertainty_df.groupby(['model_name', 'sigma']).agg({
        'epistemic_uncertainty': 'mean'
    }).reset_index()
    
    for model in sorted(epist_curves['model_name'].unique()):
        model_data = epist_curves[epist_curves['model_name'] == model].sort_values('sigma')
        if len(model_data) >= 2:
            color = MODEL_COLORS.get(model, '#999999')
            marker = MODEL_MARKERS.get(model, 'o')
            ax_b.plot(model_data['sigma'], model_data['epistemic_uncertainty'],
                     marker=marker, linewidth=2, markersize=5, alpha=0.9,
                     label=get_display_name(model), color=color)
    
    ax_b.set_xlabel('Injected Noise Level (σ)', fontsize=9)
    ax_b.set_ylabel('Mean Epistemic Uncertainty', fontsize=9)
    ax_b.set_title('B. Epistemic Stability\n(should be constant - model uncertainty)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_b.legend(fontsize=6, loc='best', ncol=2, framealpha=0.9)
    ax_b.spines['top'].set_visible(False)
    ax_b.spines['right'].set_visible(False)
    ax_b.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    # ========================================================================
    # PANEL C: Aleatoric slope ranking
    # ========================================================================
    ax_c = fig.add_subplot(gs[0, 2])
    
    # Average slopes by model
    model_slopes = decomp_response_df.groupby('model_name').agg({
        'aleatoric_slope': 'mean',
        'epistemic_slope': 'mean',
        'aleatoric_r2': 'mean',
    }).reset_index()
    
    model_slopes = model_slopes.sort_values('aleatoric_slope', ascending=True)
    
    y_pos = np.arange(len(model_slopes))
    colors = [MODEL_COLORS.get(m, '#999999') for m in model_slopes['model_name']]
    
    # Plot aleatoric slopes
    bars = ax_c.barh(y_pos, model_slopes['aleatoric_slope'],
                    color=colors, alpha=0.8, height=0.7, edgecolor='black', linewidth=0.5)
    
    # Ideal slope line
    ax_c.axvline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (slope=1)')
    
    ax_c.set_yticks(y_pos)
    ax_c.set_yticklabels([get_display_name(m) for m in model_slopes['model_name']], fontsize=8)
    ax_c.set_xlabel('Aleatoric Inflation Slope', fontsize=9)
    ax_c.set_title('C. Aleatoric Noise Tracking\n(1.0 = perfect tracking of injected σ)', 
                   fontsize=10, fontweight='bold', pad=10)
    ax_c.legend(fontsize=7, loc='lower right', framealpha=0.9)
    ax_c.spines['top'].set_visible(False)
    ax_c.spines['right'].set_visible(False)
    ax_c.grid(True, axis='x', alpha=0.3, linestyle=':', linewidth=0.5)
    
    output_path = Path(output_dir) / "figure8_decomposition_response.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_models_for_rep(metrics_df, rep, max_models=5):
    """Get available models for a representation"""
    rep_data = metrics_df[metrics_df['representation'] == rep]
    available = set(rep_data['model_name'].unique())
    
    good_models = []
    for model in available:
        model_data = rep_data[rep_data['model_name'] == model]
        if len(model_data['sigma'].unique()) >= 2:
            good_models.append(model)
    
    available = set(good_models)
    result = [m for m in EXTENDED_MODELS if m in available]
    
    for m in sorted(available):
        if m not in result:
            result.append(m)
    
    return result[:max_models]


# ============================================================================
# TABLES
# ============================================================================

def create_tables(metrics_df, response_df, decomp_response_df, cross_df, output_dir):
    """Create summary tables"""
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    # Table 1: Overall summary by model
    table1 = metrics_df.groupby('model_name').agg({
        'correlation': ['mean', 'std'],
        'ece': ['mean', 'std'],
        'coverage_1std': ['mean', 'std'],
        'mean_absolute_error': ['mean', 'std'],
    }).round(4)
    table1.columns = ['_'.join(col).strip() for col in table1.columns.values]
    table1.to_csv(output_dir / "table1_model_summary.csv")
    print(f"✓ table1_model_summary.csv")
    
    # Table 2: Uncertainty-noise response
    if len(response_df) > 0:
        table2 = response_df.groupby('model_name').agg({
            'uncertainty_inflation_slope': ['mean', 'std'],
            'uncertainty_inflation_r2': 'mean',
            'relative_inflation': 'mean',
        }).round(4)
        table2.columns = ['_'.join(col).strip() for col in table2.columns.values]
        table2.to_csv(output_dir / "table2_uncertainty_noise_response.csv")
        print(f"✓ table2_uncertainty_noise_response.csv")
    
    # Table 3: Decomposition response
    if len(decomp_response_df) > 0:
        table3 = decomp_response_df.groupby('model_name').agg({
            'aleatoric_slope': ['mean', 'std'],
            'epistemic_slope': ['mean', 'std'],
            'aleatoric_tracks_noise': 'sum',
            'epistemic_stable': 'sum',
        }).round(4)
        table3.columns = ['_'.join(str(c) for c in col).strip() for col in table3.columns.values]
        table3.to_csv(output_dir / "table3_decomposition_response.csv")
        print(f"✓ table3_decomposition_response.csv")
    
    # Table 4: Cross-phase analysis
    if len(cross_df) > 0:
        table4 = cross_df[['model_name', 'representation', 'mean_uq_correlation', 
                          'mean_ece', 'mean_coverage', 'baseline_r2', 'nds']].copy()
        table4 = table4.sort_values('mean_uq_correlation', ascending=False)
        table4.to_csv(output_dir / "table4_cross_phase_analysis.csv", index=False, float_format='%.4f')
        print(f"✓ table4_cross_phase_analysis.csv")
        
        # LaTeX version
        with open(output_dir / "table4_cross_phase_analysis.tex", 'w') as f:
            f.write("% Cross-phase analysis: UQ quality vs noise robustness\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\caption{Cross-phase analysis comparing uncertainty quantification quality with noise robustness. Higher UQ-Error correlation indicates better uncertainty estimates; lower $|$NDS$|$ indicates better noise robustness.}\n")
            f.write("\\label{tab:cross_phase}\n")
            f.write("\\begin{tabular}{llccccc}\n")
            f.write("\\toprule\n")
            f.write("Model & Rep & UQ-Err Corr & ECE & Coverage & Baseline R$^2$ & NDS \\\\\n")
            f.write("\\midrule\n")
            for _, row in table4.head(15).iterrows():
                f.write(f"{get_display_name(row['model_name'])} & {get_rep_display(row['representation'])} & ")
                f.write(f"{row['mean_uq_correlation']:.3f} & {row['mean_ece']:.3f} & ")
                f.write(f"{row['mean_coverage']:.3f} & {row['baseline_r2']:.3f} & {row['nds']:.4f} \\\\\n")
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        print(f"✓ table4_cross_phase_analysis.tex")
    
    # Table 5: Best configurations for uncertainty-aware noise robustness
    if len(cross_df) > 0:
        valid = cross_df.dropna(subset=['mean_uq_correlation', 'nds']).copy()
        if len(valid) > 0:
            # Combined score
            valid['uq_rank'] = valid['mean_uq_correlation'].rank(ascending=False)
            valid['nds_rank'] = valid['nds'].abs().rank(ascending=True)
            valid['combined_rank'] = (valid['uq_rank'] + valid['nds_rank']) / 2
            
            table5 = valid.nsmallest(20, 'combined_rank')[
                ['model_name', 'representation', 'mean_uq_correlation', 'nds', 
                 'mean_coverage', 'baseline_r2', 'combined_rank']
            ]
            table5.to_csv(output_dir / "table5_best_combined_configs.csv", index=False, float_format='%.4f')
            print(f"✓ table5_best_combined_configs.csv")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main analysis"""
    print("="*80)
    print("PHASE 2: UNCERTAINTY ANALYSIS (ENHANCED)")
    print("="*80)
    print("\nKey additions in this version:")
    print("  - Uncertainty response to noise (inflation slopes)")
    print("  - Decomposition analysis (aleatoric tracks noise, epistemic stable)")
    print("  - Cross-phase analysis (UQ quality vs Noise Degradation Slope)")
    print("="*80)
    
    # Load data
    uncertainty_df = load_uncertainty_data(results_dir)
    if len(uncertainty_df) == 0:
        raise ValueError("No data loaded")
    
    phase0_df = load_phase0_metrics(results_dir)
    
    # Calculate metrics
    metrics_df = calculate_metrics(uncertainty_df)
    if len(metrics_df) == 0:
        raise ValueError("No metrics calculated")
    
    response_df = calculate_uncertainty_noise_response(uncertainty_df)
    decomp_response_df = calculate_decomposition_response(uncertainty_df)
    cross_df = calculate_cross_phase_metrics(metrics_df, phase0_df)
    
    # Output directory
    output_dir = Path(results_dir) / "phase2_analysis_v3"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)
    print(f"\n✓ Saved metrics.csv")
    
    if len(response_df) > 0:
        response_df.to_csv(output_dir / "uncertainty_noise_response.csv", index=False)
        print(f"✓ Saved uncertainty_noise_response.csv")
    
    if len(decomp_response_df) > 0:
        decomp_response_df.to_csv(output_dir / "decomposition_response.csv", index=False)
        print(f"✓ Saved decomposition_response.csv")
    
    if len(cross_df) > 0:
        cross_df.to_csv(output_dir / "cross_phase_metrics.csv", index=False)
        print(f"✓ Saved cross_phase_metrics.csv")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure4(uncertainty_df, metrics_df, output_dir)
    create_figure5(uncertainty_df, metrics_df, output_dir)
    create_figure6_uncertainty_noise_response(uncertainty_df, metrics_df, response_df, output_dir)
    
    if len(cross_df) > 3:
        create_figure7_cross_phase(cross_df, output_dir)
    
    if len(decomp_response_df) > 0:
        create_figure8_decomposition_response(uncertainty_df, decomp_response_df, output_dir)
    
    # Generate tables
    create_tables(metrics_df, response_df, decomp_response_df, cross_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nData: {len(uncertainty_df):,} samples")
    print(f"Configurations: {len(metrics_df)}")
    
    print("\n" + "-"*40)
    print("KEY METRICS BY MODEL")
    print("-"*40)
    
    summary = metrics_df.groupby('model_name').agg({
        'correlation': 'mean',
        'coverage_1std': lambda x: x.mean() * 100,
        'ece': 'mean',
    }).round(3)
    summary.columns = ['UQ-Err Corr', 'Coverage%', 'ECE']
    print(summary.sort_values('UQ-Err Corr', ascending=False).to_string())
    
    if len(response_df) > 0:
        print("\n" + "-"*40)
        print("UNCERTAINTY NOISE TRACKING")
        print("-"*40)
        noise_summary = response_df.groupby('model_name')['uncertainty_inflation_slope'].mean().sort_values(ascending=False)
        print("Inflation slope (ideal=1.0):")
        for model, slope in noise_summary.items():
            quality = "✓ Good" if 0.7 < slope < 1.3 else "⚠️ Poor"
            print(f"  {get_display_name(model)}: {slope:.3f} {quality}")
    
    if len(cross_df) > 0:
        print("\n" + "-"*40)
        print("CROSS-PHASE: UQ vs NOISE ROBUSTNESS")
        print("-"*40)
        valid = cross_df.dropna(subset=['mean_uq_correlation', 'nds'])
        if len(valid) > 3:
            corr, pval = spearmanr(valid['mean_uq_correlation'], valid['nds'].abs())
            print(f"Correlation between UQ quality and |NDS|: ρ={corr:.3f}, p={pval:.4f}")
            if corr < 0:
                print("  → Better UQ is associated with BETTER noise robustness")
            else:
                print("  → No clear relationship between UQ and robustness")
    
    print("\n" + "="*80)
    print("✅ PHASE 2 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs: {output_dir}")
    print("\nGenerated files:")
    print("  - figure4_uncertainty_error.png")
    print("  - figure5_calibration.png")
    print("  - figure6_uncertainty_noise_response.png")
    if len(cross_df) > 3:
        print("  - figure7_uq_vs_robustness.png")
    if len(decomp_response_df) > 0:
        print("  - figure8_decomposition_response.png")
    print("  - table1-5 (csv/tex)")


if __name__ == "__main__":
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)