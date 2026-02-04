"""
Phase 3 Analysis - REVISED COMPREHENSIVE VERSION
Generates Figure 7 and Supplementary S7

FIXES:
1. Proper handling of both summary files AND raw interval files
2. Better filename parsing for model/representation extraction
3. Comprehensive data validation and reporting
4. Better handling of missing columns and calculation of derived metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

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

# Color palette
MODEL_COLORS = {
    'rf': '#3498db',
    'qrf': '#16a085',
    'xgboost': '#e74c3c',
    'ngboost': '#f39c12',
    'dnn': '#34495e',
    'gauche': '#9b59b6',
}

# ============================================================================
# DATA LOADING - COMPREHENSIVE VERSION
# ============================================================================

def load_conformal_intervals(results_dir="../results"):
    """
    Load Phase 3 conformal prediction results - COMPREHENSIVE VERSION
    
    Handles TWO types of data structures:
    1. SUMMARY FILES: phase3a/b/c_*.csv with per-row predictions and metadata
    2. RAW INTERVAL FILES: conformal_intervals/*.csv with granular predictions
    
    Strategy:
    - Try summary files first (most common, easier to work with)
    - Fall back to raw interval files if no summaries found
    - Aggregate raw files by configuration if needed
    """
    print("\n" + "="*80)
    print("LOADING PHASE 3 CONFORMAL PREDICTION DATA - COMPREHENSIVE VERSION")
    print("="*80)
    
    results_dir = Path(results_dir)
    print(f"Searching in: {results_dir.absolute()}\n")
    
    all_data = []
    
    # ========================================================================
    # STEP 1: Try to load SUMMARY files (phase3a/b/c_*.csv)
    # ========================================================================
    print("-"*80)
    print("STEP 1: Looking for summary files (phase3a/b/c/d/e/f_*.csv)")
    print("-"*80)
    
    summary_patterns = [
        "phase3a_*_conformal_*.csv",
        "phase3b_*_conformal_*.csv",
        "phase3c_*_conformal_*.csv",
        "phase3d_*_conformal_*.csv",
        "phase3e_*_conformal_*.csv",
        "phase3f_*_conformal_*.csv",
        "phase3_*_conformal_*.csv",
    ]
    
    summary_files = []
    for pattern in summary_patterns:
        found = list(results_dir.glob(pattern))
        summary_files.extend(found)
    
    # Deduplicate and filter
    summary_files = list(set(summary_files))
    summary_files = [f for f in summary_files if 'uncertainty_values' not in f.name]
    summary_files = sorted(summary_files)
    
    print(f"Found {len(summary_files)} summary files\n")
    
    if summary_files:
        print("Loading summary files...")
        for filepath in summary_files:
            loaded_df = load_summary_file(filepath)
            if loaded_df is not None and len(loaded_df) > 0:
                all_data.append(loaded_df)
    
    # ========================================================================
    # STEP 2: If no summary files, try raw interval files
    # ========================================================================
    if not all_data:
        print("\n" + "-"*80)
        print("STEP 2: No summary files found, looking for raw interval files")
        print("-"*80)
        
        conformal_dirs = list(results_dir.glob("**/conformal_intervals"))
        
        if not conformal_dirs:
            print("❌ No conformal_intervals directory found either!")
            return pd.DataFrame()
        
        print(f"Found {len(conformal_dirs)} conformal_intervals directories\n")
        
        for conf_dir in conformal_dirs:
            raw_data = load_raw_interval_directory(conf_dir)
            if raw_data:
                all_data.extend(raw_data)
    
    # ========================================================================
    # STEP 3: Combine and validate
    # ========================================================================
    if not all_data:
        print("\n❌ ERROR: No conformal data could be loaded!")
        print("\nSearched for:")
        print("  1. Summary files: phase3a/b/c_*_conformal_*.csv")
        print("  2. Raw interval files: conformal_intervals/conformal_intervals_*.csv")
        return pd.DataFrame()
    
    intervals_df = pd.concat(all_data, ignore_index=True)
    
    # Final data cleaning and standardization
    intervals_df = standardize_columns(intervals_df)
    
    # Print comprehensive summary
    print_data_summary(intervals_df)
    
    return intervals_df


def load_summary_file(filepath):
    """Load a single summary file with comprehensive error handling"""
    try:
        df = pd.read_csv(filepath)
        filename = filepath.stem
        
        # Parse filename: phase3a_ecfp4_conformal_qrf_calib10
        # Pattern: phase3X_REP_conformal_MODEL[_VARIANT]
        parts = filename.split('_')
        
        if 'conformal' not in parts:
            print(f"  ⚠️  {filepath.name}: no 'conformal' in filename")
            return None
        
        conf_idx = parts.index('conformal')
        
        # Extract representation (before 'conformal', after phase3X)
        rep = parts[conf_idx - 1] if conf_idx > 1 else None
        
        # Extract model (everything after 'conformal')
        model_parts = parts[conf_idx + 1:]
        model = '_'.join(model_parts) if model_parts else None
        
        if not rep or not model:
            print(f"  ⚠️  {filepath.name}: couldn't parse rep/model")
            return None
        
        # Add metadata if not present
        if 'model' not in df.columns:
            df['model'] = model
        if 'model_name' not in df.columns:
            df['model_name'] = f'conformal_{model}'
        if 'representation' not in df.columns:
            df['representation'] = rep
        if 'rep' not in df.columns:
            df['rep'] = rep
        
        # Ensure sigma column
        if 'sigma' not in df.columns and 'sigma_noise' not in df.columns:
            # Try to parse from filename
            import re
            sigma_match = re.search(r'sigma([0-9.]+)', filename)
            if sigma_match:
                df['sigma'] = float(sigma_match.group(1))
            else:
                df['sigma'] = 0.0
        
        # Standardize sigma naming
        if 'sigma_noise' in df.columns and 'sigma' not in df.columns:
            df['sigma'] = df['sigma_noise']
        elif 'sigma' in df.columns and 'sigma_noise' not in df.columns:
            df['sigma_noise'] = df['sigma']
        
        # Check for essential columns
        if 'alpha' not in df.columns:
            print(f"  ⚠️  {filepath.name}: missing 'alpha' column")
            return None
        
        # Ensure y_true
        if 'y_true' not in df.columns:
            if 'y_true_noisy' in df.columns:
                df['y_true'] = df['y_true_noisy']
            elif 'y_true_original' in df.columns:
                df['y_true'] = df['y_true_original']
            else:
                print(f"  ⚠️  {filepath.name}: no y_true column")
                return None
        
        # Calculate coverage if missing
        if 'coverage' not in df.columns:
            if all(col in df.columns for col in ['y_true', 'lower_bound', 'upper_bound']):
                df['coverage'] = ((df['y_true'] >= df['lower_bound']) & 
                                 (df['y_true'] <= df['upper_bound'])).astype(int)
            elif all(col in df.columns for col in ['y_true', 'lower', 'upper']):
                df['coverage'] = ((df['y_true'] >= df['lower']) & 
                                 (df['y_true'] <= df['upper'])).astype(int)
            else:
                print(f"  ⚠️  {filepath.name}: cannot calculate coverage (missing bounds)")
                return None
        
        # Calculate interval_width if missing
        if 'interval_width' not in df.columns:
            if all(col in df.columns for col in ['lower_bound', 'upper_bound']):
                df['interval_width'] = df['upper_bound'] - df['lower_bound']
            elif all(col in df.columns for col in ['lower', 'upper']):
                df['interval_width'] = df['upper'] - df['lower']
            else:
                print(f"  ⚠️  {filepath.name}: cannot calculate interval_width")
                return None
        
        # Ensure y_pred
        if 'y_pred' not in df.columns:
            if 'y_pred_mean' in df.columns:
                df['y_pred'] = df['y_pred_mean']
            elif 'prediction' in df.columns:
                df['y_pred'] = df['prediction']
        
        df['source_file'] = filepath.name
        df['source_type'] = 'summary'
        
        # Report successful load
        n_alphas = len(df['alpha'].unique())
        n_sigmas = len(df['sigma'].unique())
        print(f"  ✓ {filepath.name}: {len(df):,} rows")
        print(f"      model={model}, rep={rep}")
        print(f"      {n_alphas} alphas × {n_sigmas} sigmas")
        
        return df
        
    except Exception as e:
        print(f"  ❌ {filepath.name}: {e}")
        return None


def load_raw_interval_directory(conf_dir):
    """Load and aggregate raw interval files from a directory"""
    interval_files = list(conf_dir.glob("conformal_intervals_*.csv"))
    print(f"\nProcessing {conf_dir.name}: {len(interval_files)} files")
    
    if not interval_files:
        return []
    
    # Group files by configuration
    config_groups = {}
    
    for filepath in interval_files:
        config = parse_interval_filename(filepath.name)
        if config:
            key = (config['model'], config['rep'], config['sigma'], config.get('iter', 0))
            if key not in config_groups:
                config_groups[key] = []
            config_groups[key].append(filepath)
    
    print(f"Found {len(config_groups)} unique configurations")
    
    # Load each configuration
    all_configs = []
    
    for (model, rep, sigma, iter_num), file_list in config_groups.items():
        try:
            # Load all files for this config
            config_dfs = []
            for filepath in file_list:
                df_chunk = pd.read_csv(filepath)
                config_dfs.append(df_chunk)
            
            if not config_dfs:
                continue
            
            # Combine
            df = pd.concat(config_dfs, ignore_index=True)
            
            # Add metadata
            df['model'] = model
            df['model_name'] = f'conformal_{model}'
            df['representation'] = rep
            df['rep'] = rep
            df['sigma'] = sigma
            df['sigma_noise'] = sigma
            
            # Calculate coverage and width
            if 'coverage' not in df.columns:
                if all(col in df.columns for col in ['y_true', 'lower_bound', 'upper_bound']):
                    df['coverage'] = ((df['y_true'] >= df['lower_bound']) & 
                                     (df['y_true'] <= df['upper_bound'])).astype(int)
                elif all(col in df.columns for col in ['y_true', 'lower', 'upper']):
                    df['coverage'] = ((df['y_true'] >= df['lower']) & 
                                     (df['y_true'] <= df['upper'])).astype(int)
            
            if 'interval_width' not in df.columns:
                if all(col in df.columns for col in ['lower_bound', 'upper_bound']):
                    df['interval_width'] = df['upper_bound'] - df['lower_bound']
                elif all(col in df.columns for col in ['lower', 'upper']):
                    df['interval_width'] = df['upper'] - df['lower']
            
            # Check we have what we need
            if 'alpha' in df.columns and 'coverage' in df.columns:
                df['source_type'] = 'raw_intervals'
                all_configs.append(df)
                
                n_alphas = len(df['alpha'].unique())
                print(f"  ✓ {model}/{rep}/σ={sigma}/iter{iter_num}: {len(df):,} rows, {n_alphas} alphas")
        
        except Exception as e:
            print(f"  ❌ {model}/{rep}/σ={sigma}: {e}")
    
    return all_configs


def parse_interval_filename(filename):
    """Parse raw interval filename to extract configuration"""
    # Pattern: conformal_intervals_conformal_dnn_split_ecfp4_sigma0.0_iter0_fileXXX
    parts = filename.split('_')
    
    config = {}
    
    try:
        # Find model (usually after first 'conformal')
        conf_indices = [i for i, p in enumerate(parts) if p == 'conformal']
        if len(conf_indices) >= 2:
            model_idx = conf_indices[1] + 1
            if model_idx < len(parts):
                config['model'] = parts[model_idx]
        
        # Find representation
        for r in ['ecfp4', 'pdv', 'sns', 'smiles', 'graph']:
            if r in parts:
                config['rep'] = r
                break
        
        # Find sigma
        for p in parts:
            if p.startswith('sigma'):
                config['sigma'] = float(p.replace('sigma', ''))
                break
        
        # Find iter
        for p in parts:
            if p.startswith('iter'):
                config['iter'] = int(p.replace('iter', ''))
                break
        
        # Only return if we have minimum info
        if 'model' in config and 'rep' in config and 'sigma' in config:
            return config
    
    except Exception:
        pass
    
    return None


def standardize_columns(df):
    """Standardize column names across different data sources"""
    # Model names
    if 'model_name' in df.columns:
        df['model_name'] = df['model_name'].str.replace('_split', '', regex=False)
        df['base_model'] = df['model_name'].str.replace('conformal_', '', regex=False)
    elif 'model' in df.columns:
        df['model_name'] = 'conformal_' + df['model'].astype(str)
        df['base_model'] = df['model']
    
    # Representation
    if 'rep' not in df.columns and 'representation' in df.columns:
        df['rep'] = df['representation']
    
    # Sigma
    if 'sigma_noise' not in df.columns and 'sigma' in df.columns:
        df['sigma_noise'] = df['sigma']
    
    return df


def print_data_summary(df):
    """Print comprehensive data summary"""
    print(f"\n{'='*80}")
    print("✅ SUCCESSFULLY LOADED CONFORMAL DATA")
    print(f"{'='*80}")
    print(f"\nTotal predictions: {len(df):,}")
    
    if 'source_type' in df.columns:
        print(f"Source types: {df['source_type'].value_counts().to_dict()}")
    
    print(f"\n📊 Data Summary:")
    print(f"  Models ({len(df['base_model'].unique())}): {sorted(df['base_model'].unique())}")
    print(f"  Representations ({len(df['rep'].unique())}): {sorted(df['rep'].unique())}")
    print(f"  Alpha values ({len(df['alpha'].unique())}): {sorted(df['alpha'].unique())}")
    print(f"  Sigma values ({len(df['sigma_noise'].unique())}): {sorted(df['sigma_noise'].unique())}")
    
    print(f"\n📈 Data Completeness by Model/Representation:")
    print(f"  {'Model':<15s} {'Rep':<8s} {'Rows':>10s} {'Alphas':>8s} {'Sigmas':>8s} {'Configs':>8s}")
    print(f"  {'-'*70}")
    
    for model in sorted(df['base_model'].unique()):
        for rep in sorted(df['rep'].unique()):
            subset = df[(df['base_model'] == model) & (df['rep'] == rep)]
            if len(subset) > 0:
                n_alphas = len(subset['alpha'].unique())
                n_sigmas = len(subset['sigma_noise'].unique())
                n_configs = n_alphas * n_sigmas
                print(f"  {model:<15s} {rep:<8s} {len(subset):>10,} {n_alphas:>8} {n_sigmas:>8} {n_configs:>8}")


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_conformal_metrics(intervals_df):
    """
    Calculate conformal prediction metrics
    
    Metrics:
    - Expected coverage: 1 - alpha
    - Empirical coverage: actual fraction within intervals
    - Coverage deviation: empirical - expected
    - Mean/median interval width
    - Efficiency: width / RMSE ratio
    - Calibration status
    """
    print("\n" + "="*80)
    print("CALCULATING CONFORMAL METRICS")
    print("="*80)
    
    metrics = []
    
    groupby_cols = ['model_name', 'rep', 'sigma_noise', 'alpha']
    
    # Check which columns exist
    existing_cols = [col for col in groupby_cols if col in intervals_df.columns]
    
    if len(existing_cols) < 4:
        print(f"⚠️  Warning: Missing groupby columns. Have: {existing_cols}")
        # Try alternate column names
        if 'base_model' in intervals_df.columns and 'model_name' not in intervals_df.columns:
            groupby_cols[0] = 'base_model'
        if 'representation' in intervals_df.columns and 'rep' not in intervals_df.columns:
            groupby_cols[1] = 'representation'
        if 'sigma' in intervals_df.columns and 'sigma_noise' not in intervals_df.columns:
            groupby_cols[2] = 'sigma'
    
    for group_keys, group in intervals_df.groupby(groupby_cols):
        if len(groupby_cols) == 4:
            model, rep, sigma, alpha = group_keys
        else:
            continue  # Skip if grouping failed
        
        # Expected coverage
        expected_coverage = 1 - alpha
        
        # Empirical coverage
        empirical_coverage = group['coverage'].mean()
        
        # Coverage deviation
        coverage_deviation = empirical_coverage - expected_coverage
        
        # Interval width stats
        mean_width = group['interval_width'].mean()
        median_width = group['interval_width'].median()
        std_width = group['interval_width'].std()
        
        # Prediction error
        if 'y_pred' in group.columns and 'y_true' in group.columns:
            rmse = np.sqrt(((group['y_true'] - group['y_pred'])**2).mean())
            mae = np.abs(group['y_true'] - group['y_pred']).mean()
        else:
            rmse = np.nan
            mae = np.nan
        
        # Efficiency: narrower intervals relative to error are better
        efficiency = mean_width / rmse if rmse > 0 and not np.isnan(rmse) else np.inf
        
        # Calibration check (within 5%)
        is_calibrated = abs(coverage_deviation) < 0.05
        
        metrics.append({
            'model_name': model,
            'base_model': model.replace('conformal_', ''),
            'representation': rep,
            'sigma': sigma,
            'alpha': alpha,
            'expected_coverage': expected_coverage,
            'empirical_coverage': empirical_coverage,
            'coverage_deviation': coverage_deviation,
            'mean_width': mean_width,
            'median_width': median_width,
            'std_width': std_width,
            'rmse': rmse,
            'mae': mae,
            'efficiency': efficiency,
            'is_calibrated': is_calibrated,
            'n_samples': len(group)
        })
    
    metrics_df = pd.DataFrame(metrics)
    
    print(f"\n✓ Calculated metrics for {len(metrics_df)} configurations")
    
    # Summary stats
    if len(metrics_df) > 0:
        print(f"\nCalibration summary:")
        calibrated_count = metrics_df['is_calibrated'].sum()
        total_count = len(metrics_df)
        print(f"  Well-calibrated (|deviation| < 0.05): {calibrated_count}/{total_count} ({calibrated_count/total_count*100:.1f}%)")
        
        print(f"\nCoverage statistics:")
        print(f"  Mean coverage deviation: {metrics_df['coverage_deviation'].abs().mean():.4f}")
        print(f"  Median coverage deviation: {metrics_df['coverage_deviation'].abs().median():.4f}")
        
        print(f"\nEfficiency statistics:")
        finite_eff = metrics_df[np.isfinite(metrics_df['efficiency'])]['efficiency']
        if len(finite_eff) > 0:
            print(f"  Mean efficiency: {finite_eff.mean():.2f}")
            print(f"  Median efficiency: {finite_eff.median():.2f}")
    
    return metrics_df


# ============================================================================
# FIGURE 7: CONFORMAL PREDICTION VALIDITY AND EFFICIENCY
# ============================================================================

def create_figure7_conformal_validity_efficiency(intervals_df, metrics_df, output_dir):
    """
    Figure 7: Conformal prediction validity (SINGLE PANEL)
    
    Panel A: Coverage calibration vs target confidence and noise
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 7: CONFORMAL VALIDITY")
    print("="*80)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Select key models and sigma levels
    sigma_levels = sorted(metrics_df['sigma'].unique())[:3]
    
    available_models = metrics_df['base_model'].unique()
    key_models = [m for m in ['rf', 'qrf', 'dnn', 'gauche'] if m in available_models][:3]
    
    if len(key_models) == 0 or len(sigma_levels) == 0:
        ax.text(0.5, 0.5, 'Insufficient conformal prediction data',
               ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.axis('off')
    else:
        plotted_something = False
        
        for model in key_models:
            for sigma_idx, sigma in enumerate(sigma_levels):
                model_sigma = metrics_df[
                    (metrics_df['base_model'] == model) &
                    (metrics_df['sigma'] == sigma)
                ].sort_values('alpha', ascending=False)
                
                if len(model_sigma) == 0:
                    continue
                
                color = MODEL_COLORS.get(model, '#999999')
                linestyle = ['-', '--', ':'][sigma_idx % 3]
                
                ax.plot(model_sigma['expected_coverage'], 
                       model_sigma['empirical_coverage'],
                       marker='o', linewidth=2, markersize=6, alpha=0.9,
                       color=color, linestyle=linestyle,
                       label=f'{model}, σ={sigma:.1f}')
                
                plotted_something = True
        
        if plotted_something:
            # Perfect calibration line
            ax.plot([0.8, 1.0], [0.8, 1.0], 'k--', linewidth=2, alpha=0.7,
                   label='Perfect calibration')
            
            # Acceptable range (±5%)
            ax.fill_between([0.8, 1.0], [0.75, 0.95], [0.85, 1.05],
                          color='green', alpha=0.15, label='±5% range')
            
            ax.set_xlabel('Target Coverage (1-α)', fontsize=9)
            ax.set_ylabel('Empirical Coverage', fontsize=9)
            ax.set_title('Conformal Prediction: Coverage Calibration', 
                       fontsize=10, fontweight='bold', pad=10)
            ax.legend(fontsize=7, loc='lower right', ncol=1, frameon=True, framealpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            ax.set_xlim(0.78, 1.02)
            ax.set_ylim(0.78, 1.02)
        else:
            ax.text(0.5, 0.5, 'No plottable data found',
                   ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure7_conformal_validity.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 7 to {output_path}")
    plt.close()


# ============================================================================
# SUPPLEMENTARY S7
# ============================================================================

def create_supplementary_s7(intervals_df, metrics_df, output_dir):
    """
    Supplementary S7: Per-target conformal performance
    (Only if multiple targets available)
    """
    print("\n" + "="*80)
    print("GENERATING SUPPLEMENTARY S7: PER-TARGET CONFORMAL")
    print("="*80)
    
    if 'target' not in intervals_df.columns:
        print("⚠️  No 'target' column found. Skipping S7.")
        return
    
    targets = intervals_df['target'].unique()
    
    if len(targets) <= 1:
        print(f"⚠️  Only one target found: {targets}. Skipping S7.")
        return
    
    print(f"Found {len(targets)} targets: {targets}")
    
    # Create grid
    n_targets = len(targets)
    ncols = min(3, n_targets)
    nrows = (n_targets + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).flatten()
    
    alpha_target = 0.05
    
    for idx, target in enumerate(targets):
        ax = axes[idx]
        
        if 'target' in metrics_df.columns:
            target_metrics = metrics_df[
                (metrics_df['target'] == target) &
                (metrics_df['alpha'] == alpha_target)
            ]
        else:
            target_metrics = pd.DataFrame()
        
        if len(target_metrics) == 0:
            ax.text(0.5, 0.5, f'No data for {target}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{target}', fontsize=9, fontweight='bold')
            continue
        
        for model in target_metrics['base_model'].unique():
            model_data = target_metrics[
                target_metrics['base_model'] == model
            ].sort_values('sigma')
            
            if len(model_data) > 0:
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], model_data['median_width'],
                       marker='o', linewidth=2, label=model, color=color, alpha=0.8)
        
        ax.set_xlabel('σ', fontsize=8)
        ax.set_ylabel('Median Width', fontsize=8)
        ax.set_title(f'{target}', fontsize=9, fontweight='bold')
        ax.legend(fontsize=6, loc='best')
        ax.grid(alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused
    for idx in range(len(targets), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path(output_dir) / "supplementary_s7_per_target_conformal.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Supplementary S7 to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY TABLES
# ============================================================================

def create_summary_tables(metrics_df, output_dir):
    """Create summary tables for Phase 3"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY TABLES")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    if len(metrics_df) == 0:
        print("⚠️  No metrics to create tables from")
        return
    
    # Table 4: Simplified summary
    print("\nCreating Table 4: Simplified Conformal Summary")
    
    alpha_target = 0.05
    table4_data = []
    
    for (model, rep), group in metrics_df[metrics_df['alpha'] == alpha_target].groupby(
        ['base_model', 'representation']
    ):
        sigma_0 = group[group['sigma'] == 0.0]
        sigma_mid = group[group['sigma'] > 0.2]
        
        if len(sigma_0) > 0 and len(sigma_mid) > 0:
            sigma_mid_val = sigma_mid.iloc[0]['sigma']
            
            row = {
                'Model': model,
                'Representation': rep,
                'Target Coverage': '95%',
                'Coverage @ σ=0': f"{sigma_0.iloc[0]['empirical_coverage']:.3f}",
                f'Coverage @ σ={sigma_mid_val:.1f}': f"{sigma_mid.iloc[0]['empirical_coverage']:.3f}",
                'Width @ σ=0': f"{sigma_0.iloc[0]['median_width']:.3f}",
                f'Width @ σ={sigma_mid_val:.1f}': f"{sigma_mid.iloc[0]['median_width']:.3f}",
                'Well-Calibrated': '✓' if sigma_mid.iloc[0]['is_calibrated'] else '✗'
            }
            table4_data.append(row)
    
    if table4_data:
        table4 = pd.DataFrame(table4_data)
        table4.to_csv(output_dir / "table4_conformal_summary_simplified.csv", index=False)
        print(f"✓ Saved Table 4")
    
    # Detailed tables
    alpha_key = [0.01, 0.05, 0.1, 0.2]
    
    table1 = metrics_df[metrics_df['alpha'].isin(alpha_key)].groupby(
        ['base_model', 'representation', 'alpha']
    ).agg({
        'expected_coverage': 'mean',
        'empirical_coverage': 'mean',
        'coverage_deviation': 'mean',
        'median_width': 'mean',
        'efficiency': 'mean',
        'is_calibrated': 'mean'
    }).reset_index()
    
    table1 = table1.round(4)
    table1.to_csv(output_dir / "table_phase3_conformal_summary.csv", index=False)
    print(f"✓ Saved detailed conformal summary")


# ============================================================================
# FIGURE 8: NOISE ROBUSTNESS ANALYSIS 
# ============================================================================

def create_figure8_noise_robustness(metrics_df, output_dir):
    """
    Figure 8: How CP quality degrades with noise
    2x2: Width inflation (ECFP4/PDV), Coverage stability heatmap, Efficiency
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 8: NOISE ROBUSTNESS")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    alpha_focus = 0.1
    
    # Panel A: Width inflation - ECFP4
    ax = axes[0, 0]
    rep = 'ecfp4'
    rep_models = sorted(metrics_df[metrics_df['representation'] == rep]['base_model'].unique())
    
    for model in rep_models:
        model_data = metrics_df[
            (metrics_df['base_model'] == model) &
            (metrics_df['representation'] == rep) &
            (metrics_df['alpha'] == alpha_focus)
        ].sort_values('sigma')
        
        if len(model_data) > 2:
            baseline = model_data[model_data['sigma'] == 0.0]['median_width']
            if len(baseline) > 0:
                relative_width = (model_data['median_width'] / baseline.iloc[0] - 1) * 100
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], relative_width,
                       marker='o', linewidth=2.5, markersize=6,
                       color=color, label=model, alpha=0.85)
    
    sigma_range = np.linspace(0, metrics_df['sigma'].max(), 50)
    ax.plot(sigma_range, sigma_range * 50, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax.set_xlabel('Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Width Increase (%)', fontsize=10)
    ax.set_title('A. Interval Width Inflation (ECFP4)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel B: Width inflation - PDV
    ax = axes[0, 1]
    rep = 'pdv'
    rep_models = sorted(metrics_df[metrics_df['representation'] == rep]['base_model'].unique())
    
    for model in rep_models:
        model_data = metrics_df[
            (metrics_df['base_model'] == model) &
            (metrics_df['representation'] == rep) &
            (metrics_df['alpha'] == alpha_focus)
        ].sort_values('sigma')
        
        if len(model_data) > 2:
            baseline = model_data[model_data['sigma'] == 0.0]['median_width']
            if len(baseline) > 0:
                relative_width = (model_data['median_width'] / baseline.iloc[0] - 1) * 100
                color = MODEL_COLORS.get(model, '#999999')
                ax.plot(model_data['sigma'], relative_width,
                       marker='s', linewidth=2.5, markersize=6,
                       color=color, label=model, alpha=0.85)
    
    ax.plot(sigma_range, sigma_range * 50, 'k--', linewidth=1.5, alpha=0.5, label='Ideal')
    ax.set_xlabel('Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Width Increase (%)', fontsize=10)
    ax.set_title('B. Interval Width Inflation (PDV)', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Panel C: Coverage stability heatmap
    ax = axes[1, 0]
    plot_data = metrics_df[metrics_df['alpha'] == alpha_focus].copy()
    plot_data['abs_deviation'] = plot_data['coverage_deviation'].abs()
    
    pivot = plot_data.pivot_table(
        values='abs_deviation',
        index=['base_model', 'representation'],
        columns='sigma',
        aggfunc='mean'
    )
    
    if len(pivot) > 0:
        im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto',
                      vmin=0, vmax=0.10, interpolation='nearest')
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_xticklabels([f'{s:.1f}' for s in pivot.columns], fontsize=8)
        ax.set_yticklabels([f'{idx[0]}\n{idx[1]}' for idx in pivot.index], fontsize=7)
        
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color='white' if val > 0.05 else 'black', fontsize=6, fontweight='bold')
        
        ax.set_xlabel('Noise Level (σ)', fontsize=10)
        ax.set_ylabel('Model / Representation', fontsize=10)
        ax.set_title('C. Coverage Calibration Error', fontsize=11, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('|Coverage Deviation|', fontsize=9)
    
    # Panel D: Efficiency vs noise
    ax = axes[1, 1]
    for rep in ['ecfp4', 'pdv']:
        rep_models = sorted(metrics_df[metrics_df['representation'] == rep]['base_model'].unique())
        for model in rep_models:
            model_data = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['alpha'] == alpha_focus)
            ].sort_values('sigma')
            model_data = model_data[np.isfinite(model_data['efficiency'])]
            
            if len(model_data) > 2:
                color = MODEL_COLORS.get(model, '#999999')
                marker = 'o' if rep == 'ecfp4' else 's'
                linestyle = '-' if rep == 'ecfp4' else '--'
                ax.plot(model_data['sigma'], model_data['efficiency'],
                       marker=marker, linewidth=2, markersize=5,
                       color=color, linestyle=linestyle, alpha=0.75,
                       label=f'{model}/{rep}')
    
    ax.axhline(3.0, color='red', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.set_xlabel('Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Efficiency (Width/RMSE)', fontsize=10)
    ax.set_title('D. Efficiency Across Noise', fontsize=11, fontweight='bold')
    ax.legend(fontsize=6, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 8)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure8_noise_robustness.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 8 to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 9: DETAILED CALIBRATION CURVES
# ============================================================================

def create_figure9_detailed_calibration(metrics_df, output_dir):
    """Figure 9: Per-model calibration curves at multiple noise levels"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 9: DETAILED CALIBRATION CURVES")
    print("="*80)
    
    # First, identify which model/rep combinations have data
    available_combos = []
    models = sorted(metrics_df['base_model'].unique())
    reps = sorted(metrics_df['representation'].unique())[:2]  # ecfp4, pdv
    
    for model in models:
        for rep in reps:
            # Check if this combo has data at multiple sigmas
            combo_data = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep)
            ]
            if len(combo_data) > 3:  # Need at least a few points
                available_combos.append((model, rep))
    
    if len(available_combos) == 0:
        print("⚠️  No data for Figure 9 (need model/rep with multiple noise levels)")
        return
    
    print(f"Found data for {len(available_combos)} model/rep combinations")
    
    # Create grid based on available combos
    ncols = 2  # One column per representation
    nrows = len([m for m, r in available_combos if r == reps[0]])  # Count unique models
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 5*nrows))
    
    if nrows == 1:
        axes = axes.reshape(1, -1)
    
    sigma_levels = [0.0, 0.3, 0.6, 0.9]
    colors_sigma = ['#2c3e50', '#3498db', '#e74c3c', '#9b59b6']
    
    # Track which row we're on
    row_idx = 0
    
    # Group available combos by model
    models_with_data = sorted(list(set([m for m, r in available_combos])))
    
    for model in models_with_data:
        for col_idx, rep in enumerate(reps):
            if (model, rep) in available_combos:
                ax = axes[row_idx, col_idx]
                plotted = False
                
                for sigma_idx, sigma in enumerate(sigma_levels):
                    data = metrics_df[
                        (metrics_df['base_model'] == model) &
                        (metrics_df['representation'] == rep) &
                        (np.abs(metrics_df['sigma'] - sigma) < 0.01)
                    ].sort_values('alpha', ascending=False)
                    
                    if len(data) > 1:
                        ax.plot(data['expected_coverage'], data['empirical_coverage'],
                               marker='o', linewidth=2.5, markersize=7,
                               color=colors_sigma[sigma_idx], alpha=0.8,
                               label=f'σ={sigma:.1f}')
                        plotted = True
                
                if plotted:
                    ax.plot([0.7, 1.0], [0.7, 1.0], 'k--', linewidth=2, alpha=0.6, label='Perfect')
                    ax.fill_between([0.7, 1.0], [0.65, 0.95], [0.75, 1.05],
                                   color='green', alpha=0.12, label='±5%')
                    ax.set_xlabel('Target Coverage (1-α)', fontsize=10)
                    ax.set_ylabel('Empirical Coverage', fontsize=10)
                    ax.set_title(f'{model.upper()} / {rep.upper()}', fontsize=11, fontweight='bold')
                    ax.legend(fontsize=8, loc='lower right', ncol=2)
                    ax.grid(alpha=0.25, linestyle=':')
                    ax.set_xlim(0.69, 1.01)
                    ax.set_ylim(0.69, 1.01)
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
            else:
                # This rep doesn't have data for this model - hide the axis
                ax = axes[row_idx, col_idx]
                ax.axis('off')
        
        row_idx += 1
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure9_detailed_calibration.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 9 to {output_path}")
    plt.close()


# ============================================================================
# FIGURE 10: CP FOR NOISE DETECTION
# ============================================================================

def create_figure10_noise_detection(metrics_df, output_dir):
    """Figure 10: Can CP intervals detect/quantify noise? Width-noise correlation"""
    print("\n" + "="*80)
    print("GENERATING FIGURE 10: NOISE DETECTION POTENTIAL")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    alpha_focus = 0.1
    
    # Panels A & B: Width vs RMSE colored by noise
    for rep_idx, rep in enumerate(['ecfp4', 'pdv']):
        ax = axes[0, rep_idx]
        rep_data = metrics_df[
            (metrics_df['representation'] == rep) &
            (metrics_df['alpha'] == alpha_focus)
        ]
        
        scatter = ax.scatter(rep_data['rmse'], rep_data['median_width'],
                           c=rep_data['sigma'], cmap='viridis',
                           s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        rmse_range = np.linspace(rep_data['rmse'].min(), rep_data['rmse'].max(), 100)
        for eff in [2, 3, 4]:
            ax.plot(rmse_range, rmse_range * eff, '--', alpha=0.25, linewidth=1, color='gray')
        
        ax.set_xlabel('RMSE', fontsize=10)
        ax.set_ylabel('Median Interval Width', fontsize=10)
        ax.set_title(f'{"A" if rep_idx == 0 else "B"}. Width-Error ({rep.upper()})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Noise (σ)', fontsize=9)
    
    # Panel C: Width-noise correlation
    ax = axes[1, 0]
    correlations = []
    for model in metrics_df['base_model'].unique():
        for rep in metrics_df['representation'].unique():
            model_rep = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['alpha'] == alpha_focus)
            ]
            if len(model_rep) > 3:
                corr = model_rep[['sigma', 'median_width']].corr().iloc[0, 1]
                correlations.append({'model': model, 'rep': rep, 'correlation': corr, 'label': f'{model}/{rep}'})
    
    if correlations:
        corr_df = pd.DataFrame(correlations).sort_values('correlation')
        colors_list = [MODEL_COLORS.get(row['model'], '#999999') for _, row in corr_df.iterrows()]
        y_pos = np.arange(len(corr_df))
        ax.barh(y_pos, corr_df['correlation'], color=colors_list, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(corr_df['label'], fontsize=8)
        ax.set_xlabel('Correlation (σ vs Width)', fontsize=10)
        ax.set_title('C. CP Width Tracks Noise', fontsize=11, fontweight='bold')
        ax.axvline(0.9, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label='Strong')
        ax.grid(alpha=0.3, axis='x')
        ax.legend(fontsize=8)
    
    # Panel D: Width ratio vs true noise
    ax = axes[1, 1]
    for rep in ['ecfp4', 'pdv']:
        for model in metrics_df[metrics_df['representation'] == rep]['base_model'].unique()[:3]:
            model_data = metrics_df[
                (metrics_df['base_model'] == model) &
                (metrics_df['representation'] == rep) &
                (metrics_df['alpha'] == alpha_focus)
            ].sort_values('sigma')
            
            if len(model_data) > 2:
                baseline = model_data[model_data['sigma'] == 0.0]['median_width']
                if len(baseline) > 0:
                    width_ratio = model_data['median_width'] / baseline.iloc[0]
                    color = MODEL_COLORS.get(model, '#999999')
                    marker = 'o' if rep == 'ecfp4' else 's'
                    ax.plot(model_data['sigma'], width_ratio,
                           marker=marker, linewidth=2, markersize=6,
                           color=color, alpha=0.75, label=f'{model}/{rep}')
    
    sigma_range = np.linspace(0, metrics_df['sigma'].max(), 50)
    ax.plot(sigma_range, 1 + sigma_range * 0.5, 'k--', linewidth=2, alpha=0.5, label='Ideal')
    ax.set_xlabel('True Noise Level (σ)', fontsize=10)
    ax.set_ylabel('Width Ratio (vs σ=0)', fontsize=10)
    ax.set_title('D. Noise Detection via CP Width', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='best', ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "figure10_noise_detection.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 10 to {output_path}")
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main execution function"""
    print("="*80)
    print("PHASE 3 ANALYSIS - REVISED COMPREHENSIVE VERSION")
    print("Journal of Cheminformatics Style")
    print("="*80)
    
    # Load data
    intervals_df = load_conformal_intervals(results_dir)
    
    if len(intervals_df) == 0:
        print("\n❌ ERROR: No conformal data loaded!")
        return
    
    # Calculate metrics
    metrics_df = calculate_conformal_metrics(intervals_df)
    
    if len(metrics_df) == 0:
        print("\n❌ ERROR: No metrics calculated!")
        return
    
    # Create output directory
    output_dir = Path(results_dir) / "phase3_figures_v3"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / "phase3_conformal_metrics.csv", index=False)
    print(f"\n✓ Saved metrics to {output_dir / 'phase3_conformal_metrics.csv'}")
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    
    create_figure7_conformal_validity_efficiency(intervals_df, metrics_df, output_dir)
    create_figure8_noise_robustness(metrics_df, output_dir)
    create_figure9_detailed_calibration(metrics_df, output_dir)
    create_figure10_noise_detection(metrics_df, output_dir)
    create_supplementary_s7(intervals_df, metrics_df, output_dir)
    
    # Generate tables
    create_summary_tables(metrics_df, output_dir)
    
    # Summary
    print("\n" + "="*80)
    print("PHASE 3 ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  FIGURES:")
    print("    - figure7_conformal_validity.png")
    print("    - figure8_noise_robustness.png (width inflation, coverage stability)")
    print("    - figure9_detailed_calibration.png (per-model calibration curves)")
    print("    - figure10_noise_detection.png (CP for noise quantification)")
    print("  TABLES:")
    print("    - table4_conformal_summary_simplified.csv")
    print("    - table_phase3_conformal_summary.csv")
    print("  DATA:")
    print("    - phase3_conformal_metrics.csv")
    
    # Stats
    if len(metrics_df) > 0:
        total = len(metrics_df)
        calibrated = metrics_df['is_calibrated'].sum()
        
        print(f"\nCalibration: {calibrated}/{total} configs well-calibrated ({calibrated/total*100:.1f}%)")
        print(f"Mean coverage deviation: {metrics_df['coverage_deviation'].abs().mean():.4f}")
        
        finite_eff = metrics_df[np.isfinite(metrics_df['efficiency'])]['efficiency']
        if len(finite_eff) > 0:
            print(f"Mean efficiency: {finite_eff.mean():.2f}")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)