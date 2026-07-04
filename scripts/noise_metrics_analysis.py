"""
Noise Robustness Metric Comparison Analysis
Explores different metrics for ranking model-representation pairs by noise robustness

Metrics explored:
1. Retention % (current)
2. NSI (slope-based)
3. Relative NSI (normalized by baseline)
4. Absolute drop (R²_baseline - R²_high)
5. Thresholded retention (baseline R² > threshold)
6. Thresholded NSI (baseline R² > threshold)
7. AUC under R²(σ) curve
8. Composite/weighted approaches
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Thresholds to explore
BASELINE_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]
SIGMA_HIGH_VALUES = [0.3, 0.4, 0.5, 0.6]

# Style
sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 150,
    'font.family': 'sans-serif',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
})

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

# ============================================================================
# DATA LOADING (same as original)
# ============================================================================

def load_screening_results(results_dir="../results"):
    """Load Phase 0c screening results"""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    screening_files = list(results_dir.glob("phase0c_screen_*.csv"))
    mhggnn_files = list(results_dir.glob("phase0_mhggnn_*.csv"))
    all_files = screening_files + mhggnn_files
    
    if not all_files:
        print("ERROR: No files found!")
        return pd.DataFrame()
    
    print(f"Found {len(all_files)} files")
    
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
    
    # Filter catastrophic failures
    combined_df = combined_df[combined_df['r2'] > -10]
    
    # Filter bad baselines
    baseline_check = combined_df[combined_df['sigma'] == 0.0].groupby(['model', 'rep'])['r2'].mean()
    bad_configs = baseline_check[baseline_check < 0.1].index
    for model, rep in bad_configs:
        combined_df = combined_df[~((combined_df['model'] == model) & (combined_df['rep'] == rep))]
    
    # Average across iterations
    results = combined_df.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'rep': 'representation', 'iteration': 'n_seeds'}, inplace=True)
    
    # Filter problematic data
    results = results[~results['representation'].isin(['graph'])]
    results = results[~results['model'].str.lower().isin(['gcn', 'gin', 'gat'])]
    
    print(f"Loaded {len(results)} rows")
    print(f"Unique models: {results['model'].nunique()}")
    print(f"Unique representations: {results['representation'].nunique()}")
    print(f"Sigma values: {sorted(results['sigma'].unique())}")
    
    return results


def format_config(model, rep):
    """Format model/representation for display"""
    model_map = {
        'rf': 'RF', 'qrf': 'QRF', 'xgboost': 'XGBoost', 'ngboost': 'NGBoost',
        'dnn': 'DNN', 'mlp': 'MLP', 'gauche': 'GP', 'svm': 'SVM',
    }
    rep_map = {
        'ecfp4': 'ECFP4', 'pdv': 'PDV', 'pdv-3d': 'PDV-3D', 'sns': 'SNS',
        'smiles': 'SMILES', 'random_smiles': 'R-SMILES', 'mhggnn': 'MHGGNN',
    }
    m = model_map.get(model.lower(), model)
    r = rep_map.get(rep.lower(), rep.upper())
    return f"{m}/{r}"


# ============================================================================
# METRIC CALCULATIONS
# ============================================================================

def calculate_all_metrics(df, sigma_high=0.6, sigma_max_for_auc=0.6):
    """
    Calculate all robustness metrics for comparison
    """
    print(f"\nCalculating metrics (σ_high={sigma_high}, σ_max_auc={sigma_max_for_auc})")
    
    metrics_list = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        metrics = {
            'model': model,
            'representation': rep,
            'config': format_config(model, rep),
        }
        
        # Baseline performance at σ=0
        sigma_0 = group[group['sigma'] == 0.0]
        if len(sigma_0) > 0:
            metrics['baseline_r2'] = sigma_0['r2'].values[0]
        else:
            metrics['baseline_r2'] = np.nan
        
        # Performance at high noise
        sigma_h = group[np.abs(group['sigma'] - sigma_high) < 0.05]
        if len(sigma_h) > 0:
            metrics['r2_high'] = sigma_h['r2'].values[0]
        else:
            metrics['r2_high'] = np.nan
        
        # Also get σ=0.3 for comparison
        sigma_03 = group[np.abs(group['sigma'] - 0.3) < 0.05]
        if len(sigma_03) > 0:
            metrics['r2_03'] = sigma_03['r2'].values[0]
        else:
            metrics['r2_03'] = np.nan
        
        # ================================================================
        # METRIC 1: Retention percentage
        # ================================================================
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            if metrics['baseline_r2'] > 0:
                metrics['retention_pct'] = (metrics['r2_high'] / metrics['baseline_r2']) * 100
            else:
                metrics['retention_pct'] = np.nan
        else:
            metrics['retention_pct'] = np.nan
        
        # ================================================================
        # METRIC 2 & 3: NSI (slope) and Relative NSI
        # ================================================================
        if len(group) >= 3:
            try:
                slope, intercept, r_val, p_val, _ = stats.linregress(group['sigma'], group['r2'])
                metrics['nsi'] = slope  # More negative = worse
                metrics['nsi_r_squared'] = r_val ** 2
                
                if intercept > 0:
                    metrics['nsi_relative'] = slope / intercept  # Normalized by baseline
                else:
                    metrics['nsi_relative'] = np.nan
            except:
                metrics['nsi'] = np.nan
                metrics['nsi_relative'] = np.nan
                metrics['nsi_r_squared'] = np.nan
        else:
            metrics['nsi'] = np.nan
            metrics['nsi_relative'] = np.nan
            metrics['nsi_r_squared'] = np.nan
        
        # ================================================================
        # METRIC 4: Absolute drop
        # ================================================================
        if not np.isnan(metrics['baseline_r2']) and not np.isnan(metrics['r2_high']):
            metrics['absolute_drop'] = metrics['baseline_r2'] - metrics['r2_high']
        else:
            metrics['absolute_drop'] = np.nan
        
        # ================================================================
        # METRIC 7: AUC under R²(σ) curve
        # ================================================================
        # Filter to sigma <= sigma_max_for_auc
        auc_group = group[group['sigma'] <= sigma_max_for_auc + 0.01].sort_values('sigma')
        if len(auc_group) >= 2:
            # Use trapezoidal integration
            metrics['auc'] = trapezoid(auc_group['r2'], auc_group['sigma'])
            # Normalized AUC (divide by sigma range to get average R²)
            sigma_range = auc_group['sigma'].max() - auc_group['sigma'].min()
            if sigma_range > 0:
                metrics['auc_normalized'] = metrics['auc'] / sigma_range
            else:
                metrics['auc_normalized'] = np.nan
        else:
            metrics['auc'] = np.nan
            metrics['auc_normalized'] = np.nan

        # ================================================================
        # CURVE-SHAPE, BASELINE-DECOUPLED METRICS (A-E)
        # Computed over the FULL sigma grid on the retention curve
        # ret(sigma) = R2(sigma) / R2(0), so every curve starts at ~1.0
        # and the summaries describe SHAPE, not absolute performance.
        # ================================================================
        # Average across iterations first -> one clean point per sigma
        full = group.groupby('sigma')['r2'].mean().reset_index().sort_values('sigma')
        sig_arr = full['sigma'].values.astype(float)
        r2_arr = full['r2'].values.astype(float)
        base = r2_arr[np.isclose(sig_arr, 0.0)][0] if np.any(np.isclose(sig_arr, 0.0)) else np.nan
        shape_keys = ['auc_norm', 'half_life', 'sigma_80', 'decay_lambda',
                      'weibull_tau', 'weibull_beta', 'slope_early', 'slope_late']
        if len(sig_arr) >= 4 and not np.isnan(base) and base > 0.1:
            ret = r2_arr / base                      # retention curve (starts at 1.0)
            srange = sig_arr.max() - sig_arr.min()

            # --- B: normalize-then-integrate (mean fractional retention) ---
            metrics['auc_norm'] = trapezoid(ret, sig_arr) / srange if srange > 0 else np.nan

            # --- C: half-life / sigma_80 (sigma where retention crosses target) ---
            def _crossing(target):
                below = np.where(ret <= target)[0]
                if len(below) == 0:
                    return np.nan            # censored: never crosses within grid
                j = below[0]
                if j == 0:
                    return sig_arr[0]
                x0, x1, y0, y1 = sig_arr[j-1], sig_arr[j], ret[j-1], ret[j]
                return x1 if y1 == y0 else x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            metrics['half_life'] = _crossing(0.5)
            metrics['sigma_80'] = _crossing(0.8)

            # --- D1: exponential decay rate  ret = exp(-lambda*sigma) ---
            ret_pos = np.clip(ret, 1e-3, None)
            try:
                popt, _ = curve_fit(lambda s, lam: np.exp(-lam * s), sig_arr, ret_pos,
                                    p0=[1.0], maxfev=10000, bounds=(0, np.inf))
                metrics['decay_lambda'] = float(popt[0])
            except Exception:
                metrics['decay_lambda'] = np.nan

            # --- D2: Weibull/stretched-exp  ret = exp(-(sigma/tau)^beta) ---
            try:
                def _weib(s, tau, beta):
                    return np.exp(-np.power(np.clip(s, 0, None) / tau, beta))
                popt, _ = curve_fit(_weib, sig_arr, ret_pos, p0=[0.5, 1.5],
                                    maxfev=10000, bounds=([1e-3, 0.1], [np.inf, 10.0]))
                metrics['weibull_tau'] = float(popt[0])
                metrics['weibull_beta'] = float(popt[1])
            except Exception:
                metrics['weibull_tau'] = np.nan
                metrics['weibull_beta'] = np.nan

            # --- E: early (<=0.3) vs late (>=0.3) slope of RAW R2 ---
            early = full[full['sigma'] <= 0.3]
            late = full[full['sigma'] >= 0.3]
            metrics['slope_early'] = stats.linregress(early['sigma'], early['r2'])[0] if len(early) >= 2 else np.nan
            metrics['slope_late'] = stats.linregress(late['sigma'], late['r2'])[0] if len(late) >= 2 else np.nan
        else:
            for _k in shape_keys:
                metrics[_k] = np.nan

        # Store individual sigma values for later analysis
        for sig in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            sigma_val = group[np.abs(group['sigma'] - sig) < 0.05]
            if len(sigma_val) > 0:
                metrics[f'r2_s{sig}'] = sigma_val['r2'].values[0]
            else:
                metrics[f'r2_s{sig}'] = np.nan
        
        metrics_list.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_list)
    print(f"Calculated metrics for {len(metrics_df)} configurations")
    
    return metrics_df


def apply_threshold_filter(metrics_df, baseline_threshold):
    """
    Filter to configurations above baseline threshold
    """
    filtered = metrics_df[metrics_df['baseline_r2'] >= baseline_threshold].copy()
    return filtered


# ============================================================================
# RANKING COMPARISONS
# ============================================================================

def compare_rankings(metrics_df, top_n=20):
    """
    Compare how different metrics rank configurations
    """
    print("\n" + "="*80)
    print(f"RANKING COMPARISON (Top {top_n})")
    print("="*80)
    
    # Define ranking directions (higher is better vs lower is better)
    ranking_configs = {
        'retention_pct': ('desc', 'Retention %'),
        'nsi': ('desc', 'NSI'),  # Less negative = better
        'nsi_relative': ('desc', 'Relative NSI'),
        'absolute_drop': ('asc', 'Absolute Drop'),  # Lower = better
        'auc': ('desc', 'AUC'),
        'auc_normalized': ('desc', 'Normalized AUC'),
        # --- A-E: curve-shape, baseline-decoupled ---
        'auc_norm': ('desc', 'B: Retention AUC'),         # higher = more retained
        'half_life': ('desc', 'C: Half-life σ½'),          # higher = more robust
        'sigma_80': ('desc', 'C: σ80'),
        'decay_lambda': ('asc', 'D1: Decay rate λ'),       # lower = slower decay
        'weibull_tau': ('desc', 'D2: Weibull τ'),          # higher = breaks later
        'slope_early': ('desc', 'E: Early slope'),         # less negative = better
        'slope_late': ('desc', 'E: Late slope'),
        'baseline_r2': ('desc', 'Baseline R²'),
        'r2_high': ('desc', 'R² at High Noise'),
    }
    
    rankings = {}
    
    for metric, (direction, label) in ranking_configs.items():
        if metric not in metrics_df.columns:
            continue
        
        valid = metrics_df.dropna(subset=[metric])
        if len(valid) == 0:
            continue
        
        ascending = (direction == 'asc')
        ranked = valid.sort_values(metric, ascending=ascending).reset_index(drop=True)
        ranked['rank'] = range(1, len(ranked) + 1)
        
        rankings[metric] = ranked[['config', 'model', 'representation', metric, 'rank', 
                                   'baseline_r2', 'r2_high', 'retention_pct', 'nsi']].head(top_n)
    
    return rankings


def print_ranking_comparison(rankings, metrics_to_show=None):
    """Print side-by-side ranking comparison"""
    if metrics_to_show is None:
        metrics_to_show = ['retention_pct', 'nsi', 'absolute_drop', 'auc_normalized']
    
    print("\n" + "="*80)
    print("TOP 10 BY EACH METRIC")
    print("="*80)
    
    for metric in metrics_to_show:
        if metric not in rankings:
            continue
        
        df = rankings[metric].head(10)
        print(f"\n--- {metric.upper()} ---")
        print(f"{'Rank':<5} {'Config':<25} {'Value':>10} {'Base R²':>10} {'R²_high':>10} {'Ret%':>8}")
        print("-" * 75)
        
        for _, row in df.iterrows():
            print(f"{row['rank']:<5} {row['config']:<25} {row[metric]:>10.4f} "
                  f"{row['baseline_r2']:>10.3f} {row['r2_high']:>10.3f} {row['retention_pct']:>8.1f}")


def calculate_rank_correlations(metrics_df):
    """Calculate Spearman rank correlations between metrics"""
    print("\n" + "="*80)
    print("RANK CORRELATIONS BETWEEN METRICS")
    print("="*80)
    
    metrics_to_compare = ['retention_pct', 'nsi', 'nsi_relative', 'absolute_drop',
                          'auc', 'auc_normalized',
                          'auc_norm', 'half_life', 'sigma_80', 'decay_lambda',
                          'weibull_tau', 'weibull_beta', 'slope_early', 'slope_late',
                          'baseline_r2', 'r2_high']

    available = [m for m in metrics_to_compare if m in metrics_df.columns]

    # Create correlation matrix
    corr_matrix = pd.DataFrame(index=available, columns=available, dtype=float)

    for m1 in available:
        for m2 in available:
            valid = metrics_df.dropna(subset=[m1, m2])
            if len(valid) >= 3:
                corr, pval = stats.spearmanr(valid[m1], valid[m2])
                corr_matrix.loc[m1, m2] = corr
            else:
                corr_matrix.loc[m1, m2] = np.nan

    print("\nSpearman Rank Correlations:")
    print(corr_matrix.round(3).to_string())

    # ------------------------------------------------------------------
    # KEY DIAGNOSTIC: dependence on predictive performance.
    # A good curve-shape metric ranks robustness WITHOUT just re-ranking
    # baseline R². So |Spearman(metric, baseline_r2)| should be LOW.
    # ------------------------------------------------------------------
    if 'baseline_r2' in available:
        print("\n" + "-"*60)
        print("DEPENDENCE ON BASELINE R²  (|rho| low = well decoupled)")
        print("-"*60)
        dep = (corr_matrix['baseline_r2'].drop('baseline_r2', errors='ignore')
               .abs().sort_values())
        for m, v in dep.items():
            flag = "  <-- decoupled" if v < 0.3 else ("  <-- performance-driven" if v > 0.6 else "")
            print(f"  {m:<16} |rho_baseline| = {v:.3f}{flag}")

    return corr_matrix


# ============================================================================
# THRESHOLD ANALYSIS
# ============================================================================

def analyze_threshold_effects(metrics_df, thresholds=None):
    """
    Analyze how baseline threshold affects rankings for BOTH retention and NSI
    """
    if thresholds is None:
        thresholds = BASELINE_THRESHOLDS
    
    print("\n" + "="*80)
    print("THRESHOLD ANALYSIS (Retention AND NSI)")
    print("="*80)
    
    results = []
    
    for thresh in thresholds:
        filtered = apply_threshold_filter(metrics_df, thresh)
        n_configs = len(filtered)
        
        if n_configs < 5:
            print(f"\nThreshold {thresh}: Only {n_configs} configs, skipping")
            continue
        
        # Get top 5 by retention after filtering
        top_retention = filtered.nlargest(5, 'retention_pct')[['config', 'retention_pct', 'baseline_r2', 'r2_high', 'nsi']]
        
        # Get top 5 by NSI (least negative) after filtering
        top_nsi = filtered.nlargest(5, 'nsi')[['config', 'nsi', 'baseline_r2', 'r2_high', 'retention_pct']]
        
        # Get top 5 by absolute drop (lowest)
        top_drop = filtered.nsmallest(5, 'absolute_drop')[['config', 'absolute_drop', 'baseline_r2', 'r2_high', 'retention_pct']]
        
        results.append({
            'threshold': thresh,
            'n_configs': n_configs,
            'top_retention': top_retention,
            'top_nsi': top_nsi,
            'top_drop': top_drop,
        })
        
        print(f"\n{'='*60}")
        print(f"Baseline R² ≥ {thresh} ({n_configs} configs)")
        print(f"{'='*60}")
        
        print("\nTop 5 by RETENTION %:")
        for _, row in top_retention.iterrows():
            print(f"  {row['config']:<25} Ret={row['retention_pct']:>6.1f}%  Base={row['baseline_r2']:.3f}  High={row['r2_high']:.3f}  NSI={row['nsi']:.4f}")
        
        print("\nTop 5 by NSI (least degradation):")
        for _, row in top_nsi.iterrows():
            print(f"  {row['config']:<25} NSI={row['nsi']:>7.4f}  Base={row['baseline_r2']:.3f}  High={row['r2_high']:.3f}  Ret={row['retention_pct']:.1f}%")
        
        print("\nTop 5 by ABSOLUTE DROP (smallest):")
        for _, row in top_drop.iterrows():
            print(f"  {row['config']:<25} Drop={row['absolute_drop']:>6.4f}  Base={row['baseline_r2']:.3f}  High={row['r2_high']:.3f}  Ret={row['retention_pct']:.1f}%")
    
    return results


def analyze_auc_with_different_sigma_max(df, sigma_max_values=None):
    """
    Test AUC calculation with different max sigma values
    """
    if sigma_max_values is None:
        sigma_max_values = [0.3, 0.4, 0.5, 0.6]
    
    print("\n" + "="*80)
    print("AUC ANALYSIS WITH DIFFERENT σ_max VALUES")
    print("="*80)
    
    results_by_sigma = {}
    
    for sigma_max in sigma_max_values:
        metrics = calculate_all_metrics(df, sigma_high=sigma_max, sigma_max_for_auc=sigma_max)
        
        top_auc = metrics.nlargest(10, 'auc_normalized')[['config', 'auc_normalized', 'baseline_r2', 'r2_high', 'retention_pct']]
        results_by_sigma[sigma_max] = top_auc
        
        print(f"\n--- σ_max = {sigma_max} ---")
        print(f"{'Rank':<5} {'Config':<25} {'Norm AUC':>10} {'Base R²':>10} {'Ret%':>8}")
        print("-" * 65)
        for i, (_, row) in enumerate(top_auc.iterrows(), 1):
            print(f"{i:<5} {row['config']:<25} {row['auc_normalized']:>10.4f} {row['baseline_r2']:>10.3f} {row['retention_pct']:>8.1f}")
    
    return results_by_sigma


# ============================================================================
# SPECIAL CASE ANALYSIS
# ============================================================================

def analyze_ngboost_case(df, metrics_df):
    """
    Special analysis for NGBoost configurations (your interesting case)
    """
    print("\n" + "="*80)
    print("NGBOOST ANALYSIS (SPECIAL CASE)")
    print("="*80)
    
    ngboost_metrics = metrics_df[metrics_df['model'] == 'ngboost'].copy()
    
    if len(ngboost_metrics) == 0:
        print("No NGBoost data found")
        return
    
    print("\nNGBoost configurations across all representations:")
    print(f"{'Rep':<15} {'Base R²':>10} {'R²_high':>10} {'Ret%':>10} {'NSI':>10} {'AbsDrop':>10} {'AUC_norm':>10}")
    print("-" * 85)
    
    for _, row in ngboost_metrics.sort_values('representation').iterrows():
        print(f"{row['representation']:<15} {row['baseline_r2']:>10.4f} {row['r2_high']:>10.4f} "
              f"{row['retention_pct']:>10.1f} {row['nsi']:>10.4f} {row['absolute_drop']:>10.4f} {row.get('auc_normalized', np.nan):>10.4f}")
    
    # Compare NGBoost to other models on same representations
    print("\n" + "-"*80)
    print("NGBoost vs Others (same representation)")
    print("-"*80)
    
    for rep in ['pdv', 'sns']:
        rep_data = metrics_df[metrics_df['representation'] == rep].copy()
        if len(rep_data) == 0:
            continue
        
        rep_data = rep_data.sort_values('nsi', ascending=False)  # Best NSI first
        
        # Find NGBoost rank
        ngb_row = rep_data[rep_data['model'] == 'ngboost']
        if len(ngb_row) > 0:
            ngb_nsi_rank = (rep_data['nsi'] > ngb_row['nsi'].values[0]).sum() + 1
            ngb_ret_rank = (rep_data['retention_pct'] > ngb_row['retention_pct'].values[0]).sum() + 1
        else:
            ngb_nsi_rank = "N/A"
            ngb_ret_rank = "N/A"
        
        print(f"\n--- {rep.upper()} (NGBoost ranks: NSI #{ngb_nsi_rank}, Retention #{ngb_ret_rank} out of {len(rep_data)}) ---")
        print(f"{'Model':<15} {'Base R²':>10} {'R²_high':>10} {'Ret%':>10} {'NSI':>10}")
        print("-" * 60)
        
        for _, row in rep_data.head(15).iterrows():
            marker = " ***" if row['model'] == 'ngboost' else ""
            print(f"{row['model']:<15} {row['baseline_r2']:>10.4f} {row['r2_high']:>10.4f} "
                  f"{row['retention_pct']:>10.1f} {row['nsi']:>10.4f}{marker}")


def identify_stably_bad_configs(metrics_df, baseline_max=0.5, retention_min=80):
    """
    Identify "stably bad" configurations - high retention but low baseline
    These are the problematic cases for retention-based metrics
    """
    print("\n" + "="*80)
    print("STABLY BAD CONFIGURATIONS")
    print(f"(Baseline R² < {baseline_max} AND Retention > {retention_min}%)")
    print("="*80)
    
    stably_bad = metrics_df[
        (metrics_df['baseline_r2'] < baseline_max) & 
        (metrics_df['retention_pct'] > retention_min)
    ].sort_values('retention_pct', ascending=False)
    
    if len(stably_bad) == 0:
        print("No 'stably bad' configurations found with these criteria")
        return pd.DataFrame()
    
    print(f"\nFound {len(stably_bad)} configurations:")
    print(f"{'Config':<30} {'Base R²':>10} {'R²_high':>10} {'Ret%':>10} {'NSI':>10}")
    print("-" * 75)
    
    for _, row in stably_bad.iterrows():
        print(f"{row['config']:<30} {row['baseline_r2']:>10.4f} {row['r2_high']:>10.4f} "
              f"{row['retention_pct']:>10.1f} {row['nsi']:>10.4f}")
    
    return stably_bad


def identify_high_performers_with_degradation(metrics_df, baseline_min=0.7, retention_max=70):
    """
    Identify high baseline performers with significant degradation
    """
    print("\n" + "="*80)
    print("HIGH PERFORMERS WITH DEGRADATION")
    print(f"(Baseline R² ≥ {baseline_min} AND Retention < {retention_max}%)")
    print("="*80)
    
    degraded = metrics_df[
        (metrics_df['baseline_r2'] >= baseline_min) & 
        (metrics_df['retention_pct'] < retention_max)
    ].sort_values('baseline_r2', ascending=False)
    
    if len(degraded) == 0:
        print("No configurations found with these criteria")
        return pd.DataFrame()
    
    print(f"\nFound {len(degraded)} configurations:")
    print(f"{'Config':<30} {'Base R²':>10} {'R²_high':>10} {'Ret%':>10} {'NSI':>10}")
    print("-" * 75)
    
    for _, row in degraded.iterrows():
        print(f"{row['config']:<30} {row['baseline_r2']:>10.4f} {row['r2_high']:>10.4f} "
              f"{row['retention_pct']:>10.1f} {row['nsi']:>10.4f}")
    
    return degraded


# ============================================================================
# COMPOSITE METRICS
# ============================================================================

def calculate_composite_metrics(metrics_df):
    """
    Calculate composite metrics that balance baseline performance and robustness
    """
    print("\n" + "="*80)
    print("COMPOSITE METRICS")
    print("="*80)
    
    df = metrics_df.copy()
    
    # Composite 1: Weighted product of baseline and retention
    # This penalizes both low baseline AND low retention
    df['composite_product'] = df['baseline_r2'] * (df['retention_pct'] / 100)
    
    # Composite 2: R² at high noise (simplest - just look at final performance)
    # Already have this as r2_high
    
    # Composite 3: Weighted sum with customizable weights
    # Normalize metrics first
    baseline_norm = (df['baseline_r2'] - df['baseline_r2'].min()) / (df['baseline_r2'].max() - df['baseline_r2'].min())
    retention_norm = (df['retention_pct'] - df['retention_pct'].min()) / (df['retention_pct'].max() - df['retention_pct'].min())
    
    # Try different weight combinations
    for w_base in [0.3, 0.5, 0.7]:
        w_ret = 1 - w_base
        df[f'composite_w{int(w_base*10)}'] = w_base * baseline_norm + w_ret * retention_norm
    
    # Composite 4: Harmonic mean of baseline and retention
    df['composite_harmonic'] = 2 * df['baseline_r2'] * (df['retention_pct']/100) / (df['baseline_r2'] + df['retention_pct']/100 + 1e-10)
    
    # Composite 5: Penalized retention (subtract penalty for low baseline)
    baseline_penalty = np.maximum(0, 0.6 - df['baseline_r2']) * 50  # Penalty kicks in below 0.6
    df['composite_penalized_ret'] = df['retention_pct'] - baseline_penalty
    
    # Show rankings by each composite
    composites = ['composite_product', 'r2_high', 'composite_w3', 'composite_w5', 
                  'composite_w7', 'composite_harmonic', 'composite_penalized_ret']
    
    print("\nTop 10 by each composite metric:")
    
    for comp in composites:
        if comp not in df.columns:
            continue
        
        top = df.nlargest(10, comp)[['config', comp, 'baseline_r2', 'r2_high', 'retention_pct']]
        
        print(f"\n--- {comp} ---")
        for i, (_, row) in enumerate(top.iterrows(), 1):
            print(f"  {i:2d}. {row['config']:<25} {row[comp]:>8.4f}  "
                  f"(Base={row['baseline_r2']:.3f}, Ret={row['retention_pct']:.1f}%)")
    
    return df


# ============================================================================
# VISUALIZATION
# ============================================================================

def create_diagnostic_plots(metrics_df, output_dir):
    """Create diagnostic plots for metric comparison"""
    print("\n" + "="*80)
    print("CREATING DIAGNOSTIC PLOTS")
    print("="*80)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Baseline vs Retention with quadrant analysis
    ax = axes[0, 0]
    for rep in metrics_df['representation'].unique():
        rep_data = metrics_df[metrics_df['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['retention_pct'], 
                  alpha=0.7, s=40, color=color, label=rep.upper(), edgecolors='black', linewidth=0.3)
    
    # Add quadrant lines
    ax.axvline(0.6, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(70, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax.set_xlabel('Baseline R²')
    ax.set_ylabel('Retention %')
    ax.set_title('Baseline vs Retention\n(Red lines: threshold candidates)')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_xlim(0, 1)
    
    # Annotate quadrants
    ax.text(0.3, 85, 'STABLY BAD\n(High ret, low base)', fontsize=8, ha='center', color='red', alpha=0.7)
    ax.text(0.8, 85, 'IDEAL\n(High both)', fontsize=8, ha='center', color='green', alpha=0.7)
    ax.text(0.3, 50, 'BAD\n(Low both)', fontsize=8, ha='center', color='gray', alpha=0.7)
    ax.text(0.8, 50, 'FRAGILE\n(High base, low ret)', fontsize=8, ha='center', color='orange', alpha=0.7)
    
    # Plot 2: Baseline vs NSI
    ax = axes[0, 1]
    for rep in metrics_df['representation'].unique():
        rep_data = metrics_df[metrics_df['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['nsi'], 
                  alpha=0.7, s=40, color=color, label=rep.upper(), edgecolors='black', linewidth=0.3)
    ax.axhline(0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xlabel('Baseline R²')
    ax.set_ylabel('NSI (slope)')
    ax.set_title('Baseline vs NSI\n(Higher/less negative = more robust)')
    
    # Plot 3: Retention vs NSI
    ax = axes[0, 2]
    scatter = ax.scatter(metrics_df['retention_pct'], metrics_df['nsi'], 
                        c=metrics_df['baseline_r2'], cmap='viridis', alpha=0.7, s=40, 
                        edgecolors='black', linewidth=0.3)
    plt.colorbar(scatter, ax=ax, label='Baseline R²')
    ax.set_xlabel('Retention %')
    ax.set_ylabel('NSI')
    ax.set_title('Retention vs NSI\n(Color = baseline R²)')
    
    # Plot 4: AUC vs Baseline
    ax = axes[1, 0]
    for rep in metrics_df['representation'].unique():
        rep_data = metrics_df[metrics_df['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['auc_normalized'], 
                  alpha=0.7, s=40, color=color, label=rep.upper(), edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Baseline R²')
    ax.set_ylabel('Normalized AUC')
    ax.set_title('Baseline vs Normalized AUC')
    ax.legend(fontsize=7, loc='lower right')
    
    # Plot 5: Absolute drop vs Baseline
    ax = axes[1, 1]
    for rep in metrics_df['representation'].unique():
        rep_data = metrics_df[metrics_df['representation'] == rep]
        color = REPRESENTATION_COLORS.get(rep, '#999999')
        ax.scatter(rep_data['baseline_r2'], rep_data['absolute_drop'], 
                  alpha=0.7, s=40, color=color, label=rep.upper(), edgecolors='black', linewidth=0.3)
    ax.set_xlabel('Baseline R²')
    ax.set_ylabel('Absolute Drop (R²_base - R²_high)')
    ax.set_title('Baseline vs Absolute Drop')
    
    # Plot 6: Distribution of retention by baseline bin
    ax = axes[1, 2]
    df_plot = metrics_df.copy()
    df_plot['baseline_bin'] = pd.cut(df_plot['baseline_r2'], 
                                     bins=[0, 0.4, 0.6, 0.8, 1.0], 
                                     labels=['<0.4', '0.4-0.6', '0.6-0.8', '>0.8'])
    df_plot.boxplot(column='retention_pct', by='baseline_bin', ax=ax)
    ax.set_xlabel('Baseline R² bin')
    ax.set_ylabel('Retention %')
    ax.set_title('Retention Distribution by Baseline')
    plt.suptitle('')  # Remove automatic title
    
    plt.tight_layout()
    output_path = output_dir / "metric_comparison_diagnostics.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved diagnostic plots to {output_path}")
    plt.close()
    
    # Additional plot: Correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    metrics_for_corr = ['baseline_r2', 'r2_high', 'retention_pct', 'nsi', 'nsi_relative', 
                        'absolute_drop', 'auc', 'auc_normalized']
    available = [m for m in metrics_for_corr if m in metrics_df.columns]
    
    corr_data = metrics_df[available].corr(method='spearman')
    
    mask = np.triu(np.ones_like(corr_data, dtype=bool), k=1)
    sns.heatmap(corr_data, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Spearman Rank Correlations Between Metrics')
    
    output_path = output_dir / "metric_correlations_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved correlation heatmap to {output_path}")
    plt.close()


def create_ranking_stability_plot(metrics_df, output_dir):
    """Show how rankings change with baseline threshold"""
    print("\nCreating ranking stability plot...")
    
    output_dir = Path(output_dir)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Track top 5 configs across thresholds
    thresholds = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    # By retention
    ax = axes[0]
    for thresh in thresholds:
        filtered = metrics_df[metrics_df['baseline_r2'] >= thresh].copy()
        if len(filtered) < 5:
            continue
        
        top5 = filtered.nlargest(5, 'retention_pct')['config'].tolist()
        for rank, config in enumerate(top5, 1):
            ax.scatter(thresh, rank, s=100, alpha=0.7)
            ax.annotate(config, (thresh, rank), fontsize=6, rotation=15)
    
    ax.set_xlabel('Baseline R² Threshold')
    ax.set_ylabel('Rank')
    ax.set_title('Top 5 by RETENTION at Different Thresholds')
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # By NSI
    ax = axes[1]
    for thresh in thresholds:
        filtered = metrics_df[metrics_df['baseline_r2'] >= thresh].copy()
        if len(filtered) < 5:
            continue
        
        top5 = filtered.nlargest(5, 'nsi')['config'].tolist()
        for rank, config in enumerate(top5, 1):
            ax.scatter(thresh, rank, s=100, alpha=0.7)
            ax.annotate(config, (thresh, rank), fontsize=6, rotation=15)
    
    ax.set_xlabel('Baseline R² Threshold')
    ax.set_ylabel('Rank')
    ax.set_title('Top 5 by NSI at Different Thresholds')
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3, 4, 5])
    
    # By absolute drop (lowest)
    ax = axes[2]
    for thresh in thresholds:
        filtered = metrics_df[metrics_df['baseline_r2'] >= thresh].copy()
        if len(filtered) < 5:
            continue
        
        top5 = filtered.nsmallest(5, 'absolute_drop')['config'].tolist()
        for rank, config in enumerate(top5, 1):
            ax.scatter(thresh, rank, s=100, alpha=0.7)
            ax.annotate(config, (thresh, rank), fontsize=6, rotation=15)
    
    ax.set_xlabel('Baseline R² Threshold')
    ax.set_ylabel('Rank')
    ax.set_title('Top 5 by ABSOLUTE DROP at Different Thresholds')
    ax.invert_yaxis()
    ax.set_yticks([1, 2, 3, 4, 5])
    
    plt.tight_layout()
    output_path = output_dir / "ranking_stability_by_threshold.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved ranking stability plot to {output_path}")
    plt.close()


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def generate_summary_report(metrics_df, output_dir):
    """Generate a text summary of key findings"""
    print("\n" + "="*80)
    print("GENERATING SUMMARY REPORT")
    print("="*80)
    
    output_dir = Path(output_dir)
    
    lines = [
        "="*80,
        "NOISE ROBUSTNESS METRIC COMPARISON - SUMMARY REPORT",
        "="*80,
        "",
        f"Total configurations analyzed: {len(metrics_df)}",
        f"Models: {metrics_df['model'].nunique()}",
        f"Representations: {metrics_df['representation'].nunique()}",
        "",
        "="*80,
        "METRIC COMPARISON: TOP 5 BY EACH METRIC",
        "="*80,
    ]
    
    metrics_to_report = [
        ('retention_pct', 'Retention %', False),
        ('nsi', 'NSI (less negative = better)', False),
        ('absolute_drop', 'Absolute Drop (lower = better)', True),
        ('auc_normalized', 'Normalized AUC', False),
        ('r2_high', 'R² at High Noise', False),
    ]
    
    for metric, label, ascending in metrics_to_report:
        if metric not in metrics_df.columns:
            continue
        
        top5 = metrics_df.nsmallest(5, metric) if ascending else metrics_df.nlargest(5, metric)
        
        lines.append(f"\n--- {label} ---")
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            lines.append(f"  {i}. {row['config']:<25} {metric}={row[metric]:.4f}  Base={row['baseline_r2']:.3f}")
    
    # Agreement analysis
    lines.extend([
        "",
        "="*80,
        "METRIC AGREEMENT ANALYSIS",
        "="*80,
    ])
    
    top10_ret = set(metrics_df.nlargest(10, 'retention_pct')['config'])
    top10_nsi = set(metrics_df.nlargest(10, 'nsi')['config'])
    top10_drop = set(metrics_df.nsmallest(10, 'absolute_drop')['config'])
    
    overlap_ret_nsi = top10_ret & top10_nsi
    overlap_ret_drop = top10_ret & top10_drop
    overlap_nsi_drop = top10_nsi & top10_drop
    overlap_all = top10_ret & top10_nsi & top10_drop
    
    lines.append(f"\nTop 10 overlap:")
    lines.append(f"  Retention ∩ NSI: {len(overlap_ret_nsi)} configs")
    lines.append(f"  Retention ∩ AbsDrop: {len(overlap_ret_drop)} configs")
    lines.append(f"  NSI ∩ AbsDrop: {len(overlap_nsi_drop)} configs")
    lines.append(f"  All three: {len(overlap_all)} configs")
    
    if overlap_all:
        lines.append(f"\n  Configs in all three top-10 lists:")
        for cfg in sorted(overlap_all):
            lines.append(f"    - {cfg}")
    
    # Write report
    report_path = output_dir / "metric_comparison_summary.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Saved summary report to {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main(results_dir="../results"):
    """Main analysis"""
    print("="*80)
    print("NOISE ROBUSTNESS METRIC COMPARISON ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_screening_results(results_dir)
    if len(df) == 0:
        print("ERROR: No data loaded!")
        return
    
    # Test different sigma_high values
    print("\n" + "="*80)
    print("TESTING DIFFERENT σ_high VALUES")
    print("="*80)
    
    for sigma_high in [0.3, 0.6]:
        print(f"\n{'='*40}")
        print(f"σ_high = {sigma_high}")
        print(f"{'='*40}")
        
        metrics_df = calculate_all_metrics(df, sigma_high=sigma_high, sigma_max_for_auc=sigma_high)
        
        print(f"\nTop 5 by Retention:")
        for _, row in metrics_df.nlargest(5, 'retention_pct').iterrows():
            print(f"  {row['config']:<25} Ret={row['retention_pct']:.1f}%  Base={row['baseline_r2']:.3f}")
        
        print(f"\nTop 5 by NSI:")
        for _, row in metrics_df.nlargest(5, 'nsi').iterrows():
            print(f"  {row['config']:<25} NSI={row['nsi']:.4f}  Base={row['baseline_r2']:.3f}")
    
    # Main analysis with σ_high = 0.6
    print("\n" + "="*80)
    print("MAIN ANALYSIS (σ_high = 0.6)")
    print("="*80)
    
    metrics_df = calculate_all_metrics(df, sigma_high=0.6, sigma_max_for_auc=0.6)
    
    # Compare rankings
    rankings = compare_rankings(metrics_df, top_n=20)
    print_ranking_comparison(rankings)
    
    # Rank correlations
    corr_matrix = calculate_rank_correlations(metrics_df)
    
    # Threshold analysis (both retention AND NSI)
    threshold_results = analyze_threshold_effects(metrics_df)
    
    # AUC with different sigma_max
    auc_results = analyze_auc_with_different_sigma_max(df)
    
    # Special cases
    print("\n" + "="*80)
    print("IDENTIFYING PROBLEMATIC CONFIGURATIONS")
    print("="*80)
    
    identify_stably_bad_configs(metrics_df, baseline_max=0.5, retention_min=80)
    identify_stably_bad_configs(metrics_df, baseline_max=0.6, retention_min=70)
    identify_high_performers_with_degradation(metrics_df, baseline_min=0.7, retention_max=60)
    
    # NGBoost analysis
    analyze_ngboost_case(df, metrics_df)
    
    # Composite metrics
    metrics_df = calculate_composite_metrics(metrics_df)
    
    # Create output directory
    output_dir = Path(results_dir) / "metric_comparison_analysis"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Plots
    create_diagnostic_plots(metrics_df, output_dir)
    create_ranking_stability_plot(metrics_df, output_dir)
    
    # Summary report
    generate_summary_report(metrics_df, output_dir)
    
    # Save full metrics
    metrics_df.to_csv(output_dir / "all_metrics_comparison.csv", index=False)
    print(f"\n✓ Saved all metrics to {output_dir / 'all_metrics_comparison.csv'}")
    
    # Save correlation matrix
    corr_matrix.to_csv(output_dir / "metric_correlations.csv")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nKey questions to answer from output:")
    print("  1. Do retention and NSI agree on top performers?")
    print("  2. How many 'stably bad' configs inflate retention rankings?")
    print("  3. Where does NGBoost/PDV and NGBoost/SNS rank under different metrics?")
    print("  4. How does the baseline R² threshold change the rankings?")
    print("  5. Which composite metric best balances your goals?")
    print("  6. Does AUC behave better with lower σ_max?")


if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "../results"
    main(results_dir)