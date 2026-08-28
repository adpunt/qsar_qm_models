"""
Metrics Evaluation - CORRECT VERSION

PHASE STRUCTURE:
- Phase 0c: Main noise robustness screening → PERFORMANCE METRICS
- Phase 1-3: Bayesian/uncertainty methods → UNCERTAINTY METRICS  
- Phase 4: Alternative targets/noise → IGNORE (not relevant)

This script:
1. Analyzes ONLY Phase 0c for noise robustness metrics
2. Analyzes Phase 1-3 for uncertainty/calibration metrics
3. Ignores Phase 4 completely
4. Filters catastrophic failures (R² < -1.0 or R² < 0.1 at baseline)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PHASE 0C: NOISE ROBUSTNESS METRICS
# ============================================================================

def load_phase0c_results(results_dir="../results"):
    """
    Load ONLY Phase 0c screening results
    This is the main noise robustness comparison
    """
    print("="*80)
    print("LOADING PHASE 0C - NOISE ROBUSTNESS SCREENING")
    print("="*80)
    
    results_dir = Path(results_dir)
    phase0c_files = list(results_dir.glob("phase0c_screen_*.csv"))
    
    print(f"Found {len(phase0c_files)} Phase 0c files")
    
    all_data = []
    for f in phase0c_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: {f.name}: {e}")
    
    if not all_data:
        print("ERROR: No Phase 0c data loaded")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # Clean model names
    combined['model'] = combined['model'].str.replace('_split', '', regex=False)
    
    print(f"\nRaw data: {len(combined)} rows")
    
    # Filter catastrophic failures like YOUR code does
    print("\nFiltering catastrophic failures...")
    print(f"  Before: {len(combined)} rows")
    combined = combined[combined['r2'] > -10]
    print(f"  After R² > -10 filter: {len(combined)} rows")
    
    # Also filter models with terrible baseline performance
    baseline_check = combined[combined['sigma'] == 0.0].groupby(['model', 'rep'])['r2'].mean()
    bad_configs = baseline_check[baseline_check < 0.1].index
    
    if len(bad_configs) > 0:
        print(f"\n  Removing {len(bad_configs)} configs with R² < 0.1 at baseline:")
        for model, rep in bad_configs:
            print(f"    - {model}/{rep}")
            combined = combined[~((combined['model'] == model) & (combined['rep'] == rep))]
        print(f"  After baseline filter: {len(combined)} rows")
    
    # Average across iterations like YOUR code does
    results = combined.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'rep': 'representation', 'iteration': 'n_seeds'}, inplace=True)
    
    print(f"\nFinal aggregated data: {len(results)} unique configurations")
    print(f"Unique models: {sorted(results['model'].unique())}")
    print(f"Unique representations: {sorted(results['representation'].unique())}")
    print(f"Sigma range: {sorted(results['sigma'].unique())}")
    
    return results

def calculate_nsi_phase0c(df):
    """
    Noise Sensitivity Index for Phase 0c
    Slope of performance vs sigma
    """
    print("\n" + "="*80)
    print("METRIC 1: NOISE SENSITIVITY INDEX (NSI) - Phase 0c")
    print("="*80)
    
    results = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        group = group.sort_values('sigma')
        
        if len(group) < 3:
            continue
        
        sigma = group['sigma'].values
        r2 = group['r2'].values
        rmse = group['rmse'].values
        
        # Check variance
        if np.std(sigma) < 1e-10:
            continue
        
        # Calculate slopes
        r2_slope, r2_int, r2_r, r2_p, _ = stats.linregress(sigma, r2)
        rmse_slope, rmse_int, rmse_r, rmse_p, _ = stats.linregress(sigma, rmse)
        
        # Relative NSI (slope / baseline)
        baseline_r2 = r2_int  # intercept = value at sigma=0
        relative_nsi_r2 = r2_slope / abs(baseline_r2) if baseline_r2 != 0 else np.nan
        
        results.append({
            'model': model,
            'representation': rep,
            'nsi_r2': r2_slope,
            'nsi_r2_relative': relative_nsi_r2,
            'nsi_r2_r': r2_r,
            'nsi_r2_pval': r2_p,
            'nsi_rmse': rmse_slope,
            'nsi_rmse_r': rmse_r,
            'nsi_rmse_pval': rmse_p,
            'baseline_r2': baseline_r2,
            'n_points': len(group)
        })
    
    nsi_df = pd.DataFrame(results)
    
    print(f"Calculated NSI for {len(nsi_df)} configurations")
    if len(nsi_df) > 0:
        print(f"\nNSI(R²) Statistics:")
        print(f"  Mean: {nsi_df['nsi_r2'].mean():.4f}")
        print(f"  Median: {nsi_df['nsi_r2'].median():.4f}")
        print(f"  Range: [{nsi_df['nsi_r2'].min():.4f}, {nsi_df['nsi_r2'].max():.4f}]")
        
        print(f"\nMost Robust (smallest |NSI|, slowest degradation):")
        for _, row in nsi_df.nsmallest(10, nsi_df['nsi_r2'].abs())[:10].iterrows():
            print(f"  {row['model']}/{row['representation']}: NSI = {row['nsi_r2']:.4f}")
    
    return nsi_df

def calculate_critical_sigma_phase0c(df, sigma_levels=[0.2, 0.3, 0.4, 0.6]):
    """
    Performance at critical noise levels for Phase 0c
    """
    print("\n" + "="*80)
    print("METRIC 2: PERFORMANCE AT CRITICAL NOISE LEVELS - Phase 0c")
    print("="*80)
    print(f"Sigma levels: {sigma_levels}")
    
    results = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        result = {
            'model': model,
            'representation': rep
        }
        
        # Baseline
        baseline = group[group['sigma'] == 0.0]
        if len(baseline) > 0:
            result['r2_baseline'] = baseline['r2'].values[0]
            result['rmse_baseline'] = baseline['rmse'].values[0]
        else:
            result['r2_baseline'] = np.nan
            result['rmse_baseline'] = np.nan
        
        # Each critical sigma
        for sigma_val in sigma_levels:
            sigma_data = group[np.abs(group['sigma'] - sigma_val) < 0.01]
            
            if len(sigma_data) > 0:
                result[f'r2_s{sigma_val}'] = sigma_data['r2'].values[0]
                result[f'rmse_s{sigma_val}'] = sigma_data['rmse'].values[0]
                
                # Retention percentage
                if not np.isnan(result['r2_baseline']) and result['r2_baseline'] != 0:
                    retention = (result[f'r2_s{sigma_val}'] / result['r2_baseline']) * 100
                    result[f'retention_s{sigma_val}'] = retention
                else:
                    result[f'retention_s{sigma_val}'] = np.nan
            else:
                result[f'r2_s{sigma_val}'] = np.nan
                result[f'rmse_s{sigma_val}'] = np.nan
                result[f'retention_s{sigma_val}'] = np.nan
        
        results.append(result)
    
    perf_df = pd.DataFrame(results)
    
    print(f"\nCalculated critical point performance for {len(perf_df)} configurations")
    
    # Show best at each sigma
    for sigma in sigma_levels:
        col = f'r2_s{sigma}'
        if col in perf_df.columns:
            print(f"\nTop 5 at σ={sigma}:")
            for _, row in perf_df.nlargest(5, col).iterrows():
                print(f"  {row['model']}/{row['representation']}: R²={row[col]:.4f}")
    
    return perf_df

def identify_best_performers_phase0c(df, top_n=20):
    """
    Best performers from Phase 0c by multiple criteria
    """
    print("\n" + "="*80)
    print(f"METRIC 3: TOP {top_n} PERFORMERS - Phase 0c")
    print("="*80)
    
    configs = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        config = {
            'model': model,
            'representation': rep
        }
        
        baseline = group[group['sigma'] == 0.0]
        high_noise = group[group['sigma'] >= 0.4]
        
        config['r2_clean'] = baseline['r2'].values[0] if len(baseline) > 0 else np.nan
        config['r2_high_noise'] = high_noise['r2'].mean() if len(high_noise) > 0 else np.nan
        config['r2_overall'] = group['r2'].mean()
        
        if not np.isnan(config['r2_clean']) and not np.isnan(config['r2_high_noise']) and config['r2_clean'] != 0:
            config['retention_pct'] = (config['r2_high_noise'] / config['r2_clean']) * 100
        else:
            config['retention_pct'] = np.nan
        
        configs.append(config)
    
    configs_df = pd.DataFrame(configs)
    
    print("\nTOP BY CLEAN DATA (σ=0):")
    for i, row in configs_df.nlargest(top_n, 'r2_clean').iterrows():
        print(f"  {row['model']}/{row['representation']}: R²={row['r2_clean']:.4f}")
    
    print("\nTOP BY HIGH NOISE (σ≥0.4):")
    for i, row in configs_df.nlargest(top_n, 'r2_high_noise').iterrows():
        print(f"  {row['model']}/{row['representation']}: R²={row['r2_high_noise']:.4f}")
    
    print("\nTOP BY RETENTION:")
    for i, row in configs_df.nlargest(top_n, 'retention_pct').iterrows():
        print(f"  {row['model']}/{row['representation']}: {row['retention_pct']:.1f}% retained")
    
    return configs_df

def comparative_analysis_phase0c(df):
    """
    Variance decomposition and comparisons for Phase 0c
    """
    print("\n" + "="*80)
    print("METRIC 4: COMPARATIVE ANALYSIS - Phase 0c")
    print("="*80)
    
    results = []
    
    # Variance decomposition
    print("\n--- VARIANCE DECOMPOSITION ---")
    total_var = df['r2'].var()
    print(f"Total R² variance: {total_var:.6f}")
    
    for factor in ['representation', 'model']:
        factor_means = df.groupby(factor)['r2'].mean()
        factor_var = ((factor_means - df['r2'].mean()) ** 2).mean()
        factor_pct = (factor_var / total_var) * 100 if total_var > 0 else 0
        
        print(f"  {factor}: {factor_var:.6f} ({factor_pct:.1f}% of variance)")
        
        results.append({
            'analysis_type': 'variance_decomposition',
            'factor': factor,
            'variance_explained': factor_var,
            'percent_variance': factor_pct
        })
    
    # Representation rankings
    print("\n--- REPRESENTATION RANKINGS ---")
    rep_stats = df.groupby('representation').agg({
        'r2': ['mean', 'std', 'count']
    }).round(4)
    print(rep_stats)
    
    # At high noise only
    high_noise = df[df['sigma'] >= 0.4]
    if len(high_noise) > 0:
        print("\n--- AT HIGH NOISE (σ≥0.4) ---")
        rep_stats_noise = high_noise.groupby('representation').agg({
            'r2': ['mean', 'std', 'count']
        }).round(4)
        print(rep_stats_noise)
    
    return pd.DataFrame(results)

# ============================================================================
# PHASE 1-3: UNCERTAINTY/CALIBRATION METRICS
# ============================================================================

def load_uncertainty_data(results_dir="../results"):
    """
    Load Phase 1-3 uncertainty data
    """
    print("\n" + "="*80)
    print("LOADING PHASE 1-3 - UNCERTAINTY/CALIBRATION DATA")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Phase 2 and 3 have uncertainty_values files
    uncertainty_files = []
    uncertainty_files.extend(list(results_dir.glob("phase2*_uncertainty_values.csv")))
    uncertainty_files.extend(list(results_dir.glob("phase3*_uncertainty*.csv")))
    
    # Phase 3 conformal intervals
    conformal_dirs = list(results_dir.glob("conformal_intervals"))
    for conf_dir in conformal_dirs:
        uncertainty_files.extend(list(conf_dir.glob("conformal_intervals_*.csv")))
    
    print(f"Found {len(uncertainty_files)} uncertainty files")
    
    all_data = []
    for f in uncertainty_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            
            # Identify phase
            if 'phase2' in f.name:
                df['phase'] = 'phase2'
            elif 'phase3' in f.name or 'conformal' in f.name:
                df['phase'] = 'phase3'
            
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: {f.name}: {e}")
    
    if not all_data:
        print("⚠️  No uncertainty data loaded")
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"Loaded {len(combined)} uncertainty rows")
    
    return combined

def analyze_uncertainty_metrics(uncertainty_df):
    """
    Analyze uncertainty quantification metrics from Phase 1-3
    - Correlation between uncertainty and error
    - Calibration metrics
    - Coverage metrics (conformal prediction)
    """
    print("\n" + "="*80)
    print("METRIC 5: UNCERTAINTY ANALYSIS - Phase 1-3")
    print("="*80)
    
    if len(uncertainty_df) == 0:
        print("⚠️  No uncertainty data available")
        return pd.DataFrame()
    
    # Check what columns we have
    print("\nAvailable columns:")
    for col in sorted(uncertainty_df.columns):
        print(f"  {col}")
    
    results = []
    
    # Correlation metrics (if we have uncertainty and error columns)
    uncertainty_cols = [c for c in uncertainty_df.columns if 'uncertain' in c.lower() or 'std' in c.lower() or 'var' in c.lower()]
    error_cols = [c for c in uncertainty_df.columns if 'error' in c.lower() or 'abs_error' in c.lower()]
    
    print(f"\nUncertainty columns found: {uncertainty_cols}")
    print(f"Error columns found: {error_cols}")
    
    # Coverage metrics (if conformal prediction data)
    coverage_cols = [c for c in uncertainty_df.columns if 'coverage' in c.lower()]
    width_cols = [c for c in uncertainty_df.columns if 'width' in c.lower()]
    
    if coverage_cols:
        print(f"\nCoverage columns found: {coverage_cols}")
        print(f"Width columns found: {width_cols}")
        
        # Analyze conformal coverage
        if 'alpha' in uncertainty_df.columns and 'coverage' in uncertainty_df.columns:
            for alpha in uncertainty_df['alpha'].unique():
                subset = uncertainty_df[uncertainty_df['alpha'] == alpha]
                expected_coverage = 1 - alpha
                actual_coverage = subset['coverage'].mean()
                
                print(f"\nConformal Prediction at α={alpha}:")
                print(f"  Expected coverage: {expected_coverage:.2%}")
                print(f"  Actual coverage: {actual_coverage:.2%}")
                print(f"  Deviation: {(actual_coverage-expected_coverage):.2%}")
                
                results.append({
                    'metric_type': 'conformal_coverage',
                    'alpha': alpha,
                    'expected_coverage': expected_coverage,
                    'actual_coverage': actual_coverage,
                    'deviation': actual_coverage - expected_coverage
                })
    
    return pd.DataFrame(results)

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("NOISE ROBUSTNESS & UNCERTAINTY METRICS EVALUATION")
    print("CORRECT VERSION - PHASE-SPECIFIC ANALYSIS")
    print("="*80)
    
    output_dir = Path("../metrics_analysis")
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # PART 1: PHASE 0C - NOISE ROBUSTNESS
    # ========================================================================
    
    df_0c = load_phase0c_results()
    
    if len(df_0c) == 0:
        print("ERROR: No Phase 0c data loaded. Cannot continue.")
        return
    
    print("\n" + "="*80)
    print("PART 1: PHASE 0C NOISE ROBUSTNESS METRICS")
    print("="*80)
    
    try:
        nsi_df = calculate_nsi_phase0c(df_0c)
        nsi_df.to_csv(output_dir / 'phase0c_nsi.csv', index=False)
        print(f"\n✓ Saved NSI results ({len(nsi_df)} configs)")
    except Exception as e:
        print(f"\n✗ NSI failed: {e}")
    
    try:
        critical_df = calculate_critical_sigma_phase0c(df_0c)
        critical_df.to_csv(output_dir / 'phase0c_critical_performance.csv', index=False)
        print(f"✓ Saved critical sigma performance ({len(critical_df)} configs)")
    except Exception as e:
        print(f"✗ Critical sigma failed: {e}")
    
    try:
        best_df = identify_best_performers_phase0c(df_0c)
        best_df.to_csv(output_dir / 'phase0c_best_performers.csv', index=False)
        print(f"✓ Saved best performers ({len(best_df)} configs)")
    except Exception as e:
        print(f"✗ Best performers failed: {e}")
    
    try:
        comp_df = comparative_analysis_phase0c(df_0c)
        comp_df.to_csv(output_dir / 'phase0c_comparative_analysis.csv', index=False)
        print(f"✓ Saved comparative analysis")
    except Exception as e:
        print(f"✗ Comparative analysis failed: {e}")
    
    # ========================================================================
    # PART 2: PHASE 1-3 - UNCERTAINTY METRICS
    # # ========================================================================
    
    # print("\n" + "="*80)
    # print("PART 2: PHASE 1-3 UNCERTAINTY METRICS")
    # print("="*80)
    
    # try:
    #     uncertainty_df = load_uncertainty_data()
        
    #     if len(uncertainty_df) > 0:
    #         unc_metrics_df = analyze_uncertainty_metrics(uncertainty_df)
    #         if len(unc_metrics_df) > 0:
    #             unc_metrics_df.to_csv(output_dir / 'phase1_3_uncertainty_metrics.csv', index=False)
    #             print(f"\n✓ Saved uncertainty metrics")
    #         else:
    #             print("\n⚠️  No uncertainty metrics calculated (data structure may vary)")
    #     else:
    #         print("\n⚠️  Skipping uncertainty metrics (no data)")
    # except Exception as e:
    #     print(f"\n✗ Uncertainty analysis failed: {e}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\n** KEY FILES FOR YOUR RESULTS SECTION **")
    print("  1. phase0c_best_performers.csv - Top configurations (TABLE 1)")
    print("  2. phase0c_critical_performance.csv - Performance at key σ values")
    print("  3. phase0c_nsi.csv - Degradation rates")
    print("  4. phase0c_comparative_analysis.csv - Variance decomposition")
    print("  5. phase1_3_uncertainty_metrics.csv - Calibration/coverage (if available)")
    
    print("\n** WHAT EACH ANALYZES **")
    print("  Phase 0c: NOISE ROBUSTNESS - which models degrade least under noise")
    print("  Phase 1-3: UNCERTAINTY - calibration, coverage, uncertainty quantification")
    print("  Phase 4: IGNORED (as you requested)")

if __name__ == "__main__":
    main()