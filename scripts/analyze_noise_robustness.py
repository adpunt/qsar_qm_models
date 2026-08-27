#!/usr/bin/env python3
"""
Analyze noise robustness results - PROPER METRICS
Focus: How well do models maintain performance as noise increases?
"""

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

def read_csv_safe(filepath):
    """Read CSV handling both old and new formats"""
    try:
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
    except TypeError:
        df = pd.read_csv(filepath, engine='python')
    
    if 'params_source' not in df.columns:
        df['params_source'] = 'unknown'
    
    return df

def parse_model_info(model_name):
    """Parse model name to extract base model and transformation type"""
    if model_name.startswith('conformal_'):
        base = model_name.replace('conformal_', '').replace('_split', '')
        return 'conformal', base
    
    if '_full' in model_name or model_name.startswith('bnn_full'):
        base = model_name.replace('_full', '').replace('_bnn', '').replace('bnn_', '').replace('bnn', '')
        return 'full', base if base else 'dnn'
    elif '_last' in model_name or model_name.startswith('bnn_last'):
        base = model_name.replace('_last', '').replace('_bnn', '').replace('bnn_', '').replace('bnn', '')
        return 'last_layer', base if base else 'dnn'
    elif '_variational' in model_name or model_name.startswith('bnn_var'):
        base = model_name.replace('_variational', '').replace('_bnn', '').replace('bnn_', '').replace('bnn', '').replace('_var', '')
        return 'variational', base if base else 'dnn'
    
    return 'baseline', model_name

def load_results(results_dir='results_mini'):
    """Load all results"""
    pattern = os.path.join(results_dir, '*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No CSV files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"\nLoading {len(files)} files from {results_dir}...")
    
    dfs = []
    for file in files:
        try:
            df = read_csv_safe(file)
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
            print(f"  ✓ {os.path.basename(file)}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined = pd.concat(dfs, ignore_index=True)
    
    combined[['transformation_type', 'base_model']] = combined['model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    return combined

def calculate_noise_robustness_metrics(df):
    """
    Calculate proper noise robustness metrics
    Key metrics:
    1. RMSE degradation (absolute increase from σ=0 to σ=1)
    2. R² retention (R²@σ=1 / R²@σ=0)
    3. Performance at high noise (R²@σ=1)
    4. Robustness score (weighted combination)
    """
    
    metrics = []
    
    for (model, rep, trans), group in df.groupby(['base_model', 'rep', 'transformation_type']):
        # Get performance at different noise levels
        perf_0 = group[group['sigma'] == 0.0]
        perf_05 = group[group['sigma'] == 0.5]
        perf_1 = group[group['sigma'] == 1.0]
        
        if perf_0.empty or perf_1.empty:
            continue
            
        r2_0 = perf_0['r2'].mean()
        rmse_0 = perf_0['rmse'].mean()
        
        r2_1 = perf_1['r2'].mean()
        rmse_1 = perf_1['rmse'].mean()
        
        # Calculate robustness metrics
        rmse_degradation = rmse_1 - rmse_0  # Lower is better
        r2_retention = r2_1 / r2_0 if r2_0 > 0 else 0  # Higher is better (closer to 1)
        
        # Robustness score: combination of retention and absolute performance
        # Want high R² at high noise AND good retention
        robustness_score = r2_1 * r2_retention  # Higher is better
        
        metrics.append({
            'base_model': model,
            'rep': rep,
            'transformation_type': trans,
            'r2_clean': r2_0,
            'r2_noisy': r2_1,
            'rmse_clean': rmse_0,
            'rmse_noisy': rmse_1,
            'rmse_degradation': rmse_degradation,
            'r2_retention': r2_retention,
            'robustness_score': robustness_score
        })
        
        # Also get mid-noise if available
        if not perf_05.empty:
            metrics[-1]['r2_mid'] = perf_05['r2'].mean()
            metrics[-1]['rmse_mid'] = perf_05['rmse'].mean()
    
    return pd.DataFrame(metrics)

def analyze_by_noise_level(df):
    """Analyze performance at each noise level separately"""
    
    print(f"\n{'='*80}")
    print("PERFORMANCE BY NOISE LEVEL")
    print(f"{'='*80}\n")
    
    for sigma in [0.0, 0.5, 1.0]:
        print(f"--- Noise Level σ={sigma} ---")
        
        subset = df[df['sigma'] == sigma].groupby(['base_model', 'rep', 'transformation_type']).agg({
            'r2': 'mean',
            'rmse': 'mean'
        }).reset_index().sort_values('r2', ascending=False)
        
        print(f"{'Rank':<6} {'Model':<15} {'Rep':<10} {'Transform':<15} {'R²':>8} {'RMSE':>8}")
        print("-" * 80)
        
        for idx, row in subset.head(10).iterrows():
            model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
            print(f"{idx+1:<6} {model_label:<15} {row['rep']:<10} {row['transformation_type']:<15} {row['r2']:>8.4f} {row['rmse']:>8.4f}")
        
        print()

def identify_most_robust_pairs(robustness_df, n=15):
    """Identify pairs that maintain performance best under noise"""
    
    print(f"\n{'='*80}")
    print(f"TOP {n} MOST NOISE-ROBUST PAIRS")
    print("Ranked by: Smallest RMSE degradation (σ=0 → σ=1)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Model':<20} {'Rep':<10} {'Transform':<15} {'Δ RMSE':>10} {'R²@σ=0':>10} {'R²@σ=1':>10}")
    print("-" * 80)
    
    ranked = robustness_df.sort_values('rmse_degradation')
    
    for idx, row in ranked.head(n).iterrows():
        model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
        print(f"{ranked.index.get_loc(idx)+1:<6} {model_label:<20} {row['rep']:<10} {row['transformation_type']:<15} "
              f"{row['rmse_degradation']:>+10.4f} {row['r2_clean']:>10.4f} {row['r2_noisy']:>10.4f}")
    
    return ranked.head(n)

def identify_high_noise_performers(robustness_df, n=15):
    """Identify pairs with best absolute performance at high noise"""
    
    print(f"\n{'='*80}")
    print(f"TOP {n} PERFORMERS AT HIGH NOISE (σ=1.0)")
    print("Ranked by: R² at σ=1.0")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Model':<20} {'Rep':<10} {'Transform':<15} {'R²@σ=1':>10} {'R²@σ=0':>10} {'Retention':>10}")
    print("-" * 80)
    
    ranked = robustness_df.sort_values('r2_noisy', ascending=False)
    
    for idx, row in ranked.head(n).iterrows():
        model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
        print(f"{ranked.index.get_loc(idx)+1:<6} {model_label:<20} {row['rep']:<10} {row['transformation_type']:<15} "
              f"{row['r2_noisy']:>10.4f} {row['r2_clean']:>10.4f} {row['r2_retention']:>10.2%}")
    
    return ranked.head(n)

def analyze_bayesian_advantage(df):
    """Analyze if Bayesian methods provide advantage at high noise"""
    
    print(f"\n{'='*80}")
    print("BAYESIAN/CONFORMAL ADVANTAGE ANALYSIS")
    print("Comparing performance at σ=1.0")
    print(f"{'='*80}\n")
    
    comparisons = []
    
    # RF vs QRF
    for rep in df['rep'].unique():
        rf_data = df[(df['base_model'] == 'rf') & (df['transformation_type'] == 'baseline') & 
                     (df['rep'] == rep) & (df['sigma'] == 1.0)]
        qrf_data = df[(df['base_model'] == 'qrf') & (df['transformation_type'] == 'baseline') & 
                      (df['rep'] == rep) & (df['sigma'] == 1.0)]
        
        if not rf_data.empty and not qrf_data.empty:
            comparisons.append({
                'pair': 'RF vs QRF',
                'rep': rep,
                'det_r2': rf_data['r2'].mean(),
                'bay_r2': qrf_data['r2'].mean(),
                'det_rmse': rf_data['rmse'].mean(),
                'bay_rmse': qrf_data['rmse'].mean(),
                'r2_improvement': qrf_data['r2'].mean() - rf_data['r2'].mean(),
                'rmse_improvement': rf_data['rmse'].mean() - qrf_data['rmse'].mean()
            })
    
    # DNN baseline vs transformations
    for rep in ['ecfp4', 'pdv']:
        for trans in ['full', 'last_layer', 'variational']:
            baseline = df[(df['base_model'] == 'dnn') & (df['transformation_type'] == 'baseline') & 
                         (df['rep'] == rep) & (df['sigma'] == 1.0)]
            bayesian = df[(df['base_model'] == 'dnn') & (df['transformation_type'] == trans) & 
                         (df['rep'] == rep) & (df['sigma'] == 1.0)]
            
            if not baseline.empty and not bayesian.empty:
                comparisons.append({
                    'pair': f'DNN baseline vs {trans}',
                    'rep': rep,
                    'det_r2': baseline['r2'].mean(),
                    'bay_r2': bayesian['r2'].mean(),
                    'det_rmse': baseline['rmse'].mean(),
                    'bay_rmse': bayesian['rmse'].mean(),
                    'r2_improvement': bayesian['r2'].mean() - baseline['r2'].mean(),
                    'rmse_improvement': baseline['rmse'].mean() - bayesian['rmse'].mean()
                })
    
    # RF vs Conformal(RF)
    for rep in ['ecfp4', 'pdv']:
        rf_data = df[(df['base_model'] == 'rf') & (df['transformation_type'] == 'baseline') & 
                     (df['rep'] == rep) & (df['sigma'] == 1.0)]
        conf_data = df[(df['transformation_type'] == 'conformal') & (df['base_model'] == 'rf') & 
                       (df['rep'] == rep) & (df['sigma'] == 1.0)]
        
        if not rf_data.empty and not conf_data.empty:
            comparisons.append({
                'pair': 'RF vs Conformal(RF)',
                'rep': rep,
                'det_r2': rf_data['r2'].mean(),
                'bay_r2': conf_data['r2'].mean(),
                'det_rmse': rf_data['rmse'].mean(),
                'bay_rmse': conf_data['rmse'].mean(),
                'r2_improvement': conf_data['r2'].mean() - rf_data['r2'].mean(),
                'rmse_improvement': rf_data['rmse'].mean() - conf_data['rmse'].mean()
            })
    
    if not comparisons:
        print("No comparison data available")
        return pd.DataFrame()
    
    comp_df = pd.DataFrame(comparisons)
    comp_df = comp_df.sort_values('r2_improvement', ascending=False)
    
    print(f"{'Comparison':<35} {'Rep':<10} {'Det R²':>10} {'Bay R²':>10} {'Δ R²':>10} {'Δ RMSE':>10}")
    print("-" * 90)
    
    for _, row in comp_df.iterrows():
        print(f"{row['pair']:<35} {row['rep']:<10} {row['det_r2']:>10.4f} {row['bay_r2']:>10.4f} "
              f"{row['r2_improvement']:>+10.4f} {row['rmse_improvement']:>+10.4f}")
    
    # Summary statistics
    print("\n--- Summary ---")
    print(f"Mean R² improvement: {comp_df['r2_improvement'].mean():+.4f}")
    print(f"Mean RMSE improvement: {comp_df['rmse_improvement'].mean():+.4f}")
    print(f"Pairs with positive R² improvement: {(comp_df['r2_improvement'] > 0).sum()}/{len(comp_df)}")
    print(f"Pairs with positive RMSE improvement: {(comp_df['rmse_improvement'] > 0).sum()}/{len(comp_df)}")
    
    return comp_df

def check_for_issues(df):
    """Check for potential issues in the data"""
    
    print(f"\n{'='*80}")
    print("DATA QUALITY CHECKS")
    print(f"{'='*80}\n")
    
    # Check 1: Full Bayesian transformation
    full_results = df[(df['transformation_type'] == 'full')]
    if not full_results.empty:
        print("--- Full Bayesian Transformation ---")
        for model in full_results['base_model'].unique():
            for rep in full_results['rep'].unique():
                for sigma in [0.0, 1.0]:
                    subset = full_results[(full_results['base_model'] == model) & 
                                        (full_results['rep'] == rep) & 
                                        (full_results['sigma'] == sigma)]
                    if not subset.empty:
                        print(f"  {model}/{rep} σ={sigma}: R²={subset['r2'].mean():.4f}, RMSE={subset['rmse'].mean():.4f}")
        
        # Compare to baseline
        for model in ['dnn', 'mlp']:
            for rep in ['ecfp4', 'pdv']:
                baseline = df[(df['base_model'] == model) & (df['transformation_type'] == 'baseline') & 
                            (df['rep'] == rep) & (df['sigma'] == 0.0)]
                full = df[(df['base_model'] == model) & (df['transformation_type'] == 'full') & 
                         (df['rep'] == rep) & (df['sigma'] == 0.0)]
                
                if not baseline.empty and not full.empty:
                    diff = full['r2'].mean() - baseline['r2'].mean()
                    if diff < -0.03:
                        print(f"  ⚠️  {model}/{rep}: Full drops R² by {diff:.4f} (PROBLEM!)")
    
    # Check 2: Missing models
    print("\n--- Model Coverage ---")
    expected_models = ['rf', 'qrf', 'xgboost', 'svm', 'gauche', 'ngboost', 'dnn', 'mlp', 'gin', 'gcn']
    present_models = df['base_model'].unique()
    missing = set(expected_models) - set(present_models)
    if missing:
        print(f"  Missing models: {', '.join(missing)}")
    else:
        print(f"  ✓ All expected models present")
    
    # Check 3: Sample sizes
    print("\n--- Sample Sizes ---")
    for model in df['base_model'].unique()[:5]:
        count = len(df[df['base_model'] == model])
        print(f"  {model}: {count} rows")
    
    # Check 4: Sigma coverage
    print("\n--- Noise Level Coverage ---")
    sigma_counts = df.groupby('sigma').size()
    print(sigma_counts)

def generate_recommendations(robust_pairs, high_noise_pairs, bayesian_comp):
    """Generate final recommendations"""
    
    print(f"\n{'='*80}")
    print("FINAL RECOMMENDATIONS FOR FIGURES")
    print(f"{'='*80}\n")
    
    print("=== Fig 3b & 3d: Uncertainty Analysis ===")
    print("Select 4-5 pairs showing different approaches:\n")
    
    # Get most robust from each category
    tree_robust = robust_pairs[robust_pairs['base_model'].isin(['rf', 'qrf', 'xgboost'])].head(1)
    gp_robust = robust_pairs[robust_pairs['base_model'] == 'gauche'].head(1)
    nn_robust = robust_pairs[robust_pairs['base_model'].isin(['dnn', 'mlp'])].head(1)
    conf_robust = robust_pairs[robust_pairs['transformation_type'] == 'conformal'].head(1)
    
    for idx, row in pd.concat([tree_robust, gp_robust, nn_robust, conf_robust]).iterrows():
        model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
        print(f"  • {model_label} + {row['rep'].upper()}")
    
    print("\n=== Fig 4a: Different Noise Types ===")
    print("Select 3 pairs (diverse model types):\n")
    
    for idx, row in robust_pairs.head(3).iterrows():
        model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
        print(f"  • {model_label} + {row['rep'].upper()}")
    
    print("\n=== Fig 5d: Data Efficiency ===")
    print("Select comparison pairs with positive Bayesian advantage:\n")
    
    if not bayesian_comp.empty:
        positive = bayesian_comp[bayesian_comp['r2_improvement'] > 0].head(3)
        for _, row in positive.iterrows():
            print(f"  • {row['pair']} ({row['rep'].upper()}): Δ R²={row['r2_improvement']:+.4f}")

def main():
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results_mini'
    
    print(f"\n{'='*80}")
    print("NOISE ROBUSTNESS ANALYSIS - PROPER METRICS")
    print(f"{'='*80}")
    
    # Load data
    df = load_results(results_dir)
    
    if df.empty:
        print("No data to analyze!")
        return
    
    print(f"\nTotal rows loaded: {len(df)}")
    print(f"Unique models: {df['base_model'].nunique()}")
    print(f"Unique representations: {df['rep'].nunique()}")
    print(f"Sigma levels: {sorted(df['sigma'].unique())}")
    
    # Calculate robustness metrics
    robustness_df = calculate_noise_robustness_metrics(df)
    
    # Run analyses
    analyze_by_noise_level(df)
    robust_pairs = identify_most_robust_pairs(robustness_df, n=15)
    high_noise_pairs = identify_high_noise_performers(robustness_df, n=15)
    bayesian_comp = analyze_bayesian_advantage(df)
    check_for_issues(df)
    generate_recommendations(robust_pairs, high_noise_pairs, bayesian_comp)
    
    # Save results
    output_dir = Path('analysis_output_v2')
    output_dir.mkdir(exist_ok=True)
    
    robustness_df.to_csv(output_dir / 'noise_robustness_metrics.csv', index=False)
    robust_pairs.to_csv(output_dir / 'most_robust_pairs.csv', index=False)
    high_noise_pairs.to_csv(output_dir / 'high_noise_performers.csv', index=False)
    if not bayesian_comp.empty:
        bayesian_comp.to_csv(output_dir / 'bayesian_advantage.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")

if __name__ == '__main__':
    main()