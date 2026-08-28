#!/usr/bin/env python3
"""
Analyze results from mini data generation to identify top-performing model/rep pairs
Purpose: Select pairs to highlight in figures 3b, 3d, 4a, 5b
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
    """Load all results from mini run"""
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
    
    # Parse model info
    combined[['transformation_type', 'base_model']] = combined['model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    return combined

def identify_top_baseline_pairs(df, n=10):
    """Identify top performing model/rep pairs at sigma=0"""
    baseline = df[df['sigma'] == 0.0].copy()
    
    if baseline.empty:
        print("No baseline (sigma=0) data found")
        return pd.DataFrame()
    
    # Average across bootstraps
    summary = baseline.groupby(['base_model', 'rep', 'transformation_type']).agg({
        'rmse': 'mean',
        'r2': 'mean',
        'mae': 'mean'
    }).reset_index()
    
    # Sort by R2 (higher is better)
    summary = summary.sort_values('r2', ascending=False)
    
    print(f"\n{'='*80}")
    print("TOP {n} MODEL/REP PAIRS (by R² at sigma=0)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Model':<15} {'Rep':<20} {'Transform':<15} {'R²':>8} {'RMSE':>8}")
    print(f"{'-'*80}")
    
    for idx, row in summary.head(n).iterrows():
        model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
        print(f"{summary.index.get_loc(idx)+1:<6} {model_label:<15} {row['rep']:<20} {row['transformation_type']:<15} {row['r2']:>8.4f} {row['rmse']:>8.4f}")
    
    return summary.head(n)

def analyze_noise_robustness(df):
    """Analyze which pairs are most robust to noise"""
    if df.empty:
        return pd.DataFrame()
    
    # Calculate performance degradation
    summary = df.groupby(['base_model', 'rep', 'transformation_type', 'sigma']).agg({
        'rmse': 'mean',
        'r2': 'mean'
    }).reset_index()
    
    # Calculate AUC for R² vs noise
    auc_scores = []
    for (model, rep, trans), group in summary.groupby(['base_model', 'rep', 'transformation_type']):
        group = group.sort_values('sigma')
        if len(group) >= 3:  # Need at least 3 points
            auc = np.trapz(group['r2'], group['sigma'])
            auc_scores.append({
                'base_model': model,
                'rep': rep,
                'transformation_type': trans,
                'auc_r2': auc,
                'baseline_r2': group[group['sigma'] == 0.0]['r2'].values[0] if any(group['sigma'] == 0.0) else np.nan
            })
    
    auc_df = pd.DataFrame(auc_scores).sort_values('auc_r2', ascending=False)
    
    print(f"\n{'='*80}")
    print("MOST ROBUST PAIRS (by R² AUC across noise levels)")
    print(f"{'='*80}")
    print(f"{'Rank':<6} {'Model':<15} {'Rep':<20} {'Transform':<15} {'AUC':>8} {'R²@σ=0':>8}")
    print(f"{'-'*80}")
    
    for idx, row in auc_df.head(10).iterrows():
        model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
        print(f"{auc_df.index.get_loc(idx)+1:<6} {model_label:<15} {row['rep']:<20} {row['transformation_type']:<15} {row['auc_r2']:>8.4f} {row['baseline_r2']:>8.4f}")
    
    return auc_df

def identify_bayesian_pairs(df):
    """Identify best deterministic/Bayesian pairs for comparison"""
    print(f"\n{'='*80}")
    print("DETERMINISTIC vs BAYESIAN PAIRS FOR HIGHLIGHTING")
    print(f"{'='*80}")
    
    pairs_to_compare = []
    
    # Tree-based: RF vs QRF
    for rep in df['rep'].unique():
        rf_data = df[(df['base_model'] == 'rf') & (df['rep'] == rep) & (df['sigma'] == 0.0)]
        qrf_data = df[(df['base_model'] == 'qrf') & (df['rep'] == rep) & (df['sigma'] == 0.0)]
        
        if not rf_data.empty and not qrf_data.empty:
            rf_r2 = rf_data['r2'].mean()
            qrf_r2 = qrf_data['r2'].mean()
            pairs_to_compare.append({
                'pair_type': 'tree',
                'deterministic': 'RF',
                'bayesian': 'QRF',
                'rep': rep,
                'det_r2': rf_r2,
                'bay_r2': qrf_r2,
                'improvement': qrf_r2 - rf_r2
            })
    
    # Neural networks: baseline vs transformations
    for rep in ['ecfp4', 'pdv']:
        for model in ['dnn', 'mlp']:
            baseline_data = df[(df['base_model'] == model) & (df['transformation_type'] == 'baseline') & 
                             (df['rep'] == rep) & (df['sigma'] == 0.0)]
            
            for trans in ['full', 'last_layer', 'variational']:
                trans_data = df[(df['base_model'] == model) & (df['transformation_type'] == trans) & 
                              (df['rep'] == rep) & (df['sigma'] == 0.0)]
                
                if not baseline_data.empty and not trans_data.empty:
                    base_r2 = baseline_data['r2'].mean()
                    trans_r2 = trans_data['r2'].mean()
                    pairs_to_compare.append({
                        'pair_type': 'neural',
                        'deterministic': f'{model}/baseline',
                        'bayesian': f'{model}/{trans}',
                        'rep': rep,
                        'det_r2': base_r2,
                        'bay_r2': trans_r2,
                        'improvement': trans_r2 - base_r2
                    })
    
    pairs_df = pd.DataFrame(pairs_to_compare).sort_values('improvement', ascending=False)
    
    print(f"{'Type':<8} {'Deterministic':<20} {'Bayesian':<20} {'Rep':<10} {'Det R²':>8} {'Bay R²':>8} {'Δ':>8}")
    print(f"{'-'*80}")
    
    for _, row in pairs_df.head(15).iterrows():
        print(f"{row['pair_type']:<8} {row['deterministic']:<20} {row['bayesian']:<20} {row['rep']:<10} "
              f"{row['det_r2']:>8.4f} {row['bay_r2']:>8.4f} {row['improvement']:>+8.4f}")
    
    return pairs_df

def identify_conformal_pairs(df):
    """Identify best conformal prediction pairs"""
    print(f"\n{'='*80}")
    print("CONFORMAL PREDICTION PERFORMANCE")
    print(f"{'='*80}")
    
    cp_data = df[df['transformation_type'] == 'conformal'].copy()
    
    if cp_data.empty:
        print("No conformal prediction data found")
        return pd.DataFrame()
    
    summary = cp_data[cp_data['sigma'] == 0.0].groupby(['base_model', 'rep']).agg({
        'rmse': 'mean',
        'r2': 'mean'
    }).reset_index().sort_values('r2', ascending=False)
    
    print(f"{'Base Model':<15} {'Rep':<20} {'R²':>8} {'RMSE':>8}")
    print(f"{'-'*80}")
    
    for _, row in summary.head(10).iterrows():
        print(f"{row['base_model']:<15} {row['rep']:<20} {row['r2']:>8.4f} {row['rmse']:>8.4f}")
    
    return summary

def generate_recommendations(top_pairs, robust_pairs, bayesian_pairs):
    """Generate recommendations for figure highlights"""
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR FIGURE HIGHLIGHTS")
    print(f"{'='*80}")
    
    print("\nFig 3b & 3d - Uncertainty Analysis:")
    print("  Recommended pairs (select 3-5):")
    if not robust_pairs.empty:
        for idx, row in robust_pairs.head(5).iterrows():
            model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
            print(f"    • {model_label} + {row['rep'].upper()}")
    
    print("\nFig 4a - Different Noise Types:")
    print("  Recommended pairs (select 2-3):")
    if not top_pairs.empty:
        for idx, row in top_pairs.head(3).iterrows():
            model_label = f"{row['base_model']}/{row['transformation_type']}" if row['transformation_type'] != 'baseline' else row['base_model']
            print(f"    • {model_label} + {row['rep'].upper()}")
    
    print("\nFig 5b - Data Efficiency:")
    print("  Recommended comparison pairs:")
    if not bayesian_pairs.empty:
        tree_pairs = bayesian_pairs[bayesian_pairs['pair_type'] == 'tree'].head(2)
        for _, row in tree_pairs.iterrows():
            print(f"    • {row['deterministic']} vs {row['bayesian']} ({row['rep'].upper()})")
        
        nn_pairs = bayesian_pairs[bayesian_pairs['pair_type'] == 'neural'].head(2)
        for _, row in nn_pairs.iterrows():
            print(f"    • {row['deterministic']} vs {row['bayesian']} ({row['rep'].upper()})")

def main():
    import sys
    
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'results_mini'
    
    print(f"\n{'='*80}")
    print("MINI DATA ANALYSIS - IDENTIFYING TOP PAIRS")
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
    
    # Analysis
    top_pairs = identify_top_baseline_pairs(df, n=10)
    robust_pairs = analyze_noise_robustness(df)
    bayesian_pairs = identify_bayesian_pairs(df)
    cp_pairs = identify_conformal_pairs(df)
    
    # Generate recommendations
    generate_recommendations(top_pairs, robust_pairs, bayesian_pairs)
    
    # Save results
    output_dir = Path('analysis_output')
    output_dir.mkdir(exist_ok=True)
    
    top_pairs.to_csv(output_dir / 'top_baseline_pairs.csv', index=False)
    robust_pairs.to_csv(output_dir / 'robust_pairs.csv', index=False)
    bayesian_pairs.to_csv(output_dir / 'bayesian_comparison_pairs.csv', index=False)
    if not cp_pairs.empty:
        cp_pairs.to_csv(output_dir / 'conformal_pairs.csv', index=False)
    
    print(f"\n✓ Results saved to {output_dir}/")
    print(f"\nNext steps:")
    print(f"  1. Review recommendations above")
    print(f"  2. Select specific pairs for each figure")
    print(f"  3. Update plotting script with selected pairs")
    print(f"  4. Update full data generation script")

if __name__ == '__main__':
    main()
