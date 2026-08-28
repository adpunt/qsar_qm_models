#!/usr/bin/env python3
"""
Generate experimental summary from results files
This addresses the user's request for an experimental summary
"""

import pandas as pd
import glob
import os
from collections import defaultdict
import numpy as np

def load_all_results(results_dir='results'):
    """Load all CSV files from results directory"""
    pattern = os.path.join(results_dir, '*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No results files found in {results_dir}")
        return None
    
    all_data = []
    for file in files:
        try:
            df = pd.read_csv(file)
            df['source_file'] = os.path.basename(file)
            all_data.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    if not all_data:
        return None
    
    return pd.concat(all_data, ignore_index=True)

def parse_model_info(model_name):
    """Parse model name to extract transformation type"""
    if '_full' in model_name:
        return 'full', model_name.replace('_full', '')
    elif '_last' in model_name:
        return 'last_layer', model_name.replace('_last', '')
    elif '_variational' in model_name:
        return 'variational', model_name.replace('_variational', '')
    elif model_name.startswith('conformal'):
        return 'conformal', model_name.replace('conformal_', '')
    else:
        return 'baseline', model_name

def generate_experimental_summary(df):
    """Generate comprehensive experimental summary"""
    
    print("="*80)
    print("EXPERIMENTAL SUMMARY")
    print("="*80)
    print()
    
    # Basic statistics
    print("OVERALL STATISTICS")
    print("-"*80)
    print(f"Total experiments run: {len(df)}")
    print(f"Unique models: {df['model'].nunique()}")
    print(f"Unique representations: {df['rep'].nunique()}")
    print(f"Noise levels tested: {sorted(df['sigma'].unique())}")
    print(f"Sample sizes: {sorted(df['sample_size'].unique())}")
    print(f"Number of source files: {df['source_file'].nunique()}")
    print()
    
    # Parse model information
    df[['transformation_type', 'base_model']] = df['model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    # Models tested
    print("MODELS TESTED")
    print("-"*80)
    for trans_type in sorted(df['transformation_type'].unique()):
        models = df[df['transformation_type'] == trans_type]['base_model'].unique()
        print(f"{trans_type.capitalize()}: {', '.join(sorted(models))}")
    print()
    
    # Representations tested
    print("REPRESENTATIONS TESTED")
    print("-"*80)
    rep_counts = df.groupby('rep').size().sort_values(ascending=False)
    for rep, count in rep_counts.items():
        print(f"  {rep}: {count} experiments")
    print()
    
    # Performance summary
    print("PERFORMANCE SUMMARY (R² scores)")
    print("-"*80)
    
    # Best performing models overall
    print("\nTop 10 Best Performing Configurations (highest R²):")
    best_configs = df.nlargest(10, 'r2')[['model', 'rep', 'sigma', 'sample_size', 'r2']]
    print(best_configs.to_string(index=False))
    
    # Worst performing (potential issues)
    print("\nTop 10 Worst Performing Configurations (lowest R²):")
    worst_configs = df.nsmallest(10, 'r2')[['model', 'rep', 'sigma', 'sample_size', 'r2', 'source_file']]
    print(worst_configs.to_string(index=False))
    
    # Check for suspicious values
    suspicious = df[df['r2'] < -10]
    if len(suspicious) > 0:
        print(f"\n⚠️  WARNING: {len(suspicious)} experiments with R² < -10 (suspicious values)")
        print("Files affected:")
        for file in suspicious['source_file'].unique():
            count = len(suspicious[suspicious['source_file'] == file])
            print(f"  - {file}: {count} suspicious values")
    
    print()
    
    # Model comparison by transformation type
    print("AVERAGE R² BY TRANSFORMATION TYPE")
    print("-"*80)
    trans_summary = df.groupby('transformation_type').agg({
        'r2': ['mean', 'std', 'min', 'max', 'count']
    }).round(4)
    print(trans_summary)
    print()
    
    # Noise robustness
    print("NOISE ROBUSTNESS (Average R² by noise level)")
    print("-"*80)
    noise_summary = df.groupby(['transformation_type', 'sigma'])['r2'].mean().unstack(fill_value=np.nan)
    print(noise_summary.round(4))
    print()
    
    # Data size analysis
    print("DATA SIZE SCALING (Average R² by sample size)")
    print("-"*80)
    size_summary = df.groupby(['transformation_type', 'sample_size'])['r2'].mean().unstack(fill_value=np.nan)
    print(size_summary.round(4))
    print()
    
    # Representation performance
    print("REPRESENTATION PERFORMANCE (Average R² by representation)")
    print("-"*80)
    rep_summary = df.groupby('rep')['r2'].agg(['mean', 'std', 'count']).round(4)
    rep_summary = rep_summary.sort_values('mean', ascending=False)
    print(rep_summary)
    print()
    
    # Coverage analysis
    print("EXPERIMENTAL COVERAGE")
    print("-"*80)
    print("Checking which model/rep/transformation combinations have been tested...")
    print()
    
    # Check Figure 1c coverage
    fig1c_models = ['dnn', 'mlp', 'gin', 'gcn']
    fig1c_reps = ['ecfp4', 'pdv', 'graph']
    fig1c_trans = ['baseline', 'full', 'last_layer', 'variational']
    
    print("Figure 1c Coverage (NN Bayesian Transformations):")
    for model in fig1c_models:
        for rep in fig1c_reps:
            coverage = []
            for trans in fig1c_trans:
                has_data = len(df[(df['base_model'] == model) & 
                                  (df['rep'] == rep) & 
                                  (df['transformation_type'] == trans)]) > 0
                coverage.append('✓' if has_data else '✗')
            coverage_str = ' '.join(coverage)
            status = '✓ COMPLETE' if '✗' not in coverage else '⚠️  INCOMPLETE'
            print(f"  {model:15s} x {rep:20s}: {coverage_str}  {status}")
    print()
    
    # Check Figure 5b coverage
    fig5b_sizes = [50, 100, 200, 500]
    fig5b_trans = ['baseline', 'full', 'last_layer', 'variational', 'conformal']
    
    print("Figure 5b Coverage (Data Size Experiments):")
    for trans in fig5b_trans:
        coverage = []
        for size in fig5b_sizes:
            has_data = len(df[(df['transformation_type'] == trans) & 
                              (df['sample_size'] == size)]) > 0
            coverage.append('✓' if has_data else '✗')
        coverage_str = ' '.join(coverage)
        status = '✓ COMPLETE' if '✗' not in coverage else '⚠️  INCOMPLETE'
        print(f"  {trans:15s}: {coverage_str}  {status}")
    print()
    
    # File-by-file breakdown
    print("FILE-BY-FILE BREAKDOWN")
    print("-"*80)
    file_summary = df.groupby('source_file').agg({
        'model': 'nunique',
        'rep': 'nunique',
        'sigma': 'nunique',
        'r2': ['count', 'mean', 'min', 'max']
    }).round(4)
    file_summary.columns = ['n_models', 'n_reps', 'n_sigmas', 'n_experiments', 'mean_r2', 'min_r2', 'max_r2']
    
    # Flag suspicious files
    file_summary['status'] = '✓'
    file_summary.loc[file_summary['min_r2'] < -10, 'status'] = '⚠️'
    
    print(file_summary.sort_values('n_experiments', ascending=False).to_string())
    print()
    
    print("="*80)
    print("SUMMARY GENERATION COMPLETE")
    print("="*80)
    
    # Save to file
    output_file = 'experimental_summary.txt'
    with open(output_file, 'w') as f:
        # Redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        generate_experimental_summary(df)  # Recursive call writes to file
        sys.stdout = old_stdout
    
    return df

if __name__ == "__main__":
    print("Loading results from results/ directory...")
    df = load_all_results('results')
    
    if df is None:
        print("No data to summarize!")
        exit(1)
    
    summary_df = generate_experimental_summary(df)
    
    print()
    print("Summary has been saved to: experimental_summary.txt")
    print()
    print("Next steps:")
    print("1. Review the coverage analysis to identify missing experiments")
    print("2. Check for suspicious R² values (< -10)")
    print("3. Compare transformation types to assess Bayesian performance")
    print("4. Use this information to prioritize which experiments to re-run")