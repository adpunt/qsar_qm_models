#!/usr/bin/env python3
"""
Consolidate hyperparameters after tuning experiments
- Performs statistical significance tests
- Consolidates best hyperparameters into master config
- Creates decisions file for which models to use tuned params
"""

import pandas as pd
import numpy as np
import json
import glob
import os
from scipy import stats
from pathlib import Path

def load_experiments(default_file, tuned_file):
    """Load default and tuned experiment results"""
    try:
        df_default = pd.read_csv(default_file)
        df_tuned = pd.read_csv(tuned_file)
        return df_default, df_tuned
    except FileNotFoundError:
        return None, None

def perform_statistical_test(df_default, df_tuned, metric='r2', alpha=0.05):
    """
    Perform paired t-test on results
    Returns: (is_significant, p_value, mean_improvement, effect_size)
    """
    # Group by model and rep to get paired samples
    default_vals = df_default.groupby(['model', 'rep'])[metric].mean()
    tuned_vals = df_tuned.groupby(['model', 'rep'])[metric].mean()
    
    # Align indices
    common_idx = default_vals.index.intersection(tuned_vals.index)
    if len(common_idx) < 3:
        return False, 1.0, 0.0, 0.0
    
    default_aligned = default_vals.loc[common_idx].values
    tuned_aligned = tuned_vals.loc[common_idx].values
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(tuned_aligned, default_aligned)
    
    # Effect size (Cohen's d)
    diff = tuned_aligned - default_aligned
    effect_size = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
    
    # Mean improvement
    mean_improvement = np.mean(tuned_aligned - default_aligned)
    
    # Significant if p < alpha AND improvement is positive
    is_significant = (p_value < alpha) and (mean_improvement > 0)
    
    return is_significant, p_value, mean_improvement, effect_size

def consolidate_hyperparameters(results_dir='results'):
    """
    Main function to consolidate hyperparameters after Section 0
    """
    print("="*70)
    print("HYPERPARAMETER TUNING ANALYSIS")
    print("="*70)
    print()
    
    # Find all fig6a files
    default_pattern = os.path.join(results_dir, 'fig6a_*_default.csv')
    default_files = glob.glob(default_pattern)
    
    if not default_files:
        print("ERROR: No default files found. Make sure Section 0 has completed.")
        return None
    
    master_params = {}
    decisions = {}
    summary = []
    
    for default_file in sorted(default_files):
        # Get corresponding tuned file
        base = os.path.basename(default_file).replace('_default.csv', '')
        tuned_file = os.path.join(results_dir, f"{base}_tuned.csv")
        json_file = os.path.join(results_dir, f"{base}_tuned.json")
        
        if not os.path.exists(tuned_file):
            print(f"⚠️  Missing tuned file for {base}")
            continue
        
        # Load experiments
        df_default, df_tuned = load_experiments(default_file, tuned_file)
        if df_default is None or df_tuned is None:
            continue
        
        # Get model name
        model_name = base.replace('fig6a_', '')
        
        # Statistical test
        is_sig, p_val, improvement, effect_size = perform_statistical_test(
            df_default, df_tuned
        )
        
        # Load hyperparameters if significant
        if is_sig and os.path.exists(json_file):
            with open(json_file, 'r') as f:
                model_params = json.load(f)
            
            # Add to master params
            for model_type, reps in model_params.items():
                if model_type not in master_params:
                    master_params[model_type] = {}
                master_params[model_type].update(reps)
            
            decision = "USE_TUNED"
            symbol = "✓"
        else:
            decision = "USE_DEFAULT"
            symbol = "✗"
        
        # Store decision
        # Extract model types from the dataframe
        model_types = df_default['model'].unique()
        for mt in model_types:
            decisions[mt] = decision
        
        # Summary
        default_r2 = df_default['r2'].mean()
        tuned_r2 = df_tuned['r2'].mean()
        
        summary.append({
            'model': model_name,
            'default_r2': default_r2,
            'tuned_r2': tuned_r2,
            'improvement': improvement,
            'p_value': p_val,
            'effect_size': effect_size,
            'significant': is_sig,
            'decision': decision
        })
        
        print(f"{symbol} {model_name:15s} | Default: {default_r2:7.4f} | Tuned: {tuned_r2:7.4f} | "
              f"Δ: {improvement:+7.4f} | p={p_val:.4f} | {decision}")
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    # Count decisions
    use_tuned = sum(1 for d in decisions.values() if d == "USE_TUNED")
    use_default = sum(1 for d in decisions.values() if d == "USE_DEFAULT")
    
    print(f"Models using TUNED parameters: {use_tuned}")
    print(f"Models using DEFAULT parameters: {use_default}")
    print()
    
    # Save master hyperparameters
    master_file = os.path.join(results_dir, 'master_tuned_hyperparameters.json')
    with open(master_file, 'w') as f:
        json.dump(master_params, f, indent=4)
    print(f"✓ Saved master hyperparameters to: {master_file}")
    
    # Save decisions
    decisions_file = os.path.join(results_dir, 'hyperparameter_decisions.json')
    with open(decisions_file, 'w') as f:
        json.dump(decisions, f, indent=4)
    print(f"✓ Saved decisions to: {decisions_file}")
    
    # Save summary
    summary_df = pd.DataFrame(summary)
    summary_file = os.path.join(results_dir, 'hyperparameter_tuning_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"✓ Saved summary to: {summary_file}")
    
    print()
    print("="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. Review the decisions in hyperparameter_decisions.json")
    print("2. Run Sections 1-5 with --use-best-params flag")
    print("3. The training code will automatically use tuned params where appropriate")
    print()
    
    return master_params, decisions, summary_df

if __name__ == "__main__":
    consolidate_hyperparameters()