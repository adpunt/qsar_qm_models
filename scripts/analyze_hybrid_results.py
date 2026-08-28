#!/usr/bin/env python3
"""
Analyze Hybrid Representation Experiment Results

Usage:
    python analyze_hybrid_results.py results_hybrid_experiments_<timestamp>/
    
Or:
    python analyze_hybrid_results.py .  # if running from results directory
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import os
import sys
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_all_results(result_dir):
    """Load and merge all experiment CSV files"""
    print("Loading results...")
    
    all_files = glob(os.path.join(result_dir, "exp*.csv"))
    
    if not all_files:
        print(f"ERROR: No exp*.csv files found in {result_dir}")
        sys.exit(1)
    
    print(f"Found {len(all_files)} experiment files")
    
    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            exp_name = os.path.basename(f).replace('.csv', '')
            df['experiment'] = exp_name
            dfs.append(df)
            print(f"  ✓ Loaded {exp_name}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ Failed to load {f}: {e}")
    
    if not dfs:
        print("ERROR: No valid CSV files loaded")
        sys.exit(1)
    
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Save merged results
    merged_path = os.path.join(result_dir, "all_results_merged.csv")
    df_all.to_csv(merged_path, index=False)
    print(f"\n✓ Merged results saved to: {merged_path}")
    print(f"  Total rows: {len(df_all)}")
    print(f"  Representations: {sorted(df_all['rep'].unique())}")
    print(f"  Models: {sorted(df_all['model'].unique())}")
    print(f"  Sigmas: {sorted(df_all['sigma'].unique())}")
    
    return df_all


def analyze_overall_performance(df_all):
    """Q1: Does hybrid beat individual reps?"""
    print("\n" + "="*70)
    print("OVERALL PERFORMANCE ANALYSIS")
    print("="*70)
    
    perf_by_rep = df_all.groupby('rep').agg({
        'r2': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(4)
    
    print("\nPerformance by Representation:")
    print(perf_by_rep.sort_values(('r2', 'mean'), ascending=False))
    
    # Calculate improvement
    if 'hybrid' in perf_by_rep.index:
        best_individual = perf_by_rep.loc[perf_by_rep.index != 'hybrid', ('r2', 'mean')].max()
        hybrid_perf = perf_by_rep.loc['hybrid', ('r2', 'mean')]
        improvement = ((hybrid_perf - best_individual) / best_individual) * 100
        
        print(f"\n✓ Hybrid R²: {hybrid_perf:.4f}")
        print(f"✓ Best individual R²: {best_individual:.4f}")
        print(f"✓ Improvement: {improvement:.2f}%")
        
        return {
            'hybrid_r2': hybrid_perf,
            'best_individual_r2': best_individual,
            'improvement_pct': improvement,
            'perf_by_rep': perf_by_rep
        }
    else:
        print("\n⚠ WARNING: No 'hybrid' representation found in results")
        return {'perf_by_rep': perf_by_rep}


def analyze_importance_methods(df_all):
    """Q2: Which importance method is best?"""
    print("\n" + "="*70)
    print("IMPORTANCE METHOD COMPARISON (Experiment 1)")
    print("="*70)
    
    exp1 = df_all[df_all['experiment'].str.contains('exp1_importance', na=False)]
    
    if len(exp1) == 0:
        print("No Experiment 1 results found")
        return None
    
    # Extract method from experiment name
    exp1['importance_method'] = exp1['experiment'].str.extract(r'importance_(\w+)')
    
    importance_comparison = exp1[exp1['rep'] == 'hybrid'].groupby('importance_method').agg({
        'r2': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(4)
    
    print("\nImportance Method Performance:")
    print(importance_comparison.sort_values(('r2', 'mean'), ascending=False))
    
    best_method = importance_comparison[('r2', 'mean')].idxmax()
    print(f"\n✓ Best importance method: {best_method}")
    
    return {
        'comparison': importance_comparison,
        'best_method': best_method
    }


def analyze_feature_counts(df_all):
    """Q3: Optimal number of features?"""
    print("\n" + "="*70)
    print("FEATURES PER REP COMPARISON (Experiment 2)")
    print("="*70)
    
    exp2 = df_all[df_all['experiment'].str.contains('exp2_nfeatures', na=False)]
    
    if len(exp2) == 0:
        print("No Experiment 2 results found")
        return None
    
    # Extract n_features from experiment name
    exp2['n_features'] = exp2['experiment'].str.extract(r'nfeatures_(\d+)').astype(int)
    
    features_comparison = exp2[exp2['rep'] == 'hybrid'].groupby('n_features').agg({
        'r2': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(4)
    
    print("\nFeatures Per Rep Performance:")
    print(features_comparison.sort_values(('r2', 'mean'), ascending=False))
    
    best_n = features_comparison[('r2', 'mean')].idxmax()
    print(f"\n✓ Best features per rep: {best_n}")
    
    return {
        'comparison': features_comparison,
        'best_n': best_n
    }


def analyze_rep_combinations(df_all):
    """Q4: Which rep combination is best?"""
    print("\n" + "="*70)
    print("REP COMBINATION COMPARISON (Experiment 3)")
    print("="*70)
    
    exp3 = df_all[df_all['experiment'].str.contains('exp3_combo', na=False)]
    
    if len(exp3) == 0:
        print("No Experiment 3 results found")
        return None
    
    combo_comparison = exp3[exp3['rep'] == 'hybrid'].groupby('experiment').agg({
        'r2': ['mean', 'std', 'count'],
        'rmse': ['mean', 'std']
    }).round(4)
    
    print("\nRep Combination Performance:")
    print(combo_comparison.sort_values(('r2', 'mean'), ascending=False))
    
    best_combo = combo_comparison[('r2', 'mean')].idxmax()
    print(f"\n✓ Best combination: {best_combo}")
    
    return {
        'comparison': combo_comparison,
        'best_combo': best_combo
    }


def analyze_noise_robustness(df_all):
    """Q5: Noise robustness - R² retention"""
    print("\n" + "="*70)
    print("NOISE ROBUSTNESS ANALYSIS")
    print("="*70)
    
    noise_analysis = df_all.groupby(['rep', 'sigma'])['r2'].mean().reset_index()
    noise_pivot = noise_analysis.pivot(index='rep', columns='sigma', values='r2')
    
    # Calculate retention at each sigma level
    if 0.0 in noise_pivot.columns:
        for sigma in noise_pivot.columns:
            if sigma > 0:
                noise_pivot[f'retention_{sigma}'] = (noise_pivot[sigma] / noise_pivot[0.0]) * 100
        
        retention_cols = [col for col in noise_pivot.columns if 'retention' in str(col)]
        if retention_cols:
            print("\nR² Retention (%):")
            print(noise_pivot[retention_cols].round(2))
            
            if 'hybrid' in noise_pivot.index:
                print("\nHybrid retention:")
                for col in retention_cols:
                    sigma_val = col.replace('retention_', '')
                    ret_val = noise_pivot.loc['hybrid', col]
                    print(f"  σ={sigma_val}: {ret_val:.2f}%")
    
    return noise_pivot


def analyze_data_size_effects(df_all):
    """Q6: Data size effects (mol2vec issue)"""
    print("\n" + "="*70)
    print("DATA SIZE EFFECTS (Experiment 5)")
    print("="*70)
    
    exp5 = df_all[df_all['experiment'].str.contains('exp5_size', na=False)]
    
    if len(exp5) == 0:
        print("No Experiment 5 results found")
        return None
    
    # Extract size and mol2vec status
    exp5['data_size'] = exp5['experiment'].str.extract(r'size_(\d+)').astype(int)
    exp5['has_mol2vec'] = exp5['experiment'].str.contains('with_mol2vec')
    
    # Compare hybrid performance with/without mol2vec at each size
    size_comparison = exp5[exp5['rep'] == 'hybrid'].groupby(
        ['data_size', 'has_mol2vec']
    )['r2'].mean().reset_index()
    
    if len(size_comparison) > 0:
        size_pivot = size_comparison.pivot(
            index='data_size', 
            columns='has_mol2vec', 
            values='r2'
        )
        
        if True in size_pivot.columns and False in size_pivot.columns:
            size_pivot.columns = ['without_mol2vec', 'with_mol2vec']
            size_pivot['mol2vec_helps'] = size_pivot['with_mol2vec'] > size_pivot['without_mol2vec']
            
            print("\nData Size Effects (mol2vec):")
            print(size_pivot.round(4))
            
            return size_pivot
    
    print("Insufficient data for mol2vec comparison")
    return None


def analyze_model_improvements(df_all):
    """Q7: Which models benefit most from hybrid?"""
    print("\n" + "="*70)
    print("MODEL-SPECIFIC IMPROVEMENTS")
    print("="*70)
    
    model_improvement = []
    
    for model in df_all['model'].unique():
        model_data = df_all[df_all['model'] == model]
        
        if 'hybrid' in model_data['rep'].values:
            hybrid_perf = model_data[model_data['rep'] == 'hybrid']['r2'].mean()
            individual_data = model_data[model_data['rep'] != 'hybrid']
            
            if len(individual_data) > 0:
                best_individual = individual_data['r2'].max()
                improvement = ((hybrid_perf - best_individual) / best_individual) * 100
                
                model_improvement.append({
                    'model': model,
                    'hybrid_r2': hybrid_perf,
                    'best_individual_r2': best_individual,
                    'improvement_%': improvement
                })
    
    if model_improvement:
        model_improvement_df = pd.DataFrame(model_improvement)
        print("\nModel-Specific Improvements:")
        print(model_improvement_df.sort_values('improvement_%', ascending=False).to_string(index=False))
        
        return model_improvement_df
    
    return None


def create_visualizations(df_all, result_dir, analysis_results):
    """Create key visualizations"""
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    # Figure 1: Performance comparison
    print("\n  Creating Figure 1: Performance comparison...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Overall performance by representation
    ax = axes[0]
    perf_summary = df_all.groupby('rep')['r2'].agg(['mean', 'std']).sort_values('mean')
    
    colors = ['steelblue' if idx != 'hybrid' else 'orange' for idx in perf_summary.index]
    perf_summary.plot(kind='barh', y='mean', xerr='std', ax=ax, legend=False, color=colors)
    
    if 'hybrid' in perf_summary.index:
        best_indiv = perf_summary.loc[perf_summary.index != 'hybrid', 'mean'].max()
        ax.axvline(x=best_indiv, color='red', linestyle='--', linewidth=2, label='Best individual')
    
    ax.set_xlabel('R² Score', fontsize=12)
    ax.set_ylabel('Representation', fontsize=12)
    ax.set_title('Performance by Representation', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # Plot 2: Hybrid vs individuals across models
    ax = axes[1]
    if analysis_results.get('model_improvements') is not None:
        model_df = analysis_results['model_improvements']
        x = np.arange(len(model_df))
        width = 0.35
        
        ax.bar(x - width/2, model_df['best_individual_r2'], width, 
               label='Best Individual', alpha=0.8, color='steelblue')
        ax.bar(x + width/2, model_df['hybrid_r2'], width, 
               label='Hybrid', alpha=0.8, color='orange')
        
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(model_df['model'])
        ax.set_title('Hybrid vs Best Individual by Model', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig1_path = os.path.join(result_dir, 'fig1_performance_comparison.png')
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fig1_path}")
    
    # Figure 2: Noise robustness curves
    print("  Creating Figure 2: Noise robustness...")
    plt.figure(figsize=(10, 6))
    
    reps_to_plot = ['continuous_pdv', 'ecfp4', 'mol2vec', 'hybrid']
    colors_map = {
        'continuous_pdv': 'blue',
        'ecfp4': 'green',
        'mol2vec': 'purple',
        'hybrid': 'orange'
    }
    
    for rep in reps_to_plot:
        if rep in df_all['rep'].values:
            rep_data = df_all[df_all['rep'] == rep]
            sigma_perf = rep_data.groupby('sigma').agg({'r2': ['mean', 'std']})
            
            plt.plot(sigma_perf.index, sigma_perf[('r2', 'mean')], 
                    marker='o', linewidth=2, markersize=8,
                    label=rep, color=colors_map.get(rep, 'gray'))
            plt.fill_between(sigma_perf.index,
                            sigma_perf[('r2', 'mean')] - sigma_perf[('r2', 'std')],
                            sigma_perf[('r2', 'mean')] + sigma_perf[('r2', 'std')],
                            alpha=0.2, color=colors_map.get(rep, 'gray'))
    
    plt.xlabel('Noise Level (σ)', fontsize=12)
    plt.ylabel('R² Score', fontsize=12)
    plt.title('Noise Robustness: Performance vs Noise Level', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    fig2_path = os.path.join(result_dir, 'fig2_noise_robustness.png')
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {fig2_path}")
    
    # Figure 3: Data efficiency
    exp5 = df_all[df_all['experiment'].str.contains('exp5_size', na=False)]
    if len(exp5) > 0:
        print("  Creating Figure 3: Data efficiency...")
        plt.figure(figsize=(10, 6))
        
        exp5['data_size'] = exp5['experiment'].str.extract(r'size_(\d+)').astype(int)
        exp5['has_mol2vec'] = exp5['experiment'].str.contains('with_mol2vec')
        
        for rep in exp5['rep'].unique():
            rep_data = exp5[(exp5['rep'] == rep) & (exp5['has_mol2vec'] == True)]
            if len(rep_data) > 0:
                size_perf = rep_data.groupby('data_size')['r2'].mean()
                plt.plot(size_perf.index, size_perf.values, 
                        marker='o', linewidth=2, markersize=8, label=rep)
        
        plt.xlabel('Training Set Size', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.title('Data Efficiency: Performance vs Training Size', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        
        fig3_path = os.path.join(result_dir, 'fig3_data_efficiency.png')
        plt.savefig(fig3_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {fig3_path}")


def generate_summary_report(result_dir, analysis_results):
    """Generate text summary report"""
    print("\n" + "="*70)
    print("GENERATING SUMMARY REPORT")
    print("="*70)
    
    summary_path = os.path.join(result_dir, 'ANALYSIS_SUMMARY.txt')
    
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("HYBRID REPRESENTATION EXPERIMENTS - SUMMARY REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write("-"*70 + "\n\n")
        
        # Finding 1: Overall improvement
        if 'overall' in analysis_results:
            overall = analysis_results['overall']
            f.write(f"1. OVERALL PERFORMANCE\n")
            if 'hybrid_r2' in overall:
                f.write(f"   Hybrid R²: {overall['hybrid_r2']:.4f}\n")
                f.write(f"   Best individual R²: {overall['best_individual_r2']:.4f}\n")
                f.write(f"   Improvement: {overall['improvement_pct']:.2f}%\n\n")
        
        # Finding 2: Best methods
        f.write(f"2. BEST CONFIGURATIONS\n")
        if analysis_results.get('importance'):
            f.write(f"   Best importance method: {analysis_results['importance']['best_method']}\n")
        if analysis_results.get('features'):
            f.write(f"   Best features per rep: {analysis_results['features']['best_n']}\n")
        if analysis_results.get('combinations'):
            f.write(f"   Best rep combination: {analysis_results['combinations']['best_combo']}\n")
        f.write("\n")
        
        # Finding 3: Noise robustness
        if 'noise' in analysis_results and analysis_results['noise'] is not None:
            f.write(f"3. NOISE ROBUSTNESS\n")
            noise_pivot = analysis_results['noise']
            if 'hybrid' in noise_pivot.index:
                for col in noise_pivot.columns:
                    if 'retention' in str(col):
                        sigma_val = str(col).replace('retention_', '')
                        ret_val = noise_pivot.loc['hybrid', col]
                        f.write(f"   Hybrid retention at σ={sigma_val}: {ret_val:.2f}%\n")
            f.write("\n")
        
        # Finding 4: Model improvements
        if analysis_results.get('model_improvements') is not None:
            f.write(f"4. MODEL-SPECIFIC IMPROVEMENTS\n")
            for _, row in analysis_results['model_improvements'].iterrows():
                f.write(f"   {row['model']}: {row['improvement_%']:.2f}%\n")
            f.write("\n")
        
        # Finding 5: Data size effects
        if analysis_results.get('data_size') is not None:
            f.write(f"5. DATA SIZE EFFECTS (mol2vec)\n")
            size_df = analysis_results['data_size']
            for size in size_df.index:
                if 'mol2vec_helps' in size_df.columns:
                    helps = size_df.loc[size, 'mol2vec_helps']
                    f.write(f"   N={size}: mol2vec {'helps' if helps else 'hurts'}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("GENERATED FILES:\n")
        f.write("  - all_results_merged.csv\n")
        f.write("  - fig1_performance_comparison.png\n")
        f.write("  - fig2_noise_robustness.png\n")
        f.write("  - fig3_data_efficiency.png (if applicable)\n")
        f.write("  - ANALYSIS_SUMMARY.txt (this file)\n")
        f.write("="*70 + "\n")
    
    print(f"✓ Summary report saved to: {summary_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_hybrid_results.py <results_directory>")
        print("Example: python analyze_hybrid_results.py results_hybrid_experiments_20241211_120000/")
        sys.exit(1)
    
    result_dir = sys.argv[1]
    
    if not os.path.exists(result_dir):
        print(f"ERROR: Directory not found: {result_dir}")
        sys.exit(1)
    
    print("\n" + "="*70)
    print("HYBRID REPRESENTATION RESULTS ANALYSIS")
    print("="*70)
    print(f"Results directory: {result_dir}\n")
    
    # Load data
    df_all = load_all_results(result_dir)
    
    # Run all analyses
    analysis_results = {}
    
    analysis_results['overall'] = analyze_overall_performance(df_all)
    analysis_results['importance'] = analyze_importance_methods(df_all)
    analysis_results['features'] = analyze_feature_counts(df_all)
    analysis_results['combinations'] = analyze_rep_combinations(df_all)
    analysis_results['noise'] = analyze_noise_robustness(df_all)
    analysis_results['data_size'] = analyze_data_size_effects(df_all)
    analysis_results['model_improvements'] = analyze_model_improvements(df_all)
    
    # Create visualizations
    create_visualizations(df_all, result_dir, analysis_results)
    
    # Generate summary report
    generate_summary_report(result_dir, analysis_results)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print(f"\nAll outputs saved to: {result_dir}")
    print("\nKey files:")
    print("  - all_results_merged.csv")
    print("  - ANALYSIS_SUMMARY.txt")
    print("  - fig1_performance_comparison.png")
    print("  - fig2_noise_robustness.png")
    print("  - fig3_data_efficiency.png")
    print("\n")


if __name__ == "__main__":
    main()