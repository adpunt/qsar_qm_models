import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def check_available_files(results_dir):
    """Check which phase0a tuning files exist"""
    available = []
    missing = []
    
    # All expected combinations from bash script
    expected = []
    
    # Tree models
    for model in ['rf', 'xgboost', 'svm', 'qrf', 'ngboost']:
        for rep in ['ecfp4', 'pdv', 'sns']:
            expected.append((model, rep))
    
    # GP models
    for rep in ['ecfp4', 'pdv', 'sns']:
        expected.append(('gauche', rep))
    
    # Neural models
    for model in ['dnn', 'flexible_dnn', 'mlp', 'residual_mlp', 'factorization_mlp', 'mtl']:
        for rep in ['ecfp4', 'pdv', 'sns']:
            expected.append((model, rep))
    
    # Sequence models
    for model in ['rnn', 'gru']:
        for rep in ['smiles', 'randomized_smiles']:
            expected.append((model, rep))
    
    # Graph models
    for model in ['gin', 'gcn', 'graph_gp']:
        expected.append((model, 'graph'))
    
    # Conformal variants
    for base in ['rf', 'qrf', 'xgboost', 'dnn', 'gauche']:
        for rep in ['ecfp4', 'pdv']:
            expected.append((f'conformal_{base}', rep))
    
    for base in ['gin', 'gcn']:
        expected.append((f'conformal_{base}', 'graph'))
    
    # Check which exist
    for model, rep in expected:
        filepath = Path(results_dir) / f"phase0a_tuning_{model}_{rep}.csv"
        if filepath.exists():
            available.append((model, rep))
        else:
            missing.append((model, rep))
    
    return available, missing

def load_results(results_dir, phase, model, rep, condition):
    """Load results CSV for a given model/rep/condition"""
    filename = f"{phase}_{condition}_{model}.csv"
    filepath = Path(results_dir) / filename
    
    if not filepath.exists():
        return None
    
    df = pd.read_csv(filepath)
    
    # Filter by representation
    if 'rep' in df.columns:
        df = df[df['rep'] == rep]
    
    if len(df) == 0:
        return None
    
    return df
def compare_hyperparameters(results_dir="../results", alpha=0.05):
    """
    Compare default vs tuned hyperparameters at sigma=0.
    Returns decisions for each model/rep pair.
    """
    
    # Model-representation combinations
    tree_models = ['rf', 'xgboost', 'svm', 'qrf', 'ngboost']
    gp_models = ['gauche']
    neural_models = ['dnn', 'flexible_dnn', 'mlp', 'residual_mlp', 'factorization_mlp', 'mtl']
    sequence_models = ['rnn', 'gru']
    graph_models = ['gin', 'gcn', 'graph_gp']
    conformal_bases = ['rf', 'qrf', 'xgboost', 'dnn', 'gauche', 'gin', 'gcn']
    
    bit_reps = ['ecfp4', 'pdv', 'sns']
    seq_reps = ['smiles', 'randomized_smiles']
    graph_rep = ['graph']
    
    decisions = {}
    comparison_results = []
    
    # Test tree models
    for model in tree_models:
        for rep in bit_reps:
            decision, result = test_pair(results_dir, model, rep, alpha)
            decisions[f"{model}_{rep}"] = decision
            if result:
                comparison_results.append(result)
    
    # Test GP models
    for model in gp_models:
        for rep in bit_reps:
            decision, result = test_pair(results_dir, model, rep, alpha)
            decisions[f"{model}_{rep}"] = decision
            if result:
                comparison_results.append(result)
    
    # Test neural models
    for model in neural_models:
        for rep in bit_reps:
            decision, result = test_pair(results_dir, model, rep, alpha)
            decisions[f"{model}_{rep}"] = decision
            if result:
                comparison_results.append(result)
    
    # Test sequence models
    for model in sequence_models:
        for rep in seq_reps:
            decision, result = test_pair(results_dir, model, rep, alpha)
            decisions[f"{model}_{rep}"] = decision
            if result:
                comparison_results.append(result)
    
    # Test graph models
    for model in graph_models:
        decision, result = test_pair(results_dir, model, 'graph', alpha)
        decisions[f"{model}_graph"] = decision
        if result:
            comparison_results.append(result)
    
    # Test conformal models
    for base in conformal_bases:
        if base in ['gin', 'gcn']:
            reps = graph_rep
        else:
            reps = ['ecfp4', 'pdv']
        
        for rep in reps:
            decision, result = test_pair(
                results_dir, 
                f"conformal_{base}", 
                rep, 
                alpha,
                is_conformal=True
            )
            decisions[f"conformal_{base}_{rep}"] = decision
            if result:
                comparison_results.append(result)
    
    # Save decisions
    save_decisions(decisions, results_dir)
    
    # Create summary report
    create_summary_report(comparison_results, decisions, results_dir)
    
    return decisions, comparison_results

def test_pair(results_dir, model, rep, alpha, is_conformal=False):
    """
    Test default vs tuned for a single model/rep pair.
    Returns: (decision, result_dict)
    """
    
    # Load data
    default_df = load_results(results_dir, "phase0b", model, rep, "default")
    tuned_df = load_results(results_dir, "phase0b", model, rep, "tuned")
    
    if default_df is None or tuned_df is None:
        print(f"Warning: Missing data for {model}/{rep}")
        return "USE_DEFAULT", None
    
    # Extract R² values
    default_r2 = default_df['r2'].values if 'r2' in default_df.columns else default_df['R2'].values
    tuned_r2 = tuned_df['r2'].values if 'r2' in tuned_df.columns else tuned_df['R2'].values
    
    # Check if arrays have same length
    if len(default_r2) != len(tuned_r2):
        print(f"Warning: Different sample sizes for {model}/{rep} (default: {len(default_r2)}, tuned: {len(tuned_r2)})")
        # Use Mann-Whitney U test instead (unpaired)
        from scipy.stats import mannwhitneyu
        statistic, p_value = mannwhitneyu(default_r2, tuned_r2, alternative='two-sided')
    else:
        # Wilcoxon signed-rank test (paired)
        statistic, p_value = stats.wilcoxon(default_r2, tuned_r2)
    
    # Calculate mean difference
    mean_default = np.mean(default_r2)
    mean_tuned = np.mean(tuned_r2)
    mean_diff = mean_tuned - mean_default
    
    # Decision logic
    if p_value < alpha:
        if mean_diff > 0:
            decision = "USE_BOTH"
        else:
            decision = "USE_DEFAULT"
    else:
        decision = "USE_DEFAULT"
    
    result = {
        'model': model,
        'representation': rep,
        'mean_default': mean_default,
        'mean_tuned': mean_tuned,
        'mean_diff': mean_diff,
        'p_value': p_value,
        'decision': decision,
        'significant': p_value < alpha
    }
    
    return decision, result

def save_decisions(decisions, results_dir):
    """Save hyperparameter decisions to JSON"""
    output_file = Path(results_dir) / "hyperparameter_decisions.json"
    with open(output_file, 'w') as f:
        json.dump(decisions, f, indent=2)
    print(f"Saved decisions to {output_file}")

def create_summary_report(comparison_results, decisions, results_dir):
    """Create summary report with visualizations"""
    
    if not comparison_results:
        print("\n=== HYPERPARAMETER TUNING SUMMARY ===")
        print("No comparison results available (all files missing)")
        return
    
    df = pd.DataFrame(comparison_results)
    
    # Save detailed CSV
    output_csv = Path(results_dir) / "hyperparameter_comparison.csv"
    df.to_csv(output_csv, index=False)
    print(f"Saved detailed comparison to {output_csv}")
    
    # Summary statistics
    print("\n=== HYPERPARAMETER TUNING SUMMARY ===")
    print(f"Total model/rep pairs tested: {len(df)}")
    print(f"Significant improvements: {df['significant'].sum()}")
    print(f"USE_BOTH decisions: {sum(1 for d in decisions.values() if d == 'USE_BOTH')}")
    print(f"USE_DEFAULT decisions: {sum(1 for d in decisions.values() if d == 'USE_DEFAULT')}")
    print(f"USE_TUNED decisions: {sum(1 for d in decisions.values() if d == 'USE_TUNED')}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean improvement by model
    ax = axes[0, 0]
    df_sorted = df.sort_values('mean_diff', ascending=True)
    colors = ['green' if x > 0 else 'red' for x in df_sorted['mean_diff']]
    ax.barh(range(len(df_sorted)), df_sorted['mean_diff'], color=colors, alpha=0.6)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels([f"{row['model']}/{row['representation']}" for _, row in df_sorted.iterrows()], fontsize=8)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Mean R² Improvement (Tuned - Default)')
    ax.set_title('Hyperparameter Tuning Impact')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. P-value distribution
    ax = axes[0, 1]
    ax.hist(df['p_value'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(0.05, color='red', linestyle='--', linewidth=2, label='α=0.05')
    ax.set_xlabel('P-value')
    ax.set_ylabel('Count')
    ax.set_title('P-value Distribution')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 3. Default vs Tuned scatter
    ax = axes[1, 0]
    ax.scatter(df['mean_default'], df['mean_tuned'], 
               c=['green' if sig else 'gray' for sig in df['significant']],
               alpha=0.6, s=100)
    lim_min = min(df['mean_default'].min(), df['mean_tuned'].min())
    lim_max = max(df['mean_default'].max(), df['mean_tuned'].max())
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
    ax.set_xlabel('Default R²')
    ax.set_ylabel('Tuned R²')
    ax.set_title('Default vs Tuned Performance')
    ax.grid(alpha=0.3)
    ax.legend(['Significant', 'Not Significant'], loc='upper left')
    
    # 4. Decision summary
    ax = axes[1, 1]
    decision_counts = pd.Series(decisions).value_counts()
    ax.pie(decision_counts.values, labels=decision_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title('Hyperparameter Decisions')
    
    plt.tight_layout()
    
    output_fig = Path(results_dir) / "hyperparameter_analysis.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_fig}")
    plt.close()
    
    # Generate text report
    output_txt = Path(results_dir) / "hyperparameter_report.txt"
    with open(output_txt, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("HYPERPARAMETER TUNING ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total pairs tested: {len(df)}\n")
        f.write(f"Significant improvements (p < 0.05): {df['significant'].sum()}\n")
        f.write(f"Mean R² improvement: {df['mean_diff'].mean():.4f}\n")
        f.write(f"Median R² improvement: {df['mean_diff'].median():.4f}\n\n")
        
        f.write("DECISION SUMMARY:\n")
        for decision, count in decision_counts.items():
            f.write(f"  {decision}: {count}\n")
        f.write("\n")
        
        f.write("TOP 10 IMPROVEMENTS:\n")
        top_10 = df.nlargest(10, 'mean_diff')
        for _, row in top_10.iterrows():
            f.write(f"  {row['model']}/{row['representation']}: "
                   f"+{row['mean_diff']:.4f} (p={row['p_value']:.4f})\n")
        f.write("\n")
        
        f.write("USE_BOTH PAIRS (test with both default and tuned):\n")
        use_both = [k for k, v in decisions.items() if v == "USE_BOTH"]
        for pair in sorted(use_both):
            f.write(f"  {pair}\n")
    
    print(f"Saved text report to {output_txt}")

if __name__ == "__main__":
    decisions, results = compare_hyperparameters(alpha=0.8)
    print("\nAnalysis complete. Check results directory for outputs.")