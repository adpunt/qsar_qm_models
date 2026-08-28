import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_screening_results(results_dir="../results"):
    """Load all Phase 0C screening results"""
    
    all_data = []
    
    # Find all screening files
    screening_files = list(Path(results_dir).glob("phase0c_screen_*.csv"))
    
    if not screening_files:
        print(f"No phase0c_screen_*.csv files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Found {len(screening_files)} screening files")
    
    for filepath in screening_files:
        try:
            df = pd.read_csv(filepath)
            df['source_file'] = filepath.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    if not all_data:
        print("No data could be loaded!")
        return pd.DataFrame()
    
    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Normalize conformal model names: conformal_rf_split -> conformal_rf
    combined_df['model'] = combined_df['model'].str.replace('_split', '', regex=False)
    
    # Filter out catastrophic failures (R² < -10 indicates training failure)
    initial_count = len(combined_df)
    combined_df = combined_df[combined_df['r2'] > -10]
    filtered_count = initial_count - len(combined_df)
    if filtered_count > 0:
        print(f"Filtered out {filtered_count} rows with catastrophic R² values (< -10)")
    
    # **NEW: Check iteration counts BEFORE deduplication**
    print("\n=== Iteration Counts per Model/Rep/Sigma ===")
    iteration_counts = combined_df.groupby(['model', 'rep', 'sigma']).size().reset_index(name='n_iterations')
    
    # Summary statistics
    print(f"\nIteration count statistics:")
    print(f"  Mean: {iteration_counts['n_iterations'].mean():.1f}")
    print(f"  Median: {iteration_counts['n_iterations'].median():.0f}")
    print(f"  Min: {iteration_counts['n_iterations'].min()}")
    print(f"  Max: {iteration_counts['n_iterations'].max()}")
    
    # Show distribution
    print(f"\nDistribution of iteration counts:")
    print(iteration_counts['n_iterations'].value_counts().sort_index())
    
    # Show problematic cases (< 10 iterations)
    incomplete = iteration_counts[iteration_counts['n_iterations'] < 10]
    if len(incomplete) > 0:
        print(f"\n⚠️  {len(incomplete)} cases with < 10 iterations:")
        print(incomplete.sort_values(['model', 'rep', 'sigma']))
    
    # Show cases with > 10 iterations (duplicates)
    duplicates = iteration_counts[iteration_counts['n_iterations'] > 10]
    if len(duplicates) > 0:
        print(f"\n⚠️  {len(duplicates)} cases with > 10 iterations (duplicates from multiple files):")
        print(duplicates.sort_values(['model', 'rep', 'sigma']))
    
    # For duplicate model/rep/sigma combinations (from multiple files), keep the best R²
    # This handles cases where we have both _default.csv and _tuned.csv
    combined_df = combined_df.sort_values('r2', ascending=False)
    combined_df = combined_df.drop_duplicates(subset=['model', 'rep', 'sigma', 'iteration'], keep='first')
    
    # Group by model/rep/sigma and calculate mean metrics across iterations
    results = combined_df.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'params_source': 'first',  # Keep track of whether params were default or tuned
        'iteration': 'count'  # **NEW: Track how many iterations contributed to the mean**
    }).reset_index()
    
    results.rename(columns={
        'rep': 'representation',
        'iteration': 'n_iterations_used'
    }, inplace=True)
    
    # Data quality check
    print("\n=== Data Quality Check (after deduplication) ===")
    for (model, rep), group in results.groupby(['model', 'representation']):
        n_sigma = len(group)
        avg_iterations = group['n_iterations_used'].mean()
        if n_sigma < 11:
            print(f"  {model}/{rep}: Only {n_sigma}/11 sigma values")
        if avg_iterations < 10:
            print(f"  {model}/{rep}: Average {avg_iterations:.1f}/10 iterations per sigma")
    
    return results

def calculate_robustness_score(df):
    """Calculate robustness as RMSE increase from sigma 0 to 1"""
    
    robustness = []
    
    for (model, rep), group in df.groupby(['model', 'representation']):
        sigma_0 = group[group['sigma'] == 0.0]['rmse'].values
        sigma_1 = group[group['sigma'] == 1.0]['rmse'].values
        
        if len(sigma_0) > 0 and len(sigma_1) > 0:
            rmse_increase = sigma_1[0] - sigma_0[0]
            percent_increase = (rmse_increase / sigma_0[0]) * 100 if sigma_0[0] > 0 else np.inf
            
            robustness.append({
                'model': model,
                'representation': rep,
                'rmse_at_0': sigma_0[0],
                'rmse_at_1': sigma_1[0],
                'rmse_increase': rmse_increase,
                'percent_increase': percent_increase
            })
    
    return pd.DataFrame(robustness)

def select_diverse_pairs(df, robustness_df, n_pairs=10):
    """
    Select diverse pairs spanning model families and representations.
    
    Criteria:
    1. Top performers by R² at sigma=0
    2. Most robust (smallest RMSE increase)
    3. Coverage across model families
    4. Coverage across representations
    """
    
    # Get sigma=0 performance
    sigma_0_df = df[df['sigma'] == 0.0].copy()
    
    if len(sigma_0_df) == 0:
        print("No sigma=0 data available!")
        return []
    
    # Merge with robustness
    sigma_0_df = sigma_0_df.merge(
        robustness_df[['model', 'representation', 'rmse_increase', 'percent_increase']],
        on=['model', 'representation'],
        how='left'
    )
    
    # Define model families
    tree_models = ['rf', 'xgboost', 'svm']
    tree_prob_models = ['qrf', 'ngboost']
    gp_models = ['gauche']
    neural_models = ['dnn', 'flexible_dnn', 'mlp', 'residual_mlp', 'factorization_mlp', 'mtl']
    graph_models = ['gin', 'gcn', 'graph_gp']
    
    def get_family(model):
        if model in tree_models:
            return 'tree'
        elif model in tree_prob_models:
            return 'tree_prob'
        elif model in gp_models:
            return 'gp'
        elif model in neural_models:
            return 'neural'
        elif model in graph_models:
            return 'graph'
        elif 'conformal' in model:
            return 'conformal'
        else:
            return 'other'
    
    sigma_0_df['family'] = sigma_0_df['model'].apply(get_family)
    
    # Remove pairs with missing robustness data (incomplete sigma range)
    sigma_0_df = sigma_0_df[sigma_0_df['rmse_increase'].notna()]
    
    if len(sigma_0_df) == 0:
        print("No complete data (sigma 0->1) available!")
        return []
    
    # Selection strategy
    selected = []
    
    # 1. Top 5 by R² at sigma=0
    top_r2 = sigma_0_df.nlargest(5, 'r2')
    selected.extend(top_r2[['model', 'representation']].to_dict('records'))
    
    # 2. Top 3 by robustness (smallest percent increase)
    valid_robust = sigma_0_df[sigma_0_df['percent_increase'] != np.inf]
    if len(valid_robust) >= 3:
        top_robust = valid_robust.nsmallest(3, 'percent_increase')
        selected.extend(top_robust[['model', 'representation']].to_dict('records'))
    
    # 3. Ensure family coverage
    represented_families = set(sigma_0_df[
        sigma_0_df.apply(lambda x: {'model': x['model'], 'representation': x['representation']} in selected, axis=1)
    ]['family'])
    
    for family in ['tree', 'tree_prob', 'gp', 'neural', 'graph', 'conformal']:
        if family not in represented_families:
            family_df = sigma_0_df[sigma_0_df['family'] == family]
            if len(family_df) > 0:
                best = family_df.nlargest(1, 'r2').iloc[0]
                selected.append({'model': best['model'], 'representation': best['representation']})
                represented_families.add(family)
    
    # 4. Ensure representation coverage (prioritize SNS, ecfp4, pdv, graph)
    represented_reps = set(s['representation'] for s in selected)
    for rep in ['sns', 'ecfp4', 'pdv', 'graph', 'smiles', 'randomized_smiles']:
        if rep not in represented_reps:
            rep_df = sigma_0_df[sigma_0_df['representation'] == rep]
            if len(rep_df) > 0:
                best = rep_df.nlargest(1, 'r2').iloc[0]
                selected.append({'model': best['model'], 'representation': best['representation']})
                represented_reps.add(rep)
    
    # Remove duplicates
    selected_unique = []
    seen = set()
    for s in selected:
        key = (s['model'], s['representation'])
        if key not in seen:
            selected_unique.append(s)
            seen.add(key)
    
    # Limit to n_pairs
    selected_final = selected_unique[:n_pairs]
    
    return selected_final

def identify_bayesian_pairs(df):
    """
    Identify neural/graph models that have both deterministic and Bayesian variants,
    so we can compare them directly.
    
    Returns pairs where we can make meaningful comparisons.
    """
    
    # Models that can have Bayesian transformations
    bayesian_capable = ['dnn', 'flexible_dnn', 'mlp', 'residual_mlp', 
                       'factorization_mlp', 'mtl', 'gin', 'gcn']
    
    # Get all model/rep combinations from the data
    all_pairs = df[['model', 'representation']].drop_duplicates()
    
    comparison_pairs = []
    
    for _, row in all_pairs.iterrows():
        model = row['model']
        rep = row['representation']
        
        # Check if this is a base model that could have Bayesian variants
        if model in bayesian_capable:
            # Look for its Bayesian variants in the data
            bayesian_variants = []
            
            for transform in ['full', 'last', 'variational']:  # FIXED: 'last' not 'last_layer'
                # FIXED: Pattern is {model}_bnn_{transform}
                bayesian_model = f"{model}_bnn_{transform}"
                
                if bayesian_model in df['model'].values:
                    # Check if this specific rep exists for this Bayesian variant
                    if len(df[(df['model'] == bayesian_model) & 
                             (df['representation'] == rep)]) > 0:
                        bayesian_variants.append(transform)
            
            # If we have at least one Bayesian variant, add to comparison list
            if bayesian_variants:
                comparison_pairs.append({
                    'base_model': model,
                    'representation': rep,
                    'bayesian_variants': bayesian_variants,
                    'has_deterministic': True
                })
        
        # Also track standalone Bayesian models (without deterministic counterpart)
        # FIXED: Pattern is {base}_bnn_{transformation}
        if '_bnn_' in model:
            parts = model.split('_bnn_')
            if len(parts) == 2:
                base = parts[0]
                transform = parts[1]
                
                if base in bayesian_capable and transform in ['full', 'last', 'variational']:
                    # Check if deterministic version exists
                    has_deterministic = len(df[(df['model'] == base) & 
                                              (df['representation'] == rep)]) > 0
                    
                    if not has_deterministic:
                        # Bayesian variant without deterministic counterpart
                        comparison_pairs.append({
                            'base_model': base,
                            'representation': rep,
                            'bayesian_variants': [transform],
                            'has_deterministic': False
                        })
    
    # Remove duplicates
    unique_pairs = []
    seen = set()
    for pair in comparison_pairs:
        key = (pair['base_model'], pair['representation'])
        if key not in seen:
            unique_pairs.append(pair)
            seen.add(key)
    
    return unique_pairs

def compare_bayesian_to_deterministic(df, bayesian_pairs):
    """
    For each Bayesian pair, compare performance to deterministic counterpart
    across all sigma values.
    """
    
    comparisons = []
    
    for pair in bayesian_pairs:
        base = pair['base_model']
        rep = pair['representation']
        
        if not pair['has_deterministic']:
            continue  # Skip if no deterministic version to compare against
        
        # Get deterministic performance
        det_data = df[(df['model'] == base) & (df['representation'] == rep)].copy()
        
        for transform in pair['bayesian_variants']:
            # FIXED: Pattern is {model}_bnn_{transform}
            bay_model = f"{base}_bnn_{transform}"
            bay_data = df[(df['model'] == bay_model) & (df['representation'] == rep)].copy()
            
            # Merge on sigma to compare
            merged = det_data.merge(bay_data, on='sigma', suffixes=('_det', '_bay'))
            
            for _, row in merged.iterrows():
                comparisons.append({
                    'base_model': base,
                    'representation': rep,
                    'transformation': transform,
                    'sigma': row['sigma'],
                    'r2_det': row['r2_det'],
                    'r2_bay': row['r2_bay'],
                    'r2_improvement': row['r2_bay'] - row['r2_det'],
                    'rmse_det': row['rmse_det'],
                    'rmse_bay': row['rmse_bay'],
                    'rmse_improvement': row['rmse_det'] - row['rmse_bay']  # Positive = Bayesian better
                })
    
    return pd.DataFrame(comparisons)

def identify_conformal_pairs(df):
    """
    Identify models that have both base and conformal variants,
    so we can compare them directly.
    
    Returns pairs where we can make meaningful comparisons.
    """
    
    # Models that can have conformal prediction wrappers
    conformal_capable = ['rf', 'qrf', 'xgboost', 'dnn', 'gauche', 
                        'gin', 'gcn', 'mlp', 'flexible_dnn']
    
    all_pairs = df[['model', 'representation']].drop_duplicates()
    
    comparison_pairs = []
    
    for _, row in all_pairs.iterrows():
        model = row['model']
        rep = row['representation']
        
        # Check if this is a base model that could have conformal variants
        if model in conformal_capable:
            # Look for conformal version
            conformal_model = f"conformal_{model}"
            
            has_conformal = len(df[(df['model'] == conformal_model) & 
                                  (df['representation'] == rep)]) > 0
            
            if has_conformal:
                comparison_pairs.append({
                    'base_model': model,
                    'representation': rep,
                    'has_conformal': True
                })
        
        # Also track standalone conformal models (without base counterpart)
        if model.startswith('conformal_'):
            base = model.replace('conformal_', '')
            
            has_base = len(df[(df['model'] == base) & 
                             (df['representation'] == rep)]) > 0
            
            if not has_base:
                comparison_pairs.append({
                    'base_model': base,
                    'representation': rep,
                    'has_conformal': True,
                    'has_base': False
                })
    
    # Remove duplicates
    unique_pairs = []
    seen = set()
    for pair in comparison_pairs:
        key = (pair['base_model'], pair['representation'])
        if key not in seen:
            unique_pairs.append(pair)
            seen.add(key)
    
    return unique_pairs

def compare_conformal_to_base(df, conformal_pairs):
    """
    For each conformal pair, compare performance to base model
    across all sigma values.
    """
    
    comparisons = []
    
    for pair in conformal_pairs:
        base = pair['base_model']
        rep = pair['representation']
        
        if not pair.get('has_conformal', False):
            continue
        
        # Get base performance
        base_data = df[(df['model'] == base) & (df['representation'] == rep)].copy()
        
        # Get conformal performance
        conf_model = f"conformal_{base}"
        conf_data = df[(df['model'] == conf_model) & (df['representation'] == rep)].copy()
        
        # Merge on sigma to compare
        merged = base_data.merge(conf_data, on='sigma', suffixes=('_base', '_conf'))
        
        for _, row in merged.iterrows():
            comparisons.append({
                'base_model': base,
                'representation': rep,
                'sigma': row['sigma'],
                'r2_base': row['r2_base'],
                'r2_conf': row['r2_conf'],
                'r2_improvement': row['r2_conf'] - row['r2_base'],
                'rmse_base': row['rmse_base'],
                'rmse_conf': row['rmse_conf'],
                'rmse_improvement': row['rmse_base'] - row['rmse_conf']  # Positive = Conformal better
            })
    
    return pd.DataFrame(comparisons)

def plot_bayesian_comparisons(bayesian_comp_df, output_dir):
    """Create visualizations comparing Bayesian to deterministic"""
    
    if len(bayesian_comp_df) == 0:
        print("No Bayesian comparisons to plot")
        return
    
    # Get unique pairs
    unique_pairs = bayesian_comp_df[['base_model', 'representation', 'transformation']].drop_duplicates()
    
    n_pairs = len(unique_pairs)
    if n_pairs == 0:
        return
    
    # Create subplots
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, pair) in enumerate(unique_pairs.iterrows()):
        base = pair['base_model']
        rep = pair['representation']
        transform = pair['transformation']
        
        pair_data = bayesian_comp_df[
            (bayesian_comp_df['base_model'] == base) &
            (bayesian_comp_df['representation'] == rep) &
            (bayesian_comp_df['transformation'] == transform)
        ].sort_values('sigma')
        
        # Plot R² comparison
        ax1 = axes[idx, 0]
        ax1.plot(pair_data['sigma'], pair_data['r2_det'], 'o-', label='Deterministic', linewidth=2)
        ax1.plot(pair_data['sigma'], pair_data['r2_bay'], 's-', label=f'Bayesian ({transform})', linewidth=2)
        ax1.set_xlabel('Sigma (Noise Level)')
        ax1.set_ylabel('R²')
        ax1.set_title(f'{base}/{rep}: R²')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot RMSE comparison
        ax2 = axes[idx, 1]
        ax2.plot(pair_data['sigma'], pair_data['rmse_det'], 'o-', label='Deterministic', linewidth=2)
        ax2.plot(pair_data['sigma'], pair_data['rmse_bay'], 's-', label=f'Bayesian ({transform})', linewidth=2)
        ax2.set_xlabel('Sigma (Noise Level)')
        ax2.set_ylabel('RMSE')
        ax2.set_title(f'{base}/{rep}: RMSE')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "bayesian_vs_deterministic.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Bayesian comparison plot to {output_path}")
    plt.close()

def plot_conformal_comparisons(conformal_comp_df, output_dir):
    """Create visualizations comparing Conformal to base"""
    
    if len(conformal_comp_df) == 0:
        print("No Conformal comparisons to plot")
        return
    
    # Get unique pairs
    unique_pairs = conformal_comp_df[['base_model', 'representation']].drop_duplicates()
    
    n_pairs = len(unique_pairs)
    if n_pairs == 0:
        return
    
    # Create subplots
    fig, axes = plt.subplots(n_pairs, 2, figsize=(14, 4*n_pairs))
    if n_pairs == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (_, pair) in enumerate(unique_pairs.iterrows()):
        base = pair['base_model']
        rep = pair['representation']
        
        pair_data = conformal_comp_df[
            (conformal_comp_df['base_model'] == base) &
            (conformal_comp_df['representation'] == rep)
        ].sort_values('sigma')
        
        # Plot R² comparison
        ax1 = axes[idx, 0]
        ax1.plot(pair_data['sigma'], pair_data['r2_base'], 'o-', label='Base', linewidth=2)
        ax1.plot(pair_data['sigma'], pair_data['r2_conf'], 's-', label='Conformal', linewidth=2)
        ax1.set_xlabel('Sigma (Noise Level)')
        ax1.set_ylabel('R²')
        ax1.set_title(f'{base}/{rep}: R²')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot RMSE comparison
        ax2 = axes[idx, 1]
        ax2.plot(pair_data['sigma'], pair_data['rmse_base'], 'o-', label='Base', linewidth=2)
        ax2.plot(pair_data['sigma'], pair_data['rmse_conf'], 's-', label='Conformal', linewidth=2)
        ax2.set_xlabel('Sigma (Noise Level)')
        ax2.set_ylabel('RMSE')
        ax2.set_title(f'{base}/{rep}: RMSE')
        ax2.legend()
        ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    output_path = Path(output_dir) / "conformal_vs_base.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Conformal comparison plot to {output_path}")
    plt.close()

def select_top_performers(df, n_top=5):
    """Select top N performers by R² at sigma=0"""
    
    sigma_0_df = df[df['sigma'] == 0.0].copy()
    
    if len(sigma_0_df) == 0:
        return []
    
    top = sigma_0_df.nlargest(n_top, 'r2')
    
    return [
        {'model': row['model'], 'representation': row['representation']}
        for _, row in top.iterrows()
    ]

def create_selection_report(df, robustness_df, selected_pairs, bayesian_pairs, 
                           conformal_pairs, top_performers, results_dir):
    """Create comprehensive pair selection report"""
    
    output_dir = Path(results_dir)
    
    # Save selections to JSON
    selections = {
        'selected_pairs': selected_pairs,
        'bayesian_pairs': bayesian_pairs,
        'conformal_pairs': conformal_pairs,
        'top_performers': top_performers
    }
    
    output_json = output_dir / "selected_pairs.json"
    with open(output_json, 'w') as f:
        json.dump(selections, f, indent=2)
    print(f"\nSaved selections to {output_json}")
    
    # Create visualizations
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.4, wspace=0.4)
    
    # 1. R² at sigma=0 (all pairs)
    ax1 = fig.add_subplot(gs[0, :])
    sigma_0_df = df[df['sigma'] == 0.0].sort_values('r2', ascending=False).head(30)  # Top 30 for readability
    colors = ['red' if {'model': row['model'], 'representation': row['representation']} in selected_pairs 
              else 'lightblue' for _, row in sigma_0_df.iterrows()]
    ax1.bar(range(len(sigma_0_df)), sigma_0_df['r2'], color=colors, alpha=0.7)
    ax1.set_xticks(range(len(sigma_0_df)))
    ax1.set_xticklabels([f"{row['model']}/{row['representation']}" 
                         for _, row in sigma_0_df.iterrows()], rotation=90, fontsize=7)
    ax1.set_ylabel('R² at σ=0')
    ax1.set_title('Top 30 by Performance at Sigma=0 (Red = Selected Pairs)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Robustness (RMSE increase)
    ax2 = fig.add_subplot(gs[1, 0])
    robustness_sorted = robustness_df.sort_values('rmse_increase').head(20)  # Top 20 most robust
    colors = ['red' if {'model': row['model'], 'representation': row['representation']} in selected_pairs 
              else 'lightgreen' for _, row in robustness_sorted.iterrows()]
    ax2.barh(range(len(robustness_sorted)), robustness_sorted['rmse_increase'], color=colors, alpha=0.7)
    ax2.set_yticks(range(len(robustness_sorted)))
    ax2.set_yticklabels([f"{row['model']}/{row['representation']}" 
                         for _, row in robustness_sorted.iterrows()], fontsize=7)
    ax2.set_xlabel('RMSE Increase (σ=0 → σ=1)')
    ax2.set_title('Top 20 Most Robust')
    ax2.grid(axis='x', alpha=0.3)
    
    # 3. R² vs robustness scatter
    ax3 = fig.add_subplot(gs[1, 1])
    sigma_0_full = df[df['sigma'] == 0.0].merge(robustness_df, on=['model', 'representation'])
    colors = ['red' if {'model': row['model'], 'representation': row['representation']} in selected_pairs 
              else 'gray' for _, row in sigma_0_full.iterrows()]
    ax3.scatter(sigma_0_full['r2'], sigma_0_full['rmse_increase'], c=colors, alpha=0.6, s=100)
    ax3.set_xlabel('R² at σ=0')
    ax3.set_ylabel('RMSE Increase (σ=0 → σ=1)')
    ax3.set_title('Performance vs Robustness')
    ax3.grid(alpha=0.3)
    
    # 4. Family coverage
    ax4 = fig.add_subplot(gs[1, 2])
    
    def get_family(model):
        if model in ['rf', 'xgboost', 'svm']:
            return 'Tree'
        elif model in ['qrf', 'ngboost']:
            return 'Tree-Prob'
        elif model in ['gauche']:
            return 'GP'
        elif model in ['dnn', 'flexible_dnn', 'mlp', 'residual_mlp', 'factorization_mlp', 'mtl']:
            return 'Neural'
        elif model in ['gin', 'gcn', 'graph_gp']:
            return 'Graph'
        elif 'conformal' in model:
            return 'Conformal'
        else:
            return 'Other'
    
    selected_families = [get_family(p['model']) for p in selected_pairs]
    family_counts = pd.Series(selected_families).value_counts()
    ax4.pie(family_counts.values, labels=family_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Selected Pairs by Model Family')
    
    # 5. Representation coverage
    ax5 = fig.add_subplot(gs[2, 0])
    selected_reps = [p['representation'] for p in selected_pairs]
    rep_counts = pd.Series(selected_reps).value_counts()
    ax5.bar(rep_counts.index, rep_counts.values, color='steelblue', alpha=0.7)
    ax5.set_xlabel('Representation')
    ax5.set_ylabel('Count')
    ax5.set_title('Selected Pairs by Representation')
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. RMSE curves for selected pairs
    ax6 = fig.add_subplot(gs[2, 1:])
    for pair in selected_pairs:
        pair_df = df[(df['model'] == pair['model']) & (df['representation'] == pair['representation'])]
        pair_df = pair_df.sort_values('sigma')
        if len(pair_df) > 0:
            ax6.plot(pair_df['sigma'], pair_df['rmse'], marker='o', 
                    label=f"{pair['model']}/{pair['representation']}", linewidth=2)
    ax6.set_xlabel('Sigma (Noise Level)')
    ax6.set_ylabel('RMSE')
    ax6.set_title('Selected Pairs: RMSE vs Noise')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    ax6.grid(alpha=0.3)
    
    output_fig = output_dir / "pair_selection_analysis.png"
    plt.savefig(output_fig, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_fig}")
    plt.close()
    
    # Text report
    output_txt = output_dir / "pair_selection_report.txt"
    with open(output_txt, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("PAIR SELECTION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("SELECTED PAIRS FOR DEEP DIVE EXPERIMENTS:\n")
        for i, pair in enumerate(selected_pairs, 1):
            pair_data = df[(df['model'] == pair['model']) & 
                          (df['representation'] == pair['representation']) & 
                          (df['sigma'] == 0.0)]
            
            if len(pair_data) == 0:
                continue
                
            sigma_0_row = pair_data.iloc[0]
            
            robust_data = robustness_df[(robustness_df['model'] == pair['model']) & 
                                       (robustness_df['representation'] == pair['representation'])]
            
            f.write(f"{i}. {pair['model']}/{pair['representation']}\n")
            f.write(f"   R² at σ=0: {sigma_0_row['r2']:.4f}\n")
            
            if len(robust_data) > 0:
                robust_row = robust_data.iloc[0]
                f.write(f"   RMSE increase: {robust_row['rmse_increase']:.4f} "
                       f"({robust_row['percent_increase']:.1f}%)\n")
            
            if 'params_source' in sigma_0_row:
                f.write(f"   Params: {sigma_0_row['params_source']}\n")
            
            f.write("\n")
        
        f.write("\nBAYESIAN TRANSFORMATION PAIRS:\n")
        if bayesian_pairs:
            for pair in bayesian_pairs:
                variants_str = ', '.join(pair['bayesian_variants'])
                f.write(f"  {pair['base_model']}/{pair['representation']} → {variants_str}\n")
        else:
            f.write("  None (no neural/graph models selected)\n")
        
        f.write("\nCONFORMAL PREDICTION PAIRS:\n")
        if conformal_pairs:
            for pair in conformal_pairs:
                f.write(f"  Conformal({pair['base_model']})/{pair['representation']}\n")
        else:
            f.write("  None (no conformal-eligible models selected)\n")
        
        f.write("\nTOP PERFORMERS FOR MULTI-TARGET TESTING:\n")
        for i, pair in enumerate(top_performers, 1):
            pair_data = df[(df['model'] == pair['model']) & 
                          (df['representation'] == pair['representation']) & 
                          (df['sigma'] == 0.0)]
            if len(pair_data) > 0:
                sigma_0_row = pair_data.iloc[0]
                f.write(f"{i}. {pair['model']}/{pair['representation']} (R²={sigma_0_row['r2']:.4f})\n")
    
    print(f"Saved text report to {output_txt}")

def main(results_dir="../results"):
    """Main pair selection workflow"""
    
    print("=" * 80)
    print("PHASE 0C PAIR SELECTION")
    print("=" * 80)
    
    print("\nLoading screening results...")
    df = load_screening_results(results_dir)
    
    if len(df) == 0:
        print("No data could be loaded! Exiting.")
        return
    
    print(f"\nLoaded {len(df)} unique model/rep/sigma combinations")
    
    print("\nCalculating robustness scores...")
    robustness_df = calculate_robustness_score(df)
    
    print("\nSelecting diverse pairs...")
    selected_pairs = select_diverse_pairs(df, robustness_df, n_pairs=10)
    
    # NEW: Bayesian and Conformal analysis
    print("\n" + "=" * 80)
    print("BAYESIAN TRANSFORMATION ANALYSIS")
    print("=" * 80)
    
    bayesian_pairs = identify_bayesian_pairs(df)
    print(f"\nFound {len(bayesian_pairs)} model/rep pairs with Bayesian variants:")
    for pair in bayesian_pairs:
        status = "✓" if pair['has_deterministic'] else "✗ (no deterministic baseline)"
        print(f"  {status} {pair['base_model']}/{pair['representation']}: {', '.join(pair['bayesian_variants'])}")
    
    bayesian_comp_df = compare_bayesian_to_deterministic(df, bayesian_pairs)
    if len(bayesian_comp_df) > 0:
        print(f"\nCreated {len(bayesian_comp_df)} Bayesian vs Deterministic comparisons")
        
        # Summary statistics
        print("\nBayesian Performance Summary (averaged across all sigma):")
        summary = bayesian_comp_df.groupby(['base_model', 'representation', 'transformation']).agg({
            'r2_improvement': 'mean',
            'rmse_improvement': 'mean'
        }).round(4)
        print(summary)
        
        plot_bayesian_comparisons(bayesian_comp_df, results_dir)
    
    print("\n" + "=" * 80)
    print("CONFORMAL PREDICTION ANALYSIS")
    print("=" * 80)
    
    conformal_pairs = identify_conformal_pairs(df)
    print(f"\nFound {len(conformal_pairs)} model/rep pairs with Conformal variants:")
    for pair in conformal_pairs:
        has_base = pair.get('has_base', True)
        status = "✓" if has_base else "✗ (no base model)"
        print(f"  {status} {pair['base_model']}/{pair['representation']}")
    
    conformal_comp_df = compare_conformal_to_base(df, conformal_pairs)
    if len(conformal_comp_df) > 0:
        print(f"\nCreated {len(conformal_comp_df)} Conformal vs Base comparisons")
        
        # Summary statistics
        print("\nConformal Performance Summary (averaged across all sigma):")
        summary = conformal_comp_df.groupby(['base_model', 'representation']).agg({
            'r2_improvement': 'mean',
            'rmse_improvement': 'mean'
        }).round(4)
        print(summary)
        
        plot_conformal_comparisons(conformal_comp_df, results_dir)
    
    # Continue with original selection report
    print("\nSelecting top performers...")
    top_performers = select_top_performers(df, n_top=5)
    
    print("\nCreating selection report...")
    create_selection_report(
        df, robustness_df, selected_pairs, bayesian_pairs, 
        conformal_pairs, top_performers, results_dir
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    
if __name__ == "__main__":
    main()