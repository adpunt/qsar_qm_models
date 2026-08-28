import pandas as pd
import altair as alt
import numpy as np
import os

def load_all_noise_strategies():
    """
    Load ALL available noise strategies and categorize them meaningfully
    """
    # All possible strategies from your experiment
    all_strategies = {
        # Distribution shapes (how noise is distributed)
        'legacy_gaussian': 'Gaussian',
        'legacy_left_tailed': 'Left-Skewed', 
        'legacy_right_tailed': 'Right-Skewed',
        'legacy_u_shaped': 'U-Shaped',
        'legacy_uniform': 'Uniform',
        
        # Value-dependent strategies (noise depends on target value)
        'value_proportional': 'Value-Proportional',
        'quantile': 'Quantile-Based',
        'threshold': 'Threshold-Based', 
        'outlier': 'Outlier-Focused',
        'heteroscedastic': 'Heteroscedastic',
        
        # Distribution + strategy combinations
        'value_prop_left_tail': 'Value-Prop + Left-Skewed',
        'value_prop_uniform': 'Value-Prop + Uniform',
        'quantile_u_shaped': 'Quantile + U-Shaped',
        
        # Model comparisons
        'legacy_gaussian_xgboost': 'Gaussian (XGBoost)',
        'value_proportional_mlp': 'Value-Prop (MLP)',
        
        # Test
        'quick_test': 'Quick Test'
    }
    
    all_data = []
    found_strategies = []
    
    for file_key, display_name in all_strategies.items():
        filename = f"noise_{file_key}.csv"
        file_path = f"../results/{filename}"
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['noise_strategy'] = file_key
            df['noise_type'] = display_name
            
            # Add category for organization
            if file_key.startswith('legacy_'):
                df['category'] = 'Distribution Shapes'
            elif file_key in ['value_proportional', 'quantile', 'threshold', 'outlier', 'heteroscedastic']:
                df['category'] = 'Value-Dependent'
            elif 'prop' in file_key or 'quantile_u' in file_key:
                df['category'] = 'Hybrid Strategies'
            elif 'xgboost' in file_key or 'mlp' in file_key:
                df['category'] = 'Model Comparison'
            else:
                df['category'] = 'Other'
                
            all_data.append(df)
            found_strategies.append(display_name)
            print(f"✓ Loaded {display_name}")
        else:
            print(f"✗ Missing {filename}")
    
    print(f"\nFound {len(found_strategies)} noise strategies total")
    
    if not all_data:
        raise ValueError("No noise strategy files found!")
    
    return pd.concat(all_data, ignore_index=True)

def calculate_robustness_metrics(df):
    """
    Calculate noise robustness metrics for each model AND representation
    """
    # Clean data
    df['r2_score'] = pd.to_numeric(df['r2_score'], errors='coerce')
    df['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
    df = df[df['r2_score'].notna()]
    df = df[(df['r2_score'] >= -1.0) & (df['r2_score'] <= 1.0)]
    
    robustness_results = []
    
    for model in df['model'].unique():
        for rep in df['rep'].unique():
            for noise_type in df['noise_type'].unique():
                subset = df[(df['model'] == model) & (df['rep'] == rep) & (df['noise_type'] == noise_type)]
                
                if len(subset) >= 2:
                    # Sort by sigma
                    subset = subset.sort_values('sigma')
                    
                    # Get category from the subset
                    category = subset['category'].iloc[0] if 'category' in subset.columns else 'Unknown'
                    
                    # Calculate key robustness metrics
                    r2_at_zero = subset[subset['sigma'] == 0]['r2_score'].iloc[0] if len(subset[subset['sigma'] == 0]) > 0 else np.nan
                    r2_at_max = subset['r2_score'].iloc[-1]  # R² at highest sigma
                    
                    # Area Under Curve (robustness score)
                    auc = np.trapz(subset['r2_score'], subset['sigma'])
                    if subset['sigma'].iloc[-1] != subset['sigma'].iloc[0]:
                        auc_normalized = auc / (subset['sigma'].iloc[-1] - subset['sigma'].iloc[0])
                    else:
                        auc_normalized = r2_at_zero
                    
                    # Performance degradation
                    degradation = r2_at_zero - r2_at_max if not np.isnan(r2_at_zero) else np.nan
                    
                    robustness_results.append({
                        'model': model,
                        'representation': rep,
                        'noise_type': noise_type,
                        'category': category,
                        'r2_clean': r2_at_zero,
                        'r2_noisy': r2_at_max,
                        'degradation': degradation,
                        'auc_robustness': auc_normalized
                    })
    
    return pd.DataFrame(robustness_results)

def create_category_comparison_plot(df, representation='ecfp4'):
    """
    Create separate plots for each category of noise strategies
    """
    df_rep = df[df['rep'] == representation].copy()
    df_plot = df_rep.groupby(['sigma', 'model', 'noise_type', 'category'], as_index=False)['r2_score'].mean()
    
    charts = {}
    
    for category in df_plot['category'].unique():
        df_cat = df_plot[df_plot['category'] == category]
        
        chart = alt.Chart(df_cat).mark_line(point=True).encode(
            x=alt.X('sigma:Q', title='Noise Level (σ)'),
            y=alt.Y('r2_score:Q', title='R² Score', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model:N', title='Model'),
            column=alt.Column('noise_type:N', title='Noise Strategy', header=alt.Header(labelFontSize=10)),
            tooltip=['model', 'noise_type', 'sigma', 'r2_score']
        ).properties(
            width=150,
            height=250,
            title=f'{category} - {representation.upper()}'
        ).resolve_scale(
            color='independent'
        )
        
        charts[category] = chart
        
    return charts

def create_overall_robustness_ranking(robustness_df):
    """
    Create a comprehensive ranking of model robustness across ALL noise types
    """
    # Calculate overall robustness score for each model
    model_robustness = robustness_df.groupby('model').agg({
        'auc_robustness': 'mean',
        'degradation': 'mean', 
        'r2_clean': 'mean'
    }).reset_index()
    
    model_robustness = model_robustness.sort_values('auc_robustness', ascending=False)
    
    # Create ranking chart
    chart = alt.Chart(model_robustness).mark_bar().encode(
        x=alt.X('auc_robustness:Q', title='Average Robustness Score'),
        y=alt.Y('model:N', sort=alt.SortField('auc_robustness', order='descending'), title='Model'),
        color=alt.Color('auc_robustness:Q', scale=alt.Scale(scheme='viridis')),
        tooltip=['model', 'auc_robustness', 'degradation', 'r2_clean']
    ).properties(
        width=500,
        height=400,
        title='Overall Model Robustness Ranking (Averaged Across ALL Noise Types)'
    )
    
    return chart, model_robustness

def create_robustness_ranking_plot(robustness_df):
    """
    Create a ranking plot showing which models are most robust to each noise type
    """
    # Create ranking chart
    chart = alt.Chart(robustness_df).mark_circle(size=100).encode(
        x=alt.X('noise_type:N', title='Noise Type'),
        y=alt.Y('model:N', title='Model'),
        color=alt.Color('auc_robustness:Q', 
                       title='Robustness Score',
                       scale=alt.Scale(scheme='viridis')),
        size=alt.Size('r2_clean:Q', title='Clean Performance'),
        tooltip=['model', 'noise_type', 'auc_robustness', 'r2_clean', 'degradation']
    ).properties(
        width=400,
        height=300,
        title='Model Robustness Heatmap (Size = Clean Performance, Color = Robustness)'
    )
    
    return chart

def print_robustness_insights(robustness_df):
    """
    Print key insights about model AND representation robustness across ALL noise strategies
    """
    print("\n" + "="*60)
    print("COMPREHENSIVE NOISE ROBUSTNESS ANALYSIS")
    print("="*60)
    
    # Overall most robust MODELS (averaged across representations)
    overall_model_robustness = robustness_df.groupby('model')['auc_robustness'].mean().sort_values(ascending=False)
    print(f"\nMOST ROBUST MODELS (averaged across ALL representations & noise types):")
    for i, (model, score) in enumerate(overall_model_robustness.head(5).items()):
        print(f"{i+1}. {model:15}: {score:.3f}")
    
    # Overall most robust REPRESENTATIONS (averaged across models)
    overall_rep_robustness = robustness_df.groupby('representation')['auc_robustness'].mean().sort_values(ascending=False)
    print(f"\nMOST ROBUST REPRESENTATIONS (averaged across ALL models & noise types):")
    for i, (rep, score) in enumerate(overall_rep_robustness.head().items()):
        print(f"{i+1}. {rep:15}: {score:.3f}")
    
    # Best model-representation combinations
    print(f"\nBEST MODEL-REPRESENTATION COMBINATIONS:")
    combo_robustness = robustness_df.groupby(['model', 'representation'])['auc_robustness'].mean().sort_values(ascending=False)
    for i, ((model, rep), score) in enumerate(combo_robustness.head(5).items()):
        print(f"{i+1}. {model:12} + {rep:8}: {score:.3f}")
    
    # Performance by noise category (for main representation)
    main_rep = robustness_df['representation'].value_counts().index[0]  # Most common representation
    main_rep_data = robustness_df[robustness_df['representation'] == main_rep]
    
    print(f"\nROBUSTNESS BY NOISE CATEGORY ({main_rep.upper()}):")
    
    for category in main_rep_data['category'].unique():
        print(f"\n{category}:")
        cat_data = main_rep_data[main_rep_data['category'] == category].groupby('model')['auc_robustness'].mean().sort_values(ascending=False)
        for i, (model, score) in enumerate(cat_data.head(3).items()):
            print(f"  {i+1}. {model:15}: {score:.3f}")
    
    # Representation-specific insights
    print(f"\nREPRESENTATION-SPECIFIC INSIGHTS:")
    for rep in robustness_df['representation'].unique():
        rep_data = robustness_df[robustness_df['representation'] == rep]
        best_model = rep_data.groupby('model')['auc_robustness'].mean().idxmax()
        best_score = rep_data.groupby('model')['auc_robustness'].mean().max()
        print(f"{rep:12}: Best model = {best_model:15} ({best_score:.3f})")
    
    # Biggest performance drops by representation
    print(f"\nBIGGEST PERFORMANCE DROPS UNDER NOISE BY REPRESENTATION:")
    for rep in robustness_df['representation'].unique():
        rep_data = robustness_df[robustness_df['representation'] == rep]
        worst_degradation = rep_data.groupby('model')['degradation'].mean().sort_values(ascending=False)
        worst_model = worst_degradation.index[0]
        worst_score = worst_degradation.iloc[0]
        print(f"{rep:12}: {worst_model:15} drops {worst_score:.3f} R² points")
    
    # Clean performance leaders by representation
    print(f"\nCLEAN PERFORMANCE LEADERS BY REPRESENTATION:")
    for rep in robustness_df['representation'].unique():
        rep_data = robustness_df[robustness_df['representation'] == rep]
        best_clean = rep_data.groupby('model')['r2_clean'].mean().sort_values(ascending=False)
        best_model = best_clean.index[0]
        best_score = best_clean.iloc[0]
        print(f"{rep:12}: {best_model:15} ({best_score:.3f} R²)")

def create_representation_comparison_chart(robustness_df):
    """
    Create a chart comparing robustness across representations
    """
    # Average robustness by model and representation
    rep_comparison = robustness_df.groupby(['model', 'representation'])['auc_robustness'].mean().reset_index()
    
    chart = alt.Chart(rep_comparison).mark_circle(size=100).encode(
        x=alt.X('representation:N', title='Molecular Representation'),
        y=alt.Y('model:N', title='Model'),
        color=alt.Color('auc_robustness:Q', 
                       title='Robustness Score',
                       scale=alt.Scale(scheme='viridis')),
        tooltip=['model', 'representation', 'auc_robustness']
    ).properties(
        width=400,
        height=300,
        title='Model Robustness by Representation (Averaged Across All Noise Types)'
    )
    
    return chart

def main():
    """
    Main analysis using ALL available noise strategies
    """
    print("Loading ALL noise strategy results...")
    
    # Load all available data
    df = load_all_noise_strategies()
    
    # Calculate robustness metrics
    print("\nCalculating robustness metrics...")
    robustness_df = calculate_robustness_metrics(df)
    
    # Check available representations
    available_reps = df['rep'].unique() if 'rep' in df.columns else ['ecfp4']
    print(f"\nAnalyzing representations: {list(available_reps)}")
    
    # Focus on main representation for category plots (usually ecfp4)
    main_rep = 'ecfp4' if 'ecfp4' in available_reps else available_reps[0]
    
    # Create plots
    print(f"\nCreating plots for {main_rep}...")
    
    # Category-wise plots (for main representation)
    category_charts = create_category_comparison_plot(df, main_rep)
    for category, chart in category_charts.items():
        filename = f"robustness_{main_rep}_{category.lower().replace(' ', '_')}.html"
        chart.save(filename)
        print(f"Saved: {filename}")
    
    # Overall ranking (main representation)
    main_rep_robustness = robustness_df[robustness_df['representation'] == main_rep]
    ranking_chart, model_ranking = create_overall_robustness_ranking(main_rep_robustness)
    ranking_chart.save(f"robustness_{main_rep}_overall_ranking.html")
    print(f"Saved: robustness_{main_rep}_overall_ranking.html")
    
    # Detailed heatmap (main representation)
    heatmap_chart = create_robustness_ranking_plot(main_rep_robustness)
    heatmap_chart.save(f"robustness_{main_rep}_heatmap.html")
    print(f"Saved: robustness_{main_rep}_heatmap.html")
    
    # Representation comparison chart
    rep_chart = create_representation_comparison_chart(robustness_df)
    rep_chart.save("robustness_representation_comparison.html")
    print("Saved: robustness_representation_comparison.html")
    
    # Print comprehensive insights
    print_robustness_insights(robustness_df)
    
    return robustness_df, model_ranking

if __name__ == "__main__":
    robustness_results, model_rankings = main()