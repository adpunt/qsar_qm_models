#!/usr/bin/env python3
"""
Complete plotting framework for QSAR/QSPR noise robustness experiments
Generates Figures 1-6 using Altair for publication-quality output
Updated to handle Bayesian transformations and conformal prediction properly
"""

import altair as alt
import pandas as pd
import numpy as np
import os
import glob
from pathlib import Path
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

# Configure Altair for publication quality
alt.data_transformers.enable('json')

# Publication color scheme (colorblind-friendly cheminformatics palette)
COLORS = {
    'baseline': '#1f77b4',      # Blue - deterministic baseline
    'full': '#ff7f0e',          # Orange - full Bayesian transformation  
    'last_layer': '#2ca02c',    # Green - last layer Bayesian
    'variational': '#d62728',   # Red - variational Bayesian
    'conformal': '#9467bd',     # Purple - conformal prediction
    'tuned': '#8c564b',         # Brown - hyperparameter tuned
    # Legacy colors for compatibility
    'deterministic': '#1f77b4',
    'bayesian': '#ff7f0e',
    'rf': '#1f77b4',
    'qrf': '#ff7f0e',
    'xgboost': '#d62728',
    'svm': '#9467bd',
    'dnn': '#8c564b',
    'gin': '#e377c2',
    'gcn': '#7f7f7f',
    'gauche': '#bcbd22',
    'ngboost': '#17becf',
    'graph_gp': '#aec7e8'
}

def read_csv_safe(filepath):
    """Read CSV handling both old (10 col) and new (11 col) formats"""
    try:
        # Try modern pandas first (pandas >= 1.3.0)
        df = pd.read_csv(filepath, on_bad_lines='skip', engine='python')
    except TypeError:
        # Fall back for older pandas versions
        try:
            df = pd.read_csv(filepath, error_bad_lines=False, warn_bad_lines=False, engine='python')
        except:
            # Last resort - read line by line
            df = pd.read_csv(filepath, engine='python')
    
    # Ensure params_source column exists
    if 'params_source' not in df.columns:
        df['params_source'] = 'unknown'
    
    return df

def parse_model_info(model_name):
    """Parse model name to extract base model and transformation type"""
    # Handle conformal prediction models
    if model_name.startswith('conformal_'):
        base = model_name.replace('conformal_', '').replace('_split', '')
        return 'conformal', base
    
    # Handle Bayesian transformations
    if '_full' in model_name or model_name.startswith('bnn_full'):
        base = model_name.replace('_full', '').replace('_bnn', '').replace('bnn_', '').replace('bnn', '')
        return 'full', base if base else 'dnn'
    elif '_last' in model_name or model_name.startswith('bnn_last'):
        base = model_name.replace('_last', '').replace('_bnn', '').replace('bnn_', '').replace('bnn', '')
        return 'last_layer', base if base else 'dnn'
    elif '_variational' in model_name or model_name.startswith('bnn_var'):
        base = model_name.replace('_variational', '').replace('_bnn', '').replace('bnn_', '').replace('bnn', '').replace('_var', '')
        return 'variational', base if base else 'dnn'
    
    # Baseline/deterministic models
    return 'baseline', model_name

def load_results_data(results_dir='results'):
    pattern = os.path.join(results_dir, '*.csv')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No CSV files found in {results_dir}")
        return pd.DataFrame()
    
    print(f"Loading {len(files)} CSV files from {results_dir}...")
    
    dfs = []
    for file in files:
        try:
            df = read_csv_safe(file)
            df['source_file'] = os.path.basename(file)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")
            continue
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    combined_df[['transformation_type', 'base_model']] = combined_df['model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    combined_df['is_tuned'] = combined_df['source_file'].str.contains('tuned', case=False)
    
    return combined_df
    
def load_uncertainty_data(results_dir='results'):
    """Load uncertainty data from uncertainty CSV files"""
    uncertainty_dirs = glob.glob(os.path.join(results_dir, '*_uncertainty'))
    files = []
    for uncertainty_dir in uncertainty_dirs:
        pattern = os.path.join(uncertainty_dir, 'uncertainty_*.csv')
        files.extend(glob.glob(pattern))
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        df = read_csv_safe(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_conformal_data(results_dir='results'):
    """Load conformal prediction interval data"""
    conformal_dir = os.path.join(results_dir, 'conformal_intervals')
    pattern = os.path.join(conformal_dir, 'conformal_intervals_*.csv')
    files = glob.glob(pattern)
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        df = read_csv_safe(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_per_epoch_data(results_dir='results'):
    """Load per-epoch training data"""
    pattern = os.path.join(results_dir, '*_per_epoch.csv')
    files = glob.glob(pattern)
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        df = read_csv_safe(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def create_display_labels(df):
    """Create clean display labels for legends - handles varying column structures"""
    def make_label(row):
        # Safely get values with defaults
        model = row.get('base_model', row.get('model', 'Unknown')).replace('_', ' ').upper()
        rep = row.get('rep', '').upper()
        trans = row.get('transformation_type', 'baseline')
        
        # Build label
        if trans == 'baseline':
            label = model
        elif trans == 'conformal':
            label = f"Conformal/{model}"
        else:
            label = f"{trans.replace('_', ' ').title()}/{model}"
        
        # Add rep if present
        if rep:
            label = f"{label} ({rep})"
            
        return label
    
    df['display_label'] = df.apply(make_label, axis=1)
    return df

# =============================================================================
# FIGURE 1: Model Robustness to Noise
# =============================================================================

def plot_figure_1a(df, save_path='figure_1a.png'):
    """Fig 1a: RMSE vs. Ïƒ for bit-vector representations with conformal predictions"""
    
    bit_vector_reps = ['ecfp4', 'pdv', 'sns', 'smiles', 'randomized_smiles']
    baseline_models = ['rf', 'svm', 'xgboost', 'gauche']
    
    # Add debug prints HERE
    print("\n=== DEBUG FIGURE 1A ===")
    print(f"Total rows in df: {len(df)}")
    print(f"Unique models in df: {df['model'].unique()}")
    print(f"Unique base_models in df: {df['base_model'].unique()}")
    print(f"Unique transformation_types in df: {df['transformation_type'].unique()}")
    print(f"Unique reps in df: {df['rep'].unique()}")
    
    plot_df = df[
        (df['rep'].isin(bit_vector_reps)) & 
        (df['base_model'].isin(baseline_models)) &
        (df['transformation_type'] == 'baseline')
    ].copy()
    
    print(f"\nAfter filtering for bit-vector:")
    print(f"Rows in plot_df: {len(plot_df)}")
    print(f"Models in plot_df: {plot_df['model'].unique()}")
    print(f"Base models in plot_df: {plot_df['base_model'].unique()}")
    print(f"Transformation types in plot_df: {plot_df['transformation_type'].unique()}")
    
    if plot_df.empty:
        print("No data found for Figure 1a")
        return None
    
    summary_df = plot_df.groupby(['sigma', 'base_model', 'rep', 'transformation_type']).agg({
        'rmse': ['mean', 'std']
    }).reset_index()
    summary_df.columns = ['sigma', 'base_model', 'rep', 'transformation_type', 'rmse_mean', 'rmse_std']

    summary_df = create_display_labels(summary_df)
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        y=alt.Y('rmse_mean:Q', title='RMSE',
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        color=alt.Color('base_model:N',
               scale=alt.Scale(domain=['rf', 'gauche', 'svm', 'xgboost'],
                             range=[COLORS['rf'], COLORS['gauche'], COLORS['svm'], COLORS['xgboost']]),
               title='Base Model',
               legend=alt.Legend(titleFontSize=12, labelFontSize=11)),
        facet=alt.Facet('rep:N', title='Molecular Representation', 
                       header=alt.Header(titleFontSize=12, labelFontSize=11),
                       columns=3),
        tooltip=['base_model:N', 'transformation_type:N', 'rep:N', 'sigma:Q', 'rmse_mean:Q']
    ).properties(
        width=160,
        height=140,
        title=alt.TitleParams('Figure 1a: RMSE vs. Noise Level for Bit-Vector Representations', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

def plot_figure_1b(df, save_path='figure_1b.png'):
    """Fig 1b: RMSE vs. σ for graph representations with conformal predictions"""
    
    graph_baseline_models = ['gin', 'gcn', 'graph_gp']
    
    plot_df = df[
        (df['rep'] == 'graph') & 
        (df['base_model'].isin(graph_baseline_models)) &
        (df['transformation_type'] == 'baseline')
    ].copy()

    print("\n=== DEBUG FIGURE 1B ===")
    print(f"Graph models looking for: {graph_baseline_models}")
    print(f"Rows with rep='graph': {len(df[df['rep'] == 'graph'])}")
    print(f"Models in graph data: {df[df['rep'] == 'graph']['base_model'].unique()}")
    print(f"Transformations in graph data: {df[df['rep'] == 'graph']['transformation_type'].unique()}")
    print(f"Rows in plot_df after filtering: {len(plot_df)}")
    if not plot_df.empty:
        print(f"Models in plot_df: {plot_df['base_model'].unique()}")
    
    if plot_df.empty:
        print("No data found for Figure 1b")
        return None
    
    summary_df = plot_df.groupby(['sigma', 'base_model', 'rep', 'transformation_type']).agg({
        'rmse': ['mean', 'std']
    }).reset_index()
    summary_df.columns = ['sigma', 'base_model', 'rep', 'transformation_type', 'rmse_mean', 'rmse_std']

    summary_df = create_display_labels(summary_df)
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y('rmse_mean:Q', title='RMSE',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('base_model:N',
                scale=alt.Scale(domain=['gin', 'gcn', 'graph_gp'],
                             range=[COLORS['gin'], COLORS['gcn'], COLORS['graph_gp']]),
                title='Base Model',
                legend=alt.Legend(titleFontSize=14, labelFontSize=12)),
        tooltip=['base_model:N', 'transformation_type:N', 'sigma:Q', 'rmse_mean:Q']
    ).properties(
        width=600,
        height=400,
        title=alt.TitleParams('Figure 1b: RMSE vs. Noise Level for Graph Representations', 
                             fontSize=18, anchor='start')
    )
    
    chart.save(save_path)
    return chart

def plot_figure_1c(df, save_path='figure_1c.png'):
    """Fig 1c: RMSE vs. σ for NN vs BNN - separate plot for each model/rep pair"""
    
    # All neural network models with ALL transformations (baseline, full, last_layer, variational)
    nn_models = ['dnn', 'mlp', 'residual_mlp', 'factorization_mlp', 'gin', 'gcn']
    # Also include 'bnn' base_model (from your naming: bnn_full, bnn_last, bnn_variational)
    nn_models_expanded = nn_models + ['bnn']

    plot_df = df[df['base_model'].isin(nn_models_expanded)].copy()

    print("\n=== DEBUG FIGURE 1C ===")
    print(f"Total rows in plot_df: {len(plot_df)}")
    print(f"Unique base_models: {plot_df['base_model'].unique()}")
    print(f"Unique transformation_types: {plot_df['transformation_type'].unique()}")

    # Check data availability per base model
    for model in plot_df['base_model'].unique():
        model_data = plot_df[plot_df['base_model'] == model]
        print(f"\n{model}:")
        print(f"  Reps: {model_data['rep'].unique()}")
        print(f"  Transformations: {model_data['transformation_type'].unique()}")
        print(f"  Row count: {len(model_data)}")

    if plot_df.empty:
        print("No data found for Figure 1c")
        return None
    
    # Only keep key representations to avoid overcrowding
    key_reps = ['ecfp4', 'pdv', 'graph']
    plot_df = plot_df[plot_df['rep'].isin(key_reps)]
    
    # Only keep key models
    key_models = ['dnn', 'mlp', 'gin', 'gcn']
    plot_df = plot_df[plot_df['base_model'].isin(key_models)]
    
    if plot_df.empty:
        print("No data found after filtering for Figure 1c")
        return None
    
    summary_df = plot_df.groupby(['sigma', 'base_model', 'rep', 'transformation_type']).agg({
        'rmse': ['mean', 'std']
    }).reset_index()
    summary_df.columns = ['sigma', 'base_model', 'rep', 'transformation_type', 'rmse_mean', 'rmse_std']

    summary_df = create_display_labels(summary_df)
    
    # Create model-rep identifier for faceting
    summary_df['model_rep'] = summary_df['base_model'] + ' (' + summary_df['rep'] + ')'
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        y=alt.Y('rmse_mean:Q', title='RMSE',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational']]),
                       title='Transformation Type',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        facet=alt.Facet('model_rep:N', title=None, 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=3),
        tooltip=['base_model:N', 'transformation_type:N', 'rep:N', 'sigma:Q', 'rmse_mean:Q']
    ).properties(
        width=180,
        height=150,
        title=alt.TitleParams('Figure 1c: Neural Network Bayesian Transformations Robustness', 
                             fontSize=16, anchor='start')
    ).resolve_scale(y='independent')
    
    chart.save(save_path)
    return chart

def plot_figure_1d(df, save_path='figure_1d.png'):
    """Fig 1d: R² vs. σ for top model-representation pairs"""
    
    baseline_performance = df[df['sigma'] == 0.0].groupby(['model', 'rep'])['r2'].mean()
    
    if baseline_performance.empty:
        print("No baseline data (sigma=0) found for Figure 1d")
        return None
    
    top_pairs = baseline_performance.nlargest(15).index.tolist()
    plot_df = df[df.set_index(['model', 'rep']).index.isin(top_pairs)].copy()
    
    summary_df = plot_df.groupby(['sigma', 'model', 'rep']).agg({
        'r2': ['mean', 'std']
    }).reset_index()
    summary_df.columns = ['sigma', 'model', 'rep', 'r2_mean', 'r2_std']
    summary_df['model_rep'] = summary_df['model'] + '_' + summary_df['rep']

    summary_df = create_display_labels(summary_df)
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y('r2_mean:Q', title='R²', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('model_rep:N', 
                       legend=alt.Legend(labelLimit=300, symbolLimit=25, titleFontSize=14, labelFontSize=11),
                       title='Model-Representation'),
        tooltip=['model:N', 'rep:N', 'sigma:Q', 'r2_mean:Q']
    ).properties(
        width=700,
        height=450,
        title=alt.TitleParams('Figure 1d: R² vs. Noise Level for Top Model-Representation Pairs', 
                             fontSize=18, anchor='start')
    )
    
    chart.save(save_path)
    return chart

# =============================================================================
# FIGURE 2: Bayesian vs. Deterministic
# =============================================================================

def plot_figure_2a(per_epoch_df, save_path='figure_2a.png'):
    """Fig 2a: Training curves comparing baseline vs all Bayesian transformations"""
    
    if per_epoch_df.empty:
        print("No per-epoch data available for Figure 2a")
        return None
    
    per_epoch_df = per_epoch_df.copy()
    per_epoch_df[['transformation_type', 'base_model']] = per_epoch_df['model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    relevant_models = ['dnn', 'mlp', 'gin', 'gcn']
    plot_df = per_epoch_df[per_epoch_df['base_model'].isin(relevant_models)].copy()
    
    if plot_df.empty:
        print("No relevant models found in per-epoch data for Figure 2a")
        return None
    
    summary_df = plot_df.groupby(['epoch', 'base_model', 'transformation_type', 'sigma']).agg({
        'train_loss': 'mean',
        'val_loss': 'mean'
    }).reset_index()

    summary_df = create_display_labels(summary_df)
    
    plot_data = pd.melt(summary_df, 
                       id_vars=['epoch', 'base_model', 'transformation_type', 'sigma'],
                       value_vars=['train_loss', 'val_loss'],
                       var_name='loss_type', value_name='loss')
    
    chart = alt.Chart(plot_data).mark_line(strokeWidth=2).encode(
        x=alt.X('epoch:Q', title='Epoch',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        y=alt.Y('loss:Q', title='Loss', scale=alt.Scale(type='log'),
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational']]),
                       title='Transformation Type',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        strokeDash=alt.StrokeDash('loss_type:N', title='Loss Type'),
        facet=alt.Facet('base_model:N', title='Base Model', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=2),
        row=alt.Row('sigma:N', title='Noise Level (σ)', 
                   header=alt.Header(titleFontSize=11, labelFontSize=10)),
        tooltip=['base_model:N', 'transformation_type:N', 'epoch:Q', 'loss:Q', 'loss_type:N']
    ).properties(
        width=220,
        height=160,
        title=alt.TitleParams('Figure 2a: Training Curves - Baseline vs Bayesian Transformations', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

def plot_figure_2b(df, save_path='figure_2b.png'):
    """Fig 2b: Final R² at fixed σ comparing all transformation types"""
    
    sigma_level = 0.5
    plot_df = df[df['sigma'] == sigma_level].copy()
    
    if plot_df.empty:
        print(f"No data found for sigma level {sigma_level} in Figure 2b")
        return None
    
    key_models = ['rf', 'qrf', 'dnn']
    plot_df = plot_df[plot_df['base_model'].isin(key_models) | plot_df['model'].isin(key_models)]
    
    if plot_df.empty:
        print("No data found for key models in Figure 2b")
        return None
    
    summary_df = plot_df.groupby(['base_model', 'transformation_type'])['r2'].agg(['mean', 'std']).reset_index()
    summary_df.columns = ['base_model', 'transformation_type', 'r2_mean', 'r2_std']

    summary_df = create_display_labels(summary_df)
    
    chart = alt.Chart(summary_df).mark_bar().encode(
        x=alt.X('base_model:N', title='Base Model',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y('r2_mean:Q', title='R²',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational']]),
                       title='Transformation Type',
                       legend=alt.Legend(titleFontSize=14, labelFontSize=12)),
        xOffset='transformation_type:N',
        tooltip=['base_model:N', 'transformation_type:N', 'r2_mean:Q', 'r2_std:Q']
    ).properties(
        width=500,
        height=350,
        title=alt.TitleParams(f'Figure 2b: Final R² at σ = {sigma_level}', 
                             fontSize=18, anchor='start')
    )
    
    chart.save(save_path)
    return chart

# =============================================================================
# FIGURE 3: Uncertainty and Label Noise
# =============================================================================

def plot_figure_3a(uncertainty_df, save_path='figure_3a.png'):
    """Fig 3a: Uncertainty vs. prediction error scatter plots"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3a")
        return None
    
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['prediction_error'] = np.abs(uncertainty_df['y_true_noisy'] - uncertainty_df['y_pred_mean'])
    uncertainty_df[['transformation_type', 'base_model']] = uncertainty_df['Model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    sampled_dfs = []
    for (model, sigma), group in uncertainty_df.groupby(['Model', 'Sigma']):
        sample_size = min(300, len(group))
        if sample_size > 10:
            sampled = group.sample(n=sample_size, random_state=42)
            sampled_dfs.append(sampled)
    
    if not sampled_dfs:
        print("No data available for uncertainty scatter plot")
        return None
        
    plot_df = pd.concat(sampled_dfs)
    
    chart = alt.Chart(plot_df).mark_circle(size=15, opacity=0.6).encode(
        x=alt.X('y_pred_std:Q', title='Predicted Uncertainty',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        y=alt.Y('prediction_error:Q', title='Prediction Error',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational', 'conformal'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational'], COLORS['conformal']]),
                       title='Model Type',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        facet=alt.Facet('Sigma:N', title='Noise Level (σ)', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=3),
        tooltip=['Model:N', 'transformation_type:N', 'y_pred_std:Q', 'prediction_error:Q']
    ).properties(
        width=200,
        height=170,
        title=alt.TitleParams('Figure 3a: Uncertainty vs. Prediction Error', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

def plot_figure_3b(uncertainty_df, save_path='figure_3b.png'):
    """Fig 3b: Uncertainty boxplots for clean vs. noisy data"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3b")
        return None
    
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['data_condition'] = uncertainty_df['Sigma'].apply(
        lambda x: 'Clean' if x == 0.0 else 'Noisy'
    )
    uncertainty_df[['transformation_type', 'base_model']] = uncertainty_df['Model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    # Aggregate to reduce overlapping labels
    agg_df = uncertainty_df.groupby(['base_model', 'transformation_type', 'data_condition']).agg({
        'y_pred_std': ['mean', 'std', 'median']
    }).reset_index()
    agg_df.columns = ['base_model', 'transformation_type', 'data_condition', 'mean_unc', 'std_unc', 'median_unc']

    # Only show key models
    key_models = ['dnn', 'qrf', 'gauche']
    agg_df = agg_df[agg_df['base_model'].isin(key_models)]

    chart = alt.Chart(agg_df).mark_bar().encode(
        x=alt.X('transformation_type:N', title='Transformation Type',
                axis=alt.Axis(labelFontSize=10, titleFontSize=11, labelAngle=-20)),
        y=alt.Y('mean_unc:Q', title='Mean Predicted Uncertainty',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('data_condition:N',
                       scale=alt.Scale(domain=['Clean', 'Noisy'],
                                     range=[COLORS['baseline'], COLORS['full']]),
                       title='Data Condition',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10, orient='top')),
        xOffset='data_condition:N',
        column=alt.Column('base_model:N', title='Base Model',
                         header=alt.Header(titleFontSize=11, labelFontSize=10)),
        tooltip=['base_model:N', 'transformation_type:N', 'data_condition:N', 'mean_unc:Q']
    ).properties(
        width=200,
        height=240,
        title=alt.TitleParams('Figure 3b: Uncertainty Distribution - Clean vs. Noisy Data', 
                             fontSize=16, anchor='start')
    )
    
    chart.save(save_path)
    return chart

def plot_figure_3c(uncertainty_df, save_path='figure_3c.png'):
    """Fig 3c: Calibration curves"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3c")
        return None
    
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df[['transformation_type', 'base_model']] = uncertainty_df['Model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    def calculate_calibration(group):
        n_bins = 10
        group = group.copy()
        try:
            group['uncertainty_bin'] = pd.qcut(group['y_pred_std'], n_bins, labels=False, duplicates='drop')
        except (ValueError, TypeError):
            return pd.DataFrame()
        
        group['prediction_error'] = np.abs(group['y_true_noisy'] - group['y_pred_mean'])
        
        calibration = group.groupby('uncertainty_bin').agg({
            'y_pred_std': 'mean',
            'prediction_error': 'mean'
        }).reset_index()
        
        calibration.columns = ['uncertainty_bin', 'expected_uncertainty', 'observed_error']
        return calibration
    
    calibration_data = []
    for (model, sigma), group in uncertainty_df.groupby(['Model', 'Sigma']):
        if len(group) > 50:
            cal = calculate_calibration(group)
            if not cal.empty:
                cal['Model'] = model
                cal['Sigma'] = sigma
                transformation_type, base_model = parse_model_info(model)
                cal['transformation_type'] = transformation_type
                cal['base_model'] = base_model
                calibration_data.append(cal)
    
    if not calibration_data:
        print("Insufficient data for calibration curves")
        return None
    
    calibration_df = pd.concat(calibration_data)
    
    chart = alt.Chart(calibration_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('expected_uncertainty:Q', title='Expected Uncertainty',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        y=alt.Y('observed_error:Q', title='Observed Error',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational', 'conformal'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational'], COLORS['conformal']]),
                       title='Model Type',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        facet=alt.Facet('Sigma:N', title='Noise Level (σ)', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=3),
        tooltip=['Model:N', 'transformation_type:N', 'expected_uncertainty:Q', 'observed_error:Q']
    ).properties(
        width=200,
        height=170,
        title=alt.TitleParams('Figure 3c: Calibration Curves', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

def plot_figure_3d(uncertainty_df, save_path='figure_3d.png'):
    """Fig 3d: Uncertainty-error correlation bar chart with noise correlation plot"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3d")
        return None
    
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['prediction_error'] = np.abs(uncertainty_df['y_true_noisy'] - uncertainty_df['y_pred_mean'])
    uncertainty_df[['transformation_type', 'base_model']] = uncertainty_df['Model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    correlations = []
    for (model, rep, sigma), group in uncertainty_df.groupby(['Model', 'Rep', 'Sigma']):
        if len(group) > 10:
            try:
                corr, p_value = stats.pearsonr(group['y_pred_std'], group['prediction_error'])
                transformation_type, base_model = parse_model_info(model)
                correlations.append({
                    'Model': model,
                    'base_model': base_model,
                    'transformation_type': transformation_type,
                    'Rep': rep,
                    'Sigma': sigma,
                    'correlation': corr,
                    'p_value': p_value
                })
            except:
                continue
    
    if not correlations:
        print("Insufficient data for correlation analysis")
        return None
    
    corr_df = pd.DataFrame(correlations)
    corr_df['significant'] = corr_df['p_value'] < 0.05
    
    # Filter to reduce overcrowding
    corr_df = corr_df[corr_df['Sigma'].isin([0.0, 0.5, 1.0])]

    chart = alt.Chart(corr_df).mark_bar().encode(
        x=alt.X('base_model:N', title='Base Model', 
                axis=alt.Axis(labelAngle=-45, labelFontSize=9, titleFontSize=11)),
        y=alt.Y('correlation:Q', title='Uncertainty-Error Correlation',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational', 'conformal'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational'], COLORS['conformal']]),
                       title='Model Type',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        xOffset='transformation_type:N',
        facet=alt.Facet('Sigma:N', title='Noise Level (σ)', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=3),
        tooltip=['Model:N', 'transformation_type:N', 'correlation:Q', 'p_value:Q']
    ).properties(
        width=220,
        height=220,
        title=alt.TitleParams('Figure 3d: Uncertainty-Error Correlation Strength', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

# =============================================================================
# FIGURE 4: Impact of Noise Type
# =============================================================================

def plot_figure_4a(df, save_path='figure_4a.png'):
    """Fig 4a: RMSE vs. σ for different noise strategies with conformal predictions"""
    
    noise_strategies = ['legacy', 'value_proportional', 'quantile', 'threshold', 'outlier', 'heteroscedastic']
    strategy_files = []
    
    for strategy in noise_strategies:
        pattern = f'results/fig4a_{strategy}.csv'
        if os.path.exists(pattern):
            strategy_df = read_csv_safe(pattern)
            strategy_df['noise_strategy'] = strategy
            strategy_files.append(strategy_df)
    
    # Also load conformal strategy files
    for strategy in noise_strategies:
        pattern = f'results/fig4b_conformal_{strategy}.csv'
        if os.path.exists(pattern):
            strategy_df = read_csv_safe(pattern)
            strategy_df['noise_strategy'] = f'{strategy}_conformal'
            strategy_files.append(strategy_df)
    
    if not strategy_files:
        print("No noise strategy data found for Figure 4a")
        return None
    
    plot_df = pd.concat(strategy_files, ignore_index=True)
    plot_df[['transformation_type', 'base_model']] = plot_df['model'].apply(
        lambda x: pd.Series(parse_model_info(x))
    )
    
    summary_df = plot_df.groupby(['sigma', 'base_model', 'noise_strategy', 'transformation_type']).agg({
        'rmse': 'mean'
    }).reset_index()

    summary_df = create_display_labels(summary_df)
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        y=alt.Y('rmse:Q', title='RMSE',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('noise_strategy:N', title='Noise Strategy',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        strokeDash=alt.StrokeDash('transformation_type:N', title='Model Type'),
        facet=alt.Facet('base_model:N', title='Base Model', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=2),
        tooltip=['base_model:N', 'transformation_type:N', 'noise_strategy:N', 'sigma:Q', 'rmse:Q']
    ).properties(
        width=280,
        height=220,
        title=alt.TitleParams('Figure 4a: RMSE vs. Noise Level for Different Noise Strategies', 
                             fontSize=16, anchor='start')
    )
    
    chart.save(save_path)
    return chart

# =============================================================================
# FIGURE 5: Conformal Prediction and Data Size Analysis
# =============================================================================

def plot_figure_5a(conformal_df, save_path='figure_5a.png'):
    """Fig 5a: Coverage rates vs. noise level for different base models"""
    
    if conformal_df.empty:
        print("No conformal prediction data available for Figure 5a")
        return None
    
    coverage_df = conformal_df.groupby(['model_name', 'sigma_noise']).agg({
        'coverage': 'mean',
        'alpha': 'first'
    }).reset_index()
    
    coverage_df['expected_coverage'] = 1 - coverage_df['alpha']
    
    base_chart = alt.Chart(coverage_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma_noise:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y('coverage:Q', title='Empirical Coverage Rate', scale=alt.Scale(domain=[0.8, 1.0]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('model_name:N', title='Base Model',
                       legend=alt.Legend(titleFontSize=14, labelFontSize=12)),
        tooltip=['model_name:N', 'sigma_noise:Q', 'coverage:Q', 'expected_coverage:Q']
    )
    
    expected_line = alt.Chart(coverage_df).mark_line(
        strokeDash=[5, 5], color='black'
    ).encode(
        x=alt.X('sigma_noise:Q'),
        y=alt.Y('expected_coverage:Q')
    )
    
    chart = (base_chart + expected_line).properties(
        width=600,
        height=400,
        title=alt.TitleParams('Figure 5a: Conformal Prediction Coverage vs. Noise Level', 
                             fontSize=18, anchor='start')
    )
    
    chart.save(save_path)
    return chart

def plot_figure_5b(df, save_path='figure_5b.png'):
    """Fig 5b: Performance vs. data size analysis"""
    
    size_experiments = df[df['sample_size'].isin([50, 100, 200, 500])].copy()
    
    if size_experiments.empty:
        print("No data size experiments found")
        return None
    
    summary_df = size_experiments.groupby(['sample_size', 'base_model', 'transformation_type', 'sigma']).agg({
        'r2': 'mean'
    }).reset_index()
    
    key_models = ['rf', 'dnn']
    summary_df = summary_df[summary_df['base_model'].isin(key_models)]

    summary_df = create_display_labels(summary_df)
    
    if summary_df.empty:
        print("No key models found in data size analysis")
        return None
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sample_size:O', title='Sample Size',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        y=alt.Y('r2:Q', title='R²',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('transformation_type:N',
                       scale=alt.Scale(domain=['baseline', 'full', 'last_layer', 'variational', 'conformal'],
                                     range=[COLORS['baseline'], COLORS['full'], COLORS['last_layer'], 
                                           COLORS['variational'], COLORS['conformal']]),
                       title='Model Type',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        facet=alt.Facet('base_model:N', title='Base Model', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=2),
        row=alt.Row('sigma:N', title='Noise Level (σ)', 
                   header=alt.Header(titleFontSize=11, labelFontSize=10)),
        tooltip=['base_model:N', 'transformation_type:N', 'sample_size:O', 'r2:Q', 'sigma:N']
    ).properties(
        width=220,
        height=170,
        title=alt.TitleParams('Figure 5b: Performance vs. Data Size Analysis', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

def plot_figure_5c(conformal_df, save_path='figure_5c.png'):
    """Fig 5c: Conformal interval width vs. noise level"""
    
    if conformal_df.empty:
        print("No conformal prediction data available for Figure 5c")
        return None
    
    width_df = conformal_df.groupby(['model_name', 'sigma_noise']).agg({
        'interval_width': 'mean'
    }).reset_index()
    
    chart = alt.Chart(width_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma_noise:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        y=alt.Y('interval_width:Q', title='Mean Prediction Interval Width',
                axis=alt.Axis(labelFontSize=14, titleFontSize=16)),
        color=alt.Color('model_name:N', title='Base Model',
                       legend=alt.Legend(titleFontSize=14, labelFontSize=12)),
        tooltip=['model_name:N', 'sigma_noise:Q', 'interval_width:Q']
    ).properties(
        width=600,
        height=400,
        title=alt.TitleParams('Figure 5c: Conformal Prediction Interval Width vs. Noise Level', 
                             fontSize=18, anchor='start')
    )
    
    chart.save(save_path)
    return chart

# =============================================================================
# FIGURE 6: Hyperparameter Tuning Analysis
# =============================================================================

def plot_figure_6a(df, save_path='figure_6a.png'):
    """Fig 6a: Default vs. Tuned performance comparison"""
    
    df = df[df['source_file'].str.startswith('fig6a')].copy()
    
    tuned_df = df[df['is_tuned'] == True].copy()
    default_df = df[df['is_tuned'] == False].copy()
    
    if tuned_df.empty or default_df.empty:
        print("No tuned or default data available for Figure 6a")
        return None
    
    tuned_clean = tuned_df[tuned_df['sigma'] == 0.0]
    default_clean = default_df[default_df['sigma'] == 0.0]
    
    if tuned_clean.empty or default_clean.empty:
        print("No clean data available for tuning comparison")
        return None
    
    # Create summaries with proper structure
    tuned_summary = tuned_clean.groupby(['base_model', 'rep'])['r2'].mean().reset_index()
    tuned_summary['type'] = 'Tuned'
    tuned_summary.columns = ['base_model', 'rep', 'r2', 'type']
    
    default_summary = default_clean.groupby(['base_model', 'rep'])['r2'].mean().reset_index()
    default_summary['type'] = 'Default'
    default_summary.columns = ['base_model', 'rep', 'r2', 'type']
    
    # Combine for plotting
    comparison_df = pd.concat([tuned_summary, default_summary], ignore_index=True)
    
    chart = alt.Chart(comparison_df).mark_bar().encode(
        x=alt.X('base_model:N', title='Base Model', 
                axis=alt.Axis(labelAngle=-30, labelFontSize=10, titleFontSize=12)),
        y=alt.Y('r2:Q', title='R²',
                axis=alt.Axis(labelFontSize=11, titleFontSize=12)),
        color=alt.Color('type:N',
                       scale=alt.Scale(domain=['Default', 'Tuned'],
                                     range=[COLORS['baseline'], COLORS['tuned']]),
                       title='Configuration',
                       legend=alt.Legend(titleFontSize=11, labelFontSize=10)),
        xOffset='type:N',
        facet=alt.Facet('rep:N', title='Representation', 
                       header=alt.Header(titleFontSize=11, labelFontSize=10),
                       columns=3),
        tooltip=['base_model:N', 'rep:N', 'type:N', 'r2:Q']
    ).properties(
        width=140,
        height=170,
        title=alt.TitleParams('Figure 6a: Default vs. Hyperparameter Tuned Performance', 
                             fontSize=16, anchor='start')
    ).resolve_scale(color='shared')
    
    chart.save(save_path)
    return chart

def create_tuning_comparison_table(df, output_dir='figures'):
    """Create comprehensive tuning comparison table"""
    
    tuned_df = df[df['is_tuned'] == True].copy()
    default_df = df[df['is_tuned'] == False].copy()
    
    if tuned_df.empty or default_df.empty:
        print("No tuning data available for comparison table")
        return None
    
    tuned_clean = tuned_df[tuned_df['sigma'] == 0.0]
    default_clean = default_df[default_df['sigma'] == 0.0]
    
    # Create proper DataFrames with reset_index
    tuned_summary = tuned_clean.groupby(['base_model', 'rep'])['r2'].mean().reset_index()
    tuned_summary.columns = ['base_model', 'rep', 'r2_tuned']
    
    default_summary = default_clean.groupby(['base_model', 'rep'])['r2'].mean().reset_index()
    default_summary.columns = ['base_model', 'rep', 'r2_default']
    
    # Merge the two DataFrames
    comparison = tuned_summary.merge(
        default_summary, 
        on=['base_model', 'rep'], 
        how='inner'
    )
    
    comparison['improvement'] = comparison['r2_tuned'] - comparison['r2_default']
    
    # Sort by absolute improvement (not percentage)
    comparison = comparison.dropna()
    comparison = comparison.sort_values('improvement', ascending=False)
    
    table_path = os.path.join(output_dir, 'tuning_comparison_table.csv')
    comparison.to_csv(table_path, index=False, float_format='%.4f')
    
    summary_path = os.path.join(output_dir, 'tuning_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Hyperparameter Tuning Performance Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Average R² improvement: {comparison['improvement'].mean():.4f}\n")
        f.write(f"Median R² improvement: {comparison['improvement'].median():.4f}\n")
        f.write(f"Best improvement: {comparison['improvement'].max():.4f} ")
        f.write(f"({comparison.loc[comparison['improvement'].idxmax(), 'base_model']}-")
        f.write(f"{comparison.loc[comparison['improvement'].idxmax(), 'rep']})\n")
        f.write(f"Worst improvement: {comparison['improvement'].min():.4f} ")
        f.write(f"({comparison.loc[comparison['improvement'].idxmin(), 'base_model']}-")
        f.write(f"{comparison.loc[comparison['improvement'].idxmin(), 'rep']})\n\n")
        f.write("Top 10 improvements (by absolute R² gain):\n")
        f.write("-" * 60 + "\n")
        for idx, row in comparison.head(10).iterrows():
            f.write(f"{row['base_model']:20s} {row['rep']:20s} ")
            f.write(f"{row['r2_default']:+.4f} → {row['r2_tuned']:+.4f} ")
            f.write(f"(Δ = {row['improvement']:+.4f})\n")
    
    print(f"Tuning comparison table saved to {table_path}")
    print(f"Tuning summary saved to {summary_path}")
    
    return comparison

# =============================================================================
# ADDITIONAL ANALYSIS FUNCTIONS
# =============================================================================

def plot_wilcoxon_analysis(df, save_path='wilcoxon_analysis.png'):
    """Statistical significance analysis using Wilcoxon signed-rank tests"""
    
    pairs_to_compare = [
        ('rf', 'qrf'),
        ('dnn', 'dnn_bnn_full'),
        ('gin', 'gin_bnn_full')
    ]
    
    significance_results = []
    
    for det_model, bay_model in pairs_to_compare:
        for sigma in df['sigma'].unique():
            det_data = df[(df['model'] == det_model) & (df['sigma'] == sigma)]['r2']
            bay_data = df[(df['model'] == bay_model) & (df['sigma'] == sigma)]['r2']
            
            if len(det_data) > 0 and len(bay_data) > 0 and len(det_data) == len(bay_data):
                try:
                    stat, p_value = stats.wilcoxon(det_data, bay_data, alternative='two-sided')
                    significance_results.append({
                        'pair': f'{det_model} vs {bay_model}',
                        'sigma': sigma,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'difference': bay_data.mean() - det_data.mean()
                    })
                except:
                    continue
    
    if not significance_results:
        print("No statistical comparisons could be performed")
        return None
    
    sig_df = pd.DataFrame(significance_results)
    
    chart = alt.Chart(sig_df).mark_rect().encode(
        x=alt.X('sigma:O', title='Noise Level (σ)',
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        y=alt.Y('pair:N', title='Model Pair',
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        color=alt.Color('p_value:Q', 
                       scale=alt.Scale(scheme='redyellowblue', reverse=True),
                       title='p-value',
                       legend=alt.Legend(titleFontSize=12, labelFontSize=11)),
        tooltip=['pair:N', 'sigma:O', 'p_value:Q', 'difference:Q']
    ).properties(
        width=400,
        height=250,
        title=alt.TitleParams('Statistical Significance Analysis (Wilcoxon Tests)', 
                             fontSize=16, anchor='start')
    )
    
    chart.save(save_path)
    return chart

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_all_figures(results_dir='results', output_dir='figures'):
    """Generate all figures from experimental data"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experimental data...")
    df = load_results_data(results_dir)
    uncertainty_df = load_uncertainty_data(results_dir)
    per_epoch_df = load_per_epoch_data(results_dir)
    conformal_df = load_conformal_data(results_dir)
    
    print(f"Loaded {len(df)} main results")
    print(f"Loaded {len(uncertainty_df)} uncertainty records")
    print(f"Loaded {len(per_epoch_df)} per-epoch records")
    print(f"Loaded {len(conformal_df)} conformal records")
    
    print("\nGenerating figures...")
    
    # Figure 1
    plot_figure_1a(df, os.path.join(output_dir, 'figure_1a.png'))
    plot_figure_1b(df, os.path.join(output_dir, 'figure_1b.png'))
    plot_figure_1c(df, os.path.join(output_dir, 'figure_1c.png'))
    plot_figure_1d(df, os.path.join(output_dir, 'figure_1d.png'))
    
    # Figure 2
    plot_figure_2a(per_epoch_df, os.path.join(output_dir, 'figure_2a.png'))
    plot_figure_2b(df, os.path.join(output_dir, 'figure_2b.png'))
    
    # Figure 3
    plot_figure_3a(uncertainty_df, os.path.join(output_dir, 'figure_3a.png'))
    plot_figure_3b(uncertainty_df, os.path.join(output_dir, 'figure_3b.png'))
    plot_figure_3c(uncertainty_df, os.path.join(output_dir, 'figure_3c.png'))
    plot_figure_3d(uncertainty_df, os.path.join(output_dir, 'figure_3d.png'))
    
    # Figure 4
    plot_figure_4a(df, os.path.join(output_dir, 'figure_4a.png'))
    
    # Figure 5
    plot_figure_5a(conformal_df, os.path.join(output_dir, 'figure_5a.png'))
    plot_figure_5b(df, os.path.join(output_dir, 'figure_5b.png'))
    plot_figure_5c(conformal_df, os.path.join(output_dir, 'figure_5c.png'))
    
    # Figure 6
    plot_figure_6a(df, os.path.join(output_dir, 'figure_6a.png'))
    create_tuning_comparison_table(df, output_dir)
    
    # Additional analyses
    plot_wilcoxon_analysis(df, os.path.join(output_dir, 'wilcoxon_analysis.png'))
    
    print(f"\nAll figures saved to {output_dir}/")

def test_parse_function():
    test_cases = [
        'mlp_full',
        'mlp_last', 
        'mlp_variational',
        'residual_mlp_full',
        'bnn_full',
        'dnn',
        'conformal_rf_split',
        'rf'
    ]
    print("\n=== TESTING PARSE FUNCTION ===")
    for name in test_cases:
        trans, base = parse_model_info(name)
        print(f"{name:25s} -> trans: {trans:15s}, base: {base}")
    print()

test_parse_function()

if __name__ == "__main__":
    try:
        import subprocess
        subprocess.check_call(['pip', 'install', 'vl-convert-python'], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except:
        pass
    
    create_all_figures()