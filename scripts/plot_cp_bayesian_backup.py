#!/usr/bin/env python3
"""
Complete plotting framework for QSAR/QSPR noise robustness experiments
Generates Figures 1-5 using Altair for publication-quality output
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
    'deterministic': '#1f77b4',  # Blue
    'bayesian': '#ff7f0e',      # Orange  
    'conformal': '#2ca02c',     # Green
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

# Model categorization
DETERMINISTIC_MODELS = ['rf', 'xgboost', 'svm', 'dnn', 'mlp', 'residual_mlp', 
                       'factorization_mlp', 'gin', 'gcn']
BAYESIAN_MODELS = ['qrf', 'ngboost', 'gauche', 'graph_gp'] + \
                 [f'{model}_bnn_{variant}' for model in ['dnn', 'mlp', 'residual_mlp', 
                  'factorization_mlp', 'gin', 'gcn'] for variant in ['full', 'last', 'variational']]
CONFORMAL_MODELS = [f'conformal_{model}' for model in ['rf', 'xgboost', 'qrf', 'dnn', 
                    'gauche', 'gin', 'gcn']]

def load_results_data(results_dir='results'):
    """Load and combine all main results CSV files"""
    pattern = os.path.join(results_dir, 'fig*.csv')
    files = glob.glob(pattern)
    
    if not files:
        raise FileNotFoundError(f"No results files found in {results_dir}")
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        df['source_file'] = os.path.basename(file)
        dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Add model category
    def categorize_model(model):
        if any(conf_model in model for conf_model in ['conformal']):
            return 'conformal'
        elif model in BAYESIAN_MODELS or 'bnn' in model:
            return 'bayesian'
        else:
            return 'deterministic'
    
    combined_df['model_category'] = combined_df['model'].apply(categorize_model)
    return combined_df

def load_uncertainty_data(results_dir='results'):
    """Load uncertainty data from uncertainty CSV files"""
    uncertainty_dir = os.path.join(results_dir, '*_uncertainty')
    pattern = os.path.join(uncertainty_dir, 'uncertainty_*.csv')
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_conformal_data(results_dir='results'):
    """Load conformal prediction interval data"""
    conformal_dir = os.path.join(results_dir, 'conformal_intervals')
    pattern = os.path.join(conformal_dir, 'conformal_intervals_*.csv')
    files = glob.glob(pattern, recursive=True)
    
    if not files:
        return pd.DataFrame()
    
    dfs = []
    for file in files:
        df = pd.read_csv(file)
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
        df = pd.read_csv(file)
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

# =============================================================================
# FIGURE 1: Model Robustness to Noise
# =============================================================================

def plot_figure_1a(df, save_path='figure_1a.png'):
    """Fig 1a: RMSE vs. σ for bit-vector representations"""
    
    # Filter for bit-vector models and representations
    bit_vector_reps = ['ecfp4', 'pdv', 'sns', 'smiles', 'randomized_smiles']
    bit_vector_models = ['rf', 'qrf', 'svm', 'xgboost', 'ngboost', 'gauche']
    
    plot_df = df[
        (df['rep'].isin(bit_vector_reps)) & 
        (df['model'].isin(bit_vector_models))
    ].copy()
    
    if plot_df.empty:
        print("No data found for Figure 1a")
        return None
    
    # Calculate mean and std across iterations
    summary_df = plot_df.groupby(['sigma', 'model', 'rep']).agg({
        'rmse': ['mean', 'std'],
        'r2': ['mean', 'std']
    }).reset_index()
    
    summary_df.columns = ['sigma', 'model', 'rep', 'rmse_mean', 'rmse_std', 'r2_mean', 'r2_std']
    
    # Create the plot with lines only (no error bands for simplicity)
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('rmse_mean:Q', title='RMSE'),
        color=alt.Color('model:N', title='Model'),
        facet=alt.Facet('rep:N', title='Molecular Representation'),
        tooltip=['model:N', 'rep:N', 'sigma:Q', 'rmse_mean:Q', 'rmse_std:Q']
    ).properties(
        width=120,
        height=120,
        title='Figure 1a: RMSE vs. Noise Level for Bit-Vector Representations'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_1b(df, save_path='figure_1b.png'):
    """Fig 1b: RMSE vs. σ for graph representations"""
    
    graph_models = ['gin', 'gcn', 'graph_gp']
    plot_df = df[
        (df['rep'] == 'graph') & 
        (df['model'].isin(graph_models))
    ].copy()
    
    if plot_df.empty:
        print("No data found for Figure 1b")
        return None
    
    # Calculate mean and std across iterations
    summary_df = plot_df.groupby(['sigma', 'model']).agg({
        'rmse': ['mean', 'std']
    }).reset_index()
    
    summary_df.columns = ['sigma', 'model', 'rmse_mean', 'rmse_std']
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('rmse_mean:Q', title='RMSE'),
        color=alt.Color('model:N', 
                       scale=alt.Scale(range=['#1f77b4', '#ff7f0e', '#2ca02c']), 
                       title='Graph Model'),
        tooltip=['model:N', 'sigma:Q', 'rmse_mean:Q', 'rmse_std:Q']
    ).properties(
        width=400,
        height=300,
        title='Figure 1b: RMSE vs. Noise Level for Graph Representations'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_1c(df, save_path='figure_1c.png'):
    """Fig 1c: RMSE vs. σ for NN vs BNN"""
    
    nn_models = ['dnn', 'mlp', 'residual_mlp', 'factorization_mlp', 'gin', 'gcn']
    bnn_models = [f'{model}_bnn_full' for model in nn_models] + \
                 [f'{model}_bnn_last' for model in nn_models] + \
                 [f'{model}_bnn_variational' for model in nn_models]
    
    plot_df = df[df['model'].isin(nn_models + bnn_models)].copy()
    
    if plot_df.empty:
        print("No data found for Figure 1c")
        return None
    
    # Add model type
    plot_df['model_type'] = plot_df['model'].apply(
        lambda x: 'Bayesian' if 'bnn' in x else 'Deterministic'
    )
    
    # Calculate mean and std across iterations
    summary_df = plot_df.groupby(['sigma', 'model', 'model_type', 'rep']).agg({
        'rmse': ['mean', 'std']
    }).reset_index()
    
    summary_df.columns = ['sigma', 'model', 'model_type', 'rep', 'rmse_mean', 'rmse_std']
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('rmse_mean:Q', title='RMSE'),
        color=alt.Color('model_type:N', 
                       scale=alt.Scale(domain=['Deterministic', 'Bayesian'],
                                     range=['#1f77b4', '#ff7f0e']), 
                       title='Model Type'),
        strokeDash=alt.StrokeDash('model:N', title='Model'),
        facet=alt.Facet('rep:N', title='Representation', columns=3),
        tooltip=['model:N', 'model_type:N', 'sigma:Q', 'rmse_mean:Q']
    ).properties(
        width=120,
        height=100,
        title='Figure 1c: Neural Network vs. Bayesian Neural Network Robustness'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_1d(df, save_path='figure_1d.png'):
    """Fig 1d: R² vs. σ for top model-representation pairs"""
    
    # Calculate mean R² for each model-rep combination at sigma=0
    baseline_performance = df[df['sigma'] == 0.0].groupby(['model', 'rep'])['r2'].mean()
    
    if baseline_performance.empty:
        print("No baseline data (sigma=0) found for Figure 1d")
        return None
    
    top_pairs = baseline_performance.nlargest(15).index.tolist()  # Top 15 pairs
    
    plot_df = df[df.set_index(['model', 'rep']).index.isin(top_pairs)].copy()
    
    # Calculate mean and std across iterations
    summary_df = plot_df.groupby(['sigma', 'model', 'rep']).agg({
        'r2': ['mean', 'std']
    }).reset_index()
    
    summary_df.columns = ['sigma', 'model', 'rep', 'r2_mean', 'r2_std']
    summary_df['model_rep'] = summary_df['model'] + '_' + summary_df['rep']
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('r2_mean:Q', title='R²', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('model_rep:N', title='Model-Representation'),
        tooltip=['model:N', 'rep:N', 'sigma:Q', 'r2_mean:Q']
    ).properties(
        width=500,
        height=350,
        title='Figure 1d: R² vs. Noise Level for Top Model-Representation Pairs'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

# =============================================================================
# FIGURE 2: Bayesian vs. Deterministic
# =============================================================================

def plot_figure_2a(per_epoch_df, save_path='figure_2a.png'):
    """Fig 2a: Training curves comparing NN vs BNN"""
    
    if per_epoch_df.empty:
        print("No per-epoch data available for Figure 2a")
        return None
    
    # Filter for relevant models
    models_of_interest = ['dnn', 'dnn_bnn_full', 'gin', 'gin_bnn_full']
    plot_df = per_epoch_df[per_epoch_df['model'].isin(models_of_interest)].copy()
    
    if plot_df.empty:
        print("No relevant models found in per-epoch data for Figure 2a")
        return None
    
    # Add model type
    plot_df['model_type'] = plot_df['model'].apply(
        lambda x: 'Bayesian' if 'bnn' in x else 'Deterministic'
    )
    
    # Calculate mean across iterations
    summary_df = plot_df.groupby(['epoch', 'model', 'model_type', 'sigma']).agg({
        'train_loss': 'mean',
        'val_loss': 'mean'
    }).reset_index()
    
    # Reshape for plotting
    plot_data = pd.melt(summary_df, 
                       id_vars=['epoch', 'model', 'model_type', 'sigma'],
                       value_vars=['train_loss', 'val_loss'],
                       var_name='loss_type', value_name='loss')
    
    chart = alt.Chart(plot_data).mark_line(strokeWidth=2).encode(
        x=alt.X('epoch:Q', title='Epoch'),
        y=alt.Y('loss:Q', title='Loss'),
        color=alt.Color('model_type:N', 
                       scale=alt.Scale(domain=['Deterministic', 'Bayesian'],
                                     range=['#1f77b4', '#ff7f0e']),
                       title='Model Type'),
        strokeDash=alt.StrokeDash('loss_type:N', title='Loss Type'),
        facet=alt.Facet('sigma:N', title='Noise Level (σ)', columns=3),
        tooltip=['model:N', 'epoch:Q', 'loss:Q', 'loss_type:N']
    ).properties(
        width=150,
        height=100,
        title='Figure 2a: Training Curves - Neural Networks vs. Bayesian Neural Networks'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_2b(df, save_path='figure_2b.png'):
    """Fig 2b: Final R² at fixed σ levels"""
    
    # Filter for specific sigma levels and model pairs
    sigma_levels = [0.5]  # Using sigma from your script
    model_pairs = [
        ('rf', 'qrf'),
        ('dnn', 'dnn_bnn_full')
    ]
    
    plot_df = df[df['sigma'].isin(sigma_levels)].copy()
    
    if plot_df.empty:
        print("No data found for specified sigma levels in Figure 2b")
        return None
    
    # Filter for specific models
    all_models = [model for pair in model_pairs for model in pair]
    plot_df = plot_df[plot_df['model'].isin(all_models)]
    
    if plot_df.empty:
        print("No data found for specified models in Figure 2b")
        return None
    
    # Add model type
    plot_df['model_type'] = plot_df['model'].apply(
        lambda x: 'Bayesian' if x in ['qrf', 'dnn_bnn_full'] else 'Deterministic'
    )
    
    # Calculate mean R² across iterations
    summary_df = plot_df.groupby(['sigma', 'model', 'model_type'])['r2'].mean().reset_index()
    
    chart = alt.Chart(summary_df).mark_bar().encode(
        x=alt.X('model:N', title='Model'),
        y=alt.Y('r2:Q', title='R²'),
        color=alt.Color('model_type:N',
                       scale=alt.Scale(domain=['Deterministic', 'Bayesian'],
                                     range=['#1f77b4', '#ff7f0e']),
                       title='Model Type'),
        tooltip=['model:N', 'r2:Q', 'sigma:N']
    ).properties(
        width=300,
        height=200,
        title='Figure 2b: Final R² at Fixed Noise Levels'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

# =============================================================================
# FIGURE 3: Uncertainty and Label Noise
# =============================================================================

def plot_figure_3a(uncertainty_df, save_path='figure_3a.png'):
    """Fig 3a: Uncertainty vs. prediction error scatter plots"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3a")
        return None
    
    # Calculate prediction error
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['prediction_error'] = np.abs(uncertainty_df['y_true_noisy'] - uncertainty_df['y_pred_mean'])
    
    # Filter for relevant models
    uncertainty_models = ['qrf', 'gauche', 'ngboost'] + \
                        [f'{model}_bnn_full' for model in ['dnn', 'mlp']]
    plot_df = uncertainty_df[uncertainty_df['Model'].isin(uncertainty_models)].copy()
    
    if plot_df.empty:
        print("No relevant uncertainty models found for Figure 3a")
        return None
    
    # Sample data for plotting (too many points otherwise)
    sampled_dfs = []
    for (model, sigma), group in plot_df.groupby(['Model', 'Sigma']):
        sample_size = min(500, len(group))
        sampled = group.sample(n=sample_size, random_state=42)
        sampled_dfs.append(sampled)
    plot_df = pd.concat(sampled_dfs)
    
    chart = alt.Chart(plot_df).mark_circle(size=20, opacity=0.6).encode(
        x=alt.X('y_pred_std:Q', title='Predicted Uncertainty'),
        y=alt.Y('prediction_error:Q', title='Prediction Error'),
        color=alt.Color('Model:N', title='Model'),
        facet=alt.Facet('Sigma:N', title='Noise Level (σ)', columns=3),
        tooltip=['Model:N', 'y_pred_std:Q', 'prediction_error:Q']
    ).properties(
        width=150,
        height=150,
        title='Figure 3a: Uncertainty vs. Prediction Error'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_3b(uncertainty_df, save_path='figure_3b.png'):
    """Fig 3b: Uncertainty boxplots for clean vs. noisy data"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3b")
        return None
    
    # Create clean vs noisy labels
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['data_condition'] = uncertainty_df['Sigma'].apply(
        lambda x: 'Clean' if x == 0.0 else 'Noisy'
    )
    
    # Filter for relevant models
    uncertainty_models = ['qrf', 'gauche', 'ngboost', 'dnn_bnn_full']
    plot_df = uncertainty_df[uncertainty_df['Model'].isin(uncertainty_models)].copy()
    
    if plot_df.empty:
        print("No relevant models found for Figure 3b")
        return None
    
    chart = alt.Chart(plot_df).mark_boxplot().encode(
        x=alt.X('Model:N', title='Model'),
        y=alt.Y('y_pred_std:Q', title='Predicted Uncertainty'),
        color=alt.Color('data_condition:N',
                       scale=alt.Scale(domain=['Clean', 'Noisy'],
                                     range=['#1f77b4', '#ff7f0e']),
                       title='Data Condition'),
        tooltip=['Model:N', 'data_condition:N']
    ).properties(
        width=400,
        height=300,
        title='Figure 3b: Uncertainty Distribution - Clean vs. Noisy Data'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_3c(uncertainty_df, save_path='figure_3c.png'):
    """Fig 3c: Calibration curves"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3c")
        return None
    
    # Calculate calibration for each model
    def calculate_calibration(group):
        n_bins = 10
        group = group.copy()
        try:
            group['uncertainty_bin'] = pd.qcut(group['y_pred_std'], n_bins, labels=False, duplicates='drop')
        except ValueError:
            return pd.DataFrame()
        
        group['prediction_error'] = np.abs(group['y_true_noisy'] - group['y_pred_mean'])
        
        calibration = group.groupby('uncertainty_bin').agg({
            'y_pred_std': 'mean',
            'prediction_error': 'mean'
        }).reset_index()
        
        calibration.columns = ['uncertainty_bin', 'expected_uncertainty', 'observed_error']
        return calibration
    
    # Filter for relevant models
    calibration_models = ['qrf', 'gauche', 'ngboost', 'dnn_bnn_full']
    plot_df = uncertainty_df[uncertainty_df['Model'].isin(calibration_models)].copy()
    
    if plot_df.empty:
        print("No relevant models found for Figure 3c")
        return None
    
    # Calculate calibration for each model-sigma combination
    calibration_data = []
    for (model, sigma), group in plot_df.groupby(['Model', 'Sigma']):
        if len(group) > 50:
            cal = calculate_calibration(group)
            if not cal.empty:
                cal['Model'] = model
                cal['Sigma'] = sigma
                calibration_data.append(cal)
    
    if not calibration_data:
        print("Insufficient data for calibration curves")
        return None
    
    calibration_df = pd.concat(calibration_data)
    
    # Create calibration chart without layering
    chart = alt.Chart(calibration_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('expected_uncertainty:Q', title='Expected Uncertainty'),
        y=alt.Y('observed_error:Q', title='Observed Error'),
        color=alt.Color('Model:N', title='Model'),
        facet=alt.Facet('Sigma:N', title='Noise Level (σ)', columns=3),
        tooltip=['Model:N', 'expected_uncertainty:Q', 'observed_error:Q']
    ).properties(
        width=150,
        height=150,
        title='Figure 3c: Calibration Curves'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_3d(uncertainty_df, save_path='figure_3d.png'):
    """Fig 3d: Uncertainty-error correlation bar chart"""
    
    if uncertainty_df.empty:
        print("No uncertainty data available for Figure 3d")
        return None
    
    # Calculate correlations
    uncertainty_df = uncertainty_df.copy()
    uncertainty_df['prediction_error'] = np.abs(uncertainty_df['y_true_noisy'] - uncertainty_df['y_pred_mean'])
    
    correlations = []
    for (model, rep, sigma), group in uncertainty_df.groupby(['Model', 'Rep', 'Sigma']):
        if len(group) > 10:
            try:
                corr, p_value = stats.pearsonr(group['y_pred_std'], group['prediction_error'])
                correlations.append({
                    'Model': model,
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
    
    chart = alt.Chart(corr_df).mark_bar().encode(
        x=alt.X('Model:N', title='Model'),
        y=alt.Y('correlation:Q', title='Uncertainty-Error Correlation'),
        color=alt.Color('significant:N',
                       scale=alt.Scale(domain=[True, False],
                                     range=['#2ca02c', '#d62728']),
                       title='Significant (p<0.05)'),
        facet=alt.Facet('Sigma:N', title='Noise Level (σ)', columns=3),
        tooltip=['Model:N', 'correlation:Q', 'p_value:Q']
    ).properties(
        width=150,
        height=200,
        title='Figure 3d: Uncertainty-Error Correlation Strength'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

# =============================================================================
# FIGURE 4: Impact of Noise Type
# =============================================================================

def plot_figure_4a(df, save_path='figure_4a.png'):
    """Fig 4a: RMSE vs. σ for different noise strategies"""
    
    # Look for files with noise strategy naming
    noise_strategies = ['legacy', 'value_proportional', 'quantile', 'threshold', 'outlier', 'heteroscedastic']
    strategy_files = []
    
    for strategy in noise_strategies:
        pattern = f'results/fig4a_{strategy}.csv'
        if os.path.exists(pattern):
            strategy_df = pd.read_csv(pattern)
            strategy_df['noise_strategy'] = strategy
            strategy_files.append(strategy_df)
    
    if not strategy_files:
        print("No noise strategy data found for Figure 4a")
        return None
    
    plot_df = pd.concat(strategy_files, ignore_index=True)
    
    # Calculate mean across iterations
    summary_df = plot_df.groupby(['sigma', 'model', 'noise_strategy']).agg({
        'rmse': 'mean'
    }).reset_index()
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('rmse:Q', title='RMSE'),
        color=alt.Color('noise_strategy:N', title='Noise Strategy'),
        tooltip=['noise_strategy:N', 'sigma:Q', 'rmse:Q']
    ).properties(
        width=500,
        height=350,
        title='Figure 4a: RMSE vs. Noise Level for Different Noise Strategies'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

# =============================================================================
# FIGURE 5: Conformal Prediction Robustness
# =============================================================================

def plot_figure_5a(conformal_df, save_path='figure_5a.png'):
    """Fig 5a: Coverage rates vs. noise level for different base models"""
    
    if conformal_df.empty:
        print("No conformal prediction data available for Figure 5a")
        return None
    
    # Calculate coverage rates by model and sigma
    coverage_df = conformal_df.groupby(['model_name', 'sigma_noise']).agg({
        'coverage': 'mean',
        'alpha': 'first'
    }).reset_index()
    
    coverage_df['expected_coverage'] = 1 - coverage_df['alpha']
    
    chart = alt.Chart(coverage_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma_noise:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('coverage:Q', title='Empirical Coverage Rate', scale=alt.Scale(domain=[0.8, 1.0])),
        color=alt.Color('model_name:N', title='Base Model'),
        tooltip=['model_name:N', 'sigma_noise:Q', 'coverage:Q', 'expected_coverage:Q']
    ).properties(
        width=500,
        height=350,
        title='Figure 5a: Conformal Prediction Coverage vs. Noise Level'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_5b(conformal_df, save_path='figure_5b.png'):
    """Fig 5b: Interval width vs. noise level"""
    
    if conformal_df.empty:
        print("No conformal prediction data available for Figure 5b")
        return None
    
    # Calculate mean interval width by model and sigma
    width_df = conformal_df.groupby(['model_name', 'sigma_noise']).agg({
        'interval_width': 'mean'
    }).reset_index()
    
    chart = alt.Chart(width_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sigma_noise:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('interval_width:Q', title='Mean Prediction Interval Width'),
        color=alt.Color('model_name:N', title='Base Model'),
        tooltip=['model_name:N', 'sigma_noise:Q', 'interval_width:Q']
    ).properties(
        width=500,
        height=350,
        title='Figure 5b: Conformal Prediction Interval Width vs. Noise Level'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_figure_5c(df, conformal_df, save_path='figure_5c.png'):
    """Fig 5c: Conformal + Bayesian vs. Conformal + Deterministic comparison"""
    
    if conformal_df.empty:
        print("No conformal prediction data available for Figure 5c")
        return None
    
    # Separate conformal models by base model type
    conformal_bayesian = ['conformal_qrf', 'conformal_gauche']
    conformal_deterministic = ['conformal_rf', 'conformal_xgboost', 'conformal_dnn']
    
    conformal_filtered = conformal_df[
        conformal_df['model_name'].isin(conformal_bayesian + conformal_deterministic)
    ].copy()
    
    if conformal_filtered.empty:
        print("No conformal Bayesian/deterministic comparison data found for Figure 5c")
        return None
    
    conformal_filtered['base_type'] = conformal_filtered['model_name'].apply(
        lambda x: 'Conformal + Bayesian' if x in conformal_bayesian else 'Conformal + Deterministic'
    )
    
    # Calculate performance metrics
    performance_df = conformal_filtered.groupby(['base_type', 'sigma_noise']).agg({
        'coverage': 'mean',
        'interval_width': 'mean'
    }).reset_index()
    
    # Coverage comparison
    coverage_chart = alt.Chart(performance_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('sigma_noise:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('coverage:Q', title='Coverage Rate'),
        color=alt.Color('base_type:N', 
                       scale=alt.Scale(domain=['Conformal + Deterministic', 'Conformal + Bayesian'],
                                     range=['#1f77b4', '#ff7f0e']),
                       title='Approach'),
        tooltip=['base_type:N', 'sigma_noise:Q', 'coverage:Q']
    ).properties(
        width=200,
        height=150,
        title='Coverage Rate Comparison'
    )
    
    # Width comparison
    width_chart = alt.Chart(performance_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('sigma_noise:Q', title='Noise Level (σ)', scale=alt.Scale(domain=[0, 1])),
        y=alt.Y('interval_width:Q', title='Interval Width'),
        color=alt.Color('base_type:N', 
                       scale=alt.Scale(domain=['Conformal + Deterministic', 'Conformal + Bayesian'],
                                     range=['#1f77b4', '#ff7f0e']),
                       title='Approach'),
        tooltip=['base_type:N', 'sigma_noise:Q', 'interval_width:Q']
    ).properties(
        width=200,
        height=150,
        title='Interval Width Comparison'
    )
    
    # Combine charts horizontally
    combined_chart = alt.hconcat(coverage_chart, width_chart).resolve_scale(
        color='shared'
    )
    
    try:
        combined_chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        combined_chart.save(save_path.replace('.png', '.json'))
    
    return combined_chart

# =============================================================================
# ADDITIONAL ANALYSIS FUNCTIONS
# =============================================================================

def plot_data_size_analysis(df, save_path='data_size_analysis.png'):
    """Additional plot: Performance vs. data size analysis"""
    
    # Filter for data size experiments
    size_experiments = df[df['sample_size'].isin([50, 100, 200, 500])].copy()
    
    if size_experiments.empty:
        print("No data size experiments found")
        return None
    
    # Calculate mean performance across iterations
    summary_df = size_experiments.groupby(['sample_size', 'model', 'sigma']).agg({
        'r2': 'mean'
    }).reset_index()
    
    # Focus on key models
    key_models = ['rf', 'qrf', 'dnn', 'dnn_bnn_full']
    summary_df = summary_df[summary_df['model'].isin(key_models)]
    
    if summary_df.empty:
        print("No key models found in data size analysis")
        return None
    
    chart = alt.Chart(summary_df).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X('sample_size:O', title='Sample Size'),
        y=alt.Y('r2:Q', title='R²'),
        color=alt.Color('model:N', title='Model'),
        facet=alt.Facet('sigma:N', title='Noise Level (σ)', columns=3),
        tooltip=['model:N', 'sample_size:O', 'r2:Q', 'sigma:N']
    ).properties(
        width=150,
        height=150,
        title='Performance vs. Data Size Analysis'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

def plot_wilcoxon_analysis(df, save_path='wilcoxon_analysis.png'):
    """Statistical significance analysis using Wilcoxon signed-rank tests"""
    
    # Compare Bayesian vs Deterministic pairs
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
    
    # Create significance heatmap
    chart = alt.Chart(sig_df).mark_rect().encode(
        x=alt.X('sigma:O', title='Noise Level (σ)'),
        y=alt.Y('pair:N', title='Model Pair'),
        color=alt.Color('p_value:Q', 
                       scale=alt.Scale(scheme='redyellowblue', reverse=True),
                       title='p-value'),
        tooltip=['pair:N', 'sigma:O', 'p_value:Q', 'difference:Q']
    ).properties(
        width=300,
        height=200,
        title='Statistical Significance Analysis (Wilcoxon Tests)'
    )
    
    try:
        chart.save(save_path)
    except Exception as e:
        print(f"Saving as JSON instead due to: {e}")
        chart.save(save_path.replace('.png', '.json'))
    
    return chart

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_all_figures(results_dir='results', output_dir='figures'):
    """Generate all figures from experimental data"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all data
    print("Loading experimental data...")
    try:
        df = load_results_data(results_dir)
        print(f"Loaded {len(df)} main results")
    except Exception as e:
        print(f"Failed to load main results: {e}")
        return
    
    uncertainty_df = load_uncertainty_data(results_dir)
    conformal_df = load_conformal_data(results_dir)
    per_epoch_df = load_per_epoch_data(results_dir)
    
    print(f"Loaded {len(uncertainty_df)} uncertainty records")
    print(f"Loaded {len(conformal_df)} conformal prediction records")
    print(f"Loaded {len(per_epoch_df)} per-epoch records")
    
    # Generate Figure 1: Model Robustness to Noise
    print("\nGenerating Figure 1: Model Robustness to Noise...")
    try:
        plot_figure_1a(df, os.path.join(output_dir, 'figure_1a_bit_vector_robustness.png'))
        print("✓ Figure 1a: Bit-vector representations")
    except Exception as e:
        print(f"✗ Figure 1a failed: {e}")
    
    try:
        plot_figure_1b(df, os.path.join(output_dir, 'figure_1b_graph_robustness.png'))
        print("✓ Figure 1b: Graph representations")
    except Exception as e:
        print(f"✗ Figure 1b failed: {e}")
    
    try:
        plot_figure_1c(df, os.path.join(output_dir, 'figure_1c_nn_vs_bnn.png'))
        print("✓ Figure 1c: NN vs BNN")
    except Exception as e:
        print(f"✗ Figure 1c failed: {e}")
    
    try:
        plot_figure_1d(df, os.path.join(output_dir, 'figure_1d_top_pairs.png'))
        print("✓ Figure 1d: Top model-representation pairs")
    except Exception as e:
        print(f"✗ Figure 1d failed: {e}")
    
    # Generate Figure 2: Bayesian vs. Deterministic
    print("\nGenerating Figure 2: Bayesian vs. Deterministic...")
    try:
        plot_figure_2a(per_epoch_df, os.path.join(output_dir, 'figure_2a_training_curves.png'))
        print("✓ Figure 2a: Training curves")
    except Exception as e:
        print(f"✗ Figure 2a failed: {e}")
    
    try:
        plot_figure_2b(df, os.path.join(output_dir, 'figure_2b_fixed_sigma_comparison.png'))
        print("✓ Figure 2b: Fixed sigma comparison")
    except Exception as e:
        print(f"✗ Figure 2b failed: {e}")
    
    # Generate Figure 3: Uncertainty and Label Noise
    print("\nGenerating Figure 3: Uncertainty and Label Noise...")
    try:
        plot_figure_3a(uncertainty_df, os.path.join(output_dir, 'figure_3a_uncertainty_vs_error.png'))
        print("✓ Figure 3a: Uncertainty vs error scatter")
    except Exception as e:
        print(f"✗ Figure 3a failed: {e}")
    
    try:
        plot_figure_3b(uncertainty_df, os.path.join(output_dir, 'figure_3b_clean_vs_noisy_uncertainty.png'))
        print("✓ Figure 3b: Clean vs noisy uncertainty")
    except Exception as e:
        print(f"✗ Figure 3b failed: {e}")
    
    try:
        plot_figure_3c(uncertainty_df, os.path.join(output_dir, 'figure_3c_calibration_curves.png'))
        print("✓ Figure 3c: Calibration curves")
    except Exception as e:
        print(f"✗ Figure 3c failed: {e}")
    
    try:
        plot_figure_3d(uncertainty_df, os.path.join(output_dir, 'figure_3d_uncertainty_error_correlation.png'))
        print("✓ Figure 3d: Uncertainty-error correlation")
    except Exception as e:
        print(f"✗ Figure 3d failed: {e}")
    
    # Generate Figure 4: Impact of Noise Type
    print("\nGenerating Figure 4: Impact of Noise Type...")
    try:
        plot_figure_4a(df, os.path.join(output_dir, 'figure_4a_noise_strategies.png'))
        print("✓ Figure 4a: Noise strategies")
    except Exception as e:
        print(f"✗ Figure 4a failed: {e}")
    
    # Generate Figure 5: Conformal Prediction Robustness
    print("\nGenerating Figure 5: Conformal Prediction Robustness...")
    try:
        plot_figure_5a(conformal_df, os.path.join(output_dir, 'figure_5a_conformal_coverage.png'))
        print("✓ Figure 5a: Conformal coverage rates")
    except Exception as e:
        print(f"✗ Figure 5a failed: {e}")
    
    try:
        plot_figure_5b(conformal_df, os.path.join(output_dir, 'figure_5b_conformal_interval_width.png'))
        print("✓ Figure 5b: Conformal interval width")
    except Exception as e:
        print(f"✗ Figure 5b failed: {e}")
    
    try:
        plot_figure_5c(df, conformal_df, os.path.join(output_dir, 'figure_5c_conformal_comparison.png'))
        print("✓ Figure 5c: Conformal comparison")
    except Exception as e:
        print(f"✗ Figure 5c failed: {e}")
    
    # Generate additional analyses
    print("\nGenerating Additional Analyses...")
    try:
        plot_data_size_analysis(df, os.path.join(output_dir, 'additional_data_size_analysis.png'))
        print("✓ Data size analysis")
    except Exception as e:
        print(f"✗ Data size analysis failed: {e}")
    
    try:
        plot_wilcoxon_analysis(df, os.path.join(output_dir, 'additional_wilcoxon_analysis.png'))
        print("✓ Wilcoxon statistical analysis")
    except Exception as e:
        print(f"✗ Wilcoxon analysis failed: {e}")
    
    print(f"\nAll figures saved to {output_dir}/")

def create_figure_summary(results_dir='results', output_dir='figures'):
    """Create a summary report of all experimental results"""
    
    try:
        df = load_results_data(results_dir)
        uncertainty_df = load_uncertainty_data(results_dir)
        conformal_df = load_conformal_data(results_dir)
        
        summary = {
            'total_experiments': len(df),
            'models_tested': df['model'].nunique(),
            'representations_tested': df['rep'].nunique(),
            'sigma_levels': sorted(df['sigma'].unique()),
            'sample_sizes': sorted(df['sample_size'].unique()),
            'uncertainty_models': uncertainty_df['Model'].nunique() if not uncertainty_df.empty else 0,
            'conformal_models': conformal_df['model_name'].nunique() if not conformal_df.empty else 0,
            'best_performing_pairs': df.groupby(['model', 'rep'])['r2'].mean().nlargest(10).to_dict()
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, 'experimental_summary.txt')
        with open(summary_path, 'w') as f:
            f.write("QSAR/QSPR Noise Robustness Experimental Summary\n")
            f.write("=" * 50 + "\n\n")
            for key, value in summary.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for k, v in value.items():
                        f.write(f"  {k}: {v:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        
        print(f"Experimental summary saved to {summary_path}")
        return summary
        
    except Exception as e:
        print(f"Failed to create summary: {e}")
        return {}

def generate_individual_figure(figure_name, results_dir='results', output_dir='figures'):
    """Generate a specific figure by name"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load necessary data based on figure
    if figure_name.startswith('1'):
        df = load_results_data(results_dir)
        if figure_name == '1a':
            return plot_figure_1a(df, os.path.join(output_dir, 'figure_1a.png'))
        elif figure_name == '1b':
            return plot_figure_1b(df, os.path.join(output_dir, 'figure_1b.png'))
        elif figure_name == '1c':
            return plot_figure_1c(df, os.path.join(output_dir, 'figure_1c.png'))
        elif figure_name == '1d':
            return plot_figure_1d(df, os.path.join(output_dir, 'figure_1d.png'))
            
    elif figure_name.startswith('2'):
        df = load_results_data(results_dir)
        per_epoch_df = load_per_epoch_data(results_dir)
        if figure_name == '2a':
            return plot_figure_2a(per_epoch_df, os.path.join(output_dir, 'figure_2a.png'))
        elif figure_name == '2b':
            return plot_figure_2b(df, os.path.join(output_dir, 'figure_2b.png'))
            
    elif figure_name.startswith('3'):
        uncertainty_df = load_uncertainty_data(results_dir)
        if figure_name == '3a':
            return plot_figure_3a(uncertainty_df, os.path.join(output_dir, 'figure_3a.png'))
        elif figure_name == '3b':
            return plot_figure_3b(uncertainty_df, os.path.join(output_dir, 'figure_3b.png'))
        elif figure_name == '3c':
            return plot_figure_3c(uncertainty_df, os.path.join(output_dir, 'figure_3c.png'))
        elif figure_name == '3d':
            return plot_figure_3d(uncertainty_df, os.path.join(output_dir, 'figure_3d.png'))
            
    elif figure_name.startswith('4'):
        df = load_results_data(results_dir)
        if figure_name == '4a':
            return plot_figure_4a(df, os.path.join(output_dir, 'figure_4a.png'))
            
    elif figure_name.startswith('5'):
        df = load_results_data(results_dir)
        conformal_df = load_conformal_data(results_dir)
        if figure_name == '5a':
            return plot_figure_5a(conformal_df, os.path.join(output_dir, 'figure_5a.png'))
        elif figure_name == '5b':
            return plot_figure_5b(conformal_df, os.path.join(output_dir, 'figure_5b.png'))
        elif figure_name == '5c':
            return plot_figure_5c(df, conformal_df, os.path.join(output_dir, 'figure_5c.png'))
    
    print(f"Unknown figure name: {figure_name}")
    return None

if __name__ == "__main__":
    # First, try to install vl-convert-python
    try:
        import subprocess
        subprocess.check_call(['pip', 'install', 'vl-convert-python'])
        print("Successfully installed vl-convert-python")
    except:
        print("Warning: Could not install vl-convert-python. Plots will be saved as JSON instead of PNG.")
    
    # Generate all figures
    create_all_figures()
    
    # Create summary report
    create_figure_summary()