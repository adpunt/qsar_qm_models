import pandas as pd
import altair as alt
import numpy as np
from scipy import stats

# TODO: Run this beforehand
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/bayesianBaseline.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/bayesianApplied.csv ../results/

def plot_bayesian_difference(df_baseline: pd.DataFrame, df_applied: pd.DataFrame, model_name: str):
    """
    Plots the difference in R² between Bayesian-applied and baseline models (applied − baseline).
    Shows confidence intervals across sigma values.

    Args:
        df_baseline (pd.DataFrame): DataFrame from bayesianBaseline.csv
        df_applied (pd.DataFrame): DataFrame from bayesianApplied.csv
        model_name (str): The model to plot (e.g., 'mlp')
    """
    # Filter both dataframes
    base = df_baseline[df_baseline['model'] == model_name].copy()
    applied = df_applied[df_applied['model'] == model_name].copy()

    # Merge by sigma and iteration index (assuming same seeds)
    merged = pd.merge(
        base[['sigma', 'iteration', 'r2_score']],
        applied[['sigma', 'iteration', 'r2_score']],
        on=['sigma', 'iteration'],
        suffixes=('_baseline', '_applied')
    )

    # Calculate delta and aggregate
    merged['delta'] = merged['r2_score_applied'] - merged['r2_score_baseline']
    summary = merged.groupby('sigma')['delta'].agg(
        mean_delta='mean',
        ci=lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95)/2., len(x)-1) if len(x) > 1 else 0
    ).reset_index()

    summary['lower'] = summary['mean_delta'] - summary['ci']
    summary['upper'] = summary['mean_delta'] + summary['ci']

    chart = alt.Chart(summary).encode(
        x=alt.X('sigma:Q', title='Sigma (Noise Level)'),
        tooltip=['sigma', 'mean_delta', 'lower', 'upper']
    )

    line = chart.mark_line(color='black').encode(
        y=alt.Y('mean_delta:Q', title='ΔR² (Applied − Baseline)', scale=alt.Scale(zero=True))
    )

    band = chart.mark_area(opacity=0.2).encode(
        y='lower:Q',
        y2='upper:Q'
    )

    return (band + line).properties(
        width=600,
        height=400,
        title=f"ΔR²: Bayesian - Baseline ({model_name})"
    )


def plot_bayesian_baseline_vs_applied(df_baseline: pd.DataFrame, df_applied: pd.DataFrame, model_name: str = None):
    """
    Plots R² vs Sigma for baseline and Bayesian-applied models with confidence intervals.
    Filters by a specific model if provided.

    Args:
        df_baseline (pd.DataFrame): DataFrame loaded from bayesianBaseline.csv
        df_applied (pd.DataFrame): DataFrame loaded from bayesianApplied.csv
        model_name (str): Optional, filter by a single model (e.g., 'mlp')
    """
    def prepare_summary(df, label):
        if model_name:
            df = df[df['model'] == model_name]
        grouped = df.groupby(['sigma', 'model']).agg(list).reset_index()
        grouped['mean_r2'] = grouped['r2_score'].apply(np.mean)
        grouped['ci'] = grouped['r2_score'].apply(lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95) / 2., len(x) - 1) if len(x) > 1 else 0)
        grouped['lower'] = grouped['mean_r2'] - grouped['ci']
        grouped['upper'] = grouped['mean_r2'] + grouped['ci']
        grouped['type'] = label
        return grouped[['sigma', 'mean_r2', 'lower', 'upper', 'model', 'type']]

    df_baseline_summary = prepare_summary(df_baseline, 'baseline')
    df_applied_summary = prepare_summary(df_applied, 'applied')
    
    df_plot = pd.concat([df_baseline_summary, df_applied_summary])

    base = alt.Chart(df_plot).encode(
        x=alt.X('sigma:Q', title='Sigma (Noise Level)', scale=alt.Scale(zero=False)),
        color=alt.Color('type:N', title='Model Variant'),
        tooltip=['sigma', 'mean_r2', 'lower', 'upper', 'model', 'type']
    )

    line = base.mark_line().encode(
        y=alt.Y('mean_r2:Q', title='R² Score', scale=alt.Scale(domain=[0, 1]))
    )

    band = base.mark_area(opacity=0.2).encode(
        y='lower:Q',
        y2='upper:Q'
    )

    chart = (band + line).properties(
        width=600,
        height=400,
        title=f"Bayesian Model Comparison on ECFP4{' - ' + model_name if model_name else ''}"
    )

    return chart

# Assuming your data is loaded:
df_baseline = pd.read_csv("../results/bayesianBaseline.csv")
df_applied = pd.read_csv("../results/bayesianApplied.csv")

mlp_chart = plot_bayesian_baseline_vs_applied(df_baseline, df_applied, model_name="mlp")
mlp_chart.save('bayesian_mlp.html')

dnn_chart = plot_bayesian_baseline_vs_applied(df_baseline, df_applied, model_name="dnn")
dnn_chart.save('bayesian_dnn.html')

residual_mlp_chart = plot_bayesian_baseline_vs_applied(df_baseline, df_applied, model_name="residual_mlp")
residual_mlp_chart.save('bayesian_residual_mlp.html')

factorization_mlp_chart = plot_bayesian_baseline_vs_applied(df_baseline, df_applied, model_name="factorization_mlp")
factorization_mlp_chart.save('bayesian_factorization_mlp.html')

# mlp_diff = plot_bayesian_difference(df_baseline, df_applied, model_name="mlp")
# mlp_diff.save("bayesian_diff_mlp.html")

# dnn_diff = plot_bayesian_difference(df_baseline, df_applied, model_name="dnn")
# dnn_diff.save("bayesian_diff_dnn.html")

# residual_diff = plot_bayesian_difference(df_baseline, df_applied, model_name="residual_mlp")
# residual_diff.save("bayesian_diff_residual_mlp.html")


