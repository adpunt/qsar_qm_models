import pandas as pd
import altair as alt
import numpy as np

# TODO: Run this beforehand
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/distributionGaussian2.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/scaffold2.csv ../results/

def plot_split_strategy_comparison(df_random: pd.DataFrame, df_scaffold: pd.DataFrame, representation: str):
    """
    Plots R² vs Sigma comparing scaffold vs. random splits for a given molecular representation.

    Args:
        df_random (pd.DataFrame): Results with standard random splitting
        df_scaffold (pd.DataFrame): Results with scaffold-based splitting
        representation (str): e.g., 'ecfp4', 'sns', or 'smiles'
    """
    def prepare_summary(df, split_label):
        df = df[df['rep'] == representation].copy()
        df['split'] = split_label
        grouped = df.groupby(['sigma', 'model', 'split']).agg(list).reset_index()
        grouped['mean_r2'] = grouped['r2_score'].apply(np.mean)
        return grouped[['sigma', 'model', 'split', 'mean_r2']]

    df_random_summary = prepare_summary(df_random, 'random')
    df_scaffold_summary = prepare_summary(df_scaffold, 'scaffold')

    df_plot = pd.concat([df_random_summary, df_scaffold_summary])

    chart = alt.Chart(df_plot).mark_line().encode(
        x=alt.X('sigma:Q', title='Sigma (Noise Level)', scale=alt.Scale(zero=False)),
        y=alt.Y('mean_r2:Q', title='R² Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('model:N', title='Model'),
        strokeDash=alt.StrokeDash('split:N', title='Split Method'),
        tooltip=['sigma', 'mean_r2', 'model', 'split']
    ).properties(
        width=650,
        height=400,
        title=f"Split Strategy Comparison on {representation.upper()}"
    )

    return chart

# Load your data
df_random = pd.read_csv("../results/distributionGaussian2.csv")
df_scaffold = pd.read_csv("../results/scaffold2.csv")

# For each representation
for rep in ['ecfp4', 'sns', 'smiles']:
    chart = plot_split_strategy_comparison(df_random, df_scaffold, rep)
    chart.save(f"split_strategy_{rep}.html")
