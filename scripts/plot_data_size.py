import pandas as pd
import altair as alt
import numpy as np
from scipy import stats

# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/dataSize.csv ../results/
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/dataSizeGraph.csv ../results/

# === Load & Clean Primary Data ===
df_main = pd.read_csv("../results/dataSize.csv")
df_graph = pd.read_csv("../results/dataSizeGraph.csv")

for df in [df_main, df_graph]:
    df.dropna(subset=['sigma', 'r2_score', 'model', 'rep', 'sample_size'], inplace=True)
    df['sigma'] = pd.to_numeric(df['sigma'], errors='coerce')
    df['r2_score'] = pd.to_numeric(df['r2_score'], errors='coerce')
    df['sample_size'] = df['sample_size'].astype(str)

print("Loaded main and graph-specific CSVs")
print("Main models:", df_main['model'].unique())
print("Graph models:", df_graph['model'].unique())

# === Compute grouped stats ===
def compute_grouped(df):
    grouped = df.groupby(['sigma', 'model', 'rep', 'sample_size']).agg(list).reset_index()
    grouped['mean_r2'] = grouped['r2_score'].apply(np.mean)
    grouped['ci'] = grouped['r2_score'].apply(
        lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95) / 2., len(x)-1) if len(x) > 1 else 0
    )
    grouped['lower'] = grouped['mean_r2'] - grouped['ci']
    grouped['upper'] = grouped['mean_r2'] + grouped['ci']

    # Remove invalid R² values
    grouped = grouped[(grouped['mean_r2'] >= -1) & (grouped['mean_r2'] <= 1)]
    return grouped

# === Chart factory ===
def make_chart(grouped, title):
    base = alt.Chart(grouped).encode(
        x=alt.X('sigma:Q', title='Sigma (Noise Level)'),
        y=alt.Y('mean_r2:Q', title='R² Score', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('sample_size:N', title='Data Size'),
        tooltip=['sigma', 'mean_r2', 'lower', 'upper', 'sample_size']
    )
    bands = base.mark_area(opacity=0.2).encode(y='lower:Q', y2='upper:Q')
    lines = base.mark_line()
    return (bands + lines).properties(width=250, height=200).facet(
        row=alt.Row('model:N', title='Model'),
        column=alt.Column('rep:N', title='Representation'),
        title=title
    )

# === Main plot ===
def plot_data_efficiency(df_main, df_graph):
    grouped_main = compute_grouped(df_main)
    grouped_graph = compute_grouped(df_graph)

    print(f"Main plot size: {grouped_main.shape}")
    print(f"Graph plot size: {grouped_graph.shape}")
    print("Graph sigma values:", grouped_graph['sigma'].unique())
    print("Graph models:", grouped_graph['model'].unique())

    # MAIN PLOT
    chart_main = make_chart(grouped_main, "Data Size Sensitivity (ECFP4 / SMILES / Other)")
    chart_main.save("data_size_vs_noise_main.html")
    chart_main.show()

    # GRAPH-ONLY PLOT
    chart_graph = make_chart(grouped_graph, "Data Size Sensitivity (GIN / Graph_GP Only)")
    chart_graph.save("data_size_vs_noise_graph.html")
    chart_graph.show()

    return chart_main, chart_graph

# === Run ===
chart_main, chart_graph = plot_data_efficiency(df_main, df_graph)
