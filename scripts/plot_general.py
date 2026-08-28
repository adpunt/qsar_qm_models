import pandas as pd
import altair as alt
import numpy as np
from scipy import stats

# TODO: Run this beforehand
# scp scat9264@gateway.arc.ox.ac.uk:/data/stat-cadd/scat9264/qsar_qm_models/results/linePlot.csv ../results/

import pandas as pd
import altair as alt
import numpy as np
from scipy import stats

def plot_standard_vs_extra_by_representation(df_line: pd.DataFrame, df_extra: pd.DataFrame, representation: str):
    """
    For a given representation, generate two charts: one for standard line plots, one for extra line plots.
    Each chart has lines per model, showing R² vs sigma.
    """
    def prepare_data(df):
        df = df[df['rep'] == representation].copy()
        grouped = df.groupby(['sigma', 'model'], as_index=False).agg(list).copy()
        grouped['mean_r2'] = grouped['r2_score'].apply(np.mean)

        # --- COMMENT OUT CONFIDENCE INTERVAL CALCULATIONS ---
        # grouped['ci'] = grouped['r2_score'].apply(
        #     lambda x: stats.sem(x) * stats.t.ppf((1 + 0.95)/2., len(x)-1) if len(x) > 1 else 0
        # )
        # grouped['lower'] = grouped['mean_r2'] - grouped['ci']
        # grouped['upper'] = grouped['mean_r2'] + grouped['ci']

        grouped.columns.name = None
        grouped = grouped.reset_index(drop=True)

        return grouped

    df_std = prepare_data(df_line)
    for model in df_std['model'].unique():
        sub = df_std[df_std['model'] == model]

        x = sub['sigma'].values
        y = sub['mean_r2'].values

        # Ensure sigma is sorted correctly
        if len(x) >= 2:
            order = np.argsort(x)
            x_sorted = x[order]
            y_sorted = y[order]
            auc_score = np.trapz(y_sorted, x_sorted)

            # Optional: normalize AUC to [0, 1]
            auc_score /= (x_sorted[-1] - x_sorted[0])
        else:
            auc_score = float('nan')

        print(f"{representation.upper()} — {model}: AUC = {auc_score:.4f}")

    # df_ext = prepare_data(df_extra)

    def make_chart(df, label):
        base = alt.Chart(df).encode(
            x=alt.X('sigma:Q', title='Sigma'),
            y=alt.Y('mean_r2:Q', title='R²', scale=alt.Scale(domain=[0, 1])),
            color=alt.Color('model:N', title='Model'),
            # --- COMMENT OUT CI TOOLTIP FIELDS ---
            # tooltip=['sigma', 'mean_r2', 'lower', 'upper', 'model']
            tooltip=['sigma', 'mean_r2', 'model']
        )
        line = base.mark_line()

        # --- COMMENT OUT CONFIDENCE BAND ---
        # band = base.mark_area(opacity=0.2).encode(
        #     y='lower:Q',
        #     y2='upper:Q'
        # )
        # return (band + line).properties(
        #     width=600,
        #     height=400,
        #     title=f"{representation.upper()} — {label}"
        # )

        return line.properties(
            width=600,
            height=400,
            title=f"{representation.upper()} — {label}"
        )

    chart_standard = make_chart(df_std, "Standard")
    return chart_standard

# df_line = pd.read_csv("../results/linePlot.csv")
# df_line = pd.read_csv("../results/mu.csv")
df_line = pd.read_csv("../results/alpha.csv")
df_extra = None


# Ensure r2_score is numeric
df_line['r2_score'] = pd.to_numeric(df_line['r2_score'], errors='coerce')

# Drop NaNs and invalid R² values (e.g., below -1 or above 1)
df_line = df_line[df_line['r2_score'].notna()]
df_line = df_line[(df_line['r2_score'] >= -1.0) & (df_line['r2_score'] <= 1.0)]

# TODO: add randomized_smiles after debugging/tuning/re-running scripts
for rep in ['ecfp4', 'sns', 'smiles', 'graph', 'randomized_smiles', 'pdv']:
    chart = plot_standard_vs_extra_by_representation(df_line, df_extra, rep)
    chart.save(f"lineplots_{rep}_alpha.html")



