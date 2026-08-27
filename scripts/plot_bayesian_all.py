import pandas as pd
import altair as alt

file_map = {
    "../results/linePlot.csv": [
        ("dnn", "DNN", "DNN", "solid"),
        ("rf", "RF", "RF", "solid"),
        ("xgboost", "XGBoost", "XGBoost", "solid")
    ],
    "../results/uncertainty_full.csv": [
        ("dnn", "Full Bayesian", "DNN", "solid")
    ],
    "../results/uncertainty_last_layer.csv": [
        ("dnn", "Last Layer", "DNN", "solid")
    ],
    "../results/uncertainty_variational.csv": [
        ("dnn", "Variational", "DNN", "solid")
    ],
    "../results/uncertainty_qrf.csv": [
        ("qrf", "QRF", "RF", "dashed")
    ],
    "../results/uncertainty_ngboost.csv": [
        ("ngboost", "NGBoost", "XGBoost", "dashed")
    ]
}

rep_map = {
    'ecfp4': 'ECFP4',
    'smiles': 'SMILES',
    'sns': 'SNS',
    'pdv': 'PDV',
    'graph': 'Graph'
}

# Load and prepare data
all_rows = []
for path, models in file_map.items():
    df = pd.read_csv(path)
    model_col = 'Model' if 'Model' in df.columns else 'model'
    for model_key, full_label, family_label, line_style in models:
        if model_col not in df.columns:
            continue
        matched = df[df[model_col].str.lower().str.contains(model_key)]
        matched = matched[matched['rep'].isin(rep_map)]
        matched = matched.copy()
        matched['Model'] = full_label
        matched['Family'] = family_label
        matched['Line_Type'] = line_style
        matched['Rep_Clean'] = matched['rep'].map(rep_map)
        matched = matched[['sigma', 'r2_score', 'Model', 'Family', 'Line_Type', 'Rep_Clean']]
        all_rows.append(matched)

df_all = pd.concat(all_rows, ignore_index=True)
df_all['r2_score'] = df_all['r2_score'].clip(lower=-10, upper=2)

# Group to remove staircases: take mean R² per model/rep/sigma
agg_df = df_all.groupby(['Rep_Clean', 'Model', 'Family', 'Line_Type', 'sigma'], as_index=False).agg({'r2_score': 'mean'})

# Build charts
charts = []

for rep in sorted(agg_df['Rep_Clean'].unique()):
    rep_df = agg_df[agg_df['Rep_Clean'] == rep]

    dnn_df = rep_df[rep_df['Family'] == 'DNN']
    dnn_chart = alt.Chart(dnn_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('sigma:Q', title='Sigma (Noise Level)'),
        y=alt.Y('r2_score:Q', title='R² Score'),
        color=alt.Color('Model:N', title='DNN Variant'),
        tooltip=['Model', 'sigma', 'r2_score']
    ).properties(
        title=f'{rep}: DNN Variants, Baseline vs Bayesian',
        width=600,
        height=350
    )

    # --- RF/XGBoost with Bayesian Variants Plot ---
    others_df = rep_df[rep_df['Model'].isin(['RF', 'QRF', 'XGBoost', 'NGBoost'])].copy()

    # Add a column for baseline vs bayesian for clearer strokeDash
    others_df['Line_Type_Label'] = others_df['Model'].map({
        'RF': 'Baseline',
        'QRF': 'Bayesian',
        'XGBoost': 'Baseline',
        'NGBoost': 'Bayesian'
    })

    rf_xgb_chart = alt.Chart(others_df).mark_line(point=True, strokeWidth=3).encode(
        x=alt.X('sigma:Q', title='Sigma (Noise Level)'),
        y=alt.Y('r2_score:Q', title='R² Score'),
        color=alt.Color('Family:N', title='Model Family'),
        strokeDash=alt.StrokeDash('Line_Type_Label:N', title='Line Type'),
        tooltip=['Model', 'sigma', 'r2_score']
    ).properties(
        title=f'{rep}: Trees, Baseline vs Bayesian',
        width=600,
        height=350
    )


    rep_chart = alt.vconcat(dnn_chart, rf_xgb_chart).resolve_scale(color='independent', strokeDash='independent')
    charts.append(rep_chart)

final_chart = alt.vconcat(*charts).properties(
    title="R² Score vs Sigma (Noise Level) by Molecular Representation"
)

final_chart.save("../results/r2_vs_sigma_split_stacked.html")
