import pandas as pd
import altair as alt
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


import os
import glob


def plot_shap_trend_across_noise(df_shap: pd.DataFrame, model: str, rep: str):
    """
    Plot how the top 10 most important features at sigma=0 change across sigma levels.
    """
    df = df_shap[(df_shap['Model'] == model) & (df_shap['Rep'] == rep)].copy()

    if 'Sigma' not in df.columns:
        raise ValueError("SHAP dataframe must contain a 'Sigma' column indicating noise level.")

    feature_cols = [col for col in df.columns if col.startswith('feature_')]

    # Melt and compute mean |SHAP| per feature at sigma = 0
    df_zero = df[df['Sigma'] == 0]
    df_zero_melted = df_zero.melt(id_vars=['Sigma'], value_vars=feature_cols,
                                  var_name='Feature', value_name='SHAP_Value')
    top_features = (
        df_zero_melted.groupby('Feature')['SHAP_Value']
        .apply(lambda x: np.abs(x).mean())
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    print(f"\nTop 10 features at sigma=0 for model={model}, rep={rep}:")
    for i, f in enumerate(top_features, 1):
        print(f"{i}. {f}")

    # Track those features across all sigma levels
    df_melted = df.melt(id_vars=['Sigma'], value_vars=top_features,
                        var_name='Feature', value_name='SHAP_Value')

    df_summary = df_melted.groupby(['Sigma', 'Feature']).agg(
        mean_abs_shap=('SHAP_Value', lambda x: np.abs(x).mean())
    ).reset_index()

    chart = alt.Chart(df_summary).mark_line().encode(
        x=alt.X('Sigma:Q', title='Sigma (Noise Level)'),
        y=alt.Y('mean_abs_shap:Q', title='Mean |SHAP| of Top Feature'),
        color=alt.Color('Feature:N', title='Feature'),
        tooltip=['Sigma', 'Feature', 'mean_abs_shap']
    ).properties(
        width=700,
        height=400,
        title=f'SHAP Robustness of Top Features — {model.upper()} on {rep}'
    )

    return chart

import os
import glob
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_shap_summary_across_sigmas(model: str, rep: str, kind: str = "bar"):
    """
    For a given model and representation, plot SHAP summary plots across all sigma values.

    Parameters:
        model (str): e.g., 'rf' or 'xgboost'
        rep (str): e.g., 'ecfp4' or 'smiles'
        kind (str): either 'bar' or 'beeswarm'
    """
    npy_pattern = f"shap_{model}_{rep}_sigma*.npy"
    npy_files = sorted(glob.glob(npy_pattern))

    if not npy_files:
        print(f"No SHAP files found for {model}/{rep}")
        return

    for npy_file in npy_files:
        sigma_str = npy_file.split("sigma")[-1].replace(".npy", "")
        csv_file = f"x_{model}_{rep}_sigma{sigma_str}.csv"

        if not os.path.exists(csv_file):
            print(f"Missing CSV for {npy_file}, skipping.")
            continue

        # Load data
        shap_values = np.load(npy_file, allow_pickle=True)
        x_test = pd.read_csv(csv_file)

        # Wrap into Explanation
        shap_exp = shap.Explanation(
            values=shap_values,
            data=x_test.values,
            feature_names=x_test.columns.tolist()
        )

        # Plot
        plt.clf()
        try:
            if kind == "bar":
                shap.plots.bar(shap_exp, show=False)
                outname = f"../results/shap_bar_{model}_{rep}_sigma{sigma_str}.png"
            elif kind == "beeswarm":
                shap.plots.beeswarm(shap_exp, show=False)
                outname = f"../results/shap_beeswarm_{model}_{rep}_sigma{sigma_str}.png"
            else:
                raise ValueError(f"Unsupported plot type: {kind}")

            plt.savefig(outname, bbox_inches='tight')
            print(f"Saved {outname}")

        except Exception as e:
            print(f"Failed on sigma={sigma_str} for {model}/{rep}: {e}")


# df_shap = pd.read_csv("../results/shap_shap.csv")

# chart_ecfp4_rf = plot_shap_trend_across_noise(df_shap, model='rf', rep='ecfp4')
# chart_smiles_svm = plot_shap_trend_across_noise(df_shap, model='svm', rep='smiles')

# chart_ecfp4_rf.save("../results/shap_trend_rf_ecfp4.html")
# chart_smiles_svm.save("../results/shap_trend_svm_smiles.html")

# # Load saved SHAP values and data (sigma = 0.0)
# shap_values = np.load("shap_rf_ecfp4_sigma0.0.npy", allow_pickle=True)
# x_test = pd.read_csv("x_rf_ecfp4_sigma0.0.csv")

# # Wrap into Explanation object
# shap_exp = shap.Explanation(
#     values=shap_values,
#     data=x_test.values,
#     feature_names=x_test.columns.tolist()
# )

# shap.plots.bar(shap_exp)
# plt.savefig("../results/shap_bar_rf_ecfp4_sigma0.0.png", bbox_inches='tight')
# plt.clf()

# shap.plots.beeswarm(shap_exp)
# plt.savefig("../results/shap_beeswarm_rf_ecfp4_sigma0.0.png", bbox_inches='tight')
# plt.clf()

# # Load saved SHAP values and data (sigma = 0.5)
# shap_values = np.load("shap_xgboost_smiles_sigma0.5.npy", allow_pickle=True)
# x_test = pd.read_csv("x_xgboost_smiles_sigma0.5.csv")

# # Wrap into Explanation object
# shap_exp = shap.Explanation(
#     values=shap_values,
#     data=x_test.values,
#     feature_names=x_test.columns.tolist()
# )

# shap.plots.bar(shap_exp)
# plt.savefig("../results/shap_bar_xgboost_smiles_sigma0.5.png", bbox_inches='tight')
# plt.clf()

# shap.plots.beeswarm(shap_exp)
# plt.savefig("../results/shap_beeswarm_xgboost_smiles_sigma0.5.png", bbox_inches='tight')
# plt.clf()

# plot_shap_summary_across_sigmas(model='rf', rep='ecfp4', kind='bar')
# plot_shap_summary_across_sigmas(model='xgboost', rep='smiles', kind='beeswarm')

def plot_shap_feature_trends(model: str, rep: str, top_n: int = 10) -> alt.Chart:
    """
    Plot SHAP importance trends for top N features across different sigma levels.
    
    Args:
        model (str): e.g., 'rf' or 'xgboost'
        rep (str): e.g., 'ecfp4' or 'smiles'
        top_n (int): number of top features to track
    Returns:
        Altair chart object
    """
    npy_pattern = f"shap_{model}_{rep}_sigma*.npy"
    npy_files = sorted(glob.glob(npy_pattern))

    if not npy_files:
        print(f"No SHAP files found for {model}/{rep}")
        return None

    shap_summary = []

    for path in npy_files:
        sigma = float(path.split("sigma")[-1].replace(".npy", ""))
        shap_values = np.load(path, allow_pickle=True)
        x_path = f"x_{model}_{rep}_sigma{sigma}.csv"
        
        if not os.path.exists(x_path):
            print(f"Missing x_test: {x_path}, skipping.")
            continue

        x_df = pd.read_csv(x_path)

        # Wrap into Explanation object
        exp = shap.Explanation(
            values=shap_values,
            data=x_df.values,
            feature_names=x_df.columns.tolist()
        )

        df = pd.DataFrame(np.abs(exp.values), columns=exp.feature_names)
        df["Sigma"] = sigma
        shap_summary.append(df)

    if not shap_summary:
        print("No valid SHAP+X file pairs found.")
        return None

    # Combine and melt
    df_all = pd.concat(shap_summary)
    feature_cols = [col for col in df_all.columns if col != "Sigma"]

    df_melted = df_all.melt(id_vars="Sigma", value_vars=feature_cols,
                            var_name="Feature", value_name="SHAP_Value")

    # Get top features at sigma = 0.0
    top_features = (
        df_melted[df_melted["Sigma"] == 0.0]
        .groupby("Feature")["SHAP_Value"]
        .mean()
        .abs()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )

    df_top = df_melted[df_melted["Feature"].isin(top_features)]

    df_summary = df_top.groupby(["Sigma", "Feature"]).agg(
        mean_abs_shap=("SHAP_Value", "mean")
    ).reset_index()

    chart = alt.Chart(df_summary).mark_line(point=True).encode(
        x=alt.X("Sigma:Q", title="Sigma (Noise Level)"),
        y=alt.Y("mean_abs_shap:Q", title="Mean |SHAP|"),
        color=alt.Color("Feature:N", title="Top Features"),
        tooltip=["Sigma", "Feature", "mean_abs_shap"]
    ).properties(
        width=700,
        height=400,
        title=f"SHAP Importance Trend — {model.upper()} on {rep}"
    )

    return chart

# chart = plot_shap_feature_trends(model="rf", rep="ecfp4")
# chart.save("../results/shap_trend_rf_ecfp4.html")

# chart = plot_shap_feature_trends(model="xgboost", rep="smiles")
# chart.save("../results/shap_trend_xgboost_smiles.html")

import os
import glob
import re
import pandas as pd

# Ensure results folder exists
os.makedirs("../results", exist_ok=True)

# === Auto-detect all (model, rep) combinations based on SHAP files ===
file_pairs = sorted(glob.glob("shap_*_sigma*.npy"))
model_rep_set = set()

for fname in file_pairs:
    match = re.match(r"shap_(.+?)_(.+?)_sigma", fname)
    if match:
        model, rep = match.groups()
        model_rep_set.add((model, rep))

# === Load global df_shap if available ===
try:
    df_shap = pd.read_csv("../results/shap_shap.csv")
except FileNotFoundError:
    df_shap = None
    print("Warning: shap_shap.csv not found, skipping trend plots using that dataframe.")

# === MAIN LOOP: generate all SHAP visualizations ===
for model, rep in sorted(model_rep_set):
    print(f"\n=== Processing model={model}, rep={rep} ===")

    # Trend line plot from pre-saved shap_shap.csv
    if df_shap is not None:
        try:
            chart = plot_shap_trend_across_noise(df_shap, model=model, rep=rep)
            chart.save(f"../results/shap_trend_{model}_{rep}.html")
        except Exception as e:
            print(f"Failed trend plot from shap_shap.csv for {model}/{rep}: {e}")

    # SHAP feature trend line (across sigma)
    try:
        chart = plot_shap_feature_trends(model=model, rep=rep)
        if chart:
            chart.save(f"../results/shap_feature_trend_{model}_{rep}.html")
    except Exception as e:
        print(f"Failed feature trend plot for {model}/{rep}: {e}")

    # SHAP bar plots
    try:
        plot_shap_summary_across_sigmas(model=model, rep=rep, kind="bar")
    except Exception as e:
        print(f"Failed bar plot for {model}/{rep}: {e}")

    # SHAP beeswarm plots
    try:
        plot_shap_summary_across_sigmas(model=model, rep=rep, kind="beeswarm")
    except Exception as e:
        print(f"Failed beeswarm plot for {model}/{rep}: {e}")

    try:
        chart = plot_shap_trend_across_noise(df_shap, model=model, rep=rep)
        if chart:
            chart.save(f"../results/shap_trend_{model}_{rep}.html")
    except:
        print(f"Failed trend plot for {model}/{rep}: {e}")
