import pandas as pd
import altair as alt
import argparse
import numpy as np
from scipy.stats import sem, t

def calculate_confidence_intervals(data, confidence=0.95):
    """
    Compute the mean and 95% confidence interval (CI) for a given set of R² values.
    """
    n = len(data)
    mean_r2 = np.mean(data)
    if n > 1:
        sem_value = sem(data)  # Standard error of the mean
        margin_of_error = sem_value * t.ppf((1 + confidence) / 2, n - 1)
    else:
        margin_of_error = 0  # No confidence interval if there's only one sample

    return mean_r2, margin_of_error

def load_and_process_results(filepath):
    """
    Reads results CSV and computes mean + 95% confidence intervals for each sigma, model, and representation.
    """
    df = pd.read_csv(filepath)



    # Ensure sigma is treated as a float
    df["sigma"] = df["sigma"].astype(float)

    # Group by sigma, model, and representation
    grouped = df.groupby(["sigma", "model", "rep"])["r2_score"].apply(list).reset_index()
    grouped["mean"], grouped["ci"] = zip(*grouped["r2_score"].apply(calculate_confidence_intervals))
    
    # Compute upper and lower bounds for plotting
    grouped["lower_bound"] = grouped["mean"] - grouped["ci"]
    grouped["upper_bound"] = grouped["mean"] + grouped["ci"]

    return grouped

def plot_r2_vs_sigma_separate(df, save_folder="../results"):
    """
    Generates a separate Altair plot for each representation, showing R² vs. Sigma with 95% confidence intervals.
    """
    plots = {}
    representations = df["rep"].unique()

    for rep in representations:
        df_subset = df[df["rep"] == rep]

        base = alt.Chart(df_subset).encode(
            x=alt.X("sigma:Q", title="Sigma (Noise Level)"),
            y=alt.Y("mean:Q", title="Mean R² Score", scale=alt.Scale(domain=[0, 1])),  # Set fixed y-axis range
            color=alt.Color("model:N", title="Model"),
        )

        # Confidence interval band (95% CI) for each model
        band = base.mark_area(opacity=0.2).encode(
            y="lower_bound:Q",
            y2="upper_bound:Q"
        )

        # Line plot for mean R² for each model
        line = base.mark_line(point=True)

        chart = (band + line).properties(
            width=700,
            height=450,
            title=f"R² vs. Sigma ({rep})"
        )

        # Save each plot separately
        save_path = f"{save_folder}/r2_vs_sigma_{rep}.html"
        chart.save(save_path)
        print(f"Plot saved: {save_path}")

        # Store chart in dictionary (useful for Jupyter display)
        plots[rep] = chart

    return plots


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate separate plots of R² vs. Sigma for each representation")
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Filepath to results CSV")

    args = parser.parse_args()

    # Load and process results
    df = load_and_process_results(args.filepath)

    # Generate separate plots
    plot_r2_vs_sigma_separate(df)
