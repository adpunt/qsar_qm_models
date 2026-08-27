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
    Reads results CSV and computes mean + 95% confidence intervals for each model, representation, and sample size
    across different sigma values.
    """
    df = pd.read_csv(filepath)

    # Ensure correct data types
    df["sigma"] = df["sigma"].astype(float)
    df["sample_size"] = pd.to_numeric(df["sample_size"], errors="coerce")

    # Filter only ECFP4 / RF
    df = df[(df["rep"] == "ecfp4") & (df["model"] == "rf")]

    # Compute mean and confidence intervals for each (sample_size, sigma)
    def compute_stats(group):
        mean_r2, ci = calculate_confidence_intervals(group["r2_score"].values)
        return pd.Series({"mean": mean_r2, "ci": ci})

    grouped = df.groupby(["sample_size", "sigma"]).apply(compute_stats).reset_index()

    # Compute upper and lower bounds for confidence intervals
    grouped["lower_bound"] = grouped["mean"] - grouped["ci"]
    grouped["upper_bound"] = grouped["mean"] + grouped["ci"]

    return grouped

def plot_r2_vs_sigma_ecfp4_rf(df, save_path="r2_vs_sigma_ecfp4_rf.html"):
    """
    Generates a single Altair plot for R² vs. Sigma for ECFP4/RF, with different sample sizes as separate lines.
    """
    base = alt.Chart(df).encode(
        x=alt.X("sigma:Q", title="Sigma (Noise Level)"),
        y=alt.Y("mean:Q", title="Mean R² Score", scale=alt.Scale(domain=[0, 1])),  # Fixed y-axis
        color=alt.Color("sample_size:N", title="Sample Size"),  # Different colors for sample sizes
    )

    # Confidence interval band (95% CI)
    band = base.mark_area(opacity=0.2).encode(
        y="lower_bound:Q",
        y2="upper_bound:Q"
    )

    # Line plot for mean R² per sample size
    line = base.mark_line(point=True)

    chart = (band + line).properties(
        width=800,
        height=500,
        title="R² vs. Sigma (ECFP4 / RF, Different Sample Sizes)"
    )

    # Save the plot
    chart.save(save_path)
    print(f"Plot saved: {save_path}")

    return chart

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a plot of R² vs. Sigma for ECFP4/RF with different sample sizes")
    parser.add_argument("-f", "--filepath", type=str, required=True, help="Filepath to results CSV")

    args = parser.parse_args()

    # Load and process results
    df = load_and_process_results(args.filepath)

    print(df)

    # Generate and show the plot
    chart = plot_r2_vs_sigma_ecfp4_rf(df)
    chart.show()
