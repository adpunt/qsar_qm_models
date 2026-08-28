import pandas as pd
import altair as alt
import argparse
import os

def load_data(filepath, rep_filter=None):
    df = pd.read_csv(filepath)
    if rep_filter:
        df = df[df["rep"] == rep_filter]
    return df

def plot_train_vs_test_loss(df, output_path="train_vs_test_loss.html"):
    df_melt = df.melt(
        id_vars=["sigma", "iteration"],
        value_vars=["train_loss", "test_loss"],
        var_name="split",
        value_name="loss"
    )

    agg = df_melt.groupby(["sigma", "split"])["loss"].agg(["mean", "std"]).reset_index()
    agg.columns = ["sigma", "split", "mean_loss", "std_loss"]

    base = alt.Chart(agg).encode(
        x=alt.X("sigma:Q", title="Noise Level"),
        y=alt.Y("mean_loss:Q", title="Loss"),
        color=alt.Color("split:N", title="Split")
    )

    line = base.mark_line(point=True)

    # Fix: no alt.Y() for yError — just field name strings
    error = base.mark_errorbar().encode(
        y="mean_loss:Q",
        yError="std_loss"
    )

    chart = (line + error).properties(title="Train vs Test Loss vs Noise")
    chart.save(output_path)
    print(f"[Saved] {output_path}")

def plot_sharpness_vs_noise(df, output_path="sharpness_vs_noise.html"):
    agg = df.groupby("sigma")["sharpness_proxy"].agg(["mean", "std"]).reset_index()

    base = alt.Chart(agg).encode(
        x=alt.X("sigma:Q", title="Noise Level"),
        y=alt.Y("mean:Q", title="Sharpness Proxy")
    )

    line = base.mark_line(point=True)
    error = base.mark_errorbar().encode(
        y="mean:Q",
        yError="std:Q"
    )

    chart = (line + error).properties(title="Sharpness vs Noise")
    chart.save(output_path)
    print(f"[Saved] {output_path}")

def plot_sharpness_vs_test_loss(df, output_path="sharpness_vs_test_loss.html"):
    chart = alt.Chart(df).mark_circle(size=60).encode(
        x=alt.X("sharpness_proxy:Q", title="Sharpness Proxy"),
        y=alt.Y("test_loss:Q", title="Test Loss"),
        color=alt.Color("sigma:Q", scale=alt.Scale(scheme="viridis")),
        tooltip=["sigma", "iteration", "sharpness_proxy", "test_loss"]
    ).properties(title="Sharpness vs Test Loss")

    chart.save(output_path)
    print(f"[Saved] {output_path}")

def generate_summary_table(df, output_path="loss_summary.csv"):
    agg = df.groupby("sigma").agg({
        "train_loss": ["mean", "std"],
        "test_loss": ["mean", "std"],
        "sharpness_proxy": ["mean", "std"]
    })
    agg.columns = ["_".join(col) for col in agg.columns]
    agg.reset_index(inplace=True)

    agg.to_csv(output_path, index=False)
    print(f"[Saved summary] {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to CSV from loss_landscape()")
    parser.add_argument("--rep", default=None, help="Representation to filter by (e.g., ecfp4, smiles)")
    parser.add_argument("--outdir", default=".", help="Directory to save plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = load_data(args.csv_path, rep_filter=args.rep)

    plot_train_vs_test_loss(df, os.path.join(args.outdir, "train_vs_test_loss.html"))
    plot_sharpness_vs_noise(df, os.path.join(args.outdir, "sharpness_vs_noise.html"))
    plot_sharpness_vs_test_loss(df, os.path.join(args.outdir, "sharpness_vs_test_loss.html"))
    generate_summary_table(df, os.path.join(args.outdir, "loss_summary.csv"))
