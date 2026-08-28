#!/usr/bin/env python3
"""
Uncertainty Analysis - Layer 3 of the results

Analyzes uncertainty estimates from probabilistic models to understand
WHY some models are more noise-robust than others.

Key hypothesis: Models that attribute noise to aleatoric uncertainty are robust.
Models that try to fit the noise (high epistemic) fail.

Metrics:
1. Calibration (predicted σ vs observed error)
2. Coverage (prediction intervals)
3. ECE (Expected Calibration Error)
4. Uncertainty-Error correlation
5. Uncertainty-Noise correlation (KEY)

Run: python uncertainty_analysis.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = Path(__file__).parent.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "uncertainty_analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Style
sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
})

MODEL_COLORS = {
    'qrf': '#16a085',
    'ngboost': '#f39c12',
    'gauche': '#9b59b6',
    'dnn_bnn_full': '#e74c3c',
    'dnn_bnn_last': '#c0392b',
    'dnn_bnn_variational': '#a93226',
    'mlp_bnn_full': '#3498db',
    'mlp_bnn_last': '#2980b9',
    'mlp_bnn_variational': '#1f618d',
    'conformal_rf': '#27ae60',
    'conformal_qrf': '#1e8449',
    'conformal_dnn': '#145a32',
}


def load_uncertainty_data():
    """Load all uncertainty_*_uncertainty_values.csv files."""
    all_data = []

    # Try both naming conventions
    patterns = [
        "uncertainty_*_uncertainty_values.csv",
        "anova_*_uncertainty_values.csv",
        "phase*_uncertainty_values.csv"
    ]

    for pattern in patterns:
        for f in RESULTS_DIR.glob(pattern):
            try:
                df = pd.read_csv(f)
                df['source_file'] = f.name

                # Parse strategy from filename if not in data
                if 'strategy' not in df.columns:
                    parts = f.stem.split('_')
                    if 'uncertainty' in parts:
                        idx = parts.index('uncertainty')
                        if idx + 1 < len(parts):
                            df['strategy'] = parts[idx + 1]

                all_data.append(df)
                print(f"Loaded {f.name}: {len(df)} rows")
            except Exception as e:
                print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"\nTotal: {len(combined)} rows from {len(all_data)} files")
        return combined
    return None


def calculate_calibration_metrics(df, model_col='model', sigma_col='y_pred_std_calibrated'):
    """Calculate calibration metrics for each model."""
    results = []

    if sigma_col not in df.columns:
        # Try alternative column names
        for alt in ['y_pred_std', 'uncertainty', 'std']:
            if alt in df.columns:
                sigma_col = alt
                break

    if sigma_col not in df.columns:
        print(f"Warning: No uncertainty column found")
        return pd.DataFrame()

    for model, group in df.groupby(model_col):
        if len(group) < 100:
            continue

        # Predicted uncertainty
        pred_sigma = group[sigma_col].values

        # Actual error
        if 'y_true_original' in group.columns:
            actual_error = np.abs(group['y_pred_mean'].values - group['y_true_original'].values)
        else:
            actual_error = np.abs(group['y_pred_mean'].values - group['y_true'].values)

        # Remove NaN/Inf
        mask = np.isfinite(pred_sigma) & np.isfinite(actual_error) & (pred_sigma > 0)
        pred_sigma = pred_sigma[mask]
        actual_error = actual_error[mask]

        if len(pred_sigma) < 100:
            continue

        # Correlation
        corr, p_val = stats.spearmanr(pred_sigma, actual_error)

        # ECE (Expected Calibration Error)
        # Bin by predicted uncertainty and compare to actual error
        n_bins = 10
        bins = np.percentile(pred_sigma, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)

        ece = 0
        for i in range(len(bins) - 1):
            mask = (pred_sigma >= bins[i]) & (pred_sigma < bins[i + 1])
            if mask.sum() > 0:
                bin_pred = pred_sigma[mask].mean()
                bin_actual = actual_error[mask].mean()
                bin_weight = mask.sum() / len(pred_sigma)
                ece += bin_weight * np.abs(bin_pred - bin_actual)

        results.append({
            'model': model,
            'n_samples': len(pred_sigma),
            'mean_pred_sigma': pred_sigma.mean(),
            'mean_actual_error': actual_error.mean(),
            'uncertainty_error_corr': corr,
            'uncertainty_error_pval': p_val,
            'ece': ece,
        })

    return pd.DataFrame(results)


def calculate_noise_tracking(df, model_col='model'):
    """Calculate how well uncertainty tracks injected noise.

    This is the KEY metric for the hypothesis.
    """
    results = []

    if 'injected_noise' not in df.columns:
        print("Warning: 'injected_noise' column not found")
        return pd.DataFrame()

    # Columns for uncertainty
    uncertainty_cols = {
        'total': 'y_pred_std_calibrated',
        'aleatoric': 'aleatoric_uncertainty',
        'epistemic': 'epistemic_uncertainty'
    }

    # Check which columns exist
    available_cols = {k: v for k, v in uncertainty_cols.items() if v in df.columns}

    if not available_cols:
        # Try alternatives
        if 'y_pred_std' in df.columns:
            available_cols['total'] = 'y_pred_std'

    for model, group in df.groupby(model_col):
        if len(group) < 100:
            continue

        # Injected noise magnitude (absolute value)
        noise_mag = np.abs(group['injected_noise'].values)

        result = {'model': model, 'n_samples': len(group)}

        for unc_name, unc_col in available_cols.items():
            if unc_col not in group.columns:
                continue

            unc_values = group[unc_col].values

            # Remove NaN/Inf
            mask = np.isfinite(unc_values) & np.isfinite(noise_mag)
            unc_clean = unc_values[mask]
            noise_clean = noise_mag[mask]

            if len(unc_clean) < 100:
                continue

            # Correlation with noise
            corr, p_val = stats.spearmanr(unc_clean, noise_clean)

            result[f'{unc_name}_noise_corr'] = corr
            result[f'{unc_name}_noise_pval'] = p_val

        results.append(result)

    return pd.DataFrame(results)


def calculate_sigma_level_tracking(df, model_col='model'):
    """Calculate mean uncertainty at each sigma level.

    Shows if models "see" increasing noise.
    """
    results = []

    if 'sigma' not in df.columns:
        return pd.DataFrame()

    unc_col = 'y_pred_std_calibrated'
    if unc_col not in df.columns:
        for alt in ['y_pred_std', 'uncertainty']:
            if alt in df.columns:
                unc_col = alt
                break

    for model, group in df.groupby(model_col):
        for sigma, sigma_group in group.groupby('sigma'):
            unc_values = sigma_group[unc_col].values
            unc_values = unc_values[np.isfinite(unc_values)]

            if len(unc_values) > 10:
                results.append({
                    'model': model,
                    'sigma': sigma,
                    'mean_uncertainty': unc_values.mean(),
                    'std_uncertainty': unc_values.std(),
                    'n_samples': len(unc_values)
                })

    return pd.DataFrame(results)


def plot_calibration(df, output_dir):
    """Plot calibration: predicted uncertainty vs actual error."""
    fig, ax = plt.subplots(figsize=(6, 6))

    models = df['model'].unique()

    for model in models:
        model_df = df[df['model'] == model]
        color = MODEL_COLORS.get(model, '#333333')

        pred = model_df['y_pred_std_calibrated'].values
        if 'y_true_original' in model_df.columns:
            actual = np.abs(model_df['y_pred_mean'].values - model_df['y_true_original'].values)
        else:
            actual = np.abs(model_df['y_pred_mean'].values - model_df['y_true'].values)

        # Bin and plot
        mask = np.isfinite(pred) & np.isfinite(actual) & (pred > 0)
        pred, actual = pred[mask], actual[mask]

        if len(pred) < 100:
            continue

        bins = np.percentile(pred, np.linspace(0, 100, 11))
        bin_centers = []
        bin_errors = []

        for i in range(len(bins) - 1):
            bin_mask = (pred >= bins[i]) & (pred < bins[i + 1])
            if bin_mask.sum() > 0:
                bin_centers.append(pred[bin_mask].mean())
                bin_errors.append(actual[bin_mask].mean())

        ax.plot(bin_centers, bin_errors, 'o-', label=model, color=color, markersize=4)

    # Diagonal (perfect calibration)
    lims = [0, ax.get_xlim()[1]]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect calibration')

    ax.set_xlabel('Predicted Uncertainty (σ)')
    ax.set_ylabel('Actual Error (|y - ŷ|)')
    ax.set_title('Calibration: Predicted Uncertainty vs Actual Error')
    ax.legend(loc='upper left', fontsize=6)
    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6a_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig6a_calibration.png")


def plot_uncertainty_vs_sigma(sigma_df, output_dir):
    """Plot mean uncertainty vs injected sigma level."""
    fig, ax = plt.subplots(figsize=(8, 5))

    models = sigma_df['model'].unique()

    for model in models:
        model_df = sigma_df[sigma_df['model'] == model].sort_values('sigma')
        color = MODEL_COLORS.get(model, '#333333')

        ax.plot(model_df['sigma'], model_df['mean_uncertainty'],
                'o-', label=model, color=color, markersize=4)

    ax.set_xlabel('Injected Noise Level (σ)')
    ax.set_ylabel('Mean Predicted Uncertainty')
    ax.set_title('Do Models Track Noise Level?')
    ax.legend(loc='upper left', fontsize=6, ncol=2)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7a_uncertainty_vs_sigma.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7a_uncertainty_vs_sigma.png")


def plot_noise_correlation_bars(noise_df, output_dir):
    """Bar chart of uncertainty-noise correlations."""
    if 'total_noise_corr' not in noise_df.columns:
        print("Skipping noise correlation plot - no data")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    corr_cols = ['total_noise_corr', 'aleatoric_noise_corr', 'epistemic_noise_corr']
    titles = ['Total Uncertainty', 'Aleatoric Uncertainty', 'Epistemic Uncertainty']

    for ax, col, title in zip(axes, corr_cols, titles):
        if col not in noise_df.columns:
            ax.set_visible(False)
            continue

        data = noise_df[['model', col]].dropna().sort_values(col, ascending=False)

        colors = [MODEL_COLORS.get(m, '#333333') for m in data['model']]
        ax.barh(data['model'], data[col], color=colors)
        ax.set_xlabel('Correlation with Injected Noise')
        ax.set_title(title)
        ax.axvline(0, color='black', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig7c_noise_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig7c_noise_correlations.png")


def plot_ece_comparison(calib_df, output_dir):
    """Bar chart of ECE (Expected Calibration Error)."""
    if 'ece' not in calib_df.columns:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    data = calib_df[['model', 'ece']].dropna().sort_values('ece')
    colors = [MODEL_COLORS.get(m, '#333333') for m in data['model']]

    ax.barh(data['model'], data['ece'], color=colors)
    ax.set_xlabel('Expected Calibration Error (lower = better)')
    ax.set_title('Calibration Quality by Model')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig6c_ece.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: fig6c_ece.png")


def generate_report(calib_df, noise_df, sigma_df, output_dir):
    """Generate text report of uncertainty analysis."""
    lines = []
    lines.append("=" * 80)
    lines.append("UNCERTAINTY ANALYSIS REPORT")
    lines.append("Layer 3: Why are some models more noise-robust?")
    lines.append("=" * 80)

    lines.append("\n" + "=" * 80)
    lines.append("CALIBRATION METRICS")
    lines.append("=" * 80)
    if len(calib_df) > 0:
        lines.append("\n" + calib_df.to_string(index=False))
    else:
        lines.append("No calibration data available")

    lines.append("\n" + "=" * 80)
    lines.append("UNCERTAINTY-NOISE CORRELATIONS (KEY FINDING)")
    lines.append("=" * 80)
    lines.append("\nHypothesis: Robust models have high aleatoric-noise correlation")
    lines.append("           (they 'see' noise as uncertainty, not signal)")
    if len(noise_df) > 0:
        lines.append("\n" + noise_df.to_string(index=False))
    else:
        lines.append("No noise tracking data available")

    lines.append("\n" + "=" * 80)
    lines.append("INTERPRETATION")
    lines.append("=" * 80)

    if len(noise_df) > 0 and 'aleatoric_noise_corr' in noise_df.columns:
        # Find best/worst
        best = noise_df.loc[noise_df['aleatoric_noise_corr'].idxmax()]
        worst = noise_df.loc[noise_df['aleatoric_noise_corr'].idxmin()]

        lines.append(f"\nHighest aleatoric-noise correlation: {best['model']} (ρ={best['aleatoric_noise_corr']:.3f})")
        lines.append(f"Lowest aleatoric-noise correlation: {worst['model']} (ρ={worst['aleatoric_noise_corr']:.3f})")
        lines.append("\nModels with high aleatoric-noise correlation correctly attribute")
        lines.append("injected noise to irreducible uncertainty, hence are robust.")

    report_path = output_dir / "uncertainty_report.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"\nSaved report to {report_path}")

    # Also print
    print('\n'.join(lines))


def main():
    print("=" * 80)
    print("UNCERTAINTY ANALYSIS")
    print("=" * 80)

    # Load data
    print("\nLoading uncertainty data...")
    df = load_uncertainty_data()

    if df is None or len(df) == 0:
        print("No uncertainty data found!")
        print("Expected files matching: uncertainty_*_uncertainty_values.csv")
        print("                     or: anova_*_uncertainty_values.csv")
        return

    print(f"\nColumns: {list(df.columns)}")
    print(f"Models: {df['model'].unique() if 'model' in df.columns else 'N/A'}")

    # Calculate metrics
    print("\n" + "=" * 80)
    print("CALCULATING METRICS")
    print("=" * 80)

    print("\n1. Calibration metrics...")
    calib_df = calculate_calibration_metrics(df)

    print("\n2. Noise tracking (sample-level)...")
    noise_df = calculate_noise_tracking(df)

    print("\n3. Sigma-level tracking...")
    sigma_df = calculate_sigma_level_tracking(df)

    # Generate plots
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)

    if len(df) > 0:
        plot_calibration(df, OUTPUT_DIR)

    if len(sigma_df) > 0:
        plot_uncertainty_vs_sigma(sigma_df, OUTPUT_DIR)

    if len(noise_df) > 0:
        plot_noise_correlation_bars(noise_df, OUTPUT_DIR)

    if len(calib_df) > 0:
        plot_ece_comparison(calib_df, OUTPUT_DIR)

    # Save data
    if len(calib_df) > 0:
        calib_df.to_csv(OUTPUT_DIR / 'calibration_metrics.csv', index=False)
    if len(noise_df) > 0:
        noise_df.to_csv(OUTPUT_DIR / 'noise_tracking_metrics.csv', index=False)
    if len(sigma_df) > 0:
        sigma_df.to_csv(OUTPUT_DIR / 'sigma_level_tracking.csv', index=False)

    # Generate report
    generate_report(calib_df, noise_df, sigma_df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"ANALYSIS COMPLETE - Results in {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
