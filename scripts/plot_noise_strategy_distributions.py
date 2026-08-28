#!/usr/bin/env python3
"""
Visualize the effect of each noise strategy on data distributions.

Creates a figure showing how each noise injection strategy transforms
the clean label distribution at different sigma levels.

Usage: python plot_noise_strategy_distributions.py [--output ../results/paper_figures/noise_strategy_visualization.png]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import argparse
from pathlib import Path

# Style
sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
})

STRATEGY_COLORS = {
    'legacy': '#3498db',
    'valprop': '#e74c3c',
    'quantile': '#2ecc71',
    'threshold': '#9b59b6',
    'outlier': '#f39c12',
    'hetero': '#1abc9c',
}

STRATEGY_LABELS = {
    'legacy': 'Gaussian (Legacy)',
    'valprop': 'Value-Proportional',
    'quantile': 'Quantile-Based',
    'threshold': 'Threshold-Based',
    'outlier': 'Outlier-Targeted',
    'hetero': 'Heteroscedastic',
}


def apply_noise(y, sigma, strategy):
    """Apply noise strategy to labels."""
    n = len(y)

    if strategy == 'legacy':
        # Uniform Gaussian noise
        noise = np.random.normal(0, sigma, n)

    elif strategy == 'valprop':
        # Value-proportional: noise scales with |y|
        noise = np.random.normal(0, 1, n) * (sigma + 0.1 * np.abs(y))

    elif strategy == 'quantile':
        # More noise on extreme quantiles
        quantiles = stats.rankdata(y) / len(y)
        multipliers = np.where((quantiles < 0.1) | (quantiles > 0.9), 2.0, 0.1)
        noise = np.random.normal(0, sigma, n) * multipliers

    elif strategy == 'threshold':
        # High noise above median, low below
        median = np.median(y)
        multipliers = np.where(y > median, 2.0, 0.1)
        noise = np.random.normal(0, sigma, n) * multipliers

    elif strategy == 'outlier':
        # Target statistical outliers with heavy noise
        z_scores = np.abs(stats.zscore(y))
        multipliers = np.where(z_scores > 2.0, 3.0, 0.1)
        noise = np.random.normal(0, sigma, n) * multipliers

    elif strategy == 'hetero':
        # Heteroscedastic: variance depends on value
        alpha, beta = 0.1, 0.05
        variance = alpha * sigma**2 + beta * sigma**2 * np.abs(y)
        noise = np.random.normal(0, np.sqrt(variance))

    return y + noise


def plot_noise_strategies(output_path):
    """Create visualization of noise strategies."""
    np.random.seed(42)

    # Generate synthetic "clean" data resembling HOMO-LUMO gap distribution
    # Mixture of gaussians to simulate realistic molecular property distribution
    n_samples = 2000
    y_clean = np.concatenate([
        np.random.normal(-0.5, 0.3, n_samples // 3),
        np.random.normal(0.2, 0.4, n_samples // 3),
        np.random.normal(0.8, 0.25, n_samples // 3 + n_samples % 3),
    ])

    strategies = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
    sigmas = [0.0, 0.3, 0.6, 1.0]

    # Create figure
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(6, 4, hspace=0.4, wspace=0.3)

    for i, strategy in enumerate(strategies):
        for j, sigma in enumerate(sigmas):
            ax = fig.add_subplot(gs[i, j])

            if sigma == 0.0:
                y_noisy = y_clean.copy()
            else:
                y_noisy = apply_noise(y_clean, sigma, strategy)

            # Plot histogram
            ax.hist(y_clean, bins=50, alpha=0.4, color='gray', label='Clean', density=True)
            if sigma > 0:
                ax.hist(y_noisy, bins=50, alpha=0.6, color=STRATEGY_COLORS[strategy],
                       label=f'σ={sigma}', density=True)

            # Labels
            if j == 0:
                ax.set_ylabel(STRATEGY_LABELS[strategy], fontsize=9, fontweight='bold')
            if i == 0:
                ax.set_title(f'σ = {sigma}', fontsize=10)
            if i == 5:
                ax.set_xlabel('Normalized Value')

            # Clean up
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            # Show noise magnitude for non-zero sigma
            if sigma > 0:
                rmse = np.sqrt(np.mean((y_noisy - y_clean)**2))
                ax.text(0.95, 0.95, f'RMSE={rmse:.2f}', transform=ax.transAxes,
                       ha='right', va='top', fontsize=7, color='gray')

    # Add overall title
    fig.suptitle('Effect of Noise Injection Strategies on Label Distribution',
                 fontsize=12, fontweight='bold', y=0.98)

    # Add legend
    fig.text(0.5, 0.02, 'Gray = Clean labels | Colored = After noise injection',
             ha='center', fontsize=9, style='italic')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")

    # Also create a simpler 2x3 version showing just sigma=0.5
    fig2, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    sigma = 0.5
    for i, strategy in enumerate(strategies):
        ax = axes[i]
        y_noisy = apply_noise(y_clean, sigma, strategy)

        ax.hist(y_clean, bins=50, alpha=0.4, color='gray', label='Clean', density=True)
        ax.hist(y_noisy, bins=50, alpha=0.6, color=STRATEGY_COLORS[strategy],
               label=f'Noisy (σ={sigma})', density=True)

        ax.set_title(STRATEGY_LABELS[strategy], fontsize=10, fontweight='bold',
                    color=STRATEGY_COLORS[strategy])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        rmse = np.sqrt(np.mean((y_noisy - y_clean)**2))
        ax.text(0.95, 0.95, f'RMSE={rmse:.2f}', transform=ax.transAxes,
               ha='right', va='top', fontsize=8)

        if i == 0:
            ax.legend(loc='upper left', fontsize=7)

    fig2.suptitle('Noise Strategy Comparison (σ = 0.5)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    simple_path = output_path.replace('.png', '_simple.png')
    plt.savefig(simple_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {simple_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize noise strategies')
    parser.add_argument('--output', '-o', type=str,
                       default='../results/paper_figures/noise_strategy_visualization.png',
                       help='Output path for figure')
    args = parser.parse_args()

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    plot_noise_strategies(args.output)


if __name__ == '__main__':
    main()
