#!/usr/bin/env python3
"""
Master Figure Generation Script for Noise Robustness Paper

Produces all 7 main figures + tables for the paper structure:

PART 1: THE WHAT
  Figure 1: Global Overview (R² vs σ line plot, NDS heatmap)
  Figure 2: ANOVA Decomposition (η² bars by strategy)
  Figure 3: Ranking Consistency (bump charts, cross-dataset)

PART 2: THE WHY
  Figure 4: DNN Family Comparison (DNN vs BNN variants)
  Figure 5: MLP Family + RF/QRF Comparison
  Figure 6: Uncertainty Quality (calibration, coverage, ECE)
  Figure 7: Uncertainty Tracks Noise (KEY FIGURE)

Run: python generate_paper_figures.py [--qm9-dir ../results] [--validation-dir /path/to/kirby/results]

Outputs:
  results/paper_figures/
    fig1_global_overview.png
    fig2_anova_decomposition.png
    fig3_ranking_consistency.png
    fig4_dnn_family.png
    fig5_mlp_rf_comparison.png
    fig6_uncertainty_quality.png
    fig7_uncertainty_noise_tracking.png
    table1_anova_summary.csv
    table2_nds_by_strategy.csv
    table3_probabilistic_comparison.csv
    table4_uncertainty_metrics.csv
    paper_figures_report.txt
"""

# =============================================================================
# CONFIGURATION - REVIEW AFTER SEEING RESULTS
# =============================================================================
#
# These settings control which strategies/representations are highlighted.
# They were set BEFORE seeing results and should be reviewed.
#
# DECISION POINTS TO REVIEW:
#   1. PRIMARY_STRATEGY: Which strategy for "clean example" panels?
#      - Currently 'legacy' (Gaussian) as baseline
#      - REVIEW: Is this the most informative? Or should we lead with realistic noise?
#
#   2. CONTRAST_STRATEGY: Which strategy to show as "different from Gaussian"?
#      - Currently 'hetero' (heteroscedastic) - chosen arbitrarily
#      - REVIEW: Which strategy shows most different behavior? Check:
#        * Does ranking change significantly under this strategy?
#        * Is the effect size (NDS difference) meaningful?
#        * Candidates: 'hetero', 'valprop', 'outlier'
#
#   3. PRIMARY_REP: Which representation for main figures?
#      - Currently 'pdv' (physics-based descriptors)
#      - REVIEW: Does this generalize? Check SNS shows same pattern.
#
#   4. SUPPLEMENTARY_REP: Which rep for supplementary validation?
#      - Currently 'sns' (SMILES-based)
#      - REVIEW: Should we also include 'ecfp4'?
#
# WHAT TO LOOK FOR IN RESULTS:
#   - Kendall's W across strategies: If W > 0.7, rankings are consistent
#     and strategy choice matters less for main figures
#   - If W < 0.5, strategies produce different rankings - need to show multiple
#   - Check if any strategy breaks the "model > rep for robustness" finding
#   - Check if uncertainty-noise correlation holds across strategies
#
# =============================================================================

# Primary choices (change these based on results)
PRIMARY_STRATEGY = 'legacy'      # REVIEW: Main example strategy
CONTRAST_STRATEGY = 'hetero'     # REVIEW: Strategy to contrast with legacy
PRIMARY_REP = 'pdv'              # REVIEW: Main representation
SUPPLEMENTARY_REP = 'sns'        # REVIEW: Supplementary representation

# All strategies for completeness checks
ALL_STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']

# Flag to generate supplementary figures
GENERATE_SUPPLEMENTARY = True

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STYLE SETTINGS (Journal of Cheminformatics)
# =============================================================================

sns.set_style("ticks")
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
})

# Color palettes - Colorblind-friendly (Okabe-Ito + Wong palette)
# Avoids red-green confusion, uses blue-orange contrast
STRATEGY_COLORS = {
    'legacy': '#0072B2',       # Blue
    'valprop': '#E69F00',      # Orange
    'quantile': '#56B4E9',     # Sky blue
    'threshold': '#CC79A7',    # Pink/magenta
    'outlier': '#F0E442',      # Yellow
    'hetero': '#009E73',       # Teal/bluish-green
}

STRATEGY_LABELS = {
    'legacy': 'Gaussian',
    'valprop': 'Value-Prop.',
    'quantile': 'Quantile',
    'threshold': 'Threshold',
    'outlier': 'Outlier',
    'hetero': 'Heteroscedastic',
}

MODEL_COLORS = {
    # Colorblind-friendly palette (Okabe-Ito)
    # Deterministic - cooler tones
    'rf': '#0072B2',           # Blue
    'xgboost': '#E69F00',      # Orange
    'dnn': '#56B4E9',          # Sky blue
    'mlp': '#CC79A7',          # Pink/magenta
    'svm': '#999999',          # Gray
    'lgb': '#D55E00',          # Vermillion
    # Probabilistic - warmer/distinct tones
    'qrf': '#009E73',          # Teal
    'ngboost': '#F0E442',      # Yellow
    'gauche': '#0072B2',       # Blue (GP variant)
    # BNN variants - gradient of blues/teals
    'dnn_bnn_full': '#332288',     # Dark purple
    'dnn_bnn_last': '#6699CC',     # Light blue
    'dnn_bnn_variational': '#88CCEE',  # Cyan
    'mlp_bnn_full': '#882255',     # Wine
    'mlp_bnn_last': '#AA4499',     # Magenta
    'mlp_bnn_variational': '#CC6677',  # Rose
    # Conformal
    'conformal_rf': '#44AA99',     # Teal
    'conformal_qrf': '#117733',    # Dark green
    'conformal_dnn': '#DDCC77',    # Sand
}

MODEL_LABELS = {
    # Deterministic
    'rf': 'RF',
    'xgboost': 'XGBoost',
    'dnn': 'DNN',
    'mlp': 'MLP',
    'lgb': 'LightGBM',
    'svm': 'SVM',
    # Probabilistic
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    'gauche': 'GP (Gauche)',
    # BNN variants
    'dnn_bnn_full': 'DNN-BNN (Full)',
    'dnn_bnn_last': 'DNN-BNN (Last)',
    'dnn_bnn_variational': 'DNN-BNN (Var)',
    'mlp_bnn_full': 'MLP-BNN (Full)',
    'mlp_bnn_last': 'MLP-BNN (Last)',
    'mlp_bnn_variational': 'MLP-BNN (Var)',
    # Conformal
    'conformal_rf': 'CP-RF',
    'conformal_qrf': 'CP-QRF',
    'conformal_dnn': 'CP-DNN',
}

REP_LABELS = {
    'pdv': 'PDV',
    'ecfp4': 'ECFP4',
    'sns': 'SNS',
    'smiles': 'SMILES',
    'randomized_smiles': 'R-SMILES',
    'mhggnn': 'MHG-GNN',
}

def get_model_label(model):
    """Get clean display label for model."""
    return MODEL_LABELS.get(model, model.upper())

def get_rep_label(rep):
    """Get clean display label for representation."""
    return REP_LABELS.get(rep, rep.upper())

DATASET_MARKERS = {
    'QM9': 'o',
    'LogD': 's',
    'Caco2_Efflux': '^',
    'hERG-Ki': 'D',
}

BASELINE_THRESHOLD = 0.6

# =============================================================================
# DATA LOADING
# =============================================================================

def load_anova_data(results_dir):
    """Load all anova_*.csv files."""
    results_dir = Path(results_dir)
    all_data = []

    for f in results_dir.glob("anova_*.csv"):
        if '_uncertainty_values' in f.name:
            continue
        try:
            df = pd.read_csv(f)
            # Parse filename: anova_{strategy}_{rep}_{model}.csv
            parts = f.stem.split('_')
            if len(parts) >= 4 and parts[0] == 'anova':
                df['strategy'] = parts[1]
            df['source_file'] = f.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded QM9 ANOVA data: {len(combined)} rows from {len(all_data)} files")
        return combined
    return None


def load_uncertainty_data(results_dir):
    """Load uncertainty_*_uncertainty_values.csv files."""
    results_dir = Path(results_dir)
    all_data = []

    patterns = ["uncertainty_*_uncertainty_values.csv", "*_uncertainty_values.csv"]

    for pattern in patterns:
        for f in results_dir.glob(pattern):
            try:
                df = pd.read_csv(f)
                df['source_file'] = f.name
                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded uncertainty data: {len(combined)} rows")
        return combined
    return None


def load_validation_data(validation_dir):
    """Load validation data from KIRBy (combined_summary.csv)."""
    if validation_dir is None:
        return None

    validation_dir = Path(validation_dir)
    summary_file = validation_dir / 'combined_summary.csv'

    if summary_file.exists():
        df = pd.read_csv(summary_file)
        print(f"Loaded validation data: {len(df)} rows")
        return df

    # Try loading individual summaries
    all_data = []
    for subdir in validation_dir.iterdir():
        if subdir.is_dir():
            summary = subdir / 'summary.csv'
            if summary.exists():
                df = pd.read_csv(summary)
                all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded validation data: {len(combined)} rows from {len(all_data)} datasets")
        return combined

    return None


# =============================================================================
# METRIC CALCULATIONS
# =============================================================================

def calculate_nds(df, baseline_threshold=BASELINE_THRESHOLD):
    """Calculate NDS for each model-rep-strategy combination.

    Returns:
        nds_df: DataFrame of NDS results for configs above threshold
        excluded_df: DataFrame of all excluded configs (baseline < threshold)
            Includes 'marginal' column (True if baseline in [0.5, threshold))
    """
    nds_results = []
    excluded = []

    for (model, rep, strategy), group in df.groupby(['model', 'rep', 'strategy']):
        # Average across iterations first
        avg_group = group.groupby('sigma')['r2'].mean().reset_index()
        avg_group = avg_group.sort_values('sigma')

        if len(avg_group) < 3:
            continue

        baseline = avg_group[avg_group['sigma'] == 0.0]['r2'].values
        if len(baseline) == 0:
            continue
        baseline = baseline[0]

        # Calculate NDS regardless (needed for marginal reporting)
        try:
            slope, intercept, r_val, p_val, std_err = stats.linregress(
                avg_group['sigma'], avg_group['r2']
            )
        except:
            continue

        if baseline < baseline_threshold:
            excluded.append({
                'model': model, 'rep': rep, 'strategy': strategy,
                'baseline': baseline,
                'nds': slope,
                'marginal': baseline >= 0.5,
            })
            continue

        nds_results.append({
            'model': model,
            'rep': rep,
            'strategy': strategy,
            'nds': slope,
            'baseline_r2': baseline,
            'r2_fit': r_val**2,
        })

    return pd.DataFrame(nds_results), pd.DataFrame(excluded)


def calculate_coverage(y_true, y_pred, uncertainty, k=1):
    """
    Calculate empirical coverage at k*sigma interval.

    Coverage(kσ) = (1/N) * Σ 1[|y_i - ŷ_i| ≤ k*û_i]

    Args:
        y_true: True values
        y_pred: Predicted values
        uncertainty: Predicted uncertainty (std)
        k: Multiplier (1 for 68% target, 2 for 95% target)

    Returns:
        coverage: Fraction of predictions within k*sigma interval
    """
    errors = np.abs(y_true - y_pred)
    within_interval = errors <= k * uncertainty
    return np.mean(within_interval)


def wilcoxon_paired_test(nds_df, model_base, model_variant, rep='pdv', strategy='legacy'):
    """
    Wilcoxon signed-rank test for paired model comparison.

    Compares NDS values between a base model and its variant
    (e.g., DNN vs DNN-BNN-Full) across matched conditions.

    Returns:
        dict with statistic, p_value, and interpretation
    """
    base_data = nds_df[(nds_df['model'] == model_base) &
                       (nds_df['rep'] == rep) &
                       (nds_df['strategy'] == strategy)]
    var_data = nds_df[(nds_df['model'] == model_variant) &
                      (nds_df['rep'] == rep) &
                      (nds_df['strategy'] == strategy)]

    if len(base_data) == 0 or len(var_data) == 0:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False, 'n': 0}

    base_nds = base_data['nds'].values
    var_nds = var_data['nds'].values

    # Need paired data - if different lengths, can't do paired test
    n = min(len(base_nds), len(var_nds))
    if n < 5:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False, 'n': n}

    try:
        stat, p_val = stats.wilcoxon(base_nds[:n], var_nds[:n], alternative='two-sided')
        return {
            'statistic': stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'n': n,
            'base_mean': np.mean(base_nds),
            'var_mean': np.mean(var_nds),
            'improvement': np.mean(var_nds) - np.mean(base_nds)  # Positive = variant better (less negative NDS)
        }
    except Exception as e:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False, 'n': n, 'error': str(e)}


def calculate_kendalls_w(nds_df, rep='pdv'):
    """
    Calculate Kendall's W (coefficient of concordance) for ranking consistency.

    Measures whether model rankings are consistent across different noise strategies.
    W ranges from 0 (no agreement) to 1 (complete agreement).

    Args:
        nds_df: DataFrame with 'model', 'strategy', 'nds' columns
        rep: Representation to filter to (avoid mixing)

    Returns:
        dict with W statistic, p_value, and interpretation
    """
    # Filter to single representation
    df = nds_df[nds_df['rep'] == rep] if 'rep' in nds_df.columns else nds_df

    if len(df) == 0:
        return {'W': np.nan, 'p_value': np.nan, 'interpretation': 'No data'}

    # Create ranking matrix: rows = models, columns = strategies
    strategies = df['strategy'].unique()
    models = df['model'].unique()

    if len(strategies) < 2 or len(models) < 3:
        return {'W': np.nan, 'p_value': np.nan, 'interpretation': 'Insufficient data'}

    # Build ranking matrix
    rank_matrix = []
    for strategy in strategies:
        strat_data = df[df['strategy'] == strategy]
        model_nds = strat_data.groupby('model')['nds'].mean()
        ranks = model_nds.rank(ascending=False)  # Higher NDS (less negative) = better rank
        rank_matrix.append(ranks)

    # Convert to matrix
    rank_df = pd.DataFrame(rank_matrix).T
    rank_df.columns = strategies

    # Drop models with missing rankings
    rank_df = rank_df.dropna()

    if len(rank_df) < 3:
        return {'W': np.nan, 'p_value': np.nan, 'interpretation': 'Insufficient complete rankings'}

    # Calculate Kendall's W
    # W = 12 * S / (k^2 * (n^3 - n))
    k = len(strategies)  # number of raters
    n = len(rank_df)     # number of items

    rank_sums = rank_df.sum(axis=1)
    mean_rank_sum = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank_sum) ** 2)

    W = 12 * S / (k**2 * (n**3 - n))

    # Chi-squared test for significance
    chi2 = k * (n - 1) * W
    p_value = 1 - stats.chi2.cdf(chi2, n - 1)

    # Interpretation
    if W >= 0.7:
        interp = 'Strong agreement - rankings consistent across strategies'
    elif W >= 0.5:
        interp = 'Moderate agreement - some ranking variation'
    else:
        interp = 'Weak agreement - rankings differ substantially across strategies'

    return {
        'W': W,
        'chi2': chi2,
        'p_value': p_value,
        'n_models': n,
        'n_strategies': k,
        'interpretation': interp
    }


def run_anova_decomposition(df, sigma_value=0.3):
    """Run two-way ANOVA for model/rep effects."""
    df_sigma = df[np.abs(df['sigma'] - sigma_value) < 0.05].copy()

    if len(df_sigma) == 0:
        return None

    df_sigma = df_sigma[df_sigma['r2'] > -10].dropna(subset=['r2', 'model', 'rep'])
    df_sigma = df_sigma[~df_sigma['rep'].isin(['randomized_smiles', 'random_smiles'])]

    if len(df_sigma) < 10:
        return None

    grand_mean = df_sigma['r2'].mean()
    total_ss = ((df_sigma['r2'] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    # Model effect
    model_means = df_sigma.groupby('model')['r2'].mean()
    model_counts = df_sigma.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)

    # Representation effect
    rep_means = df_sigma.groupby('rep')['r2'].mean()
    rep_counts = df_sigma.groupby('rep').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)

    # Interaction
    interaction_means = df_sigma.groupby(['model', 'rep'])['r2'].mean()
    interaction_counts = df_sigma.groupby(['model', 'rep']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        if model in model_means.index and rep in rep_means.index:
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2

    ss_residual = total_ss - ss_model - ss_rep - ss_interaction

    return {
        'eta2_model': (ss_model / total_ss) * 100,
        'eta2_rep': (ss_rep / total_ss) * 100,
        'eta2_interaction': (ss_interaction / total_ss) * 100,
        'eta2_residual': (ss_residual / total_ss) * 100,
        'n': len(df_sigma),
    }


def run_robustness_anova(df, baseline_threshold=BASELINE_THRESHOLD):
    """Run ANOVA on NDS values."""
    nds_data = []

    for (model, rep, iteration), group in df.groupby(['model', 'rep', 'iteration']):
        group = group.sort_values('sigma')
        if len(group) < 3:
            continue

        baseline = group[group['sigma'] == 0.0]
        if len(baseline) == 0 or baseline['r2'].values[0] < baseline_threshold:
            continue

        try:
            slope, _, _, _, _ = stats.linregress(group['sigma'], group['r2'])
            nds_data.append({'model': model, 'rep': rep, 'iteration': iteration, 'nds': slope})
        except:
            continue

    if len(nds_data) < 10:
        return None

    nds_df = pd.DataFrame(nds_data)
    nds_df = nds_df[~nds_df['rep'].isin(['randomized_smiles', 'random_smiles'])]

    grand_mean = nds_df['nds'].mean()
    total_ss = ((nds_df['nds'] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    model_means = nds_df.groupby('model')['nds'].mean()
    model_counts = nds_df.groupby('model').size()
    ss_model = sum(model_counts * (model_means - grand_mean) ** 2)

    rep_means = nds_df.groupby('rep')['nds'].mean()
    rep_counts = nds_df.groupby('rep').size()
    ss_rep = sum(rep_counts * (rep_means - grand_mean) ** 2)

    interaction_means = nds_df.groupby(['model', 'rep'])['nds'].mean()
    interaction_counts = nds_df.groupby(['model', 'rep']).size()
    ss_interaction = 0
    for (model, rep), count in interaction_counts.items():
        if model in model_means.index and rep in rep_means.index:
            cell_mean = interaction_means[(model, rep)]
            expected = model_means[model] + rep_means[rep] - grand_mean
            ss_interaction += count * (cell_mean - expected) ** 2

    ss_residual = total_ss - ss_model - ss_rep - ss_interaction

    return {
        'eta2_model': (ss_model / total_ss) * 100,
        'eta2_rep': (ss_rep / total_ss) * 100,
        'eta2_interaction': (ss_interaction / total_ss) * 100,
        'eta2_residual': (ss_residual / total_ss) * 100,
        'n': len(nds_df),
    }


# =============================================================================
# METHODS FIGURE: NOISE STRATEGY DISTRIBUTIONS
# =============================================================================

def create_methods_figure(output_dir):
    """
    Methods Figure: Visualization of how each noise strategy transforms labels.

    This figure belongs in the Methods section to help readers understand
    what each noise injection strategy does before seeing results.
    """
    from scipy import stats as sp_stats

    np.random.seed(42)

    # Generate synthetic data resembling a molecular property distribution
    n_samples = 2000
    y_clean = np.concatenate([
        np.random.normal(-0.5, 0.3, n_samples // 3),
        np.random.normal(0.2, 0.4, n_samples // 3),
        np.random.normal(0.8, 0.25, n_samples // 3 + n_samples % 3),
    ])

    def apply_noise(y, sigma, strategy):
        """Apply noise strategy to labels."""
        n = len(y)

        if strategy == 'legacy':
            noise = np.random.normal(0, sigma, n)
        elif strategy == 'valprop':
            noise = np.random.normal(0, 1, n) * (sigma + 0.1 * np.abs(y))
        elif strategy == 'quantile':
            quantiles = sp_stats.rankdata(y) / len(y)
            multipliers = np.where((quantiles < 0.1) | (quantiles > 0.9), 2.0, 0.1)
            noise = np.random.normal(0, sigma, n) * multipliers
        elif strategy == 'threshold':
            median = np.median(y)
            multipliers = np.where(y > median, 2.0, 0.1)
            noise = np.random.normal(0, sigma, n) * multipliers
        elif strategy == 'outlier':
            z_scores = np.abs(sp_stats.zscore(y))
            multipliers = np.where(z_scores > 2.0, 3.0, 0.1)
            noise = np.random.normal(0, sigma, n) * multipliers
        elif strategy == 'hetero':
            alpha, beta = 0.1, 0.05
            variance = alpha * sigma**2 + beta * sigma**2 * np.abs(y)
            noise = np.random.normal(0, np.sqrt(variance))
        else:
            noise = np.zeros(n)

        return y + noise

    strategies = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
    sigma = 0.5  # Representative noise level

    # Create 2x3 figure
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for i, strategy in enumerate(strategies):
        ax = axes[i]
        y_noisy = apply_noise(y_clean, sigma, strategy)

        ax.hist(y_clean, bins=50, alpha=0.4, color='#666666', label='Clean', density=True)
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

    fig.suptitle('Noise Strategy Comparison (σ = 0.5)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_dir / 'fig_methods_noise_strategies.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved fig_methods_noise_strategies.png")

    # Also create a more detailed version showing sigma progression
    fig2 = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(6, 4, hspace=0.4, wspace=0.3)
    sigmas = [0.0, 0.3, 0.6, 1.0]

    for i, strategy in enumerate(strategies):
        for j, sig in enumerate(sigmas):
            ax = fig2.add_subplot(gs[i, j])

            if sig == 0.0:
                y_noisy = y_clean.copy()
            else:
                y_noisy = apply_noise(y_clean, sig, strategy)

            ax.hist(y_clean, bins=50, alpha=0.4, color='#666666', density=True)
            if sig > 0:
                ax.hist(y_noisy, bins=50, alpha=0.6, color=STRATEGY_COLORS[strategy], density=True)

            if j == 0:
                ax.set_ylabel(STRATEGY_LABELS[strategy], fontsize=9, fontweight='bold')
            if i == 0:
                ax.set_title(f'σ = {sig}', fontsize=10)
            if i == 5:
                ax.set_xlabel('Normalized Value')

            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)

            if sig > 0:
                rmse = np.sqrt(np.mean((y_noisy - y_clean)**2))
                ax.text(0.95, 0.95, f'RMSE={rmse:.2f}', transform=ax.transAxes,
                        ha='right', va='top', fontsize=7, color='gray')

    fig2.suptitle('Effect of Noise Injection Strategies on Label Distribution',
                  fontsize=12, fontweight='bold', y=0.98)
    fig2.text(0.5, 0.02, 'Gray = Clean labels | Colored = After noise injection',
              ha='center', fontsize=9, style='italic')

    plt.savefig(output_dir / 'fig_methods_noise_strategies_detailed.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved fig_methods_noise_strategies_detailed.png")


# =============================================================================
# FIGURE 1: GLOBAL OVERVIEW
# =============================================================================

def create_figure1(df, nds_df, output_dir):
    """Figure 1: Global Overview - R² vs σ line plot + NDS heatmap."""
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1])

    # Panel A: R² vs σ for key models on PDV
    ax_a = fig.add_subplot(gs[0])

    key_models = ['rf', 'qrf', 'dnn', 'mlp', 'ngboost', 'xgboost']
    pdv_data = df[(df['rep'] == 'pdv') & (df['strategy'] == 'legacy')]

    for model in key_models:
        model_data = pdv_data[pdv_data['model'] == model]
        if len(model_data) == 0:
            continue

        avg = model_data.groupby('sigma')['r2'].mean().reset_index()
        color = MODEL_COLORS.get(model, '#333333')
        ax_a.plot(avg['sigma'], avg['r2'], 'o-', label=get_model_label(model),
                  color=color, markersize=4)

    ax_a.set_xlabel('Noise Level (σ)')
    ax_a.set_ylabel('R²')
    ax_a.set_title('A. Performance Degradation (PDV, Gaussian Noise)', fontweight='bold')
    ax_a.legend(loc='lower left', ncol=2)
    ax_a.set_ylim(-0.1, 1.0)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Panel B: NDS heatmap (model × strategy) - PDV ONLY to avoid mixing representations
    ax_b = fig.add_subplot(gs[1])

    if len(nds_df) > 0:
        # Filter to PDV only - don't mix representations
        nds_pdv = nds_df[nds_df['rep'] == 'pdv']
        if len(nds_pdv) == 0:
            # Fallback to most common rep
            nds_pdv = nds_df

        pivot = nds_pdv.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')

        # Reorder columns
        col_order = [c for c in ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
                     if c in pivot.columns]
        pivot = pivot[col_order]

        # Rename index to clean model labels
        pivot.index = [get_model_label(m) for m in pivot.index]

        # Use colorblind-friendly diverging colormap (blue-white-orange)
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                    ax=ax_b, cbar_kws={'label': 'NDS'})
        ax_b.set_xlabel('Noise Strategy')
        ax_b.set_ylabel('Model')
        ax_b.set_title('B. NDS by Model × Strategy (PDV)', fontweight='bold')
        ax_b.set_xticklabels([STRATEGY_LABELS.get(c, c) for c in col_order], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_global_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig1_global_overview.png")


# =============================================================================
# FIGURE 2: ANOVA DECOMPOSITION
# =============================================================================

def create_figure2(df, output_dir):
    """Figure 2: ANOVA variance decomposition across strategies."""
    strategies = df['strategy'].unique()

    perf_results = {}
    robust_results = {}

    for strategy in strategies:
        strategy_df = df[df['strategy'] == strategy]

        perf = run_anova_decomposition(strategy_df)
        if perf:
            perf_results[strategy] = perf

        robust = run_robustness_anova(strategy_df)
        if robust:
            robust_results[strategy] = robust

    if not perf_results and not robust_results:
        print("⚠ Could not create Figure 2 - insufficient data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Performance ANOVA
    ax_a = axes[0]
    strats = list(perf_results.keys())
    x = np.arange(len(strats))
    width = 0.25

    if strats:
        model_vals = [perf_results[s]['eta2_model'] for s in strats]
        rep_vals = [perf_results[s]['eta2_rep'] for s in strats]
        int_vals = [perf_results[s]['eta2_interaction'] for s in strats]

        ax_a.bar(x - width, model_vals, width, label='Model', color='#3498db')
        ax_a.bar(x, rep_vals, width, label='Representation', color='#e74c3c')
        ax_a.bar(x + width, int_vals, width, label='Interaction', color='#2ecc71')

        ax_a.axhline(72, color='#e74c3c', linestyle='--', alpha=0.5)
        ax_a.text(len(strats)-0.5, 74, 'Rep ~72%', fontsize=7, color='#e74c3c')

        ax_a.set_ylabel('Variance Explained (η², %)')
        ax_a.set_title('A. Performance (R² at σ=0.3)', fontweight='bold')
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strats], rotation=45, ha='right')
        ax_a.legend()
        ax_a.set_ylim(0, 100)

    # Panel B: Robustness ANOVA
    ax_b = axes[1]
    strats_r = list(robust_results.keys())
    x_r = np.arange(len(strats_r))

    if strats_r:
        model_vals = [robust_results[s]['eta2_model'] for s in strats_r]
        rep_vals = [robust_results[s]['eta2_rep'] for s in strats_r]
        int_vals = [robust_results[s]['eta2_interaction'] for s in strats_r]

        ax_b.bar(x_r - width, model_vals, width, label='Model', color='#3498db')
        ax_b.bar(x_r, rep_vals, width, label='Representation', color='#e74c3c')
        ax_b.bar(x_r + width, int_vals, width, label='Interaction', color='#2ecc71')

        ax_b.axhline(67, color='#3498db', linestyle='--', alpha=0.5)
        ax_b.text(len(strats_r)-0.5, 69, 'Model ~67%', fontsize=7, color='#3498db')

        ax_b.set_ylabel('Variance Explained (η², %)')
        ax_b.set_title('B. Robustness (NDS)', fontweight='bold')
        ax_b.set_xticks(x_r)
        ax_b.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strats_r], rotation=45, ha='right')
        ax_b.legend()
        ax_b.set_ylim(0, 100)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig2_anova_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig2_anova_decomposition.png")

    # Save ANOVA table
    rows = []
    for s in set(list(perf_results.keys()) + list(robust_results.keys())):
        row = {'Strategy': STRATEGY_LABELS.get(s, s)}
        if s in perf_results:
            row['Perf_Model_η²'] = perf_results[s]['eta2_model']
            row['Perf_Rep_η²'] = perf_results[s]['eta2_rep']
        if s in robust_results:
            row['Robust_Model_η²'] = robust_results[s]['eta2_model']
            row['Robust_Rep_η²'] = robust_results[s]['eta2_rep']
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_dir / 'table1_anova_summary.csv', index=False)
    print("✓ Saved table1_anova_summary.csv")


# =============================================================================
# FIGURE 3: RANKING CONSISTENCY
# =============================================================================

def create_figure3(nds_df, validation_df, output_dir):
    """Figure 3: Ranking consistency across strategies, sigmas, datasets. Uses PDV only."""
    n_panels = 3 if validation_df is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 5))
    if n_panels == 2:
        axes = [axes[0], axes[1], None]

    # Filter to PDV only for consistent comparison (don't mix representations)
    nds_pdv = nds_df[nds_df['rep'] == 'pdv'] if 'rep' in nds_df.columns else nds_df

    # Panel A: Bump chart - ranks across strategies (PDV only)
    ax_a = axes[0]

    if len(nds_pdv) > 0:
        pivot = nds_pdv.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        if len(pivot) > 0:
            rankings = pivot.rank(ascending=False)  # Higher NDS = rank 1

            strategies = [c for c in ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
                          if c in rankings.columns]

            for model in rankings.index:
                ranks = [rankings.loc[model, s] for s in strategies if s in rankings.columns]
                color = MODEL_COLORS.get(model, '#333333')
                ax_a.plot(range(len(strategies)), ranks, 'o-', label=get_model_label(model), color=color, markersize=4)

            ax_a.set_xticks(range(len(strategies)))
            ax_a.set_xticklabels([STRATEGY_LABELS.get(s, s) for s in strategies], rotation=45, ha='right')
            ax_a.set_ylabel('Rank (1 = most robust)')
            ax_a.set_title('A. Rankings Across Strategies (PDV)', fontweight='bold')
            ax_a.invert_yaxis()
            ax_a.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=6)

    # Panel B: Scatter - Baseline R² vs NDS (PDV only, legacy strategy)
    ax_b = axes[1]

    nds_pdv_legacy = nds_pdv[nds_pdv['strategy'] == 'legacy'] if 'strategy' in nds_pdv.columns else nds_pdv

    for model in nds_pdv_legacy['model'].unique():
        model_data = nds_pdv_legacy[nds_pdv_legacy['model'] == model]
        color = MODEL_COLORS.get(model, '#333333')
        ax_b.scatter(model_data['baseline_r2'], model_data['nds'],
                     label=get_model_label(model), color=color, alpha=0.7, s=50)

    ax_b.set_xlabel('Baseline R² (σ=0)')
    ax_b.set_ylabel('NDS (slope)')
    ax_b.set_title('B. Baseline vs Robustness (PDV, Legacy)', fontweight='bold')
    ax_b.axhline(0, color='black', linewidth=0.5)
    ax_b.legend(loc='lower right', fontsize=6, ncol=2)

    # Panel C: Cross-dataset rankings (if validation data available)
    # Filter to PDV for consistency
    if validation_df is not None and axes[2] is not None:
        ax_c = axes[2]

        val_pdv = validation_df[validation_df['rep'] == 'pdv'] if 'rep' in validation_df.columns else validation_df
        val_pdv_legacy = val_pdv[val_pdv['strategy'] == 'legacy'] if 'strategy' in val_pdv.columns else val_pdv

        datasets = val_pdv_legacy['dataset'].unique() if len(val_pdv_legacy) > 0 else []

        for model in val_pdv_legacy['model'].unique():
            model_data = val_pdv_legacy[val_pdv_legacy['model'] == model]

            nds_vals = []
            for ds in datasets:
                ds_data = model_data[model_data['dataset'] == ds]
                if len(ds_data) > 0 and 'NDS_r2' in ds_data.columns:
                    nds_vals.append(ds_data['NDS_r2'].values[0])  # Single value per model/dataset
                else:
                    nds_vals.append(np.nan)

            if not all(np.isnan(nds_vals)):
                color = MODEL_COLORS.get(model.lower(), '#333333')
                ax_c.plot(range(len(datasets)), nds_vals, 'o-', label=get_model_label(model),
                          color=color, markersize=6)

        ax_c.set_xticks(range(len(datasets)))
        ax_c.set_xticklabels(datasets, rotation=45, ha='right')
        ax_c.set_ylabel('NDS')
        ax_c.set_title('C. Cross-Dataset (PDV, Legacy)', fontweight='bold')
        ax_c.axhline(0, color='black', linewidth=0.5)
        ax_c.legend(loc='lower right', fontsize=6)

    for ax in axes:
        if ax is not None:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_ranking_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig3_ranking_consistency.png")


# =============================================================================
# FIGURE 4: DNN FAMILY COMPARISON
# =============================================================================

def create_figure4(df, nds_df, output_dir):
    """
    Figure 4: DNN vs BNN variants comparison.

    REVIEW AFTER RESULTS:
    - Does BNN consistently beat DNN across strategies?
    - Is the improvement larger under certain noise types?
    - Check if CONTRAST_STRATEGY shows different pattern than PRIMARY_STRATEGY
    """
    dnn_variants = ['dnn', 'dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_variational']

    # 2x2 layout: top row = PRIMARY_STRATEGY, bottom row = CONTRAST_STRATEGY
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for row, strategy in enumerate([PRIMARY_STRATEGY, CONTRAST_STRATEGY]):
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)

        # Panel A/C: R² vs σ
        ax_line = axes[row, 0]
        data = df[(df['rep'] == PRIMARY_REP) & (df['strategy'] == strategy)]

        for model in dnn_variants:
            model_data = data[data['model'] == model]
            if len(model_data) == 0:
                continue

            avg = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#333333')
            ax_line.plot(avg['sigma'], avg['r2'], 'o-', label=get_model_label(model), color=color, markersize=4)

        ax_line.set_xlabel('Noise Level (σ)')
        ax_line.set_ylabel('R²')
        panel_letter = 'A' if row == 0 else 'C'
        ax_line.set_title(f'{panel_letter}. R² vs σ ({strategy_label})', fontweight='bold')
        ax_line.legend(loc='lower left', fontsize=7)
        ax_line.set_ylim(-0.1, 1.0)

        # Panel B/D: NDS comparison
        ax_bar = axes[row, 1]
        # REVIEW: Currently filtering to PRIMARY_REP + this strategy
        dnn_nds = nds_df[(nds_df['model'].isin(dnn_variants)) &
                         (nds_df['rep'] == PRIMARY_REP) &
                         (nds_df['strategy'] == strategy)]

        if len(dnn_nds) > 0:
            mean_nds = dnn_nds.groupby('model')['nds'].mean().reindex(dnn_variants)

            colors = [MODEL_COLORS.get(m, '#333333') for m in dnn_variants]
            x = range(len(dnn_variants))

            ax_bar.bar(x, mean_nds.values, color=colors)
            ax_bar.set_xticks(x)
            ax_bar.set_xticklabels([get_model_label(m) for m in dnn_variants], rotation=45, ha='right')
            ax_bar.set_ylabel('NDS (higher = more robust)')
            panel_letter = 'B' if row == 0 else 'D'
            ax_bar.set_title(f'{panel_letter}. NDS ({strategy_label})', fontweight='bold')
            ax_bar.axhline(0, color='black', linewidth=0.5)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_dnn_family.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fig4_dnn_family.png (showing {PRIMARY_STRATEGY} vs {CONTRAST_STRATEGY})")

    # REVIEW: Check if the pattern holds - does BNN > DNN in both strategies?


# =============================================================================
# FIGURE 5: MLP + RF/QRF COMPARISON
# =============================================================================

def create_figure5(df, nds_df, output_dir):
    """
    Figure 5: MLP variants + RF vs QRF comparison.

    REVIEW AFTER RESULTS:
    - Does MLP-BNN consistently beat MLP?
    - Does QRF beat RF?
    - Is the pattern consistent across strategies?
    """
    mlp_variants = ['mlp', 'mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational']
    rf_models = ['rf', 'qrf']

    # 2x3 layout: compare PRIMARY_STRATEGY vs CONTRAST_STRATEGY
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, strategy in enumerate([PRIMARY_STRATEGY, CONTRAST_STRATEGY]):
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)
        data = df[(df['rep'] == PRIMARY_REP) & (df['strategy'] == strategy)]

        # Panel A/D: MLP R² vs σ
        ax_line = axes[row, 0]

        for model in mlp_variants:
            model_data = data[data['model'] == model]
            if len(model_data) == 0:
                continue

            avg = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#333333')
            ax_line.plot(avg['sigma'], avg['r2'], 'o-', label=get_model_label(model), color=color, markersize=4)

        ax_line.set_xlabel('Noise Level (σ)')
        ax_line.set_ylabel('R²')
        panel_letter = 'A' if row == 0 else 'D'
        ax_line.set_title(f'{panel_letter}. MLP R² vs σ ({strategy_label})', fontweight='bold')
        ax_line.legend(loc='lower left', fontsize=6)
        ax_line.set_ylim(-0.1, 1.0)

        # Panel B/E: MLP NDS
        ax_mlp = axes[row, 1]
        mlp_nds = nds_df[(nds_df['model'].isin(mlp_variants)) &
                         (nds_df['rep'] == PRIMARY_REP) &
                         (nds_df['strategy'] == strategy)]

        if len(mlp_nds) > 0:
            mean_nds = mlp_nds.groupby('model')['nds'].mean().reindex(mlp_variants)
            colors = [MODEL_COLORS.get(m, '#333333') for m in mlp_variants]
            x = range(len(mlp_variants))

            ax_mlp.bar(x, mean_nds.values, color=colors)
            ax_mlp.set_xticks(x)
            ax_mlp.set_xticklabels([get_model_label(m) for m in mlp_variants], rotation=45, ha='right')
            ax_mlp.set_ylabel('NDS')
            panel_letter = 'B' if row == 0 else 'E'
            ax_mlp.set_title(f'{panel_letter}. MLP NDS ({strategy_label})', fontweight='bold')
            ax_mlp.axhline(0, color='black', linewidth=0.5)

        # Panel C/F: RF vs QRF
        ax_rf = axes[row, 2]
        rf_nds = nds_df[(nds_df['model'].isin(rf_models)) &
                        (nds_df['rep'] == PRIMARY_REP) &
                        (nds_df['strategy'] == strategy)]

        if len(rf_nds) > 0:
            mean_nds = rf_nds.groupby('model')['nds'].mean().reindex(rf_models)
            colors = [MODEL_COLORS.get(m, '#333333') for m in rf_models]

            ax_rf.bar(range(len(rf_models)), mean_nds.values, color=colors)
            ax_rf.set_xticks(range(len(rf_models)))
            ax_rf.set_xticklabels(['RF', 'QRF'])
            ax_rf.set_ylabel('NDS')
            panel_letter = 'C' if row == 0 else 'F'
            ax_rf.set_title(f'{panel_letter}. RF vs QRF ({strategy_label})', fontweight='bold')
            ax_rf.axhline(0, color='black', linewidth=0.5)

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_mlp_rf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fig5_mlp_rf_comparison.png (showing {PRIMARY_STRATEGY} vs {CONTRAST_STRATEGY})")

    # REVIEW: Does the probabilistic advantage hold under both noise types?


# =============================================================================
# FIGURE 6: UNCERTAINTY QUALITY
# =============================================================================

def _create_uncertainty_quality_figure(unc_df, output_path, strategy, rep, title_suffix=""):
    """
    Helper to create uncertainty quality figure for a given strategy/rep.
    Called by create_figure6 for main and supplementary versions.
    """
    if unc_df is None or len(unc_df) == 0:
        return False

    # Filter to specified strategy and rep
    filtered = unc_df.copy()
    if 'strategy' in filtered.columns and strategy:
        filtered = filtered[filtered['strategy'] == strategy]
    if 'rep' in filtered.columns and rep:
        filtered = filtered[filtered['rep'] == rep]

    if len(filtered) == 0:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Find uncertainty column
    unc_col = None
    for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty', 'std']:
        if col in filtered.columns:
            unc_col = col
            break

    if unc_col is None:
        plt.close()
        return False

    # Panel A: Calibration plot
    ax_a = axes[0]

    for model in filtered['model'].unique():
        model_data = filtered[filtered['model'] == model]

        pred = model_data[unc_col].values
        if 'y_true_original' in model_data.columns:
            actual = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
        elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
            actual = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
        else:
            continue

        mask = np.isfinite(pred) & np.isfinite(actual) & (pred > 0)
        pred_m, actual_m = pred[mask], actual[mask]

        if len(pred_m) < 100:
            continue

        # Bin and plot
        bins = np.percentile(pred_m, np.linspace(0, 100, 11))
        bin_centers, bin_errors = [], []

        for i in range(len(bins) - 1):
            bin_mask = (pred_m >= bins[i]) & (pred_m < bins[i + 1])
            if bin_mask.sum() > 0:
                bin_centers.append(pred_m[bin_mask].mean())
                bin_errors.append(actual_m[bin_mask].mean())

        color = MODEL_COLORS.get(model, '#333333')
        ax_a.plot(bin_centers, bin_errors, 'o-', label=get_model_label(model), color=color, markersize=4)

    # Diagonal
    lims = [0, ax_a.get_xlim()[1]] if ax_a.get_xlim()[1] > 0 else [0, 1]
    ax_a.plot(lims, lims, 'k--', alpha=0.5, label='Perfect')
    ax_a.set_xlabel('Predicted Uncertainty')
    ax_a.set_ylabel('Actual Error')
    ax_a.set_title(f'A. Calibration{title_suffix}', fontweight='bold')
    ax_a.legend(fontsize=6)

    # Panel B: ECE comparison
    ax_b = axes[1]

    ece_data = []
    for model in filtered['model'].unique():
        model_data = filtered[filtered['model'] == model]

        pred = model_data[unc_col].values
        if 'y_true_original' in model_data.columns:
            actual = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
        elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
            actual = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
        else:
            continue

        mask = np.isfinite(pred) & np.isfinite(actual) & (pred > 0)
        pred_m, actual_m = pred[mask], actual[mask]

        if len(pred_m) < 100:
            continue

        # Calculate ECE
        bins = np.percentile(pred_m, np.linspace(0, 100, 11))
        bins = np.unique(bins)
        ece = 0
        for i in range(len(bins) - 1):
            bin_mask = (pred_m >= bins[i]) & (pred_m < bins[i + 1])
            if bin_mask.sum() > 0:
                bin_pred = pred_m[bin_mask].mean()
                bin_actual = actual_m[bin_mask].mean()
                bin_weight = bin_mask.sum() / len(pred_m)
                ece += bin_weight * np.abs(bin_pred - bin_actual)

        ece_data.append({'model': model, 'ece': ece})

    if ece_data:
        ece_df = pd.DataFrame(ece_data).sort_values('ece')
        colors = [MODEL_COLORS.get(m, '#333333') for m in ece_df['model']]
        labels = [get_model_label(m) for m in ece_df['model']]
        ax_b.barh(labels, ece_df['ece'], color=colors)
        ax_b.set_xlabel('ECE (lower = better)')
        ax_b.set_title(f'B. Expected Calibration Error{title_suffix}', fontweight='bold')

    # Panel C: Uncertainty-error correlation
    ax_c = axes[2]

    corr_data = []
    for model in filtered['model'].unique():
        model_data = filtered[filtered['model'] == model]

        pred = model_data[unc_col].values
        if 'y_true_original' in model_data.columns:
            actual = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
        elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
            actual = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
        else:
            continue

        mask = np.isfinite(pred) & np.isfinite(actual)
        if mask.sum() < 100:
            continue

        corr, _ = stats.spearmanr(pred[mask], actual[mask])
        corr_data.append({'model': model, 'corr': corr})

    if corr_data:
        corr_df = pd.DataFrame(corr_data).sort_values('corr', ascending=False)
        colors = [MODEL_COLORS.get(m, '#333333') for m in corr_df['model']]
        labels = [get_model_label(m) for m in corr_df['model']]
        ax_c.barh(labels, corr_df['corr'], color=colors)
        ax_c.set_xlabel('ρ (uncertainty-error)')
        ax_c.set_title(f'C. Uncertainty-Error Correlation{title_suffix}', fontweight='bold')
        ax_c.axvline(0, color='black', linewidth=0.5)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def create_figure6(unc_df, output_dir):
    """
    Figure 6: Uncertainty quality metrics (calibration, ECE).

    REVIEW AFTER RESULTS:
    - Are uncertainty estimates well-calibrated?
    - Does calibration quality differ across strategies?
    - Check if SUPPLEMENTARY_REP shows same pattern as PRIMARY_REP
    """
    if unc_df is None or len(unc_df) == 0:
        print("⚠ Skipping Figure 6 - no uncertainty data")
        return

    strategy_label = STRATEGY_LABELS.get(PRIMARY_STRATEGY, PRIMARY_STRATEGY)

    # Main figure: PRIMARY_REP + PRIMARY_STRATEGY
    success = _create_uncertainty_quality_figure(
        unc_df,
        output_dir / 'fig6_uncertainty_quality.png',
        strategy=PRIMARY_STRATEGY,
        rep=PRIMARY_REP,
        title_suffix=f" ({PRIMARY_REP.upper()}, {strategy_label})"
    )
    if success:
        print(f"✓ Saved fig6_uncertainty_quality.png ({PRIMARY_REP}, {PRIMARY_STRATEGY})")
    else:
        print("⚠ Could not create Figure 6 main")

    # Supplementary: SUPPLEMENTARY_REP + PRIMARY_STRATEGY
    if GENERATE_SUPPLEMENTARY:
        success = _create_uncertainty_quality_figure(
            unc_df,
            output_dir / f'fig6_supp_{SUPPLEMENTARY_REP}_{PRIMARY_STRATEGY}.png',
            strategy=PRIMARY_STRATEGY,
            rep=SUPPLEMENTARY_REP,
            title_suffix=f" ({SUPPLEMENTARY_REP.upper()}, {strategy_label})"
        )
        if success:
            print(f"✓ Saved fig6_supp_{SUPPLEMENTARY_REP}_{PRIMARY_STRATEGY}.png (supplementary)")

        # Supplementary: PRIMARY_REP + CONTRAST_STRATEGY
        # REVIEW: Does calibration hold under different noise?
        contrast_label = STRATEGY_LABELS.get(CONTRAST_STRATEGY, CONTRAST_STRATEGY)
        success = _create_uncertainty_quality_figure(
            unc_df,
            output_dir / f'fig6_supp_{PRIMARY_REP}_{CONTRAST_STRATEGY}.png',
            strategy=CONTRAST_STRATEGY,
            rep=PRIMARY_REP,
            title_suffix=f" ({PRIMARY_REP.upper()}, {contrast_label})"
        )
        if success:
            print(f"✓ Saved fig6_supp_{PRIMARY_REP}_{CONTRAST_STRATEGY}.png (supplementary)")


# =============================================================================
# FIGURE 7: UNCERTAINTY TRACKS NOISE (KEY FIGURE)
# =============================================================================

def _create_uncertainty_noise_figure(unc_df, output_path, strategy, rep, title_suffix=""):
    """
    Helper to create uncertainty-noise tracking figure for a given strategy/rep.

    REVIEW AFTER RESULTS:
    - Panel C is the KEY panel: Does uncertainty correlate with injected noise?
    - Panel D: Is it aleatoric or epistemic that tracks noise?
    - Hypothesis: Robust models attribute noise to aleatoric uncertainty
    """
    if unc_df is None or len(unc_df) == 0:
        return False

    # Filter to specified strategy and rep
    filtered = unc_df.copy()
    if 'strategy' in filtered.columns and strategy:
        filtered = filtered[filtered['strategy'] == strategy]
    if 'rep' in filtered.columns and rep:
        filtered = filtered[filtered['rep'] == rep]

    if len(filtered) == 0:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Find columns
    unc_col = None
    for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty']:
        if col in filtered.columns:
            unc_col = col
            break

    if unc_col is None or 'sigma' not in filtered.columns:
        plt.close()
        return False

    # Panel A: Mean uncertainty vs sigma level
    ax_a = axes[0, 0]

    for model in filtered['model'].unique():
        model_data = filtered[filtered['model'] == model]

        sigma_means = []
        for sigma in sorted(model_data['sigma'].unique()):
            sigma_data = model_data[model_data['sigma'] == sigma]
            unc_values = sigma_data[unc_col].values
            unc_values = unc_values[np.isfinite(unc_values)]
            if len(unc_values) > 0:
                sigma_means.append({'sigma': sigma, 'mean_unc': unc_values.mean()})

        if sigma_means:
            sigma_df = pd.DataFrame(sigma_means)
            color = MODEL_COLORS.get(model, '#333333')
            ax_a.plot(sigma_df['sigma'], sigma_df['mean_unc'], 'o-',
                      label=get_model_label(model), color=color, markersize=4)

    ax_a.set_xlabel('Injected Noise Level (σ)')
    ax_a.set_ylabel('Mean Predicted Uncertainty')
    ax_a.set_title(f'A. Uncertainty vs Noise Level{title_suffix}', fontweight='bold')
    ax_a.legend(fontsize=6, ncol=2)

    # Panel B: Uncertainty-error correlation bar
    ax_b = axes[0, 1]

    corr_data = []
    for model in filtered['model'].unique():
        model_data = filtered[filtered['model'] == model]

        unc_values = model_data[unc_col].values
        if 'y_true_original' in model_data.columns:
            errors = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
        elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
            errors = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
        else:
            continue

        mask = np.isfinite(unc_values) & np.isfinite(errors)
        if mask.sum() < 100:
            continue

        corr, _ = stats.spearmanr(unc_values[mask], errors[mask])
        corr_data.append({'model': model, 'corr': corr})

    if corr_data:
        corr_df = pd.DataFrame(corr_data).sort_values('corr', ascending=False)
        colors = [MODEL_COLORS.get(m, '#333333') for m in corr_df['model']]
        labels = [get_model_label(m) for m in corr_df['model']]
        ax_b.barh(labels, corr_df['corr'], color=colors)
        ax_b.set_xlabel('ρ (uncertainty ↔ error)')
        ax_b.set_title(f'B. Uncertainty-Error Correlation{title_suffix}', fontweight='bold')
        ax_b.axvline(0, color='black', linewidth=0.5)

    # Panel C: Uncertainty-noise correlation (KEY)
    ax_c = axes[1, 0]

    if 'injected_noise' in filtered.columns:
        noise_corr_data = []
        for model in filtered['model'].unique():
            model_data = filtered[filtered['model'] == model]

            unc_values = model_data[unc_col].values
            noise_mag = np.abs(model_data['injected_noise'].values)

            mask = np.isfinite(unc_values) & np.isfinite(noise_mag)
            if mask.sum() < 100:
                continue

            corr, _ = stats.spearmanr(unc_values[mask], noise_mag[mask])
            noise_corr_data.append({'model': model, 'corr': corr})

        if noise_corr_data:
            noise_corr_df = pd.DataFrame(noise_corr_data).sort_values('corr', ascending=False)
            colors = [MODEL_COLORS.get(m, '#333333') for m in noise_corr_df['model']]
            labels = [get_model_label(m) for m in noise_corr_df['model']]
            ax_c.barh(labels, noise_corr_df['corr'], color=colors)
            ax_c.set_xlabel('ρ (uncertainty ↔ injected noise)')
            ax_c.set_title(f'C. Uncertainty-Noise Correlation (KEY){title_suffix}', fontweight='bold')
            ax_c.axvline(0, color='black', linewidth=0.5)
    else:
        ax_c.text(0.5, 0.5, 'No injected_noise data', ha='center', va='center', transform=ax_c.transAxes)
        ax_c.set_title(f'C. Uncertainty-Noise Correlation{title_suffix}', fontweight='bold')

    # Panel D: Aleatoric vs Epistemic
    ax_d = axes[1, 1]

    if 'aleatoric_uncertainty' in filtered.columns and 'epistemic_uncertainty' in filtered.columns:
        alea_corr, epis_corr = [], []
        models = []

        for model in filtered['model'].unique():
            model_data = filtered[filtered['model'] == model]

            if 'injected_noise' not in model_data.columns:
                continue

            noise_mag = np.abs(model_data['injected_noise'].values)
            alea = model_data['aleatoric_uncertainty'].values
            epis = model_data['epistemic_uncertainty'].values

            mask = np.isfinite(alea) & np.isfinite(epis) & np.isfinite(noise_mag)
            if mask.sum() < 100:
                continue

            a_corr, _ = stats.spearmanr(alea[mask], noise_mag[mask])
            e_corr, _ = stats.spearmanr(epis[mask], noise_mag[mask])

            models.append(model)
            alea_corr.append(a_corr)
            epis_corr.append(e_corr)

        if models:
            x = np.arange(len(models))
            width = 0.35

            ax_d.bar(x - width/2, alea_corr, width, label='Aleatoric', color='#E69F00')  # Orange (colorblind-safe)
            ax_d.bar(x + width/2, epis_corr, width, label='Epistemic', color='#0072B2')  # Blue (colorblind-safe)
            ax_d.set_xticks(x)
            ax_d.set_xticklabels([get_model_label(m) for m in models], rotation=45, ha='right')
            ax_d.set_ylabel('ρ (with injected noise)')
            ax_d.set_title(f'D. Aleatoric vs Epistemic{title_suffix}', fontweight='bold')
            ax_d.legend()
            ax_d.axhline(0, color='black', linewidth=0.5)
    else:
        ax_d.text(0.5, 0.5, 'No aleatoric/epistemic data', ha='center', va='center', transform=ax_d.transAxes)
        ax_d.set_title(f'D. Aleatoric vs Epistemic{title_suffix}', fontweight='bold')

    for ax in axes.flat:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def create_figure7(unc_df, output_dir):
    """
    Figure 7: Uncertainty tracks noise - KEY FIGURE.

    REVIEW AFTER RESULTS:
    - This is the mechanistic explanation figure
    - Panel C (uncertainty-noise correlation) is the key finding
    - Panel D shows whether it's aleatoric or epistemic that tracks noise
    - Check if pattern holds across strategies and representations
    """
    if unc_df is None or len(unc_df) == 0:
        print("⚠ Skipping Figure 7 - no uncertainty data")
        return

    strategy_label = STRATEGY_LABELS.get(PRIMARY_STRATEGY, PRIMARY_STRATEGY)

    # Main figure: PRIMARY_REP + PRIMARY_STRATEGY
    success = _create_uncertainty_noise_figure(
        unc_df,
        output_dir / 'fig7_uncertainty_noise_tracking.png',
        strategy=PRIMARY_STRATEGY,
        rep=PRIMARY_REP,
        title_suffix=""  # Main figure, no suffix needed
    )
    if success:
        print(f"✓ Saved fig7_uncertainty_noise_tracking.png ({PRIMARY_REP}, {PRIMARY_STRATEGY})")
    else:
        print("⚠ Could not create Figure 7 main")

    # Supplementary versions
    if GENERATE_SUPPLEMENTARY:
        # SUPPLEMENTARY_REP + PRIMARY_STRATEGY
        success = _create_uncertainty_noise_figure(
            unc_df,
            output_dir / f'fig7_supp_{SUPPLEMENTARY_REP}_{PRIMARY_STRATEGY}.png',
            strategy=PRIMARY_STRATEGY,
            rep=SUPPLEMENTARY_REP,
            title_suffix=f" ({SUPPLEMENTARY_REP.upper()})"
        )
        if success:
            print(f"✓ Saved fig7_supp_{SUPPLEMENTARY_REP}_{PRIMARY_STRATEGY}.png (supplementary)")

        # PRIMARY_REP + CONTRAST_STRATEGY
        # REVIEW: Does the uncertainty-noise correlation hold under different noise types?
        contrast_label = STRATEGY_LABELS.get(CONTRAST_STRATEGY, CONTRAST_STRATEGY)
        success = _create_uncertainty_noise_figure(
            unc_df,
            output_dir / f'fig7_supp_{PRIMARY_REP}_{CONTRAST_STRATEGY}.png',
            strategy=CONTRAST_STRATEGY,
            rep=PRIMARY_REP,
            title_suffix=f" ({contrast_label})"
        )
        if success:
            print(f"✓ Saved fig7_supp_{PRIMARY_REP}_{CONTRAST_STRATEGY}.png (supplementary)")


# =============================================================================
# TABLES
# =============================================================================

def create_tables(nds_df, unc_df, qm9_df, output_dir):
    """Create all summary tables."""

    # Table 2: NDS by model × strategy (PDV only - don't mix representations)
    if len(nds_df) > 0:
        nds_pdv = nds_df[nds_df['rep'] == 'pdv'] if 'rep' in nds_df.columns else nds_df

        if len(nds_pdv) > 0:
            pivot = nds_pdv.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
            pivot['MEAN'] = pivot.mean(axis=1)
            pivot = pivot.sort_values('MEAN', ascending=False)
            pivot.to_csv(output_dir / 'table2_nds_by_strategy_pdv.csv')
            print("✓ Saved table2_nds_by_strategy_pdv.csv")

        # Also save full table with all reps for supplementary
        pivot_all = nds_df.pivot_table(values='nds', index=['model', 'rep'], columns='strategy', aggfunc='mean')
        pivot_all['MEAN'] = pivot_all.mean(axis=1)
        pivot_all = pivot_all.sort_values('MEAN', ascending=False)
        pivot_all.to_csv(output_dir / 'table2_supp_nds_all_reps.csv')
        print("✓ Saved table2_supp_nds_all_reps.csv (supplementary)")

    # Table 3: Probabilistic comparison with Wilcoxon tests (PDV + legacy)
    prob_comparisons = {
        'DNN Family': {'base': 'dnn', 'variants': ['dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_variational']},
        'MLP Family': {'base': 'mlp', 'variants': ['mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational']},
        'RF Family': {'base': 'rf', 'variants': ['qrf']},
    }

    # Filter to PDV + legacy for fair comparison
    nds_fair = nds_df[(nds_df['rep'] == 'pdv') & (nds_df['strategy'] == 'legacy')] if 'rep' in nds_df.columns else nds_df

    rows = []
    wilcoxon_results = []

    for family, config in prob_comparisons.items():
        base_model = config['base']
        base_data = nds_fair[nds_fair['model'] == base_model]

        if len(base_data) > 0:
            rows.append({
                'Family': family,
                'Model': base_model,
                'Type': 'Deterministic',
                'NDS': base_data['nds'].mean(),
                'Baseline R²': base_data['baseline_r2'].mean(),
                'Wilcoxon p': np.nan,
                'Significant': ''
            })

        for variant in config['variants']:
            var_data = nds_fair[nds_fair['model'] == variant]
            if len(var_data) > 0:
                # Run Wilcoxon test comparing to base
                wilcox = wilcoxon_paired_test(nds_df, base_model, variant, rep='pdv', strategy='legacy')
                wilcoxon_results.append({
                    'Family': family,
                    'Comparison': f'{base_model} vs {variant}',
                    **wilcox
                })

                rows.append({
                    'Family': family,
                    'Model': variant,
                    'Type': 'Probabilistic',
                    'NDS': var_data['nds'].mean(),
                    'Baseline R²': var_data['baseline_r2'].mean(),
                    'Wilcoxon p': wilcox.get('p_value', np.nan),
                    'Significant': '*' if wilcox.get('significant', False) else ''
                })

    if rows:
        pd.DataFrame(rows).to_csv(output_dir / 'table3_probabilistic_comparison.csv', index=False)
        print("✓ Saved table3_probabilistic_comparison.csv")

    if wilcoxon_results:
        pd.DataFrame(wilcoxon_results).to_csv(output_dir / 'table3_wilcoxon_tests.csv', index=False)
        print("✓ Saved table3_wilcoxon_tests.csv")

    # Table 4: Uncertainty metrics (legacy strategy only)
    if unc_df is not None and len(unc_df) > 0:
        unc_legacy = unc_df[unc_df['strategy'] == 'legacy'] if 'strategy' in unc_df.columns else unc_df

        # Find uncertainty column
        unc_col = None
        for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty']:
            if col in unc_legacy.columns:
                unc_col = col
                break

        if unc_col and len(unc_legacy) > 0:
            unc_metrics = []
            for model in unc_legacy['model'].unique():
                model_data = unc_legacy[unc_legacy['model'] == model]

                unc_values = model_data[unc_col].values

                # Calculate error
                if 'y_true_original' in model_data.columns:
                    errors = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
                elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                    errors = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
                else:
                    continue

                mask = np.isfinite(unc_values) & np.isfinite(errors)
                if mask.sum() < 100:
                    continue

                # Uncertainty-error correlation
                unc_err_corr, _ = stats.spearmanr(unc_values[mask], errors[mask])

                # Uncertainty-noise correlation (if available)
                unc_noise_corr = np.nan
                if 'injected_noise' in model_data.columns:
                    noise_mag = np.abs(model_data['injected_noise'].values)
                    noise_mask = mask & np.isfinite(noise_mag)
                    if noise_mask.sum() > 100:
                        unc_noise_corr, _ = stats.spearmanr(unc_values[noise_mask], noise_mag[noise_mask])

                # Coverage at 1σ and 2σ intervals
                # Get predictions for coverage calculation
                if 'y_true_original' in model_data.columns and 'y_pred_mean' in model_data.columns:
                    y_true = model_data['y_true_original'].values[mask]
                    y_pred = model_data['y_pred_mean'].values[mask]
                elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                    y_true = model_data['y_true'].values[mask]
                    y_pred = model_data['y_pred'].values[mask]
                else:
                    y_true = y_pred = None

                if y_true is not None:
                    cov_1sigma = calculate_coverage(y_true, y_pred, unc_values[mask], k=1)
                    cov_2sigma = calculate_coverage(y_true, y_pred, unc_values[mask], k=2)
                else:
                    cov_1sigma = cov_2sigma = np.nan

                unc_metrics.append({
                    'Model': model,
                    'Unc-Error ρ': unc_err_corr,
                    'Unc-Noise ρ': unc_noise_corr,
                    'Coverage 1σ': cov_1sigma,
                    'Coverage 2σ': cov_2sigma,
                    'Mean Uncertainty': unc_values[mask].mean(),
                })

            if unc_metrics:
                unc_metrics_df = pd.DataFrame(unc_metrics).sort_values('Unc-Noise ρ', ascending=False)
                unc_metrics_df.to_csv(output_dir / 'table4_uncertainty_metrics.csv', index=False)
                print("✓ Saved table4_uncertainty_metrics.csv")

    # Table 5: Model rankings at specific sigma levels (shows ranking stability)
    # Uses PDV + legacy only
    if qm9_df is not None and len(qm9_df) > 0:
        # Filter to PDV + legacy
        pdv_legacy = qm9_df[(qm9_df['strategy'] == 'legacy') & (qm9_df['rep'] == 'pdv')] if 'strategy' in qm9_df.columns else qm9_df

        sigma_levels = [0.0, 0.3, 0.5, 0.7, 1.0]
        rank_data = {}
        valid_models = []

        for sigma in sigma_levels:
            sigma_df = pdv_legacy[np.abs(pdv_legacy['sigma'] - sigma) < 0.05]
            if len(sigma_df) == 0:
                continue

            # Average R² per model at this sigma
            model_r2 = sigma_df.groupby('model')['r2'].mean()

            # Filter to baseline > threshold at sigma=0
            if sigma == 0.0:
                valid_models = model_r2[model_r2 > BASELINE_THRESHOLD].index.tolist()

            # Rank models (1 = best)
            model_ranks = model_r2.rank(ascending=False)
            rank_data[f'σ={sigma}'] = model_ranks

        if rank_data:
            rank_df = pd.DataFrame(rank_data)
            # Filter to models that pass baseline
            if 'valid_models' in dir():
                rank_df = rank_df[rank_df.index.isin(valid_models)]

            # Add mean rank column
            rank_df['Mean Rank'] = rank_df.mean(axis=1)
            rank_df = rank_df.sort_values('Mean Rank')

            rank_df.to_csv(output_dir / 'table5_sigma_rankings.csv')
            print("✓ Saved table5_sigma_rankings.csv")

    # Table 6: Kendall's W for ranking consistency across strategies
    if len(nds_df) > 0:
        # Get rankings per strategy
        strategies = nds_df['strategy'].unique()
        if len(strategies) > 1:
            rank_matrix = []
            models = nds_df['model'].unique()

            for strategy in strategies:
                strat_data = nds_df[nds_df['strategy'] == strategy]
                model_nds = strat_data.groupby('model')['nds'].mean()
                # Higher NDS = more robust = rank 1
                ranks = model_nds.rank(ascending=False)
                rank_matrix.append([ranks.get(m, np.nan) for m in models])

            rank_matrix = np.array(rank_matrix)

            # Remove models with any NaN
            valid_cols = ~np.any(np.isnan(rank_matrix), axis=0)
            rank_matrix = rank_matrix[:, valid_cols]
            valid_models = [m for m, v in zip(models, valid_cols) if v]

            if rank_matrix.shape[1] > 2:
                # Calculate Kendall's W
                n_raters = rank_matrix.shape[0]  # strategies
                n_items = rank_matrix.shape[1]   # models

                rank_sums = rank_matrix.sum(axis=0)
                mean_rank_sum = rank_sums.mean()
                ss_between = np.sum((rank_sums - mean_rank_sum) ** 2)

                max_ss = (n_raters ** 2) * (n_items ** 3 - n_items) / 12
                kendalls_w = ss_between / max_ss if max_ss > 0 else 0

                # Save summary
                with open(output_dir / 'table6_kendalls_w.txt', 'w') as f:
                    f.write("Kendall's W Concordance Analysis\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Number of raters (strategies): {n_raters}\n")
                    f.write(f"Number of items (models): {n_items}\n")
                    f.write(f"Kendall's W: {kendalls_w:.4f}\n\n")
                    f.write("Interpretation:\n")
                    f.write("  W > 0.7: Strong agreement\n")
                    f.write("  W 0.5-0.7: Moderate agreement\n")
                    f.write("  W < 0.5: Weak agreement\n")
                print("✓ Saved table6_kendalls_w.txt")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(nds_df, excluded_df, output_dir):
    """Generate text report summarizing findings."""
    lines = []
    lines.append("=" * 80)
    lines.append("PAPER FIGURES GENERATION REPORT")
    lines.append("=" * 80)

    lines.append(f"\nBaseline R² threshold: {BASELINE_THRESHOLD}")
    lines.append(f"NDS calculated for {len(nds_df)} configurations")
    lines.append(f"Excluded {len(excluded_df)} configurations (baseline R² < {BASELINE_THRESHOLD})")

    if len(excluded_df) > 0 and 'marginal' in excluded_df.columns:
        marginal = excluded_df[excluded_df['marginal'] == True]
        clearly_excluded = excluded_df[excluded_df['marginal'] == False]
        lines.append(f"\n  Marginal exclusions (0.5 <= R² < {BASELINE_THRESHOLD}): {len(marginal)}")
        lines.append(f"  Clearly excluded (R² < 0.5): {len(clearly_excluded)}")

        if len(marginal) > 0:
            lines.append(f"\n  --- MARGINAL EXCLUSIONS (would pass at 0.5 threshold) ---")
            marginal_sorted = marginal.sort_values('baseline', ascending=False)
            for _, row in marginal_sorted.iterrows():
                lines.append(f"    {row['model']:25s} {row['rep']:20s} {row['strategy']:12s}  "
                             f"R²={row['baseline']:.3f}  NDS={row['nds']:.4f}")

    # Save excluded configs CSV for reference
    if len(excluded_df) > 0:
        excluded_df.to_csv(output_dir / 'excluded_configs.csv', index=False)
        lines.append(f"\n  Full exclusion list saved to excluded_configs.csv")

    if len(nds_df) > 0:
        lines.append("\n" + "=" * 80)
        lines.append("KEY FINDINGS (PDV representation, Legacy strategy)")
        lines.append("=" * 80)

        # Filter to PDV + legacy for consistent reporting
        nds_pdv_legacy = nds_df[(nds_df['rep'] == 'pdv') & (nds_df['strategy'] == 'legacy')]
        if len(nds_pdv_legacy) == 0:
            nds_pdv_legacy = nds_df  # Fallback

        # Most robust model
        mean_nds = nds_pdv_legacy.groupby('model')['nds'].mean().sort_values(ascending=False)
        lines.append(f"\nMost robust model: {mean_nds.index[0]} (NDS = {mean_nds.iloc[0]:.4f})")
        lines.append(f"Least robust model: {mean_nds.index[-1]} (NDS = {mean_nds.iloc[-1]:.4f})")

        # BNN vs DNN comparison
        dnn_data = nds_pdv_legacy[nds_pdv_legacy['model'] == 'dnn']
        bnn_data = nds_pdv_legacy[nds_pdv_legacy['model'] == 'dnn_bnn_full']

        if len(dnn_data) > 0 and len(bnn_data) > 0:
            dnn_nds = dnn_data['nds'].values[0]
            bnn_nds = bnn_data['nds'].values[0]
            lines.append(f"\nDNN NDS: {dnn_nds:.4f}")
            lines.append(f"DNN-BNN-full NDS: {bnn_nds:.4f}")
            lines.append(f"Improvement: {(bnn_nds - dnn_nds):.4f}")

    report_path = output_dir / 'paper_figures_report.txt'
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"✓ Saved {report_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate all paper figures')
    parser.add_argument('--qm9-dir', type=str, default='../results',
                        help='Directory containing QM9 results')
    parser.add_argument('--validation-dir', type=str, default=None,
                        help='Directory containing validation results from KIRBy')
    parser.add_argument('--output-dir', type=str, default='../results/paper_figures',
                        help='Output directory for figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GENERATING PAPER FIGURES")
    print("=" * 80)

    # Load data
    print("\n[1/3] Loading data...")
    qm9_df = load_anova_data(args.qm9_dir)
    unc_df = load_uncertainty_data(args.qm9_dir)
    validation_df = load_validation_data(args.validation_dir)

    if qm9_df is None:
        print("ERROR: No QM9 ANOVA data found!")
        return

    # Calculate NDS
    print("\n[2/3] Calculating metrics...")
    nds_df, excluded_df = calculate_nds(qm9_df)
    print(f"  NDS calculated: {len(nds_df)} configs")
    print(f"  Excluded (baseline R² < {BASELINE_THRESHOLD}): {len(excluded_df)} configs")
    if len(excluded_df) > 0 and 'marginal' in excluded_df.columns:
        n_marginal = excluded_df['marginal'].sum()
        print(f"    Marginal (0.5 <= R² < {BASELINE_THRESHOLD}): {n_marginal}")
        print(f"    Clearly excluded (R² < 0.5): {len(excluded_df) - n_marginal}")

    # Generate figures
    print("\n[3/3] Generating figures...")

    print("\n--- METHODS FIGURE ---")
    create_methods_figure(output_dir)

    print("\n--- PART 1: THE WHAT ---")
    create_figure1(qm9_df, nds_df, output_dir)
    create_figure2(qm9_df, output_dir)
    create_figure3(nds_df, validation_df, output_dir)

    print("\n--- PART 2: THE WHY ---")
    create_figure4(qm9_df, nds_df, output_dir)
    create_figure5(qm9_df, nds_df, output_dir)
    create_figure6(unc_df, output_dir)
    create_figure7(unc_df, output_dir)

    print("\n--- TABLES ---")
    create_tables(nds_df, unc_df, qm9_df, output_dir)

    print("\n--- REPORT ---")
    generate_report(nds_df, excluded_df, output_dir)

    print("\n" + "=" * 80)
    print(f"COMPLETE - All outputs in {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
