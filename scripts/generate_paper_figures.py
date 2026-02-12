#!/usr/bin/env python3
"""
Master Figure Generation Script for Noise Robustness Paper

Produces all main figures + tables for the paper structure:

PART 1: THE WHAT
  Figure 1: Global Overview (R² vs σ line plot, NDS heatmap)
  Figure 2: ANOVA Decomposition (η² bars by strategy)
  Figure 3: Ranking Consistency (heatmaps, cross-dataset)

PART 2: THE WHY
  Figure 4: DNN Family Comparison (DNN vs BNN variants) — CSV table + R² vs σ plot
  Figure 5: MLP Family + RF/QRF Comparison — CSV table + R² vs σ plot
  Figure: Uncertainty Tracks Noise (single-panel: mean uncertainty vs σ)

Calibration (ECE, coverage), unc-error/unc-noise correlations, and
aleatoric/epistemic decomposition are reported in table4 CSVs.

Run: python generate_paper_figures.py [--qm9-dir ../results] [--validation-dir /path/to/kirby/results]

Outputs:
  results/paper_figures/
    fig1_global_overview.png
    fig2_anova_decomposition.png
    fig3_ranking_consistency.png
    fig4_dnn_family.png / .csv
    fig5_mlp_rf_comparison.png / .csv
    fig_uncertainty_combined.png
    table1_anova_summary.csv
    table1_supp_simple_effects.csv
    table1_supp_simple_effects_all_reps.csv
    table2_nds_by_strategy.csv
    table3_probabilistic_comparison.csv
    table4_uncertainty_metrics.csv (ECE, coverage, correlations, aleatoric/epistemic)
    table4_supp_uncertainty_by_strategy_rep.csv (same metrics × all strategies/reps)
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
SUPPLEMENTARY_REP = 'ecfp4'      # Changed from SNS: ECFP4 is more robust and representative of QSAR practice

# All strategies for completeness checks
ALL_STRATEGIES = ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']

# Flag to generate supplementary figures
GENERATE_SUPPLEMENTARY = True

# =============================================================================
# ANOVA DESIGN — Curated models/reps for balanced, non-redundant design
# =============================================================================
# Models EXCLUDED from ANOVA (remain in the full study for other analyses):
#   - conformal_*: Wrappers around base models; NDS Spearman rho > 0.99 with
#     their base models. Kept for Part 2 uncertainty analysis.
#   - qrf: NDS rho = 0.996 with rf. Kept for probabilistic vs deterministic
#     comparison in Part 1 deep dives.
#
# Reps EXCLUDED from ANOVA:
#   - sns: Spearman rho = 0.90 with ecfp4 across 14 models (both substructure
#     fingerprints). Kept for supplementary per-rep analysis.
#   - randomized_smiles: Incomplete data coverage across models.
#
# Full BNN variants (last + variational) are ALL included in ANOVA.
# Although pairwise rho ~0.99, user requires all transformation variants
# to be represented for completeness.
#
# See supplementary ICC table (table_supp_icc.csv) and redundancy table
# (table_supp_pairwise_redundancy.csv) for justification.
# =============================================================================

ANOVA_MODELS_EXCLUDE = {
    'conformal_rf_split', 'conformal_qrf_split', 'conformal_dnn_split',  # Wrappers (rho > 0.99)
    'qrf',  # Redundant with rf (rho = 0.996)
}

ANOVA_REPS_EXCLUDE = {
    'sns',                # Redundant with ecfp4 (rho = 0.90)
    'randomized_smiles',  # Incomplete coverage
    'random_smiles',      # Alias
}

# Old var-BNN implementation was identical to last-layer BNN (both used torchbnn.BayesLinear).
# Exclude until new VBLL experiments complete. Remove this exclusion once VBLL data arrives.
VBLL_PENDING_EXCLUDE = {
    'dnn_bnn_variational', 'bnn_variational',
    'mlp_bnn_variational',
}

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
    'quantile': '#882255',     # Wine/purple
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
    # Tree-based — blues/greens
    'rf': '#0072B2',               # Blue
    'xgboost': '#56B4E9',          # Sky blue
    'lgb': '#009E73',              # Teal
    'qrf': '#117733',              # Dark green
    'ngboost': '#D55E00',          # Vermillion (stands out)
    # Neural networks — warm tones
    'dnn': '#E69F00',              # Orange
    'mlp': '#CC79A7',              # Pink
    'flexible_dnn': '#E69F00',     # Orange (same as dnn base)
    'flexible_dnn_256_128_64': '#C68500',  # Darker orange
    'flexible_dnn_512_256': '#F0C050',     # Lighter orange
    # SVM / GP
    'svm': '#999999',              # Gray
    'gauche': '#882255',           # Wine
    # DNN-BNN variants — purples
    'dnn_bnn_full': '#332288',     # Dark purple
    'dnn_bnn_last': '#6633CC',     # Medium purple
    'dnn_bnn_variational': '#9966FF',  # Light purple
    'bnn_full': '#332288',         # Alias
    'bnn_last': '#6633CC',         # Alias
    'bnn_variational': '#9966FF',  # Alias
    # MLP-BNN variants — reds/roses
    'mlp_bnn_full': '#CC3311',     # Red
    'mlp_bnn_last': '#EE6677',     # Salmon
    'mlp_bnn_variational': '#EE99AA',  # Light pink
    # Conformal — teals
    'conformal_rf': '#44AA99',     # Teal
    'conformal_qrf': '#2D6A4F',    # Dark teal
    'conformal_dnn': '#74C69D',    # Light teal
    'conformal_rf_split': '#44AA99',   # Alias
    'conformal_qrf_split': '#2D6A4F',  # Alias
    'conformal_dnn_split': '#74C69D',  # Alias
}

MODEL_LABELS = {
    # Tree-based
    'rf': 'RF',
    'xgboost': 'XGBoost',
    'lgb': 'LightGBM',
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    # Neural networks
    'dnn': 'DNN',
    'mlp': 'MLP',
    'flexible_dnn': 'DNN [128,64]',
    'flexible_dnn_256_128_64': 'DNN [256,128,64]',
    'flexible_dnn_512_256': 'DNN [512,256]',
    # SVM / GP
    'svm': 'SVM',
    'gauche': 'GP (Gauche)',
    # DNN-BNN variants
    'dnn_bnn_full': 'DNN-BNN (Full)',
    'dnn_bnn_last': 'DNN-BNN (Last)',
    'dnn_bnn_variational': 'DNN-BNN (Var.)',
    'bnn_full': 'DNN-BNN (Full)',
    'bnn_last': 'DNN-BNN (Last)',
    'bnn_variational': 'DNN-BNN (Var.)',
    # MLP-BNN variants
    'mlp_bnn_full': 'MLP-BNN (Full)',
    'mlp_bnn_last': 'MLP-BNN (Last)',
    'mlp_bnn_variational': 'MLP-BNN (Var.)',
    # Conformal
    'conformal_rf': 'CP-RF',
    'conformal_qrf': 'CP-QRF',
    'conformal_dnn': 'CP-DNN',
    'conformal_rf_split': 'CP-RF',
    'conformal_qrf_split': 'CP-QRF',
    'conformal_dnn_split': 'CP-DNN',
}

REP_LABELS = {
    'pdv': 'PDV',
    'ecfp4': 'ECFP4',
    'sns': 'SNS',
    'smiles': 'SMILES',
    'randomized_smiles': 'R-SMILES',
    'mhggnn': 'MHG-GNN',
    'mol2vec': 'Mol2Vec',
}

def get_model_label(model):
    """Get clean display label for model."""
    if model in MODEL_LABELS:
        return MODEL_LABELS[model]
    if model.lower() in MODEL_LABELS:
        return MODEL_LABELS[model.lower()]
    # Fallback: replace underscores, title case (avoids ugly DNN_BNN_FULL)
    return model.replace('_', '-').replace('bnn', 'BNN').replace('dnn', 'DNN').replace('mlp', 'MLP').title() if '_' in model else model.upper()

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
CATASTROPHIC_R2_THRESHOLD = -0.5  # Per-iteration R² below this = training failure

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

        # Normalize model names: bnn_full → dnn_bnn_full etc.
        # process_and_train.py saves BNN-DNN variants as 'bnn_full' etc.
        # but all downstream code expects 'dnn_bnn_full' prefix.
        BNN_NAME_MAP = {
            'bnn_full': 'dnn_bnn_full',
            'bnn_last': 'dnn_bnn_last',
            'bnn_variational': 'dnn_bnn_variational',
        }
        if 'model' in combined.columns:
            n_renamed = combined['model'].isin(BNN_NAME_MAP).sum()
            combined['model'] = combined['model'].map(lambda m: BNN_NAME_MAP.get(m, m))
            if n_renamed > 0:
                print(f"  Normalized {n_renamed} BNN model names (bnn_* → dnn_bnn_*)")

        # Normalize column: representation → rep
        if 'representation' in combined.columns and 'rep' not in combined.columns:
            combined.rename(columns={'representation': 'rep'}, inplace=True)

        print(f"Loaded QM9 ANOVA data: {len(combined)} rows from {len(all_data)} files")
        return combined
    return None


def filter_catastrophic_iterations(df, r2_threshold=CATASTROPHIC_R2_THRESHOLD):
    """Filter out catastrophic training iterations (R² below threshold).

    DNN training on certain representations (e.g. mol2vec) occasionally produces
    catastrophic failures with wildly negative R² (e.g. -63.6). These poison
    mean R² calculations and can flip NDS positive. This function removes
    entire iterations where any sigma level has R² below the threshold.

    Returns:
        filtered_df: DataFrame with catastrophic iterations removed
        filtered_log: DataFrame logging what was removed (for paper reporting)
    """
    if 'iteration' not in df.columns or 'r2' not in df.columns:
        return df, pd.DataFrame()

    # Identify iterations where ANY sigma level has R² below threshold
    catastrophic_mask = df['r2'] < r2_threshold
    if catastrophic_mask.sum() == 0:
        return df, pd.DataFrame()

    # Log the catastrophic rows before filtering
    catastrophic_rows = df[catastrophic_mask].copy()
    log_entries = []
    for _, row in catastrophic_rows.iterrows():
        log_entries.append({
            'model': row.get('model', ''),
            'rep': row.get('rep', ''),
            'strategy': row.get('strategy', ''),
            'sigma': row.get('sigma', np.nan),
            'iteration': row.get('iteration', ''),
            'r2': row['r2'],
        })
    filtered_log = pd.DataFrame(log_entries)

    # Remove entire iterations that contain catastrophic R² values
    # (not just the single bad sigma — the whole iteration is suspect)
    group_cols = ['model', 'rep', 'strategy', 'iteration']
    available_cols = [c for c in group_cols if c in df.columns]
    catastrophic_iters = catastrophic_rows[available_cols].drop_duplicates()

    pre_count = len(df)
    filtered_df = df.merge(catastrophic_iters, on=available_cols, how='left', indicator=True)
    filtered_df = filtered_df[filtered_df['_merge'] == 'left_only'].drop(columns='_merge')
    n_removed = pre_count - len(filtered_df)

    print(f"\n  Catastrophic iteration filter (R² < {r2_threshold}):")
    print(f"    Removed {n_removed} rows ({len(filtered_log)} catastrophic R² values across "
          f"{len(catastrophic_iters)} iterations)")
    for _, entry in filtered_log.iterrows():
        print(f"    {entry['model']}/{entry['rep']}/{entry['strategy']} "
              f"σ={entry['sigma']:.1f} iter={entry['iteration']} R²={entry['r2']:.2f}")

    return filtered_df, filtered_log


def load_uncertainty_data(results_dir):
    """Load uncertainty_*_uncertainty_values.csv files."""
    results_dir = Path(results_dir)
    all_data = []

    # Valid strategy names for filename parsing
    VALID_STRATEGIES = {
        'legacy', 'outlier', 'quantile', 'threshold', 'hetero', 'valprop',
        'heteroscedastic', 'value_proportional',
    }
    STRATEGY_NORMALIZE = {'heteroscedastic': 'hetero', 'value_proportional': 'valprop'}

    patterns = ["uncertainty_*_uncertainty_values.csv", "*_uncertainty_values.csv"]

    for pattern in patterns:
        for f in results_dir.glob(pattern):
            try:
                df = pd.read_csv(f)
                df['source_file'] = f.name

                # Normalize column: representation → rep
                if 'representation' in df.columns and 'rep' not in df.columns:
                    df.rename(columns={'representation': 'rep'}, inplace=True)

                # Extract strategy from filename if not in data
                # Pattern: uncertainty_{strategy}_{rep}_{model}_uncertainty_values.csv
                if 'strategy' not in df.columns:
                    parts = f.stem.replace('_uncertainty_values', '').split('_')
                    if len(parts) >= 2 and parts[0] == 'uncertainty':
                        candidate = parts[1]
                        if candidate in VALID_STRATEGIES:
                            df['strategy'] = STRATEGY_NORMALIZE.get(candidate, candidate)

                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Loaded uncertainty data: {len(combined)} rows")
        if 'strategy' in combined.columns:
            print(f"  Strategies: {sorted(combined['strategy'].dropna().unique())}")
        if 'rep' in combined.columns:
            print(f"  Representations: {sorted(combined['rep'].dropna().unique())}")
        return combined
    return None


def fix_injected_noise(df):
    """
    Recompute injected_noise to correct the scale mismatch bug.

    The saved injected_noise = y_true_noisy - y_true_original, but these are in
    different scales (normalized vs raw). The actual noise is:
        noise = y_true_noisy - normalize(y_true_original)

    We recover the normalization transform per group via linear regression:
        y_true_noisy = a * y_true_original + b + noise
    The residuals are the correct noise values.
    """
    if 'y_true_noisy' not in df.columns or 'y_true_original' not in df.columns:
        return df

    required_cols = {'model', 'rep', 'sigma', 'iteration'}
    group_cols = [c for c in required_cols if c in df.columns]
    if not group_cols:
        return df

    corrected_noise = df['injected_noise'].copy() if 'injected_noise' in df.columns else pd.Series(np.nan, index=df.index)
    n_fixed = 0

    for group_key, group_idx in df.groupby(group_cols).groups.items():
        group = df.loc[group_idx]
        y_noisy = group['y_true_noisy'].values
        y_orig = group['y_true_original'].values

        mask = np.isfinite(y_noisy) & np.isfinite(y_orig)
        if mask.sum() < 10:
            continue

        # Linear regression to recover normalization parameters
        slope, intercept, _, _, _ = stats.linregress(y_orig[mask], y_noisy[mask])
        # Residuals = actual noise in normalized space
        residuals = y_noisy - (slope * y_orig + intercept)
        corrected_noise.loc[group_idx] = residuals
        n_fixed += 1

    df['injected_noise'] = corrected_noise
    print(f"  Fixed injected_noise via linear regression ({n_fixed} groups)")
    return df


def _normalize_validation_names(df):
    """Normalize KIRBy naming conventions to match QM9 conventions."""
    val_model_map = {
        'RF': 'rf', 'XGBoost': 'xgboost', 'DNN': 'dnn', 'MLP': 'mlp',
        'GP': 'gauche', 'QRF': 'qrf', 'NGBoost': 'ngboost', 'SVM': 'svm',
        'BNN-Full': 'dnn_bnn_full', 'BNN-Last': 'dnn_bnn_last',
        'BNN-Var': 'dnn_bnn_variational',
    }
    val_rep_map = {
        'ECFP4': 'ecfp4', 'PDV': 'pdv', 'SNS': 'sns',
        'MHG-GNN-pretrained': 'mhggnn', 'MHGGNNpretrained': 'mhggnn',
        'SMILES': 'smiles',
    }
    if 'model' in df.columns:
        df['model'] = df['model'].map(val_model_map).fillna(df['model'].str.lower())
    if 'rep' in df.columns:
        df['rep'] = df['rep'].map(val_rep_map).fillna(df['rep'].str.lower())
    # Normalize NDS column name
    if 'NDS_r2' in df.columns and 'nds' not in df.columns:
        df = df.rename(columns={'NDS_r2': 'nds'})
    return df


def load_validation_data(validation_dir):
    """Load validation data from KIRBy results directory.

    Supports two layouts:
    1. combined_summary.csv (pre-merged, must have 'dataset' column)
    2. Per-dataset subdirectories, each with all_results.csv or summary.csv
       The subdirectory name becomes the 'dataset' column.
    """
    if validation_dir is None:
        return None

    validation_dir = Path(validation_dir)
    summary_file = validation_dir / 'combined_summary.csv'

    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df = _normalize_validation_names(df)
        print(f"Loaded validation data: {len(df)} rows from combined_summary.csv")
        return df

    # Try loading per-dataset subdirectories
    all_data = []
    for subdir in sorted(validation_dir.iterdir()):
        if not subdir.is_dir():
            continue
        # Prefer all_results.csv (per-sigma format) for NDS computation
        results_file = subdir / 'all_results.csv'
        if results_file.exists():
            df = pd.read_csv(results_file)
            df['dataset'] = subdir.name
            all_data.append(df)
            continue
        # Fall back to summary.csv
        summary = subdir / 'summary.csv'
        if summary.exists():
            df = pd.read_csv(summary)
            df['dataset'] = subdir.name
            all_data.append(df)

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = _normalize_validation_names(combined)
        datasets = combined['dataset'].unique()
        print(f"Loaded validation data: {len(combined)} rows from {len(datasets)} datasets ({', '.join(sorted(datasets))})")
        return combined

    print("⚠ No validation data found")
    return None


def calculate_validation_nds(validation_df):
    """Convert validation data into NDS-compatible format.

    The KIRBy validation script outputs:
    - all_results.csv: per-sigma R² (columns: sigma, r2, model, rep, strategy)
    - summary.csv: NSI (= NDS), baseline_r2 (columns: model, rep, strategy, baseline_r2, NSI)

    This function handles both formats and adds a 'dataset' column.
    """
    if validation_df is None or len(validation_df) == 0:
        return None

    # If summary format (has NSI or NDS_r2 or nds column), convert directly
    nds_col = None
    for col in ['nds', 'NSI', 'NDS_r2', 'nsi_r2']:
        if col in validation_df.columns:
            nds_col = col
            break

    if nds_col is not None:
        nds_df = validation_df.rename(columns={nds_col: 'nds'})
        if 'baseline_r2' not in nds_df.columns:
            nds_df['baseline_r2'] = np.nan
        if 'dataset' not in nds_df.columns:
            nds_df['dataset'] = 'validation'
        return nds_df

    # If per-sigma format (has sigma and r2 columns), compute NDS
    if 'sigma' in validation_df.columns and 'r2' in validation_df.columns:
        group_cols = ['model', 'rep', 'strategy']
        if 'dataset' in validation_df.columns:
            group_cols = ['dataset'] + group_cols

        nds_results = []
        for keys, group in validation_df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            avg = group.groupby('sigma')['r2'].mean().reset_index().sort_values('sigma')
            if len(avg) < 3:
                continue
            baseline = avg[avg['sigma'] == 0.0]['r2'].values
            if len(baseline) == 0:
                continue
            baseline = baseline[0]
            if baseline < BASELINE_THRESHOLD:
                continue
            try:
                slope, _, r_val, _, _ = stats.linregress(avg['sigma'], avg['r2'])
            except:
                continue
            row = dict(zip(group_cols, keys))
            row.update({'nds': slope, 'baseline_r2': baseline, 'r2_fit': r_val**2})
            nds_results.append(row)

        if nds_results:
            return pd.DataFrame(nds_results)

    return None


def create_validation_figures(validation_df, val_nds_df, output_dir):
    """Generate all validation-related figures and tables.

    Validation data (LogD, Caco2_Efflux, hERG-Ki) is integrated into the paper
    as generalisation evidence — NOT a separate section.

    Outputs:
    - fig_validation_overview.png: NDS heatmap per dataset (like Figure 1B)
    - fig_validation_anova.png: η² decomposition on validation datasets
    - table_validation_nds.csv: Full NDS table across datasets
    - table_validation_anova.csv: Validation ANOVA statistics
    """
    if val_nds_df is None or len(val_nds_df) == 0:
        print("⚠ No validation NDS data available — skipping validation figures")
        return

    datasets = val_nds_df['dataset'].unique() if 'dataset' in val_nds_df.columns else ['validation']
    print(f"  Validation datasets: {sorted(datasets)}, {len(val_nds_df)} configs total")

    # --- Table: Validation NDS (cross-dataset comparison) ---
    if 'dataset' in val_nds_df.columns:
        pivot = val_nds_df.pivot_table(
            values='nds', index=['model', 'rep'], columns='dataset', aggfunc='mean'
        )
        pivot['MEAN'] = pivot.mean(axis=1)
        pivot['STD'] = pivot.drop(columns=['MEAN']).std(axis=1)
        pivot = pivot.sort_values('MEAN', ascending=False)
        pivot.to_csv(output_dir / 'table_validation_nds.csv')
        print("✓ Saved table_validation_nds.csv")
    else:
        val_nds_df.to_csv(output_dir / 'table_validation_nds.csv', index=False)
        print("✓ Saved table_validation_nds.csv")

    # --- Figure: Validation NDS heatmap per dataset ---
    n_datasets = len(datasets)
    if n_datasets > 0 and 'dataset' in val_nds_df.columns:
        fig, axes = plt.subplots(1, n_datasets, figsize=(5 * n_datasets, 6), squeeze=False)
        axes = axes[0]

        for i, dataset in enumerate(sorted(datasets)):
            ax = axes[i]
            ds_data = val_nds_df[val_nds_df['dataset'] == dataset]
            if len(ds_data) == 0:
                continue

            pivot = ds_data.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
            col_order = [c for c in ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
                         if c in pivot.columns]
            if not col_order:
                continue
            pivot = pivot[col_order]
            pivot.index = [get_model_label(m) for m in pivot.index]

            # Clip extreme NDS values for colormap, but annotate with actual values
            NDS_CLIP = 2.0
            pivot_display = pivot.clip(lower=-NDS_CLIP)
            vals = pivot_display.values[~np.isnan(pivot_display.values)]
            vmin = vals.min() if len(vals) > 0 else -NDS_CLIP
            # Custom annotation: actual values (compact format for extremes)
            annot_text = pivot.applymap(
                lambda x: '' if pd.isna(x) else (f'{x:.0f}' if abs(x) >= 10 else f'{x:.2f}'))
            # Black background so NaN/missing cells are clearly marked
            ax.set_facecolor('black')
            sns.heatmap(pivot_display, annot=annot_text, fmt='', cmap='RdBu', center=0,
                        vmin=vmin, vmax=0,
                        ax=ax, cbar_kws={'label': 'NDS'}, linewidths=0.5,
                        linecolor='#333333')
            ax.set_title(f'{dataset}', fontweight='bold')
            ax.set_xticklabels([STRATEGY_LABELS.get(c, c) for c in col_order], rotation=45, ha='right')
            if i > 0:
                ax.set_ylabel('')

        plt.suptitle('Validation Datasets: NDS by Model × Strategy', fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(output_dir / 'fig_validation_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved fig_validation_overview.png")

    # --- Figure: Validation ANOVA (η² decomposition) ---
    if validation_df is not None and 'sigma' in validation_df.columns and 'dataset' in validation_df.columns:
        anova_results = {}
        for dataset in sorted(datasets):
            ds_df = validation_df[validation_df['dataset'] == dataset]
            if len(ds_df) == 0:
                continue
            # Run robustness ANOVA on this dataset
            robust = run_robustness_anova(ds_df)
            if robust:
                anova_results[dataset] = robust

        if anova_results:
            fig, axes = plt.subplots(1, len(anova_results), figsize=(4 * len(anova_results), 5), squeeze=False)
            axes = axes[0]

            for i, (dataset, result) in enumerate(anova_results.items()):
                ax = axes[i]
                factors = ['Model', 'Rep', 'Interaction']
                values = [result['eta2_model'], result['eta2_rep'], result['eta2_interaction']]
                colors = ['#3498db', '#E69F00', '#0072B2']  # Blue, Orange, Dark blue
                ax.bar(factors, values, color=colors)
                ax.set_ylabel('Variance Explained (η², %)')
                ax.set_title(f'{dataset}', fontweight='bold')
                ax.set_ylim(0, 100)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            plt.suptitle('Validation: ANOVA Variance Decomposition (Robustness)', fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_validation_anova.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved fig_validation_anova.png")

            # Save ANOVA table
            rows = []
            for dataset, result in anova_results.items():
                rows.append({
                    'Dataset': dataset,
                    'Model_η²': result['eta2_model'],
                    'Rep_η²': result['eta2_rep'],
                    'Interaction_η²': result['eta2_interaction'],
                    'n_models': result.get('n_models', ''),
                    'n_reps': result.get('n_reps', ''),
                })
            pd.DataFrame(rows).to_csv(output_dir / 'table_validation_anova.csv', index=False)
            print("✓ Saved table_validation_anova.csv")

    # --- Cross-dataset ranking comparison ---
    if 'dataset' in val_nds_df.columns and len(datasets) >= 2:
        # Compare model rankings across datasets using Gaussian strategy
        rank_data = {}
        for dataset in sorted(datasets):
            ds_nds = val_nds_df[(val_nds_df['dataset'] == dataset) &
                                (val_nds_df['strategy'] == 'legacy')]
            if len(ds_nds) == 0:
                ds_nds = val_nds_df[val_nds_df['dataset'] == dataset]
            model_means = ds_nds.groupby('model')['nds'].mean()
            rank_data[dataset] = model_means.rank(ascending=False)

        if rank_data:
            rank_df = pd.DataFrame(rank_data)
            # Also add QM9 rankings for comparison if we can compute them later
            rank_df['Mean Rank'] = rank_df.mean(axis=1)
            rank_df = rank_df.sort_values('Mean Rank')
            rank_df.to_csv(output_dir / 'table_validation_rankings.csv')
            print("✓ Saved table_validation_rankings.csv")

    # --- Probabilistic comparison on validation datasets ---
    if 'dataset' in val_nds_df.columns:
        prob_pairs = [
            ('rf', 'qrf', 'RF vs QRF'),
        ]
        # Add DNN/MLP BNN comparisons if available
        for base in ['dnn', 'mlp']:
            for suffix in ['_bnn_full', '_bnn_last', '_bnn_variational']:
                variant = base + suffix
                if variant in val_nds_df['model'].values:
                    prob_pairs.append((base, variant, f'{base.upper()} vs {variant}'))

        rows = []
        for base, variant, label in prob_pairs:
            for dataset in sorted(datasets):
                ds_nds = val_nds_df[val_nds_df['dataset'] == dataset]
                base_nds = ds_nds[ds_nds['model'] == base]['nds'].mean()
                var_nds = ds_nds[ds_nds['model'] == variant]['nds'].mean()
                if np.isfinite(base_nds) and np.isfinite(var_nds):
                    rows.append({
                        'Dataset': dataset,
                        'Comparison': label,
                        'Base NDS': base_nds,
                        'Variant NDS': var_nds,
                        'Δ NDS': var_nds - base_nds,
                        'Variant Better': var_nds > base_nds,
                    })

        if rows:
            pd.DataFrame(rows).to_csv(output_dir / 'table_validation_probabilistic.csv', index=False)
            print("✓ Saved table_validation_probabilistic.csv")


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

    nds_df = pd.DataFrame(nds_results)

    # Flag suspicious positive NDS values (performance improving with noise)
    if len(nds_df) > 0:
        positive = nds_df[nds_df['nds'] > 0]
        if len(positive) > 0:
            print(f"\n  ⚠ WARNING: {len(positive)} configs have POSITIVE NDS (improving with noise):")
            for _, row in positive.iterrows():
                print(f"    {row['model']}/{row['rep']}/{row['strategy']}: NDS={row['nds']:.4f}")
            print("    These are likely statistical artifacts from incomplete data.\n")

    return nds_df, pd.DataFrame(excluded)


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

    Pairs models across all shared (rep, strategy) conditions to get enough
    matched observations. Falls back to single rep×strategy if pairing across
    conditions yields too few points.

    Returns:
        dict with statistic, p_value, and interpretation
    """
    # Pair across all shared conditions (rep × strategy) for sufficient n
    base_data = nds_df[nds_df['model'] == model_base]
    var_data = nds_df[nds_df['model'] == model_variant]

    if len(base_data) == 0 or len(var_data) == 0:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False, 'n': 0}

    # Create keys for pairing
    base_keyed = base_data.set_index(['rep', 'strategy'])['nds']
    var_keyed = var_data.set_index(['rep', 'strategy'])['nds']
    shared_keys = base_keyed.index.intersection(var_keyed.index)

    base_nds = base_keyed.loc[shared_keys].values
    var_nds = var_keyed.loc[shared_keys].values

    n = len(shared_keys)
    if n < 5:
        return {'statistic': np.nan, 'p_value': np.nan, 'significant': False, 'n': n}

    try:
        stat, p_val = stats.wilcoxon(base_nds, var_nds, alternative='two-sided')
        return {
            'statistic': stat,
            'p_value': p_val,
            'significant': p_val < 0.05,
            'n': n,
            'base_mean': np.mean(base_nds),
            'var_mean': np.mean(var_nds),
            'improvement': np.mean(var_nds) - np.mean(base_nds)
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

    # Find models present in ALL strategies
    model_nds_by_strategy = {}
    for strategy in strategies:
        strat_data = df[df['strategy'] == strategy]
        model_nds_by_strategy[strategy] = strat_data.groupby('model')['nds'].mean()

    valid_models = [m for m in models
                    if all(m in model_nds_by_strategy[s].index for s in strategies)]

    if len(valid_models) < 3:
        return {'W': np.nan, 'p_value': np.nan, 'interpretation': 'Insufficient complete rankings'}

    # Re-rank using only valid models (avoids gaps that inflate W > 1)
    rank_df = pd.DataFrame(index=valid_models, columns=strategies, dtype=float)
    for strategy in strategies:
        nds_vals = model_nds_by_strategy[strategy].loc[valid_models]
        rank_df[strategy] = nds_vals.rank(ascending=False)

    # Calculate Kendall's W = 12 * S / (k^2 * (n^3 - n))
    k = len(strategies)  # number of raters
    n = len(valid_models)  # number of items

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
    """Run two-way ANOVA for model/rep effects.

    Filters to ANOVA-curated models and reps (see ANOVA_MODELS_EXCLUDE,
    ANOVA_REPS_EXCLUDE). Excluded models/reps remain available for other analyses.
    """
    df_sigma = df[np.abs(df['sigma'] - sigma_value) < 0.05].copy()

    if len(df_sigma) == 0:
        return None

    df_sigma = df_sigma[df_sigma['r2'] > -10].dropna(subset=['r2', 'model', 'rep'])
    df_sigma = df_sigma[~df_sigma['rep'].isin(ANOVA_REPS_EXCLUDE)]
    df_sigma = df_sigma[~df_sigma['model'].isin(ANOVA_MODELS_EXCLUDE)]

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
    """Run ANOVA on NDS values.

    Filters to ANOVA-curated models and reps (see ANOVA_MODELS_EXCLUDE,
    ANOVA_REPS_EXCLUDE). Excluded models/reps remain available for other analyses.
    """
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
    nds_df = nds_df[~nds_df['rep'].isin(ANOVA_REPS_EXCLUDE)]
    nds_df = nds_df[~nds_df['model'].isin(ANOVA_MODELS_EXCLUDE)]

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


def run_simple_effects(df, response_col, group_col, factor_col):
    """
    Run one-way ANOVA (simple effects) for `factor_col` within each level of `group_col`.

    When interaction η² is large, main effects are misleading. Simple effects
    answer: "How much does Factor A matter at each level of Factor B?"

    Returns list of dicts with group level, η², F-statistic, and p-value.
    """
    results = []
    for group_level, group_data in df.groupby(group_col):
        groups = [g[response_col].values for _, g in group_data.groupby(factor_col)
                  if len(g) >= 2]
        if len(groups) < 2:
            continue

        grand_mean = group_data[response_col].mean()
        total_ss = ((group_data[response_col] - grand_mean) ** 2).sum()
        if total_ss == 0:
            continue

        factor_means = group_data.groupby(factor_col)[response_col].mean()
        factor_counts = group_data.groupby(factor_col).size()
        ss_factor = sum(factor_counts * (factor_means - grand_mean) ** 2)

        eta2 = (ss_factor / total_ss) * 100

        # F-test
        try:
            f_stat, p_val = stats.f_oneway(*groups)
        except:
            f_stat, p_val = np.nan, np.nan

        results.append({
            'group': group_level,
            'eta2': eta2,
            'f_stat': f_stat,
            'p_value': p_val,
            'n': len(group_data),
            'n_levels': len(groups),
        })

    return results


def run_simple_effects_analysis(df, output_dir):
    """
    Simple effects analysis for performance and robustness.

    When two-way ANOVA shows large interaction, main effects are uninterpretable.
    Simple effects decompose: effect of model at each rep, and effect of rep at each model.
    """
    all_rows = []

    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)

        # --- Performance simple effects (R² at σ=0.3) ---
        df_sigma = strategy_df[np.abs(strategy_df['sigma'] - 0.3) < 0.05].copy()
        df_sigma = df_sigma[df_sigma['r2'] > -10].dropna(subset=['r2', 'model', 'rep'])
        df_sigma = df_sigma[~df_sigma['rep'].isin(ANOVA_REPS_EXCLUDE)]
        df_sigma = df_sigma[~df_sigma['model'].isin(ANOVA_MODELS_EXCLUDE)]

        if len(df_sigma) >= 10:
            # Simple effect of model at each representation
            model_at_rep = run_simple_effects(df_sigma, 'r2', 'rep', 'model')
            for r in model_at_rep:
                all_rows.append({
                    'Strategy': strategy_label,
                    'Analysis': 'Performance',
                    'Type': 'Model effect',
                    'Within': r['group'],
                    'eta2': r['eta2'],
                    'F': r['f_stat'],
                    'p': r['p_value'],
                    'n': r['n'],
                })

            # Simple effect of representation at each model
            rep_at_model = run_simple_effects(df_sigma, 'r2', 'model', 'rep')
            for r in rep_at_model:
                all_rows.append({
                    'Strategy': strategy_label,
                    'Analysis': 'Performance',
                    'Type': 'Rep effect',
                    'Within': r['group'],
                    'eta2': r['eta2'],
                    'F': r['f_stat'],
                    'p': r['p_value'],
                    'n': r['n'],
                })

        # --- Robustness simple effects (NDS) ---
        nds_data = []
        for (model, rep, iteration), group in strategy_df.groupby(['model', 'rep', 'iteration']):
            group = group.sort_values('sigma')
            if len(group) < 3:
                continue
            baseline = group[group['sigma'] == 0.0]
            if len(baseline) == 0 or baseline['r2'].values[0] < BASELINE_THRESHOLD:
                continue
            try:
                slope, _, _, _, _ = stats.linregress(group['sigma'], group['r2'])
                nds_data.append({'model': model, 'rep': rep, 'iteration': iteration, 'nds': slope})
            except:
                continue

        if len(nds_data) >= 10:
            nds_df = pd.DataFrame(nds_data)
            nds_df = nds_df[~nds_df['rep'].isin(ANOVA_REPS_EXCLUDE)]
            nds_df = nds_df[~nds_df['model'].isin(ANOVA_MODELS_EXCLUDE)]

            # Simple effect of model at each representation
            model_at_rep = run_simple_effects(nds_df, 'nds', 'rep', 'model')
            for r in model_at_rep:
                all_rows.append({
                    'Strategy': strategy_label,
                    'Analysis': 'Robustness',
                    'Type': 'Model effect',
                    'Within': r['group'],
                    'eta2': r['eta2'],
                    'F': r['f_stat'],
                    'p': r['p_value'],
                    'n': r['n'],
                })

            # Simple effect of representation at each model
            rep_at_model = run_simple_effects(nds_df, 'nds', 'model', 'rep')
            for r in rep_at_model:
                all_rows.append({
                    'Strategy': strategy_label,
                    'Analysis': 'Robustness',
                    'Type': 'Rep effect',
                    'Within': r['group'],
                    'eta2': r['eta2'],
                    'F': r['f_stat'],
                    'p': r['p_value'],
                    'n': r['n'],
                })

    if all_rows:
        se_df = pd.DataFrame(all_rows)
        se_df.to_csv(output_dir / 'table1_supp_simple_effects.csv', index=False)
        print(f"✓ Saved table1_supp_simple_effects.csv ({len(se_df)} rows)")

        # Print summary
        print("\n  Simple Effects Summary (η² for model effect within each representation):")
        perf_model = se_df[(se_df['Analysis'] == 'Performance') & (se_df['Type'] == 'Model effect')]
        if len(perf_model) > 0:
            summary = perf_model.groupby('Within')['eta2'].mean()
            for rep, eta2 in summary.items():
                print(f"    {rep}: Model η² = {eta2:.1f}%")

        print("  Simple Effects Summary (η² for rep effect within each model):")
        perf_rep = se_df[(se_df['Analysis'] == 'Performance') & (se_df['Type'] == 'Rep effect')]
        if len(perf_rep) > 0:
            summary = perf_rep.groupby('Within')['eta2'].mean()
            for model, eta2 in sorted(summary.items(), key=lambda x: -x[1]):
                print(f"    {model}: Rep η² = {eta2:.1f}%")
    else:
        print("⚠ No simple effects data computed")

    # --- Supplementary: simple effects for ALL reps/models (no ANOVA exclusions) ---
    # This captures trends for SNS, randomized_smiles, conformal, QRF etc.
    all_rows_full = []
    for strategy in df['strategy'].unique():
        strategy_df = df[df['strategy'] == strategy]
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)

        # Performance (R² at σ=0.3) — no exclusions
        df_sigma = strategy_df[np.abs(strategy_df['sigma'] - 0.3) < 0.05].copy()
        df_sigma = df_sigma[df_sigma['r2'] > -10].dropna(subset=['r2', 'model', 'rep'])

        if len(df_sigma) >= 10:
            for r in run_simple_effects(df_sigma, 'r2', 'rep', 'model'):
                all_rows_full.append({
                    'Strategy': strategy_label, 'Analysis': 'Performance',
                    'Type': 'Model effect', 'Within': r['group'],
                    'eta2': r['eta2'], 'F': r['f_stat'], 'p': r['p_value'], 'n': r['n'],
                })
            for r in run_simple_effects(df_sigma, 'r2', 'model', 'rep'):
                all_rows_full.append({
                    'Strategy': strategy_label, 'Analysis': 'Performance',
                    'Type': 'Rep effect', 'Within': r['group'],
                    'eta2': r['eta2'], 'F': r['f_stat'], 'p': r['p_value'], 'n': r['n'],
                })

        # Robustness (NDS) — no exclusions
        nds_data = []
        for (model, rep, iteration), group in strategy_df.groupby(['model', 'rep', 'iteration']):
            group = group.sort_values('sigma')
            if len(group) < 3:
                continue
            baseline = group[group['sigma'] == 0.0]
            if len(baseline) == 0 or baseline['r2'].values[0] < BASELINE_THRESHOLD:
                continue
            try:
                slope, _, _, _, _ = stats.linregress(group['sigma'], group['r2'])
                nds_data.append({'model': model, 'rep': rep, 'iteration': iteration, 'nds': slope})
            except:
                continue

        if len(nds_data) >= 10:
            nds_df_full = pd.DataFrame(nds_data)
            for r in run_simple_effects(nds_df_full, 'nds', 'rep', 'model'):
                all_rows_full.append({
                    'Strategy': strategy_label, 'Analysis': 'Robustness',
                    'Type': 'Model effect', 'Within': r['group'],
                    'eta2': r['eta2'], 'F': r['f_stat'], 'p': r['p_value'], 'n': r['n'],
                })
            for r in run_simple_effects(nds_df_full, 'nds', 'model', 'rep'):
                all_rows_full.append({
                    'Strategy': strategy_label, 'Analysis': 'Robustness',
                    'Type': 'Rep effect', 'Within': r['group'],
                    'eta2': r['eta2'], 'F': r['f_stat'], 'p': r['p_value'], 'n': r['n'],
                })

    if all_rows_full:
        se_full_df = pd.DataFrame(all_rows_full)
        se_full_df.to_csv(output_dir / 'table1_supp_simple_effects_all_reps.csv', index=False)
        print(f"✓ Saved table1_supp_simple_effects_all_reps.csv ({len(se_full_df)} rows)")

        # Print SNS specifically since it was excluded from ANOVA
        rob_model = se_full_df[(se_full_df['Analysis'] == 'Robustness') & (se_full_df['Type'] == 'Model effect')]
        sns_rows = rob_model[rob_model['Within'] == 'sns']
        if len(sns_rows) > 0:
            print(f"  SNS robustness — Model η² (mean across strategies): {sns_rows['eta2'].mean():.1f}%")
    else:
        print("⚠ No supplementary simple effects data computed")


# =============================================================================
# SUPPLEMENTARY: ICC AND PAIRWISE REDUNDANCY TABLES
# =============================================================================

def compute_icc_and_redundancy(nds_df, output_dir):
    """Compute ICC(1,1) for all model pairs and pairwise Spearman redundancy.

    Outputs three supplementary CSVs:
      - table_supp_model_redundancy.csv: Spearman rho between model NDS profiles
      - table_supp_rep_redundancy.csv: Spearman rho between rep NDS profiles
      - table_supp_icc.csv: ICC(1,1) for each model pair

    These tables justify which models/reps were excluded from the ANOVA.
    """
    if len(nds_df) == 0:
        print("⚠ No NDS data for ICC computation")
        return

    strategies = [s for s in ALL_STRATEGIES if s in nds_df['strategy'].unique()]

    # Build NDS pivot: mean NDS per (model, rep, strategy)
    mean_nds = nds_df.groupby(['model', 'rep', 'strategy'])['nds'].mean().reset_index()

    # ── Pairwise MODEL redundancy using full (rep x strategy) vectors ──
    models = sorted(mean_nds['model'].unique())
    model_profiles = {}
    for model in models:
        mdata = mean_nds[mean_nds['model'] == model]
        profile = {}
        for _, row in mdata.iterrows():
            profile[(row['rep'], row['strategy'])] = row['nds']
        model_profiles[model] = profile

    model_pairs = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            shared = sorted(set(model_profiles[m1].keys()) & set(model_profiles[m2].keys()))
            if len(shared) < 3:
                continue
            v1 = [model_profiles[m1][k] for k in shared]
            v2 = [model_profiles[m2][k] for k in shared]
            rho, p = stats.spearmanr(v1, v2)
            model_pairs.append({
                'model_a': m1, 'model_b': m2,
                'n_shared_points': len(shared),
                'spearman_rho': rho, 'p_value': p,
                'excluded_from_anova': (
                    'yes' if m1 in ANOVA_MODELS_EXCLUDE or m2 in ANOVA_MODELS_EXCLUDE
                    else 'no'
                ),
            })

    if model_pairs:
        mp_df = pd.DataFrame(model_pairs).sort_values('spearman_rho', ascending=False)
        mp_df.to_csv(output_dir / 'table_supp_model_redundancy.csv', index=False)
        print(f"✓ Saved table_supp_model_redundancy.csv ({len(mp_df)} pairs)")

    # ── Pairwise REP redundancy ──
    reps = sorted(mean_nds['rep'].unique())
    rep_profiles = {}
    for rep in reps:
        rdata = mean_nds[mean_nds['rep'] == rep]
        profile = {}
        for _, row in rdata.iterrows():
            profile[(row['model'], row['strategy'])] = row['nds']
        rep_profiles[rep] = profile

    rep_pairs = []
    for i, r1 in enumerate(reps):
        for r2 in reps[i+1:]:
            shared = sorted(set(rep_profiles[r1].keys()) & set(rep_profiles[r2].keys()))
            if len(shared) < 3:
                continue
            v1 = [rep_profiles[r1][k] for k in shared]
            v2 = [rep_profiles[r2][k] for k in shared]
            rho, p = stats.spearmanr(v1, v2)
            rep_pairs.append({
                'rep_a': r1, 'rep_b': r2,
                'n_shared_points': len(shared),
                'spearman_rho': rho, 'p_value': p,
                'excluded_from_anova': (
                    'yes' if r1 in ANOVA_REPS_EXCLUDE or r2 in ANOVA_REPS_EXCLUDE
                    else 'no'
                ),
            })

    if rep_pairs:
        rp_df = pd.DataFrame(rep_pairs).sort_values('spearman_rho', ascending=False)
        rp_df.to_csv(output_dir / 'table_supp_rep_redundancy.csv', index=False)
        print(f"✓ Saved table_supp_rep_redundancy.csv ({len(rp_df)} pairs)")

    # ── ICC(1,1) per model pair ──
    # Treats reps as "subjects", compares NDS agreement between two models
    overall_nds = mean_nds.groupby(['model', 'rep'])['nds'].mean().reset_index()
    icc_rows = []

    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            m1_data = overall_nds[overall_nds['model'] == m1][['rep', 'nds']].rename(
                columns={'nds': 'nds_m1'})
            m2_data = overall_nds[overall_nds['model'] == m2][['rep', 'nds']].rename(
                columns={'nds': 'nds_m2'})
            merged = m1_data.merge(m2_data, on='rep')

            if len(merged) < 2:
                continue

            n = len(merged)
            k = 2
            data = merged[['nds_m1', 'nds_m2']].values
            grand_mean = data.mean()
            row_means = data.mean(axis=1)
            bms = k * ((row_means - grand_mean) ** 2).sum() / (n - 1)
            wms = ((data - row_means.reshape(-1, 1)) ** 2).sum() / (n * (k - 1))

            if bms + (k - 1) * wms == 0:
                icc = np.nan
            else:
                icc = (bms - wms) / (bms + (k - 1) * wms)

            mad = np.mean(np.abs(merged['nds_m1'] - merged['nds_m2']))

            icc_rows.append({
                'model_a': m1, 'model_b': m2,
                'n_shared_reps': n,
                'icc_1_1': icc,
                'mean_abs_nds_diff': mad,
                'mean_nds_a': merged['nds_m1'].mean(),
                'mean_nds_b': merged['nds_m2'].mean(),
            })

    if icc_rows:
        icc_df = pd.DataFrame(icc_rows).sort_values('icc_1_1', ascending=False)
        icc_df.to_csv(output_dir / 'table_supp_icc.csv', index=False)
        print(f"✓ Saved table_supp_icc.csv ({len(icc_df)} model pairs)")

        high_icc = icc_df[icc_df['icc_1_1'] > 0.9].head(10)
        if len(high_icc) > 0:
            print("  High ICC pairs (>0.9, candidates for ANOVA exclusion):")
            for _, r in high_icc.iterrows():
                print(f"    {r['model_a']} <-> {r['model_b']}: "
                      f"ICC={r['icc_1_1']:.3f}, MAD={r['mean_abs_nds_diff']:.4f}")


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

        # Use common bins for both distributions
        all_vals = np.concatenate([y_clean, y_noisy])
        bins = np.linspace(all_vals.min(), all_vals.max(), 51)

        # Filled histograms: very transparent so overlap is visible
        ax.hist(y_clean, bins=bins, alpha=0.15, color='#0072B2', density=True)
        ax.hist(y_noisy, bins=bins, alpha=0.15, color=STRATEGY_COLORS[strategy], density=True)

        # Bold step outlines on TOP of curves
        n_clean, _, _ = ax.hist(y_clean, bins=bins, density=True,
                                histtype='step', linewidth=2.0, color='#0072B2',
                                label='Clean')
        n_noisy, _, _ = ax.hist(y_noisy, bins=bins, density=True,
                                histtype='step', linewidth=2.0, color=STRATEGY_COLORS[strategy],
                                label=f'Noisy (σ={sigma})')

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

            detail_bins = np.linspace(min(y_clean.min(), y_noisy.min()),
                                      max(y_clean.max(), y_noisy.max()), 51)
            ax.hist(y_clean, bins=detail_bins, alpha=0.15, color='#0072B2', density=True)
            ax.hist(y_clean, bins=detail_bins, density=True,
                    histtype='step', linewidth=1.5, color='#0072B2')
            if sig > 0:
                ax.hist(y_noisy, bins=detail_bins, alpha=0.15, color=STRATEGY_COLORS[strategy], density=True)
                ax.hist(y_noisy, bins=detail_bins, density=True,
                        histtype='step', linewidth=1.5, color=STRATEGY_COLORS[strategy])

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
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1.2])

    # Panel A: R² vs σ for key models on PDV
    ax_a = fig.add_subplot(gs[0])

    key_models = ['rf', 'qrf', 'dnn', 'mlp', 'ngboost', 'xgboost']
    pdv_data = df[(df['rep'] == 'pdv') & (df['strategy'] == 'legacy')]

    # Background: all non-key models in light grey so full range is visible
    all_models = pdv_data['model'].unique()
    for model in all_models:
        if model in key_models:
            continue
        model_data = pdv_data[pdv_data['model'] == model]
        if len(model_data) == 0:
            continue
        avg = model_data.groupby('sigma')['r2'].mean().reset_index()
        ax_a.plot(avg['sigma'], avg['r2'], '-', color='#cccccc', alpha=0.4,
                  linewidth=1.0, zorder=1)

    # Foreground: key models highlighted with bold lines
    for model in key_models:
        model_data = pdv_data[pdv_data['model'] == model]
        if len(model_data) == 0:
            continue

        avg = model_data.groupby('sigma')['r2'].mean().reset_index()
        color = MODEL_COLORS.get(model, '#333333')
        ax_a.plot(avg['sigma'], avg['r2'], 'o-', label=get_model_label(model),
                  color=color, markersize=4, linewidth=2.0, alpha=0.85, zorder=5)

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

        # Use colorblind-friendly diverging colormap centered at data midpoint
        # so both colors are visible (all NDS values are negative)
        vals = pivot.values[~np.isnan(pivot.values)]
        center_val = (vals.min() + vals.max()) / 2 if len(vals) > 0 else 0
        ax_b.set_facecolor('black')
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu', center=center_val,
                    ax=ax_b, cbar_kws={'label': 'NDS'})
        ax_b.set_xlabel('Noise Strategy')
        ax_b.set_ylabel('Model')
        ax_b.set_title('B. NDS by Model × Strategy (PDV)', fontweight='bold')
        ax_b.set_xticklabels([STRATEGY_LABELS.get(c, c) for c in col_order], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_global_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig1_global_overview.png")

    # --- ECFP4 variant (Issue C: compare to PDV; supplementary unless very different) ---
    fig_e, axes_e = plt.subplots(1, 2, figsize=(12, 5))
    ax_ea = axes_e[0]

    ecfp4_data = df[(df['rep'] == 'ecfp4') & (df['strategy'] == 'legacy')]

    # Background: all non-key models in light grey
    all_ecfp4_models = ecfp4_data['model'].unique()
    for model in all_ecfp4_models:
        if model in key_models:
            continue
        model_data = ecfp4_data[ecfp4_data['model'] == model]
        if len(model_data) == 0:
            continue
        avg = model_data.groupby('sigma')['r2'].mean().reset_index()
        ax_ea.plot(avg['sigma'], avg['r2'], '-', color='#cccccc', alpha=0.4,
                   linewidth=1.0, zorder=1)

    # Foreground: key models highlighted
    for model in key_models:
        model_data = ecfp4_data[ecfp4_data['model'] == model]
        if len(model_data) == 0:
            continue
        avg = model_data.groupby('sigma')['r2'].mean().reset_index()
        color = MODEL_COLORS.get(model, '#333333')
        ax_ea.plot(avg['sigma'], avg['r2'], 'o-', label=get_model_label(model),
                   color=color, markersize=4, linewidth=2.0, alpha=0.85, zorder=5)

    ax_ea.set_xlabel('Noise Level (σ)')
    ax_ea.set_ylabel('R²')
    ax_ea.set_title('A. Performance Degradation (ECFP4, Gaussian Noise)', fontweight='bold')
    ax_ea.legend(loc='lower left', ncol=2)
    ax_ea.set_ylim(-0.1, 1.0)
    ax_ea.spines['top'].set_visible(False)
    ax_ea.spines['right'].set_visible(False)

    ax_eb = axes_e[1]
    if len(nds_df) > 0:
        nds_ecfp4 = nds_df[nds_df['rep'] == 'ecfp4']
        if len(nds_ecfp4) > 0:
            pivot = nds_ecfp4.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
            col_order = [c for c in ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
                         if c in pivot.columns]
            pivot = pivot[col_order]
            pivot.index = [get_model_label(m) for m in pivot.index]
            vals_e = pivot.values[~np.isnan(pivot.values)]
            cv_e = (vals_e.min() + vals_e.max()) / 2 if len(vals_e) > 0 else 0
            ax_eb.set_facecolor('black')
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdBu', center=cv_e,
                        ax=ax_eb, cbar_kws={'label': 'NDS'})
            ax_eb.set_xlabel('Noise Strategy')
            ax_eb.set_ylabel('Model')
            ax_eb.set_title('B. NDS by Model × Strategy (ECFP4)', fontweight='bold')
            ax_eb.set_xticklabels([STRATEGY_LABELS.get(c, c) for c in col_order], rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_supp_ecfp4_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig1_supp_ecfp4_overview.png (supplementary — compare to PDV)")


# =============================================================================
# FIGURE 2: ANOVA DECOMPOSITION
# =============================================================================

def create_figure2(df, output_dir):
    """Figure 2: ANOVA variance decomposition.

    Issue E: Gaussian is primary. Other 4 concordant strategies noted via mean rho.
    Outlier flagged as different.
    """
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

    # Preferred strategy order: Gaussian first, then concordant 4, then outlier
    strat_order = ['legacy', 'valprop', 'quantile', 'threshold', 'hetero', 'outlier']

    def _plot_anova_panel(ax, results, title):
        strats = [s for s in strat_order if s in results]
        if not strats:
            return

        x = np.arange(len(strats))
        width = 0.25

        model_vals = [results[s]['eta2_model'] for s in strats]
        rep_vals = [results[s]['eta2_rep'] for s in strats]
        int_vals = [results[s]['eta2_interaction'] for s in strats]

        ax.bar(x - width, model_vals, width, label='Model', color='#2166AC')
        ax.bar(x, rep_vals, width, label='Representation', color='#B2182B')
        ax.bar(x + width, int_vals, width, label='Interaction', color='#7A3B9E')

        ax.set_ylabel('Variance Explained (η², %)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        labels = [STRATEGY_LABELS.get(s, s) for s in strats]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 100)

    _plot_anova_panel(axes[0], perf_results, 'A. Performance (R² at σ=0.3)')
    _plot_anova_panel(axes[1], robust_results, 'B. Robustness (NDS)')

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

    anova_table = pd.DataFrame(rows)
    anova_table.to_csv(output_dir / 'table1_anova_summary.csv', index=False)
    print("✓ Saved table1_anova_summary.csv")

    # Log ANOVA design decisions
    print(f"\n  ANOVA design:")
    print(f"    Models excluded: {sorted(ANOVA_MODELS_EXCLUDE)}")
    print(f"    Reps excluded: {sorted(ANOVA_REPS_EXCLUDE)}")
    # Report what's actually in the ANOVA
    all_models_in = sorted(set(df['model'].unique()) - ANOVA_MODELS_EXCLUDE
                           - {'graph_gp', 'gcn', 'gin', 'ginct', 'gin2d'})
    all_reps_in = sorted(set(df['rep'].unique()) - ANOVA_REPS_EXCLUDE)
    print(f"    Models in ANOVA: {all_models_in}")
    print(f"    Reps in ANOVA: {all_reps_in}")

    # Simple effects analysis (recommended when interaction η² > 30%)
    run_simple_effects_analysis(df, output_dir)


# =============================================================================
# FIGURE 3: RANKING CONSISTENCY
# =============================================================================

def create_figure3(nds_df, validation_df, output_dir):
    """Figure 3: Ranking consistency across strategies, sigmas, datasets. Uses PDV only."""
    n_panels = 3 if validation_df is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 6))
    if n_panels == 2:
        axes = [axes[0], axes[1], None]

    # Filter to PDV only for consistent comparison (don't mix representations)
    nds_pdv = nds_df[nds_df['rep'] == 'pdv'] if 'rep' in nds_df.columns else nds_df

    # Panel A: Heatmap - NDS by model × strategy (PDV only, ANOVA models)
    ax_a = axes[0]

    if len(nds_pdv) > 0:
        # Filter to ANOVA-included models for readability
        nds_pdv_anova = nds_pdv[~nds_pdv['model'].isin(ANOVA_MODELS_EXCLUDE)]
        pivot = nds_pdv_anova.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        # Only keep models with data for all strategies
        pivot = pivot.dropna()
        if len(pivot) > 0:
            strat_order = [c for c in ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
                          if c in pivot.columns]
            pivot = pivot[strat_order]
            pivot.index = [get_model_label(m) for m in pivot.index]
            pivot.columns = [STRATEGY_LABELS.get(s, s) for s in pivot.columns]
            # Sort by mean NDS (best at top)
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

            ax_a.set_facecolor('black')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu', vmax=0,
                        ax=ax_a, cbar_kws={'label': 'NDS'}, linewidths=0.5)
            ax_a.set_title('A. NDS by Model × Strategy (PDV)', fontweight='bold')
            ax_a.set_ylabel('')

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
    ax_b.set_title('B. Baseline vs Robustness (PDV, Gaussian)', fontweight='bold')
    ax_b.axhline(0, color='black', linewidth=0.5)
    ax_b.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), fontsize=5, ncol=4,
                borderaxespad=0, frameon=False)

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
        ax_c.set_title('C. Cross-Dataset (PDV, Gaussian)', fontweight='bold')
        ax_c.axhline(0, color='black', linewidth=0.5)
        ax_c.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), fontsize=5, ncol=3,
                    borderaxespad=0, frameon=False)

    for ax in axes:
        if ax is not None:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_ranking_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig3_ranking_consistency.png")

    # --- ECFP4 variant (Issue C) ---
    nds_ecfp4 = nds_df[nds_df['rep'] == 'ecfp4'] if 'rep' in nds_df.columns else nds_df
    if len(nds_ecfp4) > 0:
        fig_e, axes_e = plt.subplots(1, 2, figsize=(10, 5))

        # Heatmap across strategies
        ax_ea = axes_e[0]
        pivot = nds_ecfp4.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        if len(pivot) > 0:
            strat_list = [c for c in ['legacy', 'valprop', 'quantile', 'threshold', 'outlier', 'hetero']
                          if c in pivot.columns]
            pivot = pivot[strat_list].dropna()
            pivot.index = [get_model_label(m) for m in pivot.index]
            pivot.columns = [STRATEGY_LABELS.get(s, s) for s in pivot.columns]
            pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

            ax_ea.set_facecolor('black')
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdBu', vmax=0,
                        ax=ax_ea, cbar_kws={'label': 'NDS'}, linewidths=0.5)
            ax_ea.set_title('A. NDS by Model × Strategy (ECFP4)', fontweight='bold')
            ax_ea.set_ylabel('')

        # Baseline vs NDS scatter
        ax_eb = axes_e[1]
        nds_ecfp4_leg = nds_ecfp4[nds_ecfp4['strategy'] == 'legacy'] if 'strategy' in nds_ecfp4.columns else nds_ecfp4
        for model in nds_ecfp4_leg['model'].unique():
            md = nds_ecfp4_leg[nds_ecfp4_leg['model'] == model]
            color = MODEL_COLORS.get(model, '#333333')
            ax_eb.scatter(md['baseline_r2'], md['nds'], label=get_model_label(model),
                          color=color, alpha=0.7, s=50)
        ax_eb.set_xlabel('Baseline R² (σ=0)')
        ax_eb.set_ylabel('NDS (slope)')
        ax_eb.set_title('B. Baseline vs Robustness (ECFP4, Gaussian)', fontweight='bold')
        ax_eb.axhline(0, color='black', linewidth=0.5)
        ax_eb.legend(loc='upper left', bbox_to_anchor=(0.0, -0.15), fontsize=5, ncol=4,
                     borderaxespad=0, frameon=False)

        for ax in axes_e:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig(output_dir / 'fig3_supp_ecfp4_ranking.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved fig3_supp_ecfp4_ranking.png (supplementary)")


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
    dnn_variants = ['dnn', 'dnn_bnn_full', 'dnn_bnn_last']
    # dnn_bnn_variational excluded pending VBLL re-implementation

    # 1x2 layout: R² vs σ for PRIMARY_STRATEGY and CONTRAST_STRATEGY
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, strategy in enumerate([PRIMARY_STRATEGY, CONTRAST_STRATEGY]):
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)

        ax_line = axes[col]
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
        panel_letter = 'A' if col == 0 else 'B'
        ax_line.set_title(f'{panel_letter}. R² vs σ ({strategy_label})', fontweight='bold')
        ax_line.legend(loc='lower left', fontsize=7)
        ax_line.set_ylim(-0.1, 1.0)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig4_dnn_family.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fig4_dnn_family.png (showing {PRIMARY_STRATEGY} vs {CONTRAST_STRATEGY})")

    # Output NDS table for DNN family (replaces bar chart panels)
    dnn_nds_data = nds_df[(nds_df['model'].isin(dnn_variants)) & (nds_df['rep'] == PRIMARY_REP)]
    if len(dnn_nds_data) > 0:
        dnn_table = dnn_nds_data.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        dnn_table.index = [get_model_label(m) for m in dnn_table.index]
        dnn_table.columns = [STRATEGY_LABELS.get(s, s) for s in dnn_table.columns]
        dnn_table.to_csv(output_dir / 'table_fig4_dnn_nds.csv')
        print("✓ Saved table_fig4_dnn_nds.csv")

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
    mlp_variants = ['mlp', 'mlp_bnn_full', 'mlp_bnn_last']
    # mlp_bnn_variational excluded pending VBLL re-implementation
    rf_models = ['rf', 'qrf']

    # 1x2 layout: MLP R² vs σ for PRIMARY_STRATEGY and CONTRAST_STRATEGY
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for col, strategy in enumerate([PRIMARY_STRATEGY, CONTRAST_STRATEGY]):
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)
        data = df[(df['rep'] == PRIMARY_REP) & (df['strategy'] == strategy)]

        ax_line = axes[col]

        for model in mlp_variants:
            model_data = data[data['model'] == model]
            if len(model_data) == 0:
                continue

            avg = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = MODEL_COLORS.get(model, '#333333')
            ax_line.plot(avg['sigma'], avg['r2'], 'o-', label=get_model_label(model), color=color, markersize=4)

        ax_line.set_xlabel('Noise Level (σ)')
        ax_line.set_ylabel('R²')
        panel_letter = 'A' if col == 0 else 'B'
        ax_line.set_title(f'{panel_letter}. MLP R² vs σ ({strategy_label})', fontweight='bold')
        ax_line.legend(loc='lower left', fontsize=6)
        ax_line.set_ylim(-0.1, 1.0)

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig5_mlp_rf_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fig5_mlp_rf_comparison.png (showing {PRIMARY_STRATEGY} vs {CONTRAST_STRATEGY})")

    # Output NDS tables for MLP family and RF/QRF (replaces bar chart panels)
    mlp_nds_data = nds_df[(nds_df['model'].isin(mlp_variants)) & (nds_df['rep'] == PRIMARY_REP)]
    if len(mlp_nds_data) > 0:
        mlp_table = mlp_nds_data.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        mlp_table.index = [get_model_label(m) for m in mlp_table.index]
        mlp_table.columns = [STRATEGY_LABELS.get(s, s) for s in mlp_table.columns]
        mlp_table.to_csv(output_dir / 'table_fig5_mlp_nds.csv')
        print("✓ Saved table_fig5_mlp_nds.csv")

    rf_nds_data = nds_df[(nds_df['model'].isin(rf_models)) & (nds_df['rep'] == PRIMARY_REP)]
    if len(rf_nds_data) > 0:
        rf_table = rf_nds_data.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        rf_table.index = [get_model_label(m) for m in rf_table.index]
        rf_table.columns = [STRATEGY_LABELS.get(s, s) for s in rf_table.columns]
        rf_table.to_csv(output_dir / 'table_fig5_rf_qrf_nds.csv')
        print("✓ Saved table_fig5_rf_qrf_nds.csv")

    # REVIEW: Does the probabilistic advantage hold under both noise types?


# =============================================================================
# FIGURE 6: UNCERTAINTY QUALITY
# =============================================================================


def _create_combined_uncertainty_figure(unc_df, output_path, strategy, rep, title_suffix=""):
    """
    Uncertainty figure — single panel: mean uncertainty vs noise level.
    Calibration metrics (ECE, coverage) and aleatoric/epistemic decomposition
    are reported in table4 CSVs instead of as figure panels.
    """
    if unc_df is None or len(unc_df) == 0:
        return False

    filtered = unc_df.copy()
    if 'strategy' in filtered.columns and strategy:
        filtered = filtered[filtered['strategy'] == strategy]
    if 'rep' in filtered.columns and rep:
        filtered = filtered[filtered['rep'] == rep]

    if len(filtered) == 0 or 'sigma' not in filtered.columns:
        return False

    unc_col = None
    for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty', 'std']:
        if col in filtered.columns:
            unc_col = col
            break
    if unc_col is None:
        return False

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    # Pre-filter: exclude models with negligible uncertainty (e.g. plain DNN/MLP)
    valid_models = []
    for model in filtered['model'].unique():
        mdata = filtered[filtered['model'] == model]
        uvals = mdata[unc_col].values
        finite_mask = np.isfinite(uvals)
        if finite_mask.sum() > 100 and uvals[finite_mask].mean() > 1e-3:
            valid_models.append(model)
    filtered = filtered[filtered['model'].isin(valid_models)]

    for model in sorted(filtered['model'].unique()):
        model_data = filtered[filtered['model'] == model]
        unc_values = model_data[unc_col].values
        if np.sum(np.isfinite(unc_values) & (unc_values > 0)) < 100:
            continue

        sigma_means = []
        for sigma in sorted(model_data['sigma'].unique()):
            sigma_data = model_data[model_data['sigma'] == sigma]
            vals = sigma_data[unc_col].values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                sigma_means.append({'sigma': sigma, 'mean_unc': vals.mean()})

        if sigma_means:
            sigma_df = pd.DataFrame(sigma_means)
            color = MODEL_COLORS.get(model, '#333333')
            ax.plot(sigma_df['sigma'], sigma_df['mean_unc'], 'o-',
                    label=get_model_label(model), color=color,
                    markersize=4, linewidth=1.2, alpha=0.8)

    ax.set_xlabel('Injected Noise Level (σ)')
    ax.set_ylabel('Mean Predicted Uncertainty')
    ax.set_title('Uncertainty Tracking of Injected Noise', fontweight='bold')
    ax.legend(fontsize=7, ncol=2, loc='upper left', framealpha=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return True


def create_uncertainty_figure(unc_df, output_dir):
    """
    Uncertainty figure: single-panel mean uncertainty vs noise level.
    Calibration metrics and aleatoric/epistemic are in table4 CSVs.
    """
    if unc_df is None or len(unc_df) == 0:
        print("⚠ Skipping uncertainty figure - no data")
        return

    # Main figure: PRIMARY_REP + PRIMARY_STRATEGY
    success = _create_combined_uncertainty_figure(
        unc_df,
        output_dir / 'fig_uncertainty_combined.png',
        strategy=PRIMARY_STRATEGY,
        rep=PRIMARY_REP,
    )
    if success:
        print(f"✓ Saved fig_uncertainty_combined.png ({PRIMARY_REP}, {PRIMARY_STRATEGY})")
    else:
        print("⚠ Could not create combined uncertainty figure")

    # Supplementary uncertainty figures removed — calibration, ECE, coverage,
    # aleatoric/epistemic, and unc-error/unc-noise correlations are all in
    # table4_uncertainty_metrics.csv and table4_supp_uncertainty_by_strategy_rep.csv


# NOTE: Old _create_uncertainty_noise_figure, create_figure7, and 3-panel
# uncertainty figures (calibration scatter, aleatoric/epistemic) removed.
# ECE, coverage, correlations, and aleatoric/epistemic decomposition
# are now in table4_uncertainty_metrics.csv and table4_supp CSVs.
# Only the uncertainty-vs-noise-level line plot remains as a figure.


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
            pivot['STD'] = pivot.drop(columns=['MEAN']).std(axis=1)
            pivot = pivot.sort_values('MEAN', ascending=False)
            pivot.rename(columns=STRATEGY_LABELS, inplace=True)
            pivot.to_csv(output_dir / 'table2_nds_by_strategy_pdv.csv')
            print("✓ Saved table2_nds_by_strategy_pdv.csv")

        # Also save full table with all reps for supplementary
        pivot_all = nds_df.pivot_table(values='nds', index=['model', 'rep'], columns='strategy', aggfunc='mean')
        pivot_all['MEAN'] = pivot_all.mean(axis=1)
        pivot_all['STD'] = pivot_all.drop(columns=['MEAN']).std(axis=1)
        pivot_all = pivot_all.sort_values('MEAN', ascending=False)
        pivot_all.rename(columns=STRATEGY_LABELS, inplace=True)
        pivot_all.to_csv(output_dir / 'table2_supp_nds_all_reps.csv')
        print("✓ Saved table2_supp_nds_all_reps.csv (supplementary)")

    # Table 3: Probabilistic comparison with Wilcoxon tests (PDV + legacy)
    prob_comparisons = {
        'DNN Family': {'base': 'dnn', 'variants': ['dnn_bnn_full', 'dnn_bnn_last']},
        'MLP Family': {'base': 'mlp', 'variants': ['mlp_bnn_full', 'mlp_bnn_last']},
        'RF Family': {'base': 'rf', 'variants': ['qrf']},
        # Note: dnn_bnn_variational and mlp_bnn_variational excluded pending VBLL re-run
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
                # Use y_true_noisy (normalized space) to match y_pred_mean (normalized space)
                if 'y_true_noisy' in model_data.columns and 'y_pred_mean' in model_data.columns:
                    errors = np.abs(model_data['y_pred_mean'].values - model_data['y_true_noisy'].values)
                elif 'y_true_original' in model_data.columns and 'y_pred_mean' in model_data.columns:
                    errors = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
                elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                    errors = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
                else:
                    continue

                mask = np.isfinite(unc_values) & np.isfinite(errors)
                if mask.sum() < 100:
                    continue

                # Skip models with negligible uncertainty (e.g. plain DNN/MLP
                # that don't produce meaningful uncertainty estimates)
                mean_unc = unc_values[mask].mean()
                if mean_unc < 1e-3:
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
                # IMPORTANT: Use y_true_noisy (normalized space) with y_pred_mean (normalized space)
                # NOT y_true_original (original scale) — that causes a scale mismatch
                # giving near-zero coverage for all models.
                if 'y_true_noisy' in model_data.columns and 'y_pred_mean' in model_data.columns:
                    y_true = model_data['y_true_noisy'].values[mask]
                    y_pred = model_data['y_pred_mean'].values[mask]
                elif 'y_true_original' in model_data.columns and 'y_pred_mean' in model_data.columns:
                    # Fallback — may have scale mismatch if data is normalized
                    y_true = model_data['y_true_original'].values[mask]
                    y_pred = model_data['y_pred_mean'].values[mask]
                elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                    y_true = model_data['y_true'].values[mask]
                    y_pred = model_data['y_pred'].values[mask]
                else:
                    y_true = y_pred = None

                if y_true is not None:
                    is_conformal = 'conformal' in model
                    if is_conformal:
                        # For conformal models, uncertainty is pseudo-std derived from
                        # interval_width / (2*1.645). Recover the actual half-width
                        # to compute coverage properly against the interval bounds.
                        half_width = unc_values[mask] * 1.645  # undo the /1.645
                        cov_nominal = calculate_coverage(y_true, y_pred, half_width, k=1)
                        # Also compute at 2x the interval for comparison
                        cov_2x = calculate_coverage(y_true, y_pred, half_width, k=2)
                        cov_1sigma = cov_nominal
                        cov_2sigma = cov_2x
                    else:
                        cov_1sigma = calculate_coverage(y_true, y_pred, unc_values[mask], k=1)
                        cov_2sigma = calculate_coverage(y_true, y_pred, unc_values[mask], k=2)
                else:
                    cov_1sigma = cov_2sigma = np.nan

                # ECE: binned calibration error
                pred_pos_mask = mask & (unc_values > 0)
                pred_m = unc_values[pred_pos_mask]
                actual_m = errors[pred_pos_mask]
                ece = np.nan
                if len(pred_m) >= 100:
                    ece_bins = np.percentile(pred_m, np.linspace(0, 100, 11))
                    ece_bins = np.unique(ece_bins)
                    ece = 0
                    for i in range(len(ece_bins) - 1):
                        bin_mask = (pred_m >= ece_bins[i]) & (pred_m < ece_bins[i + 1])
                        if bin_mask.sum() > 0:
                            bin_pred = pred_m[bin_mask].mean()
                            bin_actual = actual_m[bin_mask].mean()
                            bin_weight = bin_mask.sum() / len(pred_m)
                            ece += bin_weight * np.abs(bin_pred - bin_actual)

                # Aleatoric / epistemic decomposition
                mean_alea = mean_epis = np.nan
                if 'aleatoric_uncertainty' in model_data.columns:
                    alea = model_data['aleatoric_uncertainty'].values
                    alea_valid = alea[np.isfinite(alea) & (alea > 0)]
                    if len(alea_valid) > 10:
                        mean_alea = alea_valid.mean()
                if 'epistemic_uncertainty' in model_data.columns:
                    epis = model_data['epistemic_uncertainty'].values
                    epis_valid = epis[np.isfinite(epis) & (epis > 0)]
                    if len(epis_valid) > 10:
                        mean_epis = epis_valid.mean()

                unc_metrics.append({
                    'Model': model,
                    'Unc-Error ρ': unc_err_corr,
                    'Unc-Noise ρ': unc_noise_corr,
                    'ECE': ece,
                    'Coverage 1σ': cov_1sigma,
                    'Coverage 2σ': cov_2sigma,
                    'Mean Uncertainty': unc_values[mask].mean(),
                    'Mean Aleatoric': mean_alea,
                    'Mean Epistemic': mean_epis,
                })

            if unc_metrics:
                unc_metrics_df = pd.DataFrame(unc_metrics).sort_values('Unc-Error ρ', ascending=False)
                # Add clean labels
                unc_metrics_df['Model'] = unc_metrics_df['Model'].map(
                    lambda m: get_model_label(m))
                unc_metrics_df.to_csv(output_dir / 'table4_uncertainty_metrics.csv', index=False)
                print("✓ Saved table4_uncertainty_metrics.csv")

    # Table 4b: Uncertainty metrics across ALL strategies and reps
    # Answers: do uncertainty patterns hold across noise types and representations?
    if unc_df is not None and len(unc_df) > 0:
        unc_col = None
        for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty']:
            if col in unc_df.columns:
                unc_col = col
                break

        if unc_col:
            all_unc_rows = []
            strategies = unc_df['strategy'].unique() if 'strategy' in unc_df.columns else ['all']
            reps = unc_df['rep'].unique() if 'rep' in unc_df.columns else ['all']

            for strategy in strategies:
                for rep in reps:
                    subset = unc_df.copy()
                    if 'strategy' in subset.columns:
                        subset = subset[subset['strategy'] == strategy]
                    if 'rep' in subset.columns:
                        subset = subset[subset['rep'] == rep]
                    if len(subset) == 0:
                        continue

                    for model in subset['model'].unique():
                        model_data = subset[subset['model'] == model]
                        unc_values = model_data[unc_col].values
                        mask = np.isfinite(unc_values)
                        if mask.sum() < 100 or unc_values[mask].mean() < 1e-3:
                            continue

                        # Error computation
                        if 'y_true_noisy' in model_data.columns and 'y_pred_mean' in model_data.columns:
                            errors = np.abs(model_data['y_pred_mean'].values - model_data['y_true_noisy'].values)
                        elif 'y_true_original' in model_data.columns and 'y_pred_mean' in model_data.columns:
                            errors = np.abs(model_data['y_pred_mean'].values - model_data['y_true_original'].values)
                        elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                            errors = np.abs(model_data['y_pred'].values - model_data['y_true'].values)
                        else:
                            continue

                        valid = mask & np.isfinite(errors)
                        if valid.sum() < 100:
                            continue

                        unc_err_corr, _ = stats.spearmanr(unc_values[valid], errors[valid])

                        unc_noise_corr = np.nan
                        if 'injected_noise' in model_data.columns:
                            noise_mag = np.abs(model_data['injected_noise'].values)
                            noise_mask = valid & np.isfinite(noise_mag)
                            if noise_mask.sum() > 100:
                                unc_noise_corr, _ = stats.spearmanr(unc_values[noise_mask], noise_mag[noise_mask])

                        # ECE: binned calibration error
                        pred_m = unc_values[valid]
                        actual_m = errors[valid]
                        ece = np.nan
                        if len(pred_m) >= 100:
                            ece_bins = np.percentile(pred_m, np.linspace(0, 100, 11))
                            ece_bins = np.unique(ece_bins)
                            ece = 0
                            for i in range(len(ece_bins) - 1):
                                bin_mask = (pred_m >= ece_bins[i]) & (pred_m < ece_bins[i + 1])
                                if bin_mask.sum() > 0:
                                    bin_pred = pred_m[bin_mask].mean()
                                    bin_actual = actual_m[bin_mask].mean()
                                    bin_weight = bin_mask.sum() / len(pred_m)
                                    ece += bin_weight * np.abs(bin_pred - bin_actual)

                        # Coverage
                        y_true_col = y_pred_col = None
                        if 'y_true_noisy' in model_data.columns and 'y_pred_mean' in model_data.columns:
                            y_true_col, y_pred_col = 'y_true_noisy', 'y_pred_mean'
                        elif 'y_true_original' in model_data.columns and 'y_pred_mean' in model_data.columns:
                            y_true_col, y_pred_col = 'y_true_original', 'y_pred_mean'
                        elif 'y_true' in model_data.columns and 'y_pred' in model_data.columns:
                            y_true_col, y_pred_col = 'y_true', 'y_pred'

                        cov_1sigma = cov_2sigma = np.nan
                        if y_true_col:
                            yt = model_data[y_true_col].values[valid]
                            yp = model_data[y_pred_col].values[valid]
                            cov_1sigma = calculate_coverage(yt, yp, unc_values[valid], k=1)
                            cov_2sigma = calculate_coverage(yt, yp, unc_values[valid], k=2)

                        # Aleatoric / epistemic decomposition
                        mean_alea = mean_epis = np.nan
                        if 'aleatoric_uncertainty' in model_data.columns:
                            alea = model_data['aleatoric_uncertainty'].values
                            alea_valid = alea[np.isfinite(alea) & (alea > 0)]
                            if len(alea_valid) > 10:
                                mean_alea = alea_valid.mean()
                        if 'epistemic_uncertainty' in model_data.columns:
                            epis = model_data['epistemic_uncertainty'].values
                            epis_valid = epis[np.isfinite(epis) & (epis > 0)]
                            if len(epis_valid) > 10:
                                mean_epis = epis_valid.mean()

                        all_unc_rows.append({
                            'Strategy': STRATEGY_LABELS.get(strategy, strategy),
                            'Rep': rep,
                            'Model': get_model_label(model),
                            'Unc-Error ρ': unc_err_corr,
                            'Unc-Noise ρ': unc_noise_corr,
                            'ECE': ece,
                            'Coverage 1σ': cov_1sigma,
                            'Coverage 2σ': cov_2sigma,
                            'Mean Uncertainty': unc_values[valid].mean(),
                            'Mean Aleatoric': mean_alea,
                            'Mean Epistemic': mean_epis,
                        })

            if all_unc_rows:
                unc_full_df = pd.DataFrame(all_unc_rows)
                unc_full_df.to_csv(output_dir / 'table4_supp_uncertainty_by_strategy_rep.csv', index=False)
                print(f"✓ Saved table4_supp_uncertainty_by_strategy_rep.csv ({len(unc_full_df)} rows)")

                # Summary: which model has best unc-error correlation across strategies?
                mean_by_model = unc_full_df.groupby('Model')['Unc-Error ρ'].mean().sort_values(ascending=False)
                print("  Uncertainty-Error ρ (mean across strategies/reps):")
                for model, rho in mean_by_model.head(5).items():
                    print(f"    {model}: {rho:.3f}")

                # Are BNNs better on any strategy?
                bnn_rows = unc_full_df[unc_full_df['Model'].str.contains('BNN', case=False)]
                if len(bnn_rows) > 0:
                    bnn_by_strat = bnn_rows.groupby('Strategy')['Unc-Error ρ'].mean()
                    print("  BNN Unc-Error ρ by strategy:")
                    for strat, rho in bnn_by_strat.items():
                        print(f"    {strat}: {rho:.3f}")

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
            models = nds_df['model'].unique()

            # First pass: find models present in ALL strategies
            model_nds_by_strategy = {}
            for strategy in strategies:
                strat_data = nds_df[nds_df['strategy'] == strategy]
                model_nds_by_strategy[strategy] = strat_data.groupby('model')['nds'].mean()

            valid_models = [m for m in models
                           if all(m in model_nds_by_strategy[s].index for s in strategies)]

            if len(valid_models) > 2:
                # Second pass: rank only valid models within each strategy
                rank_matrix = []
                for strategy in strategies:
                    nds_vals = model_nds_by_strategy[strategy].loc[valid_models]
                    ranks = nds_vals.rank(ascending=False)  # Higher NDS = more robust = rank 1
                    rank_matrix.append(ranks.values)

                rank_matrix = np.array(rank_matrix)  # shape: (n_strategies, n_models)

                # Calculate Kendall's W = 12 * S / (k² * (n³ - n))
                n_raters = rank_matrix.shape[0]  # strategies
                n_items = rank_matrix.shape[1]   # models

                rank_sums = rank_matrix.sum(axis=0)
                mean_rank_sum = rank_sums.mean()
                ss_between = np.sum((rank_sums - mean_rank_sum) ** 2)

                max_ss = (n_raters ** 2) * (n_items ** 3 - n_items) / 12
                kendalls_w = ss_between / max_ss if max_ss > 0 else 0

                # Chi-squared significance test
                chi2 = n_raters * (n_items - 1) * kendalls_w
                p_value = 1 - stats.chi2.cdf(chi2, n_items - 1)

                # Save summary
                with open(output_dir / 'table6_kendalls_w.txt', 'w') as f:
                    f.write("Kendall's W Concordance Analysis\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Number of raters (strategies): {n_raters}\n")
                    f.write(f"Number of items (models): {n_items}\n")
                    f.write(f"Kendall's W: {kendalls_w:.4f}\n")
                    f.write(f"Chi-squared: {chi2:.2f}\n")
                    f.write(f"p-value: {p_value:.2e}\n\n")
                    f.write(f"Models included: {', '.join(sorted(valid_models))}\n\n")
                    f.write("Interpretation:\n")
                    f.write("  W > 0.7: Strong agreement\n")
                    f.write("  W 0.5-0.7: Moderate agreement\n")
                    f.write("  W < 0.5: Weak agreement\n")
                print(f"✓ Saved table6_kendalls_w.txt (W={kendalls_w:.4f}, n={n_items} models)")

    # Table 7: Strategy Sensitivity Ratio (NDS_strategy / NDS_legacy)
    # Tests whether certain noise types differentially affect model families
    if len(nds_df) > 0 and 'legacy' in nds_df['strategy'].values:
        # Compute mean NDS per model per strategy on PRIMARY_REP
        nds_primary = nds_df[nds_df['rep'] == PRIMARY_REP] if 'rep' in nds_df.columns else nds_df
        pivot = nds_primary.groupby(['model', 'strategy'])['nds'].mean().unstack('strategy')

        if 'legacy' in pivot.columns:
            ratio_df = pivot.div(pivot['legacy'], axis=0)
            # Add model family classification
            tree_models = ['rf', 'xgboost', 'qrf', 'ngboost', 'lgb',
                           'conformal_rf_split', 'conformal_qrf_split']
            nn_models = ['dnn', 'mlp', 'flexible_dnn',
                         'flexible_dnn_256_128_64', 'flexible_dnn_512_256',
                         'dnn_bnn_full', 'dnn_bnn_last', 'dnn_bnn_variational',
                         'mlp_bnn_full', 'mlp_bnn_last', 'mlp_bnn_variational',
                         'conformal_dnn_split']

            ratio_df['family'] = 'other'
            ratio_df.loc[ratio_df.index.isin(tree_models), 'family'] = 'tree'
            ratio_df.loc[ratio_df.index.isin(nn_models), 'family'] = 'nn'

            ratio_df.rename(columns=STRATEGY_LABELS, inplace=True)
            ratio_df.to_csv(output_dir / 'table7_strategy_sensitivity_ratio.csv')
            print("✓ Saved table7_strategy_sensitivity_ratio.csv")

            # Summary by family
            strategy_display = {STRATEGY_LABELS.get(s, s) for s in ALL_STRATEGIES if s != 'legacy'}
            for family in ['tree', 'nn']:
                fam_data = ratio_df[ratio_df['family'] == family]
                if len(fam_data) > 0:
                    for strat in [s for s in fam_data.columns if s in strategy_display]:
                        vals = fam_data[strat].dropna()
                        if len(vals) > 0:
                            print(f"  {family} {strat}/Gaussian ratio: "
                                  f"mean={vals.mean():.3f} (range {vals.min():.3f}-{vals.max():.3f})")


# =============================================================================
# INTERACTION FIGURE (Issue D)
# =============================================================================

def create_interaction_figure(nds_df, output_dir):
    """Visualize the model x representation interaction effect.

    Panel A: Heatmap — NDS by model × representation (Gaussian strategy).
    Panel B: Scatter — NDS on ECFP4 vs NDS on PDV per model.

    BNN variants use different marker shapes but same color as base model.
    """
    if len(nds_df) == 0:
        print("⚠ Could not create interaction figure - no NDS data")
        return

    nds_legacy = nds_df[nds_df['strategy'] == 'legacy'] if 'strategy' in nds_df.columns else nds_df

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Heatmap — NDS by model × representation (Gaussian strategy)
    ax_a = axes[0]

    # Get mean NDS per model x rep (Gaussian strategy)
    pivot = nds_legacy.pivot_table(values='nds', index='model', columns='rep', aggfunc='mean')

    # Include all reps with enough models
    valid_reps = [r for r in pivot.columns if pivot[r].notna().sum() >= 3]
    rep_order = [r for r in ['ecfp4', 'pdv', 'smiles', 'mol2vec', 'mhggnn'] if r in valid_reps]
    rep_order += [r for r in valid_reps if r not in rep_order]

    if len(rep_order) >= 2:
        hm_pivot = pivot[rep_order].dropna(how='all')
        hm_pivot.index = [get_model_label(m) for m in hm_pivot.index]
        hm_pivot.columns = [get_rep_label(r) for r in hm_pivot.columns]
        # Sort by mean NDS (best at top)
        hm_pivot = hm_pivot.loc[hm_pivot.mean(axis=1).sort_values(ascending=False).index]

        ax_a.set_facecolor('black')
        sns.heatmap(hm_pivot, annot=True, fmt='.3f', cmap='RdBu', vmax=0,
                    ax=ax_a, cbar_kws={'label': 'NDS'}, linewidths=0.5)
        ax_a.set_title('A. Model × Rep Interaction (Gaussian NDS)', fontweight='bold')
        ax_a.set_ylabel('')

    # Panel B: Scatter — NDS on ECFP4 vs NDS on PDV, with legend (not annotations)
    ax_b = axes[1]

    # BNN marker mapping: different shape, same color as base
    BNN_BASE_MAP = {
        'dnn_bnn_full': 'dnn', 'dnn_bnn_last': 'dnn',
        'mlp_bnn_full': 'mlp', 'mlp_bnn_last': 'mlp',
    }
    BNN_MARKERS = {
        'dnn_bnn_full': 's', 'dnn_bnn_last': '^',
        'mlp_bnn_full': 's', 'mlp_bnn_last': '^',
    }

    ecfp4_nds = nds_legacy[nds_legacy['rep'] == 'ecfp4'].groupby('model')['nds'].mean()
    pdv_nds = nds_legacy[nds_legacy['rep'] == 'pdv'].groupby('model')['nds'].mean()

    shared_models = sorted(set(ecfp4_nds.index) & set(pdv_nds.index))

    if len(shared_models) >= 3:
        for m in shared_models:
            ev, pv = ecfp4_nds[m], pdv_nds[m]
            base_model = BNN_BASE_MAP.get(m, m)
            color = MODEL_COLORS.get(base_model, MODEL_COLORS.get(m, '#333333'))
            marker = BNN_MARKERS.get(m, 'o')
            ax_b.scatter(ev, pv, color=color, marker=marker, s=60, zorder=3,
                         label=get_model_label(m), edgecolors='white', linewidths=0.3)

        # Compute and annotate rho
        e_vals = [ecfp4_nds[m] for m in shared_models]
        p_vals = [pdv_nds[m] for m in shared_models]
        rho, p = stats.spearmanr(e_vals, p_vals)
        ax_b.text(0.05, 0.95, f'Spearman ρ = {rho:.2f} (p = {p:.3f})',
                  transform=ax_b.transAxes, fontsize=8, va='top',
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Identity line
        all_vals = e_vals + p_vals
        lim = [min(all_vals) - 0.02, max(all_vals) + 0.02]
        ax_b.plot(lim, lim, '--', color='gray', alpha=0.5, linewidth=0.8)

        # Legend outside plot area
        ax_b.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), fontsize=6,
                    borderaxespad=0, frameon=True, fancybox=False)

    ax_b.set_xlabel('NDS on ECFP4 (Gaussian)')
    ax_b.set_ylabel('NDS on PDV (Gaussian)')
    ax_b.set_title('B. ECFP4 vs PDV Robustness', fontweight='bold')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_interaction.png")


# =============================================================================
# FULL EXPERIMENT OVERVIEW (all configs, including non-ANOVA)
# =============================================================================

def _plot_full_overview_panels(nds_df, strategies, output_dir, filename, panel_labels=None):
    """Helper: scatter plot of ANOVA model x rep configurations for given strategies.

    Each point is one model x rep configuration; color = model, shape = rep.
    Filtered to ANOVA-included models/reps for readability.
    """
    strategies_available = [s for s in strategies if s in nds_df['strategy'].unique()]
    if not strategies_available:
        return

    # Filter to ANOVA-included models and reps for readability
    anova_nds = nds_df[~nds_df['model'].isin(ANOVA_MODELS_EXCLUDE) &
                       ~nds_df['rep'].isin(ANOVA_REPS_EXCLUDE)]
    # Exclude graph models
    graph_models = {'graph_gp', 'gcn', 'gin', 'ginct', 'gin2d'}
    anova_nds = anova_nds[~anova_nds['model'].isin(graph_models)]

    n_panels = len(strategies_available)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    rep_markers = {'ecfp4': 'o', 'pdv': 's', 'smiles': '^',
                   'mhggnn': 'v'}

    last_mean_nds = None
    for idx, strategy in enumerate(strategies_available):
        ax = axes[idx]
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)
        label_prefix = f'{panel_labels[idx]}. ' if panel_labels and idx < len(panel_labels) else ''

        strat_data = anova_nds[anova_nds['strategy'] == strategy]
        mean_nds = strat_data.groupby(['model', 'rep']).agg(
            nds_mean=('nds', 'mean'),
            baseline_r2=('baseline_r2', 'mean')
        ).reset_index()
        last_mean_nds = mean_nds

        for _, row in mean_nds.iterrows():
            color = MODEL_COLORS.get(row['model'], '#333333')
            marker = rep_markers.get(row['rep'], 'o')
            ax.scatter(row['baseline_r2'], row['nds_mean'],
                       color=color, marker=marker, s=50, alpha=0.7,
                       edgecolors='black', linewidths=0.5)

        ax.set_xlabel('Baseline R² (σ=0)')
        if idx == 0:
            ax.set_ylabel('NDS')
        else:
            ax.set_ylabel('')
        ax.set_title(f'{label_prefix}{strategy_label}', fontweight='bold')
        ax.axhline(0, color='black', linewidth=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Dual legend: rep shapes (first panel) + model colors (last panel)
    if last_mean_nds is not None:
        from matplotlib.lines import Line2D

        # Rep legend on first panel
        rep_handles = [Line2D([0], [0], marker=rep_markers.get(r, 'o'), color='gray',
                              linestyle='None', markersize=6,
                              label=REP_LABELS.get(r, r))
                       for r in ['ecfp4', 'pdv', 'smiles', 'mhggnn']
                       if r in last_mean_nds['rep'].unique()]
        axes[0].legend(handles=rep_handles, loc='lower left', fontsize=5,
                       title='Rep', title_fontsize=6)

        # Model color legend on last panel (only models present in data)
        models_present = sorted(last_mean_nds['model'].unique())
        model_handles = [Line2D([0], [0], marker='o', color=MODEL_COLORS.get(m, '#333333'),
                                linestyle='None', markersize=5,
                                label=get_model_label(m))
                         for m in models_present]
        axes[-1].legend(handles=model_handles, loc='lower right', fontsize=4,
                        title='Model', title_fontsize=5, ncol=2)

    fig.suptitle('Baseline Performance vs Noise Robustness (ANOVA Configurations)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {filename}")


def create_full_overview(nds_df, output_dir):
    """Scatter plots of ALL model x rep configurations.

    Main figure: one panel per severity tier (Gaussian, Outlier, Threshold).
    Supplementary: remaining strategies (Quantile, Heteroscedastic, Value-prop).
    """
    if len(nds_df) == 0:
        print("⚠ Could not create full overview - no NDS data")
        return

    # Main figure: one strategy per tier (moderate, mild, severe)
    _plot_full_overview_panels(
        nds_df,
        strategies=['legacy', 'outlier', 'threshold'],
        output_dir=output_dir,
        filename='fig_full_overview.png',
        panel_labels=['A', 'B', 'C']
    )

    # Supplementary: remaining strategies
    _plot_full_overview_panels(
        nds_df,
        strategies=['quantile', 'hetero', 'valprop'],
        output_dir=output_dir,
        filename='fig_full_overview_supp.png',
        panel_labels=['A', 'B', 'C']
    )

    # Also save the underlying data as a table
    all_configs = nds_df.groupby(['model', 'rep', 'strategy']).agg(
        nds_mean=('nds', 'mean'),
        nds_std=('nds', 'std'),
        baseline_r2=('baseline_r2', 'mean'),
        n_iterations=('nds', 'count')
    ).reset_index()
    all_configs['in_anova'] = (~all_configs['model'].isin(ANOVA_MODELS_EXCLUDE) &
                                ~all_configs['rep'].isin(ANOVA_REPS_EXCLUDE))
    all_configs.to_csv(output_dir / 'table_all_configurations.csv', index=False)
    print(f"✓ Saved table_all_configurations.csv ({len(all_configs)} rows)")


# =============================================================================
# REPORT
# =============================================================================

def generate_report(nds_df, excluded_df, output_dir):
    """Generate text report summarizing findings."""
    lines = []
    lines.append("=" * 80)
    lines.append("PAPER FIGURES GENERATION REPORT")
    lines.append("=" * 80)

    lines.append(f"\nCatastrophic iteration threshold: R² < {CATASTROPHIC_R2_THRESHOLD}")
    catastrophic_path = output_dir / 'filtered_catastrophic_iterations.csv'
    if catastrophic_path.exists():
        cat_df = pd.read_csv(catastrophic_path)
        lines.append(f"Catastrophic iterations filtered: {len(cat_df)} rows removed before NDS calculation")
        for _, row in cat_df.iterrows():
            lines.append(f"  {row['model']}/{row['rep']}/{row['strategy']} "
                         f"σ={row['sigma']:.1f} iter={row['iteration']} R²={row['r2']:.2f}")
    else:
        lines.append("Catastrophic iterations filtered: 0")

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
        lines.append("KEY FINDINGS (PDV representation, Gaussian strategy)")
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
    if unc_df is not None:
        unc_df = fix_injected_noise(unc_df)
    validation_df = load_validation_data(args.validation_dir)

    if qm9_df is None:
        print("ERROR: No QM9 ANOVA data found!")
        return

    # Exclude graph models (Graph_GP, GCN, GIN) - not in scope
    EXCLUDED_MODELS = {'graph_gp', 'gcn', 'gin', 'ginct', 'gin2d'}
    # Also exclude old var-BNN data (incorrect implementation, identical to last-layer)
    EXCLUDED_MODELS = EXCLUDED_MODELS | VBLL_PENDING_EXCLUDE
    pre_filter = len(qm9_df)
    qm9_df = qm9_df[~qm9_df['model'].isin(EXCLUDED_MODELS)]
    if len(qm9_df) < pre_filter:
        print(f"  Filtered out {pre_filter - len(qm9_df)} rows from excluded models (graph + old var-BNN)")
    if unc_df is not None:
        pre_filter_unc = len(unc_df)
        unc_df = unc_df[~unc_df['model'].isin(EXCLUDED_MODELS)]
        if len(unc_df) < pre_filter_unc:
            print(f"  Filtered out {pre_filter_unc - len(unc_df)} uncertainty rows from excluded models")

    # Filter catastrophic training iterations (e.g. DNN/mol2vec with R² = -63)
    qm9_df, catastrophic_log = filter_catastrophic_iterations(qm9_df)
    if len(catastrophic_log) > 0:
        catastrophic_log.to_csv(output_dir / 'filtered_catastrophic_iterations.csv', index=False)
        print(f"    Saved filtered_catastrophic_iterations.csv")

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

    # Process validation data into NDS format
    val_nds_df = calculate_validation_nds(validation_df)
    if val_nds_df is not None:
        print(f"  Validation NDS: {len(val_nds_df)} configs across "
              f"{val_nds_df['dataset'].nunique() if 'dataset' in val_nds_df.columns else 1} datasets")

    print("\n--- PART 1: THE WHAT ---")
    create_figure1(qm9_df, nds_df, output_dir)
    create_figure2(qm9_df, output_dir)
    create_figure3(nds_df, validation_df, output_dir)
    create_interaction_figure(nds_df, output_dir)
    create_full_overview(nds_df, output_dir)

    print("\n--- PART 2: THE WHY ---")
    create_figure4(qm9_df, nds_df, output_dir)
    create_figure5(qm9_df, nds_df, output_dir)
    create_uncertainty_figure(unc_df, output_dir)

    print("\n--- TABLES ---")
    create_tables(nds_df, unc_df, qm9_df, output_dir)

    print("\n--- VALIDATION (GENERALISATION) ---")
    create_validation_figures(validation_df, val_nds_df, output_dir)

    print("\n--- SUPPLEMENTARY: ICC & REDUNDANCY ---")
    compute_icc_and_redundancy(nds_df, output_dir)

    print("\n--- REPORT ---")
    generate_report(nds_df, excluded_df, output_dir)

    print("\n" + "=" * 80)
    print(f"COMPLETE - All outputs in {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
