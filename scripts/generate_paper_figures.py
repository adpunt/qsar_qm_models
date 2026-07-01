#!/usr/bin/env python3
"""
Master Figure Generation Script for Noise Robustness Paper

Produces all main figures + tables for the paper structure:

PART 1: THE WHAT
  Figure 1: Global Overview (R² vs σ line plot, NDS heatmap)
  Figure 2: ANOVA Decomposition (η² bars by strategy)
  Figure 3: Ranking Consistency (heatmaps, cross-dataset)

PART 2: THE WHY
  Figure: NN Family Comparison (NN-α, NN-β, RF families) — 1×3 panel + NDS CSV
  Figure: Uncertainty Tracks Noise (single-panel: mean uncertainty vs σ)

Calibration (ECE, coverage), unc-error/unc-noise correlations, and
aleatoric/epistemic decomposition are reported in table4 CSVs.

Run: python generate_paper_figures.py [--qm9-dir ../results] [--validation-dir /path/to/kirby/results]

Outputs:
  results/paper_figures/
    fig1_global_overview.png
    fig2_anova_decomposition.png
    fig3_ranking_consistency.png
    fig_nn_family_comparison.png / table_nn_family_nds.csv
    fig_uncertainty_combined.png
    fig_full_overview_all_strategies.png
    fig_validation_combined.png
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
#      - Currently 'continuous_pdv' (continuous physics-based descriptors)
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
PRIMARY_REP = 'continuous_pdv'   # Continuous physicochemical descriptors
SUPPLEMENTARY_REP = 'ecfp4'      # ECFP4 is more robust and representative of QSAR practice

# Canonical ordering for all noise strategies. Use this everywhere instead of
# re-typing the literal list — keeps text/table/figure orderings in sync.
# Supervisor preference (Hetero last, matches the explanatory paragraph order):
#   Gaussian → Outlier → Quantile → Threshold → Value-Prop. → Heteroscedastic
STRATEGY_ORDER = ['legacy', 'outlier', 'quantile', 'threshold', 'valprop', 'hetero']

# All strategies for completeness checks (legacy alias — order-insensitive)
ALL_STRATEGIES = STRATEGY_ORDER

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

# Models excluded from ALL figures globally
GLOBAL_MODELS_EXCLUDE = {
    # Conformal wrappers — rho > 0.99 with base models, add no information
    'conformal_rf', 'conformal_qrf', 'conformal_dnn',
    'conformal_rf_split', 'conformal_qrf_split', 'conformal_dnn_split',
    # Pre-VBLL variational — was identical to last-layer (bug), replaced by VBLL
    'dnn_bnn_variational', 'mlp_bnn_variational',
    # Last-layer BNN — no significant improvement over base (Wilcoxon p > 0.1)
    'dnn_bnn_last', 'mlp_bnn_last',
    # Flexible DNN architecture variants — not discussed in paper
    'flexible_dnn', 'flexible_dnn_256_128_64', 'flexible_dnn_512_256',
}

ANOVA_MODELS_EXCLUDE = {
    'qrf',  # Redundant with rf (rho = 0.996)
    'gauche',  # Tanimoto kernel incompatible with continuous PDV (non-binary features)
    'gauche_rbf',  # RBF GP for PDV only; excluded from cross-rep ANOVA
} | GLOBAL_MODELS_EXCLUDE

# Models that only have PDV data by design (show N/A for other reps in heatmaps)
PDV_ONLY_MODELS = {'gauche_rbf'}

ANOVA_REPS_EXCLUDE = {
    'sns',                # Redundant with ecfp4 (rho = 0.90)
    'randomized_smiles',  # Incomplete coverage
    'random_smiles',      # Alias
    'pdv',                # Binary PDV; continuous_pdv used instead
    'morgan',             # Redundant with ecfp4 (rho = 0.995)
}

import argparse
import time
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
# Springer Nature sn-mathphys-num textwidth is ~6.3 in (16 cm). Using 6.5 in
# as the canonical width: figures are saved at this width so
# \includegraphics[width=\textwidth] renders 1:1 (no scaling), which keeps
# 10pt matplotlib text at 10pt on the printed page.
TEXTWIDTH_IN = 6.5
HALFWIDTH_IN = TEXTWIDTH_IN / 2
plt.rcParams.update({
    'figure.dpi': 300,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
})

# Color palettes — maximally distinct, all different from clean blue
CLEAN_COLOR = '#0072B2'        # Steel blue — used ONLY for clean (no-noise) data
STRATEGY_COLORS = {
    'legacy': '#E31A1C',       # Bright red
    'valprop': '#FF7F00',      # Pure orange
    'quantile': '#33A02C',     # Forest green
    'threshold': '#6A3D9A',    # Royal purple
    'outlier': '#B15928',      # Sienna/brown
    'hetero': '#E91E63',       # Hot pink
}

STRATEGY_LABELS = {
    'legacy': 'Gaussian',
    'valprop': 'Value-Prop.',
    'quantile': 'Quantile',
    'threshold': 'Threshold',
    'outlier': 'Outlier',
    'hetero': 'Heteroscedastic',
}

# ANOVA factor colors — consistent across fig2 and validation ANOVA figures
ANOVA_FACTOR_COLORS = {
    'Model': '#6BAED6',          # Light blue
    'Representation': '#FC8D59', # Orange
    'Strategy': '#66C2A5',       # Mint green (validation only)
    'Interaction': '#B39DDB',    # Light purple
}

# Colors: variants of the same model family share ONE color.
# Distinction between variants comes from MODEL_MARKERS (shape).
MODEL_COLORS = {
    # Tree-based — each distinct (different model families)
    'rf': '#0072B2',               # Blue
    'qrf': '#0072B2',              # Blue (RF variant)
    'xgboost': '#56B4E9',          # Sky blue
    'lgb': '#009E73',              # Teal
    'ngboost': '#D55E00',          # Vermillion
    # NN-α family — all orange
    'dnn': '#E69F00',
    'dnn_bnn_full': '#E69F00',
    'dnn_bnn_last': '#E69F00',
    'dnn_vbll': '#E69F00',
    # NN-β family — all pink
    'mlp': '#CC79A7',
    'mlp_bnn_full': '#CC79A7',
    'mlp_bnn_last': '#CC79A7',
    'mlp_vbll': '#CC79A7',
    # SVM / GP — unique
    'svm': '#999999',              # Gray
    'gauche': '#882255',           # Wine
    'gauche_rbf': '#882255',       # Wine (RBF variant, PDV only)
}

# Within-family colors for figures that compare variants of the same family.
# When variants are shown alongside other models, use MODEL_COLORS (shared family color).
# When variants are shown only against each other, use these to differentiate.
DNN_FAMILY_COLORS = {
    'dnn': '#333333',              # Dark grey (base/deterministic)
    'dnn_bnn_full': '#E69F00',     # Amber (full Bayesian)
    'dnn_vbll': '#882255',         # Wine (VBLL)
}
MLP_FAMILY_COLORS = {
    'mlp': '#333333',              # Dark grey (base/deterministic)
    'mlp_bnn_full': '#E69F00',     # Amber (full Bayesian)
    'mlp_vbll': '#882255',         # Wine (VBLL)
}
RF_FAMILY_COLORS = {
    'rf': '#0072B2',               # Blue (base)
    'qrf': '#AA4499',              # Purple (quantile variant; distinct from NGBoost vermillion)
}

# Unique colors for the uncertainty figure (9 models, all visually distinct)
UNCERTAINTY_COLORS = {
    'qrf':           '#E69F00',    # Amber
    'ngboost':       '#D55E00',    # Vermillion
    'gauche':        '#882255',    # Wine
    'dnn_bnn_full':  '#0072B2',    # Blue
    'dnn_bnn_last':  '#56B4E9',    # Sky blue
    'dnn_vbll':      '#009E73',    # Teal
    'mlp_bnn_full':  '#CC79A7',    # Pink
    'mlp_bnn_last':  '#AA4499',    # Purple
    'mlp_vbll':      '#332288',    # Indigo
}

# Markers: variants of the same family get different shapes.
# Base model = circle, BNN Full = square, BNN Last = triangle, VBLL = diamond.
MODEL_MARKERS = {
    # Base models
    'rf': 'o', 'qrf': 'D', 'xgboost': 'o', 'lgb': 'o', 'ngboost': 'o',
    'svm': 'o', 'gauche': 'o', 'gauche_rbf': 's',
    'dnn': 'o', 'mlp': 'o',
    # NN-α family variants
    'dnn_bnn_full': 's', 'dnn_bnn_last': '^', 'dnn_vbll': 'D',
    # NN-β family variants
    'mlp_bnn_full': 's', 'mlp_bnn_last': '^', 'mlp_vbll': 'D',
}

# Canonical ordering for legends — grouped by family, base model first.
MODEL_ORDER = [
    'rf', 'qrf',
    'xgboost', 'lgb', 'ngboost',
    'svm', 'gauche', 'gauche_rbf',
    'dnn', 'dnn_bnn_full', 'dnn_vbll',
    'mlp', 'mlp_bnn_full', 'mlp_vbll',
]

def sort_models_by_family(models):
    """Sort models by MODEL_ORDER (family grouping), then alphabetical for unknown."""
    order_map = {m: i for i, m in enumerate(MODEL_ORDER)}
    return sorted(models, key=lambda m: (order_map.get(m, len(MODEL_ORDER)), m))


def _to_colmajor(items, ncol):
    """Reorder a reading-order (row-major) sequence into the column-major order
    matplotlib needs so a multi-column legend DISPLAYS left-to-right, top-to-bottom.

    matplotlib fills legends column-major (down column 0, then column 1, ...),
    which scrambles a logically-ordered handle list. Passing the output of this
    helper restores intuitive reading order. Same permutation must be applied to
    both handles and labels.
    """
    n = len(items)
    if ncol <= 1 or n == 0:
        return list(items)
    nrow = -(-n // ncol)  # ceil
    out = []
    for c in range(ncol):
        for r in range(nrow):
            idx = r * ncol + c
            if idx < n:
                out.append(items[idx])
    return out


def _ordered_legend(target, ncol, **kwargs):
    """Draw a legend on `target` (Axes or Figure) whose entries read left-to-right,
    top-to-bottom across `ncol` columns (countering matplotlib's column-major fill).
    Collects the current axes handles/labels when `target` is an Axes."""
    import matplotlib.axes
    if isinstance(target, matplotlib.axes.Axes):
        handles, labels = target.get_legend_handles_labels()
    else:  # Figure — caller must have set labels via plot()
        handles, labels = target.axes[0].get_legend_handles_labels()
    handles = _to_colmajor(handles, ncol)
    labels = _to_colmajor(labels, ncol)
    return target.legend(handles, labels, ncol=ncol, **kwargs)


def get_variant_color(model):
    """Get color distinguishing variants within NN-α/NN-β families.

    For figures where multiple NN-α or NN-β variants appear together (e.g. uncertainty),
    use family-specific colors so variants are visually distinct. Non-family models
    use their standard MODEL_COLORS.
    """
    if model in DNN_FAMILY_COLORS:
        return DNN_FAMILY_COLORS[model]
    if model in MLP_FAMILY_COLORS:
        return MLP_FAMILY_COLORS[model]
    if model in RF_FAMILY_COLORS:
        return RF_FAMILY_COLORS[model]
    return MODEL_COLORS.get(model, '#333333')


MODEL_LABELS = {
    # Tree-based
    'rf': 'RF',
    'xgboost': 'XGBoost',
    'lgb': 'LightGBM',
    'qrf': 'QRF',
    'ngboost': 'NGBoost',
    # Neural networks
    'dnn': 'NN-α',
    'mlp': 'NN-β',
    # SVM / GP
    'svm': 'SVM',
    'gauche': 'GP',
    'gauche_rbf': 'GP (RBF)',
    # α Bayesian variants
    'dnn_bnn_full': 'BNN-α',
    'dnn_vbll': 'VBLL-α',
    # β Bayesian variants
    'mlp_bnn_full': 'BNN-β',
    'mlp_vbll': 'VBLL-β',
}

REP_LABELS = {
    'continuous_pdv': 'PDV',
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
    return model.replace('_', '-').upper() if '_' in model else model.upper()

def get_rep_label(rep):
    """Get clean display label for representation."""
    return REP_LABELS.get(rep, rep.upper())

DATASET_MARKERS = {
    'QM9': 'o',
    'LogD': 's',
    'Caco2_Efflux': '^',
    'hERG-Ki': 'D',
}

# Canonical order + display labels for the three external validation datasets.
# Used in every per-dataset panel layout so figures and paper.tex captions stay
# in sync (alphabetical sort gives ChEMBL → Caco2 → LogD, which mismatches the
# paper's narrative order LogD → Caco-2 → hERG).
VALIDATION_DATASET_ORDER = [
    'OpenADMET-LogD',
    'OpenADMET-Caco2_Efflux',
    'ChEMBL-hERG-Ki',
]
VALIDATION_DATASET_LABELS = {
    'OpenADMET-LogD': 'LogD',
    'OpenADMET-Caco2_Efflux': 'Caco-2 Efflux',
    'ChEMBL-hERG-Ki': 'hERG K$_i$',
}


def _order_validation_datasets(datasets):
    """Sort a dataset iterable into VALIDATION_DATASET_ORDER, appending any unknowns."""
    known = [d for d in VALIDATION_DATASET_ORDER if d in datasets]
    extras = [d for d in datasets if d not in VALIDATION_DATASET_ORDER]
    return known + sorted(extras)

BASELINE_THRESHOLD = 0.6
VALIDATION_BASELINE_THRESHOLD = 0.3  # External datasets are harder; lower threshold
CATASTROPHIC_R2_THRESHOLD = -0.5  # Per-iteration R² below this = training failure
VALIDATION_NDS_THRESHOLD = 2.0  # |NDS| above this = artifact, filter to N/A

# Per-fold training-set size for the dataset-size-effect analysis
# (compute_validation_size_effect, Fig 8 reviewer rebuttal).
# KIRBy uses 5-fold GroupKFold (scaffold) → train ≈ (4/5)·N_total, then 20% of
# train is carved out as val → effective per-fold train ≈ 0.64·N_total.
# QM9 default: --sample-size 10000, 80/10/10 split → train = 8000.
# *** REPLACE PLACEHOLDERS *** with the actual "Train: X" lines from the SLURM
# stdout for each KIRBy dataset before the analysis is used in the paper.
DATASET_N_TRAIN = {
    'QM9': 8000,
    'OpenADMET-LogD': None,            # TODO fill from SLURM log
    'OpenADMET-Caco2_Efflux': None,    # TODO fill from SLURM log
    'ChEMBL-hERG-Ki': None,            # TODO fill from SLURM log
}


def make_heatmap_annotations(pivot, raw_df, index_col, columns_col, rep_filter=None,
                              fmt='.2f', extra_filters=None):
    """Create annotation text for heatmaps distinguishing missing vs filtered data.

    For NaN cells in pivot:
    - If (index, column) combination exists in raw_df → "N/A" (filtered, e.g. baseline < 0.6)
    - If not in raw_df → "missing" (experiment not yet run)

    Args:
        pivot: DataFrame with NaN for missing/filtered cells
        raw_df: Raw experiment data (pre-NDS) to check what was actually run
        index_col: Column name in raw_df corresponding to pivot index (e.g. 'model')
        columns_col: Column name in raw_df corresponding to pivot columns (e.g. 'strategy')
        rep_filter: If set, filter raw_df to this rep value
        fmt: Format string for numeric values
        extra_filters: Dict of {column: value} for additional filters on raw_df
    Returns:
        DataFrame of annotation strings (same shape as pivot)
    """
    if raw_df is None:
        # No raw data: can't distinguish, use generic format
        try:
            return pivot.map(lambda x: '' if pd.isna(x) else f'{x:{fmt}}')
        except AttributeError:
            return pivot.applymap(lambda x: '' if pd.isna(x) else f'{x:{fmt}}')

    # Build set of (index, column) combos that exist in raw data
    filtered_raw = raw_df.copy()
    if rep_filter is not None and 'rep' in filtered_raw.columns:
        filtered_raw = filtered_raw[filtered_raw['rep'] == rep_filter]
    if extra_filters:
        for col, val in extra_filters.items():
            if col in filtered_raw.columns:
                filtered_raw = filtered_raw[filtered_raw[col] == val]

    existing_combos = set()
    if index_col in filtered_raw.columns and columns_col in filtered_raw.columns:
        for _, row in filtered_raw[[index_col, columns_col]].drop_duplicates().iterrows():
            existing_combos.add((row[index_col], row[columns_col]))

    annot = pd.DataFrame('', index=pivot.index, columns=pivot.columns)
    for idx in pivot.index:
        for col in pivot.columns:
            val = pivot.loc[idx, col]
            if pd.isna(val):
                # Check if this combo exists in raw data (map labels back to raw names)
                raw_idx = idx  # May need reverse mapping
                raw_col = col
                # Try reverse-mapping labels to raw names
                for raw_name, label in MODEL_LABELS.items():
                    if label == idx:
                        raw_idx = raw_name
                        break
                for raw_name, label in STRATEGY_LABELS.items():
                    if label == col:
                        raw_col = raw_name
                        break
                for raw_name, label in REP_LABELS.items():
                    if label == col:
                        raw_col = raw_name
                        break

                # PDV-only models (e.g. gauche_rbf) show N/A for non-PDV reps
                is_pdv_only_on_nonpdv = (raw_idx in PDV_ONLY_MODELS and
                                         columns_col == 'rep' and
                                         raw_col != 'continuous_pdv')
                # PDV-only models on non-PDV rep_filter (model×strategy heatmaps)
                is_pdv_only_wrong_rep = (raw_idx in PDV_ONLY_MODELS and
                                         rep_filter is not None and
                                         rep_filter != 'continuous_pdv')
                # Tanimoto GP is incompatible with continuous PDV
                is_tanimoto_on_pdv = (raw_idx == 'gauche' and
                                      ((columns_col == 'rep' and raw_col == 'continuous_pdv') or
                                       (rep_filter == 'continuous_pdv')))

                if (raw_idx, raw_col) in existing_combos or \
                   is_pdv_only_on_nonpdv or is_pdv_only_wrong_rep or \
                   is_tanimoto_on_pdv:
                    annot.loc[idx, col] = 'N/A'
                else:
                    annot.loc[idx, col] = 'missing'
            else:
                if abs(val) >= 10:
                    annot.loc[idx, col] = f'{val:.0f}'
                else:
                    annot.loc[idx, col] = f'{val:{fmt}}'
    return annot


def _white_text_for_missing(ax, pivot, annot_text):
    """Add white 'missing' and 'N/A' annotations for NaN cells in heatmaps.

    Seaborn's heatmap skips annotations entirely for NaN data cells, even when
    a custom annotation DataFrame is provided. This function manually adds white
    text at the correct cell positions for cells labelled 'missing' or 'N/A'.
    """
    for i, idx in enumerate(pivot.index):
        for j, col in enumerate(pivot.columns):
            val = pivot.loc[idx, col]
            if pd.isna(val):
                label = annot_text.loc[idx, col] if annot_text is not None else ''
                if label in ('missing', 'N/A'):
                    ax.text(j + 0.5, i + 0.5, label,
                            ha='center', va='center',
                            color='white', fontsize=7, fontweight='bold')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_anova_data(results_dir):
    """Load all anova_*.csv files, deduplicating appended runs."""
    results_dir = Path(results_dir)
    all_data = []

    # Valid strategy names for filename parsing (same as uncertainty loader)
    VALID_STRATEGIES = {
        'legacy', 'outlier', 'quantile', 'threshold', 'hetero', 'valprop',
        'heteroscedastic', 'value_proportional',
    }
    STRATEGY_NORMALIZE = {'heteroscedastic': 'hetero', 'value_proportional': 'valprop'}

    anova_files = sorted(results_dir.glob("anova_*.csv"), key=lambda p: p.stat().st_mtime)
    total_mb = sum(p.stat().st_size for p in anova_files) / 1e6
    print(f"  load_anova_data: {len(anova_files)} anova_*.csv files, {total_mb:.0f} MB total", flush=True)
    for i, f in enumerate(anova_files):
        if '_uncertainty_values' in f.name:
            continue
        if i % 50 == 0 and i > 0:
            print(f"    ...read {i}/{len(anova_files)} files", flush=True)
        try:
            df = pd.read_csv(f)
            # Parse filename: anova_{strategy}_{rep}_{model}.csv
            # Use longest-prefix matching to handle multi-word strategies
            # (e.g., value_proportional, heteroscedastic)
            rest = f.stem[len('anova_'):]
            strategy = None
            for s in sorted(VALID_STRATEGIES, key=len, reverse=True):
                if rest.startswith(s + '_'):
                    strategy = STRATEGY_NORMALIZE.get(s, s)
                    break
            if strategy:
                df['strategy'] = strategy

            # Fix: process_and_train.py with --bayesian-transformation saves
            # model as base name (dnn/mlp) without the BNN suffix. Infer from filename.
            if 'model' in df.columns and len(df) > 0:
                csv_model = df['model'].iloc[0]
                fname = f.stem
                if csv_model in ('dnn', 'mlp'):
                    if '_bnn_full_variational' in fname:
                        df['model'] = csv_model + '_bnn_full_variational'
                    elif '_bnn_last' in fname:
                        df['model'] = csv_model + '_bnn_last'
                    elif '_bnn_full' in fname and '_bnn_full_variational' not in fname:
                        df['model'] = csv_model + '_bnn_full'

            df['source_file'] = f.name
            all_data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        # Normalize model names from CSV conventions to clean internal names.
        # process_and_train.py saves NN-α BNN variants as 'bnn_full' etc.
        # VBLL variants saved as '*_full_variational' → renamed to '*_vbll'.
        BNN_NAME_MAP = {
            'bnn_full': 'dnn_bnn_full',
            'bnn_last': 'dnn_bnn_last',
            'bnn_variational': 'dnn_bnn_variational',  # Old pre-VBLL variant → gets caught by GLOBAL_MODELS_EXCLUDE
            'bnn_full_variational': 'dnn_vbll',
            'dnn_bnn_full_variational': 'dnn_vbll',
            'mlp_bnn_full_variational': 'mlp_vbll',
        }
        if 'model' in combined.columns:
            n_renamed = combined['model'].isin(BNN_NAME_MAP).sum()
            combined['model'] = combined['model'].map(lambda m: BNN_NAME_MAP.get(m, m))
            if n_renamed > 0:
                print(f"  Normalized {n_renamed} model names (bnn_* → dnn_bnn_*, *_full_variational → *_vbll)")

        # Normalize column: representation → rep
        if 'representation' in combined.columns and 'rep' not in combined.columns:
            combined.rename(columns={'representation': 'rep'}, inplace=True)

        # Global model exclusion (CP, old variational) — applied at load time
        if 'model' in combined.columns:
            n_before = len(combined)
            combined = combined[~combined['model'].isin(GLOBAL_MODELS_EXCLUDE)]
            n_excluded = n_before - len(combined)
            if n_excluded > 0:
                print(f"  Global exclusion: removed {n_excluded} rows "
                      f"(conformal wrappers + pre-VBLL variational)")

        # --- Deduplicate appended runs ---
        # Files may have been appended to across multiple SLURM runs.
        # Keep only the LAST occurrence of each (model, rep, strategy, sigma, iteration).
        pre_dedup = len(combined)
        dedup_cols = ['model', 'rep', 'strategy', 'sigma', 'iteration']
        if all(c in combined.columns for c in dedup_cols):
            combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
            n_dupes = pre_dedup - len(combined)
            if n_dupes > 0:
                print(f"  Deduplicated: removed {n_dupes} duplicate rows (kept last run)")

        print(f"Loaded QM9 ANOVA data: {len(combined)} rows from {len(all_data)} files")
        return combined
    return None


def audit_data_completeness(df, output_dir, min_iterations=5):
    """Audit ANOVA data for missing sigmas/iterations. Run after all filtering.

    Checks ALL expected model×rep×strategy combos including completely missing ones.
    Prints a summary and saves detailed gap report to CSV.
    """
    EXPECTED_SIGMAS = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}
    EXPECTED_REPS = {'ecfp4', 'continuous_pdv', 'smiles', 'mhggnn', 'mol2vec'}
    EXPECTED_STRATEGIES = set(STRATEGY_ORDER)

    # Only audit ANOVA-included data
    anova_df = df[
        ~df['model'].isin(ANOVA_MODELS_EXCLUDE) &
        ~df['rep'].isin(ANOVA_REPS_EXCLUDE)
    ]
    all_models = sorted(anova_df['model'].unique())

    gap_rows = []
    ok_count = 0
    warn_count = 0
    unusable_count = 0
    missing_count = 0

    for model in all_models:
        for rep in sorted(EXPECTED_REPS):
            for strategy in sorted(EXPECTED_STRATEGIES):
                grp = anova_df[(anova_df['model'] == model) &
                               (anova_df['rep'] == rep) &
                               (anova_df['strategy'] == strategy)]

                if len(grp) == 0:
                    gap_rows.append({
                        'model': model, 'rep': rep, 'strategy': strategy,
                        'status': 'MISSING', 'n_iterations': 0, 'n_sigmas': 0,
                        'missing_sigmas': str(sorted(EXPECTED_SIGMAS)),
                        'problems': 'NO DATA',
                    })
                    missing_count += 1
                    continue

                found_sigmas = set(grp['sigma'].unique())
                missing_sigmas = EXPECTED_SIGMAS - found_sigmas
                n_iters = grp['iteration'].nunique() if 'iteration' in grp.columns else 0

                problems = []
                if missing_sigmas:
                    problems.append(f"missing sigmas: {sorted(missing_sigmas)}")
                if n_iters < min_iterations:
                    problems.append(f"only {n_iters} iterations (need >= {min_iterations})")

                if n_iters < min_iterations:
                    status = 'UNUSABLE'
                    unusable_count += 1
                elif missing_sigmas or n_iters < 10:
                    status = 'WARNING'
                    warn_count += 1
                else:
                    status = 'OK'
                    ok_count += 1

                if status != 'OK':
                    gap_rows.append({
                        'model': model, 'rep': rep, 'strategy': strategy,
                        'status': status, 'n_iterations': n_iters,
                        'n_sigmas': len(found_sigmas),
                        'missing_sigmas': str(sorted(missing_sigmas)) if missing_sigmas else '',
                        'problems': '; '.join(problems),
                    })

    total = ok_count + warn_count + unusable_count + missing_count
    print(f"\n  DATA AUDIT ({total} model x rep x strategy configs):")
    print(f"    OK (10 iters, 11 sigmas): {ok_count}")
    print(f"    WARNING (usable but incomplete): {warn_count}")
    print(f"    UNUSABLE (<{min_iterations} iterations): {unusable_count}")
    print(f"    MISSING (no data at all): {missing_count}")

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        gap_path = output_dir / 'data_gaps.csv'
        gap_df.to_csv(gap_path, index=False)
        print(f"    Gap details saved to: {gap_path}")

        for status_label in ['MISSING', 'UNUSABLE', 'WARNING']:
            subset = gap_df[gap_df['status'] == status_label]
            if len(subset) == 0:
                continue
            print(f"\n    {status_label} CONFIGS ({len(subset)}):")
            for i, (_, row) in enumerate(subset.iterrows()):
                if i >= 30:
                    print(f"      ... and {len(subset) - 30} more (see data_gaps.csv)")
                    break
                detail = f"{row['n_iterations']} iters, {row['n_sigmas']} sigmas" if row['n_iterations'] > 0 else 'NO DATA'
                print(f"      {row['model']:30} / {row['rep']:10} / {row['strategy']:10} — {detail}")
    else:
        print("    No gaps found!")

    return gap_rows


def audit_uncertainty_completeness(unc_df, output_dir):
    """Audit uncertainty data for missing model/rep/strategy combos."""
    EXPECTED_SIGMAS = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}

    # Find uncertainty column
    unc_col = None
    for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty', 'std']:
        if col in unc_df.columns:
            unc_col = col
            break

    gap_rows = []
    ok_count = 0

    group_cols = [c for c in ['model', 'rep', 'strategy'] if c in unc_df.columns]
    if not group_cols:
        print("\n  UNCERTAINTY AUDIT: Cannot audit — missing group columns")
        return

    for keys, grp in unc_df.groupby(group_cols):
        if len(group_cols) == 3:
            model, rep, strategy = keys
        elif len(group_cols) == 2:
            model, rep = keys
            strategy = 'unknown'
        else:
            continue

        found_sigmas = set(grp['sigma'].unique()) if 'sigma' in grp.columns else set()
        missing_sigmas = EXPECTED_SIGMAS - found_sigmas
        n_samples = len(grp)

        # Check for all-NaN uncertainty
        all_nan = unc_col and grp[unc_col].isna().all()
        near_zero = unc_col and (grp[unc_col].abs().mean() < 1e-3 if not grp[unc_col].isna().all() else False)

        problems = []
        if missing_sigmas:
            problems.append(f"missing sigmas: {sorted(missing_sigmas)}")
        if all_nan:
            problems.append("all uncertainty values NaN")
        if near_zero:
            problems.append("mean uncertainty < 1e-3 (effectively zero)")
        if n_samples < 100:
            problems.append(f"only {n_samples} rows")

        if problems:
            gap_rows.append({
                'model': model, 'rep': rep, 'strategy': strategy,
                'n_sigmas': len(found_sigmas), 'n_rows': n_samples,
                'problems': '; '.join(problems),
            })
        else:
            ok_count += 1

    total = ok_count + len(gap_rows)
    print(f"\n  UNCERTAINTY AUDIT ({total} configs):")
    print(f"    OK: {ok_count}")
    print(f"    Problems: {len(gap_rows)}")

    if gap_rows:
        gap_df = pd.DataFrame(gap_rows)
        gap_path = output_dir / 'uncertainty_gaps.csv'
        gap_df.to_csv(gap_path, index=False)
        print(f"    Details saved to: {gap_path}")
        for _, row in gap_df.iterrows():
            print(f"      {row['model']:30} / {row.get('rep','?'):10} / {row.get('strategy','?'):10} — {row['problems']}")


def filter_catastrophic_iterations(df, r2_threshold=CATASTROPHIC_R2_THRESHOLD):
    """Filter out catastrophic training iterations (R² below threshold).

    NN-α training on certain representations (e.g. mol2vec) occasionally produces
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
        for f in sorted(results_dir.glob(pattern), key=lambda p: p.stat().st_mtime):
            try:
                df = pd.read_csv(f)
                df['source_file'] = f.name

                # Normalize column: representation → rep
                if 'representation' in df.columns and 'rep' not in df.columns:
                    df.rename(columns={'representation': 'rep'}, inplace=True)

                # Extract strategy from filename if not in data
                # Patterns:
                #   uncertainty_{strategy}_{rep}_{model}_uncertainty_values.csv
                #   anova_{strategy}_{rep}_{model}_uncertainty_values.csv
                if 'strategy' not in df.columns:
                    stem_clean = f.stem.replace('_uncertainty_values', '')
                    prefix = None
                    if stem_clean.startswith('uncertainty_'):
                        prefix = 'uncertainty_'
                    elif stem_clean.startswith('anova_'):
                        prefix = 'anova_'
                    if prefix:
                        rest = stem_clean[len(prefix):]
                        for s in sorted(VALID_STRATEGIES, key=len, reverse=True):
                            if rest.startswith(s + '_'):
                                df['strategy'] = STRATEGY_NORMALIZE.get(s, s)
                                break

                # Fix: process_and_train.py with --bayesian-transformation saves
                # model as base name (dnn/mlp) without the BNN suffix. Infer from filename.
                if 'model' in df.columns and len(df) > 0:
                    csv_model = df['model'].iloc[0]
                    fname = f.stem
                    if csv_model in ('dnn', 'mlp'):
                        if '_bnn_full_variational' in fname:
                            df['model'] = csv_model + '_bnn_full_variational'
                        elif '_bnn_last' in fname:
                            df['model'] = csv_model + '_bnn_last'
                        elif '_bnn_full' in fname and '_bnn_full_variational' not in fname:
                            df['model'] = csv_model + '_bnn_full'

                all_data.append(df)
            except Exception as e:
                print(f"Warning: Could not load {f.name}: {e}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)

        # Deduplicate appended runs
        dedup_cols = [c for c in ['model', 'rep', 'strategy', 'sigma', 'iteration', 'sample_idx']
                      if c in combined.columns]
        if len(dedup_cols) >= 4:
            pre_dedup = len(combined)
            combined = combined.drop_duplicates(subset=dedup_cols, keep='last')
            n_dupes = pre_dedup - len(combined)
            if n_dupes > 0:
                print(f"  Deduplicated uncertainty: removed {n_dupes} duplicate rows")

        # Normalize BNN model names (same map as ANOVA loader)
        BNN_NAME_MAP = {
            'bnn_full': 'dnn_bnn_full',
            'bnn_last': 'dnn_bnn_last',
            'bnn_variational': 'dnn_bnn_variational',
            'bnn_full_variational': 'dnn_vbll',
            'dnn_bnn_full_variational': 'dnn_vbll',
            'mlp_bnn_full_variational': 'mlp_vbll',
        }
        if 'model' in combined.columns:
            n_renamed = combined['model'].isin(BNN_NAME_MAP).sum()
            combined['model'] = combined['model'].map(lambda m: BNN_NAME_MAP.get(m, m))
            if n_renamed > 0:
                print(f"  Normalized {n_renamed} uncertainty model names")

        # Global model exclusion (CP, old variational)
        if 'model' in combined.columns:
            n_before = len(combined)
            combined = combined[~combined['model'].isin(GLOBAL_MODELS_EXCLUDE)]
            n_excluded = n_before - len(combined)
            if n_excluded > 0:
                print(f"  Global exclusion: removed {n_excluded} uncertainty rows")

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


# σ levels for the within-σ uncertainty–noise correlation (per-sample tracking).
# Reported SEPARATELY, never averaged across σ or across noise strategies.
# σ=0.0 is the clean control (expect ρ≈0/undefined — no noise injected); 0.3 (moderate)
# and 0.6 (high) are the real per-sample tests.
WITHIN_SIGMA_LEVELS = [0.0, 0.3, 0.6]


def within_sigma_unc_noise_rho(model_data, unc_values, base_mask,
                               sigma_levels=WITHIN_SIGMA_LEVELS, min_points=100):
    """Per-σ Spearman ρ between predicted uncertainty and |injected_noise|.

    The pooled correlation (all σ stacked together) measures the POPULATION-level trend:
    uncertainty AND |injected_noise| both rise with σ, so pooling makes the points a
    staircase and Spearman reports that shared σ ramp — the Kolmar effect — as a strong
    positive value, NOT per-sample detection. Computing ρ within each σ slice isolates
    whether uncertainty actually tracks the per-sample noise draw εᵢ.

    Returns {sigma: {'rho', 'p', 'n'}} for each requested σ. Fully disaggregated —
    the caller must NOT average across σ (or across strategies).
    """
    out = {s: {'rho': np.nan, 'p': np.nan, 'n': 0} for s in sigma_levels}
    if 'injected_noise' not in model_data.columns or 'sigma' not in model_data.columns:
        return out
    noise_mag = np.abs(model_data['injected_noise'].values)
    sigmas = model_data['sigma'].values
    for s in sigma_levels:
        m = base_mask & np.isclose(sigmas, s, atol=1e-6) & np.isfinite(noise_mag)
        n = int(m.sum())
        out[s]['n'] = n
        if n > min_points:
            rho, p = stats.spearmanr(unc_values[m], noise_mag[m])
            out[s]['rho'] = rho
            out[s]['p'] = p
    return out


def _normalize_validation_names(df):
    """Normalize KIRBy naming conventions to match QM9 conventions."""
    val_model_map = {
        'RF': 'rf', 'XGBoost': 'xgboost', 'DNN': 'dnn', 'MLP': 'mlp',  # KIRBy uses DNN/MLP internally
        # Validation GP is sklearn GaussianProcessRegressor with an RBF + WhiteKernel
        # composite, run on PDV only. Semantically that's gauche_rbf (RBF GP on PDV),
        # NOT gauche (Tanimoto GP on fingerprints). Conflating them in figures would
        # mix two different models.
        'GP': 'gauche_rbf', 'QRF': 'qrf', 'NGBoost': 'ngboost', 'SVM': 'svm',
        'LightGBM': 'lgb', 'LGBM': 'lgb',
        'BNN-Full': 'dnn_bnn_full', 'BNN-Last': 'dnn_bnn_last',
        'VBLL-Full': 'dnn_vbll',
        # NN-β family from KIRBy validation (MLP base, hidden=32, 2 hidden layers)
        'MLP-BNN-Full': 'mlp_bnn_full',
        'MLP-VBLL-Full': 'mlp_vbll',
    }
    val_rep_map = {
        'ECFP4': 'ecfp4', 'PDV': 'continuous_pdv', 'SNS': 'sns',
        'MHG-GNN-pretrained': 'mhggnn', 'MHGGNNpretrained': 'mhggnn',
        'SMILES': 'smiles',
    }
    # Map directory names → display names for datasets
    # herg_fluid is classification (no r2), excluded from regression NDS analysis
    val_dataset_map = {
        'openadmet_logd': 'OpenADMET-LogD',
        'openadmet_caco2': 'OpenADMET-Caco2_Efflux',
        'herg': 'ChEMBL-hERG-Ki',
        'logd': 'OpenADMET-LogD',              # duplicate of openadmet_logd
        'caco2': 'OpenADMET-Caco2_Efflux',     # duplicate of openadmet_caco2
    }
    VALIDATION_EXCLUDE_DIRS = {'herg_fluid'}  # classification dataset, not regression
    if 'model' in df.columns:
        df['model'] = df['model'].map(val_model_map).fillna(df['model'].str.lower())
    if 'rep' in df.columns:
        df['rep'] = df['rep'].map(val_rep_map).fillna(df['rep'].str.lower())
    if 'dataset' in df.columns:
        df['dataset'] = df['dataset'].map(val_dataset_map).fillna(df['dataset'])
    # Normalize NDS column name
    if 'NDS_r2' in df.columns and 'nds' not in df.columns:
        df = df.rename(columns={'NDS_r2': 'nds'})
    return df


def load_validation_data(validation_dir):
    """Load validation data from KIRBy results directory.

    Prefers per-dataset subdirectories (always up-to-date) over
    combined_summary.csv (may be stale if new models were added).
    """
    if validation_dir is None:
        return None

    validation_dir = Path(validation_dir)

    # Try loading per-dataset subdirectories (preferred — always current)
    # Import exclusion set from _normalize_validation_names
    # herg_fluid: classification (no r2)
    # openadmet_*: older runs without scaffold CV folds, superseded by caco2/herg/logd
    exclude_dirs = {'herg_fluid', 'openadmet_caco2', 'openadmet_logd'}
    all_data = []
    for subdir in sorted(validation_dir.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name in exclude_dirs:
            print(f"  Skipping {subdir.name}/ (excluded: classification or superseded)")
            continue
        # Prefer all_results.csv (per-sigma format) for NDS computation
        results_file = subdir / 'all_results.csv'
        if results_file.exists():
            df = pd.read_csv(results_file)
            if 'r2' not in df.columns:
                print(f"  Skipping {subdir.name}/all_results.csv (no r2 column — classification data)")
                continue
            df['dataset'] = subdir.name
            all_data.append(df)
            print(f"  Loaded {len(df)} rows from {subdir.name}/all_results.csv")
            continue
        # Fall back to summary.csv
        summary = subdir / 'summary.csv'
        if summary.exists():
            df = pd.read_csv(summary)
            df['dataset'] = subdir.name
            all_data.append(df)
            print(f"  Loaded {len(df)} rows from {subdir.name}/summary.csv")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        combined = _normalize_validation_names(combined)
        # Deduplicate: if logd + openadmet_logd map to same display name, keep longer data
        if 'dataset' in combined.columns:
            before = len(combined)
            dedup_cols = [c for c in ['dataset', 'model', 'rep', 'strategy', 'sigma'] if c in combined.columns]
            combined = combined.drop_duplicates(subset=dedup_cols, keep='first')
            if len(combined) < before:
                print(f"  Deduplicated: {before} → {len(combined)} rows (overlapping directories)")
        datasets = combined['dataset'].unique()
        print(f"Loaded validation data: {len(combined)} rows from {len(datasets)} datasets ({', '.join(sorted(datasets))})")
        return combined

    # Fall back to combined_summary.csv (may be stale)
    summary_file = validation_dir / 'combined_summary.csv'
    if summary_file.exists():
        df = pd.read_csv(summary_file)
        df = _normalize_validation_names(df)
        datasets = df['dataset'].unique() if 'dataset' in df.columns else ['unknown']
        print(f"⚠ Using combined_summary.csv ({len(df)} rows, datasets: {sorted(datasets)}) — "
              f"this file may be stale. Delete it to force per-directory loading.")
        return df

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

    # Prefer per-sigma computation (produces clean 1 row per config)
    # over pre-computed NDS columns (which may have per-sigma duplicates)
    has_sigma = 'sigma' in validation_df.columns and 'r2' in validation_df.columns

    # If summary-only format (has NDS column but NO per-sigma data), convert directly
    nds_col = None
    for col in ['nds', 'NSI', 'NDS_r2', 'nsi_r2']:
        if col in validation_df.columns:
            nds_col = col
            break

    if nds_col is not None and not has_sigma:
        nds_df = validation_df.rename(columns={nds_col: 'nds'})
        if 'baseline_r2' not in nds_df.columns:
            nds_df['baseline_r2'] = np.nan
        if 'dataset' not in nds_df.columns:
            nds_df['dataset'] = 'validation'
        # Apply baseline R² filtering (lower threshold for external datasets)
        if 'baseline_r2' in nds_df.columns:
            low_baseline = nds_df['baseline_r2'] < VALIDATION_BASELINE_THRESHOLD
            n_filtered = low_baseline.sum()
            if n_filtered > 0:
                print(f"  ⚠ Filtering {n_filtered} validation configs with baseline R² < {VALIDATION_BASELINE_THRESHOLD}:")
                for _, row in nds_df[low_baseline].iterrows():
                    print(f"    {row.get('model','?')}/{row.get('rep','?')}/{row.get('strategy','?')} "
                          f"on {row.get('dataset','?')}: baseline_r2={row['baseline_r2']:.3f}")
                nds_df = nds_df[~low_baseline].copy()
        return nds_df

    # Per-sigma format (has sigma and r2 columns): compute NDS from scratch
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
            if baseline < VALIDATION_BASELINE_THRESHOLD:
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


def create_validation_figures(validation_df, val_nds_df, qm9_nds_df, output_dir):
    """Generate all validation-related figures and tables.

    Validation data (LogD, Caco2_Efflux, hERG-Ki) is integrated into the paper
    as generalisation evidence — NOT a separate section.

    Outputs:
    - fig_validation_overview.png: NDS heatmap per dataset (like Figure 1B)
    - fig_validation_anova.png: η² decomposition on validation datasets
    - fig_validation_degradation.png: R² vs σ curves per dataset
    - fig_validation_strategy.png: Strategy comparison across datasets
    - fig_validation_qm9_correlation.png: QM9 vs external NDS correlation
    - fig_validation_rep_comparison.png: Representation effect on external datasets
    - table_validation_nds.csv: Full NDS table across datasets
    - table_validation_anova.csv: Validation ANOVA statistics
    """
    if val_nds_df is None or len(val_nds_df) == 0:
        print("⚠ No validation NDS data available — skipping validation figures")
        return

    # Filter extreme NDS values (|NDS| > threshold = artifact, e.g. NN-α divergence on hERG-Ki)
    # These are set to NaN so they appear as "N/A" in heatmaps
    val_nds_df = val_nds_df.copy()
    extreme_mask = val_nds_df['nds'].abs() > VALIDATION_NDS_THRESHOLD
    if extreme_mask.any():
        n_extreme = extreme_mask.sum()
        print(f"  ⚠ Filtering {n_extreme} extreme validation NDS values (|NDS| > {VALIDATION_NDS_THRESHOLD}):")
        for _, row in val_nds_df[extreme_mask].iterrows():
            ds = row.get('dataset', '?')
            print(f"    {row['model']}/{row.get('rep','?')}/{row.get('strategy','?')} "
                  f"on {ds}: NDS={row['nds']:.1f}")
        val_nds_df.loc[extreme_mask, 'nds'] = np.nan

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

        # Diagnostic: save full per-model/rep/strategy/dataset breakdown for inspection
        diag_cols = ['dataset', 'model', 'rep', 'strategy', 'nds']
        if 'baseline_r2' in val_nds_df.columns:
            diag_cols.append('baseline_r2')
        diag_df = val_nds_df[diag_cols].sort_values(['dataset', 'model', 'rep', 'strategy'])
        diag_df.to_csv(output_dir / 'table_validation_nds_full.csv', index=False)
        print(f"✓ Saved table_validation_nds_full.csv ({len(diag_df)} rows)")

        # Flag extreme individual NDS values for investigation
        extreme = diag_df[diag_df['nds'].abs() > 1.0].copy() if 'nds' in diag_df.columns else pd.DataFrame()
        if len(extreme) > 0:
            print(f"  ⚠ {len(extreme)} validation configs with |NDS| > 1.0:")
            for _, row in extreme.iterrows():
                bl = f", baseline_r2={row['baseline_r2']:.3f}" if 'baseline_r2' in row and pd.notna(row['baseline_r2']) else ""
                print(f"    {row['model']}/{row['rep']}/{row['strategy']} on {row['dataset']}: NDS={row['nds']:.3f}{bl}")
    else:
        val_nds_df.to_csv(output_dir / 'table_validation_nds.csv', index=False)
        print("✓ Saved table_validation_nds.csv")

    # --- Figure: Validation NDS heatmap per dataset ---
    # Show a single representation (PRIMARY_REP = continuous_pdv) so the figure
    # represents an actual configuration rather than a cross-rep mean. NDS is
    # near rep-invariant (see ANOVA: representation η² for robustness is small),
    # so PDV is a defensible single choice and the rep is named in the title.
    n_datasets = len(datasets)
    val_nds_primary = val_nds_df[val_nds_df['rep'] == PRIMARY_REP] if 'rep' in val_nds_df.columns else val_nds_df
    if n_datasets > 0 and 'dataset' in val_nds_df.columns and len(val_nds_primary) > 0:
        # Stack panels vertically — three 6-column heatmaps + colorbars don't fit
        # in 6.5" of textwidth (annotations overlap the colorbar). Each panel
        # gets full textwidth so 5-char cell labels render cleanly.
        # Height: ≥0.25" per model row so 10pt labels + cell annotations don't
        # overlap. Caco-2 has up to 13 rows → ~3.25" of plot area + title/xticks.
        ordered_datasets = _order_validation_datasets(datasets)
        max_rows = 0
        for ds in ordered_datasets:
            ds_models = val_nds_primary[val_nds_primary['dataset'] == ds]['model'].nunique()
            max_rows = max(max_rows, ds_models)
        row_h = 0.26   # inches per heatmap row
        chrome_h = 1.1  # title + x ticks + axes padding per panel
        per_panel_h = max(max_rows * row_h + chrome_h, 0.5 * TEXTWIDTH_IN)
        fig, axes = plt.subplots(n_datasets, 1,
                                 figsize=(TEXTWIDTH_IN, per_panel_h * n_datasets),
                                 squeeze=False)
        axes = axes[:, 0]
        panel_letters = ['a)', 'b)', 'c)', 'd)']

        # Compute shared color scale across all datasets for uniform comparison
        all_nds_vals = val_nds_primary['nds'].dropna().values
        if len(all_nds_vals) > 0:
            NDS_CLIP = 2.0
            clipped_vals = np.clip(all_nds_vals, -NDS_CLIP, 0)
            shared_vmin = clipped_vals.min()
        else:
            NDS_CLIP = 2.0
            shared_vmin = -NDS_CLIP

        for i, dataset in enumerate(_order_validation_datasets(datasets)):
            ax = axes[i]
            ds_data = val_nds_primary[val_nds_primary['dataset'] == dataset]
            if len(ds_data) == 0:
                continue

            pivot = ds_data.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
            col_order = [c for c in STRATEGY_ORDER if c in pivot.columns]
            if not col_order:
                continue
            pivot = pivot[col_order]
            # Sort by family grouping
            family_order = [m for m in sort_models_by_family(pivot.index.tolist())]
            pivot = pivot.loc[family_order]

            # Clip extreme NDS for colormap but show actual values in annotations
            pivot_display = pivot.clip(lower=-NDS_CLIP)

            # Build annotations: missing vs N/A vs actual value.
            # Filter raw validation to PRIMARY_REP so 'N/A' vs 'missing' is computed
            # against the rows that actually feed this panel.
            ds_raw = validation_df[validation_df['dataset'] == dataset] if validation_df is not None else None
            if ds_raw is not None and 'rep' in ds_raw.columns:
                ds_raw = ds_raw[ds_raw['rep'] == PRIMARY_REP]
            annot_text = make_heatmap_annotations(pivot, ds_raw, 'model', 'strategy', fmt='.2f')
            pivot.index = [get_model_label(m) for m in pivot.index]
            pivot_display.index = pivot.index
            annot_text.index = pivot.index
            pivot.columns = [STRATEGY_LABELS.get(c, c) for c in col_order]
            pivot_display.columns = pivot.columns
            annot_text.columns = pivot.columns

            ax.set_facecolor('black')
            sns.heatmap(pivot_display, annot=annot_text, fmt='', cmap='PuOr', center=0,
                        vmin=shared_vmin, vmax=0,
                        ax=ax, cbar_kws={'label': 'NDS'}, linewidths=0.5,
                        linecolor='#333333')
            _white_text_for_missing(ax, pivot_display, annot_text)
            letter = panel_letters[i] if i < len(panel_letters) else f'({i})'
            label = VALIDATION_DATASET_LABELS.get(dataset, dataset)
            ax.set_title(f'{letter} {label}', fontweight='bold')
            if i > 0:
                ax.set_ylabel('')

        plt.tight_layout()
        plt.savefig(output_dir / 'fig_validation_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved fig_validation_overview.png")

    # --- Figure: Validation ANOVA (η² decomposition: Model × Rep, legacy strategy only) ---
    # Filter to legacy strategy so the ANOVA mirrors the QM9 per-strategy design.
    if 'dataset' in val_nds_df.columns and 'model' in val_nds_df.columns and 'rep' in val_nds_df.columns:
        anova_results = {}
        for dataset in _order_validation_datasets(datasets):
            ds_nds = val_nds_df[val_nds_df['dataset'] == dataset].copy()
            ds_nds = ds_nds[~ds_nds['rep'].isin(ANOVA_REPS_EXCLUDE)]
            # Use legacy strategy only — consistent with QM9 per-strategy ANOVA
            if 'strategy' in ds_nds.columns:
                ds_nds = ds_nds[ds_nds['strategy'] == 'legacy']
            ds_nds = ds_nds.dropna(subset=['nds'])
            if len(ds_nds) < 10:
                continue

            grand_mean = ds_nds['nds'].mean()
            total_ss = ((ds_nds['nds'] - grand_mean) ** 2).sum()
            if total_ss == 0:
                continue

            model_means = ds_nds.groupby('model')['nds'].mean()
            model_counts = ds_nds.groupby('model').size()
            ss_model = (model_counts * (model_means - grand_mean) ** 2).sum()

            rep_means = ds_nds.groupby('rep')['nds'].mean()
            rep_counts = ds_nds.groupby('rep').size()
            ss_rep = (rep_counts * (rep_means - grand_mean) ** 2).sum()

            interaction_means = ds_nds.groupby(['model', 'rep'])['nds'].mean()
            interaction_counts = ds_nds.groupby(['model', 'rep']).size()
            ss_interaction = 0
            for (model, rep), count in interaction_counts.items():
                if model in model_means.index and rep in rep_means.index:
                    cell_mean = interaction_means[(model, rep)]
                    expected = model_means[model] + rep_means[rep] - grand_mean
                    ss_interaction += count * (cell_mean - expected) ** 2

            result = {
                'eta2_model': (ss_model / total_ss) * 100,
                'eta2_rep': (ss_rep / total_ss) * 100,
                'eta2_interaction': (ss_interaction / total_ss) * 100,
                'eta2_residual': ((total_ss - ss_model - ss_rep - ss_interaction) / total_ss) * 100,
                'n_models': ds_nds['model'].nunique(),
                'n_reps': ds_nds['rep'].nunique(),
                'n': len(ds_nds),
            }
            anova_results[dataset] = result

        if anova_results:
            # Stacked vertically (portrait page): one full-width panel per dataset.
            fig, axes = plt.subplots(len(anova_results), 1,
                                     figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.30 * len(anova_results)),
                                     squeeze=False, sharey=True)
            axes = axes[:, 0]
            panel_letters = ['a)', 'b)', 'c)', 'd)']

            for i, (dataset, result) in enumerate(anova_results.items()):
                ax = axes[i]
                factors = ['Model', 'Rep', 'Interaction']
                values = [result['eta2_model'], result['eta2_rep'], result['eta2_interaction']]
                colors = [ANOVA_FACTOR_COLORS[f if f != 'Rep' else 'Representation'] for f in factors]
                ax.bar(factors, values, color=colors)
                letter = panel_letters[i] if i < len(panel_letters) else f'({i})'
                label = VALIDATION_DATASET_LABELS.get(dataset, dataset)
                ax.set_title(f'{letter} {label}', fontweight='bold')
                ax.set_ylim(0, 100)
                ax.tick_params(axis='x', rotation=30)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            # Single shared y-label — per-panel labels overlap on short stacked panels.
            fig.supylabel('Variance Explained (η², %)')
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_validation_anova.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved fig_validation_anova.png")

            # Save ANOVA table
            rows = []
            for dataset, result in anova_results.items():
                row = {
                    'Dataset': dataset,
                    'Model_η²': round(result['eta2_model'], 1),
                    'Rep_η²': round(result['eta2_rep'], 1),
                }
                row['Interaction_η²'] = round(result['eta2_interaction'], 1)
                row['Residual_η²'] = round(result['eta2_residual'], 1)
                row['n_models'] = result['n_models']
                row['n_reps'] = result['n_reps']
                row['n'] = result['n']
                rows.append(row)
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
        # Add NN-α/NN-β BNN comparisons if available
        for base in ['dnn', 'mlp']:
            for suffix in ['_bnn_full', '_bnn_variational']:
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

    # --- Figure: R² degradation curves per dataset (unified axes) ---
    DEGRADATION_R2_FLOOR = 0.5  # Floor for y-axis; models whose R² drops below 0.5 are filtered
    if validation_df is not None and 'sigma' in validation_df.columns and 'dataset' in validation_df.columns:

        for dataset in sorted(datasets):
            ds_df = validation_df[validation_df['dataset'] == dataset]
            if len(ds_df) == 0:
                continue

            # One panel per representation
            reps = sorted(ds_df['rep'].unique()) if 'rep' in ds_df.columns else ['all']
            n_reps = len(reps)
            fig, axes_deg = plt.subplots(1, n_reps, figsize=(5 * n_reps, 5), squeeze=False)
            axes_deg = axes_deg[0]

            filtered_models = []  # Track models excluded for paper.md note

            for j, rep in enumerate(reps):
                ax = axes_deg[j]
                rep_df = ds_df[ds_df['rep'] == rep] if 'rep' in ds_df.columns else ds_df
                # Use legacy strategy for clean comparison
                leg_df = rep_df[rep_df['strategy'] == 'legacy'] if 'strategy' in rep_df.columns else rep_df

                # Two-pass approach: first check which models pass the floor,
                # then relax if ALL models are filtered (e.g. hERG-Ki)
                effective_floor = DEGRADATION_R2_FLOOR
                model_sigma_data = {}  # model -> sigma_means DataFrame
                for model in sort_models_by_family(leg_df['model'].unique().tolist()):
                    m_df = leg_df[leg_df['model'] == model]
                    sigma_means = m_df.groupby('sigma')['r2'].mean().reset_index().sort_values('sigma')
                    if len(sigma_means) >= 2:
                        model_sigma_data[model] = sigma_means

                models_passing = [m for m, sd in model_sigma_data.items()
                                  if sd['r2'].min() >= DEGRADATION_R2_FLOOR]

                if len(models_passing) == 0 and len(model_sigma_data) > 0:
                    effective_floor = 0.0
                    models_passing = [m for m, sd in model_sigma_data.items()
                                      if sd['r2'].min() >= effective_floor]
                    print(f"  ⚠ {dataset}/{rep}: all models filtered at floor={DEGRADATION_R2_FLOOR}, "
                          f"relaxing to {effective_floor}")

                for model in sort_models_by_family(list(model_sigma_data.keys())):
                    sigma_means = model_sigma_data[model]
                    if sigma_means['r2'].min() < effective_floor:
                        filtered_models.append({
                            'dataset': dataset, 'rep': rep, 'model': model,
                            'min_r2': sigma_means['r2'].min(),
                        })
                        continue

                    color = MODEL_COLORS.get(model, '#333333')
                    marker = MODEL_MARKERS.get(model, 'o')
                    ax.plot(sigma_means['sigma'], sigma_means['r2'],
                            marker=marker, linestyle='-',
                            label=get_model_label(model), color=color,
                            markersize=4, linewidth=1.5, alpha=0.8)

                ax.set_xlabel('Injected Noise Level (σ)')
                ax.set_ylabel('R²')
                ax.set_title(f'{rep.upper()}', fontweight='bold')
                ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.set_ylim(effective_floor - 0.05, 1.05)
                # Legend on every panel so each panel's models are labeled
                _ordered_legend(ax, 2, fontsize=6, loc='lower left', framealpha=0.9)

            if filtered_models:
                safe_name_f = dataset.replace('/', '_').replace(' ', '_')
                print(f"  ⚠ {safe_name_f}: filtered {len(filtered_models)} divergent model×rep combos "
                      f"from degradation plot (R² < {DEGRADATION_R2_FLOOR}):")
                for fm in filtered_models:
                    print(f"    {fm['model']}/{fm['rep']}: min R²={fm['min_r2']:.3f}")

            plt.suptitle(f'{dataset}: R² Degradation with Noise (Gaussian Strategy)',
                         fontweight='bold', y=1.02)
            plt.tight_layout()
            safe_name = dataset.replace('/', '_').replace(' ', '_')
            plt.savefig(output_dir / f'fig_validation_degradation_{safe_name}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved fig_validation_degradation_{safe_name}.png")

    # --- Figure: Strategy comparison across datasets ---
    if val_nds_df is not None and 'dataset' in val_nds_df.columns and 'strategy' in val_nds_df.columns:
        strategies = [s for s in STRATEGY_ORDER
                      if s in val_nds_df['strategy'].unique()]
        if len(strategies) >= 2:
            # Heatmap: mean NDS by strategy × dataset (averaged across models and reps)
            pivot_strat = val_nds_df.pivot_table(
                values='nds', index='strategy', columns='dataset', aggfunc='mean')
            # Reorder strategies
            strat_order = [s for s in strategies if s in pivot_strat.index]
            pivot_strat = pivot_strat.loc[strat_order]
            pivot_strat['MEAN'] = pivot_strat.mean(axis=1)
            pivot_strat = pivot_strat.sort_values('MEAN', ascending=False)
            pivot_strat.index = [STRATEGY_LABELS.get(s, s) for s in pivot_strat.index]

            fig, ax = plt.subplots(figsize=(max(6, 2 * len(datasets)), 4))
            ax.set_facecolor('black')
            sns.heatmap(pivot_strat, annot=True, fmt='.2f', cmap='PuOr', center=0, vmax=0,
                        ax=ax, cbar_kws={'label': 'Mean NDS'}, linewidths=0.5,
                        linecolor='#333333')
            ax.set_title('Noise Strategy Robustness Across Validation Datasets', fontweight='bold')
            ax.set_ylabel('Strategy')
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_validation_strategy.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved fig_validation_strategy.png")

    # --- Figure: Representation comparison on external datasets ---
    if val_nds_df is not None and 'rep' in val_nds_df.columns and 'dataset' in val_nds_df.columns:
        reps_available = sorted(val_nds_df['rep'].unique())
        if len(reps_available) >= 2:
            # Mean NDS by rep × dataset (averaged across models and strategies)
            pivot_rep = val_nds_df.pivot_table(
                values='nds', index='rep', columns='dataset', aggfunc='mean')
            pivot_rep['MEAN'] = pivot_rep.mean(axis=1)
            pivot_rep = pivot_rep.sort_values('MEAN', ascending=False)
            pivot_rep.index = [get_rep_label(r) for r in pivot_rep.index]

            fig, ax = plt.subplots(figsize=(max(6, 2 * len(datasets)), 4))
            ax.set_facecolor('black')
            sns.heatmap(pivot_rep, annot=True, fmt='.2f', cmap='PuOr', center=0, vmax=0,
                        ax=ax, cbar_kws={'label': 'Mean NDS'}, linewidths=0.5,
                        linecolor='#333333')
            ax.set_title('Representation Effect on Robustness (Validation Datasets)', fontweight='bold')
            ax.set_ylabel('Representation')
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_validation_rep_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved fig_validation_rep_comparison.png")

    # --- Figure: QM9 vs External NDS correlation ---
    if qm9_nds_df is not None and val_nds_df is not None and 'dataset' in val_nds_df.columns:
        # Compute mean QM9 NDS per model (across reps and strategies)
        qm9_model_nds = qm9_nds_df.groupby('model')['nds'].mean().reset_index()
        qm9_model_nds = qm9_model_nds.rename(columns={'nds': 'qm9_nds'})

        # For each external dataset, compute mean NDS per model
        for dataset in sorted(datasets):
            ds_nds = val_nds_df[val_nds_df['dataset'] == dataset]
            ds_model_nds = ds_nds.groupby('model')['nds'].mean().reset_index()
            ds_model_nds = ds_model_nds.rename(columns={'nds': 'ext_nds'})

            merged = qm9_model_nds.merge(ds_model_nds, on='model', how='inner')
            if len(merged) < 3:
                continue

            # Drop rows with NaN NDS (filtered artifacts)
            merged = merged.dropna(subset=['qm9_nds', 'ext_nds'])
            if len(merged) < 3:
                continue

            fig, ax = plt.subplots(figsize=(7, 6))
            for _, row in merged.iterrows():
                color = get_variant_color(row['model'])
                ax.scatter(row['qm9_nds'], row['ext_nds'], color=color, s=80, zorder=5,
                           label=get_model_label(row['model']),
                           edgecolors='black', linewidth=0.3)

            # Correlation line
            r, p = stats.pearsonr(merged['qm9_nds'], merged['ext_nds'])
            slope, intercept = np.polyfit(merged['qm9_nds'], merged['ext_nds'], 1)
            x_range = np.linspace(merged['qm9_nds'].min(), merged['qm9_nds'].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, '--', color='grey', alpha=0.6)

            ax.set_xlabel('QM9 Mean NDS')
            ax.set_ylabel(f'{dataset} Mean NDS')
            ax.set_title(f'QM9 vs {dataset} Robustness (r={r:.2f}, p={p:.3f})', fontweight='bold')
            ax.axhline(0, color='grey', linewidth=0.3)
            ax.legend(fontsize=7, loc='best', framealpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Tight axis around data
            x_pad = (merged['qm9_nds'].max() - merged['qm9_nds'].min()) * 0.15
            y_pad = (merged['ext_nds'].max() - merged['ext_nds'].min()) * 0.15
            ax.set_xlim(merged['qm9_nds'].min() - x_pad, merged['qm9_nds'].max() + x_pad)
            ax.set_ylim(merged['ext_nds'].min() - y_pad, merged['ext_nds'].max() + y_pad)

            plt.tight_layout()
            safe_name = dataset.replace('/', '_').replace(' ', '_')
            plt.savefig(output_dir / f'fig_validation_qm9_correlation_{safe_name}.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved fig_validation_qm9_correlation_{safe_name}.png")

    # --- Figure: Combined NDS comparison (QM9 vs all external) ---
    if qm9_nds_df is not None and val_nds_df is not None and 'dataset' in val_nds_df.columns:
        # Model-level summary: QM9 NDS vs mean external NDS
        qm9_model_nds = qm9_nds_df.groupby('model')['nds'].mean().reset_index()
        qm9_model_nds = qm9_model_nds.rename(columns={'nds': 'qm9_nds'})
        ext_model_nds = val_nds_df.groupby('model')['nds'].mean().reset_index()
        ext_model_nds = ext_model_nds.rename(columns={'nds': 'ext_nds'})
        merged = qm9_model_nds.merge(ext_model_nds, on='model', how='inner')
        merged = merged.dropna(subset=['qm9_nds', 'ext_nds'])

        if len(merged) >= 3:
            fig, ax = plt.subplots(figsize=(7, 7))
            for _, row in merged.iterrows():
                color = get_variant_color(row['model'])
                ax.scatter(row['qm9_nds'], row['ext_nds'], color=color, s=100, zorder=5,
                           edgecolors='black', linewidth=0.5,
                           label=get_model_label(row['model']))

            r, p = stats.pearsonr(merged['qm9_nds'], merged['ext_nds'])
            slope, intercept = np.polyfit(merged['qm9_nds'], merged['ext_nds'], 1)
            x_range = np.linspace(merged['qm9_nds'].min(), merged['qm9_nds'].max(), 100)
            ax.plot(x_range, slope * x_range + intercept, '--', color='grey', alpha=0.6, linewidth=1.5)

            ax.set_xlabel('QM9 Mean NDS', fontsize=12)
            ax.set_ylabel('External Datasets Mean NDS', fontsize=12)
            ax.set_title(f'Robustness Transferability: QM9 → External (r={r:.2f}, p={p:.3f})',
                         fontweight='bold', fontsize=12)
            ax.axhline(0, color='grey', linewidth=0.3)
            ax.legend(fontsize=8, loc='best', framealpha=0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Tight axis around data (don't extend to 0)
            x_pad = (merged['qm9_nds'].max() - merged['qm9_nds'].min()) * 0.15
            y_pad = (merged['ext_nds'].max() - merged['ext_nds'].min()) * 0.15
            ax.set_xlim(merged['qm9_nds'].min() - x_pad, merged['qm9_nds'].max() + x_pad)
            ax.set_ylim(merged['ext_nds'].min() - y_pad, merged['ext_nds'].max() + y_pad)

            plt.tight_layout()
            plt.savefig(output_dir / 'fig_validation_qm9_transferability.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved fig_validation_qm9_transferability.png")

            # Save correlation table
            merged_sorted = merged.sort_values('qm9_nds', ascending=False)
            merged_sorted['model_label'] = merged_sorted['model'].apply(get_model_label)
            merged_sorted.to_csv(output_dir / 'table_validation_qm9_correlation.csv', index=False)
            print("✓ Saved table_validation_qm9_correlation.csv")

    # --- Figure: Per-model robustness profile across datasets (incl. QM9) ---
    if val_nds_df is not None and 'dataset' in val_nds_df.columns:
        # Grouped bar chart: NDS by model for each dataset + QM9
        models_in_val = sorted(val_nds_df['model'].unique())
        if len(models_in_val) >= 2 and len(datasets) >= 2:
            model_ds = val_nds_df.pivot_table(values='nds', index='model', columns='dataset', aggfunc='mean')
            model_ds = model_ds.dropna(how='all')

            # Add QM9 mean NDS per model (averaged across reps and strategies)
            if qm9_nds_df is not None and len(qm9_nds_df) > 0:
                qm9_model_means = qm9_nds_df.groupby('model')['nds'].mean()
                # Only include models present in validation data
                qm9_col = qm9_model_means.reindex(model_ds.index)
                model_ds.insert(0, 'QM9', qm9_col)

            # Sort by family grouping (MODEL_ORDER)
            family_order = sort_models_by_family(model_ds.index.tolist())
            model_ds = model_ds.loc[family_order]
            model_ds.index = [get_model_label(m) for m in model_ds.index]

            fig, ax = plt.subplots(figsize=(max(10, len(model_ds) * 0.8), 6))
            model_labels = model_ds.index.tolist()
            datasets_list = model_ds.columns.tolist()
            n_models = len(model_labels)
            n_datasets = len(datasets_list)
            x = np.arange(n_models)
            width = 0.8 / n_datasets
            dataset_colors = ['#6BAED6', '#FC8D59', '#66C2A5', '#B39DDB'][:n_datasets]
            for i, dataset in enumerate(datasets_list):
                offset = (i - n_datasets / 2 + 0.5) * width
                values = model_ds[dataset].values
                ax.bar(x + offset, values, width,
                       color=dataset_colors[i], label=dataset)
            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, rotation=45, ha='right')
            ax.set_ylabel('NDS')
            ax.set_title('Model Robustness Across Datasets', fontweight='bold')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.legend(title='Dataset')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(output_dir / 'fig_validation_model_comparison.png',
                        dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved fig_validation_model_comparison.png")

    # --- Combined validation figure (Panel A: grouped bar, Panel B: transferability scatter) ---
    # Requires both the grouped bar data and the transferability scatter data
    if (qm9_nds_df is not None and val_nds_df is not None
            and 'dataset' in val_nds_df.columns):
        # Re-compute model_ds for Panel A
        models_in_val_c = sorted(val_nds_df['model'].unique())
        if len(models_in_val_c) >= 2:
            model_ds_c = val_nds_df.pivot_table(values='nds', index='model', columns='dataset', aggfunc='mean')
            model_ds_c = model_ds_c.dropna(how='all')
            # Reorder external dataset columns to the canonical paper order
            # before prepending QM9, so legends read QM9 → LogD → Caco-2 → hERG.
            ext_cols = _order_validation_datasets(model_ds_c.columns.tolist())
            model_ds_c = model_ds_c[ext_cols]
            if qm9_nds_df is not None and len(qm9_nds_df) > 0:
                qm9_means_c = qm9_nds_df.groupby('model')['nds'].mean()
                qm9_col_c = qm9_means_c.reindex(model_ds_c.index)
                model_ds_c.insert(0, 'QM9', qm9_col_c)
            family_order_c = sort_models_by_family(model_ds_c.index.tolist())
            model_ds_c = model_ds_c.loc[family_order_c]
            model_ds_c.index = [get_model_label(m) for m in model_ds_c.index]
            # Use friendly display labels for the dataset legend (LogD, Caco-2 Efflux, hERG K_i).
            model_ds_c.columns = [VALIDATION_DATASET_LABELS.get(c, c) for c in model_ds_c.columns]

            # Re-compute merged for Panel B
            qm9_mn = qm9_nds_df.groupby('model')['nds'].mean().reset_index().rename(columns={'nds': 'qm9_nds'})
            ext_mn = val_nds_df.groupby('model')['nds'].mean().reset_index().rename(columns={'nds': 'ext_nds'})
            merged_c = qm9_mn.merge(ext_mn, on='model', how='inner').dropna(subset=['qm9_nds', 'ext_nds'])

            if len(merged_c) >= 3 and len(model_ds_c) >= 2:
                # Stack panels vertically so each gets full textwidth.
                fig_comb, (ax_a, ax_b) = plt.subplots(
                    2, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 1.25))

                # Panel A: grouped bar
                ml = model_ds_c.index.tolist()
                dl = model_ds_c.columns.tolist()
                x_c = np.arange(len(ml))
                w_c = 0.8 / len(dl)
                dc = ['#6BAED6', '#FC8D59', '#66C2A5', '#B39DDB'][:len(dl)]
                for i, ds in enumerate(dl):
                    offset = (i - len(dl) / 2 + 0.5) * w_c
                    ax_a.bar(x_c + offset, model_ds_c[ds].values, w_c, color=dc[i], label=ds)
                ax_a.set_xticks(x_c)
                ax_a.set_xticklabels(ml, rotation=45, ha='right')
                ax_a.set_ylabel('NDS')
                ax_a.set_title('a) Model Robustness Across Datasets', fontweight='bold')
                ax_a.axhline(0, color='black', linewidth=0.5)
                # Bars hang below 0 — open headroom above it and put the legend
                # there so it never lands on the data.
                _ymin = ax_a.get_ylim()[0]
                ax_a.set_ylim(_ymin, abs(_ymin) * 0.34)
                ax_a.legend(title='Dataset', loc='upper center', ncol=len(dl),
                            framealpha=0.9, fontsize=9)
                ax_a.spines['top'].set_visible(False)
                ax_a.spines['right'].set_visible(False)

                # Panel B: QM9 vs external NDS scatter (no regression line).
                # Use get_variant_color so RF and QRF are visually distinct
                # (MODEL_COLORS maps both to the family blue).
                for _, row in merged_c.iterrows():
                    color = get_variant_color(row['model'])
                    ax_b.scatter(row['qm9_nds'], row['ext_nds'], color=color, s=100, zorder=5,
                                 edgecolors='black', linewidth=0.5,
                                 label=get_model_label(row['model']))
                ax_b.set_xlabel('QM9 Mean NDS')
                ax_b.set_ylabel('External Datasets Mean NDS')
                ax_b.set_title('b) QM9 vs External Robustness', fontweight='bold')
                ax_b.axhline(0, color='grey', linewidth=0.3)
                ax_b.axvline(0, color='grey', linewidth=0.3)
                # Multi-column legend below panel B so it stays within textwidth
                _ordered_legend(ax_b, 5, loc='upper center', bbox_to_anchor=(0.5, -0.18),
                                frameon=False, columnspacing=0.8,
                                handletextpad=0.3)
                ax_b.spines['top'].set_visible(False)
                ax_b.spines['right'].set_visible(False)
                xp = (merged_c['qm9_nds'].max() - merged_c['qm9_nds'].min()) * 0.15
                yp = (merged_c['ext_nds'].max() - merged_c['ext_nds'].min()) * 0.15
                ax_b.set_xlim(merged_c['qm9_nds'].min() - xp, merged_c['qm9_nds'].max() + xp)
                ax_b.set_ylim(merged_c['ext_nds'].min() - yp, merged_c['ext_nds'].max() + yp)

                plt.tight_layout()
                plt.savefig(output_dir / 'fig_validation_combined.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Saved fig_validation_combined.png")

    # --- Dataset-size effect analysis (Fig 8 reviewer rebuttal) ---
    # *** PAPER-INSERTION FLAG: dataset-size-effect ***
    # The supervisor asked whether "ensemble-dependent models degrade on smaller
    # datasets" is actually a size effect, or could be confounded with task
    # difficulty (baseline R²). This function builds the table and figure needed
    # to defend that claim. Once DATASET_N_TRAIN is filled in, the outputs
    # (table_validation_size_effect.csv, fig_validation_size_effect.png) are the
    # artefacts to cite in the rebuttal letter / supplement.
    compute_validation_size_effect(qm9_nds_df, val_nds_df, output_dir)


def compute_validation_size_effect(qm9_nds_df, val_nds_df, output_dir):
    """Test whether NDS depends on training-set size, partialing out task difficulty.

    Builds a (model × dataset) panel of (mean_NDS, baseline_R², log N_train) and:
      1. Per-model Pearson r and Spearman ρ of NDS vs log(N_train) across QM9
         + external datasets. Strongly negative for ensemble-dependent models
         (XGBoost, LightGBM, RF) supports the "degrades on small data" claim.
      2. Partial correlation NDS vs log(N) controlling for baseline R² — if the
         size effect persists after controlling for difficulty, it's a real
         size effect rather than a confound.
      3. Small scatter (one panel per model) of mean_NDS vs log(N_train).

    Outputs:
      table_validation_size_effect.csv    — per-model correlations + panel data
      fig_validation_size_effect.png      — scatter, one panel per model

    Requires DATASET_N_TRAIN to be populated with non-None values for the
    external datasets. Skips with a warning if any are still placeholders.
    """
    missing = [k for k, v in DATASET_N_TRAIN.items() if v is None]
    if missing:
        print(f"  ⚠ Skipping dataset-size-effect analysis — DATASET_N_TRAIN missing: {missing}")
        print(f"    Fill in per-fold train sizes from KIRBy SLURM logs ('Train: X' line).")
        return
    if qm9_nds_df is None or val_nds_df is None or 'dataset' not in val_nds_df.columns:
        print("  ⚠ Skipping dataset-size-effect analysis — missing QM9 or validation NDS data")
        return

    # Per-model mean NDS + baseline R² on each dataset
    qm9_panel = qm9_nds_df.groupby('model').agg(
        nds=('nds', 'mean'),
        baseline_r2=('baseline_r2', 'mean') if 'baseline_r2' in qm9_nds_df.columns else ('nds', 'mean'),
    ).reset_index()
    qm9_panel['dataset'] = 'QM9'

    val_panel = val_nds_df.groupby(['dataset', 'model']).agg(
        nds=('nds', 'mean'),
        baseline_r2=('baseline_r2', 'mean') if 'baseline_r2' in val_nds_df.columns else ('nds', 'mean'),
    ).reset_index()

    panel = pd.concat([qm9_panel, val_panel], ignore_index=True)
    panel['n_train'] = panel['dataset'].map(DATASET_N_TRAIN)
    panel = panel.dropna(subset=['n_train', 'nds'])
    panel['log_n_train'] = np.log(panel['n_train'])

    # Per-model correlations (need ≥3 datasets per model)
    rows = []
    for model, grp in panel.groupby('model'):
        if len(grp) < 3:
            continue
        try:
            r_pearson, p_pearson = stats.pearsonr(grp['log_n_train'], grp['nds'])
        except Exception:
            r_pearson, p_pearson = np.nan, np.nan
        try:
            r_spearman, p_spearman = stats.spearmanr(grp['log_n_train'], grp['nds'])
        except Exception:
            r_spearman, p_spearman = np.nan, np.nan
        # Partial r(NDS, log_n | baseline_r2) — only if baseline_r2 has variance
        partial_r = np.nan
        if grp['baseline_r2'].nunique() > 1:
            try:
                r_yz = stats.pearsonr(grp['nds'], grp['baseline_r2'])[0]
                r_xz = stats.pearsonr(grp['log_n_train'], grp['baseline_r2'])[0]
                denom = np.sqrt((1 - r_yz ** 2) * (1 - r_xz ** 2))
                if denom > 0:
                    partial_r = (r_pearson - r_yz * r_xz) / denom
            except Exception:
                pass
        rows.append({
            'model': model,
            'n_datasets': len(grp),
            'pearson_r_NDS_vs_logN': r_pearson,
            'pearson_p': p_pearson,
            'spearman_rho_NDS_vs_logN': r_spearman,
            'spearman_p': p_spearman,
            'partial_r_NDS_vs_logN_given_R2': partial_r,
        })

    if not rows:
        print("  ⚠ dataset-size-effect: no model has ≥3 datasets")
        return

    corr_df = pd.DataFrame(rows).sort_values('pearson_r_NDS_vs_logN')
    corr_df['model_label'] = corr_df['model'].apply(get_model_label)
    corr_df.to_csv(output_dir / 'table_validation_size_effect.csv', index=False)
    print("✓ Saved table_validation_size_effect.csv")

    # Also dump the underlying panel so the user can re-plot manually
    panel_out = panel.copy()
    panel_out['model_label'] = panel_out['model'].apply(get_model_label)
    panel_out.to_csv(output_dir / 'table_validation_size_effect_panel.csv', index=False)

    # Figure: NDS vs log(N_train) per model, one panel per model
    models = corr_df['model'].tolist()
    n_models = len(models)
    if n_models == 0:
        return
    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes_se = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows),
                                 squeeze=False)
    axes_flat = axes_se.flatten()
    for i, model in enumerate(models):
        ax = axes_flat[i]
        grp = panel[panel['model'] == model]
        color = get_variant_color(model)
        ax.scatter(grp['log_n_train'], grp['nds'], color=color, s=60,
                   edgecolors='black', linewidth=0.4, zorder=5)
        for _, row in grp.iterrows():
            ax.annotate(row['dataset'][:6], (row['log_n_train'], row['nds']),
                        fontsize=6, xytext=(3, 3), textcoords='offset points')
        # Fit line
        if len(grp) >= 2:
            slope, intercept = np.polyfit(grp['log_n_train'], grp['nds'], 1)
            x_fit = np.linspace(grp['log_n_train'].min(), grp['log_n_train'].max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, '--', color=color, alpha=0.5, linewidth=1)
        r_row = corr_df[corr_df['model'] == model].iloc[0]
        ax.set_title(f"{get_model_label(model)}  r={r_row['pearson_r_NDS_vs_logN']:.2f}",
                     fontsize=9, fontweight='bold')
        ax.axhline(0, color='grey', linewidth=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    for j in range(n_models, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.supxlabel('log(per-fold N_train)')
    fig.supylabel('Mean NDS across strategies')
    plt.suptitle('Dataset-size effect on noise robustness',
                 fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_validation_size_effect.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig_validation_size_effect.png")


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


def wilcoxon_paired_test(nds_df, model_base, model_variant, rep=None, strategy='legacy'):
    """
    Wilcoxon signed-rank test for paired model comparison.

    Pairs models across all shared (rep, strategy) conditions to get enough
    matched observations. Falls back to single rep×strategy if pairing across
    conditions yields too few points.

    Returns:
        dict with statistic, p_value, and interpretation
    """
    if rep is None:
        rep = PRIMARY_REP
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


def calculate_kendalls_w(nds_df, rep=None):
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
    if rep is None:
        rep = PRIMARY_REP
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

    # Enforce balanced design: restrict to models present in ALL reps
    models = df_sigma['model'].unique()
    reps = df_sigma['rep'].unique()
    cell_presence = df_sigma.groupby(['model', 'rep']).size()
    valid_models = [m for m in models if all((m, r) in cell_presence.index for r in reps)]
    dropped = sorted(set(models) - set(valid_models))
    if dropped:
        print(f"  ⚠ Performance ANOVA (σ={sigma_value}): dropping {len(dropped)} models "
              f"with missing reps: {dropped}")
        df_sigma = df_sigma[df_sigma['model'].isin(valid_models)]
    print(f"  Performance ANOVA (σ={sigma_value}): {len(valid_models)} models × {len(reps)} reps "
          f"({len(df_sigma)} rows)")

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

    # Enforce balanced design: restrict to models present in ALL reps
    models = nds_df['model'].unique()
    reps = nds_df['rep'].unique()
    cell_presence = nds_df.groupby(['model', 'rep']).size()
    valid_models = [m for m in models if all((m, r) in cell_presence.index for r in reps)]
    dropped = sorted(set(models) - set(valid_models))
    if dropped:
        print(f"  ⚠ Robustness ANOVA: dropping {len(dropped)} models "
              f"with missing reps: {dropped}")
        nds_df = nds_df[nds_df['model'].isin(valid_models)]
    print(f"  Robustness ANOVA: {len(valid_models)} models × {len(reps)} reps "
          f"({len(nds_df)} rows)")

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
    # Still exclude PDV-only models (e.g. gauche_rbf) which lack cross-rep data
    all_rows_full = []
    for strategy in df['strategy'].unique():
        strategy_df = df[(df['strategy'] == strategy) & (~df['model'].isin(PDV_ONLY_MODELS))]
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

        # Generate LaTeX table (high-correlation pairs only)
        high_corr = mp_df[mp_df['spearman_rho'] >= 0.85].copy()
        if len(high_corr) > 0:
            lines_tex = []
            lines_tex.append(r'\begin{tabular}{llrl}')
            lines_tex.append(r'\toprule')
            lines_tex.append(r'\textbf{Model A} & \textbf{Model B} & \textbf{$\rho$} & \textbf{Excluded} \\')
            lines_tex.append(r'\midrule')
            for _, row in high_corr.iterrows():
                ma = get_model_label(row['model_a'])
                mb = get_model_label(row['model_b'])
                rho = f"{row['spearman_rho']:.3f}"
                excl = 'Yes' if row['excluded_from_anova'] == 'yes' else ''
                lines_tex.append(f'{ma} & {mb} & {rho} & {excl} \\\\')
            lines_tex.append(r'\bottomrule')
            lines_tex.append(r'\end{tabular}')

            tex_path = output_dir / 'table_supp_model_redundancy.tex'
            tex_path.write_text('\n'.join(lines_tex))
            print(f"✓ Saved table_supp_model_redundancy.tex")

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

        # Generate LaTeX table (all rep pairs)
        lines_tex = []
        lines_tex.append(r'\begin{tabular}{llrl}')
        lines_tex.append(r'\toprule')
        lines_tex.append(r'\textbf{Rep A} & \textbf{Rep B} & \textbf{$\rho$} & \textbf{Excluded} \\')
        lines_tex.append(r'\midrule')
        for _, row in rp_df.iterrows():
            ra = get_rep_label(row['rep_a'])
            rb = get_rep_label(row['rep_b'])
            rho = f"{row['spearman_rho']:.3f}"
            excl = 'Yes' if row['excluded_from_anova'] == 'yes' else ''
            lines_tex.append(f'{ra} & {rb} & {rho} & {excl} \\\\')
        lines_tex.append(r'\bottomrule')
        lines_tex.append(r'\end{tabular}')

        tex_path = output_dir / 'table_supp_rep_redundancy.tex'
        tex_path.write_text('\n'.join(lines_tex))
        print(f"✓ Saved table_supp_rep_redundancy.tex")

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

        # Generate LaTeX table (high-ICC pairs)
        high_icc_tex = icc_df[icc_df['icc_1_1'] >= 0.7].copy()
        if len(high_icc_tex) > 0:
            lines_tex = []
            lines_tex.append(r'\begin{tabular}{llrr}')
            lines_tex.append(r'\toprule')
            lines_tex.append(r'\textbf{Model A} & \textbf{Model B} & \textbf{ICC(1,1)} & \textbf{Mean $|\Delta$NDS$|$} \\')
            lines_tex.append(r'\midrule')
            for _, row in high_icc_tex.iterrows():
                ma = get_model_label(row['model_a'])
                mb = get_model_label(row['model_b'])
                icc_val = f"{row['icc_1_1']:.3f}"
                mad = f"{row['mean_abs_nds_diff']:.4f}"
                lines_tex.append(f'{ma} & {mb} & {icc_val} & {mad} \\\\')
            lines_tex.append(r'\bottomrule')
            lines_tex.append(r'\end{tabular}')

            tex_path = output_dir / 'table_supp_icc.tex'
            tex_path.write_text('\n'.join(lines_tex))
            print(f"✓ Saved table_supp_icc.tex")

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

    strategies = list(STRATEGY_ORDER)
    sigma = 0.5  # Representative noise level

    # Pre-compute one bin grid wide enough to contain every panel's noisy
    # distribution. Without this, recomputing bins per panel from
    # (y_clean ∪ y_noisy) shifts bin edges between panels — re-binning the
    # SAME y_clean against shifted edges then produces visibly different
    # "clean" outlines across panels even though the data is identical.
    # Save/restore RNG state so each strategy gets a reproducible draw and
    # the real plotting loop below sees the same RNG as before this fix.
    rng_state = np.random.get_state()
    all_y = [y_clean]
    for strategy in strategies:
        np.random.set_state(rng_state)
        all_y.append(apply_noise(y_clean, sigma, strategy))
    np.random.set_state(rng_state)
    lo = min(arr.min() for arr in all_y)
    hi = max(arr.max() for arr in all_y)
    bins = np.linspace(lo, hi, 51)

    # 3x2 grid (portrait page: 3 rows tall rather than 3 columns wide)
    fig, axes = plt.subplots(3, 2, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.95))
    axes = axes.flatten()

    panel_labels = ['a)', 'b)', 'c)', 'd)', 'e)', 'f)']

    for i, strategy in enumerate(strategies):
        ax = axes[i]
        y_noisy = apply_noise(y_clean, sigma, strategy)

        # Filled histograms: very transparent so overlap is visible
        ax.hist(y_clean, bins=bins, alpha=0.15, color=CLEAN_COLOR, density=True)
        ax.hist(y_noisy, bins=bins, alpha=0.15, color=STRATEGY_COLORS[strategy], density=True)

        # Step outlines on TOP of curves (slightly transparent so crossings visible)
        ax.hist(y_clean, bins=bins, density=True,
                histtype='step', linewidth=2.0, color=CLEAN_COLOR, alpha=0.7)
        ax.hist(y_noisy, bins=bins, density=True,
                histtype='step', linewidth=2.0, color=STRATEGY_COLORS[strategy], alpha=0.7)

        ax.set_title(STRATEGY_LABELS[strategy], fontweight='bold',
                     color=STRATEGY_COLORS[strategy])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Add vertical headroom so the RMSE annotation clears the histogram bars.
        ax.set_ylim(top=ax.get_ylim()[1] * 1.30)
        rmse = np.sqrt(np.mean((y_noisy - y_clean)**2))
        ax.text(0.97, 0.96, f'RMSE={rmse:.2f}', transform=ax.transAxes,
                ha='right', va='top')

        # Panel label (a), (b), ... in upper-left, outside the title.
        ax.text(-0.10, 1.08, panel_labels[i], transform=ax.transAxes,
                fontweight='bold', ha='left', va='bottom')

    plt.tight_layout()

    plt.savefig(output_dir / 'fig_methods_noise_strategies.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved fig_methods_noise_strategies.png")

    # Detailed sigma progression version cut (was fig_methods_noise_strategies_detailed.png)


# =============================================================================
# FIGURE 1: GLOBAL OVERVIEW
# =============================================================================

def create_figure1(df, nds_df, output_dir):
    """Figure 1: Global Overview - R² vs σ line plot + NDS heatmap."""
    # Stack panels vertically so each one gets full textwidth — same rationale
    # as Figs 2/3: side-by-side at \textwidth squeezed the heatmap and legend.
    fig = plt.figure(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 1.25))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1.2])

    # Panel A: R² vs σ for key models on PDV
    ax_a = fig.add_subplot(gs[0])

    key_models = ['rf', 'qrf', 'dnn', 'mlp', 'ngboost', 'xgboost', 'gauche_rbf']
    pdv_data = df[(df['rep'] == PRIMARY_REP) & (df['strategy'] == 'legacy')]

    # Plot key models only (no grey background lines)

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
    ax_a.set_title('a) Performance Degradation (PDV, Gaussian Noise)', fontweight='bold')
    _ordered_legend(ax_a, 2, loc='lower left')
    ax_a.set_ylim(0.3, 1.0)
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)

    # Panel B: NDS heatmap (model × strategy) - PDV ONLY to avoid mixing representations
    ax_b = fig.add_subplot(gs[1])

    if len(nds_df) > 0:
        # Filter to PDV only - don't mix representations
        nds_pdv = nds_df[nds_df['rep'] == PRIMARY_REP]
        if len(nds_pdv) == 0:
            # Fallback to most common rep
            nds_pdv = nds_df

        # Build full pivot (all models × all strategies) so missing cells appear
        # Exclude Tanimoto GP from PDV heatmap (kernel incompatible with continuous features)
        all_models = sort_models_by_family([m for m in nds_df['model'].unique() if m != 'gauche'])
        all_strategies = [s for s in STRATEGY_ORDER
                          if s in nds_df['strategy'].unique()]
        pivot = nds_pdv.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        pivot = pivot.reindex(index=all_models, columns=all_strategies)

        # Create annotations distinguishing missing vs filtered
        annot_text = make_heatmap_annotations(pivot, df, 'model', 'strategy',
                                               rep_filter=PRIMARY_REP, fmt='.2f')

        # Rename index/columns to clean labels
        pivot.index = [get_model_label(m) for m in pivot.index]
        annot_text.index = pivot.index
        col_order = all_strategies
        pivot.columns = [STRATEGY_LABELS.get(c, c) for c in col_order]
        annot_text.columns = pivot.columns

        # Use colorblind-friendly diverging colormap centered at data midpoint
        vals = pivot.values[~np.isnan(pivot.values)]
        center_val = (vals.min() + vals.max()) / 2 if len(vals) > 0 else 0
        ax_b.set_facecolor('black')
        sns.heatmap(pivot, annot=annot_text, fmt='', cmap='PuOr', center=center_val,
                    ax=ax_b, cbar_kws={'label': 'NDS'}, linewidths=0.5)
        _white_text_for_missing(ax_b, pivot, annot_text)
        ax_b.set_xlabel('Noise Strategy')
        ax_b.set_ylabel('Model')
        ax_b.set_title('b) NDS by Model × Strategy (PDV)', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'fig1_global_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig1_global_overview.png")

    # --- ECFP4 variant (Issue C: compare to PDV; supplementary unless very different) ---
    fig_e, axes_e = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 1.25))
    ax_ea = axes_e[0]

    ecfp4_data = df[(df['rep'] == 'ecfp4') & (df['strategy'] == 'legacy')]

    # Plot key models only (no grey background lines)

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
    ax_ea.set_title('a) Performance Degradation (ECFP4, Gaussian Noise)', fontweight='bold')
    _ordered_legend(ax_ea, 2, loc='lower left')
    ax_ea.set_ylim(0.3, 1.0)
    ax_ea.spines['top'].set_visible(False)
    ax_ea.spines['right'].set_visible(False)

    ax_eb = axes_e[1]
    if len(nds_df) > 0:
        nds_ecfp4 = nds_df[nds_df['rep'] == 'ecfp4']
        if len(nds_ecfp4) > 0:
            all_strategies = [s for s in STRATEGY_ORDER
                              if s in nds_df['strategy'].unique()]
            all_models_e = sort_models_by_family([m for m in nds_df['model'].unique()
                                                     if m not in PDV_ONLY_MODELS])
            pivot = nds_ecfp4.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
            pivot = pivot.reindex(index=all_models_e, columns=all_strategies)

            annot_text = make_heatmap_annotations(pivot, df, 'model', 'strategy',
                                                   rep_filter='ecfp4', fmt='.2f')
            pivot.index = [get_model_label(m) for m in pivot.index]
            annot_text.index = pivot.index
            pivot.columns = [STRATEGY_LABELS.get(c, c) for c in all_strategies]
            annot_text.columns = pivot.columns

            vals_e = pivot.values[~np.isnan(pivot.values)]
            cv_e = (vals_e.min() + vals_e.max()) / 2 if len(vals_e) > 0 else 0
            ax_eb.set_facecolor('black')
            sns.heatmap(pivot, annot=annot_text, fmt='', cmap='PuOr', center=cv_e,
                        ax=ax_eb, cbar_kws={'label': 'NDS'}, linewidths=0.5)
            _white_text_for_missing(ax_eb, pivot, annot_text)
            ax_eb.set_xlabel('Noise Strategy')
            ax_eb.set_ylabel('Model')
            ax_eb.set_title('b) NDS by Model × Strategy (ECFP4)', fontweight='bold')

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

    # Stack panels vertically so each one is full-textwidth — keeps bar labels
    # legible at print scale instead of being squeezed into a half-width column.
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85),
                             sharex=True)

    # Preferred strategy order: Gaussian first, then concordant 4, then outlier
    strat_order = ['legacy', 'valprop', 'quantile', 'threshold', 'hetero', 'outlier']

    def _plot_anova_panel(ax, results, title, show_legend=False):
        strats = [s for s in strat_order if s in results]
        if not strats:
            return

        x = np.arange(len(strats))
        width = 0.25

        model_vals = [results[s]['eta2_model'] for s in strats]
        rep_vals = [results[s]['eta2_rep'] for s in strats]
        int_vals = [results[s]['eta2_interaction'] for s in strats]

        ax.bar(x - width, model_vals, width, label='Model',
               color=ANOVA_FACTOR_COLORS['Model'])
        ax.bar(x, rep_vals, width, label='Representation',
               color=ANOVA_FACTOR_COLORS['Representation'])
        ax.bar(x + width, int_vals, width, label='Interaction',
               color=ANOVA_FACTOR_COLORS['Interaction'])

        ax.set_ylabel('Variance Explained (η², %)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        labels = [STRATEGY_LABELS.get(s, s) for s in strats]
        ax.set_xticklabels(labels, rotation=45, ha='right')
        if show_legend:
            ax.legend(loc='upper right', ncol=3)
        ax.set_ylim(0, 100)

    # Single shared legend on the top panel only (factors are identical across A and B).
    _plot_anova_panel(axes[0], perf_results, 'a) Performance (R² at σ=0.3)', show_legend=True)
    _plot_anova_panel(axes[1], robust_results, 'b) Robustness (NDS)', show_legend=False)

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
            row['Perf_Interaction_η²'] = perf_results[s]['eta2_interaction']
            row['Perf_Residual_η²'] = perf_results[s]['eta2_residual']
        if s in robust_results:
            row['Robust_Model_η²'] = robust_results[s]['eta2_model']
            row['Robust_Rep_η²'] = robust_results[s]['eta2_rep']
            row['Robust_Interaction_η²'] = robust_results[s]['eta2_interaction']
            row['Robust_Residual_η²'] = robust_results[s]['eta2_residual']
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

def create_figure3(nds_df, validation_df, val_nds_df, raw_df, output_dir):
    """Figure 3: Ranking consistency — Baseline R² vs NDS scatter (PDV, Gaussian)."""
    # Single panel sized to textwidth so labels stay 10pt on the page.
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.85))

    nds_pdv = nds_df[nds_df['rep'] == PRIMARY_REP] if 'rep' in nds_df.columns else nds_df
    nds_pdv_legacy = nds_pdv[nds_pdv['strategy'] == 'legacy'] if 'strategy' in nds_pdv.columns else nds_pdv

    for model in sort_models_by_family(nds_pdv_legacy['model'].unique().tolist()):
        model_data = nds_pdv_legacy[nds_pdv_legacy['model'] == model]
        color = MODEL_COLORS.get(model, '#333333')
        marker = MODEL_MARKERS.get(model, 'o')
        ax.scatter(model_data['baseline_r2'], model_data['nds'],
                   label=get_model_label(model), color=color, marker=marker, alpha=0.7, s=50)

    ax.autoscale()
    ax.margins(x=0.08, y=0.08)
    ax.set_xlabel('Baseline R² (σ=0)')
    ax.set_ylabel('NDS (slope)')
    ax.set_title('Baseline vs Robustness (PDV, Gaussian)', fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.5)
    _ordered_legend(ax, 4, loc='upper left', bbox_to_anchor=(0.0, -0.15),
                    borderaxespad=0, frameon=False, columnspacing=1.0,
                    handletextpad=0.5, labelspacing=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig3_ranking_consistency.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved fig3_ranking_consistency.png")

    # ECFP4 supplementary ranking figure cut (was fig3_supp_ecfp4_ranking.png)


# =============================================================================
# FIGURE: NN FAMILY COMPARISON (combines old fig4 + fig5)
# =============================================================================

def create_nn_family_comparison(df, nds_df, output_dir):
    """Combined NN family comparison: NN-α, NN-β, and RF families under Gaussian noise.

    1×3 subplot:
      Panel A: NN-α family (NN-α, BNN-α, VBLL-α)
      Panel B: NN-β family (NN-β, BNN-β, VBLL-β)
      Panel C: RF vs QRF
    All on PRIMARY_REP, Gaussian strategy only.
    """
    dnn_variants = ['dnn', 'dnn_bnn_full', 'dnn_vbll']
    mlp_variants = ['mlp', 'mlp_bnn_full', 'mlp_vbll']
    rf_models = ['rf', 'qrf']

    # textwidth-wide 1×3 panel; single shared legend below all three panels
    # (per-panel legends were redundant and ate plot area).
    # Stacked vertically (portrait page): three full-width panels.
    fig, axes = plt.subplots(3, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 1.05),
                             sharex=True)
    strategy = PRIMARY_STRATEGY
    strategy_label = STRATEGY_LABELS.get(strategy, strategy)
    data = df[(df['rep'] == PRIMARY_REP) & (df['strategy'] == strategy)]

    families = [
        ('a', 'NN-α Family', dnn_variants, DNN_FAMILY_COLORS),
        ('b', 'NN-β Family', mlp_variants, MLP_FAMILY_COLORS),
        ('c', 'RF vs QRF', rf_models, RF_FAMILY_COLORS),
    ]

    handle_by_model = {}
    for idx, (panel, title, models, colors) in enumerate(families):
        ax = axes[idx]
        for model in models:
            model_data = data[data['model'] == model]
            if len(model_data) == 0:
                continue
            avg = model_data.groupby('sigma')['r2'].mean().reset_index()
            color = colors.get(model, '#333333')
            marker = MODEL_MARKERS.get(model, 'o')
            line, = ax.plot(avg['sigma'], avg['r2'], marker=marker, linestyle='-',
                            label=get_model_label(model), color=color, markersize=5)
            handle_by_model.setdefault(model, line)

        if idx == len(families) - 1:
            ax.set_xlabel('Noise Level (σ)')
        ax.set_ylabel('R²')
        ax.set_title(f'{panel}) {title}', fontweight='bold')
        ax.set_ylim(0.3, 1.0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # matplotlib fills multi-column legends COLUMN-major (top-to-bottom, then
    # the next column). To line up matched α/β pairs on each row, put the α
    # family in the left column and the β family in the right column, then pass
    # handles column-by-column (all of col-0, then all of col-1) with ncol=2:
    #   row0: NN-α   | NN-β
    #   row1: BNN-α  | BNN-β
    #   row2: VBLL-α | VBLL-β
    #   row3: RF     | QRF
    alpha_col = ['dnn', 'dnn_bnn_full', 'dnn_vbll', 'rf']
    beta_col = ['mlp', 'mlp_bnn_full', 'mlp_vbll', 'qrf']
    ordered_models = [m for m in alpha_col if m in handle_by_model] + \
                     [m for m in beta_col if m in handle_by_model]
    all_handles = [handle_by_model[m] for m in ordered_models]
    all_labels = [get_model_label(m) for m in ordered_models]
    fig.legend(all_handles, all_labels, loc='lower center',
               bbox_to_anchor=(0.5, 0.01), ncol=2,
               frameon=False, columnspacing=3.0, handletextpad=0.4)
    # The paired 2-column legend is 4 rows tall; reserve enough bottom margin so
    # it clears panel c's x-axis ticks and "Noise Level (σ)" title (6% overlapped
    # them). bbox_inches='tight' on save trims any surplus whitespace.
    plt.tight_layout(rect=[0, 0.20, 1, 1])
    plt.savefig(output_dir / 'fig_nn_family_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved fig_nn_family_comparison.png ({strategy_label}, {get_rep_label(PRIMARY_REP)})")

    # Output NDS table for all three families
    all_family_models = dnn_variants + mlp_variants + rf_models
    family_nds = nds_df[(nds_df['model'].isin(all_family_models)) & (nds_df['rep'] == PRIMARY_REP)]
    if len(family_nds) > 0:
        nds_table = family_nds.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
        nds_table = nds_table.reindex([m for m in all_family_models if m in nds_table.index])
        nds_table.index = [get_model_label(m) for m in nds_table.index]
        nds_table.columns = [STRATEGY_LABELS.get(s, s) for s in nds_table.columns]
        nds_table.to_csv(output_dir / 'table_nn_family_nds.csv')
        print("✓ Saved table_nn_family_nds.csv")


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

    # Pre-filter: exclude models with negligible uncertainty (e.g. plain NN-α/NN-β)
    valid_models = []
    for model in filtered['model'].unique():
        mdata = filtered[filtered['model'] == model]
        uvals = mdata[unc_col].values
        finite_mask = np.isfinite(uvals)
        if finite_mask.sum() > 100 and uvals[finite_mask].mean() > 1e-3:
            valid_models.append(model)
    filtered = filtered[filtered['model'].isin(valid_models)]

    # Check if any model has aleatoric/epistemic decomposition data
    has_decomposition = ('aleatoric_uncertainty' in filtered.columns and
                         'epistemic_uncertainty' in filtered.columns)
    any_decomposition = False
    if has_decomposition:
        for model in filtered['model'].unique():
            mdata = filtered[filtered['model'] == model]
            alea = mdata['aleatoric_uncertainty'].values
            epis = mdata['epistemic_uncertainty'].values
            if (np.isfinite(alea).sum() > 10 and np.nanmean(alea[np.isfinite(alea)]) > 1e-6 and
                np.isfinite(epis).sum() > 10 and np.nanmean(epis[np.isfinite(epis)]) > 1e-6):
                any_decomposition = True
                break

    if any_decomposition:
        # Stacked vertically (portrait page): each panel gets full textwidth.
        fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.9),
                                         sharex=True)
    else:
        fig, ax_a = plt.subplots(1, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 0.6))

    # --- Panel A: Mean uncertainty vs noise level ---
    for model in sort_models_by_family(filtered['model'].unique().tolist()):
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
            color = UNCERTAINTY_COLORS.get(model, MODEL_COLORS.get(model, '#333333'))
            marker = MODEL_MARKERS.get(model, 'o')
            ax_a.plot(sigma_df['sigma'], sigma_df['mean_unc'],
                    marker=marker, linestyle='-',
                    label=get_model_label(model), color=color,
                    markersize=4, linewidth=1.2, alpha=0.8)

    if not any_decomposition:
        ax_a.set_xlabel('Injected Noise Level (σ)')
        ax_a.set_ylabel('Mean Predicted Uncertainty')
    ax_a.set_title('a) Total Uncertainty vs Noise Level', fontweight='bold')
    ax_a.spines['top'].set_visible(False)
    ax_a.spines['right'].set_visible(False)
    # Per-axis legend deferred — built as a single shared legend below the
    # whole figure after both panels are populated.

    # --- Panel B: Aleatoric vs Epistemic decomposition ---
    if any_decomposition:
        for model in sort_models_by_family(filtered['model'].unique().tolist()):
            model_data = filtered[filtered['model'] == model]
            alea = model_data['aleatoric_uncertainty'].values
            epis = model_data['epistemic_uncertainty'].values
            valid_mask = (np.isfinite(alea) & np.isfinite(epis) &
                         (alea > 1e-6) & (epis > 1e-6))
            if valid_mask.sum() < 10:
                continue

            color = UNCERTAINTY_COLORS.get(model, MODEL_COLORS.get(model, '#333333'))
            label = get_model_label(model)
            marker = MODEL_MARKERS.get(model, 'o')
            sigmas = sorted(model_data['sigma'].unique())
            if len(sigmas) < 3:
                continue

            alea_means, epis_means = [], []
            for sigma in sigmas:
                sigma_data = model_data[model_data['sigma'] == sigma]
                a = sigma_data['aleatoric_uncertainty'].values
                e = sigma_data['epistemic_uncertainty'].values
                a_valid = a[np.isfinite(a) & (a > 1e-6)]
                e_valid = e[np.isfinite(e) & (e > 1e-6)]
                if len(a_valid) > 0 and len(e_valid) > 0:
                    alea_means.append(a_valid.mean())
                    epis_means.append(e_valid.mean())
                else:
                    alea_means.append(np.nan)
                    epis_means.append(np.nan)

            # Drop redundant per-model labels here — the shared legend handles
            # model identity; line style alone communicates aleatoric vs epistemic.
            ax_b.plot(sigmas, alea_means, marker=marker, linestyle='-', color=color,
                     markersize=4, linewidth=1.2, alpha=0.8)
            ax_b.plot(sigmas, epis_means, marker=marker, linestyle='--', color=color,
                     markersize=3, linewidth=1.0, alpha=0.8)

        ax_b.set_xlabel('Injected Noise Level (σ)')
        ax_b.set_title('b) Aleatoric vs Epistemic Decomposition', fontweight='bold')
        ax_b.spines['top'].set_visible(False)
        ax_b.spines['right'].set_visible(False)

    # Single shared legend below the figure: one entry per model (no triplication).
    handles, labels = ax_a.get_legend_handles_labels()
    if handles:
        ncol = min(len(labels), 4)
        if any_decomposition:
            # Add line-style key explaining aleatoric vs epistemic
            from matplotlib.lines import Line2D
            style_handles = [
                Line2D([0], [0], color='black', linestyle='-', label='Aleatoric'),
                Line2D([0], [0], color='black', linestyle='--', label='Epistemic'),
            ]
            handles = handles + style_handles
            labels = labels + ['Aleatoric', 'Epistemic']
            ncol = min(len(labels), 5)
        # Reorder so the legend reads left-to-right (matplotlib fills column-major).
        handles = _to_colmajor(handles, ncol)
        labels = _to_colmajor(labels, ncol)
        fig.legend(handles, labels, loc='lower center',
                   bbox_to_anchor=(0.5, -0.06), ncol=ncol,
                   frameon=False, columnspacing=1.0, handletextpad=0.4)
    # Shared y-label for the stacked panels (per-panel labels overlap; the
    # panel titles + caption distinguish total vs aleatoric/epistemic).
    if any_decomposition:
        fig.supylabel('Mean Uncertainty')
    plt.tight_layout(rect=[0, 0.09, 1, 1])
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

def create_tables(nds_df, unc_df, qm9_df, output_dir, val_nds_df=None):
    """Create all summary tables."""

    # Table 2: NDS by model × strategy (PDV only - don't mix representations)
    # Filter to ANOVA-included models for main table
    nds_anova_tbl = nds_df[~nds_df['model'].isin(ANOVA_MODELS_EXCLUDE)]
    if len(nds_anova_tbl) > 0:
        nds_pdv = nds_anova_tbl[nds_anova_tbl['rep'] == PRIMARY_REP] if 'rep' in nds_anova_tbl.columns else nds_anova_tbl

        if len(nds_pdv) > 0:
            pivot = nds_pdv.pivot_table(values='nds', index='model', columns='strategy', aggfunc='mean')
            pivot['MEAN'] = pivot.mean(axis=1)
            pivot['STD'] = pivot.drop(columns=['MEAN']).std(axis=1)
            pivot_labeled = pivot.rename(columns=STRATEGY_LABELS)

            # Variant A: ranked by mean NDS
            variant_a = pivot_labeled.sort_values('MEAN', ascending=False)
            variant_a.to_csv(output_dir / 'table2_nds_by_strategy_pdv.csv')
            print("✓ Saved table2_nds_by_strategy_pdv.csv (ranked by mean)")

            # Variant B: ranked by Gaussian NDS
            gauss_col = STRATEGY_LABELS.get('legacy', 'Gaussian')
            if gauss_col in pivot_labeled.columns:
                variant_b = pivot_labeled.sort_values(gauss_col, ascending=False)
                variant_b.to_csv(output_dir / 'table2_nds_by_gaussian_pdv.csv')
                print("✓ Saved table2_nds_by_gaussian_pdv.csv (ranked by Gaussian)")

            # Variant C: per-strategy ranks and mean rank
            strategy_cols = [c for c in pivot.columns if c not in ('MEAN', 'STD')]
            rank_df = pivot[strategy_cols].rank(ascending=False)  # Higher NDS (less negative) = more robust = rank 1
            rank_df = rank_df.rename(columns=STRATEGY_LABELS)
            rank_df['Mean_Rank'] = rank_df.mean(axis=1)
            rank_df = rank_df.sort_values('Mean_Rank')
            rank_df.to_csv(output_dir / 'table2_nds_ranks_pdv.csv')
            print("✓ Saved table2_nds_ranks_pdv.csv (per-strategy ranks)")

        # Also save full table with all reps for supplementary (ANOVA models only)
        pivot_all = nds_anova_tbl.pivot_table(values='nds', index=['model', 'rep'], columns='strategy', aggfunc='mean')
        pivot_all['MEAN'] = pivot_all.mean(axis=1)
        pivot_all['STD'] = pivot_all.drop(columns=['MEAN']).std(axis=1)
        pivot_all = pivot_all.sort_values('MEAN', ascending=False)
        pivot_all.rename(columns=STRATEGY_LABELS, inplace=True)
        pivot_all.to_csv(output_dir / 'table2_supp_nds_all_reps.csv')
        print("✓ Saved table2_supp_nds_all_reps.csv (supplementary)")

    # Table: Top NDS configurations (model × rep) with per-strategy columns + external validation
    if len(nds_anova_tbl) > 0:
        # Pivot: model×rep with one column per strategy
        top_pivot = nds_anova_tbl.pivot_table(
            values='nds', index=['model', 'rep'], columns='strategy', aggfunc='mean')
        top_pivot.rename(columns=STRATEGY_LABELS, inplace=True)

        gauss_col = STRATEGY_LABELS.get('legacy', 'Gaussian')

        # Add external validation mean NDS if available
        if val_nds_df is not None and 'dataset' in val_nds_df.columns:
            val_mean = val_nds_df.pivot_table(
                values='nds', index=['model', 'rep'], columns='dataset', aggfunc='mean')
            val_mean['Ext. Mean'] = val_mean.mean(axis=1)
            top_pivot = top_pivot.join(val_mean[['Ext. Mean']], how='left')
        else:
            top_pivot['Ext. Mean'] = np.nan

        # Sort by Gaussian NDS (least negative = most robust)
        if gauss_col in top_pivot.columns:
            top_pivot = top_pivot.sort_values(gauss_col, ascending=False)

        # Add display labels
        top_pivot = top_pivot.reset_index()
        top_pivot['Model'] = top_pivot['model'].apply(get_model_label)
        top_pivot['Rep'] = top_pivot['rep'].apply(get_rep_label)

        # Select and order columns
        strat_cols = [STRATEGY_LABELS.get(s, s) for s in STRATEGY_ORDER
                      if STRATEGY_LABELS.get(s, s) in top_pivot.columns]
        out_cols = ['Model', 'Rep'] + strat_cols + ['Ext. Mean']
        top_configs = top_pivot[out_cols].copy()

        # Save top 15
        top15 = top_configs.head(15)
        top15.to_csv(output_dir / 'table_top_nds_configs.csv', index=False)
        print(f"✓ Saved table_top_nds_configs.csv (top 15 by {gauss_col})")

        # Check if Gaussian ranking differs from other strategies
        if gauss_col in top_pivot.columns:
            rank_by_gauss = top_pivot.set_index(['model', 'rep'])[gauss_col].rank(ascending=False)
            for sc in strat_cols:
                if sc != gauss_col and sc in top_pivot.set_index(['model', 'rep']).columns:
                    rank_by_other = top_pivot.set_index(['model', 'rep'])[sc].rank(ascending=False)
                    common = rank_by_gauss.index.intersection(rank_by_other.index)
                    if len(common) >= 5:
                        from scipy.stats import spearmanr
                        rho, p = spearmanr(rank_by_gauss.loc[common], rank_by_other.loc[common])
                        print(f"  Rank correlation Gaussian vs {sc}: ρ={rho:.3f} (p={p:.2e})")

        # Save full table (all configs)
        top_configs.to_csv(output_dir / 'table_all_nds_configs.csv', index=False)
        print(f"✓ Saved table_all_nds_configs.csv ({len(top_configs)} configs)")

    # Table 3: Probabilistic comparison with Wilcoxon tests (PDV + legacy)
    prob_comparisons = {
        'NN-α Family': {'base': 'dnn', 'variants': ['dnn_bnn_full', 'dnn_vbll']},
        'NN-β Family': {'base': 'mlp', 'variants': ['mlp_bnn_full', 'mlp_vbll']},
        'RF Family': {'base': 'rf', 'variants': ['qrf']},
    }

    # Filter to PDV + legacy for fair comparison
    nds_fair = nds_df[(nds_df['rep'] == PRIMARY_REP) & (nds_df['strategy'] == 'legacy')] if 'rep' in nds_df.columns else nds_df

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
                wilcox = wilcoxon_paired_test(nds_df, base_model, variant, rep=PRIMARY_REP, strategy='legacy')
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
    # Generate per-rep tables AND averaged table for comparison
    if unc_df is not None and len(unc_df) > 0:
        unc_legacy = unc_df[unc_df['strategy'] == 'legacy'] if 'strategy' in unc_df.columns else unc_df

        # Find uncertainty column
        unc_col = None
        for col in ['y_pred_std_calibrated', 'y_pred_std', 'uncertainty']:
            if col in unc_legacy.columns:
                unc_col = col
                break

        if unc_col and len(unc_legacy) > 0:
            # Determine reps to iterate: each individual rep + 'all' (averaged)
            if 'rep' in unc_legacy.columns:
                available_reps = sorted(unc_legacy['rep'].unique())
            else:
                available_reps = []
            reps_to_compute = available_reps + ['all']

            for rep_name in reps_to_compute:
                if rep_name == 'all':
                    # Exclude non-ANOVA reps (binary pdv, sns, morgan) from the average
                    if 'rep' in unc_legacy.columns:
                        rep_data = unc_legacy[~unc_legacy['rep'].isin(ANOVA_REPS_EXCLUDE)]
                    else:
                        rep_data = unc_legacy
                else:
                    rep_data = unc_legacy[unc_legacy['rep'] == rep_name]

                if len(rep_data) == 0:
                    continue

                unc_metrics = []
                for model in rep_data['model'].unique():
                    model_data = rep_data[rep_data['model'] == model]

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

                    # Skip models with negligible uncertainty (e.g. plain NN-α/NN-β
                    # that don't produce meaningful uncertainty estimates)
                    mean_unc = unc_values[mask].mean()
                    if mean_unc < 1e-3:
                        continue

                    # Uncertainty-error correlation
                    unc_err_corr, _ = stats.spearmanr(unc_values[mask], errors[mask])

                    # Uncertainty-noise correlation, POOLED across σ (population-level /
                    # Kolmar trend — kept only for comparison, do NOT report as per-sample).
                    unc_noise_corr = np.nan
                    if 'injected_noise' in model_data.columns:
                        noise_mag = np.abs(model_data['injected_noise'].values)
                        noise_mask = mask & np.isfinite(noise_mag)
                        if noise_mask.sum() > 100:
                            unc_noise_corr, _ = stats.spearmanr(unc_values[noise_mask], noise_mag[noise_mask])

                    # Within-σ per-sample correlation (the honest metric) — each σ separate.
                    per_sigma_noise = within_sigma_unc_noise_rho(model_data, unc_values, mask)

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

                    row = {
                        'Model': model,
                        'Unc-Error ρ': unc_err_corr,
                        'Unc-Noise ρ (pooled)': unc_noise_corr,
                        'ECE': ece,
                        'Coverage 1σ': cov_1sigma,
                        'Coverage 2σ': cov_2sigma,
                        'Mean Uncertainty': unc_values[mask].mean(),
                        'Mean Aleatoric': mean_alea,
                        'Mean Epistemic': mean_epis,
                    }
                    # Per-σ within-slice ρ + slice size n, each σ kept separate.
                    for s in WITHIN_SIGMA_LEVELS:
                        row[f'Unc-Noise ρ σ={s}'] = per_sigma_noise[s]['rho']
                        row[f'Unc-Noise n σ={s}'] = per_sigma_noise[s]['n']
                    unc_metrics.append(row)

                if unc_metrics:
                    unc_metrics_df = pd.DataFrame(unc_metrics).sort_values('Unc-Error ρ', ascending=False)
                    # Add clean labels
                    unc_metrics_df['Model'] = unc_metrics_df['Model'].map(
                        lambda m: get_model_label(m))
                    suffix = f'_{rep_name}' if rep_name != 'all' else ''
                    unc_metrics_df.to_csv(output_dir / f'table4_uncertainty_metrics{suffix}.csv', index=False)
                    print(f"✓ Saved table4_uncertainty_metrics{suffix}.csv")

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

                        # POOLED across σ (population-level / Kolmar trend — comparison only).
                        unc_noise_corr = np.nan
                        if 'injected_noise' in model_data.columns:
                            noise_mag = np.abs(model_data['injected_noise'].values)
                            noise_mask = valid & np.isfinite(noise_mag)
                            if noise_mask.sum() > 100:
                                unc_noise_corr, _ = stats.spearmanr(unc_values[noise_mask], noise_mag[noise_mask])

                        # Within-σ per-sample correlation (honest metric) — each σ separate.
                        per_sigma_noise = within_sigma_unc_noise_rho(model_data, unc_values, valid)

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

                        strat_row = {
                            'Strategy': STRATEGY_LABELS.get(strategy, strategy),
                            'Rep': rep,
                            'Model': get_model_label(model),
                            'Unc-Error ρ': unc_err_corr,
                            'Unc-Noise ρ (pooled)': unc_noise_corr,
                            'ECE': ece,
                            'Coverage 1σ': cov_1sigma,
                            'Coverage 2σ': cov_2sigma,
                            'Mean Uncertainty': unc_values[valid].mean(),
                            'Mean Aleatoric': mean_alea,
                            'Mean Epistemic': mean_epis,
                        }
                        # Per-σ within-slice ρ + slice size n, each σ kept separate.
                        for s in WITHIN_SIGMA_LEVELS:
                            strat_row[f'Unc-Noise ρ σ={s}'] = per_sigma_noise[s]['rho']
                            strat_row[f'Unc-Noise n σ={s}'] = per_sigma_noise[s]['n']
                        all_unc_rows.append(strat_row)

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

    # Table 4c: Top model×rep Unc-Noise correlations (Gaussian strategy)
    # Shows specific combinations where uncertainty best detects injected noise.
    # Ranked by the within-σ per-sample ρ at σ=0.3 (moderate — the primary real test),
    # NOT the pooled ρ (which just reflects the population-level σ trend). The pooled and
    # all per-σ columns are carried alongside for the paper Table 7 reframing (plan §9).
    # NOTE: sort key (σ=0.3) is provisional — revisit with the user once numbers are in.
    if unc_df is not None and len(unc_df) > 0:
        supp_path = output_dir / 'table4_supp_uncertainty_by_strategy_rep.csv'
        if supp_path.exists():
            supp_unc = pd.read_csv(supp_path)
            gauss_unc = supp_unc[supp_unc['Strategy'] == 'Gaussian']
            rank_col = 'Unc-Noise ρ σ=0.3'
            if len(gauss_unc) > 0 and rank_col in gauss_unc.columns:
                per_sigma_cols = [c for c in gauss_unc.columns if c.startswith('Unc-Noise ')]
                table4c_cols = (['Model', 'Rep'] + per_sigma_cols
                                + ['Unc-Error ρ', 'ECE', 'Coverage 1σ', 'Coverage 2σ'])
                top_noise = gauss_unc.nlargest(15, rank_col)[table4c_cols].copy()
                # Apply rep labels
                top_noise['Rep'] = top_noise['Rep'].map(lambda r: get_rep_label(r))
                top_noise.to_csv(output_dir / 'table4c_top_unc_noise_correlations.csv', index=False)
                print(f"✓ Saved table4c_top_unc_noise_correlations.csv ({len(top_noise)} rows, ranked by {rank_col})")

                # Also save bottom combinations (near-zero/negative) for contrast
                bottom_noise = gauss_unc.nsmallest(10, rank_col)[table4c_cols].copy()
                bottom_noise['Rep'] = bottom_noise['Rep'].map(lambda r: get_rep_label(r))
                bottom_noise.to_csv(output_dir / 'table4c_bottom_unc_noise_correlations.csv', index=False)
                print(f"✓ Saved table4c_bottom_unc_noise_correlations.csv ({len(bottom_noise)} rows)")

    # Table 5: Model rankings at specific sigma levels (shows ranking stability)
    # Uses PDV + legacy only
    if qm9_df is not None and len(qm9_df) > 0:
        # Filter to PDV + legacy
        pdv_legacy = qm9_df[(qm9_df['strategy'] == 'legacy') & (qm9_df['rep'] == PRIMARY_REP)] if 'strategy' in qm9_df.columns else qm9_df

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
    # Use ANOVA-included models only for consistency with other analyses
    nds_anova = nds_df[~nds_df['model'].isin(ANOVA_MODELS_EXCLUDE)]
    if len(nds_anova) > 0:
        # Get rankings per strategy
        strategies = nds_anova['strategy'].unique()
        if len(strategies) > 1:
            models = nds_anova['model'].unique()

            # First pass: find models present in ALL strategies
            model_nds_by_strategy = {}
            for strategy in strategies:
                strat_data = nds_anova[nds_anova['strategy'] == strategy]
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

    # Table 7 (sensitivity ratio) cut — sensitivity ratios now reported inline in paper text


# =============================================================================
# INTERACTION FIGURE (Issue D)
# =============================================================================

def create_interaction_figure(nds_df, raw_df, output_dir):
    """Visualize the model x representation interaction effect.

    Panel A: Heatmap — NDS by model × representation (Gaussian strategy).
    Panel B: Scatter — NDS on ECFP4 vs NDS on PDV per model.

    BNN variants use different marker shapes but same color as base model.
    """
    if len(nds_df) == 0:
        print("⚠ Could not create interaction figure - no NDS data")
        return

    nds_legacy = nds_df[nds_df['strategy'] == 'legacy'] if 'strategy' in nds_df.columns else nds_df
    # Filter to ANOVA-included reps only; keep ALL models (including QRF) for interaction view
    nds_legacy = nds_legacy[~nds_legacy['rep'].isin(ANOVA_REPS_EXCLUDE)]
    # Exclude PDV-only models from cross-rep interaction figure
    nds_legacy = nds_legacy[~nds_legacy['model'].isin(PDV_ONLY_MODELS)]

    # Stack panels vertically: heatmap (A) and scatter (B) each get the full
    # textwidth so cell annotations and legends stay legible at print scale.
    fig, axes = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN * 1.25))

    # Panel A: Heatmap — NDS by model × representation (Gaussian strategy)
    ax_a = axes[0]

    # Get mean NDS per model x rep (Gaussian strategy)
    pivot = nds_legacy.pivot_table(values='nds', index='model', columns='rep', aggfunc='mean')

    # Include ANOVA reps with enough models
    valid_reps = [r for r in pivot.columns if pivot[r].notna().sum() >= 3]
    rep_order = [r for r in ['ecfp4', 'continuous_pdv', 'smiles', 'mol2vec', 'mhggnn'] if r in valid_reps]

    if len(rep_order) >= 2:
        # Reindex to show all models
        all_models_int = sort_models_by_family(nds_legacy['model'].unique().tolist())
        hm_pivot = pivot.reindex(index=all_models_int, columns=rep_order)
        hm_pivot = hm_pivot.dropna(how='all')

        annot_text = make_heatmap_annotations(hm_pivot, raw_df, 'model', 'rep',
                                               fmt='.2f',
                                               extra_filters={'strategy': 'legacy'})
        hm_pivot.index = [get_model_label(m) for m in hm_pivot.index]
        annot_text.index = hm_pivot.index
        hm_pivot.columns = [get_rep_label(r) for r in hm_pivot.columns]
        annot_text.columns = hm_pivot.columns

        ax_a.set_facecolor('black')
        sns.heatmap(hm_pivot, annot=annot_text, fmt='', cmap='PuOr', vmax=0,
                    ax=ax_a, cbar_kws={'label': 'NDS'}, linewidths=0.5)
        _white_text_for_missing(ax_a, hm_pivot, annot_text)
        ax_a.set_title('a) Model × Rep Interaction (Gaussian NDS)', fontweight='bold')
        ax_a.set_ylabel('')

    # Panel B: Scatter — NDS on ECFP4 vs NDS on PDV, with legend (not annotations)
    ax_b = axes[1]

    ecfp4_nds = nds_legacy[nds_legacy['rep'] == 'ecfp4'].groupby('model')['nds'].mean()
    pdv_nds = nds_legacy[nds_legacy['rep'] == 'continuous_pdv'].groupby('model')['nds'].mean()

    shared_models_set = set(ecfp4_nds.index) & set(pdv_nds.index)
    # Use MODEL_ORDER for consistent legend ordering
    shared_models = [m for m in MODEL_ORDER if m in shared_models_set]
    # Append any models not in MODEL_ORDER (shouldn't happen, but safe)
    shared_models += sorted(shared_models_set - set(shared_models))

    if len(shared_models) >= 3:
        for m in shared_models:
            ev, pv = ecfp4_nds[m], pdv_nds[m]
            color = MODEL_COLORS.get(m, '#333333')
            marker = MODEL_MARKERS.get(m, 'o')
            ax_b.scatter(ev, pv, color=color, marker=marker, s=60, zorder=3, alpha=0.75,
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

        # Multi-column legend below the panel so it stays inside textwidth
        # (previously hung off the right, which forced \textwidth scaling
        # to shrink the whole figure).
        _ordered_legend(ax_b, 5, loc='upper center', bbox_to_anchor=(0.5, -0.20),
                        frameon=False, columnspacing=0.8,
                        handletextpad=0.4, borderaxespad=0)

    ax_b.set_xlabel('NDS on ECFP4 (Gaussian)')
    ax_b.set_ylabel('NDS on PDV (Gaussian)')
    ax_b.set_title('b) ECFP4 vs PDV Robustness', fontweight='bold')

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
    if n_panels > 3:
        nrows, ncols = 2, 3
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(16.5, 10))
        axes = axes_grid.flatten()[:n_panels]
        # Hide unused axes
        for i in range(n_panels, nrows * ncols):
            axes_grid.flatten()[i].set_visible(False)
    else:
        fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

    rep_markers = {'ecfp4': 'o', 'continuous_pdv': 's', 'smiles': '^',
                   'mhggnn': 'v', 'mol2vec': 'D'}

    # First pass: collect all data to compute shared axes
    all_baseline_r2 = []
    all_nds_mean = []
    panel_data = []
    last_mean_nds = None
    for idx, strategy in enumerate(strategies_available):
        strat_data = anova_nds[anova_nds['strategy'] == strategy]
        mean_nds = strat_data.groupby(['model', 'rep']).agg(
            nds_mean=('nds', 'mean'),
            baseline_r2=('baseline_r2', 'mean')
        ).reset_index()
        panel_data.append(mean_nds)
        last_mean_nds = mean_nds
        all_baseline_r2.extend(mean_nds['baseline_r2'].dropna().tolist())
        all_nds_mean.extend(mean_nds['nds_mean'].dropna().tolist())

    # Compute shared limits from data with generous padding
    if all_baseline_r2 and all_nds_mean:
        x_pad = (max(all_baseline_r2) - min(all_baseline_r2)) * 0.08
        y_pad = (max(all_nds_mean) - min(all_nds_mean)) * 0.08
        shared_xlim = (min(all_baseline_r2) - x_pad, max(all_baseline_r2) + x_pad)
        shared_ylim = (min(all_nds_mean) - y_pad, max(all_nds_mean) + y_pad)
    else:
        shared_xlim = None
        shared_ylim = None

    # Second pass: plot with shared axes
    for idx, strategy in enumerate(strategies_available):
        ax = axes[idx]
        strategy_label = STRATEGY_LABELS.get(strategy, strategy)
        label_prefix = f'{panel_labels[idx]}. ' if panel_labels and idx < len(panel_labels) else ''
        mean_nds = panel_data[idx]

        for _, row in mean_nds.iterrows():
            color = MODEL_COLORS.get(row['model'], '#333333')
            marker = rep_markers.get(row['rep'], 'o')
            ax.scatter(row['baseline_r2'], row['nds_mean'],
                       color=color, marker=marker, s=50, alpha=0.7,
                       edgecolors='black', linewidths=0.5)

        if shared_xlim:
            ax.set_xlim(shared_xlim)
        if shared_ylim:
            ax.set_ylim(shared_ylim)
        ax.set_xlabel('Baseline R² (σ=0)')
        if n_panels > 3:
            # 2×3 grid: ylabel on first column only
            if idx % 3 == 0:
                ax.set_ylabel('NDS')
            else:
                ax.set_ylabel('')
        else:
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

        # Combined legend on first panel (top-right)
        rep_handles = [Line2D([0], [0], marker=rep_markers.get(r, 'o'), color='gray',
                              linestyle='None', markersize=6,
                              markeredgecolor='black', markeredgewidth=0.5,
                              label=REP_LABELS.get(r, r))
                       for r in ['ecfp4', 'continuous_pdv', 'smiles', 'mhggnn', 'mol2vec']
                       if r in last_mean_nds['rep'].unique()]

        # Model color+marker legend — use MODEL_ORDER for grouping
        models_in_data = set(last_mean_nds['model'].unique())
        models_present = [m for m in MODEL_ORDER if m in models_in_data]
        models_present += sorted(models_in_data - set(models_present))
        model_handles = [Line2D([0], [0], marker=MODEL_MARKERS.get(m, 'o'),
                                color=MODEL_COLORS.get(m, '#333333'),
                                linestyle='None', markersize=6,
                                markeredgecolor='black', markeredgewidth=0.5,
                                label=get_model_label(m))
                         for m in models_present]
        axes[-1].legend(handles=_to_colmajor(rep_handles + model_handles, 2),
                        loc='upper right', fontsize=6,
                        title='Rep / Model', title_fontsize=7, ncol=2)

    fig.suptitle('Baseline Performance vs Noise Robustness (ANOVA Configurations)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved {filename}")


def create_full_overview(nds_df, output_dir):
    """Scatter plots of ALL model x rep configurations across all 6 strategies.

    Single 2×3 figure covering all strategies (replaces old main + supp split).
    """
    if len(nds_df) == 0:
        print("⚠ Could not create full overview - no NDS data")
        return

    # Combined 2×3 figure with all 6 strategies
    _plot_full_overview_panels(
        nds_df,
        strategies=list(STRATEGY_ORDER),
        output_dir=output_dir,
        filename='fig_full_overview_all_strategies.png',
        panel_labels=['A', 'B', 'C', 'D', 'E', 'F']
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

        # Generate LaTeX summary table
        excl_summary = excluded_df.groupby(['model', 'rep']).size().reset_index(name='n_excluded')
        excl_pivot = excl_summary.pivot_table(values='n_excluded', index='model', columns='rep', fill_value=0)
        excl_pivot.index = [get_model_label(m) for m in excl_pivot.index]
        excl_pivot.columns = [get_rep_label(r) for r in excl_pivot.columns]
        excl_pivot['Total'] = excl_pivot.sum(axis=1)
        excl_pivot = excl_pivot.sort_values('Total', ascending=False)

        lines_tex = []
        cols = list(excl_pivot.columns)
        col_fmt = 'l' + 'r' * len(cols)
        lines_tex.append(r'\begin{tabular}{' + col_fmt + '}')
        lines_tex.append(r'\toprule')
        lines_tex.append(r'\textbf{Model} & ' + ' & '.join(f'\\textbf{{{c}}}' for c in cols) + r' \\')
        lines_tex.append(r'\midrule')
        for model_label, row in excl_pivot.iterrows():
            vals = ' & '.join(str(int(v)) if v > 0 else '--' for v in row.values)
            lines_tex.append(f'{model_label} & {vals} \\\\')
        lines_tex.append(r'\bottomrule')
        lines_tex.append(r'\end{tabular}')

        tex_path = output_dir / 'table_supp_excluded_configs.tex'
        tex_path.write_text('\n'.join(lines_tex))
        print(f"✓ Saved table_supp_excluded_configs.tex")

    if len(nds_df) > 0:
        lines.append("\n" + "=" * 80)
        lines.append("KEY FINDINGS (PDV representation, Gaussian strategy)")
        lines.append("=" * 80)

        # Filter to PDV + legacy for consistent reporting
        nds_pdv_legacy = nds_df[(nds_df['rep'] == PRIMARY_REP) & (nds_df['strategy'] == 'legacy')]
        if len(nds_pdv_legacy) == 0:
            nds_pdv_legacy = nds_df  # Fallback

        # Most robust model
        mean_nds = nds_pdv_legacy.groupby('model')['nds'].mean().sort_values(ascending=False)
        lines.append(f"\nMost robust model: {mean_nds.index[0]} (NDS = {mean_nds.iloc[0]:.4f})")
        lines.append(f"Least robust model: {mean_nds.index[-1]} (NDS = {mean_nds.iloc[-1]:.4f})")

        # BNN vs NN-α comparison
        dnn_data = nds_pdv_legacy[nds_pdv_legacy['model'] == 'dnn']
        bnn_data = nds_pdv_legacy[nds_pdv_legacy['model'] == 'dnn_bnn_full']

        if len(dnn_data) > 0 and len(bnn_data) > 0:
            dnn_nds = dnn_data['nds'].values[0]
            bnn_nds = bnn_data['nds'].values[0]
            lines.append(f"\nNN-α NDS: {dnn_nds:.4f}")
            lines.append(f"BNN-α NDS: {bnn_nds:.4f}")
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

    # Load data (timed + flushed so the SLURM log shows exactly which step is slow)
    print("\n[1/3] Loading data...", flush=True)
    _t = time.time()
    qm9_df = load_anova_data(args.qm9_dir)
    print(f"  ✓ load_anova_data: {time.time()-_t:.1f}s "
          f"({len(qm9_df) if qm9_df is not None else 0} rows)", flush=True)

    _t = time.time()
    unc_df = load_uncertainty_data(args.qm9_dir)
    print(f"  ✓ load_uncertainty_data: {time.time()-_t:.1f}s "
          f"({len(unc_df) if unc_df is not None else 0} rows)", flush=True)
    if unc_df is not None:
        unc_df = fix_injected_noise(unc_df)

    _t = time.time()
    validation_df = load_validation_data(args.validation_dir)
    print(f"  ✓ load_validation_data: {time.time()-_t:.1f}s "
          f"({len(validation_df) if validation_df is not None else 0} rows)", flush=True)

    if qm9_df is None:
        print("ERROR: No QM9 ANOVA data found!")
        return

    # Exclude graph models (Graph_GP, GCN, GIN) - not in scope
    EXCLUDED_MODELS = {'graph_gp', 'gcn', 'gin', 'ginct', 'gin2d'}
    pre_filter = len(qm9_df)
    qm9_df = qm9_df[~qm9_df['model'].isin(EXCLUDED_MODELS)]
    if len(qm9_df) < pre_filter:
        print(f"  Filtered out {pre_filter - len(qm9_df)} rows from excluded models (graph + old var-BNN)")
    if unc_df is not None:
        pre_filter_unc = len(unc_df)
        unc_df = unc_df[~unc_df['model'].isin(EXCLUDED_MODELS)]
        if len(unc_df) < pre_filter_unc:
            print(f"  Filtered out {pre_filter_unc - len(unc_df)} uncertainty rows from excluded models")

    # Filter catastrophic training iterations (e.g. NN-α/mol2vec with R² = -63)
    qm9_df, catastrophic_log = filter_catastrophic_iterations(qm9_df)
    if len(catastrophic_log) > 0:
        catastrophic_log.to_csv(output_dir / 'filtered_catastrophic_iterations.csv', index=False)
        print(f"    Saved filtered_catastrophic_iterations.csv")

    # Audit data completeness — flags missing sigmas/iterations, saves data_gaps.csv
    audit_data_completeness(qm9_df, output_dir)
    if unc_df is not None:
        audit_uncertainty_completeness(unc_df, output_dir)

    # Full data inventory — print everything that's loaded so we can see it in SLURM output
    print("\n" + "=" * 80)
    print("FULL DATA INVENTORY")
    print("=" * 80)
    all_models = sorted(qm9_df['model'].unique())
    all_reps = sorted(qm9_df['rep'].unique())
    all_strategies = sorted(qm9_df['strategy'].unique())
    print(f"  Models ({len(all_models)}): {all_models}")
    print(f"  Reps ({len(all_reps)}): {all_reps}")
    print(f"  Strategies ({len(all_strategies)}): {all_strategies}")
    print(f"  Total rows: {len(qm9_df)}")

    # Per model×rep: number of strategies and mean iterations
    print(f"\n  --- Iterations per model × rep × strategy (ANOVA-included only) ---")
    anova_only = qm9_df[
        ~qm9_df['model'].isin(ANOVA_MODELS_EXCLUDE) &
        ~qm9_df['rep'].isin(ANOVA_REPS_EXCLUDE)
    ]
    inv_rows = []
    for (model, rep, strat), grp in anova_only.groupby(['model', 'rep', 'strategy']):
        n_iters = grp['iteration'].nunique() if 'iteration' in grp.columns else 0
        n_sigmas = grp['sigma'].nunique() if 'sigma' in grp.columns else 0
        inv_rows.append({'model': model, 'rep': rep, 'strategy': strat,
                         'n_iterations': n_iters, 'n_sigmas': n_sigmas})
    inv_df = pd.DataFrame(inv_rows)
    inv_df.to_csv(output_dir / 'data_inventory.csv', index=False)

    # Pivot: model×rep showing min iterations across strategies
    if len(inv_df) > 0:
        pivot = inv_df.groupby(['model', 'rep'])['n_iterations'].min().unstack(fill_value=0)
        print(pivot.to_string())
        print(f"\n  Saved full inventory to data_inventory.csv")

    # SNS summary (not in ANOVA but may be used elsewhere)
    sns_data = qm9_df[qm9_df['rep'] == 'sns'] if 'sns' in all_reps else pd.DataFrame()
    if len(sns_data) > 0:
        print(f"\n  --- SNS data (excluded from ANOVA, available for supplementary) ---")
        sns_models = sorted(sns_data['model'].unique())
        sns_strategies = sorted(sns_data['strategy'].unique())
        print(f"  Models ({len(sns_models)}): {sns_models}")
        print(f"  Strategies ({len(sns_strategies)}): {sns_strategies}")
        for (model, strat), grp in sns_data.groupby(['model', 'strategy']):
            n_iters = grp['iteration'].nunique() if 'iteration' in grp.columns else 0
            n_sigmas = grp['sigma'].nunique() if 'sigma' in grp.columns else 0
            if n_iters < 10 or n_sigmas < 11:
                print(f"    {model:30} / {strat:10}: {n_iters} iters, {n_sigmas} sigmas")

    # Uncertainty summary
    if unc_df is not None:
        print(f"\n  --- Uncertainty data ---")
        unc_models = sorted(unc_df['model'].dropna().unique()) if 'model' in unc_df.columns else []
        print(f"  Models ({len(unc_models)}): {unc_models}")
        if 'rep' in unc_df.columns:
            print(f"  Reps: {sorted(unc_df['rep'].dropna().unique())}")
        if 'strategy' in unc_df.columns:
            print(f"  Strategies: {sorted(unc_df['strategy'].dropna().unique())}")

    # Validation summary
    if validation_df is not None:
        print(f"\n  --- Validation data ---")
        if 'model' in validation_df.columns:
            print(f"  Models: {sorted(validation_df['model'].unique())}")
        if 'dataset' in validation_df.columns:
            print(f"  Datasets: {sorted(validation_df['dataset'].unique())}")
        if 'fold' in validation_df.columns:
            print(f"  Folds: {sorted(validation_df['fold'].unique())}")
        if 'sigma' in validation_df.columns:
            print(f"  Sigmas: {sorted(validation_df['sigma'].unique())}")
    print("=" * 80)

    # Calculate NDS
    print("\n[2/3] Calculating metrics...", flush=True)
    _t = time.time()
    nds_df, excluded_df = calculate_nds(qm9_df)
    print(f"  NDS calculated: {len(nds_df)} configs (in {time.time()-_t:.1f}s)", flush=True)
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
    create_figure3(nds_df, validation_df, val_nds_df, qm9_df, output_dir)
    create_interaction_figure(nds_df, qm9_df, output_dir)
    create_full_overview(nds_df, output_dir)

    print("\n--- PART 2: THE WHY ---")
    create_nn_family_comparison(qm9_df, nds_df, output_dir)
    create_uncertainty_figure(unc_df, output_dir)

    print("\n--- TABLES ---")
    create_tables(nds_df, unc_df, qm9_df, output_dir, val_nds_df=val_nds_df)

    print("\n--- VALIDATION (GENERALISATION) ---")
    create_validation_figures(validation_df, val_nds_df, nds_df, output_dir)

    print("\n--- SUPPLEMENTARY: ICC & REDUNDANCY ---")
    compute_icc_and_redundancy(nds_df, output_dir)

    print("\n--- REPORT ---")
    generate_report(nds_df, excluded_df, output_dir)

    print("\n" + "=" * 80)
    print(f"COMPLETE - All outputs in {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
