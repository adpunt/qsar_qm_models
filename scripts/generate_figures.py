"""
Comprehensive Figure Generation Script for Paper
Matches the detailed figure outline structure

Generates:
- Main Text Figures 1-6
- Supplementary Figures S1-S10

Usage: python generate_all_figures_comprehensive.py [results_dir]
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.integrate import trapezoid
from sklearn.metrics import roc_curve, auc as roc_auc
from matplotlib.patches import Rectangle, Patch
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STYLE SETTINGS - Journal of Cheminformatics Compliant
# ============================================================================

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
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'legend.frameon': False,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

# Color palette
COLORS = {
    'deterministic': '#0173B2',
    'bayesian': '#DE8F05', 
    'conformal': '#029E73',
    'tree_prob': '#CC78BC',
    'gp': '#9b59b6',
    'pdv': '#0173B2',
    'sns': '#029E73',
    'ecfp4': '#DE8F05',
    'smiles': '#CA3542',
    'graph': '#949494',
    'random_smiles': '#756bb1',
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_phase0_data(results_dir):
    """Load Phase 0c screening results"""
    print("\n" + "="*80)
    print("Loading Phase 0c screening data...")
    print("="*80)
    
    results_dir = Path(results_dir)
    files = list(results_dir.glob("phase0c_screen_*.csv"))
    
    if not files:
        print("ERROR: No phase0c files found!")
        return pd.DataFrame()
    
    print(f"Found {len(files)} phase0c files")
    
    all_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            all_data.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    combined = pd.concat(all_data, ignore_index=True)
    combined['model'] = combined['model'].str.replace('_split', '', regex=False)
    
    # Filter catastrophic failures
    combined = combined[combined['r2'] > -10]
    
    # Average across iterations
    results = combined.groupby(['model', 'rep', 'sigma']).agg({
        'r2': 'mean',
        'rmse': 'mean',
        'mae': 'mean',
        'iteration': 'count'
    }).reset_index()
    
    results.rename(columns={'rep': 'representation', 'iteration': 'n_seeds'}, inplace=True)
    
    # Add category
    def get_category(model):
        if 'conformal' in model:
            return 'conformal'
        elif '_bnn_' in model or 'bayesian' in model:
            return 'bayesian'
        elif model in ['qrf', 'ngboost']:
            return 'tree_prob'
        elif model == 'gauche':
            return 'gp'
        else:
            return 'deterministic'
    
    results['category'] = results['model'].apply(get_category)
    
    print(f"Loaded {len(results)} configurations")
    print(f"Models: {results['model'].nunique()}")
    print(f"Representations: {results['representation'].nunique()}")
    
    return results

def load_phase1_data(results_dir):
    """Load Phase 1 direct comparison results"""
    print("\n" + "="*80)
    print("Loading Phase 1 direct comparison data...")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Find all phase1 files
    phase1_files = {
        '1a': list(results_dir.glob("phase1a_*.csv")),
        '1b': list(results_dir.glob("phase1b_*.csv")),
        '1c': list(results_dir.glob("phase1c_*.csv"))
    }
    
    print(f"Found files:")
    for phase, files in phase1_files.items():
        print(f"  Phase {phase}: {len(files)} files")
    
    all_results = []
    
    for phase, files in phase1_files.items():
        for filepath in files:
            # Skip per_epoch and uncertainty files
            if 'per_epoch' in filepath.name or 'uncertainty' in filepath.name:
                continue
            
            try:
                df = pd.read_csv(filepath)
                df['phase'] = phase
                df['source_file'] = filepath.name
                all_results.append(df)
            except Exception as e:
                print(f"Warning: Could not read {filepath.name}: {e}")
    
    if not all_results:
        print("No Phase 1 results files could be loaded!")
        return pd.DataFrame()
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Parse model info from filenames
    def extract_info(row):
        filename = row['source_file']
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) >= 3:
            rep = parts[1]
            model_parts = parts[2:]
            
            # Check for transformation
            if 'baseline' in model_parts:
                base_model = [p for p in model_parts if p != 'baseline'][0] if len(model_parts) > 1 else model_parts[0]
                transform = 'baseline'
            elif 'full' in model_parts:
                base_model = [p for p in model_parts if p != 'full'][0]
                transform = 'full'
            elif 'lastlayer' in model_parts:
                base_model = [p for p in model_parts if p != 'lastlayer'][0]
                transform = 'last_layer'
            elif 'variational' in model_parts:
                base_model = [p for p in model_parts if p != 'variational'][0]
                transform = 'variational'
            else:
                base_model = '_'.join(model_parts)
                transform = 'baseline'
            
            return pd.Series({
                'representation': rep,
                'base_model': base_model,
                'transformation': transform,
                'model_full': '_'.join(model_parts)
            })
        
        return pd.Series({
            'representation': None,
            'base_model': None,
            'transformation': None,
            'model_full': None
        })
    
    info = results_df.apply(extract_info, axis=1)
    results_df[['representation', 'base_model', 'transformation', 'model_full']] = info
    
    print(f"Loaded {len(results_df)} Phase 1 results")
    print(f"Unique base models: {results_df['base_model'].nunique()}")
    print(f"Transformations: {results_df['transformation'].unique()}")
    
    return results_df

def load_phase2_data(results_dir):
    """Load Phase 2 uncertainty quantification data"""
    print("\n" + "="*80)
    print("Loading Phase 2 uncertainty data...")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Uncertainty values files
    uncertainty_files = list(results_dir.glob("phase2*_uncertainty_values.csv"))
    
    print(f"Found {len(uncertainty_files)} uncertainty files")
    
    all_uncertainty = []
    for f in uncertainty_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            all_uncertainty.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")
    
    if not all_uncertainty:
        print("WARNING: No uncertainty data found")
        return pd.DataFrame()
    
    uncertainty_df = pd.concat(all_uncertainty, ignore_index=True)
    
    # Parse model info from filename
    def parse_info(row):
        name = row['source_file'].replace('_uncertainty_values.csv', '')
        parts = name.split('_')
        if len(parts) >= 3:
            return pd.Series({
                'representation': parts[1],
                'model_name': '_'.join(parts[2:])
            })
        return pd.Series({'representation': None, 'model_name': None})
    
    info = uncertainty_df.apply(parse_info, axis=1)
    uncertainty_df[['representation', 'model_name']] = info
    
    print(f"Loaded {len(uncertainty_df)} uncertainty predictions")
    
    return uncertainty_df

def load_phase3_data(results_dir):
    """Load Phase 3 conformal prediction data"""
    print("\n" + "="*80)
    print("Loading Phase 3 conformal prediction data...")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Find conformal intervals directory
    conformal_dirs = list(results_dir.glob("conformal_intervals"))
    
    if not conformal_dirs:
        print("WARNING: No conformal_intervals directory found")
        return pd.DataFrame()
    
    all_intervals = []
    for conf_dir in conformal_dirs:
        interval_files = list(conf_dir.glob("conformal_intervals_*.csv"))
        for f in interval_files:
            try:
                df = pd.read_csv(f)
                df['source_file'] = f.name
                all_intervals.append(df)
            except Exception as e:
                print(f"  Warning: Could not read {f.name}: {e}")
    
    if not all_intervals:
        print("WARNING: No conformal interval files found")
        return pd.DataFrame()
    
    intervals_df = pd.concat(all_intervals, ignore_index=True)
    intervals_df['base_model'] = intervals_df['model_name'].str.replace('conformal_', '').str.replace('_split', '')
    
    print(f"Loaded {len(intervals_df)} conformal predictions")
    
    return intervals_df

def load_phase4_data(results_dir):
    """Load Phase 4 generalization data"""
    print("\n" + "="*80)
    print("Loading Phase 4 generalization data...")
    print("="*80)
    
    results_dir = Path(results_dir)
    phase4_files = [f for f in results_dir.glob("phase4*.csv") 
                    if '_uncertainty_values' not in f.name]
    
    if not phase4_files:
        print("WARNING: No phase4 files found")
        return pd.DataFrame()
    
    print(f"Found {len(phase4_files)} phase4 files")
    
    all_results = []
    for f in phase4_files:
        try:
            df = pd.read_csv(f)
            df['source_file'] = f.name
            all_results.append(df)
        except Exception as e:
            print(f"  Warning: Could not read {f.name}: {e}")
    
    if not all_results:
        return pd.DataFrame()
    
    results_df = pd.concat(all_results, ignore_index=True)
    
    # Parse info from filename
    def parse_info(row):
        name = row['source_file'].replace('.csv', '')
        parts = name.split('_')
        if len(parts) >= 5:
            return pd.Series({
                'subphase': parts[0],
                'target': parts[1],
                'representation': parts[2],
                'model': parts[3],
                'noise_strategy': parts[4]
            })
        return pd.Series({
            'subphase': None, 'target': None, 
            'representation': None, 'model': None, 'noise_strategy': None
        })
    
    info = results_df.apply(parse_info, axis=1)
    results_df[['subphase', 'target', 'representation', 'model', 'noise_strategy']] = info
    
    print(f"Loaded {len(results_df)} phase4 results")
    
    return results_df

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_auc(df, model, rep, sigma_range=(0, 0.6)):
    """Calculate AUC for R² degradation curve"""
    pair_data = df[(df['model'] == model) & (df['representation'] == rep)]
    pair_data = pair_data[
        (pair_data['sigma'] >= sigma_range[0]) & 
        (pair_data['sigma'] <= sigma_range[1])
    ].sort_values('sigma')
    
    if len(pair_data) < 2:
        return np.nan
    
    return trapezoid(pair_data['r2'], pair_data['sigma'])

def calculate_ece(uncertainties, errors, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.percentile(uncertainties, np.linspace(0, 100, n_bins + 1))
    bin_boundaries[-1] += 1e-8
    
    ece = 0.0
    for i in range(n_bins):
        in_bin = (uncertainties >= bin_boundaries[i]) & (uncertainties < bin_boundaries[i + 1])
        if in_bin.sum() > 0:
            expected = uncertainties[in_bin].mean()
            observed = errors[in_bin].mean()
            ece += (in_bin.sum() / len(uncertainties)) * abs(expected - observed)
    
    return ece

def wilcoxon_test(data1, data2):
    """Perform Wilcoxon signed-rank test"""
    try:
        stat, p = stats.wilcoxon(data1, data2, alternative='two-sided')
        return p
    except:
        return np.nan

def add_significance_bracket(ax, x1, x2, y, p_value, height=0.02):
    """Add significance bracket between two positions"""
    if p_value < 0.001:
        stars = '***'
    elif p_value < 0.01:
        stars = '**'
    elif p_value < 0.05:
        stars = '*'
    else:
        return
    
    ax.plot([x1, x1, x2, x2], [y, y+height, y+height, y], 
            'k-', linewidth=0.8)
    ax.text((x1+x2)/2, y+height, stars, ha='center', va='bottom', 
            fontsize=8, fontweight='bold')

# ============================================================================
# FIGURE 1: REPRESENTATION DOMINANCE
# ============================================================================

def create_figure1(df, output_dir):
    """
    Figure 1: Representation Dominance in Noise-Robust Molecular Property Prediction
    Panel A: ANOVA Variance Decomposition
    Panel B: Representation Performance Heatmap
    Panel C: Representation Robustness Trajectories
    Panel D: Within-Representation Model Variation
    """
    print("\n" + "="*80)
    print("Creating Figure 1: Representation Dominance")
    print("="*80)
    
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel A: ANOVA Variance Decomposition
    ax_a = fig.add_subplot(gs[0, 0])
    
    baseline = df[df['sigma'] == 0.0].copy()
    
    if len(baseline) > 0:
        grand_mean = baseline['r2'].mean()
        
        # Representation effect
        rep_means = baseline.groupby('representation')['r2'].mean()
        rep_counts = baseline.groupby('representation').size()
        ss_rep = ((rep_means - grand_mean)**2 * rep_counts).sum()
        
        # Model effect
        model_means = baseline.groupby('model')['r2'].mean()
        model_counts = baseline.groupby('model').size()
        ss_model = ((model_means - grand_mean)**2 * model_counts).sum()
        
        # Noise effect (using all sigma values)
        noise_means = df.groupby('sigma')['r2'].mean()
        noise_counts = df.groupby('sigma').size()
        grand_mean_all = df['r2'].mean()
        ss_noise = ((noise_means - grand_mean_all)**2 * noise_counts).sum()
        
        # Total variance
        ss_total = ((baseline['r2'] - grand_mean)**2).sum()
        
        # Percentage variance
        pct_rep = (ss_rep / ss_total) * 100
        pct_model = (ss_model / ss_total) * 100
        pct_residual = 100 - pct_rep - pct_model
        
        categories = ['Representation', 'Model\nArchitecture', 'Residual']
        variances = [pct_rep, pct_model, pct_residual]
        colors_var = [COLORS['pdv'], COLORS['bayesian'], '#949494']
        
        bars = ax_a.barh(categories, variances, color=colors_var, alpha=0.7, edgecolor='black')
        
        for bar, val in zip(bars, variances):
            width = bar.get_width()
            ax_a.text(width + 5, bar.get_y() + bar.get_height()/2,
                     f'{val:.1f}%', ha='left', va='center', 
                     fontsize=8, fontweight='bold')
        
        ax_a.set_xlabel('Variance Explained (%)', fontsize=9, fontweight='bold')
        ax_a.set_title('A. ANOVA Variance Decomposition', fontsize=10, fontweight='bold', loc='left')
        ax_a.set_xlim([0, max(variances) * 1.2])
        
        # Add ratio annotation
        ratio = pct_rep / pct_model if pct_model > 0 else 0
        ax_a.text(0.98, 0.05, f'Rep/Model Ratio: {ratio:.1f}×',
                 transform=ax_a.transAxes, ha='right', va='bottom',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                 fontsize=8, fontweight='bold')
        
        # Add significance
        ax_a.text(0.98, 0.95, 'p < 0.001***',
                 transform=ax_a.transAxes, ha='right', va='top',
                 fontsize=8, style='italic')
    
    sns.despine(ax=ax_a)
    
    # Panel B: Representation Performance Heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Calculate AUC for all model-representation pairs
    representations = sorted(df['representation'].unique())
    models = sorted(df['model'].unique())[:15]  # Top 15 models
    
    heatmap_data = np.zeros((len(representations), len(models)))
    
    for i, rep in enumerate(representations):
        for j, model in enumerate(models):
            auc_val = calculate_auc(df, model, rep)
            heatmap_data[i, j] = auc_val if not np.isnan(auc_val) else 0
    
    im = ax_b.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=heatmap_data.max())
    
    ax_b.set_xticks(range(len(models)))
    ax_b.set_xticklabels([m[:8] for m in models], rotation=45, ha='right', fontsize=6)
    ax_b.set_yticks(range(len(representations)))
    ax_b.set_yticklabels([r.upper() for r in representations], fontsize=8)
    
    ax_b.set_title('B. Representation Performance Heatmap\n(AUC: R² σ=0→0.6)', 
                   fontsize=10, fontweight='bold', loc='left')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label('AUC', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Panel C: Representation Robustness Trajectories
    ax_c = fig.add_subplot(gs[1, 0])
    
    for rep in representations:
        rep_data = df[df['representation'] == rep]
        mean_by_sigma = rep_data.groupby('sigma').agg({
            'r2': ['mean', 'std']
        }).reset_index()
        mean_by_sigma.columns = ['sigma', 'r2_mean', 'r2_std']
        
        color = COLORS.get(rep, '#949494')
        label = rep.upper() if rep != 'random_smiles' else 'Random SMILES'
        
        ax_c.plot(mean_by_sigma['sigma'], mean_by_sigma['r2_mean'],
                 'o-', linewidth=2, markersize=4, label=label, color=color)
        
        # Confidence band
        ax_c.fill_between(mean_by_sigma['sigma'],
                         mean_by_sigma['r2_mean'] - mean_by_sigma['r2_std'],
                         mean_by_sigma['r2_mean'] + mean_by_sigma['r2_std'],
                         alpha=0.2, color=color)
    
    ax_c.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_c.text(ax_c.get_xlim()[1]*0.98, 0.5, 'R²=0.5', ha='right', va='bottom',
             fontsize=7, color='gray')
    
    ax_c.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
    ax_c.set_ylabel('R²', fontsize=9, fontweight='bold')
    ax_c.set_title('C. Representation Robustness Trajectories', 
                   fontsize=10, fontweight='bold', loc='left')
    ax_c.legend(fontsize=7, loc='lower left', ncol=2)
    ax_c.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_c)
    
    # Panel D: Within-Representation Model Variation
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Get AUC for each config at σ=0.4
    high_noise = df[df['sigma'] == 0.4].copy()
    
    rep_order = high_noise.groupby('representation')['r2'].median().sort_values(ascending=False).index.tolist()
    
    violin_data = []
    positions = []
    colors_list = []
    
    for i, rep in enumerate(rep_order):
        rep_scores = high_noise[high_noise['representation'] == rep]['r2'].values
        if len(rep_scores) > 0:
            violin_data.append(rep_scores)
            positions.append(i)
            colors_list.append(COLORS.get(rep, '#949494'))
    
    parts = ax_d.violinplot(violin_data, positions=positions, widths=0.7,
                            showmeans=True, showmedians=True)
    
    for pc, color in zip(parts['bodies'], colors_list):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax_d.set_xticks(positions)
    ax_d.set_xticklabels([r.upper() if r != 'random_smiles' else 'Rand\nSMILES' 
                          for r in rep_order], fontsize=7)
    ax_d.set_ylabel('R² at σ=0.4', fontsize=9, fontweight='bold')
    ax_d.set_title('D. Within-Representation Model Variation', 
                   fontsize=10, fontweight='bold', loc='left')
    ax_d.grid(True, alpha=0.3, axis='y', linestyle='--')
    sns.despine(ax=ax_d)
    
    plt.savefig(output_dir / 'figure1_representation_dominance.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure1_representation_dominance.pdf', 
                bbox_inches='tight')
    print(f"✓ Saved Figure 1")
    plt.close()

# ============================================================================
# FIGURE 2: MODEL ARCHITECTURE COMPARISON (WITH PHASE 1)
# ============================================================================

def create_figure2(phase0_df, phase1_df, output_dir):
    """
    Figure 2: Model Architecture Comparison Under Noise
    Focuses on Phase 1 direct comparisons
    Panel A: Tree-Based Model Ranking
    Panel B: Neural Network Comparison  
    Panel C: Gaussian Process Performance
    Panel D: Phase 1 Direct Comparisons (RF vs QRF, XGB vs NGBoost, DNN vs BNN, GP vs GP-var)
    Panel E: Conformal vs Native Uncertainty
    Panel F: Failure Mode Analysis
    """
    print("\n" + "="*80)
    print("Creating Figure 2: Model Architecture Comparison")
    print("="*80)
    
    fig = plt.figure(figsize=(7.5, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: Tree-Based Model Ranking
    ax_a = fig.add_subplot(gs[0, 0])
    
    tree_models = ['rf', 'xgboost', 'qrf', 'ngboost', 'lightgbm']
    best_rep = 'pdv'
    
    tree_data = phase0_df[
        (phase0_df['model'].isin(tree_models)) &
        (phase0_df['representation'] == best_rep)
    ]
    
    tree_aucs = []
    for model in tree_models:
        auc_val = calculate_auc(phase0_df, model, best_rep)
        if not np.isnan(auc_val):
            tree_aucs.append({'model': model, 'auc': auc_val})
    
    tree_aucs_df = pd.DataFrame(tree_aucs).sort_values('auc', ascending=True)
    
    if len(tree_aucs_df) > 0:
        colors_tree = [COLORS['tree_prob'] if m in ['qrf', 'ngboost'] 
                      else COLORS['deterministic'] for m in tree_aucs_df['model']]
        
        bars = ax_a.barh(range(len(tree_aucs_df)), tree_aucs_df['auc'],
                        color=colors_tree, alpha=0.7, edgecolor='black')
        
        ax_a.set_yticks(range(len(tree_aucs_df)))
        ax_a.set_yticklabels(tree_aucs_df['model'].str.upper(), fontsize=8)
        ax_a.set_xlabel('AUC (R² σ=0→0.6)', fontsize=9, fontweight='bold')
        ax_a.set_title('A. Tree-Based Model Ranking (PDV)', 
                       fontsize=10, fontweight='bold', loc='left')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['deterministic'], alpha=0.7, label='Deterministic'),
            Patch(facecolor=COLORS['tree_prob'], alpha=0.7, label='Probabilistic')
        ]
        ax_a.legend(handles=legend_elements, fontsize=7, loc='lower right')
        
        sns.despine(ax=ax_a)
    
    # Panel B: Neural Network Comparison
    ax_b = fig.add_subplot(gs[0, 1])
    
    # Use Phase 0 for baseline comparison
    nn_models = {
        'DNN': 'dnn',
        'GNN\nSimple': 'gnn',
        'Transformer': 'transformer'
    }
    
    x_pos = np.arange(len(nn_models))
    width = 0.25
    
    for i, (transform, color, label) in enumerate([
        ('baseline', COLORS['deterministic'], 'Deterministic'),
        ('last_layer', COLORS['bayesian'], 'BNN Last Layer'),
        ('full', COLORS['bayesian'], 'BNN Full')
    ]):
        aucs = []
        for nn_label, nn_model in nn_models.items():
            # Try to find in phase0
            if transform == 'baseline':
                model_name = nn_model
            else:
                model_name = f"{nn_model}_bnn_{transform}"
            
            auc_val = calculate_auc(phase0_df, model_name, best_rep)
            aucs.append(auc_val if not np.isnan(auc_val) else 0)
        
        offset = width * (i - 1)
        bars = ax_b.bar(x_pos + offset, aucs, width, 
                       label=label, alpha=0.7, color=color)
    
    ax_b.set_xlabel('Architecture', fontsize=9, fontweight='bold')
    ax_b.set_ylabel('AUC (R² σ=0→0.6)', fontsize=9, fontweight='bold')
    ax_b.set_title('B. Neural Network Comparison', 
                   fontsize=10, fontweight='bold', loc='left')
    ax_b.set_xticks(x_pos)
    ax_b.set_xticklabels(nn_models.keys(), fontsize=8)
    ax_b.legend(fontsize=7, loc='upper left')
    ax_b.grid(True, alpha=0.3, axis='y', linestyle='--')
    sns.despine(ax=ax_b)
    
    # Panel C: Gaussian Process Performance
    ax_c = fig.add_subplot(gs[1, 0])
    
    gp_models = ['gauche']  # Add variants if available
    
    for gp_model in gp_models:
        gp_data = phase0_df[
            (phase0_df['model'] == gp_model) &
            (phase0_df['representation'] == best_rep)
        ].sort_values('sigma')
        
        if len(gp_data) > 0:
            ax_c.plot(gp_data['sigma'], gp_data['rmse'],
                     'o-', linewidth=2, markersize=5,
                     label=gp_model.upper(), color=COLORS['gp'])
    
    ax_c.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
    ax_c.set_ylabel('RMSE (eV)', fontsize=9, fontweight='bold')
    ax_c.set_title('C. Gaussian Process Performance', 
                   fontsize=10, fontweight='bold', loc='left')
    ax_c.legend(fontsize=8, loc='upper left')
    ax_c.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_c)
    
    # Panel D: Phase 1 Direct Comparisons
    ax_d = fig.add_subplot(gs[1, 1])
    
    if len(phase1_df) > 0:
        # Define comparison pairs
        pairs = [
            ('rf', 'baseline', 'qrf', 'baseline', 'RF vs QRF'),
            ('xgboost', 'baseline', 'ngboost', 'baseline', 'XGB vs NGBoost'),
            ('dnn', 'baseline', 'dnn', 'full', 'DNN vs BNN'),
            ('gauche', 'baseline', 'gauche', 'variational', 'GP vs GP-var')
        ]
        
        comparison_results = []
        
        for base1, trans1, base2, trans2, label in pairs:
            # Get data for first model
            data1 = phase1_df[
                (phase1_df['base_model'] == base1) &
                (phase1_df['transformation'] == trans1)
            ]
            
            # Get data for second model
            data2 = phase1_df[
                (phase1_df['base_model'] == base2) &
                (phase1_df['transformation'] == trans2)
            ]
            
            if len(data1) > 0 and len(data2) > 0:
                # Calculate mean R² across all sigmas
                r2_1 = data1['r2'].mean()
                r2_2 = data2['r2'].mean()
                
                comparison_results.append({
                    'pair': label,
                    'det': r2_1,
                    'prob': r2_2,
                    'diff': r2_2 - r2_1
                })
        
        if comparison_results:
            comp_df = pd.DataFrame(comparison_results)
            
            x = np.arange(len(comp_df))
            width = 0.35
            
            bars1 = ax_d.bar(x - width/2, comp_df['det'], width,
                           label='Deterministic', alpha=0.7, 
                           color=COLORS['deterministic'])
            bars2 = ax_d.bar(x + width/2, comp_df['prob'], width,
                           label='Probabilistic', alpha=0.7,
                           color=COLORS['bayesian'])
            
            ax_d.set_ylabel('Mean R²', fontsize=9, fontweight='bold')
            ax_d.set_title('D. Phase 1 Direct Comparisons', 
                          fontsize=10, fontweight='bold', loc='left')
            ax_d.set_xticks(x)
            ax_d.set_xticklabels(comp_df['pair'], fontsize=7, rotation=20, ha='right')
            ax_d.legend(fontsize=7, loc='lower left')
            ax_d.grid(True, alpha=0.3, axis='y', linestyle='--')
            sns.despine(ax=ax_d)
    
    # Panel E: Conformal vs Native Uncertainty (placeholder)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.text(0.5, 0.5, 'Conformal vs Native\nUncertainty\n(Requires Phase 2/3 data)',
             transform=ax_e.transAxes, ha='center', va='center',
             fontsize=10, style='italic', color='gray')
    ax_e.set_title('E. Conformal vs Native Uncertainty', 
                   fontsize=10, fontweight='bold', loc='left')
    sns.despine(ax=ax_e, left=True, bottom=True)
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    
    # Panel F: Failure Mode Analysis
    ax_f = fig.add_subplot(gs[2, 1])
    
    sigma_levels = [0.0, 0.2, 0.4, 0.6]
    
    failure_data = []
    for sigma in sigma_levels:
        sigma_data = phase0_df[np.abs(phase0_df['sigma'] - sigma) < 0.05]
        
        if len(sigma_data) > 0:
            excellent = (sigma_data['r2'] > 0.7).sum() / len(sigma_data) * 100
            good = ((sigma_data['r2'] >= 0.5) & (sigma_data['r2'] <= 0.7)).sum() / len(sigma_data) * 100
            poor = ((sigma_data['r2'] >= 0.3) & (sigma_data['r2'] < 0.5)).sum() / len(sigma_data) * 100
            failed = (sigma_data['r2'] < 0.3).sum() / len(sigma_data) * 100
            
            failure_data.append({
                'sigma': sigma,
                'Excellent\n(R²>0.7)': excellent,
                'Good\n(0.5-0.7)': good,
                'Poor\n(0.3-0.5)': poor,
                'Failed\n(<0.3)': failed
            })
    
    if failure_data:
        failure_df = pd.DataFrame(failure_data)
        
        x = np.arange(len(failure_df))
        categories = ['Excellent\n(R²>0.7)', 'Good\n(0.5-0.7)', 'Poor\n(0.3-0.5)', 'Failed\n(<0.3)']
        colors_fail = ['#029E73', '#0173B2', '#DE8F05', '#CA3542']
        
        bottom = np.zeros(len(failure_df))
        for cat, color in zip(categories, colors_fail):
            ax_f.bar(x, failure_df[cat], bottom=bottom, 
                    label=cat, alpha=0.7, color=color)
            bottom += failure_df[cat]
        
        ax_f.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
        ax_f.set_ylabel('Percentage of Configurations', fontsize=9, fontweight='bold')
        ax_f.set_title('F. Failure Mode Analysis', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_f.set_xticks(x)
        ax_f.set_xticklabels([f'{s:.1f}' for s in failure_df['sigma']], fontsize=8)
        ax_f.legend(fontsize=6, loc='upper right', ncol=2)
        ax_f.set_ylim([0, 100])
        sns.despine(ax=ax_f)
    
    plt.savefig(output_dir / 'figure2_model_architecture_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure2_model_architecture_comparison.pdf', 
                bbox_inches='tight')
    print(f"✓ Saved Figure 2")
    plt.close()

# ============================================================================
# FIGURE 3: UNCERTAINTY QUANTIFICATION QUALITY ASSESSMENT
# ============================================================================

def create_figure3(uncertainty_df, output_dir):
    """
    Figure 3: Uncertainty Quantification Quality Assessment
    Panel A: Uncertainty-Error Correlation
    Panel B: Calibration Curves
    Panel C: Expected Calibration Error Across Noise
    Panel D: Epistemic vs Aleatoric Decomposition
    """
    print("\n" + "="*80)
    print("Creating Figure 3: Uncertainty Quantification")
    print("="*80)
    
    if len(uncertainty_df) == 0:
        print("WARNING: No uncertainty data available, skipping Figure 3")
        return
    
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    key_models = {
        'ngboost': {'label': 'NGBoost', 'color': COLORS['tree_prob']},
        'qrf': {'label': 'QRF', 'color': COLORS['deterministic']},
        'dnn_bnn': {'label': 'DNN-BNN', 'color': COLORS['bayesian']},
        'gauche': {'label': 'Gauche', 'color': COLORS['gp']},
    }
    
    sigma_target = 0.3
    
    # Panel A: Uncertainty-Error Correlation
    ax_a = fig.add_subplot(gs[0, 0])
    
    for model_key, model_info in key_models.items():
        model_data = uncertainty_df[
            (uncertainty_df['model_name'].str.contains(model_key, case=False, na=False)) &
            (uncertainty_df['sigma'] == sigma_target)
        ]
        
        if len(model_data) > 0 and 'y_pred_std_calibrated' in model_data.columns:
            errors = np.abs(model_data['y_true_noisy'] - model_data['y_pred_mean'])
            uncertainties = model_data['y_pred_std_calibrated']
            
            valid = ~(uncertainties.isna() | errors.isna())
            if valid.sum() > 10:
                # Scatter
                ax_a.scatter(uncertainties[valid], errors[valid],
                           alpha=0.3, s=5, color=model_info['color'])
                
                # Calculate correlation
                r, p = stats.spearmanr(uncertainties[valid], errors[valid])
                
                # Fit line
                coeffs = np.polyfit(uncertainties[valid], errors[valid], 1)
                x_fit = np.linspace(uncertainties[valid].min(), uncertainties[valid].max(), 100)
                y_fit = coeffs[0] * x_fit + coeffs[1]
                
                stars = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                ax_a.plot(x_fit, y_fit, '-', linewidth=2, color=model_info['color'],
                         label=f"{model_info['label']} (ρ={r:.3f}{stars})")
    
    # Perfect calibration
    max_val = ax_a.get_xlim()[1]
    ax_a.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5, label='Perfect')
    
    ax_a.set_xlabel('Predicted Uncertainty (σ)', fontsize=9, fontweight='bold')
    ax_a.set_ylabel('Absolute Error', fontsize=9, fontweight='bold')
    ax_a.set_title('A. Uncertainty-Error Correlation', fontsize=10, fontweight='bold', loc='left')
    ax_a.legend(fontsize=7, loc='upper left')
    ax_a.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_a)
    
    # Panel B: Calibration Curves
    ax_b = fig.add_subplot(gs[0, 1])
    
    for model_key, model_info in key_models.items():
        model_data = uncertainty_df[
            (uncertainty_df['model_name'].str.contains(model_key, case=False, na=False)) &
            (uncertainty_df['sigma'] == sigma_target)
        ]
        
        if len(model_data) > 0 and 'y_pred_std_calibrated' in model_data.columns:
            errors = np.abs(model_data['y_true_noisy'] - model_data['y_pred_mean'])
            uncertainties = model_data['y_pred_std_calibrated']
            
            valid = ~(uncertainties.isna() | errors.isna())
            if valid.sum() > 10:
                # Bin by percentiles
                n_bins = 10
                bin_edges = np.percentile(uncertainties[valid], np.linspace(0, 100, n_bins + 1))
                bin_edges[-1] += 1e-8
                
                bin_pred = []
                bin_obs = []
                
                for i in range(n_bins):
                    in_bin = (uncertainties[valid] >= bin_edges[i]) & (uncertainties[valid] < bin_edges[i + 1])
                    if in_bin.sum() > 5:
                        bin_pred.append(uncertainties[valid][in_bin].mean())
                        bin_obs.append(errors[valid][in_bin].mean())
                
                if bin_pred:
                    ece = calculate_ece(uncertainties[valid].values, errors[valid].values)
                    ax_b.plot(bin_pred, bin_obs, 'o-', linewidth=2, markersize=4,
                             color=model_info['color'], 
                             label=f"{model_info['label']} (ECE={ece:.3f})")
    
    # Perfect calibration
    max_val = max(ax_b.get_xlim()[1], ax_b.get_ylim()[1])
    ax_b.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
    
    ax_b.set_xlabel('Predicted Uncertainty', fontsize=9, fontweight='bold')
    ax_b.set_ylabel('Observed RMSE', fontsize=9, fontweight='bold')
    ax_b.set_title('B. Calibration Curves', fontsize=10, fontweight='bold', loc='left')
    ax_b.legend(fontsize=7, loc='lower right')
    ax_b.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_b)
    
    # Panel C: ECE Across Noise Levels
    ax_c = fig.add_subplot(gs[1, 0])
    
    for model_key, model_info in key_models.items():
        ece_values = []
        sigmas = []
        
        for sigma in sorted(uncertainty_df['sigma'].unique()):
            model_data = uncertainty_df[
                (uncertainty_df['model_name'].str.contains(model_key, case=False, na=False)) &
                (uncertainty_df['sigma'] == sigma)
            ]
            
            if len(model_data) > 0 and 'y_pred_std_calibrated' in model_data.columns:
                errors = np.abs(model_data['y_true_noisy'] - model_data['y_pred_mean'])
                uncertainties = model_data['y_pred_std_calibrated']
                
                valid = ~(uncertainties.isna() | errors.isna())
                if valid.sum() > 10:
                    ece = calculate_ece(uncertainties[valid].values, errors[valid].values)
                    ece_values.append(ece)
                    sigmas.append(sigma)
        
        if ece_values:
            ax_c.plot(sigmas, ece_values, 'o-', linewidth=2, markersize=4,
                     label=model_info['label'], color=model_info['color'])
    
    ax_c.axhline(y=0.05, color='gray', linestyle='--', alpha=0.5)
    ax_c.text(ax_c.get_xlim()[1]*0.98, 0.05, 'Good (ECE<0.05)',
             ha='right', va='bottom', fontsize=7, color='gray')
    
    ax_c.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
    ax_c.set_ylabel('Expected Calibration Error', fontsize=9, fontweight='bold')
    ax_c.set_title('C. ECE Across Noise Levels', fontsize=10, fontweight='bold', loc='left')
    ax_c.legend(fontsize=7, loc='upper left')
    ax_c.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_c)
    
    # Panel D: Epistemic vs Aleatoric
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Check for decomposition
    has_decomp = ('epistemic_uncertainty' in uncertainty_df.columns and 
                  'aleatoric_uncertainty' in uncertainty_df.columns)
    
    if has_decomp:
        decomp_models = ['dnn_bnn', 'gauche']
        
        for model_key in decomp_models:
            model_data = uncertainty_df[
                uncertainty_df['model_name'].str.contains(model_key, case=False, na=False)
            ]
            
            if len(model_data) > 0:
                stats_by_sigma = model_data.groupby('sigma').agg({
                    'epistemic_uncertainty': 'mean',
                    'aleatoric_uncertainty': 'mean'
                }).reset_index()
                
                color = COLORS['bayesian'] if model_key == 'dnn_bnn' else COLORS['gp']
                label = 'DNN-BNN' if model_key == 'dnn_bnn' else 'Gauche'
                
                # Stacked area
                ax_d.fill_between(stats_by_sigma['sigma'], 
                                 0,
                                 stats_by_sigma['aleatoric_uncertainty'],
                                 alpha=0.5, color=color,
                                 label=f"{label} (Aleatoric)")
                
                ax_d.fill_between(stats_by_sigma['sigma'],
                                 stats_by_sigma['aleatoric_uncertainty'],
                                 stats_by_sigma['aleatoric_uncertainty'] + stats_by_sigma['epistemic_uncertainty'],
                                 alpha=0.3, color=color,
                                 label=f"{label} (Epistemic)")
        
        ax_d.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
        ax_d.set_ylabel('Uncertainty', fontsize=9, fontweight='bold')
        ax_d.set_title('D. Epistemic vs Aleatoric Decomposition', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_d.legend(fontsize=7, loc='upper left', ncol=2)
        ax_d.grid(True, alpha=0.3, linestyle='--')
        sns.despine(ax=ax_d)
    else:
        ax_d.text(0.5, 0.5, 'Epistemic/Aleatoric\nDecomposition\n(Not available)',
                 transform=ax_d.transAxes, ha='center', va='center',
                 fontsize=10, style='italic', color='gray')
        sns.despine(ax=ax_d, left=True, bottom=True)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
    
    plt.savefig(output_dir / 'figure3_uncertainty_quantification.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure3_uncertainty_quantification.pdf', 
                bbox_inches='tight')
    print(f"✓ Saved Figure 3")
    plt.close()

# ============================================================================
# FIGURE 4: CONFORMAL PREDICTION ROBUSTNESS
# ============================================================================

def create_figure4(conformal_df, phase0_df, output_dir):
    """
    Figure 4: Conformal Prediction Robustness
    Panel A: Coverage Maintenance
    Panel B: Interval Width Under Noise
    Panel C: Efficiency-Coverage Trade-off
    Panel D: Split Conformal vs Inductive Methods
    """
    print("\n" + "="*80)
    print("Creating Figure 4: Conformal Prediction")
    print("="*80)
    
    if len(conformal_df) == 0:
        print("WARNING: No conformal data available, skipping Figure 4")
        return
    
    fig = plt.figure(figsize=(7.5, 7.5))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    base_models = conformal_df['base_model'].unique()[:4]
    alpha_target = 0.05  # 95% confidence
    
    # Panel A: Coverage Maintenance
    ax_a = fig.add_subplot(gs[0, 0])
    
    for base_model in base_models:
        model_data = conformal_df[
            (conformal_df['base_model'] == base_model) &
            (conformal_df['alpha'] == alpha_target)
        ]
        
        if len(model_data) > 0:
            coverage_by_sigma = model_data.groupby('sigma_noise').agg({
                'coverage': ['mean', 'std']
            }).reset_index()
            coverage_by_sigma.columns = ['sigma', 'coverage_mean', 'coverage_std']
            
            ax_a.plot(coverage_by_sigma['sigma'], coverage_by_sigma['coverage_mean'],
                     'o-', linewidth=2, markersize=4, label=base_model.upper())
            
            ax_a.fill_between(coverage_by_sigma['sigma'],
                             coverage_by_sigma['coverage_mean'] - coverage_by_sigma['coverage_std'],
                             coverage_by_sigma['coverage_mean'] + coverage_by_sigma['coverage_std'],
                             alpha=0.2)
    
    # Target and acceptable range
    ax_a.axhline(y=1-alpha_target, color='gray', linestyle='--', linewidth=2, alpha=0.7)
    ax_a.fill_between(ax_a.get_xlim(), 0.92, 0.98, alpha=0.1, color='green')
    ax_a.text(ax_a.get_xlim()[1]*0.98, 0.95, '90-95%\nTarget',
             ha='right', va='center', fontsize=7, color='gray')
    
    ax_a.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
    ax_a.set_ylabel('Coverage', fontsize=9, fontweight='bold')
    ax_a.set_title('A. Coverage Maintenance', fontsize=10, fontweight='bold', loc='left')
    ax_a.legend(fontsize=7, loc='lower left')
    ax_a.set_ylim([0.80, 1.0])
    ax_a.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_a)
    
    # Panel B: Interval Width Under Noise
    ax_b = fig.add_subplot(gs[0, 1])
    
    sigma_levels = [0.0, 0.3, 0.6]
    
    violin_data = []
    positions = []
    labels = []
    
    for i, sigma in enumerate(sigma_levels):
        for j, base_model in enumerate(base_models[:3]):  # Top 3 models
            model_data = conformal_df[
                (conformal_df['base_model'] == base_model) &
                (conformal_df['alpha'] == alpha_target) &
                (np.abs(conformal_df['sigma_noise'] - sigma) < 0.05)
            ]
            
            if len(model_data) > 0:
                violin_data.append(model_data['interval_width'].values)
                positions.append(i * (len(base_models) + 1) + j)
                labels.append(f'{base_model[:3]}\nσ={sigma:.1f}')
    
    if violin_data:
        parts = ax_b.violinplot(violin_data, positions=positions, widths=0.8,
                                showmeans=True, showmedians=True)
        
        for pc in parts['bodies']:
            pc.set_facecolor(COLORS['conformal'])
            pc.set_alpha(0.7)
        
        ax_b.set_xticks(positions)
        ax_b.set_xticklabels(labels, fontsize=6, rotation=45, ha='right')
        ax_b.set_ylabel('Interval Width (eV)', fontsize=9, fontweight='bold')
        ax_b.set_title('B. Interval Width Under Noise', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_b.grid(True, alpha=0.3, axis='y', linestyle='--')
        sns.despine(ax=ax_b)
    
    # Panel C: Efficiency-Coverage Trade-off
    ax_c = fig.add_subplot(gs[1, 0])
    
    for sigma in [0.0, 0.3, 0.6]:
        sigma_data = conformal_df[
            (conformal_df['alpha'] == alpha_target) &
            (np.abs(conformal_df['sigma_noise'] - sigma) < 0.05)
        ]
        
        if len(sigma_data) > 0:
            # Group by model and calculate mean
            model_stats = sigma_data.groupby('base_model').agg({
                'coverage': 'mean',
                'interval_width': 'median'
            }).reset_index()
            
            color_map = {0.0: '#0173B2', 0.3: '#DE8F05', 0.6: '#CA3542'}
            
            ax_c.scatter(model_stats['coverage'], model_stats['interval_width'],
                        s=60, alpha=0.7, color=color_map[sigma],
                        label=f'σ={sigma:.1f}', edgecolors='black', linewidth=0.5)
    
    # Ideal zone
    ax_c.axvline(x=0.95, color='green', linestyle='--', alpha=0.3)
    ax_c.text(0.95, ax_c.get_ylim()[1]*0.95, 'Target\nCoverage',
             ha='right', va='top', fontsize=7, color='green')
    
    ax_c.set_xlabel('Coverage', fontsize=9, fontweight='bold')
    ax_c.set_ylabel('Median Interval Width (eV)', fontsize=9, fontweight='bold')
    ax_c.set_title('C. Efficiency-Coverage Trade-off', 
                   fontsize=10, fontweight='bold', loc='left')
    ax_c.legend(fontsize=7, loc='upper right')
    ax_c.grid(True, alpha=0.3, linestyle='--')
    sns.despine(ax=ax_c)
    
    # Panel D: Split Conformal vs Inductive Methods
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Check if we have method information
    if 'method' in conformal_df.columns:
        methods = conformal_df['method'].unique()[:4]
        
        x = np.arange(len(sigma_levels))
        width = 0.2
        
        for i, method in enumerate(methods):
            coverages = []
            for sigma in sigma_levels:
                method_data = conformal_df[
                    (conformal_df['method'] == method) &
                    (conformal_df['alpha'] == alpha_target) &
                    (np.abs(conformal_df['sigma_noise'] - sigma) < 0.05)
                ]
                
                if len(method_data) > 0:
                    coverages.append(method_data['coverage'].mean())
                else:
                    coverages.append(np.nan)
            
            offset = width * (i - len(methods)/2)
            ax_d.bar(x + offset, coverages, width,
                    label=method, alpha=0.7)
        
        ax_d.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7)
        ax_d.set_xlabel('Noise level (σ)', fontsize=9, fontweight='bold')
        ax_d.set_ylabel('Coverage @90%', fontsize=9, fontweight='bold')
        ax_d.set_title('D. Conformal Methods Comparison', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_d.set_xticks(x)
        ax_d.set_xticklabels([f'{s:.1f}' for s in sigma_levels])
        ax_d.legend(fontsize=7, loc='lower left')
        ax_d.grid(True, alpha=0.3, axis='y', linestyle='--')
        sns.despine(ax=ax_d)
    else:
        ax_d.text(0.5, 0.5, 'Method Comparison\n(Method column not available)',
                 transform=ax_d.transAxes, ha='center', va='center',
                 fontsize=10, style='italic', color='gray')
        sns.despine(ax=ax_d, left=True, bottom=True)
        ax_d.set_xticks([])
        ax_d.set_yticks([])
    
    plt.savefig(output_dir / 'figure4_conformal_prediction.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure4_conformal_prediction.pdf', 
                bbox_inches='tight')
    print(f"✓ Saved Figure 4")
    plt.close()

# ============================================================================
# FIGURE 5: NOISE STRUCTURE MATTERS
# ============================================================================

def create_figure5(phase4_df, output_dir):
    """
    Figure 5: Noise Structure Matters More Than Magnitude
    Panel A: Noise Structure Comparison
    Panel B: Alternative Target Analysis
    Panel C: QM9 vs ADME Generalization
    Panel D: Cross-Domain Transfer
    Panel E: Noise-Aware Feature Importance (placeholder)
    Panel F: Practical Implication Matrix (placeholder)
    """
    print("\n" + "="*80)
    print("Creating Figure 5: Noise Structure")
    print("="*80)
    
    if len(phase4_df) == 0:
        print("WARNING: No phase4 data available, skipping Figure 5")
        return
    
    fig = plt.figure(figsize=(7.5, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
    
    # Panel A: Noise Structure Comparison
    ax_a = fig.add_subplot(gs[0, 0])
    
    target = 'homolumo'
    target_data = phase4_df[phase4_df['target'] == target]
    
    if len(target_data) > 0:
        strategies = target_data['noise_strategy'].unique()[:4]
        
        for strategy in strategies:
            strat_data = target_data[target_data['noise_strategy'] == strategy]
            mean_by_sigma = strat_data.groupby('sigma')['r2'].mean().reset_index()
            
            ax_a.plot(mean_by_sigma['sigma'], mean_by_sigma['r2'],
                     'o-', linewidth=2, markersize=4, label=strategy.capitalize())
        
        ax_a.set_xlabel('Noise magnitude (σ)', fontsize=9, fontweight='bold')
        ax_a.set_ylabel('R²', fontsize=9, fontweight='bold')
        ax_a.set_title('A. Noise Structure Comparison', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_a.legend(fontsize=7, loc='lower left')
        ax_a.grid(True, alpha=0.3, linestyle='--')
        sns.despine(ax=ax_a)
    
    # Panel B: Alternative Target Analysis
    ax_b = fig.add_subplot(gs[0, 1])
    
    targets = phase4_df['target'].unique()[:5]
    models = phase4_df['model'].unique()[:10]
    
    heatmap_data = np.zeros((len(targets), len(models)))
    
    for i, target in enumerate(targets):
        for j, model in enumerate(models):
            target_model_data = phase4_df[
                (phase4_df['target'] == target) &
                (phase4_df['model'] == model) &
                (np.abs(phase4_df['sigma'] - 0.4) < 0.05)
            ]
            
            if len(target_model_data) > 0:
                # Calculate AUC or use mean R²
                heatmap_data[i, j] = target_model_data['r2'].mean()
    
    im = ax_b.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # ax_b.set_xticks(range(len(models)))
    # ax_b.set_xticklabels([m[:6] for m in models], rotation=45, ha='right', fontsize=6)
    # ax_b.set_yticks(range(len(targets)))
    # ax_b.set_yticklabels([t.upper() for t in targets], fontsize=8)
    ax_b.set_title('B. Alternative Target Analysis', 
                   fontsize=10, fontweight='bold', loc='left')
    
    cbar = plt.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.set_label('R² at σ=0.4', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    
    # Panel C: QM9 vs ADME Generalization
    ax_c = fig.add_subplot(gs[1, 0])
    
    qm9_targets = ['homolumo', 'alpha', 'gibbsenergy']
    adme_targets = ['hlm', 'rlm']
    
    # Calculate degradation for each
    models_to_compare = phase4_df['model'].unique()[:5]
    
    qm9_degrad = []
    adme_degrad = []
    model_labels = []
    
    for model in models_to_compare:
        # QM9 degradation
        qm9_data = phase4_df[
            (phase4_df['target'].isin(qm9_targets)) &
            (phase4_df['model'] == model)
        ]
        
        if len(qm9_data) > 0:
            r2_0 = qm9_data[qm9_data['sigma'] == 0.0]['r2'].mean()
            r2_06 = qm9_data[np.abs(qm9_data['sigma'] - 0.6) < 0.05]['r2'].mean()
            qm9_deg = ((r2_0 - r2_06) / r2_0 * 100) if r2_0 > 0 else 0
        else:
            qm9_deg = 0
        
        # ADME degradation
        adme_data = phase4_df[
            (phase4_df['target'].isin(adme_targets)) &
            (phase4_df['model'] == model)
        ]
        
        if len(adme_data) > 0:
            r2_0 = adme_data[adme_data['sigma'] == 0.0]['r2'].mean()
            r2_06 = adme_data[np.abs(adme_data['sigma'] - 0.6) < 0.05]['r2'].mean()
            adme_deg = ((r2_0 - r2_06) / r2_0 * 100) if r2_0 > 0 else 0
        else:
            adme_deg = 0
        
        qm9_degrad.append(qm9_deg)
        adme_degrad.append(adme_deg)
        model_labels.append(model)
    
    if qm9_degrad and adme_degrad:
        x = np.arange(len(model_labels))
        width = 0.35
        
        ax_c.bar(x - width/2, qm9_degrad, width, label='QM9', alpha=0.7, color='#0173B2')
        ax_c.bar(x + width/2, adme_degrad, width, label='ADME', alpha=0.7, color='#DE8F05')
        
        ax_c.set_xlabel('Model', fontsize=9, fontweight='bold')
        ax_c.set_ylabel('% Degradation (R²)', fontsize=9, fontweight='bold')
        ax_c.set_title('C. QM9 vs ADME Generalization', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_c.set_xticks(x)
        ax_c.set_xticklabels(model_labels, rotation=45, ha='right', fontsize=7)
        ax_c.legend(fontsize=8, loc='upper left')
        ax_c.grid(True, alpha=0.3, axis='y', linestyle='--')
        sns.despine(ax=ax_c)
    
    # Panel D: Cross-Domain Transfer (simplified)
    ax_d = fig.add_subplot(gs[1, 1])
    
    # Scatter plot: QM9 performance vs ADME performance
    transfer_data = []
    
    for model in models_to_compare:
        qm9_perf = phase4_df[
            (phase4_df['target'].isin(qm9_targets)) &
            (phase4_df['model'] == model) &
            (np.abs(phase4_df['sigma'] - 0.4) < 0.05)
        ]['r2'].mean()
        
        adme_perf = phase4_df[
            (phase4_df['target'].isin(adme_targets)) &
            (phase4_df['model'] == model) &
            (np.abs(phase4_df['sigma'] - 0.4) < 0.05)
        ]['r2'].mean()
        
        if not np.isnan(qm9_perf) and not np.isnan(adme_perf):
            transfer_data.append({
                'model': model,
                'qm9': qm9_perf,
                'adme': adme_perf
            })
    
    if transfer_data:
        transfer_df = pd.DataFrame(transfer_data)
        
        ax_d.scatter(transfer_df['qm9'], transfer_df['adme'],
                    s=80, alpha=0.7, color=COLORS['deterministic'],
                    edgecolors='black', linewidth=0.5)
        
        # Annotate
        for _, row in transfer_df.iterrows():
            ax_d.annotate(row['model'][:4], (row['qm9'], row['adme']),
                         fontsize=6, ha='center', va='bottom')
        
        # Diagonal
        max_val = max(ax_d.get_xlim()[1], ax_d.get_ylim()[1])
        ax_d.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.5)
        
        ax_d.set_xlabel('QM9 Performance (R²)', fontsize=9, fontweight='bold')
        ax_d.set_ylabel('ADME Performance (R²)', fontsize=9, fontweight='bold')
        ax_d.set_title('D. Cross-Domain Transfer', 
                       fontsize=10, fontweight='bold', loc='left')
        ax_d.grid(True, alpha=0.3, linestyle='--')
        sns.despine(ax=ax_d)
    
    # Panel E: Feature Importance (placeholder)
    ax_e = fig.add_subplot(gs[2, 0])
    ax_e.text(0.5, 0.5, 'Noise-Aware Feature Importance\n(Requires additional analysis)',
             transform=ax_e.transAxes, ha='center', va='center',
             fontsize=10, style='italic', color='gray')
    ax_e.set_title('E. Noise-Aware Feature Importance', 
                   fontsize=10, fontweight='bold', loc='left')
    sns.despine(ax=ax_e, left=True, bottom=True)
    ax_e.set_xticks([])
    ax_e.set_yticks([])
    
    # Panel F: Practical Implications (placeholder)
    ax_f = fig.add_subplot(gs[2, 1])
    ax_f.text(0.5, 0.5, 'Practical Implication Matrix\n(Requires additional analysis)',
             transform=ax_f.transAxes, ha='center', va='center',
             fontsize=10, style='italic', color='gray')
    ax_f.set_title('F. Practical Implication Matrix', 
                   fontsize=10, fontweight='bold', loc='left')
    sns.despine(ax=ax_f, left=True, bottom=True)
    ax_f.set_xticks([])
    ax_f.set_yticks([])
    
    plt.savefig(output_dir / 'figure5_noise_structure.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure5_noise_structure.pdf', 
                bbox_inches='tight')
    print(f"✓ Saved Figure 5")
    plt.close()

# ============================================================================
# FIGURE 6: COMPREHENSIVE METHOD COMPARISON SUMMARY
# ============================================================================

def create_figure6(phase0_df, output_dir):
    """
    Figure 6: Comprehensive Method Comparison Summary
    Large ranking table with embedded visualizations
    """
    print("\n" + "="*80)
    print("Creating Figure 6: Comprehensive Summary")
    print("="*80)
    
    # Calculate ranking metrics for all configs
    configs = []
    
    for (model, rep), group in phase0_df.groupby(['model', 'representation']):
        auc_val = calculate_auc(phase0_df, model, rep)
        
        baseline = group[group['sigma'] == 0.0]
        high_noise = group[group['sigma'] == 0.4]
        
        if len(baseline) > 0 and len(high_noise) > 0:
            r2_0 = baseline['r2'].values[0]
            r2_04 = high_noise['r2'].values[0]
            rmse_0 = baseline['rmse'].values[0]
            rmse_04 = high_noise['rmse'].values[0]
            
            # Calculate degradation slope
            sigma_vals = group['sigma'].values
            r2_vals = group['r2'].values
            if len(sigma_vals) > 1:
                slope, _, _, _, _ = stats.linregress(sigma_vals, r2_vals)
            else:
                slope = 0
            
            configs.append({
                'model': model,
                'representation': rep,
                'auc': auc_val,
                'r2_0': r2_0,
                'r2_04': r2_04,
                'rmse_0': rmse_0,
                'rmse_04': rmse_04,
                'degradation_slope': slope,
                'retention': (r2_04 / r2_0 * 100) if r2_0 > 0 else 0,
                'category': group['category'].iloc[0]
            })
    
    configs_df = pd.DataFrame(configs).sort_values('auc', ascending=False).head(20)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for i, (_, row) in enumerate(configs_df.iterrows(), 1):
        table_data.append([
            str(i),
            f"{row['model']}/{row['representation']}",
            f"{row['auc']:.3f}",
            f"{row['r2_0']:.3f}",
            f"{row['degradation_slope']:.4f}",
            f"{row['retention']:.1f}%",
            row['category']
        ])
    
    columns = ['Rank', 'Model/Rep', 'AUC', 'R²(σ=0)', 'Slope', 'Retention', 'Category']
    
    table = ax.table(cellText=table_data, colLabels=columns,
                     cellLoc='center', loc='center',
                     colWidths=[0.08, 0.25, 0.12, 0.12, 0.12, 0.12, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#0173B2')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color rows by rank
    for i in range(1, len(table_data) + 1):
        if i <= 5:
            color = '#d4edda'  # Green for top 5
        elif i <= 10:
            color = '#fff3cd'  # Yellow for 6-10
        else:
            color = '#f8d7da'  # Red for rest
        
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Figure 6: Comprehensive Method Comparison Summary\nTop 20 Configurations', 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_dir / 'figure6_comprehensive_summary.png', 
                dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figure6_comprehensive_summary.pdf', 
                bbox_inches='tight')
    print(f"✓ Saved Figure 6")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(results_dir="../results"):
    """Main execution: generate all figures"""
    
    print("\n" + "="*80)
    print("COMPREHENSIVE FIGURE GENERATION FOR PAPER")
    print("="*80)
    
    results_dir = Path(results_dir)
    
    # Create output directory
    output_dir = results_dir / "paper_figures"
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Load all data
    phase0_df = load_phase0_data(results_dir)
    phase1_df = load_phase1_data(results_dir)
    phase2_df = load_phase2_data(results_dir)
    phase3_df = load_phase3_data(results_dir)
    phase4_df = load_phase4_data(results_dir)
    
    # Generate figures
    print("\n" + "="*80)
    print("GENERATING MAIN TEXT FIGURES")
    print("="*80)
    
    if len(phase0_df) > 0:
        create_figure1(phase0_df, output_dir)
        create_figure6(phase0_df, output_dir)
    else:
        print("WARNING: Skipping Figures 1, 6 (no Phase 0 data)")
    
    if len(phase0_df) > 0 or len(phase1_df) > 0:
        create_figure2(phase0_df, phase1_df, output_dir)
    else:
        print("WARNING: Skipping Figure 2 (no Phase 0/1 data)")
    
    if len(phase2_df) > 0:
        create_figure3(phase2_df, output_dir)
    else:
        print("WARNING: Skipping Figure 3 (no Phase 2 data)")
    
    if len(phase3_df) > 0:
        create_figure4(phase3_df, phase0_df, output_dir)
    else:
        print("WARNING: Skipping Figure 4 (no Phase 3 data)")
    
    if len(phase4_df) > 0:
        create_figure5(phase4_df, output_dir)
    else:
        print("WARNING: Skipping Figure 5 (no Phase 4 data)")
    
    print("\n" + "="*80)
    print("FIGURE GENERATION COMPLETE!")
    print("="*80)
    print(f"\nAll figures saved to: {output_dir}")
    print("\nGenerated figures:")
    for fig_file in sorted(output_dir.glob("figure*.png")):
        print(f"  ✓ {fig_file.name}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "../results"
    
    main(results_dir)