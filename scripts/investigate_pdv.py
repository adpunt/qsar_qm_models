#!/usr/bin/env python3
"""
Investigation: Why does binary PDV (threshold > 0) work so well?

This script:
1. Loads QM9 homo_lumo_gap data (n=10k, scaffold split)
2. Computes raw PDVs and prints detailed statistics
3. Categorizes each descriptor by its behavior under binarization
4. Runs RF experiments with different PDV transformations, measuring both
   baseline R² and NDS (Noise Degradation Slope) from sigma=0 to sigma=0.5
5. Uses NoiseInject (legacy strategy) for noise injection
6. Prints RF feature importances and sample PDVs

PDV transformations tested:
   - binary_gt0:        (pdv > 0)          — the "broken" version used for a year
   - continuous_raw:    raw float values   — no normalization
   - continuous_zscore: z-score normalized (mean=0, std=1)
   - continuous_minmax: min-max to [-1, 1]
   - binary_mean:       (pdv > per-feature mean)
   - binary_median:     (pdv > per-feature median)
   - binary_nonzero:    (pdv != 0)        — same as gt0 for counts, different for negatives
   - binary_gt0_no_dead: binary > 0 but drops constant columns

Usage:
    python investigate_pdv.py [--n-samples 10000] [--n-iterations 5] [--seed 42]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# RDKit
from rdkit import Chem, RDLogger
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

# Sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# RDKit scaffold splitting (avoids deepchem dependency)
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict

import os.path as osp

# NoiseInject
sys.path.insert(0, osp.join(osp.dirname(osp.realpath(__file__)), '..', '..', 'NoiseInject'))
from noiseInject import NoiseInjectorRegression, calculate_noise_metrics

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
DEFAULT_DESCRIPTOR_LIST = [
    'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
    'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10',
    'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
    'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
    'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt',
    'Ipc', 'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge',
    'MaxEStateIndex', 'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex',
    'MinPartialCharge', 'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles',
    'NumAliphaticHeterocycles', 'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles',
    'NumAromaticRings', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons',
    'NumRotatableBonds', 'NumSaturatedCarbocycles', 'NumSaturatedHeterocycles', 'NumSaturatedRings',
    'NumValenceElectrons', 'PEOE_VSA1', 'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13',
    'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4', 'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7',
    'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2', 'SMR_VSA3', 'SMR_VSA4',
    'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10', 'SlogP_VSA11',
    'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
    'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3',
    'VSA_EState4', 'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9', 'fr_Al_COO',
    'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO', 'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO',
    'fr_COO2', 'fr_C_O', 'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1', 'fr_NH2',
    'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2', 'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde',
    'fr_alkyl_carbamate', 'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine', 'fr_aniline',
    'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur', 'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic',
    'fr_diazo', 'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether', 'fr_furan', 'fr_guanido',
    'fr_halogen', 'fr_hdrzine', 'fr_hdrzone', 'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
    'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone', 'fr_methoxy', 'fr_morpholine', 'fr_nitrile',
    'fr_nitro', 'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole', 'fr_oxime',
    'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond', 'fr_phos_acid', 'fr_phos_ester',
    'fr_piperdine', 'fr_piperzine', 'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN', 'fr_sulfide',
    'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene', 'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan',
    'fr_thiophene', 'fr_unbrch_alkane', 'fr_urea', 'qed'
]

PROPERTIES = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

SIGMA_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

QM9_RAW_DIR = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9', 'raw')

# QM9 CSV column mapping (gap = homo_lumo_gap = lumo - homo)
QM9_TARGET_COLS = {
    'homo_lumo_gap': 'gap',
    'alpha': 'alpha',
    'mu': 'mu',
}


# ──────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────
def compute_pdv(smiles):
    """Compute raw PDV from SMILES string."""
    calc = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    vals = calc.CalcDescriptors(mol)
    return np.array(vals, dtype=np.float64)


def load_qm9_data(target='homo_lumo_gap', n_samples=10000, seed=42):
    """Load QM9 from raw CSV + SDF, filtering uncharacterized molecules."""
    print(f"Loading QM9 (target={target}, n={n_samples})...")

    # Read properties CSV
    csv_path = osp.join(QM9_RAW_DIR, 'gdb9.sdf.csv')
    df = pd.read_csv(csv_path)
    target_col = QM9_TARGET_COLS[target]
    print(f"  CSV loaded: {len(df)} molecules, target column: '{target_col}'")

    # Read uncharacterized indices to exclude
    unchar_path = osp.join(QM9_RAW_DIR, 'uncharacterized.txt')
    exclude_indices = set()
    with open(unchar_path) as f:
        for line in f.readlines()[4:]:
            parts = line.strip().split()
            if parts and parts[0].isdigit():
                exclude_indices.add(int(parts[0]))
    print(f"  Excluding {len(exclude_indices)} uncharacterized molecules")

    # Get SMILES from SDF
    sdf_path = osp.join(QM9_RAW_DIR, 'gdb9.sdf')
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True)

    smiles_all = []
    targets_all = []
    for i, mol in enumerate(supplier):
        mol_idx = i + 1  # gdb9 is 1-indexed
        if mol_idx in exclude_indices:
            continue
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)
        if smi and i < len(df):
            smiles_all.append(smi)
            targets_all.append(df[target_col].iloc[i])

    print(f"  Valid molecules: {len(smiles_all)}")

    # Shuffle and sample
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(smiles_all))[:n_samples]

    smiles_list = [smiles_all[i] for i in indices]
    target_list = np.array([targets_all[i] for i in indices])

    print(f"  Sampled {len(smiles_list)} molecules")
    return smiles_list, target_list


def scaffold_split(smiles_list, targets, seed=42):
    """Scaffold split: 80/10/10 using RDKit MurckoScaffold (same logic as DeepChem)."""
    print("Performing scaffold split...")

    # Group molecule indices by Murcko scaffold
    scaffold_to_indices = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                mol=mol, includeChirality=False)
            scaffold_to_indices[scaffold].append(i)
        else:
            scaffold_to_indices['NONE'].append(i)

    # Sort scaffolds by size (largest first) for deterministic splitting
    scaffold_sets = sorted(scaffold_to_indices.values(),
                           key=lambda x: (len(x), x[0]), reverse=True)

    train_idx, val_idx, test_idx = [], [], []
    n = len(smiles_list)
    train_cutoff = int(n * 0.8)
    val_cutoff = int(n * 0.9)

    for scaffold_indices in scaffold_sets:
        if len(train_idx) < train_cutoff:
            train_idx.extend(scaffold_indices)
        elif len(train_idx) + len(val_idx) < val_cutoff:
            val_idx.extend(scaffold_indices)
        else:
            test_idx.extend(scaffold_indices)

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


def compute_all_pdvs(smiles_list):
    """Compute PDVs for all molecules. Returns (n_molecules, 200) array."""
    print("Computing PDVs for all molecules...")
    pdvs = []
    failed = 0
    for i, smi in enumerate(smiles_list):
        if i % 2000 == 0:
            print(f"  {i}/{len(smiles_list)}...")
        pdv = compute_pdv(smi)
        if pdv is None:
            failed += 1
            pdvs.append(np.full(200, np.nan))
        else:
            pdvs.append(pdv)
    print(f"  Done. {failed} failures.")
    return np.array(pdvs)


# ──────────────────────────────────────────────────────────────
# Descriptor analysis
# ──────────────────────────────────────────────────────────────
def print_sample_pdvs(pdvs, smiles_list, n=3):
    """Print detailed PDV values for a few sample molecules."""
    print("\n" + "=" * 80)
    print("SAMPLE PDVs")
    print("=" * 80)
    for i in range(min(n, len(pdvs))):
        print(f"\n--- Molecule {i}: {smiles_list[i]} ---")
        for j, name in enumerate(DEFAULT_DESCRIPTOR_LIST):
            val = pdvs[i, j]
            binary = 1 if val > 0 else 0
            print(f"  {name:35s}  raw={val:12.4f}  binary(>0)={binary}")


def analyze_descriptors(pdvs):
    """Analyze each descriptor's behavior under binarization."""
    print("\n" + "=" * 80)
    print("DESCRIPTOR ANALYSIS")
    print("=" * 80)

    always_one = []
    always_zero = []
    mostly_one = []
    mostly_zero = []
    informative = []

    print(f"\n{'Descriptor':35s} {'Min':>10s} {'Max':>10s} {'Mean':>10s} {'%>0':>6s} {'%==0':>6s} {'%<0':>6s} {'Category':>15s}")
    print("-" * 110)

    for j, name in enumerate(DEFAULT_DESCRIPTOR_LIST):
        col = pdvs[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) == 0:
            continue

        pct_pos = np.mean(valid > 0) * 100
        pct_zero = np.mean(valid == 0) * 100
        pct_neg = np.mean(valid < 0) * 100

        if pct_pos == 100:
            cat = "ALWAYS_1"
            always_one.append(name)
        elif pct_pos == 0:
            cat = "ALWAYS_0"
            always_zero.append(name)
        elif pct_pos > 90:
            cat = "MOSTLY_1"
            mostly_one.append(name)
        elif pct_pos < 10:
            cat = "MOSTLY_0"
            mostly_zero.append(name)
        else:
            cat = "INFORMATIVE"
            informative.append(name)

        print(f"{name:35s} {np.min(valid):10.3f} {np.max(valid):10.3f} {np.mean(valid):10.3f} {pct_pos:5.1f}% {pct_zero:5.1f}% {pct_neg:5.1f}% {cat:>15s}")

    print("\n" + "=" * 80)
    print("BINARIZATION SUMMARY")
    print("=" * 80)
    print(f"  ALWAYS_1  (dead — constant 1 after >0):       {len(always_one):3d} features")
    print(f"  ALWAYS_0  (dead — constant 0 after >0):       {len(always_zero):3d} features")
    print(f"  MOSTLY_1  (>90% positive, low info):           {len(mostly_one):3d} features")
    print(f"  MOSTLY_0  (<10% positive, sparse):             {len(mostly_zero):3d} features")
    print(f"  INFORMATIVE (10-90% positive, high info):      {len(informative):3d} features")
    print(f"  TOTAL:                                         {len(DEFAULT_DESCRIPTOR_LIST):3d} features")

    print(f"\n  ALWAYS_1: {', '.join(always_one[:20])}{'...' if len(always_one) > 20 else ''}")
    print(f"\n  ALWAYS_0: {', '.join(always_zero[:20])}{'...' if len(always_zero) > 20 else ''}")
    print(f"\n  INFORMATIVE: {', '.join(informative[:20])}{'...' if len(informative) > 20 else ''}")

    n_active = len(informative) + len(mostly_one) + len(mostly_zero)
    print(f"\n  >> After binarization, ~{n_active} features carry any discriminating information.")
    print(f"  >> {len(always_one)} features are constant=1, {len(always_zero)} are constant=0 (wasted).")

    return {
        'always_one': always_one,
        'always_zero': always_zero,
        'mostly_one': mostly_one,
        'mostly_zero': mostly_zero,
        'informative': informative,
    }


# ──────────────────────────────────────────────────────────────
# PDV transformations
# ──────────────────────────────────────────────────────────────
def transform_pdv(pdvs_train, pdvs_test, pdvs_val, method):
    """Apply a transformation to PDVs. Returns (X_train, X_test, X_val)."""
    if method == 'binary_gt0':
        return (pdvs_train > 0).astype(np.float32), \
               (pdvs_test > 0).astype(np.float32), \
               (pdvs_val > 0).astype(np.float32)

    elif method == 'continuous_raw':
        return pdvs_train.astype(np.float32), \
               pdvs_test.astype(np.float32), \
               pdvs_val.astype(np.float32)

    elif method == 'continuous_zscore':
        mean = np.nanmean(pdvs_train, axis=0)
        std = np.nanstd(pdvs_train, axis=0)
        std[std == 0] = 1.0
        return ((pdvs_train - mean) / std).astype(np.float32), \
               ((pdvs_test - mean) / std).astype(np.float32), \
               ((pdvs_val - mean) / std).astype(np.float32)

    elif method == 'continuous_minmax':
        vmin = np.nanmin(pdvs_train, axis=0)
        vmax = np.nanmax(pdvs_train, axis=0)
        rng = vmax - vmin
        rng[rng == 0] = 1.0
        def scale(x):
            return (2.0 * (x - vmin) / rng - 1.0).astype(np.float32)
        return scale(pdvs_train), scale(pdvs_test), scale(pdvs_val)

    elif method == 'binary_mean':
        mean = np.nanmean(pdvs_train, axis=0)
        return (pdvs_train > mean).astype(np.float32), \
               (pdvs_test > mean).astype(np.float32), \
               (pdvs_val > mean).astype(np.float32)

    elif method == 'binary_median':
        median = np.nanmedian(pdvs_train, axis=0)
        return (pdvs_train > median).astype(np.float32), \
               (pdvs_test > median).astype(np.float32), \
               (pdvs_val > median).astype(np.float32)

    elif method == 'binary_nonzero':
        return (pdvs_train != 0).astype(np.float32), \
               (pdvs_test != 0).astype(np.float32), \
               (pdvs_val != 0).astype(np.float32)

    elif method == 'binary_gt0_no_dead':
        binary_train = (pdvs_train > 0).astype(np.float32)
        binary_test = (pdvs_test > 0).astype(np.float32)
        binary_val = (pdvs_val > 0).astype(np.float32)
        col_std = np.std(binary_train, axis=0)
        keep = col_std > 0
        print(f"    [binary_gt0_no_dead] Keeping {np.sum(keep)}/{len(keep)} features (dropping {np.sum(~keep)} constant)")
        return binary_train[:, keep], binary_test[:, keep], binary_val[:, keep]

    else:
        raise ValueError(f"Unknown method: {method}")


# ──────────────────────────────────────────────────────────────
# Experiment runner
# ──────────────────────────────────────────────────────────────
def train_and_predict_rf(X_train, y_train, X_test, seed=42):
    """Train RF with default hyperparameters (matching main ANOVA pipeline)."""
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        max_features='sqrt',
        min_samples_leaf=1,
        min_samples_split=2,
        bootstrap=True,
        n_jobs=-1,
        random_state=seed,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test), rf


def compute_nds(sigma_r2_pairs):
    """Compute NDS (slope of R² vs sigma) via linear regression."""
    sigmas = np.array([s for s, _ in sigma_r2_pairs])
    r2s = np.array([r for _, r in sigma_r2_pairs])
    result = scipy_stats.linregress(sigmas, r2s)
    return result.slope, result.intercept, result.pvalue


def run_noise_experiment(X_train, y_train_clean, X_test, y_test, sigma_levels, iteration_seed):
    """
    For each sigma level, inject noise into y_train, train RF, predict on clean test.
    Returns dict of {sigma: r2} and the baseline RF.
    """
    sigma_r2 = {}
    baseline_rf = None

    for sigma in sigma_levels:
        if sigma == 0.0:
            y_train_noisy = y_train_clean
        else:
            injector = NoiseInjectorRegression(strategy='legacy', random_state=iteration_seed)
            y_train_noisy = injector.inject(y_train_clean, sigma)

        y_pred, rf = train_and_predict_rf(X_train, y_train_noisy, X_test, seed=iteration_seed)
        r2 = r2_score(y_test, y_pred)
        sigma_r2[sigma] = r2

        if sigma == 0.0:
            baseline_rf = rf

    return sigma_r2, baseline_rf


def print_feature_importance(rf, method, categories, top_n=15):
    """Print top features by RF importance."""
    importances = rf.feature_importances_
    if method == 'binary_gt0_no_dead':
        return  # can't map indices back
    indices = np.argsort(importances)[::-1][:top_n]
    print(f"\n  Top {top_n} features by RF importance ({method}):")
    for rank, idx in enumerate(indices):
        name = DEFAULT_DESCRIPTOR_LIST[idx]
        cat = "?"
        for cat_name, cat_list in categories.items():
            if name in cat_list:
                cat = cat_name
                break
        print(f"    {rank+1:2d}. {name:35s}  importance={importances[idx]:.4f}  binarization_cat={cat}")


# ──────────────────────────────────────────────────────────────
# Model comparison (deeper dive)
# ──────────────────────────────────────────────────────────────

def run_model_comparison(pdvs_train, pdvs_test, pdvs_val, y_train, y_test, y_val, args):
    """
    Compare RF, XGBoost, SVM, BNN Full, and VBLL on select PDV variants.

    Matches the main pipeline:
      - Y targets z-score normalized (fit on train), predictions inverse-transformed
      - Optuna-tuned hyperparameters from scripts/results/*.json
      - No X normalization (binary PDV is inherently {0,1} scaled;
        continuous gets z-score for SVM/NNs only)
    """
    import math
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    import torchbnn as bnn
    import gpytorch
    from botorch.fit import fit_gpytorch_model
    from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
    from sklearn.svm import SVR
    from sklearn.ensemble import RandomForestRegressor
    from xgboost import XGBRegressor

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # ── Y normalization (matches Rust pipeline) ──
    y_mean = float(y_train.mean())
    y_std = float(y_train.std())
    print(f"  Y normalization: mean={y_mean:.6f}, std={y_std:.6f}")

    def y_normalize(y):
        return (y - y_mean) / y_std

    def y_denormalize(y):
        return y * y_std + y_mean

    y_train_n = y_normalize(y_train)
    y_test_n = y_normalize(y_test)
    y_val_n = y_normalize(y_val)

    # ── Default hyperparameters (matching main ANOVA pipeline defaults) ──
    DEFAULT_PARAMS = {
        'rf': {
            'max_depth': None,
            'max_features': 'sqrt',
            'min_samples_leaf': 1,
            'min_samples_split': 2,
            'n_estimators': 100,
            'bootstrap': True,
        },
        'xgboost': {
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 1.0,
            'n_estimators': 100,
            'colsample_bytree': 1.0,
            'colsample_bylevel': 1.0,
            'min_child_weight': 1,
            'gamma': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
        },
        'svm': {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'rbf',
        },
        'dnn': {  # Used for BNN Full and VBLL
            'hidden_size1': 128,
            'hidden_size2': 64,
            'activation': 'relu',
        },
    }

    # ── Model definitions ──
    class DNNModel(nn.Module):
        def __init__(self, input_size, h1=64, h2=32, activation='tanh'):
            super().__init__()
            self.fc1 = nn.Linear(input_size, h1)
            self.fc2 = nn.Linear(h1, h2)
            self.fc3 = nn.Linear(h2, 1)
            self.dropout = nn.Dropout(0.2)
            self.act = nn.Tanh() if activation == 'tanh' else nn.ReLU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.dropout(x)
            x = self.act(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x)

    class VBLLLayer(nn.Module):
        """Variational Bayesian layer with learned observation noise."""
        def __init__(self, in_features, out_features, prior_sigma=1.0):
            super().__init__()
            self.prior_sigma = prior_sigma
            self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
            self.weight_log_sigma = nn.Parameter(torch.full((out_features, in_features), -3.0))
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_log_sigma = nn.Parameter(torch.full((out_features,), -3.0))
            self.log_noise_var = nn.Parameter(torch.tensor(-2.0))
            nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
            bound = 1.0 / math.sqrt(in_features)
            nn.init.uniform_(self.bias_mu, -bound, bound)

        @property
        def noise_var(self):
            return torch.exp(self.log_noise_var)

        def kl_divergence(self):
            prior_var = self.prior_sigma ** 2
            w_var = torch.exp(2.0 * self.weight_log_sigma)
            kl_w = 0.5 * torch.sum(
                w_var / prior_var + self.weight_mu**2 / prior_var - 1.0
                + math.log(prior_var) - 2.0 * self.weight_log_sigma)
            b_var = torch.exp(2.0 * self.bias_log_sigma)
            kl_b = 0.5 * torch.sum(
                b_var / prior_var + self.bias_mu**2 / prior_var - 1.0
                + math.log(prior_var) - 2.0 * self.bias_log_sigma)
            return kl_w + kl_b

        def forward(self, x):
            weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_mu)
            bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_mu)
            return F.linear(x, weight, bias)

    class VBLLLoss(nn.Module):
        """ELBO loss: Gaussian NLL (learned noise) + KL / n_data."""
        def __init__(self, model, n_data):
            super().__init__()
            self.vbll_layers = [m for m in model.modules() if isinstance(m, VBLLLayer)]
            self.output_layer = self.vbll_layers[-1]
            self.n_data = n_data

        def forward(self, pred, target):
            noise_var = self.output_layer.noise_var
            nll = 0.5 * torch.log(noise_var) + 0.5 * (pred - target)**2 / noise_var
            kl = sum(l.kl_divergence() for l in self.vbll_layers)
            return nll.mean() + kl / self.n_data

    # ── Training utilities ──
    def train_nn_model(model, X_train, y_train, X_val, y_val, criterion,
                       epochs=200, lr=0.001, batch_size=32, patience=20):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        X_tr = torch.FloatTensor(X_train).to(device)
        y_tr = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
        X_v = torch.FloatTensor(X_val).to(device)
        y_v = torch.FloatTensor(y_val).reshape(-1, 1).to(device)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        wait = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_v)
                val_loss = nn.MSELoss()(val_pred, y_v).item()

            if val_loss < best_val_loss - 0.0001:
                best_val_loss = val_loss
                wait = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model

    def mc_predict(model, X_test, n_samples=100):
        """MC sampling: n forward passes through stochastic model, return mean."""
        model.eval()
        X_t = torch.FloatTensor(X_test).to(device)
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(model(X_t).cpu().numpy().flatten())
        return np.mean(preds, axis=0)

    # ── Per-model runners (all train on normalized Y, return denormalized predictions) ──
    def run_rf(X_train, y_train_norm, X_val, y_val_norm, X_test, seed):
        p = DEFAULT_PARAMS['rf']
        X_tr = np.vstack((X_train, X_val))
        y_tr = np.hstack((y_train_norm, y_val_norm))
        model = RandomForestRegressor(
            n_estimators=p['n_estimators'], max_depth=p['max_depth'],
            max_features=p['max_features'], min_samples_leaf=p['min_samples_leaf'],
            min_samples_split=p['min_samples_split'], bootstrap=p['bootstrap'],
            n_jobs=-1, random_state=seed)
        model.fit(X_tr, y_tr)
        return y_denormalize(model.predict(X_test))

    def run_xgboost(X_train, y_train_norm, X_val, y_val_norm, X_test, seed):
        p = DEFAULT_PARAMS['xgboost']
        X_tr = np.vstack((X_train, X_val))
        y_tr = np.hstack((y_train_norm, y_val_norm))
        model = XGBRegressor(
            n_estimators=p['n_estimators'], max_depth=p['max_depth'],
            learning_rate=p['learning_rate'],
            subsample=p['subsample'], colsample_bytree=p['colsample_bytree'],
            colsample_bylevel=p['colsample_bylevel'], min_child_weight=p['min_child_weight'],
            gamma=p['gamma'], reg_alpha=p['reg_alpha'], reg_lambda=p['reg_lambda'],
            random_state=seed, n_jobs=-1, verbosity=0)
        model.fit(X_tr, y_tr)
        return y_denormalize(model.predict(X_test))

    def run_svm(X_train, y_train_norm, X_val, y_val_norm, X_test):
        p = DEFAULT_PARAMS['svm']
        X_tr = np.vstack((X_train, X_val))
        y_tr = np.hstack((y_train_norm, y_val_norm))
        model = SVR(kernel=p['kernel'], C=p['C'], gamma=p['gamma'])
        model.fit(X_tr, y_tr)
        return y_denormalize(model.predict(X_test))

    def run_bnn_full(X_train, y_train_norm, X_val, y_val_norm, X_test, seed):
        torch.manual_seed(seed)
        p = DEFAULT_PARAMS['dnn']
        model = DNNModel(X_train.shape[1], h1=p['hidden_size1'], h2=p['hidden_size2'],
                         activation=p['activation']).to(device)
        for name in ['fc1', 'fc2', 'fc3']:
            module = getattr(model, name)
            bl = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                 in_features=module.in_features,
                                 out_features=module.out_features,
                                 bias=module.bias is not None)
            with torch.no_grad():
                bl.weight_mu.copy_(module.weight.data)
                if module.bias is not None:
                    bl.bias_mu.copy_(module.bias.data)
            setattr(model, name, bl)
        model.to(device)
        model = train_nn_model(model, X_train, y_train_norm, X_val, y_val_norm, nn.MSELoss())
        return y_denormalize(mc_predict(model, X_test))

    def run_vbll(X_train, y_train_norm, X_val, y_val_norm, X_test, seed):
        torch.manual_seed(seed)
        p = DEFAULT_PARAMS['dnn']
        model = DNNModel(X_train.shape[1], h1=p['hidden_size1'], h2=p['hidden_size2'],
                         activation=p['activation']).to(device)
        for name in ['fc1', 'fc2', 'fc3']:
            module = getattr(model, name)
            vl = VBLLLayer(module.in_features, module.out_features)
            with torch.no_grad():
                vl.weight_mu.copy_(module.weight.data)
                if module.bias is not None:
                    vl.bias_mu.copy_(module.bias.data)
            setattr(model, name, vl)
        model.to(device)
        criterion = VBLLLoss(model, n_data=len(X_train))
        model = train_nn_model(model, X_train, y_train_norm, X_val, y_val_norm, criterion)
        return y_denormalize(mc_predict(model, X_test))

    # ── Gauche GP ──
    class GaucheGP(gpytorch.models.ExactGP):
        """Exact GP matching the main pipeline's Gauche model."""
        def __init__(self, train_x, train_y, likelihood, kernel_class):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(kernel_class())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def run_gauche(X_train, y_train_norm, X_val, y_val_norm, X_test, kernel_class):
        X_tr = np.vstack((X_train, X_val))
        y_tr = np.hstack((y_train_norm, y_val_norm))
        x_tensor = torch.from_numpy(X_tr).double()
        y_tensor = torch.from_numpy(y_tr).double()
        x_test_tensor = torch.from_numpy(X_test).double()

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = GaucheGP(x_tensor, y_tensor, likelihood, kernel_class)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_model(mll)

        model.eval()
        likelihood.eval()
        with torch.no_grad():
            preds = model(x_test_tensor)
            y_pred = preds.mean.numpy()
        return y_denormalize(y_pred)

    # ── Main comparison loop ──
    methods = ['continuous_raw', 'binary_mean']

    # Gauche kernel → method compatibility
    # Tanimoto (fingerprint kernel): only valid on binary features — collapses on continuous
    # RBF: valid on all PDV variants; run everywhere for comparison
    GAUCHE_SKIP = {
        ('gauche_tanimoto', 'continuous_raw'),
    }

    tanimoto_kernel = TanimotoKernel
    model_runners = {
        'rf': lambda Xtr, ytr, Xv, yv, Xte: run_rf(Xtr, ytr, Xv, yv, Xte, args.seed),
        'xgboost': lambda Xtr, ytr, Xv, yv, Xte: run_xgboost(Xtr, ytr, Xv, yv, Xte, args.seed),
        'svm': lambda Xtr, ytr, Xv, yv, Xte: run_svm(Xtr, ytr, Xv, yv, Xte),
        'bnn_full': lambda Xtr, ytr, Xv, yv, Xte: run_bnn_full(Xtr, ytr, Xv, yv, Xte, args.seed),
        'vbll': lambda Xtr, ytr, Xv, yv, Xte: run_vbll(Xtr, ytr, Xv, yv, Xte, args.seed),
        'gauche_tanimoto': lambda Xtr, ytr, Xv, yv, Xte: run_gauche(Xtr, ytr, Xv, yv, Xte, tanimoto_kernel),
        'gauche_rbf': lambda Xtr, ytr, Xv, yv, Xte: run_gauche(Xtr, ytr, Xv, yv, Xte, gpytorch.kernels.RBFKernel),
    }

    print("\n" + "=" * 100)
    print("MODEL COMPARISON v4: with Y normalization + default hyperparameters + Gauche GP kernels")
    print(f"  Models: {', '.join(model_runners.keys())}")
    print(f"  PDV variants: {', '.join(methods)}")
    print(f"  Sigma levels: {SIGMA_LEVELS}")
    print(f"  Y normalization: z-score (mean={y_mean:.4f}, std={y_std:.4f})")
    print(f"  Default params: RF (100 trees, sqrt), XGB (100 trees, depth=6, lr=0.1), SVM (RBF, C=1.0)")
    print(f"  BNN/VBLL architecture: DNN [{DEFAULT_PARAMS['dnn']['hidden_size1']}, "
          f"{DEFAULT_PARAMS['dnn']['hidden_size2']}], {DEFAULT_PARAMS['dnn']['activation']}")
    print(f"  BNN Full: torchbnn BayesLinear, prior N(0, 0.1), 100 MC samples")
    print(f"  VBLL: VBLLLayer all layers, ELBO loss, learned noise, 100 MC samples")
    print(f"  Gauche Tanimoto: fingerprint kernel (binary PDV only)")
    print(f"  Gauche RBF: standard RBF kernel (all PDV variants)")
    print(f"  Skipped combos: {sorted(GAUCHE_SKIP)}")
    print("=" * 100)

    all_results = []

    for method in methods:
        X_train_raw, X_test_raw, X_val_raw = transform_pdv(
            pdvs_train, pdvs_test, pdvs_val, method)

        # For continuous features, z-score normalize X (SVM and NNs need it)
        # For binary features, no X normalization needed (matches pipeline)
        if method == 'continuous_raw':
            x_mean = np.nanmean(X_train_raw, axis=0)
            x_std = np.nanstd(X_train_raw, axis=0)
            x_std[x_std == 0] = 1.0
            X_train_x = ((X_train_raw - x_mean) / x_std).astype(np.float32)
            X_test_x = ((X_test_raw - x_mean) / x_std).astype(np.float32)
            X_val_x = ((X_val_raw - x_mean) / x_std).astype(np.float32)
        else:
            # Binary features: use as-is (matches pipeline — no X normalization)
            X_train_x = X_train_raw.astype(np.float32)
            X_test_x = X_test_raw.astype(np.float32)
            X_val_x = X_val_raw.astype(np.float32)

        X_train_x = np.nan_to_num(X_train_x, 0.0)
        X_test_x = np.nan_to_num(X_test_x, 0.0)
        X_val_x = np.nan_to_num(X_val_x, 0.0)

        n_feat = X_train_x.shape[1]
        print(f"\n{'─' * 60}")
        print(f"PDV variant: {method} ({n_feat} features)"
              f"{' [X z-score normalized]' if method == 'continuous_raw' else ' [no X normalization]'}")
        print(f"{'─' * 60}")

        for model_name, runner in model_runners.items():
            if (model_name, method) in GAUCHE_SKIP:
                print(f"  Skipping {model_name} × {method} (incompatible kernel)")
                continue
            print(f"  Training {model_name}...", end="", flush=True)
            sigma_r2 = {}

            for sigma in SIGMA_LEVELS:
                if sigma == 0.0:
                    y_noisy_n = y_train_n
                else:
                    # Inject noise into ORIGINAL scale, then normalize
                    injector = NoiseInjectorRegression(strategy='legacy', random_state=args.seed)
                    y_noisy = injector.inject(y_train, sigma)
                    y_noisy_n = y_normalize(y_noisy)

                # Models train on normalized Y, return denormalized predictions
                y_pred = runner(X_train_x, y_noisy_n, X_val_x, y_val_n, X_test_x)
                r2 = r2_score(y_test, y_pred)
                sigma_r2[sigma] = r2

            nds, _, _ = compute_nds([(s, sigma_r2[s]) for s in SIGMA_LEVELS])
            result = {
                'model': model_name, 'method': method, 'n_features': n_feat,
                'baseline_r2': sigma_r2[0.0], 'nds': nds,
            }
            result.update({f'r2_sigma_{s}': sigma_r2[s] for s in SIGMA_LEVELS})
            all_results.append(result)

            sigma_str = "  ".join([f"s={s}: {sigma_r2[s]:.4f}" for s in SIGMA_LEVELS])
            print(f" done  |  {sigma_str}  | NDS={nds:+.2f}")

    # ── Save and print summary ──
    df = pd.DataFrame(all_results)
    out_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'pdv_investigation')
    os.makedirs(out_dir, exist_ok=True)
    out_path = osp.join(out_dir, 'model_comparison.csv')
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 100)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 100)
    print(f"{'Model':12s} {'PDV Variant':25s} {'#Feat':>6s} {'Baseline R²':>12s} {'NDS':>8s}")
    print("-" * 70)

    for method in methods:
        sub = df[df['method'] == method].sort_values('baseline_r2', ascending=False)
        for _, row in sub.iterrows():
            print(f"{row['model']:12s} {row['method']:25s} {int(row['n_features']):6d} "
                  f"{row['baseline_r2']:12.4f} {row['nds']:+8.2f}")
        print()

    # Pipeline reference
    print("  Pipeline reference (tuned, 10 iterations, binary PDV):")
    print(f"  {'gauche_gp':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.8655':>12s} {'-0.37':>8s}")
    print(f"  {'lgb':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.8619':>12s} {'-0.38':>8s}")
    print(f"  {'xgboost':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.8615':>12s} {'-0.38':>8s}")
    print(f"  {'rf':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.8418':>12s} {'-0.39':>8s}")
    print(f"  {'svm':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.8298':>12s} {'-0.36':>8s}")
    print(f"  {'dnn_bnn_full':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.8143':>12s} {'-0.37':>8s}")
    print(f"  {'dnn_vbll':12s} {'binary_gt0 (pipeline)':25s} {'200':>6s} {'0.7929':>12s} {'-0.38':>8s}")

    return df


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='PDV binarization investigation')
    parser.add_argument('--n-samples', type=int, default=10000)
    parser.add_argument('--n-iterations', type=int, default=5,
                        help='Number of RF iterations (different seeds)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-analysis', action='store_true',
                        help='Skip descriptor analysis, go straight to experiments')
    parser.add_argument('--model-comparison', action='store_true',
                        help='Run deeper dive comparing XGBoost/SVM/BNN/VBLL on select PDV variants')
    parser.add_argument('--skip-rf', action='store_true',
                        help='Skip the main RF experiments (use with --model-comparison)')
    args = parser.parse_args()

    # ── Load data ──
    smiles_list, targets = load_qm9_data('homo_lumo_gap', args.n_samples, args.seed)

    # ── Scaffold split ──
    train_idx, val_idx, test_idx = scaffold_split(smiles_list, targets, args.seed)

    # ── Compute PDVs ──
    pdvs = compute_all_pdvs(smiles_list)

    # ── Print samples ──
    print_sample_pdvs(pdvs, smiles_list, n=3)

    # ── Analyze descriptors ──
    if not args.skip_analysis:
        categories = analyze_descriptors(pdvs)
    else:
        categories = {}

    # ── Split into train/val/test ──
    pdvs_train = pdvs[train_idx]
    pdvs_test = pdvs[test_idx]
    pdvs_val = pdvs[val_idx]
    y_train = targets[train_idx]
    y_test = targets[test_idx]
    y_val = targets[val_idx]

    # Drop NaN PDVs
    valid_train = ~np.any(np.isnan(pdvs_train), axis=1)
    valid_test = ~np.any(np.isnan(pdvs_test), axis=1)
    valid_val = ~np.any(np.isnan(pdvs_val), axis=1)
    pdvs_train, y_train = pdvs_train[valid_train], y_train[valid_train]
    pdvs_test, y_test = pdvs_test[valid_test], y_test[valid_test]
    pdvs_val, y_val = pdvs_val[valid_val], y_val[valid_val]

    print(f"\nAfter NaN removal: train={len(y_train)}, test={len(y_test)}, val={len(y_val)}")
    print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    print(f"Target std: {y_train.std():.3f}")

    # ── RF experiments ──
    if not args.skip_rf:
        methods = [
            'binary_gt0',
            'continuous_raw',
            'continuous_zscore',
            'continuous_minmax',
            'binary_mean',
            'binary_median',
            'binary_nonzero',
            'binary_gt0_no_dead',
        ]

        print("\n" + "=" * 100)
        print("EXPERIMENTS: RF with different PDV transformations")
        print(f"  n_iterations={args.n_iterations}, base_seed={args.seed}")
        print(f"  sigma_levels={SIGMA_LEVELS}")
        print(f"  noise_strategy=legacy (simple Gaussian)")
        print("=" * 100)

        all_results = []
        all_per_sigma = []  # for detailed CSV output

        for method in methods:
            print(f"\n{'='*60}")
            print(f"Method: {method}")
            print(f"{'='*60}")

            iter_baseline_r2 = []
            iter_nds = []
            iter_nds_pval = []
            last_rf = None

            for it in range(args.n_iterations):
                seed_i = args.seed + it
                X_train, X_test, X_val = transform_pdv(pdvs_train, pdvs_test, pdvs_val, method)

                # Run across all sigma levels
                sigma_r2, rf = run_noise_experiment(
                    X_train, y_train, X_test, y_test, SIGMA_LEVELS, seed_i
                )
                last_rf = rf

                # NDS = slope of R² vs sigma
                sigma_r2_pairs = [(s, sigma_r2[s]) for s in SIGMA_LEVELS]
                nds, intercept, pval = compute_nds(sigma_r2_pairs)

                baseline = sigma_r2[0.0]
                iter_baseline_r2.append(baseline)
                iter_nds.append(nds)
                iter_nds_pval.append(pval)

                # Print per-sigma detail
                sigma_str = "  ".join([f"s={s}: {sigma_r2[s]:.4f}" for s in SIGMA_LEVELS])
                print(f"  iter {it} (seed {seed_i}): {sigma_str}  | NDS={nds:+.4f}")

                # Record per-sigma for CSV
                for s in SIGMA_LEVELS:
                    all_per_sigma.append({
                        'method': method,
                        'iteration': it,
                        'seed': seed_i,
                        'sigma': s,
                        'r2': sigma_r2[s],
                    })

            # Feature importance (from baseline, last iteration)
            if categories and last_rf is not None:
                print_feature_importance(last_rf, method, categories)

            result = {
                'method': method,
                'n_features': X_train.shape[1],
                'baseline_r2_mean': np.mean(iter_baseline_r2),
                'baseline_r2_std': np.std(iter_baseline_r2),
                'nds_mean': np.mean(iter_nds),
                'nds_std': np.std(iter_nds),
                'nds_pval_mean': np.mean(iter_nds_pval),
            }
            all_results.append(result)

        # ── Summary table ──
        print("\n" + "=" * 100)
        print("RESULTS SUMMARY (sorted by baseline R²)")
        print("=" * 100)
        print(f"{'Method':25s} {'#Feat':>6s} {'Baseline R²':>16s} {'NDS (slope)':>16s} {'NDS p-val':>10s}")
        print("-" * 80)
        for r in sorted(all_results, key=lambda x: -x['baseline_r2_mean']):
            print(f"{r['method']:25s} {r['n_features']:6d} "
                  f"{r['baseline_r2_mean']:.4f} +/- {r['baseline_r2_std']:.4f} "
                  f"{r['nds_mean']:+.4f} +/- {r['nds_std']:.4f} "
                  f"{r['nds_pval_mean']:.4f}")

        # ── Save results ──
        out_dir = osp.join(osp.dirname(osp.realpath(__file__)), 'pdv_investigation')
        os.makedirs(out_dir, exist_ok=True)

        summary_path = osp.join(out_dir, 'summary.csv')
        pd.DataFrame(all_results).to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")

        detail_path = osp.join(out_dir, 'per_sigma_detail.csv')
        pd.DataFrame(all_per_sigma).to_csv(detail_path, index=False)
        print(f"Per-sigma detail saved to {detail_path}")

        # ── Key comparisons ──
        def get_result(name):
            return [r for r in all_results if r['method'] == name][0]

        binary = get_result('binary_gt0')
        continuous = get_result('continuous_raw')
        zscore = get_result('continuous_zscore')
        minmax = get_result('continuous_minmax')
        bmean = get_result('binary_mean')
        bmedian = get_result('binary_median')

        print("\n" + "=" * 100)
        print("KEY FINDINGS")
        print("=" * 100)

        print(f"\n  1. BASELINE PERFORMANCE (R² at sigma=0)")
        print(f"     Binary (>0):          {binary['baseline_r2_mean']:.4f}")
        print(f"     Continuous raw:       {continuous['baseline_r2_mean']:.4f}")
        print(f"     Continuous z-score:   {zscore['baseline_r2_mean']:.4f}")
        print(f"     Continuous [-1,1]:    {minmax['baseline_r2_mean']:.4f}")
        diff = continuous['baseline_r2_mean'] - binary['baseline_r2_mean']
        print(f"     >> {'Continuous' if diff > 0 else 'Binary'} is better by {abs(diff):.4f}")

        print(f"\n  2. NOISE ROBUSTNESS (NDS, more negative = worse)")
        print(f"     Binary (>0):          {binary['nds_mean']:+.4f}")
        print(f"     Continuous raw:       {continuous['nds_mean']:+.4f}")
        print(f"     Continuous z-score:   {zscore['nds_mean']:+.4f}")
        print(f"     Continuous [-1,1]:    {minmax['nds_mean']:+.4f}")
        nds_diff = binary['nds_mean'] - continuous['nds_mean']
        print(f"     >> {'Binary' if nds_diff > 0 else 'Continuous'} is more robust (NDS diff: {abs(nds_diff):.4f})")

        print(f"\n  3. ALTERNATIVE THRESHOLDS")
        print(f"     Binary (>0):          R²={binary['baseline_r2_mean']:.4f}  NDS={binary['nds_mean']:+.4f}")
        print(f"     Binary (>mean):       R²={bmean['baseline_r2_mean']:.4f}  NDS={bmean['nds_mean']:+.4f}")
        print(f"     Binary (>median):     R²={bmedian['baseline_r2_mean']:.4f}  NDS={bmedian['nds_mean']:+.4f}")

    # ── Model comparison (deeper dive) ──
    if args.model_comparison:
        run_model_comparison(pdvs_train, pdvs_test, pdvs_val,
                             y_train, y_test, y_val, args)


if __name__ == '__main__':
    main()
