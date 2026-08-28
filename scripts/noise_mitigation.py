import os
import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from collections import deque
import torch
import warnings

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import DBSCAN
from sklearn.model_selection import KFold

import scipy.spatial.distance as distance

# save_uncertainty_values lives in scripts/utils.py beside this file and was never
# imported, so the one call to it (in train_baseline_model) would have raised
# NameError before it could raise TypeError. Resolve utils from this file's own
# location so the import works whatever the working directory is.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import save_uncertainty_values

from torch_geometric.datasets import QM9
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

# Add Gauche imports
try:
    import gauche
    from gauche.kernels.fingerprint_kernels import *
    from gauche.kernels.graph_kernels import *
    from gauche import SIGP, NonTensorialInputs
    from gauche.dataloader import MolPropLoader
    from gauche.dataloader.data_utils import transform_data
    from gauche.kernels.graph_kernels import WeisfeilerLehmanKernel, VertexHistogramKernel
    import gpytorch
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.fit import fit_gpytorch_model
    GAUCHE_AVAILABLE = True
except ImportError:
    GAUCHE_AVAILABLE = False
    print("Warning: Gauche not available. Install with: pip install gauche-ml")


warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

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

def load_qm9(target):
    """Load QM9 dataset with proper filtering"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    
    # Ensure data directory exists
    os.makedirs(data_dir, exist_ok=True)
    
    valid_indices_path = os.path.join(data_dir, 'valid_qm9_indices.pth')
    qm9_data_path = os.path.join(data_dir, 'QM9')
    
    print(f"Loading QM9 from: {qm9_data_path}")
    qm9 = QM9(root=qm9_data_path)
    
    # Filter out molecules that cannot be processed by RDKit
    if os.path.exists(valid_indices_path):
        print(f"Loading valid indices from: {valid_indices_path}")
        valid_indices_tensor = torch.load(valid_indices_path)
        qm9 = qm9.index_select(valid_indices_tensor)
        print(f"Filtered to {len(qm9)} valid molecules")
    else:
        print("Warning: valid_qm9_indices.pth not found, using all molecules")
    
    # Isolate target property
    y_target = pd.DataFrame(qm9.data.y.numpy())
    property_index = properties[target]
    qm9.data.y = torch.Tensor(y_target[property_index])
    
    print(f"Target property '{target}' loaded with {len(qm9)} samples")
    return qm9

def split_qm9_simple(qm9, sample_size, random_seed):
    """Simple random split - FIXED to match working pipeline"""
    # First select the sample size, then shuffle
    qm9_subset = qm9[:sample_size]
    
    # Shuffle indices with random seed
    indices = torch.randperm(sample_size, generator=torch.Generator().manual_seed(random_seed))
    
    train_size = int(sample_size * 0.8)
    test_size = int(sample_size * 0.1)
    val_size = sample_size - train_size - test_size
    
    # Use shuffled indices for splitting
    train_idx = indices[:train_size].tolist()
    val_idx = indices[train_size:train_size + val_size].tolist()
    test_idx = indices[train_size + val_size:].tolist()
    
    return train_idx, test_idx, val_idx

def create_sns_featurizer(train_molecules, vec_dimension=1024):
    """Create SNS featurizer trained only on training molecules"""
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True),
        useBondTypes=True,
        includeChirality=False
    )
    
    # Build vocabulary from training molecules only
    sub_ids_to_counts = {}
    for mol in train_molecules:
        if mol is not None:
            fingerprint = morgan_generator.GetSparseCountFingerprint(mol)
            for sub_id, count in fingerprint.GetNonzeroElements().items():
                sub_ids_to_counts[sub_id] = sub_ids_to_counts.get(sub_id, 0) + 1
    
    # Sort by frequency
    sorted_sub_ids = sorted(sub_ids_to_counts.keys(), 
                           key=lambda x: sub_ids_to_counts[x], reverse=True)
    
    # Take top vec_dimension substructures
    vocab_sub_ids = sorted_sub_ids[:vec_dimension]
    sub_id_to_index = {sub_id: i for i, sub_id in enumerate(vocab_sub_ids)}
    
    def featurize(mol):
        if mol is None:
            return np.zeros(vec_dimension, dtype=np.float32)
        
        try:
            fingerprint = morgan_generator.GetSparseCountFingerprint(mol)
            feature_vector = np.zeros(vec_dimension, dtype=np.float32)
            
            for sub_id, count in fingerprint.GetNonzeroElements().items():
                if sub_id in sub_id_to_index:
                    feature_vector[sub_id_to_index[sub_id]] = count
            
            return feature_vector
        except:
            return np.zeros(vec_dimension, dtype=np.float32)
    
    return featurize

def extract_pdv_features(smiles):
    """Extract PDV features from SMILES"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(len(DEFAULT_DESCRIPTOR_LIST), dtype=np.float32)
    
    try:
        calculator = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
        descriptors = calculator.CalcDescriptors(mol)
        descriptors = np.array(descriptors, dtype=np.float32)
        # Handle NaN/Inf values
        descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
        return descriptors
    except:
        return np.zeros(len(DEFAULT_DESCRIPTOR_LIST), dtype=np.float32)

def extract_molecular_features(qm9_data, train_indices, representation='sns'):
    """Extract molecular features with proper train/test isolation"""
    print(f"Extracting {representation} features...")
    
    if representation == 'sns':
        # Create featurizer using only training molecules
        train_molecules = []
        for idx in train_indices:
            mol = Chem.MolFromSmiles(qm9_data[idx].smiles)
            if mol is not None:
                train_molecules.append(mol)
        
        featurizer = create_sns_featurizer(train_molecules)
        
        # Apply to all molecules
        features = []
        for data in qm9_data:
            mol = Chem.MolFromSmiles(data.smiles)
            features.append(featurizer(mol))
        
        return np.array(features)
    
    elif representation == 'pdv':
        features = []
        for data in qm9_data:
            features.append(extract_pdv_features(data.smiles))
        
        return np.array(features)
    
    else:
        raise ValueError(f"Unsupported representation: {representation}")

def add_artificial_noise(y, noise_level, noise_type='gaussian', random_seed=42):
    """Add artificial noise to target values"""
    np.random.seed(random_seed)
    
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * np.std(y), size=y.shape)
    elif noise_type == 'uniform':
        noise_std = noise_level * np.std(y)
        noise_range = noise_std * np.sqrt(3)
        noise = np.random.uniform(-noise_range, noise_range, size=y.shape)
    elif noise_type == 'outlier':
        noise = np.zeros_like(y)
        n_outliers = max(1, int(noise_level * len(y)))
        outlier_indices = np.random.choice(len(y), n_outliers, replace=False)
        noise[outlier_indices] = np.random.normal(0, np.std(y) * 3, size=n_outliers)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return y + noise, noise

def detect_noise_distance_based(X, y, k_neighbors=5, threshold=2.0):
    """Detect noise based on distance to neighbors"""
    n_samples = X.shape[0]
    
    if n_samples <= k_neighbors + 1:
        return np.zeros(n_samples, dtype=bool)
    
    try:
        # Use only feature scaling for distance calculation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calculate pairwise distances
        distances = distance.squareform(distance.pdist(X_scaled, 'euclidean'))
        
        noise_scores = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Find k nearest neighbors (excluding self)
            neighbor_distances = distances[i]
            neighbor_indices = np.argsort(neighbor_distances)[1:k_neighbors+1]
            
            # Calculate label deviation from neighbors
            neighbor_labels = y[neighbor_indices]
            current_label = y[i]
            
            # Score based on label deviation weighted by distance
            weights = 1.0 / (neighbor_distances[neighbor_indices] + 1e-8)
            weights = weights / np.sum(weights)
            
            label_deviation = np.abs(neighbor_labels - current_label)
            noise_scores[i] = np.sum(weights * label_deviation)
        
        # Threshold based on distribution of scores
        if np.std(noise_scores) > 0:
            threshold_value = np.mean(noise_scores) + threshold * np.std(noise_scores)
            return noise_scores > threshold_value
        else:
            return np.zeros(n_samples, dtype=bool)
            
    except Exception as e:
        print(f"Error in distance-based detection: {e}")
        return np.zeros(n_samples, dtype=bool)

def detect_noise_uncertainty_based(X, y, n_estimators=50, uncertainty_threshold=0.9):
    """Detect noise based on model uncertainty using ensemble"""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        n_samples = len(X)
        predictions = np.zeros((n_samples, n_estimators))
        
        # Create ensemble of models with bootstrap sampling
        for i in range(n_estimators):
            # Bootstrap sample
            bootstrap_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_scaled[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=i,
                n_jobs=1
            )
            model.fit(X_boot, y_boot)
            predictions[:, i] = model.predict(X_scaled)
        
        # Calculate uncertainty as standard deviation of predictions
        uncertainties = np.std(predictions, axis=1)
        
        # Threshold based on percentile
        threshold = np.percentile(uncertainties, uncertainty_threshold * 100)
        return uncertainties > threshold
        
    except Exception as e:
        print(f"Error in uncertainty-based detection: {e}")
        return np.zeros(len(X), dtype=bool)

def detect_noise_clustering_based(X, y, eps=0.5, min_samples=5):
    """Detect noise using DBSCAN clustering"""
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Normalize y values
        y_normalized = (y - np.mean(y)) / (np.std(y) + 1e-8)
        
        # Combine features with normalized targets
        combined = np.column_stack([X_scaled, y_normalized.reshape(-1, 1)])
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(combined)
        
        # Points labeled as -1 are noise
        return clusters == -1
        
    except Exception as e:
        print(f"Error in clustering-based detection: {e}")
        return np.zeros(len(X), dtype=bool)

def clean_data_removal(X, y, noise_mask):
    """Remove detected noisy samples"""
    clean_mask = ~noise_mask
    if np.sum(clean_mask) < 10:  # Keep at least 10 samples
        return X, y
    return X[clean_mask], y[clean_mask]

def clean_data_smoothing(X, y, noise_mask, k_neighbors=5):
    """Replace noisy labels with weighted average of neighbors"""
    if np.sum(noise_mask) == 0:
        return X, y
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        y_cleaned = y.copy()
        distances = distance.squareform(distance.pdist(X_scaled, 'euclidean'))
        
        for i in np.where(noise_mask)[0]:
            # Find k nearest clean neighbors
            clean_indices = np.where(~noise_mask)[0]
            if len(clean_indices) == 0:
                continue
                
            neighbor_distances = distances[i][clean_indices]
            nearest_k_indices = np.argsort(neighbor_distances)[:k_neighbors]
            neighbor_indices = clean_indices[nearest_k_indices]
            
            # Weighted average based on distance
            weights = 1.0 / (distances[i][neighbor_indices] + 1e-8)
            weights = weights / np.sum(weights)
            
            y_cleaned[i] = np.sum(weights * y[neighbor_indices])
        
        return X, y_cleaned
        
    except Exception as e:
        print(f"Error in smoothing: {e}")
        return X, y

def clean_data_replacement(X, y, noise_mask, model_type='random_forest'):
    """Replace noisy labels with model predictions"""
    clean_mask = ~noise_mask
    if np.sum(clean_mask) < 10:
        return X, y
    
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_clean = X_scaled[clean_mask]
        y_clean = y[clean_mask]
        
        # Train model on clean data
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X_clean, y_clean)
        
        # Replace noisy labels with predictions
        y_cleaned = y.copy()
        noisy_indices = np.where(noise_mask)[0]
        if len(noisy_indices) > 0:
            y_cleaned[noisy_indices] = model.predict(X_scaled[noisy_indices])
        
        return X, y_cleaned
        
    except Exception as e:
        print(f"Error in replacement: {e}")
        return X, y

def create_performance_plots(results_df, output_dir):
    """Create comprehensive performance plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Performance vs Noise Level
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # R² vs Noise Level
    ax1 = axes[0, 0]
    for detector in results_df['detector'].unique():
        if detector in ['clean_data', 'no_detection']:
            data = results_df[results_df['detector'] == detector]
            ax1.plot(data['noise_level'], data['r2'], 'o-', linewidth=2, markersize=8, 
                    label=detector.replace('_', ' ').title())
    
    # Best cleaning methods
    for noise_level in results_df['noise_level'].unique():
        level_data = results_df[results_df['noise_level'] == noise_level]
        cleaned_data = level_data[~level_data['detector'].isin(['clean_data', 'no_detection'])]
        if not cleaned_data.empty:
            best_idx = cleaned_data['r2'].idxmax()
            best = cleaned_data.loc[best_idx]
            ax1.scatter(best['noise_level'], best['r2'], s=100, marker='*', 
                       c='red', label='Best Cleaning' if noise_level == results_df['noise_level'].iloc[0] else "")
    
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Performance vs Noise Level')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # RMSE vs Noise Level
    ax2 = axes[0, 1]
    for detector in ['clean_data', 'no_detection']:
        data = results_df[results_df['detector'] == detector]
        ax2.plot(data['noise_level'], data['rmse'], 'o-', linewidth=2, markersize=8,
                label=detector.replace('_', ' ').title())
    ax2.set_xlabel('Noise Level')
    ax2.set_ylabel('RMSE')
    ax2.set_title('RMSE vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Detection Effectiveness
    ax3 = axes[1, 0]
    detectors = results_df[~results_df['detector'].isin(['clean_data', 'no_detection'])]
    if not detectors.empty:
        detector_performance = detectors.groupby('detector')['r2'].mean().sort_values(ascending=False)
        detector_performance.plot(kind='bar', ax=ax3, color='skyblue')
        ax3.set_title('Average R² by Detector')
        ax3.set_ylabel('Average R² Score')
        ax3.tick_params(axis='x', rotation=45)
    
    # Cleaning Method Effectiveness
    ax4 = axes[1, 1]
    if not detectors.empty:
        cleaner_performance = detectors.groupby('cleaner')['r2'].mean().sort_values(ascending=False)
        cleaner_performance.plot(kind='bar', ax=ax4, color='lightcoral')
        ax4.set_title('Average R² by Cleaning Method')
        ax4.set_ylabel('Average R² Score')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_overview.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Detailed Heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table for heatmap
    cleaned_results = results_df[~results_df['detector'].isin(['clean_data', 'no_detection'])]
    if not cleaned_results.empty:
        pivot_data = cleaned_results.pivot_table(
            values='r2', 
            index='detector', 
            columns='cleaner', 
            aggfunc='mean'
        )
        
        sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   center=pivot_data.mean().mean(), ax=ax)
        ax.set_title('R² Performance Heatmap: Detector vs Cleaner')
        ax.set_xlabel('Cleaning Method')
        ax.set_ylabel('Detection Method')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Recovery Analysis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for noise_level in sorted(results_df['noise_level'].unique()):
        level_data = results_df[results_df['noise_level'] == noise_level]
        clean_r2 = level_data[level_data['detector'] == 'clean_data']['r2'].iloc[0]
        noisy_r2 = level_data[level_data['detector'] == 'no_detection']['r2'].iloc[0]
        
        cleaned_data = level_data[~level_data['detector'].isin(['clean_data', 'no_detection'])]
        if not cleaned_data.empty:
            best_cleaned_r2 = cleaned_data['r2'].max()
            if clean_r2 != noisy_r2:
                recovery = (best_cleaned_r2 - noisy_r2) / (clean_r2 - noisy_r2) * 100
                ax.scatter(noise_level, recovery, s=100, alpha=0.7)
    
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.7, label='Full Recovery')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='No Recovery')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Recovery Percentage (%)')
    ax.set_title('Performance Recovery by Noise Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recovery_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_uncertainty_plots(uncertainty_file, output_dir):
    """Create uncertainty analysis plots"""
    if not os.path.exists(uncertainty_file):
        print("No uncertainty file found, skipping uncertainty plots")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        df = pd.read_csv(uncertainty_file)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Uncertainty vs Error
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Uncertainty vs Absolute Error
        ax1 = axes[0, 0]
        abs_error = np.abs(df['y_true'] - df['y_pred_mean'])
        ax1.scatter(df['y_pred_std'], abs_error, alpha=0.6, s=20)
        ax1.set_xlabel('Prediction Uncertainty (Std)')
        ax1.set_ylabel('Absolute Error')
        ax1.set_title('Uncertainty vs Absolute Error')
        
        # Add correlation
        corr = np.corrcoef(df['y_pred_std'], abs_error)[0, 1]
        ax1.text(0.05, 0.95, f'Correlation: {corr:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        ax1.grid(True, alpha=0.3)
        
        # Uncertainty Distribution by Noise Level
        ax2 = axes[0, 1]
        for sigma in sorted(df['Sigma'].unique()):
            sigma_data = df[df['Sigma'] == sigma]
            ax2.hist(sigma_data['y_pred_std'], alpha=0.6, bins=30, 
                    label=f'Noise {sigma}', density=True)
        ax2.set_xlabel('Prediction Uncertainty (Std)')
        ax2.set_ylabel('Density')
        ax2.set_title('Uncertainty Distribution by Noise Level')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Calibration Plot
        ax3 = axes[1, 0]
        # Bin by uncertainty and compute actual vs predicted error
        n_bins = 10
        uncertainty_bins = np.percentile(df['y_pred_std'], np.linspace(0, 100, n_bins + 1))
        bin_centers = []
        actual_errors = []
        predicted_errors = []
        
        for i in range(n_bins):
            mask = (df['y_pred_std'] >= uncertainty_bins[i]) & (df['y_pred_std'] < uncertainty_bins[i + 1])
            if mask.sum() > 0:
                bin_data = df[mask]
                bin_centers.append(bin_data['y_pred_std'].mean())
                actual_errors.append(np.sqrt(np.mean((bin_data['y_true'] - bin_data['y_pred_mean'])**2)))
                predicted_errors.append(bin_data['y_pred_std'].mean())
        
        ax3.scatter(predicted_errors, actual_errors, s=50, alpha=0.7)
        min_val = min(min(predicted_errors), min(actual_errors))
        max_val = max(max(predicted_errors), max(actual_errors))
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7, label='Perfect Calibration')
        ax3.set_xlabel('Predicted Error (Uncertainty)')
        ax3.set_ylabel('Actual RMSE')
        ax3.set_title('Calibration Plot')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Uncertainty vs Noise Level
        ax4 = axes[1, 1]
        uncertainty_by_noise = df.groupby('Sigma')['y_pred_std'].agg(['mean', 'std']).reset_index()
        ax4.errorbar(uncertainty_by_noise['Sigma'], uncertainty_by_noise['mean'], 
                    yerr=uncertainty_by_noise['std'], marker='o', capsize=5)
        ax4.set_xlabel('Noise Level')
        ax4.set_ylabel('Average Uncertainty')
        ax4.set_title('Average Uncertainty by Noise Level')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Confidence Analysis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # High vs Low uncertainty samples
        high_uncertainty_threshold = df['y_pred_std'].quantile(0.8)
        high_unc_mask = df['y_pred_std'] > high_uncertainty_threshold
        low_unc_mask = df['y_pred_std'] <= df['y_pred_std'].quantile(0.2)
        
        high_unc_error = np.abs(df[high_unc_mask]['y_true'] - df[high_unc_mask]['y_pred_mean'])
        low_unc_error = np.abs(df[low_unc_mask]['y_true'] - df[low_unc_mask]['y_pred_mean'])
        
        ax.hist(high_unc_error, alpha=0.6, bins=30, label='High Uncertainty', density=True, color='red')
        ax.hist(low_unc_error, alpha=0.6, bins=30, label='Low Uncertainty', density=True, color='blue')
        
        ax.set_xlabel('Absolute Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution: High vs Low Uncertainty Predictions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        high_mean_error = np.mean(high_unc_error)
        low_mean_error = np.mean(low_unc_error)
        ax.axvline(high_mean_error, color='red', linestyle='--', alpha=0.7)
        ax.axvline(low_mean_error, color='blue', linestyle='--', alpha=0.7)
        ax.text(0.6, 0.8, f'High Unc Mean Error: {high_mean_error:.3f}\nLow Unc Mean Error: {low_mean_error:.3f}', 
                transform=ax.transAxes, bbox=dict(boxstyle="round", facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_confidence.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Uncertainty plots saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error creating uncertainty plots: {e}")

class Gauche(gpytorch.models.ExactGP):
    """Gauche GP model for molecular property prediction"""
    def __init__(self, train_x, train_y, likelihood, kernel_class):
        super(Gauche, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel_class())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_baseline_model(X_train, y_train, X_test, y_test, model_type='random_forest', save_uncertainty=False, **kwargs):
    """Train and evaluate baseline model"""
    if len(X_train) == 0 or len(y_train) == 0:
        return {'r2': -999, 'rmse': 999, 'mae': 999}
    
    try:
        if model_type == 'gauche':
            if not GAUCHE_AVAILABLE:
                print("Gauche not available, falling back to Random Forest")
                model_type = 'random_forest'
            else:
                # Gauche expects binary features for fingerprint kernels
                X_train_binary = (X_train > 0).astype(np.float64)
                X_test_binary = (X_test > 0).astype(np.float64)
                
                X_train_tensor = torch.from_numpy(X_train_binary).double()
                X_test_tensor = torch.from_numpy(X_test_binary).double()
                y_train_tensor = torch.from_numpy(y_train).double()
                
                # Use default Gauche settings
                likelihood = gpytorch.likelihoods.GaussianLikelihood(noise=1e-3)
                kernel_class = TanimotoKernel  # Using the imported TanimotoKernel
                model = Gauche(X_train_tensor, y_train_tensor, likelihood, kernel_class)
                
                # Fit model
                mll = ExactMarginalLogLikelihood(likelihood, model)
                fit_gpytorch_model(mll)
                
                # Make predictions
                model.eval()
                likelihood.eval()
                with torch.no_grad():
                    preds = model(X_test_tensor)
                    y_pred = preds.mean.numpy()
                    pred_vars = preds.variance.numpy()
                
                # Save uncertainty if requested
                if save_uncertainty and 'uncertainty_info' in kwargs:
                    info = kwargs['uncertainty_info']
                    # This call raised TypeError on contact until 2026-08-26: `y_true`
                    # is not a parameter and three required ones were missing, so it
                    # had never run. The test split is never noised, so the clean and
                    # the noisy label are the same array and injected_noise is 0.0.
                    save_uncertainty_values(
                        y_pred_mean=y_pred,
                        y_pred_std=np.sqrt(pred_vars),
                        y_true_original=y_test,
                        y_true_noisy=y_test,
                        filepath=info['filepath'],
                        model_name="gauche",
                        rep=info['rep'],
                        sigma_noise=info['sigma'],
                        iteration=info['iteration'],
                        file_no=info.get('file_no', 0),
                        # This path writes a total and performs no split, which
                        # is what the SUPPORT entry records. The model column
                        # says 'gauche', so the key is given explicitly rather
                        # than letting the guard read this as the roster's
                        # Gaussian process and demand two components it does not
                        # compute (RERUN_PLAN.md 5.5).
                        support_model='noise_mitigation_gauche',
                        split='test'
                    )
                
                # Calculate metrics
                r2 = r2_score(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                
                return {'r2': r2, 'rmse': rmse, 'mae': mae}
        
        # Non-Gauche models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            model = SVR(kernel='rbf', C=1.0, gamma='scale')
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0
            )
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        return {'r2': r2, 'rmse': rmse, 'mae': mae}
        
    except Exception as e:
        print(f"Error in model training: {e}")
        return {'r2': -999, 'rmse': 999, 'mae': 999}

def run_noise_mitigation_experiment(args):
    """Main experiment function"""
    # Set seeds
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    print(f"Loading QM9 with target: {args.target}")
    qm9 = load_qm9(args.target)
    
    print(f"Splitting dataset with sample size: {args.sample_size}")
    train_idx, test_idx, val_idx = split_qm9_simple(qm9, args.sample_size, args.random_seed)
    
    # Get data subsets - USE SHUFFLED INDICES
    qm9_subset = qm9[:args.sample_size]
    
    # Extract features with proper train/test isolation
    print("Extracting molecular features...")
    X_all = extract_molecular_features(qm9_subset, train_idx, args.molecular_representation)
    
    # Split features using the shuffled indices
    X_train = X_all[train_idx]
    X_test = X_all[test_idx]
    X_val = X_all[val_idx]
    
    # Extract target values using shuffled indices - CLEAN versions (no noise yet)
    y_train_clean = np.array([qm9_subset[i].y.item() for i in train_idx])
    y_test_clean = np.array([qm9_subset[i].y.item() for i in test_idx])
    y_val_clean = np.array([qm9_subset[i].y.item() for i in val_idx])
    
    print(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Clean target ranges - Train: [{y_train_clean.min():.3f}, {y_train_clean.max():.3f}]")
    print(f"Clean target ranges - Test: [{y_test_clean.min():.3f}, {y_test_clean.max():.3f}]")
    
    # Compute normalization parameters from CLEAN training data
    train_mean = y_train_clean.mean()
    train_std = y_train_clean.std()
    
    if train_std == 0:
        train_std = 1.0
    
    print(f"Normalization parameters: mean={train_mean:.3f}, std={train_std:.3f}")
    
    # Normalize CLEAN test/val data (these never get noise)
    if args.normalize:
        y_test_norm = (y_test_clean - train_mean) / train_std
        y_val_norm = (y_val_clean - train_mean) / train_std
    else:
        y_test_norm = y_test_clean
        y_val_norm = y_val_clean
    
    results = []
    
    for noise_level in args.noise_levels:
        print(f"\n=== Testing noise level: {noise_level} ===")
        
        # CRITICAL FIX: Add noise ONLY to training data
        y_train_noisy, true_noise = add_artificial_noise(
            y_train_clean, noise_level, args.noise_type, args.random_seed
        )
        
        # Normalize CLEAN training data for clean baseline
        if args.normalize:
            y_train_clean_norm = (y_train_clean - train_mean) / train_std
            # Normalize NOISY training data using CLEAN training statistics
            y_train_noisy_norm = (y_train_noisy - train_mean) / train_std
        else:
            y_train_clean_norm = y_train_clean
            y_train_noisy_norm = y_train_noisy
        
        print(f"Noisy train target range: [{y_train_noisy.min():.3f}, {y_train_noisy.max():.3f}]")
        print(f"Clean test target range: [{y_test_clean.min():.3f}, {y_test_clean.max():.3f}]")
        
        # Baseline: clean data (train on clean, test on clean)
        print("  Training baseline on clean data...")
        uncertainty_info = {
            'filepath': args.output_path,
            'rep': args.molecular_representation,
            'sigma': noise_level,
            'iteration': 0,
            'file_no': 0
        }
        clean_metrics = train_baseline_model(
            X_train, y_train_clean_norm, X_test, y_test_norm, 
            args.baseline_model, 
            save_uncertainty=args.save_uncertainty,
            uncertainty_info=uncertainty_info
        )
        results.append({
            'noise_level': noise_level, 'detector': 'clean_data', 'cleaner': 'none',
            'r2': clean_metrics['r2'], 'rmse': clean_metrics['rmse'], 'mae': clean_metrics['mae']
        })
        print(f"    Clean data R² = {clean_metrics['r2']:.4f}")
        
        # If clean baseline is terrible, something is fundamentally wrong
        if clean_metrics['r2'] < 0:
            print("WARNING: Clean baseline R² is negative - check data quality")
        
        # Baseline: noisy data (train on noisy, test on clean)
        print("  Training baseline on noisy data...")
        noisy_metrics = train_baseline_model(X_train, y_train_noisy_norm, X_test, y_test_norm, args.baseline_model)
        results.append({
            'noise_level': noise_level, 'detector': 'no_detection', 'cleaner': 'none',
            'r2': noisy_metrics['r2'], 'rmse': noisy_metrics['rmse'], 'mae': noisy_metrics['mae']
        })
        print(f"    Noisy data R² = {noisy_metrics['r2']:.4f}")
        
        # Noise detection and mitigation
        print("  Running noise detection and mitigation...")
        
        # Define detection methods (operate on ORIGINAL scale for better detection)
        detectors = [
            ('distance_euclidean', lambda: detect_noise_distance_based(X_train, y_train_noisy)),
            ('uncertainty_ensemble', lambda: detect_noise_uncertainty_based(X_train, y_train_noisy)),
            ('clustering', lambda: detect_noise_clustering_based(X_train, y_train_noisy))
        ]
        
        # Define cleaning methods
        cleaners = [
            ('removal', clean_data_removal),
            ('smoothing', clean_data_smoothing),
            ('replacement', lambda X, y, mask: clean_data_replacement(X, y, mask, args.baseline_model))
        ]
        
        for detector_name, detector_func in detectors:
            try:
                print(f"    Running detector: {detector_name}")
                detected_noise_mask = detector_func()
                
                n_detected = np.sum(detected_noise_mask)
                n_total = len(detected_noise_mask)
                print(f"      Detected {n_detected}/{n_total} noisy samples")
                
                if n_detected == 0:
                    print(f"      No noise detected by {detector_name}")
                    continue
                elif n_detected == n_total:
                    print(f"      All samples detected as noise by {detector_name}, skipping")
                    continue
                
                for cleaner_name, cleaner_func in cleaners:
                    try:
                        print(f"      Running cleaner: {cleaner_name}")
                        
                        # Apply cleaning to normalized data for training
                        X_cleaned, y_cleaned = cleaner_func(X_train, y_train_noisy_norm, detected_noise_mask)
                        
                        if len(X_cleaned) < 10:
                            print(f"        Too few samples left after cleaning ({len(X_cleaned)})")
                            continue
                        
                        # Train and evaluate
                        cleaned_metrics = train_baseline_model(X_cleaned, y_cleaned, X_test, y_test_norm, args.baseline_model)
                        
                        results.append({
                            'noise_level': noise_level, 'detector': detector_name, 'cleaner': cleaner_name,
                            'r2': cleaned_metrics['r2'], 'rmse': cleaned_metrics['rmse'], 'mae': cleaned_metrics['mae'],
                            'n_detected': n_detected, 'n_total': n_total, 'n_remaining': len(X_cleaned)
                        })
                        
                        print(f"        R² = {cleaned_metrics['r2']:.4f} (n={len(X_cleaned)})")
                        
                    except Exception as e:
                        print(f"        Error with cleaner {cleaner_name}: {e}")
                        
            except Exception as e:
                print(f"      Error with detector {detector_name}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to: {args.output_path}")
    
    # Create plots if requested
    if args.create_plots:
        print("Creating performance plots...")
        plot_dir = os.path.join(os.path.dirname(args.output_path), 'plots')
        create_performance_plots(results_df, plot_dir)
        
        # Create uncertainty plots if uncertainty data exists
        if args.save_uncertainty:
            uncertainty_file = args.output_path.replace('.csv', '_uncertainty.csv')
            create_uncertainty_plots(uncertainty_file, plot_dir)
        
        print(f"Plots saved to: {plot_dir}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for noise_level in args.noise_levels:
        level_results = results_df[results_df['noise_level'] == noise_level]
        
        clean_results = level_results[level_results['detector'] == 'clean_data']
        noisy_results = level_results[level_results['detector'] == 'no_detection']
        
        if len(clean_results) > 0 and len(noisy_results) > 0:
            clean_r2 = clean_results['r2'].iloc[0]
            noisy_r2 = noisy_results['r2'].iloc[0]
            
            print(f"\nNoise {noise_level}:")
            print(f"  Clean R² = {clean_r2:.4f}")
            print(f"  Noisy R² = {noisy_r2:.4f}")
            
            if clean_r2 > noisy_r2:
                print(f"  Performance drop = {clean_r2 - noisy_r2:.4f}")
            else:
                print(f"  Performance change = {noisy_r2 - clean_r2:.4f}")
            
            # Find best cleaning method
            cleaned_results = level_results[~level_results['detector'].isin(['clean_data', 'no_detection'])]
            if not cleaned_results.empty:
                best_idx = cleaned_results['r2'].idxmax()
                best = cleaned_results.loc[best_idx]
                improvement = best['r2'] - noisy_r2
                
                if clean_r2 != noisy_r2:
                    recovery = improvement / (clean_r2 - noisy_r2) * 100
                else:
                    recovery = 0
                
                print(f"  Best method: {best['detector']} + {best['cleaner']}")
                print(f"  Best R² = {best['r2']:.4f}")
                print(f"  Improvement = {improvement:.4f}")
                print(f"  Recovery = {recovery:.1f}%")
                
                if 'n_remaining' in best:
                    print(f"  Samples remaining: {best['n_remaining']}")
            else:
                print("  No successful cleaning methods")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Fixed QM9 Noise Mitigation Comparison')
    
    parser.add_argument('--target', type=str, default='homo_lumo_gap', 
                       choices=list(properties.keys()), help='QM9 target property')
    parser.add_argument('--sample_size', type=int, default=10000, help='Sample size from QM9')
    parser.add_argument('--molecular_representation', type=str, default='sns', 
                       choices=['sns', 'pdv'], help='Molecular representation')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[round(x, 1) for x in np.arange(0.1, 2.1, 0.1)], 
                       help='Noise levels to test')
    parser.add_argument('--noise_type', type=str, default='gaussian', 
                       choices=['gaussian', 'uniform', 'outlier'], help='Type of noise')
    parser.add_argument('--baseline_model', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'xgboost', 'gauche'], help='Baseline model')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_path', type=str, default='fixed_noise_mitigation_results.csv',
                       help='Output CSV path')
    parser.add_argument('--normalize', action='store_true', default=True,
                       help='Normalize targets (default: True)')
    parser.add_argument('--save_uncertainty', action='store_true', default=True,
                       help='Save uncertainty values for analysis (default: True)')
    parser.add_argument('--create_plots', action='store_true', default=True,
                       help='Create comprehensive plots (default: True)')
    
    return parser.parse_args()

def main():
    """Main function"""
    print("=== Fixed QM9 Noise Mitigation Script ===")
    print("Key improvements:")
    print("1. Consistent feature extraction pipeline")
    print("2. Proper data normalization flow")
    print("3. Fixed model training parameters")
    print("4. Corrected noise detection on appropriate scales")
    print("=" * 50)
    
    args = parse_arguments()
    
    # Handle "both" option for molecular representation
    if args.molecular_representation == 'both':
        representations = ['sns', 'pdv']
    else:
        representations = [args.molecular_representation]
    
    for rep in representations:
        print(f"\n{'='*20} TESTING {rep.upper()} {'='*20}")
        
        # Update args for this representation
        args.molecular_representation = rep
        
        # Update output path to include representation
        base_path = args.output_path
        if base_path.endswith('.csv'):
            rep_path = base_path.replace('.csv', f'_{rep}.csv')
        else:
            rep_path = f"{base_path}_{rep}.csv"
        args.output_path = rep_path
        
        print(f"Running experiment with {rep} representation...")
        print(f"Results will be saved to: {rep_path}")
        
        run_noise_mitigation_experiment(args)
        
        print(f"Completed {rep} representation")
    
    print("\n" + "="*50)
    print("All experiments completed!")

if __name__ == "__main__":
    main()