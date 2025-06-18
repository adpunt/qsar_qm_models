#!/usr/bin/env python3
"""
QM9 Noise Mitigation Comparison Script
"""

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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

import scipy.spatial.distance as distance
from scipy import stats

from torch_geometric.datasets import QM9
import xgboost as xgb
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import MLE

sys.path.append('../models/')

from models import *
from utils import *

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
    qm9 = QM9(root=os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', 'QM9'))
    y_target = pd.DataFrame(qm9.data.y.numpy())
    property_index = properties[target]
    qm9.data.y = torch.Tensor(y_target[property_index])
    return qm9

def create_sort_and_slice_ecfp_featuriser(mols_train, max_radius=2, pharm_atom_invs=False, 
                                          bond_invs=True, chirality=False, sub_counts=True, 
                                          vec_dimension=1024, break_ties_with=lambda sub_id: sub_id, 
                                          print_train_set_info=True):
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=max_radius,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if pharm_atom_invs else rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership=True),
        useBondTypes=bond_invs,
        includeChirality=chirality
    )
    
    sub_id_enumerator = lambda mol: morgan_generator.GetSparseCountFingerprint(mol).GetNonzeroElements() if mol is not None else {}
    
    sub_ids_to_prevs_dict = {}
    for mol in mols_train:
        for sub_id in sub_id_enumerator(mol).keys():
            sub_ids_to_prevs_dict[sub_id] = sub_ids_to_prevs_dict.get(sub_id, 0) + 1

    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict, key=lambda sub_id: (sub_ids_to_prevs_dict[sub_id], break_ties_with(sub_id)), reverse=True)
    
    def standard_unit_vector(dim, k):
        vec = np.zeros(dim, dtype=int)
        vec[k] = 1
        return vec
    
    def sub_id_one_hot_encoder(sub_id):
        return standard_unit_vector(vec_dimension, sub_ids_sorted_list.index(sub_id)) if sub_id in sub_ids_sorted_list[0: vec_dimension] else np.zeros(vec_dimension)
    
    def ecfp_featuriser(mol):
        if sub_counts:
            sub_id_list = [sub_idd for (sub_id, count) in sub_id_enumerator(mol).items() for sub_idd in [sub_id]*count]
        else:
            sub_id_list = list(sub_id_enumerator(mol).keys())
        
        ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis=0)
        return ecfp_vector
    
    if print_train_set_info:
        print(f"Number of compounds in molecular training set: {len(mols_train)}")
        print(f"Number of unique circular substructures: {len(sub_ids_to_prevs_dict)}")

    return ecfp_featuriser

def rdkit_mol_descriptors_from_smiles(smiles_string):
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is None:
        # Return zeros for invalid molecules
        mol_descriptor_calculator = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
        return np.zeros(len(DEFAULT_DESCRIPTOR_LIST))
    
    mol_descriptor_calculator = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
    descriptor_vals = mol_descriptor_calculator.CalcDescriptors(mol)
    return np.array(descriptor_vals)

def split_qm9_simple(qm9, sample_size, random_seed):
    indices = torch.randperm(len(qm9), generator=torch.Generator().manual_seed(random_seed))
    qm9 = qm9.index_select(indices)
    
    train_size = int(sample_size * 0.8)
    test_size = int(sample_size * 0.1)
    val_size = int(sample_size * 0.1)
    
    train_idx = list(range(train_size))
    val_idx = list(range(train_size, train_size + val_size))
    test_idx = list(range(train_size + val_size, train_size + val_size + test_size))
    
    return train_idx, test_idx, val_idx

def extract_molecular_features(qm9_subset, molecular_representation='sns'):
    smiles_list = [data.smiles for data in qm9_subset]
    
    if molecular_representation == 'sns':
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        valid_mols = [mol for mol in mols if mol is not None]
        
        ecfp_featuriser = create_sort_and_slice_ecfp_featuriser(
            mols_train=valid_mols, max_radius=2, pharm_atom_invs=False, 
            bond_invs=True, chirality=False, sub_counts=True, 
            vec_dimension=1024, print_train_set_info=False
        )
        
        features = []
        for mol in mols:
            if mol is not None:
                features.append(ecfp_featuriser(mol))
            else:
                features.append(np.zeros(1024))
        
        return np.array(features)
    
    elif molecular_representation == 'pdv':
        features = []
        for smiles in smiles_list:
            features.append(rdkit_mol_descriptors_from_smiles(smiles))
        return np.array(features)
    
    else:
        raise ValueError(f"Unsupported molecular representation: {molecular_representation}")

def add_artificial_noise(y, noise_level, noise_type='gaussian'):
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level * np.std(y), size=y.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_level, noise_level, size=y.shape)
    elif noise_type == 'outlier':
        noise = np.zeros_like(y)
        n_outliers = int(noise_level * len(y))
        outlier_indices = np.random.choice(len(y), n_outliers, replace=False)
        noise[outlier_indices] = np.random.normal(0, np.std(y) * 3, size=n_outliers)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return y + noise, noise

def detect_noise_distance_based(X, y, distance_metric='tanimoto', k_neighbors=5, threshold=2.0):
    n_samples = X.shape[0]
    
    if distance_metric == 'tanimoto':
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                intersection = np.sum(np.minimum(X[i], X[j]))
                union = np.sum(np.maximum(X[i], X[j]))
                tanimoto = intersection / union if union > 0 else 0
                dist_matrix[i, j] = dist_matrix[j, i] = 1 - tanimoto
    else:
        dist_matrix = distance.squareform(distance.pdist(X, distance_metric))
    
    noise_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        distances = dist_matrix[i]
        neighbor_indices = np.argsort(distances)[1:k_neighbors+1]
        
        neighbor_labels = y[neighbor_indices]
        current_label = y[i]
        
        if len(neighbor_labels) > 1:
            neighbor_std = np.std(neighbor_labels)
            neighbor_mean = np.mean(neighbor_labels)
            label_deviation = abs(current_label - neighbor_mean)
            
            weights = 1 / (distances[neighbor_indices] + 1e-8)
            weighted_deviation = np.sum(weights * np.abs(neighbor_labels - current_label)) / np.sum(weights)
            
            noise_scores[i] = weighted_deviation + label_deviation / (neighbor_std + 1e-8)
    
    threshold_value = np.mean(noise_scores) + threshold * np.std(noise_scores)
    return noise_scores > threshold_value

def detect_noise_uncertainty_based(X, y, model_type='qrf', uncertainty_threshold=0.95):
    if model_type == 'qrf':
        from quantile_forest import RandomForestQuantileRegressor
        model = RandomForestQuantileRegressor(n_estimators=100, random_state=42)
        
        uncertainties = np.zeros(len(X))
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            
            model.fit(X_train, y_train)
            q16, q50, q84 = model.predict(X_test, quantiles=[0.16, 0.5, 0.84]).T
            uncertainties[test_idx] = 0.5 * (q84 - q16)
        
        threshold = np.percentile(uncertainties, uncertainty_threshold * 100)
        return uncertainties > threshold
    
    elif model_type == 'ngboost':
        model = NGBRegressor(Dist=Normal, Score=MLE, natural_gradient=True, 
                           n_estimators=100, learning_rate=0.01, verbose=False, random_state=42)
        
        uncertainties = np.zeros(len(X))
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]
            
            model.fit(X_train, y_train)
            pred_dist = model.pred_dist(X_test)
            uncertainties[test_idx] = pred_dist.scale
        
        threshold = np.percentile(uncertainties, uncertainty_threshold * 100)
        return uncertainties > threshold
    
    elif model_type == 'ensemble':
        n_models = 10
        predictions = np.zeros((len(X), n_models))
        
        for i in range(n_models):
            bootstrap_idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[bootstrap_idx]
            y_boot = y[bootstrap_idx]
            
            model = RandomForestRegressor(n_estimators=50, random_state=i)
            model.fit(X_boot, y_boot)
            predictions[:, i] = model.predict(X)
        
        uncertainties = np.std(predictions, axis=1)
        threshold = np.percentile(uncertainties, uncertainty_threshold * 100)
        return uncertainties > threshold
    
    else:
        raise ValueError(f"Unknown uncertainty model: {model_type}")

def detect_noise_clustering_based(X, y, eps=0.5, min_samples=5):
    combined_features = np.column_stack([StandardScaler().fit_transform(X), y.reshape(-1, 1)])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(combined_features)
    return clusters == -1

def clean_data_removal(X, y, noise_mask):
    clean_mask = ~noise_mask
    return X[clean_mask], y[clean_mask]

def clean_data_smoothing(X, y, noise_mask, k_neighbors=5, distance_metric='tanimoto'):
    y_cleaned = y.copy()
    n_samples = X.shape[0]
    
    if distance_metric == 'tanimoto':
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i+1, n_samples):
                intersection = np.sum(np.minimum(X[i], X[j]))
                union = np.sum(np.maximum(X[i], X[j]))
                tanimoto = intersection / union if union > 0 else 0
                dist_matrix[i, j] = dist_matrix[j, i] = 1 - tanimoto
    else:
        dist_matrix = distance.squareform(distance.pdist(X, distance_metric))
    
    for i in np.where(noise_mask)[0]:
        distances = dist_matrix[i]
        
        clean_indices = np.where(~noise_mask)[0]
        if len(clean_indices) >= k_neighbors:
            neighbor_candidates = clean_indices
        else:
            neighbor_candidates = np.arange(len(X))
        
        neighbor_candidates = neighbor_candidates[neighbor_candidates != i]
        
        if len(neighbor_candidates) > 0:
            neighbor_distances = distances[neighbor_candidates]
            nearest_k = np.argsort(neighbor_distances)[:k_neighbors]
            neighbor_indices = neighbor_candidates[nearest_k]
            
            weights = 1 / (distances[neighbor_indices] + 1e-8)
            weights /= np.sum(weights)
            
            y_cleaned[i] = np.sum(weights * y[neighbor_indices])
    
    return X, y_cleaned

def clean_data_replacement(X, y, noise_mask, model_type='random_forest'):
    y_cleaned = y.copy()
    
    clean_mask = ~noise_mask
    if np.sum(clean_mask) < 10:
        return X, y
    
    X_clean = X[clean_mask]
    y_clean = y[clean_mask]
    
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVR(kernel='rbf')
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_clean, y_clean)
    
    noisy_indices = np.where(noise_mask)[0]
    if len(noisy_indices) > 0:
        y_cleaned[noisy_indices] = model.predict(X[noisy_indices])
    
    return X, y_cleaned

def train_baseline_model(X_train, y_train, X_test, y_test, model_type='random_forest'):
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type == 'svm':
        model = SVR(kernel='rbf')
    elif model_type == 'xgboost':
        model = xgb.XGBRegressor(random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {'r2': r2, 'rmse': rmse, 'mae': mae}

def run_noise_mitigation_experiment(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    
    print(f"Loading QM9 with target: {args.target}")
    qm9 = load_qm9(args.target)
    
    print(f"Splitting dataset with sample size: {args.sample_size}")
    train_idx, test_idx, val_idx = split_qm9_simple(qm9, args.sample_size, args.random_seed)
    
    print(f"Extracting {args.molecular_representation} features...")
    train_subset = [qm9[i] for i in train_idx]
    test_subset = [qm9[i] for i in test_idx]
    
    X_train = extract_molecular_features(train_subset, args.molecular_representation)
    X_test = extract_molecular_features(test_subset, args.molecular_representation)
    
    y_train = np.array([qm9[i].y.item() for i in train_idx])
    y_test = np.array([qm9[i].y.item() for i in test_idx])
    
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Original target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    results = []
    
    for noise_level in args.noise_levels:
        print(f"\n=== Testing noise level: {noise_level} ===")
        
        # Add noise to original (non-normalized) targets
        y_train_noisy, true_noise = add_artificial_noise(y_train, noise_level, args.noise_type)
        
        # Normalize using the NOISY training data statistics (like your original script)
        mean = y_train_noisy.mean()
        std = y_train_noisy.std()
        
        y_train_clean_norm = (y_train - mean) / std
        y_train_noisy_norm = (y_train_noisy - mean) / std
        y_test_norm = (y_test - mean) / std
        
        print(f"Normalized target range: [{y_train_noisy_norm.min():.3f}, {y_train_noisy_norm.max():.3f}]")
        
        # Baseline: clean data (but normalized with noisy stats)
        clean_metrics = train_baseline_model(X_train, y_train_clean_norm, X_test, y_test_norm, args.baseline_model)
        results.append({
            'noise_level': noise_level, 'detector': 'clean_data', 'cleaner': 'none',
            'r2': clean_metrics['r2'], 'rmse': clean_metrics['rmse'], 'mae': clean_metrics['mae']
        })
        
        # Baseline: noisy data
        noisy_metrics = train_baseline_model(X_train, y_train_noisy_norm, X_test, y_test_norm, args.baseline_model)
        results.append({
            'noise_level': noise_level, 'detector': 'no_detection', 'cleaner': 'none',
            'r2': noisy_metrics['r2'], 'rmse': noisy_metrics['rmse'], 'mae': noisy_metrics['mae']
        })
        
        # Detection methods (use normalized noisy data)
        detectors = [
            ('distance_tanimoto', lambda: detect_noise_distance_based(X_train, y_train_noisy_norm, 'tanimoto')),
            ('distance_euclidean', lambda: detect_noise_distance_based(X_train, y_train_noisy_norm, 'euclidean')),
            ('uncertainty_qrf', lambda: detect_noise_uncertainty_based(X_train, y_train_noisy_norm, 'qrf')),
            ('uncertainty_ngboost', lambda: detect_noise_uncertainty_based(X_train, y_train_noisy_norm, 'ngboost')),
            ('uncertainty_ensemble', lambda: detect_noise_uncertainty_based(X_train, y_train_noisy_norm, 'ensemble')),
            ('clustering', lambda: detect_noise_clustering_based(X_train, y_train_noisy_norm))
        ]
        
        # Cleaning methods
        cleaners = [
            ('removal', clean_data_removal),
            ('smoothing', lambda X, y, mask: clean_data_smoothing(X, y, mask, distance_metric='tanimoto')),
            ('replacement', lambda X, y, mask: clean_data_replacement(X, y, mask, args.baseline_model))
        ]
        
        for detector_name, detector_func in detectors:
            try:
                print(f"  Running detector: {detector_name}")
                detected_noise_mask = detector_func()
                
                for cleaner_name, cleaner_func in cleaners:
                    try:
                        print(f"    Running cleaner: {cleaner_name}")
                        X_cleaned, y_cleaned = cleaner_func(X_train, y_train_noisy_norm, detected_noise_mask)
                        
                        cleaned_metrics = train_baseline_model(X_cleaned, y_cleaned, X_test, y_test_norm, args.baseline_model)
                        
                        results.append({
                            'noise_level': noise_level, 'detector': detector_name, 'cleaner': cleaner_name,
                            'r2': cleaned_metrics['r2'], 'rmse': cleaned_metrics['rmse'], 'mae': cleaned_metrics['mae']
                        })
                        
                    except Exception as e:
                        print(f"      Error with cleaner {cleaner_name}: {e}")
                        
            except Exception as e:
                print(f"    Error with detector {detector_name}: {e}")
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_path, index=False)
    print(f"\nResults saved to: {args.output_path}")
    
    # Print summary
    print("\n=== SUMMARY ===")
    for noise_level in args.noise_levels:
        level_results = results_df[results_df['noise_level'] == noise_level]
        clean_r2 = level_results[level_results['detector'] == 'clean_data']['r2'].iloc[0]
        noisy_r2 = level_results[level_results['detector'] == 'no_detection']['r2'].iloc[0]
        
        cleaned_results = level_results[~level_results['detector'].isin(['clean_data', 'no_detection'])]
        if not cleaned_results.empty:
            best_idx = cleaned_results['r2'].idxmax()
            best = cleaned_results.loc[best_idx]
            improvement = best['r2'] - noisy_r2
            
            print(f"Noise {noise_level}: Clean R²={clean_r2:.4f}, Noisy R²={noisy_r2:.4f}")
            print(f"  Best: {best['detector']} + {best['cleaner']} (R²={best['r2']:.4f}, improvement={improvement:.4f})")
            
def parse_arguments():
    parser = argparse.ArgumentParser(description='QM9 Noise Mitigation Comparison')
    
    parser.add_argument('--target', type=str, default='homo_lumo_gap', 
                       choices=list(properties.keys()), help='QM9 target property')
    parser.add_argument('--sample_size', type=int, default=1000, help='Sample size from QM9')
    parser.add_argument('--molecular_representation', type=str, default='sns', 
                       choices=['sns', 'pdv'], help='Molecular representation')
    parser.add_argument('--noise_levels', type=float, nargs='+', default=[0.05, 0.1, 0.2], 
                       help='Noise levels to test')
    parser.add_argument('--noise_type', type=str, default='gaussian', 
                       choices=['gaussian', 'uniform', 'outlier'], help='Type of noise')
    parser.add_argument('--baseline_model', type=str, default='random_forest',
                       choices=['random_forest', 'svm', 'xgboost'], help='Baseline model')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_path', type=str, default='noise_mitigation_results.csv',
                       help='Output CSV path')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    run_noise_mitigation_experiment(args)

if __name__ == "__main__":
    main()