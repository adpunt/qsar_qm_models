from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
from scipy.stats import pearsonr
import numpy as np
import os
import csv
import shap
import pandas as pd

import torch
from torch_geometric.data import Data
from rdkit import Chem

def save_results(filepath, s, iteration, model, rep, n, metrics, params_source='default', loss_function='mse'):
    """
    Save results to a CSV file with loss function tracking
    """
    if filepath:
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if the file is new
            if not file_exists:
                writer.writerow(["sigma", "iteration", "model", "rep", "sample_size", "mae", "mse", "rmse", "r2", "pearson_corr", "params_source", "loss_function"])
            
            # Save the results
            writer.writerow([s, iteration, model, rep, n, metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], params_source, loss_function])

def calculate_regression_metrics(y_test, prediction, logging=False):
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    pearson_corr, _ = pearsonr(y_test, prediction)

    # Optionally log the metrics
    if logging:
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("RMSE:", rmse)
        print("R-squared:", r2)
        print("Pearson Correlation:", pearson_corr)

    return mae, mse, rmse, r2, pearson_corr


# ============================================================================
# UNCERTAINTY DECOMPOSITION HELPERS
# ============================================================================

def decompose_uncertainty_sampling(predictions_array, num_samples):
    """
    Decompose uncertainty from sampling-based methods (BNN, ensembles).
    
    Args:
        predictions_array: numpy array of shape (num_samples, num_datapoints)
        num_samples: number of forward passes
    
    Returns:
        epistemic: uncertainty due to model parameters (variance of means)
        aleatoric: uncertainty due to data noise (mean of variances)
        total: total predictive uncertainty
    """
    # Epistemic: variance across different model samples
    epistemic = predictions_array.std(axis=0)
    
    # Aleatoric: for simple sampling, we can't separate this without explicit noise modeling
    # So we return None - only models with heteroscedastic outputs can provide this
    aleatoric = None
    
    # Total uncertainty
    total = epistemic  # When aleatoric is unknown
    
    return epistemic, aleatoric, total


def decompose_uncertainty_sampling_heteroscedastic(mean_predictions, var_predictions):
    """
    Decompose uncertainty from heteroscedastic sampling methods.
    
    Args:
        mean_predictions: array of shape (num_samples, num_datapoints) - predicted means
        var_predictions: array of shape (num_samples, num_datapoints) - predicted variances
    
    Returns:
        epistemic: uncertainty due to model parameters
        aleatoric: uncertainty due to data noise (from learned variance)
        total: total predictive uncertainty
    """
    # Epistemic: variance of the predicted means across samples
    epistemic = mean_predictions.std(axis=0)
    
    # Aleatoric: average of the predicted variances
    aleatoric = np.sqrt(var_predictions.mean(axis=0))
    
    # Total: sqrt(epistemic^2 + aleatoric^2)
    total = np.sqrt(epistemic**2 + aleatoric**2)
    
    return epistemic, aleatoric, total


def decompose_uncertainty_gp(posterior_variance, likelihood_noise):
    """
    Decompose uncertainty from Gaussian Process models.
    
    Args:
        posterior_variance: numpy array of posterior predictive variance
        likelihood_noise: scalar noise parameter from likelihood
    
    Returns:
        epistemic: uncertainty due to model (posterior variance)
        aleatoric: uncertainty due to data noise (likelihood noise)
        total: total predictive uncertainty
    """
    # Epistemic: posterior variance (model uncertainty)
    epistemic = np.sqrt(posterior_variance)
    
    # Aleatoric: likelihood noise (observation noise)
    aleatoric = np.full_like(epistemic, np.sqrt(likelihood_noise))
    
    # Total: GP predictive includes both
    total = np.sqrt(posterior_variance + likelihood_noise)
    
    return epistemic, aleatoric, total


def decompose_uncertainty_distributional(pred_mean, pred_std_or_var, model_type='ngboost', is_variance=False):
    """
    Decompose uncertainty from distributional models (NGBoost, QRF, heteroscedastic NN).
    These models only capture aleatoric uncertainty directly.
    
    Args:
        pred_mean: predicted means
        pred_std_or_var: predicted standard deviations or variances
        model_type: 'ngboost', 'qrf', 'heteroscedastic'
        is_variance: True if pred_std_or_var contains variances, False if std
    
    Returns:
        epistemic: None (single model can't estimate this)
        aleatoric: predicted uncertainty (data noise)
        total: same as aleatoric for single models
    """
    # Convert to std if variance was provided
    if is_variance:
        aleatoric = np.sqrt(pred_std_or_var)
    else:
        aleatoric = pred_std_or_var
    
    # Single distributional models can't estimate epistemic uncertainty
    epistemic = None
    
    # Total is just aleatoric for these models
    total = aleatoric
    
    return epistemic, aleatoric, total


# ============================================================================
# SAVE UNCERTAINTY WITH DECOMPOSITION
# ============================================================================

def save_uncertainty_values(y_pred_mean, y_pred_std, y_true_original, y_true_noisy, 
                           filepath, model_name, rep, sigma_noise, iteration, file_no,
                           y_pred_std_calibrated=None, temperature=None,
                           epistemic_uncertainty=None, aleatoric_uncertainty=None):
    """
    Save uncertainty values with optional epistemic/aleatoric decomposition.
    
    UPDATED to handle decomposition columns.
    """
    uncertainty_file = filepath.replace('.csv', '_uncertainty_values.csv')
    
    rows = []
    for i in range(len(y_pred_mean)):
        row = {
            'model': model_name,
            'representation': rep,
            'sigma': sigma_noise,
            'iteration': iteration,
            'file_no': file_no,
            'sample_idx': i,
            'y_pred_mean': y_pred_mean[i],
            'y_pred_std_uncalibrated': y_pred_std[i],
            'y_true_original': y_true_original[i],
            'y_true_noisy': y_true_noisy[i],
            'injected_noise': y_true_noisy[i] - y_true_original[i]
        }
        
        # Add calibrated values if provided
        if y_pred_std_calibrated is not None:
            row['y_pred_std_calibrated'] = y_pred_std_calibrated[i]
            row['temperature'] = temperature
        else:
            # No calibration - set calibrated = uncalibrated
            row['y_pred_std_calibrated'] = y_pred_std[i]
            row['temperature'] = 1.0
        
        # Add decomposition if provided
        if epistemic_uncertainty is not None:
            row['epistemic_uncertainty'] = epistemic_uncertainty[i]
        else:
            row['epistemic_uncertainty'] = np.nan
        
        if aleatoric_uncertainty is not None:
            row['aleatoric_uncertainty'] = aleatoric_uncertainty[i]
        else:
            row['aleatoric_uncertainty'] = np.nan
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    if os.path.exists(uncertainty_file):
        df.to_csv(uncertainty_file, mode='a', header=False, index=False)
    else:
        df.to_csv(uncertainty_file, mode='w', header=True, index=False)


def save_conformal_intervals(y_pred, y_lower, y_upper, y_true, filepath, model_name, rep, sigma_noise, iteration, file_no, alpha):
    # Create directory if it doesn't exist
    intervals_dir = os.path.join(os.path.dirname(filepath), "conformal_intervals")
    os.makedirs(intervals_dir, exist_ok=True)
    
    # Create dataframe with results
    results_df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'y_lower': y_lower,
        'y_upper': y_upper,
        'interval_width': y_upper - y_lower,
        'coverage': ((y_true >= y_lower) & (y_true <= y_upper)).astype(int),
        'alpha': alpha,
        'model_name': model_name,
        'rep': rep,
        'sigma_noise': sigma_noise,
        'iteration': iteration,
        'file_no': file_no
    })
    
    # Calculate empirical coverage
    empirical_coverage = results_df['coverage'].mean()
    results_df['empirical_coverage'] = empirical_coverage
    
    # Save to file following your naming convention
    filename = f"conformal_intervals_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}_file{file_no}.csv"
    filepath_full = os.path.join(intervals_dir, filename)
    results_df.to_csv(filepath_full, index=False)
    
    print(f"Conformal intervals saved. Target coverage: {1-alpha:.1%}, Empirical coverage: {empirical_coverage:.1%}")

def save_per_epoch_metrics(train_losses, val_losses, filepath, model_name, rep, sigma_noise, iteration, file_no):
   """
   Save per-epoch training and validation metrics to a CSV file.
   """
   if filepath:
       # Create per-epoch specific filepath
       base_path = filepath.replace('.csv', '_per_epoch.csv')
       file_exists = os.path.isfile(base_path)
       
       with open(base_path, mode='a', newline='') as f:
           writer = csv.writer(f)
           
           # Write header if the file is new
           if not file_exists:
               writer.writerow(["sigma", "iteration", "model", "rep", "file_no", "epoch", "train_loss", "val_loss"])
           
           # Save metrics for each epoch
           for epoch, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses)):
               writer.writerow([sigma_noise, iteration, model_name, rep, file_no, epoch, train_loss, val_loss])

def calibrate_uncertainty_simple(y_pred_mean, y_pred_std, y_true):
    """
    Find optimal temperature T for variance scaling.
    Works for any model that outputs mean and std.
    
    Args:
        y_pred_mean: predicted means (numpy array)
        y_pred_std: predicted stds (numpy array)
        y_true: true values (numpy array)
    
    Returns:
        float: optimal temperature T
    """
    from scipy.optimize import minimize_scalar
    
    def nll(T):
        scaled_std = np.maximum(y_pred_std * T, 1e-6)
        return (0.5 * np.log(2 * np.pi * scaled_std**2) + 
                0.5 * ((y_true - y_pred_mean)**2 / scaled_std**2)).mean()
    
    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return result.x


def save_calibration_metadata(filepath, model_name, rep, sigma_noise, iteration, 
                              n_train, n_cal, n_val, n_test, alpha_list):
    """Save calibration set size and split information"""
    metadata_dir = os.path.join(os.path.dirname(filepath), "conformal_metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    
    metadata_df = pd.DataFrame({
        'model_name': [model_name],
        'rep': [rep],
        'sigma_noise': [sigma_noise],
        'iteration': [iteration],
        'n_train': [n_train],
        'n_calibration': [n_cal],
        'n_validation': [n_val],
        'n_test': [n_test],
        'cal_pct_of_total': [n_cal / (n_train + n_cal + n_val + n_test) * 100],
        'alphas_tested': [str(alpha_list)]
    })
    
    filename = f"calibration_metadata_{model_name}_{rep}_sigma{sigma_noise}_iter{iteration}.csv"
    filepath_full = os.path.join(metadata_dir, filename)
    metadata_df.to_csv(filepath_full, index=False)

"""
Graph Utilities for Molecular Property Prediction

Converts SMILES to PyTorch Geometric Data objects with proper features.
"""

def smiles_to_graph(smiles, y_value=None):
    """
    Convert SMILES to PyTorch Geometric Data object.
    
    Args:
        smiles: SMILES string
        y_value: Optional target value
        
    Returns:
        torch_geometric.data.Data or None if invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    
    # Node features: [atomic_num, degree, formal_charge, is_aromatic, hybridization, num_h]
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            int(atom.GetIsAromatic()),
            int(atom.GetHybridization()),
            atom.GetTotalNumHs()
        ])
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Edge indices
    edge_index = []
    edge_attr = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Bond features: [bond_type, is_conjugated, is_in_ring]
        bond_features = [
            float(bond.GetBondTypeAsDouble()),
            int(bond.GetIsConjugated()),
            int(bond.IsInRing())
        ]
        
        # Add both directions (undirected graph)
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 3), dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if y_value is not None:
        data.y = torch.tensor([y_value], dtype=torch.float)
    
    return data


def smiles_list_to_graphs(smiles_list, y_values=None):
    """
    Convert list of SMILES to list of PyG graphs.
    
    Args:
        smiles_list: List of SMILES strings
        y_values: Optional array of target values
        
    Returns:
        List of Data objects (None for invalid SMILES)
    """
    graphs = []
    failed = 0
    
    for i, smiles in enumerate(smiles_list):
        y_val = y_values[i] if y_values is not None else None
        graph = smiles_to_graph(smiles, y_val)
        
        if graph is None:
            failed += 1
        
        graphs.append(graph)
    
    if failed > 0:
        print(f"Warning: {failed}/{len(smiles_list)} molecules failed conversion")
    
    return graphs


def create_graph_loaders(train_graphs, test_graphs, val_graphs, 
                        y_train_noisy, y_test_noisy, y_val_noisy,
                        batch_size=32):
    """
    Create PyG DataLoaders with noisy targets attached.
    
    This properly handles the noisy targets by attaching them to the graphs.
    
    Args:
        train_graphs, test_graphs, val_graphs: Lists of Data objects
        y_train_noisy, y_test_noisy, y_val_noisy: Noisy target arrays
        batch_size: Batch size
        
    Returns:
        train_loader, test_loader, val_loader
    """
    from torch_geometric.loader import DataLoader
    
    # Filter out None graphs and attach noisy targets
    def attach_targets(graphs, y_noisy):
        valid_data = []
        for graph, y_val in zip(graphs, y_noisy):
            if graph is not None:
                graph.y = torch.tensor([y_val], dtype=torch.float)
                valid_data.append(graph)
        return valid_data
    
    train_data = attach_targets(train_graphs, y_train_noisy)
    test_data = attach_targets(test_graphs, y_test_noisy)
    val_data = attach_targets(val_graphs, y_val_noisy)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, val_loader


def extract_graph_features_for_kernel(smiles_list):
    """
    Extract graph structure for kernel computation.
    
    Returns node labels and edge lists suitable for grakel/graph kernels.
    
    Args:
        smiles_list: List of SMILES strings
        
    Returns:
        List of tuples (node_labels_dict, edge_list)
    """
    graphs = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumBonds() == 0:
            graphs.append(None)
            continue
        
        # Node labels: atomic symbol
        node_labels = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}
        
        # Edge list
        edge_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_list.append((i, j))
        
        graphs.append((node_labels, edge_list))
    
    return graphs