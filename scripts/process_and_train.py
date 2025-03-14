import argparse
import os
import os.path as osp
import random
import json
import subprocess
import struct
import warnings
import numpy as np
import torch
import pandas as pd
import csv
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdFingerprintGenerator
from collections import deque
import gc
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
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
from xgboost import XGBRegressor, XGBClassifier
import diskcache
import deepchem as dc
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from torch_geometric.utils import to_scipy_sparse_matrix
import polaris as po
from polaris.hub.client import PolarisHubClient
from scipy.stats import pearsonr
import shap
import lightgbm as lgb
import optuna
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader as TorchDataLoader
from torch.nn.utils import parameters_to_vector as Params2Vec, vector_to_parameters as Vec2Params
import matplotlib.pyplot as plt

import sys
sys.path.append('../models/')
sys.path.append('../preprocessing/')
sys.path.append('../results/')

from qm_models import ModelTrainer, RNNRegressionModel, GRURegressionModel, GIN, GCN, GINCoTeaching, MLPRegressor, MLPClassifier, Gauche, train_epochs, train_epochs_co_teaching, testing, testing_co_teaching, train_mlp, predict_mlp, GATv2, GATv2a, DNNRegressionModel, train_dnn

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
valid_indices_path = os.path.join(data_dir, 'valid_qm9_indices.pth')

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

DELIMITER = b"\x1F"  # ASCII 31 (Unit Separator)
NEWLINE = b"\n"
MULTIPLE_SMILES_REPS = 3

# Note: only run multiple_smiles on it's own, otherwise triplicate entries for every single entry
# TODO: potentially fix this logic

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

bit_vectors = ['ecfp4', 'mpnn', 'sns']
graph_models = ['gin', 'gcn', 'ginct', 'gauche_graph', 'gin2d', 'gtat']

# Initialize the cache
cache = diskcache.Cache('./smiles_cache')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make sure everything above function definitions is properly formatted
# TODO: make sure val is working for hyperparameter tuning
# TODO: reformat import statements

# TODO: add BNN 

def parse_arguments():
    parser = argparse.ArgumentParser(description="Framework for running QSAR/QSPR prediction models")
    parser.add_argument("-d", "--dataset", type=str, default='QM9', help="Dataset to run experiments on (default is QM9)")
    parser.add_argument("-t", "--target", type=str, default="homo_lumo_gap", help="Target property to predict")
    parser.add_argument("-m", "--models", nargs='*', help="Model(s) to use for prediction", required=True)
    parser.add_argument("-r", "--molecular_representations", nargs='*', help="Molecular representation as a list of strings", required=True)
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default is 42)")
    parser.add_argument("-n", "--sample-size", type=int, default=1000, help="Sample size per iteration (default is 1000)")
    parser.add_argument("-b", "--bootstrapping", type=int, default=1, help="Bootstrapping iterations (default is 1 ie. no bootstrapping)")
    parser.add_argument("--sigma", nargs='*', default=[0.0], help="Standard deviation(s) of artificially added Gaussian noise (default is None)")
    parser.add_argument("--distribution", type=str, default='gaussian', help="Distribution of artificial noise (default is Gaussian)")
    parser.add_argument("--tuning", type=bool, default=False, help="Hyperparameter tuning (default is False)")
    parser.add_argument("--kernel", type=str, default="tanimoto", help="Specify the kernel for certain models (Gaussian Process)")
    parser.add_argument("-k", "--k_domains", type=int, default=1, help="Number of domains for clustering (default is 1)")
    parser.add_argument("-s", "--split", type=str, default="random", help="Method for splitting data (default is random)")
    parser.add_argument("-c", "--clustering_method", type=str, default="Agglomerative", help="Method to cluster the chemical domain (default is Agglomerative)")
    parser.add_argument("--max_vocab", type=int, default=30, help="Max vocab length of SMILES OHE generation (default is 30)")
    parser.add_argument("--custom_model", type=str, default=None, help="Filepath to custom PyTorch model in .pt file")
    parser.add_argument("--metadata_file", type=str, default=None, help="Filepath to custom model's metadata ie. hyperparameters")
    parser.add_argument("-f", "--filepath", type=str, default='../results/test.csv', help="Filepath to save raw results in csv (default is None)")
    parser.add_argument("--logging", type=bool, default=False, help="Extra logging to check individual entries in mmap files (default is False)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training grpah-based models (default is 15)")
    parser.add_argument("--clean-smiles", type=bool, default=False, help="Clean the SMILES string (default is False)")
    parser.add_argument("--shap", type=bool, default=False, help="Calculate SHAP values for relevant tree-based models (default is False)")
    parser.add_argument("--loss-landscape", type=bool, default=False, help="Plot loss landscape (default is False)")
    return parser.parse_args()

# TODO: save the original QM9 index
def write_to_mmap(
    smiles_isomeric,
    smiles_canonical,
    randomized_smiles,
    property_value,
    category,
    files,
    molecular_representations,
    k_domains,
    sns_fp,
    max_vocab,
):
    entry = b""

    smiles_isomeric_binary = smiles_isomeric.encode('utf-8') + DELIMITER
    entry += smiles_isomeric_binary

    smiles_canonical_binary = smiles_canonical.encode('utf-8') + DELIMITER
    entry += smiles_canonical_binary

    property_value_binary = struct.pack('f', property_value) + DELIMITER
    entry += property_value_binary  # Append delimiter separately

    if "randomized_smiles" in molecular_representations:
        randomized_smiles_binary = randomized_smiles.encode('utf-8') + DELIMITER
        entry += randomized_smiles_binary

    if "sns" in molecular_representations:
        if sns_fp is not None:
            sns_fp = sns_fp.astype(np.uint8).tolist()
            sns_fp_binary = struct.pack('16B', *sns_fp[:16]) + DELIMITER
            entry += sns_fp_binary
        else:
            return

    files[category].write(entry)
    files[category].flush()

# targets: BDR4, HSA, sEH
# TODO: need to normalize data
def load_and_split_polaris(args, files):
    dataset_name = "BELKA"

    if dataset_name != "BELKA":
        raise ValueError("Invalid dataset name")

    # Define the binding target
    proteins = {'BDR4': 'binds_BRD4', 'HSA': 'binds_HSA', 'sEH': 'binds_sEH'}
    binding_target = proteins[args.target]

    # Load dataset from PolarisHub
    dataset = po.load_dataset("leash-bio/BELKA-v1")
    dataset_size = dataset.size()[0]

    # Select `args.sample_size` random indices
    N = args.sample_size
    random_indices = np.random.choice(dataset_size, N, replace=False)

    # Allocate space-efficient storage for molecules & labels
    smiles_arr = np.empty(N, dtype="U128")
    target_arr = np.zeros(N, dtype=np.float32)

    # Load SMILES & binding data into NumPy arrays
    for i, idx in enumerate(random_indices):
        smiles_str = dataset.get_data(idx, "molecule_smiles")
        if args.clean_smiles:
            smiles_arr[i] = smiles_arr[i].replace("[Dy]", "")
        target_arr[i] = dataset.get_data(idx, binding_target)

    if args.split == 'random':
        train_index = int(args.sample_size * 0.8)
        test_index = train_index + int(args.sample_size * 0.1)
        val_index = test_index + int(args.sample_size * 0.1)
        train_idx = list(range(train_index))
        val_idx = list(range(train_index, test_index))
        test_idx = list(range(test_index, val_index))

    elif args.split == 'scaffold':
        Xs = np.array(target_arr).reshape(-1, 1)
        dataset = dc.data.DiskDataset.from_numpy(X=Xs,ids=smiles_arr)

        splitter = dc.splits.ScaffoldSplitter()
            
        train_idx, val_idx, test_idx = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    else:
        raise ValueError("Invalid split type")

    mols_train = deque()

    if 'sns' in args.molecular_representations:
        for index, smiles in enumerate(smiles_arr[:args.sample_size]):
            if index in train_idx:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mols_train.append(mol)
        ecfp_featuriser = create_sort_and_slice_ecfp_featuriser(mols_train = mols_train, 
                                                               max_radius = 2, 
                                                               pharm_atom_invs = False, 
                                                               bond_invs = True, 
                                                               chirality = False, 
                                                               sub_counts = True, 
                                                               vec_dimension = 1024,
                                                               print_train_set_info = args.logging)

    for index, smiles_isomeric in enumerate(smiles_arr[:args.sample_size]):
        smiles_canonical = None

        category = "excluded"
        if index in train_idx:
            category = "train"
        elif index in test_idx:
            category = "test"
        elif index in val_idx:
            category = "val"

        smiles_randomized = None
        if smiles_isomeric in cache and not 'randomized_smiles' in args.molecular_representations:
            smiles_canonical = cache[smiles_isomeric]
            # TODO: potentially keep randomized SMILES in the same cache if they look promising (smiles_isomeric, smiles_randomized)
        else:
            # Generate canonical SMILES and store it in cache
            mol = Chem.MolFromSmiles(smiles_isomeric)
            if not mol:
                cache[smiles_isomeric] = None
                continue
            if 'randomized_smiles' in args.molecular_representations:
                smiles_randomized = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)
            smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
            cache[smiles_isomeric] = smiles_canonical

        sns_fp = None
        if 'sns' in args.molecular_representations:
            if index in train_idx:
                mol = mols_train.popleft()
            if not mol: 
                mol = Chem.MolFromSmiles(smiles_isomeric)
            sns_fp = ecfp_featuriser(mol)
        
        if smiles_canonical and not (category == "excluded"):
            # TODO: don't call for graphs
            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, target_arr[index], category, files, args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)
    if 'sns' in args.molecular_representations:
        del mols_train

    del smiles_arr, target_arr
    gc.collect()

    return train_idx, test_idx, val_idx

def load_qm9(target):
    qm9 = QM9(root=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9'))

    # Filter out molecules that cannot be processed by RDKit
    valid_indices_tensor = torch.load(valid_indices_path)
    qm9 = qm9.index_select(valid_indices_tensor)

    # Isolate a single regression target
    y_target = pd.DataFrame(qm9.data.y.numpy())
    property_index = properties[target]
    qm9.data.y = torch.Tensor(y_target[property_index])

    return qm9

# TODO: somewhere in here record the indices so they can be used later
def split_qm9(qm9, args, files):

    # Shuffle with random seed
    indices = torch.randperm(len(qm9))
    qm9 = qm9.index_select(indices)

    if args.split == 'random':
        qm9 = qm9.shuffle()
        train_index = int(args.sample_size * 0.8)
        test_index = train_index + int(args.sample_size * 0.1)
        val_index = test_index + int(args.sample_size * 0.1)
        train_idx = list(range(train_index))
        val_idx = list(range(train_index, test_index))
        test_idx = list(range(test_index, val_index))

    elif args.split == 'scaffold':
        qm9_smiles = [data.smiles for data in qm9[:args.sample_size]]
        np.array([data.y.item() for data in qm9[:args.sample_size]]).reshape(-1, 1)
        dataset = dc.data.DiskDataset.from_numpy(X=Xs,ids=qm9_smiles)

        splitter = dc.splits.ScaffoldSplitter()
            
        train_idx, val_idx, test_idx = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    else:
        raise ValueError("Invalid split type")

    # Normalize the data
    data_mean = qm9.data.y[train_idx].mean()
    data_std = qm9.data.y[train_idx].std()
    qm9.data.y = (qm9.data.y - data_mean) / data_std

    mols_train = deque()

    ecfp_featuriser = None
    if 'sns' in args.molecular_representations:
        for index, data in enumerate(qm9[:args.sample_size]):
            if index in train_idx:
                mols_train.append(Chem.MolFromSmiles(data.smiles))
        ecfp_featuriser = create_sort_and_slice_ecfp_featuriser(mols_train = mols_train, 
                                                               max_radius = 2, 
                                                               pharm_atom_invs = False, 
                                                               bond_invs = True, 
                                                               chirality = False, 
                                                               sub_counts = True, 
                                                               vec_dimension = 1024, 
                                                               print_train_set_info = args.logging)


    for index, data in enumerate(qm9[:args.sample_size]):
        smiles_isomeric = data.smiles
        smiles_canonical = None
        mol = None

        category = "excluded"
        if index in train_idx:
            category = "train"
        elif index in test_idx:
            category = "test"
        elif index in val_idx:
            category = "val"

        smiles_reps = 1
        if 'multiple_smiles' in args.molecular_representations:
            smiles_reps = MULTIPLE_SMILES_REPS

        # TODO: only do this if not a graph
        randomized_smiles = []
        if smiles_isomeric in cache and not 'randomized_smiles' in args.molecular_representations:
            smiles_canonical = cache[smiles_isomeric]
            # TODO: potentially keep randomized SMILES in the same cache if they look promising (smiles_isomeric, smiles_randomized)
        else:
            # Generate canonical SMILES and store it in cache
            mol = Chem.MolFromSmiles(smiles_isomeric)
            if not mol:
                cache[smiles_isomeric] = None
                continue
            if 'multiple_smiles' in args.molecular_representations or 'randomized_smiles' in args.molecular_representations:
                for _ in range(smiles_reps):
                    randomized_smiles.append(Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True))
            smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
            cache[smiles_isomeric] = smiles_canonical

        sns_fp = None
        if 'sns' in args.molecular_representations:
            if index in train_idx:
                mol = mols_train.popleft()
            if not mol: 
                mol = Chem.MolFromSmiles(smiles_isomeric)
            sns_fp = ecfp_featuriser(mol)

        if smiles_canonical and not (category == "excluded"):
            for smiles_i in range(smiles_reps):
                randomized_entry = None
                if randomized_smiles != []:
                    randomized_entry = randomized_smiles[smiles_i]

                write_to_mmap(smiles_isomeric, smiles_canonical, randomized_entry, data.y.item(), category, files, args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)

    if 'sns' in args.molecular_representations:
        del mols_train


    return train_idx, test_idx, val_idx

def create_sort_and_slice_ecfp_featuriser(mols_train, 
                                          max_radius = 2, 
                                          pharm_atom_invs = False, 
                                          bond_invs = True, 
                                          chirality = False, 
                                          sub_counts = True, 
                                          vec_dimension = 1024, 
                                          break_ties_with = lambda sub_id: sub_id, 
                                          print_train_set_info = True):
    # Create a function sub_id_enumerator that maps a mol object to a dictionary whose keys are the integer substructure identifiers in mol and whose values are the associated substructure counts (i.e., how often each substructure appears in mol)
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius = max_radius,
                                                                 atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if pharm_atom_invs == True else rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True),
                                                                 useBondTypes = bond_invs,
                                                                 includeChirality = chirality)
    
    sub_id_enumerator = lambda mol: morgan_generator.GetSparseCountFingerprint(mol).GetNonzeroElements() if mol is not None else {}
    
    # Construct dictionary that maps each integer substructure identifier sub_id in mols_train to its associated prevalence (i.e., to the total number of compounds in mols_train that contain sub_id at least once)
    sub_ids_to_prevs_dict = {}
    for mol in mols_train:
        for sub_id in sub_id_enumerator(mol).keys():
            sub_ids_to_prevs_dict[sub_id] = sub_ids_to_prevs_dict.get(sub_id, 0) + 1

    # Create list of integer substructure identifiers sorted by prevalence in mols_train
    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict, key = lambda sub_id: (sub_ids_to_prevs_dict[sub_id], break_ties_with(sub_id)), reverse = True)
    
    # Create auxiliary function that generates standard unit vectors in NumPy
    def standard_unit_vector(dim, k):
        
        vec = np.zeros(dim, dtype = int)
        vec[k] = 1
        
        return vec
    
    # Create one-hot encoder for the first vec_dimension substructure identifiers in sub_ids_sorted_list; all other substructure identifiers are mapped to a vector of 0s
    def sub_id_one_hot_encoder(sub_id):
        
        return standard_unit_vector(vec_dimension, sub_ids_sorted_list.index(sub_id)) if sub_id in sub_ids_sorted_list[0: vec_dimension] else np.zeros(vec_dimension)
    
    # Create a function ecfp_featuriser that maps RDKit mol objects to vectorial ECFPs via a Sort & Slice substructure pooling operator trained on mols_train
    def ecfp_featuriser(mol):

        # create list of integer substructure identifiers contained in input mol object (multiplied by how often they are structurally contained in mol if sub_counts = True)
        if sub_counts == True:
            sub_id_list = [sub_idd for (sub_id, count) in sub_id_enumerator(mol).items() for sub_idd in [sub_id]*count]
        else:
            sub_id_list = list(sub_id_enumerator(mol).keys())
        
        # create molecule-wide vectorial representation by summing up one-hot encoded substructure identifiers
        ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis = 0)
    
        return ecfp_vector
    
    # Print information on training set
    if print_train_set_info == True:
        print(f"Number of compounds in molecular training set: {len(mols_train)}")
        print(f"Number of unique circular substructures with the specified parameters in molecular training set: {len(sub_ids_to_prevs_dict)}")

    return ecfp_featuriser

def calculate_classification_metrics(y_test, prediction, logging=False):
    accuracy = accuracy_score(y_test, y_test_preds)
    roc_auc = roc_auc_score(y_test, y_test_probs[:, 1])  # Assuming binary classification
    precision = precision_score(y_test, y_test_preds, average="weighted")
    recall = recall_score(y_test, y_test_preds, average="weighted")
    f1 = f1_score(y_test, y_test_preds, average="weighted")
    pr_auc = average_precision_score(
        y_test, y_test_probs[:, 1], average="weighted"
    )
    # TODO: pearson?

    # Optionally log the metrics
    if logging:
        print("Accuracy:", accuracy)
        print("ROC AUC:", roc_auc)
        print("Precision", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("PR AUC:", pr_auc)

    return [accuracy, roc_auc, precision, recall, f1, pr_auc]

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

# TOOD: this doesn't work with non-gaussian distributions at the moment, nor do a lot of things that rely solely on sigma
def save_results(filepath, s, iteration, model, rep, n, r2, mae, corr):
    """
    Save results to a CSV file specified by args.filepath.
    """
    if filepath:
        file_exists = os.path.isfile(filepath)

        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if the file is new
            if not file_exists:
                writer.writerow(["sigma", "iteration", "model", "rep", "sample_size", "r2_score", "mae", "pearson_corr"])
            
            # Save the results
            writer.writerow([s, iteration, model, rep, n, r2, mae, corr])

def save_shap_values(shap_values, feature_names, x_test, filepath, model, iteration, rep):
    """
    Save SHAP values to a CSV file or NumPy file for large datasets.
    """
    shap_filepath = filepath.replace('.csv', '_shap.csv')  # Store SHAP values separately

    if shap_values is not None:
        shap_df = pd.DataFrame(shap_values, columns=feature_names)
        shap_df.insert(0, "Sample_Index", np.arange(len(shap_values)))  # Track sample index
        shap_df.insert(1, "Model", model)
        shap_df.insert(2, "Iteration", iteration)
        shap_df.insert(3, "Rep", rep)

        # Save to CSV (appending if file exists)
        if os.path.exists(shap_filepath):
            shap_df.to_csv(shap_filepath, mode='a', header=False, index=False)
        else:
            shap_df.to_csv(shap_filepath, index=False)

    # Save as NumPy file for efficiency
    npy_filepath = filepath.replace('.csv', '_shap.npy')
    np.save(npy_filepath, shap_values)

def load_custom_model(model_path):
    """
    Loads a PyTorch model from a .pt file.
    If saved using state_dict, assumes architecture is pre-defined.
    """
    model = torch.load(model_path, map_location=torch.device('cpu'))
    if isinstance(model, dict) and "state_dict" in model:  # Handle state_dict models
        model_class = get_predefined_model_class()  # Ensure user-defined model class is available
        model_instance = model_class()
        model_instance.load_state_dict(model["state_dict"])
        model_instance.eval()
        return model_instance
    model.eval()
    return model

def run_gauche(args, x_train, x_test, y_train, y_test, sigma, iteration, rep):
    """
    Run Gauche model with kernel selection
    """
    # Convert input data to tensors
    x_train_tensor = torch.from_numpy(x_train).double()
    x_test_tensor = torch.from_numpy(x_test).double()
    y_train_tensor = torch.from_numpy(y_train).double()

    # Initialize the kernel wrapped with ScaleKernel
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.ScaleKernel(kernel_map[kernel_name]())
    model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel)

    # Fit GP model using BoTorch
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    fit_gpytorch_model(mll)

    # Make predictions with the trained GP model
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(x_test_tensor)
        y_pred = preds.mean.numpy()
        pred_vars = preds.variance.numpy()

    # Handle logging and domain-specific evaluations
    if args.distribution in ["domain_mpnn", "domain_tanimoto"]:
        calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, args.dataset)

    # Compute metrics
    if args.dataset == 'QM9':
        metrics = calculate_regression_metrics(y_test, y_pred, logging=True)
    else:
        metrics = calculate_classification_metrics(y_test, y_pred, logging=True)

    save_results(args.filepath, sigma, iteration, "gauche", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3]  # Negative MSE for optimization

# TODO: need to call save_results in here
def run_custom(x_train, x_test, y_train, logging, sigma, dataset, model_path, metadata_path=None):
    """
    Runs a PyTorch model stored in a .pt file, trains it on x_train/y_train, and evaluates it on x_test.
    Handles hyperparameter tuning if a metadata file is provided.
    """

    # Load model
    model = load_custom_model(model_path)

    # Convert inputs to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    # Load hyperparameters if available
    hyperparams = get_custom_hyperparameter_bounds(metadata_path) if metadata_path else {}

    # Define optimizer and loss function
    learning_rate = hyperparams.get("learning_rate", [0.001, 0.001])[0]  # Default 0.001 if missing
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()  # Adjust if classification

    # Train model
    model.train()
    for _ in range(10):  # Change 10 to desired number of epochs
        optimizer.zero_grad()
        y_pred_train = model(x_train_tensor).squeeze()
        loss = loss_fn(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        y_pred = model(x_test_tensor).squeeze().cpu().numpy()  # Ensure proper shape & detach from graph

    # Optionally log results
    if logging:
        print("Results for custom model:", model_path)
        print("Sigma:", sigma)

    logging = True
    if distribution == "domain_mpnn" or distribution == "domain_tanimoto":
        calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, dataset)
        logging = False

    if dataset == 'QM9':
        metrics = calculate_regression_metrics(y_test, y_pred, logging=logging)
    else:
        metrics = calculate_classification_metrics(y_test, y_pred, logging=logging)

    save_results(args.filepath, sigma, iteration, "custom", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3]  # Return negative MSE

# Sample hyperparameter file
# {
#     "learning_rate": [0.0001, 0.01],
#     "batch_size": [8, 64],
#     "dropout": [0.1, 0.5]
# }
def get_custom_hyperparameter_bounds(metadata_path):
    """
    Reads hyperparameter tuning bounds from a JSON file.
    Assumes the JSON file contains a dictionary with parameter names and their bounds.
    """
    try:
        with open(metadata_path, 'r') as f:
            hyperparams = json.load(f)
        return hyperparams
    except FileNotFoundError:
        raise ValueError("Metadata file not found. Please specify a valid path.")
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format in metadata file.")

def parse_mmap(mmap_file, entry_count, rep, molecular_representations, k_domains, logging):
    x_data = []
    y_data = []

    for i in range(entry_count):
        line = mmap_file.readline().strip(NEWLINE)
        try:
            feature_vector = []

            fields = line.split(DELIMITER)  # Split by delimiter

            field_idx = 0  # Track current field position

            # Read required fields
            isomeric_smiles = fields[field_idx].decode("utf-8")
            field_idx += 1

            canonical_smiles = fields[field_idx].decode("utf-8")
            field_idx += 1

            target_value = struct.unpack("f", fields[field_idx])[0]
            field_idx += 1

            # Read optional fields
            randomized_smiles = None
            if "randomized_smiles" in molecular_representations:
                randomized_smiles = fields[field_idx].decode("utf-8")
                field_idx += 1
                if logging: 
                    print(f"randomized_smiles: {randomized_smiles}")

            domain_label = None
            if "k_domains" in molecular_representations:
                domain_label = struct.unpack("B", fields[field_idx])[0]
                field_idx += 1
                if logging:
                    print(f"domain_label: {domain_label}")

            # TODO: you broke something in here
            sns_fp = None
            if "sns" in molecular_representations:
                sns_fp = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8), bitorder='little')

                field_idx += 1
                if logging:
                    print(f"sns_fp: {sns_fp}")
            
            processed_target = struct.unpack("f", fields[field_idx])[0]
            field_idx += 1
            if logging:
                print(f"processed_target: {processed_target}")

            # Check for NaN
            if processed_target != processed_target:
                continue

            # Finish processing SNS after reading the target value
            if "sns" == rep:
                y_data.append(processed_target)
                feature_vector.append(sns_fp)
                x_data.append(np.concatenate([f for f in feature_vector if f is not None]))
                continue
            
            if "graph" == rep:
                x_data.append(i)
                y_data.append(processed_target)
                continue

            smiles_ohe = None
            if "smiles" in molecular_representations:
                # smiles_ohe = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8))
                smiles_packed = np.frombuffer(fields[field_idx], dtype=np.uint8)
                smiles_ohe = np.unpackbits(smiles_packed, bitorder='little')
                field_idx += 1
                if logging: 
                    print(f"smiles: {smiles_ohe}")

                if "smiles" == rep:
                    x_data.append(smiles_ohe)
                    y_data.append(processed_target)
                    continue

            randomized_smiles_ohe = None
            if "randomized_smiles" in molecular_representations:
                # randomized_smiles_ohe = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8))
                randomized_smiles_packed = np.frombuffer(fields[field_idx], dtype=np.uint8)
                randomized_smiles_ohe = np.unpackbits(randomized_smiles_packed, bitorder='little')
                field_idx += 1
                if logging:
                    print(f"randomized_smiles: {randomized_smiles}")

                if "randomized_smiles" == rep:
                    x_data.append(randomized_smiles_ohe)
                    y_data.append(processed_target)
                    continue

            ecfp4 = None
            if "ecfp4" in molecular_representations:
                ecfp4 = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8), bitorder='little')
                field_idx += 1
                if logging:
                    print(f"ecfp4: {ecfp4}")

                if "ecfp4" == rep:
                    # Ensure correct size before unpacking
                    if ecfp4.size == 256: 
                        u64_array = np.frombuffer(ecfp4, dtype=np.uint64)
                        ecfp4 = np.unpackbits(ecfp4, bitorder='little')[:2048]
                    else:
                        # print(f"Warning: ECFP4 fingerprint has unexpected size {ecfp4.size} at index {field_idx}")
                        continue

                    feature_vector.append(ecfp4)
                    x_data.append(np.concatenate([f for f in feature_vector if f is not None]))
                    y_data.append(processed_target)
                    continue

        except Exception as e:
            # TODO: look into why there's so many of these
            # print(f"Skipping malformed line: {line}")
            continue

    # TODO: may need a try/catch here with sufficient error messaging
    if rep != 'graph':
        x_data = np.vstack(x_data).astype(np.uint8)
    y_data = np.array(y_data, dtype=np.float32)

    return x_data, y_data

# TODO: save plots to appropriate place
def loss_landscape(model, model_type, rep, s, x_test_tensor, y_test_tensor, device, iteration_seed, loss_landscape_flag):
    """
    Computes and visualizes 1D and 2D loss landscapes for a given neural network.

    Parameters:
    - model: PyTorch model (DNN or any NN)
    - x_test_tensor: Input test tensor
    - y_test_tensor: Ground truth labels for test set
    - device: Device (CPU/GPU) to run computations
    - iteration_seed: Identifier for saving plots
    - loss_landscape_flag: Boolean flag to enable/disable landscape computation
    """
    if not loss_landscape_flag:
        return  # Exit if landscape computation is disabled

    print("Computing loss landscape...")

    model_save_path = f"trained_dnn_{iteration_seed}.pt"
    torch.save(model.state_dict(), model_save_path)

    # Recreate the model with known architecture parameters
    if isinstance(model, DNNRegressionModel):
        infer_net = DNNRegressionModel(input_size=model.fc1.in_features,
                                       hidden_size1=model.fc1.out_features,
                                       hidden_size2=model.fc2.out_features).to(device)
    elif isinstance(model, DNNClassificationModel):
        infer_net = DNNClassificationModel(input_size=model.fc1.in_features,
                                           hidden_size1=model.fc1.out_features,
                                           hidden_size2=model.fc2.out_features,
                                           num_classes=model.fc3.out_features).to(device)
    else:
        raise ValueError("Unsupported model type for loss landscape analysis")

    # Load trained weights
    infer_net.load_state_dict(torch.load(model_save_path))

    infer_net.load_state_dict(torch.load(model_save_path))

    # Convert parameters to vectors
    theta_ast = Params2Vec(model.parameters()).detach()
    theta = Params2Vec(infer_net.parameters()).detach()

    loss_fn = torch.nn.MSELoss() if isinstance(model, DNNRegressionModel) else torch.nn.CrossEntropyLoss()

    # 1D Loss Landscape
    alphas = torch.linspace(-20, 20, 40)
    losses_1d = []

    for alpha in alphas:
        Vec2Params(alpha * theta_ast + (1 - alpha) * theta, infer_net.parameters())
        infer_net.eval()
        with torch.no_grad():
            y_pred = infer_net(x_test_tensor)
            loss = loss_fn(y_pred, y_test_tensor).item()
            losses_1d.append(loss)

    # 2D Loss Landscape
    x_range = torch.linspace(-20, 20, 20)
    y_range = torch.linspace(-20, 20, 20)
    alpha, beta = torch.meshgrid(x_range, y_range, indexing="ij")

    def tau_2d(alpha, beta, theta_ast):
        return alpha * theta_ast[:, None, None] + beta * alpha * theta_ast[:, None, None]

    space = tau_2d(alpha, beta, theta_ast)
    losses_2d = torch.empty_like(space[0, :, :])

    for a, _ in enumerate(x_range):
        print(f'Processing alpha = {a}')
        for b, _ in enumerate(y_range):
            Vec2Params(space[:, a, b] + theta_ast, infer_net.parameters())
            infer_net.eval()
            with torch.no_grad():
                y_pred = infer_net(x_test_tensor)
                losses_2d[a, b] = loss_fn(y_pred, y_test_tensor).item()

    # Plot 1D loss landscape
    plt.figure(figsize=(8, 6))
    plt.plot(alphas.numpy(), losses_1d)
    plt.xlabel("Alpha")
    plt.ylabel("Loss")
    plt.title("1D Loss Landscape")
    plt.grid()
    plt.savefig(f"../resuls/loss_landscape_1d_{model_type}_{rep}_{s}.png")
    plt.close()

    # Plot 2D loss contour
    plt.figure(figsize=(8, 6))
    plt.contourf(alpha.numpy(), beta.numpy(), losses_2d.numpy(), levels=50, cmap="viridis")
    plt.colorbar(label="Loss")
    plt.xlabel("Alpha")
    plt.ylabel("Beta")
    plt.title("2D Loss Contour")
    plt.savefig(f"../resuls/loss_landscape_2d_{model_type}_{rep}_{s}.png")
    plt.close()

    print("Loss landscape computation complete!")


def run_model(x_train, y_train, x_test, y_test, model_type, args, iteration_seed, rep, iteration, s):
    def black_box_function(trial=None):
        params = {}

        if model_type == 'rf':
            params = {}
            if args.tuning:
                params['max_depth'] = trial.suggest_int('max_depth', 10, 200)
                params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 1.0, None])  # Allow no max feature restriction
                params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 50)
                params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
                params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
                params['bootstrap'] = trial.suggest_categorical('bootstrap', [True, False])  # Typically tuned
            if args.dataset == 'QM9':
                model = RandomForestRegressor(random_state=iteration_seed, **params)
            else:
                params['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy'])  # Only for classification
                model = RandomForestClassifier(random_state=iteration_seed, **params)

        elif model_type == 'svm':
            params = {}
            if args.tuning:
                params['C'] = trial.suggest_float('C', 0.1, 100, log=True)
                params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto'])  # Allow automatic gamma scaling
                params['kernel'] = trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid'])

                if params['kernel'] == 'poly':  
                    params['degree'] = trial.suggest_int('degree', 2, 5)  # Poly kernel degree (2-5 common in literature)
                    params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)  # Bias term for 'poly' and 'sigmoid'
                
                if params['kernel'] == 'sigmoid':
                    params['coef0'] = trial.suggest_float('coef0', 0.0, 10.0)  

            if args.dataset == 'QM9':
                model = SVR(**params)
            else:
                model = SVC(**params)

        elif model_type == 'xgboost':
            params = {}
            if args.tuning:
                params['max_depth'] = trial.suggest_int('max_depth', 2, 20)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)
                params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)  # Feature sampling per tree
                params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.5, 1.0)  # Feature sampling per level
                params['min_child_weight'] = trial.suggest_int('min_child_weight', 1, 10)  # Minimum samples for a split
                params['gamma'] = trial.suggest_float('gamma', 0, 5.0)  # L1 regularization for pruning
                params['reg_alpha'] = trial.suggest_float('reg_alpha', 0.0, 1.0)  # L1 regularization
                params['reg_lambda'] = trial.suggest_float('reg_lambda', 0.0, 1.0)  # L2 regularization

            if args.dataset == 'QM9':
                model = XGBRegressor(random_state=iteration_seed, **params)
            else:
                model = XGBClassifier(random_state=iteration_seed, **params)

        elif model_type == 'gauche':
            params = {}
            if args.tuning:
                kernel_name = trial.suggest_categorical('kernel', [
                    'Tanimoto', 'BraunBlanquet', 'Dice', 'Faith', 'Forbes',
                    'InnerProduct', 'Intersection', 'MinMax', 'Otsuka',
                    'Rand', 'RogersTanimoto', 'RussellRao', 'Sogenfrei', 'SokalSneath'
                ])

                # Kernel mapping
                kernel_map = {
                    'Tanimoto': gauche.kernels.fingerprint_kernels.tanimoto_kernel.TanimotoKernel,
                    'BraunBlanquet': gauche.kernels.fingerprint_kernels.braun_blanquet_kernel.BraunBlanquetKernel,
                    'Dice': gauche.kernels.fingerprint_kernels.dice_kernel.DiceKernel,
                    'Faith': gauche.kernels.fingerprint_kernels.faith_kernel.FaithKernel,
                    'Forbes': gauche.kernels.fingerprint_kernels.forbes_kernel.ForbesKernel,
                    'InnerProduct': gauche.kernels.fingerprint_kernels.inner_product_kernel.InnerProductKernel,
                    'Intersection': gauche.kernels.fingerprint_kernels.intersection_kernel.IntersectionKernel,
                    'MinMax': gauche.kernels.fingerprint_kernels.minmax_kernel.MinMaxKernel,
                    'Otsuka': gauche.kernels.fingerprint_kernels.otsuka_kernel.OtsukaKernel,
                    'Rand': gauche.kernels.fingerprint_kernels.rand_kernel.RandKernel,
                    'RogersTanimoto': gauche.kernels.fingerprint_kernels.rogers_tanimoto_kernel.RogersTanimotoKernel,
                    'RussellRao': gauche.kernels.fingerprint_kernels.russell_rao_kernel.RussellRaoKernel,
                    'Sogenfrei': gauche.kernels.fingerprint_kernels.sogenfrei_kernel.SogenfreiKernel,
                    'SokalSneath': gauche.kernels.fingerprint_kernels.sokal_sneath_kernel.SokalSneathKernel
                }
            return run_gauche(args, x_train, x_test, y_train, y_test, s, iteration, rep)

        # TODO: DNN for classification
        # TODO: reduce number of print statements, find correct amount of epochs
        # Original source used 100 epochs
        elif model_type == "dnn":
            if args.tuning:
                params['hidden_size1'] = trial.suggest_categorical('hidden_size1', [32, 64, 128, 256, 512, 1024, 2048, 4096])
                params['hidden_size2'] = trial.suggest_categorical('hidden_size2', [32, 64, 128, 256, 512, 1024, 2048, 4096])
                params['activation'] = trial.suggest_categorical('activation', ['relu', 'tanh', 'softmax'])
            else:
                params['hidden_size1'], params['hidden_size2'] = 128, 64  # Default values if no tuning
                params['activation'] = 'relu'

            activation_map = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'softmax': nn.Softmax(dim=1)}
            activation = activation_map[params['activation']]

            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
            y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)
            y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)

            train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)

            if args.dataset == 'QM9':
                model = DNNRegressionModel(input_size=x_train.shape[1], hidden_size1=params['hidden_size1'], hidden_size2=params['hidden_size2'])
            else:
                model = DNNClassificationModel(input_size=x_train.shape[1], hidden_size1=params['hidden_size1'], hidden_size2=params['hidden_size2'])
            
            model.activation = activation
            model.to(device)

            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_dnn(model, train_loader, train_loader, criterion, optimizer, device, args.epochs)

            model.eval()
            with torch.no_grad():
                y_pred_tensor = model(x_test_tensor).cpu().numpy()
            y_pred = y_pred_tensor.flatten()

                        # TODO: may need to modify save paths and such 
            if args.loss_landscape:
                loss_landscape(model, model_type, rep, s, x_test_tensor, y_test_tensor, device, iteration_seed, args.loss_landscape)

        elif model_type == "lgb":
            params = {}
            if args.tuning:
                params['num_leaves'] = trial.suggest_int('num_leaves', 10, 200)
                params['max_depth'] = trial.suggest_int('max_depth', 2, 20)
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)
                params['subsample'] = trial.suggest_float('subsample', 0.5, 1.0)
                params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.5, 1.0)
                params['n_estimators'] = trial.suggest_int('n_estimators', 10, 2000)
                params['min_child_samples'] = trial.suggest_int('min_child_samples', 1, 50)

            param_dict = {
                'objective': 'regression' if args.dataset == 'QM9' else 'binary',
                'metric': 'r2' if args.dataset == 'QM9' else 'binary_logloss',  # Ensure metric is defined
                'random_state': iteration_seed
            }
            param_dict.update(params)

            # Create LightGBM datasets
            train_data = lgb.Dataset(x_train, label=y_train)

            # Train the model WITHOUT early stopping
            # TODO: does this work with classification
            model = lgb.train(
                param_dict,
                train_data,
                num_boost_round=100
            )

            # Save the trained model
            model.save_model(f"lgb_model_{iteration_seed}.txt")

            # Make predictions
            y_pred = model.predict(x_test)

        elif model_type == "mlp":
            if args.tuning:
                params['hidden_size'] = trial.suggest_categorical('hidden_size', [32, 64, 128, 256, 512, 1024])
                params['num_hidden_layers'] = trial.suggest_int('num_hidden_layers', 1, 5)
                params['dropout_rate'] = trial.suggest_float('dropout_rate', 0.1, 0.5)
                params['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
            else:
                params['hidden_size'], params['num_hidden_layers'], params['dropout_rate'], params['lr'] = 128, 2, 0.2, 0.001

            x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
            x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

            if args.dataset == 'QM9':  # Regression
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(device)
            else:  # Classification
                y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

            train_loader = TorchDataLoader(TensorDataset(x_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
            test_loader = TorchDataLoader(TensorDataset(x_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

            if args.dataset == 'QM9':
                model = MLPRegressor(
                    input_size=x_train.shape[1],
                    hidden_size=params['hidden_size'],
                    num_hidden_layers=params['num_hidden_layers'],
                    dropout_rate=params['dropout_rate']
                )
                criterion = nn.MSELoss()
            else:
                model = MLPClassifier(
                    input_size=x_train.shape[1],
                    hidden_size=params['hidden_size'],
                    num_hidden_layers=params['num_hidden_layers'],
                    num_classes=len(set(y_train)),  # Auto-detect number of classes
                    dropout_rate=params['dropout_rate']
                )
                criterion = nn.CrossEntropyLoss()

            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

            train_mlp(model, train_loader, train_loader, epochs=100, lr=params['lr'])

            # Use predict_mlp instead of manual prediction loop
            y_pred_tensor = predict_mlp(model, test_loader)
            
            if args.dataset != 'QM9':  # Convert probabilities to class predictions for classification
                y_pred = np.argmax(y_pred_tensor, axis=1)
            else:
                y_pred = np.array(y_pred_tensor).flatten()

        elif model_type == 'custom':
            return run_custom_model(x_train, x_test, y_train, False, None, args.dataset, args.distribution)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        if model_type not in ['lgb', 'dnn', 'mlp']:
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

        if args.shap:
            try:
                explainer = None
                shap_values = None
                if model_type in ['rf', 'xgboost', 'lgb']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(x_test)
                if shap_values is not None:
                    save_shap_values(shap_values, [f'feature_{i}' for i in range(x_test.shape[1])], x_test, args.filepath, model_type, iteration, rep)
            except Exception as e:
                print(f"SHAP calculation failed for {model_type}: {e}")

        metrics = calculate_regression_metrics(y_test, y_pred, logging=True)

        save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics[3], metrics[0], metrics[4])

        return metrics[3] if args.dataset == 'QM9' else metrics[0]

    if args.tuning:
        study = optuna.create_study(direction="minimize")
        study.optimize(black_box_function, n_trials=7)  # Adjust `n_trials` as needed

        best_params = study.best_params
        print(f"Best params for {model_type} and {rep} with sigma {s}: {best_params}")

        res = black_box_function(optuna.trial.FixedTrial(best_params))

    elif args.bootstrapping == 1:
        black_box_function()

    else:
        black_box_function()

def run_qm9_graph_model(args, qm9, train_idx, test_idx, val_idx, s):
    for model_type in args.models:
        if model_type == "gin" or model_type == "gin2d":
            model = GIN(dim_h=64)
        elif model_type == "gcn":
            model = GCN(dim_h=128)
        elif model_type == "gin_co_teaching":
            model = GINCoTeaching(dim_h=64)
        # elif model_type == "gauche_graph":
        #     # TODO: need to add label noise for gauche graph
        #     # Potentially just take this whole section out for now
        #     likelihood = gpytorch.likelihoods.GaussianLikelihood()
        #     # TODO: change this
        #     kernel = gpytorch.kernels.RBFKernel  # or whichever kernel you need
        #     kernel_kwargs = {}  # specify any necessary kernel arguments

        #     # Convert graph-structured inputs to custom data class for non-tensorial inputs and convert labels to PyTorch tensors
        #     X_train = NonTensorialInputs(qm9[train_idx].data)
        #     X_test = NonTensorialInputs(qm9[test_idx].data)
        #     # TODO: use noisy_train instead
        #     y_train = torch.tensor(qm9[train_idx].data.y.numpy()).flatten().float()
        #     y_test = torch.tensor(qm9[test_idx].data.y.numpy()).flatten().float()

        #     # Initialize the GaucheGraph model
        #     model = GaucheGraph(X_train, y_train, likelihood, kernel, **kernel_kwargs)

        #     # Define the marginal log likelihood used to optimize the model hyperparameters
        #     mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        #     # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimizer (recommended)
        #     fit_gpytorch_model(mll)

        #     # Get into evaluation (predictive posterior) mode and compute predictions
        #     model.eval()
        #     likelihood.eval()
        #     with torch.no_grad():
        #         f_pred = model(X_test)
        #         y_pred = f_pred.mean
        #         y_var = f_pred.variance

        #     # Optionally log results
        #     if logging:
        #         print("Results for ", model_type, " and ", molecular_representation)

        #     if pred_tracking:
        #         predictions_dict.put((molecular_representation, model_type, (y_test, y_pred)))

        #     logging = True
        #     if args.distribution == "domain_mpnn" or args.distribution == "domain_tanimoto":
        #         calculate_domain_metrics(y_test, y_pred, domain_labels_subset, target_domain)
        #         logging = False
            
        #     metrics = calculate_regression_metrics(y_test, y_pred, logging=logging)

        #     pass

        # else: 
        #     pass

        # Add label noise
        train_set = qm9[train_idx]
        if s > 0:
            # Generate Gaussian noise with mean 0 and standard deviation s
            noise = torch.normal(mean=0, std=s, size=train_set.data.y.shape)
            
            # Add noise to the original labels
            train_set.data.y = train_set.data.y + noise

            # Ensure the tensor is properly formatted
            train_set.data.y = train_set.data.y.to(dtype=torch.float32)

        # TODO: do I need to add noise for val? 

        # datasets into DataLoader
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        test_loader = DataLoader(qm9[test_idx], batch_size=64, shuffle=True)
        val_loader = DataLoader(qm9[val_idx], batch_size=64, shuffle=True)

        test_loss = test_target = test_y = None
        if model_type != "gin_co_teaching":
            train_loss, val_loss, train_target, train_y_target, trained_model = train_epochs(
                args.epochs, model, train_loader, test_loader, "GIN_model.pt"
            )

            test_loss, test_target, test_y = testing(test_loader, trained_model)
        else:

            train_loss, val_loss, train_target, train_y_target, trained_model = train_epochs_co_teaching(
                args.epochs, model, train_loader, test_loader, "GIN_co_teching_model.pt", optimal_co_teaching_hyperparameters['ratio'], optimal_co_teaching_hyperparameters['tolerance'], optimal_co_teaching_hyperparameters['forget_rate']
            )

            test_loss, test_target, test_y = testing_co_teaching(test_loader, trained_model)

        logging = True
        if args.distribution == "domain_mpnn" or args.distribution == "domain_tanimoto":
            calculate_domain_metrics(test_target, test_y, domain_labels_subset, target_domain)
            logging = False

        metrics = calculate_regression_metrics(test_target, test_y, logging=logging)

        save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics[3], metrics[0], metrics[4])

def process_and_run(args, iteration, iteration_seed, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset=None):
    graph_only = True
    for model in args.models:
        if model not in graph_models:
            graph_only = False 

    if 'multiple_smiles' in args.molecular_representations:
        train_count = MULTIPLE_SMILES_REPS * len(train_idx)
        test_count = MULTIPLE_SMILES_REPS * len(test_idx)
    else:
        train_count = len(train_idx)
        test_count = len(test_idx)

    config = {
        'sample_size': args.sample_size,
        'noise': s > 0,
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'max_vocab': args.max_vocab,
        'iteration_seed': iteration_seed,
        'molecular_representations': args.molecular_representations,
        'k_domains': args.k_domains,
        'logging': args.logging,
        'regression': args.dataset == 'QM9',
    }

    with open('config.json', 'w') as f:
        json.dump(config, f)

    print(f"Rust executable path: {rust_executable_path}")

    proc_a = subprocess.Popen(
        [
            rust_executable_path,
            '--seed', str(iteration_seed),
            '--model', "rf",
            '--sigma', str(s),
            '--noise_distribution', args.distribution,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc_a.communicate()

    print(f"Rust stderr: {stderr}")
    print(f"Rust stdout: {stdout}")

    # Read mmap files and train/test models for all molecular representations
    for rep in args.molecular_representations:
        if not graph_only:
            # Reset pointed for each mmap file
            for file in files.values():
                file.seek(0)

            try:
                x_train, y_train = parse_mmap(files["train"], len(train_idx), rep, args.molecular_representations, args.k_domains, logging=args.logging)
                x_test, y_test = parse_mmap(files["test"], len(test_idx), rep, args.molecular_representations, args.k_domains, logging=args.logging)
            except Exception as e:
                print(f"Error with parsing mmap file for {rep} and {model}; more details: {e}")
                continue

        for model in args.models:
            if model not in graph_models:
                # TODO: mlp, remove try to see specific error ADD THIS BACK IN
                # try: 
                print(f"model: {model}")
                run_model(
                    x_train, 
                    y_train, 
                    x_test, 
                    y_test, 
                    model, 
                    args, 
                    iteration_seed,
                    rep,
                    iteration,
                    s,
                )
                # except Exception as e:
                #     print(f"Error with {rep} and {model}; more details: {e}")
            else:
                if args.dataset == 'QM9':
                    run_qm9_graph_model(args, dataset, train_idx, test_idx, val_idx, s)
                else:
                    # TODO: need to convert polaris molecules to 3D and 2D
                    return 

    for file in files.values():
        # Distinction between direct refcount of the mmap object itself, and the number of "view" objects that are pointing to it
        # TODO: check if this is properly closing the mmap file (https://github.com/numpy/numpy/issues/13510)
        # mv = memoryview(file)
        # del mv
        file.close()
        del file
    gc.collect()

def main():
    args = parse_arguments()

    # Prepare for communication with Rust
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"  # Enable Rust backtraces for debugging

    rust_executable_path = os.path.join(base_dir, '../rust/target/release/rust_processor')

    qm9 = None
    if args.dataset == 'QM9':
        qm9 = load_qm9(args.target)
        print("QM9 loaded")

    for s in args.sigma:
        s = float(s)
        print(f"Sigma: {s}")

        for iteration in range(args.bootstrapping):
            # Set seeds
            iteration_seed = (args.random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF  # XOR and mask for 32-bit seed
            random.seed(iteration_seed)
            np.random.seed(iteration_seed)
            torch.manual_seed(iteration_seed)

            files = {
                "train": open('train_' + str(iteration_seed) + '.mmap', 'wb+'),
                "test": open('test_' + str(iteration_seed) + '.mmap', 'wb+'),
                "val": open('val_' + str(iteration_seed) + '.mmap', 'wb+'),
            }

            train_size = int(args.sample_size * 0.8)
            test_size = int(args.sample_size * 0.1)
            val_size = int(args.sample_size * 0.1)

            if args.dataset == 'QM9':
                train_idx, test_idx, val_idx = split_qm9(qm9, args, files)

            else:
                train_idx, test_idx, val_idx = load_and_split_polaris(args, files)

            gc.collect()
            
            target_domain = 1 # TODO: change, this is just a placeholder
            process_and_run(args, iteration, iteration_seed, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset=qm9)

if __name__ == "__main__":
    main()

# TODO: add polaris login to README
# TODO: sanity check to see if val set/epochs can be used for any others
# What's the best practice - if RF doesn't use a val set should the val be merged with training? 
# TODO: properly format all print statements