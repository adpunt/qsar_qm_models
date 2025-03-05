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
from bayes_opt import BayesianOptimization
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

import sys
sys.path.append('../models/')
sys.path.append('../preprocessing/')
sys.path.append('../results/')

from qm_models import ModelTrainer, RNNRegressionModel, GRURegressionModel, GIN, GCN, GINCoTeaching, MLPRegressor, Gauche, train_epochs, train_epochs_co_teaching, testing, testing_co_teaching, train_mlp, predict_mlp, GATv2, GATv2a

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
valid_indices_path = os.path.join(data_dir, 'valid_qm9_indices.pth')

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

DELIMITER = b"\x1F"  # ASCII 31 (Unit Separator)
NEWLINE = b"\n"

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

bit_vectors = ['ecfp4', 'mpnn', 'sns']
graph_models = ['gin', 'gcn', 'ginct', 'gauche_graph', 'gin2d', 'gtat']

# Initialize the cache
cache = diskcache.Cache('./smiles_cache')

# TODO: make sure everything above function definitions is properly formatted
# TODO: make sure val is working for hyperparameter tuning
# TODO: reformat import statements
# TODO: redirect results (csv and plots) to correct folder 

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
    parser.add_argument("--hyperparameter-tuning", type=bool, default=False, help="Hyperparameter tuning (default is False)")
    parser.add_argument("-f", "--filepath", type=str, default='../results/test.csv', help="Filepath to save raw results in csv (default is None)")
    parser.add_argument("--logging", type=bool, default=False, help="Extra logging to check individual entries in mmap files (default is False)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs for training grpah-based models (default is 15)")
    parser.add_argument("--clean-smiles", type=bool, default=False, help="Clean the SMILES string (default is False)")
    parser.add_argument("--shap", type=bool, default=False, help="Calculate SHAP values for relevant tree-based models (default is False)")
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
                                                               vec_dimension = 1024)

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
                                                               vec_dimension = 1024)


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

        # TODO: only do this if not a graph
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
            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, data.y.item(), category, files, args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)

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
    # create a function sub_id_enumerator that maps a mol object to a dictionary whose keys are the integer substructure identifiers in mol and whose values are the associated substructure counts (i.e., how often each substructure appears in mol)
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius = max_radius,
                                                                 atomInvariantsGenerator = rdFingerprintGenerator.GetMorganFeatureAtomInvGen() if pharm_atom_invs == True else rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True),
                                                                 useBondTypes = bond_invs,
                                                                 includeChirality = chirality)
    
    sub_id_enumerator = lambda mol: morgan_generator.GetSparseCountFingerprint(mol).GetNonzeroElements() if mol is not None else {}
    
    # construct dictionary that maps each integer substructure identifier sub_id in mols_train to its associated prevalence (i.e., to the total number of compounds in mols_train that contain sub_id at least once)
    sub_ids_to_prevs_dict = {}
    for mol in mols_train:
        for sub_id in sub_id_enumerator(mol).keys():
            sub_ids_to_prevs_dict[sub_id] = sub_ids_to_prevs_dict.get(sub_id, 0) + 1

    # create list of integer substructure identifiers sorted by prevalence in mols_train
    sub_ids_sorted_list = sorted(sub_ids_to_prevs_dict, key = lambda sub_id: (sub_ids_to_prevs_dict[sub_id], break_ties_with(sub_id)), reverse = True)
    
    # create auxiliary function that generates standard unit vectors in NumPy
    def standard_unit_vector(dim, k):
        
        vec = np.zeros(dim, dtype = int)
        vec[k] = 1
        
        return vec
    
    # create one-hot encoder for the first vec_dimension substructure identifiers in sub_ids_sorted_list; all other substructure identifiers are mapped to a vector of 0s
    def sub_id_one_hot_encoder(sub_id):
        
        return standard_unit_vector(vec_dimension, sub_ids_sorted_list.index(sub_id)) if sub_id in sub_ids_sorted_list[0: vec_dimension] else np.zeros(vec_dimension)
    
    # create a function ecfp_featuriser that maps RDKit mol objects to vectorial ECFPs via a Sort & Slice substructure pooling operator trained on mols_train
    def ecfp_featuriser(mol):

        # create list of integer substructure identifiers contained in input mol object (multiplied by how often they are structurally contained in mol if sub_counts = True)
        if sub_counts == True:
            sub_id_list = [sub_idd for (sub_id, count) in sub_id_enumerator(mol).items() for sub_idd in [sub_id]*count]
        else:
            sub_id_list = list(sub_id_enumerator(mol).keys())
        
        # create molecule-wide vectorial representation by summing up one-hot encoded substructure identifiers
        ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis = 0)
    
        return ecfp_vector
    
    # print information on training set
    if print_train_set_info == True:
        print("Number of compounds in molecular training set = ", len(mols_train))
        print("Number of unique circular substructures with the specified parameters in molecular training set = ", len(sub_ids_to_prevs_dict))

    return ecfp_featuriser

def calculate_classification_metrics(y_test, prediction, logging_bb=False):
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
    if logging_bb:
        print("Accuracy:", accuracy)
        print("ROC AUC:", roc_auc)
        print("Precision", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("PR AUC:", pr_auc)

    return [accuracy, roc_auc, precision, recall, f1, pr_auc]

def calculate_regression_metrics(y_test, prediction, logging_bb=False):
    mae = mean_absolute_error(y_test, prediction)
    mse = mean_squared_error(y_test, prediction)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, prediction)
    pearson_corr, _ = pearsonr(y_test, prediction)

    # Optionally log the metrics
    if logging_bb:
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
                writer.writerow(["sigma", "iteration", "model", "rep", "sample_size", "r2_score"])
            
            # Save the results
            writer.writerow([s, iteration, model, rep, n, r2])

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

# TODO: see if this works with classification/regression
def run_gauche(args, x_train, x_test, y_train, y_test, logging_bb, sigma, iteration, rep):
    kernel = "tanimoto"

    x_train_tensor = torch.from_numpy(x_train).double()
    x_test_tensor = torch.from_numpy(x_test).double()
    y_train_tensor = torch.from_numpy(y_train).double()  

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = Gauche(x_train_tensor, y_train_tensor, likelihood, kernel)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # Fit GP model with BoTorch
    fit_gpytorch_model(mll)

    # Make predictions with the trained GP model
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(x_test_tensor)
        y_pred = preds.mean.numpy()
        pred_vars = preds.variance.numpy()

    logging_bb = True
    # TODO: are you sure distribution is the right term here?
    if args.distribution == "domain_mpnn" or args.distribution == "domain_tanimoto":
        calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, args.dataset)
        logging_bb = False

    if args.dataset == 'QM9':
        metrics = calculate_regression_metrics(y_test, y_pred, logging_bb=logging_bb)
    else:
        metrics = calculate_classification_metrics(y_test, y_pred, logging_bb=logging_bb)

    if logging_bb:
        print(f"Results for gauche and {rep}")

    save_results(args.filepath, sigma, iteration, "gauche", rep, args.sample_size, metrics[3], metrics[0], metrics[4])

    return metrics[3]

# TODO: need to call save_results in here
def run_custom(x_train, x_test, y_train, logging_bb, sigma, dataset, model_path, metadata_path=None):
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
    if logging_bb:
        print("Results for custom model:", model_path)
        print("Sigma:", sigma)

    logging_bb = True
    if distribution == "domain_mpnn" or distribution == "domain_tanimoto":
        calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, dataset)
        logging_bb = False

    if dataset == 'QM9':
        metrics = calculate_regression_metrics(y_test, y_pred, logging_bb=logging_bb)
    else:
        metrics = calculate_classification_metrics(y_test, y_pred, logging_bb=logging_bb)

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

            sns_fp = None
            if "sns" in molecular_representations:
                sns_fp = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8), bitorder='little')

                field_idx += 1
                if logging:
                    print(f"sns_fp: {sns_fp}")

                if "sns" == rep:
                    # Ensure correct size before unpacking
                    if sns_fp.size == 128:  # Expecting 16 bytes (128 bits)
                        # sns_fp = np.unpackbits(sns_fp)
                        u64_array = np.frombuffer(sns_fp, dtype=np.uint64)
                        sns_fp = np.unpackbits(ecfp4, bitorder='little')[:1024]
                    else:
                        # print(f"Warning: SNS fingerprint has unexpected size {sns_fp.size} at index {field_idx}")
                        continue
            
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

# TODO: make this run_regression_model and make a separate run_classification_model
def run_model(x_train, y_train, x_test, y_test, model_type, args, iteration_seed, rep, iteration, s):
    def black_box_function(tuning_active=False, logging_bb=False, sigma=None, **params):
        # nonlocal x_train, x_test, y_train

        if model_type == 'rf':
            # Adjust RandomForest parameters
            if params: 
                params['min_samples_split'] = int(round(params['min_samples_split']))
                params['min_samples_leaf'] = int(round(params['min_samples_leaf']))
                params['n_estimators'] = int(round(params['n_estimators']))
                params['max_depth'] = int(round(params['max_depth']))
                if params['max_depth'] > 150:
                    params['max_depth'] = None
                if params['max_features'] < 0.5:
                    params['max_features'] = 1.0
                else:
                    params['max_features'] = 'sqrt'
            if args.dataset == 'QM9':
                model = RandomForestRegressor(random_state=iteration_seed, **params) if params else RandomForestRegressor(random_state=iteration_seed)
            else:
                model = RandomForestClassifier(random_state=iteration_seed, **params) if params else RandomForestClassifier(random_state=iteration_seed)

        elif model_type == 'svm':
            # Adjust SVR parameters
            if params:
                if params['kernel'] < 1:
                    params['kernel'] = 'rbf'
                elif params['kernel'] < 2:
                    params['kernel'] = 'poly'
                else:
                    params['kernel'] = 'sigmoid'
            if args.dataset == 'QM9':
                model = SVR(**params) if params else SVR()
            else:
                model = SVC(**params) if params else SVC()

        # TODO: may need to adjust pbounds and params and the call for classification
        elif model_type == 'xgboost':
            # Adjust Gradient Boosting parameters
            if params: 
                params['n_estimators'] = int(round(params['n_estimators']))
                params['max_depth'] = int(round(params['max_depth']))
                if params['max_depth'] > 15: 
                    params['max_depth'] = None
            if args.dataset == 'QM9':
                model = XGBRegressor(random_state=iteration_seed, **params) if params else XGBRegressor(random_state=iteration_seed)
            else:
                model = XGBClassifier(random_state=iteration_seed, **params) if params else XGBClassifier(random_state=iteration_seed)

        elif model_type == 'gauche':
            return run_gauche(args, x_train, x_test, y_train, y_test, logging_bb, s, iteration, rep)

        elif model_type == 'custom':
            return run_custom_model(x_train, x_test, y_train, logging_bb, sigma, args.dataset, args.distribution)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        if args.shap:
            # Calculate SHAP values
            explainer = None
            shap_values = None

            try:
                if model_type in ['rf', 'xgboost']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(x_test)

                # TODO: uncomment, temporarily commented out for the sake of time
                # elif model_type == 'svm':
                #     explainer = shap.KernelExplainer(model.predict, x_test[:50])  # Use a small subset for efficiency
                #     shap_values = explainer.shap_values(x_test)

                if logging_bb and shap_values is not None:
                    print(f"SHAP values calculated for {model_type}")

                if shap_values is not None:
                    print(f"SHAP value length: {x_test.shape[1]}")
                    save_shap_values(shap_values, [f'feature_{i}' for i in range(x_test.shape[1])], x_test, args.filepath, model_type, iteration, rep)


            except Exception as e:
                print(f"SHAP calculation failed for {model_type}: {e}")

        # Optionally log results
        if logging_bb:
            print(f"Results for {model_type} and {rep}")

        metrics = calculate_regression_metrics(y_test, y_pred, logging_bb=logging_bb)

        # Save results with all relevant information
        save_results(args.filepath, s, iteration, model_type, rep, args.sample_size, metrics[3], metrics[0], metrics[4])

        if args.dataset == 'QM9':
            return metrics[3]  # Return the negative MSE for optimization with regression
        else:
            return metrics[0]  # Return the log loss for optimization with classification

    # Check if hyperparameter tuning is enabled
    if args.hyperparameter_tuning:
        # Define parameter bounds for hyperparameter tuning based on the model type
        if model_type == 'rf':
            pbounds = {
                'max_depth': (10, 200),
                'max_features': (0, 1),
                'min_samples_leaf': (1, 50),
                'min_samples_split': (2, 20),
                'n_estimators': (10, 2000),
            }
        elif model_type == 'svm':
            pbounds = {
                'C': (0.1, 100),
                'gamma': (0, 1),
                'kernel': (0, 3),
            }
        elif model_type == 'xgboost':
            pbounds = {
                'max_depth': (0, 20),
                'learning_rate': (0.001, 0.2),
                'subsample': (0.5, 1),
                'n_estimators': (10, 2000),
            }
        elif model_type == 'gauche':
            pbounds = {}

        elif model_type == 'custom':
            pbounds = get_custom_hyperparameter_bounds(args.metadata_file)
        
        # Initialize Bayesian optimization
        optimizer = BayesianOptimization(f=lambda **params: black_box_function(tuning_active=True, sigma=s, **params), pbounds=pbounds, random_state=42)

        # Perform the optimization
        optimizer.maximize(init_points=2, n_iter=5)

        # Extract the best parameters found by the optimizer
        best_params = optimizer.max['params']
        print(f"Best params: {best_params}")
        
        # Use the best parameters to run the model with logging if bootstrapping is enabled
        if args.bootstrapping == 1:
            black_box_function(logging_bb=True, sigma=s, **best_params)
        else:
            black_box_function(sigma=s, **best_params)
    elif args.bootstrapping == 1:
        # Run the model with logging enabled if bootstrapping is enabled but without hyperparameter tuning
        black_box_function(logging_bb=True, sigma=s)
    else:
        # Run the model without hyperparameter tuning or logging
        black_box_function(sigma=s, logging_bb=True)

# TODO: need to call save_results
def run_qm9_graph_model(args, qm9, train_idx, test_idx, val_idx, s, train_y):
    for model_type in args.models:
        if model_type == "gin" or model_type == "gin2d":
            model = GIN(dim_h=64)
        elif model_type == "gcn":
            model = GCN(dim_h=128)
        elif model_type == "gin_co_teaching":
            model = GINCoTeaching(dim_h=64)
        elif model_type == "gauche_graph":
            likelihood = gpytorch.likelihoods.GaussianLikelihood()
            # TODO: change this
            kernel = gpytorch.kernels.RBFKernel  # or whichever kernel you need
            kernel_kwargs = {}  # specify any necessary kernel arguments

            # Convert graph-structured inputs to custom data class for non-tensorial inputs and convert labels to PyTorch tensors
            X_train = NonTensorialInputs(qm9[train_idx].data)
            X_test = NonTensorialInputs(qm9[test_idx].data)
            # TODO: use noisy_train instead
            y_train = torch.tensor(qm9[train_idx].data.y.numpy()).flatten().float()
            y_test = torch.tensor(qm9[test_idx].data.y.numpy()).flatten().float()

            # Initialize the GaucheGraph model
            model = GaucheGraph(X_train, y_train, likelihood, kernel, **kernel_kwargs)

            # Define the marginal log likelihood used to optimize the model hyperparameters
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            # Use the BoTorch utility for fitting GPs in order to use the LBFGS-B optimizer (recommended)
            fit_gpytorch_model(mll)

            # Get into evaluation (predictive posterior) mode and compute predictions
            model.eval()
            likelihood.eval()
            with torch.no_grad():
                f_pred = model(X_test)
                y_pred = f_pred.mean
                y_var = f_pred.variance

            # Optionally log results
            if logging_bb:
                print("Results for ", model_type, " and ", molecular_representation)

            if pred_tracking:
                predictions_dict.put((molecular_representation, model_type, (y_test, y_pred)))

            logging_bb = True
            if args.distribution == "domain_mpnn" or args.distribution == "domain_tanimoto":
                calculate_domain_metrics(y_test, y_pred, domain_labels_subset, target_domain)
                logging_bb = False
            
            metrics = calculate_regression_metrics(y_test, y_pred, logging_bb=logging_bb)
            pass

        else: 
            pass

        # Add label noise
        train_set = qm9[train_idx]
        print(f"train labels before noise: {train_set.data.y}")
        if s > 0:
            # List of values drawn from Gaussian distribution with sigma = s of length train_idx
            # Then add those values onto train_set.data.y
            # Replace train_set.data.y like below with a tensoe of that list
            print(f"train_y: {train_y}")
            train_set.data.y = torch.tensor(train_y.flatten(), dtype=torch.float32)
        print(f"train labels after noise: {train_set.data.y}")

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

        logging_bb = True
        if args.distribution == "domain_mpnn" or args.distribution == "domain_tanimoto":
            calculate_domain_metrics(test_target, test_y, domain_labels_subset, target_domain)
            logging_bb = False

        metrics = calculate_regression_metrics(test_target, test_y, logging_bb=logging_bb)

def process_and_run(args, iteration, iteration_seed, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset=None):
    graph_only = True
    for model in args.models:
        if model not in graph_models:
            graph_only = False 

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
                try: 
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
                except Exception as e:
                    print(f"Error with {rep} and {model}; more details: {e}")
            else:
                if args.dataset == 'QM9':
                    print(f"y_train: {y_train}")
                    run_qm9_graph_model(args, dataset, x_train, test_idx, val_idx, s, y_train)
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

            # TODO: use iteration seed so you can run multiple versions of the program at the same time with different seeds
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