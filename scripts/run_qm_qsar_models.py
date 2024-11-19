import torch
import numpy as np
import random
import struct
import deepchem as dc
import diskcache
from rdkit import Chem
import polaris as po
from concurrent.futures import ThreadPoolExecutor, as_completed
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from threading import Lock
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import os.path as osp
import subprocess
import json
import struct
from sklearn.svm import SVR
import os
from torch_geometric.datasets import QM9
import pandas as pd
from botorch import fit_gpytorch_model # Newer versions of botorch don't have this function stick to 0.10.0
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
)

import sys
base_dir = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(base_dir, '..')
sys.path.append(models_path)

from qm_models import ModelTrainer, RNNRegressionModel, GRURegressionModel, GIN, GCN, GINCoTeaching, MLPRegressor, Gauche, train_epochs, train_epochs_co_teaching, testing, testing_co_teaching, train_mlp, predict_mlp, GATv2, GATv2a
from similarity_calculations import DistanceNetworkLightning

# Initialize the cache
cache = diskcache.Cache('./smiles_cache')

# Parameters
batch_size = 20  # Adjust as needed, default is 2048
N = 100  # The number of samples per bootstrapping iteration

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

graph_models = ['gin', 'gcn', 'gin_co_teaching', 'gauche_graph']
bit_vector_models = ['rf', 'gb', 'catboost', 'svm', 'gp', 'gauche']

def parse_arguments():
    parser = argparse.ArgumentParser(description="Framework for running QSAR/QM property prediction models")
    parser.add_argument("-q", "--qm_property", type=str, default="homo_lumo_gap", help="QM property to predict (default is homo_lumo_gap)")
    parser.add_argument("-m", "--models", nargs='*', help="Model(s) to use for prediction", required=True)
    parser.add_argument("-r", "--molecular_representations", nargs='*', help="Molecular representation as a list of strings", required=True)
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed (default is 42)")
    parser.add_argument("-n", "--sample-size", type=int, required=True, help="Sample size")
    parser.add_argument("-b", "--bootstrapping", type=int, default=1, help="Bootstrapping iterations (default is 1 ie. no bootstrapping)")
    parser.add_argument("--sampling-proportion", nargs='*', default=None, help="Sampling proportion to add artificial noise to (default is None)")
    parser.add_argument("--noise", type=bool, default=False, help="Generate artifical Gaussian noise")
    parser.add_argument("--sigma", nargs='*', default=None, help="Standard deviation(s) of artificially added Gaussian noise (default is None)")
    parser.add_argument("--distribution", type=str, default='gaussian', help="Distribution of artificial noise (default is Gaussian)")
    parser.add_argument("-t", "--hyperparameter-tuning", type=bool, default=False, help="Hyperparameter tuning (default is False)")
    parser.add_argument("-s", "--split", type=str, default="random", help="Method for splitting data (default is random)")
    parser.add_argument("-d", "--dataset", type=str, default="QM9", help="Dataset to run experiments on (default is QM9)")
    parser.add_argument("--shuffle", type=bool, default=True, help="Shuffle the data, required for bootstrapping (default is True)")
    return parser.parse_args()

def write_to_mmap(smiles_isomeric, smiles_canonical, smiles_alternative, property_value, domain_label, category, files):
    smiles_isomeric_binary = smiles_isomeric.encode('utf-8')
    smiles_canonical_binary = smiles_canonical.encode('utf-8')
    smiles_alternative_binary = smiles_alternative.encode('utf-8')
    property_value_binary = struct.pack('d', property_value)
    domain_label_binary = struct.pack('i', domain_label)

    # TODO: property value can be shortened, also potentially domain_label if there's a data structure that captures a small range of integers
    entry = (
        struct.pack('I', len(smiles_isomeric_binary)) + smiles_isomeric_binary +
        struct.pack('I', len(smiles_canonical_binary)) + smiles_canonical_binary +
        struct.pack('I', len(smiles_alternative_binary)) + smiles_alternative_binary +
        property_value_binary + domain_label_binary
    )

    files[category].write(entry)
    files[category].flush()

def load_qm9(qm_property):
    qm9 = QM9(root=osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9'))

    # Filter out molecules that cannot be processed by RDKit
    valid_indices_tensor = torch.load('valid_qm9_indices.pth')
    qm9 = qm9.index_select(valid_indices_tensor)

    # Isolate a single regression target
    y_target = pd.DataFrame(qm9.data.y.numpy())
    property_index = properties[qm_property]
    qm9.data.y = torch.Tensor(y_target[property_index])
    return qm9

def split_qm9(qm9, args, files):
    raw_results = {}

    if args.shuffle:
        qm9 = qm9.shuffle()
    else:
        # Note: cannot do this with bootstrapping
        torch.manual_seed(args.random_seed)
        num_data = len(qm9)
        indices = torch.randperm(num_data)
        qm9 = qm9.index_select(indices)
    data_size = args.sample_size

    if args.split == 'random':
        train_index = int(data_size * 0.8)
        test_index = train_index + int(data_size * 0.1)
        val_index = test_index + int(data_size * 0.1)
        train_idx = list(range(train_index))
        val_idx = list(range(train_index, test_index))
        test_idx = list(range(test_index, val_index))

    else:
        qm9_smiles = [data.smiles for data in qm9[:data_size]]
        Xs = np.zeros(len(qm9_smiles))
        dataset = dc.data.DiskDataset.from_numpy(X=Xs,ids=qm9_smiles)

        splitter = dc.splits.ScaffoldSplitter()
            
        train_idx, val_idx, test_idx = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    property_index = properties[args.qm_property]

    for index, data in enumerate(qm9[:data_size]):
        smiles_isomeric = data.smiles
        smiles_canonical = None

        category = "excluded"
        if index in train_idx:
            category = "train"
        elif index in test_idx:
            category = "test"
        elif index in val_idx:
            category = "val"

        smiles_alternative = ""
        if smiles_isomeric in cache and not 'alternative_smiles' in args.molecular_representations:
            smiles_canonical = cache[smiles_isomeric]
            # TODO: potentially keep alternative SMILES in the same cache if they look promising (smiles_isomeric, smiles_alternative)
        else:
            # Generate canonical SMILES and store it in cache
            mol = Chem.MolFromSmiles(smiles_isomeric)
            if 'alternative_smiles' in args.molecular_representations:
                smiles_alternative = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)
            smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
            cache[smiles_isomeric] = smiles_canonical

        # TODO: replace -1 with domain labels
        write_to_mmap(smiles_isomeric, smiles_canonical, smiles_alternative, data.y.item(), -1, category, files)

    return len(train_idx), len(test_idx)

def load_and_split_polaris(files, splitting_type):
    dataset_name = "BELKA"

    if dataset_name == "BELKA":
    	# Define the binding target to get the label from (e.g., 'binds_BRD4', 'binds_HSA', 'binds_sEH')
    	proteins = {'BDR4': 'binds_BRD4', 'HSA': 'binds_HSA', 'sEH': 'binds_sEH'}
    	binding_protein = 'BDR4'  # User-specified protein
    	binding_target = proteins[binding_protein]

    	# Load the dataset from PolarisHub
    	dataset = po.load_dataset("leash-bio/BELKA-v1")
    	dataset_size = dataset.size()[0]

    else:
    	dataset = None
    	dataset_size = None
    	# TODO: throw error instead of defining None

	# Select random indices for this iteration
    random_indices = np.random.choice(dataset_size, N, replace=False)

    # Pre-allocated space-efficient storage for scaffold splitting
    Xs = np.empty(N, dtype="U1000")  # Adjust dtype size if necessary for SMILES length
    smiles = np.empty(N, dtype="U1000")  # Adjust dtype size if necessary
    domain_labels_list = -np.ones(N, dtype=int)

    train_count = 0
    test_count = 0

    # Iterate over the selected random indices in batches
    for batch_start in range(0, N, batch_size):
        batch_indices = random_indices[batch_start:batch_start + batch_size]
        if dataset_name == "BELKA":
            smiles_isomeric_list = [dataset.get_data(row, "molecule_smiles") for row in batch_indices]
            properties_list = [dataset.get_data(row, binding_target) for row in batch_indices]
        else:
            break
            # TODO: handle this appropriately 
        split_list = []
        category = "train"
        # TODO: reach out to someone on polaris hub to get further information on predefined splits 
        if splitting_type == "central_core":
        	split_list = [dataset.get_data(row, "split") for row in batch_indices]
        elif splitting_type == "library":
        	split_list = [dataset.get_data(row, "split_group") for row in batch_indices]

        for i, smiles_isomeric in enumerate(smiles_isomeric_list):

            if smiles_isomeric in cache and not 'alternative_smiles' in args.molecular_representations:
                smiles_canonical = cache[smiles_isomeric]
                # TODO: potentially keep alternative SMILES in the same cache if they look promising (smiles_isomeric, smiles_alternative)
            else:
                # Generate canonical SMILES and store it in cache
                mol = Chem.MolFromSmiles(smiles_isomeric)
                if 'alternative_smiles' in args.molecular_representations:
                    smiles_alternative = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)
                else:
                    smiles_alternative = None
                smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
                cache[smiles_isomeric] = smiles_canonical

            if splitting_type == "scaffold":
            	# Accumulate for later scaffold splitting
                Xs[i] = smiles_isomeric
                smiles[i] = smiles_canonical
                pass

            elif splitting_type == "random":
                # Randomly assign molecule to a split (pre-determined split indices)
                if i % 10 < 8:
                    category = "train"
                    train += 1
                elif i % 10 == 8:
                    category = "val"
                else:
                    category = "test"
                    test += 1

            else:
            	category = split_list[i]

            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_alternative, properties_list[i], domain_labels_list[i], category, files)

    return train_count, test_count

    # Scaffold splitting process (if needed)
    if splitting_type == "scaffold":
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles)
        splitter = dc.splits.ScaffoldSplitter()
        train_idx, val_idx, test_idx = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        
        # Write data to mmap files efficiently
        for idx, category in zip([train_idx, val_idx, test_idx], ["train", "val", "test"]):
            for index in idx:
                smiles_isomeric = Xs[index]
                smiles_canonical = smiles[index]
                property_value = properties_list[index]
                domain_label = domain_labels_list[index]

                write_to_mmap(smiles_isomeric, smiles_canonical, property_value, domain_label, category, files)

def chemdist_func(s: str, network) -> np.ndarray or None:
    """
    Compute the molecular embedding using a provided neural network based on the molecule's graph representation.
    
    Parameters:
        s (str): The SMILES string of the molecule.
        network: The neural network model to use for computing the embedding.
    
    Returns:
        np.ndarray or None: The computed molecular embedding as a numpy array, or None if computation fails.
    """
    try:
        g = smiles_to_bigraph(smiles=s, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer())
        nfeats = g.ndata.pop('h')
        efeats = g.edata.pop('e')
        membed = network._net(g, nfeats, efeats).cpu().detach().numpy().ravel()
        return membed
    except Exception as e:
        # print(f"Error computing embedding for SMILES '{s}': {e}")
        return None

def pad_fingerprints(fp_array, max_fp_length):
    """
    Pads a list of fingerprint vectors to a uniform length.

    Parameters:
        fp_array (list of np.ndarray): List of fingerprint vectors.
        max_fp_length (int): Maximum fingerprint length to pad the vectors to.

    Returns:
        list of np.ndarray: A list of padded fingerprint vectors.
    """
    padded_fingerprints = []
    for fp in fp_array:
        # Adjust the fingerprint to be exactly the max_fp_length
        adjusted_fp = np.pad(fp[:max_fp_length], (0, max_fp_length - len(fp)), 'constant')
        padded_fingerprints.append(adjusted_fp)
    return padded_fingerprints

def calculate_classification_metrics(y_test, prediction, logging_bb=False):
    accuracy = accuracy_score(y_test, y_test_preds)
    roc_auc = roc_auc_score(y_test, y_test_probs[:, 1])  # Assuming binary classification
    precision = precision_score(y_test, y_test_preds, average="weighted")
    recall = recall_score(y_test, y_test_preds, average="weighted")
    f1 = f1_score(y_test, y_test_preds, average="weighted")
    pr_auc = average_precision_score(
        y_test, y_test_probs[:, 1], average="weighted"
    )

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

    # Optionally log the metrics
    if logging_bb:
        print("Mean Absolute Error:", mae)
        print("Mean Squared Error:", mse)
        print("RMSE", rmse)
        print("R-squared:", r2)

    # Return the list of calculated metrics
    return [mae, mse, rmse, r2]

def calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, dataset):
    # Calculate metrics for the target domain
    y_test_domain = []
    y_pred_domain = []
    y_test_non_target = []
    y_pred_non_target = []
    for i, y_i in enumerate(y_test):
        if domain_labels[i] == target_domain:
            y_test_domain.append(y_i)
            y_pred_domain.append(y_pred[i])
        else:
            y_test_non_target.append(y_i)
            y_pred_non_target.append(y_pred[i])

    print("size of original results")
    print(len(y_test))
    print(len(y_pred))

    print("size of split domain results")
    print(len(y_test_domain))
    print(len(y_pred_domain))
    print(len(y_test_non_target))
    print(len(y_pred_non_target))

    if len(y_test_domain) >= 3:
        if dataset == "QM9":
            domain_metrics = calculate_regression_metrics(y_test_domain, y_pred_domain, logging_bb=True)
        else:
            domain_metrics = calculate_classification_metrics(y_test_domain, y_pred_domain, logging_bb=True)
        print(f"Metrics for target domain {target_domain}: {domain_metrics}")

        if dataset == "QM9":
            non_target_metrics = calculate_regression_metrics(y_test_non_target, y_pred_non_target, logging_bb=True)
        else:
            non_target_metrics = calculate_classification_metrics(y_test_non_target, y_pred_non_target, logging_bb=True)
        print(f"Metrics for non-target domains: {non_target_metrics}")
    else:
        print(f"Not enough samples in target domain: {target_domain}")

def run_gauche(x_train, x_test, y_train, logging_bb, sigma, proportion, dataset):
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

    # Optionally log results
    if logging_bb:
        print("Results for GP and ", molecular_representation)
        print(sigma)
        print(proportion)

    logging_bb = True
    if distribution == "domain_mpnn" or distribution == "domain_tanimoto":
        calculate_domain_metrics(y_test, y_pred, domain_labels, target_domain, dataset)
        logging_bb = False

    if dataset == "QM9":
        metrics = calculate_regression_metrics(y_test, y_pred, logging_bb=logging_bb)
    else:
        metrics = calculate_classification_metrics(y_test, y_pred, logging_bb=logging_bb)
    bootstrapped_results.put((molecular_representation, model_type, metrics))

    if sigma is not None and proportion is not None:
        target_metric.put((molecular_representation, model_type, (sigma, proportion), metrics[3]))

    return metrics[3]

def run_model(x_train, y_train, x_test, y_test, model_type, molecular_representation, hyperparameter_tuning, bootstrapping, sigma, proportion, current_seed, target_metric, bootstrapped_results, network, distribution, dataset):
    def black_box_function(tuning_active=False, logging_bb=False, sigma=None, proportion=None, **params):
        nonlocal x_train, x_test, y_train

        # Configure model parameters based on the model type
        # if molecular_representation == 'mpnn':
        #     # Load pre-computed MPNN fingerprints if available
        #     with open('mpnn_dict.pkl', 'rb') as file:
        #         loaded_mpnn_dict = pickle.load(file)
        #     filtered_train = []
        #     filtered_test = []
        #     filtered_y = []
        #     # Filter or compute fingerprints as needed
        #     mpnn_fp_max_length = 0
        #     for i, smi in enumerate(x_train):
        #         if smi in loaded_mpnn_dict.keys():
        #             fp = loaded_mpnn_dict[smi]
        #         else:
        #             fp = chemdist_func(smi, network)
        #         if fp is not None:
        #             filtered_y.append(y_train[i])
        #             filtered_train.append(fp)
        #             mpnn_fp_max_length = max(len(fp), mpnn_fp_max_length)
        #     for i, smi in enumerate(x_test):
        #         if smi in loaded_mpnn_dict.keys():
        #             fp = loaded_mpnn_dict[smi]
        #         else:
        #             fp = chemdist_func(smi, network)
        #         if fp is not None:
        #             filtered_test.append(fp)

        #     # Pad fingerprints to uniform length
        #     x_train = pad_fingerprints(filtered_train, mpnn_fp_max_length)
        #     x_test = pad_fingerprints(filtered_test, mpnn_fp_max_length)
        #     y_train = filtered_y

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
            model = RandomForestRegressor(random_state=current_seed, **params) if params else RandomForestRegressor(random_state=current_seed)

        elif model_type == 'svm':
            # Adjust SVR parameters
            if params:
                if params['kernel'] < 1:
                    params['kernel'] = 'rbf'
                elif params['kernel'] < 2:
                    params['kernel'] = 'poly'
                else:
                    params['kernel'] = 'sigmoid'
            model = SVR(**params) if params else SVR()

        elif model_type == 'gb':
            # Adjust Gradient Boosting parameters
            if params: 
                params['n_estimators'] = int(round(params['n_estimators']))
                params['max_depth'] = int(round(params['max_depth']))
                if params['max_depth'] > 15: 
                    params['max_depth'] = None
            model = XGBRegressor(random_state=current_seed, **params) if params else XGBRegressor(random_state=current_seed)

        elif model_type == 'gauche':
            return run_gauche(x_train, x_test, y_train, logging_bb, sigma, proportion, dataset)

        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        # Train the model and make predictions
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
       
        # Optionally log results
        if logging_bb:
            print("Results for ", model_type, " and ", molecular_representation)
            print(sigma)
            print(proportion)
        
        metrics = calculate_regression_metrics(y_test, y_pred, logging_bb=logging_bb)
        bootstrapped_results.put((molecular_representation, model_type, metrics))

        if sigma is not None and proportion is not None:
            target_metric.put((molecular_representation, model_type, (sigma, proportion), metrics[3]))

        return metrics[3]  # Return the negative MSE for optimization

    # Check if hyperparameter tuning is enabled
    if hyperparameter_tuning:
        # Define parameter bounds for hyperparameter tuning based on the model type
        if model_type == 'rf':
            pbounds = {
                'max_depth': (10, 200),  # >150 max_depth is interpreted as None, rounded to int
                'max_features': (0, 1),  # Mapped to auto or sqrt
                'min_samples_leaf': (1, 50),  # Rounded to int
                'min_samples_split': (2, 20),  # Rounded to int
                'n_estimators': (10, 2000),  # Rounded to int
            }
        elif model_type == 'svm':
            pbounds = {
                'C': (0.1, 100),
                'gamma': (0, 1),
                'kernel': (0, 3),  # Mapped to rbf, poly, sigmoid
            }
        elif model_type == 'gb':
            pbounds = {
                'max_depth': (0, 20),  # >15 max_depth is interpreted as None, rounded to int
                'learning_rate': (0.001, 0.2),
                'subsample': (0.5, 1),
                'n_estimators': (10, 2000),  # Rounded to int
            }
        elif model_type == 'catboost':
            pbounds = {
                'iterations': (100, 1000),  # A reasonable range might be from 100 to 1000 trees
                'depth': (4, 10),  # Tree depth from 4 to 10 is commonly used
                'learning_rate': (0.01, 0.3),  # Learning rate from 0.01 to 0.3 can capture a wide range of speeds
                'l2_leaf_reg': (1, 10),  # L2 regularization from 1 to 10 adds a moderate regularization effect
                'border_count': (32, 255),  # The number of splits for numerical features can be in the range of 32 to 255
            }
        elif model_type == 'gp':
            pbounds = {
                'alpha': (1e-10, 1),
                'n_restarts_optimizer': (0, 10),
                'kernel': (0, 3)
            }
        elif model_type == 'gauche':
            pbounds = {}
        
        # Initialize Bayesian optimization with the defined parameter bounds and optimization function
        optimizer = BayesianOptimization(f=lambda **params: black_box_function(tuning_active=True, sigma=sigma, proportion=proportion, **params), pbounds=pbounds, random_state=42)
        # Perform the optimization
        optimizer.maximize(init_points=2, n_iter=5)

        # Extract the best parameters found by the optimizer
        best_params = optimizer.max['params']
        print("Best params: " + str(best_params))
        
        # Use the best parameters to run the model with logging if bootstrapping is enabled
        if bootstrapping == 1:
            black_box_function(logging_bb=True, sigma=sigma, proportion=proportion, **best_params)
        else:
            black_box_function(sigma=sigma, proportion=proportion, **best_params)
    elif bootstrapping == 1:
        # Run the model with logging enabled if bootstrapping is enabled but without hyperparameter tuning
        black_box_function(logging_bb=True, sigma=sigma, proportion=proportion)
    else:
        # Run the model without hyperparameter tuning or logging
        black_box_function(sigma=sigma, proportion=proportion, logging_bb=True)
    

def process_and_run(args, iteration, iteration_seed, train_count, test_count, target_domain, env, rust_executable_path, files, target_metric, bootstrapped_results, network, dataset):
    config = {
        'sample_size': args.sample_size,
        'noise': args.noise,
        'train_count': train_count,
        'test_count': test_count,
        'max_vocab': 30,
        'bootstrapping_iteration': iteration,
        'target_domain': target_domain
    }

    with open('config.json', 'w') as f:
        json.dump(config, f)

    for molecular_representation in args.molecular_representations:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []

            for s in args.sigma:
                for p in args.sampling_proportion:
                    print("Rust executable path: ", rust_executable_path)
                    proc_a = subprocess.Popen(
                        [
                            rust_executable_path,
                            '--seed', str(iteration_seed),
                            '--molecular_representation', molecular_representation,
                            '--model', "rf",
                            '--sigma', str(s),
                            '--sampling_proportion', str(p),
                            '--noise_distribution', args.distribution,
                        ],
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    stdout, stderr = proc_a.communicate()
                    print("Rust stderr:", stderr)
                    print("Rust stdout:", stdout)

                    temp_file_path = None
                    for line in stdout.splitlines():
                        if line.startswith("TEMP_FILE_PATH:"):
                            temp_file_path = line.split("TEMP_FILE_PATH:")[1].strip()
                            break

                    if temp_file_path:
                        if os.path.exists(temp_file_path):
                            with open(temp_file_path, 'r') as file:
                                try:
                                    data = json.load(file)
                                    fps_train_np, y_train_np, fps_test_np, y_fixed_test_np = process_data_from_rust(data, molecular_representation)

                                    for model in args.models:
                                        if model not in graph_models:
                                            future = executor.submit(
                                                run_model, 
                                                fps_train_np, 
                                                y_train_np, 
                                                fps_test_np, 
                                                y_fixed_test_np, 
                                                model, 
                                                molecular_representation, 
                                                args.hyperparameter_tuning, 
                                                args.bootstrapping, 
                                                s, 
                                                p, 
                                                iteration_seed, 
                                                target_metric, 
                                                bootstrapped_results, 
                                                bootstrapped_results, 
                                                network, 
                                                args.distribution,
                                                dataset,
                                            )
                                            futures.append(future)
                                except json.JSONDecodeError as e:
                                    print(f"Failed to decode JSON from file: {e}")
                                    print(f"File content: {file.read()}")

                            os.remove(temp_file_path)
                        else:
                            print(f"Temporary file not found: {temp_file_path}")
                    else:
                        print("Temporary file path not found in Rust stdout")

            for future in as_completed(futures):
                future.result()
    
    for file in files.values():
        file.close()

def process_data_from_rust(data, molecular_representation):
    """
    Process the data received from Rust.
    `data` is a dictionary containing the serialized training and test data.
    """
    # Deserialize data
    if molecular_representation == 'ecfp4' or molecular_representation == 'sns':
        fps_train = [fp['ECFP4'] for fp in data['fps_train']]
        fps_test = [fp['ECFP4'] for fp in data['fps_test']]
    elif molecular_representation == "smiles":
        fps_train = [fp['SMILES']['data'] for fp in data['fps_train']]
        fps_test = [fp['SMILES']['data'] for fp in data['fps_test']]
    else:
        fps_train = [fp['SRAW'] for fp in data['fps_train']]
        fps_test = [fp['SRAW'] for fp in data['fps_test']]
    y_train = data['y_train']
    y_fixed_test = data['y_fixed_test']
    # r_squared = data['r_squared']  # New line to capture R-squared value

    fps_train_np = np.array(fps_train)
    y_train_np = np.array(y_train)
    fps_test_np = np.array(fps_test)
    y_fixed_test_np = np.array(y_fixed_test)

    if molecular_representation == 'ecfp4':
        fps_train_np = np.unpackbits(fps_train_np.astype(np.uint64).view(np.uint8), axis=1)
        fps_test_np = np.unpackbits(fps_test_np.astype(np.uint64).view(np.uint8), axis=1)

    # print("Training data (features):", fps_train_np)
    # print("Training data (labels):", y_train_np)
    # print("Test data (features):", fps_test_np)
    # print("Test data (labels):", y_fixed_test_np)
    # print("R-squared:", r_squared)  # Print R-squared value
    return fps_train_np, y_train_np, fps_test_np, y_fixed_test_np

def main():
    args = parse_arguments()

    # Initialise queues to store results across threads
    target_metric = Queue()
    bootstrapped_results = Queue()

    # Define the sigma and sampling proportion associated with artificial noise
    if args.sigma is None and args.noise:
        args.sigma = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    elif args.noise:
        args.sigma = [float(s) for s in args.sigma]
    else:
        args.sigma = [0.0]

    if args.sampling_proportion is None and args.noise:
        args.sampling_proportion = [0.0, 0.25, 0.5, 0.75, 1.0]
    elif args.noise:
        args.sampling_proportion = [float(p) for p in args.sampling_proportion]
    else:
        args.sampling_proportion = [0.0]

    # Prepare for commmunication with Rust
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"
    rust_executable_path = os.path.join(base_dir, 'rust_processor/target/release/rust_processor')

    # Load DistanceNetworkLightning for MPNN generation
    network = None
    if 'mpnn' in args.molecular_representations:
        network = DistanceNetworkLightning.load_from_checkpoint("model_trained.pt")

    # Load QM9 
    if args.dataset == "QM9":
        qm9 = load_qm9(args.qm_property)

    for iteration in range(args.bootstrapping):
        files = {
            "train": open('train_' + str(iteration) + '.mmap', 'wb+'),
            "test": open('test_' + str(iteration) + '.mmap', 'wb+'),
            "val": open('val_' + str(iteration) + '.mmap', 'wb+'),
        }

        iteration_seed = (args.random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF  # XOR and mask for 32-bit seed

        random.seed(iteration_seed)
        np.random.seed(iteration_seed)
        torch.manual_seed(iteration_seed)

        if args.dataset == "QM9":
            train_count, test_count = split_qm9(qm9, args, files)

        else:
            train_count, test_count = load_and_split_polaris(files, args.split)
        
        target_domain = 1 # TODO: change, this is just a placeholder
        process_and_run(args, iteration, iteration_seed, train_count, test_count, target_domain, env, rust_executable_path, files, target_metric, bootstrapped_results, network, args.dataset)


# TODO: add the following Polaris datasets: tdcommons/cyp2d6-veith, tdcommons/cyp2c9-veith, tdcommons/cyp3a4-veith, graphium/tox21-v1
# TODO: I need to determine if getting QM9 from polaris (graphium/qm9-v1) is better than what I'm currently doing in terms of loading the graphs in 
# TODO: modify existing functions to make them compatible with classification 

# Next steps: copy over rest of functionality from stream_processing, starting with bootstrapping iterations, and cleanly refactor it 
# Next next steps: graphs, just stick with FPs/SMILES for now

if __name__ == "__main__":
    main()