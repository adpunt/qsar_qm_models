
import argparse
import os
import os.path as osp
import random
import json
import subprocess
import struct
import warnings
import numpy as np
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
import diskcache
import deepchem as dc
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import polaris as po
from polaris.hub.client import PolarisHubClient
import optuna
import logging
import sqlite3
import pickle
from torch_geometric.utils import to_networkx

import sys
sys.path.append('../models/')
sys.path.append('../preprocessing/')
sys.path.append('../results/')

from models import *

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
valid_indices_path = os.path.join(data_dir, 'valid_qm9_indices.pth')

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

DELIMITER = b"\x1F"  # ASCII 31 (Unit Separator)
NEWLINE = b"\n"

# TODO: get git to stop tracking Cargo.lock 
# TODO: figure out tuning for GINs

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

bit_vectors = ['ecfp4', 'mpnn', 'sns', 'plec']
graph_models = ['gin', 'gcn', 'ginct', 'graph_gp', 'gin2d', 'gtat']
neural_nets = ["dnn", "mlp", "rnn", "gru", 'factorization_mlp', 'residual_mlp']

cache_path = "../data/smiles_cache.sqlite"
conn = sqlite3.connect(cache_path)

cursor = conn.cursor()

# Check if the cache exists
if not os.path.exists(cache_path):
    print("Cache does not exist. Creating a new one...")
    os.makedirs(cache_path, exist_ok=True)  # Ensure the directory exists
    cache = diskcache.Cache(cache_path)  # Initialize a new cache
else:
    try:
        cache = diskcache.Cache(cache_path)
    except Exception as e:
        cache = None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make sure everything above function definitions is properly formatted
# TODO: make sure val is working for hyperparameter tuning
# TODO: reformat import statements

# TODO: add argument for classification/regression, then dataset source 
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
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training grpah-based models (default is 100)")
    parser.add_argument("--clean-smiles", type=bool, default=False, help="Clean the SMILES string (default is False)")
    parser.add_argument("--n-trials", type=bool, default=20, help="Number of trials in hyperparameter tuning (default is 20)")
    parser.add_argument("-p", "--params", type=str, default=None, help="Filepath for model parameters (default is None)")
    return parser.parse_args()

# TODO: add PLEC
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
        if randomized_smiles:
            randomized_smiles_binary = randomized_smiles.encode('utf-8') + DELIMITER
            entry += randomized_smiles_binary
        else:
            entry += DELIMITER

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
        # TODO: this try/catch logic is wrong
        if smiles_isomeric in cache and not 'randomized_smiles' in args.molecular_representations:
            try: 
                smiles_canonical = cache[smiles_isomeric]
            except Exception as e:
                mol = Chem.MolFromSmiles(smiles_isomeric)
            # TODO: potentially keep randomized SMILES in the same cache if they look promising (smiles_isomeric, smiles_randomized)
        else:
            # Generate canonical SMILES and store it in cache
            mol = Chem.MolFromSmiles(smiles_isomeric)
        
        if not mol:
            try:
                cache[smiles_isomeric] = None
            except Exception as e:
                continue
            continue
        if 'randomized_smiles' in args.molecular_representations:
            smiles_randomized = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)
        smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        cache[smiles_isomeric] = smiles_canonical

        sns_fp = None
        # TODO: need to define protein by pdb 
        if 'sns' in args.molecular_representations or 'plec' in args.molecular_representations:
            if index in train_idx:
                mol = mols_train.popleft()
            if not mol: 
                mol = Chem.MolFromSmiles(smiles_isomeric)
            if 'sns' in args.molecular_representations:
                sns_fp = ecfp_featuriser(mol)
            # if 'plec' in args.molecular_representations:
            #     od_mol = oddt.toolkit.Molecule(rd_mol)
            #     plec = PLEC(od_mol,protein) 
        
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
        Xs = np.zeros(len(qm9_smiles))  # Dummy features just for splitting
        dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=qm9_smiles)

        splitter = dc.splits.ScaffoldSplitter()
        split = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        train_idx, val_idx, test_idx = split

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
        smiles_randomized = None
        mol = None

        category = "excluded"
        if index in train_idx:
            category = "train"
        elif index in test_idx:
            category = "test"
        elif index in val_idx:
            category = "val"

        # TODO: only do this if not a graph
        if 'randomized_smiles' not in args.molecular_representations:
            cursor.execute("SELECT canonical FROM smiles_cache WHERE isomeric = ?", (smiles_isomeric,))
            result = cursor.fetchone()
            if result:
                smiles_canonical = result[0]

        if smiles_canonical is None:
            mol = Chem.MolFromSmiles(smiles_isomeric)
            smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles_canonical is None:
                continue

        if 'randomized_smiles' in args.molecular_representations:
            smiles_randomized = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)

        sns_fp = None
        if 'sns' in args.molecular_representations:
            if index in train_idx:
                mol = mols_train.popleft()
            if not mol: 
                mol = Chem.MolFromSmiles(smiles_isomeric)
            sns_fp = ecfp_featuriser(mol)

        if smiles_canonical and not (category == "excluded"):
            if 'randomized_smiles' in args.molecular_representations and not smiles_randomized:
                continue
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

# TODO: refactor this, only do the processing if it's that rep 
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
                if "randomized_smiles" == rep:
                    randomized_smiles = fields[field_idx].decode("utf-8")
                    if logging:
                        print(f"randomized_smiles1: {randomized_smiles}")
                field_idx += 1

            domain_label = None
            if "k_domains" in molecular_representations:
                domain_label = struct.unpack("B", fields[field_idx])[0]
                field_idx += 1
                if logging:
                    print(f"domain_label: {domain_label}")

            sns_fp = None
            if "sns" in molecular_representations:
                if "sns" == rep:
                    sns_fp = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8), bitorder='little')
                    if logging:
                        print(f"sns_fp: {sns_fp}")
                field_idx += 1
            
            # Read required fields
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

            # Continue reading optional fields
            smiles_ohe = None
            if "smiles" in molecular_representations:
                if "smiles" == rep:
                    smiles_packed = np.frombuffer(fields[field_idx], dtype=np.uint8)
                    smiles_ohe = np.unpackbits(smiles_packed, bitorder='little')
                    x_data.append(smiles_ohe)
                    y_data.append(processed_target)
                    if logging:
                        print(f"smiles_ohe: {smiles_ohe}")
                    continue
                field_idx += 1

            # TODO: figure out what's broken here
            # Need to compare with SMILES, something with the OHE is broken
            randomized_smiles_ohe = []
            if "randomized_smiles" in molecular_representations:
                if "randomized_smiles" == rep:
                    randomized_smiles_packed = np.frombuffer(fields[field_idx], dtype=np.uint8)
                    randomized_smiles_ohe = np.unpackbits(randomized_smiles_packed, bitorder='little')
                    if logging:
                        print(f"randomized_smiles2: {randomized_smiles_ohe}")
                    if len(randomized_smiles_ohe) > 0:
                        x_data.append(randomized_smiles_ohe)
                        y_data.append(processed_target)
                    continue
                field_idx += 1

            ecfp4 = None
            if "ecfp4" in molecular_representations:
                if "ecfp4" == rep:
                    ecfp4 = np.unpackbits(np.frombuffer(fields[field_idx], dtype=np.uint8), bitorder='little')
                    if logging:
                        print(f"ecfp4: {ecfp4}")
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
                field_idx += 1

        except Exception as e:
            # TODO: look into why there's so many of these
            # print(f"Skipping malformed line: {line}")
            continue

    if rep != 'graph':
        x_data = np.vstack(x_data).astype(np.uint8)
    y_data = np.array(y_data, dtype=np.float32)

    return x_data, y_data

def run_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, iteration_seed, rep, iteration, s):
    # try: 
    def black_box_function(trial=None):
        if model_type == 'rf':
            return train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == 'svm':
            return train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == 'xgboost':
            return train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == 'gauche':
            return train_gauche_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == "dnn":
            return train_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == "flexible_dnn":
            return train_flexible_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == "lgb":
            return train_lgb_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type in ["mlp", "residual_mlp", "factorization_mlp", "mtl"]:
            return train_mlp_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, trial)

        elif model_type in ["rnn", "gru"] and rep in ['smiles', 'randomized_smiles']:
            return train_rnn_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, trial)

    if args.tuning:
        study = optuna.create_study(direction="minimize")
        study.optimize(black_box_function, n_trials=args.n_trials)  # Adjust `n_trials` as needed

        best_params = study.best_params
        print(f"Best params for {model_type} and {rep} with sigma {s}: {best_params}")

        # Save the best params as JSON next to the CSV
        if args.filepath:
            json_path = os.path.splitext(args.filepath)[0] + ".json"
            dir_path = os.path.dirname(json_path)
            if dir_path:
                os.makedirs(dir_path, exist_ok=True)

            # Load existing params if file exists
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    all_params = json.load(f)
            else:
                all_params = {}

            # Create nested structure if missing
            if model_type not in all_params:
                all_params[model_type] = {}
            all_params[model_type][rep] = best_params

            # Save updated structure
            with open(json_path, 'w') as f:
                json.dump(all_params, f, indent=4)


        res = black_box_function(optuna.trial.FixedTrial(best_params))

    elif args.params:
        with open(args.params, 'r') as f:
            all_params = json.load(f)

        best_params = all_params[model_type][rep]

        # Reconstruct use_default flags
        fixed_params = {}
        for key, value in best_params.items():
            if value is None:
                fixed_params[f"use_default_{key}"] = True
            else:
                fixed_params[f"use_default_{key}"] = False
                fixed_params[key] = value

        return black_box_function(optuna.trial.FixedTrial(fixed_params))

    else:
        return black_box_function()
    # except Exception as e:
    #    print(f"Error with {rep} and {model_type}; more details: {e}")
    #    # TODO: remove this!!
    #    print(f"x_train: {x_train}")

def qm9_to_networkx(data):
    G = to_networkx(data, to_undirected=True)

    # Add node labels (atomic numbers)
    atomic_numbers = data.x[:, 0].long().tolist()
    for i, atomic_num in enumerate(atomic_numbers):
        G.nodes[i]['label'] = atomic_num

    # Add edge labels (bond types)
    if hasattr(data, 'edge_attr') and data.edge_attr is not None:
        bond_types = data.edge_attr[:, 0].long().tolist()
        edge_list = data.edge_index.t().tolist()
        for idx, (u, v) in enumerate(edge_list):
            if G.has_edge(u, v):
                G[u][v]['label'] = bond_types[idx]
    
    return G

def run_qm9_graph_model(args, qm9, train_idx, test_idx, val_idx, s, iteration):
    for model_type in args.models:
        if model_type == "graph_gp":
            # CASE 2: GraphGP (SIGP subclass over graphs)
            # This is what YOU focus on
            train_set = qm9[train_idx]
            test_set = qm9[test_idx]
            val_set = qm9[val_idx]

            if s > 0:
                noise = torch.normal(mean=0, std=s, size=train_set.data.y.shape)
                train_set.data.y = train_set.data.y + noise
                train_set.data.y = train_set.data.y.to(dtype=torch.float32)

            # 1. Convert PyG to NetworkX graphs
            train_graphs = [qm9_to_networkx(g) for g in train_set]
            test_graphs = [qm9_to_networkx(g) for g in test_set]
            val_graphs = [qm9_to_networkx(g) for g in val_set]

            # 2. Get labels
            y_train = torch.stack([g.y for g in train_set])
            y_test = torch.stack([g.y for g in test_set])
            y_val = torch.stack([g.y for g in val_set])

            # 3. Train GraphGP model
            train_graph_gp(train_graphs, y_train, test_graphs, y_test, val_graphs, y_val, args, s, iteration, trial=None)
        else:
            if model_type == "gin" or model_type == "gin2d":
                model = GIN(dim_h=64)
            elif model_type == "gcn":
                model = GCN(dim_h=128)
            elif model_type == "gin_co_teaching":
                model = GINCoTeaching(dim_h=64)

            # Add label noise
            train_set = qm9[train_idx]
            if s > 0:
                # Generate Gaussian noise with mean 0 and standard deviation s
                noise = torch.normal(mean=0, std=s, size=train_set.data.y.shape)
                
                # Add noise to the original labels
                train_set.data.y = train_set.data.y + noise

                # Ensure the tensor is properly formatted
                train_set.data.y = train_set.data.y.to(dtype=torch.float32)

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
            # else:

            #     train_loss, val_loss, train_target, train_y_target, trained_model = train_epochs_co_teaching(
            #         args.epochs, model, train_loader, test_loader, "GIN_co_teching_model.pt", optimal_co_teaching_hyperparameters['ratio'], optimal_co_teaching_hyperparameters['tolerance'], optimal_co_teaching_hyperparameters['forget_rate']
            #     )

            #     test_loss, test_target, test_y = testing_co_teaching(test_loader, trained_model)

            logging = True
            if args.distribution == "domain_mpnn" or args.distribution == "domain_tanimoto":
                calculate_domain_metrics(test_target, test_y, domain_labels_subset, target_domain)
                logging = False

            metrics = calculate_regression_metrics(test_target, test_y, logging=logging)

            # TODO: iteration is broken!!!
            save_results(args.filepath, s, iteration, model_type, 'graph', args.sample_size, metrics[3], metrics[0], metrics[4])



def process_and_run(args, iteration, iteration_seed, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset=None):
    graph_only = True
    for model in args.models:
        if model not in graph_models:
            graph_only = False 

    train_count = len(train_idx)
    test_count = len(test_idx)

    config = {
        'sample_size': args.sample_size,
        'noise': s > 0,
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'val_count': len(val_idx),
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
                x_val, y_val = parse_mmap(files["val"], len(test_idx), rep, args.molecular_representations, args.k_domains, logging=args.logging)

            except Exception as e:
                print(f"Error with parsing mmap file for {rep} and {model}; more details: {e}")

        for model in args.models:
            if model not in graph_models:
                print(f"model: {model}")
                run_model(
                    x_train, 
                    y_train, 
                    x_test, 
                    y_test, 
                    x_val,
                    y_val,
                    model, 
                    args, 
                    iteration_seed,
                    rep,
                    iteration,
                    s,
                )
            else:
                if args.dataset == 'QM9':
                    run_qm9_graph_model(args, dataset, train_idx, test_idx, val_idx, s, iteration)
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
# TODO: check if things in Cargo.toml are necessary
# TODO: address numerous server warnings 