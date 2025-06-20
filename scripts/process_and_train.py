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
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
from collections import deque
import gc
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
import uuid
import time

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

# TODO: figure out tuning for GINs

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

bit_vectors = ['ecfp4', 'mpnn', 'sns', 'plec', 'pdv']
graph_models = ['gin', 'gcn', 'ginct', 'graph_gp', 'gin2d']
neural_nets = ["dnn", "mlp", "rnn", "gru", 'factorization_mlp', 'residual_mlp']

smiles_db_path = "../data/smiles_db.sqlite"

# Ensure parent directory exists
os.makedirs(os.path.dirname(smiles_db_path), exist_ok=True)

try:
    # Connect to the SQLite db
    conn = sqlite3.connect(smiles_db_path)
    cursor = conn.cursor()

    # Create table if it doesn't exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS smiles_db (
        isomeric TEXT PRIMARY KEY,
        canonical TEXT
    )
    """)
    conn.commit()

except Exception as e:
    conn = None
    cursor = None
    print("Failed to initialize SMILES db")
    print(e)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: make sure everything above function definitions is properly formatted
# TODO: make sure val is working for hyperparameter tuning
# TODO: reformat import statements

# TODO: add argument for classification/regression, then dataset source 
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

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
    parser.add_argument("--tuning", type=str2bool, default=False, help="Hyperparameter tuning (default is False)")
    parser.add_argument("--kernel", type=str, default="tanimoto", help="Specify the kernel for certain models (Gaussian Process)")
    parser.add_argument("-k", "--k_domains", type=int, default=1, help="Number of domains for clustering (default is 1)")
    parser.add_argument("-s", "--split", type=str, default="random", help="Method for splitting data (default is random)")
    parser.add_argument("-c", "--clustering_method", type=str, default="Agglomerative", help="Method to cluster the chemical domain (default is Agglomerative)")
    parser.add_argument("--max_vocab", type=int, default=30, help="Max vocab length of SMILES OHE generation (default is 30)")
    parser.add_argument("--custom_model", type=str, default=None, help="Filepath to custom PyTorch model in .pt file")
    parser.add_argument("--metadata_file", type=str, default=None, help="Filepath to custom model's metadata ie. hyperparameters")
    parser.add_argument("-f", "--filepath", type=str, default='../results/test.csv', help="Filepath to save raw results in csv (default is None)")
    parser.add_argument("--logging", type=str2bool, default=False, help="Extra logging to check individual entries in mmap files (default is False)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training graph-based models (default is 100)")
    parser.add_argument("--clean-smiles", type=str2bool, default=False, help="Clean the SMILES string (default is False)")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of trials in hyperparameter tuning (default is 20)")
    parser.add_argument("-p", "--params", type=str, default=None, help="Filepath for model parameters (default is None)")
    parser.add_argument("-u", "--uncertainty", type=bool, default=False, help="Save uncertainty values for applicable modesl (default is False)")
    parser.add_argument("--shap", type=bool, default=False, help="Calculate SHAP values for relevant tree-based models (default is False)")
    parser.add_argument("--normalize", type=str2bool, default=True, help="Normalize the data before processing (default is True)")    
    parser.add_argument(
        "--bayesian-transformation",
        type=str,
        default=None,
        help=(
            "Apply Bayesian transformation to applicable models (e.g., DNN, MLP). "
            "Options:\n"
            "  full        - Replace all nn.Linear layers with Bayesian layers (BayesLinear).\n"
            "  last_layer  - Replace only the final nn.Linear layer with a Bayesian layer (VBLL-style).\n"
            "  variational - Use variational Bayes for uncertainty (sampling-based, not deterministic).\n"
            "Default is None (no transformation)."
        )
)
    return parser.parse_args()

def write_to_mmap(
    smiles_isomeric,
    smiles_canonical,
    randomized_smiles,
    pdv,
    property_value,
    category,
    files,
    molecular_representations,
    k_domains,
    sns_fp,
    max_vocab,
):
    entry = b""

    # Encode isomeric SMILES with length prefix
    smiles_isomeric_bytes = smiles_isomeric.encode("utf-8")
    entry += struct.pack("I", len(smiles_isomeric_bytes))
    entry += smiles_isomeric_bytes

    # Encode canonical SMILES with length prefix
    smiles_canonical_bytes = smiles_canonical.encode("utf-8")
    entry += struct.pack("I", len(smiles_canonical_bytes))
    entry += smiles_canonical_bytes

    # Encode property value (float)
    entry += struct.pack("f", property_value)

    # Encode randomized SMILES (optional, with length prefix)
    if "randomized_smiles" in molecular_representations:
        if randomized_smiles:
            randomized_smiles_bytes = randomized_smiles.encode("utf-8")
            entry += struct.pack("I", len(randomized_smiles_bytes))
            entry += randomized_smiles_bytes
        else:
            entry += struct.pack("I", 0)  # Zero length = missing

    # SNS fingerprint (packed bits, fixed length)
    if "sns" in molecular_representations:
        if sns_fp is not None:
            sns_fp_array = np.array(sns_fp, dtype=np.uint8)
            sns_fp_packed = np.packbits(sns_fp_array, bitorder='little')
            entry += sns_fp_packed.tobytes()
        else:
            return  # skip incomplete entry

    if "pdv" in molecular_representations:
        if pdv is not None:
            pdv_binary = (pdv > 0).astype(np.uint8)  # or any threshold rule
            pdv_packed = np.packbits(pdv_binary, bitorder='little')
            entry += pdv_packed.tobytes()
        else:
            return

    files[category].write(entry)
    files[category].flush()

# targets: BDR4, HSA, sEH
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
        smiles_canonical = None
        mol = None

        if 'randomized_smiles' not in args.molecular_representations:
            cursor.execute("SELECT canonical FROM smiles_db WHERE isomeric = ?", (smiles_isomeric,))
            result = cursor.fetchone()

            print(f"smiles db result: {result}")

            if result:
                smiles_canonical = result[0]
                mol = Chem.MolFromSmiles(smiles_canonical)
            else:
                mol = Chem.MolFromSmiles(smiles_isomeric)
        else:
            mol = Chem.MolFromSmiles(smiles_isomeric)

        if not mol:
            cursor.execute(
                "INSERT OR REPLACE INTO smiles_db (isomeric, canonical) VALUES (?, ?)",
                (smiles_isomeric, None)
            )
            conn.commit()
            continue

        if 'randomized_smiles' in args.molecular_representations:
            smiles_randomized = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)

        smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        cursor.execute(
            "INSERT OR REPLACE INTO smiles_db (isomeric, canonical) VALUES (?, ?)",
            (smiles_isomeric, smiles_canonical)
        )
        conn.commit()

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

        smiles_canonical = None
        if 'smiles' in args.molecular_representations:
            cursor.execute("SELECT canonical FROM smiles_db WHERE isomeric = ?", (smiles_isomeric,))
            result = cursor.fetchone()
            if result:
                smiles_canonical = result[0]

        if smiles_canonical is None or 'randomized_smiles' in args.molecular_representations:
            mol = Chem.MolFromSmiles(smiles_isomeric)
            if not mol:
                continue

            if not smiles_canonical:
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

        pdv = None
        if 'pdv' in args.molecular_representations:
            pdv = rdkit_mol_descriptors_from_smiles(smiles_canonical)

        if smiles_canonical and not (category == "excluded"):
            if 'randomized_smiles' in args.molecular_representations and not smiles_randomized:
                continue
            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv, data.y.item(), category, files, args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)

    if 'sns' in args.molecular_representations:
        del mols_train

    return train_idx, test_idx, val_idx

def rdkit_mol_descriptors_from_smiles(smiles_string):
    mol_descriptor_calculator = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
    mol = Chem.MolFromSmiles(smiles_string)
    descriptor_vals = mol_descriptor_calculator.CalcDescriptors(mol)
    return np.array(descriptor_vals)

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

    for entry in range(entry_count):
        try:
            feature_vector = []

            # --- isomeric SMILES ---
            iso_len_bytes = mmap_file.read(4)
            iso_len = struct.unpack("I", iso_len_bytes)[0]
            iso_bytes = mmap_file.read(iso_len)
            isomeric_smiles = iso_bytes.decode("utf-8")
            if logging:
                print(f"[{entry}] isomeric_smiles: {isomeric_smiles}")

            # --- canonical SMILES ---
            canon_len_bytes = mmap_file.read(4)
            canon_len = struct.unpack("I", canon_len_bytes)[0]
            canon_bytes = mmap_file.read(canon_len)
            canonical_smiles = canon_bytes.decode("utf-8")
            if logging:
                print(f"[{entry}] canonical_smiles: {canonical_smiles}")

            # --- target value (raw) ---
            target_bytes = mmap_file.read(4)
            target_value = struct.unpack("f", target_bytes)[0]
            if logging:
                print(f"[{entry}] target_value: {target_value}")

            # --- randomized SMILES (length-prefixed) ---
            randomized_smiles = None
            if "randomized_smiles" in molecular_representations:
                rand_len_bytes = mmap_file.read(4)
                rand_len = struct.unpack("I", rand_len_bytes)[0]
                if rand_len > 0:
                    rand_bytes = mmap_file.read(rand_len)
                    randomized_smiles = rand_bytes.decode("utf-8")
                else:
                    rand_bytes = b""
                if logging:
                    print(f"[{entry}] randomized_smiles: {randomized_smiles}")

            # --- sns_fp ---
            if "sns" in molecular_representations:
                sns_bytes = mmap_file.read(128)
                if rep == "sns":
                    sns_fp = np.unpackbits(np.frombuffer(sns_bytes, dtype=np.uint8), bitorder="little")
                    feature_vector.append(sns_fp)
                    if logging:
                        print(f"[{entry}] sns_fp: {sns_fp}")
            
            # --- pdv ---
            pdv = None
            if "pdv" in molecular_representations:
                pdv_bytes = mmap_file.read(25)
                if "pdv" == rep:
                    pdv = np.unpackbits(np.frombuffer(pdv_bytes, dtype=np.uint8), bitorder="little")
                    feature_vector.append(pdv)
                    if logging: 
                        print(f"pdv: {pdv}")

            # --- processed target ---
            processed_bytes = mmap_file.read(4)
            processed_target = struct.unpack("f", processed_bytes)[0]
            if logging:
                print(f"[{entry}] processed_target: {processed_target}")

            # --- domain label ---
            if k_domains > 1:
                domain_byte = mmap_file.read(1)
                if logging:
                    print(f"[{entry}] domain_flag bytes: {[f'{b:02X}' for b in domain_byte]}")

            # --- sns_fp ---
            if rep == "sns" or rep == "pdv":
                x_data.append(np.concatenate([f for f in feature_vector if f is not None]))
                y_data.append(processed_target)

            # --- SMILES OHE ---
            if "smiles" in molecular_representations:
                ohe_len_bytes = mmap_file.read(4)
                ohe_len = struct.unpack("I", ohe_len_bytes)[0]
                packed = mmap_file.read(ohe_len)
                if rep == "smiles":
                    smiles_ohe = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
                    x_data.append(smiles_ohe)
                    y_data.append(processed_target)
                    if logging:
                        print(f"[{entry}] smiles_ohe: {smiles_ohe}")

            # --- randomized SMILES OHE ---
            if "randomized_smiles" in molecular_representations:
                ohe_len_bytes = mmap_file.read(4)
                ohe_len = struct.unpack("I", ohe_len_bytes)[0]
                packed = mmap_file.read(ohe_len)
                if rep == "randomized_smiles":
                    rand_ohe = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
                    x_data.append(rand_ohe)
                    y_data.append(processed_target)
                    if logging:
                        print(f"[{entry}] randomized_ohe: {rand_ohe}")

            # --- ECFP4 fingerprint ---
            if "ecfp4" in molecular_representations:
                raw_bytes = mmap_file.read(256)
                if rep == "ecfp4":
                    ecfp4_packed = np.frombuffer(raw_bytes, dtype=np.uint8)
                    ecfp4 = np.unpackbits(ecfp4_packed, bitorder="little")
                    feature_vector.append(ecfp4)
                    x_data.append(np.concatenate([f for f in feature_vector if f is not None]))
                    y_data.append(processed_target)
                    if logging:
                        print(f"[{entry}] ecfp4: {ecfp4}")

            # --- graph fallback ---
            if rep == "graph":
                x_data.append(entry)
                y_data.append(processed_target)
                continue

        except Exception as e:
            if logging:
                print(f"[{entry}] Skipping malformed entry: {e}")
            continue

    if rep != "graph":
        x_data = np.vstack(x_data).astype(np.uint8)
    y_data = np.array(y_data, dtype=np.float32)

    return x_data, y_data

# TODO: pdv
def run_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, iteration_seed, rep, iteration, s):
    def _black_box_function(trial):
        print(f"Running Optuna trial {trial.number}")
        return model_selector(trial)

    def model_selector(trial=None):
        if model_type in ['rf', 'qrf']:
            return train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, model_type, trial)

        elif model_type == 'svm':
            return train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == 'xgboost':
            return train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)
        
        elif model_type == 'ngboost':
            return train_ngboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

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
        temp_study_name = f"temp_qspr_{uuid.uuid4().hex}"
        study = optuna.create_study(
            direction="maximize",
            storage="sqlite:///optuna_study.db",
            study_name=temp_study_name,
            load_if_exists=False,
        )

        study.optimize(_black_box_function, n_trials=args.n_trials, show_progress_bar=True)

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

        res = _black_box_function(optuna.trial.FixedTrial(best_params))
        optuna.delete_study(study_name=study.study_name, storage="sqlite:///optuna_study.db")

    elif args.params:
        with open(args.params, 'r') as f:
            all_params = json.load(f)

        # PATCHED VERSION
        if model_type in all_params and rep in all_params[model_type]:
            best_params = all_params[model_type][rep]

            # Reconstruct use_default flags
            fixed_params = {}
            for key, value in best_params.items():
                if value is None:
                    fixed_params[f"use_default_{key}"] = True
                else:
                    fixed_params[f"use_default_{key}"] = False
                    fixed_params[key] = value

            return _black_box_function(optuna.trial.FixedTrial(fixed_params))
        else:
            print(f"No saved parameters for model_type '{model_type}' and rep '{rep}'. Using default settings.")
            return model_selector()

    else:
        return model_selector()

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
    def _black_box_function(trial, model_type):
        print(f"Running Optuna trial {trial.number} for {model_type}")
        return model_selector(trial, model_type)

    def model_selector(trial, model_type):
        # try: 
        if model_type == "graph_gp":
            # CASE 2: GraphGP (SIGP subclass over graphs)
            train_set = qm9[train_idx]
            test_set = qm9[test_idx]
            val_set = qm9[val_idx]
            if s > 0:
                noise = torch.normal(mean=0, std=s, size=train_set.data.y.shape)
                train_set.data.y = train_set.data.y + noise
                train_set.data.y = train_set.data.y.to(dtype=torch.float32)
            # Convert PyG to NetworkX graphs
            train_graphs = [qm9_to_networkx(g) for g in train_set]
            test_graphs = [qm9_to_networkx(g) for g in test_set]
            val_graphs = [qm9_to_networkx(g) for g in val_set]
            # Get labels
            y_train = torch.stack([g.y for g in train_set])
            y_test = torch.stack([g.y for g in test_set])
            y_val = torch.stack([g.y for g in val_set])
            # Normalize
            if args.normalize:
                mean = y_train.mean()
                std = y_train.std()
                y_train = (y_train - mean) / std
                y_test = (y_test - mean) / std
                y_val = (y_val - mean) / std
            # Train GraphGP model
            return train_graph_gp(train_graphs, y_train, test_graphs, y_test, val_graphs, y_val, args, s, iteration, trial=trial)
        else:
            # Add label noise
            train_set = qm9[train_idx]
            if s > 0:
                # Generate Gaussian noise with mean 0 and standard deviation s
                noise = torch.normal(mean=0, std=s, size=train_set.data.y.shape)
                
                # Add noise to the original labels
                train_set.data.y = train_set.data.y + noise
                # Ensure the tensor is properly formatted
                train_set.data.y = train_set.data.y.to(dtype=torch.float32)
            # Normalize
            if args.normalize:
                mean = train_set.data.y.mean()
                std = train_set.data.y.std()
                train_set.data.y = (train_set.data.y - mean) / std
                for i in test_idx:
                    qm9[i].y = (qm9[i].y - mean) / std
                for i in val_idx:
                    qm9[i].y = (qm9[i].y - mean) / std
            # datasets into DataLoader
            train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
            test_loader = DataLoader(qm9[test_idx], batch_size=64, shuffle=True)
            val_loader = DataLoader(qm9[val_idx], batch_size=64, shuffle=True)
            return train_gnn(model_type, train_loader, test_loader, val_loader, args, s, iteration, trial)
        
        # except Exception as e:
        #     print(f"Error with graph and {model_type}; more details: {e}")
        #     return None

    # Main execution loop
    for model_type in args.models:
        if args.tuning:
            temp_study_name = f"temp_qspr_graph_{uuid.uuid4().hex}"
            study = optuna.create_study(
                direction="maximize",
                storage="sqlite:///optuna_study.db",
                study_name=temp_study_name,
                load_if_exists=False,
            )

            # Create a wrapper function for this specific model_type
            def objective(trial):
                return _black_box_function(trial, model_type)

            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

            # Check if we have any successful trials
            if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
                print(f"No successful trials for {model_type}. Running with default parameters.")
                res = model_selector(None, model_type)
            else:
                best_params = study.best_params
                print(f"Best params for {model_type} with sigma {s}: {best_params}")

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
                    all_params[model_type]['graph'] = best_params  # Using 'graph' as the representation type

                    # Save updated structure
                    with open(json_path, 'w') as f:
                        json.dump(all_params, f, indent=4)

                # Run with best params
                res = _black_box_function(optuna.trial.FixedTrial(best_params), model_type)
            
            optuna.delete_study(study_name=study.study_name, storage="sqlite:///optuna_study.db")

        elif args.params:
            with open(args.params, 'r') as f:
                all_params = json.load(f)

            # PATCHED VERSION
            if model_type in all_params and 'graph' in all_params[model_type]:
                best_params = all_params[model_type]['graph']

                # Reconstruct use_default flags
                fixed_params = {}
                for key, value in best_params.items():
                    if value is None:
                        fixed_params[f"use_default_{key}"] = True
                    else:
                        fixed_params[f"use_default_{key}"] = False
                        fixed_params[key] = value

                res = _black_box_function(optuna.trial.FixedTrial(fixed_params), model_type)
            else:
                print(f"No saved parameters for model_type '{model_type}' and rep 'graph'. Using default settings.")
                res = model_selector(None, model_type)

        else:
            res = model_selector(None, model_type)

def process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset=None):
    train_count = len(train_idx)
    test_count = len(test_idx)

    print(f"normalising: {args.normalize}")

    config = {
        'sample_size': args.sample_size,
        'noise': s > 0,
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'val_count': len(val_idx),
        'max_vocab': args.max_vocab,
        'file_no': file_no,
        'molecular_representations': args.molecular_representations,
        'k_domains': args.k_domains,
        'logging': args.logging,
        'regression': args.dataset == 'QM9',
        'normalize': args.normalize,
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

    files = {
        "train": open('train_' + str(file_no) + '.mmap', 'rb'),
        "test": open('test_' + str(file_no) + '.mmap', 'rb'),
        "val": open('val_' + str(file_no) + '.mmap', 'rb'),
    }

    if 'graph' in args.molecular_representations:
        run_qm9_graph_model(args, dataset, train_idx, test_idx, val_idx, s, iteration)

    # Read mmap files and train/test models for all molecular representations
    for rep in args.molecular_representations:
        if rep != "graph":
            try: 
                for model in args.models:
                    if model not in graph_models:
                        # Reset mmap pointers
                        for file in files.values():
                            file.seek(0)

                        x_train, y_train = parse_mmap(files["train"], len(train_idx), rep, args.molecular_representations, args.k_domains, logging=args.logging)
                        x_test, y_test = parse_mmap(files["test"], len(test_idx), rep, args.molecular_representations, args.k_domains, logging=args.logging)
                        x_val, y_val = parse_mmap(files["val"], len(test_idx), rep, args.molecular_representations, args.k_domains, logging=args.logging)

                        print(f"model: {model}")
                        print(f"rep: {rep}")
                        run_model(
                            x_train, y_train, x_test, y_test, x_val, y_val,
                            model, args, iteration_seed, rep, iteration, s,
                        )
            except Exception as e:
                print(f"Error with {rep} and {model}; more details: {e}")

    for key in list(files.keys()):
        filename = f"{key}_{file_no}.mmap"
        files[key].close()
        os.remove(filename)  # <-- deletes the actual file
        del files[key]
    files.clear()
    gc.collect()

def main():
    start_time = time.time()
    args = parse_arguments()

    # Prepare for communication with Rust
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"  # Enable Rust backtraces for debugging

    rust_executable_path = os.path.join(base_dir, '../rust/target/release/rust_processor')

    qm9 = None
    if args.dataset == 'QM9':
        qm9 = load_qm9(args.target)
        print("QM9 loaded")

    sigma_time = time.time()
    for s in args.sigma:
        s = float(s)
        print(f"Sigma: {s}")

        for iteration in range(args.bootstrapping):
            # Set seeds
            iteration_seed = (args.random_seed ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF  # XOR and mask for 32-bit seed
            random.seed(iteration_seed)
            np.random.seed(iteration_seed)
            torch.manual_seed(iteration_seed)
            file_no = (iteration_seed ^ int(time.time() * 1e6)) & 0xFFFFFFFF

            files = {
                "train": open('train_' + str(file_no) + '.mmap', 'wb+'),
                "test": open('test_' + str(file_no) + '.mmap', 'wb+'),
                "val": open('val_' + str(file_no) + '.mmap', 'wb+'),
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
            process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset=qm9)

        current_time = time.time()
        print(f"Time for sigma {s}: {current_time - sigma_time:.2f} seconds")
        sigma_time = current_time

    print(f"Time for total run: {time.time() - start_time}")

if __name__ == "__main__":
    main()

# TODO: add polaris login to README
# TODO: sanity check to see if val set/epochs can be used for any others
# What's the best practice - if RF doesn't use a val set should the val be merged with training? 
# TODO: properly format all print statements
# TODO: check if things in Cargo.toml are necessary
# TODO: address numerous server warnings 
