import argparse
import os
import os.path as osp
import random
import json
import subprocess
import struct
import warnings
import traceback
import numpy as np
import pandas as pd
import csv
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader as GeometricDataLoader
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
from gensim.models import word2vec
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec

import sys
sys.path.append('../models/')
sys.path.append('../preprocessing/')
sys.path.append('../results/')

from models import *
from distance_metrics import *
from extract_and_cluster_for_domains import extract_and_cluster_for_domains
from hybrid_representation import create_hybrid_representation
from hybrid_diagnostics import *

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
valid_indices_path = os.path.join(data_dir, 'valid_qm9_indices.pth')

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

# Global cache for ChemBERTa model (loaded once, reused)
_CHEMBERTA_MODEL = None
_CHEMBERTA_TOKENIZER = None
_MHG_GNN_MODEL = None


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

properties = {
    'homo_lumo_gap': 4, 'alpha': 1, 'G': 10, 'H': 9, 'U': 8,
    'G_a': 15, 'H_a': 14, 'U_a': 13, 'mu': 0, 'A': 16, 'B': 17, 'C': 18
}

bit_vectors = ['ecfp4', 'mpnn', 'sns', 'plec', 'pdv', 'smiles', 'randomized_smiles', 'continuous_pdv', 'mol2vec', 'chemberta', 'mhggnn']
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

ADME_TARGETS = {
    'hlm': 'LOG_HLM_CLint',
    'rlm': 'LOG_RLM_CLint',
    'solubility': 'LOG_SOLUBILITY',
    'mdr1': 'LOG_MDR1-MDCK_ER',
    'hppb': 'LOG_HPPB',
    'rppb': 'LOG_RPPB'
}

def load_adme(target):
    """Load ADME dataset from Polaris Hub"""
    import polaris as po
    
    if target not in ADME_TARGETS:
        raise ValueError(f"Unknown ADME target: {target}. Valid targets: {list(ADME_TARGETS.keys())}")
    
    target_column = ADME_TARGETS[target]
    
    print(f"Loading ADME dataset with target: {target_column}")
    dataset = po.load_dataset("biogen/adme-fang-v1")
    
    # Find SMILES column
    smiles_col = None
    for col in dataset.columns:
        if 'smiles' in col.lower():
            smiles_col = col
            break
    
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Available: {dataset.columns}")
    
    print(f"SMILES column: {smiles_col}")
    print(f"Dataset size: {dataset.n_rows}")
    
    return dataset, smiles_col, target_column

def load_moleculenet(task_name):
    """
    Load MoleculeNet dataset via DeepChem.
    
    Args:
        task_name: One of: esol, freesolv, lipo, qm7, qm8, bace, bbbp, clintox, hiv
    
    Returns:
        (dataset, smiles_col, target_col) - same as ADME loader
    """
    import deepchem as dc
    
    # Map task names to DeepChem loader functions
    LOADERS = {
        # Regression
        'esol': dc.molnet.load_delaney,
        'freesolv': dc.molnet.load_sampl,
        'lipo': dc.molnet.load_lipo,
        'qm7': dc.molnet.load_qm7,
        'qm8': dc.molnet.load_qm8,
        # Classification  
        'bace': dc.molnet.load_bace_classification,
        'bbbp': dc.molnet.load_bbbp,
        'clintox': dc.molnet.load_clintox,
        'hiv': dc.molnet.load_hiv,
    }
    
    if task_name not in LOADERS:
        raise ValueError(f"Unknown MoleculeNet task: {task_name}. "
                        f"Available: {list(LOADERS.keys())}")
    
    print(f"Loading MoleculeNet: {task_name}...")
    
    # Load with default splitter first to get all data
    loader = LOADERS[task_name]
    tasks, datasets, transformers = loader(
        featurizer='Raw',
        splitter='scaffold',  # Use scaffold split to get 3-tuple
        reload=False
    )
    
    print(f"DEBUG: type(datasets) = {type(datasets)}")
    print(f"DEBUG: len(datasets) = {len(datasets)}")
    
    # Now merge the splits into one dataset
    if isinstance(datasets, tuple) and len(datasets) == 3:
        full_dataset = dc.data.DiskDataset.merge([datasets[0], datasets[1], datasets[2]])
    else:
        # Fallback: if it's already a single dataset
        full_dataset = datasets[0] if isinstance(datasets, tuple) else datasets
    
    print(f"Loaded {len(full_dataset)} molecules with tasks: {tasks}")
    
    # Return in same format as load_adme: (dataset, smiles_col, target_col)
    # For DeepChem, SMILES are in .ids and we'll use first task
    return (full_dataset, 'ids', tasks[0])

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
    parser.add_argument("-u", "--uncertainty", type=str2bool, default=False, help="Save uncertainty values for applicable modesl (default is False)")
    parser.add_argument("--shap", type=str2bool, default=False, help="Calculate SHAP values for relevant tree-based models (default is False)")
    parser.add_argument("--normalize", type=str2bool, default=True, help="Normalize the data before processing (default is True)")   
    parser.add_argument("--noise-strategy", type=str, default="legacy", help="Noise strategy: legacy, value_proportional, quantile, threshold, outlier, heteroscedastic (default is legacy)")
    parser.add_argument("--strategy-params", type=str, default="noise_strategy_params.json", help="JSON file path with strategy-specific parameters (default is noise_strategy_params.json)")
    parser.add_argument("--save-per-epoch-metrics", type=str2bool, default=False, help='Save training/validation loss for each epoch')
    parser.add_argument('--cp-base-model', type=str, default='rf',
                    choices=['rf', 'xgboost', 'dnn', 'qrf', 'gauche', 'gin', 'gcn'],
                    help='Base model for conformal prediction')
    parser.add_argument('--use-best-params', action='store_true')
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
    parser.add_argument("--calibration-size", type=int, default=20,
                       help="Percentage of validation set for conformal calibration (default is 20)")
    parser.add_argument("--domain-method", type=str, default='none',
                        choices=['none', 'random', 'fingerprint_kmeans', 'descriptor', 
                                 'butina', 'splito', 'scaffold', 'molecular_weight'],
                        help="Method for domain clustering")
    parser.add_argument("--domain-representation", type=str, default='ecfp4',
                        choices=['ecfp4', 'sns', 'pdv', 'mol2vec'],
                        help="Representation for domain clustering")
    parser.add_argument("--loss", type=str, default='mse',
                   choices=['mse', 'mae', 'smooth_l1', 'huber', 'cauchy', 'log_cosh',
                           'focal', 'truncated',
                           'quantile_0.1', 'quantile_0.5', 'quantile_0.9',
                           'heteroscedastic', 'evidential', 'barron', 
                           'domain_weighted', 'domain_balanced', 'het_per_domain',
                           'adaptive_domain', 'mixture_domain', 'evidential_cauchy', 
                           'evidential_laplace', 'sample_adaptive_barron', 
                           'stratified'],
                   help="Loss function to use (default is mse)")
    parser.add_argument("--use-uncertainty-weighting", type=str2bool, default=False,
                   help="Use uncertainty in addition to loss for sample weighting (default: False)")
    parser.add_argument("--distance-metric", type=str, default='tanimoto',
                   choices=['tanimoto', 'euclidean', 'cosine', 'mahalanobis', 'mmd', 'optimal_transport'],
                   help="Distance metric for molecular similarity (default: tanimoto)")

    parser.add_argument("--use-distance", type=str2bool, default=False,
                       help="Use distance metrics in sample selection (default: False)")
    parser.add_argument("--alpha", nargs='*', type=float, default=[0.1],
                       help="Confidence levels for conformal prediction (default is [0.1])")
    parser.add_argument("--mol2vec-dim", type=int, default=300,
                       help="Dimensions for mol2vec embeddings (default is 300)")
    parser.add_argument("--mol2vec-model", type=str, default=None,
                       help="Path to pre-trained mol2vec model (default: data/model_300dim.pkl)")
    parser.add_argument("--create-hybrid", type=str2bool, default=False)
    parser.add_argument("--hybrid-n-per-rep", type=int, default=100)
    parser.add_argument("--hybrid-importance", type=str, default="shap",
                       choices=['shap', 'random_forest', 'mutual_info', 'correlation', 'lasso'])
    parser.add_argument("--hybrid-normalize", type=str, default="standard",
                       choices=['standard', 'minmax', 'none'])
    parser.add_argument("--save-hybrid-analysis", type=str2bool, default=False,
                   help="Save hybrid feature rankings and diagnostics")
    return parser.parse_args()

def write_to_mmap(
    smiles_isomeric,
    smiles_canonical,
    randomized_smiles,
    pdv,
    continuous_pdv,
    mol2vec,
    chemberta,
    mhggnn,
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

    if "continuous_pdv" in molecular_representations:
        if continuous_pdv is not None:
            continuous_pdv_fp16 = continuous_pdv.astype(np.float16)
            entry += continuous_pdv_fp16.tobytes()
        else:
            return

    if "mol2vec" in molecular_representations:
        if mol2vec is not None:
            # Already quantized in mol2vec_fingerprint(), just write it
            entry += mol2vec.tobytes()
        else:
            return

    if "chemberta" in molecular_representations:
        if chemberta is not None:
            entry += chemberta.tobytes()
        else:
            return

    if "mhggnn" in molecular_representations:
        if mhggnn is not None:
            entry += mhggnn.tobytes()
        else:
            return

    files[category].write(entry)
    files[category].flush()

def load_and_split_polaris(dataset_tuple, args, files):
    """
    Now handles both ADME (Polaris) and MoleculeNet (DeepChem) datasets
    """
    dataset, smiles_col, target_col = dataset_tuple
    
    # Check if this is DeepChem dataset (MoleculeNet) or Polaris dataset (ADME)
    is_deepchem = hasattr(dataset, 'ids')  # DeepChem datasets have .ids attribute
    
    if is_deepchem:
        # MoleculeNet path
        n_total = min(args.sample_size, len(dataset))
        
        smiles_list = dataset.ids[:n_total]
        
        # Handle single-task vs multi-task
        if len(dataset.y.shape) == 1:
            targets = dataset.y[:n_total]
        else:
            targets = dataset.y[:n_total, 0]  # Use first task
        
        # Filter valid
        valid_smiles = []
        valid_targets = []
        for i in range(len(smiles_list)):
            smi = smiles_list[i]
            target = targets[i]
            
            if smi is None or target is None:
                continue
            if isinstance(target, (float, np.floating)) and np.isnan(target):
                continue
            
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            
            valid_smiles.append(str(smi))
            valid_targets.append(float(target))
        
        smiles_list = valid_smiles
        target_list = valid_targets
        
    else:
        # ADME/Polaris path (your existing code)
        table = dataset.table
        
        print(f"Available columns: {table.columns.tolist()}")
        print(f"SMILES column: {smiles_col}")
        print(f"Target column: {target_col}")
        
        n_total = min(args.sample_size, len(table))
        
        smiles_list = []
        target_list = []
        
        for i in range(n_total):
            try:
                smiles = table[smiles_col][i]
                target = table[target_col][i]
                
                if smiles is None or target is None:
                    continue
                if isinstance(target, (float, np.floating)) and np.isnan(target):
                    continue
                
                smiles_list.append(str(smiles))
                target_list.append(float(target))
            except Exception as e:
                print(f"Row {i} error: {e}")
                continue
    
    print(f"Total valid molecules: {len(smiles_list)}")
    
    # Split (same for both)
    n_valid = len(smiles_list)
    
    if args.split == 'random':
        train_end = int(n_valid * 0.8)
        val_end = int(n_valid * 0.9)
        train_idx = list(range(train_end))
        val_idx = list(range(train_end, val_end))
        test_idx = list(range(val_end, n_valid))
    
    elif args.split == 'scaffold':
        Xs = np.zeros(n_valid).reshape(-1, 1)
        dc_dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=smiles_list)
        splitter = dc.splits.ScaffoldSplitter()
        train_idx, val_idx, test_idx = splitter.split(dc_dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
    
    # Prepare SNS
    mols_train = deque()
    ecfp_featuriser = None
    
    if 'sns' in args.molecular_representations:
        for idx in train_idx:
            mol = Chem.MolFromSmiles(smiles_list[idx])
            if mol:
                mols_train.append(mol)
        ecfp_featuriser = create_sort_and_slice_ecfp_featuriser(
            mols_train=mols_train, max_radius=2, pharm_atom_invs=False,
            bond_invs=True, chirality=False, sub_counts=True,
            vec_dimension=1024, print_train_set_info=args.logging
        )
    
    # Load mol2vec
    mol2vec_model = None
    mol2vec_dimensions = args.mol2vec_dim
    if 'mol2vec' in args.molecular_representations:
        model_path = get_mol2vec_model_path(args)
        mol2vec_model = load_mol2vec_model(model_path)

    # Load mhggnn
    if 'mhggnn' in args.molecular_representations:
        _ = get_mhg_gnn_model()  # Load once
    
    print("Writing to mmap...")
    
    # Write to mmap
    for local_idx in range(n_valid):
        category = "excluded"
        if local_idx in train_idx:
            category = "train"
        elif local_idx in test_idx:
            category = "test"
        elif local_idx in val_idx:
            category = "val"
        
        if category == "excluded":
            continue
        
        smiles_isomeric = smiles_list[local_idx]
        mol = Chem.MolFromSmiles(smiles_isomeric)
        if not mol:
            continue
        
        smiles_canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
        if not smiles_canonical:
            continue
        
        smiles_randomized = None
        if 'randomized_smiles' in args.molecular_representations:
            smiles_randomized = Chem.MolToSmiles(mol, isomericSmiles=False, doRandom=True)
        
        sns_fp = None
        if 'sns' in args.molecular_representations:
            if local_idx in train_idx and mols_train:
                mol = mols_train.popleft()
            sns_fp = ecfp_featuriser(mol)
        
        pdv = None
        if 'pdv' in args.molecular_representations:
            pdv = rdkit_mol_descriptors_from_smiles(smiles_canonical)
        
        continuous_pdv = None
        if 'continuous_pdv' in args.molecular_representations:
            if 'pdv' in args.molecular_representations:
                continuous_pdv = pdv
            else:
                continuous_pdv = rdkit_mol_descriptors_from_smiles(smiles_canonical)
        
        mol2vec = None
        if 'mol2vec' in args.molecular_representations:
            mol2vec = mol2vec_fingerprint(mol, mol2vec_model, dimensions=mol2vec_dimensions)
        
        chemberta = None
        if 'chemberta' in args.molecular_representations:
            chemberta = chemberta_fingerprint(smiles_canonical, dimensions=768)

        mhggnn = None
        if 'mhggnn' in args.molecular_representations:
            mhggnn = mhggnn_fingerprint(smiles_canonical, dimensions=1024)
        
        write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv, continuous_pdv, mol2vec, chemberta, mhggnn,
                     target_list[local_idx], category, files,
                     args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)
    
    if 'sns' in args.molecular_representations:
        del mols_train
    
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

    # Load mol2vec model if needed
    mol2vec_model = None
    mol2vec_dimensions = args.mol2vec_dim
    if 'mol2vec' in args.molecular_representations or 'mol2vec' in args.molecular_representations:
        model_path = get_mol2vec_model_path(args)
        mol2vec_model = load_mol2vec_model(model_path)

    # Load mhg-gnn
    if 'mhggnn' in args.molecular_representations:
        _ = get_mhg_gnn_model()  # Load once

    successful_train_idx = []
    successful_test_idx = []
    successful_val_idx = []

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

        continuous_pdv = None
        if 'continuous_pdv' in args.molecular_representations:
            if 'pdv' in args.molecular_representations:
                continuous_pdv = pdv
            else:
                continuous_pdv = rdkit_mol_descriptors_from_smiles(smiles_canonical)

        mol2vec = None
        if 'mol2vec' in args.molecular_representations:
            mol2vec = mol2vec_fingerprint(mol, mol2vec_model, 
                                               dimensions=mol2vec_dimensions)

        chemberta = None
        if 'chemberta' in args.molecular_representations:
            chemberta = chemberta_fingerprint(smiles_canonical, dimensions=768)

        mhggnn = None
        if 'mhggnn' in args.molecular_representations:
            mhggnn = mhggnn_fingerprint(smiles_canonical, dimensions=1024)

        if smiles_canonical and not (category == "excluded"):
            if 'randomized_smiles' in args.molecular_representations and not smiles_randomized:
                continue
            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv, continuous_pdv, mol2vec, chemberta, mhggnn, data.y.item(), category, files, args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)

            if category == "train":
                successful_train_idx.append(index)
            elif category == "test":
                successful_test_idx.append(index)
            elif category == "val":
                successful_val_idx.append(index)

    if 'sns' in args.molecular_representations:
        del mols_train

    return successful_train_idx, successful_test_idx, successful_val_idx

def get_chemberta_model():
    """Load ChemBERTa model once and cache it globally"""
    global _CHEMBERTA_MODEL, _CHEMBERTA_TOKENIZER
    
    if _CHEMBERTA_MODEL is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        print("Loading ChemBERTa model (one-time, ~30 seconds)...")
        model_name = "seyonec/ChemBERTa-zinc-base-v1"
        _CHEMBERTA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _CHEMBERTA_MODEL = AutoModel.from_pretrained(model_name)
        _CHEMBERTA_MODEL.eval()
        
        if torch.cuda.is_available():
            _CHEMBERTA_MODEL = _CHEMBERTA_MODEL.cuda()
            print("ChemBERTa loaded on GPU")
        else:
            print("ChemBERTa loaded on CPU")
    
    return _CHEMBERTA_TOKENIZER, _CHEMBERTA_MODEL

def chemberta_fingerprint(smiles, dimensions=768):
    """
    Generate ChemBERTa embedding for SMILES string.
    Uses globally cached model.
    """
    import torch
    
    try:
        tokenizer, model = get_chemberta_model()
        
        # Tokenize
        inputs = tokenizer(smiles, return_tensors="pt", padding=True,
                          truncation=True, max_length=512)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate embedding
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            
            if torch.cuda.is_available():
                embedding = embedding.cpu()
            embedding = embedding.numpy()
        
        # Ensure 768 dimensions
        if len(embedding) != dimensions:
            if len(embedding) < dimensions:
                embedding = np.pad(embedding, (0, dimensions - len(embedding)), mode='constant')
            else:
                embedding = embedding[:dimensions]
        
        # Quantize to uint8
        vec_min, vec_max = embedding.min(), embedding.max()
        if vec_max - vec_min > 1e-6:
            vec_normalized = (embedding - vec_min) / (vec_max - vec_min)
            return (vec_normalized * 255).astype(np.uint8)
        else:
            return np.zeros(dimensions, dtype=np.uint8)
            
    except Exception as e:
        print(f"ChemBERTa error: {e}")
        return np.zeros(dimensions, dtype=np.uint8)

def mhggnn_fingerprint(smiles, dimensions=1024):
    """Generate MHG-GNN embedding for SMILES string"""
    try:
        model = get_mhg_gnn_model()
        
        # Encode returns list of tensors
        embedding = model.encode([smiles])[0]
        embedding = embedding.cpu().detach().numpy()
        
        # Quantize to uint8
        vec_min, vec_max = embedding.min(), embedding.max()
        if vec_max - vec_min > 1e-6:
            vec_normalized = (embedding - vec_min) / (vec_max - vec_min)
            return (vec_normalized * 255).astype(np.uint8)
        else:
            return np.zeros(dimensions, dtype=np.uint8)
            
    except Exception as e:
        print(f"MHG-GNN error: {e}")
        return np.zeros(dimensions, dtype=np.uint8)

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

def get_mol2vec_model_path(args):
    """Get mol2vec model path from args or default location"""
    if args.mol2vec_model:
        return args.mol2vec_model
    return os.path.join(data_dir, 'model_300dim.pkl')

def load_mol2vec_model(model_path):
    """Load pre-trained mol2vec model"""
    if not os.path.exists(model_path):
        print(f"\n{'='*70}")
        print(f"ERROR: mol2vec model not found at {model_path}")
        print(f"{'='*70}")
        print("\nPlease download the pre-trained model:")
        print("1. Visit: https://github.com/samoturk/mol2vec")
        print("2. Download 'model_300dim.pkl'")
        print(f"3. Place it at: {model_path}")
        print(f"{'='*70}\n")
        raise FileNotFoundError(f"mol2vec model not found at {model_path}")
    
    print(f"Loading mol2vec model from {model_path}...")
    model = word2vec.Word2Vec.load(model_path)
    print("mol2vec model loaded successfully")
    return model

def mol2vec_fingerprint(mol, model, dimensions):
    """
    Generate mol2vec embedding for molecule.
    Compatible with both Gensim 3.x and 4.x
    
    Args:
        mol: RDKit molecule object
        model: Pre-trained mol2vec word2vec model
        dimensions: Embedding dimensions (typically 300)
    
    Returns:
        numpy array: mol2vec embedding as uint8 
    """
    try:
        # Generate sentence from molecule
        sentence = MolSentence(mol2alt_sentence(mol, radius=1))
        
        # FIXED: Manual implementation to work with Gensim 4.x
        # Instead of using sentences2vec, we'll do it ourselves
        vec = np.zeros(model.wv.vector_size)
        count = 0
        
        for word in sentence:
            if word in model.wv:  # Check if word is in vocabulary
                vec += model.wv[word]  # Get vector for word
                count += 1
        
        if count > 0:
            vec = vec / count  # Average the vectors
        else:
            # If no words found, use a default vector or zeros
            vec = np.zeros(model.wv.vector_size)
        
        # Ensure correct dimensions
        if len(vec) != dimensions:
            print(f"Warning: mol2vec returned {len(vec)} dims, expected {dimensions}")
            if len(vec) < dimensions:
                vec = np.pad(vec, (0, dimensions - len(vec)), mode='constant')
            else:
                vec = vec[:dimensions]
        
        # Quantize to uint8: scale to [0, 255]
        vec_min, vec_max = vec.min(), vec.max()
        if vec_max - vec_min > 1e-6:  # Avoid division by zero
            vec_normalized = (vec - vec_min) / (vec_max - vec_min)
            vec_uint8 = (vec_normalized * 255).astype(np.uint8)
        else:
            vec_uint8 = np.zeros(dimensions, dtype=np.uint8)
        return vec_uint8
            
    except Exception as e:
        print(f"Error generating mol2vec: {e}")
        import traceback
        traceback.print_exc()
        return np.zeros(dimensions, dtype=np.uint8)

def get_mhg_gnn_model(materials_repo_path=None, model_pickle_path=None):
    """Load MHG-GNN model once and cache globally"""
    global _MHG_GNN_MODEL
    
    if _MHG_GNN_MODEL is None:
        import sys
        import pickle
        
        print("Loading MHG-GNN model (one-time)...")
        
        # Find materials repo
        if materials_repo_path is None:
            search_paths = [
                os.path.expanduser('~/repos/materials'),
                os.path.join(data_dir, 'materials'),
                '../materials',
            ]
            for path in search_paths:
                if os.path.exists(os.path.join(path, 'models', 'mhg_model')):
                    materials_repo_path = path
                    break
            
            if materials_repo_path is None:
                raise RuntimeError("MHG-GNN: materials repo not found. Clone from https://github.com/IBM/materials.git")
        
        # Add to path
        models_path = os.path.join(materials_repo_path, 'models')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        from mhg_model.load import PretrainedModelWrapper
        
        # Find pickle
        if model_pickle_path is None:
            search_paths = [
                os.path.expanduser('~/repos/materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle'),
                os.path.join(data_dir, 'mhggnn_pretrained_model_0724_2023.pickle'),
            ]
            for path in search_paths:
                if os.path.exists(path):
                    model_pickle_path = path
                    break
            
            if model_pickle_path is None:
                raise RuntimeError("MHG-GNN model pickle not found. Download from https://huggingface.co/ibm-research/materials.mhg-ged")
        
        with open(model_pickle_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        _MHG_GNN_MODEL = PretrainedModelWrapper(model_dict)
        _MHG_GNN_MODEL.model.eval()
        print("MHG-GNN loaded successfully")
    
    return _MHG_GNN_MODEL

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

def parse_mmap(mmap_file, entry_count, rep, molecular_representations, k_domains, s, logging):
    x_data = []
    y_data = []
    y_data_original = []

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

            # --- continuous pdv ---
            continuous_pdv = None
            if "continuous_pdv" in molecular_representations:
                continuous_pdv_bytes = mmap_file.read(400)
                if "continuous_pdv" == rep:
                    continuous_pdv = np.frombuffer(continuous_pdv_bytes, dtype=np.float16)
                    feature_vector.append(continuous_pdv)
                    if logging: 
                        print(f"continuous_pdv: {continuous_pdv}")

            # --- mol2vec ---
            if "mol2vec" in molecular_representations:
                mol2vec_bytes = mmap_file.read(300)
                if "mol2vec" == rep:
                    mol2vec = np.frombuffer(mol2vec_bytes, dtype=np.uint8)
                    # Already per-molecule quantized, use as-is or scale back to [0,1] if needed
                    feature_vector.append(mol2vec)
                    if logging: 
                        print(f"mol2vec: {mol2vec}")

            # --- chemberta ---
            if "chemberta" in molecular_representations:
                chemberta_bytes = mmap_file.read(768)
                if "chemberta" == rep:
                    chemberta = np.frombuffer(chemberta_bytes, dtype=np.uint8)
                    feature_vector.append(chemberta)
                    if logging: 
                        print(f"chemberta: {chemberta}")

            # --- mhg-gnn ---
            if "mhggnn" in molecular_representations:
                mhggnn_bytes = mmap_file.read(1024)
                if "mhggnn" == rep:
                    mhggnn = np.frombuffer(mhggnn_bytes, dtype=np.uint8)
                    feature_vector.append(mhggnn)
                    if logging: 
                        print(f"mhggnn: {mhggnn}")

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
            if rep == "sns" or rep == "pdv" or rep == "continuous_pdv" or rep == "mol2vec" or rep =="chemberta":
                x_data.append(np.concatenate([f for f in feature_vector if f is not None]))
                y_data.append(processed_target)
                y_data_original.append(target_value)

            # --- SMILES OHE ---
            if "smiles" in molecular_representations:
                ohe_len_bytes = mmap_file.read(4)
                ohe_len = struct.unpack("I", ohe_len_bytes)[0]
                packed = mmap_file.read(ohe_len)
                if rep == "smiles":
                    smiles_ohe = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little")
                    x_data.append(smiles_ohe)
                    y_data.append(processed_target)
                    y_data_original.append(target_value)
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
                    y_data_original.append(target_value)
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
                    y_data_original.append(target_value)
                    if logging:
                        print(f"[{entry}] ecfp4: {ecfp4}")

            # --- graph fallback ---
            if rep == "graph":
                x_data.append(entry)
                y_data.append(processed_target)
                y_data_original.append(target_value)
                continue

        except Exception as e:
            if logging:
                print(f"[{entry}] Skipping malformed entry: {e}")
            continue

    if rep != "graph":
        x_data = np.vstack(x_data).astype(np.uint8)
    y_data = np.array(y_data, dtype=np.float32)
    y_data_original = np.array(y_data_original, dtype=np.float32)

    return x_data, y_data, y_data_original

def run_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, iteration_seed, rep, iteration, s, file_no, y_test_original, domain_labels=None):
    def _black_box_function(trial):
        print(f"Running Optuna trial {trial.number}")
        return model_selector(trial)

    def model_selector(trial=None):
        # Extract domain labels for each split
        domain_labels_train = domain_labels.get('train', None) if domain_labels else None
        domain_labels_val = domain_labels.get('val', None) if domain_labels else None
        domain_labels_test = domain_labels.get('test', None) if domain_labels else None
        
        if model_type in ['rf', 'qrf']:
            return train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, model_type, file_no, y_test_original, trial)

        elif model_type == 'svm':
            return train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type == 'xgboost':
            return train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)
        
        elif model_type == 'ngboost':
            return train_ngboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial)

        elif model_type == 'gauche':
            return train_gauche_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial)

        elif model_type == "dnn":
            return train_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial, 
                                 domain_labels_train=domain_labels_train, domain_labels_val=domain_labels_val, domain_labels_test=domain_labels_test)

        elif model_type == "flexible_dnn":
            return train_flexible_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial,
                                          domain_labels_train=domain_labels_train, domain_labels_val=domain_labels_val, domain_labels_test=domain_labels_test)

        elif model_type == "lgb":
            return train_lgb_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial)

        elif model_type in ["mlp", "residual_mlp", "factorization_mlp", "mtl"]:
            return train_mlp_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial,
                                         domain_labels_train=domain_labels_train, domain_labels_val=domain_labels_val, domain_labels_test=domain_labels_test)

        elif model_type in ["rnn", "gru"] and rep in ['smiles', 'randomized_smiles']:
            return train_rnn_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, trial)

        elif model_type == 'conformal':
            return train_conformal_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, args.cp_base_model, args.calibration_size, y_test_original, trial)

        elif model_type == 'meta_weight_net':
            return train_meta_weight_net(x_train, y_train, x_test, y_test, x_val, y_val,
                                          args, s, rep, iteration, iteration_seed, file_no,
                                          y_test_original, trial)

        elif model_type == 'dividemix_dnn':
            return train_dividemix_dnn(x_train, y_train, x_test, y_test, x_val, y_val,
                                       args, s, rep, iteration, iteration_seed, file_no,
                                       y_test_original, trial)

        elif model_type == 'early_learning':
            return train_early_learning_regularization(x_train, y_train, x_test, y_test, x_val, y_val,
                                                       args, s, rep, iteration, iteration_seed, file_no,
                                                       y_test_original, trial)

        elif model_type == 'multistage_cleaning':
            return train_multistage_cleaning(x_train, y_train, x_test, y_test, x_val, y_val,
                                             args, s, rep, iteration, iteration_seed, file_no,
                                             y_test_original, trial)

        elif model_type == 'uncertainty_curriculum':
            return train_uncertainty_curriculum(x_train, y_train, x_test, y_test, x_val, y_val,
                                               args, s, rep, iteration, iteration_seed, file_no,
                                               y_test_original, trial)

        elif model_type == 'confident_learning':
            return train_confident_learning(x_train, y_train, x_test, y_test, x_val, y_val,
                                           args, s, rep, iteration, iteration_seed, file_no,
                                           y_test_original, trial)

        elif model_type == 'small_loss':
            return train_small_loss_trick(x_train, y_train, x_test, y_test, x_val, y_val,
                                          args, s, rep, iteration, iteration_seed, file_no,
                                          y_test_original, trial)

        elif model_type == 'mentornet':
            return train_mentornet(x_train, y_train, x_test, y_test, x_val, y_val,
                                  args, s, rep, iteration, iteration_seed, file_no,
                                  y_test_original, trial)

        elif model_type == 'contrast_divide':
            return train_contrast_to_divide(x_train, y_train, x_test, y_test, x_val, y_val,
                                            args, s, rep, iteration, iteration_seed, file_no,
                                            y_test_original, trial)

        elif model_type == 'distance_select':
            return train_distance_based_selection(x_train, y_train, x_test, y_test, x_val, y_val,
                                                 args, s, rep, iteration, iteration_seed, file_no,
                                                 y_test_original, trial)

        elif model_type == 'het_gp':
            return train_heteroscedastic_gp(x_train, y_train, x_test, y_test, x_val, y_val,
                                           args, s, rep, iteration, iteration_seed, file_no,
                                           y_test_original, trial)

        elif model_type == 'evidential_kernel':
            return train_evidential_kernel(x_train, y_train, x_test, y_test, x_val, y_val,
                                           args, s, rep, iteration, iteration_seed, file_no,
                                           y_test_original, trial)

        elif model_type == 'ntk_gnn':
            return train_ntk_gnn(train_loader, test_loader, val_loader, args, s, iteration, 
                                file_no, y_test_original, trial,
                                y_train_noisy=y_train_noisy, y_test_noisy=y_test_noisy, 
                                y_val_noisy=y_val_noisy)

        elif model_type == 'conformal_hetero':
            return train_conformal_heteroscedastic(x_train, y_train, x_test, y_test, x_val, y_val,
                                                  args, s, rep, iteration, iteration_seed, file_no,
                                                  y_test_original, trial)

        elif model_type == 'mixup':
            return train_mixup(x_train, y_train, x_test, y_test, x_val, y_val,
                              args, s, rep, iteration, iteration_seed, file_no,
                              y_test_original, trial)

        elif model_type == 'sam':
            return train_sam(x_train, y_train, x_test, y_test, x_val, y_val,
                            args, s, rep, iteration, iteration_seed, file_no,
                            y_test_original, trial)

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

def run_qm9_graph_model(args, qm9, train_idx, test_idx, val_idx, s, iteration, file_no):
    # Read the noisy targets that Rust already generated
    train_file = open(f'train_{file_no}.mmap', 'rb')
    test_file = open(f'test_{file_no}.mmap', 'rb')
    val_file = open(f'val_{file_no}.mmap', 'rb')
    
    # parse_mmap with rep="graph" extracts the processed targets (with noise + normalization from Rust)
    _, y_train_noisy, y_train_original = parse_mmap(
        train_file, len(train_idx), "graph", 
        args.molecular_representations, args.k_domains, s, args.logging
    )
    _, y_test_noisy, y_test_original = parse_mmap(
        test_file, len(test_idx), "graph",
        args.molecular_representations, args.k_domains, s, args.logging
    )
    _, y_val_noisy, y_val_original = parse_mmap(
        val_file, len(val_idx), "graph",
        args.molecular_representations, args.k_domains, s, args.logging
    )
    
    train_file.close()
    test_file.close()
    val_file.close()
    
    # Convert to tensors
    y_train_noisy = torch.tensor(y_train_noisy, dtype=torch.float32)
    y_test_noisy = torch.tensor(y_test_noisy, dtype=torch.float32)
    y_val_noisy = torch.tensor(y_val_noisy, dtype=torch.float32)
    y_test_original_tensor = torch.tensor(y_test_original, dtype=torch.float32)

    # Attach noisy labels to Data objects so they travel with graphs through shuffling
    for i, idx in enumerate(train_idx):
        qm9[idx].y_noisy = y_train_noisy[i].item()
    for i, idx in enumerate(test_idx):
        qm9[idx].y_noisy = y_test_noisy[i].item()
    for i, idx in enumerate(val_idx):
        qm9[idx].y_noisy = y_val_noisy[i].item()
    
    # Create datasets and loaders BEFORE model_selector
    train_set = qm9[train_idx]
    test_set = qm9[test_idx]
    val_set = qm9[val_idx]
    
    train_loader = GeometricDataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = GeometricDataLoader(test_set, batch_size=64, shuffle=False)
    val_loader = GeometricDataLoader(val_set, batch_size=64, shuffle=False)
    
    def _black_box_function(trial, model_type):
        print(f"Running Optuna trial {trial.number} for {model_type}")
        return model_selector(trial, model_type)

    def model_selector(trial, model_type):
        if model_type == "graph_gp":
            # For Graph GP, use PyG Data objects directly
            train_graphs = [qm9[i] for i in train_idx]
            test_graphs = [qm9[i] for i in test_idx]
            val_graphs = [qm9[i] for i in val_idx]
            
            return train_graph_gp(train_graphs, y_train_noisy, test_graphs, y_test_noisy, 
                                 val_graphs, y_val_noisy, args, s, iteration, file_no, 
                                 y_test_original_tensor, trial=trial)
        
        elif model_type == "conformal":
            return train_conformal_graph_model(
                train_loader, test_loader, val_loader, args, s, iteration, 
                file_no, args.cp_base_model, args.calibration_size, y_test_original_tensor, trial,
                y_train_noisy=y_train_noisy, y_test_noisy=y_test_noisy, y_val_noisy=y_val_noisy
            )
        
        else:
            return train_gnn(
                model_type, train_loader, test_loader, val_loader, args, s, 
                iteration, file_no, y_test_original_tensor, trial=trial,
                y_train_noisy=y_train_noisy, y_test_noisy=y_test_noisy, y_val_noisy=y_val_noisy
            )

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

            def objective(trial):
                return _black_box_function(trial, model_type)

            study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

            if len(study.trials) == 0 or all(trial.state != optuna.trial.TrialState.COMPLETE for trial in study.trials):
                print(f"No successful trials for {model_type}. Running with default parameters.")
                res = model_selector(None, model_type)
            else:
                best_params = study.best_params
                print(f"Best params for {model_type} with sigma {s}: {best_params}")

                if args.filepath:
                    json_path = os.path.splitext(args.filepath)[0] + ".json"
                    dir_path = os.path.dirname(json_path)
                    if dir_path:
                        os.makedirs(dir_path, exist_ok=True)

                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            all_params = json.load(f)
                    else:
                        all_params = {}

                    if model_type not in all_params:
                        all_params[model_type] = {}
                    all_params[model_type]['graph'] = best_params

                    with open(json_path, 'w') as f:
                        json.dump(all_params, f, indent=4)

                res = _black_box_function(optuna.trial.FixedTrial(best_params), model_type)
            
            optuna.delete_study(study_name=study.study_name, storage="sqlite:///optuna_study.db")

        elif args.params:
            with open(args.params, 'r') as f:
                all_params = json.load(f)

            if model_type in all_params and 'graph' in all_params[model_type]:
                best_params = all_params[model_type]['graph']

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

def process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset,):
    rust_molecular_representations = args.molecular_representations.copy()
    if args.domain_representation and args.domain_representation not in rust_molecular_representations:
        rust_molecular_representations.append(args.domain_representation)
    
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
        'regression': args.dataset in ['QM9', 'ADME'],
        'normalize': args.normalize,
        'uncertainty': args.uncertainty
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
            '--noise_strategy', args.noise_strategy,
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc_a.communicate()

    print(f"Rust stderr: {stderr}")
    print(f"Rust stdout: {stdout}")

    # Close the write-mode files before reopening in read mode
    for f in files.values():
        f.close()
    
    # Now reopen in read mode
    files = {
        "train": open('train_' + str(file_no) + '.mmap', 'rb'),
        "test": open('test_' + str(file_no) + '.mmap', 'rb'),
        "val": open('val_' + str(file_no) + '.mmap', 'rb'),
    }

    try:
        if 'graph' in args.molecular_representations:
            if args.dataset != 'QM9':
                print(f"WARNING: {args.dataset} has no graph structure, skipping graph models")
            else:
                run_qm9_graph_model(args, dataset, train_idx, test_idx, val_idx, s, iteration, file_no)

        domain_labels = None
        if args.k_domains > 1 and args.domain_method != 'none':
            domain_labels = extract_and_cluster_for_domains(
                args=args,
                file_no=file_no,
                train_idx=train_idx,
                test_idx=test_idx,
                val_idx=val_idx,
                parse_mmap=parse_mmap
            )
        
        # ========== NEW: Create hybrid representation ==========
        parsed_reps = {}
        
        if args.create_hybrid:
            try: 
                print("\n" + "="*70)
                print("CREATING HYBRID REPRESENTATION")
                print("="*70)
                
                sources = ['continuous_pdv', 'ecfp4', 'mol2vec', 'chemberta']
                available = [r for r in sources if r in args.molecular_representations]
                
                if len(available) >= 2:
                    reps_dict = {}
                    
                    for rep in available:
                        print(f"  Parsing {rep}...")
                        for file in files.values():
                            file.seek(0)
                        
                        x_train, y_train, _ = parse_mmap(
                            files["train"], len(train_idx), rep,
                            args.molecular_representations, args.k_domains, s, args.logging
                        )
                        x_test, y_test, y_test_orig = parse_mmap(
                            files["test"], len(test_idx), rep,
                            args.molecular_representations, args.k_domains, s, args.logging
                        )
                        x_val, y_val, _ = parse_mmap(
                            files["val"], len(val_idx), rep,
                            args.molecular_representations, args.k_domains, s, args.logging
                        )
                        
                        reps_dict[rep] = {
                            'x_train': x_train, 'y_train': y_train,
                            'x_test': x_test, 'x_val': x_val
                        }
                        
                        parsed_reps[rep] = {
                            'x_train': x_train, 'y_train': y_train,
                            'x_test': x_test, 'y_test': y_test,
                            'x_val': x_val, 'y_val': y_val,
                            'y_test_original': y_test_orig
                        }
                    
                    print(f"  Combining {len(available)} representations...")
                    h_train, h_test, h_val, _ = create_hybrid_representation(
                        reps_dict, 
                        n_per_rep=args.hybrid_n_per_rep,
                        importance_method=args.hybrid_importance,
                        normalize_method=args.hybrid_normalize,
                        verbose=True,
                        random_state=iteration_seed
                    )
                    
                    parsed_reps['hybrid'] = {
                        'x_train': h_train, 'y_train': y_train,
                        'x_test': h_test, 'y_test': y_test,
                        'x_val': h_val, 'y_val': y_val,
                        'y_test_original': y_test_orig
                    }

                    if args.save_hybrid_analysis:
                        from hybrid_diagnostics import save_feature_rankings, check_multicollinearity
                        
                        # Save feature rankings
                        save_feature_rankings(feature_info, 
                            f'feature_rankings_{iteration}_{s}_{args.hybrid_importance}.csv')
                        
                        # Check and save multicollinearity
                        corr_pairs = check_multicollinearity(h_train, threshold=0.9)
                        with open(f'multicollinearity_{iteration}_{s}.txt', 'w') as f:
                            f.write(f"Highly correlated pairs (|r| > 0.9): {len(corr_pairs)}\n")
                            for i, j, corr in corr_pairs[:20]:  # Top 20
                                f.write(f"Feature {i} <-> Feature {j}: r = {corr:.3f}\n")
                                    
                    print(f"  ✓ Hybrid: {h_train.shape[1]} features")
                    print("="*70 + "\n")
            except Exception as e:  # ADD THIS - catches any hybrid creation errors
                print(f"  ✗ ERROR creating hybrid at sigma={s}, iteration={iteration}: {e}")
                print(f"  Continuing without hybrid for this run...")
                traceback.print_exc()  # Show full error for debugging
                print("="*70 + "\n")
        # ========== END NEW ==========
        
        # Read mmap files and train/test models for all molecular representations
        # ========== MODIFIED: Add hybrid to list ==========
        reps_to_process = list(args.molecular_representations)
        if 'hybrid' in parsed_reps:
            reps_to_process.append('hybrid')
        
        for rep in reps_to_process:
        # ========== END MODIFIED ==========
            if rep != "graph":
                try: 
                    for model in args.models:
                        if model not in graph_models:
                            
                            # ========== MODIFIED: Check cache first ==========
                            if rep in parsed_reps:
                                # Use cached data
                                x_train = parsed_reps[rep]['x_train']
                                y_train = parsed_reps[rep]['y_train']
                                x_test = parsed_reps[rep]['x_test']
                                y_test = parsed_reps[rep]['y_test']
                                x_val = parsed_reps[rep]['x_val']
                                y_val = parsed_reps[rep]['y_val']
                                y_test_original = parsed_reps[rep]['y_test_original']
                            else:
                                # Parse from mmap as usual
                                for file in files.values():
                                    file.seek(0)

                                x_train, y_train, y_train_original = parse_mmap(
                                    files["train"], len(train_idx), rep,
                                    args.molecular_representations, args.k_domains, s, args.logging
                                )
                                x_test, y_test, y_test_original = parse_mmap(
                                    files["test"], len(test_idx), rep,
                                    args.molecular_representations, args.k_domains, s, args.logging
                                )
                                x_val, y_val, y_val_original = parse_mmap(
                                    files["val"], len(val_idx), rep,
                                    args.molecular_representations, args.k_domains, s, args.logging
                                )
                            # ========== END MODIFIED ==========

                            print(f"model: {model}")
                            print(f"rep: {rep}")
                            run_model(
                                x_train, y_train, x_test, y_test, x_val, y_val,
                                model, args, iteration_seed, rep, iteration, s,
                                file_no, y_test_original, domain_labels=domain_labels
                            )
                except Exception as e:
                    print(f"Error with {rep} and {model}; more details: {e}")
    finally:
        # Always close and delete files, even if an error occurred
        for key in list(files.keys()):
            filename = f"{key}_{file_no}.mmap"
            try:
                files[key].close()
            except:
                pass  # File might already be closed
            
            try:
                os.remove(filename)
            except FileNotFoundError:
                print(f"Warning: {filename} not found for deletion")
            except Exception as e:
                print(f"Error deleting {filename}: {e}")
        
        files.clear()
        gc.collect()

def main():
    start_time = time.time()
    args = parse_arguments()

    # Prepare for communication with Rust
    env = os.environ.copy()
    env["RUST_BACKTRACE"] = "1"  # Enable Rust backtraces for debugging

    rust_executable_path = os.path.join(base_dir, '../rust/target/release/rust_processor')

    dataset = None
    if args.dataset == 'QM9':
        dataset = load_qm9(args.target)
        print("QM9 loaded")
    elif args.dataset == 'ADME':
        dataset = load_adme(args.target)
        print("ADME loaded")
    elif args.dataset == 'MoleculeNet':
        dataset = load_moleculenet(args.target)
        print(f"MoleculeNet {args.target} loaded")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

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
                train_idx, test_idx, val_idx = split_qm9(dataset, args, files)

            else:
                train_idx, test_idx, val_idx = load_and_split_polaris(dataset, args, files)

            gc.collect()
            
            target_domain = 1 # TODO: change, this is just a placeholder
            # TODO: uncomment this!!
            try: 
                process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset)
            except Exception as e:
                if logging:
                    print(f"Error with sigma {s}: {e}")
                continue

        current_time = time.time()
        print(f"Time for sigma {s}: {current_time - sigma_time:.2f} seconds")
        sigma_time = current_time

    print(f"Time for total run: {time.time() - start_time}")

if __name__ == "__main__":
    main()