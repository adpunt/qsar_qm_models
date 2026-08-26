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
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
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

import sys
import torch  # used at module level (device, below) and only ever arrived here
             # via the `from models import *` star-import a few lines down, which
             # made this module importable ONLY from the scripts/ directory.

# Anchored to this file, not to the working directory. They were relative
# ('../models/'), so importing this module from anywhere but scripts/ silently
# failed to find the package and then died at `device = torch.device(...)` with
# a bare NameError.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _sub in ('models', 'preprocessing', 'results'):
    _p = os.path.join(_ROOT, _sub)
    if _p not in sys.path:
        sys.path.append(_p)

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

bit_vectors = ['ecfp4', 'mpnn', 'sns', 'plec', 'pdv', 'smiles', 'randomized_smiles', 'continuous_pdv', 'chemberta', 'mhggnn', 'avalon']

# Representations stored as 32-bit floats rather than bits or bytes, and therefore
# the ones that must be standardised per feature before a model sees them. The
# radial-basis kernel shares ONE lengthscale across every dimension
# (models/models.py RBFKernel), so without this the widest few dimensions decide
# every distance and the rest are invisible. Read the reader's dtype choice and the
# standardisation block off this list so they cannot drift apart.
# Which representations get standardised is now shared with the experimental
# pipeline (models/model_defaults.py). The two had diverged: this side scaled
# only the continuous ones, the other scaled everything including binary
# fingerprints, which costs an RBF-kernel SVM about 0.6 R-squared.
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'models'))
from model_defaults import should_standardise, is_binary_matrix

# Kept only for the handful of places that still ask "is this a continuous
# representation by name". The standardisation decision itself is made from the
# FEATURES -- `pdv` means the binarised vector here and the continuous one on
# the experimental side, so no name-keyed list can be right for both.
CONTINUOUS_REPS = ('continuous_pdv', 'chemberta', 'mhggnn')
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
    parser.add_argument("-n", "--sample-size", type=int, default=10000, help="Sample size per iteration (default is 10000)")
    # NOT bootstrapping. Each repetition refits the whole model on a freshly
    # seeded split; nothing is resampled with replacement. The old name is kept
    # as an alias so the existing job arrays keep working.
    parser.add_argument("-b", "--repetitions", "--bootstrapping", dest="repetitions",
                        type=int, default=1,
                        help="Independent repetitions per cell, each with its own seed "
                             "(default 1, i.e. a single fit). The spread across "
                             "repetitions is the run-to-run variance term in the ANOVA.")
    parser.add_argument("--start-iteration", type=int, default=0, help="Starting repetition index (for splitting repetitions across parallel jobs)")
    parser.add_argument("--dump-features", type=str, default=None,
                        help="Write the parsed feature matrix for each representation to "
                             "<PREFIX>__<rep>.npz, exactly as the model receives it and "
                             "BEFORE standardisation. Used by "
                             "scripts/audit_representation_identity.py to check a "
                             "representation against a reference implementation rather than "
                             "by reading the featuriser source. Off by default; it writes "
                             "nothing unless a prefix is given.")
    parser.add_argument("--noise-level", nargs='*', default=[0.0],
                        help="Noise level(s) to run. For every noise type except censoring this is the "
                             "dose DELIVERED, read according to --dose-units. For censoring it is the "
                             "fraction of labels clipped at the assay limit.")
    parser.add_argument("--dose-units", type=str, default="spread", choices=["spread", "label"],
                        help="How --noise-level is read: 'spread' = a fraction of the clean training "
                             "label standard deviation (the only honest axis on QM9); 'label' = the "
                             "label's own units, e.g. log units on the experimental datasets.")
    parser.add_argument("--noise-shape", type=str, default="gaussian",
                        choices=["gaussian", "student_t", "laplace"],
                        help="Shape of each individual draw.")
    parser.add_argument("--noise-targeting", type=str, default="uniform",
                        choices=["uniform", "grouped_wide", "grouped_shift", "outlier", "censoring"],
                        help="Who gets hit and how hard.")
    parser.add_argument("--nu", type=float, default=5.0,
                        help="Degrees of freedom for Student-t. Must be > 2 or the variance is "
                             "undefined and the dose cannot be matched.")
    parser.add_argument("--noise-lambda", type=float, default=3.0,
                        help="How many times wider the affected molecules' error is "
                             "(grouped_wide, outlier). Default 3, from Avdeef 2019.")
    parser.add_argument("--group-fraction", type=float, default=0.2,
                        help="Fraction of scaffold GROUPS affected by grouped_wide. No published "
                             "number exists; 0.2 is a stated choice.")
    parser.add_argument("--group-variance-share", type=float, default=0.62,
                        help="Share of total variance carried by the group-level offset in "
                             "grouped_shift. Default 0.62, from Bentz et al. 2013 Table 7.")
    parser.add_argument("--outlier-p", type=float, default=0.05,
                        help="Fraction of labels contaminated by the outlier type. Hampel (2001): "
                             "1-10% for routine scientific data.")
    parser.add_argument("--censor-side", type=str, default="upper", choices=["upper", "lower"],
                        help="Which end of the label range the assay limit sits at.")
    parser.add_argument("--tuning", type=str2bool, default=False, help="Hyperparameter tuning (default is False)")
    parser.add_argument("--kernel", type=str, default="tanimoto", help="Specify the kernel for certain models (Gaussian Process)")
    parser.add_argument("-k", "--k_domains", type=int, default=1, help="Number of domains for clustering (default is 1)")
    parser.add_argument("-s", "--split", type=str, default="scaffold", help="Method for splitting data (default is scaffold)")
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
    parser.add_argument("--save-per-epoch-metrics", type=str2bool, default=False, help='Save training/validation loss for each epoch')
    parser.add_argument('--cp-base-model', type=str, default='rf',
                    choices=['rf', 'xgboost', 'dnn', 'qrf', 'gauche', 'gin', 'gcn'],
                    help='Base model for conformal prediction')
    parser.add_argument('--use-best-params', action='store_true')
    parser.add_argument(
        "--bayesian-transformation",
        type=str,
        default=None,
        # choices= was absent, and the job scripts pass `last`. Nothing in
        # models.py branches on `last` -- it tests `last_layer` -- so the value
        # fell through every branch, NO Bayesian layer was applied, and the model
        # trained as an ordinary deterministic network. `is_bayesian` is set from
        # `is not None`, so the code then took 100 stochastic forward passes over
        # a model in eval mode, where dropout is inactive: all 100 identical, and
        # the reported uncertainty exactly zero. The row was written to a file
        # named ..._bnn_last.csv and the figure script identifies the model by
        # filename, so a plain network was read as a last-layer Bayesian one.
        # An unrecognised value is now refused by name.
        choices=['full', 'last_layer', 'variational', 'full_variational'],
        help=(
            "Apply Bayesian transformation to applicable models (e.g., DNN, MLP). "
            "Options:\n"
            "  full              - Replace all nn.Linear layers with BayesLinear (torchbnn).\n"
            "  last_layer        - Replace only the final nn.Linear with BayesLinear.\n"
            "  variational       - VBLL: replace last layer with VBLLLayer (learned noise).\n"
            "  full_variational  - Full VBLL: replace ALL layers with VBLLLayer (learned noise on output).\n"
            "Default is None (no transformation)."
        )
    )
    parser.add_argument("--hidden-sizes", type=int, nargs='+', default=None,
                       help="Hidden layer sizes for flexible_dnn/dnn/mlp (e.g. --hidden-sizes 256 128 64 32)")
    parser.add_argument("--calibration-size", type=int, default=20,
                       help="Percentage of validation set for conformal calibration (default is 20)")
    parser.add_argument("--domain-method", type=str, default='none',
                        choices=['none', 'random', 'fingerprint_kmeans', 'descriptor', 
                                 'butina', 'splito', 'scaffold', 'molecular_weight'],
                        help="Method for domain clustering")
    parser.add_argument("--domain-representation", type=str, default='ecfp4',
                        choices=['ecfp4', 'sns', 'pdv'],
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
    parser.add_argument("--create-hybrid", type=str2bool, default=False)
    parser.add_argument("--hybrid-n-per-rep", type=int, default=100)
    parser.add_argument("--hybrid-importance", type=str, default="shap",
                       choices=['shap', 'random_forest', 'mutual_info', 'correlation', 'lasso'])
    parser.add_argument("--hybrid-normalize", type=str, default="standard",
                       choices=['standard', 'minmax', 'none'])
    parser.add_argument("--save-hybrid-analysis", type=str2bool, default=False,
                   help="Save hybrid feature rankings and diagnostics")

    # The noise scheme was redesigned (NOISE_DESIGN.md). The retired flags are
    # refused rather than ignored: a job script written against the old scheme would
    # otherwise run silently under the new one, where the level means something
    # different. Refusing is the whole point.
    retired = {
        "--sigma": "--noise-level (and note the meaning changed: the level is now the dose DELIVERED, "
                   "not a knob each noise type interprets its own way)",
        "--distribution": "--noise-shape (gaussian, student_t, laplace)",
        "--noise-strategy": "--noise-targeting (uniform, grouped_wide, grouped_shift, outlier, censoring)",
        "--strategy-params": "nothing — the file was never passed to the injector and is deleted",
    }
    for flag, replacement in retired.items():
        if any(a == flag or a.startswith(flag + "=") for a in sys.argv[1:]):
            parser.error(f"{flag} has been removed. Use {replacement}.")

    return parser.parse_args()


def build_scaffold_groups(smiles_canonical_list):
    """Map each canonical SMILES to a Murcko scaffold group id.

    The grouped noise types need to know which molecules share a scaffold, and the
    split itself is scaffold-based, so a training split holds whole groups. The map
    is keyed by CANONICAL SMILES rather than by row position, because keying noise
    by row position is precisely what let one molecule's noise land on another.

    Molecules whose scaffold cannot be computed get their own singleton group, so
    they are never silently folded in with anything else.
    """
    group_of_scaffold = {}
    assignments = {}
    for smiles in smiles_canonical_list:
        if smiles is None or smiles in assignments:
            continue
        try:
            scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        except Exception:
            scaffold = None
        # Rule 2 of NOISE_DESIGN.md §2a: an acyclic molecule has an EMPTY Murcko
        # scaffold, and RDKit returns the same empty string for all of them. Left
        # alone they form one enormous group — 32.2% of the first 10,000 QM9
        # molecules — so a single group-level offset would move a third of the
        # dataset at once and the delivered dose would swing by 11% run to run.
        # Each acyclic molecule is its own group instead.
        if not scaffold:
            scaffold = f"__acyclic__{smiles}"
        if scaffold not in group_of_scaffold:
            group_of_scaffold[scaffold] = len(group_of_scaffold)
        assignments[smiles] = group_of_scaffold[scaffold]
    return assignments

def write_to_mmap(
    smiles_isomeric,
    smiles_canonical,
    randomized_smiles,
    pdv,
    continuous_pdv,
    chemberta,
    mhggnn,
    avalon,
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
            continuous_pdv_fp32 = continuous_pdv.astype(np.float32)
            entry += continuous_pdv_fp32.tobytes()
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

    if "avalon" in molecular_representations:
        if avalon is not None:
            entry += avalon.tobytes()
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
    
    # Load mhggnn
    if 'mhggnn' in args.molecular_representations:
        _ = get_mhg_gnn_model()  # Load once
    
    print("Writing to mmap...")
    
    written_canonical = []

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
            # Featurise THIS row's molecule.
            #
            # This used to pop from `mols_train`, a queue filled by iterating
            # train_idx. The writer iterates positions in ASCENDING order, and a
            # scaffold split does not return its indices ascending, so the queue
            # was drained in a different order from the one it was filled in and
            # each training row received a DIFFERENT molecule's fingerprint.
            # Measured on QM9: 95.8% of training rows at 500 molecules, 99.0% at
            # 2,000. The queue was only ever needed to FIT the substructure
            # vocabulary, which happens before this loop; the featuriser itself
            # is a plain function of one molecule.
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
        
        chemberta = None
        if 'chemberta' in args.molecular_representations:
            chemberta = chemberta_fingerprint(smiles_canonical, dimensions=768)

        mhggnn = None
        if 'mhggnn' in args.molecular_representations:
            mhggnn = mhggnn_fingerprint(smiles_canonical, dimensions=1024)

        avalon = None
        if 'avalon' in args.molecular_representations:
            avalon = avalon_fingerprint(smiles_canonical)
        
        write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv, continuous_pdv, chemberta, mhggnn, avalon,
                     target_list[local_idx], category, files,
                     args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)
        written_canonical.append(smiles_canonical)
    
    if 'sns' in args.molecular_representations:
        del mols_train
    
    gc.collect()
    return train_idx, test_idx, val_idx, build_scaffold_groups(written_canonical)

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

    # Load mhg-gnn
    if 'mhggnn' in args.molecular_representations:
        _ = get_mhg_gnn_model()  # Load once

    successful_train_idx = []
    successful_test_idx = []
    successful_val_idx = []
    written_canonical = []

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
            # Same fix as the QM9 writer above: featurise this row's own
            # molecule rather than popping a queue drained in a different order.
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

        chemberta = None
        if 'chemberta' in args.molecular_representations:
            chemberta = chemberta_fingerprint(smiles_canonical, dimensions=768)

        mhggnn = None
        if 'mhggnn' in args.molecular_representations:
            mhggnn = mhggnn_fingerprint(smiles_canonical, dimensions=1024)

        avalon = None
        if 'avalon' in args.molecular_representations:
            avalon = avalon_fingerprint(smiles_canonical)

        if smiles_canonical and not (category == "excluded"):
            if 'randomized_smiles' in args.molecular_representations and not smiles_randomized:
                continue
            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv, continuous_pdv, chemberta, mhggnn, avalon, data.y.item(), category, files, args.molecular_representations, args.k_domains, sns_fp, args.max_vocab)

            written_canonical.append(smiles_canonical)

            if category == "train":
                successful_train_idx.append(index)
            elif category == "test":
                successful_test_idx.append(index)
            elif category == "val":
                successful_val_idx.append(index)

    if 'sns' in args.molecular_representations:
        del mols_train

    return (successful_train_idx, successful_test_idx, successful_val_idx,
            build_scaffold_groups(written_canonical))

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
        
        # Store the embedding exactly as the model produced it. NO per-molecule
        # rescaling: dimension k must mean the same thing on the same scale for every
        # molecule, or distance between molecules is meaningless (RERUN_PLAN.md 2.8c).
        # Per-feature standardisation is applied later, fitted on the training split.
        return np.asarray(embedding, dtype=np.float32)
            
    except Exception as e:
        print(f"ChemBERTa error: {e}")
        return np.zeros(dimensions, dtype=np.float32)

def mhggnn_fingerprint(smiles, dimensions=1024):
    """Generate MHG-GNN embedding for SMILES string"""
    try:
        model = get_mhg_gnn_model()
        
        # Encode returns list of tensors
        embedding = model.encode([smiles])[0]
        embedding = embedding.cpu().detach().numpy()
        
        # Store the embedding exactly as the model produced it. NO per-molecule
        # rescaling: dimension k must mean the same thing on the same scale for every
        # molecule, or distance between molecules is meaningless (RERUN_PLAN.md 2.8c).
        # Per-feature standardisation is applied later, fitted on the training split.
        return np.asarray(embedding, dtype=np.float32)
            
    except Exception as e:
        print(f"MHG-GNN error: {e}")
        return np.zeros(dimensions, dtype=np.float32)

def rdkit_mol_descriptors_from_smiles(smiles_string):
    mol_descriptor_calculator = MolecularDescriptorCalculator(DEFAULT_DESCRIPTOR_LIST)
    mol = Chem.MolFromSmiles(smiles_string)
    descriptor_vals = mol_descriptor_calculator.CalcDescriptors(mol)
    return np.array(descriptor_vals)

def avalon_fingerprint(smiles, n_bits=2048):
    """Avalon fingerprint, packed to bits.

    Binary, so it needs neither the float storage the learned embeddings need nor
    the per-feature standardisation — it is stored exactly like ECFP4. Same call
    KIRBy makes (kirby/representations/molecular.py create_avalon), so the two
    pipelines build the same features.
    """
    from rdkit.Avalon import pyAvalonTools

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(n_bits // 8, dtype=np.uint8)
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        bits = np.array(fp, dtype=np.uint8)
        return np.packbits(bits, bitorder="little")
    except Exception as e:
        print(f"Avalon error: {e}")
        return np.zeros(n_bits // 8, dtype=np.uint8)

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
                '/data/stat-cadd/scat9264/materials',  # Add this
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
        print(f'mhggnn path: {models_path}')
        if models_path not in sys.path:
            sys.path.insert(0, models_path)
        
        from mhg_model.load import PretrainedModelWrapper
        
        # Find pickle
        if model_pickle_path is None:
            search_paths = [
                os.path.expanduser('~/repos/materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle'),
                '/data/stat-cadd/scat9264/materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle',  # Add this
                os.path.join(data_dir, 'mhggnn_pretrained_model_0724_2023.pickle'),
                os.path.join(data_dir, '../materials.mhg-ged/mhggnn_pretrained_model_0724_2023.pickle')
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

# Every representation this reader knows how to step over. The record is a
# packed byte stream with no delimiters, so a representation the WRITER emits
# and the reader does not skip shifts the offset of everything after it in that
# record — silently, and for every molecule. `morgan` was exactly that case: the
# Rust writer emitted 256 bytes for it and there was never a reader. It was
# DELETED from the Rust side on 2026-08-26 (the author does not trust it), so the
# guard below is now what stops the next one rather than that one
# (RERUN_PLAN.md §2.7, §5.6).
# Dropped from the study on 2026-08-26 when the representation set was settled:
# PDV, MHG-GNN, Avalon, ECFP4, ChemBERTa, Sort & Slice. mol2vec has been deleted
# outright. One-hot SMILES still builds -- pulling out the tokenizer would mean
# editing the record layout and the vocabulary handling -- so it is refused by
# name instead, which is what stops a job running it by accident.
DROPPED_REPS = {"smiles", "randomized_smiles"}

PARSEABLE_REPS = {
    "randomized_smiles", "sns", "pdv", "continuous_pdv",
    "chemberta", "mhggnn", "avalon", "smiles", "ecfp4",
}


def parse_mmap(mmap_file, entry_count, rep, molecular_representations, k_domains, s, logging,
               return_smiles=False):
    """Decode the packed record the Rust writer produced.

    return_smiles adds the canonical SMILES as a fourth return value, in the SAME
    ORDER as the feature rows. Only the feature-dump path asks for it. It exists
    so a feature matrix can be checked against a reference implementation
    computed on the same molecules -- without it, a row cannot be traced to a
    molecule at all, and representation identity can only be audited by reading
    the featuriser source.
    """
    dropped = [r for r in molecular_representations if r in DROPPED_REPS]
    if dropped:
        raise RuntimeError(
            f"representation(s) {sorted(dropped)} were dropped from the study on 2026-08-26. "
            f"The set is: PDV (continuous_pdv), MHG-GNN, Avalon, ECFP4, ChemBERTa, Sort & Slice. "
            f"mol2vec is gone from the code entirely; one-hot SMILES still builds but is not part "
            f"of the study, so running it would produce results with nowhere to go."
        )

    unreadable = [r for r in molecular_representations if r not in PARSEABLE_REPS]
    if unreadable:
        raise RuntimeError(
            f"no reader for representation(s) {sorted(unreadable)}. The Rust writer may still "
            f"emit bytes for them, and this reader would then decode every field after them "
            f"from the wrong offset. Add a reader or drop them from -r."
        )

    x_data = []
    y_data = []
    y_data_original = []
    smiles_data = []

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
            smiles_data.append(canonical_smiles)
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
                continuous_pdv_bytes = mmap_file.read(800)
                if "continuous_pdv" == rep:
                    continuous_pdv = np.frombuffer(continuous_pdv_bytes, dtype=np.float32)
                    feature_vector.append(continuous_pdv)
                    if logging: 
                        print(f"continuous_pdv: {continuous_pdv}")

            # --- chemberta ---
            if "chemberta" in molecular_representations:
                chemberta_bytes = mmap_file.read(3072)
                if "chemberta" == rep:
                    chemberta = np.frombuffer(chemberta_bytes, dtype=np.float32)
                    feature_vector.append(chemberta)
                    if logging: 
                        print(f"chemberta: {chemberta}")

            # --- mhg-gnn ---
            if "mhggnn" in molecular_representations:
                mhggnn_bytes = mmap_file.read(4096)
                if "mhggnn" == rep:
                    mhggnn = np.frombuffer(mhggnn_bytes, dtype=np.float32)
                    feature_vector.append(mhggnn)
                    if logging: 
                        print(f"mhggnn: {mhggnn}")

            # --- avalon ---
            if "avalon" in molecular_representations:
                avalon_bytes = mmap_file.read(256)
                if "avalon" == rep:
                    avalon = np.unpackbits(np.frombuffer(avalon_bytes, dtype=np.uint8), bitorder="little")
                    feature_vector.append(avalon)
                    if logging:
                        print(f"avalon: {avalon}")

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
            if rep in ("sns", "pdv", "continuous_pdv", "chemberta", "mhggnn", "avalon"):
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
            # This used to be `continue`, and that was the defect: these records
            # are a packed byte stream with no delimiters, so once a read has
            # gone wrong the offset is unrecoverable and EVERY molecule after it
            # is decoded from the wrong bytes. Skipping the entry does not
            # resynchronise anything — it just stops anyone finding out
            # (RERUN_PLAN.md §2.7). Wildly negative runs that the catastrophic
            # filter later deleted are the shape this produces.
            raise RuntimeError(
                f"malformed record at entry {entry} of {entry_count} "
                f"(byte offset {mmap_file.tell()}, representation '{rep}'). The stream "
                f"cannot be resynchronised from here, so every later molecule would be "
                f"read from the wrong bytes."
            ) from e

    # The stream must have been consumed exactly. A record written short — the
    # failure mode the writer's all-or-nothing rule now prevents — shows up here
    # as leftover bytes, and this is the assertion that catches it (gate 8).
    end = mmap_file.tell()
    mmap_file.seek(0, os.SEEK_END)
    size = mmap_file.tell()
    if end != size:
        raise RuntimeError(
            f"read {entry_count} records but consumed {end} of {size} bytes "
            f"({size - end} left over, representation '{rep}'). The record stream and the "
            f"expected layout disagree — a record was written short or a representation is "
            f"missing from the configuration."
        )

    if rep != "graph":
        if len(x_data) != entry_count:
            raise RuntimeError(
                f"parsed {len(x_data)} feature rows from {entry_count} records "
                f"(representation '{rep}')"
            )

        # Every molecule of one representation has the same feature width. A row
        # that does not is a record read at the wrong offset, and this is the
        # earliest point it can be named. Without it the mismatch reaches
        # np.vstack, which reports a shape error with no entry number — and only
        # when the widths happen to differ. Two misaligned records of the SAME
        # wrong width would have gone through silently.
        widths = {len(row) for row in x_data}
        if len(widths) > 1:
            first_bad = next(
                i for i, row in enumerate(x_data) if len(row) != len(x_data[0])
            )
            raise RuntimeError(
                f"entry {first_bad} decoded to {len(x_data[first_bad])} features but entry 0 "
                f"decoded to {len(x_data[0])} (representation '{rep}'). The record stream is "
                f"misaligned — a record before this one was written short or read at the wrong "
                f"offset."
            )

        # This is a STORAGE dtype decision, not standardisation. It used to be
        # keyed on a name list, which meant an unlisted representation carrying
        # values above 255 would WRAP silently on the cast -- a count of 256
        # reading back as absent. Decide it from the values instead: the narrow
        # type is used only when it provably cannot lose anything.
        x_stacked = np.vstack(x_data)
        fits_uint8 = (np.isfinite(x_stacked).all()
                      and np.all(x_stacked >= 0) and np.all(x_stacked <= 255)
                      and np.array_equal(x_stacked, np.round(x_stacked)))
        x_data = x_stacked.astype(np.uint8 if fits_uint8 else np.float32)
    y_data = np.array(y_data, dtype=np.float32)
    y_data_original = np.array(y_data_original, dtype=np.float32)

    if return_smiles:
        return x_data, y_data, y_data_original, smiles_data
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

        elif model_type == "mlp_bnn_last_standalone":
            return train_bnn_last_standalone(x_train, y_train, x_test, y_test, x_val, y_val,
                                              args, s, rep, iteration, iteration_seed, file_no,
                                              y_test_original)

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

def record_noise_manifest(args, manifest_path, iteration, file_no, level):
    """Append the run-level noise provenance to a CSV beside the results file.

    Nothing recorded how much noise was actually delivered, which is the single
    reason it took the life of the project to notice that the six noise types were
    one type at six doses (RERUN_PLAN.md §2.2). Every run now writes what it
    delivered, next to the results it delivered it for.
    """
    if not os.path.exists(manifest_path):
        print(f"WARNING: the injector wrote no manifest at {manifest_path}")
        return None

    with open(manifest_path) as f:
        manifest = json.load(f)

    row = {
        'iteration': iteration,
        'file_no': file_no,
        'noise_level': level,
        'dose_units': args.dose_units,
        'dataset': args.dataset,
        'target': args.target,
    }
    params = manifest.pop('parameters', {}) or {}
    row.update(manifest)
    for k, v in params.items():
        row[f'param_{k}'] = v

    out_path = args.filepath.replace('.csv', '_noise_manifest.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    write_header = not os.path.exists(out_path)
    with open(out_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset, scaffold_groups=None,):
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

    # One file per task, and the path is handed to the binary explicitly.
    #
    # This used to be a fixed 'config.json'. Every job script does `cd scripts`
    # first and the array jobs run several tasks at once, so every concurrent
    # task wrote and read the SAME file. The file carries `file_no`, which the
    # binary uses to choose which memory-mapped training files to open and
    # rewrite — so one task reading another's configuration meant one task
    # silently overwriting another's training data, with no error from either
    # (RERUN_PLAN.md §2.8a). Everything else the binary touches was already
    # keyed by `file_no`; this was the last shared name.
    config_path = f'config_{file_no}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f)

    scaffold_path = f'scaffold_groups_{file_no}.json'
    if scaffold_groups:
        with open(scaffold_path, 'w') as f:
            json.dump(scaffold_groups, f)
    elif args.noise_targeting in ('grouped_wide', 'grouped_shift'):
        raise RuntimeError(
            f"{args.noise_targeting} needs scaffold group assignments and the split produced none"
        )

    manifest_path = f'noise_manifest_{file_no}.json'
    provenance_path = f'noise_provenance_{file_no}.csv'

    print(f"Rust executable path: {rust_executable_path}")

    rust_cmd = [
        rust_executable_path,
        '--seed', str(iteration_seed),
        '--config', config_path,
        '--model', "rf",
        '--noise-level', str(s),
        '--dose-units', args.dose_units,
        '--noise-shape', args.noise_shape,
        '--noise-targeting', args.noise_targeting,
        '--nu', str(args.nu),
        '--lambda', str(args.noise_lambda),
        '--group-fraction', str(args.group_fraction),
        '--group-variance-share', str(args.group_variance_share),
        '--outlier-p', str(args.outlier_p),
        '--censor-side', args.censor_side,
        '--scaffold-file', scaffold_path,
        '--noise-manifest', manifest_path,
        '--noise-provenance', provenance_path,
    ]
    print(f"Rust command: {' '.join(rust_cmd)}")

    proc_a = subprocess.Popen(
        rust_cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = proc_a.communicate()

    print(f"Rust stderr: {stderr}")
    print(f"Rust stdout: {stdout}")

    # The injector asserts its own gates and dies on any of them. A failure here is
    # a confounded run, so it must stop the pipeline rather than be trained on.
    if proc_a.returncode != 0:
        raise RuntimeError(
            f"noise injection failed (exit {proc_a.returncode}) at level {s}: {stderr.strip()}"
        )

    record_noise_manifest(args, manifest_path, iteration, file_no, s)

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
                
                sources = ['continuous_pdv', 'ecfp4', 'chemberta']
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

                        print(f"DEBUG sigma={s} rep={rep}: train={x_train.shape}, val={x_val.shape}, test={x_test.shape}")
                        print(f"DEBUG y_train[:5]={y_train[:5]}, y_val[:5]={y_val[:5]}")
                    
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

        reps_to_process = list(args.molecular_representations)
        for rep in reps_to_process:
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

                            # Optional: write the features EXACTLY as the model will
                            # receive them, before standardisation. Off unless asked for.
                            # The point is that representation identity can then be
                            # checked against a reference implementation instead of by
                            # reading the Rust source -- which is how the fingerprint
                            # being the wrong function went unnoticed for months
                            # (RERUN_PLAN.md §3.4.1).
                            if getattr(args, 'dump_features', None):
                                _dump = f"{args.dump_features}__{rep}.npz"
                                # Re-read the training split for its SMILES, in the
                                # same order as the rows, through the SAME decoder --
                                # a second copy of the offset logic is exactly the
                                # kind of thing this audit exists to catch.
                                files["train"].seek(0)
                                *_, _smiles = parse_mmap(
                                    files["train"], len(train_idx), rep,
                                    args.molecular_representations, args.k_domains, s,
                                    False, return_smiles=True
                                )
                                np.savez_compressed(
                                    _dump,
                                    x_train=np.asarray(x_train),
                                    x_test=np.asarray(x_test),
                                    y_train=np.asarray(y_train),
                                    smiles_train=np.array(_smiles, dtype=object),
                                    rep=rep,
                                    sigma=s,
                                )
                                print(f"dumped features for {rep} -> {_dump} "
                                      f"(pre-standardisation, {np.asarray(x_train).shape})")

                            # Per-feature standardisation, fitted on the TRAINING split only
                            # and applied to validation and test with the training constants.
                            if should_standardise(x_train, rep):
                                x_mean = np.nanmean(x_train, axis=0)
                                x_std = np.nanstd(x_train, axis=0)
                                x_std[x_std == 0] = 1.0
                                x_train = ((x_train - x_mean) / x_std).astype(np.float32)
                                x_test = ((x_test - x_mean) / x_std).astype(np.float32)
                                x_val = ((x_val - x_mean) / x_std).astype(np.float32)
                                x_train = np.nan_to_num(x_train, 0.0)
                                x_test = np.nan_to_num(x_test, 0.0)
                                x_val = np.nan_to_num(x_val, 0.0)

                            print(f"model: {model}")
                            print(f"rep: {rep}")
                            print(f"DEBUG sigma={s}: x_val type={type(x_val)}, shape={x_val.shape if hasattr(x_val, 'shape') else len(x_val)}")
                            print(f"DEBUG sigma={s}: y_val={y_val[:5] if len(y_val) > 0 else 'EMPTY'}")
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

        # The per-task configuration file goes with them. A task calls this
        # function once per noise level per replicate — 110 times on the main
        # grid — so leaving them behind would litter `scripts/` with a file per
        # invocation.
        try:
            os.remove(config_path)
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error deleting {config_path}: {e}")

        gc.collect()

def main():
    start_time = time.time()
    args = parse_arguments()

    # Refuse a dropped representation BEFORE any work starts.
    #
    # parse_mmap raises on one too, but that raise happens inside the per-model
    # loop, whose `except Exception` prints the message and moves on. A job asking
    # only for a dropped representation would print an error, produce no rows, and
    # exit 0 -- which in an array job of hundreds of tasks is indistinguishable
    # from success (RERUN_PLAN.md 0.6, failure mode 9). Fail here instead, before
    # a single molecule is read.
    dropped = sorted(set(args.molecular_representations) & DROPPED_REPS)
    if dropped:
        raise SystemExit(
            f"\nERROR: {dropped} were dropped from the study on 2026-08-26.\n"
            f"The set is: continuous_pdv (PDV), mhggnn, avalon, ecfp4, chemberta, sns.\n"
            f"mol2vec no longer exists in the code. One-hot SMILES still builds, but it is\n"
            f"not part of the study, so this job would produce results with nowhere to go.\n"
        )

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

    level_time = time.time()
    for s in args.noise_level:
        s = float(s)
        print(f"Noise level: {s} ({args.noise_targeting} / {args.noise_shape}, units: {args.dose_units})")

        for iteration in range(args.start_iteration, args.start_iteration + args.repetitions):
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
                train_idx, test_idx, val_idx, scaffold_groups = split_qm9(dataset, args, files)

            else:
                train_idx, test_idx, val_idx, scaffold_groups = load_and_split_polaris(dataset, args, files)

            gc.collect()
            
            target_domain = 1 # TODO: change, this is just a placeholder
            try: 
                process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset, scaffold_groups)
            except Exception as e:
                if logging:
                    print(f"Error at noise level {s}: {e}")
                continue

        current_time = time.time()
        print(f"Time for noise level {s}: {current_time - level_time:.2f} seconds")
        level_time = current_time

    print(f"Time for total run: {time.time() - start_time}")

if __name__ == "__main__":
    main()