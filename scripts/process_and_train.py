import argparse
import os
import zlib
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
# polaris is imported INSIDE load_adme (the only place it is used) and not here.
#
# At module scope it broke every QM9 job on ARC on 2026-08-28: polaris pulls
# pydantic, whose pydantic_core wants a typing_extensions new enough to carry
# `Sentinel`, and the environment there still has the 4.12.2 that an old
# pip-constraints pin had downgraded it to. The pin was removed from the recipe
# for exactly this reason (pip-constraints.txt, "Ten pins were removed"), but the
# built environment predates that, so the import failed at line 28 and no QM9
# task could start -- for a dataset loader QM9 never calls.
#
# PolarisHubClient went with it: imported here since the file was written and
# referenced nowhere in the repository.
import optuna
import logging
import sqlite3
import pickle
from torch_geometric.utils import to_networkx
import uuid
import time

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

bit_vectors = ['ecfp4', 'mpnn', 'sns', 'plec', 'smiles', 'randomized_smiles', 'pdv', 'chemberta', 'mhggnn', 'avalon']

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
CONTINUOUS_REPS = ('pdv', 'chemberta', 'mhggnn')

# ChemBERTa: ONE encoder across both pipelines, settled 2026-08-27 by the author.
# This side used to load seyonec/ChemBERTa-zinc-base-v1 -- a masked language model
# pretrained on ZINC, 768 wide, 6 layers -- while the experimental pipeline loaded
# DeepChem/ChemBERTa-77M-MTR, a multi-task REGRESSION model pretrained on PubChem,
# 384 wide, 3 layers. A 'ChemBERTa' row from one pipeline and a 'ChemBERTa' row
# from the other came from unrelated networks (RERUN_PLAN.md 2.12).
#
# The width is in the record layout, so this constant and `chemberta_buf` in
# rust/src/main.rs must move together or every field after it decodes at the
# wrong offset.
CHEMBERTA_MODEL_ID = "DeepChem/ChemBERTa-77M-MTR"
CHEMBERTA_DIMS = 384
CHEMBERTA_BYTES = CHEMBERTA_DIMS * 4

# The Sort & Slice record: 1024 substructures, stored as COUNTS rather than
# presence bits. The dimension must match the vec_dimension the featuriser is
# built with; both the writer and the reader read it from here so they cannot
# drift apart, which is how the record layout has gone wrong before.
SNS_DIM = 1024
SNS_COUNT_DTYPE = np.uint16
SNS_RECORD_BYTES = SNS_DIM * np.dtype(SNS_COUNT_DTYPE).itemsize
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
                        # Both spellings, matching the injector. Every results row,
                        # manifest and figure says grouped_wider / grouped_shifted,
                        # so the name read off a row has to be one this accepts --
                        # and argparse refuses an unlisted choice before the value
                        # ever reaches the injector (NOISE_DESIGN.md 6.2b).
                        choices=["uniform", "grouped_wide", "grouped_wider",
                                 "grouped_shift", "grouped_shifted", "outlier",
                                 "censoring"],
                        help="Who gets hit and how hard.")
    parser.add_argument("--nu", type=float, default=5.0,
                        help="Degrees of freedom for Student-t. Must be > 2 or the variance is "
                             "undefined and the dose cannot be matched. Default 5 -- the ONE "
                             "setting the study runs, settled 2026-08-27 in noise_conditions.json: "
                             "nu = 10, 5 and 3 came within 0.006 R2 of each other and of Gaussian "
                             "over twelve replicates on real QM9 (RERUN_PLAN.md 13.9).")
    parser.add_argument("--noise-lambda", type=float, default=3.0,
                        help="How many times wider the affected molecules' error is "
                             "(grouped_wide, outlier). Default 3, from Avdeef 2019.")
    parser.add_argument("--group-fraction", type=float, default=0.2,
                        help="Fraction of scaffold GROUPS affected by grouped_wide. No published "
                             "number exists; 0.2 is a stated choice.")
    parser.add_argument("--group-variance-share", type=float, default=0.62,
                        help="Share of total variance carried by the group-level offset in "
                             "grouped_shift. Default 0.62, from Bentz et al. 2013 Table 7.")
    parser.add_argument("--outlier-p", type=float, default=0.10,
                        help="Fraction of labels contaminated by the outlier type. Hampel (2001): "
                             "1-10% for routine scientific data. Default 0.10 -- the ONE setting "
                             "the study runs, settled 2026-08-27 in noise_conditions.json, because "
                             "1%, 5% and 10% came within 0.005 R2 of each other over twelve "
                             "replicates on real QM9 (RERUN_PLAN.md 13.9).")
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
    # Out-of-fold scoring of the TRAINING molecules. Off by default because it
    # refits every model this many extra times.
    #
    # Without it QM9 cannot answer "does predicted uncertainty find the corrupted
    # labels?" at all: corruption enters the training split only, uncertainty was
    # saved for test molecules only, and a molecule scored by a model that fitted
    # its own corrupted label measures memorisation rather than uncertainty (a GP
    # has zero posterior variance at its own training inputs; a forest has fitted
    # those exact rows). See RERUN_PLAN.md §2.6 / §3.1.
    parser.add_argument("--oof-folds", type=int, default=0,
                        help="Inner folds used to score TRAINING molecules out of fold "
                             "(0 = off, the default). Needs -u/--uncertainty. A value of "
                             "1 is refused: one fold cannot score anything out of fold.")
    parser.add_argument("--oof-folds-scored", type=int, default=0,
                        help="How many of the --oof-folds parts to actually fit and "
                             "score (0 = all of them, the default). The split geometry "
                             "is unchanged, so a scored molecule is scored by a model "
                             "fitted on the same fraction of the data either way; there "
                             "are simply fewer scored molecules and the pass costs that "
                             "many fits instead of --oof-folds. Molecules in a fold that "
                             "was not run are written as NaN, so the file says which "
                             "molecules carry an out-of-fold score.")
    parser.add_argument("--score-validation", action='store_true',
                        help="Also score the held-out VALIDATION molecules with the "
                             "model that is already fitted, and write them as "
                             "split='validation' rows. A validation molecule meets the "
                             "two conditions a scored molecule has to meet -- no model "
                             "fitted it, and the injector recorded the noise it "
                             "received -- so it can answer the same question as a "
                             "train_oof row at one forward pass instead of --oof-folds "
                             "extra fits. It needs the per-molecule provenance, which "
                             "is loaded when --oof-folds is set; pass both to write "
                             "both routes from one run (RERUN_PLAN.md 13 chat O). "
                             "The four neural families early-stop on these molecules, "
                             "so their error here is optimistic; the temperature is "
                             "fitted on them too but is one multiplier and cannot "
                             "change a rank. IT CANNOT ANSWER THE 'DOES PREDICTED "
                             "UNCERTAINTY FIND THE CORRUPTED LABELS' QUESTION FOR THE "
                             "TWO GROUPED CONDITIONS. The split holds whole scaffold "
                             "families out, so validation shares no family with "
                             "training; the injector therefore draws validation its "
                             "OWN affected families (rust/src/main.rs, scale_map, the "
                             "GroupedWide fallback), and they are families no model "
                             "ever fitted. A correlation between predicted uncertainty "
                             "and the recorded shape on those rows is zero in "
                             "expectation BY CONSTRUCTION, and a near-zero would read "
                             "as 'uncertainty does not track the noise' when the "
                             "honest statement is 'the noise was unlearnable there'. "
                             "Read the grouped conditions off the out-of-fold TRAINING "
                             "rows, whose affected families ARE ones the model saw "
                             "(RERUN_PLAN.md 3.1d). The job generator passes cross-"
                             "fitting instead of this flag, for a separate reason of "
                             "its own -- scoring only the models that never see a "
                             "validation label would confound the model comparison "
                             "with the route that produced each number.")
    parser.add_argument("--shap", type=str2bool, default=False, help="Calculate SHAP values for relevant tree-based models (default is False)")
    parser.add_argument("--normalize", type=str2bool, default=True, help="Normalize the data before processing (default is True)")   
    parser.add_argument("--save-per-epoch-metrics", type=str2bool, default=False, help='Save training/validation loss for each epoch')
    # --cp-base-model is COMMENTED OUT 2026-08-28, with the conformal models it
    # selects a base for. Nothing else reads it (RERUN_PLAN.md 2.20).
    # parser.add_argument('--cp-base-model', type=str, default='rf',
    #                 choices=['rf', 'xgboost', 'dnn', 'qrf', 'gauche', 'gin', 'gcn'],
    #                 help='Base model for conformal prediction')
    parser.add_argument('--use-best-params', action='store_true')
    # The variational layer's observation noise, per molecule instead of one
    # number for the whole fit.
    #
    # Without this the layer holds a single learned noise, so its data-noise term
    # is identical for every molecule and its correlation with per-molecule
    # injected noise is ZERO however good the model is -- a mechanism, not a
    # result, and one that plausibly explains the coverage anomaly open since
    # June (RERUN_PLAN.md 5.5, 5.5f). The layer has been able to do this since
    # 2026-08-28 and nothing could reach it: there was no flag and no roster
    # entry. The row name gains `_hetero`, because the two models report
    # different KINDS of term and one name over both would make a per-molecule
    # column and a broadcast constant indistinguishable in the output.
    #
    # Only meaningful with --bayesian-transformation variational or
    # full_variational; refused otherwise rather than silently ignored.
    parser.add_argument(
        '--heteroscedastic-vbll', action='store_true',
        help=("Give the variational last layer a noise head that predicts the "
              "observation noise from the molecule, following HetRegression in "
              "research_archive/f692d614/vbll_regression.py. Requires "
              "--bayesian-transformation variational or full_variational."))
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
    # --calibration-size is COMMENTED OUT, 2026-08-27. It was accepted, passed
    # down to train_conformal_model and never read: conformity scores are
    # |y_val - y_val_pred| over the WHOLE validation split, so the advertised 20%
    # carve-out never happened. Nothing in the paper reads conformal rows -- the
    # figure script drops them at load time and the job generator does not run
    # them -- so the flag only invited someone to trust a setting that does
    # nothing (RERUN_PLAN.md 2.13).
    #
    # The whole held-out split is the calibration set, and that is the better
    # estimator now that no model trains on validation.
    # parser.add_argument("--calibration-size", type=int, default=20,
    #                    help="Percentage of validation set for conformal calibration (default is 20)")
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
    # --alpha is COMMENTED OUT 2026-08-28, with the conformal models. It set the
    # conformal interval's miss rate and is read nowhere else; both readers in
    # models.py fall back to [0.1] when it is absent (RERUN_PLAN.md 2.20).
    # parser.add_argument("--alpha", nargs='*', type=float, default=[0.1],
    #                    help="Confidence levels for conformal prediction (default is [0.1])")
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

    args = parser.parse_args()

    # Conformal is COMMENTED OUT, 2026-08-28, on the author's instruction: it is
    # not used. Refusing by name rather than letting the model names fall
    # through, because the two dispatchers fail differently and both quietly --
    # the tabular one returns None and writes no row, and the graph one trains an
    # ordinary graph network and writes it under the name `conformal`
    # (RERUN_PLAN.md 2.20).
    CONFORMAL_MODELS = ('conformal', 'conformal_hetero')
    _asked = [m for m in (args.models or []) if m in CONFORMAL_MODELS]
    if _asked:
        parser.error(
            f"{', '.join(_asked)} is commented out and cannot be run. Conformal "
            f"prediction is not part of this study: the job generator lists it in "
            f"EXCLUDED_MODELS, the figure script drops its rows when it loads them, "
            f"and no table reads one. The training code is still in models.py if it "
            f"is ever wanted back; the dispatch in this file is what was removed."
        )

    # A flag that is silently ignored is a flag that produced a whole run of the
    # WRONG model under the right name. --heteroscedastic-vbll only means
    # anything to a variational layer, so asking for it anywhere else stops the
    # job rather than writing rows labelled `_hetero` that carry a broadcast
    # constant (RERUN_PLAN.md 5.5f).
    if getattr(args, 'heteroscedastic_vbll', False) and \
            args.bayesian_transformation not in ('variational', 'full_variational'):
        parser.error(
            "--heteroscedastic-vbll needs a variational last layer to attach its "
            "noise head to. Pass --bayesian-transformation full_variational (or "
            "variational), or drop the flag. It was "
            f"{args.bayesian_transformation!r}."
        )

    # One fold cannot score anything out of fold: the single "held-out" part is
    # the whole training set, so there is no fit set left. KIRBy used to accept
    # the value and silently do nothing with it (its trigger reads
    # `if oof_folds and oof_folds > 1`), which is the failure this refuses.
    if args.oof_folds == 1:
        parser.error(
            "--oof-folds 1 is not a fold count. One fold leaves no rows to fit on, "
            "so nothing can be scored out of fold. Use 0 to switch the pass off, or "
            "at least 2."
        )
    if args.oof_folds < 0:
        parser.error(f"--oof-folds must be 0 or at least 2, got {args.oof_folds}")
    if args.oof_folds_scored < 0:
        parser.error(f"--oof-folds-scored must be 0 or more, got {args.oof_folds_scored}")
    if args.oof_folds_scored and not args.oof_folds:
        parser.error(
            "--oof-folds-scored without --oof-folds asks for a fraction of a pass "
            "that is switched off.")
    if args.oof_folds and args.oof_folds_scored > args.oof_folds:
        parser.error(
            f"--oof-folds-scored {args.oof_folds_scored} exceeds --oof-folds "
            f"{args.oof_folds}: there are only {args.oof_folds} folds to score.")
    if args.oof_folds > 1 and not args.uncertainty:
        parser.error(
            "--oof-folds needs -u/--uncertainty: the out-of-fold pass exists to write "
            "per-molecule uncertainty for the TRAINING molecules, and with uncertainty "
            "off there is nowhere for it to go."
        )
    if args.score_validation and not args.uncertainty:
        parser.error(
            "--score-validation needs -u/--uncertainty: it exists to write "
            "per-molecule uncertainty for the VALIDATION molecules, and with "
            "uncertainty off there is nowhere for it to go."
        )

    return args


def scaffold_split_indices(smiles_list, frac_train=0.8, frac_valid=0.1,
                           frac_test=0.1):
    """Split by Murcko scaffold, with each ACYCLIC molecule its own group.

    This used to be `dc.splits.ScaffoldSplitter()`. DeepChem keys on
    `MurckoScaffoldSmiles`, which returns the EMPTY STRING for an acyclic
    molecule and does not special-case it, so every acyclic molecule joins one
    pseudo-group -- and its splitter fills the training set from the largest
    groups first, so that whole group lands in training.

    MEASURED on the first 2,000 QM9 molecules: 851 of them (42.5%) are acyclic,
    and under DeepChem's splitter they were 53.2% of the training set and
    **0.0%** of validation and of test. The models were trained on a population
    half of which never appeared in what they were scored on (RERUN_PLAN.md
    2.13). QM9's own noise grouping already gives acyclic molecules singletons,
    and so does the experimental pipeline's CV grouping.

    The groups are filled in RANDOM order, not largest-first. DeepChem's
    largest-first rule leaves the smallest groups for the held-out splits, and
    once acyclic molecules are singletons that means validation and test come out
    almost entirely acyclic -- measured at 100% and 82% on the same 2,000
    molecules. A random group order gives each split roughly the population's own
    composition while still never splitting a group across two, and it is what
    the experimental pipeline's GroupKFold does in effect. The order is drawn
    from the global numpy generator, which main() seeds per replicate, so each
    replicate is an independently reseeded split -- which is what a QM9
    replicate means.

    An acyclic molecule is keyed on its stereochemistry-free canonical SMILES,
    the same key `build_scaffold_groups` uses for the noise map. It used to be
    keyed on the ROW INDEX here, so two rows holding the SAME acyclic molecule
    were ONE noise group but TWO split groups, and nothing stopped the split
    putting one copy in training and the other in a held-out part -- QM9 is
    never deduplicated, so the copies do exist. MEASURED on 1,998 QM9 molecules
    plus one duplicated acyclic pair, over 200 split seeds: the row-index key put
    the two copies in DIFFERENT splits on 68 of them; the canonical key on 0.
    On the same fixture the split partition and the noise partition now agree
    exactly -- 0 of 1,058 noise groups spread over more than one split.

    An identical molecule on both sides of a split is memorised, not predicted.
    It is rare enough at today's sample size to move no reported number, but the
    rate grows with the sample, and the two partitions now agree by construction
    rather than by luck.
    """
    groups = {}
    for i, smiles in enumerate(smiles_list):
        try:
            scaffold = MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        except Exception:
            scaffold = None
        if not scaffold:
            # Canonicalise the way the writer does (`Chem.MolToSmiles(mol,
            # isomericSmiles=False)`) so the key is character-for-character the
            # one the noise map is built from. A SMILES RDKit cannot parse keeps
            # its own string, which still makes it a singleton.
            mol = Chem.MolFromSmiles(smiles)
            key = smiles if mol is None else Chem.MolToSmiles(mol, isomericSmiles=False)
            scaffold = f"__acyclic__{key}"
        groups.setdefault(scaffold, []).append(i)

    ordered = sorted(groups.values(), key=lambda idx: idx[0])
    np.random.shuffle(ordered)
    n = len(smiles_list)
    train_cutoff = frac_train * n
    valid_cutoff = (frac_train + frac_valid) * n

    train_idx, val_idx, test_idx = [], [], []
    for members in ordered:
        if len(train_idx) + len(members) <= train_cutoff:
            train_idx.extend(members)
        elif len(train_idx) + len(val_idx) + len(members) <= valid_cutoff:
            val_idx.extend(members)
        else:
            test_idx.extend(members)

    if not val_idx or not test_idx:
        raise RuntimeError(
            f"the scaffold split produced {len(train_idx)} training, "
            f"{len(val_idx)} validation and {len(test_idx)} test molecules out "
            f"of {n}. A split with an empty held-out part cannot be scored.")
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def noise_seeds_for_level(iteration_seed, level):
    """The two seeds the injector needs: (shape_seed, selection_seed).

    THE SHAPE SEED varies with the level. It used to be `iteration_seed` alone,
    recomputed identically at every level of the outer loop, and for uniform
    targeting the scale map consumes no randomness -- so the epsilon at every
    level was the SAME standard draw times that level's solved scale, and
    epsilon(0.6) was exactly 2 x epsilon(0.3). The whole degradation curve within
    a replicate rode on one realisation: an unusually smooth curve, and a
    replicate-to-replicate spread of auc_norm smaller than an independent-draw
    design gives. The experimental pipeline draws each level independently, so
    the two were not measuring the same thing (RERUN_PLAN.md 2.13).

    It is keyed on the level's own repr, not its position in the grid, so a
    subset run reproduces the full run's rows.

    THE SELECTION SEED does not vary with the level, and that is the whole reason
    this function exists rather than one inline expression. It seeds WHO GETS HIT
    -- the outlier draw and the affected scaffold groups. Both seeds used to be
    the one level-dependent value, so the affected molecules were redrawn at every
    point of the same condition's curve. Measured on 160 real QM9 molecules,
    level 0.3 against level 0.5: outlier_p10 Spearman 0.014 between the two shape
    columns with 3 affected molecules in common, grouped_wider -0.691 with none
    of its affected scaffold families in common (RERUN_PLAN.md 2.26a).

    Two things break when it moves. `noise_pattern_raw` is the level-free column
    the zero-level subtraction rests on, so subtracting the clean row subtracts a
    correlation against a different set of molecules. And a condition whose
    affected set is redrawn at every level is not one condition swept, so its
    auc_norm is not the same quantity as gaussian's.

    It still varies per REPLICATE, which is what makes the affected set a draw
    rather than a fixture.

    `noiseInject` carries the same pair as `random_state` and `selection_state`
    (NoiseInject/noiseInject/core.py, `_selection_rng`), where this was fixed
    first; the Rust injector takes them as --seed and --selection-seed.
    """
    shape_seed = (iteration_seed * 1000003
                  + zlib.crc32(repr(float(level)).encode())) & 0xFFFFFFFF
    return shape_seed, int(iteration_seed)


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
    ecfp4=None,
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

    # Sort & Slice substructure COUNTS, fixed length.
    #
    # These used to be packed to one presence bit per substructure, which threw
    # away everything the featuriser had been asked for: it is called with
    # sub_counts=True, so it counts how many times each substructure occurs, and
    # packbits then flattened every nonzero to 1. The experimental pipeline kept
    # the counts, so the same name meant two different representations
    # (RERUN_PLAN.md 3.4.1). The counts are worth having on their own terms --
    # scripts/parity_test_count_scaling.py measures +0.011 R2 on QM9 for a
    # radial-kernel support vector machine and +0.019 for the forest.
    #
    # The old path also cast to uint8 BEFORE packing, so a count that was an
    # exact multiple of 256 wrapped to zero and the substructure recorded as
    # absent. Storing uint16 and refusing anything that will not fit closes that
    # instead of moving it to a larger number.
    if "sns" in molecular_representations:
        if sns_fp is not None:
            counts = np.asarray(sns_fp)
            if counts.shape != (SNS_DIM,):
                # A molecule with no enumerable substructures makes the featuriser
                # sum an empty list, which returns the scalar 0.0. The old code
                # packed that into ONE byte instead of 128 and every field after it
                # decoded from the wrong offset for the rest of the file.
                #
                # Zero-filling fixed the alignment and left the corruption: the
                # molecule then trained as if every substructure were genuinely
                # absent. Refuse instead. Alignment is still safe because the run
                # stops. Fixed 2026-08-28, close-out audit item A5.
                raise ValueError(
                    f"Sort & Slice produced {counts.shape} for {smiles_canonical}, "
                    f"not ({SNS_DIM},). A molecule with no enumerable substructures "
                    f"gives a scalar; zero-filling it would train as if every "
                    f"substructure were genuinely absent.")
            if not counts.any():
                raise ValueError(
                    f"Sort & Slice produced an all-zero count vector for "
                    f"{smiles_canonical}. That is a molecule with no features "
                    f"carrying a real label into training.")
            if not np.isfinite(counts).all() or counts.min() < 0:
                raise ValueError(
                    f"substructure counts must be finite and non-negative, got "
                    f"min {counts.min()} for {smiles_canonical}")
            if counts.max() > np.iinfo(np.uint16).max:
                raise ValueError(
                    f"substructure count {counts.max()} exceeds what the record "
                    f"holds ({np.iinfo(np.uint16).max}) for {smiles_canonical}. "
                    f"Widen SNS_COUNT_DTYPE rather than letting it wrap.")
            entry += counts.astype(SNS_COUNT_DTYPE).tobytes()
        else:
            return  # skip incomplete entry

    # PDV is the 200 RDKit descriptors as float32, 800 bytes.
    #
    # It used to be stored as `(pdv > 0)` bit-packed into 25 bytes under this
    # same name, with the float32 form carried alongside as `continuous_pdv`.
    # Bit-packing threw away every descriptor magnitude and handed the model 200
    # raw 0/1 values, 47 of which are constant across QM9 because MolWt,
    # HeavyAtomCount and the like are positive for every molecule. That form is
    # DELETED, not disabled, on the author's instruction 2026-08-28: `pdv` now
    # means the float32 vector, which is what the experimental pipeline has
    # always meant by it and what the paper has always called PDV.
    if "pdv" in molecular_representations:
        if pdv is not None:
            pdv_fp32 = pdv.astype(np.float32)
            entry += pdv_fp32.tobytes()
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

    # LAST in the record, because that is where the Rust re-write puts it in the
    # output too, and the two orderings have to agree.
    if "ecfp4" in molecular_representations:
        if ecfp4 is None:
            return
        if len(ecfp4) != ECFP4_BYTES:
            raise ValueError(
                f"ECFP4 block is {len(ecfp4)} bytes, the record holds "
                f"{ECFP4_BYTES}")
        entry += ecfp4.tobytes()

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
        # This used to be `dc.splits.ScaffoldSplitter()`, the splitter the QM9
        # branch of this same file abandoned and `scaffold_split_indices` was
        # written to replace: DeepChem puts every acyclic molecule in ONE
        # pseudo-group and fills training from the largest group first, so that
        # whole group lands in training. Re-measured on the same 2,000 QM9
        # molecules while making this change: against a population that is 42.5%
        # acyclic, DeepChem gives 53.2% of training and 0.0% of validation and
        # of test, while scaffold_split_indices gives 46.4 / 25.0 / 29.5. No job in the
        # run design reaches this branch -- every generated script passes
        # `-d QM9` and the three experimental datasets go through the other
        # repository -- so nothing in results/ came from the old splitter. It
        # was a second, defective splitter living beside its own replacement,
        # reachable by one flag.
        train_idx, val_idx, test_idx = scaffold_split_indices(
            smiles_list, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

    else:
        # `--split` takes a free-form string and only these two branches exist.
        # Anything else used to fall through with train_idx unbound and die
        # further down on a NameError that named nothing.
        raise ValueError(
            f"--split {args.split!r} is not a split this pipeline knows: "
            f"use 'scaffold' or 'random'.")


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
            vec_dimension=SNS_DIM, print_train_set_info=args.logging
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
        
        chemberta = None
        if 'chemberta' in args.molecular_representations:
            chemberta = chemberta_fingerprint(smiles_canonical, dimensions=CHEMBERTA_DIMS)

        mhggnn = None
        if 'mhggnn' in args.molecular_representations:
            mhggnn = mhggnn_fingerprint(smiles_canonical, dimensions=1024)

        avalon = None
        if 'avalon' in args.molecular_representations:
            avalon = avalon_fingerprint(smiles_canonical)

        ecfp4 = None
        if 'ecfp4' in args.molecular_representations:
            ecfp4 = ecfp4_fingerprint(smiles_canonical)

        write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv, chemberta, mhggnn, avalon,
                     target_list[local_idx], category, files,
                     args.molecular_representations, args.k_domains, sns_fp, args.max_vocab,
                     ecfp4=ecfp4)
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
    """Shuffle, split, featurise and write QM9.

    Returns the SHUFFLED dataset along with the indices, and the caller must
    use the object it returns. `index_select` and `shuffle` both return a NEW
    dataset and rebind the local name only, so every index below is a position
    in the shuffled order. main() used to keep the indices and throw this
    object away, then hand the original, unshuffled dataset to the graph
    models -- which paired each graph with a different molecule's label
    (RERUN_PLAN.md §2.10b). scripts/test_qm9_split_alignment.py fails if that
    comes back.
    """

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
        train_idx, val_idx, test_idx = scaffold_split_indices(
            qm9_smiles, frac_train=0.8, frac_valid=0.1, frac_test=0.1)

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
                                                               vec_dimension = SNS_DIM, 
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

        # The SMILES canonicalisation cache was READ here and never written --
        # grep found no INSERT or UPDATE on that table anywhere, and the on-disk
        # file is schema only. So the lookup always missed, and if it ever HAD
        # hit, `mol` would have stayed None and the randomized-SMILES branch
        # below would have raised (RERUN_PLAN.md 2.13). The molecule is built
        # once and canonicalised from it, which is what actually happened on
        # every pass.
        smiles_canonical = None

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

        chemberta = None
        if 'chemberta' in args.molecular_representations:
            chemberta = chemberta_fingerprint(smiles_canonical, dimensions=CHEMBERTA_DIMS)

        mhggnn = None
        if 'mhggnn' in args.molecular_representations:
            mhggnn = mhggnn_fingerprint(smiles_canonical, dimensions=1024)

        avalon = None
        if 'avalon' in args.molecular_representations:
            avalon = avalon_fingerprint(smiles_canonical)

        ecfp4 = None
        if 'ecfp4' in args.molecular_representations:
            ecfp4 = ecfp4_fingerprint(smiles_canonical)

        if smiles_canonical and not (category == "excluded"):
            if 'randomized_smiles' in args.molecular_representations and not smiles_randomized:
                continue
            write_to_mmap(smiles_isomeric, smiles_canonical, smiles_randomized, pdv,
                          chemberta, mhggnn, avalon, data.y.item(), category, files,
                          args.molecular_representations, args.k_domains, sns_fp, args.max_vocab,
                          ecfp4=ecfp4)

            written_canonical.append(smiles_canonical)

            if category == "train":
                successful_train_idx.append(index)
            elif category == "test":
                successful_test_idx.append(index)
            elif category == "val":
                successful_val_idx.append(index)

    if 'sns' in args.molecular_representations:
        del mols_train

    return (qm9, successful_train_idx, successful_test_idx, successful_val_idx,
            build_scaffold_groups(written_canonical))

def get_chemberta_model():
    """Load ChemBERTa model once and cache it globally"""
    global _CHEMBERTA_MODEL, _CHEMBERTA_TOKENIZER
    
    if _CHEMBERTA_MODEL is None:
        from transformers import AutoTokenizer, AutoModel
        import torch
        
        print("Loading ChemBERTa model (one-time, ~30 seconds)...")
        model_name = CHEMBERTA_MODEL_ID
        _CHEMBERTA_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _CHEMBERTA_MODEL = AutoModel.from_pretrained(model_name)
        _CHEMBERTA_MODEL.eval()
        
        if torch.cuda.is_available():
            _CHEMBERTA_MODEL = _CHEMBERTA_MODEL.cuda()
            print("ChemBERTa loaded on GPU")
        else:
            print("ChemBERTa loaded on CPU")
    
    return _CHEMBERTA_TOKENIZER, _CHEMBERTA_MODEL

def chemberta_fingerprint(smiles, dimensions=CHEMBERTA_DIMS):
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
            # Mean over the real tokens only. The experimental pipeline pools
            # this way (KIRBy molecular.py create_chemberta); a plain mean(dim=1)
            # agrees with it only because this side tokenises one molecule at a
            # time and so never pads. Same pooling, both sides, on purpose.
            mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            embedding = ((token_embeddings * mask).sum(dim=1)
                         / mask.sum(dim=1)).squeeze()
            
            if torch.cuda.is_available():
                embedding = embedding.cpu()
            embedding = embedding.numpy()
        
        # A width other than the record slot's is a hard error, not something to
        # pad or truncate: silently trimming an embedding to fit changes what the
        # representation IS, and padding writes zeros that read as real features.
        if len(embedding) != dimensions:
            raise RuntimeError(
                f"{CHEMBERTA_MODEL_ID} returned a {len(embedding)}-wide "
                f"embedding where the record slot holds {dimensions}. Padding or "
                f"truncating it would change what 'chemberta' means; change "
                f"CHEMBERTA_DIMS here and chemberta_buf in rust/src/main.rs "
                f"together, and re-featurise.")
        
        # Store the embedding exactly as the model produced it. NO per-molecule
        # rescaling: dimension k must mean the same thing on the same scale for every
        # molecule, or distance between molecules is meaningless (RERUN_PLAN.md 2.8c).
        # Per-feature standardisation is applied later, fitted on the training split.
        return np.asarray(embedding, dtype=np.float32)
            
    except Exception as e:
        # A zero row is a molecule with no features carrying a real label into
        # training, and nothing downstream can tell it from a real one.
        raise RuntimeError(f"ChemBERTa failed on {smiles!r}: {e}") from e

MHGGNN_DIMS = 1024
MHGGNN_BYTES = MHGGNN_DIMS * 4

# ECFP4 = Morgan, radius 2, 2,048 bits. Computed HERE, in Python, and carried
# through the record, because that is the only way to compute the same thing the
# experimental pipeline computes (KIRBy create_ecfp4 uses
# rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)).
#
# The Rust side used to compute it with `rdk_fingerprint_mol`, which is RDKit's
# PATH fingerprint (RDKFingerprintMol) -- a different fingerprint entirely.
# Measured on the first 1,500 QM9 molecules: the path fingerprint and Morgan
# radius 2 agreed on ZERO of them, and methane, ammonia and water came back
# all-zero from the path fingerprint because a molecule with one heavy atom has
# no bond paths (RERUN_PLAN.md 2.13). The rdkit-sys binding exposes only
# `morgan_fingerprint_mol`, hardcoded to radius 3, so there was no Rust route to
# radius 2 at all.
ECFP4_BITS = 2048
ECFP4_RADIUS = 2
ECFP4_BYTES = ECFP4_BITS // 8

_ECFP4_GENERATOR = None


def ecfp4_fingerprint(smiles):
    """Morgan radius 2, 2,048 bits, packed little-endian into 256 bytes."""
    global _ECFP4_GENERATOR
    if _ECFP4_GENERATOR is None:
        from rdkit.Chem import rdFingerprintGenerator
        _ECFP4_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
            radius=ECFP4_RADIUS, fpSize=ECFP4_BITS)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise RuntimeError(f"ECFP4: RDKit could not parse {smiles!r}")
    bits = np.array(_ECFP4_GENERATOR.GetFingerprint(mol), dtype=np.uint8)
    if bits.size != ECFP4_BITS:
        raise RuntimeError(
            f"ECFP4: got {bits.size} bits, the record slot holds {ECFP4_BITS}")
    if not bits.any():
        # An all-zero row is a molecule with no features carrying a real label
        # into training, and every such molecule presents the model with an
        # identical input. The Rust gate never caught this case because the
        # fingerprint had been computed SUCCESSFULLY.
        raise RuntimeError(
            f"ECFP4 is all zeros for {smiles!r}. That row would train as if it "
            f"were featurised, and every all-zero molecule would look identical "
            f"to the model.")
    return np.packbits(bits, bitorder='little')


def mhggnn_fingerprint(smiles, dimensions=MHGGNN_DIMS):
    """Generate MHG-GNN embedding for SMILES string"""
    try:
        model = get_mhg_gnn_model()

        # Encode returns list of tensors
        embedding = model.encode([smiles])[0]
        embedding = embedding.cpu().detach().numpy()

        # The width is part of the RECORD LAYOUT: write_to_mmap writes whatever
        # this returns, and both readers consume a fixed 4,096 bytes. An
        # embedding of any other length would shift every field after it in that
        # record, for every molecule, silently -- the failure the deleted
        # `morgan` representation caused. Nothing checked it here; every other
        # float representation enforces its width before writing
        # (RERUN_PLAN.md 2.13).
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if embedding.size != dimensions:
            raise RuntimeError(
                f"the MHG-GNN encoder returned {embedding.size} values where the "
                f"record slot holds {dimensions}. Change MHGGNN_DIMS here and "
                f"mhggnn_buf in rust/src/main.rs together, and re-featurise.")
        return embedding

    except Exception as e:
        # A zero row is a molecule with no features carrying a real label into
        # training. It is not a fallback; it is a silent corruption.
        raise RuntimeError(f"MHG-GNN failed on {smiles!r}: {e}") from e

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
            # An all-zero fingerprint is a molecule with no features carrying a real
            # label into training, and nothing downstream can tell it from a real one:
            # write_to_mmap accepts the block unchecked, and the Rust zero-block refusal
            # is keyed to ecfp4 alone. ChemBERTa and MHG-GNN were fixed on 2026-08-26 and
            # this path was missed. Found by the chat D/G close-out audit, 2026-08-27.
            raise RuntimeError(f"Avalon: RDKit could not parse {smiles!r}")
        fp = pyAvalonTools.GetAvalonFP(mol, nBits=n_bits)
        bits = np.array(fp, dtype=np.uint8)
        if not bits.any():
            # RDKit parses '' into a valid molecule with no atoms, so `mol is None`
            # does not catch it and the fingerprint comes back all zeros. Same silent
            # corruption by a different route. This mirrors the Rust writer's zero-block
            # refusal, which is keyed to ecfp4 only.
            raise RuntimeError(
                f"Avalon: {smiles!r} produced an all-zero fingerprint "
                f"({mol.GetNumAtoms()} atoms)")
        return np.packbits(bits, bitorder="little")
    except Exception as e:
        raise RuntimeError(f"Avalon failed on {smiles!r}: {e}") from e

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
    
    def sub_id_enumerator(mol):
        # An unparseable molecule used to return {} here, which becomes an
        # all-zero count vector: a molecule with no features carrying a real label
        # into training, and nothing downstream can tell it from a real one. Every
        # other featuriser raises on this path -- ChemBERTa and MHG-GNN since
        # 2026-08-26, Avalon since 2026-08-27 -- and Sort & Slice was the one that
        # was missed. Found by the close-out audit and fixed 2026-08-28.
        if mol is None:
            raise RuntimeError(
                "Sort & Slice: RDKit could not parse the molecule, so it has no "
                "substructures. An all-zero count vector would train as if it were "
                "real features.")
        return morgan_generator.GetSparseCountFingerprint(mol).GetNonzeroElements()
    
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

        # `sub_id_enumerator` raises for an unparseable molecule, but RDKit parses
        # '' into a VALID Mol with no atoms, so that route returns {} and never
        # touches the guard above -- the same case Avalon's fix calls out by name.
        # The sum below over an empty list returns the scalar 0.0, which used to
        # reach write_to_mmap as a shape mismatch.
        #
        # This is NOT the case of a molecule whose substructures all fall outside
        # the top `vec_dimension`: that returns a full-width vector of zeros and
        # is the method working as designed.
        if not sub_id_list:
            raise RuntimeError(
                f"Sort & Slice: {Chem.MolToSmiles(mol)!r} has no enumerable "
                f"substructures, so it has no representation to store")

        # create molecule-wide vectorial representation by summing up one-hot encoded substructure identifiers
        ecfp_vector = np.sum(np.array([sub_id_one_hot_encoder(sub_id) for sub_id in sub_id_list]), axis = 0)

        if ecfp_vector.shape != (vec_dimension,):
            raise RuntimeError(
                f"Sort & Slice: {Chem.MolToSmiles(mol)!r} produced shape "
                f"{ecfp_vector.shape}, expected ({vec_dimension},)")

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
# The representation set settled on 2026-08-26 is: PDV, MHG-GNN, Avalon, ECFP4,
# ChemBERTa and Sort & Slice. Everything below is OUTSIDE that set and refused by
# name, which is what stops a job running one by accident. mol2vec has been
# deleted outright.
#
# `continuous_pdv` IS NOT A THIRD THING. It was this pipeline's name for the
# float32 descriptor vector while `pdv` meant the same 200 descriptors
# bit-packed to 25 bytes. The binary form is deleted (2026-08-28, the author's
# instruction), `pdv` now means the float32 vector -- which is what the
# experimental pipeline has always meant by it, so the two pipelines finally
# agree on the name -- and the old spelling is refused rather than aliased.
#
# Refused, not silently accepted, because the meaning of `pdv` CHANGED. Every
# QM9 job script and results file written before 2026-08-28 that says `pdv`
# means the binary vector. A job written against the old naming must stop and
# be read by a person, not run and produce rows that look like the others.
#
# One-hot SMILES and randomized SMILES are a different case: they still BUILD,
# and they stay that way on the author's instruction of 2026-08-28 -- "SMILES
# should not be deleted, just not called. It will not be published." Not being
# called is the whole requirement, and this set is what enforces it, together
# with the job generator, which never emits either name.
DROPPED_REPS = {"smiles", "randomized_smiles", "continuous_pdv"}

PARSEABLE_REPS = {
    "randomized_smiles", "sns", "pdv",
    "chemberta", "mhggnn", "avalon", "smiles", "ecfp4",
    # "graph" contributes no bytes to the record -- the graph models read the
    # dataset object and use parse_mmap only to pull the processed targets back
    # out. It still has to be listed: `-r graph` puts it in
    # molecular_representations, run_qm9_graph_model passes that straight in,
    # and without this entry the guard below rejects every graph run by name.
    # That regression was introduced with the guard and caught by the audit.
    "graph",
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
            f"representation(s) {sorted(dropped)} are not in the study. The study's set is "
            f"pdv, mhggnn, avalon, ecfp4, chemberta and sns.\n"
            f"`continuous_pdv` was renamed to `pdv` on 2026-08-28 when the binary form was "
            f"deleted. It is refused rather than aliased because `pdv` used to mean the binary "
            f"vector, so a job or a file written before that date means something else by it.\n"
            f"One-hot SMILES and randomized SMILES still build and are kept on purpose "
            f"(2026-08-28, the author's instruction), but they are not in the study and "
            f"produce results nothing reads. mol2vec is gone from the code entirely."
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
                sns_bytes = mmap_file.read(SNS_RECORD_BYTES)
                if rep == "sns":
                    sns_fp = np.frombuffer(sns_bytes, dtype=SNS_COUNT_DTYPE).astype(np.float32)
                    feature_vector.append(sns_fp)
                    if logging:
                        print(f"[{entry}] sns_fp: {sns_fp}")
            
            # --- pdv (200 descriptors as float32, 800 bytes) ---
            pdv = None
            if "pdv" in molecular_representations:
                pdv_bytes = mmap_file.read(800)
                if "pdv" == rep:
                    pdv = np.frombuffer(pdv_bytes, dtype=np.float32)
                    feature_vector.append(pdv)
                    if logging:
                        print(f"pdv: {pdv}")

            # --- chemberta ---
            if "chemberta" in molecular_representations:
                chemberta_bytes = mmap_file.read(CHEMBERTA_BYTES)
                if "chemberta" == rep:
                    chemberta = np.frombuffer(chemberta_bytes, dtype=np.float32)
                    feature_vector.append(chemberta)
                    if logging: 
                        print(f"chemberta: {chemberta}")

            # --- mhg-gnn ---
            if "mhggnn" in molecular_representations:
                mhggnn_bytes = mmap_file.read(MHGGNN_BYTES)
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
            if rep in ("sns", "pdv", "chemberta", "mhggnn", "avalon"):
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

def run_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, iteration_seed, rep, iteration, s, file_no, y_test_original, domain_labels=None, train_noise=None):
    """`train_noise` is a TrainingNoiseRecord, or None when --oof-folds is off.

    It carries, for every TRAINING and VALIDATION molecule: the canonical SMILES,
    the scaffold group, the noise the injector recorded, the level-free shape of
    that noise, and the clean training mean and spread. It is handed to every
    dispatched model, and the models that emit a per-molecule uncertainty use it to
    score their own training molecules out of fold. Models that emit no
    per-molecule uncertainty accept it and ignore it, so one signature covers the
    whole roster.
    """
    def _black_box_function(trial):
        print(f"Running Optuna trial {trial.number}")
        return model_selector(trial)

    def model_selector(trial=None):
        # Extract domain labels for each split
        domain_labels_train = domain_labels.get('train', None) if domain_labels else None
        domain_labels_val = domain_labels.get('val', None) if domain_labels else None
        domain_labels_test = domain_labels.get('test', None) if domain_labels else None
        
        if model_type in ['rf', 'qrf']:
            return train_rf_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, model_type, file_no, y_test_original, trial, train_noise=train_noise)

        elif model_type == 'svm':
            return train_svm_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial, train_noise=train_noise)

        elif model_type == 'xgboost':
            return train_xgboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial, train_noise=train_noise)
        
        elif model_type == 'ngboost':
            return train_ngboost_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial, train_noise=train_noise)

        elif model_type == 'gauche':
            return train_gauche_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial, train_noise=train_noise)

        elif model_type == "dnn":
            return train_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial, 
                                 domain_labels_train=domain_labels_train, domain_labels_val=domain_labels_val, domain_labels_test=domain_labels_test, train_noise=train_noise)

        elif model_type == "flexible_dnn":
            return train_flexible_dnn_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial,
                                          domain_labels_train=domain_labels_train, domain_labels_val=domain_labels_val, domain_labels_test=domain_labels_test, train_noise=train_noise)

        elif model_type == "lgb":
            return train_lgb_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, trial, train_noise=train_noise)

        elif model_type in ["mlp", "residual_mlp", "factorization_mlp", "mtl"]:
            return train_mlp_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, y_test_original, trial,
                                         domain_labels_train=domain_labels_train, domain_labels_val=domain_labels_val, domain_labels_test=domain_labels_test, train_noise=train_noise)

        elif model_type in ["rnn", "gru"] and rep in ['smiles', 'randomized_smiles']:
            return train_rnn_variant_model(x_train, y_train, x_test, y_test, x_val, y_val, model_type, args, s, rep, iteration, iteration_seed, file_no, trial, train_noise=train_noise)

        # COMMENTED OUT 2026-08-28, on the author's instruction: conformal is not
        # used. It is in EXCLUDED_MODELS in the job generator, the figure script
        # drops its rows at load time, and no table in the paper reads one.
        # `--calibration-size` was commented out on 2026-08-27 for the same
        # reason. The refusal above is what stops `-m conformal` from returning
        # None here and writing nothing (RERUN_PLAN.md 2.20).
        # elif model_type == 'conformal':
        #     return train_conformal_model(x_train, y_train, x_test, y_test, x_val, y_val, args, s, rep, iteration, iteration_seed, file_no, args.cp_base_model, None, y_test_original, trial, train_noise=train_noise)

        elif model_type == 'meta_weight_net':
            return train_meta_weight_net(x_train, y_train, x_test, y_test, x_val, y_val,
                                          args, s, rep, iteration, iteration_seed, file_no,
                                          y_test_original, trial, train_noise=train_noise)

        elif model_type == 'dividemix_dnn':
            return train_dividemix_dnn(x_train, y_train, x_test, y_test, x_val, y_val,
                                       args, s, rep, iteration, iteration_seed, file_no,
                                       y_test_original, trial, train_noise=train_noise)

        elif model_type == 'early_learning':
            return train_early_learning_regularization(x_train, y_train, x_test, y_test, x_val, y_val,
                                                       args, s, rep, iteration, iteration_seed, file_no,
                                                       y_test_original, trial, train_noise=train_noise)

        elif model_type == 'multistage_cleaning':
            return train_multistage_cleaning(x_train, y_train, x_test, y_test, x_val, y_val,
                                             args, s, rep, iteration, iteration_seed, file_no,
                                             y_test_original, trial, train_noise=train_noise)

        elif model_type == 'uncertainty_curriculum':
            return train_uncertainty_curriculum(x_train, y_train, x_test, y_test, x_val, y_val,
                                               args, s, rep, iteration, iteration_seed, file_no,
                                               y_test_original, trial, train_noise=train_noise)

        elif model_type == 'confident_learning':
            return train_confident_learning(x_train, y_train, x_test, y_test, x_val, y_val,
                                           args, s, rep, iteration, iteration_seed, file_no,
                                           y_test_original, trial, train_noise=train_noise)

        elif model_type == 'small_loss':
            return train_small_loss_trick(x_train, y_train, x_test, y_test, x_val, y_val,
                                          args, s, rep, iteration, iteration_seed, file_no,
                                          y_test_original, trial, train_noise=train_noise)

        elif model_type == 'mentornet':
            return train_mentornet(x_train, y_train, x_test, y_test, x_val, y_val,
                                  args, s, rep, iteration, iteration_seed, file_no,
                                  y_test_original, trial, train_noise=train_noise)

        elif model_type == 'contrast_divide':
            return train_contrast_to_divide(x_train, y_train, x_test, y_test, x_val, y_val,
                                            args, s, rep, iteration, iteration_seed, file_no,
                                            y_test_original, trial, train_noise=train_noise)

        elif model_type == 'distance_select':
            return train_distance_based_selection(x_train, y_train, x_test, y_test, x_val, y_val,
                                                 args, s, rep, iteration, iteration_seed, file_no,
                                                 y_test_original, trial, train_noise=train_noise)

        elif model_type == 'het_gp':
            return train_heteroscedastic_gp(x_train, y_train, x_test, y_test, x_val, y_val,
                                           args, s, rep, iteration, iteration_seed, file_no,
                                           y_test_original, trial, train_noise=train_noise)

        elif model_type == 'evidential_kernel':
            return train_evidential_kernel(x_train, y_train, x_test, y_test, x_val, y_val,
                                           args, s, rep, iteration, iteration_seed, file_no,
                                           y_test_original, trial, train_noise=train_noise)

        elif model_type == 'ntk_gnn':
            return train_ntk_gnn(train_loader, test_loader, val_loader, args, s, iteration, 
                                file_no, y_test_original, trial,
                                y_train_noisy=y_train_noisy, y_test_noisy=y_test_noisy, 
                                y_val_noisy=y_val_noisy, train_noise=train_noise)

        # COMMENTED OUT 2026-08-28, with `conformal` above. This one also wrote its
        # per-molecule learned spread to a file name no reader looks for, so the
        # only column it produced that nothing else produces never reached a
        # table (RERUN_PLAN.md 2.20).
        # elif model_type == 'conformal_hetero':
        #     return train_conformal_heteroscedastic(x_train, y_train, x_test, y_test, x_val, y_val,
        #                                           args, s, rep, iteration, iteration_seed, file_no,
        #                                           y_test_original, trial, train_noise=train_noise)

        elif model_type == 'mixup':
            return train_mixup(x_train, y_train, x_test, y_test, x_val, y_val,
                              args, s, rep, iteration, iteration_seed, file_no,
                              y_test_original, trial, train_noise=train_noise)

        elif model_type == 'sam':
            return train_sam(x_train, y_train, x_test, y_test, x_val, y_val,
                            args, s, rep, iteration, iteration_seed, file_no,
                            y_test_original, trial, train_noise=train_noise)

        elif model_type == "mlp_bnn_last_standalone":
            return train_bnn_last_standalone(x_train, y_train, x_test, y_test, x_val, y_val,
                                              args, s, rep, iteration, iteration_seed, file_no,
                                              y_test_original, train_noise=train_noise)

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

    # Attach the noisy labels to Data objects the loaders will actually see.
    #
    # This used to be `qm9[idx].y_noisy = ...`. Indexing a PyG InMemoryDataset
    # BUILDS a new Data object every time, so the assignment landed on a
    # temporary and was discarded, and `qm9[train_idx]` afterwards produced
    # objects with no `y_noisy` at all. Every graph model then died on the first
    # training batch with `'GlobalStorage' object has no attribute 'y_noisy'`,
    # which the caller's blanket `except Exception` turned into a missing result
    # row (RERUN_PLAN.md 2.13). Materialise the graphs ONCE and keep them.
    train_graphs = [qm9[i] for i in train_idx]
    test_graphs = [qm9[i] for i in test_idx]
    val_graphs = [qm9[i] for i in val_idx]
    for graphs, labels, name in ((train_graphs, y_train_noisy, 'train'),
                                 (test_graphs, y_test_noisy, 'test'),
                                 (val_graphs, y_val_noisy, 'val')):
        if len(graphs) != len(labels):
            raise RuntimeError(
                f"{len(graphs)} {name} graphs against {len(labels)} {name} "
                f"labels read back from the mmap -- pairing them would put "
                f"every label on a different molecule.")
        for g, y in zip(graphs, labels):
            g.y_noisy = float(y)

    train_loader = GeometricDataLoader(train_graphs, batch_size=64, shuffle=True)
    test_loader = GeometricDataLoader(test_graphs, batch_size=64, shuffle=False)
    val_loader = GeometricDataLoader(val_graphs, batch_size=64, shuffle=False)
    
    def _black_box_function(trial, model_type):
        print(f"Running Optuna trial {trial.number} for {model_type}")
        return model_selector(trial, model_type)

    def model_selector(trial, model_type):
        if model_type == "graph_gp":
            # The SAME objects the loaders hold -- rebuilding them from the
            # dataset here would hand the GP graphs with no y_noisy on them.
            return train_graph_gp(train_graphs, y_train_noisy, test_graphs, y_test_noisy, 
                                 val_graphs, y_val_noisy, args, s, iteration, file_no, 
                                 y_test_original_tensor, trial=trial)
        
        # COMMENTED OUT 2026-08-28, with the two tabular branches. The `else`
        # below trains an ordinary graph network, so without the refusal above
        # `-m conformal -r graph` would train a GNN and write it under the name
        # `conformal` (RERUN_PLAN.md 2.20).
        # elif model_type == "conformal":
        #     return train_conformal_graph_model(
        #         train_loader, test_loader, val_loader, args, s, iteration,
        #         file_no, args.cp_base_model, None, y_test_original_tensor, trial,
        #         y_train_noisy=y_train_noisy, y_test_noisy=y_test_noisy, y_val_noisy=y_val_noisy
        #     )
        
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

def _censor_side_from(targeting_name, fallback=None):
    """'censoring_upper' -> 'upper'. Anything else falls back to the CLI value."""
    name = str(targeting_name or '')
    if name.startswith('censoring_'):
        return name[len('censoring_'):]
    return fallback


class TrainingNoiseRecord:
    """What the injector RECORDED for the molecules the models train on.

    Read, never reconstructed. The values come straight out of the per-molecule
    provenance CSV the Rust injector writes where the noise is applied, keyed by
    canonical SMILES; the standardisation constants come out of the run manifest.
    Fitting a line to the labels to recover the noise -- what the analysis used to
    do -- cannot work at all now that held-out labels are clean, and never worked
    for a noise type that transforms the label (RERUN_PLAN.md §0.6, failure mode 2).

    TWO SCALES, BOTH CARRIED, NEITHER GUESSED BACK FROM THE OTHER.

      * `*_raw` arrays are in RAW label units -- the units the injector works in
        and the units the provenance file records. `injected_noise`,
        `noise_scale`, `noise_pattern` and `noise_pattern_pred` reach the
        uncertainty CSV in these units, bit-identical to the provenance file.
      * `y_written` is the STANDARDISED label the model was actually trained on,
        (y_clean_raw + epsilon_raw - std_mean) / std_sd, and it is the column that
        is on the same footing as a prediction. Predictions, uncertainties and
        `y_true_noisy` in the uncertainty CSV are standardised on BOTH splits, so
        no column changes meaning between a test row and a train_oof row.

    `std_mean` and `std_sd` are the CLEAN TRAINING mean and spread, so they do not
    move with the noise level.
    """

    #: The provenance columns this class needs, in the order the injector writes.
    REQUIRED_COLUMNS = (
        'split', 'record_index', 'canonical_smiles', 'y_clean_raw', 'epsilon_raw',
        'noise_scale_raw', 'noise_pattern_raw', 'y_noisy_raw', 'y_written',
    )

    def __init__(self, splits, std_mean, std_sd, level, targeting,
                 censor_side=None, censor_reference_limit=None,
                 provenance_path=None, noise_type=None):
        self.splits = splits
        # The condition's registry name, as the injector wrote it. Every result row
        # has to carry it: a correlation computed over rows from two noise types is
        # pooling across a dimension it should have conditioned on, which is how the
        # paper's per-molecule claim was produced (RERUN_PLAN.md §0.6, failure 1).
        self.noise_type = noise_type
        self.std_mean = float(std_mean)
        self.std_sd = float(std_sd)
        self.level = float(level)
        self.targeting = targeting
        self.censor_side = censor_side
        self.censor_reference_limit = (None if censor_reference_limit is None
                                       else float(censor_reference_limit))
        self.provenance_path = provenance_path
        if self.std_sd <= 0 or not np.isfinite(self.std_sd):
            raise RuntimeError(
                f"the manifest's standardisation spread is {self.std_sd}, which cannot "
                f"put a raw label and a prediction on the same scale")
        self._check_recorded_shape_is_reproducible()

    # ---- construction -----------------------------------------------------
    @staticmethod
    def load(provenance_path, manifest_path, scaffold_groups, level, censor_side=None):
        """Read the per-molecule provenance CSV and the run manifest.

        `censor_side` comes from the run's own --censor-side; the injector records
        the two cut-points but not which end they came from. Passing the wrong one
        does not go unnoticed -- `_check_recorded_shape_is_reproducible` recomputes
        the recorded shape and refuses the run when it disagrees.
        """
        if not os.path.exists(provenance_path):
            raise RuntimeError(
                f"no per-molecule noise provenance at {provenance_path}. Out-of-fold "
                f"scoring writes the noise the injector RECORDED for each training "
                f"molecule; without the file there is nothing to record and the run "
                f"must not fall back to reconstructing it.")
        if not os.path.exists(manifest_path):
            raise RuntimeError(
                f"no noise manifest at {manifest_path}; the standardisation constants "
                f"live there and a raw label cannot be put on the model's scale without "
                f"them.")

        prov = pd.read_csv(provenance_path)
        missing = [c for c in TrainingNoiseRecord.REQUIRED_COLUMNS
                   if c not in prov.columns]
        if missing:
            raise RuntimeError(
                f"{provenance_path} is missing {missing}. It was written by an injector "
                f"predating the per-molecule noise scale and shape (2026-08-26), so the "
                f"level-free shape -- the only thing that makes the zero-level "
                f"subtraction possible -- does not exist in it. Rebuild the Rust "
                f"injector and re-run rather than reconstructing the column.")

        with open(manifest_path) as f:
            manifest = json.load(f)
        params = manifest.get('parameters', {}) or {}

        splits = {}
        for name, part in prov.groupby('split', sort=False):
            part = part.sort_values('record_index')
            if not np.array_equal(part['record_index'].to_numpy(),
                                  np.arange(len(part))):
                raise RuntimeError(
                    f"the '{name}' rows of {provenance_path} are not the contiguous "
                    f"record indices 0..{len(part) - 1}; the file cannot be lined up "
                    f"with the record stream.")
            smiles = part['canonical_smiles'].astype(str).to_numpy()
            groups = np.array(
                [(scaffold_groups or {}).get(sm, -1) for sm in smiles], dtype=np.int64)
            splits[str(name)] = {
                'canonical_smiles': smiles,
                'group': groups,
                'y_clean_raw': part['y_clean_raw'].to_numpy(dtype=np.float64),
                'epsilon_raw': part['epsilon_raw'].to_numpy(dtype=np.float64),
                'noise_scale_raw': part['noise_scale_raw'].to_numpy(dtype=np.float64),
                'noise_pattern_raw': part['noise_pattern_raw'].to_numpy(dtype=np.float64),
                'y_noisy_raw': part['y_noisy_raw'].to_numpy(dtype=np.float64),
                'y_written': part['y_written'].to_numpy(dtype=np.float64),
            }

        for needed in ('train', 'val'):
            if needed not in splits:
                raise RuntimeError(
                    f"{provenance_path} has no '{needed}' rows (it has "
                    f"{sorted(splits)}); the models fit on train and val together.")

        return TrainingNoiseRecord(
            splits=splits,
            std_mean=manifest['standardisation_mean'],
            std_sd=manifest['standardisation_sd'],
            level=level,
            targeting=manifest.get('noise_targeting'),
            noise_type=manifest.get('noise_type'),
            # The condition name carries the side ('censoring_upper' /
            # 'censoring_lower'), so read it from the file rather than trusting a
            # flag that a resubmitted job might have set differently.
            censor_side=_censor_side_from(manifest.get('noise_targeting'),
                                          censor_side or params.get('censor_side')),
            censor_reference_limit=params.get('censor_reference_limit'),
            provenance_path=provenance_path,
        )

    # ---- the level-free shape ---------------------------------------------
    #: Every targeting name `NoiseTargeting::name()` can produce
    #: (rust/src/main.rs), and whether that condition's level-free shape is a
    #: function of the molecule's LABEL.
    #:
    #: Censoring is the only one. Everywhere else the shape is `reference_dose *
    #: s_i / rms(s)` where `s_i` comes from the scale map, and the scale map is a
    #: function of the scaffold group (grouped-wider), of a contamination draw
    #: (outlier), or of nothing at all (uniform, grouped-shifted) -- never of the
    #: label. For those conditions the shape computed from the model's own
    #: predicted label IS the recorded shape, so the ceiling is exact rather than
    #: approximated.
    LABEL_DEPENDENT_SHAPE = {
        'uniform': False,
        'grouped_wider': False,
        'grouped_shifted': False,
        'outlier': False,
        'censoring_upper': True,
        'censoring_lower': True,
    }

    def shape_depends_on_the_label(self):
        """Does a molecule's level-free noise shape move when its LABEL moves?

        An unrecognised condition name STOPS the run. Answering 'no' by default
        is how `noise_pattern_pred` silently became a copy of `noise_pattern`
        when this compared against 'censoring' and the injector writes
        'censoring_upper'.
        """
        try:
            return self.LABEL_DEPENDENT_SHAPE[self.targeting]
        except KeyError:
            raise RuntimeError(
                f"unrecognised noise condition {self.targeting!r} in the manifest. "
                f"Known names are {sorted(self.LABEL_DEPENDENT_SHAPE)}. Whether this "
                f"condition's level-free shape depends on the label decides how "
                f"noise_pattern_pred is computed, and guessing it wrong makes the "
                f"ceiling column silently meaningless.")

    def _clip_to_reference_limit(self, y_raw):
        """The censoring shape: how far a label sits past the far end of the
        training range, at a FIXED reference cut that does not move with the
        level. Mirrors the `clip(y, reference_cut).abs()` in `build_noise_plan`.

        A MAGNITUDE, not a signed shift. It used to be signed, which for the upper
        side is negative, and that made this column rank molecules in the opposite
        order to the Python injector's censoring shape (`noise_scale` in
        `NoiseInject/noiseInject/core.py`, which returns `max(y - limit, 0)`).
        The statistic that reads it is a signed rank correlation over both
        pipelines' rows at once, so the two halves of the study disagreed about
        which direction counted as detection.
        """
        if self.censor_reference_limit is None:
            raise RuntimeError(
                "the manifest records no censor_reference_limit, so the censoring "
                "shape cannot be recomputed from a predicted label.")
        y_raw = np.asarray(y_raw, dtype=np.float64)
        limit = self.censor_reference_limit
        if self.censor_side == 'lower':
            return np.abs(np.where(y_raw < limit, limit - y_raw, 0.0))
        return np.abs(np.where(y_raw > limit, limit - y_raw, 0.0))

    def _check_recorded_shape_is_reproducible(self):
        """Executable guard against this file drifting from the injector.

        For censoring the shape is recomputed here from the recorded clean labels
        and must reproduce the recorded shape. If the two ever disagree, the
        recomputation from a PREDICTED label is meaningless and the run stops
        instead of writing a silently wrong ceiling column.
        """
        if not self.shape_depends_on_the_label():
            return
        for name, part in self.splits.items():
            mine = self._clip_to_reference_limit(part['y_clean_raw'])
            theirs = part['noise_pattern_raw']
            worst = float(np.max(np.abs(mine - theirs))) if len(mine) else 0.0
            tol = 1e-6 * max(1.0, float(np.max(np.abs(theirs))) if len(theirs) else 1.0)
            if worst > tol:
                raise RuntimeError(
                    f"recomputing the censoring shape on the '{name}' split disagrees "
                    f"with the shape the injector recorded by {worst:.3e} (tolerance "
                    f"{tol:.3e}). This module and rust/src/main.rs have drifted; the "
                    f"noise_pattern_pred column would be wrong.")

    def pattern_pred_from_standardised(self, y_pred_standardised, recorded_pattern_raw):
        """`noise_pattern_pred`: the level-free shape recomputed from the model's
        OWN out-of-fold predicted label, in RAW label units.

        The ceiling for `noise_pattern`. An effect against the real shape that is
        no larger than the effect against this one is the model tracking its own
        prediction, not the noise.
        """
        y_pred_raw = (np.asarray(y_pred_standardised, dtype=np.float64) * self.std_sd
                      + self.std_mean)
        if self.shape_depends_on_the_label():
            return self._clip_to_reference_limit(y_pred_raw)
        return np.asarray(recorded_pattern_raw, dtype=np.float64).copy()

    # ---- row selection ----------------------------------------------------
    def check_alignment(self, **smiles_by_split):
        """Assert the provenance rows are the same molecules, in the same order,
        as the rows decoded from the record stream."""
        for name, smiles in smiles_by_split.items():
            if smiles is None:
                continue
            recorded = self.splits[name]['canonical_smiles']
            seen = np.asarray([str(sm) for sm in smiles])
            if len(seen) != len(recorded):
                raise RuntimeError(
                    f"the record stream decoded {len(seen)} '{name}' molecules but the "
                    f"noise provenance holds {len(recorded)}")
            bad = np.flatnonzero(seen != recorded)
            if len(bad):
                i = int(bad[0])
                raise RuntimeError(
                    f"'{name}' row {i} decoded as {seen[i]} but the noise was recorded "
                    f"for {recorded[i]} ({len(bad)} rows differ). Keying noise by row "
                    f"position instead of by molecule is the original QM9 defect; this "
                    f"is the assertion that stops it recurring.")

    def test_rows(self):
        """The recorded arrays for the held-out molecules, in record order.

        Test labels are never corrupted, so `epsilon_raw` and `noise_scale_raw` are
        exactly zero here. `noise_pattern_raw` is not: it is the level-free shape the
        molecule's region WOULD receive, computed against the TRAINING distribution's
        cut-points. The model never saw it, which is what makes it a genuine target
        for "does the model become less certain where the data is unreliable". Without
        it on test rows that question cannot be asked on QM9 at all, and the zero-level
        subtraction that removes the label-magnitude confound cannot be formed.

        Returns None when the run has no test rows recorded.
        """
        part = self.splits.get('test')
        if part is None:
            return None
        return dict(part)

    def rows(self, train_slice=slice(None), val_slice=slice(None)):
        """The recorded arrays for the fit rows, in `vstack((x_train, x_val))` order.

        Seven model families merge validation into training before fitting
        (RERUN_PLAN.md §2.5). Validation now carries its own independently drawn
        noise, dosed against the clean training spread, so those rows are corrupted
        training rows and belong in the out-of-fold pass on the same footing.
        """
        out = {}
        for key in ('canonical_smiles', 'group', 'y_clean_raw', 'epsilon_raw',
                    'noise_scale_raw', 'noise_pattern_raw', 'y_noisy_raw', 'y_written'):
            pieces = []
            if train_slice is not None:
                pieces.append(self.splits['train'][key][train_slice])
            if val_slice is not None:
                pieces.append(self.splits['val'][key][val_slice])
            out[key] = np.concatenate(pieces) if pieces else np.array([])
        return out


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

    # The join keys go back LAST. `row.update(manifest)` overwrote noise_level
    # with the injector's value, which has been through an f32 -- so a level of
    # 0.3 came back as 0.30000001192092896 and a join against the results row's
    # sigma of 0.3 matched nothing. Measured: one of two rows joined
    # (RERUN_PLAN.md 2.13).
    row['iteration'] = iteration
    row['file_no'] = file_no
    row['noise_level'] = level

    out_path = args.filepath.replace('.csv', '_noise_manifest.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # The header is the UNION of every row's keys, rewritten when a later row
    # brings a column the header does not have.
    #
    # This used to be `csv.DictWriter(f, fieldnames=list(row.keys()))` in append
    # mode, with the header written on the first call only. Fieldnames were
    # recomputed per row, so no exception was raised -- the row was simply
    # written with ITS OWN column set. The level loop runs 0.0 first, and the
    # injector returns early at level 0 before it inserts lambda, the outlier
    # fraction and the censoring limit, so the header came from the NARROWEST
    # row: for outlier, grouped_wide, grouped_shift and censoring every later
    # level appended 2-4 extra values with no header cells above them
    # (RERUN_PLAN.md 2.13).
    existing_rows, header = [], []
    if os.path.exists(out_path):
        with open(out_path, newline='') as f:
            reader = csv.DictReader(f)
            header = list(reader.fieldnames or [])
            existing_rows = list(reader)

    new_keys = [k for k in row.keys() if k not in header]
    if header and new_keys:
        header = header + new_keys
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, restval='')
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow(old_row)
            writer.writerow(row)
        return row

    if not header:
        header = list(row.keys())
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header, restval='')
            writer.writeheader()
            writer.writerow(row)
        return row

    with open(out_path, 'a', newline='') as f:
        # restval fills a column this row does not carry; extrasaction cannot
        # fire, because new_keys is empty here.
        csv.DictWriter(f, fieldnames=header, restval='').writerow(row)
    return row


def level_units_for(args):
    """What the number in the `sigma` column MEASURES for this run.

    QM9 doses in fractions of the clean training label spread (`--dose-units
    spread`, the default) or in raw label units (`--dose-units label`); the
    experimental pipeline doses in raw log units anchored to published assay
    error. Censoring has no dose axis at all -- its level is the fraction of
    labels clipped. All three used to be written into one column called `sigma`
    with nothing to tell them apart, and `auc_norm` is mean retention over each
    configuration's OWN level range, so two of them on one axis compare
    retention over different spans (RERUN_PLAN.md 2.12).
    """
    if getattr(args, 'noise_targeting', None) == 'censoring':
        return 'fraction_censored'
    return 'label_sd' if getattr(args, 'dose_units', 'spread') == 'spread' else 'raw_label'


def condition_from_manifest_row(manifest_row, manifest_path, level, iteration):
    """The condition name the injector recorded, or stop.

    A row written with no condition on it cannot be conditioned on later, and
    gets pooled with every other condition -- which is how the paper's
    per-molecule claim was produced (RERUN_PLAN.md 2.11). The name is never
    composed here from the CLI flags: a second implementation of the naming is a
    second thing to drift out of step with the injector's.
    """
    condition = (manifest_row or {}).get('noise_type')
    if not condition:
        raise RuntimeError(
            f"the injector's manifest at {manifest_path} carries no noise_type, so "
            f"the rows for noise level {level}, replicate {iteration} would be "
            f"written with no condition on them. A row that cannot be conditioned "
            f"on its noise type gets pooled with every other condition.")
    return str(condition)


# Messages raised by the reader guards. A misparse destroys the byte offset, so
# it cannot be confined to one (representation, model) cell -- it must take the
# task down rather than be printed and stepped over.
_INTEGRITY_MARKERS = (
    "no reader for representation",
    "malformed record at entry",
    "consumed",
    "feature rows from",
    "decoded to",
    "misaligned",
    "SMILES",
    "alignment",
)


def _is_data_integrity_error(exc):
    text = str(exc)
    return any(m in text for m in _INTEGRITY_MARKERS)


def process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, dataset, scaffold_groups=None,):
    # Every (representation, model) that failed for a reason that is NOT a data
    # integrity problem. Reported at the end so a task cannot finish quietly
    # having produced fewer rows than it was asked for.
    failed_pairs = []
    # `rust_molecular_representations` used to be built here -- args.
    # molecular_representations plus the domain representation -- and then never
    # used: the config below wrote args.molecular_representations, and
    # write_to_mmap is driven by the same list, so the domain representation was
    # never actually written or read. It is gone rather than left looking live;
    # --k_domains > 1 is refused in main() for the same reason (the domain byte
    # is a placeholder, RERUN_PLAN.md 2.13).

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

    # Two seeds, and the difference between them is the point: the shape draw
    # moves with the level, the affected set does not. The rule lives in one
    # function so it can be executed by a test rather than read
    # (scripts/test_injector_wiring.py).
    level_seed, selection_seed = noise_seeds_for_level(iteration_seed, s)

    rust_cmd = [
        rust_executable_path,
        '--seed', str(level_seed),
        '--selection-seed', str(selection_seed),
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

    # The return code was never looked at. Everything the injector refuses to do
    # -- a failed dose gate, a molecule it could not fingerprint, a truncated
    # record, a configuration it could not open, a segmentation fault -- was
    # reported to a pipe nobody read, and this function carried straight on to
    # reopen the files and train on whatever was on disk. Every hard failure in
    # the Rust half was decorative until this check existed.
    #
    # The injector also asserts its own dose and identity gates and dies on any
    # of them, so a non-zero exit is a confounded run, not just a missing file.
    #
    # It is worse than a missed message: preprocess_data renames the rewritten
    # training file over the original BEFORE it processes val and test, so a run
    # that dies partway leaves train noised and the held-out splits not, which
    # trains and scores without complaint.
    if proc_a.returncode != 0:
        raise RuntimeError(
            f"the noise injector exited {proc_a.returncode} for noise level {s}, "
            f"replicate {iteration}, file_no {file_no}. The memory-mapped files may be "
            f"half-rewritten, so nothing downstream of this point is trustworthy.\n"
            f"--- stderr ---\n{(stderr or '').strip()[-4000:]}"
        )

    manifest_row = record_noise_manifest(args, manifest_path, iteration, file_no, s)

    # Stamp the condition onto every results row this level writes. The name is
    # the injector's own (`condition_name` in rust/src/main.rs, carried through
    # the manifest); nothing downstream has to recover it from a filename.
    set_current_noise_type(
        condition_from_manifest_row(manifest_row, manifest_path, s, iteration),
        level_units=level_units_for(args),
        delivered_dose=(manifest_row or {}).get('delivered_dose_in_label_units'),
        standardisation=((manifest_row or {}).get('standardisation_mean'),
                         (manifest_row or {}).get('standardisation_sd')),
        file_no=file_no)

    # The noise the injector RECORDED, read back before a single model is fitted.
    # Only when the out-of-fold pass is on: it is the only consumer, and reading
    # the file otherwise would refuse older provenance for no reason.
    noise_record = None
    # The per-molecule provenance is what BOTH routes read: the out-of-fold pass
    # needs the training draws, and --score-validation needs the validation ones.
    if getattr(args, 'oof_folds', 0) > 1 or getattr(args, 'score_validation', False):
        noise_record = TrainingNoiseRecord.load(
            provenance_path, manifest_path, scaffold_groups, s,
            censor_side=getattr(args, 'censor_side', None))
        print(f"[oof] read the recorded noise for "
              f"{len(noise_record.splits['train']['epsilon_raw'])} training and "
              f"{len(noise_record.splits['val']['epsilon_raw'])} validation molecules "
              f"from {provenance_path} "
              f"(standardisation mean {noise_record.std_mean:.6g}, "
              f"sd {noise_record.std_sd:.6g})")

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
                
                sources = ['pdv', 'ecfp4', 'chemberta']
                available = [r for r in sources if r in args.molecular_representations]
                
                if len(available) >= 2:
                    reps_dict = {}
                    
                    for rep in available:
                        print(f"  Parsing {rep}...")
                        for file in files.values():
                            file.seek(0)
                        
                        x_train, y_train, _, smiles_train = parse_mmap(
                            files["train"], len(train_idx), rep,
                            args.molecular_representations, args.k_domains, s, args.logging,
                            return_smiles=True
                        )
                        x_test, y_test, y_test_orig = parse_mmap(
                            files["test"], len(test_idx), rep,
                            args.molecular_representations, args.k_domains, s, args.logging
                        )
                        x_val, y_val, _, smiles_val = parse_mmap(
                            files["val"], len(val_idx), rep,
                            args.molecular_representations, args.k_domains, s, args.logging,
                            return_smiles=True
                        )

                        reps_dict[rep] = {
                            'x_train': x_train, 'y_train': y_train,
                            'x_test': x_test, 'x_val': x_val
                        }
                        
                        parsed_reps[rep] = {
                            'x_train': x_train, 'y_train': y_train,
                            'x_test': x_test, 'y_test': y_test,
                            'x_val': x_val, 'y_val': y_val,
                            'y_test_original': y_test_orig,
                            'smiles_train': smiles_train, 'smiles_val': smiles_val
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
                        'y_test_original': y_test_orig,
                        'smiles_train': smiles_train, 'smiles_val': smiles_val
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
                                smiles_train = parsed_reps[rep].get('smiles_train')
                                smiles_val = parsed_reps[rep].get('smiles_val')
                            else:
                                # Parse from mmap as usual
                                for file in files.values():
                                    file.seek(0)

                                # The canonical SMILES come back with every parse now.
                                # They cost nothing (the decoder already reads them) and
                                # they are what lets a feature row be matched to the
                                # molecule the noise was recorded for. `sample_idx` is a
                                # row position and cannot do that job.
                                x_train, y_train, y_train_original, smiles_train = parse_mmap(
                                    files["train"], len(train_idx), rep,
                                    args.molecular_representations, args.k_domains, s, args.logging,
                                    return_smiles=True
                                )
                                x_test, y_test, y_test_original, smiles_test = parse_mmap(
                                    files["test"], len(test_idx), rep,
                                    args.molecular_representations, args.k_domains, s, args.logging,
                                    return_smiles=True
                                )
                                x_val, y_val, y_val_original, smiles_val = parse_mmap(
                                    files["val"], len(val_idx), rep,
                                    args.molecular_representations, args.k_domains, s, args.logging,
                                    return_smiles=True
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
                                # PER CELL, and BEFORE the scaling constants are
                                # computed. It used to be after: a single +inf
                                # anywhere in a feature made nanmean inf and
                                # nanstd nan, the `x_std[x_std == 0] = 1.0` guard
                                # does not catch a nan, and the nan_to_num that
                                # followed then zeroed that ENTIRE feature for
                                # every molecule rather than the one cell. The
                                # experimental pipeline replaces non-finite cells
                                # before scaling (KIRBy create_pdv); this now
                                # does the same (RERUN_PLAN.md 2.13).
                                _clean = lambda a: np.nan_to_num(
                                    np.asarray(a, dtype=np.float64),
                                    nan=0.0, posinf=0.0, neginf=0.0)
                                x_train, x_test, x_val = (
                                    _clean(x_train), _clean(x_test), _clean(x_val))
                                x_mean = x_train.mean(axis=0)
                                x_std = x_train.std(axis=0)
                                x_std[~np.isfinite(x_std) | (x_std == 0)] = 1.0
                                x_train = ((x_train - x_mean) / x_std).astype(np.float32)
                                x_test = ((x_test - x_mean) / x_std).astype(np.float32)
                                x_val = ((x_val - x_mean) / x_std).astype(np.float32)

                            print(f"model: {model}")
                            print(f"rep: {rep}")
                            print(f"DEBUG sigma={s}: x_val type={type(x_val)}, shape={x_val.shape if hasattr(x_val, 'shape') else len(x_val)}")
                            print(f"DEBUG sigma={s}: y_val={y_val[:5] if len(y_val) > 0 else 'EMPTY'}")
                            # The recorded noise only reaches a model once the rows
                            # it will fit have been shown to BE the molecules the
                            # noise was recorded for.
                            if noise_record is not None:
                                if smiles_train is None or smiles_val is None:
                                    raise RuntimeError(
                                        f"the '{rep}' rows reached the models without "
                                        f"their canonical SMILES, so they cannot be "
                                        f"shown to be the molecules the noise was "
                                        f"recorded for. Refusing to score them out of "
                                        f"fold rather than trusting row position.")
                                noise_record.check_alignment(
                                    train=smiles_train, val=smiles_val)

                            run_model(
                                x_train, y_train, x_test, y_test, x_val, y_val,
                                model, args, iteration_seed, rep, iteration, s,
                                file_no, y_test_original, domain_labels=domain_labels,
                                train_noise=noise_record
                            )
                except Exception as e:
                    # This handler wraps the parse AND the whole model loop, and
                    # it used to only print. Every guard added to the reader --
                    # the unreadable-representation rejection, the malformed
                    # record, the leftover bytes, the row-count and ragged-width
                    # checks, and the out-of-fold alignment assertions -- landed
                    # here and the task still exited 0 with an empty results
                    # file. The audit found five reviewers reporting it
                    # independently: fixing parse_mmap to raise, and then
                    # swallowing the raise one frame up, changed a silent
                    # `continue` into a printed line and nothing else.
                    #
                    # A misparse is not recoverable and is not per-model: the
                    # byte offset is gone, so every later representation read
                    # from the same handles is suspect too. It propagates.
                    print(f"ERROR with {rep} and {model}: {type(e).__name__}: {e}",
                          flush=True)
                    traceback.print_exc()
                    if isinstance(e, (RuntimeError, ValueError, AssertionError, OSError)) \
                            and _is_data_integrity_error(e):
                        raise
                    failed_pairs.append((rep, model, f"{type(e).__name__}: {e}"))
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

    # Cells that failed for reasons other than data integrity (a model that
    # would not converge, an out-of-memory, a library error). Not fatal on their
    # own -- but the caller records them so the job cannot exit 0 having lost
    # part of the grid it was asked for.
    if failed_pairs:
        print(f"WARNING: {len(failed_pairs)} (representation, model) pair(s) produced no rows "
              f"at noise level {s}, replicate {iteration}:")
        for rep, model, msg in failed_pairs:
            print(f"  {rep} / {model}: {msg}")
    return failed_pairs

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
    # The domain label is written as a literal zero for every molecule by the
    # Rust writer, so a run with more than one domain carries no domain
    # assignment through the record at all -- the clustering result is discarded
    # and every reader sees one constant (RERUN_PLAN.md 2.13). Refused rather
    # than run into a silent no-op.
    if getattr(args, 'k_domains', 1) > 1:
        raise SystemExit(
            f"\nERROR: --k_domains {args.k_domains} was requested, but the domain "
            f"label the record carries is a literal zero for every molecule "
            f"(rust/src/main.rs). The clustering would be computed and thrown "
            f"away. Run with k_domains 1, or wire the label through first.\n")

    dropped = sorted(set(args.molecular_representations) & DROPPED_REPS)
    if dropped:
        raise SystemExit(
            f"\nERROR: {dropped} are not in the study.\n"
            f"The study's set is: pdv, mhggnn, avalon, ecfp4, chemberta, sns.\n"
            f"`continuous_pdv` was renamed to `pdv` on 2026-08-28, when the binary descriptor\n"
            f"vector was deleted. It is refused rather than treated as an alias: `pdv` used to\n"
            f"mean the binary form, so a job script written before that date means the other\n"
            f"thing by it and must be read by a person.\n"
            f"One-hot SMILES and randomized SMILES still build and are kept on purpose, but\n"
            f"neither is part of the study, so this job would produce results with nowhere\n"
            f"to go. mol2vec no longer exists in the code.\n"
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

    # Every (noise level, replicate) that did not complete. A run that loses
    # cells must not look like a run that did not.
    failed_cells = []

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
            # A NAME for this task's scratch files, never a seed -- every use of
            # it is a filename (verified across process_and_train.py and
            # models/models.py), so randomness here costs no reproducibility.
            #
            # It used to be `iteration_seed ^ int(time.time() * 1e6)`, and
            # RERUN_PLAN.md §2.8a called that "effectively unique per task". It is
            # not. Array tasks differ by representation and strategy, not by seed,
            # so they all compute the SAME iteration_seed -- leaving the clock as
            # the only distinguishing term, and consecutive calls to
            # `int(time.time() * 1e6)` on this machine return the same value.
            # Two tasks starting together therefore get the same file_no, open
            # the same train_<file_no>.mmap, and rewrite each other's training
            # data: exactly the corruption §2.8a was written about, surviving the
            # fix that was supposed to close it.
            #
            # uuid4 draws from the OS entropy source, so it does not collide
            # between processes however close together they start.
            # 64 bits, not 32: the Rust side types file_no as usize, and a
            # full grid draws roughly 4,000 of these (36 tasks x 110
            # invocations). At 32 bits that is about a 1-in-500 chance of a
            # birthday collision across the run; at 64 bits it is negligible.
            # 63 bits, not 64: a value above 2^63 does not fit in an int64, so
            # pandas reads such a column as float or object and a join on it
            # loses precision or fails outright -- measured, one of two rows
            # joined (RERUN_PLAN.md 2.13). The birthday risk is unchanged at this
            # width.
            file_no = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF

            files = {
                "train": open('train_' + str(file_no) + '.mmap', 'wb+'),
                "test": open('test_' + str(file_no) + '.mmap', 'wb+'),
                "val": open('val_' + str(file_no) + '.mmap', 'wb+'),
            }

            train_size = int(args.sample_size * 0.8)
            test_size = int(args.sample_size * 0.1)
            val_size = int(args.sample_size * 0.1)

            if args.dataset == 'QM9':
                # split_dataset is the SHUFFLED dataset the indices belong to.
                # Passing `dataset` here instead voids the split for every graph
                # model: qm9[train_idx] on the unshuffled object is an arbitrary
                # subset, and the graph at each position belongs to a different
                # molecule than the label written at that position.
                split_dataset, train_idx, test_idx, val_idx, scaffold_groups = \
                    split_qm9(dataset, args, files)

            else:
                split_dataset = dataset
                train_idx, test_idx, val_idx, scaffold_groups = load_and_split_polaris(dataset, args, files)

            gc.collect()
            
            target_domain = 1 # TODO: change, this is just a placeholder
            try: 
                pairs = process_and_run(args, iteration, iteration_seed, file_no, train_idx, test_idx, val_idx, target_domain, env, rust_executable_path, files, s, split_dataset, scaffold_groups)
                for rep, model, msg in (pairs or []):
                    failed_cells.append((s, iteration, f"{rep}/{model}: {msg}"))
            except Exception as e:
                # This used to print only `if logging`, which is off by default,
                # and then `continue`. A noise level that failed produced no rows
                # and no message: the same shape as the two Gaussian-process jobs
                # that ran to completion and wrote nothing (RERUN_PLAN.md §2.8d).
                # It is always reported now, with a traceback, and the run is
                # remembered so the job cannot exit 0 having lost cells.
                print(f"ERROR at noise level {s}, replicate {iteration}: "
                      f"{type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                failed_cells.append((s, iteration, f"{type(e).__name__}: {e}"))
                continue

        current_time = time.time()
        print(f"Time for noise level {s}: {current_time - level_time:.2f} seconds")
        level_time = current_time

    print(f"Time for total run: {time.time() - start_time}")

    if failed_cells:
        planned = len(args.noise_level) * args.repetitions * \
            max(1, len(args.molecular_representations) * len(args.models))
        print(f"\nERROR: {len(failed_cells)} of about {planned} planned "
              f"(noise level, replicate, representation, model) cells failed and produced "
              f"no rows:")
        for level, rep, msg in failed_cells:
            print(f"  noise level {level}, replicate {rep}: {msg}")
        print("The results file is INCOMPLETE. Exiting non-zero so the scheduler and the "
              "runbook's resubmit step can see it.")
        sys.exit(1)

if __name__ == "__main__":
    main()