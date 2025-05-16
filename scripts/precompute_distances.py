import torch
import numpy as np
import random
import argparse
import os
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity
from tqdm import tqdm
import pickle
from process_and_train import load_qm9
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import pairwise_distances
from sklearn.utils import shuffle

# -------------------- Constants --------------------
parser = argparse.ArgumentParser()
parser.add_argument("--target", type=str, default="homo_lumo_gap")
parser.add_argument("--sample-size", type=int, default=5000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output", type=str, default="../results/ecfp4_tanimoto.pkl")
args = parser.parse_args()

# -------------------- Fixed Setup --------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# -------------------- Load QM9 --------------------
qm9 = load_qm9(args.target)
print(f"Loaded QM9 with {len(qm9)} molecules")

# -------------------- Scaffold split logic (copied) --------------------
qm9_smiles = [data.smiles for data in qm9[:args.sample_size]]
Xs = np.zeros(len(qm9_smiles))  # Dummy features just for splitting
dataset = dc.data.DiskDataset.from_numpy(X=Xs, ids=qm9_smiles)

splitter = dc.splits.ScaffoldSplitter()
split = splitter.split(dataset, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
train_idx, val_idx, test_idx = split
all_indices = train_idx + val_idx + test_idx

print(f"Scaffold split selected {len(all_indices)} molecules")

# -------------------- Generate ECFP4 fingerprints --------------------
def ecfp4_fp(smiles: str, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)

smiles_list = [qm9[i].smiles for i in all_indices]
fps = []
for s in tqdm(smiles_list, desc="Generating ECFP4"):
    fp = ecfp4_fp(s)
    if fp is None:
        raise ValueError(f"Invalid SMILES: {s}")
    fps.append(fp)

# -------------------- Compute pairwise Tanimoto distances --------------------
N = len(fps)
distance_matrix = np.zeros((N, N), dtype=np.float32)

for i in tqdm(range(N), desc="Computing distances"):
    sims = BulkTanimotoSimilarity(fps[i], fps)
    for j in range(N):
        distance_matrix[i, j] = 1 - sims[j]  # similarity → distance

# -------------------- Compute pairwise proximity matrcies --------------------

def compute_proximity_matrix(fingerprints, method="rf", n_estimators=100, max_depth=20, seed=42):
    """
    Compute a tree-based proximity matrix using sklearn RF, quantile RF, or XGBoost.

    Parameters:
        fingerprints (list of rdkit bit vectors)
        method: 'rf', 'qrf', or 'xgboost'
    Returns:
        distance_matrix (np.ndarray)
    """
    print(f"Training {method.upper()} for proximity matrix...")

    X = np.array([np.asarray(fp) for fp in fingerprints])
    y = np.random.rand(X.shape[0])  # dummy targets

    if method == "rf":
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      random_state=seed, n_jobs=-1)
    elif method == "qrf":
        from quantile_forest import RandomForestQuantileRegressor
        model = RandomForestQuantileRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                              random_state=seed, n_jobs=-1)
    elif method == "xgboost":
        from xgboost import XGBRegressor
        model = XGBRegressor(n_estimators=n_estimators, max_depth=max_depth,
                             random_state=seed, n_jobs=-1, tree_method="auto", verbosity=0)
    else:
        raise ValueError(f"Unsupported method: {method}")

    model.fit(X, y)

    if method in {"rf", "qrf"}:
        leaf_indices = model.apply(X)  # shape (n_samples, n_trees)
    elif method == "xgboost":
        # returns shape (n_samples, n_trees)
        leaf_indices = model.apply(X).astype(np.int32)

    print("Computing proximity matrix...")
    N = X.shape[0]
    proximity = np.zeros((N, N), dtype=np.float32)

    for tree_idx in range(leaf_indices.shape[1]):
        leaf = leaf_indices[:, tree_idx]
        leaf_to_indices = {}
        for i, node in enumerate(leaf):
            leaf_to_indices.setdefault(node, []).append(i)
        for indices in leaf_to_indices.values():
            for i in indices:
                for j in indices:
                    proximity[i, j] += 1

    proximity /= leaf_indices.shape[1]
    distance_matrix = 1.0 - np.round(proximity, 3)

    print(f"Finished computing {method.upper()} distance matrix.")
    return distance_matrix


# -------------------- Save Tanimoto distance --------------------
distance_matrix = np.round(distance_matrix, 3)
with open(args.output, "wb") as f:
    pickle.dump({
        "smiles": smiles_list,
        "distance_matrix": distance_matrix,
        "indices": all_indices,
        "seed": args.seed,
        "target": args.target,
        "method": "tanimoto"
    }, f)
print(f"Tanimoto distance matrix saved to {args.output}")

# -------------------- Save RF proximity distance --------------------
rf_dist = compute_proximity_matrix(fps, method="rf")
with open(args.output.replace(".pkl", "_rf.pkl"), "wb") as f:
    pickle.dump({
        "smiles": smiles_list,
        "distance_matrix": rf_dist,
        "indices": all_indices,
        "seed": args.seed,
        "target": args.target,
        "method": "rf_proximity"
    }, f)
print(f"RF proximity matrix saved to {args.output.replace('.pkl', '_rf.pkl')}")

# -------------------- Save QRF proximity distance --------------------
qrf_dist = compute_proximity_matrix(fps, method="qrf")
with open(args.output.replace(".pkl", "_qrf.pkl"), "wb") as f:
    pickle.dump({
        "smiles": smiles_list,
        "distance_matrix": qrf_dist,
        "indices": all_indices,
        "seed": args.seed,
        "target": args.target,
        "method": "qrf_proximity"
    }, f)
print(f"QRF proximity matrix saved to {args.output.replace('.pkl', '_qrf.pkl')}")

# -------------------- Save XGBoost proximity distance --------------------
xgb_dist = compute_proximity_matrix(fps, method="xgboost")
with open(args.output.replace(".pkl", "_xgb.pkl"), "wb") as f:
    pickle.dump({
        "smiles": smiles_list,
        "distance_matrix": xgb_dist,
        "indices": all_indices,
        "seed": args.seed,
        "target": args.target,
        "method": "xgb_proximity"
    }, f)
print(f"XGBoost proximity matrix saved to {args.output.replace('.pkl', '_xgb.pkl')}")
