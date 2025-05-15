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
    print(sims)
    for j in range(N):
        distance_matrix[i, j] = 1 - sims[j]  # similarity → distance

# -------------------- Save result --------------------
distance_matrix = np.round(distance_matrix, 3)
with open(args.output, "wb") as f:
    pickle.dump({
        "smiles": smiles_list,
        "distance_matrix": distance_matrix,
        "indices": all_indices,
        "seed": args.seed,
        "target": args.target
    }, f)

print(f"Tanimoto distance matrix saved to {args.output}")
