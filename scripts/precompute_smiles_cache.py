import sqlite3
import torch
import numpy as np
import random
from rdkit import Chem
from process_and_train import load_qm9
import argparse
import os

# -------------------- Constants --------------------
BOOTSTRAPPING = 20
RANDOM_SEED = 42
SAMPLE_SIZE = 30_000
TARGET = 0  # Modify if needed

# -------------------- Load QM9 dataset --------------------
dataset = load_qm9('homo_lumo_gap')
print(f"Loaded QM9 with {len(dataset)} molecules")

# -------------------- Setup SQLite DB --------------------
cache_path = "../data/smiles_cache.sqlite"
conn = sqlite3.connect(cache_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS smiles_cache (
    isomeric TEXT PRIMARY KEY,
    canonical TEXT
)
""")
conn.commit()

# -------------------- Collect unique isomeric SMILES --------------------
unique_smiles = set()

for iteration in range(BOOTSTRAPPING):
    seed = (RANDOM_SEED ^ (iteration * 0x5DEECE66D)) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    indices = torch.randperm(len(dataset))[:SAMPLE_SIZE]
    for idx in indices:
        data = dataset[idx]
        unique_smiles.add(data.smiles)

print(f"Collected {len(unique_smiles)} unique SMILES from {BOOTSTRAPPING} bootstraps")

# -------------------- Precompute and store canonical SMILES --------------------
for i, smiles in enumerate(unique_smiles):
    cursor.execute("SELECT 1 FROM smiles_cache WHERE isomeric = ?", (smiles,))
    if cursor.fetchone():
        continue  # Already cached

    mol = Chem.MolFromSmiles(smiles)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False) if mol else None
    cursor.execute("INSERT OR REPLACE INTO smiles_cache (isomeric, canonical) VALUES (?, ?)", (smiles, canonical))

    if i % 1000 == 0:
        conn.commit()
        print(f"Processed {i} SMILES...")

conn.commit()
conn.close()
print("Precomputation complete.")
