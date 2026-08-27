import os
import subprocess
from Bio.PDB import PDBList
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import MolToPDBFile

# ---- CONFIGURATION ----
TARGET_PDB_ID = "3ehy"  # Target protein
OUTPUT_DIR = "./docking_results"
SMILES_LIST = [
    "CCO",  # Example SMILES (ethanol)
    "CCCC",  # Example SMILES (butane)
]  # Replace with actual dataset

# TODO: if I run this on a server the URL will need to change
LIGAND_DIR = "./ligands"
VINA_DIR = "./vina_1.2.6_mac_x86_64"
VINA_URL = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.6/vina_1.2.6_mac_x86_64"
VINA_EXE = "./vina_1.2.6_mac_x86_64"

# Download Vina (DO NOT assume it's a ZIP file)
if not os.path.exists(VINA_EXE):
    print("Downloading AutoDock Vina 1.2.6 for macOS...")
    subprocess.run(["curl", "-L", "-o", VINA_EXE, VINA_URL], check=True)

# Make sure Vina is executable
subprocess.run(["chmod", "+x", VINA_EXE], check=True)

# Add Vina to PATH (equivalent to aliasing in Jupyter)
os.environ["PATH"] += os.pathsep + os.path.abspath(".")

# Test if Vina runs
try:
    subprocess.run([VINA_EXE, "--help"], check=True)
    print("AutoDock Vina is set up correctly!")
except subprocess.CalledProcessError:
    print("Error: AutoDock Vina did not execute correctly.")

# ---- STEP 3: CHECK PDB2PQR INSTALLATION ----
try:
    subprocess.run(["pdb2pqr30", "-h"], check=True, capture_output=True)
    print("pdb2pqr is installed correctly.")
except subprocess.CalledProcessError:
    print("Error: pdb2pqr is not installed.")

# ---- STEP 4: FETCH TARGET PROTEIN ----
pdbl = PDBList()
pdb_file = pdbl.retrieve_pdb_file(TARGET_PDB_ID, pdir=".", file_format="pdb", overwrite=True)
pdb_filename = f"{TARGET_PDB_ID}.pdb"

if not os.path.exists(pdb_filename):
    os.rename(f"pdb{TARGET_PDB_ID}.ent", pdb_filename)

print(f"Downloaded target PDB file: {pdb_filename}")

# ---- STEP 5: CONVERT SMILES TO PDBQT ----
os.makedirs(LIGAND_DIR, exist_ok=True)

for i, smiles in enumerate(SMILES_LIST):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)  # Add hydrogens
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())  # Generate 3D conformation

    ligand_pdb = os.path.join(LIGAND_DIR, f"ligand_{i}.pdb")
    ligand_pdbqt = os.path.join(LIGAND_DIR, f"ligand_{i}.pdbqt")

    MolToPDBFile(mol, ligand_pdb)  # Save as PDB

    # Convert PDB to PDBQT using OpenBabel
    subprocess.run(["obabel", ligand_pdb, "-O", ligand_pdbqt, "--gen3D"], check=True)
    print(f"Converted {smiles} to {ligand_pdbqt}")

# ---- STEP 6: RUN AUTODOCK VINA ----
os.makedirs(OUTPUT_DIR, exist_ok=True)

for i, smiles in enumerate(SMILES_LIST):
    ligand_pdbqt = os.path.join(LIGAND_DIR, f"ligand_{i}.pdbqt")
    output_pdbqt = os.path.join(OUTPUT_DIR, f"docked_ligand_{i}.pdbqt")

    vina_command = [
        VINA_EXE,
        "--receptor", pdb_filename,
        "--ligand", ligand_pdbqt,
        "--out", output_pdbqt,
        "--center_x", "0", "--center_y", "0", "--center_z", "0",
        "--size_x", "20", "--size_y", "20", "--size_z", "20",
    ]

    subprocess.run(vina_command, check=True)
    print(f"Docking completed: {output_pdbqt}")

print("All docking runs finished successfully!")
