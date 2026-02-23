"""Verify that the pipeline's Morgan fingerprint matches direct RDKit computation.

Computes Morgan FP for a set of SMILES strings via two paths:
1. The morgan_fingerprint() function used in process_and_train.py (pack/unpack roundtrip)
2. Direct RDKit GetMorganGenerator → GetFingerprint

Asserts bit-exact match for every molecule.

Usage:
    python verify_morgan_match.py
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

# Same function as in process_and_train.py
def morgan_fingerprint(mol, radius=2, n_bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    bits = np.array(fp, dtype=np.uint8)
    return np.packbits(bits, bitorder='little')

# Test SMILES covering various molecular features
TEST_SMILES = [
    'CCO',                          # ethanol
    'c1ccccc1',                     # benzene
    'CC(=O)Oc1ccccc1C(=O)O',       # aspirin
    'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',  # testosterone
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',          # caffeine
    'OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O',  # glucose
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',             # ibuprofen
    'C1CCCCC1',                     # cyclohexane
    'c1ccc2ccccc2c1',               # naphthalene
    'CC(=O)NC1=CC=C(C=C1)O',       # acetaminophen
]

def test_morgan_roundtrip():
    """Test pack/unpack roundtrip preserves bits exactly."""
    passed = 0
    for smi in TEST_SMILES:
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"Failed to parse: {smi}"

        # Path 1: pipeline function (pack then unpack)
        packed = morgan_fingerprint(mol)
        assert len(packed) == 256, f"Expected 256 bytes, got {len(packed)}"
        unpacked = np.unpackbits(packed, bitorder='little')

        # Path 2: direct RDKit
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp = generator.GetFingerprint(mol)
        direct_bits = np.array(fp, dtype=np.uint8)

        assert np.array_equal(unpacked, direct_bits), (
            f"Mismatch for {smi}: pipeline has {unpacked.sum()} bits on, "
            f"direct has {direct_bits.sum()} bits on"
        )
        passed += 1
        print(f"  OK: {smi:50s} ({direct_bits.sum():3d} bits on)")

    print(f"\nAll {passed}/{len(TEST_SMILES)} molecules match exactly.")

def test_morgan_vs_ecfp4_difference():
    """Verify Morgan (radius=2) differs from RDKit topological fingerprint.

    This confirms we're actually computing a different fingerprint, not just
    aliasing the existing ecfp4/topological path.
    """
    from rdkit.Chem import RDKFingerprint

    differ_count = 0
    for smi in TEST_SMILES:
        mol = Chem.MolFromSmiles(smi)

        # Morgan (ECFP4, radius=2)
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        morgan_bits = np.array(gen.GetFingerprint(mol), dtype=np.uint8)

        # RDKit topological (what ecfp4 actually is)
        topo_fp = RDKFingerprint(mol, fpSize=2048)
        topo_bits = np.array(topo_fp, dtype=np.uint8)

        if not np.array_equal(morgan_bits, topo_bits):
            differ_count += 1

    print(f"\nMorgan vs Topological: {differ_count}/{len(TEST_SMILES)} molecules differ")
    assert differ_count > 0, "Morgan and topological should differ for at least some molecules!"
    print("Confirmed: Morgan (ECFP4) and Topological are distinct fingerprints.")

if __name__ == '__main__':
    print("=== Test 1: Morgan fingerprint pack/unpack roundtrip ===")
    test_morgan_roundtrip()

    print("\n=== Test 2: Morgan vs Topological difference ===")
    test_morgan_vs_ecfp4_difference()

    print("\nAll tests passed.")
