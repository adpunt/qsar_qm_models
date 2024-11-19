from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprint
from rdkit.DataStructs.cDataStructs import UIntSparseIntVect

def ecfp_invariants(smiles):
    mol = Chem.MolFromSmiles(smiles)
    invs = []
    for k in range(mol.GetNumAtoms()):
        fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius = 0, atomInvariantsGenerator = rdFingerprintGenerator.GetMorganAtomInvGen(includeRingMembership = True))
        fingerprint = fp_gen.GetSparseCountFingerprint(mol, fromAtoms = [k])        
        invs.append(list(fingerprint.GetNonzeroElements().keys())[0])
    return invs

def ecfp_atom_ids_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    info_dict = {}
    ecfp_settings = {"radius": 2, "use_counts": False, "use_chirality": False, "use_bond_invs": True}
    fp = GetMorganFingerprint(mol,
                              radius=ecfp_settings["radius"],
                              useCounts=ecfp_settings["use_counts"],
                              invariants=ecfp_invariants(smiles),
                              useChirality=ecfp_settings["use_chirality"],
                              useBondTypes=ecfp_settings["use_bond_invs"],
                              bitInfo=info_dict)
    
    # Convert to a standard Python dictionary
    result = dict(fp.GetNonzeroElements())
    return result