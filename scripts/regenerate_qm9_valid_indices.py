#!/usr/bin/env python
"""Rebuild the QM9 pool index and compare it against data/valid_qm9_indices.pth.

WHY THIS EXISTS
---------------
`data/valid_qm9_indices.pth` is a binary dated 12 November 2024. Five scripts
read it (`process_and_train.py`, `clean_noise.py`, `domain_clustering.py`,
`run_qm_qsar_models.py`, `noise_mitigation.py`) and nothing writes it. Every one
of them calls it "molecules that cannot be processed by RDKit" in a comment and
none of them says how the list was made.

WHAT THE FILTER IS
------------------
Each position in PyG's QM9 carries a `smiles` string that PyG itself produced
with `Chem.MolToSmiles(mol, isomericSmiles=True)` from a molecule read out of
`gdb9.sdf` with `sanitize=False`. A position is kept when
`Chem.MolFromSmiles(smiles)` returns a molecule, and dropped when it returns
None. That is the same call the pipeline makes on every molecule it featurises
(`process_and_train.py:1244`), so a position that fails here fails there.

The molecules flagged uncharacterised in the QM9 release are NOT dropped by this
script. PyG removes them one step earlier, inside `QM9.process()`, by reading
`data/QM9/raw/uncharacterized.txt` and skipping those rows of `gdb9.sdf` before
anything is written to `data/QM9/processed/data_v3.pt`. This script reads that
same file only to report, for each dropped position, whether it is also
uncharacterised -- a cross-check that should come back zero.

WHAT IT WRITES
--------------
Nothing is overwritten. Three new files:
  data/valid_qm9_indices_regenerated.pth   the rebuilt index, same format
  data/qm9_excluded_molecules.csv          one row per dropped position
  data/qm9_pool_provenance.json            the counts, for a script to read

Exit status is 0 when the rebuilt index matches `data/valid_qm9_indices.pth`
element for element, and 1 when it does not.

Usage:
    python scripts/regenerate_qm9_valid_indices.py
    python scripts/regenerate_qm9_valid_indices.py --qm9-root /path/to/QM9
"""

import argparse
import collections
import csv
import json
import os.path as osp
import sys
import warnings

warnings.filterwarnings("ignore")

import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_HERE = osp.dirname(osp.abspath(__file__))
_ROOT = osp.dirname(_HERE)
_DATA = osp.join(_ROOT, "data")

REFERENCE = osp.join(_DATA, "valid_qm9_indices.pth")
REGENERATED = osp.join(_DATA, "valid_qm9_indices_regenerated.pth")
EXCLUDED_CSV = osp.join(_DATA, "qm9_excluded_molecules.csv")
PROVENANCE = osp.join(_DATA, "qm9_pool_provenance.json")

# The largest bond-order sum each element may carry when it is uncharged.
NEUTRAL_VALENCE = {"C": 4, "N": 3, "O": 2, "F": 1, "H": 1}


def permitted_bonds(atom):
    """Bond-order sum this atom may carry, allowing for its formal charge.

    A nitrogen carrying +1 takes four bonds and an oxygen carrying -1 takes
    one. Comparing against the uncharged number alone counted five ammonium
    nitrogens in the dropped molecules as violations when they are ordinary,
    and flagged about one in ninety of the molecules that are KEPT.
    """
    symbol = atom.GetSymbol()
    charge = atom.GetFormalCharge()
    base = NEUTRAL_VALENCE.get(symbol, 8)
    if symbol in ("N", "O", "F"):
        return base + charge
    if symbol == "C":
        return base - abs(charge)
    return base


def read_uncharacterized(qm9_root):
    """Row numbers of gdb9.sdf that the QM9 release flags as uncharacterised.

    Returned zero-based, to match the `idx` PyG stores on every molecule. The
    slice `[9:-2]` and the `-1` are copied from `torch_geometric.datasets.qm9`
    so that this reads the file exactly as PyG read it.
    """
    path = osp.join(qm9_root, "raw", "uncharacterized.txt")
    if not osp.exists(path):
        return None
    with open(path, "r") as handle:
        return set(int(line.split()[0]) - 1 for line in handle.read().split("\n")[9:-2])


def sanitization_failure(smiles):
    """The reason RDKit refuses a SMILES string, as a short label."""
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return "unparseable before sanitization"
    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:  # RDKit raises several unrelated exception types
        text = str(exc).strip()
        return text.split("\n")[0][:200]
    return "parses with sanitize=False and sanitizes, but MolFromSmiles returned None"


def over_valent_atoms(mol):
    """The atoms of `mol` carrying more bonds than their element and charge permit."""
    return [
        atom
        for atom in mol.GetAtoms()
        if sum(int(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds())
        > permitted_bonds(atom)
    ]


def check_raw_sdf(qm9_root, sdf_rows, kept_rows=()):
    """Read the dropped molecules back out of gdb9.sdf and count over-valent atoms.

    The point of this is provenance. If the release's own connection table
    already gives an atom more bonds than its element permits, the exclusion is
    a property of the published file and not of anything this repository did to
    it. Bond orders are summed straight off the SDF bond block, with no
    sanitization, which is the only way to read a file RDKit refuses to sanitize.

    Over-valent nitrogen does NOT discriminate. RDKit accepts a neutral
    nitrogen carrying five bonds when it is written as a nitro group,
    `N(=O)=O`, and molecules like that are kept. So `kept_rows` runs the same
    scan over molecules the filter keeps, and the number that matters is the
    carbon one: a carbon with five bonds is what RDKit refuses.
    """
    path = osp.join(qm9_root, "raw", "gdb9.sdf")
    if not osp.exists(path):
        print("gdb9.sdf not found under %s/raw, skipping the connection-table check" % qm9_root)
        return None

    supplier = Chem.SDMolSupplier(path, removeHs=False, sanitize=False)
    over_valent_molecules = 0
    over_valent_carbon_molecules = 0
    by_element = collections.Counter()
    heavy_atoms = collections.Counter()
    ring_closures = collections.Counter()
    for row in sdf_rows:
        mol = supplier[row]
        if mol is None:
            continue
        over = over_valent_atoms(mol)
        if over:
            over_valent_molecules += 1
        if any(atom.GetSymbol() == "C" for atom in over):
            over_valent_carbon_molecules += 1
        for atom in over:
            by_element[atom.GetSymbol()] += 1
        heavy_atoms[sum(1 for a in mol.GetAtoms() if a.GetSymbol() != "H")] += 1
        # Rings without ring perception, which needs a sanitized molecule:
        # for one connected fragment, bonds minus atoms plus one.
        ring_closures[mol.GetNumBonds() - mol.GetNumAtoms() + 1] += 1

    kept_over_valent = 0
    kept_over_valent_carbon = 0
    kept_checked = 0
    for row in kept_rows:
        mol = supplier[row]
        if mol is None:
            continue
        kept_checked += 1
        over = over_valent_atoms(mol)
        kept_over_valent += int(bool(over))
        kept_over_valent_carbon += int(any(atom.GetSymbol() == "C" for atom in over))

    report = {
        "dropped_over_valent_in_sdf": over_valent_molecules,
        "dropped_over_valent_carbon_in_sdf": over_valent_carbon_molecules,
        "over_valent_atoms_by_element": dict(by_element),
        "dropped_by_heavy_atom_count": {str(k): v for k, v in sorted(heavy_atoms.items())},
        "dropped_by_ring_count": {str(k): v for k, v in sorted(ring_closures.items())},
    }
    if kept_checked:
        report["kept_molecules_scanned_as_a_control"] = kept_checked
        report["kept_over_valent_in_sdf"] = kept_over_valent
        report["kept_over_valent_carbon_in_sdf"] = kept_over_valent_carbon
    return report


def write_provenance(path, payload):
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote %s" % path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qm9-root",
        default=osp.join(_DATA, "QM9"),
        help="PyG QM9 root directory (the one holding raw/ and processed/)",
    )
    parser.add_argument(
        "--reference", default=REFERENCE, help="index file to compare against"
    )
    parser.add_argument(
        "--out", default=REGENERATED, help="where to write the rebuilt index"
    )
    parser.add_argument(
        "--excluded-csv", default=EXCLUDED_CSV, help="where to write the dropped rows"
    )
    parser.add_argument(
        "--provenance", default=PROVENANCE, help="where to write the counts as JSON"
    )
    parser.add_argument(
        "--no-sdf-check",
        action="store_true",
        help="skip reading gdb9.sdf back to see whether the dropped molecules were "
        "already over-valent in the release's own connection table",
    )
    args = parser.parse_args()

    from torch_geometric.datasets import QM9

    dataset = QM9(root=args.qm9_root)
    store = dataset._data if hasattr(dataset, "_data") else dataset.data
    smiles = list(store.smiles)
    sdf_rows = store.idx.tolist()
    names = list(store.name)
    total = len(smiles)
    print("PyG QM9 positions: %d" % total)

    uncharacterized = read_uncharacterized(args.qm9_root)
    if uncharacterized is None:
        print("uncharacterized.txt not found under %s/raw" % args.qm9_root)
    else:
        print("uncharacterised rows listed in the release: %d" % len(uncharacterized))

    valid = []
    dropped = []
    for position, smi in enumerate(smiles):
        if Chem.MolFromSmiles(smi) is not None:
            valid.append(position)
        else:
            dropped.append(position)

    print("kept: %d" % len(valid))
    print("dropped: %d" % len(dropped))

    index = torch.tensor(valid, dtype=torch.int64)
    torch.save(index, args.out)
    print("wrote %s" % args.out)

    overlap = 0
    with open(args.excluded_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pyg_position",
                "gdb9_sdf_row_zero_based",
                "qm9_name",
                "smiles_from_pyg",
                "listed_as_uncharacterised",
                "rdkit_failure",
            ]
        )
        for position in dropped:
            row = sdf_rows[position]
            flagged = uncharacterized is not None and row in uncharacterized
            overlap += int(flagged)
            writer.writerow(
                [
                    position,
                    row,
                    names[position],
                    smiles[position],
                    int(flagged),
                    sanitization_failure(smiles[position]),
                ]
            )
    print("wrote %s" % args.excluded_csv)
    print("dropped positions that are ALSO flagged uncharacterised: %d" % overlap)

    sdf_report = None
    if not args.no_sdf_check:
        control_rows = [sdf_rows[p] for p in valid[:: max(1, len(valid) // 2000)]]
        sdf_report = check_raw_sdf(
            args.qm9_root, [sdf_rows[p] for p in dropped], control_rows
        )
        if sdf_report is not None:
            print(
                "of the %d dropped, carrying a carbon with more bonds than it permits "
                "in gdb9.sdf itself: %d"
                % (len(dropped), sdf_report["dropped_over_valent_carbon_in_sdf"])
            )
            print(
                "over-bonded atoms in those connection tables, by element: %s"
                % sdf_report["over_valent_atoms_by_element"]
            )
            print(
                "control, %d KEPT molecules scanned the same way: %d carry an "
                "over-bonded carbon, %d an over-bonded atom of any element"
                % (
                    sdf_report.get("kept_molecules_scanned_as_a_control", 0),
                    sdf_report.get("kept_over_valent_carbon_in_sdf", 0),
                    sdf_report.get("kept_over_valent_in_sdf", 0),
                )
            )

    import rdkit
    import torch_geometric

    provenance = {
        "written_by": "scripts/regenerate_qm9_valid_indices.py",
        "qm9_root": args.qm9_root,
        "pyg_qm9_positions": total,
        "uncharacterised_rows_in_release": None if uncharacterized is None else len(uncharacterized),
        "kept": len(valid),
        "dropped": len(dropped),
        "dropped_also_uncharacterised": overlap,
        "filter": "Chem.MolFromSmiles(data.smiles) is not None",
        "matches_reference": None,
        "rdkit_version": rdkit.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "torch_version": torch.__version__,
    }
    if sdf_report is not None:
        provenance.update(sdf_report)

    if not osp.exists(args.reference):
        print("no reference at %s, nothing to compare" % args.reference)
        write_provenance(args.provenance, provenance)
        return 0

    reference = torch.load(args.reference)
    print("reference holds %d indices" % len(reference))
    if len(reference) == len(index) and bool(torch.equal(reference, index)):
        print("MATCH: the rebuilt index is identical to %s" % args.reference)
        provenance["matches_reference"] = True
        write_provenance(args.provenance, provenance)
        return 0

    mine = set(index.tolist())
    theirs = set(reference.tolist())
    print("MISMATCH")
    print("  in the rebuild and not in the reference: %d" % len(mine - theirs))
    print("  in the reference and not in the rebuild: %d" % len(theirs - mine))
    print("  first ten of each: %s | %s" % (sorted(mine - theirs)[:10], sorted(theirs - mine)[:10]))
    provenance["matches_reference"] = False
    provenance["only_in_rebuild"] = len(mine - theirs)
    provenance["only_in_reference"] = len(theirs - mine)
    write_provenance(args.provenance, provenance)
    return 1


if __name__ == "__main__":
    sys.exit(main())
