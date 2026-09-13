#!/usr/bin/env python
"""Guard on the QM9 pool index.

`scripts/regenerate_qm9_valid_indices.py` rebuilds `data/valid_qm9_indices.pth`
from PyG's QM9 and writes `data/valid_qm9_indices_regenerated.pth` beside it.
This checks that the two agree and that the rule behind them has not moved.

Four checks:

  1. The reference index holds 129,428 positions, sorted, unique, and every one
     of them is inside PyG's QM9 (0 to 130,830).
  2. The regenerated index is element-for-element identical to the reference.
  3. Every position the reference drops fails `Chem.MolFromSmiles` on the SMILES
     PyG stores for it, and every position it keeps passes. By default this is
     checked on a deterministic sample of 2,000 positions, which takes a couple
     of seconds; `--full` checks all 130,831 and takes about two minutes.
  4. None of the dropped positions is a molecule the QM9 release flags as
     uncharacterised. Those are removed one step earlier, by PyG, and the count
     of dropped-and-uncharacterised must be zero.
  5. The reason is in the release's own connection table. Every dropped
     molecule carries a carbon with five bonds in `gdb9.sdf` itself, and no
     kept molecule does. Checked on 200 of each, read with no sanitization.
     Over-bonded NITROGEN is not checked, because RDKit accepts a neutral
     nitrogen with five bonds written as a nitro group and keeps those
     molecules.

Usage:
    python scripts/test_qm9_valid_indices.py
    python scripts/test_qm9_valid_indices.py --full
"""

import argparse
import csv
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

EXPECTED_KEPT = 129428
EXPECTED_DROPPED = 1403
EXPECTED_POSITIONS = 130831
SAMPLE_STRIDE = 65  # 130831 / 65 is about 2,013 positions
SDF_SAMPLE = 200  # molecules of each kind read back out of gdb9.sdf in check 5


def fail(message):
    print("FAIL: %s" % message)
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="re-derive all 130,831")
    parser.add_argument("--qm9-root", default=osp.join(_DATA, "QM9"))
    args = parser.parse_args()

    ok = True

    if not osp.exists(REFERENCE):
        print("FAIL: %s is missing" % REFERENCE)
        return 1
    reference = torch.load(REFERENCE)

    # 1. shape of the reference
    if len(reference) != EXPECTED_KEPT:
        ok = fail("reference holds %d indices, expected %d" % (len(reference), EXPECTED_KEPT))
    if len(set(reference.tolist())) != len(reference):
        ok = fail("reference has repeated indices")
    if not bool((reference[1:] > reference[:-1]).all()):
        ok = fail("reference is not sorted ascending")
    if int(reference.min()) < 0 or int(reference.max()) >= EXPECTED_POSITIONS:
        ok = fail(
            "reference runs %d to %d, outside 0 to %d"
            % (int(reference.min()), int(reference.max()), EXPECTED_POSITIONS - 1)
        )
    if ok:
        print("PASS 1: reference holds %d sorted unique indices in range" % len(reference))

    # 2. the regenerated file
    if not osp.exists(REGENERATED):
        ok = fail(
            "%s is missing -- run scripts/regenerate_qm9_valid_indices.py" % REGENERATED
        )
    else:
        regenerated = torch.load(REGENERATED)
        if not torch.equal(regenerated, reference):
            mine, theirs = set(regenerated.tolist()), set(reference.tolist())
            ok = fail(
                "regenerated and reference differ: %d only in the rebuild, %d only in the reference"
                % (len(mine - theirs), len(theirs - mine))
            )
        else:
            print("PASS 2: the regenerated index is identical to the reference")

    # 3. the rule itself, against PyG's QM9
    from torch_geometric.datasets import QM9

    dataset = QM9(root=args.qm9_root)
    store = dataset._data if hasattr(dataset, "_data") else dataset.data
    smiles = list(store.smiles)
    if len(smiles) != EXPECTED_POSITIONS:
        ok = fail("PyG QM9 holds %d positions, expected %d" % (len(smiles), EXPECTED_POSITIONS))

    kept = set(reference.tolist())
    positions = range(len(smiles)) if args.full else range(0, len(smiles), SAMPLE_STRIDE)
    disagreements = []
    checked = 0
    for position in positions:
        checked += 1
        parses = Chem.MolFromSmiles(smiles[position]) is not None
        if parses != (position in kept):
            disagreements.append(position)
    if disagreements:
        ok = fail(
            "%d of %d checked positions disagree with the RDKit-parses rule, first ten %s"
            % (len(disagreements), checked, disagreements[:10])
        )
    else:
        print(
            "PASS 3: all %d checked positions agree with the RDKit-parses rule%s"
            % (checked, " (full sweep)" if args.full else " (sampled)")
        )

    # 4. the uncharacterised cross-check
    if not osp.exists(EXCLUDED_CSV):
        ok = fail("%s is missing -- run scripts/regenerate_qm9_valid_indices.py" % EXCLUDED_CSV)
    else:
        rows = list(csv.DictReader(open(EXCLUDED_CSV)))
        if len(rows) != EXPECTED_DROPPED:
            ok = fail("%s holds %d rows, expected %d" % (EXCLUDED_CSV, len(rows), EXPECTED_DROPPED))
        overlap = sum(int(row["listed_as_uncharacterised"]) for row in rows)
        if overlap != 0:
            ok = fail("%d dropped molecules are also flagged uncharacterised, expected 0" % overlap)
        else:
            print(
                "PASS 4: %d dropped molecules, none of them flagged uncharacterised"
                % len(rows)
            )

    # 5. the reason, in the release's own connection table
    sdf_path = osp.join(args.qm9_root, "raw", "gdb9.sdf")
    if not osp.exists(sdf_path):
        ok = fail("%s is missing, cannot check the connection tables" % sdf_path)
    elif not osp.exists(EXCLUDED_CSV):
        pass  # already reported by check 4
    else:
        supplier = Chem.SDMolSupplier(sdf_path, removeHs=False, sanitize=False)
        sdf_rows = store.idx.tolist()
        dropped_rows = [int(row["gdb9_sdf_row_zero_based"]) for row in rows][:SDF_SAMPLE]
        kept_positions = sorted(kept)[:: max(1, len(kept) // SDF_SAMPLE)][:SDF_SAMPLE]
        kept_rows = [sdf_rows[position] for position in kept_positions]

        def over_bonded_carbon(row):
            mol = supplier[row]
            if mol is None:
                return False
            return any(
                atom.GetSymbol() == "C"
                and sum(int(b.GetBondTypeAsDouble()) for b in atom.GetBonds()) > 4
                for atom in mol.GetAtoms()
            )

        dropped_without = [row for row in dropped_rows if not over_bonded_carbon(row)]
        kept_with = [row for row in kept_rows if over_bonded_carbon(row)]
        if dropped_without:
            ok = fail(
                "%d of %d dropped molecules have no over-bonded carbon in gdb9.sdf, "
                "first ten rows %s"
                % (len(dropped_without), len(dropped_rows), dropped_without[:10])
            )
        elif kept_with:
            ok = fail(
                "%d of %d KEPT molecules carry an over-bonded carbon in gdb9.sdf, "
                "so that is not what the filter removes, first ten rows %s"
                % (len(kept_with), len(kept_rows), kept_with[:10])
            )
        else:
            print(
                "PASS 5: all %d dropped molecules checked carry a five-bond carbon in "
                "gdb9.sdf, none of %d kept molecules does"
                % (len(dropped_rows), len(kept_rows))
            )

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
