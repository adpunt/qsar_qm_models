#!/usr/bin/env python
"""'ECFP4' is a Morgan radius-2 fingerprint, on both pipelines.

The QM9 side computed it with `rdk_fingerprint_mol`, which is RDKit's
`RDKFingerprintMol` -- the PATH fingerprint, a different fingerprint entirely
(RERUN_PLAN.md §2.13). Measured on the first 1,500 QM9 molecules: the path
fingerprint and Morgan radius 2 agreed on ZERO of them, and methane, ammonia and
water came back all-zero from the path fingerprint, because a molecule with one
heavy atom has no bond paths. Those rows passed the featurisation gate, which
only refuses a fingerprint that FAILED to compute.

The rdkit-sys binding exposes only `morgan_fingerprint_mol`, hardcoded to radius
3, so there was no route to radius 2 on the Rust side at all. It is computed in
Python now and carried through the record.

Three properties:

  1. the pipeline's ECFP4 equals a direct Morgan radius-2, 2048-bit fingerprint;
  2. CONTROL: it does NOT equal the path fingerprint, so this check can tell the
     two apart;
  3. an all-zero fingerprint is refused by name rather than written.

Run it directly:  python scripts/test_ecfp4_identity.py
"""

import os
import sys
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from process_and_train import (ECFP4_BITS, ECFP4_RADIUS,  # noqa: E402
                               ecfp4_fingerprint)

# Real molecules, including the three that break the path fingerprint. QM9 stores
# explicit-hydrogen SMILES, so they are written the way the dataset writes them.
MOLECULES = [
    "[H]C([H])([H])[H]",            # methane
    "[H]N([H])[H]",                 # ammonia
    "[H]O[H]",                      # water
    "[H]OC([H])([H])C([H])([H])[H]",
    "c1ccccc1",
    "CC(=O)Oc1ccccc1C(=O)O",        # aspirin
]


def unpack(packed):
    return np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")


def morgan(smiles):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=ECFP4_RADIUS,
                                                    fpSize=ECFP4_BITS)
    return np.array(gen.GetFingerprint(Chem.MolFromSmiles(smiles)), dtype=np.uint8)


def path_fingerprint(smiles):
    return np.array(Chem.RDKFingerprint(Chem.MolFromSmiles(smiles)), dtype=np.uint8)


def it_is_a_morgan_radius_two_fingerprint():
    for smiles in MOLECULES:
        got = unpack(ecfp4_fingerprint(smiles))
        assert got.size == ECFP4_BITS, (smiles, got.size)
        assert np.array_equal(got, morgan(smiles)), (
            f"{smiles}: the pipeline's ECFP4 is not a Morgan radius-"
            f"{ECFP4_RADIUS} fingerprint")
    print(f"    {len(MOLECULES)} molecules, every bit matches Morgan radius "
          f"{ECFP4_RADIUS} at {ECFP4_BITS} bits")


def it_is_not_the_path_fingerprint():
    differ = sum(1 for s in MOLECULES
                 if not np.array_equal(unpack(ecfp4_fingerprint(s)),
                                       path_fingerprint(s)))
    assert differ == len(MOLECULES), (
        f"only {differ} of {len(MOLECULES)} differ from RDKFingerprintMol, so "
        f"this check cannot tell the two fingerprints apart")
    zeros = [s for s in MOLECULES if not path_fingerprint(s).any()]
    assert zeros, ("no molecule here gives an all-zero PATH fingerprint, so the "
                   "control has lost its point")
    print(f"    all {len(MOLECULES)} differ from the path fingerprint; "
          f"{len(zeros)} of them are all-zero under it "
          f"({', '.join(zeros)})")


def an_all_zero_fingerprint_is_refused():
    # A molecule whose Morgan fingerprint is genuinely empty does not exist for
    # r=2 (every atom sets its own bit), so drive the guard directly.
    import process_and_train as P

    real = P._ECFP4_GENERATOR

    class Empty:
        def GetFingerprint(self, mol):
            return [0] * ECFP4_BITS

    P._ECFP4_GENERATOR = Empty()
    try:
        ecfp4_fingerprint("c1ccccc1")
    except RuntimeError as exc:
        assert "all zeros" in str(exc), str(exc)
        print("    an all-zero fingerprint raises instead of being written")
    else:
        raise AssertionError("an all-zero fingerprint was written")
    finally:
        P._ECFP4_GENERATOR = real


def check(name, fn):
    print(f"  {name}")
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        print(f"    FAIL: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False
    print("    ok")
    return True


def main():
    print("ECFP4 identity (RERUN_PLAN.md §2.13)")
    results = [
        check("it is a Morgan radius-2 fingerprint",
              it_is_a_morgan_radius_two_fingerprint),
        check("it is not the path fingerprint", it_is_not_the_path_fingerprint),
        check("an all-zero fingerprint is refused",
              an_all_zero_fingerprint_is_refused),
    ]
    if not all(results):
        print("\nFAIL: 'ECFP4' does not name a Morgan radius-2 fingerprint")
        return 1
    print("\nOK: 'ECFP4' is one fingerprint across both pipelines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
