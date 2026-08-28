#!/usr/bin/env python3
"""Avalon and Sort & Slice must refuse a molecule they cannot featurise, not return zeros.

Guard for the defect found by the chat D/G close-out audit on 2026-08-27: Avalon
returned np.zeros both for `mol is None` and for any exception, after only a print.
Nothing downstream catches that -- `write_to_mmap` accepts the block unchecked and the
Rust zero-block refusal is keyed to ecfp4 alone -- so an unparseable molecule trained as
if it had real features. ChemBERTa and MHG-GNN were fixed on 2026-08-26; Avalon was not.

This test goes red if the raise is removed.
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])

from rdkit import Chem
from process_and_train import (avalon_fingerprint as avalon,
                               create_sort_and_slice_ecfp_featuriser)

FAILURES = []

def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILURES.append(f"{name}: {e}")
        print(f"  FAIL  {name}: {e}")

def unparseable_raises():
    for bad in ('this is not a molecule', 'C(((', ''):
        try:
            out = avalon(bad)
        except RuntimeError:
            continue
        assert False, (f"avalon({bad!r}) returned {type(out).__name__} instead of raising"
                       + (f"; all-zero = {not out.any()}" if isinstance(out, np.ndarray) else ""))

def valid_still_works():
    out = avalon('CCO')
    assert isinstance(out, np.ndarray), f"expected an array, got {type(out).__name__}"
    assert out.any(), "ethanol produced an all-zero Avalon fingerprint"

# ---------------------------------------------------------------------------
# Sort & Slice -- the same defect, in the one featuriser the 2026-08-27 close-out
# audit did not check (RERUN_PLAN.md 13.12 A5). Two routes, and `mol is None`
# only closes the first:
#   * RDKit cannot parse the SMILES at all -> MolFromSmiles returns None
#   * RDKit parses '' into a VALID Mol with no atoms -> the enumerator returns {},
#     the sum over an empty list returns the SCALAR 0.0, and write_to_mmap turned
#     that shape mismatch into np.zeros(SNS_DIM) and wrote it as real features
# A molecule whose substructures all fall outside the top `vec_dimension` is NOT
# this: it gives a full-width vector of zeros, which is the method as designed.
# ---------------------------------------------------------------------------
SNS_TRAIN = [Chem.MolFromSmiles(s) for s in
             ('CCO', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCCCO', 'c1ccncc1')]


def sns_featuriser(dim=64):
    return create_sort_and_slice_ecfp_featuriser(
        SNS_TRAIN, vec_dimension=dim, print_train_set_info=False)


def sns_unparseable_raises():
    featurise = sns_featuriser()
    for bad in ('this is not a molecule', 'C((('):
        mol = Chem.MolFromSmiles(bad)
        assert mol is None, f"{bad!r} unexpectedly parsed; pick another"
        try:
            out = featurise(mol)
        except RuntimeError:
            continue
        assert False, (f"Sort & Slice on an unparseable molecule returned "
                       f"{type(out).__name__} instead of raising")


def sns_atomless_raises():
    featurise = sns_featuriser()
    mol = Chem.MolFromSmiles('')
    assert mol is not None, "RDKit no longer parses '' into an empty Mol; pick another case"
    try:
        out = featurise(mol)
    except RuntimeError:
        return
    shape = getattr(out, 'shape', '(scalar)')
    assert False, (f"Sort & Slice on an atomless molecule returned {shape} instead of "
                   f"raising -- this is what write_to_mmap zero-filled into a record")


def sns_valid_still_works():
    dim = 64
    out = sns_featuriser(dim)(Chem.MolFromSmiles('CCO'))
    assert out.shape == (dim,), f"expected shape ({dim},), got {getattr(out, 'shape', None)}"
    assert out.any(), "ethanol produced an all-zero Sort & Slice vector"


if __name__ == '__main__':
    print("Avalon failure handling")
    check('an unparseable SMILES raises rather than returning zeros', unparseable_raises)
    check('a valid SMILES still produces a non-zero fingerprint', valid_still_works)
    print("Sort & Slice failure handling")
    check('an unparseable molecule raises rather than returning zeros', sns_unparseable_raises)
    check('an atomless molecule raises rather than returning a scalar', sns_atomless_raises)
    check('a valid molecule still produces a full-width non-zero vector', sns_valid_still_works)
    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)})")
        sys.exit(1)
    print("\nOK")

