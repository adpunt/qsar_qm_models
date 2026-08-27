#!/usr/bin/env python3
"""Avalon must refuse an unparseable molecule, not return an all-zero fingerprint.

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

from process_and_train import avalon_fingerprint as avalon

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

if __name__ == '__main__':
    print("Avalon failure handling")
    check('an unparseable SMILES raises rather than returning zeros', unparseable_raises)
    check('a valid SMILES still produces a non-zero fingerprint', valid_still_works)
    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)})")
        sys.exit(1)
    print("\nOK")
