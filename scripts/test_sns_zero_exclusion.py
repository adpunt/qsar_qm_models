#!/usr/bin/env python
"""The molecules Sort & Slice cannot represent are dropped from EVERY representation.

    python scripts/test_sns_zero_exclusion.py

WHAT THIS CATCHES. 51 tasks of the QM9 main grid died on

    ValueError: Sort & Slice produced an all-zero count vector for N

N is ammonia. Sort & Slice keeps the top SNS_DIM substructures by TRAINING frequency.
Methane, ammonia and water have exactly one Morgan substructure each, occurring in one
molecule of QM9, so they never reach a top-1024 under any split or seed and their
vector is all zeros. Counted over all 132,480 QM9 SMILES: exactly three molecules.

Four things are checked, because fixing fewer would leave the defect:
  1. the property is real -- a single-atom molecule really does get an all-zero vector
     from a featuriser fitted on anything else;
  2. the guard in write_to_mmap still raises, so nothing can slip through silently;
  3. the exclusion is applied in split_qm9, which is the function that failed, and to
     ALL THREE index lists -- not to the training split alone;
  4. the featuriser is built for EVERY run, not only when Sort & Slice is asked for.
     If it were built only for Sort & Slice, an ECFP4 job would keep those molecules
     and the cross-representation tables would stop comparing the same molecule set,
     which is a confound in the one table this study exists to produce.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))

FAILS = []


def check(name, ok, detail=''):
    if not ok:
        FAILS.append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f'\n          {detail}' if detail and not ok else ''))


def main():
    src = (ROOT / 'scripts' / 'process_and_train.py').read_text()
    lines = src.splitlines()

    start = next(i for i, l in enumerate(lines, 1) if l.startswith('def split_qm9'))
    end = next(i for i, l in enumerate(lines, 1) if i > start and l.startswith('def '))
    body = '\n'.join(lines[start - 1:end - 1])

    # 3. The exclusion is in the function that actually failed.
    check('the exclusion is inside split_qm9, the function that failed',
          'dropped_sns' in body,
          'split_qm9 has no exclusion -- load_and_split_polaris is a different path '
          'and no job in the run design reaches it')
    for name in ('train_idx', 'val_idx', 'test_idx'):
        check(f'{name} has the dropped molecules removed',
              re.search(rf'{name}\s*=\s*\[i for i in {name} if i not in drop\]', body)
              is not None,
              f'{name} is not filtered, so the splits would disagree')

    # 4. Built for every run, not only for Sort & Slice.
    head = body[:body.index('dropped_sns')] if 'dropped_sns' in body else body
    check('the featuriser is built for EVERY run, not only when sns is requested',
          "if 'sns' in args.molecular_representations:" not in head,
          'still guarded by the representation, so an ECFP4 job would keep the '
          'molecules an sns job drops and the two would score different data')

    # 2. The guard stays.
    check('the write_to_mmap guard still raises on an all-zero vector',
          'all-zero count vector' in src,
          'the guard was removed -- it is the thing that caught this')

    # 1. The property itself, against the real featuriser.
    try:
        from rdkit import Chem, RDLogger
        RDLogger.DisableLog('rdApp.*')
        import numpy as np
    except ImportError:
        print('\n  rdkit or numpy missing -- the three source checks above still ran.')
        print(f"\n  {len(FAILS)} failure(s)" if FAILS else '\n  all checks passed.')
        return 1 if FAILS else 0

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'pat', ROOT / 'scripts' / 'process_and_train.py')
    try:
        pat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pat)
        featurise = pat.create_sort_and_slice_ecfp_featuriser
        dim = pat.SNS_DIM
    except Exception as exc:                                  # noqa: BLE001
        print(f'\n  could not import process_and_train ({type(exc).__name__}); '
              f'the source checks above still ran.')
        print(f"\n  {len(FAILS)} failure(s)" if FAILS else '\n  all checks passed.')
        return 1 if FAILS else 0

    # A training set of ordinary organic molecules, none of them a lone heavy atom.
    train = ['CCO', 'CCC', 'c1ccccc1', 'CC(=O)O', 'CCN', 'CCOC', 'C1CCCCC1',
             'CC(C)O', 'CCCCO', 'c1ccncc1', 'CC#N', 'CNC', 'COC', 'CC=O']
    mols = [Chem.MolFromSmiles(s) for s in train]
    f = featurise(mols_train=mols, max_radius=2, pharm_atom_invs=False,
                  bond_invs=True, chirality=False, sub_counts=True,
                  vec_dimension=dim, print_train_set_info=False)

    for smi, name in (('C', 'methane'), ('N', 'ammonia'), ('O', 'water')):
        v = np.asarray(f(Chem.MolFromSmiles(smi)))
        check(f'{name} ({smi}) gets an all-zero Sort & Slice vector',
              not np.any(v), f'{int(np.count_nonzero(v))} non-zero entries')
    v = np.asarray(f(Chem.MolFromSmiles('CCO')))
    check('a molecule that IS representable is not dropped', bool(np.any(v)),
          'ethanol came back all-zero, so the test itself is wrong')

    print(f"\n  {len(FAILS)} failure(s)" if FAILS else '\n  all checks passed.')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
