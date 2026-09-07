#!/usr/bin/env python
"""The molecules Sort & Slice cannot represent are dropped from EVERY representation.

    python scripts/test_sns_zero_exclusion.py

WHAT THIS CATCHES. 51 tasks of the QM9 main grid died on

    ValueError: Sort & Slice produced an all-zero count vector for N

N is ammonia. Sort & Slice keeps the top SNS_DIM substructures by TRAINING frequency.
Methane, ammonia and water have exactly one Morgan substructure each, occurring in one
molecule of QM9, so they never reach a top-1024 under any split or seed and their
vector is all zeros. Counted over all 132,480 QM9 SMILES: exactly three molecules.

The exclusion sits in `split_qm9`, which is called OUTSIDE the per-replicate `try` in
main(), so anything that raises in there kills the whole task rather than losing one
cell. That is why every check below matters.

HOW THIS TEST WORKS. It does two things.

  PART ONE reads the source of `split_qm9` and asks whether the exclusion is written
  there at all, in the shape it has to have.

  PART TWO actually RUNS `split_qm9` on a fake forty-molecule dataset that contains
  methane, ammonia, water and an empty SMILES. It cannot import
  scripts/process_and_train.py as a module -- that module pulls in TensorFlow through
  DeepChem and is far too heavy for a laptop test -- so it compiles the REAL top-level
  functions out of the real file and runs those. Nothing here is a reimplementation:
  if the exclusion is deleted from the file, the code this test executes loses it too.

Neither part needs the cluster, a GPU, or the QM9 file.

WHAT WOULD FAIL IF THE EXCLUSION WERE REMOVED.
  * with Sort & Slice requested, `split_qm9` raises the ValueError the guard was
    written to raise, and the "completes" check fails;
  * with only ECFP4 requested, ammonia is kept, and the "same molecules on every
    representation" check fails -- which is the confound the exclusion exists to
    prevent, because representation is a FACTOR in this study;
  * the empty SMILES reaches ECFP4's own all-zero guard, which raises out of
    `split_qm9` and kills the task from a completely different featuriser.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'scripts' / 'process_and_train.py'

FAILS = []


def check(name, ok, detail=''):
    if not ok:
        FAILS.append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f'\n          {detail}' if detail and not ok else ''))


# ---------------------------------------------------------------- part one


def source_checks(src):
    lines = src.splitlines()
    start = next(i for i, l in enumerate(lines, 1) if l.startswith('def split_qm9'))
    end = next(i for i, l in enumerate(lines, 1) if i > start and l.startswith('def '))
    body = '\n'.join(lines[start - 1:end - 1])

    check('the exclusion is inside split_qm9, the function that failed',
          'dropped_sns' in body,
          'split_qm9 has no exclusion -- load_and_split_polaris is a different path '
          'and no job in the run design reaches it')

    for name in ('train_idx', 'val_idx', 'test_idx'):
        check(f'{name} has the dropped molecules removed',
              re.search(rf'{name}\s*=\s*\[i for i in {name} if i not in drop\]', body)
              is not None,
              f'{name} is not filtered, so the splits would disagree')

    head = body[:body.index('dropped_sns')] if 'dropped_sns' in body else body
    check('the featuriser is built for EVERY run, not only when sns is requested',
          "if 'sns' in args.molecular_representations:" not in head,
          'still guarded by the representation, so an ECFP4 job would keep the '
          'molecules an sns job drops and the two would score different data')

    check('the write_to_mmap guard still raises on an all-zero vector',
          'all-zero count vector' in src,
          'the guard was removed -- it is the thing that caught this')

    check('what was dropped is printed on every run',
          'Sort & Slice cannot represent' in body
          and 'nothing excluded' in body,
          'the run has to say which molecules went and how many, every time, '
          'including when the answer is none')

    check('a molecule the featuriser itself refuses is dropped, not raised',
          re.search(r'except RuntimeError:', body) is not None,
          'the featuriser raises for a molecule with no enumerable substructures. '
          'split_qm9 runs outside the per-replicate try, so that kills the task -- '
          'the same failure this exclusion exists to remove, on a new line')

    code = '\n'.join(l.split('#', 1)[0] for l in body.splitlines())
    check('membership is tested against sets, not lists',
          'train_set, val_set, test_set = set(train_idx)' in code
          and not re.search(r'index (?:not )?in (?:train|val|test)_idx', code),
          'scanning an 8,000-entry list costs 1.55 s per pass and there are four '
          'passes, paid once per noise level per replicate by every representation')


# ---------------------------------------------------------------- part two


def load_real_functions(src, sns_dim):
    """Compile the real top-level functions of process_and_train.py.

    The module cannot be imported: it does `from models import *`, which pulls
    DeepChem and TensorFlow in and takes minutes and gigabytes. Every function
    below is nevertheless the file's own source, compiled from the file, so the
    behaviour checked is the behaviour that runs on the cluster.
    """
    import numpy as np
    import struct
    from collections import deque
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    RDLogger.DisableLog('rdApp.*')

    class _Torch:
        """Only `randperm` is reached, and only to shuffle. Seeded by numpy so the
        two runs compared below draw the identical permutation."""
        @staticmethod
        def randperm(n):
            return [int(i) for i in np.random.permutation(n)]

    ns = {
        '__name__': 'process_and_train_subset',
        'np': np, 'struct': struct, 'deque': deque, 'Chem': Chem,
        'rdFingerprintGenerator': rdFingerprintGenerator,
        'MurckoScaffoldSmiles': MurckoScaffoldSmiles,
        'torch': _Torch,
    }

    tree = ast.parse(src)

    # Module-level constants first: they are the default arguments of several of
    # the functions and are evaluated when the def runs.
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name):
            try:
                ns[node.targets[0].id] = eval(  # noqa: S307
                    compile(ast.Expression(node.value), str(SOURCE), 'eval'), ns)
            except Exception:                                       # noqa: BLE001
                pass

    # A small top-k so the cut actually bites on forty molecules. On QM9 the cut is
    # SNS_DIM=1024 against a vocabulary far larger than that; here the vocabulary is
    # small, and without this every substructure present in training would be kept
    # and the property under test would never appear. write_to_mmap reads the same
    # name from the same namespace, so the record and the featuriser stay in step.
    ns['SNS_DIM'] = sns_dim
    ns['SNS_COUNT_DTYPE'] = np.uint16

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            try:
                exec(compile(ast.Module(body=[node], type_ignores=[]),   # noqa: S102
                             str(SOURCE), 'exec'), ns)
            except Exception:                                       # noqa: BLE001
                pass  # a def whose defaults need something we did not stub

    return ns


class _Label:
    def __init__(self, value):
        self.value = float(value)

    def item(self):
        return self.value


class _Molecule:
    def __init__(self, smiles, value):
        self.smiles = smiles
        self.y = _Label(value)


class _FakeQM9:
    """Enough of a torch_geometric dataset for split_qm9: length, index_select,
    shuffle and slicing. index_select really does reorder, as the real one does."""

    def __init__(self, items):
        self.items = list(items)

    def __len__(self):
        return len(self.items)

    def index_select(self, indices):
        return _FakeQM9([self.items[int(i)] for i in indices])

    def shuffle(self):
        return self

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _FakeQM9(self.items[key])
        return self.items[key]


# Ordinary organic molecules, none of them a lone heavy atom, plus the four cases
# under test. The four are what QM9 contains (methane, ammonia, water) plus the
# empty SMILES, which RDKit parses into a VALID molecule with no atoms -- the case
# the featuriser's own comment names, and the one that makes it raise rather than
# return zeros.
ORDINARY = [
    'CCO', 'CCC', 'CCCC', 'CCCCC', 'CC(C)C', 'CC(C)(C)C', 'CCOC', 'COC',
    'CCOCC', 'CC(=O)O', 'CC(=O)C', 'CC=O', 'CCC=O', 'CCN', 'CCNC', 'CNC',
    'CCCN', 'CC#N', 'CCC#N', 'CCCO', 'CCCCO', 'CC(C)O', 'CCOC(C)=O',
    'c1ccccc1', 'Cc1ccccc1', 'c1ccncc1', 'C1CCCCC1', 'C1CCCC1', 'C1CCOC1',
    'CC(N)=O', 'CCC(N)=O', 'OCCO', 'OCCCO', 'NCCN', 'CSC', 'CCS',
]
CANNOT_REPRESENT = ['C', 'N', 'O', '']


def run_split(ns, representations, seed=7):
    """Run the real split_qm9 once and report what it kept and what it dropped."""
    import numpy as np

    molecules = [_Molecule(s, i * 0.1) for i, s in enumerate(ORDINARY + CANNOT_REPRESENT)]
    dataset = _FakeQM9(molecules)

    args = argparse.Namespace(
        split='scaffold',
        sample_size=len(molecules),
        molecular_representations=list(representations),
        k_domains=1,
        max_vocab=64,
        logging=False,
    )

    np.random.seed(seed)
    with tempfile.TemporaryDirectory() as tmp:
        files = {name: open(Path(tmp) / f'{name}.mmap', 'wb+')
                 for name in ('train', 'val', 'test')}
        try:
            shuffled, train_idx, test_idx, val_idx, _groups = \
                ns['split_qm9'](dataset, args, files)
        finally:
            for handle in files.values():
                handle.close()

    kept = sorted(set(train_idx) | set(test_idx) | set(val_idx))
    kept_smiles = sorted(shuffled[i].smiles for i in kept)
    return kept_smiles


def behaviour_checks(src):
    try:
        import numpy  # noqa: F401
        from rdkit import Chem  # noqa: F401
    except ImportError as exc:                                      # noqa: BLE001
        print(f'\n  numpy or rdkit missing ({exc}); the source checks above still ran.')
        return

    # A top-32 cut over this forty-molecule sample. On QM9 the cut is SNS_DIM=1024
    # against a much larger vocabulary; here the vocabulary is small, so the cut has
    # to be small too or every substructure present in training would be kept and the
    # property under test would never appear. Measured on this fixture: at 32 exactly
    # the four unrepresentable molecules drop and all 36 ordinary ones survive, at 64
    # methane's substructure reaches the top-k and it survives, at 128 water's does
    # too. That is the same mechanism as QM9's, which is the point.
    ns = load_real_functions(src, sns_dim=32)
    for needed in ('split_qm9', 'write_to_mmap', 'scaffold_split_indices',
                   'create_sort_and_slice_ecfp_featuriser', 'ecfp4_fingerprint',
                   'build_scaffold_groups'):
        if needed not in ns:
            check(f'the real {needed} could be compiled from the file', False,
                  'the test could not build the code it means to exercise')
            return
    check('the real split_qm9 and its callees compile out of the file', True)

    # 1. The property itself, against the real featuriser: a lone heavy atom whose
    #    one substructure is not in the kept top-k comes back all zeros.
    import numpy as np
    from rdkit import Chem
    trainers = [Chem.MolFromSmiles(s) for s in ORDINARY]
    featurise = ns['create_sort_and_slice_ecfp_featuriser'](
        mols_train=trainers, max_radius=2, pharm_atom_invs=False, bond_invs=True,
        chirality=False, sub_counts=True, vec_dimension=32,
        print_train_set_info=False)
    for smiles, name in (('C', 'methane'), ('N', 'ammonia'), ('O', 'water')):
        vector = np.asarray(featurise(Chem.MolFromSmiles(smiles)))
        check(f'{name} ({smiles}) gets an all-zero Sort & Slice vector',
              not np.any(vector), f'{int(np.count_nonzero(vector))} non-zero entries')
    check('a molecule that IS representable is not all-zero',
          bool(np.any(np.asarray(featurise(Chem.MolFromSmiles('CCO'))))),
          'ethanol came back all-zero, so the test itself is wrong')

    # 2. split_qm9 completes with Sort & Slice requested. Without the exclusion the
    #    write_to_mmap guard raises, and split_qm9 is outside the per-replicate try.
    try:
        with_sns = run_split(ns, ['sns', 'ecfp4'])
        check('split_qm9 completes with Sort & Slice on a sample containing them', True)
    except Exception as exc:                                        # noqa: BLE001
        with_sns = None
        check('split_qm9 completes with Sort & Slice on a sample containing them',
              False, f'{type(exc).__name__}: {exc}')

    # 3. And with Sort & Slice NOT requested. Without the write-loop skip the empty
    #    SMILES reaches ECFP4's own all-zero guard and kills the task from there.
    try:
        without_sns = run_split(ns, ['ecfp4'])
        check('split_qm9 completes when Sort & Slice is NOT requested', True)
    except Exception as exc:                                        # noqa: BLE001
        without_sns = None
        check('split_qm9 completes when Sort & Slice is NOT requested',
              False, f'{type(exc).__name__}: {exc}')

    # 4. The molecules that cannot be represented are gone from what was written.
    #    Checked on each run that got as far as writing anything, so a crash in one
    #    of the two does not hide what the other did.
    for label, kept in (('sns run', with_sns), ('ecfp4-only run', without_sns)):
        if kept is None:
            continue
        left = [s for s in CANNOT_REPRESENT if s in kept]
        check(f'the {label} writes none of the molecules it cannot represent',
              not left,
              f'still written: {left!r}')

    if with_sns is None or without_sns is None:
        return

    # 5. THE POINT OF THE WHOLE THING. Representation is a factor in this study, so
    #    an ECFP4 job and a Sort & Slice job at the same replicate have to write the
    #    same molecule set, or the comparison between the two carries a difference in
    #    the data as well as in the featuriser.
    check('the same molecules are written whether or not sns is requested',
          with_sns == without_sns,
          f'sns run wrote {len(with_sns)}, ecfp4-only run wrote {len(without_sns)}; '
          f'only in one: '
          f'{sorted(set(with_sns) ^ set(without_sns))}')

    # 6. The exclusion took the four it should and not one molecule more. An
    #    exclusion that threw away representable molecules would be worse than the
    #    crash it replaces, because it would be silent.
    missing = [s for s in ORDINARY if s not in with_sns]
    check('every representable molecule is still written',
          not missing,
          f'{len(missing)} representable molecule(s) were dropped as well: {missing}')


def main():
    src = SOURCE.read_text()
    print('Sort & Slice exclusion -- what the file says')
    source_checks(src)
    print('\nSort & Slice exclusion -- what the code does')
    behaviour_checks(src)
    print(f"\n  {len(FAILS)} failure(s): " + ', '.join(FAILS) if FAILS
          else '\n  all checks passed.')
    return 1 if FAILS else 0


if __name__ == '__main__':
    sys.exit(main())
