#!/usr/bin/env python3
"""Check what each representation ACTUALLY is, by computing it and comparing.

WHY THIS EXISTS
---------------
Every worst-in-class defect this project has found was a representation identity
error, and not one of them could have been caught by comparing settings:

  * a fingerprint named ECFP4 that was computed by RDKit's path-based function
    rather than a circular one -- a different substructure family entirely
  * a descriptor vector silently reduced to one presence bit per descriptor
  * substructure COUNTS computed and then thrown away by bit-packing, so the
    model trained on presence while the code asked for counts
  * learned embeddings rescaled using each molecule's own minimum and maximum,
    so no two molecules were on the same scale

All four were found by a person reading the featuriser source. That does not
scale, it does not repeat, and it did not happen for months at a time.

WHAT THIS DOES INSTEAD
----------------------
It computes each representation on the same molecules and compares the RESULT
against reference implementations built here, in the open, from RDKit directly.
It does not read the pipeline's source and it does not trust any name.

  1. IDENTITY. For each fingerprint, compare against several references at once
     -- circular at radius 2 and 3, path-based, and so on -- and report which
     one it actually matches. A fingerprint that matches the path-based
     reference is a path fingerprint, whatever it is called.
  2. INFORMATION LOST. Whether the stored values are binary, small counts, or
     continuous; whether anything that should be a count has been flattened to
     0/1; whether an integer cast could wrap.
  3. PER-MOLECULE SCALING. Whether each row has been rescaled on its own -- if
     every row has the same minimum and maximum, the features are not
     comparable between molecules.
  4. CROSS-PIPELINE. Where both pipelines produce a representation under one
     name, compare the two matrices directly.

USAGE
-----
    # 1. get the QM9 pipeline's real features (needs the Rust binary)
    cd scripts
    python process_and_train.py -d QM9 -t homo_lumo_gap -m rf \\
        -r ecfp4 sns pdv continuous_pdv --noise-level 0.0 -n 300 -b 1 \\
        -s scaffold --dump-features /tmp/qm9feat -f /tmp/throwaway.csv

    # 2. audit them
    python audit_representation_identity.py --qm9-dump /tmp/qm9feat

    # reference checks alone, no pipeline run needed:
    python audit_representation_identity.py --references-only
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors, Descriptors
from rdkit.Chem import RDKFingerprint
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator

RDLogger.DisableLog('rdApp.*')
REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# reference implementations, written out in the open so they can be checked
# ---------------------------------------------------------------------------

def ref_morgan(mols, radius, n_bits=2048, counts=False):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    if counts:
        return np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=np.int32)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.int32)


def ref_rdk_path(mols, n_bits=2048):
    """RDKit's Daylight-style TOPOLOGICAL PATH fingerprint. Not circular."""
    out = np.zeros((len(mols), n_bits), dtype=np.int32)
    for i, m in enumerate(mols):
        fp = RDKFingerprint(m, fpSize=n_bits)
        out[i, list(fp.GetOnBits())] = 1
    return out


def ref_atompair(mols, n_bits=2048):
    gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
    return np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.int32)


def ref_descriptors(mols):
    names = [d[0] for d in Descriptors._descList]
    calc = MolecularDescriptorCalculator(names)
    raw = np.array([calc.CalcDescriptors(m) for m in mols], dtype=float)
    return np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)


REFERENCES = {
    'morgan_r2_bits': lambda m: ref_morgan(m, 2),
    'morgan_r3_bits': lambda m: ref_morgan(m, 3),
    'morgan_r2_counts': lambda m: ref_morgan(m, 2, counts=True),
    'rdk_path_bits': ref_rdk_path,
    'atompair_bits': ref_atompair,
    'rdkit_descriptors': ref_descriptors,
}


# ---------------------------------------------------------------------------
# what a matrix IS, regardless of what it is called
# ---------------------------------------------------------------------------

def describe(X):
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(len(X), -1)
    vals = np.unique(X[:min(len(X), 200)])
    finite = X[np.isfinite(X)]
    kind = ('binary' if set(np.unique(vals)) <= {0, 1}
            else 'small integer counts' if (np.allclose(X, np.round(X)) and X.max() < 256)
            else 'integer' if np.allclose(X, np.round(X))
            else 'continuous')
    row_min, row_max = X.min(axis=1), X.max(axis=1)
    per_row_scaled = bool(len(X) > 2
                          and np.allclose(row_min, row_min[0])
                          and np.allclose(row_max, row_max[0])
                          and kind != 'binary')
    return {
        'shape': X.shape,
        'kind': kind,
        'distinct values (first 200 rows)': int(len(vals)),
        'min': float(finite.min()) if finite.size else float('nan'),
        'max': float(finite.max()) if finite.size else float('nan'),
        'density (non-zero fraction)': float((X != 0).mean()),
        'every row shares one min and max': per_row_scaled,
    }


def agreement(A, B):
    """How alike are two 0/1 matrices? Jaccard over the on-bits, averaged."""
    A = (np.asarray(A) != 0)
    B = (np.asarray(B) != 0)
    if A.shape != B.shape:
        return None
    inter = (A & B).sum(axis=1).astype(float)
    union = (A | B).sum(axis=1).astype(float)
    ok = union > 0
    return float(inter[ok].mean() / union[ok].mean()) if ok.any() else 1.0


def identify(X, refs, name):
    """Say which reference this matrix actually is."""
    print(f'\n  which reference does "{name}" match?')
    X = np.asarray(X)
    scores = []
    for ref_name, R in refs.items():
        R = np.asarray(R)
        if R.shape != X.shape:
            print(f'    {ref_name:20s} shape {R.shape} != {X.shape}, cannot compare')
            continue
        exact = bool(np.array_equal((X != 0), (R != 0)))
        j = agreement(X, R)
        scores.append((j, ref_name, exact))
        flag = '   <-- EXACT MATCH' if exact else ''
        print(f'    {ref_name:20s} overlap {j:.4f}{flag}')
    if not scores:
        return None
    scores.sort(reverse=True)
    best_j, best, exact = scores[0]
    if exact:
        print(f'    => "{name}" IS {best}')
    elif best_j > 0.95:
        print(f'    => "{name}" is essentially {best} (overlap {best_j:.3f})')
    elif best_j < 0.5:
        print(f'    => "{name}" matches NOTHING here well (best {best} at {best_j:.3f}).')
        print(f'       Either it is a different algorithm, or the molecule order differs.')
    else:
        print(f'    => "{name}" is closest to {best} at {best_j:.3f}, which is not a match.')
    return best


# ---------------------------------------------------------------------------
# molecules to test on
# ---------------------------------------------------------------------------

PROBE_SMILES = [
    # small, QM9-like
    'C', 'CC', 'CCO', 'C1CC1', 'c1ccccc1', 'CC(=O)O', 'CCN', 'C1COC1',
    'CC(C)O', 'C#N', 'CC=O', 'OCC(O)CO', 'C1CCNC1', 'c1ccncc1', 'CNC',
    # drug-like, where counts and radius actually differ
    'CC(=O)Nc1ccc(O)cc1',
    'CC(C)Cc1ccc(cc1)C(C)C(=O)O',
    'CN1C=NC2=C1C(=O)N(C)C(=O)N2C',
    'CC(=O)Oc1ccccc1C(=O)O',
    'Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)Cl',
    'CCCCCCCCCCCCCCCC(=O)O',
    'c1ccc2c(c1)ccc3c2cccc3',
    'CCCCCCCCCCCCCCCCCCCC(=O)OCCCCCCCCCCCCCCCCCC',
]


def load_probe(n_repeat=1):
    mols = [Chem.MolFromSmiles(s) for s in PROBE_SMILES * n_repeat]
    keep = [(s, m) for s, m in zip(PROBE_SMILES * n_repeat, mols) if m is not None]
    return [s for s, _ in keep], [m for _, m in keep]


# ---------------------------------------------------------------------------
def audit_references():
    """Show that the references themselves are distinguishable. If a circular
    fingerprint and a path fingerprint looked alike, none of this would mean
    anything."""
    print('=' * 78)
    print('REFERENCES — are these actually different from one another?')
    print('=' * 78)
    smiles, mols = load_probe()
    print(f'  {len(mols)} probe molecules')
    built = {k: f(mols) for k, f in REFERENCES.items()}
    bit_refs = {k: v for k, v in built.items() if k.endswith('_bits')}
    names = list(bit_refs)
    print()
    print('  pairwise overlap between the bit fingerprints:')
    print('    ' + ' ' * 20 + ''.join(f'{n[:12]:>14s}' for n in names))
    worst = 1.0
    for a in names:
        row = ''
        for b in names:
            j = agreement(bit_refs[a], bit_refs[b])
            row += f'{j:>14.3f}'
            if a != b:
                worst = min(worst, 1 - j)
        print(f'    {a:20s}{row}')
    print()
    if worst < 0.2:
        print('  ⚠ two references look too alike — this check would not distinguish them')
        return 1
    print('  OK: the references are clearly different, so a match below means something')
    return 0


def audit_dumps(prefix):
    print('\n' + '=' * 78)
    print(f'QM9 PIPELINE FEATURES — {prefix}__*.npz')
    print('=' * 78)
    files = sorted(glob.glob(f'{prefix}__*.npz'))
    if not files:
        print(f'  no dumps found at {prefix}__*.npz')
        print('  produce them first:')
        print('    cd scripts && python process_and_train.py -d QM9 -t homo_lumo_gap \\')
        print('        -m rf -r ecfp4 sns pdv continuous_pdv --noise-level 0.0 -n 300 \\')
        print(f'        -b 1 -s scaffold --dump-features {prefix} -f /tmp/throwaway.csv')
        return 1

    problems = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        rep = str(d['rep'])
        X = d['x_train']
        print(f'\n{"-" * 78}')
        print(f'{rep}   from {Path(f).name}')
        print(f'{"-" * 78}')
        for k, v in describe(X).items():
            print(f'  {k:34s} {v}')

        info = describe(X)
        # The checks that do not need a reference at all.
        if rep in ('sns',) and info['kind'] == 'binary':
            print('  ✗ DEFECT: this representation is asked for as COUNTS and stored as')
            print('            presence bits. The model never sees a count.')
            problems += 1
        if rep == 'pdv' and info['kind'] == 'binary':
            print('  ✗ DEFECT: the descriptor vector is binarised. The continuous one is')
            print('            a separate representation under a different name.')
            problems += 1
        if info['every row shares one min and max']:
            print('  ✗ DEFECT: every row has the same minimum and maximum, so each molecule')
            print('            has been rescaled on its own and no two are comparable.')
            problems += 1

        # Identity, computed on the SAME molecules in the SAME order.
        smiles = d['smiles_train'] if 'smiles_train' in d.files else None
        if smiles is None:
            print('\n  NOTE: this dump carries no molecules, so identity cannot be checked.')
            print('        Re-run the pipeline with a build that includes smiles_train.')
            continue
        mols = [Chem.MolFromSmiles(str(x)) for x in smiles]
        bad = sum(m is None for m in mols)
        if bad:
            print(f'  ✗ {bad} of {len(mols)} stored SMILES do not parse')
            problems += 1
        keep = [i for i, m in enumerate(mols) if m is not None]
        mols = [mols[i] for i in keep]
        Xk = X[keep]
        if Xk.shape[1] == 2048:
            refs = {k: f(mols) for k, f in REFERENCES.items()
                    if k.endswith('_bits') or k == 'morgan_r2_counts'}
            best = identify(Xk, refs, rep)
            if rep == 'ecfp4' and best not in ('morgan_r2_bits', 'morgan_r2_counts'):
                print('  ✗ DEFECT: this is named ECFP4 but is not a circular radius-2')
                print(f'            fingerprint. It matches {best}.')
                problems += 1
        elif Xk.shape[1] in (200, 208, 210):
            ref = ref_descriptors(mols)
            print(f'\n  descriptor block: {Xk.shape[1]} columns against a '
                  f'{ref.shape[1]}-descriptor reference')
            if Xk.shape[1] == ref.shape[1]:
                same = np.isclose(Xk, ref, rtol=1e-3, atol=1e-6).mean()
                print(f'    fraction of entries equal to the reference: {same:.4f}')
                if same < 0.9 and describe(Xk)['kind'] != 'binary':
                    print('    ✗ the values do not match a plain RDKit descriptor set')
                    problems += 1
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--qm9-dump', default=None,
                    help='prefix passed to process_and_train.py --dump-features')
    ap.add_argument('--references-only', action='store_true',
                    help='just show that the reference implementations are distinguishable')
    ap.add_argument('--strict', action='store_true')
    args = ap.parse_args()

    problems = audit_references()
    if not args.references_only:
        if args.qm9_dump:
            problems += audit_dumps(args.qm9_dump)
        else:
            print('\n  (no --qm9-dump given, so the pipeline\'s own features were not checked)')

    print('\n' + '=' * 78)
    print(f'SUMMARY: {problems} problem(s)')
    print('=' * 78)
    return 1 if (args.strict and problems) else 0


if __name__ == '__main__':
    sys.exit(main())
