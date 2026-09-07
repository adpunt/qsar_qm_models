#!/usr/bin/env python
"""Which molecules Sort & Slice cannot represent, and how many there are.

    python scripts/sns_zero_molecules.py --smiles-file <one SMILES per line>
    python scripts/sns_zero_molecules.py --smiles-file qm9.smi --train-frac 0.8

WHAT KILLED 48 QM9 TASKS ON 2026-09-04.

    ValueError: Sort & Slice produced an all-zero count vector for N. That is a
    molecule with no features carrying a real label into training.

`N` is ammonia. Sort & Slice sorts substructures by how often they occur in the TRAINING
set and slices the top 1024 (SNS_DIM). Ammonia has exactly ONE substructure -- a nitrogen
with three hydrogens and no heavy neighbour -- and in QM9, which is C/N/O/F molecules of
up to nine heavy atoms, that environment occurs in essentially one molecule. Frequency
one does not reach the top 1024, so ammonia's vector is all zeros, and the guard added by
the close-out audit on 2026-08-28 refuses it rather than training a real label against no
features.

The guard is right. This is not a bug in Sort & Slice or in the pipeline: it is a
property of a top-k featuriser meeting a molecule whose whole substructure set is rare.

WHY IT PASSED THE SCREEN AND FAILED THE MAIN GRID. The screen is replicate 0 and the main
grid is replicates 1-9, and each replicate samples its own molecules. The affected
molecules were not in replicate 0's sample. Nothing changed in the code between them.

WHAT THIS SCRIPT IS FOR. The decision -- exclude these molecules, or drop Sort & Slice --
turns entirely on how many there are, and nobody has counted. Ten molecules out of
133,885 is a Methods sentence. Ten thousand is a different study.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--smiles-file', required=True,
                    help='one SMILES per line, or a CSV whose first column is SMILES')
    ap.add_argument('--dim', type=int, default=None,
                    help='SNS_DIM (default: read from process_and_train.py)')
    ap.add_argument('--train-frac', type=float, default=0.8,
                    help='fraction used to FIT the feature selection, as the pipeline '
                         'fits it on the training split alone')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--show', type=int, default=25)
    cli = ap.parse_args()

    try:
        from rdkit import Chem
        from rdkit.Chem import rdFingerprintGenerator
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        raise SystemExit('needs rdkit -- source setup.sh first.')
    import numpy as np

    dim = cli.dim
    if dim is None:
        import re
        m = re.search(r'^SNS_DIM = (\d+)',
                      (ROOT / 'scripts' / 'process_and_train.py').read_text(), re.M)
        dim = int(m.group(1)) if m else 1024
    print(f"  SNS_DIM = {dim}   (the top-k the featuriser keeps)")

    smiles = []
    for line in Path(cli.smiles_file).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        smiles.append(line.split(',')[0].strip().strip('"'))
    if smiles and not Chem.MolFromSmiles(smiles[0]):
        smiles = smiles[1:]                      # a header
    print(f"  {len(smiles):,} SMILES read from {cli.smiles_file}")

    # The same generator settings the pipeline uses (process_and_train.py:1045).
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganAtomInvGen(
            includeRingMembership=True),
        useBondTypes=True, includeChirality=False)

    subs = []
    bad = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad += 1
            subs.append(None)
            continue
        subs.append(gen.GetSparseCountFingerprint(mol).GetNonzeroElements())
    if bad:
        print(f"  {bad} SMILES RDKit could not parse (excluded)")

    rng = np.random.default_rng(cli.seed)
    order = rng.permutation(len(smiles))
    n_train = int(len(smiles) * cli.train_frac)
    train = set(order[:n_train].tolist())

    freq = Counter()
    for i in train:
        if subs[i]:
            freq.update(subs[i].keys())
    kept = {sub for sub, _ in freq.most_common(dim)}
    print(f"  {len(freq):,} distinct substructures in the training split; "
          f"the top {len(kept):,} are kept")

    zero = [(smiles[i], len(subs[i] or {})) for i in range(len(smiles))
            if subs[i] is not None and not (set(subs[i]) & kept)]
    print(f"\n  {len(zero)} molecule(s) of {len(smiles):,} get an ALL-ZERO Sort & Slice "
          f"vector\n  ({100.0 * len(zero) / max(1, len(smiles)):.4f}% of the dataset)")
    if zero:
        print(f"\n  {'SMILES':30s} {'substructures it has':>20s}")
        for smi, n in sorted(zero, key=lambda t: t[1])[:cli.show]:
            print(f"      {smi:30s} {n:>16d}")
        if len(zero) > cli.show:
            print(f"      ... and {len(zero) - cli.show} more")

    print(f"\n  WHAT TO DO WITH THAT NUMBER")
    print(f"    A handful  -- exclude them from EVERY representation, not just Sort &")
    print(f"                  Slice, so the cross-representation tables still compare")
    print(f"                  the same molecules. One Methods sentence.")
    print(f"    Many       -- Sort & Slice cannot represent this dataset at "
          f"SNS_DIM={dim},")
    print(f"                  and the choice is a larger dim or dropping it from the")
    print(f"                  representation set. Both are the author's call.")
    print(f"    Either way, do NOT relax the guard: an all-zero vector trains a real")
    print(f"    label against no features, and nothing downstream can tell it from a")
    print(f"    molecule whose substructures genuinely are all absent.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
