#!/usr/bin/env python3
"""Build the two files `rust_processor --self-test` wants, from real QM9.

    python scripts/make_selftest_inputs.py [n] [out-stem]

WHY THIS EXISTS
---------------
`--self-test` is the preflight gate the runbook points every operator at, and it
takes a labels file and a scaffold map that nothing in the repository produced.
Before this, the only inputs lying around were the synthetic 4,000-label file
`crosscheck_noise_pattern.py` writes into `.crosscheck_tmp/`. Running the
preflight on that is not the preflight: on 2026-08-30 it failed there --
`grouped_shifted`'s mean delivered dose landed 3.2 standard errors low over 200
seeds -- while passing on real QM9 at both 5,000 and all 133,885 labels.
`grouped_shifted` has by far the widest per-run spread of any condition, so it is
the one most likely to trip a three-sigma band by chance, and the difference was
the input rather than the injector.

The scaffold rule is NOT reimplemented here. It is
`crosscheck_injectors.qm9_scaffold_groups`, which iterates the SDF by record
position (1,405 records do not parse in RDKit, and skipping them would shift
every later scaffold against its label) and gives each acyclic molecule its own
singleton group (32.2% of QM9 is acyclic, and one bucket for all of them would
let a single offset draw move a third of the dataset).

The self-test keys everything by the string in the first column, so a stable
per-record id is enough and cannot collide with a real SMILES.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import crosscheck_injectors as cc                     # noqa: E402


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    stem = sys.argv[2] if len(sys.argv) > 2 else 'qm9_selftest'

    y = cc.qm9_labels()
    if n > len(y):
        sys.exit(f"asked for {n} labels; QM9 has {len(y)}")
    cache = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'scratchpad_crosscheck', 'groups.txt')
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    groups = cc.qm9_scaffold_groups(len(y), cache=cache)
    y, groups = y[:n], groups[:n]

    names = [f'q{i}' for i in range(n)]
    with open(stem + '.csv', 'w') as fh:
        fh.write('canonical_smiles,y\n')
        for name, value in zip(names, y):
            fh.write(f'{name},{value!r}\n')
    with open(stem + '.groups.json', 'w') as fh:
        json.dump({name: int(g) for name, g in zip(names, groups)}, fh)

    print(f"{n} labels  mean {y.mean():.4f}  SD {y.std(ddof=1):.4f}  "
          f"{len(set(groups.tolist()))} scaffold groups")
    print(f"  {stem}.csv")
    print(f"  {stem}.groups.json")
    print(f"\n  ./rust/target/release/rust_processor --self-test {stem}.csv "
          f"--scaffold-file {stem}.groups.json")


if __name__ == '__main__':
    main()
