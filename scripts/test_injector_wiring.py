#!/usr/bin/env python
"""The Python side of the noise wiring, tested without the training stack.

`process_and_train.py` cannot be imported on a machine without torch_geometric,
deepchem and polaris, so the three helpers chat A added to it had never actually
been run -- only read. Two of them carry real rules:

  build_scaffold_groups   splits acyclic molecules into singleton groups
                          (NOISE_DESIGN.md 2a rule 2). Left as one group they are
                          a third of QM9, and a single offset draw moves a third
                          of the dataset at once.
  retired-flag refusal    a job script written against the old noise scheme must
                          not run silently under the new one, where the level
                          means something different.
  record_noise_manifest   every results file gets the delivered dose beside it,
                          which is the column whose absence hid the confound.

This lifts those definitions out of the source with `ast` and exercises them, so
the rules are executed rather than assumed. Exits non-zero on failure.

    python scripts/test_injector_wiring.py
"""
import ast
import csv
import json
import os
import sys
import tempfile
import types
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, 'scripts', 'process_and_train.py')

WANTED = {'build_scaffold_groups', 'record_noise_manifest', 'noise_seeds_for_level'}


def load_helpers():
    """Pull the helper definitions out of the source without importing it."""
    tree = ast.parse(open(SOURCE).read())
    picked = [n for n in tree.body
              if isinstance(n, ast.FunctionDef) and n.name in WANTED]
    missing = WANTED - {n.name for n in picked}
    if missing:
        sys.exit(f"process_and_train.py no longer defines: {sorted(missing)}")

    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
    RDLogger.DisableLog('rdApp.*')

    ns = {'MurckoScaffoldSmiles': MurckoScaffoldSmiles,
          'os': os, 'json': json, 'csv': csv, 'zlib': zlib, 'print': print}
    exec(compile(ast.Module(body=picked, type_ignores=[]), SOURCE, 'exec'), ns)
    return ns


def check(name, condition, detail=''):
    print(f"  {'ok  ' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail else ''))
    return 0 if condition else 1


def main():
    ns = load_helpers()
    build_scaffold_groups = ns['build_scaffold_groups']
    record_noise_manifest = ns['record_noise_manifest']
    noise_seeds_for_level = ns['noise_seeds_for_level']
    failures = 0

    print("build_scaffold_groups")
    # Four molecules sharing a benzene scaffold, plus three acyclic ones. The
    # acyclic molecules have an EMPTY Murcko scaffold and RDKit returns the same
    # empty string for all of them.
    ringed = ['c1ccccc1C', 'c1ccccc1CC', 'c1ccccc1CCC', 'c1ccccc1O']
    acyclic = ['CCCC', 'CCCCO', 'CC(C)CO']
    two_ring = ['C1CCCCC1C', 'C1CCCCC1CC']
    groups = build_scaffold_groups(ringed + acyclic + two_ring)

    failures += check("every molecule is assigned",
                      len(groups) == len(ringed + acyclic + two_ring),
                      f"{len(groups)} assignments")
    failures += check("molecules sharing a ring scaffold share a group",
                      len({groups[s] for s in ringed}) == 1,
                      f"benzene group ids {sorted({groups[s] for s in ringed})}")
    failures += check("a different ring scaffold is a different group",
                      len({groups[s] for s in two_ring}) == 1
                      and groups[two_ring[0]] != groups[ringed[0]])
    # rule 2, the load-bearing one
    acyclic_ids = [groups[s] for s in acyclic]
    failures += check("RULE 2: acyclic molecules are SINGLETONS, not one big group",
                      len(set(acyclic_ids)) == len(acyclic),
                      f"{len(set(acyclic_ids))} distinct ids for {len(acyclic)} acyclic molecules")
    failures += check("an acyclic molecule never joins a ring group",
                      not (set(acyclic_ids) & {groups[s] for s in ringed + two_ring}))
    failures += check("duplicate SMILES do not create duplicate entries",
                      build_scaffold_groups(ringed + ringed) == build_scaffold_groups(ringed))
    failures += check("an unparseable SMILES gets its own group, and does not crash",
                      'not a molecule' in build_scaffold_groups(ringed + ['not a molecule']))

    print("\nrecord_noise_manifest")
    with tempfile.TemporaryDirectory() as tmp:
        manifest = os.path.join(tmp, 'noise_manifest_7.json')
        payload = {
            'noise_type': 'grouped_shifted', 'noise_shape': 'gaussian',
            'noise_targeting': 'grouped_shifted', 'noise_level': 0.5,
            'unit_dose': 1.0, 'solved_scale': 0.679,
            'target_dose_in_label_units': 0.679,
            'delivered_dose_in_label_units': 0.684,
            'delivered_dose_as_fraction_of_label_spread': 0.503,
            'mean_epsilon': 0.01, 'affected_molecule_fraction': 1.0,
            'effective_n': 51.1, 'standardisation_mean': 7.28,
            'standardisation_sd': 1.36, 'clean_label_mean': 7.28,
            'clean_label_sd': 1.36, 'seed': 42, 'n_train': 4000,
            'parameters': {'group_variance_share': 0.62, 'n_scaffold_groups': 1703},
        }
        json.dump(payload, open(manifest, 'w'))

        args = types.SimpleNamespace(
            dose_units='spread', dataset='QM9', target='homo_lumo_gap',
            filepath=os.path.join(tmp, 'results', 'run.csv'))

        row = record_noise_manifest(args, manifest, iteration=0, file_no=7, level=0.5)
        out = os.path.join(tmp, 'results', 'run_noise_manifest.csv')
        failures += check("a manifest CSV is written beside the results file",
                          os.path.exists(out))

        rows = list(csv.DictReader(open(out)))
        failures += check("one row per run", len(rows) == 1)

        # RERUN_PLAN.md 5.2: these are the columns whose absence hid the confound
        required = ['noise_type', 'unit_dose', 'delivered_dose_in_label_units',
                    'delivered_dose_as_fraction_of_label_spread',
                    'affected_molecule_fraction', 'standardisation_mean',
                    'standardisation_sd', 'seed']
        missing = [c for c in required if c not in rows[0]]
        failures += check("every §5.2 run-level column reaches the CSV",
                          not missing, f"missing {missing}" if missing else '')
        failures += check("the condition parameters are flattened, not dropped",
                          rows[0].get('param_group_variance_share') == '0.62')
        failures += check("the row is joinable to a results row",
                          rows[0]['iteration'] == '0' and rows[0]['file_no'] == '7'
                          and rows[0]['noise_level'] == '0.5')

        # a second run appends rather than overwriting, and does not re-header
        record_noise_manifest(args, manifest, iteration=1, file_no=8, level=1.0)
        rows = list(csv.DictReader(open(out)))
        failures += check("a second run appends", len(rows) == 2,
                          f"{len(rows)} rows")

        # a missing manifest must warn and return None, never invent a row
        got = record_noise_manifest(args, os.path.join(tmp, 'nope.json'),
                                    iteration=2, file_no=9, level=0.2)
        failures += check("a missing manifest returns None rather than inventing a row",
                          got is None and len(list(csv.DictReader(open(out)))) == 2)

    print("\nnoise_seeds_for_level")
    # The rule that decides who gets damaged. It has to hold still across the level
    # grid: both seeds used to come off one level-dependent value, so the affected
    # molecules were redrawn at every point of a condition's own degradation curve
    # (RERUN_PLAN.md 2.26a). Nothing in the injector can see that -- one invocation
    # sees one level -- so the check belongs here, where the rule lives.
    grid = [0.0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
    shape_seeds = [noise_seeds_for_level(42, v)[0] for v in grid]
    selection_seeds = [noise_seeds_for_level(42, v)[1] for v in grid]

    failures += check("the SELECTION seed is the same at every noise level",
                      len(set(selection_seeds)) == 1,
                      f"{len(set(selection_seeds))} distinct values across "
                      f"{len(grid)} levels: {sorted(set(selection_seeds))}")
    failures += check("the SHAPE seed is different at every noise level",
                      len(set(shape_seeds)) == len(grid),
                      f"{len(set(shape_seeds))} distinct values across {len(grid)} levels")
    # ...and it still has to be a draw, not a fixture: a different replicate picks a
    # different set of molecules, or the affected set is the same one all study long.
    per_replicate = {noise_seeds_for_level(rep_seed, 0.5)[1]
                     for rep_seed in (42, 43, 1234, 99999)}
    failures += check("the SELECTION seed still varies between replicates",
                      len(per_replicate) == 4,
                      f"{len(per_replicate)} distinct values across 4 replicates")
    # Keyed on the level's VALUE, not its position, so a gap-filling run that sweeps
    # a subset of the grid reproduces the full run's rows rather than new noise.
    failures += check("the SHAPE seed is keyed on the level's value, not its position",
                      noise_seeds_for_level(42, 0.5)[0]
                      == [noise_seeds_for_level(42, v)[0] for v in [0.5, 1.0]][0]
                      == [noise_seeds_for_level(42, v)[0] for v in [0.0, 0.2, 0.5]][2])
    failures += check("both seeds are passed to the injector on the command line",
                      "'--selection-seed', str(selection_seed)" in open(SOURCE).read(),
                      "the level-free seed is computed but never handed over"
                      if "'--selection-seed'" not in open(SOURCE).read() else '')

    print("\nthe retired noise flags are refused")
    source = open(SOURCE).read()
    for flag in ('--sigma', '--distribution', '--noise-strategy', '--strategy-params'):
        failures += check(f"{flag} is named in the refusal table", f'"{flag}"' in source)
    failures += check("the refusal calls parser.error, so it exits non-zero",
                      'parser.error(f"{flag} has been removed' in source)
    failures += check("the refusal matches --flag=value as well as --flag",
                      'a.startswith(flag + "=")' in source)

    print()
    if failures:
        print(f"FAILED — {failures} check(s)")
        return 1
    print("the Python side of the noise wiring is sound")
    return 0


if __name__ == '__main__':
    sys.exit(main())
