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


class _ParserStopped(Exception):
    """argparse's error() exits; the stand-in raises so the test can look at it."""


class _StandInParser:
    """Just enough parser for the refusal loop: it records and stops."""

    def __init__(self):
        self.message = None

    def error(self, message):
        self.message = message
        raise _ParserStopped(message)


def load_retired_flag_guard():
    """Lift the retired-flag refusal out of main() so it can be RUN.

    This was three substring searches over the text of process_and_train.py
    ("--sigma" appears, parser.error appears, startswith(flag + "=") appears).
    A substring search is not a check. Measured 2026-08-29: with the real
    refusal loop replaced by a dead string literal holding those same fragments,
    every check in this section still printed ok and the file exited 0. The
    guard the test is named after can be deleted and the test stays green.

    process_and_train.py cannot be imported here -- torch_geometric, deepchem and
    optuna are the reason this file exists at all -- so the two statements are
    pulled out with ast and executed against a stand-in parser. That runs the
    real loop over the real dict, and a deleted loop is a missing node, which
    fails.
    """
    tree = ast.parse(open(SOURCE).read())
    # Walk the whole module, not one named function: the refusal sits in
    # parse_arguments(), and an earlier version of this search looked only inside
    # main() and reported the guard as DELETED while it was there and working.
    assign = loop = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == 'retired'
                and isinstance(node.value, ast.Dict)):
            assign = node
        if (isinstance(node, ast.For) and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Attribute)
                and node.iter.func.attr == 'items'
                and getattr(node.iter.func.value, 'id', None) == 'retired'):
            loop = node
    return assign, loop


def run_retired_flag_guard(assign, loop, argv):
    """Execute the real refusal against this command line. Returns the message
    it refused with, or None if it let the command through."""
    parser = _StandInParser()
    ns = {'parser': parser,
          'sys': types.SimpleNamespace(argv=['process_and_train.py'] + argv)}
    module = ast.Module(body=[assign, loop], type_ignores=[])
    try:
        exec(compile(ast.fix_missing_locations(module), SOURCE, 'exec'), ns)
    except _ParserStopped:
        return parser.message
    return None


def selection_seed_reaches_the_injector():
    """Is the level-free seed actually on the command line the injector gets?

    Structural, not textual: find the list assigned to `rust_cmd` and check that
    the literal '--selection-seed' is followed by str(selection_seed) and
    '--seed' by str(level_seed). A substring search passes on a commented-out
    line, and would not notice the two seeds being swapped -- which is the whole
    defect this rule exists to prevent.
    """
    tree = ast.parse(open(SOURCE).read())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], 'id', None) == 'rust_cmd'
                and isinstance(node.value, ast.List)):
            continue
        pairs = {}
        items = node.value.elts
        for i, e in enumerate(items[:-1]):
            if not (isinstance(e, ast.Constant) and isinstance(e.value, str)
                    and e.value.startswith('--')):
                continue
            nxt = items[i + 1]
            if (isinstance(nxt, ast.Call)
                    and getattr(nxt.func, 'id', None) == 'str'
                    and nxt.args and isinstance(nxt.args[0], ast.Name)):
                pairs[e.value] = nxt.args[0].id
        return pairs
    return {}


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
    pairs = selection_seed_reaches_the_injector()
    failures += check("the level-free seed is handed to the injector as "
                      "--selection-seed",
                      pairs.get('--selection-seed') == 'selection_seed',
                      f"--selection-seed carries {pairs.get('--selection-seed')!r}")
    failures += check("the level-dependent seed is handed over as --seed",
                      pairs.get('--seed') == 'level_seed',
                      f"--seed carries {pairs.get('--seed')!r}")

    print("\nthe retired noise flags are refused")
    # RUN the refusal rather than search for its text. See load_retired_flag_guard.
    assign, loop = load_retired_flag_guard()
    failures += check("the refusal table is still there",
                      assign is not None,
                      "" if assign is not None
                      else "no `retired = {...}` anywhere in the file")
    failures += check("the refusal loop is still there, and still runs",
                      loop is not None,
                      "" if loop is not None
                      else "no `for flag, replacement in retired.items()` "
                           "anywhere in the file")
    if assign is None or loop is None:
        print("      the refusal has been deleted or moved; the checks below "
              "cannot run")
        failures += 1
    else:
        table = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[assign],
                                                          type_ignores=[])),
                     SOURCE, 'exec'), {}, table)
        named = set(table['retired'])
        expected = {'--sigma', '--distribution', '--noise-strategy',
                    '--strategy-params'}
        failures += check("all four retired flags are in the table",
                          named == expected,
                          "" if named == expected else
                          f"missing {sorted(expected - named)}, "
                          f"extra {sorted(named - expected)}")
        for flag in sorted(expected & named):
            msg = run_retired_flag_guard(assign, loop, [flag, '0.5'])
            failures += check(f"{flag} is REFUSED when a job passes it",
                              msg is not None and 'has been removed' in msg,
                              "the run was allowed to continue" if msg is None
                              else '')
            eq = run_retired_flag_guard(assign, loop, [f'{flag}=0.5'])
            failures += check(f"{flag}=value is refused too",
                              eq is not None and 'has been removed' in eq,
                              "the =value form slipped through" if eq is None
                              else '')
            failures += check(f"{flag}'s refusal names what to use instead",
                              bool(msg) and 'Use ' in msg and msg.rstrip('.')
                              .split('Use ')[-1].strip() not in ('', flag),
                              '' if msg else 'nothing was refused')
        allowed = run_retired_flag_guard(assign, loop,
                                         ['--noise-level', '0.5', '--noise-shape',
                                          'gaussian'])
        failures += check("a current flag is NOT refused",
                          allowed is None,
                          "" if allowed is None
                          else f"refused a live flag: {allowed}")

    print()
    if failures:
        print(f"FAILED — {failures} check(s)")
        return 1
    print("the Python side of the noise wiring is sound")
    return 0


if __name__ == '__main__':
    sys.exit(main())
