#!/usr/bin/env python
"""No top-level name in the pipeline may be defined twice (RERUN_PLAN.md §2.13).

`create_gnn_model` was defined FOUR times in models/models.py. Python binds the
last one, so three architectures sat in the file looking authoritative and never
ran -- and anyone reading the class definitions to describe the GNN in the paper
described a model that was never trained.

It was not harmless. `train_conformal_graph_model` built its base network as
`GIN(dim_h=...)`, the signature of a `class GCN` that a second definition later
in the same file shadows. The live classes take (num_node_features, hidden_dim,
...), so that call raises TypeError -- swallowed by the caller's blanket
`except Exception` and seen only as a missing result row.

This walks the module with `ast` and fails on any repeated top-level def or
class, in either pipeline.

Run it directly:  python scripts/test_no_shadowed_definitions.py
"""

import ast
import collections
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

FILES = [
    os.path.join(ROOT, 'models', 'models.py'),
    os.path.join(ROOT, 'models', 'model_defaults.py'),
    os.path.join(HERE, 'process_and_train.py'),
    os.path.join(HERE, 'utils.py'),
    os.path.join(HERE, 'generate_paper_figures_v2.py'),
    # The experimental pipeline, when it is checked out beside this one.
    os.path.join(ROOT, '..', 'KIRBy', 'tests',
                 'alternative_data_noise_robustness.py'),
]


def shadowed(path):
    tree = ast.parse(open(path).read())
    seen = collections.defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[node.name].append(node.lineno)
    return {name: lines for name, lines in seen.items() if len(lines) > 1}


def no_module_defines_a_name_twice():
    bad = []
    checked = 0
    for path in FILES:
        if not os.path.exists(path):
            print(f"    (skipped, not present: {os.path.relpath(path, ROOT)})")
            continue
        checked += 1
        for name, lines in sorted(shadowed(path).items()):
            bad.append(f"{os.path.relpath(path, ROOT)}: {name} at lines {lines} "
                       f"-- only the one at {lines[-1]} runs")
    assert not bad, ("a definition is shadowed by a later one with the same "
                     "name:\n    " + "\n    ".join(bad))
    print(f"    {checked} modules, no name defined twice")


def the_conformal_graph_wrapper_says_it_cannot_run():
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, 'models'))
    import inspect

    import models as M

    # The live classes do not take the keyword the conformal wrapper used.
    for name in ('GCN', 'GIN'):
        params = list(inspect.signature(getattr(M, name).__init__).parameters)
        assert 'dim_h' not in params, (
            f"{name} takes dim_h again; the wrapper's original call would now "
            f"build a model and this check no longer means anything")
    src = inspect.getsource(M.train_conformal_graph_model)
    assert 'NotImplementedError' in src, (
        "the conformal graph wrapper no longer refuses by name -- if it has "
        "been made to work, replace this check with one that runs it")
    print("    the live GCN/GIN take no dim_h, and the wrapper refuses by name")


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
    print("shadowed definitions (RERUN_PLAN.md §2.13)")
    results = [
        check("no module defines a top-level name twice",
              no_module_defines_a_name_twice),
        check("the conformal graph wrapper says it cannot run",
              the_conformal_graph_wrapper_says_it_cannot_run),
    ]
    if not all(results):
        print("\nFAIL: a definition in the pipeline is shadowed and never runs")
        return 1
    print("\nOK: every top-level definition in the pipeline is the one that runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
