#!/usr/bin/env python3
"""A replicate is QM9's. The other three datasets have folds. The two never merge.

Settled 2026-08-27, restated here because it came back three times: QM9's ten
replicates are ten fresh draws of 10,000 molecules with their own scaffold split,
and hERG/LogD/Caco-2 have five GroupKFold folds over one fixed dataset. Both can
carry an error bar. They are different statistics and must be labelled
differently. RERUN_PLAN.md 3.2b.

The way the two would get merged is by the column names drifting together --
either pipeline calling its axis by the other's name, and then one reader
concatenating them into a single spread. So that is what this checks:

  1. QM9's results rows carry `iteration` and neither `fold` nor `replicate`.
  2. The laboratory rows carry `fold` and neither `iteration` nor `replicate`.
  3. QM9 really does resample: the replicate loop reseeds, and split_qm9 draws a
     fresh permutation, so two replicates hold different molecules.
  4. The laboratory side really does not: one GroupKFold over one dataset, with
     every molecule tested exactly once.

Run: python3 scripts/test_replicate_is_not_a_fold.py
"""

import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QSAR = os.path.dirname(HERE)
KIRBY = os.path.join(os.path.dirname(QSAR), "KIRBy")

QM9_UTILS = os.path.join(QSAR, "scripts", "utils.py")
QM9_RUNNER = os.path.join(QSAR, "scripts", "process_and_train.py")
LAB_RUNNER = os.path.join(KIRBY, "tests", "alternative_data_noise_robustness.py")

failures = []
skipped = []


def check(name, ok, detail=""):
    print(f"  {name}\n    {'ok' if ok else 'FAILED: ' + detail}")
    if not ok:
        failures.append(name)


def literal_list(path, name):
    """The value of a module-level list of strings, read without importing."""
    tree = ast.parse(open(path).read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return [e.value for e in node.value.elts
                            if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return None


# ------------------------------------------------------------------ 1
print("QM9 rows carry the replicate axis and no fold axis:")
cols = literal_list(QM9_UTILS, "RESULT_COLUMNS")
check("RESULT_COLUMNS was read", cols is not None, "not found in scripts/utils.py")
if cols:
    check("`iteration` is the QM9 axis", "iteration" in cols, f"columns: {cols}")
    stray = [c for c in cols if c in ("fold", "replicate")]
    check("no `fold` or `replicate` column on a QM9 row", not stray,
          f"a QM9 row would carry {stray}, which is the other pipeline's axis")

unc = literal_list(QM9_UTILS, "UNCERTAINTY_COLUMNS")
if unc:
    stray = [c for c in unc if c in ("fold", "replicate")]
    check("no `fold` or `replicate` column on a QM9 uncertainty row", not stray,
          f"carries {stray}")


# ------------------------------------------------------------------ 2
print("\nlaboratory rows carry the fold axis and no replicate axis:")
if not os.path.exists(LAB_RUNNER):
    skipped.append("the laboratory runner is not on this machine")
    print(f"  SKIPPED: {LAB_RUNNER} not found")
else:
    lab = open(LAB_RUNNER).read()
    check("`fold` is written onto the rows",
          re.search(r"['\"]fold['\"]\s*[:=]", lab) is not None,
          "no row is stamped with a fold")
    # `iteration`/`replicate` as a COLUMN, not as an ordinary variable.
    stray = sorted(set(re.findall(r"['\"](iteration|replicate)['\"]\s*:", lab)))
    check("no `iteration` or `replicate` column on a laboratory row", not stray,
          f"a laboratory row would carry {stray}, which is QM9's axis")


# ------------------------------------------------------------------ 3
print("\nQM9 replicates really are fresh draws of molecules:")
runner = open(QM9_RUNNER).read()
check("the replicate loop reseeds",
      re.search(r"for iteration in range\(", runner) is not None
      and re.search(r"torch\.manual_seed\(iteration_seed\)", runner) is not None,
      "no reseeded per-replicate loop found")
check("split_qm9 draws a fresh permutation",
      re.search(r"indices = torch\.randperm\(len\(qm9\)\)", runner) is not None,
      "split_qm9 no longer resamples, so a replicate is not a resample any more")
check("and then takes a sample off the front",
      re.search(r"qm9\[:\s*args\.sample_size\s*\]", runner) is not None,
      "the sample-size truncation is gone, so all replicates hold the same molecules")


# ------------------------------------------------------------------ 4
print("\nthe laboratory side really is one partition, not repeats:")
if os.path.exists(LAB_RUNNER):
    lab = open(LAB_RUNNER).read()
    check("one GroupKFold over one dataset",
          re.search(r"GroupKFold\(n_splits=N_FOLDS\)", lab) is not None,
          "the fold loop is not a single GroupKFold any more")
    check("and it is iterated once, not repeated",
          len(re.findall(r"for fold_idx, \(train_idx, test_idx\) in enumerate\(", lab)) == 1,
          "more than one fold loop -- check whether repeats were added")

print()
for s in skipped:
    print(f"UNCHECKED: {s}")
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("a replicate is QM9's, a fold is the other three datasets', and the two axes "
      "carry different names.")
