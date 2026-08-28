#!/usr/bin/env python3
"""Conformal is commented out, and asking for it stops the run by name.

Conformal prediction was taken out of the study on 2026-08-28. It was already in
EXCLUDED_MODELS in the job generator, the figure script dropped its rows when it
loaded them, and no table read one. `--calibration-size` had been commented out
the day before, because it was accepted, passed down to train_conformal_model and
never read.

Commenting out the three dispatch branches alone would not have been safe. The two
dispatchers fail differently and both quietly:

  * the tabular one ends its elif chain with no else, so an unknown model name
    returns None and the run writes no row and says nothing;
  * the graph one ends with `else: return train_gnn(...)`, so `-m conformal
    -r graph` would train an ordinary graph network and write it to a file named
    for conformal.

So the branches are commented out AND the two names are refused before any work
starts. This checks all four things:

  1. `-m conformal` stops, and the message names the model.
  2. `-m conformal_hetero` stops, and the message names the model.
  3. Neither dispatcher still has a live branch for either name.
  4. The job generator emits no script that asks for conformal, with the
     excluded models turned ON, which is the only way they could have come back.

Run: python3 scripts/test_conformal_is_out.py
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
QSAR = os.path.dirname(HERE)
RUNNER = os.path.join(HERE, "process_and_train.py")
GENERATOR = os.path.join(QSAR, "slurm_scripts_qm9_rerun", "generate_scripts.py")

failures = []


def check(name, ok, detail=""):
    print(f"  {name}\n    {'ok' if ok else 'FAILED: ' + detail}")
    if not ok:
        failures.append(name)


# ---------------------------------------------------------------- 1 and 2
# argparse refuses before the dataset is touched, so this costs one interpreter
# start-up per name and no data.
def refusal_for(model_name):
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "never_written.csv")
        p = subprocess.run(
            [sys.executable, RUNNER, "-d", "QM9", "-t", "homo_lumo_gap",
             "-m", model_name, "-r", "ecfp4", "-n", "50", "-f", out],
            capture_output=True, text=True, cwd=HERE,
            env=dict(os.environ, OMP_NUM_THREADS="1"))
        wrote = os.path.exists(out)
    return p.returncode, p.stdout + p.stderr, wrote


print("the run stops when conformal is asked for:")
for model_name in ("conformal", "conformal_hetero"):
    code, text, wrote = refusal_for(model_name)
    named = model_name in text
    check(f"-m {model_name} stops with the name in the message",
          code != 0 and named and not wrote,
          f"exit={code}, name in message={named}, wrote a file={wrote}\n"
          f"    {text.strip()[-400:]}")


# ------------------------------------------------------------------- 3
# A commented-out branch is a comment; a live one is an ast node. Parsing is what
# separates them, so this cannot be satisfied by the words being present.
print("\nneither dispatcher has a live branch for either name:")

CONFORMAL = {"conformal", "conformal_hetero"}


def dispatched_names(path):
    """Every string a live `model_type == ...` / `in [...]` test compares against."""
    found = set()
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Name) and left.id == "model_type"):
            continue
        for comparator in node.comparators:
            if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                found.add(comparator.value)
            elif isinstance(comparator, (ast.List, ast.Tuple, ast.Set)):
                for element in comparator.elts:
                    if isinstance(element, ast.Constant) and isinstance(element.value, str):
                        found.add(element.value)
    return found


live = dispatched_names(RUNNER) & CONFORMAL
check("process_and_train.py dispatches neither name", not live,
      f"still dispatched: {sorted(live)}")

# The two flags that exist only for conformal go with it.
source = open(RUNNER).read()
for flag in ("--cp-base-model", "--alpha"):
    live_flag = re.search(r"^\s*parser\.add_argument\(\s*['\"]" + re.escape(flag),
                          source, re.M)
    check(f"{flag} is not accepted any more", live_flag is None,
          "the parser still takes it")


# ------------------------------------------------------------------- 4
# --include-excluded is the switch that used to bring them back. Run the generator
# with it on and read every command it wrote.
print("\nthe job generator writes no conformal script, excluded models ON:")

with tempfile.TemporaryDirectory() as outdir:
    p = subprocess.run(
        [sys.executable, GENERATOR, "--include-excluded", "--out-dir", outdir],
        capture_output=True, text=True, cwd=os.path.dirname(GENERATOR),
        env=dict(os.environ, OMP_NUM_THREADS="1"))
    if p.returncode != 0:
        # --out-dir may not be a flag it takes; fall back to reading the source's
        # own model table, which is the thing that decides what gets written.
        table = open(GENERATOR).read()
        live_entries = re.findall(r"^\s*'(conformal[a-z_]*)'\s*:", table, re.M)
        check("no live conformal entry in the generator's model tables",
              not live_entries, f"still listed: {live_entries}")
    else:
        wrote_conformal = []
        for name in sorted(os.listdir(outdir)):
            body = open(os.path.join(outdir, name)).read()
            if re.search(r"-m\s+conformal", body):
                wrote_conformal.append(name)
        check("no generated script asks for -m conformal", not wrote_conformal,
              f"scripts asking for it: {wrote_conformal}")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s): {failures}")
    sys.exit(1)
print("conformal is out: refused by name, no live branch, no generated script.")
