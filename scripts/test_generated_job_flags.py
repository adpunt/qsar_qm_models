#!/usr/bin/env python
"""Every model flag the QM9 job generator emits must be one the program accepts.

`--bayesian-transformation last` sat in this table for the whole study.
models.py branches on `last_layer`; `last` matched nothing, so no Bayesian
layer was applied and a plain deterministic network was put through 100
forward passes in eval mode -- identical every time, uncertainty exactly zero
-- and written to a file the figure script reads as a last-layer BNN
(RERUN_PLAN.md §2.10c). argparse had no `choices=` and let the value through.

This walks the generator's own model tables and runs each flag string through
the program's real argument parser. A value the program refuses, or a flag it
does not have, fails here rather than at 3 a.m. on the cluster.

Run it directly:  python scripts/test_generated_job_flags.py
"""

import argparse
import contextlib
import importlib.util
import io
import os
import shlex
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

GENERATOR = os.path.join(HERE, "..", "slurm_scripts_qm9_rerun",
                         "generate_scripts.py")


def load_generator():
    spec = importlib.util.spec_from_file_location("qm9_generate_scripts",
                                                  GENERATOR)
    module = importlib.util.module_from_spec(spec)
    saved, sys.argv = sys.argv, ["generate_scripts.py"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = saved
    return module


def parser_of_process_and_train():
    import process_and_train

    captured = {}
    real = argparse.ArgumentParser.parse_args

    def grab(self, *a, **k):
        captured["parser"] = self
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = grab
    try:
        with contextlib.suppress(SystemExit):
            process_and_train.parse_arguments()
    finally:
        argparse.ArgumentParser.parse_args = real
    assert "parser" in captured, "could not reach the program's argument parser"
    return captured["parser"]


def every_model_flag_string_parses():
    module = load_generator()
    parser = parser_of_process_and_train()

    tables = [("MODELS", module.MODELS),
              ("EXCLUDED_MODELS", getattr(module, "EXCLUDED_MODELS", {}))]
    bad = []
    total = 0
    for table_name, table in tables:
        for job_name, entry in table.items():
            flags = entry[0]
            total += 1
            argv = shlex.split(flags) + ["-r", "ecfp4", "-d", "QM9"]
            err = io.StringIO()
            try:
                with contextlib.redirect_stderr(err), \
                        contextlib.redirect_stdout(io.StringIO()):
                    parser.parse_args(argv)
            except SystemExit:
                bad.append(f"{table_name}[{job_name}]: {flags}\n"
                           f"      {err.getvalue().strip().splitlines()[-1]}")
    assert not bad, (
        f"{len(bad)} of {total} job definitions pass flags the program "
        f"refuses:\n    " + "\n    ".join(bad))
    print(f"    {total} job definitions, every flag accepted")


def every_generated_script_parses():
    """Generate real scripts and check every flag NAME they pass.

    The committed job scripts passed `--sigma` and `--noise-strategy`, which
    process_and_train.py refuses by name, so every QM9 job died at argument
    parsing (RERUN_PLAN.md §2.10c). The scripts are not kept in version control
    -- they are regenerated from this generator -- so the generator is what has
    to be checked.

    The values are shell variables (`$LEVELS`, `$NOISE_FLAGS`, `$rep`) and cannot
    be resolved without running the script, so this checks the flag names, which
    is where the fault was. `--noise-shape` and `--noise-targeting` arrive inside
    `$NOISE_FLAGS`, so the case statement that builds it is checked too.
    """
    import re
    import subprocess
    import tempfile

    parser = parser_of_process_and_train()
    known = set()
    for action in parser._actions:
        known.update(action.option_strings)

    with tempfile.TemporaryDirectory() as tmp:
        result = subprocess.run(
            [sys.executable, GENERATOR, "--stage", "1", "--out-dir", tmp,
             "--sample-size", "500", "--replicates", "1"],
            capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-2000:]
        scripts = sorted(f for f in os.listdir(tmp) if f.endswith(".sh"))
        assert scripts, f"the generator wrote no scripts: {result.stdout[-800:]}"

        flag = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")
        bad, seen = [], set()
        for name in scripts:
            text = open(os.path.join(tmp, name)).read()
            # The python invocation, with its backslash continuations joined.
            joined = text.replace("\\\n", " ")
            for line in joined.splitlines():
                stripped = line.strip()
                if stripped.startswith("#") or "process_and_train.py" not in stripped:
                    continue
                after = stripped.split("process_and_train.py", 1)[1]
                for f in flag.findall(after):
                    seen.add(f)
                    if f not in known:
                        bad.append(f"{name}: {f}")
            # NOISE_FLAGS is built in a case statement, one line per condition.
            for line in text.splitlines():
                if "NOISE_FLAGS=" not in line:
                    continue
                for f in flag.findall(line.split("NOISE_FLAGS=", 1)[1]):
                    seen.add(f)
                    if f not in known:
                        bad.append(f"{name}: {f} (in NOISE_FLAGS)")

        assert seen, "no flags were found in the generated scripts"
        assert not bad, (
            f"{len(bad)} flag(s) the program does not have:\n    "
            + "\n    ".join(sorted(set(bad))))
        for retired in ("--sigma", "--noise-strategy"):
            assert retired not in seen, f"{retired} is still emitted"
        print(f"    {len(scripts)} scripts, {len(seen)} distinct flags, all known")


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
    print("QM9 job generator flags (RERUN_PLAN.md §2.10c)")
    results = [
        check("every model flag string the generator emits parses",
              every_model_flag_string_parses),
        check("every generated script parses", every_generated_script_parses),
    ]
    if not all(results):
        print("\nFAIL: the generator emits settings the program refuses")
        return 1
    print("\nOK: every generated model flag is one the program accepts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
