#!/usr/bin/env python
"""Verification gate 10 (RERUN_PLAN.md §2.8a, §8).

Two tasks running at once must not corrupt each other.

The defect this guards: `process_and_train.py` wrote its configuration to a
fixed `config.json` and the Rust binary read a fixed `config.json`. Every job
script does `cd scripts` first, and the array jobs are submitted at four to six
concurrent tasks, so every task on a node shared one file. The file carries
`file_no` — the number the binary uses to choose which memory-mapped training
files to open **and rewrite**. One task reading another's configuration meant
one task silently overwriting another's training data, with no error from
either.

Two halves, because they fail differently:

  * **fast** — two binaries launched concurrently in one directory, each handed
    its own configuration naming its own files and its own representation. Each
    must come out with its own data intact. This is the defect at the level it
    actually occurred, and it is deterministic.
  * **--end-to-end** — two real `process_and_train.py` tasks, as the job scripts
    run them. Slow (minutes, and it needs the QM9 cache), so it is opt-in.

A static check runs in both: no source file may write a bare `config.json`.

    python scripts/test_config_isolation.py
    python scripts/test_config_isolation.py --end-to-end
"""

import argparse
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(ROOT, "rust", "target", "release", "rust_processor")

FP_BYTES = 256
PDV_BYTES = 800


def write_split(path, rows, rep):
    """Rows as the Python side writes them, for one representation.

    `ecfp4` is 256 bytes at the END of the record. It used to be computed by the
    Rust side and inserted into the OUTPUT only, so the input record carried
    nothing for it; since 2026-08-27 it is Morgan radius 2, computed in Python
    and carried through, because the Rust binding has no route to radius 2
    (RERUN_PLAN.md 2.13b). A fixture without that block leaves the reader short
    and the run refuses the split.
    """
    out = b""
    for smiles, y, payload in rows:
        b = smiles.encode()
        out += struct.pack("I", len(b)) + b
        out += struct.pack("I", len(b)) + b
        out += struct.pack("f", y)
        if rep == "pdv":
            out += payload  # written BEFORE the label by the Rust writer
        elif rep == "ecfp4":
            # Not all zeros: an all-zero block is a featurisation failure and
            # stops the run by design.
            out += bytes(((payload[0] if payload else 0x11) + i) % 256 | 1
                         for i in range(FP_BYTES))
    with open(path, "wb") as f:
        f.write(out)


def build_task(tmp, file_no, rep, marker):
    """One task's inputs: a distinct file_no, representation and payload byte."""
    n_train, n_held = 12, 3
    payload_len = PDV_BYTES if rep == "pdv" else 0

    # Five characters minimum: read_smiles_data rejects anything shorter.
    def rows(n, base):
        return [
            (
                f"CCCC{'C' * (i % 5)}O",
                base + i * 0.31,
                bytes([marker]) * payload_len,
            )
            for i in range(n)
        ]

    train = rows(n_train, 3.0)
    for split, n, data in (("train", n_train, train),
                           ("val", n_held, rows(n_held, 9.0)),
                           ("test", n_held, rows(n_held, 9.0))):
        write_split(os.path.join(tmp, f"{split}_{file_no}.mmap"), data, rep)

    config = {
        "sample_size": n_train + 2 * n_held,
        "noise": True,
        "train_count": n_train,
        "test_count": n_held,
        "val_count": n_held,
        "max_vocab": 30,
        "file_no": file_no,
        "molecular_representations": [rep],
        "k_domains": 1,
        "logging": False,
        "regression": True,
        "normalize": True,
        "uncertainty": False,
    }
    config_path = os.path.join(tmp, f"config_{file_no}.json")
    with open(config_path, "w") as f:
        json.dump(config, f)
    return config_path, train


def record_len(smiles, rep):
    n = len(smiles.encode())
    extra = PDV_BYTES if rep == "pdv" else FP_BYTES
    return 4 + n + 4 + n + 4 + 4 + extra


def concurrent_tasks_do_not_corrupt_each_other():
    """The gate. Two binaries, one directory, at the same moment."""
    if not os.access(BIN, os.X_OK):
        raise AssertionError(
            f"{BIN} is missing. Run `cargo build --release` in rust/ first."
        )

    with tempfile.TemporaryDirectory(prefix="config_isolation_") as tmp:
        # Deliberately different in every way a mixed-up configuration would
        # show: different file_no, different representation, different payload.
        a_cfg, a_train = build_task(tmp, 101, "ecfp4", 0xAA)
        b_cfg, b_train = build_task(tmp, 202, "pdv", 0xBB)

        def launch(cfg):
            return subprocess.Popen(
                [BIN, "--seed", "42", "--config", cfg, "--model", "rf",
                 "--noise-level", "0.4", "--noise-shape", "gaussian",
                 "--noise-targeting", "uniform"],
                cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )

        # Launched before either is waited on, so their windows overlap.
        pa, pb = launch(a_cfg), launch(b_cfg)
        out_a, err_a = pa.communicate()
        out_b, err_b = pb.communicate()

        assert pa.returncode == 0, f"task A failed: {err_a}"
        assert pb.returncode == 0, f"task B failed: {err_b}"

        for file_no, rep, train in ((101, "ecfp4", a_train), (202, "pdv", b_train)):
            path = os.path.join(tmp, f"train_{file_no}.mmap")
            size = os.path.getsize(path)
            expected = sum(record_len(s, rep) for s, _, _ in train)
            assert size == expected, (
                f"train_{file_no}.mmap is {size} bytes, expected {expected} for "
                f"representation '{rep}'. The task was rewritten under another "
                f"task's configuration."
            )

        # Each task's own provenance must describe its own molecules, and the two
        # must not have written over one another's.
        for file_no, train in ((101, a_train), (202, b_train)):
            prov = os.path.join(tmp, f"noise_provenance_{file_no}.csv")
            assert os.path.isfile(prov), f"no provenance for task {file_no}"
            lines = [l for l in open(prov).read().splitlines()[1:] if l]
            n_train = sum(1 for l in lines if l.startswith("train,"))
            assert n_train == len(train), (
                f"task {file_no} recorded {n_train} training molecules, expected {len(train)}"
            )

        assert not os.path.exists(os.path.join(tmp, "config.json")), (
            "a shared config.json was created — the fixed name is back"
        )


def no_source_file_writes_a_bare_config_json():
    """Cheap, and it fails even without running anything."""
    offenders = []
    pattern = re.compile(r"""(open|File::open)\(\s*['"]config\.json['"]""")
    for base, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in {".git", "target", "research_archive", "__pycache__",
                                "_build_paper", "_build_addfiles", "results", "data"}]
        for name in files:
            if not name.endswith((".py", ".rs")):
                continue
            path = os.path.join(base, name)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{os.path.relpath(path, ROOT)}:{i}")

    assert not offenders, (
        "these write or read a fixed 'config.json', which concurrent array tasks "
        f"share: {offenders}"
    )


def two_real_tasks(tmp_results):
    """The end-to-end half: two `process_and_train.py` runs, as the jobs run them."""
    def launch(rep, tag):
        return subprocess.Popen(
            [sys.executable, "-u", "process_and_train.py",
             "-d", "QM9", "-t", "homo_lumo_gap",
             "-m", "rf", "-r", rep,
             "--noise-level", "0.0", "0.4",
             "-n", "400", "-b", "1", "-s", "scaffold", "--normalize", "True",
             "-f", os.path.join(tmp_results, f"{tag}.csv")],
            cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    pa = launch("ecfp4", "task_a")
    pb = launch("pdv", "task_b")
    out_a, err_a = pa.communicate()
    out_b, err_b = pb.communicate()

    # A TASK THAT NEVER STARTED IS NOT TWO TASKS INTERFERING.
    #
    # On ARC on 2026-08-28 both tasks died at `import polaris` -- a module-scope
    # import of a dataset loader QM9 never calls -- and this check reported
    # "concurrent tasks can interfere. Do NOT raise job concurrency." The real
    # answer was that no task could start at all, concurrently or alone, and the
    # verdict pointed at the wrong thing entirely. Separate the two, because they
    # have opposite fixes: one is an environment repair, the other is a throttle.
    for tag, proc, out, err in (("A", pa, out_a, err_a), ("B", pb, out_b, err_b)):
        if proc.returncode == 0:
            continue
        tail = f"{out[-3000:]}\n{err[-3000:]}"
        if "Traceback" in tail and "ImportError" in tail or "ModuleNotFoundError" in tail:
            raise AssertionError(
                f"task {tag} could not START -- this is NOT a concurrency result.\n"
                f"The interpreter cannot import what the pipeline needs, so a task "
                f"run entirely alone would fail the same way. Fix the environment "
                f"and re-run; do not read anything into the concurrency verdict.\n"
                f"{tail}")
        raise AssertionError(f"task {tag} failed:\n{tail}")

    for tag in ("task_a", "task_b"):
        path = os.path.join(tmp_results, f"{tag}.csv")
        assert os.path.isfile(path), f"{tag} produced no results file"
        assert len(open(path).read().splitlines()) > 1, f"{tag} produced no rows"

    assert not os.path.exists(os.path.join(HERE, "config.json")), (
        "a shared config.json was left in scripts/"
    )


def check(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL  {name}: {e}")
        return False
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}")
        return False
    print(f"  OK    {name}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--end-to-end", action="store_true",
                    help="Also run two real process_and_train.py tasks (slow).")
    args = ap.parse_args()

    print("configuration isolation (RERUN_PLAN.md gate 10)")
    results = [
        check("no source file writes a bare config.json",
              no_source_file_writes_a_bare_config_json),
        check("two concurrent tasks do not corrupt each other",
              concurrent_tasks_do_not_corrupt_each_other),
    ]

    if args.end_to_end:
        if platform.system() == "Darwin":
            # keopscore (gpytorch -> pykeops) hardcodes /tmp/compiler_version.txt
            # and /tmp/brew_prefix.txt, writing then deleting them during import.
            # Two simultaneous imports race and one dies with FileNotFoundError
            # before any of this project's code runs. Both functions are guarded
            # by platform.system() == "Darwin", so the cluster is unaffected --
            # but on a laptop this half can never pass, and it would fail for a
            # reason that has nothing to do with what it is testing.
            print("  SKIP  two real pipeline tasks: keopscore races on hardcoded")
            print("        /tmp files during import, macOS only. Run this half on")
            print("        the cluster, where that code does not execute.")
        else:
            with tempfile.TemporaryDirectory(prefix="config_isolation_e2e_") as tmp:
                results.append(check("two real pipeline tasks run side by side",
                                     lambda: two_real_tasks(tmp)))
    else:
        print("  (skipped the two-real-task run; pass --end-to-end to include it)")

    if not all(results):
        print("\nFAIL: concurrent tasks can interfere. Do NOT raise job concurrency.")
        return 1
    print("\nOK: concurrent tasks are isolated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
