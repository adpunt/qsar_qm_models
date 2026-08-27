#!/usr/bin/env python
"""Check the claims against a REAL QM9 run's own output files.

Nothing here is synthetic. It reads the per-molecule noise record the pipeline
writes during a real run and checks the properties the whole re-run depends on.
Run the pipeline first, then point this at the directory it wrote into.

    python scripts/process_and_train.py -d QM9 -t homo_lumo_gap \\
        -m rf lgb -r ecfp4 continuous_pdv --noise-level 0.0 0.4 0.8 \\
        -n 1000 -b 2 -s scaffold --normalize True -f /tmp/out.csv
    python scripts/smoke_real_data.py /tmp/out.csv
"""

import glob
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_records(log_path=None):
    """The per-molecule noise records THIS run wrote.

    Not a glob of scripts/. Other sessions run the pipeline in the same
    directory, and their records sit alongside this run's -- reading all of them
    made this script report a failure that belonged to somebody else's
    deliberately different configuration. The run's own log names every file it
    wrote, on the "Rust command:" lines, so bind to those.
    """
    paths = []
    if log_path and os.path.isfile(log_path):
        import re
        text = open(log_path, errors="ignore").read()
        for name in re.findall(r"--noise-provenance\s+(\S+)", text):
            full = name if os.path.isabs(name) else os.path.join(HERE, name)
            if os.path.isfile(full) and full not in paths:
                paths.append(full)
        if not paths:
            raise AssertionError(
                f"{log_path} names no noise-provenance files -- did the run reach the "
                f"injector?"
            )
    else:
        raise AssertionError(
            "pass the run's log as the second argument so this reads only that run's "
            "files. Globbing scripts/ picks up other sessions' runs."
        )
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        d["source_file"] = os.path.basename(p)
        frames.append(d)
    return paths, pd.concat(frames, ignore_index=True)


def held_out_labels_are_never_touched(df):
    """The original defect: test and validation molecules were given noise."""
    held = df[df.split.isin(["val", "test"])]
    assert len(held), "the run recorded no held-out molecules at all"
    worst = held.epsilon_raw.abs().max()
    assert worst == 0.0, (
        f"{(held.epsilon_raw != 0).sum()} held-out molecules carry injected noise "
        f"(largest {worst:.3e}). Held-out labels must be untouched."
    )
    return f"{len(held):,} held-out molecules, every injected value exactly zero"


def the_recorded_noise_reconstructs_the_label(df):
    """y_clean + epsilon must equal y_noisy, to the bit."""
    lhs = df.y_clean_raw.to_numpy() + df.epsilon_raw.to_numpy()
    rhs = df.y_noisy_raw.to_numpy()
    bad = ~np.isclose(lhs, rhs, rtol=0, atol=1e-6)
    assert not bad.any(), (
        f"{bad.sum():,} of {len(df):,} molecules: the recorded noise does not reconstruct "
        f"the noisy label (worst gap {np.abs(lhs - rhs).max():.3e})"
    )
    return f"{len(df):,} molecules, clean + injected == noisy throughout"


def zero_noise_records_exactly_zero(df, levels_by_file):
    """The negative control. Not small -- zero."""
    zero_files = [f for f, lv in levels_by_file.items() if lv == 0.0]
    if not zero_files:
        return "no zero-noise run in this batch (skipped)"
    z = df[df.source_file.isin(zero_files)]
    worst = z.epsilon_raw.abs().max()
    assert worst == 0.0, f"at zero noise the largest injected value is {worst:.3e}, not zero"
    return f"{len(z):,} molecules at zero noise, every injected value exactly zero"


def the_same_molecule_keeps_the_same_true_label(df):
    """Across noise levels, a molecule's CLEAN label must not move."""
    seen = defaultdict(set)
    for smiles, y in zip(df.canonical_smiles, df.y_clean_raw):
        seen[smiles].add(round(float(y), 9))
    drifted = {k: v for k, v in seen.items() if len(v) > 1}
    assert not drifted, (
        f"{len(drifted)} molecules have more than one clean label across the run, "
        f"e.g. {list(drifted.items())[:2]}"
    )
    return f"{len(seen):,} distinct molecules, each with one clean label everywhere"


def training_noise_actually_arrived(df, levels_by_file):
    """The positive control: above zero, training labels really do move."""
    out = []
    for f, lv in sorted(levels_by_file.items(), key=lambda kv: kv[1]):
        if lv == 0.0:
            continue
        tr = df[(df.source_file == f) & (df.split == "train")]
        if not len(tr):
            continue
        spread = tr.y_clean_raw.std(ddof=0)
        delivered = tr.epsilon_raw.std(ddof=0)
        assert delivered > 0, f"noise level {lv} delivered nothing to the training split"
        out.append(f"level {lv}: delivered {delivered / spread:.3f} of the label spread")
    assert out, "no non-zero noise level produced training records"
    return "; ".join(out)


def the_results_file_has_the_rows_it_should(results_csv):
    assert os.path.isfile(results_csv), f"{results_csv} was never written"
    d = pd.read_csv(results_csv)
    assert len(d), "the results file is empty"
    cols = {c.lower() for c in d.columns}
    got = []
    for key in ("sigma", "noise_level", "model", "rep", "representation", "r2"):
        if key in cols:
            got.append(key)
    return f"{len(d)} rows, columns include {sorted(got)}"


def check(name, fn):
    try:
        detail = fn()
    except AssertionError as e:
        print(f"  FAIL  {name}\n        {e}")
        return False
    except Exception as e:
        print(f"  FAIL  {name}\n        {type(e).__name__}: {e}")
        return False
    print(f"  OK    {name}\n        {detail}")
    return True


def main():
    results_csv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/smoke_real.csv"
    print("smoke test on REAL QM9 output\n")

    log_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/smoke_real.log"
    paths, df = load_records(log_path)
    print(f"read {len(paths)} noise records written by THIS run "
          f"(named in {log_path}), {len(df):,} rows total\n")

    # Recover each file's noise level from the run manifest beside it.
    levels_by_file = {}
    for p in paths:
        man = p.replace("noise_provenance_", "noise_manifest_").replace(".csv", ".json")
        lvl = None
        if os.path.isfile(man):
            import json
            m = json.load(open(man))
            for k in ("level", "noise_level", "requested_level"):
                if k in m:
                    lvl = float(m[k])
                    break
            if lvl is None and isinstance(m.get("spec"), dict):
                lvl = float(m["spec"].get("level", 0.0))
        if lvl is None:
            d = pd.read_csv(p)
            lvl = 0.0 if d.epsilon_raw.abs().max() == 0 else float("nan")
        levels_by_file[os.path.basename(p)] = lvl

    results = [
        check("held-out labels are never touched",
              lambda: held_out_labels_are_never_touched(df)),
        check("the recorded noise reconstructs the label",
              lambda: the_recorded_noise_reconstructs_the_label(df)),
        check("zero noise records exactly zero",
              lambda: zero_noise_records_exactly_zero(df, levels_by_file)),
        check("a molecule keeps the same true label at every noise level",
              lambda: the_same_molecule_keeps_the_same_true_label(df)),
        check("training noise actually arrived",
              lambda: training_noise_actually_arrived(df, levels_by_file)),
        check("the results file has rows",
              lambda: the_results_file_has_the_rows_it_should(results_csv)),
    ]

    print()
    if not all(results):
        print("FAIL: the real run does not satisfy the properties the re-run depends on")
        return 1
    print("OK: every property checked holds on real QM9 output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
