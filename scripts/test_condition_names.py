"""One noise condition, one name, on both sides.

Results rows from the two pipelines are joined on the condition name. Until
2026-08-27 they could not be, for anything outside the eleven settled
conditions: the QM9 injector composed `outlier_p05_laplace` while the Python
injector returned `outlier/laplace` -- a string QM9 never writes, and one that
1%, 5% and 10% contamination all came out as, so the fraction was lost from the
results entirely (RERUN_PLAN.md 2.14, audit entry 39).

This runs BOTH injectors and compares what they call each condition against
`condition_names.json`, which is the only place the expected name is written
down. The QM9 side goes through the real binary's `--self-test --json` mode and
reads the name off the emitted row, so this checks the shipped executable, not a
restatement of what it is supposed to do.

It also covers the second half of that entry. The command line took
`grouped_wide` while every results row said `grouped_wider`, so typing the name
read off a row killed the run; the table spells the targeting the way the rows
do, so removing that fix fails this check.

    python3 scripts/test_condition_names.py

Not covered by `scripts/crosscheck_injectors.py`, which compares the two
injectors' STATISTICS and does so only for the settled conditions, reaching them
through `from_condition` -- where the name is handed over rather than composed,
so the composition this checks never runs there.

Needs the release binary: cd rust && cargo build --release.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
BIN = REPO / 'rust' / 'target' / 'release' / 'rust_processor'
TABLE = REPO / 'condition_names.json'

sys.path.insert(0, str(REPO / 'scripts'))

try:
    from noiseInject import NoiseInjectorRegression
except ImportError as exc:                                    # pragma: no cover
    sys.exit(f"cannot import noiseInject, which is the Python injector this "
             f"compares against: {exc}")


# A group per twenty molecules, so the grouped conditions have something to
# group by. Nothing here depends on the labels or the grouping -- a condition's
# NAME is a property of its settings -- but both injectors refuse to run a
# grouped condition without an assignment, which is correct of them.
N = 300
SMILES = [f"C{i}CCO" for i in range(N)]
GROUPS = np.array([i // 20 for i in range(N)])


def rust_name(row, workdir):
    """What the QM9 binary calls this condition, read off a row it emitted."""
    labels = np.random.default_rng(0).normal(3.0, 1.0, N)
    lab = workdir / 'labels.txt'
    lab.write_text(''.join(f"{s},{y:.6f}\n" for s, y in zip(SMILES, labels)))
    grp = workdir / 'groups.json'
    grp.write_text(json.dumps({s: int(g) for s, g in zip(SMILES, GROUPS)}))

    cmd = [str(BIN), '--self-test', str(lab), '--json', '--seeds', '1',
           '--scaffold-file', str(grp),
           '--noise-shape', row['shape'],
           # The targeting is spelled as the RESULTS ROWS spell it, which the
           # command line refused until 2026-08-27.
           '--noise-targeting', row['targeting'],
           '--noise-level', str(row.get('level', 0.5))]
    if 'nu' in row:
        cmd += ['--nu', str(row['nu'])]
    if 'p' in row:
        cmd += ['--outlier-p', str(row['p'])]
    if 'side' in row:
        cmd += ['--censor-side', row['side']]

    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise AssertionError(
            f"the QM9 injector refused {row['name']}:\n"
            f"  {' '.join(cmd)}\n{out.stdout}\n{out.stderr}")
    rows = json.loads(out.stdout.strip().splitlines()[-1])['rows']
    assert len(rows) == 1, f"expected one row for {row['name']}, got {len(rows)}"
    return rows[0]['condition']


def python_name(row):
    """What the Python injector calls the same condition."""
    kwargs = {'strategy': row['targeting'], 'distribution': row['shape'],
              'random_state': 0}
    if 'nu' in row:
        kwargs['nu'] = row['nu']
    if row['targeting'] == 'grouped_wider':
        kwargs.update(lam=3.0, group_fraction=0.2)
    if row['targeting'] == 'grouped_shifted':
        kwargs['rho'] = 0.62
    if row['targeting'] == 'outlier':
        kwargs.update(p=row['p'], lam=3.0)
    if row['targeting'] == 'censoring':
        kwargs['side'] = row['side']

    inj = NoiseInjectorRegression(**kwargs)
    if row['targeting'] == 'censoring':
        # The percentage clipped IS the level, so this name is only complete
        # once a level has been injected.
        y = np.random.default_rng(0).normal(3.0, 1.0, N)
        inj.inject_verbose(y, row['level'], groups=GROUPS)
    return inj.condition


def main():
    if '--rebuild' in sys.argv:
        # For the revert harness. It edits main.rs, and this reads the SHIPPED
        # binary -- so without a rebuild an edited source would pass here and
        # the check would prove nothing about the fix it guards.
        build = subprocess.run(['cargo', 'build', '--release'],
                               cwd=REPO / 'rust', capture_output=True, text=True)
        if build.returncode != 0:
            sys.exit(f"cargo build --release failed:\n{build.stderr}")
    if not BIN.exists():
        sys.exit(f"{BIN} does not exist. Build it: cd rust && cargo build --release")
    table = json.loads(TABLE.read_text())['names']

    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for row in table:
            want = row['name']
            got_rust = rust_name(row, workdir)
            got_py = python_name(row)
            ok = got_rust == want and got_py == want
            print(f"  {'ok  ' if ok else 'FAIL'}  {want:32s} "
                  f"QM9={got_rust:32s} validation={got_py}")
            if not ok:
                failures.append((want, got_rust, got_py))

    print(f"\n{len(table)} conditions checked on both injectors")
    if failures:
        for want, r, p in failures:
            print(f"  {want}: QM9 says {r!r}, the validation injector says {p!r}")
        sys.exit(f"{len(failures)} condition(s) are named differently by the two "
                 f"injectors, or differently from condition_names.json")
    print("every condition has one name on both sides")


if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
