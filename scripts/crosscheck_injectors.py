#!/usr/bin/env python
"""Gate 2: the Rust and Python noise injectors deliver the same thing.

`RERUN_PLAN.md` section 8 gate 2, `NOISE_DESIGN.md` section 6.3 item 6.

WHY THIS EXISTS
---------------
The noise scheme is implemented twice -- `rust/src/main.rs` for QM9, and
`NoiseInject/noiseInject/core.py` for the three experimental datasets and every
uncertainty number. They have already drifted apart once, on a constant and on
how cut-points are computed, and nothing noticed for the life of the project.
Two implementations of one specification stay together only if something
executable checks that they do.

WHY IT IS NOT AN ELEMENT-WISE CHECK
-----------------------------------
Rust's StdRng and numpy's RandomState produce different streams, so identical
draws are impossible. A check written that way would fail for a reason that
does not matter and would then be disabled -- which is worse than no check.
So it compares STATISTICS on the same labels, the same target, the same group
assignment and the same seeds: the delivered dose, the unit dose, how many
labels end up badly wrong, the median error, how concentrated the damage is,
and how many records were affected.

Exits non-zero on any failure. Run it before any cluster time is spent.

    python scripts/crosscheck_injectors.py                 # QM9, all conditions
    python scripts/crosscheck_injectors.py --seeds 40      # tighter
    python scripts/crosscheck_injectors.py --labels my.txt --groups my_groups.txt
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO, 'rust', 'reference')
BINARY = os.path.join(REFERENCE_DIR, 'target', 'release', 'noise_arms')

sys.path.insert(0, os.path.expanduser('~/repos/NoiseInject'))
try:
    from noiseInject import CONDITIONS, NoiseInjectorRegression, dose_tolerance
except ImportError as exc:                                   # pragma: no cover
    sys.exit(f"noiseInject is not importable ({exc}).\n"
             "Install it: pip install -e ~/repos/NoiseInject")


# --- tolerances -------------------------------------------------------------
#
# DERIVED per condition, not hand-kept. The sampling spread of a root-mean-square
# estimate depends on the fourth moment and on how many independent
# contributions it is averaged over, so a heavy-tailed shape and a group-level
# term are imprecise for the same structural reason, not because either is
# defective. A fixed list of "unstable conditions" would need editing whenever a
# condition is added, and would silently stop covering the new one.
# `noiseInject.dose_tolerance` and `dose_tolerance` in `rust/src/main.rs` are the
# same function.

# The unit dose is arithmetic, so it must agree tightly -- except where the two
# implementations select a different set of groups or records (different RNG
# streams), so the realised affected fraction differs slightly.
UNIT_DOSE_TOL = 0.002
UNIT_DOSE_TOL_SELECTED = 0.03
SELECTED = ('grouped_wider', 'outlier_p01', 'outlier_p05', 'outlier_p10')

SHAPE_TOL = 0.10          # relative, on the shape diagnostics
SHAPE_FLOOR = 0.002       # absolute floor: these are small fractions
AFFECTED_TOL = 0.02       # absolute, on the affected molecule fraction


class Report:
    def __init__(self):
        self.failures = []
        self.checks = 0

    def check(self, name, ok, detail=''):
        self.checks += 1
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  ' + detail) if detail else ''}")
        if not ok:
            self.failures.append(f"{name}  {detail}")


# --- labels and groups -------------------------------------------------------

def qm9_labels():
    """The real QM9 HOMO-LUMO gaps, in electronvolts.

    The pipeline works in eV, not Hartree, and that conversion is not cosmetic:
    the retired threshold condition fired on |y| >= 1.0 against eV and so caught
    99.9992% of molecules, while in Hartree the same cut would have caught none.
    """
    import pandas as pd
    csv = os.path.join(REPO, 'data', 'QM9', 'raw', 'gdb9.sdf.csv')
    if not os.path.exists(csv):
        sys.exit(f"QM9 labels not found at {csv}")
    return (pd.read_csv(csv)['gap'].values * 27.211386).astype(np.float64)


def qm9_scaffold_groups(n, cache=None):
    """Real Murcko scaffold groups, aligned to the CSV by RECORD POSITION.

    Two things this has to get right.

    Section 2a rule 2: 32.2% of QM9 molecules have an empty Murcko scaffold,
    because they are acyclic. Treating that bucket as one group would let a
    single offset draw move a third of the dataset, so each acyclic molecule
    becomes its own singleton group.

    And 1,405 of the 133,885 SDF records do not parse in RDKit. Skipping them
    would shift every later scaffold against its label -- exactly the index
    drift recorded in RERUN_PLAN.md section 2.7. So iterate by position and give
    an unparseable record its own singleton group rather than dropping it.
    """
    if cache and os.path.exists(cache):
        groups = np.loadtxt(cache, dtype=np.int64)
        if len(groups) == n:
            return groups
        print(f"  cached groups have length {len(groups)}, need {n}; recomputing")

    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    sdf = os.path.join(REPO, 'data', 'QM9', 'raw', 'gdb9.sdf')
    print(f"  computing Murcko scaffolds for {n} molecules (one-off, then cached) ...")
    scaffolds = []
    unparseable = acyclic = 0
    for position, mol in enumerate(Chem.SDMolSupplier(sdf, removeHs=False)):
        if position >= n:
            break
        if mol is None:
            unparseable += 1
            scaffolds.append(f'__unparseable_{position}')
            continue
        try:
            s = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            s = ''
        if s == '':
            acyclic += 1
            s = f'__acyclic_{position}'          # rule 2
        scaffolds.append(s)

    if len(scaffolds) != n:
        sys.exit(f"{sdf} holds {len(scaffolds)} records, but there are {n} labels")

    _, groups = np.unique(np.array(scaffolds), return_inverse=True)
    groups = groups.astype(np.int64)
    ringed = n - acyclic - unparseable
    print(f"  {len(np.unique(groups))} groups: {ringed} molecules in real scaffold groups, "
          f"{acyclic} acyclic and {unparseable} unparseable as singletons")
    if cache:
        np.savetxt(cache, groups, fmt='%d')
    return groups


# --- the two implementations -------------------------------------------------

def run_rust(labels_path, groups_path, ks, seeds):
    if not os.path.exists(BINARY):
        print("  building the reference implementation ...")
        build = subprocess.run(['cargo', 'build', '--release'], cwd=REFERENCE_DIR,
                               capture_output=True, text=True)
        if build.returncode != 0:
            sys.exit("cargo build failed:\n" + build.stderr)
    cmd = [BINARY, labels_path, '--groups', groups_path, '--json',
           '--seeds', str(seeds), '--k', ','.join(str(k) for k in ks)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"the reference implementation failed:\n{out.stderr}")
    return json.loads(out.stdout)


def python_stats(condition, y, groups, k, label_sd, seeds):
    """The same statistics the reference emits, from the Python injector.

    `condition` is the reported name. For censoring that name carries the
    censored fraction (`censoring_25`), which is derived from the level rather
    than stored beside it -- so it maps back to the single `censoring` entry in
    the registry, with the fraction passed as the level.
    """
    if condition.startswith('censoring_'):
        registry_name = 'censoring'
        target = int(condition.rsplit('_', 1)[1]) / 100.0
    else:
        registry_name = condition
        target = k * label_sd
    spec = CONDITIONS[registry_name]
    dose_matched = spec['strategy'] != 'censoring'

    realised, shift, f3, med, top5, affected = [], [], [], [], [], []
    unit_dose = solved = limit = float('nan')
    effective_n = tol = float('nan')

    for seed in range(seeds):
        inj = NoiseInjectorRegression.from_condition(registry_name, random_state=42 + seed)
        r = inj.inject_verbose(y, target, groups=groups)
        eps = r.epsilon
        realised.append(r.realised_dose_label_units)
        shift.append(r.mean_epsilon)
        # For censoring the dose is a diagnostic, so scale the "badly wrong"
        # threshold to what was actually delivered -- as the reference does.
        thr = 3.0 * (target if dose_matched else r.realised_dose_label_units)
        f3.append(float(np.mean(np.abs(eps) > thr)))
        med.append(float(np.median(np.abs(eps))))
        sq = np.sort(eps ** 2)[::-1]
        total = sq.sum()
        n_top = max(1, int(round(len(sq) * 0.05)))
        top5.append(float(sq[:n_top].sum() / total) if total > 0 else 0.0)
        affected.append(r.affected_molecule_fraction)
        unit_dose, solved = r.unit_dose_g, r.solved_scale
        limit, effective_n = r.censoring_limit, r.effective_n
        tol = dose_tolerance(eps, r.effective_n, nu=spec.get('nu'))

    return {
        'condition': condition, 'k': k, 'dose_matched': dose_matched,
        'target_dose': target, 'unit_dose': unit_dose, 'solved_scale': solved,
        'censoring_limit': limit,
        'delivered_dose': float(np.mean(realised)),
        'delivered_dose_sd': float(np.std(realised)),
        'mean_shift': float(np.mean(shift)),
        'frac_beyond_3': float(np.mean(f3)),
        'median_abs': float(np.mean(med)),
        'top5_energy_share': float(np.mean(top5)),
        'affected_molecule_fraction': float(np.mean(affected)),
        'effective_n': effective_n,
        'dose_tolerance': tol,
        'seeds': seeds,
    }


# --- comparisons -------------------------------------------------------------

def _fmt(v):
    """Format a number that may legitimately be missing."""
    if v is None or v != v:
        return 'none'
    return f"{v:.4f}"


def close(a, b, rel, floor=0.0):
    if a is None or b is None:
        # Missing on both sides is agreement -- a quantity that does not exist
        # for this condition (a censoring limit at fraction 0, a unit dose for a
        # condition that is not dose-matched) must be missing in BOTH.
        return a is None and b is None
    if any(x != x for x in (a, b)):          # NaN on both sides is agreement
        return (a != a) and (b != b)
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), floor)


def compare(rep, rust_row, py_row):
    c = rust_row['condition']
    tag = f"{c} @ k={rust_row['k']}"
    # Derived per condition, and shrunk by the number of seeds each side
    # averaged over. The comparison BETWEEN implementations is a difference of
    # two independent means, so its spread is sqrt(2) wider than either.
    seeds = rust_row['seeds']
    base_tol = max(rust_row.get('dose_tolerance') or 0.005,
                   py_row.get('dose_tolerance') or 0.005)
    dose_tol = max(base_tol / np.sqrt(seeds), 0.002)
    pair_tol = max(base_tol * np.sqrt(2.0 / seeds), 0.002)

    if rust_row['dose_matched']:
        # 1. The solved scale hits the target exactly. Pure arithmetic, and the
        #    half of the gate that does not depend on any draw.
        exact = py_row['unit_dose'] * py_row['solved_scale']
        rep.check(f"{tag}: solved scale hits the target exactly",
                  close(exact, py_row['target_dose'], 1e-9),
                  f"{exact:.6f} vs {py_row['target_dose']:.6f}")

        # 2. Both implementations deliver the target amount.
        for impl, row in (('rust', rust_row), ('python', py_row)):
            rep.check(f"{tag}: {impl} delivers the target dose",
                      close(row['delivered_dose'], row['target_dose'], dose_tol),
                      f"asked {row['target_dose']:.4f}, delivered {row['delivered_dose']:.4f}"
                      f" ({100 * (row['delivered_dose'] / row['target_dose'] - 1):+.2f}%,"
                      f" tolerance {100 * dose_tol:.2f}%, effective n {row['effective_n']:.0f})")

        # 3. Unit dose: arithmetic, so it must agree tightly.
        u_tol = UNIT_DOSE_TOL_SELECTED if c in SELECTED else UNIT_DOSE_TOL
        rep.check(f"{tag}: unit dose agrees",
                  close(rust_row['unit_dose'], py_row['unit_dose'], u_tol),
                  f"rust {rust_row['unit_dose']:.4f} vs python {py_row['unit_dose']:.4f}")
    else:
        # Censoring is deterministic given the labels, so it must agree closely.
        # At a censored fraction of 0 there is no limit at all, on either side --
        # that is the clean baseline, and both must report nothing rather than a
        # number. `_fmt` prints a missing value as "none" instead of crashing.
        rep.check(f"{tag}: censoring limit agrees",
                  close(rust_row['censoring_limit'], py_row['censoring_limit'], 0.001),
                  f"rust {_fmt(rust_row['censoring_limit'])} vs "
                  f"python {_fmt(py_row['censoring_limit'])}")
        rep.check(f"{tag}: a censored fraction of 0 clips nothing",
                  (rust_row['affected_molecule_fraction'] > 0) or
                  (rust_row['censoring_limit'] is None and
                   rust_row['delivered_dose'] == 0 and py_row['delivered_dose'] == 0),
                  '' if rust_row['affected_molecule_fraction'] > 0 else
                  'the clean baseline is genuinely clean')

    # 4. The delivered amount agrees BETWEEN the implementations.
    rep.check(f"{tag}: delivered dose agrees across implementations",
              close(rust_row['delivered_dose'], py_row['delivered_dose'], pair_tol),
              f"rust {rust_row['delivered_dose']:.4f} vs python {py_row['delivered_dose']:.4f}"
              f" (tolerance {100 * pair_tol:.2f}%)")

    # 5. Shape: matched amount must not mean matched shape, and the two
    #    implementations must agree on the shape as well as the amount.
    for field, tol, floor in (('frac_beyond_3', SHAPE_TOL, SHAPE_FLOOR),
                              ('median_abs', SHAPE_TOL, 1e-6),
                              ('top5_energy_share', SHAPE_TOL, 0.01)):
        rep.check(f"{tag}: {field} agrees",
                  close(rust_row[field], py_row[field], tol, floor),
                  f"rust {rust_row[field]:.5f} vs python {py_row[field]:.5f}")

    # 6. The affected fraction is what the solver divides by, so it has to be
    #    right AND recorded -- section 2a rule 1.
    rep.check(f"{tag}: affected molecule fraction agrees",
              close(rust_row['affected_molecule_fraction'],
                    py_row['affected_molecule_fraction'], 0.0, AFFECTED_TOL),
              f"rust {rust_row['affected_molecule_fraction']:.4f} vs "
              f"python {py_row['affected_molecule_fraction']:.4f}")


def check_dose_is_flat(rep, rows, k, impl):
    """The single check that proves the confound is gone.

    At one target, every dose-matched condition must deliver the SAME amount. If
    this fails the whole re-run is confounded and worthless: the six superseded
    conditions delivered between 0.49x and 2.00x the same amount at one setting,
    and their entire apparent severity ordering was that.
    """
    matched = [r for r in rows if r['dose_matched'] and r['k'] == k]
    if not matched:
        return
    target = matched[0]['target_dose']
    seeds = matched[0]['seeds']

    def within(r):
        tol = max((r.get('dose_tolerance') or 0.005) / np.sqrt(seeds), 0.002)
        return close(r['delivered_dose'], target, tol)

    # Name the conditions that actually FAIL, not merely the largest deviation:
    # a condition can deviate most and still be well inside its own tolerance,
    # and reporting it as "worst" points the reader at the wrong row.
    offenders = [f"{r['condition']} {100 * (r['delivered_dose'] / target - 1):+.2f}%"
                 f" (tolerance {100 * max((r.get('dose_tolerance') or 0.005) / np.sqrt(seeds), 0.002):.2f}%)"
                 for r in matched if not within(r)]
    worst = max(matched, key=lambda r: abs(r['delivered_dose'] / target - 1))
    rep.check(f"{impl}: dose is flat across all {len(matched)} conditions at k={k}",
              not offenders,
              '; '.join(offenders) if offenders else
              f"largest deviation {worst['condition']} "
              f"{100 * (worst['delivered_dose'] / target - 1):+.2f}%, inside its tolerance")

    # And the SPREAD between conditions is what a reader would see as a
    # difference in severity. Before the redesign it was 0.49x to 2.00x.
    lo = min(r['delivered_dose'] for r in matched)
    hi = max(r['delivered_dose'] for r in matched)
    rep.check(f"{impl}: spread across conditions at k={k} is under 3%",
              (hi / lo - 1) < 0.03,
              f"{100 * (hi / lo - 1):.2f}% (was 0.49x to 2.00x, i.e. 308%)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', help='one label per line; default is real QM9 gaps in eV')
    ap.add_argument('--groups', help='one integer group id per line, same length as the labels')
    ap.add_argument('--k', default='0.25,0.5,1.0', help='noise levels, as fractions of the label spread')
    ap.add_argument('--seeds', type=int, default=20,
                    help='seeds to average the realised dose over. The gate is on the '
                         'POPULATION dose, so one realisation is not enough: at n=133,885 '
                         'the sampling spread of an RMS estimate is already 0.2%%')
    ap.add_argument('--scratch', default=os.path.join(REPO, 'scratchpad_crosscheck'),
                    help='where the shared label and group files are written')
    args = ap.parse_args()

    ks = [float(x) for x in args.k.split(',')]
    os.makedirs(args.scratch, exist_ok=True)

    print("=" * 78)
    print("CROSS-CHECK: the Rust reference against the Python injector")
    print("=" * 78)

    if args.labels:
        y = np.loadtxt(args.labels, dtype=np.float64)
        labels_path = os.path.abspath(args.labels)
    else:
        y = qm9_labels()
        labels_path = os.path.join(args.scratch, 'labels.txt')
        np.savetxt(labels_path, y, fmt='%.10f')

    if args.groups:
        groups = np.loadtxt(args.groups, dtype=np.int64)
        groups_path = os.path.abspath(args.groups)
    else:
        groups = qm9_scaffold_groups(len(y), cache=os.path.join(args.scratch, 'groups.txt'))
        groups_path = os.path.join(args.scratch, 'groups.txt')
        if not os.path.exists(groups_path):
            np.savetxt(groups_path, groups, fmt='%d')

    if len(groups) != len(y):
        sys.exit(f"groups has length {len(groups)} but there are {len(y)} labels")

    label_sd = float(np.std(y))
    sizes = np.bincount(groups)
    print(f"\nlabels: n={len(y)}  mean={y.mean():.4f}  SD={label_sd:.4f}")
    print(f"groups: {len(np.unique(groups))} distinct, largest holds "
          f"{sizes.max() / len(y):.1%} of molecules")
    print(f"seeds:  {args.seeds}   levels: {ks}\n")

    rust = run_rust(labels_path, groups_path, ks, args.seeds)
    rust_rows = rust['rows']

    rep = Report()
    py_rows = []
    for rust_row in rust_rows:
        condition, k = rust_row['condition'], rust_row['k']
        if condition not in CONDITIONS and not condition.startswith('censoring_'):
            rep.check(f"{condition}: present in the Python registry", False,
                      "the reference emits a condition the Python side does not know")
            continue
        py_rows.append(python_stats(condition, y, groups, k, label_sd, args.seeds))

    # A condition missing from the reference must fail loudly, not be skipped:
    # a silent no-op is how the placebo column stayed blank for months.
    emitted = {r['condition'] for r in rust_rows}
    for condition in CONDITIONS:
        covered = (condition in emitted or
                   (condition == 'censoring' and
                    any(e.startswith('censoring_') for e in emitted)))
        rep.check(f"{condition}: covered by the reference implementation", covered,
                  '' if covered else "the reference does not implement it")

    print("\n--- the confound check -------------------------------------------------")
    for k in ks:
        check_dose_is_flat(rep, rust_rows, k, 'rust  ')
        check_dose_is_flat(rep, py_rows, k, 'python')

    print("\n--- condition by condition ---------------------------------------------")
    by_key = {(r['condition'], r['k']): r for r in py_rows}
    for rust_row in rust_rows:
        key = (rust_row['condition'], rust_row['k'])
        if key in by_key:
            compare(rep, rust_row, by_key[key])

    print("\n" + "=" * 78)
    if rep.failures:
        print(f"FAILED: {len(rep.failures)} of {rep.checks} checks")
        for f in rep.failures:
            print(f"  - {f}")
        print("=" * 78)
        return 1
    print(f"PASSED: all {rep.checks} checks. The two injectors agree.")
    print("=" * 78)
    return 0


if __name__ == '__main__':
    sys.exit(main())
