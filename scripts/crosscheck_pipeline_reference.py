#!/usr/bin/env python
"""The training pipeline's injector against the reference implementation.

`RERUN_PLAN.md` section 8 gate 1, `NOISE_DESIGN.md` section 6.3 item 1.

WHY THIS EXISTS
---------------
The noise scheme exists in three places, and only two of them were tied together.
`scripts/crosscheck_injectors.py` ties the REFERENCE (`rust/reference/noise_arms.rs`)
to the PYTHON injector (`NoiseInject/noiseInject/core.py`). Nothing tied either of
them to the thing that actually noises QM9: `rust/src/main.rs`.

That gap matters more than it looks. The reference is a clean-room prover with no
memmap, no RDKit and no pipeline around it; the pipeline is the code that touches
the data. They can drift in exactly the way the two injectors already drifted once,
and the existing gate would pass throughout.

Chain, once this runs:  python injector <-> reference <-> pipeline.

WHAT IT COMPARES, AND WHAT IT DELIBERATELY DOES NOT
---------------------------------------------------
Not element-wise. The two draw from separate generator streams in a different call
order, so identical draws are impossible and a check written that way would fail for
a reason that does not matter -- and would then be switched off.

So it compares statistics on the same labels, the same groups and the same seeds:

  * unit dose G          -- exact where G is deterministic (the shape-only conditions
                            and grouped-shifted). Where the scale map is itself drawn
                            (grouped-wider, outlier) G is a random variable, so it is
                            compared loosely and the affected fraction is checked
                            instead.
  * delivered dose       -- the mean over N seeds, which is the quantity the design
                            says to fix and report (NOISE_DESIGN.md 2a rule 3).
  * censoring            -- deterministic given the labels, so it must agree to
                            floating-point. This is the sharpest check in the file:
                            it is what caught the two implementations putting the
                            assay limit in different places.
  * shape diagnostics    -- median absolute error and the worst-hit 5%'s share of the
                            noise energy, so a condition cannot match on dose while
                            delivering a different shape.

Exits non-zero on any failure. Run it before any cluster time is spent.

    python scripts/crosscheck_pipeline_reference.py
    python scripts/crosscheck_pipeline_reference.py --seeds 40 --k 0.5
"""
import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REFERENCE_DIR = os.path.join(REPO, 'rust', 'reference')
REFERENCE_BIN = os.path.join(REFERENCE_DIR, 'target', 'release', 'noise_arms')
PIPELINE_BIN = os.path.join(REPO, 'rust', 'target', 'release', 'rust_processor')

# Where G is fixed by algebra rather than by a draw, so the two implementations must
# agree on it exactly.
DETERMINISTIC_G = {
    'gaussian', 'student_t_nu10', 'student_t_nu5', 'student_t_nu3', 'laplace',
    'grouped_shifted',
}

# Conditions with no selection rule: the shape is applied to every molecule, so the
# affected fraction is 1.0. All three implementations record it that way as of
# 2026-08-26 (NOISE_DESIGN.md 5.1c), so the column is compared like any other AND the
# value is pinned, because "no targeting applies -> 0.0" is a defensible reading of the
# name that would silently reintroduce the disagreement.
NO_SELECTION = {'gaussian', 'student_t_nu10', 'student_t_nu5', 'student_t_nu3', 'laplace',
                'grouped_shifted'}

# At nu <= 4 a Student-t has an infinite fourth moment, so any sample statistic that
# leans on it -- the empirical dose, the kurtosis, the share of energy in the worst-hit
# tail -- is unstable BY CONSTRUCTION. NOISE_DESIGN.md 5.1b already rules that the
# population value is what gets reported at those degrees of freedom; the same ruling
# has to govern what a cross-check may demand of them.
HEAVY_TAIL = {'student_t_nu3', 'student_t_nu4'}
HEAVY_TAIL_DOSE_TOL = 0.06
HEAVY_TAIL_SHAPE_TOL = 0.30


def build(path, cwd, what):
    if os.path.exists(path):
        return
    print(f"  building {what} ...")
    env = dict(os.environ)
    env.setdefault('CONDA_PREFIX', os.environ.get('CONDA_PREFIX', ''))
    r = subprocess.run(['cargo', 'build', '--release'], cwd=cwd, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        sys.exit(f"could not build {what}:\n{r.stderr}")


def run_json(cmd, what):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{what} failed:\n{r.stdout}\n{r.stderr}")
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError as e:
        sys.exit(f"{what} did not emit valid JSON ({e}):\n{r.stdout[:2000]}")


def rel(a, b):
    """Relative gap, guarding the zero denominator."""
    if a is None or b is None:
        return None
    if abs(b) < 1e-12:
        return abs(a - b)
    return abs(a / b - 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', default=os.path.join(REPO, 'data', 'qm9_gap_ev.csv'),
                    help="canonical_smiles,y per line for the pipeline; the reference "
                         "reads the y column of the same file")
    ap.add_argument('--groups', default=None,
                    help="JSON mapping canonical SMILES to scaffold group id. Without it "
                         "the two grouped conditions are not covered, and that is reported")
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--k', type=float, default=0.5)
    ap.add_argument('--dose-tol', type=float, default=0.03,
                    help="allowed gap between the two mean delivered doses")
    ap.add_argument('--shape-tol', type=float, default=0.10,
                    help="allowed relative gap on the shape diagnostics")
    ap.add_argument('--allow-partial', action='store_true',
                    help="pass even though the grouped conditions were not covered. "
                         "Without a group file they cannot be, and a gate that covers "
                         "15 of 17 conditions while reporting agreement is worse than "
                         "no gate")
    args = ap.parse_args()

    if not os.path.exists(args.labels):
        sys.exit(f"no labels file at {args.labels} — pass --labels")

    build(REFERENCE_BIN, REFERENCE_DIR, 'the reference implementation')
    build(PIPELINE_BIN, os.path.join(REPO, 'rust'), 'the pipeline')

    # The reference takes plain columns; the pipeline takes the smiles,y file and a
    # SMILES-keyed group map. Derive the reference's inputs from the same source, so
    # there is no chance of the two reading different data.
    rows = [l.rstrip('\n') for l in open(args.labels) if l.strip()]
    scratch = os.path.join(REPO, '.crosscheck_tmp')
    os.makedirs(scratch, exist_ok=True)
    ref_labels = os.path.join(scratch, 'labels.txt')
    smiles = []
    with open(ref_labels, 'w') as f:
        for line in rows:
            smi, _, y = line.rpartition(',')
            smiles.append(smi)
            f.write(y + '\n')

    ref_cmd = [REFERENCE_BIN, ref_labels, '--json', '--seeds', str(args.seeds),
               '--k', str(args.k)]
    pipe_cmd = [PIPELINE_BIN, '--self-test', args.labels, '--json',
                '--seeds', str(args.seeds), '--noise-level', str(args.k)]

    if args.groups:
        groups = json.load(open(args.groups))
        missing = [s for s in smiles if s not in groups]
        if missing:
            sys.exit(f"{len(missing)} molecules are missing from the group file — "
                     "it does not describe these labels")
        ref_groups = os.path.join(scratch, 'groups.txt')
        with open(ref_groups, 'w') as f:
            f.write('\n'.join(str(groups[s]) for s in smiles) + '\n')
        ref_cmd += ['--groups', ref_groups]
        pipe_cmd += ['--scaffold-file', args.groups]

    print("CROSS-CHECK: the reference implementation against the training pipeline")
    print(f"  labels {args.labels}  n={len(rows)}  k={args.k}  seeds={args.seeds}")
    if not args.groups:
        print("  NOTE: no --groups given, so grouped-wider and grouped-shifted are NOT covered")
    print()

    ref = {r['condition']: r for r in run_json(ref_cmd, 'the reference')['rows']}
    pipe = {r['condition']: r for r in run_json(pipe_cmd, 'the pipeline')['rows']}

    failures = []
    uncovered = []

    # Without a real scaffold assignment the two sides behave differently ON PURPOSE.
    # The reference falls back to 2,000 synthetic clusters so it still has something
    # to draw; the pipeline REFUSES, because "grouped noise" over invented groups is
    # uniform noise wearing a grouped name. So the grouped conditions are not
    # comparable here -- they are simply uncovered, and this says so rather than
    # reporting agreement over a roster with two conditions quietly missing.
    if not args.groups:
        grouped = {c for c in set(ref) | set(pipe) if c.startswith('grouped_')}
        uncovered = sorted(grouped)
        for c in grouped:
            ref.pop(c, None)
            pipe.pop(c, None)

    only_ref = sorted(set(ref) - set(pipe))
    only_pipe = sorted(set(pipe) - set(ref))
    if only_ref:
        failures.append(f"the reference emits conditions the pipeline does not: {only_ref}")
    if only_pipe:
        failures.append(f"the pipeline emits conditions the reference does not: {only_pipe}")

    hdr = f"{'condition':<18}{'G ref':>9}{'G pipe':>9}  {'dose ref':>10}{'dose pipe':>11}{'gap':>8}  {'med|e|':>8}{'top5%':>8}"
    print(hdr)
    print('-' * len(hdr))

    for cond in sorted(set(ref) & set(pipe)):
        r, p = ref[cond], pipe[cond]
        bad = []

        # unit dose
        if cond in DETERMINISTIC_G and r['unit_dose'] is not None:
            if rel(p['unit_dose'], r['unit_dose']) > 1e-4:
                bad.append(f"unit dose {p['unit_dose']} vs {r['unit_dose']}")

        # delivered dose. Censoring is deterministic, so it is held to a far tighter
        # line than the conditions that are averaged over draws; the heaviest tails
        # are held to a looser one, for the reason recorded at HEAVY_TAIL.
        if not r['dose_matched']:
            tol = 1e-5
        elif cond in HEAVY_TAIL:
            tol = max(args.dose_tol, HEAVY_TAIL_DOSE_TOL)
        else:
            tol = args.dose_tol
        gap = rel(p['delivered_dose'], r['delivered_dose'])
        if gap is not None and gap > tol:
            bad.append(f"delivered dose differs by {gap * 100:.3f}% (tolerance {tol * 100:g}%)")

        # At a censored fraction of zero nothing is clipped, so neither side records a
        # limit; there is nothing to compare and that is not a disagreement.
        if (not r['dose_matched']
                and r['censoring_limit'] is not None
                and p['censoring_limit'] is not None):
            if rel(p['censoring_limit'], r['censoring_limit']) > 1e-5:
                bad.append(f"censoring limit {p['censoring_limit']} vs {r['censoring_limit']}")

        shape_tol = max(args.shape_tol, HEAVY_TAIL_SHAPE_TOL) if cond in HEAVY_TAIL else args.shape_tol
        for key, tolerance in (('median_abs', shape_tol),
                               ('top5_energy_share', shape_tol),
                               ('affected_molecule_fraction', 0.05)):
            a, b = p.get(key), r.get(key)
            if a is None or b is None:
                continue
            if key == 'affected_molecule_fraction':
                if cond in NO_SELECTION:
                    # both must say 1.0, not merely agree with each other
                    for who, v in (('the pipeline', a), ('the reference', b)):
                        if abs(v - 1.0) > 1e-6:
                            bad.append(f"{who} reports {v:.4f} of molecules affected "
                                       f"for a condition with no selection rule, not 1.0")
                elif abs(a - b) > tolerance:
                    bad.append(f"{key} {a:.4f} vs {b:.4f}")
            elif rel(a, b) > tolerance:
                bad.append(f"{key} differs by {rel(a, b) * 100:.1f}%")

        gfmt = lambda v: f"{v:>9.4f}" if v is not None else f"{'-':>9}"
        print(f"{cond:<18}{gfmt(r['unit_dose'])}{gfmt(p['unit_dose'])}  "
              f"{r['delivered_dose']:>10.5f}{p['delivered_dose']:>11.5f}"
              f"{(gap or 0) * 100:>+7.2f}%  "
              f"{p['median_abs']:>8.4f}{p['top5_energy_share'] * 100:>7.1f}%"
              f"{'' if not bad else '   FAIL'}")
        for b in bad:
            print(f"    - {b}")
            failures.append(f"{cond}: {b}")

    print()
    if failures:
        print(f"FAILED — {len(failures)} disagreement(s) between the reference and the pipeline")
        for f in failures:
            print(f"  {f}")
        return 1

    covered = len(set(ref) & set(pipe))
    if uncovered:
        print(f"the {covered} compared conditions agree, but {len(uncovered)} were NOT "
              f"covered: {uncovered}")
        print("  pass --groups <scaffold_groups.json> to cover them")
        if not args.allow_partial:
            print("PARTIAL — not a pass. Re-run with a group file, or --allow-partial "
                  "if you really mean to skip them")
            return 2
        print("PARTIAL, accepted by --allow-partial")
        return 0

    print(f"the reference implementation and the training pipeline agree "
          f"on all {covered} conditions")
    return 0


if __name__ == '__main__':
    sys.exit(main())
