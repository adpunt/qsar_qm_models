#!/usr/bin/env python3
"""The two injectors write the same SHAPE column.

WHY THIS EXISTS
---------------
`noise_pattern` is the level-free shape of the noise: how much each molecule's
region would receive at a reference dose. It is the column the uncertainty
question is answered against -- `confound_controlled_effect` correlates the
predicted uncertainty against it and subtracts the same correlation at level
zero.

Two implementations write it: `rust/src/main.rs` for QM9 and `noiseInject` for
logD, Caco-2 and hERG. **Nothing compared them.** `scripts/crosscheck_injectors.py`
compares delivered dose, unit dose, median error, top-5% energy share, affected
fraction and the censoring limit -- six quantities, and not this one. That is how
the censoring shape ran with the OPPOSITE SIGN on the two sides for a week: a
model that found the clipped molecules perfectly scored +0.88 on the laboratory
datasets and -0.88 on QM9, and QM9's "does the uncertainty find the bad labels"
answer came out inverted (RERUN_PLAN.md 2.26c).

It matters more now than it did then. Censoring is the ONLY condition on which
that question has an answer at all -- the four uniform conditions have a flat
shape, and the grouped and outlier conditions are structural nulls under a
scaffold split (RERUN_PLAN.md 3.1f). So this column IS the answer, and until this
script existed nothing checked that the two halves of the study wrote the same
one.

WHAT IT COMPARES, AND WHY THESE FIVE
------------------------------------
Each catches a different way of getting the column wrong:

* ``frac_negative`` and the sign of the extremes -- a sign flip, which is the
  defect that actually happened.
* ``spearman_label`` -- the rank order against the clean label. Exactly +1 on one
  side and -1 on the other under a flip, whatever the scale, and it is the
  statistic the analysis effectively reads.
* ``rms`` -- a scale that has drifted. The shape is quoted at a reference dose
  and both sides must quote it at the same one.
* ``max/min`` -- the shape's own contrast, free of scale. For the outlier and
  grouped conditions the affected SET is a random draw and the two sides draw
  independently, so their extremes sit on different molecules; the ratio between
  the two levels does not.
* flatness -- a condition that gives every molecule the same amount must be flat
  on BOTH sides, and one that does not must be flat on neither. A shape that is
  flat where it should not be is a condition that silently did nothing.

    python scripts/crosscheck_noise_pattern.py \\
        --labels <canonical_smiles,y csv> --groups <smiles->group json>
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(REPO, 'rust', 'target', 'release', 'rust_processor')

# The reference dose the shape is quoted at.
#
# Every dose-matched condition quotes it at one clean training label spread --
# "the amount delivered at a level of 1.0". CENSORING IS EXEMPT: its level is the
# FRACTION of labels clipped, which is already dimensionless, so its reference is
# 1.0 and the shape is the distance past the far end of the training range. The
# laboratory runner makes the same exemption for the same reason
# (`alternative_data_noise_robustness.py`, the `_censoring` branch); multiplying a
# fraction by a spread gives a quantile outside [0, 1] and numpy raises.
CENSORING_REFERENCE_DOSE = 1.0


def spearman(a, b):
    """Rank correlation, NaN on a constant input rather than a warning."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def stats(pattern, labels):
    p = np.asarray(pattern, dtype=float)
    lo, hi = float(np.min(p)), float(np.max(p))
    flat = (hi - lo) <= 1e-9 * max(1.0, abs(hi))
    return {
        'min': lo,
        'max': hi,
        'rms': float(np.sqrt((p ** 2).mean())),
        'frac_negative': float((p < 0).mean()),
        'contrast': float('nan') if flat or lo <= 0 else hi / lo,
        'spearman_label': float('nan') if flat else spearman(p, labels),
        'flat': flat,
    }


def rust_rows(labels_csv, groups_json, level, seeds):
    if not os.path.exists(BINARY):
        sys.exit(f"no binary at {BINARY} -- run: cd rust && cargo build --release")
    cmd = [BINARY, '--self-test', labels_csv, '--json',
           '--seeds', str(seeds), '--noise-level', str(level)]
    if groups_json:
        cmd += ['--scaffold-file', groups_json]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"the injector failed:\n{out.stderr[-3000:]}")
    doc = json.loads(out.stdout)
    rows = {}
    for r in doc['rows']:
        if 'pattern_rms' not in r:
            sys.exit("this binary emits no pattern columns -- it predates the "
                     "shape cross-check. Rebuild: cd rust && cargo build --release")
        lo, hi = r['pattern_min'], r['pattern_max']
        flat = lo is not None and hi is not None and (hi - lo) <= 1e-9 * max(1.0, abs(hi))
        rows[r['condition']] = {
            'min': lo, 'max': hi, 'rms': r['pattern_rms'],
            'frac_negative': r['pattern_frac_negative'],
            'contrast': (float('nan') if flat or not lo or lo <= 0 else hi / lo),
            'spearman_label': (float('nan') if r['pattern_spearman_label'] is None
                               else r['pattern_spearman_label']),
            'flat': flat,
        }
    return rows, doc['label_sd']


def python_stats(condition, y, groups, label_sd):
    sys.path.insert(0, os.path.join(os.path.dirname(REPO), 'NoiseInject'))
    from noiseInject import NoiseInjectorRegression, CONDITIONS
    if condition not in CONDITIONS:
        return None, f'{condition} is not a condition noiseInject knows'
    inj = NoiseInjectorRegression.from_condition(condition, random_state=1000)
    dose = (CENSORING_REFERENCE_DOSE
            if CONDITIONS[condition]['strategy'] == 'censoring' else label_sd)
    try:
        p = inj.noise_scale(y, dose, reference=y, groups=groups,
                            reference_groups=groups)
    except Exception as exc:
        return None, f'{type(exc).__name__}: {exc}'
    return stats(np.asarray(p, dtype=float), y), None


def close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    if a != a or b != b:                       # NaN on both sides is agreement
        return (a != a) and (b != b)
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--labels', required=True,
                    help='canonical_smiles,y csv, as the pipeline reads it')
    ap.add_argument('--groups', default=None,
                    help='JSON of canonical SMILES -> scaffold group id. Without '
                         'it the two grouped conditions are NOT covered, and this '
                         'says so rather than reporting agreement over a short set')
    ap.add_argument('--level', type=float, default=0.5)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--tol', type=float, default=1e-4,
                    help='relative tolerance on the scale-bearing statistics')
    args = ap.parse_args()

    rows = [l.rstrip('\n') for l in open(args.labels) if l.strip()]
    if rows and rows[0].lower().startswith('canonical_smiles'):
        rows = rows[1:]
    smiles, y = [], []
    for line in rows:
        smi, _, val = line.rpartition(',')
        smiles.append(smi)
        y.append(float(val))
    y = np.asarray(y, dtype=float)
    groups = None
    if args.groups:
        gmap = json.load(open(args.groups))
        missing = [s for s in smiles if s not in gmap]
        if missing:
            sys.exit(f"{len(missing)} molecules are not in the group file")
        groups = np.asarray([gmap[s] for s in smiles])

    rust, label_sd = rust_rows(args.labels, args.groups, args.level, args.seeds)

    print("CROSS-CHECK: the level-free shape column, on both injectors")
    print(f"  labels {args.labels}  n={len(y)}  label_sd={label_sd:.6f}")
    if groups is None:
        print("  NOTE: no --groups, so the grouped conditions are NOT covered")
    print()

    # The affected SET is an independent draw on the two sides, so the extremes
    # sit on different molecules. Those two statistics are compared as a ratio,
    # which the draw does not move, and the extremes themselves are reported but
    # not asserted.
    DRAWN = ('grouped_wider', 'outlier_p01', 'outlier_p05', 'outlier_p10')
    # The settled parameters, read from the injector rather than retyped.
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(REPO), 'NoiseInject'))
        from noiseInject import CONDITIONS as CONDITION_PARAMS
    except ImportError:
        CONDITION_PARAMS = {}

    failures, checked, uncovered = [], 0, []
    for name, r in sorted(rust.items()):
        base = name.rsplit('_', 1)[0] if name.startswith('censoring') else name
        base = 'censoring' if base.startswith('censoring') else base
        p, why = python_stats(base, y, groups, label_sd)
        if p is None:
            uncovered.append(f'{name}: {why}')
            continue
        drawn = base in DRAWN
        keys = ['flat', 'frac_negative', 'rms', 'contrast']
        bad = []
        if drawn:
            # HOW BIG A CORRELATION IS "ZERO" DEPENDS ON HOW MANY INDEPENDENT
            # THINGS WERE DRAWN, and for a grouped condition that is the number
            # of SCAFFOLD GROUPS, not the number of molecules.
            #
            # This was a flat 0.05 and it made the gate a coin flip on
            # grouped_wider. Measured on 5,000 real QM9 labels over 200 seeds:
            # the correlation is centred on +0.0041 (0.87 standard errors from
            # zero, so there is no leak) with a spread of 0.0662, and 48% of
            # correct draws land past 0.05 -- on BOTH sides at once, because
            # both are drawing. It never showed up before because the only
            # inputs lying around were synthetic, where the labels are
            # independent of the scaffolds and the clustering that widens this
            # does not exist.
            #
            # `outlier_p10` picks molecules one at a time, so its spread is
            # 0.0141 and the old threshold was safe there. That is the whole
            # difference, and it is exactly n_units.
            # The unit is what the condition DRAWS, and the count is how many
            # of them carry the contrast. `grouped_wider` marks a fraction of
            # scaffold groups, so the shape is a two-level contrast driven by
            # the marked groups alone -- 0.2 x 1,816 = 363 of them at 5,000
            # molecules, not 5,000 and not 1,816. `outlier_p10` marks molecules
            # one at a time, so its unit is the molecule.
            #
            # The constant is CALIBRATED, not chosen, the same way the heavy-tail
            # dose tolerance is: over 200 seeds on 5,000 real QM9 labels the
            # correlation's spread was 0.0662 at 363 marked groups, which fixes
            # it at 0.0662 x sqrt(363) = 1.26. Three of those, so a correct draw
            # lands outside about one time in three hundred. Checked at the same
            # size on the other side: outlier_p10's spread was 0.0141 against
            # 1/sqrt(5000) = 0.0141.
            RHO_SE_AT_UNIT_N = 1.26
            if groups is not None and base.startswith('grouped'):
                frac = float(CONDITION_PARAMS.get(base, {}).get('group_fraction', 1.0))
                n_units = max(2.0, frac * len(np.unique(groups)))
            else:
                n_units = max(2.0, float(len(y)))
            rho_band = max(3.0 * RHO_SE_AT_UNIT_N / math.sqrt(n_units), 0.02)
            # WHICH molecules were hit is an independent draw on the two sides,
            # so the extremes sit on different molecules and the rank
            # correlation with the label is a property of that draw. Requiring
            # the two to be EQUAL would be requiring the two injectors to make
            # the same random selection, which they are not built to do and
            # which no result depends on.
            #
            # What must hold on both sides is that the shape is uncorrelated
            # with the label. That is what makes these conditions structural
            # nulls rather than label-keyed ones (RERUN_PLAN.md 3.1f), and a
            # side whose selection had leaked the label would show up here.
            for side, v in (('rust', r['spearman_label']),
                            ('python', p['spearman_label'])):
                checked += 1
                if v == v and abs(v) > rho_band:
                    bad.append(f'spearman_label: {side}={v:.4f}, past the '
                               f'{rho_band:.4f} that {n_units} independent '
                               f'selection unit(s) allow. This condition selects '
                               f'independently of the label, so its shape must '
                               f'not track it.')
        else:
            keys += ['min', 'max', 'spearman_label']
        for k in keys:
            checked += 1
            a, b = r.get(k), p.get(k)
            if k == 'flat':
                if bool(a) != bool(b):
                    bad.append(f'{k}: rust={a} python={b}')
            elif not close(a, b, args.tol):
                bad.append(f'{k}: rust={a} python={b}')
        mark = 'ok  ' if not bad else 'FAIL'
        note = ' (extremes are an independent draw)' if drawn else ''
        print(f"  {mark} {name:26} flat={str(r['flat']):5} "
              f"rho_label={r['spearman_label']!s:>8.8} rms={r['rms']:.6f}{note}")
        for b in bad:
            print(f"         {b}")
            failures.append(f'{name} {b}')

    print()
    for u in uncovered:
        print(f"  NOT COVERED  {u}")
    if failures:
        print(f"\nFAIL — {len(failures)} disagreement(s) over {checked} comparisons")
        return 1
    print(f"PASS — {checked} comparisons over {len(rust) - len(uncovered)} "
          f"conditions, both injectors agree on the shape column")
    return 0


if __name__ == '__main__':
    sys.exit(main())
