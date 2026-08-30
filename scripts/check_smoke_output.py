#!/usr/bin/env python3
"""Do the smoke run's own files say the pipeline did what it claims?

    python scripts/check_smoke_output.py <directory>

The directory is the one stage 5 of `scripts/smoke_arc.sh` writes into: one
`smoke_<condition>.csv` per settled condition, each with its
`_uncertainty_values.csv` and `_noise_manifest.csv` beside it.

This is deliberately NOT a re-implementation of the analysis. It asks the small
number of questions that separate "the grid ran" from "the grid ran and the
numbers mean something", and every one of them is a failure mode this project
has actually had:

  1  every settled condition produced rows, and rows for the levels it sweeps
     -- a condition that quietly produces nothing is failure mode 9
  2  the clean level records EXACTLY zero delivered dose, not something small
     -- failure mode 2, the zero-noise control that was rounding error
  3  held-out labels are untouched: `y_true_original` at a given molecule is the
     same at every level -- the defect the whole re-run exists to undo
  4  accuracy falls as the level rises -- if it does not, the noise is not
     reaching the model
  5  the delivered dose is flat ACROSS conditions at a given level -- failure
     mode 5, six noise types that were one type at six doses
  6  every row names its condition, its scale and its support -- failure mode 12
  7  the uncertainty rows carry per-molecule injected noise on the splits whose
     labels were corrupted, and exactly zero on the test split
  8  a component the support table calls per-molecule actually varies within
     each fit, and one it calls constant does not -- failure mode 1, the
     constant column whose correlation is zero however good the model is
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from uncertainty_decomposition import (  # noqa: E402
    CONSTANT, NONE, PER_MOLECULE, support)


_CENSORING_LEVEL_SUFFIX = re.compile(r'^(censoring(?:_lower)?)_\d+$')


def _strip_level(name):
    """`censoring_25` -> `censoring`. The same rule both readers apply."""
    m = _CENSORING_LEVEL_SUFFIX.match(str(name))
    return m.group(1) if m else name


def _manifest_path(out, cond):
    for stem in (f'smoke_{cond}', cond):
        p = os.path.join(out, f'{stem}_noise_manifest.csv')
        if os.path.exists(p):
            return p
    return None


def _dose_band(row):
    """The percentage band this draw's own sample size allows, in per cent.

    Read from the injector's own rule rather than reimplemented: the two
    injectors agree on it by an executable gate, and a third copy here is how
    they drifted in the first place. Falls back to three standard errors under a
    Gaussian kurtosis when the shape is unknown.
    """
    n_eff = float(row.get('effective_n') or 0.0)
    if n_eff < 1:
        return 100.0
    # The manifest's `noise_shape` carries the setting: 'student_t_nu5', not
    # 'student_t'. Matched by prefix for that reason -- an exact test silently
    # fell through to the Gaussian band and failed a correct Student-t draw.
    shape = str(row.get('noise_shape', 'gaussian'))
    family = ('student_t' if shape.startswith('student_t')
              else 'laplace' if shape.startswith('laplace')
              else 'gaussian')
    nu = _nu_of(row)

    # Below five degrees of freedom the second moment has no ordinary sampling
    # law and the injector uses a rule of its own. Ask it rather than restating
    # it -- the two injectors agree on that rule by an executable gate, and a
    # third copy here is exactly how they drifted in the first place.
    if family == 'student_t' and nu is not None and nu <= 4.0:
        try:
            from noiseInject import dose_tolerance
            return 100.0 * float(dose_tolerance(np.zeros(1), n_eff, nu=nu))
        except ImportError:
            return 100.0

    # The ordinary branch is three standard errors of the second moment,
    # sqrt((kurtosis - 1) / (4 n_eff)). The injector takes the kurtosis from the
    # SAMPLE; the manifest does not carry the drawn residuals, so it is taken
    # from the shape's population value instead:
    #
    #   gaussian   3
    #   laplace    6
    #   student_t  3 + 6/(nu - 4), which is 9 at the settled nu = 5
    #
    # Using 3 for every shape is what the first version of this file did, and it
    # failed a correct Student-t draw at 3.77% against a band of 3.35% when the
    # right band is 6.71%.
    if family == 'student_t' and nu is not None and nu > 4.0:
        kurt = 3.0 + 6.0 / (nu - 4.0)
    else:
        kurt = {'laplace': 6.0}.get(family, 3.0)
    kurt = min(kurt, 60.0)          # the same clamp the injector applies
    return 100.0 * max(3.0 * float(np.sqrt((kurt - 1.0) / (4.0 * n_eff))), 0.005)


def _nu_of(row):
    """The tail weight this row was drawn with.

    The manifest does not carry it, so it comes from the settled file, which is
    the one place the single-setting decisions live (noise_conditions.json,
    settled 2026-08-27). The condition name is the fallback for a hand-run
    invocation that used something else.
    """
    name = f"{row.get('noise_type', '')} {row.get('noise_shape', '')}"
    m = re.search(r'_nu(\d+)\b', name)
    if m:
        return float(m.group(1))
    try:
        doc = json.loads(open(os.path.join(ROOT, 'noise_conditions.json')).read())
        return float(doc['settings_that_follow']['nu'])
    except Exception:
        return None


FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def ok(msg, detail=''):
    print(f"  ok    {msg}" + (f"  [{detail}]" if detail else ''))


def settled_conditions():
    doc = json.loads(open(os.path.join(ROOT, 'noise_conditions.json')).read())
    return ([c['name'] for c in doc['stage_1_full_grid']]
            + [c['name'] for c in doc['stage_2_depth_only']])


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, 'results', 'smoke_arc')
    want = settled_conditions()
    print(f"\nthe smoke run in {out}")
    print(f"settled conditions: {', '.join(want)}\n")

    # --- 1. every condition produced rows -----------------------------------
    print("1. every settled condition produced rows")
    frames = {}
    for cond in want:
        # accept both the stage-5 name and a bare <condition>.csv
        hits = [p for p in (os.path.join(out, f'smoke_{cond}.csv'),
                            os.path.join(out, f'{cond}.csv')) if os.path.exists(p)]
        if not hits:
            fail(f"{cond}: no results file. A condition that produces no rows is "
                 f"indistinguishable from one nobody ran.")
            continue
        df = pd.read_csv(hits[0])
        if df.empty:
            fail(f"{cond}: the file exists and is empty")
            continue
        frames[cond] = df
        ok(f"{cond}", f"{len(df)} row(s), levels {sorted(df['sigma'].unique())}")
    if not frames:
        print("\nnothing to check.")
        return 1

    # --- 2. the clean level is EXACTLY zero ---------------------------------
    print("\n2. the clean level records exactly zero delivered dose")
    for cond, df in frames.items():
        zero = df[df['sigma'] == 0.0]
        if zero.empty:
            print(f"  note  {cond} runs no clean level (only the reference and "
                  f"censoring do; copy_zero_rows.py supplies the rest)")
            continue
        worst = float(np.max(np.abs(zero['delivered_dose'].astype(float))))
        if worst != 0.0:
            fail(f"{cond}: the clean level delivered {worst!r}, not exactly 0. "
                 f"A zero control that is 'small' is rounding error, and rounding "
                 f"error grows with the label -- which is where uncertainty is "
                 f"largest.")
        else:
            ok(f"{cond}: clean level delivers exactly 0")

    # --- 3. held-out labels are untouched -----------------------------------
    print("\n3. held-out labels are the same at every level")
    for cond in frames:
        unc = _unc_path(out, cond)
        if not unc:
            continue
        u = pd.read_csv(unc)
        test = u[u['split'] == 'test']
        if test.empty or test['sigma'].nunique() < 2:
            print(f"  note  {cond}: fewer than two levels of test rows to compare")
            continue
        piv = test.pivot_table(index='sample_idx', columns='sigma',
                               values='y_true_original', aggfunc='first')
        spread = float(np.nanmax(piv.max(axis=1) - piv.min(axis=1)))
        if spread > 1e-9:
            fail(f"{cond}: a test molecule's true label moves by {spread:.6g} "
                 f"between noise levels. Held-out labels must be clean; this is "
                 f"the defect the whole re-run exists to undo.")
        else:
            ok(f"{cond}: test labels bit-identical across "
               f"{test['sigma'].nunique()} levels")

    # --- 4. accuracy falls as the level rises -------------------------------
    print("\n4. accuracy falls as the noise level rises")
    for cond, df in frames.items():
        curve = df.groupby('sigma')['r2'].mean().sort_index()
        if len(curve) < 3:
            print(f"  note  {cond}: {len(curve)} level(s), too few to read a curve")
            continue
        drop = float(curve.iloc[0] - curve.iloc[-1])
        rho = float(pd.Series(curve.index).corr(pd.Series(curve.values),
                                                method='spearman'))
        line = '  '.join(f"{k:g}:{v:.3f}" for k, v in curve.items())
        if drop <= 0:
            fail(f"{cond}: R2 does not fall from the lowest level to the highest "
                 f"({curve.iloc[0]:.4f} -> {curve.iloc[-1]:.4f}). Either the noise "
                 f"is not reaching the model, or it is reaching the held-out "
                 f"labels too. {line}")
        else:
            ok(f"{cond}: R2 falls {drop:+.4f}, rank correlation with level "
               f"{rho:+.2f}", line)

    # --- 5. each condition delivered what it asked for ----------------------
    #
    # NOT "the conditions agree with each other to X%". This is ONE replicate,
    # and a single draw's delivered amount wobbles: measured on the same 5,000
    # labels over 200 seeds, the per-run spread runs from 0.98% (Gaussian) to
    # 5.27% (grouped-shifted), so six independent single draws routinely differ
    # by several per cent for no reason but sampling. Checking them against each
    # other at a fixed few per cent fails correct code -- which is what the first
    # version of this file did, on all six levels.
    #
    # The flat-across-conditions claim is about the MEAN over many seeds and is
    # what the preflight self-test measures (200 seeds, spread 0.44% on real QM9).
    # One replicate cannot test it, and pretending otherwise is how a smoke test
    # comes to cry wolf.
    #
    # What one replicate CAN test is the thing that matters per run: each
    # condition delivered the amount it was asked for, inside the band its own
    # manifest says its sample size allows.
    print("\n5. each condition delivered the amount it asked for")
    for cond in frames:
        man = _manifest_path(out, cond)
        if not man:
            print(f"  note  {cond}: no manifest beside the results")
            continue
        m = pd.read_csv(man)
        if 'target_dose_in_label_units' not in m.columns:
            print(f"  note  {cond}: the manifest predates the target column")
            continue
        rows = m[m['target_dose_in_label_units'].astype(float) > 0]
        if rows.empty:
            print(f"  note  {cond}: nothing but the clean level to check")
            continue
        worst, worst_lvl, band_at_worst = 0.0, None, 0.0
        for _, r in rows.iterrows():
            target = float(r['target_dose_in_label_units'])
            got = float(r['delivered_dose_in_label_units'])
            err = 100.0 * (got - target) / target
            band = _dose_band(r)
            if abs(err) > abs(worst):
                worst, worst_lvl, band_at_worst = err, float(r['noise_level']), band
        if abs(worst) > band_at_worst:
            fail(f"{cond}: at level {worst_lvl:g} it delivered {worst:+.2f}% of "
                 f"the amount asked for, outside the {band_at_worst:.2f}% band "
                 f"its own effective sample size allows.")
        else:
            ok(f"{cond}: every level inside its band",
               f"worst {worst:+.2f}% at level {worst_lvl:g}, band "
               f"{band_at_worst:.2f}%")

    # And the flat-across-conditions property, REPORTED rather than asserted,
    # so the numbers are on the record without one replicate being asked to
    # prove a many-seed claim.
    dose = {}
    for cond, df in frames.items():
        if cond == 'censoring':
            continue          # not dose-matched; swept on its own axis
        for lvl, g in df.groupby('sigma'):
            if lvl == 0.0:
                continue
            dose.setdefault(float(lvl), {})[cond] = float(g['delivered_dose'].mean())
    for lvl in sorted(dose):
        vals = dose[lvl]
        if len(vals) < 2:
            continue
        lo, hi = min(vals.values()), max(vals.values())
        spread = 100.0 * (hi - lo) / ((hi + lo) / 2)
        print(f"  note  level {lvl:g}: the six dose-matched conditions span "
              f"{spread:.2f}% in this single replicate  ("
              + '  '.join(f"{c}:{v:.4f}" for c, v in sorted(vals.items())) + ")")

    # --- 6. every row names what produced it --------------------------------
    print("\n6. every row names its condition, its units and its provenance")
    need = ['noise_type', 'level_units', 'delivered_dose', 'params_source',
            'spec_hash', 'standardisation_mean', 'standardisation_sd']
    for cond, df in frames.items():
        missing = [c for c in need if c not in df.columns]
        if missing:
            fail(f"{cond}: result rows carry no {missing}")
            continue
        # Censoring's name carries the clipped percentage -- `censoring_25` --
        # because the QM9 injector writes it that way, and both readers strip it
        # (`_strip_censoring_level` in generate_paper_figures_v2.py,
        # `_normalise_condition` in uncertainty_stats.py). So the test is that
        # every row normalises to THIS condition, not that it equals it.
        named = set(df['noise_type'].astype(str).map(_strip_level))
        wrong = sorted(named - {cond})
        if wrong:
            fail(f"{cond}: rows in this file name condition(s) {wrong}")
        else:
            ok(f"{cond}: every row names its condition",
               f"as {sorted(set(df['noise_type'].astype(str)))[0]!r}, "
               f"units={df['level_units'].iloc[0]}, "
               f"params={df['params_source'].iloc[0]}, "
               f"spec={df['spec_hash'].iloc[0]}")

    # --- 7 & 8. the uncertainty rows ----------------------------------------
    print("\n7. the uncertainty rows carry the noise that was actually injected")
    print("8. a per-molecule component varies within each fit; a constant one does not")
    for cond in frames:
        path = _unc_path(out, cond)
        if not path:
            print(f"  note  {cond}: no uncertainty file (this model emits none, "
                  f"or the pair is not a settled one)")
            continue
        u = pd.read_csv(path)
        _check_uncertainty(cond, u)

    print()
    if FAILURES:
        print(f"FAIL — {len(FAILURES)} check(s) failed")
        return 1
    print("PASS — the smoke run's own files say the pipeline did what it claims")
    return 0


def _unc_path(out, cond):
    for stem in (f'smoke_{cond}', cond):
        p = os.path.join(out, f'{stem}_uncertainty_values.csv')
        if os.path.exists(p):
            return p
    return None


def _check_uncertainty(cond, u):
    model = str(u['model'].iloc[0])
    alea_kind, epis_kind = support(model)

    # 7a. test rows carry no injected noise, by construction.
    test = u[u['split'] == 'test']
    if not test.empty:
        worst = float(np.nanmax(np.abs(test['injected_noise'].astype(float))))
        if worst != 0.0:
            fail(f"{cond}: a TEST row carries injected noise {worst:.6g}. Test "
                 f"labels are never corrupted, so this is the held-out-noise "
                 f"defect back again.")
        else:
            ok(f"{cond}: {len(test)} test row(s), injected noise exactly 0")

    # 7b. the corrupted splits carry real per-molecule noise above level 0.
    for split in ('train_oof', 'validation'):
        s = u[(u['split'] == split) & (u['sigma'] > 0)]
        if s.empty:
            print(f"  note  {cond}: no {split} rows above the clean level")
            continue
        var = float(np.nanstd(s['injected_noise'].astype(float)))
        if not np.isfinite(var) or var == 0.0:
            fail(f"{cond}: {split} rows above the clean level carry a CONSTANT "
                 f"injected noise. It must be the per-molecule amount the "
                 f"injector recorded, never a level broadcast onto every row.")
        else:
            ok(f"{cond}: {len(s)} {split} row(s), injected noise varies per "
               f"molecule", f"sd {var:.4f}")

    # 7c. the level-free shape is present where the condition has one.
    if 'noise_pattern' in u.columns:
        pat = u.loc[u['sigma'] > 0, 'noise_pattern'].astype(float)
        if len(pat) and np.isfinite(pat).any():
            spread = float(np.nanstd(pat))
            ok(f"{cond}: the level-free shape is on the row", f"sd {spread:.4f}")

    # 8. the support columns, checked against what is in the column.
    for kind, col, name in ((alea_kind, 'aleatoric_uncertainty', 'aleatoric'),
                            (epis_kind, 'epistemic_uncertainty', 'epistemic')):
        declared = u.get(f'{name}_support')
        if declared is not None and declared.notna().any():
            said = sorted(set(declared.dropna().astype(str)))
            if said != [kind]:
                fail(f"{cond}: the {name}_support column says {said} and the "
                     f"support table says {kind!r}")
        if kind == NONE:
            if col in u.columns and u[col].notna().any():
                fail(f"{cond}: {model} has no {name} term but the column is "
                     f"populated")
            else:
                ok(f"{cond}: {model} has no {name} term, and the column is blank")
            continue
        if col not in u.columns or not u[col].notna().any():
            fail(f"{cond}: {model}'s {name} column is empty, which is "
                 f"indistinguishable from a model that cannot produce one")
            continue
        # WITHIN each fit. Out of fold the column is several fits stacked.
        blocks = (u['oof_folds_ok'] if 'oof_folds_ok' in u.columns
                  else pd.Series(0, index=u.index))
        key = list(zip(u['sigma'], u['split'], blocks.fillna(-99)))
        u = u.assign(_fit=pd.Series(key, index=u.index).astype(str))
        widths = []
        for _fit, part in u.groupby('_fit')[col]:
            p = part.dropna()
            if len(p) > 1:
                widths.append(float(p.max() - p.min()))
        if not widths:
            print(f"  note  {cond}: no fit with more than one {name} value")
            continue
        if kind == PER_MOLECULE and min(widths) <= 1e-12:
            fail(f"{cond}: {model}'s {name} term is called per molecule but is "
                 f"CONSTANT within at least one fit. A constant correlates with "
                 f"per-molecule noise at exactly zero however good the model is.")
        elif kind == CONSTANT and max(widths) > 1e-12:
            fail(f"{cond}: {model}'s {name} term is called one number per fit but "
                 f"varies by {max(widths):.6g} within a fit. Update the support "
                 f"table -- a term that really varies is worth more, not less.")
        else:
            ok(f"{cond}: {model}'s {name} term is {kind} as declared",
               f"{len(widths)} fit(s), within-fit width "
               f"{min(widths):.3g}..{max(widths):.3g}")


if __name__ == '__main__':
    sys.exit(main())
