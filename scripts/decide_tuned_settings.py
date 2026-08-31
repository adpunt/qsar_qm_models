#!/usr/bin/env python3
"""Apply the author's decision rules to the finished search. One row per pairing.

    python scripts/decide_tuned_settings.py

Writes results/tuning_local/DECISIONS.md and prints the same thing.

THE RULES (author, 2026-08-31)
------------------------------
1. If the default beats the best drawn setting, keep the default. It is written
   out as the tuned value anyway, marked so, rather than left absent -- absent
   and "deliberately the default" look identical on disk otherwise.
2. A pairing with no significant extreme, beating the default, AND not much
   slower than the default is adopted.
3. Everything else needs a decision, and for each one this script looks for an
   alternative that is not significantly extreme, is no slower, and still beats
   the default.

WHAT "SIGNIFICANTLY EXTREME" MEANS HERE
---------------------------------------
Not every value at an end of its range is a problem. The ones that matter are
those sitting at the end that makes the model BIGGER or SLOWER, because those
are the ones that would keep going if the range were widened, and the ones that
cost compute on the full grid. A width at its SMALLEST option is at an end too,
and it is asking for a cheaper model -- there is nothing to worry about and
nothing to review.

So an extreme counts as significant only when it is at the expensive end.

COMPUTE
-------
Measured, not guessed: every fit recorded its own seconds, and each pairing's
winner is compared with that same pairing's default fit. The laptop was running
up to six searches at once, so a ratio near 1 is noise; a ratio of 5 or 20 is
not. The threshold is deliberately loose for that reason.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')
REPORT = os.path.join(OUT_DIR, 'DECISIONS.md')
SUPERSEDED = {'aborted_10000'}
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, 'models'))

# The end of the range that makes the model bigger or slower.
COST_AT_MAX = {
    'n_estimators', 'max_depth', 'num_leaves', 'max_features',
    'hidden_size', 'hidden_size1', 'hidden_size2', 'num_hidden_layers',
    'C', 'subsample', 'colsample_bytree', 'colsample_bylevel',
    'base_max_depth',
}
# Low values here mean deeper trees or more boosting rounds, so low is expensive.
COST_AT_MIN = {
    'min_samples_leaf', 'min_samples_split', 'min_child_samples',
    'min_child_weight',
}
# Boosters only: a small step size needs more rounds. Networks run a fixed
# number of epochs, so their learning rate costs nothing either way.
COST_AT_MIN_BOOSTERS = {'learning_rate'}
BOOSTERS = {'xgboost', 'lgb', 'ngboost'}

SLOW_ENOUGH_TO_REVIEW = 2.0     # times the default's fit
NO_SLOWER = 1.5                 # an alternative may cost this much more


def significant(params, sp, family):
    """The values at the expensive end of their range."""
    import analyse_tuned_settings as A
    out = []
    for p in A.edges_in(params, sp):
        v, opts = params[p], sp[p]
        try:
            lo, hi = min(opts), max(opts)
        except (TypeError, ValueError):
            continue
        at_max = v == hi or (isinstance(v, float) and v > (lo + hi) / 2)
        costly_high = p in COST_AT_MAX
        costly_low = p in COST_AT_MIN or (
            p in COST_AT_MIN_BOOSTERS and family in BOOSTERS)
        if (at_max and costly_high) or ((not at_max) and costly_low):
            out.append(p)
    return out


NOISE_LEVEL = 0.5      # the level the author judges the settings at


def noise_check():
    """R2 at noise level 0.5, tuned against default, per (dataset, model, rep).

    THE THIRD FILTER (author, 2026-08-31). The search scored every candidate on
    CLEAN labels, and a setting that wins there can lose badly once labels are
    noisy -- measured, LightGBM on ECFP4 ends at +0.226 where its own default
    holds +0.634. So a setting is not adopted on its clean score alone: it is
    refitted with noisy training labels and scored on a clean test split, and if
    it is worse than the default there it is flagged for discussion.

    Written by scripts/tuned_under_noise.py. Absent for pairings not yet tested.
    """
    out = {}
    for path in glob.glob(os.path.join(OUT_DIR, 'noise_vs_tuned_*.csv')):
        for r in csv.DictReader(open(path)):
            if r['status'] != 'ok':
                continue
            try:
                if abs(float(r['level']) - NOISE_LEVEL) > 1e-9:
                    continue
                ds = 'qm9'      # every noise run so far is QM9
                key = (ds, r['model'], r['rep'])
                out.setdefault(key, {})[r['setting']] = float(r['r2'])
            except (TypeError, ValueError):
                pass
    return out


def trial_rows():
    """(dataset, model, rep) -> [(setting label, r2, seconds)] for every good fit.

    Built once. An earlier version re-read every result file inside the pairing
    loop, which is the same work several thousand times over.
    """
    V = ('herg', 'logd', 'caco2')
    out = {}
    paths = (glob.glob(os.path.join(OUT_DIR, 'trials_*.csv'))
             + glob.glob(os.path.join(OUT_DIR, '*', 'trials_*.csv')))
    for path in sorted(paths):
        if os.path.basename(os.path.dirname(path)) in SUPERSEDED:
            continue
        name = os.path.basename(path)
        ds = next((v for v in V if v in name), 'qm9')
        for r in csv.DictReader(open(path)):
            if r['status'] != 'ok':
                continue
            try:
                out.setdefault((ds, r['model'], r['rep']), []).append(
                    (r['setting'], float(r['r2']), float(r['seconds'])))
            except (TypeError, ValueError):
                pass
    return out


def seconds_for(rows, score):
    """Seconds for the non-default fit that scored this, if it was recorded."""
    for lbl, r2, sec in rows:
        if lbl != 'default' and abs(r2 - score) < 1e-9:
            return sec
    return None


def main():
    import analyse_tuned_settings as A
    import report_tuning_results as RT
    import tune_hyperparameters as T
    import tuning_rosters as R

    trials = trial_rows()
    noise = noise_check()
    out = ['# The tuned settings, decided',
           '',
           'Regenerate with `python scripts/decide_tuned_settings.py`.',
           '',
           'A value counts as **significantly extreme** only when it sits at the '
           'end of its range that makes the model bigger or slower. A width at '
           'its smallest option is at an end, but it is asking for a cheaper '
           'model, so it is not flagged.',
           '',
           f'**Slower** means the winning fit took at least '
           f'{SLOW_ENOUGH_TO_REVIEW:g} times the default fit for the same '
           f'pairing. An alternative is accepted only if it beats the default, '
           f'has no significantly extreme value, and costs at most '
           f'{NO_SLOWER:g} times the default.',
           '']
    tally = {}
    for ds in ('qm9', 'logd', 'caco2', 'herg'):
        default, cands, sizes, ranges = RT.load(ds)
        if not cands:
            continue
        adopt, keep, review = [], [], []
        for k, c in sorted(cands.items()):
            fam = T.search_family(k[0], R) or k[0]
            sp = ranges.get(fam, {})
            if not sp:
                continue
            d = default.get(k)
            ranked = sorted(c, key=lambda t: -t[0])
            score, params = ranked[0]
            if d is None or not params:
                continue

            rows = trials.get((ds,) + k, [])
            base = next((sec for lbl, _r, sec in rows if lbl == 'default'), None)
            won = seconds_for(rows, score)
            ratio = (won / base) if (base and won and base > 0) else None

            sig = significant(params, sp, fam)
            slow = ratio is not None and ratio >= SLOW_ENOUGH_TO_REVIEW

            nz = noise.get((ds,) + k, {})
            tuned_at_noise = next((v for kk, v in nz.items()
                                   if kk != 'default'), None)
            worse_under_noise = (
                tuned_at_noise is not None and 'default' in nz
                and tuned_at_noise < nz['default'])

            if score <= d:
                keep.append((k, d, score, ratio))
            elif not sig and not slow and not worse_under_noise:
                adopt.append((k, d, score, ratio))
            else:
                alt = None
                for s2, p2 in ranked:
                    if s2 <= d or not p2 or significant(p2, sp, fam):
                        continue
                    a_sec = seconds_for(rows, s2)
                    a_ratio = (a_sec / base) if (base and a_sec and base > 0) \
                        else None
                    if a_ratio is not None and a_ratio > NO_SLOWER:
                        continue
                    alt = (s2, p2, a_ratio)
                    break
                review.append((k, d, score, ratio, sig, slow, alt,
                               worse_under_noise))

        tally[ds] = (len(adopt), len(keep), len(review))
        out.append(f'\n## {ds}\n')
        out.append(f'{len(adopt)} adopted, {len(keep)} keep the default, '
                   f'{len(review)} need a decision.\n')

        out.append('\n### Adopt — beats the default, nothing significantly '
                   f'extreme, not slower  ({len(adopt)})\n')
        out.append('| model | representation | default | tuned | times the '
                   'default fit |')
        out.append('|---|---|---|---|---|')
        for (m, rep), d, s, ratio in adopt:
            out.append(f'| {m} | {rep} | {d:.4f} | {s:.4f} | '
                       f'{"—" if ratio is None else f"{ratio:.2f}x"} |')

        out.append(f'\n### Keep the default — tuning lost  ({len(keep)})\n')
        out.append('| model | representation | default | best drawn | shortfall |')
        out.append('|---|---|---|---|---|')
        for (m, rep), d, s, ratio in keep:
            out.append(f'| {m} | {rep} | {d:.4f} | {s:.4f} | {s - d:+.4f} |')

        out.append(f'\n### Needs a decision  ({len(review)})\n')
        out.append('| model | representation | default | tuned | times the '
                   'default fit | why flagged | alternative | alternative beats '
                   'the default by | alternative cost |')
        out.append('|---|---|---|---|---|---|---|---|---|')
        for (m, rep), d, s, ratio, sig, slow, alt, wun in review:
            why = ', '.join(sig) if sig else ''
            if slow:
                why = (why + ', ' if why else '') + 'slow'
            if wun:
                why = (why + ', ' if why else '') + 'WORSE AT NOISE 0.5'
            if alt is None:
                a, gap, ac = 'none drawn', '—', '—'
            else:
                a = f'{alt[0]:.4f}'
                gap = f'{alt[0] - d:+.4f}'
                ac = '—' if alt[2] is None else f'{alt[2]:.2f}x'
            out.append(f'| {m} | {rep} | {d:.4f} | {s:.4f} | '
                       f'{"—" if ratio is None else f"{ratio:.2f}x"} | {why} | '
                       f'{a} | {gap} | {ac} |')
        out.append('')

    at = 7
    out.insert(at, '| dataset | adopt | keep the default | needs a decision |')
    out.insert(at + 1, '|---|---|---|---|')
    for i, (ds, (a, kk, r)) in enumerate(tally.items()):
        out.insert(at + 2 + i, f'| {ds} | {a} | {kk} | {r} |')
    out.insert(at + 2 + len(tally), '')

    text = '\n'.join(out) + '\n'
    with open(REPORT, 'w') as fh:
        fh.write(text)
    print(text)
    print(f'written to {REPORT}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
