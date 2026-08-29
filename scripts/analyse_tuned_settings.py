#!/usr/bin/env python3
"""Is the winning setting sensible, or is it pressed against the edge of the box?

    python scripts/analyse_tuned_settings.py

WHAT THIS ANSWERS
-----------------
A random search can only return a setting it was allowed to try. If the winner
sits at the smallest or largest value the search was allowed to draw, that is
evidence the real optimum is OUTSIDE the range and the range was drawn too
narrow. The winning score is then not wrong, but it is a floor rather than an
answer, and widening the range would be expected to beat it.

The opposite failure matters too: a winner in the middle of every range, beating
the default by a hair, is a winner that is probably noise.

So this reports, for every pairing that has produced a winner:

  * each tuned value, next to the range the search could draw from;
  * whether that value is AT an edge, which is the warning;
  * how many settings were actually tried, since a winner from three tries is
    weaker evidence than a winner from twelve;
  * the gain over the shared default, so a hair-thin win is visible as one.

It reads the search's own definitions rather than restating them, so the ranges
here cannot drift from the ranges that were sampled.

WHAT IT DOES NOT DO
-------------------
It does not decide anything. A setting at an edge may be perfectly correct --
more trees is usually better and the largest value winning is unremarkable for
that one. The judgement is the author's; this puts the evidence in front of them.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
OUT_DIR = os.path.join(_ROOT, 'results', 'tuning_local')

# NOT EVERY EDGE IS A WARNING, and reporting all of them equally buries the ones
# that matter.
#
# At the TOP: more trees or more leaves is almost always weakly better, so the
# search reaching the ceiling says nothing about the ceiling.
BENIGN_AT_MAX = {'n_estimators', 'num_leaves'}
# At the BOTTOM: these are natural floors, not choices the range imposed. A leaf
# cannot hold fewer than one molecule and a split cannot be made on fewer than
# two, so winning there means "grow the tree out", which is a real answer rather
# than a range that was drawn too small.
BENIGN_AT_MIN = {'min_samples_leaf', 'min_samples_split', 'gamma',
                 'reg_alpha', 'reg_lambda'}
# A numeric maximum that was offered ALONGSIDE an unbounded option is not an
# edge in any useful sense: the search could have chosen no limit at all and
# did not.
UNBOUNDED_ALTERNATIVE = {'max_depth': ('None', '-1')}


def observed_ranges(n=4000):
    """What each search space can actually produce, by sampling it.

    The spaces are callables, so the only honest way to know their range is to
    draw from them. Sampling rather than restating means this cannot disagree
    with what the sweep drew.
    """
    sys.path.insert(0, _HERE)
    import tune_hyperparameters as tuner

    rng = random.Random(0)
    out = {}
    for key, space in tuner.SEARCH_SPACES.items():
        seen = defaultdict(list)
        for _ in range(n):
            for name, draw in space.items():
                seen[name].append(draw(rng))
        out[key] = {name: vals for name, vals in seen.items()}
    return out


def describe(values):
    """A readable range, and the set of edge values.

    Three shapes turn up and they need different descriptions. A handful of
    repeated values is a menu. Many distinct numbers is a continuous range. A
    MIXTURE of the two -- which the support vector machine's kernel width is,
    since it draws either a named setting or a number -- has to be described as
    both, or the printout becomes thousands of sampled floats.
    """
    numeric = sorted({v for v in values
                      if isinstance(v, (int, float)) and not isinstance(v, bool)})
    other = sorted({str(v) for v in values
                    if not isinstance(v, (int, float)) or isinstance(v, bool)})
    fmt = lambda x: f'{x:g}'

    parts, edges = [], set()
    if numeric:
        if len(numeric) <= 8:
            parts.append('{' + ', '.join(fmt(v) for v in numeric) + '}')
        else:
            parts.append(f'{fmt(numeric[0])} to {fmt(numeric[-1])}')
        edges = {numeric[0], numeric[-1]}
    if other:
        parts.append('{' + ', '.join(other) + '}')
    return ' or '.join(parts), edges



# A CONTINUOUS SETTING IS NEVER EXACTLY ON ITS BOUNDARY.
#
# Dropout and the learning rate are drawn from a range rather than a list, so a
# draw lands on the exact minimum or maximum essentially never -- which meant the
# edge test could not fire on them AT ALL, and reported settings sitting at 97%
# of their range as comfortably interior. Anything in the outer tenth of a
# continuous range counts. Measured in log space where the range was drawn in log
# space, or the test is meaningless for a learning rate.
CONTINUOUS_TAIL = 0.10
LOG_SCALE = {'lr', 'learning_rate', 'C', 'gamma', 'likelihood_noise',
             'outputscale'}


def _position(name, value, values):
    """Where a value sits in its range, 0 at the bottom and 1 at the top."""
    import math
    lo, hi = min(values), max(values)
    if hi <= lo:
        return 0.5
    if name in LOG_SCALE and lo > 0:
        return (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    return (value - lo) / (hi - lo)


def edges_in(params, space):
    """Which of a setting's values sit at the edge of what could be drawn."""
    hit = []
    for name, value in params.items():
        if name not in space:
            continue
        text, edges = describe(space[name])
        if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
            continue
        numeric = [v for v in space[name]
                   if isinstance(v, (int, float)) and not isinstance(v, bool)]
        continuous = len(set(numeric)) > 8
        if continuous:
            pos = _position(name, value, numeric)
            if CONTINUOUS_TAIL < pos < 1 - CONTINUOUS_TAIL:
                continue
            is_max = pos > 0.5
        else:
            if value not in edges:
                continue
            is_max = value == max(edges)
        if is_max and name in BENIGN_AT_MAX:
            continue
        if is_max and any(alt in text for alt in
                          UNBOUNDED_ALTERNATIVE.get(name, ())):
            continue
        if not is_max and name in BENIGN_AT_MIN:
            continue
        hit.append(name)
    return hit


def report_alternatives(candidates, default, ranges, rosters):
    """Where the winner is at an edge, what is the best setting that is not?

    A setting whose values all sit inside their ranges is one the search chose on
    the data rather than because the box stopped it. If such a setting scores
    close to an edge-pinned winner, it is the better thing to adopt: the same
    evidence, without the range doing the deciding.
    """
    print('\n' + '=' * 70)
    print('CLEARED — the winner sits inside every range')
    print('=' * 70)
    cleared, constrained = [], []
    for pair, cands in sorted(candidates.items()):
        if not cands:
            continue
        space = ranges.get(rosters.TUNED_KEY.get(pair[0], pair[0]), {})
        ranked = sorted(cands, key=lambda t: -t[0])
        top_score, top_params = ranked[0]
        d = default.get(pair)
        gain = None if d is None else top_score - d
        if not edges_in(top_params, space):
            cleared.append((pair, top_score, gain, len(cands)))
        else:
            constrained.append((pair, ranked, space, d))

    if not cleared:
        print('  none yet')
    for (model, rep), score, gain, n in cleared:
        g = '?' if gain is None else f'{gain:+.4f}'
        verdict = 'beats the default' if (gain or 0) > 0 else 'DEFAULT STILL WINS'
        print(f'  {model:26s} {rep:10s} best {score:+.4f}  '
              f'{g:>8s} vs default   {n:2d} tried   {verdict}')

    print('\n' + '=' * 70)
    print('CONSTRAINED — winner at an edge, and the best interior alternative')
    print('=' * 70)
    if not constrained:
        print('  none')
    for (model, rep), ranked, space, d in constrained:
        top_score, top_params = ranked[0]
        interior = [(sc, pr) for sc, pr in ranked if not edges_in(pr, space)]
        head = (f'  {model} x {rep}: winner {top_score:+.4f} '
                f'(at edge: {", ".join(edges_in(top_params, space))})')
        if not interior:
            print(head + '  — no interior setting was tried at all')
            continue
        alt_score, alt_params = interior[0]
        cost = top_score - alt_score
        vs_default = '' if d is None else f', {alt_score - d:+.4f} vs default'
        print(head)
        print(f'      best interior {alt_score:+.4f} '
              f'(costs {cost:.4f} against the winner{vs_default})')
        print(f'      {json.dumps(alt_params, sort_keys=True)}')


def main():
    sys.path.insert(0, os.path.join(_ROOT, 'models'))
    import tuning_rosters as rosters

    rows = []
    for path in sorted(glob.glob(os.path.join(OUT_DIR, 'trials_*.csv'))):
        with open(path) as fh:
            rows += [r for r in csv.DictReader(fh)]
    if not rows:
        print(f'no trials_*.csv under {OUT_DIR} yet.')
        return 0

    # REFUSE TO MIX SAMPLE SIZES. Runs at different numbers of molecules are not
    # comparable, and reading them together silently produced counts like "82 of
    # 72 settings" -- results from a run that was killed at 10,000 molecules
    # sitting alongside the live one at 5,000. Nothing warned; the numbers simply
    # blended.
    sizes = {r['sample_size'] for r in rows if r.get('sample_size')}
    if len(sizes) > 1:
        print(f'REFUSING to read these together: the files disagree on how many '
              f'molecules were used ({sorted(sizes)}). Settings chosen on '
              f'different amounts of data cannot be compared. Move the stale '
              f'files out of {OUT_DIR} and run again.')
        raise SystemExit(1)

    ok = [r for r in rows if r['status'] == 'ok']
    screened = {(r['model'], r['rep']) for r in ok
                if r['setting'].startswith('final_')}

    # EVERY scored setting is kept, not only the winner. A winner pressed against
    # the edge of its range is a poor thing to adopt; the question that follows
    # is always "what was the best setting that was NOT at an edge, and how much
    # worse was it?" -- and that cannot be answered from the winner alone.
    best, default, tried = {}, {}, defaultdict(int)
    candidates = defaultdict(list)
    for r in ok:
        pair = (r['model'], r['rep'])
        if r['setting'] == 'default':
            default[pair] = float(r['r2'])
            continue
        if r['setting'].startswith('screen_') and pair in screened:
            tried[pair] += 1
            continue
        tried[pair] += 1
        score = float(r['r2'])
        params = json.loads(r['detail']) if r['detail'] else {}
        candidates[pair].append((score, params))
        if pair not in best or score > best[pair][0]:
            best[pair] = (score, params)

    ranges = observed_ranges()
    edge_hits, thin_wins, n_pairs = [], [], 0

    for pair in sorted(best):
        model, rep = pair
        score, params = best[pair]
        d = default.get(pair)
        if not params:
            continue
        n_pairs += 1
        gain = None if d is None else score - d
        space = ranges.get(rosters.TUNED_KEY.get(model, model), {})

        flag = ''
        if gain is not None and 0 < gain < 0.005:
            thin_wins.append((model, rep, gain))
            flag = '   (wins by less than 0.005)'
        print(f'\n{model} x {rep} — best {score:+.4f}, default '
              f'{"?" if d is None else f"{d:+.4f}"}, '
              f'{"?" if gain is None else f"{gain:+.4f}"} '
              f'from {tried[pair]} setting(s){flag}')

        for name, value in sorted(params.items()):
            if name not in space:
                print(f'    {name:20s} {str(value):>14s}   (pinned, not searched)')
                continue
            text, edges = describe(space[name])
            at_edge = name in edges_in({name: value}, space)
            note = ''
            if at_edge:
                is_max = value == max(e for e in edges)
                unbounded = any(alt in text for alt
                                in UNBOUNDED_ALTERNATIVE.get(name, ()))
                if is_max and name in BENIGN_AT_MAX:
                    note = '   <- at the top (more is always weakly better)'
                elif is_max and unbounded:
                    note = '   <- at the top, but no limit at all was on offer'
                elif not is_max and name in BENIGN_AT_MIN:
                    note = '   <- at the floor (this is the natural floor)'
                else:
                    note = '   <- AT THE EDGE OF THE RANGE'
                    edge_hits.append((model, rep, name, value, text))
            print(f'    {name:20s} {str(value):>14s}   range {text}{note}')

    report_alternatives(candidates, default, ranges, rosters)

    print('\n' + '=' * 70)
    print(f'{n_pairs} pairing(s) with a winner so far')
    if edge_hits:
        print(f'\n{len(edge_hits)} value(s) sit at the edge of what the search '
              f'could draw. Where that happens the range is the binding '
              f'constraint, not the data:')
        for model, rep, name, value, text in edge_hits:
            print(f'    {model} x {rep}: {name} = {value}, range {text}')
    else:
        print('\nNo tuned value is pressed against the edge of its range.')
    if thin_wins:
        print(f'\n{len(thin_wins)} winner(s) beat the default by less than '
              f'0.005, which one split cannot distinguish from luck:')
        for model, rep, gain in thin_wins:
            print(f'    {model} x {rep}: {gain:+.5f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
