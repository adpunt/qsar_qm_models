#!/usr/bin/env python3
"""There is exactly ONE reporting level, it lives in the shared spec, and it is unset.

WHY THIS TEST EXISTS
--------------------
Between 2026-08-20 and 2026-08-28 the reporting level -- the single noise level a
table quotes accuracy at -- was stated in more than thirty places across two state
documents, two scripts and paper.tex, on two different scales, with four different
values live at the same time. paper.tex says "R2 at sigma = 0.3"; that 0.3 is the
default argument of a figure-script function, on a scale that no longer exists, and
nobody ever chose it. The author, 2026-08-27: "there's clearly multiple sources of
information on it and you need to find all of them and rectify it. I'm really sick
of this coming up."

This test fails if a second source appears.

WHAT IT CHECKS
  1. models/model_defaults.py holds REPORTING_LEVELS and reporting_level().
  2. reporting_level() RAISES for any dataset whose level is unset, rather than
     returning a default. A default silently becomes the answer -- that is exactly
     how 0.3 got into the paper.
  3. Any level it does hold is a point on the ladder in NOISE_DESIGN.md 6.4.
  4. No other live script defines its own reporting level.
"""
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

FAILURES = []


def check(name, fn):
    try:
        fn()
        print(f"  PASS  {name}")
    except AssertionError as e:
        FAILURES.append(f"{name}: {e}")
        print(f"  FAIL  {name}: {e}")


def spec_is_the_source():
    import models.model_defaults as MD
    assert hasattr(MD, 'REPORTING_LEVELS'), \
        "models/model_defaults.py has no REPORTING_LEVELS; the one source of truth is gone"
    assert hasattr(MD, 'reporting_level'), \
        "models/model_defaults.py has no reporting_level(); callers will hardcode instead"
    assert MD.REPORTING_LEVEL_SCALE == 'fraction_of_clean_training_label_spread', \
        f"the scale changed to {MD.REPORTING_LEVEL_SCALE!r}. A level in log units is a "
    for ds in ('qm9', 'logd', 'caco2', 'herg'):
        assert ds in MD.REPORTING_LEVELS, f"{ds} is missing from REPORTING_LEVELS"


def unset_raises_rather_than_defaults():
    import models.model_defaults as MD
    for ds, value in MD.REPORTING_LEVELS.items():
        if value is not None:
            continue
        try:
            got = MD.reporting_level(ds)
        except ValueError:
            continue
        assert False, (f"reporting_level({ds!r}) returned {got!r} instead of raising. "
                       f"A default silently becomes the answer -- that is how "
                       f"'R2 at sigma = 0.3' got into paper.tex")


def any_level_set_is_on_the_ladder():
    import models.model_defaults as MD
    design = (REPO / 'NOISE_DESIGN.md').read_text()
    m = re.search(r'\*\*Levels\*\*\s*\|\s*\*\*([0-9.,\s]+)\*\*', design)
    assert m, "could not find the ladder in NOISE_DESIGN.md 6.4 to check against"
    ladder = {float(x) for x in m.group(1).replace(',', ' ').split()}
    for ds, value in MD.REPORTING_LEVELS.items():
        if value is None:
            continue
        assert value in ladder, (
            f"{ds}'s reporting level {value} is not on the ladder {sorted(ladder)}. "
            f"A level that does not run cannot be reported")


def no_second_source():
    """No live script may define its own reporting level."""
    allowed = {'models/model_defaults.py', 'scripts/test_one_reporting_level.py'}
    # setting_selection_test.py is a screening harness with its own declared level;
    # it is listed here deliberately so that removing this line makes the test fail.
    known_exception = 'scripts/setting_selection_test.py'
    offenders = []
    for path in sorted(list((REPO / 'scripts').glob('*.py'))
                       + list((REPO / 'models').glob('*.py'))):
        rel = str(path.relative_to(REPO))
        if rel in allowed:
            continue
        text = path.read_text()
        for i, line in enumerate(text.splitlines(), 1):
            if re.search(r'^\s*REPORTING_LEVEL\s*=', line):
                offenders.append(f"{rel}:{i}  {line.strip()}")
    unexpected = [o for o in offenders if not o.startswith(known_exception)]
    assert not unexpected, (
        "a second reporting level has appeared outside the shared spec:\n    "
        + "\n    ".join(unexpected)
        + "\n  Import it from models.model_defaults.reporting_level() instead.")
    assert any(o.startswith(known_exception) for o in offenders), (
        f"{known_exception} no longer declares its own level. If that is deliberate, "
        f"drop it from this test's exception list so the check stays honest")


if __name__ == '__main__':
    print("One reporting level, in the shared spec")
    check('the spec holds it', spec_is_the_source)
    check('an unset level raises rather than defaulting', unset_raises_rather_than_defaults)
    check('any level that is set is on the ladder', any_level_set_is_on_the_ladder)
    check('no second source outside the spec', no_second_source)
    if FAILURES:
        print(f"\nFAILED ({len(FAILURES)})")
        sys.exit(1)
    print("\nOK")
