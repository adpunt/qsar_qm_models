#!/usr/bin/env python3
"""The generated .tex fragments have to compile when they are pasted.

WHY THIS EXISTS
---------------
`figlib_tables.write` called `to_latex(escape=False)`, which is right -- the
table text carries deliberate LaTeX like `AUC$_{norm}$`, `Student-$t$ ($\\nu$=5)`
and `hERG K$_i$`, and escaping those would print the source instead of setting
it. But nothing then escaped the characters that are NOT deliberate. On the
17 September harvest, of sixteen generated fragments, NINE could not compile at
all and FOURTEEN carried Unicode pdflatex cannot set. Only T1 was clean:

  * `Sort & Slice` -- the `&` is a column separator, so that row carried one
    more field than the tabular declared. The header row of T5, four rows of T8
    and 48 rows of T6.
  * `Outlier (10%)` -- `%` starts a LaTeX comment, so it ate the rest of its
    line INCLUDING the row's own `\\\\`. T2, T3 and all four T4s.
  * `R²`, `α`, `β`, `η`, `ρ`, `Δ`, `±`, `÷`, `→`, `–` -- "Unicode character not
    set up for use with LaTeX" under pdflatex.

None of that shows up in the CSV, which is what everything else reads, so it
survived every check until someone pasted a table.

WHAT IT CHECKS
--------------
1. `latex_safe` escapes outside math and leaves math alone.
2. `write` produces a fragment whose every row has exactly as many fields as the
    tabular declares, and no unescaped `%`, on a frame built to contain all four
    hazards.
3. If `results/decisions_arc/tables/` is on disk, every `.tex` beside a `.csv`
    is audited the same way. Skipped, loudly, when the results are not there.

    python scripts/test_table_latex.py
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_tables as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / 'results' / 'decisions_arc' / 'tables'

#: A `%` anywhere that is not already escaped comments out the rest of the line.
_BARE_PERCENT = re.compile(r'(?<!\\)%')


def rows_and_columns(text):
    """(declared column count, [fields in each body row]) for one fragment."""
    lines = [line.rstrip('\n') for line in text.split('\n')
             if not line.startswith('%')]
    spec = re.search(r'\{tabular\}\{([^}]*)\}',
                     next(l for l in lines if l.startswith(r'\begin{tabular}')))
    declared = len([c for c in spec.group(1) if c in 'lcr'])
    body = [l.strip() for l in lines if l.strip().endswith(r'\\')]
    return declared, body


def audit(text, where):
    """Every complaint about one fragment, as a list of sentences."""
    problems = []
    declared, body = rows_and_columns(text)
    if not body:
        return [f'{where}: no rows ending in a row terminator']
    for row in body:
        fields = row.count('&') + 1 - row.count(r'\&')
        if fields != declared:
            problems.append(
                f'{where}: a row carries {fields} field(s) where the tabular '
                f'declares {declared} -- {row[:80]}')
            break
    for line in text.split('\n'):
        if line.startswith('%'):
            continue
        if _BARE_PERCENT.search(line):
            problems.append(
                f'{where}: an unescaped % comments out the rest of its line '
                f'-- {line[:80]}')
            break
    for character in T._LATEX_UNICODE:
        if character in '\n'.join(l for l in text.split('\n')
                                  if not l.startswith('%')):
            problems.append(
                f'{where}: {character!r} is not set up for use with LaTeX')
            break
    return problems


def check_latex_safe():
    """Escaping outside math, and nothing touched inside it."""
    expected = {
        'Sort & Slice': r'Sort \& Slice',
        'Outlier (10%)': r'Outlier (10\%)',
        'Clean R²': 'Clean R$^2$',
        'hERG Kᵢ': 'hERG K$_i$',
        # Already math: returned character for character.
        'Student-$t$ ($\\nu$=5)': 'Student-$t$ ($\\nu$=5)',
        'AUC$_{norm}$': 'AUC$_{norm}$',
        '$\\eta^2$ model': '$\\eta^2$ model',
        'NN-α → BNN-α':
            'NN-$\\alpha$ $\\rightarrow$ BNN-$\\alpha$',
        '49.5 ± 0.6': '49.5 $\\pm$ 0.6',
    }
    for raw, want in expected.items():
        got = T.latex_safe(raw)
        assert got == want, f'latex_safe({raw!r}) gave {got!r}, wanted {want!r}'
    print(f'  latex_safe: {len(expected)} case(s), math left alone')


def check_write_round_trip():
    """A frame carrying all four hazards writes a fragment that survives audit."""
    table = pd.DataFrame([
        {'Model': 'NN-α → BNN-α', 'Representation': 'Sort & Slice',
         'Condition': 'Outlier (10%)', 'Clean R²': 0.832,
         'AUC$_{norm}$': 0.963, 'η² model': '49.5 ± 0.6'},
        {'Model': 'GP (het.)', 'Representation': 'ECFP4',
         'Condition': 'Student-$t$ ($\\nu$=5)', 'Clean R²': 0.839,
         'AUC$_{norm}$': 0.940, 'η² model': '30.5 ± 4.0'},
    ])
    with tempfile.TemporaryDirectory() as tmp:
        T.write(table, tmp, 'TX_hazards', caption='100% of the hazards & then some')
        text = (Path(tmp) / 'TX_hazards.tex').read_text()
    problems = audit(text, 'TX_hazards')
    assert not problems, '\n'.join(problems)
    assert r'Sort \& Slice' in text, 'the ampersand was not escaped'
    assert r'Outlier (10\%)' in text, 'the percent was not escaped'
    assert 'AUC$_{norm}$' in text, 'the deliberate math was mangled'
    print('  write: a frame with all four hazards produces a clean fragment')


def check_on_disk():
    """Every generated fragment, if the results are here."""
    if not TABLES.is_dir():
        print(f'  on disk: {TABLES} is not here, so the generated fragments '
              f'are NOT checked. This runs after a harvest.')
        return
    fragments = sorted(TABLES.glob('*.tex'))
    if not fragments:
        print(f'  on disk: no .tex in {TABLES}')
        return
    problems = []
    for path in fragments:
        problems += audit(path.read_text(encoding='utf-8'), path.name)
    if problems:
        raise AssertionError(
            f'{len(problems)} generated fragment(s) will not compile. Re-run '
            f'scripts/run_paper_analysis.py --only tables to rewrite them:\n  '
            + '\n  '.join(problems))
    print(f'  on disk: {len(fragments)} fragment(s), all compile-clean')


def main():
    print(__doc__.split('\n')[0])
    check_latex_safe()
    check_write_round_trip()
    check_on_disk()
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
