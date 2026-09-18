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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import figlib_tables as T  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
def _tables_dir():
    """The newest harvest's tables directory, whatever it is called.

    `run_paper_analysis.py` writes to a dated directory, and this was pinned to
    the undated `results/decisions_arc/tables`, which no longer exists. The
    check then printed "not here, so the generated fragments are NOT checked"
    and passed, so the one test that catches an uncompilable fragment had
    quietly stopped testing anything. Take the newest directory that has a
    `tables` in it, and prefer `tables_latex_fixed` where a harvest has one.
    """
    results = ROOT / 'results'
    if not results.is_dir():
        return results / 'decisions_arc' / 'tables'
    harvests = sorted((d for d in results.glob('decisions_arc*') if d.is_dir()),
                      key=lambda d: d.name, reverse=True)
    for harvest in harvests:
        for name in ('tables_latex_fixed', 'tables'):
            if (harvest / name).is_dir():
                return harvest / name
    return results / 'decisions_arc' / 'tables'


TABLES = _tables_dir()

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


def check_missing_cells_are_dashes():
    """A cell with no value prints an em dash, and the CSV keeps the real NaN.

    `NaN` in a printed table reads to a referee as a calculation that failed
    rather than as a combination that was never run, and the journal's tables
    use a dash with the reason in the footnote. The CSV is what every other
    reader of these tables uses, so it must not carry the dash.
    """
    table = pd.DataFrame([
        {'Model': 'RF', 'Sort & Slice': 0.961, 'Outlier (10%)': np.nan},
        {'Model': 'LightGBM', 'Sort & Slice': np.nan, 'Outlier (10%)': 0.603},
    ])
    with tempfile.TemporaryDirectory() as tmp:
        T.write(table, tmp, 'TX_missing', caption='a table with gaps in it')
        text = (Path(tmp) / 'TX_missing.tex').read_text()
        back = pd.read_csv(Path(tmp) / 'TX_missing.csv')
    assert 'NaN' not in text, 'the LaTeX fragment still prints NaN'
    assert text.count(T.MISSING_CELL) == 2, (
        f'expected two em dashes in the fragment, found '
        f'{text.count(T.MISSING_CELL)}')
    assert back['Outlier (10%)'].isna().sum() == 1, (
        'the CSV lost its NaN and got a dash instead')
    assert not audit(text, 'TX_missing'), 'the dashed fragment will not compile'
    print('  missing cells: em dash in the .tex, real NaN kept in the .csv')


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
    check_missing_cells_are_dashes()
    check_on_disk()
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
