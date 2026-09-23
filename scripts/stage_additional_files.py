#!/usr/bin/env python3
"""Assemble everything the journal's Additional files upload needs into one folder.

    python3 scripts/stage_additional_files.py

Journal of Cheminformatics is a BMC title, so the submission system takes each
Additional file as its own upload rather than as one appendix bound to the paper.
This writes `additional_files/`, which is a STAGING folder and not a source of
truth: it is deleted and rebuilt on every run, and nothing should ever be edited
inside it. The sources stay where they are, so `scripts/generate_supp_table1.py`
keeps writing `additional_files.tex` at the repository root and keeps passing
`scripts/test_supp_table1.py`.

What lands in the folder:

  additional_files.tex   the manuscript source, with \\graphicspath rewritten to
                         the local `figures/` directory so the folder compiles on
                         its own, in Overleaf or anywhere else
  figures/               every PNG the .tex includes, copied from wherever it was
                         found, with the source path recorded in MANIFEST.md
  MANIFEST.md            one row per Additional file: its number, its title, what
                         it is built from, and the date that source was written

The folder is listed in .gitignore, because everything in it is a copy.
"""
import os
import re
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, 'additional_files.tex')
OUT = os.path.join(ROOT, 'additional_files')

# Where a PNG may live, in the order searched. The v2 directory is the live one;
# `results/paper_figures/` is the superseded NDS output and is searched last so
# that a figure present in both is taken from v2.
FIG_SEARCH = [
    os.path.join(ROOT, 'results', 'decisions_arc', 'figures'),
    os.path.join(ROOT, 'results', 'decisions_arc_20260916', 'figures'),
    os.path.join(ROOT, 'results', 'paper_figures_v2'),
    os.path.join(ROOT, 'results', 'paper_figures'),
]

SECTION = re.compile(r'^% Additional file (\d+)\s*[-\u2014]\s*(.+?)\s*$', re.M)
GRAPHIC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')


def main():
    if not os.path.exists(TEX):
        sys.exit(f'{TEX} does not exist; nothing to stage')
    body = open(TEX, encoding='utf-8').read()

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, 'figures'))

    # Every PNG the .tex includes, copied in and its origin recorded.
    found, missing = [], []
    for name in sorted(set(GRAPHIC.findall(body))):
        for directory in FIG_SEARCH:
            src = os.path.join(directory, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(OUT, 'figures', name))
                stamp = datetime.fromtimestamp(os.path.getmtime(src)).strftime('%Y-%m-%d')
                found.append((name, os.path.relpath(src, ROOT), stamp))
                break
        else:
            missing.append(name)

    # The folder has to compile without the repository around it.
    staged = re.sub(r'\\graphicspath\{\{[^}]*\}\}',
                    r'\\graphicspath{{figures/}}', body)
    if '\\graphicspath' not in staged:
        staged = staged.replace(r'\begin{document}',
                                '\\graphicspath{{figures/}}\n\n\\begin{document}', 1)
    open(os.path.join(OUT, 'additional_files.tex'), 'w', encoding='utf-8').write(staged)

    sections = SECTION.findall(body)
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    with open(os.path.join(OUT, 'MANIFEST.md'), 'w', encoding='utf-8') as fh:
        fh.write('# Additional files, staged for upload\n\n')
        fh.write(f'Written by `scripts/stage_additional_files.py` on {now}. '
                 'Rebuilt from scratch on every run, so nothing here should be edited.\n\n')
        fh.write('Compile `additional_files.tex` twice, for the longtable, then split the '
                 'PDF into one file per Additional file before uploading.\n\n')
        fh.write(f'## The {len(sections)} Additional files\n\n')
        fh.write('| # | Title |\n|---|---|\n')
        for number, title in sections:
            fh.write(f'| {number} | {title} |\n')
        fh.write(f'\n## The {len(found)} figures, and where each was copied from\n\n')
        fh.write('| figure | source | source last written |\n|---|---|---|\n')
        for name, src, stamp in found:
            fh.write(f'| `{name}` | `{src}` | {stamp} |\n')
        if missing:
            fh.write('\n## Figures the .tex includes that are on no search path\n\n')
            for name in missing:
                fh.write(f'- `{name}`\n')

    print(f'staged {len(sections)} Additional files and {len(found)} figures into {OUT}')
    for name in missing:
        print(f'  MISSING FIGURE  {name}')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
