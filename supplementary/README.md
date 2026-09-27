# Supplementary material for the NoiseInject paper

This folder is the one source for the paper's Additional files. The thesis has its own copy of
this system, for the same material, at `KIRBy/thesis_appendix/noise/`. The two are kept separate
on purpose: they start with the same content and are then edited independently.

Set up on 2026-09-27. Before that, the Additional files were typed straight into
`additional_files.tex` with their numbers built into captions and labels. Those numbers drifted
from what paper.tex cites (see the Notes column of `INDEX.md`).

## What is here

| Path | What it is | Edit it? |
|---|---|---|
| `INDEX.md` | One row per supplementary table or figure: its name, title, what makes it, its Additional file number, status, and where paper.tex cites it | Yes. This is where numbers and order are decided |
| `items/<name>.tex` | One table or figure, as a LaTeX fragment | Yes, unless its header says GENERATED |
| `preamble.tex` | Packages and settings for the Additional files document | Yes |
| `../additional_files.tex` | The assembled document | **Never.** It is rebuilt from the three above |
| `../additional_files/` | Upload folder made by `scripts/stage_additional_files.py` | **Never.** Deleted and rebuilt on every run |

## The rules

1. **An item's name is permanent.** It is lower case with underscores and says what the item
   shows: `pdv_descriptors`, `excluded_configurations`. It is the file name, the label
   (`\label{supp:<name>}`) and the name the thesis copy keeps. Never rename an item. If the
   content changes so much that the name is wrong, make a new item and drop the old row.
2. **Numbers live only in `INDEX.md`.** No item file contains "Additional file 5", "Table S3" or
   any other number. A caption starts with `\SuppFile{}`, which the build turns into
   "Additional file N," for the paper. Tables within one Additional file are lettered in the
   caption text ("Table A") because the letter belongs to the item, not to its position.
3. **One item is one table or one figure.** An Additional file can hold several items: give them
   the same number in `INDEX.md`, and they print in the order the rows appear.
4. **A generated item is never hand-edited.** Its first lines say which script writes it. Change
   the script or its inputs and re-run it. Today that is the three `hyperparameters_*` /
   `pdv_descriptors` items, written by `scripts/generate_supp_table1.py --write` and guarded by
   `scripts/test_supp_table1.py`.
5. **Results tables are pulled in, not copied.** An item may `\input{results/.../T11_x}` a table
   that `scripts/run_paper_analysis.py` writes. The build pastes the file's text in place, so
   the output is still one file that Overleaf compiles.
6. **Every status is honest.** `current` means built from the study as it stands. `stale`
   means built from the earlier study (NDS, six noise strategies, mol2vec). `planned` means the
   text cites it and no file exists. Update the row in the same change as the item.
7. **Re-check the "Cited in paper.tex" column after every Overleaf download.** paper.tex is
   read-only here (see `CLAUDE.md`), so a renumbering has to be made in Overleaf by hand. The
   column says which lines to change there.

## How to

**Add an item.** Write `items/<name>.tex` (copy the shape of an existing one: a float or
longtable, caption starting `\textbf{\SuppFile{} ...}`, `\label{supp:<name>}`). Add a row to
`INDEX.md`. Give it a number or `—`. Run the build.

**Renumber or reorder.** Change the Paper file column in `INDEX.md` and run the build. Numbers
must run 1, 2, 3 … with no gaps. Then fix the citations in paper.tex in Overleaf, using the
"Cited in paper.tex" column.

**Drop an item from the paper.** Set its Paper file to `—`. Keep the row and the file, so the
thesis still knows where its copy came from.

**Build.**
```
python3 scripts/generate_supp_table1.py --write   # only if model settings changed
python3 scripts/build_additional_files.py         # writes additional_files.tex
python3 scripts/build_additional_files.py --check # exit 1 if additional_files.tex is out of date
python3 scripts/test_supp_table1.py               # guards on Additional file 1
python3 scripts/stage_additional_files.py         # upload folder with the figures copied in
```
There is no LaTeX here. `additional_files.tex` is compiled in Overleaf (twice, for the
longtables). After compiling, copy the PDF back to `additional_files.pdf`; the last check in
`scripts/test_supp_table1.py` fails until you do.

**Figures.** An item names its PNG by file name only. The build leaves the path to
`\graphicspath` in `preamble.tex`, and `scripts/stage_additional_files.py` finds each PNG on its
search list and copies it into `additional_files/figures/`.

## For the thesis

The thesis never reads this folder at build time. To start a thesis copy of an item, run from
the KIRBy repository:
```
python3 thesis_appendix/seed_from_paper.py <name>
```
That copies `items/<name>.tex` once. After that the two files are independent. Rules and index
for the thesis side: `KIRBy/thesis_appendix/README.md`.
