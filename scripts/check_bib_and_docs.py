#!/usr/bin/env python3
"""Assert the bibliography resolves and the two design documents have not re-drifted.

Chat K, 2026-08-26. Every check here corresponds to a defect that was actually found and
fixed, so removing a fix makes this script fail. Run it before any paper pass:

    python3 scripts/check_bib_and_docs.py

Exit status is 1 if any check fails. Pending manuscript edits are reported separately and
do not fail the run, because `paper.tex` is the author's file and is never edited here.

Why each check exists is recorded beside it. The two documents are:

  NOISE_DESIGN.md  - owns what the noise IS: conditions, algebra, parameters, sources,
                     level grids, and the checks that are properties of the noise scheme.
  RERUN_PLAN.md    - owns what gets RUN and in what order.

Neither restates the other. The drift checks below are what stops that rule eroding.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper.tex"
BIB = ROOT / "citations.bib"
DESIGN = ROOT / "NOISE_DESIGN.md"
PLAN = ROOT / "RERUN_PLAN.md"

failures = []
pending = []


def fail(check, detail):
    failures.append((check, detail))


def read(path):
    if not path.exists():
        fail("file present", f"{path.name} does not exist")
        return ""
    return path.read_text(encoding="utf-8")


def strip_bib_comments(text):
    """Drop % comment lines. The blocklist is *named* in a comment block at the head of
    the added entries, so scanning the raw file for those names would always trip."""
    return "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))


# BibTeX tolerates `@article {Key,` with a space; a scan that assumes `@article{Key,`
# reports a defined key as missing. That exact false positive put five phantom entries
# into RERUN_PLAN.md section 9.1 and survived weeks of re-reading.
BIB_ENTRY = re.compile(r"^[ \t]*@([A-Za-z]+)[ \t]*\{[ \t]*([^,\s]+)[ \t]*,", re.MULTILINE)
CITE = re.compile(r"\\cite[a-zA-Z]*\s*(?:\[[^\]]*\]\s*)*\{([^}]*)\}")


def bib_keys(text):
    return [m.group(2) for m in BIB_ENTRY.finditer(text)]


def cited_keys(text):
    keys = set()
    for m in CITE.finditer(text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


paper = read(PAPER)
bib = read(BIB)
design = read(DESIGN)
plan = read(PLAN)
bib_body = strip_bib_comments(bib)

# ---------------------------------------------------------------- 1. keys resolve
# `Rogers2010` was cited while the entry key was `rogers2010`. Traditional BibTeX
# matches case-insensitively so it resolved; biber does not. Compare case-sensitively
# so the mismatch cannot come back silently.
defined = bib_keys(bib_body)
defined_set = set(defined)
cited = cited_keys(paper)
undefined = sorted(cited - defined_set)
if undefined:
    fail("every cited key is defined in citations.bib (case-sensitively)",
         "undefined: " + " ".join(undefined))

# ---------------------------------------------------------------- 2. no key collisions
# `Xu2019` was defined twice, on two DIFFERENT papers. BibTeX keeps the first and says
# nothing, so a citation would have silently resolved to the wrong source.
seen, dupes = set(), []
for k in defined:
    if k in seen:
        dupes.append(k)
    seen.add(k)
if dupes:
    fail("no two entries share a key in citations.bib",
         "duplicated: " + " ".join(sorted(set(dupes))))

# ---------------------------------------------------------------- 3. sources are reachable
# NOISE_DESIGN.md's Sources table gives the BibTeX key beside every source so a paper
# pass can cite straight from the evidence document. A key that names nothing makes the
# table worse than useless.
sources_section = design.split("## Sources", 1)[-1]
source_keys = set(re.findall(r"\|\s*`([A-Za-z][A-Za-z0-9_]*)`\s*\|", sources_section))
if not source_keys:
    fail("NOISE_DESIGN.md Sources table lists BibTeX keys",
         "no `key` cells found - has the table been reformatted?")
missing_sources = sorted(source_keys - defined_set)
if missing_sources:
    fail("every source in NOISE_DESIGN.md has a citations.bib entry",
         "no entry for: " + " ".join(missing_sources))

# ---------------------------------------------------------------- 4. rejected sources stay out
# NOISE_DESIGN.md section 4a lists sources traced to source and rejected. Until now
# nothing stopped a later pass re-adding one.
BLOCKLIST = {
    "Matsson": "no such paper exists; Matsson's only 2019 paper is about unbound intracellular drug fraction",
    "Pham-The": "contains no reproducibility experiment; its 18.5% is a regression standard error",
    "Lanevskij": "unit error - converted a range-normalised dimensionless RMSE into 'log units'",
    "Fagerholm": "preprint, not peer reviewed",
    "2022.09.27.509731": "the Fagerholm bioRxiv preprint, by identifier",
}
for name, why in BLOCKLIST.items():
    if name.lower() in bib_body.lower():
        fail("no source rejected in NOISE_DESIGN.md section 4a appears in citations.bib",
             f"{name} is present - {why}")

# ---------------------------------------------------------------- 5. the manuscript's \bibliography
# paper.tex is the author's file and is never edited from here, so a wrong target is
# reported as an outstanding edit rather than a failure. This turns green on its own.
m = re.search(r"^\s*\\bibliography\{([^}]*)\}", paper, re.MULTILINE)
if not m:
    fail("paper.tex has a \\bibliography line", "none found")
else:
    named = m.group(1).strip()
    if not (ROOT / f"{named}.bib").exists():
        pending.append(
            f"paper.tex names \\bibliography{{{named}}} but {named}.bib does not exist. "
            f"The bibliography is citations.bib - change that one line to "
            f"\\bibliography{{citations}}. See RERUN_PLAN.md section 9.1. "
            f"Do NOT make the same change to paper_inline_bbl.tex, which inlines a "
            f"compiled .bbl on purpose.")

# ---------------------------------------------------------------- 6. the documents own their facts
# Each of these was found in both documents, saying different things. The fact now lives
# in one owner and the other points at it.
def absent_from(text, needle, doc, check, why):
    if needle in text:
        fail(check, f"{doc} contains {needle!r} - {why}")


# The level grids are NOISE_DESIGN.md section 6.4's and nowhere else.
for grid in ("0, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5", "0, 10, 20, 25, 30, 40, 50%"):
    absent_from(plan, grid, "RERUN_PLAN.md",
                "the level grids live only in NOISE_DESIGN.md section 6.4",
                "point at section 6.4 instead of restating the grid")

# Section 1 of the design used to propose its own ladders, which section 6.4 superseded.
for stale in ("0, 0.25, 0.5, 0.68, 1.0", "k \u2208 {0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0}"):
    absent_from(design, stale, "NOISE_DESIGN.md",
                "NOISE_DESIGN.md states each level grid once, in section 6.4",
                "section 1 proposed a ladder that section 6.4 supersedes")

# The threshold-degeneracy figure is quoted in both documents by design, so the check is
# that they quote the SAME value. These are the superseded roundings.
for doc, text in (("NOISE_DESIGN.md", design), ("RERUN_PLAN.md", plan)):
    for stale in ("99.9992%", "0.67 eV"):
        absent_from(text, stale, doc,
                    "both documents quote the same threshold-degeneracy figures",
                    "the verified values are 99.99925% and 0.669 eV")

# The design named rust/src/main.rs and nothing else, which left the injector behind every
# uncertainty number unspecified. That is how the two implementations drifted apart.
if "NoiseInject/noiseInject/core.py" not in design:
    fail("NOISE_DESIGN.md specifies BOTH injectors",
         "the Python injector path is not named - the specification covers only the Rust half")

# Six state documents were recorded as deleted; they were restored and are on disk.
restored = ["RESULTS_REWORK.md", "DISCUSSION_REWORK.md", "DISCUSSION_TRACKER.md",
            "REVISION_STATUS.md", "immediate_next_steps.md", "UNCERTAINTY_METRIC_FIX_PLAN.md"]
on_disk = [f for f in restored if (ROOT / f).exists()]
if on_disk and "all deleted 2026-08-24" in design:
    fail("no document claims the restored state documents were deleted",
         f"NOISE_DESIGN.md says 'all deleted 2026-08-24' but {len(on_disk)} of them are on disk")

# ---------------------------------------------------------------- report
print(f"citations.bib: {len(defined)} entries, {len(defined_set)} distinct keys")
print(f"paper.tex:     {len(cited)} cited keys")
print(f"NOISE_DESIGN.md Sources: {len(source_keys)} keys")
print()

for note in pending:
    print(f"PENDING (author's edit, not a failure): {note}\n")

if failures:
    print(f"FAILED: {len(failures)} check(s)\n")
    for check, detail in failures:
        print(f"  x {check}")
        print(f"    {detail}")
    sys.exit(1)

print("OK: all checks passed.")
